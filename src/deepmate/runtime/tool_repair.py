"""Runtime repair helpers for model tool-call failures."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from hashlib import sha256

from deepmate.domain import ErrorInfo, RuntimeEvent
from deepmate.providers import ModelResponse, ModelToolRequest, ModelToolResult

DEFAULT_MAX_IDENTICAL_TOOL_CALLS = 2
DEFAULT_MAX_SIMILAR_TOOL_CALLS = 2

_SEARCH_KEYS = ("query", "pattern", "keyword", "keywords")
_SEARCH_SCOPE_KEYS = ("path", "root", "directory", "folder")
_TARGET_KEYS = ("path", "file_path", "target_path", "target")
_CONTENT_KEYS = ("content", "text", "new_text", "replacement")


@dataclass(frozen=True, slots=True)
class ToolCallRepairResult:
    """A repair decision that can be consumed by the agent loop."""

    error: ErrorInfo | None
    events: tuple[RuntimeEvent, ...]
    tool_results: tuple[ModelToolResult, ...] = ()
    status: str = ""

    def has_replay_result(self) -> bool:
        """Return whether the repair produced a model-replayable tool result."""
        return bool(self.tool_results)


@dataclass(frozen=True, slots=True)
class ToolArgumentRepair:
    """A deterministic argument repair applied before tool execution."""

    request: ModelToolRequest
    events: tuple[RuntimeEvent, ...]
    status: str = ""


@dataclass(frozen=True, slots=True)
class ToolCallScavenge:
    """Recovered tool calls that a provider did not surface structurally."""

    tool_requests: tuple[ModelToolRequest, ...]
    events: tuple[RuntimeEvent, ...]
    status: str = ""


@dataclass(frozen=True, slots=True)
class ToolRepairPolicy:
    """Configurable tool-call repair behavior for one runtime turn."""

    enabled: bool = True
    reasoning_scavenge: bool = True
    argument_repair: bool = True
    max_identical_tool_calls: int = DEFAULT_MAX_IDENTICAL_TOOL_CALLS
    max_similar_tool_calls: int = DEFAULT_MAX_SIMILAR_TOOL_CALLS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_identical_tool_calls",
            max(0, self.max_identical_tool_calls),
        )
        object.__setattr__(
            self,
            "max_similar_tool_calls",
            max(0, self.max_similar_tool_calls),
        )

    def new_state(self) -> "ToolCallRepairState":
        """Return fresh per-turn state matching this policy."""
        if not self.enabled:
            return ToolCallRepairState(
                max_identical_tool_calls=0,
                max_similar_tool_calls=0,
            )
        return ToolCallRepairState(
            max_identical_tool_calls=self.max_identical_tool_calls,
            max_similar_tool_calls=self.max_similar_tool_calls,
        )

    def trace_refs(self) -> tuple[str, ...]:
        """Return concise refs for repair policy observability."""
        return (
            f"tool_repair_enabled={str(self.enabled).lower()}",
            f"reasoning_scavenge={str(self.reasoning_scavenge).lower()}",
            f"argument_repair={str(self.argument_repair).lower()}",
            f"max_identical_tool_calls={self.max_identical_tool_calls}",
            f"max_similar_tool_calls={self.max_similar_tool_calls}",
        )


@dataclass(frozen=True, slots=True)
class _JsonDelimiterScan:
    stack: tuple[str, ...]
    in_string: bool
    escape: bool


@dataclass(slots=True)
class ToolCallRepairState:
    """State for one user turn's repair decisions."""

    max_identical_tool_calls: int = DEFAULT_MAX_IDENTICAL_TOOL_CALLS
    max_similar_tool_calls: int = DEFAULT_MAX_SIMILAR_TOOL_CALLS
    _seen_calls: dict[str, int] = field(default_factory=dict)
    _seen_similar_calls: dict[str, int] = field(default_factory=dict)

    def repeated_call_result(
        self,
        request: ModelToolRequest,
    ) -> ToolCallRepairResult | None:
        """Return a suppression result when one tool call repeats too often."""
        if not request.is_ready():
            return None
        key = _tool_call_key(request)
        next_count = self._seen_calls.get(key, 0) + 1
        self._seen_calls[key] = next_count
        limit = max(0, self.max_identical_tool_calls)
        if limit > 0 and next_count > limit:
            return _suppressed_call_result(
                request,
                repeat_count=next_count,
                repeat_limit=limit,
                similar=False,
            )

        similar_limit = max(0, self.max_similar_tool_calls)
        if similar_limit <= 0 or next_count > 1:
            return None
        similar_key = _similar_tool_call_key(request)
        if similar_key is None:
            return None
        similar_count = self._seen_similar_calls.get(similar_key, 0) + 1
        self._seen_similar_calls[similar_key] = similar_count
        if similar_count <= similar_limit:
            return None
        return _suppressed_call_result(
            request,
            repeat_count=similar_count,
            repeat_limit=similar_limit,
            similar=True,
        )


def invalid_tool_request_result(request: ModelToolRequest) -> ToolCallRepairResult | None:
    """Return an invalid-request diagnostic, if the request is malformed."""
    tool_name = _text(request.name)
    request_id = _text(request.id)
    if not tool_name:
        message = "Tool request requires a tool name and tool call id."
        refs = (request_id,) if request_id else ("tool_name=<empty>",)
        return ToolCallRepairResult(
            error=ErrorInfo(
                code="tool_request_invalid",
                message=message,
                refs=refs,
            ),
            events=(RuntimeEvent(kind="tool_request_invalid", summary=message, refs=refs),),
            status="tool request ignored: missing tool name",
        )
    if not request_id:
        message = "Tool request requires a tool call id."
        refs = (tool_name, "tool_call_id=<empty>")
        return ToolCallRepairResult(
            error=ErrorInfo(
                code="tool_request_invalid",
                message=message,
                refs=refs,
            ),
            events=(RuntimeEvent(kind="tool_request_invalid", summary=message, refs=refs),),
            status=f"tool request ignored: {tool_name} missing tool call id",
        )
    return None


def tool_argument_diagnostic(request: ModelToolRequest) -> RuntimeEvent | None:
    """Return a non-blocking diagnostic for provider argument parse errors."""
    error = _text(request.argument_error)
    if not error:
        return None
    tool_name = _text(request.name) or "<empty>"
    request_id = _text(request.id)
    return RuntimeEvent(
        kind="tool_arguments_invalid",
        summary=f"Tool arguments invalid: {tool_name}: {error}",
        refs=(
            f"tool={tool_name}",
            f"tool_call_id={request_id}",
            f"argument_error={error}",
        ),
    )


def repair_tool_arguments(request: ModelToolRequest) -> ToolArgumentRepair | None:
    """Repair a conservatively recoverable truncated JSON argument object."""
    error = _text(request.argument_error)
    raw_arguments = _text(request.raw_arguments)
    if not error or not raw_arguments:
        return None
    repaired = _parse_repaired_json_object(raw_arguments)
    if repaired is None:
        return None

    tool_name = _text(request.name) or "<empty>"
    request_id = _text(request.id)
    raw_repaired = json.dumps(repaired, ensure_ascii=False, separators=(",", ":"))
    repaired_request = replace(
        request,
        arguments=repaired,
        raw_arguments=raw_repaired,
        argument_error="",
    )
    message = f"Tool arguments repaired before execution: {tool_name}."
    event = RuntimeEvent(
        kind="tool_arguments_repaired",
        summary=message,
        refs=(
            f"tool={tool_name}",
            f"tool_call_id={request_id}",
            f"argument_error={error}",
        ),
    )
    return ToolArgumentRepair(
        request=repaired_request,
        events=(event,),
        status=f"tool arguments repaired: {tool_name}",
    )


def scavenge_tool_requests_from_response(
    response: ModelResponse,
    visible_tool_names: Iterable[str],
    *,
    step_index: int,
) -> ToolCallScavenge | None:
    """Recover explicit tool-call JSON that was emitted as text or reasoning."""
    if response.tool_requests:
        return None
    allowed = {name.strip() for name in visible_tool_names if isinstance(name, str)}
    if not allowed:
        return None

    requests: list[ModelToolRequest] = []
    events: list[RuntimeEvent] = []
    seen: set[str] = set()
    sources = _scavenge_sources(response)
    for source, text in sources:
        for payload in _tool_call_payload_candidates(text):
            for call in _tool_call_objects(payload):
                request = _tool_request_from_payload(
                    call,
                    allowed,
                    step_index=step_index,
                    request_index=len(requests) + 1,
                )
                if request is None:
                    continue
                key = _tool_call_key(request)
                if key in seen:
                    continue
                seen.add(key)
                requests.append(request)
                events.append(
                    RuntimeEvent(
                        kind="tool_call_scavenged",
                        summary=(
                            "Tool call recovered from model text/reasoning: "
                            f"{request.name}."
                        ),
                        refs=(
                            f"tool={request.name}",
                            f"tool_call_id={request.id}",
                            f"source={source}",
                            f"step={step_index}",
                        ),
                    )
                )
                if len(requests) >= 3:
                    break
            if len(requests) >= 3:
                break
        if len(requests) >= 3:
            break
    if not requests:
        return None
    names = ", ".join(request.name for request in requests)
    return ToolCallScavenge(
        tool_requests=tuple(requests),
        events=tuple(events),
        status=f"tool call scavenged: {names}",
    )


def mcp_schema_not_loaded_result(request: ModelToolRequest) -> ToolCallRepairResult:
    """Return a recovery result for an MCP tool whose schema is not loaded."""
    tool_name = _text(request.name) or "<empty>"
    request_id = _text(request.id)
    message = (
        f"MCP tool schema is not loaded for this step: {tool_name}. "
        f"Call load_mcp_tool with name=\"{tool_name}\" before invoking this MCP tool."
    )
    refs = (tool_name, f"tool_call_id={request_id}", "recovery=load_mcp_tool")
    return ToolCallRepairResult(
        error=None,
        events=(
            RuntimeEvent(
                kind="mcp_tool_schema_not_loaded",
                summary=message,
                refs=refs,
            ),
        ),
        tool_results=(
            ModelToolResult(
                name=tool_name,
                request_id=request_id,
                content=message,
                refs=refs,
                is_error=True,
            ),
        ),
        status=f"MCP schema not loaded: call load_mcp_tool for {tool_name}",
    )


def _scavenge_sources(response: ModelResponse) -> tuple[tuple[str, str], ...]:
    content = _text(response.content)
    reasoning = _text(response.reasoning)
    sources: list[tuple[str, str]] = []
    if content:
        sources.extend(("content", block) for block in _tagged_tool_call_blocks(content))
    if reasoning:
        sources.extend(("reasoning", block) for block in _tagged_tool_call_blocks(reasoning))
        if not content and _looks_like_standalone_json(reasoning):
            sources.append(("reasoning", reasoning))
    return tuple(sources)


def _looks_like_standalone_json(text: str) -> bool:
    clean = text.strip()
    return (clean.startswith("{") and clean.endswith("}")) or (
        clean.startswith("[") and clean.endswith("]")
    )


def _tagged_tool_call_blocks(text: str) -> tuple[str, ...]:
    blocks: list[str] = []
    for pattern in (
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        r"<tool_calls>\s*(.*?)\s*</tool_calls>",
    ):
        blocks.extend(
            match.group(1).strip()
            for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if match.group(1).strip()
        )
    return tuple(blocks)


def _tool_call_payload_candidates(text: str) -> tuple[object, ...]:
    candidates: list[object] = []
    for candidate in _json_repair_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, (Mapping, list)):
            candidates.append(parsed)
    return tuple(candidates)


def _tool_call_objects(payload: object) -> tuple[Mapping[str, object], ...]:
    if isinstance(payload, list):
        return tuple(item for item in payload if isinstance(item, Mapping))
    if not isinstance(payload, Mapping):
        return ()
    tool_calls = payload.get("tool_calls")
    if isinstance(tool_calls, list):
        return tuple(item for item in tool_calls if isinstance(item, Mapping))
    tool_call = payload.get("tool_call")
    if isinstance(tool_call, list):
        return tuple(item for item in tool_call if isinstance(item, Mapping))
    if isinstance(tool_call, Mapping):
        return (tool_call,)
    return (payload,)


def _tool_request_from_payload(
    payload: Mapping[str, object],
    allowed: set[str],
    *,
    step_index: int,
    request_index: int,
) -> ModelToolRequest | None:
    function = payload.get("function")
    function_payload = function if isinstance(function, Mapping) else None
    name = _text(
        payload.get("name")
        or payload.get("tool_name")
        or payload.get("tool")
        or (function_payload.get("name") if function_payload is not None else "")
    )
    if name not in allowed:
        return None
    raw_arguments = (
        function_payload.get("arguments") if function_payload is not None else None
    )
    if raw_arguments is None:
        raw_arguments = payload.get("arguments")
    if raw_arguments is None:
        raw_arguments = payload.get("args")
    if not isinstance(raw_arguments, (Mapping, str)):
        return None
    arguments = _mapping_arguments(raw_arguments)
    if arguments is None:
        return None
    request_id = _text(payload.get("id") or payload.get("tool_call_id"))
    if not request_id:
        request_id = f"scavenged_{step_index}_{request_index}"
    return ModelToolRequest(
        name=name,
        arguments=arguments,
        id=request_id,
        raw_arguments=json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
    )


def _mapping_arguments(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return None
    parsed = _parse_repaired_json_object(value)
    if parsed is None:
        return None
    return parsed


def _suppressed_call_result(
    request: ModelToolRequest,
    *,
    repeat_count: int,
    repeat_limit: int,
    similar: bool,
) -> ToolCallRepairResult:
    tool_name = _text(request.name) or "<empty>"
    kind = "tool_call_similar_suppressed" if similar else "tool_call_repeated_suppressed"
    wording = "similar arguments" if similar else "identical arguments"
    message = (
        "Repeated tool call suppressed: "
        f"{tool_name} with {wording} was requested {repeat_count} times "
        f"in this user turn. Inspect the previous tool result and choose a "
        "different next step instead of calling a near-duplicate tool again."
    )
    refs = (
        f"tool={tool_name}",
        f"tool_call_id={_text(request.id)}",
        f"repeat_count={repeat_count}",
        f"repeat_limit={repeat_limit}",
        f"repeat_kind={'similar' if similar else 'identical'}",
    )
    return ToolCallRepairResult(
        error=None,
        events=(
            RuntimeEvent(
                kind=kind,
                summary=message,
                refs=refs,
            ),
        ),
        tool_results=(
            ModelToolResult(
                name=tool_name,
                request_id=_text(request.id),
                content=message,
                refs=refs,
                is_error=True,
            ),
        ),
        status=(
            "tool call suppressed: "
            f"{tool_name} repeated {repeat_count} times with {wording}"
        ),
    )


def _tool_call_key(request: ModelToolRequest) -> str:
    return json.dumps(
        {
            "name": _text(request.name),
            "arguments": _normalized_mapping(request.arguments),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _similar_tool_call_key(request: ModelToolRequest) -> str | None:
    search_key = _search_tool_call_key(request)
    if search_key is not None:
        return search_key
    return _target_content_tool_call_key(request)


def _search_tool_call_key(request: ModelToolRequest) -> str | None:
    tool_name = _text(request.name)
    search_text = _first_text_value(request.arguments, _SEARCH_KEYS)
    if not search_text:
        return None
    if not _looks_search_like(tool_name) and not any(
        key in request.arguments for key in _SEARCH_KEYS
    ):
        return None
    normalized_query = _normalized_search_text(search_text)
    if len(normalized_query) < 3:
        return None
    scope = _normalized_search_text(_first_text_value(request.arguments, _SEARCH_SCOPE_KEYS))
    return json.dumps(
        {
            "name": tool_name,
            "kind": "search",
            "query": normalized_query,
            "scope": scope,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _target_content_tool_call_key(request: ModelToolRequest) -> str | None:
    target = _first_text_value(request.arguments, _TARGET_KEYS)
    content = _first_text_value(request.arguments, _CONTENT_KEYS)
    if not target or not content:
        return None
    digest = sha256(content.encode("utf-8")).hexdigest()[:16]
    return json.dumps(
        {
            "name": _text(request.name),
            "kind": "target_content",
            "target": target.strip(),
            "content_sha256": digest,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _first_text_value(arguments: Mapping[str, object], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _looks_search_like(tool_name: str) -> bool:
    lowered = tool_name.lower()
    return any(marker in lowered for marker in ("search", "find", "grep", "query"))


def _normalized_search_text(value: str) -> str:
    lowered = value.lower()
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", " ", lowered, flags=re.UNICODE)
    return " ".join(normalized.split())


def _parse_repaired_json_object(text: str) -> dict[str, object] | None:
    for candidate in _json_repair_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _json_repair_candidates(text: str) -> tuple[str, ...]:
    cleaned = text.strip()
    if not cleaned:
        return ()
    candidates = [cleaned]
    repaired = _remove_trailing_commas(_balanced_json_candidate(cleaned))
    if repaired != cleaned:
        candidates.append(repaired)
    trailing_fixed = _remove_trailing_commas(cleaned)
    if trailing_fixed not in candidates:
        candidates.append(trailing_fixed)
    return tuple(candidates)


def _balanced_json_candidate(text: str) -> str:
    candidate = text.strip()
    if candidate.endswith(":"):
        candidate = f"{candidate}null"
    state = _scan_json_delimiters(candidate)
    if state.in_string:
        candidate += '\\"' if state.escape else '"'
    if len(state.stack) > 20:
        return candidate
    closers = {"{": "}", "[": "]"}
    return candidate + "".join(closers[opener] for opener in reversed(state.stack))


def _scan_json_delimiters(text: str) -> _JsonDelimiterScan:
    stack: list[str] = []
    in_string = False
    escape = False
    for character in text:
        if in_string:
            if escape:
                escape = False
            elif character == "\\":
                escape = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "{[":
            stack.append(character)
        elif character in "}]":
            if stack and _matching_delimiter(stack[-1], character):
                stack.pop()
            else:
                return _JsonDelimiterScan(stack=(), in_string=False, escape=False)
    return _JsonDelimiterScan(
        stack=tuple(stack),
        in_string=in_string,
        escape=escape,
    )


def _matching_delimiter(opener: str, closer: str) -> bool:
    return (opener == "{" and closer == "}") or (opener == "[" and closer == "]")


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _normalized_mapping(value: Mapping[str, object]) -> object:
    try:
        json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)
    return value


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""
