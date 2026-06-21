"""Headroom-aware tool output normalization and compaction."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from deepmate.domain import RuntimeEvent
from deepmate.foundation import estimate_text_tokens
from deepmate.providers import ModelToolResult
from deepmate.runtime.conversation_budget import (
    ConversationBudgetPolicy,
    RequestBudgetReport,
)
from deepmate.storage.tool_output_store import ToolOutputRecord, ToolOutputStore

LOW_PRESSURE_RATIO = 0.25
CHECKPOINT_PRESSURE_RATIO = 0.50
EMERGENCY_PRESSURE_RATIO = 0.75
SMALL_OUTPUT_RATIO = 0.0025
MEDIUM_OUTPUT_RATIO = 0.01
HUGE_OUTPUT_RATIO = 0.03
MIN_LARGE_NOISY_TOKENS = 4_000
MIN_HUGE_TOKENS = 10_000
COMPACT_TARGET_RATIO = 0.003
MIN_COMPACT_TARGET_TOKENS = 1_200
MAX_COMPACT_TARGET_TOKENS = 4_000

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_SEARCH_RE = re.compile(r"^[^:\n]+:\d+(?::\d+)?:")
_ERROR_RE = re.compile(
    r"(traceback|assertionerror|exception|error|failed|failure|stderr|panic)",
    re.IGNORECASE,
)
_BROWSER_REF_RE = re.compile(
    r"(@[A-Za-z0-9_-]+|\[ref=([A-Za-z0-9@_.:-]+)\]|ref[:=]\s*([A-Za-z0-9@_.:-]+))"
)
_DATA_IMAGE_RE = re.compile(
    r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=\s]{120,}",
    re.IGNORECASE,
)
_LONG_BASE64_LINE_RE = re.compile(r"^[A-Za-z0-9+/]{200,}={0,2}$")


class ToolOutputKind(Enum):
    """Detected model-facing tool output kind."""

    BROWSER = "browser"
    LOG = "log"
    JSON = "json"
    SEARCH = "search"
    DIFF = "diff"
    FILE = "file"
    TABLE = "table"
    PLAIN = "plain"


@dataclass(frozen=True, slots=True)
class ToolOutputProcessResult:
    """Processed tool result and runtime observability payload."""

    result: ModelToolResult
    events: tuple[RuntimeEvent, ...] = field(default_factory=tuple)
    status_messages: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ToolOutputCompactionPolicy:
    """Cost-oriented thresholds for tool-output governance."""

    small_output_ratio: float = SMALL_OUTPUT_RATIO
    medium_output_ratio: float = MEDIUM_OUTPUT_RATIO
    huge_output_ratio: float = HUGE_OUTPUT_RATIO
    compact_target_ratio: float = COMPACT_TARGET_RATIO

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "small_output_ratio",
            _bounded_ratio(self.small_output_ratio),
        )
        object.__setattr__(
            self,
            "medium_output_ratio",
            _bounded_ratio(self.medium_output_ratio),
        )
        object.__setattr__(
            self,
            "huge_output_ratio",
            _bounded_ratio(self.huge_output_ratio),
        )
        object.__setattr__(
            self,
            "compact_target_ratio",
            _bounded_ratio(self.compact_target_ratio),
        )


@dataclass(frozen=True, slots=True)
class _NormalizedOutput:
    content: str
    steps: tuple[str, ...]
    destructive: bool = False


@dataclass(frozen=True, slots=True)
class _Pressure:
    current: float
    raw: float
    normalized: float
    usable_input_tokens: int


class ToolOutputCompactor:
    """Apply conservative tool-output governance before model replay."""

    def __init__(
        self,
        *,
        store: ToolOutputStore,
        policy: ConversationBudgetPolicy | None = None,
        enabled: bool = True,
        lossless_normalization: bool = True,
        compaction_policy: ToolOutputCompactionPolicy | None = None,
    ) -> None:
        self._store = store
        self._policy = (policy or ConversationBudgetPolicy()).normalized()
        self._enabled = enabled
        self._lossless_normalization = lossless_normalization
        self._compaction_policy = compaction_policy or ToolOutputCompactionPolicy()

    def with_policy(
        self,
        policy: ConversationBudgetPolicy | None,
    ) -> "ToolOutputCompactor":
        """Return a compactor sharing storage/settings with a new budget policy."""
        return ToolOutputCompactor(
            store=self._store,
            policy=policy,
            enabled=self._enabled,
            lossless_normalization=self._lossless_normalization,
            compaction_policy=self._compaction_policy,
        )

    def process(
        self,
        result: ModelToolResult,
        *,
        tool_source: str,
        request_budget_report: RequestBudgetReport | None = None,
    ) -> ToolOutputProcessResult:
        """Return a governed tool result for model replay."""
        if not self._enabled or not result.is_ready():
            return ToolOutputProcessResult(result=result)
        if _is_retrieved_tool_output(result):
            return ToolOutputProcessResult(result=result)

        raw_text = _raw_tool_text(result)
        original_tokens = estimate_text_tokens(raw_text)
        if original_tokens <= 0:
            return ToolOutputProcessResult(result=result)

        normalized = (
            _normalize_output(raw_text, result)
            if self._lossless_normalization
            else _NormalizedOutput(content=raw_text, steps=())
        )
        normalized_tokens = estimate_text_tokens(normalized.content)
        kind = _detect_kind(normalized.content, result)
        pressure = _pressure(
            original_tokens=original_tokens,
            normalized_tokens=normalized_tokens,
            report=request_budget_report,
            policy=self._policy,
        )

        if not _should_compact(
            normalized=normalized,
            normalized_tokens=normalized_tokens,
            kind=kind,
            pressure=pressure,
            is_error=result.is_error,
            compaction_policy=self._compaction_policy,
        ):
            normalized_result, record = self._normalized_result(
                result,
                normalized=normalized,
                kind=kind,
                original_tokens=original_tokens,
                normalized_tokens=normalized_tokens,
                tool_source=tool_source,
            )
            events, statuses = _normalization_observability(
                result=normalized_result,
                record=record,
                kind=kind,
                original_tokens=original_tokens,
                normalized_tokens=normalized_tokens,
                normalized=normalized,
                pressure=pressure,
                tool_source=tool_source,
            )
            return ToolOutputProcessResult(
                result=normalized_result,
                events=events,
                status_messages=statuses,
            )

        record = self._store.save(
            tool_name=result.name,
            tool_source=tool_source,
            content_kind=kind.value,
            content=raw_text,
            estimated_tokens=original_tokens,
            request_id=result.request_id,
        )
        target_tokens = _compact_target_tokens(
            pressure.usable_input_tokens,
            self._compaction_policy,
        )
        compaction_reason = _compaction_reason(
            pressure,
            kind,
            normalized_tokens,
            self._compaction_policy,
        )
        compacted_content = _compact_content(
            normalized.content,
            kind=kind,
            original_tokens=original_tokens,
            normalized_tokens=normalized_tokens,
            target_tokens=target_tokens,
            record=record,
            reason=compaction_reason,
        )
        compacted_tokens = estimate_text_tokens(compacted_content)
        refs = _merge_refs(
            result.refs,
            (
                "tool_output_compacted=true",
                f"tool_output_ref={record.ref}",
                f"tool_output_kind={kind.value}",
                f"original_estimated_tokens={original_tokens}",
                f"normalized_estimated_tokens={normalized_tokens}",
                f"compacted_estimated_tokens={compacted_tokens}",
            ),
        )
        compacted = ModelToolResult(
            name=result.name,
            request_id=result.request_id,
            content=compacted_content,
            data=_compacted_data(result.data),
            refs=refs,
            attachments=result.attachments,
            is_error=result.is_error,
        )
        event = RuntimeEvent(
            kind="tool_output_compacted",
            summary=f"Tool output compacted: {result.name}.",
            refs=(
                f"tool_source={tool_source}",
                f"tool={result.name}",
                f"tool_output_kind={kind.value}",
                f"tool_output_ref={record.ref}",
                f"original_estimated_tokens={original_tokens}",
                f"normalized_estimated_tokens={normalized_tokens}",
                f"compacted_estimated_tokens={compacted_tokens}",
                f"normalization_ratio={_ratio(normalized_tokens, original_tokens):.4f}",
                f"compression_ratio={_ratio(compacted_tokens, original_tokens):.4f}",
                f"current_pressure={pressure.current:.4f}",
                f"pressure_if_normalized={pressure.normalized:.4f}",
                f"compaction_reason={compaction_reason}",
                *_browser_observability_refs(result),
            ),
        )
        status = (
            f"tool output compacted: {result.name} "
            f"{original_tokens}->{compacted_tokens} tokens "
            f"(ratio={_ratio(compacted_tokens, original_tokens):.3f}, ref={record.ref})"
        )
        return ToolOutputProcessResult(
            result=compacted,
            events=(event,),
            status_messages=(status,),
        )

    def _normalized_result(
        self,
        result: ModelToolResult,
        *,
        normalized: _NormalizedOutput,
        kind: ToolOutputKind,
        original_tokens: int,
        normalized_tokens: int,
        tool_source: str,
    ) -> tuple[ModelToolResult, ToolOutputRecord | None]:
        if not normalized.steps:
            return result, None
        record = None
        extra_refs: tuple[str, ...] = (
            "tool_output_normalized=true",
            f"tool_output_kind={kind.value}",
            f"original_estimated_tokens={original_tokens}",
            f"normalized_estimated_tokens={normalized_tokens}",
            f"normalization={','.join(normalized.steps[:6])}",
        )
        if normalized.destructive:
            record = self._store.save(
                tool_name=result.name,
                tool_source=tool_source,
                content_kind=kind.value,
                content=_raw_tool_text(result),
                estimated_tokens=original_tokens,
                request_id=result.request_id,
            )
            extra_refs = (*extra_refs, f"tool_output_ref={record.ref}")
        return (
            ModelToolResult(
                name=result.name,
                request_id=result.request_id,
                content=_normalized_visible_content(normalized, record=record),
                data={} if not result.content and result.data else result.data,
                refs=_merge_refs(result.refs, extra_refs),
                attachments=result.attachments,
                is_error=result.is_error,
            ),
            record,
        )


def _normalization_observability(
    *,
    result: ModelToolResult,
    record: ToolOutputRecord | None,
    kind: ToolOutputKind,
    original_tokens: int,
    normalized_tokens: int,
    normalized: _NormalizedOutput,
    pressure: _Pressure,
    tool_source: str,
) -> tuple[tuple[RuntimeEvent, ...], tuple[str, ...]]:
    if not normalized.steps:
        return (), ()
    refs = (
        f"tool_source={tool_source}",
        f"tool={result.name}",
        f"tool_output_kind={kind.value}",
        f"original_estimated_tokens={original_tokens}",
        f"normalized_estimated_tokens={normalized_tokens}",
        f"normalization_ratio={_ratio(normalized_tokens, original_tokens):.4f}",
        f"normalization={','.join(normalized.steps[:6])}",
        f"current_pressure={pressure.current:.4f}",
        f"pressure_if_normalized={pressure.normalized:.4f}",
        *((f"tool_output_ref={record.ref}",) if record is not None else ()),
        *_browser_observability_refs(result),
    )
    event = RuntimeEvent(
        kind="tool_output_normalized",
        summary=f"Tool output normalized: {result.name}.",
        refs=refs,
    )
    status = (
        f"tool output normalized: source={tool_source}; tool={result.name}; "
        f"kind={kind.value}; original_tokens={original_tokens}; "
        f"normalized_tokens={normalized_tokens}; normalization={','.join(normalized.steps[:3])}"
    )
    return (event,), (status,)


def _raw_tool_text(result: ModelToolResult) -> str:
    parts: list[str] = []
    if result.content:
        parts.append(result.content)
    if result.data:
        parts.append("[structured data]")
        parts.append(_json_text(result.data))
    if not parts and result.refs:
        parts.append("[refs]")
        parts.append("\n".join(result.refs))
    return "\n".join(parts)


def _is_retrieved_tool_output(result: ModelToolResult) -> bool:
    if result.name.strip() == "retrieve_tool_output":
        return True
    return any(ref.startswith("tool_output_retrieved_tokens=") for ref in result.refs)


def _normalize_output(text: str, result: ModelToolResult) -> _NormalizedOutput:
    if _is_browser_result(result):
        return _normalize_browser_output(text, result)
    return _normalize_losslessly(text)


def _normalize_losslessly(text: str) -> _NormalizedOutput:
    steps: list[str] = []
    destructive = False
    content = text

    parsed = _json_loads(content)
    if parsed is not None:
        minified = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        if minified != content:
            content = minified
            steps.append("json_minify")
        return _NormalizedOutput(content=content, steps=tuple(steps))

    without_ansi = _ANSI_RE.sub("", content)
    if without_ansi != content:
        content = without_ansi
        steps.append("ansi_strip")
        destructive = True

    cr_normalized = _normalize_carriage_returns(content)
    if cr_normalized != content:
        content = cr_normalized
        steps.append("carriage_return_fold")
        destructive = True

    folded = _fold_repeated_lines(content)
    if folded != content:
        content = folded
        steps.append("repeated_line_fold")

    return _NormalizedOutput(
        content=content,
        steps=tuple(steps),
        destructive=destructive,
    )


def _is_browser_result(result: ModelToolResult) -> bool:
    name = result.name.strip()
    if name.startswith("browser_"):
        return True
    if any(ref.startswith("browser_tool=") for ref in result.refs):
        return True
    if isinstance(result.data, dict):
        backend = result.data.get("backend")
        return isinstance(backend, str) and backend.strip() == "agent-browser"
    return False


def _normalize_browser_output(text: str, result: ModelToolResult) -> _NormalizedOutput:
    base = _normalize_losslessly(text)
    content = base.content
    steps = list(base.steps)
    destructive = base.destructive

    binary_cleaned = _strip_browser_binary_payloads(content)
    if binary_cleaned != content:
        content = binary_cleaned
        steps.append("browser_binary_payload_strip")
        destructive = True

    tool_name = _browser_tool_name(result)
    if tool_name == "browser_screenshot":
        normalized = _normalize_browser_screenshot(content, result)
    elif tool_name == "browser_snapshot":
        normalized = _normalize_browser_snapshot(content, result)
    else:
        normalized = _normalize_browser_page_text(content, result)
    if normalized.content != content or normalized.steps:
        content = normalized.content
        steps.extend(normalized.steps)
        destructive = True

    return _NormalizedOutput(
        content=content,
        steps=tuple(_dedupe_preserve_order(steps)),
        destructive=destructive,
    )


def _strip_browser_binary_payloads(text: str) -> str:
    content = _DATA_IMAGE_RE.sub("[browser image data omitted]", text)
    lines = []
    changed = content != text
    for line in content.splitlines():
        if _LONG_BASE64_LINE_RE.match(line.strip()):
            lines.append("[browser base64 line omitted]")
            changed = True
        else:
            lines.append(line)
    return "\n".join(lines) if changed else text


def _normalize_browser_screenshot(
    text: str,
    result: ModelToolResult,
) -> _NormalizedOutput:
    if result.is_error:
        return _NormalizedOutput(content=text, steps=())
    data = result.data if isinstance(result.data, dict) else {}
    lines = ["[browser output normalized: screenshot_metadata]"]
    path = _text_value(data.get("path")) or _ref_value(result.refs, "browser_output")
    if path:
        lines.append(f"path: {path}")
    bytes_count = _int_value(data.get("bytes"))
    if bytes_count:
        lines.append(f"bytes: {bytes_count}")
    image_format = _text_value(data.get("image_format"))
    if image_format:
        lines.append(f"image_format: {image_format}")
    width = _int_value(data.get("width"))
    height = _int_value(data.get("height"))
    if width and height:
        lines.append(f"dimensions: {width}x{height}")
    current_url = _text_value(data.get("current_url")) or _ref_value(result.refs, "browser_url")
    if current_url:
        lines.append(f"url: {current_url}")
    if len(lines) == 1:
        return _NormalizedOutput(content=text, steps=())
    return _NormalizedOutput(content="\n".join(lines), steps=("browser_screenshot_metadata",))


def _normalize_browser_snapshot(
    text: str,
    result: ModelToolResult,
) -> _NormalizedOutput:
    lines = text.splitlines()
    if len(lines) < 8:
        return _normalize_browser_page_text(text, result)
    page_title = _browser_title(lines)
    url = _ref_value(result.refs, "browser_url")
    interactive_lines, text_lines = _browser_snapshot_lines(lines)
    text_lines = _dedupe_browser_lines(text_lines)
    page_lines = [line for line in text_lines if line.strip()]
    sections: list[tuple[str, list[str]]] = []
    card = []
    if page_title:
        card.append(f"title: {page_title}")
    if url:
        card.append(f"url: {url}")
    card.append(f"raw_lines: {len(lines)}")
    card.append(f"interactive_refs: {len(interactive_lines)}")
    sections.append(("Page", card))
    if interactive_lines:
        sections.append(("Interactive refs", interactive_lines[:80]))
    if page_lines:
        sections.append(("Visible text", page_lines[:120]))
    omitted = max(0, len(interactive_lines) - 80) + max(0, len(page_lines) - 120)
    if omitted:
        sections.append(("Suppressed", [f"browser snapshot lines omitted={omitted}"]))
    normalized = "[browser output normalized: snapshot]\n" + _sections(
        tuple((title, tuple(items)) for title, items in sections)
    )
    return _NormalizedOutput(content=normalized, steps=("browser_snapshot_cleanup",))


def _normalize_browser_page_text(
    text: str,
    result: ModelToolResult,
) -> _NormalizedOutput:
    lines = text.splitlines()
    if len(lines) < 12:
        return _NormalizedOutput(content=text, steps=())
    kept = _dedupe_browser_lines(lines)
    if kept == lines:
        return _NormalizedOutput(content=text, steps=())
    url = _ref_value(result.refs, "browser_url")
    body = ["[browser output normalized: page_text]"]
    if url:
        body.append(f"url: {url}")
    body.extend(kept)
    omitted = len(lines) - len(kept)
    if omitted > 0:
        body.append(f"[browser low-value repeated lines omitted: {omitted}]")
    return _NormalizedOutput(content="\n".join(body), steps=("browser_page_text_cleanup",))


def _browser_snapshot_lines(lines: list[str]) -> tuple[list[str], list[str]]:
    interactive: list[str] = []
    text_lines: list[str] = []
    seen_interactive: set[str] = set()
    for line in lines:
        clean = line.strip()
        if not clean:
            continue
        if _is_browser_low_value_line(clean):
            continue
        if _BROWSER_REF_RE.search(clean) or _looks_interactive(clean):
            key = _browser_line_key(clean)
            if key not in seen_interactive:
                interactive.append(clean)
                seen_interactive.add(key)
            continue
        text_lines.append(clean)
    return interactive, text_lines


def _dedupe_browser_lines(lines: list[str]) -> list[str]:
    kept: list[str] = []
    seen: set[str] = set()
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if _is_browser_low_value_line(line):
            continue
        key = _browser_line_key(line)
        if key in seen:
            continue
        seen.add(key)
        kept.append(line)
    return kept


def _is_browser_low_value_line(line: str) -> bool:
    clean = line.strip().lower()
    if not clean:
        return True
    if clean in {"navigation", "main navigation", "footer", "header", "menu", "breadcrumb"}:
        return True
    if clean in {"home", "search", "skip to content", "privacy policy", "terms", "cookie settings"}:
        return True
    if len(clean) <= 2 and not clean.startswith("@"):
        return True
    return False


def _looks_interactive(line: str) -> bool:
    lower = line.lower()
    return any(
        marker in lower
        for marker in (
            "button",
            "link",
            "textbox",
            "input",
            "checkbox",
            "combobox",
            "menuitem",
            "role=",
            "[button",
            "[link",
        )
    )


def _browser_title(lines: list[str]) -> str:
    for line in lines[:40]:
        clean = line.strip()
        lower = clean.lower()
        if lower.startswith("title:"):
            return clean.split(":", 1)[1].strip()
        if lower.startswith("# "):
            return clean[2:].strip()
    return ""


def _browser_tool_name(result: ModelToolResult) -> str:
    for ref in result.refs:
        if ref.startswith("browser_tool="):
            return ref.split("=", 1)[1].strip()
    return result.name.strip()


def _browser_line_key(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip().lower())


def _ref_value(refs: tuple[str, ...], key: str) -> str:
    prefix = f"{key}="
    for ref in refs:
        if ref.startswith(prefix):
            return ref.split("=", 1)[1]
    return ""


def _text_value(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _int_value(value: object) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _dedupe_preserve_order(values: list[str]) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        output.append(value)
        seen.add(value)
    return tuple(output)


def _browser_observability_refs(result: ModelToolResult) -> tuple[str, ...]:
    if not _is_browser_result(result):
        return ()
    refs = []
    for key in (
        "browser_tool",
        "browser_url",
        "browser_output",
        "browser_session",
    ):
        value = _ref_value(result.refs, key)
        if value:
            refs.append(f"{key}={value}")
    if isinstance(result.data, dict):
        for key in ("path", "image_format", "width", "height", "bytes"):
            value = result.data.get(key)
            if isinstance(value, str) and value.strip():
                refs.append(f"browser_{key}={value.strip()}")
            elif isinstance(value, int) and value:
                refs.append(f"browser_{key}={value}")
    return tuple(_dedupe_preserve_order(refs))


def _normalize_carriage_returns(text: str) -> str:
    if "\r" not in text:
        return text
    lines: list[str] = []
    for line in text.split("\n"):
        if "\r" not in line:
            lines.append(line)
            continue
        parts = [part for part in line.split("\r") if part]
        if not parts:
            continue
        lines.append(parts[-1])
    return "\n".join(lines)


def _fold_repeated_lines(text: str) -> str:
    lines = text.splitlines()
    if len(lines) < 4:
        return text
    output: list[str] = []
    index = 0
    changed = False
    while index < len(lines):
        line = lines[index]
        count = 1
        while index + count < len(lines) and lines[index + count] == line:
            count += 1
        if count >= 3 and line.strip():
            output.append(f"{line} [repeated {count} times]")
            changed = True
        else:
            output.extend(lines[index : index + count])
        index += count
    return "\n".join(output) if changed else text


def _detect_kind(text: str, result: ModelToolResult) -> ToolOutputKind:
    if _is_browser_result(result):
        return ToolOutputKind.BROWSER
    if _json_loads(text) is not None:
        return ToolOutputKind.JSON
    stripped = text.lstrip()
    if stripped.startswith(("diff --git", "--- ", "+++ ")) or "\n@@" in text:
        return ToolOutputKind.DIFF
    lines = text.splitlines()
    search_lines = sum(1 for line in lines[:200] if _SEARCH_RE.match(line))
    if search_lines >= 5:
        return ToolOutputKind.SEARCH
    if _looks_like_log(text, lines, result.is_error):
        return ToolOutputKind.LOG
    if _looks_like_table(lines):
        return ToolOutputKind.TABLE
    if _looks_like_file(lines, result):
        return ToolOutputKind.FILE
    return ToolOutputKind.PLAIN


def _looks_like_log(text: str, lines: list[str], is_error: bool) -> bool:
    if is_error:
        return True
    if _ERROR_RE.search(text):
        return True
    lower = text.lower()
    return any(marker in lower for marker in ("pytest", "npm ", "cargo ", "go test"))


def _looks_like_table(lines: list[str]) -> bool:
    non_empty = [line for line in lines if line.strip()]
    if len(non_empty) < 4:
        return False
    comma_rows = sum(1 for line in non_empty[:20] if line.count(",") >= 2)
    pipe_rows = sum(1 for line in non_empty[:20] if line.count("|") >= 2)
    return comma_rows >= 4 or pipe_rows >= 4


def _looks_like_file(lines: list[str], result: ModelToolResult) -> bool:
    if any(ref.startswith("path=") for ref in result.refs):
        return True
    if len(lines) < 20:
        return False
    signal = sum(
        1
        for line in lines[:120]
        if line.startswith(("#", "class ", "def ", "import ", "from "))
        or line.strip().startswith(("function ", "const ", "let ", "export "))
    )
    return signal >= 3


def _pressure(
    *,
    original_tokens: int,
    normalized_tokens: int,
    report: RequestBudgetReport | None,
    policy: ConversationBudgetPolicy,
) -> _Pressure:
    if report is not None:
        current_tokens = max(0, report.estimated_input_tokens)
        usable = max(1, report.usable_input_tokens)
    else:
        usable = max(
            1,
            policy.model_context_tokens
            - policy.response_token_reserve
            - policy.safety_margin_tokens,
        )
        current_tokens = 0
    return _Pressure(
        current=current_tokens / usable,
        raw=(current_tokens + original_tokens) / usable,
        normalized=(current_tokens + normalized_tokens) / usable,
        usable_input_tokens=usable,
    )


def _should_compact(
    *,
    normalized: _NormalizedOutput,
    normalized_tokens: int,
    kind: ToolOutputKind,
    pressure: _Pressure,
    is_error: bool,
    compaction_policy: ToolOutputCompactionPolicy,
) -> bool:
    if normalized_tokens <= _small_output_limit(
        pressure.usable_input_tokens,
        compaction_policy,
    ):
        return False
    if is_error and normalized_tokens < _huge_output_limit(
        pressure.usable_input_tokens,
        compaction_policy,
    ):
        return False
    if pressure.normalized >= EMERGENCY_PRESSURE_RATIO:
        return True
    if _crosses_major_pressure_band(pressure.current, pressure.normalized):
        return True
    if normalized_tokens >= _huge_output_limit(
        pressure.usable_input_tokens,
        compaction_policy,
    ):
        return True
    if pressure.normalized >= CHECKPOINT_PRESSURE_RATIO:
        return True
    if pressure.normalized >= LOW_PRESSURE_RATIO and _is_large_noisy(
        kind,
        normalized,
        normalized_tokens,
        pressure.usable_input_tokens,
        compaction_policy,
    ):
        return True
    return False


def _is_large_noisy(
    kind: ToolOutputKind,
    normalized: _NormalizedOutput,
    tokens: int,
    usable_input_tokens: int,
    compaction_policy: ToolOutputCompactionPolicy,
) -> bool:
    if kind not in {
        ToolOutputKind.LOG,
        ToolOutputKind.BROWSER,
        ToolOutputKind.SEARCH,
        ToolOutputKind.TABLE,
        ToolOutputKind.PLAIN,
    }:
        return False
    if tokens < max(
        MIN_LARGE_NOISY_TOKENS,
        int(usable_input_tokens * compaction_policy.medium_output_ratio),
    ):
        return False
    text = normalized.content.lower()
    return (
        "repeated" in text
        or "passed" in text
        or "warning" in text
        or kind in {ToolOutputKind.SEARCH, ToolOutputKind.TABLE, ToolOutputKind.BROWSER}
    )


def _small_output_limit(
    usable_input_tokens: int,
    compaction_policy: ToolOutputCompactionPolicy,
) -> int:
    return max(
        1_000,
        int(max(1, usable_input_tokens) * compaction_policy.small_output_ratio),
    )


def _huge_output_limit(
    usable_input_tokens: int,
    compaction_policy: ToolOutputCompactionPolicy,
) -> int:
    return max(
        MIN_HUGE_TOKENS,
        int(max(1, usable_input_tokens) * compaction_policy.huge_output_ratio),
    )


def _compact_target_tokens(
    usable_input_tokens: int,
    compaction_policy: ToolOutputCompactionPolicy,
) -> int:
    return min(
        MAX_COMPACT_TARGET_TOKENS,
        max(
            MIN_COMPACT_TARGET_TOKENS,
            int(max(1, usable_input_tokens) * compaction_policy.compact_target_ratio),
        ),
    )


def _crosses_major_pressure_band(before: float, after: float) -> bool:
    return (
        before < CHECKPOINT_PRESSURE_RATIO <= after
        or before < EMERGENCY_PRESSURE_RATIO <= after
    )


def _compact_content(
    text: str,
    *,
    kind: ToolOutputKind,
    original_tokens: int,
    normalized_tokens: int,
    target_tokens: int,
    record: ToolOutputRecord,
    reason: str,
) -> str:
    if kind == ToolOutputKind.LOG:
        body = _compact_log(text)
    elif kind == ToolOutputKind.BROWSER:
        body = _compact_browser(text)
    elif kind == ToolOutputKind.JSON:
        body = _compact_json(text)
    elif kind == ToolOutputKind.SEARCH:
        body = _compact_search(text)
    elif kind == ToolOutputKind.DIFF:
        body = _compact_diff(text)
    elif kind == ToolOutputKind.FILE:
        body = _compact_file(text)
    elif kind == ToolOutputKind.TABLE:
        body = _compact_table(text)
    else:
        body = _compact_plain(text)
    header = (
        f"[tool output compacted: {kind.value}]\n"
        f"Original estimated tokens: {original_tokens}\n"
        f"Normalized estimated tokens: {normalized_tokens}\n"
        f"Compacted estimated tokens: <estimated-after-render>\n"
        f"Compression reason: {reason}\n"
        f"Retrieval ref: {record.ref}\n\n"
    )
    footer = (
        "\n\nNeed exact details:\n"
        f'Call retrieve_tool_output(ref="{record.ref}", query="specific text to recover").'
    )
    content = f"{header}{body}{footer}"
    if estimate_text_tokens(content) <= target_tokens:
        return content.replace(
            "Compacted estimated tokens: <estimated-after-render>",
            f"Compacted estimated tokens: {estimate_text_tokens(content)}",
        )
    bounded = _bounded_text(body, max(200, target_tokens - estimate_text_tokens(header + footer)))
    content = f"{header}{bounded}{footer}"
    return content.replace(
        "Compacted estimated tokens: <estimated-after-render>",
        f"Compacted estimated tokens: {estimate_text_tokens(content)}",
    )


def _compact_log(text: str) -> str:
    lines = text.splitlines()
    error_indexes = [
        index for index, line in enumerate(lines) if _ERROR_RE.search(line)
    ]
    windows = _line_windows(lines, error_indexes[:12], radius=3)
    tail = lines[-20:] if len(lines) > 20 else []
    passing = sum(1 for line in lines if "pass" in line.lower())
    warnings = sum(1 for line in lines if "warning" in line.lower())
    return _sections(
        (
            ("Summary", [f"lines={len(lines)}", f"error_windows={len(windows)}"]),
            ("Key details", windows or tail),
            (
                "Suppressed",
                [
                    f"passing-like lines={passing}",
                    f"warning-like lines={warnings}",
                    f"non-selected lines={max(0, len(lines) - len(windows) - len(tail))}",
                ],
            ),
        )
    )


def _compact_browser(text: str) -> str:
    lines = text.splitlines()
    refs = [line for line in lines if _BROWSER_REF_RE.search(line)]
    important = [
        line
        for line in lines
        if re.search(
            r"(title:|url:|button|link|textbox|input|error|warning|login|sign in|submit|checkout)",
            line,
            re.IGNORECASE,
        )
    ]
    head = lines[:30]
    tail = lines[-20:] if len(lines) > 30 else []
    selected = _dedupe_preserve_order([*important[:80], *refs[:80], *head, *tail])
    return _sections(
        (
            ("Summary", [f"lines={len(lines)}", f"interactive_or_ref_lines={len(refs)}"]),
            ("Key details", tuple(selected)),
            ("Suppressed", [f"non-selected browser lines={max(0, len(lines) - len(selected))}"]),
        )
    )


def _compact_json(text: str) -> str:
    parsed = _json_loads(text)
    if parsed is None:
        return _compact_plain(text)
    details: list[str] = []
    samples: list[str] = []
    if isinstance(parsed, dict):
        keys = [str(key) for key in parsed.keys()]
        details.append(f"top_level_keys={', '.join(keys[:24])}")
        for key, value in parsed.items():
            details.append(f"{key}: {_shape(value)}")
        samples.extend(_json_samples(parsed))
    elif isinstance(parsed, list):
        details.append(f"array_length={len(parsed)}")
        if parsed:
            details.append(f"item_shape={_shape(parsed[0])}")
        samples.extend(_json_samples(parsed))
    return _sections(
        (
            ("Summary", details[:30]),
            ("Key details", samples[:8]),
            ("Suppressed", ["full JSON values not shown in compacted view"]),
        )
    )


def _compact_search(text: str) -> str:
    file_counts: Counter[str] = Counter()
    samples: dict[str, list[str]] = {}
    for line in text.splitlines():
        match = _SEARCH_RE.match(line)
        if not match:
            continue
        prefix = match.group(0)
        path = prefix.split(":", 1)[0]
        file_counts[path] += 1
        samples.setdefault(path, [])
        if len(samples[path]) < 3:
            samples[path].append(line)
    top_files = [f"{path}: {count} matches" for path, count in file_counts.most_common(12)]
    detail_lines: list[str] = []
    for path, _ in file_counts.most_common(8):
        detail_lines.append(f"{path}:")
        detail_lines.extend(f"  {line}" for line in samples.get(path, ()))
    return _sections(
        (
            ("Summary", [f"files={len(file_counts)}", f"matches={sum(file_counts.values())}"]),
            ("Key details", (*top_files, *detail_lines)),
            ("Suppressed", ["lower-priority search matches omitted"]),
        )
    )


def _compact_diff(text: str) -> str:
    files: list[str] = []
    hunks: list[str] = []
    additions = 0
    deletions = 0
    for line in text.splitlines():
        if line.startswith("diff --git "):
            files.append(line)
        elif line.startswith("@@"):
            hunks.append(line)
        elif line.startswith("+") and not line.startswith("+++"):
            additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
    return _sections(
        (
            ("Summary", [f"changed_files={len(files)}", f"additions={additions}", f"deletions={deletions}"]),
            ("Key details", (*files[:20], *hunks[:30])),
            ("Suppressed", ["unchanged context lines and lower-priority hunks omitted"]),
        )
    )


def _compact_file(text: str) -> str:
    lines = text.splitlines()
    outline = [
        f"{index + 1}: {line.strip()}"
        for index, line in enumerate(lines)
        if line.startswith(("#", "class ", "def ", "import ", "from "))
        or line.strip().startswith(("function ", "const ", "let ", "export "))
    ]
    head = [f"{index + 1}: {line}" for index, line in enumerate(lines[:12])]
    tail_start = max(0, len(lines) - 12)
    tail = [f"{tail_start + index + 1}: {line}" for index, line in enumerate(lines[-12:])]
    return _sections(
        (
            ("Summary", [f"lines={len(lines)}"]),
            ("Key details", (*outline[:40], *head, *tail)),
            ("Suppressed", ["middle file content omitted"]),
        )
    )


def _compact_table(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    return _sections(
        (
            ("Summary", [f"rows={max(0, len(lines) - 1)}"]),
            ("Key details", lines[:15] + (["..."] if len(lines) > 30 else []) + lines[-10:]),
            ("Suppressed", [f"middle rows omitted={max(0, len(lines) - 25)}"]),
        )
    )


def _compact_plain(text: str) -> str:
    lines = text.splitlines()
    important = [
        line
        for line in lines
        if re.search(r"(error|warning|todo|decision|blocked|failed)", line, re.IGNORECASE)
    ]
    head = lines[:20]
    tail = lines[-15:] if len(lines) > 20 else []
    return _sections(
        (
            ("Summary", [f"lines={len(lines)}"]),
            ("Key details", important[:20] or (*head, *tail)),
            ("Suppressed", [f"non-selected lines={max(0, len(lines) - len(important[:20]) - len(head) - len(tail))}"]),
        )
    )


def _line_windows(lines: list[str], indexes: list[int], *, radius: int) -> list[str]:
    selected: list[str] = []
    seen: set[int] = set()
    for index in indexes:
        for line_index in range(max(0, index - radius), min(len(lines), index + radius + 1)):
            if line_index in seen:
                continue
            seen.add(line_index)
            selected.append(f"{line_index + 1}: {lines[line_index]}")
    return selected


def _sections(sections: tuple[tuple[str, list[str] | tuple[str, ...]], ...]) -> str:
    rendered: list[str] = []
    for title, lines in sections:
        clean_lines = [line for line in lines if str(line).strip()]
        if not clean_lines:
            continue
        rendered.append(f"{title}:")
        rendered.extend(f"- {line}" for line in clean_lines)
    return "\n".join(rendered)


def _json_samples(value: object) -> list[str]:
    samples: list[str] = []
    if isinstance(value, dict):
        for key, item in list(value.items())[:8]:
            if isinstance(item, (dict, list)):
                samples.append(f"{key}: {_shape(item)}")
            else:
                samples.append(f"{key}: {_short_json(item)}")
    elif isinstance(value, list):
        if value:
            samples.append(f"first: {_short_json(value[0])}")
        if len(value) > 1:
            samples.append(f"last: {_short_json(value[-1])}")
    return samples


def _shape(value: object) -> str:
    if isinstance(value, dict):
        return "object{" + ",".join(str(key) for key in list(value.keys())[:12]) + "}"
    if isinstance(value, list):
        item_shape = _shape(value[0]) if value else "empty"
        return f"array[{len(value)}] of {item_shape}"
    return type(value).__name__


def _short_json(value: object, limit: int = 240) -> str:
    text = _json_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _bounded_text(text: str, max_tokens: int) -> str:
    if estimate_text_tokens(text) <= max_tokens:
        return text
    lines = text.splitlines()
    if len(lines) <= 2:
        return text[: max(1, int(len(text) * max_tokens / max(1, estimate_text_tokens(text))))]
    keep_head = min(80, max(1, len(lines) // 3))
    keep_tail = min(40, max(1, len(lines) // 4))
    candidate = "\n".join(
        (
            *lines[:keep_head],
            f"[... compact body bounded; omitted {max(0, len(lines) - keep_head - keep_tail)} line(s) ...]",
            *lines[-keep_tail:],
        )
    )
    if estimate_text_tokens(candidate) <= max_tokens:
        return candidate
    low = 1
    high = 100
    best = "\n".join(
        (
            lines[0],
            f"[... compact body bounded; omitted {max(0, len(lines) - 2)} line(s) ...]",
            lines[-1],
        )
    )
    while low <= high:
        mid = (low + high) // 2
        head = max(1, keep_head * mid // 100)
        tail = max(1, keep_tail * mid // 100)
        candidate = "\n".join(
            (
                *lines[:head],
                f"[... compact body bounded; omitted {max(0, len(lines) - head - tail)} line(s) ...]",
                *lines[-tail:],
            )
        )
        if estimate_text_tokens(candidate) <= max_tokens:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best


def _compacted_data(data: object) -> dict[str, object]:
    if not isinstance(data, dict) or not data:
        return {}
    return {
        "tool_output_compacted": True,
        "structured_data_omitted": True,
        "data_keys": [str(key) for key in list(data.keys())[:24]],
    }


def _normalized_visible_content(
    normalized: _NormalizedOutput,
    *,
    record: ToolOutputRecord | None,
) -> str:
    if not normalized.steps:
        return normalized.content
    retrieval_note = (
        f"\nRaw output ref: {record.ref}\n"
        f'Need exact original bytes: call retrieve_tool_output(ref="{record.ref}").'
        if record is not None
        else ""
    )
    return (
        f"[tool output normalized: {','.join(normalized.steps)}]\n"
        f"{normalized.content}"
        f"{retrieval_note}"
    )


def _compaction_reason(
    pressure: _Pressure,
    kind: ToolOutputKind,
    normalized_tokens: int,
    compaction_policy: ToolOutputCompactionPolicy,
) -> str:
    if pressure.normalized >= EMERGENCY_PRESSURE_RATIO:
        return "next_pressure_crosses_75_percent"
    if pressure.current < CHECKPOINT_PRESSURE_RATIO <= pressure.normalized:
        return "next_pressure_crosses_50_percent"
    if normalized_tokens >= _huge_output_limit(
        pressure.usable_input_tokens,
        compaction_policy,
    ):
        return "huge_tool_output"
    if kind in {ToolOutputKind.LOG, ToolOutputKind.SEARCH, ToolOutputKind.TABLE}:
        return "large_noisy_tool_output"
    return "context_pressure"


def _merge_refs(current: tuple[str, ...], additions: tuple[str, ...]) -> tuple[str, ...]:
    refs: list[str] = []
    seen: set[str] = set()
    for ref in (*current, *additions):
        if not ref or ref in seen:
            continue
        refs.append(ref)
        seen.add(ref)
    return tuple(refs)


def _bounded_ratio(value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, parsed))


def _json_loads(text: str) -> object | None:
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except (TypeError, ValueError):
        return None


def _json_text(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return str(value)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0
