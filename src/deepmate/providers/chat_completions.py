"""Chat completions provider for compatible HTTP APIs."""

from __future__ import annotations

import base64
import mimetypes
import json
import re
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from deepmate.providers.messages import (
    AuthError,
    ModelConversationItem,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelToolExchange,
    ModelToolRequest,
    ModelToolResult,
    NetworkError,
    ProviderError,
    RateLimitError,
    ServerError,
    StreamDelta,
)
from deepmate.providers.usage import TokenUsage

CHAT_COMPLETIONS_PATH = "/chat/completions"
MAX_MODEL_RESPONSE_BYTES = 10 * 1024 * 1024
MAX_MODEL_STREAM_LINE_BYTES = 2 * 1024 * 1024
MAX_MODEL_STREAM_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _ParsedArguments:
    arguments: Mapping[str, object]
    error: str = ""


class ChatCompletionsProvider:
    """Synchronous provider for HTTP chat completions APIs.

    The first concrete target is DeepSeek V4 through its OpenAI-compatible
    `/chat/completions` endpoint, but the provider name stays protocol-level
    instead of vendor-level so future adapters can coexist cleanly.
    """

    def __init__(self, base_url: str, api_key: str, timeout: int = 60) -> None:
        if not base_url.strip():
            raise ValueError("base_url is required")
        if not api_key.strip():
            raise ValueError("api_key is required")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Send one chat completion request and return a normalized response."""
        request = sanitize_model_request(request)
        if not request.is_ready():
            raise ValueError("model request requires a model and conversation")
        payload = _request_payload(request)
        raw = _post_json(
            url=f"{self._base_url}{CHAT_COMPLETIONS_PATH}",
            api_key=self._api_key,
            payload=payload,
            timeout=self._timeout,
        )
        return _parse_response(raw)

    def complete_stream(
        self,
        request: ModelRequest,
        on_delta: Callable[[StreamDelta], None],
    ) -> ModelResponse:
        """Stream one chat completion, calling ``on_delta`` per text fragment.

        Returns the fully accumulated ``ModelResponse`` (identical in shape to
        ``complete``) so callers stay agnostic to how it was produced. Tool-call
        fragments are accumulated internally and surface in the return value;
        only assistant content and reasoning are streamed to ``on_delta``.

        Connection-level failures are raised before any delta is emitted, so a
        caller may safely retry. A failure mid-stream (after deltas were emitted)
        is raised as a terminal ``ProviderError`` to avoid duplicate output on
        retry.
        """
        request = sanitize_model_request(request)
        if not request.is_ready():
            raise ValueError("model request requires a model and conversation")
        payload = dict(_request_payload(request))
        payload["stream"] = True
        if request.capabilities.normalized().supports_stream_usage:
            payload["stream_options"] = {"include_usage": True}
        accumulator = _StreamAccumulator()
        _stream_sse(
            url=f"{self._base_url}{CHAT_COMPLETIONS_PATH}",
            api_key=self._api_key,
            payload=payload,
            timeout=self._timeout,
            on_event=lambda event: accumulator.consume(event, on_delta),
        )
        return accumulator.finalize()


def sanitize_model_request(request: ModelRequest) -> ModelRequest:
    """Return a provider-boundary-safe request without mutating transcript state."""
    capabilities = request.capabilities.normalized()
    if not capabilities.sanitize_request:
        return replace(request, capabilities=capabilities)

    conversation = tuple(
        item
        for item in (
            _sanitize_conversation_item(item, capabilities)
            for item in request.conversation
        )
        if item is not None
    )
    options = _sanitize_options(request.options, capabilities)
    tool_schemas = request.tool_schemas if capabilities.supports_tools else ()
    return ModelRequest(
        model=request.model,
        conversation=conversation,
        tool_schemas=tuple(tool_schemas),
        options=options,
        capabilities=capabilities,
    )


def _sanitize_conversation_item(
    item: ModelConversationItem,
    capabilities: ModelCapabilities,
) -> ModelConversationItem | None:
    if item.message is not None:
        return item if item.message.is_ready() else None
    if item.tool_exchange is not None:
        if not capabilities.supports_tools or not item.tool_exchange.is_ready():
            return None
        if capabilities.supports_assistant_reasoning_replay:
            return item
        exchange = item.tool_exchange
        if not exchange.assistant_reasoning:
            return item
        return ModelConversationItem.from_tool_exchange(
            ModelToolExchange(
                assistant_content=exchange.assistant_content,
                assistant_reasoning="",
                tool_requests=exchange.tool_requests,
                tool_results=exchange.tool_results,
            )
        )
    return None


def _sanitize_options(
    options: Mapping[str, object],
    capabilities: ModelCapabilities,
) -> Mapping[str, object]:
    clean = dict(options)
    if not capabilities.supports_thinking:
        clean.pop("thinking", None)
        clean.pop("reasoning_effort", None)
    if clean.get("stream") is None:
        clean.pop("stream", None)
    if not capabilities.supports_stream_usage:
        clean.pop("stream_options", None)
    return clean


def _request_payload(request: ModelRequest) -> Mapping[str, object]:
    payload: dict[str, object] = {
        "model": request.model,
        "messages": _request_messages(request),
    }
    if request.tool_schemas:
        payload["tools"] = _tool_schema_payloads(request.tool_schemas)

    for key, value in request.options.items():
        if key == "stream" and value:
            raise ValueError(
                "set streaming via complete_stream(), not request options"
            )
        payload[key] = value
    return payload


def _request_messages(request: ModelRequest) -> list[Mapping[str, object]]:
    return [
        message
        for item in request.conversation
        for message in _conversation_item_messages(
            item,
            supports_image_input=request.capabilities.supports_image_input,
        )
    ]


def _conversation_item_messages(
    item: ModelConversationItem,
    *,
    supports_image_input: bool = True,
) -> list[Mapping[str, object]]:
    if item.message is not None:
        return [_message_payload(item.message)]
    if item.tool_exchange is not None:
        return _tool_exchange_messages(
            item.tool_exchange,
            supports_image_input=supports_image_input,
        )
    raise ValueError("conversation item requires message or tool exchange")


def _message_payload(message: Message) -> Mapping[str, object]:
    return {"role": message.role.value, "content": message.content}


def _tool_exchange_messages(
    exchange: ModelToolExchange,
    *,
    supports_image_input: bool = True,
) -> list[Mapping[str, object]]:
    requests_by_id = _tool_requests_by_id(exchange)
    results_by_id = _tool_results_by_id(exchange, requests_by_id)
    if set(requests_by_id) != set(results_by_id):
        raise ValueError("tool exchange results must match tool request ids")

    assistant_message: dict[str, object] = {
        "role": "assistant",
        "content": exchange.assistant_content,
        "tool_calls": [
            _tool_call_payload(request) for request in exchange.tool_requests
        ],
    }
    if exchange.assistant_reasoning:
        assistant_message["reasoning_content"] = exchange.assistant_reasoning

    messages: list[Mapping[str, object]] = [assistant_message]
    attachment_messages: list[Mapping[str, object]] = []
    for request in exchange.tool_requests:
        result = results_by_id[request.id.strip()]
        messages.append(
            {
                "role": "tool",
                "tool_call_id": request.id.strip(),
                "content": _tool_result_content(result),
            }
        )
        if supports_image_input:
            attachment_messages.extend(_tool_result_attachment_messages(result))
    messages.extend(attachment_messages)
    return messages


def _tool_requests_by_id(
    exchange: ModelToolExchange,
) -> Mapping[str, ModelToolRequest]:
    if not exchange.tool_requests:
        raise ValueError("tool exchange requires tool requests")

    requests: dict[str, ModelToolRequest] = {}
    for request in exchange.tool_requests:
        if not request.is_ready():
            raise ValueError("tool exchange contains an invalid tool request")
        request_id = request.id.strip()
        if not request_id:
            raise ValueError("tool request id is required for tool replay")
        if request_id in requests:
            raise ValueError(f"duplicate tool request id: {request_id}")
        requests[request_id] = request
    return requests


def _tool_results_by_id(
    exchange: ModelToolExchange,
    requests_by_id: Mapping[str, ModelToolRequest],
) -> Mapping[str, ModelToolResult]:
    if not exchange.tool_results:
        raise ValueError("tool exchange requires tool results")

    results: dict[str, ModelToolResult] = {}
    for result in exchange.tool_results:
        if not result.is_ready():
            raise ValueError("tool exchange contains an invalid tool result")
        request_id = result.request_id.strip()
        if not request_id:
            raise ValueError("tool result request_id is required for tool replay")
        if request_id not in requests_by_id:
            raise ValueError(f"tool result has no matching request id: {request_id}")
        if request_id in results:
            raise ValueError(f"duplicate tool result request_id: {request_id}")
        results[request_id] = result
    return results


def _tool_call_payload(request: ModelToolRequest) -> Mapping[str, object]:
    return {
        "id": request.id.strip(),
        "type": "function",
        "function": {
            "name": _provider_tool_name(request.name.strip()),
            "arguments": _tool_call_arguments(request),
        },
    }


def _tool_call_arguments(request: ModelToolRequest) -> str:
    if request.raw_arguments.strip():
        return request.raw_arguments
    try:
        return json.dumps(request.arguments, ensure_ascii=False, separators=(",", ":"))
    except TypeError as exc:
        raise ValueError(f"tool call arguments must be JSON serializable: {exc}") from exc


def _tool_result_content(result: ModelToolResult) -> str:
    if result.content:
        return result.content

    payload: dict[str, object] = {}
    if result.data:
        payload["data"] = result.data
    if result.refs:
        payload["refs"] = list(result.refs)
    if result.is_error:
        payload["is_error"] = True
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _tool_result_attachment_messages(
    result: ModelToolResult,
) -> list[Mapping[str, object]]:
    image_parts = _tool_result_image_parts(result)
    if not image_parts:
        return []
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Visual attachment from tool result "
                        f"{result.name} ({result.request_id})."
                    ),
                },
                *image_parts,
            ],
        }
    ]


def _tool_result_image_parts(result: ModelToolResult) -> list[Mapping[str, object]]:
    parts: list[Mapping[str, object]] = []
    for attachment in result.attachments:
        if not isinstance(attachment, Mapping):
            continue
        if str(attachment.get("type") or "").strip().lower() != "image":
            continue
        path = str(attachment.get("path") or "").strip()
        if not path:
            continue
        data_url = _image_data_url(Path(path), attachment)
        if not data_url:
            continue
        parts.append({"type": "image_url", "image_url": {"url": data_url}})
    return parts


def _image_data_url(path: Path, attachment: Mapping[str, object]) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    if not data:
        return ""
    mime_type = str(attachment.get("mime_type") or "").strip()
    if not mime_type:
        guessed, _encoding = mimetypes.guess_type(path.name)
        mime_type = guessed or "image/png"
    if not mime_type.startswith("image/"):
        return ""
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _tool_schema_payload(schema: Mapping[str, object]) -> Mapping[str, object]:
    if schema.get("type") == "function":
        return dict(schema)

    name = schema.get("name")
    description = schema.get("description")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("tool schema requires name")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("tool schema requires description")

    parameters = schema.get("input_schema")
    if parameters is None:
        parameters = _empty_object_schema()
    return {
        "type": "function",
        "function": {
            "name": _provider_tool_name(name.strip()),
            "description": description.strip(),
            "parameters": parameters,
        },
    }


def _tool_schema_payloads(
    schemas: tuple[Mapping[str, object], ...],
) -> list[Mapping[str, object]]:
    payloads = [_tool_schema_payload(schema) for schema in schemas]
    names: dict[str, int] = {}
    for index, payload in enumerate(payloads):
        name = _tool_payload_name(payload)
        if not name:
            raise ValueError("tool schema requires function name")
        if name in names:
            raise ValueError(f"duplicate provider tool name after encoding: {name}")
        names[name] = index
    return payloads


def _tool_payload_name(payload: Mapping[str, object]) -> str:
    function = payload.get("function")
    if not isinstance(function, Mapping):
        return ""
    name = function.get("name")
    return name.strip() if isinstance(name, str) else ""


def _post_json(
    url: str,
    api_key: str,
    payload: Mapping[str, object],
    timeout: int,
) -> Mapping[str, object]:
    body = json.dumps(payload).encode("utf-8")
    http_request = Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(http_request, timeout=timeout) as response:
            status_code = response.getcode()
            response_body = _read_response_text(response, MAX_MODEL_RESPONSE_BYTES)
    except HTTPError as exc:
        detail = _read_error_detail(exc)
        raise _http_error(exc.code, detail, headers=exc.headers) from exc
    except URLError as exc:
        raise NetworkError(f"model request failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise NetworkError(
            "model request timed out while waiting for response data"
        ) from exc
    if status_code not in {200, 201}:
        raise _http_error(status_code, _truncate_utf8_text(response_body, 800))

    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError as exc:
        snippet = response_body[:800]
        raise ProviderError(f"model response must be valid JSON: {snippet}") from exc
    if not isinstance(parsed, Mapping):
        raise ProviderError("model response must be a JSON object")
    return parsed


def _stream_sse(
    url: str,
    api_key: str,
    payload: Mapping[str, object],
    timeout: int,
    on_event: Callable[[Mapping[str, object]], None],
) -> None:
    """Open an SSE chat-completions stream and dispatch each JSON data event.

    Errors raised before the first event are connection-level and retryable;
    a parse/transport failure after streaming has begun is terminal.
    """
    body = json.dumps(payload).encode("utf-8")
    http_request = Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    try:
        response = urlopen(http_request, timeout=timeout)
    except HTTPError as exc:
        detail = _read_error_detail(exc)
        raise _http_error(exc.code, detail, headers=exc.headers) from exc
    except URLError as exc:
        raise NetworkError(f"model request failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise NetworkError("model request timed out before the stream opened") from exc

    with closing(response):
        status_code = response.getcode()
        if status_code not in {200, 201}:
            detail = _truncate_utf8_text(
                _read_response_text(response, MAX_MODEL_RESPONSE_BYTES),
                800,
            )
            raise _http_error(
                status_code,
                detail,
                headers=getattr(response, "headers", None),
            )

        try:
            total_bytes = 0
            for raw_line in _iter_sse_lines(response):
                total_bytes += len(raw_line)
                if total_bytes > MAX_MODEL_STREAM_BYTES:
                    raise ProviderError("model stream response is too large")
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    # Tolerate keep-alive/comment noise rather than aborting a
                    # stream that may have already produced visible output.
                    continue
                if isinstance(event, Mapping):
                    on_event(event)
        except (URLError, TimeoutError, OSError) as exc:
            raise ProviderError(
                f"model stream interrupted after partial output: {exc}"
            ) from exc


def _read_response_text(response: object, max_bytes: int) -> str:
    data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ProviderError(f"model response exceeds {max_bytes} bytes")
    return data.decode("utf-8", errors="replace")


def _read_error_detail(error: HTTPError) -> str:
    data = error.read(801)
    if len(data) > 800:
        data = data[:800]
        while data:
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError as exc:
                if exc.reason != "unexpected end of data":
                    break
                data = data[:-1]
    return data.decode("utf-8", errors="replace")


def _iter_sse_lines(response: object):
    readline = getattr(response, "readline", None)
    if callable(readline):
        while True:
            raw_line = readline(MAX_MODEL_STREAM_LINE_BYTES + 1)
            if not raw_line:
                break
            if len(raw_line) > MAX_MODEL_STREAM_LINE_BYTES:
                raise ProviderError(
                    f"model stream line exceeds {MAX_MODEL_STREAM_LINE_BYTES} bytes"
                )
            yield raw_line
        return
    for raw_line in response:
        if len(raw_line) > MAX_MODEL_STREAM_LINE_BYTES:
            raise ProviderError(
                f"model stream line exceeds {MAX_MODEL_STREAM_LINE_BYTES} bytes"
            )
        yield raw_line


class _StreamAccumulator:
    """Assemble streamed SSE delta chunks into one ModelResponse."""

    def __init__(self) -> None:
        self._content: list[str] = []
        self._reasoning: list[str] = []
        self._finish_reason = ""
        self._usage: Mapping[str, object] | None = None
        # Tool calls arrive fragmented by index: id/name in the first fragment,
        # argument strings concatenated across later fragments.
        self._tool_calls: dict[int, dict[str, str]] = {}
        self._last_tool_call_index: int | None = None

    def consume(
        self,
        event: Mapping[str, object],
        on_delta: Callable[[StreamDelta], None],
    ) -> None:
        usage = event.get("usage")
        if isinstance(usage, Mapping):
            self._usage = usage
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            return
        choice = choices[0]
        if not isinstance(choice, Mapping):
            return
        finish = choice.get("finish_reason")
        if isinstance(finish, str) and finish:
            self._finish_reason = finish
        delta = choice.get("delta")
        if not isinstance(delta, Mapping):
            return

        content = delta.get("content")
        reasoning = delta.get("reasoning_content")
        content_text = content if isinstance(content, str) else ""
        reasoning_text = reasoning if isinstance(reasoning, str) else ""
        if content_text:
            self._content.append(content_text)
        if reasoning_text:
            self._reasoning.append(reasoning_text)
        if content_text or reasoning_text:
            on_delta(StreamDelta(content=content_text, reasoning=reasoning_text))

        self._consume_tool_calls(delta.get("tool_calls"))

    def _consume_tool_calls(self, value: object) -> None:
        if not isinstance(value, list):
            return
        for item in value:
            if not isinstance(item, Mapping):
                continue
            index = item.get("index")
            if not isinstance(index, int):
                index = self._no_index_tool_call_slot(item)
            self._last_tool_call_index = index
            slot = self._tool_calls.setdefault(
                index, {"id": "", "name": "", "arguments": ""}
            )
            call_id = item.get("id")
            if isinstance(call_id, str) and call_id:
                slot["id"] = call_id
            function = item.get("function")
            if isinstance(function, Mapping):
                name = function.get("name")
                if isinstance(name, str) and name:
                    slot["name"] += name
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    slot["arguments"] += arguments

    def _no_index_tool_call_slot(self, item: Mapping[str, object]) -> int:
        function = item.get("function")
        name = ""
        if isinstance(function, Mapping):
            raw_name = function.get("name")
            name = raw_name if isinstance(raw_name, str) else ""
        if self._last_tool_call_index is None:
            return len(self._tool_calls)
        last_slot = self._tool_calls.get(self._last_tool_call_index)
        if name and last_slot is not None and last_slot["name"]:
            return max(self._tool_calls, default=-1) + 1
        return self._last_tool_call_index

    def finalize(self) -> ModelResponse:
        return ModelResponse(
            content="".join(self._content),
            reasoning="".join(self._reasoning),
            tool_requests=self._finalize_tool_requests(),
            usage=_parse_usage(self._usage),
            finish_reason=self._finish_reason,
        )

    def _finalize_tool_requests(self) -> tuple[ModelToolRequest, ...]:
        requests: list[ModelToolRequest] = []
        for _index, slot in sorted(self._tool_calls.items()):
            if not slot["name"].strip():
                continue
            parsed = _parse_arguments(slot["arguments"])
            requests.append(
                ModelToolRequest(
                    name=_runtime_tool_name(slot["name"]),
                    arguments=parsed.arguments,
                    id=slot["id"],
                    raw_arguments=slot["arguments"],
                    argument_error=parsed.error,
                )
            )
        return tuple(requests)


def _http_error(
    status_code: int,
    detail: str,
    *,
    headers: Mapping[str, str] | None = None,
) -> ProviderError:
    message = f"model request failed with HTTP {status_code}: {_redact_error_detail(detail)}"
    if status_code in {401, 403}:
        return AuthError(message)
    if status_code == 429:
        return RateLimitError(message, retry_after_seconds=_retry_after_seconds(headers))
    if 500 <= status_code <= 599:
        return ServerError(message)
    return ProviderError(message)


def _retry_after_seconds(headers: Mapping[str, str] | None) -> float | None:
    if headers is None:
        return None
    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    if not isinstance(retry_after, str):
        return None
    value = retry_after.strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())


def _truncate_utf8_text(text: str, limit: int) -> str:
    if len(text.encode("utf-8")) <= limit:
        return text
    data = text.encode("utf-8")[:limit]
    while data:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            if exc.reason != "unexpected end of data":
                break
            data = data[:-1]
    return data.decode("utf-8", errors="replace")


def _redact_error_detail(detail: str) -> str:
    text = detail.strip()
    if not text:
        return ""
    redacted = re.sub(
        r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+",
        "Bearer [redacted]",
        text,
    )
    redacted = re.sub(
        r"\bsk-[A-Za-z0-9_-]{12,}\b",
        "sk-[redacted]",
        redacted,
    )
    redacted = re.sub(
        r"(?i)(api[_-]?key|access[_-]?token|refresh[_-]?token|secret|password)"
        r'(\s*[:=]\s*["\']?)[^"\'\s,}]+',
        r"\1\2[redacted]",
        redacted,
    )
    redacted = re.sub(
        r"(?i)(authorization\s*[:=]\s*)([^,\n}]+)",
        _redact_authorization_match,
        redacted,
    )
    return redacted[:800]


def _redact_authorization_match(match: re.Match[str]) -> str:
    prefix = match.group(1)
    value = match.group(2).strip()
    if value.lower() == "bearer [redacted]":
        return f"{prefix}Bearer [redacted]"
    return f"{prefix}[redacted]"


def _parse_response(payload: Mapping[str, object]) -> ModelResponse:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderError("model response has no choices")

    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        raise ProviderError("model response choice must be an object")
    message = first_choice.get("message")
    if not isinstance(message, Mapping):
        raise ProviderError("model response choice has no message")
    finish_reason = first_choice.get("finish_reason") or ""
    if not isinstance(finish_reason, str):
        finish_reason = str(finish_reason)

    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    if not isinstance(reasoning, str):
        reasoning = json.dumps(reasoning, ensure_ascii=False)

    return ModelResponse(
        content=content,
        reasoning=reasoning,
        tool_requests=_parse_tool_requests(message.get("tool_calls")),
        usage=_parse_usage(payload.get("usage")),
        finish_reason=finish_reason,
    )


def _parse_tool_requests(value: object) -> tuple[ModelToolRequest, ...]:
    if not isinstance(value, list):
        return ()

    requests: list[ModelToolRequest] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        function = item.get("function")
        if not isinstance(function, Mapping):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        parsed_arguments = _parse_arguments(function.get("arguments"))
        requests.append(
            ModelToolRequest(
                name=_runtime_tool_name(name),
                arguments=parsed_arguments.arguments,
                id=str(item.get("id") or ""),
                raw_arguments=_raw_arguments(function.get("arguments")),
                argument_error=parsed_arguments.error,
            )
        )
    return tuple(requests)


def _parse_arguments(value: object) -> _ParsedArguments:
    if isinstance(value, Mapping):
        return _ParsedArguments(arguments=dict(value))
    if not isinstance(value, str) or not value.strip():
        return _ParsedArguments(arguments={})
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        return _ParsedArguments(
            arguments={},
            error=f"tool arguments must be valid JSON: {exc.msg}",
        )
    if isinstance(parsed, Mapping):
        return _ParsedArguments(arguments=dict(parsed))
    return _ParsedArguments(
        arguments={},
        error="tool arguments JSON must decode to an object",
    )


def _raw_arguments(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            return ""
    return ""


def _parse_usage(value: object) -> TokenUsage | None:
    if not isinstance(value, Mapping):
        return None
    details = value.get("completion_tokens_details")
    detail_map = details if isinstance(details, Mapping) else {}
    return TokenUsage(
        input_tokens=_int_field(value, "prompt_tokens"),
        output_tokens=_int_field(value, "completion_tokens"),
        cache_hit_input_tokens=_int_field(value, "prompt_cache_hit_tokens"),
        cache_miss_input_tokens=_int_field(value, "prompt_cache_miss_tokens"),
        reasoning_tokens=_int_field(detail_map, "reasoning_tokens"),
    )


def _int_field(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _empty_object_schema() -> Mapping[str, object]:
    return {"type": "object", "properties": {}}


def _provider_tool_name(name: str) -> str:
    if "." not in name:
        if _looks_like_encoded_mcp_tool_name(name):
            raise ValueError(f"tool name uses reserved MCP provider namespace: {name}")
        return name
    return "mcp__" + name.replace(".", "__")


def _runtime_tool_name(name: str) -> str:
    if not name.startswith("mcp__"):
        return name
    parts = name[len("mcp__") :].split("__", 1)
    if len(parts) != 2 or not all(part.strip() for part in parts):
        return name
    return f"{parts[0]}.{parts[1]}"


def _looks_like_encoded_mcp_tool_name(name: str) -> bool:
    return name.startswith("mcp__") and "__" in name[len("mcp__") :]
