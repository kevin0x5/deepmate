"""Provider request and response objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from deepmate.domain import Message
from deepmate.providers.usage import TokenUsage


class ProviderError(RuntimeError):
    """Base error raised by model providers."""


class NetworkError(ProviderError):
    """Transient network error raised before receiving a provider response."""


class AuthError(ProviderError):
    """Non-retryable provider authentication or authorization error."""


class RateLimitError(ProviderError):
    """Provider rate limit error that may succeed after waiting."""

    def __init__(self, message: str, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class ServerError(ProviderError):
    """Transient provider-side server error."""


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Provider/model features used to shape one request safely."""

    supports_tools: bool = True
    supports_thinking: bool = True
    supports_stream_usage: bool = True
    supports_assistant_reasoning_replay: bool = True
    supports_image_input: bool = True
    sanitize_request: bool = True

    def normalized(self) -> "ModelCapabilities":
        """Return a capability object with plain bool fields."""
        return ModelCapabilities(
            supports_tools=bool(self.supports_tools),
            supports_thinking=bool(self.supports_thinking),
            supports_stream_usage=bool(self.supports_stream_usage),
            supports_assistant_reasoning_replay=bool(
                self.supports_assistant_reasoning_replay
            ),
            supports_image_input=bool(self.supports_image_input),
            sanitize_request=bool(self.sanitize_request),
        )


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """Provider-neutral request sent to a model adapter."""

    model: str
    conversation: tuple["ModelConversationItem", ...]
    tool_schemas: tuple[Mapping[str, object], ...] = field(default_factory=tuple)
    options: Mapping[str, object] = field(default_factory=dict)
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)

    def is_ready(self) -> bool:
        """Return whether the request has a target model and conversation."""
        return bool(_ready_text(self.model) and self.conversation)


@dataclass(frozen=True, slots=True)
class ModelToolRequest:
    """Provider-neutral request from a model to call a tool."""

    name: str
    arguments: Mapping[str, object] = field(default_factory=dict)
    id: str = ""
    raw_arguments: str = ""
    argument_error: str = ""

    def is_ready(self) -> bool:
        """Return whether the tool call has a callable name and id."""
        return bool(_ready_text(self.name) and _ready_text(self.id))


@dataclass(frozen=True, slots=True)
class ModelToolResult:
    """Provider-neutral result returned after a requested tool call."""

    name: str
    content: str = ""
    request_id: str = ""
    data: Mapping[str, object] = field(default_factory=dict)
    refs: tuple[str, ...] = field(default_factory=tuple)
    attachments: tuple[Mapping[str, object], ...] = field(default_factory=tuple)
    is_error: bool = False

    def is_ready(self) -> bool:
        """Return whether the tool result can be sent back to a model."""
        has_result = (
            _ready_text(self.content) or self.data or self.refs or self.is_error
            or self.attachments
        )
        return bool(
            _ready_text(self.name) and _ready_text(self.request_id) and has_result
        )


@dataclass(frozen=True, slots=True)
class ModelToolExchange:
    """Assistant tool calls paired with runtime tool results for replay."""

    assistant_content: str = ""
    assistant_reasoning: str = ""
    tool_requests: tuple[ModelToolRequest, ...] = field(default_factory=tuple)
    tool_results: tuple[ModelToolResult, ...] = field(default_factory=tuple)

    def is_ready(self) -> bool:
        """Return whether the exchange has enough data for tool replay."""
        request_ids = tuple(_ready_text(request.id) for request in self.tool_requests)
        result_ids = tuple(
            _ready_text(result.request_id) for result in self.tool_results
        )
        return bool(
            self.tool_requests
            and self.tool_results
            and all(request.is_ready() for request in self.tool_requests)
            and all(result.is_ready() for result in self.tool_results)
            and all(request_ids)
            and all(result_ids)
            and len(set(request_ids)) == len(request_ids)
            and len(set(result_ids)) == len(result_ids)
            and set(request_ids) == set(result_ids)
        )


@dataclass(frozen=True, slots=True)
class ModelConversationItem:
    """One ordered model-facing conversation item."""

    message: Message | None = None
    tool_exchange: ModelToolExchange | None = None

    @classmethod
    def from_message(cls, message: Message) -> "ModelConversationItem":
        """Build a conversation item from one text message."""
        return cls(message=message)

    @classmethod
    def from_tool_exchange(
        cls,
        exchange: ModelToolExchange,
    ) -> "ModelConversationItem":
        """Build a conversation item from one assistant/tool exchange."""
        return cls(tool_exchange=exchange)

    def is_ready(self) -> bool:
        """Return whether exactly one ready item payload is present."""
        has_message = self.message is not None
        has_exchange = self.tool_exchange is not None
        if has_message == has_exchange:
            return False
        if self.message is not None:
            return self.message.is_ready()
        if self.tool_exchange is not None:
            return self.tool_exchange.is_ready()
        return False


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Provider-neutral model response returned to runtime."""

    content: str = ""
    tool_requests: tuple[ModelToolRequest, ...] = field(default_factory=tuple)
    usage: TokenUsage | None = None
    reasoning: str = ""
    finish_reason: str = ""

    def has_output(self) -> bool:
        """Return whether the response contains text, reasoning, or tool requests."""
        return bool(
            _ready_text(self.content)
            or _ready_text(self.reasoning)
            or self.tool_requests
        )


@dataclass(frozen=True, slots=True)
class StreamDelta:
    """One incremental fragment emitted while a streamed response is produced.

    Only the visible text channels are streamed (assistant content and
    reasoning); tool-call fragments are accumulated inside the provider and
    surface in the final ``ModelResponse``. Either field may be empty.
    """

    content: str = ""
    reasoning: str = ""

    def is_empty(self) -> bool:
        return not self.content and not self.reasoning


def _ready_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()
