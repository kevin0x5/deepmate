"""Provider boundary types for Deepmate."""

from deepmate.providers.base import ModelProvider
from deepmate.providers.messages import (
    AuthError,
    ModelCapabilities,
    ModelConversationItem,
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
from deepmate.providers.chat_completions import ChatCompletionsProvider
from deepmate.providers.usage import TokenUsage

__all__ = [
    "AuthError",
    "ChatCompletionsProvider",
    "ModelCapabilities",
    "ModelConversationItem",
    "ModelProvider",
    "ModelRequest",
    "ModelResponse",
    "ModelToolExchange",
    "ModelToolRequest",
    "ModelToolResult",
    "NetworkError",
    "ProviderError",
    "RateLimitError",
    "ServerError",
    "StreamDelta",
    "TokenUsage",
]
