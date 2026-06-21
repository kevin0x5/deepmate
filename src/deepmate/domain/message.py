"""Message domain objects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class MessageRole(StrEnum):
    """Stable roles used in Deepmate conversations."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    """Text message shared across channels, context, and providers."""

    role: MessageRole
    content: str

    def is_ready(self) -> bool:
        """Return whether the message has content worth sending or storing."""
        return bool(self.content.strip())
