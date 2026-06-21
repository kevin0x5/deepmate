"""Memory domain objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class MemorySource(StrEnum):
    """Known provenance types for long-term memory entries."""

    USER_DECLARED = "user_declared"
    USER_CORRECTED = "user_corrected"
    INFERRED = "inferred"


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """Source-aware long-term memory entry or candidate."""

    content: str
    source: MemorySource
    refs: tuple[str, ...] = field(default_factory=tuple)

    def is_ready(self) -> bool:
        """Return whether the memory entry has content worth reviewing."""
        return bool(self.content.strip())

    def is_user_confirmed(self) -> bool:
        """Return whether the entry came from explicit user input."""
        return self.source in {
            MemorySource.USER_DECLARED,
            MemorySource.USER_CORRECTED,
        }
