"""Error domain objects."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ErrorInfo:
    """Serializable summary of a failure observed by an outer module."""

    code: str
    message: str
    refs: tuple[str, ...] = field(default_factory=tuple)

    def is_ready(self) -> bool:
        """Return whether the error has enough detail to record or display."""
        return bool(self.code.strip() and self.message.strip())
