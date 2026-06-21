"""Runtime event domain objects."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    """Append-only fact emitted while runtime work moves forward."""

    kind: str
    summary: str
    refs: tuple[str, ...] = field(default_factory=tuple)

    def is_ready(self) -> bool:
        """Return whether the event is meaningful enough to record."""
        return bool(self.kind.strip() and self.summary.strip())
