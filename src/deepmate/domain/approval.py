"""Approval domain objects."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """Request for user approval before a high-impact action runs."""

    action: str
    reason: str
    refs: tuple[str, ...] = field(default_factory=tuple)

    def is_ready(self) -> bool:
        """Return whether the approval request is clear enough to show."""
        return bool(self.action.strip() and self.reason.strip())
