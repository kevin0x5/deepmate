"""Display restraint policy for desktop pet events."""

from __future__ import annotations

from dataclasses import dataclass

from deepmate.pet.events import PetEvent


@dataclass(frozen=True, slots=True)
class PetDisplayDecision:
    """Decision for rendering one pet event."""

    show_bubble: bool
    priority: int
    reason: str
    hold: bool = False


@dataclass(frozen=True, slots=True)
class PetDisplayPolicy:
    """Small deterministic display policy for pet events."""

    def decide(self, event: PetEvent) -> PetDisplayDecision:
        """Return whether an event should show a text bubble."""
        kind = event.kind.strip()
        if kind in {"task.failed", "task.waiting"}:
            return PetDisplayDecision(True, 100, "requires_attention", hold=True)
        if kind in {"task.completed", "task.achievement"}:
            return PetDisplayDecision(True, 90, "completed", hold=True)
        if kind == "task.progress":
            return PetDisplayDecision(True, 60, "important_progress")
        if kind.startswith("learning."):
            return PetDisplayDecision(True, 40, "learning_suggestion")
        if kind.startswith("care."):
            return PetDisplayDecision(True, 30, "proactive_care")
        if kind.startswith("maintenance."):
            return PetDisplayDecision(False, 20, "background_maintenance")
        return PetDisplayDecision(False, 10, "animation_only")
