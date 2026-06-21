"""Running follow-up support for active agent turns."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from uuid import uuid4

from deepmate.domain import Message, MessageRole


@dataclass(frozen=True, slots=True)
class TurnFollowupMessage:
    """One user follow-up submitted while a turn is running."""

    text: str
    source: str = "user"

    def is_ready(self) -> bool:
        """Return whether the follow-up can be injected."""
        return bool(self.text.strip())

    def to_message(self) -> Message:
        """Return the model-facing user message."""
        clean = self.text.strip()
        return Message(
            role=MessageRole.USER,
            content=(
                "User follow-up while this turn was running:\n"
                f"{clean}"
            ),
        )


@dataclass(slots=True)
class TurnFollowupBuffer:
    """Thread-safe follow-up buffer scoped to one active user turn."""

    active_turn_id: str | None = None
    _pending: list[TurnFollowupMessage] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock)

    def start_turn(self) -> str:
        """Start a new turn that can accept running follow-ups."""
        with self._lock:
            self.active_turn_id = uuid4().hex
            self._pending.clear()
            return self.active_turn_id

    def finish_turn(self, turn_id: str | None) -> tuple[TurnFollowupMessage, ...]:
        """Finish an active turn and return unconsumed follow-ups."""
        with self._lock:
            if turn_id is None or turn_id != self.active_turn_id:
                return ()
            pending = tuple(self._pending)
            self._pending.clear()
            self.active_turn_id = None
            return pending

    def submit(
        self,
        turn_id: str | None,
        text: str,
        *,
        source: str = "user",
    ) -> bool:
        """Submit a follow-up for the expected active turn."""
        clean = text.strip()
        if not clean:
            return False
        with self._lock:
            if turn_id is None or turn_id != self.active_turn_id:
                return False
            self._pending.append(TurnFollowupMessage(text=clean, source=source))
            return True

    def drain(self, turn_id: str | None) -> tuple[TurnFollowupMessage, ...]:
        """Return and clear pending follow-ups for the expected active turn."""
        with self._lock:
            if turn_id is None or turn_id != self.active_turn_id:
                return ()
            pending = tuple(self._pending)
            self._pending.clear()
            return pending

    def pending_count(self) -> int:
        """Return the current pending follow-up count."""
        with self._lock:
            return len(self._pending)
