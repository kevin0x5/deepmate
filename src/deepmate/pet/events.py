"""Desktop pet event contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any


class PetSeverity(StrEnum):
    """User-facing severity for pet events."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class PetVisualState(StrEnum):
    """Small visual state set used by the pixel pet renderer."""

    IDLE = "idle"
    THINKING = "thinking"
    WORKING = "working"
    WAITING = "waiting"
    REPORTING = "reporting"
    CELEBRATE = "celebrate"
    BLOCKED = "blocked"
    RESTING = "resting"
    OFFLINE = "offline"


@dataclass(frozen=True, slots=True)
class PetAction:
    """One user-facing action shown by the pet surface."""

    id: str
    label: str
    payload: Mapping[str, object] = field(default_factory=dict)

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable action record."""
        return {
            "id": self.id.strip(),
            "label": self.label.strip(),
            "payload": dict(self.payload),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "PetAction":
        """Build an action from a JSON-like record."""
        payload = record.get("payload", {})
        return cls(
            id=_text(record.get("id")),
            label=_text(record.get("label")),
            payload=payload if isinstance(payload, Mapping) else {},
        )


@dataclass(frozen=True, slots=True)
class PetEvent:
    """A short, scrubbed state event consumed by the desktop pet."""

    kind: str
    summary: str
    state: PetVisualState = PetVisualState.IDLE
    severity: PetSeverity = PetSeverity.INFO
    event_id: str = ""
    created_at: str = ""
    workspace: str = ""
    session_id: str = ""
    task_id: str = ""
    run_id: str = ""
    current_work_title: str = ""
    refs: tuple[str, ...] = field(default_factory=tuple)
    actions: tuple[PetAction, ...] = field(default_factory=tuple)

    def normalized(self) -> "PetEvent":
        """Return this event with required generated fields filled."""
        event_id = self.event_id.strip() or _event_id(self.kind)
        created_at = self.created_at.strip() or _now_iso()
        refs = tuple(_unique_texts(self.refs))
        actions = tuple(action for action in self.actions if action.id and action.label)
        return PetEvent(
            kind=self.kind.strip(),
            summary=self.summary.strip(),
            state=self.state,
            severity=self.severity,
            event_id=event_id,
            created_at=created_at,
            workspace=self.workspace.strip(),
            session_id=self.session_id.strip(),
            task_id=self.task_id.strip(),
            run_id=self.run_id.strip(),
            current_work_title=self.current_work_title.strip(),
            refs=refs,
            actions=actions,
        )

    def is_ready(self) -> bool:
        """Return whether the event has enough data to display or store."""
        event = self.normalized()
        return bool(event.kind and event.summary and event.event_id)

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable event record."""
        event = self.normalized()
        return {
            "event_id": event.event_id,
            "kind": event.kind,
            "severity": event.severity.value,
            "created_at": event.created_at,
            "workspace": event.workspace,
            "session_id": event.session_id,
            "task_id": event.task_id,
            "run_id": event.run_id,
            "current_work_title": event.current_work_title,
            "summary": event.summary,
            "state": event.state.value,
            "refs": list(event.refs),
            "actions": [action.to_record() for action in event.actions],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "PetEvent":
        """Build an event from a JSON-like record."""
        state = _enum_value(PetVisualState, record.get("state"), PetVisualState.IDLE)
        severity = _enum_value(PetSeverity, record.get("severity"), PetSeverity.INFO)
        actions = tuple(
            PetAction.from_record(item)
            for item in _mapping_items(record.get("actions"))
        )
        return cls(
            event_id=_text(record.get("event_id")),
            kind=_text(record.get("kind")),
            severity=severity,
            created_at=_text(record.get("created_at")),
            workspace=_text(record.get("workspace")),
            session_id=_text(record.get("session_id")),
            task_id=_text(record.get("task_id")),
            run_id=_text(record.get("run_id")),
            current_work_title=_text(record.get("current_work_title")),
            summary=_text(record.get("summary")),
            state=state,
            refs=tuple(_unique_texts(_string_items(record.get("refs")))),
            actions=actions,
        ).normalized()


def event_for_turn_started(
    *,
    workspace: str | Path,
    session_id: str,
    prompt: str,
    title: str = "",
) -> PetEvent:
    """Return a scrubbed current-work event for one user turn start."""
    clean_title = title.strip() or _compact(prompt, 80) or "Deepmate task"
    return PetEvent(
        kind="task.started",
        severity=PetSeverity.INFO,
        state=PetVisualState.WORKING,
        workspace=str(workspace),
        session_id=session_id,
        current_work_title=clean_title,
        summary=f"Started: {clean_title}",
        actions=(PetAction(id="open_current_work", label="Open current work"),),
    )


def event_for_turn_finished(
    *,
    workspace: str | Path,
    session_id: str,
    title: str,
    summary: str,
    failed: bool = False,
) -> PetEvent:
    """Return a scrubbed event for one completed or failed user turn."""
    return PetEvent(
        kind="task.failed" if failed else "task.completed",
        severity=PetSeverity.ERROR if failed else PetSeverity.INFO,
        state=PetVisualState.BLOCKED if failed else PetVisualState.CELEBRATE,
        workspace=str(workspace),
        session_id=session_id,
        current_work_title=title.strip() or "Deepmate task",
        summary=_compact(summary, 220)
        or ("Task needs attention." if failed else "Task completed."),
        actions=(PetAction(id="open_current_work", label="Open current work"),),
    )


def event_for_task_achievement(
    *,
    workspace: str | Path,
    session_id: str,
    title: str,
    summary: str,
    path: str | Path = "",
) -> PetEvent:
    """Return an event for a Task Mode stage achievement artifact."""
    clean_title = title.strip() or "Task achievement"
    refs = (f"path={path}",) if str(path).strip() else ()
    return PetEvent(
        kind="task.achievement",
        severity=PetSeverity.INFO,
        state=PetVisualState.CELEBRATE,
        workspace=str(workspace),
        session_id=session_id,
        current_work_title=clean_title,
        summary=_compact(summary, 220) or f"Created task achievement: {clean_title}.",
        refs=refs,
        actions=(PetAction(id="open_current_work", label="Open current work"),),
    )


def event_for_care_reminder(
    *,
    workspace: str | Path = "",
    session_id: str = "",
    title: str = "",
    summary: str = "",
) -> PetEvent:
    """Return a low-priority local care reminder event."""
    return PetEvent(
        kind="care.reminder",
        severity=PetSeverity.INFO,
        state=PetVisualState.RESTING,
        workspace=str(workspace),
        session_id=session_id,
        current_work_title=title.strip() or "Deepmate",
        summary=_compact(summary, 160) or "Take a short break before continuing.",
    )


def event_for_turn_progress(
    *,
    workspace: str | Path,
    session_id: str,
    title: str,
    summary: str,
) -> PetEvent:
    """Return a low-priority progress event for a running turn."""
    return PetEvent(
        kind="task.progress",
        severity=PetSeverity.INFO,
        state=PetVisualState.REPORTING,
        workspace=str(workspace),
        session_id=session_id,
        current_work_title=title.strip() or "Deepmate task",
        summary=_compact(summary, 180) or "Deepmate is making progress.",
        actions=(PetAction(id="open_current_work", label="Open current work"),),
    )


def event_for_turn_waiting(
    *,
    workspace: str | Path,
    session_id: str,
    title: str,
    summary: str,
) -> PetEvent:
    """Return a high-priority waiting event for approval or user attention."""
    return PetEvent(
        kind="task.waiting",
        severity=PetSeverity.WARNING,
        state=PetVisualState.WAITING,
        workspace=str(workspace),
        session_id=session_id,
        current_work_title=title.strip() or "Deepmate task",
        summary=_compact(summary, 180) or "Deepmate is waiting for you.",
        actions=(PetAction(id="open_current_work", label="Open current work"),),
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _event_id(kind: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    clean = "".join(ch if ch.isalnum() else "_" for ch in kind.strip())[:40]
    return f"pet_{clean or 'event'}_{stamp}"


def _compact(value: object, limit: int) -> str:
    text = " ".join(_text(value).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_items(value: object) -> tuple[str, ...]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _mapping_items(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _unique_texts(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = value.strip()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return tuple(result)


def _enum_value(enum_type: type[Any], value: object, fallback: Any) -> Any:
    if isinstance(value, str):
        try:
            return enum_type(value.strip())
        except ValueError:
            return fallback
    return fallback
