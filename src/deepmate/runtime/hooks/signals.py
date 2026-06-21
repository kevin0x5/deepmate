"""Bounded hook signal storage for memory and evolution consumers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from deepmate.storage import JsonlWriter

from .types import HookEnvelope, utc_now_iso

HOOK_SIGNALS_FILE = "signals.jsonl"
MAX_SIGNAL_SUMMARY_CHARS = 500
MAX_SIGNAL_REF_CHARS = 240
MAX_SIGNAL_REFS = 20
MAX_RECENT_SIGNALS = 200
SIGNAL_TYPES = frozenset({"memory", "evolution"})


@dataclass(frozen=True, slots=True)
class HookSignalRecord:
    """One sanitized hook signal candidate."""

    signal_id: str
    signal_type: str
    signal_kind: str
    summary: str
    refs: tuple[str, ...] = field(default_factory=tuple)
    hook_id: str = ""
    event_name: str = ""
    source_layer: str = ""
    source_actor: str = ""
    session_id: str = ""
    turn_id: str = ""
    step_id: str = ""
    task_id: str = ""
    branch_id: str = ""
    recorded_at: str = field(default_factory=utc_now_iso)

    def is_ready(self) -> bool:
        """Return whether this record has enough content to be useful."""
        return (
            bool(self.signal_id.strip())
            and self.signal_type in SIGNAL_TYPES
            and bool(self.signal_kind.strip())
            and bool(self.summary.strip())
        )

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable signal record."""
        return {
            "recorded_at": self.recorded_at,
            "signal_id": self.signal_id,
            "signal_type": self.signal_type,
            "signal_kind": self.signal_kind,
            "summary": self.summary,
            "refs": list(self.refs),
            "hook_id": self.hook_id,
            "event_name": self.event_name,
            "source_layer": self.source_layer,
            "source_actor": self.source_actor,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "step_id": self.step_id,
            "task_id": self.task_id,
            "branch_id": self.branch_id,
        }

    @classmethod
    def from_record(cls, record: object) -> "HookSignalRecord":
        """Build a signal record from a parsed JSON object."""
        if not isinstance(record, dict):
            return cls(
                signal_id="",
                signal_type="",
                signal_kind="",
                summary="",
            )
        return cls(
            recorded_at=_text(record.get("recorded_at")),
            signal_id=_text(record.get("signal_id")),
            signal_type=_text(record.get("signal_type")),
            signal_kind=_text(record.get("signal_kind")),
            summary=_bounded_text(record.get("summary"), MAX_SIGNAL_SUMMARY_CHARS),
            refs=_bounded_refs(record.get("refs")),
            hook_id=_text(record.get("hook_id")),
            event_name=_text(record.get("event_name")),
            source_layer=_text(record.get("source_layer")),
            source_actor=_text(record.get("source_actor")),
            session_id=_text(record.get("session_id")),
            turn_id=_text(record.get("turn_id")),
            step_id=_text(record.get("step_id")),
            task_id=_text(record.get("task_id")),
            branch_id=_text(record.get("branch_id")),
        )


class HookSignalStore:
    """Append-only local store for bounded memory/evolution hook signals."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def in_data_dir(cls, data_dir: str | Path) -> "HookSignalStore":
        """Return the default hook signal sink under Deepmate runtime data."""
        return cls(Path(data_dir) / "hooks" / HOOK_SIGNALS_FILE)

    def append(
        self,
        *,
        signal_type: str,
        signal_kind: str,
        summary: str,
        refs: tuple[str, ...] = (),
        hook_id: str = "",
        event_name: str = "",
        source_layer: str = "",
        source_actor: str = "",
        session_id: str = "",
        turn_id: str = "",
        step_id: str = "",
        task_id: str = "",
        branch_id: str = "",
    ) -> HookSignalRecord:
        """Append one sanitized signal and return the stored record."""
        clean_type = signal_type.strip().lower()
        if clean_type not in SIGNAL_TYPES:
            raise ValueError(f"unsupported hook signal type: {signal_type}")
        record = HookSignalRecord(
            signal_id=uuid4().hex,
            signal_type=clean_type,
            signal_kind=_bounded_text(signal_kind, 80) or "unspecified",
            summary=_bounded_text(summary, MAX_SIGNAL_SUMMARY_CHARS),
            refs=_bounded_refs(refs),
            hook_id=_bounded_text(hook_id, 120),
            event_name=_bounded_text(event_name, 120),
            source_layer=_bounded_text(source_layer, 80),
            source_actor=_bounded_text(source_actor, 80),
            session_id=_bounded_text(session_id, 120),
            turn_id=_bounded_text(turn_id, 120),
            step_id=_bounded_text(step_id, 120),
            task_id=_bounded_text(task_id, 120),
            branch_id=_bounded_text(branch_id, 120),
        )
        if not record.is_ready():
            raise ValueError("hook signal requires non-empty summary and kind")
        JsonlWriter(self.path).append(record.to_record())
        return record

    def append_from_envelope(
        self,
        *,
        signal_type: str,
        signal_kind: str,
        summary: str,
        refs: tuple[str, ...],
        hook_id: str,
        source_layer: str,
        envelope: HookEnvelope,
    ) -> HookSignalRecord:
        """Append a signal using envelope metadata."""
        return self.append(
            signal_type=signal_type,
            signal_kind=signal_kind,
            summary=summary,
            refs=refs,
            hook_id=hook_id,
            event_name=envelope.event_name.value,
            source_layer=source_layer,
            source_actor=envelope.actor.value,
            session_id=envelope.session_id,
            turn_id=envelope.turn_id,
            step_id=envelope.step_id,
            task_id=envelope.task_id,
            branch_id=envelope.branch_id,
        )

    def load_recent(self, limit: int = MAX_RECENT_SIGNALS) -> tuple[HookSignalRecord, ...]:
        """Load recent well-formed records from the signal file."""
        if not self.path.exists():
            return ()
        max_count = max(1, int(limit))
        records: list[HookSignalRecord] = []
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = HookSignalRecord.from_record(json.loads(line))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                if record.is_ready():
                    records.append(record)
        return tuple(records[-max_count:])


def _bounded_refs(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_items: tuple[object, ...] = (value,)
    elif isinstance(value, (list, tuple, set)):
        raw_items = tuple(value)
    else:
        raw_items = ()
    refs: list[str] = []
    for item in raw_items:
        text = _bounded_text(item, MAX_SIGNAL_REF_CHARS)
        if text:
            refs.append(text)
        if len(refs) >= MAX_SIGNAL_REFS:
            break
    return tuple(refs)


def _bounded_text(value: object, limit: int) -> str:
    text = _text(value)
    if not text:
        return ""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""
