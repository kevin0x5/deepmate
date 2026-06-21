"""Trace event and span schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import time_ns
from typing import Mapping
from uuid import uuid4


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _unix_nano_now() -> int:
    return time_ns()


def new_trace_id() -> str:
    """Return a W3C-compatible 16-byte trace id as lowercase hex."""
    return uuid4().hex


def new_span_id() -> str:
    """Return a W3C-compatible 8-byte span id as lowercase hex."""
    return uuid4().hex[:16]


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """Observable fact about a runtime, model, tool, or storage operation."""

    kind: str
    summary: str
    refs: tuple[str, ...] = field(default_factory=tuple)
    recorded_at: str = field(default_factory=_utc_now_iso)

    def is_ready(self) -> bool:
        """Return whether the trace event is meaningful enough to record."""
        return bool(self.kind.strip() and self.summary.strip())

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable record."""
        return {
            "recorded_at": self.recorded_at,
            "kind": self.kind.strip(),
            "summary": self.summary.strip(),
            "refs": list(self.refs),
        }


@dataclass(frozen=True, slots=True)
class TraceSpan:
    """Duration-bearing operation that can be exported as an OTel span."""

    name: str
    kind: str
    trace_id: str
    span_id: str
    parent_span_id: str = ""
    started_at_unix_nano: int = field(default_factory=_unix_nano_now)
    ended_at_unix_nano: int = 0
    status: str = "UNSET"
    attributes: Mapping[str, object] = field(default_factory=dict)
    events: tuple[TraceEvent, ...] = field(default_factory=tuple)

    def is_ready(self) -> bool:
        """Return whether this span has enough structure to persist/export."""
        return bool(
            self.name.strip()
            and self.kind.strip()
            and _is_hex(self.trace_id, 32)
            and _is_hex(self.span_id, 16)
            and (not self.parent_span_id or _is_hex(self.parent_span_id, 16))
        )

    def is_complete(self) -> bool:
        """Return whether this span has a valid completed time range."""
        return (
            self.started_at_unix_nano > 0
            and self.ended_at_unix_nano >= self.started_at_unix_nano
        )

    def finish(
        self,
        *,
        status: str | None = None,
        attributes: Mapping[str, object] | None = None,
        events: tuple[TraceEvent, ...] | None = None,
        ended_at_unix_nano: int | None = None,
    ) -> "TraceSpan":
        """Return a completed span, preserving existing attributes."""
        merged_attributes = dict(self.attributes)
        if attributes:
            merged_attributes.update(attributes)
        return TraceSpan(
            name=self.name,
            kind=self.kind,
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            started_at_unix_nano=self.started_at_unix_nano,
            ended_at_unix_nano=ended_at_unix_nano or _unix_nano_now(),
            status=(status or self.status).strip().upper() or "UNSET",
            attributes=merged_attributes,
            events=events if events is not None else self.events,
        )

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable span record."""
        return {
            "record_type": "span",
            "name": self.name.strip(),
            "kind": self.kind.strip().upper(),
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "started_at_unix_nano": self.started_at_unix_nano,
            "ended_at_unix_nano": self.ended_at_unix_nano,
            "status": self.status.strip().upper() or "UNSET",
            "attributes": dict(self.attributes),
            "events": [event.to_record() for event in self.events],
        }


def _is_hex(value: str, length: int) -> bool:
    if len(value) != length:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
