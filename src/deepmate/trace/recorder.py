"""Trace recorder."""

from __future__ import annotations

from contextvars import ContextVar
from contextlib import contextmanager
from collections.abc import Mapping
from typing import Protocol

from deepmate.trace.schema import TraceEvent, TraceSpan, new_span_id, new_trace_id


class TraceSink(Protocol):
    """Minimal sink interface used by the trace recorder."""

    def write(self, record: TraceEvent | TraceSpan) -> None:
        """Persist or forward one trace record."""
        ...


class TraceRecorder:
    """Facade for recording trace events and lightweight spans."""

    def __init__(self, sink: TraceSink) -> None:
        self.sink = sink
        self._span_stack: ContextVar[tuple[TraceSpan, ...]] = ContextVar(
            "deepmate_trace_span_stack",
            default=(),
        )

    def record(self, event: TraceEvent) -> None:
        """Record an event if it has enough information."""
        if event.is_ready():
            self.sink.write(event)

    def record_span(self, span: TraceSpan) -> None:
        """Record a span if it has enough structure."""
        if span.is_ready():
            self.sink.write(span)

    @contextmanager
    def start_span(
        self,
        name: str,
        *,
        kind: str = "INTERNAL",
        attributes: dict[str, object] | None = None,
        trace_id: str = "",
        parent_span_id: str | None = None,
    ) -> object:
        """Start a local span and write it when the context exits.

        This is intentionally small: it tracks parent/child relationships inside
        one process and does not require OpenTelemetry SDK state.
        """
        stack = self._span_stack.get()
        parent = stack[-1] if stack else None
        span = TraceSpan(
            name=name,
            kind=kind,
            trace_id=trace_id or (parent.trace_id if parent is not None else new_trace_id()),
            span_id=new_span_id(),
            parent_span_id=(
                parent_span_id
                if parent_span_id is not None
                else (parent.span_id if parent is not None else "")
            ),
            attributes=attributes or {},
        )
        scope = TraceSpanScope(span)
        token = self._span_stack.set((*stack, span))
        try:
            yield scope
        except Exception as exc:
            scope.set_status("ERROR")
            scope.set_attribute("error.type", type(exc).__name__)
            scope.set_attribute("error.message", str(exc))
            raise
        finally:
            self._span_stack.reset(token)
            self.record_span(scope.finish())


class TraceSpanScope:
    """Mutable scope for filling attributes before a frozen span is recorded."""

    def __init__(self, span: TraceSpan) -> None:
        self._span = span
        self._status = "OK"
        self._attributes = dict(span.attributes)

    @property
    def span(self) -> TraceSpan:
        """Return the initial span identity for child linkage."""
        return self._span

    def set_attribute(self, key: str, value: object) -> None:
        """Set one span attribute when the key is non-empty."""
        clean = key.strip()
        if clean:
            self._attributes[clean] = value

    def set_attributes(self, attributes: Mapping[str, object]) -> None:
        """Merge multiple span attributes."""
        for key, value in attributes.items():
            self.set_attribute(str(key), value)

    def set_status(self, status: str) -> None:
        """Set the final span status."""
        clean = status.strip().upper()
        if clean:
            self._status = clean

    def finish(self) -> TraceSpan:
        """Return the completed immutable span."""
        return self._span.finish(status=self._status, attributes=self._attributes)
