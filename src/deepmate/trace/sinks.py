"""Trace sinks."""

from __future__ import annotations

from pathlib import Path

from deepmate.storage import JsonlWriter
from deepmate.trace.schema import TraceEvent, TraceSpan


class JsonlTraceSink:
    """Persist trace events and spans to a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.writer = JsonlWriter(path)

    def write(self, record: TraceEvent | TraceSpan) -> None:
        """Write one trace record."""
        self.writer.append(record.to_record())
