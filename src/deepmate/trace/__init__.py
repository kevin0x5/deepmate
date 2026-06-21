"""Trace recording primitives for Deepmate."""

from deepmate.trace.recorder import TraceRecorder, TraceSink
from deepmate.trace.schema import TraceEvent, TraceSpan, new_span_id, new_trace_id
from deepmate.trace.otel import build_otlp_traces_payload
from deepmate.trace.exporter import OtlpExportResult, export_otlp_traces
from deepmate.trace.semantic import (
    TraceUsageSummary,
    summarize_trace_usage,
    trace_record_kind,
    trace_record_matches_kinds,
    trace_record_matches_session,
    trace_record_refs,
    trace_refs_to_map,
    trace_semantic_attributes,
)
from deepmate.trace.sinks import JsonlTraceSink

__all__ = [
    "JsonlTraceSink",
    "OtlpExportResult",
    "TraceEvent",
    "TraceRecorder",
    "TraceSink",
    "TraceSpan",
    "build_otlp_traces_payload",
    "export_otlp_traces",
    "new_span_id",
    "new_trace_id",
    "TraceUsageSummary",
    "summarize_trace_usage",
    "trace_record_kind",
    "trace_record_matches_kinds",
    "trace_record_matches_session",
    "trace_record_refs",
    "trace_refs_to_map",
    "trace_semantic_attributes",
]
