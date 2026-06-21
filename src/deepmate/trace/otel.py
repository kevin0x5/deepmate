"""OpenTelemetry OTLP JSON helpers for Deepmate spans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from deepmate.trace.schema import TraceSpan

OTEL_SPAN_KIND_INTERNAL = 1
OTEL_SPAN_KIND_CLIENT = 3
OTEL_STATUS_UNSET = 0
OTEL_STATUS_OK = 1
OTEL_STATUS_ERROR = 2


def build_otlp_traces_payload(
    spans: Sequence[TraceSpan],
    *,
    service_name: str = "deepmate",
    service_version: str = "",
) -> dict[str, object]:
    """Return an OTLP HTTP/JSON traces payload for the given spans.

    The output follows OTLP JSON protobuf encoding: field names are lower camel
    case, enum values are integers, and trace/span ids are hex strings.
    """
    ready_spans = tuple(span for span in spans if span.is_ready() and span.is_complete())
    if not ready_spans:
        return {"resourceSpans": []}
    resource_attributes: list[dict[str, object]] = [
        _attribute("service.name", service_name),
    ]
    if service_version.strip():
        resource_attributes.append(_attribute("service.version", service_version.strip()))
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": resource_attributes},
                "scopeSpans": [
                    {
                        "scope": {"name": "deepmate.trace"},
                        "spans": [_span_to_otlp(span) for span in ready_spans],
                    }
                ],
            }
        ]
    }


def _span_to_otlp(span: TraceSpan) -> dict[str, object]:
    attributes = _span_attributes(span)
    record: dict[str, object] = {
        "traceId": span.trace_id,
        "spanId": span.span_id,
        "name": span.name.strip(),
        "kind": _span_kind(span.kind),
        "startTimeUnixNano": str(span.started_at_unix_nano),
        "endTimeUnixNano": str(span.ended_at_unix_nano or span.started_at_unix_nano),
        "attributes": [_attribute(key, value) for key, value in attributes.items()],
        "status": _status(span.status, attributes),
    }
    if span.parent_span_id:
        record["parentSpanId"] = span.parent_span_id
    if span.events:
        record["events"] = [
            {
                "timeUnixNano": str(span.started_at_unix_nano),
                "name": event.kind.strip(),
                "attributes": [
                    _attribute("deepmate.event.summary", event.summary.strip()),
                    *(_attribute("deepmate.event.ref", ref) for ref in event.refs),
                ],
            }
            for event in span.events
            if event.is_ready()
        ]
    return record


def _attributes_with_langfuse_session(
    attributes: Mapping[str, object],
) -> dict[str, object]:
    values = {str(key): value for key, value in attributes.items() if str(key).strip()}
    session_id = _first_text(
        values,
        "session.id",
        "langfuse.session.id",
        "gen_ai.conversation.id",
        "deepmate.session_id",
    )
    if session_id:
        values.setdefault("session.id", session_id)
        values.setdefault("gen_ai.conversation.id", session_id)
    trace_name = _first_text(values, "langfuse.trace.name", "deepmate.turn.title")
    if trace_name:
        values.setdefault("langfuse.trace.name", trace_name)
    return values


def _span_attributes(span: TraceSpan) -> dict[str, object]:
    values = _attributes_with_langfuse_session(span.attributes)
    values.setdefault("gen_ai.operation.name", _operation_name(span))
    return values


def _operation_name(span: TraceSpan) -> str:
    explicit = _first_text(span.attributes, "gen_ai.operation.name")
    if explicit:
        return explicit
    name = span.name.strip().lower()
    if name.startswith("chat"):
        return "chat"
    if name.startswith("execute_tool"):
        return "execute_tool"
    if name.startswith("invoke_workflow"):
        return "invoke_workflow"
    if name.startswith("invoke_agent") or name.startswith("deepmate turn"):
        return "invoke_agent"
    return "invoke_agent"


def _first_text(values: Mapping[str, object], *keys: str) -> str:
    for key in keys:
        value = values.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _attribute(key: str, value: object) -> dict[str, object]:
    return {"key": key, "value": _any_value(value)}


def _any_value(value: object) -> dict[str, object]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [_any_value(item) for item in value]}}
    return {"stringValue": str(value)}


def _span_kind(kind: str) -> int:
    clean = kind.strip().upper()
    if clean == "CLIENT":
        return OTEL_SPAN_KIND_CLIENT
    return OTEL_SPAN_KIND_INTERNAL


def _status(status: str, attributes: Mapping[str, object]) -> dict[str, object]:
    clean = status.strip().upper()
    message = str(attributes.get("error.message", "")).strip()
    if clean == "ERROR":
        result: dict[str, object] = {"code": OTEL_STATUS_ERROR}
        if message:
            result["message"] = message
        return result
    if clean == "OK":
        return {"code": OTEL_STATUS_OK}
    return {"code": OTEL_STATUS_UNSET}
