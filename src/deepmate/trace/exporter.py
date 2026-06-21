"""Minimal OTLP HTTP/JSON trace exporter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from deepmate.trace.otel import build_otlp_traces_payload
from deepmate.trace.schema import TraceSpan


@dataclass(frozen=True, slots=True)
class OtlpExportResult:
    """Result of one explicit OTLP trace export attempt."""

    endpoint: str
    spans_seen: int
    spans_exported: int
    status_code: int = 0
    message: str = ""

    def is_success(self) -> bool:
        """Return whether the endpoint accepted the payload."""
        return 200 <= self.status_code < 300


def export_otlp_traces(
    spans: tuple[TraceSpan, ...],
    *,
    endpoint: str,
    headers: tuple[tuple[str, str], ...] = (),
    service_name: str = "deepmate",
    service_version: str = "",
    timeout_seconds: float = 10,
) -> OtlpExportResult:
    """Send completed spans to an OTLP HTTP/JSON traces endpoint."""
    clean_endpoint = _traces_endpoint(endpoint)
    payload = build_otlp_traces_payload(
        spans,
        service_name=service_name,
        service_version=service_version,
    )
    spans_exported = _payload_span_count(payload)
    if spans_exported == 0:
        return OtlpExportResult(
            endpoint=clean_endpoint,
            spans_seen=len(spans),
            spans_exported=0,
            message="no complete spans to export",
        )
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request_headers = {
        "Content-Type": "application/json",
        "User-Agent": "Deepmate OTLP Exporter",
    }
    request_headers.update({key: value for key, value in headers if key.strip() and value.strip()})
    request = Request(clean_endpoint, data=body, headers=request_headers, method="POST")
    with urlopen(request, timeout=timeout_seconds) as response:
        status = int(getattr(response, "status", 0) or response.getcode() or 0)
        message = str(getattr(response, "reason", "") or "").strip()
    return OtlpExportResult(
        endpoint=clean_endpoint,
        spans_seen=len(spans),
        spans_exported=spans_exported,
        status_code=status,
        message=message,
    )


def _traces_endpoint(endpoint: str) -> str:
    clean = endpoint.strip().rstrip("/")
    parsed = urlparse(clean)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("OTLP endpoint must be an http:// or https:// URL")
    if parsed.path.endswith("/v1/traces"):
        return clean
    return f"{clean}/v1/traces"


def _payload_span_count(payload: dict[str, object]) -> int:
    count = 0
    for resource in _list(payload.get("resourceSpans")):
        for scope in _list(resource.get("scopeSpans")):
            count += len(_list(scope.get("spans")))
    return count


def _list(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))
