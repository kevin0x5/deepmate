"""Helpers for interpreting lightweight trace records.

This module keeps Deepmate's JSONL trace small while making the fields easier
to map to OpenTelemetry GenAI attributes later.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

MODEL_EVENT_KINDS = {
    "model_request_prefix_fingerprint",
    "model_request_started",
    "model_response_received",
    "model_request_failed",
}
SESSION_REF_KEYS = {
    "session_id",
    "parent_session_id",
    "current_session_id",
    "from_session_id",
    "to_session_id",
}


@dataclass(frozen=True, slots=True)
class TraceUsageSummary:
    """Token usage summarized from model response trace records."""

    model_response_events: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_hit_input_tokens: int = 0
    cache_miss_input_tokens: int = 0

    def is_empty(self) -> bool:
        """Return whether this summary contains no model usage signal."""
        return self.model_response_events == 0

    def cache_hit_ratio(self) -> float | None:
        """Return cache hit ratio over input tokens when available."""
        if self.input_tokens <= 0:
            return None
        return self.cache_hit_input_tokens / self.input_tokens


def trace_record_kind(record: Mapping[str, object]) -> str:
    """Return a normalized trace record kind."""
    return str(record.get("kind", "")).strip()


def trace_record_refs(record: Mapping[str, object]) -> tuple[str, ...]:
    """Return normalized refs from a JSON-decoded trace record."""
    refs = record.get("refs")
    if not isinstance(refs, list):
        return ()
    return tuple(str(ref).strip() for ref in refs if str(ref).strip())


def trace_refs_to_map(refs: Sequence[str]) -> dict[str, str]:
    """Parse `key=value` refs into a small lookup map."""
    values: dict[str, str] = {}
    for ref in refs:
        key, separator, value = ref.partition("=")
        key = key.strip()
        value = value.strip()
        if separator and key and value:
            values[key] = value
    return values


def trace_record_matches_session(
    record: Mapping[str, object],
    session_id: str,
) -> bool:
    """Return whether a trace record belongs to a session or child session."""
    clean_session_id = session_id.strip()
    if not clean_session_id:
        return False
    refs = trace_refs_to_map(trace_record_refs(record))
    if any(refs.get(key) == clean_session_id for key in SESSION_REF_KEYS):
        return True
    attributes = record.get("attributes")
    if not isinstance(attributes, Mapping):
        return False
    return any(
        str(attributes.get(key, "")).strip() == clean_session_id
        for key in (
            "session.id",
            "langfuse.session.id",
            "gen_ai.conversation.id",
            *(f"deepmate.{key}" for key in SESSION_REF_KEYS),
        )
    )


def trace_record_matches_kinds(
    record: Mapping[str, object],
    kinds: Sequence[str],
) -> bool:
    """Return whether a trace record matches an optional exact kind filter."""
    selected = {kind.strip() for kind in kinds if kind.strip()}
    if not selected:
        return True
    kind = trace_record_kind(record)
    if kind in selected:
        return True
    if str(record.get("record_type", "")).strip() == "span":
        return "span" in selected or str(record.get("name", "")).strip() in selected
    return False


def trace_semantic_attributes(
    kind: str,
    refs: Sequence[str] | Mapping[str, str],
) -> dict[str, object]:
    """Return attributes that can later feed an OTEL/OpenLLMetry exporter.

    Only fields that clearly match OpenTelemetry GenAI semantics use `gen_ai.*`.
    Provider-specific or uncertain fields stay under `deepmate.*`.
    """
    ref_map = dict(refs) if isinstance(refs, Mapping) else trace_refs_to_map(refs)
    clean_kind = kind.strip()
    attributes: dict[str, object] = {"deepmate.trace.kind": clean_kind}

    if clean_kind in MODEL_EVENT_KINDS:
        attributes["gen_ai.operation.name"] = "chat"
        _copy_string(ref_map, attributes, "model", "gen_ai.request.model")
        _copy_string(ref_map, attributes, "provider", "gen_ai.provider.name")
        if clean_kind == "model_response_received":
            _copy_string(ref_map, attributes, "model", "gen_ai.response.model")
            _copy_int(ref_map, attributes, "input_tokens", "gen_ai.usage.input_tokens")
            _copy_int(
                ref_map,
                attributes,
                "output_tokens",
                "gen_ai.usage.output_tokens",
            )

    _copy_int(
        ref_map,
        attributes,
        "reasoning_tokens",
        "deepmate.usage.reasoning_tokens",
    )
    _copy_int(
        ref_map,
        attributes,
        "cache_hit_input_tokens",
        "deepmate.usage.cache_hit_input_tokens",
    )
    _copy_int(
        ref_map,
        attributes,
        "cache_miss_input_tokens",
        "deepmate.usage.cache_miss_input_tokens",
    )
    _copy_string(ref_map, attributes, "finish_reason", "deepmate.finish_reason")
    _copy_string(ref_map, attributes, "tool_source", "deepmate.tool.source")
    _copy_string(ref_map, attributes, "mcp_tool", "deepmate.mcp.tool")
    _copy_string(ref_map, attributes, "subagent_run_id", "deepmate.subagent.run_id")
    for key in SESSION_REF_KEYS:
        _copy_string(ref_map, attributes, key, f"deepmate.{key}")
    return attributes


def summarize_trace_usage(
    records: Sequence[Mapping[str, object]],
) -> TraceUsageSummary:
    """Summarize model token usage from trace records."""
    model_events = 0
    input_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0
    cache_hit = 0
    cache_miss = 0
    for record in records:
        if trace_record_kind(record) != "model_response_received":
            continue
        refs = trace_refs_to_map(trace_record_refs(record))
        model_events += 1
        input_tokens += _int_ref(refs, "input_tokens")
        output_tokens += _int_ref(refs, "output_tokens")
        reasoning_tokens += _int_ref(refs, "reasoning_tokens")
        cache_hit += _int_ref(refs, "cache_hit_input_tokens")
        cache_miss += _int_ref(refs, "cache_miss_input_tokens")
    return TraceUsageSummary(
        model_response_events=model_events,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cache_hit_input_tokens=cache_hit,
        cache_miss_input_tokens=cache_miss,
    )


def _copy_string(
    refs: Mapping[str, str],
    attributes: dict[str, object],
    ref_key: str,
    attribute_key: str,
) -> None:
    value = refs.get(ref_key, "").strip()
    if value:
        attributes[attribute_key] = value


def _copy_int(
    refs: Mapping[str, str],
    attributes: dict[str, object],
    ref_key: str,
    attribute_key: str,
) -> None:
    value = _int_ref(refs, ref_key)
    if value > 0:
        attributes[attribute_key] = value


def _int_ref(refs: Mapping[str, str], key: str) -> int:
    try:
        return int(refs.get(key, "0"))
    except ValueError:
        return 0
