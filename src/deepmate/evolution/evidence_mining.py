"""Deterministic evidence aggregation for self-evolution.

This module keeps the learning trigger cheap and auditable. It does not ask an
LLM to decide what is worth learning; it only groups explicit evidence and
returns candidates once fixed thresholds are met.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from deepmate.foundation import compact_text, normalize_name

DEFAULT_USER_CORRECTION_THRESHOLD = 2
DEFAULT_TOOL_FAILURE_THRESHOLD = 3
DEFAULT_WORKFLOW_THRESHOLD = 2


@dataclass(frozen=True, slots=True)
class ToolFailureEvidence:
    """One observed tool failure."""

    tool_name: str
    error_signature: str
    source_ref: str

    def signature(self) -> str:
        """Return the same-tool same-error grouping signature."""
        return evidence_signature(self.tool_name, self.error_signature)


@dataclass(frozen=True, slots=True)
class UserCorrectionEvidence:
    """One user correction that indicates a repeated failure mode."""

    signature: str
    correction: str
    source_ref: str


@dataclass(frozen=True, slots=True)
class WorkflowEvidence:
    """One successful workflow that may become a generated skill."""

    signature: str
    name: str
    description: str
    steps: tuple[str, ...]
    source_ref: str
    reference_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvidenceAggregate:
    """Evidence grouped by normalized signature."""

    kind: str
    signature: str
    count: int
    source_refs: tuple[str, ...]
    examples: tuple[str, ...] = ()

    def is_ready(self) -> bool:
        """Return whether this aggregate has enough data to act on."""
        return bool(self.kind.strip() and self.signature.strip() and self.count > 0)


@dataclass(frozen=True, slots=True)
class WorkflowAggregate:
    """Repeated successful workflow candidate."""

    signature: str
    name: str
    description: str
    steps: tuple[str, ...]
    count: int
    source_refs: tuple[str, ...]
    reference_paths: tuple[str, ...] = ()

    def is_ready(self) -> bool:
        """Return whether this aggregate can be rendered as a skill draft."""
        return bool(
            self.signature.strip()
            and self.name.strip()
            and self.description.strip()
            and self.steps
            and self.count > 0
        )


@dataclass(frozen=True, slots=True)
class EvolutionEvidenceBatch:
    """Structured evidence collected from explicit trace-like records."""

    tool_failures: tuple[ToolFailureEvidence, ...] = ()
    user_corrections: tuple[UserCorrectionEvidence, ...] = ()
    workflows: tuple[WorkflowEvidence, ...] = ()

    def is_empty(self) -> bool:
        """Return whether the batch has no usable evidence."""
        return not (self.tool_failures or self.user_corrections or self.workflows)


def evidence_signature(*parts: str) -> str:
    """Return a compact normalized evidence signature."""
    text = " ".join(part for part in parts if part.strip())
    return compact_text(normalize_name(text), 160)


def collect_evidence_from_records(
    records: Iterable[Mapping[str, object]],
) -> EvolutionEvidenceBatch:
    """Collect explicit evolution evidence from trace/session/daily-like records.

    The collector intentionally accepts only clear fields or refs such as
    `tool_name=...`, `error_signature=...`, `signature=...`, and `step=...`.
    It does not infer failure modes from arbitrary prose.
    """
    tool_failures: list[ToolFailureEvidence] = []
    user_corrections: list[UserCorrectionEvidence] = []
    workflows: list[WorkflowEvidence] = []
    for index, record in enumerate(records):
        kind = normalize_name(_record_text(record, "kind")).replace(" ", "_")
        refs = _record_refs(record)
        source_ref = _source_ref(record, refs, index)
        if kind in {"tool_failure", "tool_failed", "mcp_tool_failed", "native_tool_failed"}:
            tool_name = _record_value(record, refs, "tool_name", "tool") or _first_bare_ref(refs)
            error = _record_value(
                record,
                refs,
                "error_signature",
                "error",
                "message",
            ) or _record_text(record, "summary")
            if tool_name and error:
                tool_failures.append(ToolFailureEvidence(tool_name, error, source_ref))
            continue
        if kind in {"user_correction", "user_corrected", "correction"}:
            signature = _record_value(record, refs, "signature", "topic")
            correction = _record_text(record, "correction") or _record_text(
                record,
                "summary",
            )
            if signature and correction:
                user_corrections.append(
                    UserCorrectionEvidence(signature, correction, source_ref)
                )
            continue
        if kind in {"workflow_success", "successful_workflow", "workflow_completed"}:
            signature = _record_value(record, refs, "signature", "workflow_signature")
            name = _record_value(record, refs, "name", "workflow_name")
            description = _record_value(record, refs, "description") or _record_text(
                record,
                "summary",
            )
            steps = _record_tuple(record, refs, "steps", "step")
            reference_paths = _record_tuple(record, refs, "reference_paths", "path")
            if signature and name and description and steps:
                workflows.append(
                    WorkflowEvidence(
                        signature=signature,
                        name=name,
                        description=description,
                        steps=steps,
                        source_ref=source_ref,
                        reference_paths=reference_paths,
                    )
                )
    return EvolutionEvidenceBatch(
        tool_failures=tuple(tool_failures),
        user_corrections=tuple(user_corrections),
        workflows=tuple(workflows),
    )


def user_correction_candidates(
    corrections: tuple[UserCorrectionEvidence, ...],
    threshold: int = DEFAULT_USER_CORRECTION_THRESHOLD,
) -> tuple[EvidenceAggregate, ...]:
    """Return correction aggregates that meet the deterministic threshold."""
    grouped: dict[str, list[UserCorrectionEvidence]] = defaultdict(list)
    for correction in corrections:
        signature = evidence_signature(correction.signature)
        if signature:
            grouped[signature].append(correction)
    return tuple(
        _correction_aggregate(signature, items)
        for signature, items in sorted(grouped.items())
        if len(items) >= threshold
    )


def tool_failure_candidates(
    failures: tuple[ToolFailureEvidence, ...],
    threshold: int = DEFAULT_TOOL_FAILURE_THRESHOLD,
) -> tuple[EvidenceAggregate, ...]:
    """Return tool-failure aggregates that meet the deterministic threshold."""
    grouped: dict[str, list[ToolFailureEvidence]] = defaultdict(list)
    for failure in failures:
        signature = failure.signature()
        if signature:
            grouped[signature].append(failure)
    return tuple(
        _tool_failure_aggregate(signature, items)
        for signature, items in sorted(grouped.items())
        if len(items) >= threshold
    )


def workflow_candidates(
    workflows: tuple[WorkflowEvidence, ...],
    threshold: int = DEFAULT_WORKFLOW_THRESHOLD,
) -> tuple[WorkflowAggregate, ...]:
    """Return repeated workflow aggregates that can become generated skills."""
    grouped: dict[str, list[WorkflowEvidence]] = defaultdict(list)
    for workflow in workflows:
        signature = evidence_signature(workflow.signature)
        if signature:
            grouped[signature].append(workflow)
    candidates: list[WorkflowAggregate] = []
    for signature, items in sorted(grouped.items()):
        if len(items) < threshold:
            continue
        first = items[0]
        steps = _merge_steps(tuple(item.steps for item in items))
        refs = _merge_strings(item.source_ref for item in items)
        reference_paths = _merge_strings(
            path for item in items for path in item.reference_paths
        )
        candidate = WorkflowAggregate(
            signature=signature,
            name=first.name,
            description=first.description,
            steps=steps,
            count=len(items),
            source_refs=refs,
            reference_paths=reference_paths,
        )
        if candidate.is_ready():
            candidates.append(candidate)
    return tuple(candidates)


def _correction_aggregate(
    signature: str,
    items: list[UserCorrectionEvidence],
) -> EvidenceAggregate:
    return EvidenceAggregate(
        kind="user_correction",
        signature=signature,
        count=len(items),
        source_refs=_merge_strings(item.source_ref for item in items),
        examples=_merge_strings(item.correction for item in items),
    )


def _tool_failure_aggregate(
    signature: str,
    items: list[ToolFailureEvidence],
) -> EvidenceAggregate:
    return EvidenceAggregate(
        kind="tool_failure",
        signature=signature,
        count=len(items),
        source_refs=_merge_strings(item.source_ref for item in items),
        examples=_merge_strings(item.error_signature for item in items),
    )


def _merge_steps(step_groups: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    return _merge_strings(step for steps in step_groups for step in steps)


def _merge_strings(values) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value).strip().split())
        key = normalize_name(text)
        if not text or key in seen:
            continue
        seen.add(key)
        merged.append(text)
    return tuple(merged)


def _record_text(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    return value.strip() if isinstance(value, str) else ""


def _record_refs(record: Mapping[str, object]) -> tuple[str, ...]:
    refs = record.get("refs")
    if isinstance(refs, list) or isinstance(refs, tuple):
        return tuple(ref.strip() for ref in refs if isinstance(ref, str) and ref.strip())
    return ()


def _source_ref(
    record: Mapping[str, object],
    refs: tuple[str, ...],
    index: int,
) -> str:
    explicit = _record_text(record, "source_ref")
    if explicit:
        return explicit
    record_id = _record_text(record, "record_id") or _record_text(record, "id")
    if record_id:
        return f"record:{record_id}"
    recorded_at = _record_text(record, "recorded_at")
    kind = _record_text(record, "kind")
    if recorded_at and kind:
        return f"trace:{recorded_at}:{kind}"
    for ref in refs:
        if ref.startswith("source_ref="):
            return ref.split("=", 1)[1].strip()
    return f"record_index:{index}"


def _record_value(
    record: Mapping[str, object],
    refs: tuple[str, ...],
    *keys: str,
) -> str:
    for key in keys:
        value = _record_text(record, key)
        if value:
            return value
        for ref_value in _ref_values(refs, key):
            if ref_value:
                return ref_value
    return ""


def _record_tuple(
    record: Mapping[str, object],
    refs: tuple[str, ...],
    field_key: str,
    ref_key: str,
) -> tuple[str, ...]:
    value = record.get(field_key)
    if isinstance(value, list) or isinstance(value, tuple):
        return _merge_strings(item for item in value if isinstance(item, str))
    return _merge_strings(_ref_values(refs, ref_key))


def _ref_values(refs: tuple[str, ...], key: str) -> tuple[str, ...]:
    prefix = f"{key}="
    return tuple(ref.split("=", 1)[1].strip() for ref in refs if ref.startswith(prefix))


def _first_bare_ref(refs: tuple[str, ...]) -> str:
    for ref in refs:
        if "=" not in ref:
            return ref.strip()
    return ""
