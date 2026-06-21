"""Parent-side verification for subagent results."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from deepmate.runtime.tool_policy import ToolAccessMode
from deepmate.subagents.types import (
    SubagentRunRequest,
    SubagentRunResult,
    SubagentRunStatus,
)


class SubagentReviewStatus(StrEnum):
    """Merge-readiness status for one subagent result."""

    ACCEPTED = "accepted"
    INCOMPLETE = "incomplete"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class SubagentResultReview:
    """Parent-side review of whether a child run result can be merged."""

    status: SubagentReviewStatus
    summary: str
    missing: tuple[str, ...] = field(default_factory=tuple)
    retryable: bool = False
    retry_instruction: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, SubagentReviewStatus):
            object.__setattr__(self, "status", SubagentReviewStatus(str(self.status)))
        object.__setattr__(self, "summary", self.summary.strip())
        object.__setattr__(self, "missing", _unique_texts(self.missing))
        object.__setattr__(self, "retry_instruction", self.retry_instruction.strip())

    def is_accepted(self) -> bool:
        """Return whether the parent agent may merge the child result."""
        return self.status == SubagentReviewStatus.ACCEPTED

    def refs(self) -> tuple[str, ...]:
        """Return compact trace refs for this review."""
        refs = [
            f"review_status={self.status.value}",
            f"retryable={self.retryable}",
            f"missing={len(self.missing)}",
        ]
        return tuple(refs)

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-serializable review payload."""
        payload: dict[str, object] = {
            "status": self.status.value,
            "summary": self.summary,
            "missing": list(self.missing),
            "retryable": self.retryable,
        }
        if self.retry_instruction:
            payload["retry_instruction"] = self.retry_instruction
        return payload


def review_subagent_result(
    request: SubagentRunRequest,
    result: SubagentRunResult,
) -> SubagentResultReview:
    """Review whether a subagent result satisfies the parent request."""
    if result.status == SubagentRunStatus.MAX_STEPS_REACHED:
        return SubagentResultReview(
            status=SubagentReviewStatus.FAILED,
            summary="Subagent reached max_steps before producing a final answer.",
            missing=("final_answer",),
            retryable=True,
            retry_instruction=(
                "Narrow the goal or input_context before retrying; do not simply "
                "increase max_steps."
            ),
        )

    if result.status == SubagentRunStatus.FAILED or result.error is not None:
        message = result.error.message if result.error is not None else result.summary
        return SubagentResultReview(
            status=SubagentReviewStatus.FAILED,
            summary=message or "Subagent run failed.",
            missing=("successful_child_run",),
            retryable=False,
        )

    missing = _missing_merge_inputs(request, result)
    if missing:
        return SubagentResultReview(
            status=SubagentReviewStatus.INCOMPLETE,
            summary="Subagent result is incomplete for the requested output contract.",
            missing=missing,
            retryable=_retryable_missing(missing, request),
            retry_instruction=_retry_instruction(missing),
        )

    if result.status != SubagentRunStatus.COMPLETED:
        return SubagentResultReview(
            status=SubagentReviewStatus.REJECTED,
            summary=f"Unsupported subagent status for merging: {result.status.value}.",
            missing=("supported_status",),
            retryable=False,
        )

    return SubagentResultReview(
        status=SubagentReviewStatus.ACCEPTED,
        summary="Subagent result is mergeable.",
    )


def _missing_merge_inputs(
    request: SubagentRunRequest,
    result: SubagentRunResult,
) -> tuple[str, ...]:
    missing: list[str] = []
    if not result.summary.strip():
        missing.append("summary")

    contract = (request.output_contract or "").lower()
    if _mentions_any(
        contract,
        (
            "evidence",
            "evidence_ref",
            "evidence_refs",
            "ref",
            "refs",
            "reference",
            "references",
            "引用",
            "证据",
        ),
    ):
        if not _has_external_ref(result.evidence_refs):
            missing.append("evidence_refs")
    if _mentions_any(
        contract,
        (
            "artifact",
            "artifacts",
            "artifact_ref",
            "artifact_refs",
            "file",
            "files",
            "diff",
            "产物",
            "文件",
        ),
    ):
        if not result.artifact_refs:
            missing.append("artifact_refs")
    if request.tool_access_mode == ToolAccessMode.WORKSPACE_WRITE:
        if not result.artifact_refs:
            missing.append("artifact_refs_for_write")

    return _unique_texts(tuple(missing))


def _retryable_missing(
    missing: tuple[str, ...],
    request: SubagentRunRequest,
) -> bool:
    if request.tool_access_mode == ToolAccessMode.WORKSPACE_WRITE:
        return "artifact_refs_for_write" not in missing
    return all(item in {"summary", "evidence_refs", "artifact_refs"} for item in missing)


def _retry_instruction(missing: tuple[str, ...]) -> str:
    if not missing:
        return ""
    return "Retry once with a tighter output_contract requiring: " + ", ".join(missing)


def _mentions_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(_mentions_term(text, needle) for needle in needles)


def _mentions_term(text: str, needle: str) -> bool:
    if _ascii_word(needle):
        return re.search(rf"(?<![a-z0-9_]){re.escape(needle)}(?![a-z0-9_])", text) is not None
    return needle in text


def _ascii_word(text: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9_]+", text))


def _has_external_ref(refs: tuple[str, ...]) -> bool:
    """Return whether refs include evidence beyond internal trace markers."""
    return any(
        not (
            ref.startswith("subagent_run_id=")
            or ref.startswith("parent_session_id=")
            or ref.startswith("parent_activation_id=")
            or ref.startswith("tool_access_mode=")
            or ref.startswith("max_steps=")
            or ref.startswith("goal=")
        )
        for ref in refs
    )


def _unique_texts(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = value.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return tuple(output)
