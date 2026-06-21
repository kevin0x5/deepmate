"""Lightweight final-response review before user delivery."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from collections.abc import Iterable, Mapping

from deepmate.domain import ErrorInfo
from deepmate.providers import ModelToolExchange, ModelToolResult


class DeliveryReviewStatus(StrEnum):
    """Status for a final response draft before user delivery."""

    ACCEPTED = "accepted"
    NEEDS_REVISION = "needs_revision"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class DeliveryReviewInput:
    """Compressed packet used to review a final response draft."""

    user_request: str
    final_response_draft: str
    accepted_subagent_reviews: tuple[str, ...] = field(default_factory=tuple)
    non_accepted_subagent_reviews: tuple[str, ...] = field(default_factory=tuple)
    evidence_refs: tuple[str, ...] = field(default_factory=tuple)
    artifact_refs: tuple[str, ...] = field(default_factory=tuple)
    tool_test_summary: str = ""
    known_limits: tuple[str, ...] = field(default_factory=tuple)
    workspace_write_used: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "user_request", self.user_request.strip())
        object.__setattr__(
            self,
            "final_response_draft",
            self.final_response_draft.strip(),
        )
        object.__setattr__(
            self,
            "accepted_subagent_reviews",
            _unique_texts(self.accepted_subagent_reviews),
        )
        object.__setattr__(
            self,
            "non_accepted_subagent_reviews",
            _unique_texts(self.non_accepted_subagent_reviews),
        )
        object.__setattr__(self, "evidence_refs", _unique_texts(self.evidence_refs))
        object.__setattr__(self, "artifact_refs", _unique_texts(self.artifact_refs))
        object.__setattr__(self, "tool_test_summary", self.tool_test_summary.strip())
        object.__setattr__(self, "known_limits", _unique_texts(self.known_limits))


@dataclass(frozen=True, slots=True)
class DeliveryReview:
    """Deterministic review result for a final response draft."""

    status: DeliveryReviewStatus
    summary: str
    issues: tuple[str, ...] = field(default_factory=tuple)
    revision_instruction: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, DeliveryReviewStatus):
            object.__setattr__(self, "status", DeliveryReviewStatus(str(self.status)))
        object.__setattr__(self, "summary", self.summary.strip())
        object.__setattr__(self, "issues", _unique_texts(self.issues))
        object.__setattr__(
            self,
            "revision_instruction",
            self.revision_instruction.strip(),
        )

    def is_accepted(self) -> bool:
        """Return whether the draft can be delivered."""
        return self.status == DeliveryReviewStatus.ACCEPTED


def review_final_response(review_input: DeliveryReviewInput) -> DeliveryReview:
    """Run cheap deterministic checks on a final response draft."""
    issues: list[str] = []
    if not review_input.final_response_draft:
        issues.append("final_response_empty")
    if review_input.non_accepted_subagent_reviews:
        issues.append("non_accepted_subagent_results_present")
    if review_input.workspace_write_used and not review_input.artifact_refs:
        issues.append("workspace_write_without_artifact_refs")
    if review_input.workspace_write_used and not review_input.tool_test_summary:
        issues.append("workspace_write_without_validation_summary")
    if review_input.known_limits and not _mentions_limits(review_input.final_response_draft):
        issues.append("known_limits_not_disclosed")

    if not review_input.user_request:
        issues.append("user_request_missing")

    unique_issues = _unique_texts(tuple(issues))
    if not review_input.final_response_draft:
        return DeliveryReview(
            status=DeliveryReviewStatus.BLOCKED,
            summary="Final response draft is empty.",
            issues=unique_issues,
            revision_instruction="Create a user-facing response before delivery.",
        )
    if unique_issues:
        return DeliveryReview(
            status=DeliveryReviewStatus.NEEDS_REVISION,
            summary="Final response needs revision before delivery.",
            issues=unique_issues,
            revision_instruction=_revision_instruction(unique_issues),
        )
    return DeliveryReview(
        status=DeliveryReviewStatus.ACCEPTED,
        summary="Final response is ready for delivery.",
    )


def build_delivery_review_input(
    user_request: str,
    final_response_draft: str,
    tool_exchanges: Iterable[ModelToolExchange] = (),
    errors: Iterable[ErrorInfo] = (),
    reached_max_steps: bool = False,
) -> DeliveryReviewInput:
    """Build a compact delivery-review packet from a completed user turn."""
    exchanges = tuple(tool_exchanges)
    error_items = tuple(errors)
    accepted_subagent_reviews, non_accepted_subagent_reviews = (
        _subagent_review_summaries(exchanges, error_items)
    )
    evidence_refs, artifact_refs, workspace_write_used = _result_refs(exchanges)
    known_limits = _known_limits(error_items, reached_max_steps)
    tool_test_summary = _tool_test_summary(
        exchanges,
        error_items,
        workspace_write_used,
        reached_max_steps,
    )
    return DeliveryReviewInput(
        user_request=user_request,
        final_response_draft=final_response_draft,
        accepted_subagent_reviews=accepted_subagent_reviews,
        non_accepted_subagent_reviews=non_accepted_subagent_reviews,
        evidence_refs=evidence_refs,
        artifact_refs=artifact_refs,
        tool_test_summary=tool_test_summary,
        known_limits=known_limits,
        workspace_write_used=workspace_write_used,
    )


def should_run_llm_delivery_review(review_input: DeliveryReviewInput) -> bool:
    """Return whether a costlier LLM delivery review is worth considering."""
    if review_input.workspace_write_used:
        return True
    if len(review_input.accepted_subagent_reviews) >= 2:
        return True
    if review_input.non_accepted_subagent_reviews:
        return True
    if review_input.known_limits:
        return True
    return False


def _revision_instruction(issues: tuple[str, ...]) -> str:
    return "Revise the final response to address: " + ", ".join(issues)


def _mentions_limits(text: str) -> bool:
    lowered = text.lower()
    if re.search(r"\b(limit|limitation|constraint|caveat)\b", lowered):
        return True
    if re.search(r"\b(not run|not verified|not tested|skipped|unable|could not)\b", lowered):
        return True
    return any(marker in lowered for marker in ("未验证", "未运行", "限制", "风险"))


def _unique_texts(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = value.strip() if isinstance(value, str) else ""
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return tuple(output)


def _subagent_review_summaries(
    exchanges: tuple[ModelToolExchange, ...],
    errors: tuple[ErrorInfo, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    accepted: list[str] = []
    non_accepted: list[str] = []
    for exchange in exchanges:
        for tool_result in _iter_values(exchange.tool_results):
            if not isinstance(tool_result, ModelToolResult):
                continue
            if _text(tool_result.name) != "run_subagent":
                continue
            review = _tool_result_review(tool_result)
            if review is None:
                continue
            summary = str(review.get("summary", "")).strip()
            status = str(review.get("status", "")).strip().lower()
            if status == "accepted":
                if summary:
                    accepted.append(summary)
                continue
            if summary:
                non_accepted.append(summary)
    for error in errors:
        if not _text(error.code).startswith("subagent_result_"):
            continue
        message = _text(error.message)
        if message:
            non_accepted.append(message)
    return _unique_texts(tuple(accepted)), _unique_texts(tuple(non_accepted))


def _tool_result_review(tool_result: ModelToolResult) -> Mapping[str, object] | None:
    payload = _tool_result_payload(tool_result)
    if payload is None:
        return None
    review = payload.get("review")
    if not isinstance(review, dict):
        return None
    return review


def _tool_result_payload(tool_result: ModelToolResult) -> Mapping[str, object] | None:
    content = _text(tool_result.content)
    if not content:
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _result_refs(
    exchanges: tuple[ModelToolExchange, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    evidence_refs: list[str] = []
    artifact_refs: list[str] = []
    workspace_write_used = False
    for exchange in exchanges:
        for result in _iter_values(exchange.tool_results):
            if not isinstance(result, ModelToolResult):
                continue
            refs = tuple(str(ref).strip() for ref in _iter_values(result.refs) if str(ref).strip())
            if refs:
                evidence_refs.extend(refs)
            result_name = _text(result.name)
            if result_name in {"write_text_file", "edit_text_file"}:
                workspace_write_used = True
                artifact_refs.extend(refs)
            if result_name == "run_subagent":
                payload = _tool_result_payload(result)
                if payload is None:
                    continue
                nested_artifacts = payload.get("artifact_refs")
                if isinstance(nested_artifacts, list):
                    artifact_texts = [
                        str(item).strip()
                        for item in nested_artifacts
                        if str(item).strip()
                    ]
                    if artifact_texts:
                        workspace_write_used = True
                        artifact_refs.extend(artifact_texts)
    return (
        _unique_texts(tuple(evidence_refs)),
        _unique_texts(tuple(artifact_refs)),
        workspace_write_used,
    )


def _known_limits(
    errors: tuple[ErrorInfo, ...],
    reached_max_steps: bool,
) -> tuple[str, ...]:
    limits: list[str] = []
    if reached_max_steps:
        limits.append("reached_max_steps")
    for error in errors:
        text = _text(error.message)
        if text:
            limits.append(text)
    return _unique_texts(tuple(limits))


def _tool_test_summary(
    exchanges: tuple[ModelToolExchange, ...],
    errors: tuple[ErrorInfo, ...],
    workspace_write_used: bool,
    reached_max_steps: bool,
) -> str:
    if errors:
        codes = [_text(error.code) for error in errors if _text(error.code)]
        if codes:
            return "errors: " + ", ".join(codes[:3])
        messages = [_text(error.message) for error in errors if _text(error.message)]
        if messages:
            return "errors: " + ", ".join(messages[:2])
        return "errors: unknown runtime error."
    if reached_max_steps:
        return "reached max_steps before final answer."
    if workspace_write_used:
        return "workspace write tool execution completed without runtime errors."
    if exchanges:
        return "tool execution completed without runtime errors."
    return ""


def _iter_values(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return ()


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""
