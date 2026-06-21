"""Lightweight orchestration over bounded Deepmate subagent runs."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from enum import StrEnum

from deepmate.context import ContextWarning
from deepmate.providers import TokenUsage
from deepmate.runtime.agent_loop import HistorySink
from deepmate.runtime.tool_policy import ToolAccessMode
from deepmate.subagents.runtime import (
    SubagentResultObserver,
    SubagentRuntime,
    run_subagent,
)
from deepmate.subagents.types import SubagentRunRequest, SubagentRunResult
from deepmate.subagents.verification import (
    SubagentResultReview,
    review_subagent_result,
)
from deepmate.trace import TraceEvent


class SubagentAssignmentStage(StrEnum):
    """Small set of orchestration stages for a child assignment."""

    PLAN = "plan"
    EXECUTE = "execute"
    REFLECT = "reflect"


class SubagentWorkflowStatus(StrEnum):
    """Final status for one subagent orchestration run."""

    COMPLETED = "completed"
    REVISED = "revised"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SubagentAssignment:
    """One bounded unit of delegated work."""

    assignment_id: str
    goal: str
    input_context: str = ""
    output_contract: str | None = None
    acceptance_criteria: tuple[str, ...] = field(default_factory=tuple)
    allowed_tools: tuple[str, ...] = field(default_factory=tuple)
    tool_access_mode: ToolAccessMode = ToolAccessMode.READ_ONLY
    max_steps: int = 3
    depends_on: tuple[str, ...] = field(default_factory=tuple)
    stage: SubagentAssignmentStage = SubagentAssignmentStage.EXECUTE

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignment_id", self.assignment_id.strip())
        object.__setattr__(self, "goal", self.goal.strip())
        object.__setattr__(self, "input_context", self.input_context.strip())
        object.__setattr__(
            self,
            "output_contract",
            _optional_text(self.output_contract),
        )
        object.__setattr__(
            self,
            "acceptance_criteria",
            _unique_texts(self.acceptance_criteria),
        )
        object.__setattr__(
            self,
            "allowed_tools",
            _unique_texts(self.allowed_tools),
        )
        object.__setattr__(self, "depends_on", _unique_texts(self.depends_on))
        if not isinstance(self.tool_access_mode, ToolAccessMode):
            object.__setattr__(
                self,
                "tool_access_mode",
                ToolAccessMode(str(self.tool_access_mode)),
            )
        if not isinstance(self.stage, SubagentAssignmentStage):
            object.__setattr__(
                self,
                "stage",
                SubagentAssignmentStage(str(self.stage)),
            )

    def is_ready(self) -> bool:
        """Return whether this assignment can be converted to a child run."""
        return bool(self.assignment_id and self.goal and self.max_steps >= 1)


@dataclass(frozen=True, slots=True)
class SubagentOrchestrationPolicy:
    """Budget and quality gates for bounded subagent orchestration."""

    max_child_runs: int = 4
    max_workspace_write_child_runs: int = 1
    max_revise_attempts: int = 1
    max_child_steps: int = 12
    revise_step_extension: int = 2
    auto_revise: bool = True
    allow_workspace_write_revise: bool = False
    enable_reflector: bool = True
    reflect_on_workspace_write: bool = True
    reflect_on_multiple_results: bool = True
    reflect_on_non_accepted: bool = True

    def __post_init__(self) -> None:
        if self.max_child_runs < 1:
            raise ValueError("max_child_runs must be at least 1")
        if self.max_workspace_write_child_runs < 0:
            raise ValueError("max_workspace_write_child_runs must be non-negative")
        if self.max_revise_attempts < 0:
            raise ValueError("max_revise_attempts must be non-negative")
        if self.max_child_steps < 1:
            raise ValueError("max_child_steps must be at least 1")
        if self.revise_step_extension < 0:
            raise ValueError("revise_step_extension must be non-negative")


@dataclass(frozen=True, slots=True)
class SubagentAssignmentRun:
    """Observed result for one assignment attempt."""

    assignment: SubagentAssignment
    request: SubagentRunRequest
    result: SubagentRunResult
    review: SubagentResultReview
    attempt: int = 1

    def is_accepted(self) -> bool:
        """Return whether this attempt can be merged."""
        return self.result.is_success() and self.review.is_accepted()


@dataclass(frozen=True, slots=True)
class SubagentWorkflowResult:
    """Aggregated result from one orchestration pass."""

    status: SubagentWorkflowStatus
    plan_summary: str = ""
    execution_summary: str = ""
    accepted_results: tuple[SubagentRunResult, ...] = field(default_factory=tuple)
    non_accepted_reviews: tuple[SubagentResultReview, ...] = field(
        default_factory=tuple
    )
    reflector_summary: str = ""
    artifact_refs: tuple[str, ...] = field(default_factory=tuple)
    evidence_refs: tuple[str, ...] = field(default_factory=tuple)
    revised_assignment_ids: tuple[str, ...] = field(default_factory=tuple)
    assignment_runs: tuple[SubagentAssignmentRun, ...] = field(default_factory=tuple)
    blocking_gaps: tuple[str, ...] = field(default_factory=tuple)
    usage: TokenUsage | None = None

    def is_success(self) -> bool:
        """Return whether orchestration reached a mergeable final state."""
        return self.status in {
            SubagentWorkflowStatus.COMPLETED,
            SubagentWorkflowStatus.REVISED,
        }


def run_subagent_orchestration(
    runtime: SubagentRuntime,
    assignments: tuple[SubagentAssignment, ...],
    *,
    plan_summary: str = "",
    reflector_assignment: SubagentAssignment | None = None,
    policy: SubagentOrchestrationPolicy | None = None,
    warning_sink: Callable[[ContextWarning], None] | None = None,
    history_sink: HistorySink | None = None,
    result_observer: SubagentResultObserver | None = None,
) -> SubagentWorkflowResult:
    """Run a bounded assignment -> review -> optional revise/reflect workflow."""
    active_policy = policy or SubagentOrchestrationPolicy()
    validated = _validated_assignments(assignments)
    _record_trace(
        runtime,
        "subagent_orchestration_started",
        "Subagent orchestration started.",
        refs=(
            f"assignments={len(validated)}",
            f"max_child_runs={active_policy.max_child_runs}",
            f"max_workspace_write_child_runs={active_policy.max_workspace_write_child_runs}",
            f"max_child_steps={active_policy.max_child_steps}",
            f"max_revise_attempts={active_policy.max_revise_attempts}",
        ),
    )
    if not validated:
        result = SubagentWorkflowResult(
            status=SubagentWorkflowStatus.FAILED,
            plan_summary=plan_summary.strip(),
            execution_summary="No ready subagent assignments were provided.",
            blocking_gaps=("ready_assignment",),
        )
        _record_workflow_finished(runtime, result)
        return result

    runs: list[SubagentAssignmentRun] = []
    accepted_by_id: dict[str, SubagentRunResult] = {}
    blocking_gaps: list[str] = []
    revised_assignment_ids: list[str] = []
    child_runs_used = 0
    workspace_write_child_runs_used = 0

    for assignment in validated:
        missing_dependencies = tuple(
            dependency
            for dependency in assignment.depends_on
            if dependency not in accepted_by_id
        )
        if missing_dependencies:
            blocking_gaps.append(
                f"{assignment.assignment_id}: missing accepted dependencies "
                + ", ".join(missing_dependencies)
            )
            _record_trace(
                runtime,
                "subagent_assignment_blocked",
                "Subagent assignment blocked by missing dependencies.",
                refs=(
                    f"assignment_id={assignment.assignment_id}",
                    f"missing_dependencies={len(missing_dependencies)}",
                ),
            )
            continue

        if child_runs_used >= active_policy.max_child_runs:
            blocking_gaps.append(
                f"{assignment.assignment_id}: child run budget exhausted"
            )
            _record_trace(
                runtime,
                "subagent_assignment_blocked",
                "Subagent assignment blocked by child run budget.",
                refs=(
                    f"assignment_id={assignment.assignment_id}",
                    f"child_runs={child_runs_used}",
                    f"max_child_runs={active_policy.max_child_runs}",
                ),
            )
            continue

        if _workspace_write_child_budget_exhausted(
            assignment,
            active_policy,
            workspace_write_child_runs_used,
        ):
            blocking_gaps.append(
                f"{assignment.assignment_id}: workspace_write child run budget exhausted"
            )
            _record_trace(
                runtime,
                "subagent_assignment_blocked",
                "Subagent assignment blocked by workspace-write child run budget.",
                refs=(
                    f"assignment_id={assignment.assignment_id}",
                    f"workspace_write_child_runs={workspace_write_child_runs_used}",
                    f"max_workspace_write_child_runs={active_policy.max_workspace_write_child_runs}",
                ),
            )
            continue

        first_run = _run_assignment(
            runtime=runtime,
            assignment=assignment,
            attempt=1,
            dependency_results=accepted_by_id,
            policy=active_policy,
            warning_sink=warning_sink,
            history_sink=history_sink,
            result_observer=result_observer,
        )
        runs.append(first_run)
        child_runs_used += 1
        if assignment.tool_access_mode == ToolAccessMode.WORKSPACE_WRITE:
            workspace_write_child_runs_used += 1
        final_run = first_run

        if (
            not final_run.is_accepted()
            and _can_revise(final_run, active_policy)
            and child_runs_used < active_policy.max_child_runs
        ):
            _record_trace(
                runtime,
                "subagent_assignment_revised",
                "Subagent assignment scheduled for one bounded revision.",
                refs=(
                    f"assignment_id={assignment.assignment_id}",
                    f"attempt={final_run.attempt + 1}",
                    f"subagent_run_id={final_run.result.run_id}",
                ),
            )
            revision = _revision_assignment(final_run, active_policy)
            revised_run = _run_assignment(
                runtime=runtime,
                assignment=revision,
                attempt=2,
                dependency_results=accepted_by_id,
                policy=active_policy,
                warning_sink=warning_sink,
                history_sink=history_sink,
                result_observer=result_observer,
            )
            runs.append(revised_run)
            child_runs_used += 1
            if revision.tool_access_mode == ToolAccessMode.WORKSPACE_WRITE:
                workspace_write_child_runs_used += 1
            final_run = revised_run
            revised_assignment_ids.append(assignment.assignment_id)

        if final_run.is_accepted():
            accepted_by_id[assignment.assignment_id] = final_run.result
        else:
            blocking_gaps.append(_blocking_gap_for_run(final_run))

    reflector_summary = ""
    if _should_run_reflector(
        runs,
        reflector_assignment,
        active_policy,
        blocking_gaps=tuple(blocking_gaps),
    ):
        if child_runs_used >= active_policy.max_child_runs:
            blocking_gaps.append("reflector: child run budget exhausted")
        else:
            reflector = _reflector_assignment(
                reflector_assignment,
                plan_summary=plan_summary,
                accepted_results=tuple(accepted_by_id.values()),
                blocking_gaps=tuple(blocking_gaps),
            )
            _record_trace(
                runtime,
                "subagent_reflector_started",
                "Subagent reflector assignment started.",
                refs=(
                    f"assignment_id={reflector.assignment_id}",
                    f"accepted_results={len(accepted_by_id)}",
                ),
            )
            reflector_run = _run_assignment(
                runtime=runtime,
                assignment=reflector,
                attempt=1,
                dependency_results={},
                policy=active_policy,
                warning_sink=warning_sink,
                history_sink=history_sink,
                result_observer=result_observer,
            )
            runs.append(reflector_run)
            child_runs_used += 1
            if reflector_run.is_accepted():
                reflector_summary = reflector_run.result.summary
            else:
                blocking_gaps.append(f"reflector: {reflector_run.review.summary}")

    accepted_results = tuple(
        run.result
        for run in runs
        if run.is_accepted() and run.assignment.stage != SubagentAssignmentStage.REFLECT
    )
    non_accepted_reviews = tuple(run.review for run in runs if not run.is_accepted())
    status = _workflow_status(
        accepted_results=accepted_results,
        blocking_gaps=tuple(blocking_gaps),
        revised_assignment_ids=tuple(revised_assignment_ids),
    )
    execution_summary = _execution_summary(
        status=status,
        accepted_results=accepted_results,
        blocking_gaps=tuple(blocking_gaps),
        assignment_runs=tuple(runs),
    )
    result = SubagentWorkflowResult(
        status=status,
        plan_summary=plan_summary.strip(),
        execution_summary=execution_summary,
        accepted_results=accepted_results,
        non_accepted_reviews=non_accepted_reviews,
        reflector_summary=reflector_summary,
        artifact_refs=_aggregate_refs(result.artifact_refs for result in accepted_results),
        evidence_refs=_aggregate_refs(result.evidence_refs for result in accepted_results),
        revised_assignment_ids=_unique_texts(tuple(revised_assignment_ids)),
        assignment_runs=tuple(runs),
        blocking_gaps=_unique_texts(tuple(blocking_gaps)),
        usage=_aggregate_usage(run.result.usage for run in runs),
    )
    _record_workflow_finished(runtime, result)
    return result


def _validated_assignments(
    assignments: tuple[SubagentAssignment, ...],
) -> tuple[SubagentAssignment, ...]:
    seen: set[str] = set()
    valid: list[SubagentAssignment] = []
    for assignment in assignments:
        if not assignment.is_ready() or assignment.assignment_id in seen:
            continue
        seen.add(assignment.assignment_id)
        valid.append(assignment)
    return tuple(valid)


def _run_assignment(
    *,
    runtime: SubagentRuntime,
    assignment: SubagentAssignment,
    attempt: int,
    dependency_results: dict[str, SubagentRunResult],
    policy: SubagentOrchestrationPolicy,
    warning_sink: Callable[[ContextWarning], None] | None,
    history_sink: HistorySink | None,
    result_observer: SubagentResultObserver | None,
) -> SubagentAssignmentRun:
    _record_trace(
        runtime,
        "subagent_assignment_started",
        "Subagent assignment started.",
        refs=(
            f"assignment_id={assignment.assignment_id}",
            f"attempt={attempt}",
            f"stage={assignment.stage.value}",
            f"max_steps={min(assignment.max_steps, policy.max_child_steps)}",
            f"tool_access_mode={assignment.tool_access_mode.value}",
            f"depends_on={len(assignment.depends_on)}",
        ),
    )
    request = _request_for_assignment(
        runtime,
        assignment,
        attempt=attempt,
        dependency_results=dependency_results,
        policy=policy,
    )
    result = run_subagent(
        runtime,
        request,
        warning_sink=warning_sink,
        history_sink=history_sink,
        result_observer=result_observer,
    )
    review = review_subagent_result(request, result)
    _record_trace(
        runtime,
        "subagent_assignment_reviewed",
        "Subagent assignment reviewed.",
        refs=(
            f"assignment_id={assignment.assignment_id}",
            f"attempt={attempt}",
            f"stage={assignment.stage.value}",
            f"subagent_run_id={result.run_id}",
            f"status={result.status.value}",
            f"review_status={review.status.value}",
            f"retryable={str(review.retryable).lower()}",
            f"missing={len(review.missing)}",
        ),
    )
    return SubagentAssignmentRun(
        assignment=assignment,
        request=request,
        result=result,
        review=review,
        attempt=attempt,
    )


def _request_for_assignment(
    runtime: SubagentRuntime,
    assignment: SubagentAssignment,
    *,
    attempt: int,
    dependency_results: dict[str, SubagentRunResult],
    policy: SubagentOrchestrationPolicy,
) -> SubagentRunRequest:
    input_context = _input_context_with_dependencies(assignment, dependency_results)
    activation = runtime.activation
    return SubagentRunRequest(
        run_id=f"{assignment.assignment_id}-attempt-{attempt}",
        goal=assignment.goal,
        input_context=input_context,
        output_contract=assignment.output_contract,
        acceptance_criteria=assignment.acceptance_criteria,
        allowed_tools=assignment.allowed_tools,
        tool_access_mode=assignment.tool_access_mode,
        max_steps=min(assignment.max_steps, policy.max_child_steps),
        parent_session_id=activation.session_id if activation else None,
        parent_activation_id=activation.activation_id if activation else None,
    )


def _input_context_with_dependencies(
    assignment: SubagentAssignment,
    dependency_results: dict[str, SubagentRunResult],
) -> str:
    pieces: list[str] = []
    if assignment.input_context:
        pieces.append(assignment.input_context)
    dependency_lines: list[str] = []
    for dependency in assignment.depends_on:
        result = dependency_results.get(dependency)
        if result is None:
            continue
        dependency_lines.append(f"- {dependency}: {result.summary}")
        if result.evidence_refs:
            dependency_lines.append(
                "  evidence_refs: " + ", ".join(result.evidence_refs)
            )
        if result.artifact_refs:
            dependency_lines.append(
                "  artifact_refs: " + ", ".join(result.artifact_refs)
            )
    if dependency_lines:
        pieces.append("Dependency summaries:\n" + "\n".join(dependency_lines))
    return "\n\n".join(piece for piece in pieces if piece.strip())


def _can_revise(
    run: SubagentAssignmentRun,
    policy: SubagentOrchestrationPolicy,
) -> bool:
    if not policy.auto_revise or policy.max_revise_attempts < 1:
        return False
    if not run.review.retryable or not run.review.retry_instruction:
        return False
    if run.attempt > policy.max_revise_attempts:
        return False
    if (
        run.assignment.tool_access_mode == ToolAccessMode.WORKSPACE_WRITE
        and not policy.allow_workspace_write_revise
    ):
        return False
    return True


def _workspace_write_child_budget_exhausted(
    assignment: SubagentAssignment,
    policy: SubagentOrchestrationPolicy,
    workspace_write_child_runs_used: int,
) -> bool:
    return (
        assignment.tool_access_mode == ToolAccessMode.WORKSPACE_WRITE
        and workspace_write_child_runs_used >= policy.max_workspace_write_child_runs
    )


def _revision_assignment(
    run: SubagentAssignmentRun,
    policy: SubagentOrchestrationPolicy,
) -> SubagentAssignment:
    assignment = run.assignment
    input_context = assignment.input_context.strip()
    revision_context = "\n".join(
        piece
        for piece in (
            "Previous attempt was not mergeable.",
            f"Previous summary: {run.result.summary}",
            f"Revision instruction: {run.review.retry_instruction}",
        )
        if piece.strip()
    )
    if input_context:
        input_context = f"{input_context}\n\n{revision_context}"
    else:
        input_context = revision_context
    output_contract = assignment.output_contract or ""
    output_contract = "\n".join(
        piece
        for piece in (
            output_contract,
            "Revision must address: " + ", ".join(run.review.missing),
            run.review.retry_instruction,
        )
        if piece.strip()
    )
    return replace(
        assignment,
        input_context=input_context,
        output_contract=output_contract or None,
        max_steps=min(
            assignment.max_steps + policy.revise_step_extension,
            policy.max_child_steps,
        ),
    )


def _blocking_gap_for_run(run: SubagentAssignmentRun) -> str:
    summary = f"{run.assignment.assignment_id}: {run.review.summary}"
    if run.review.missing:
        summary += "; missing: " + ", ".join(run.review.missing)
    return summary


def _should_run_reflector(
    runs: list[SubagentAssignmentRun],
    reflector_assignment: SubagentAssignment | None,
    policy: SubagentOrchestrationPolicy,
    blocking_gaps: tuple[str, ...] = (),
) -> bool:
    if not policy.enable_reflector or reflector_assignment is None:
        return False
    accepted_execute_runs = [
        run
        for run in runs
        if run.is_accepted() and run.assignment.stage != SubagentAssignmentStage.REFLECT
    ]
    if policy.reflect_on_workspace_write and any(
        run.assignment.tool_access_mode == ToolAccessMode.WORKSPACE_WRITE
        for run in accepted_execute_runs
    ):
        return True
    if policy.reflect_on_multiple_results and len(accepted_execute_runs) >= 2:
        return True
    if policy.reflect_on_non_accepted and (
        blocking_gaps or any(not run.is_accepted() for run in runs)
    ):
        return True
    return False


def _reflector_assignment(
    assignment: SubagentAssignment,
    *,
    plan_summary: str,
    accepted_results: tuple[SubagentRunResult, ...],
    blocking_gaps: tuple[str, ...] = (),
) -> SubagentAssignment:
    result_lines: list[str] = []
    for index, result in enumerate(accepted_results, start=1):
        result_lines.append(f"{index}. {result.summary}")
        if result.artifact_refs:
            result_lines.append("   artifact_refs: " + ", ".join(result.artifact_refs))
        if result.evidence_refs:
            result_lines.append("   evidence_refs: " + ", ".join(result.evidence_refs))
    context_parts = [
        assignment.input_context,
        f"Plan summary:\n{plan_summary.strip()}" if plan_summary.strip() else "",
        "Accepted child results:\n" + "\n".join(result_lines)
        if result_lines
        else "Accepted child results: none",
        "Blocking gaps:\n" + "\n".join(f"- {gap}" for gap in blocking_gaps)
        if blocking_gaps
        else "",
    ]
    criteria = assignment.acceptance_criteria or (
        "Check whether accepted child results satisfy the plan.",
        "List blocking gaps if the parent should not deliver yet.",
        "Do not request writes or recursive subagents.",
    )
    return replace(
        assignment,
        input_context="\n\n".join(part for part in context_parts if part.strip()),
        acceptance_criteria=criteria,
        tool_access_mode=ToolAccessMode.READ_ONLY,
        stage=SubagentAssignmentStage.REFLECT,
    )


def _workflow_status(
    *,
    accepted_results: tuple[SubagentRunResult, ...],
    blocking_gaps: tuple[str, ...],
    revised_assignment_ids: tuple[str, ...],
) -> SubagentWorkflowStatus:
    if blocking_gaps:
        return SubagentWorkflowStatus.BLOCKED
    if revised_assignment_ids:
        return SubagentWorkflowStatus.REVISED
    return SubagentWorkflowStatus.COMPLETED


def _execution_summary(
    *,
    status: SubagentWorkflowStatus,
    accepted_results: tuple[SubagentRunResult, ...],
    blocking_gaps: tuple[str, ...],
    assignment_runs: tuple[SubagentAssignmentRun, ...],
) -> str:
    summary = (
        f"status={status.value}; "
        f"accepted_results={len(accepted_results)}; "
        f"child_runs={len(assignment_runs)}"
    )
    if blocking_gaps:
        summary += "; blocking_gaps=" + " | ".join(blocking_gaps)
    return summary


def _aggregate_refs(ref_groups: Iterable[Iterable[str]]) -> tuple[str, ...]:
    refs: list[str] = []
    for group in ref_groups:
        refs.extend(group)
    return _unique_texts(tuple(refs))


def _aggregate_usage(usages: Iterable[TokenUsage | None]) -> TokenUsage | None:
    found = False
    input_tokens = 0
    output_tokens = 0
    cache_hit_input_tokens = 0
    cache_miss_input_tokens = 0
    reasoning_tokens = 0
    for usage in usages:
        if usage is None:
            continue
        found = True
        input_tokens += usage.input_tokens
        output_tokens += usage.output_tokens
        cache_hit_input_tokens += usage.cache_hit_input_tokens
        cache_miss_input_tokens += usage.cache_miss_input_tokens
        reasoning_tokens += usage.reasoning_tokens
    if not found:
        return None
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_hit_input_tokens=cache_hit_input_tokens,
        cache_miss_input_tokens=cache_miss_input_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def _record_workflow_finished(
    runtime: SubagentRuntime,
    result: SubagentWorkflowResult,
) -> None:
    _record_trace(
        runtime,
        "subagent_orchestration_finished",
        "Subagent orchestration finished.",
        refs=(
            f"status={result.status.value}",
            f"child_runs={len(result.assignment_runs)}",
            f"accepted_results={len(result.accepted_results)}",
            f"blocking_gaps={len(result.blocking_gaps)}",
            f"revised={len(result.revised_assignment_ids)}",
            f"artifact_refs={len(result.artifact_refs)}",
            f"evidence_refs={len(result.evidence_refs)}",
        ),
    )


def _record_trace(
    runtime: SubagentRuntime,
    kind: str,
    summary: str,
    refs: tuple[str, ...] = (),
) -> None:
    recorder = runtime.trace_recorder
    if recorder is None:
        return
    activation_refs = (
        runtime.activation.trace_refs() if runtime.activation is not None else ()
    )
    clean_refs = tuple(ref for ref in (*refs, *activation_refs) if ref)
    recorder.record(TraceEvent(kind=kind, summary=summary, refs=clean_refs))


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _unique_texts(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        text = value.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return tuple(normalized)
