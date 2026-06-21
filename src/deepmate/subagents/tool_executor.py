"""Model-visible bridge for running Deepmate subagents."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from deepmate.domain import ErrorInfo, RuntimeEvent
from deepmate.providers import ModelToolRequest, ModelToolResult, TokenUsage
from deepmate.runtime.hooks import (
    HookActor,
    HookDirective,
    HookEnvelope,
    HookEvent,
    HookOutcome,
    HookRuntimeContext,
)
from deepmate.runtime.tool_executor import ToolExecutionResult
from deepmate.runtime.tool_policy import ToolAccessMode, ToolAccessPolicy
from deepmate.subagents.orchestration import (
    SubagentAssignment,
    SubagentAssignmentStage,
    SubagentOrchestrationPolicy,
    SubagentWorkflowResult,
    run_subagent_orchestration,
)
from deepmate.subagents.runtime import SubagentRuntime, run_subagent
from deepmate.subagents.store import (
    SubagentResultStore,
    read_subagent_record_payload,
)
from deepmate.subagents.types import (
    SubagentRunRequest,
    SubagentRunResult,
    SubagentRunStatus,
)
from deepmate.subagents.verification import (
    SubagentResultReview,
    SubagentReviewStatus,
    review_subagent_result,
)

if TYPE_CHECKING:
    from deepmate.capabilities import CapabilitySurface
    from deepmate.mcp import McpToolExecutor
    from deepmate.runtime import RuntimeActivation
    from deepmate.skills import SkillDocument
    from deepmate.tools import NativeToolRegistry

SUBAGENT_TOOL_NAME = "run_subagent"
SUBAGENT_WORKFLOW_TOOL_NAME = "run_subagent_workflow"
READ_SUBAGENT_RESULT_TOOL_NAME = "read_subagent_result"
SUBAGENT_TOOL_NAMES = frozenset(
    (
        SUBAGENT_TOOL_NAME,
        SUBAGENT_WORKFLOW_TOOL_NAME,
        READ_SUBAGENT_RESULT_TOOL_NAME,
    )
)
DEFAULT_SUBAGENT_TOOL_DESCRIPTION = (
    "Run a bounded child agent for an isolated, well-scoped subtask. "
    "Use it when delegation reduces noise or tool churn, not for trivial calls."
)
DEFAULT_SUBAGENT_WORKFLOW_TOOL_DESCRIPTION = (
    "Run a bounded assignment workflow over child agents. Use it when a task "
    "benefits from several scoped child runs, dependency summaries, deterministic "
    "review, one narrow revision, or a final reflector check."
)
DEFAULT_READ_SUBAGENT_RESULT_DESCRIPTION = (
    "Read persisted details for a subagent result handle from the current session. "
    "Use only when the default subagent summary/evidence is not enough."
)


@dataclass(frozen=True, slots=True)
class SubagentRuntimeBinding:
    """Current parent runtime state to apply before one child run."""

    capability_surface: "CapabilitySurface | None" = None
    native_tools: "NativeToolRegistry | None" = None
    mcp_tools: "McpToolExecutor | None" = None
    tool_schemas: tuple[Mapping[str, object], ...] = field(default_factory=tuple)
    selected_skill_documents: tuple["SkillDocument", ...] = field(default_factory=tuple)
    activation: "RuntimeActivation | None" = None
    parent_tool_access_policy: ToolAccessPolicy | None = None
    default_allowed_tools: tuple[str, ...] = field(default_factory=tuple)
    result_store: SubagentResultStore | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_schemas", tuple(self.tool_schemas))
        object.__setattr__(
            self,
            "selected_skill_documents",
            tuple(self.selected_skill_documents),
        )
        object.__setattr__(
            self,
            "default_allowed_tools",
            _tool_names(self.default_allowed_tools),
        )


@dataclass(frozen=True, slots=True)
class SubagentToolExecutor:
    """Execute the model-visible subagent bridge through SubagentRuntime."""

    runtime_factory: Callable[[], SubagentRuntime]
    default_allowed_tools: tuple[str, ...] = field(default_factory=tuple)
    parent_tool_access_mode: ToolAccessMode = ToolAccessMode.READ_ONLY
    default_max_steps: int = 3
    max_steps_limit: int = 5
    max_retries: int = 0
    workflow_policy: SubagentOrchestrationPolicy = field(
        default_factory=SubagentOrchestrationPolicy
    )
    hook_context: HookRuntimeContext | None = None
    result_store: SubagentResultStore | None = None
    runtime_binding_factory: Callable[[], SubagentRuntimeBinding | None] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "default_allowed_tools",
            _tool_names(self.default_allowed_tools),
        )
        if not isinstance(self.parent_tool_access_mode, ToolAccessMode):
            object.__setattr__(
                self,
                "parent_tool_access_mode",
                ToolAccessMode(str(self.parent_tool_access_mode)),
            )
        if self.default_max_steps < 1:
            raise ValueError("default_max_steps must be at least 1")
        if self.max_steps_limit < self.default_max_steps:
            raise ValueError("max_steps_limit must be >= default_max_steps")
        if self.max_retries < 0 or self.max_retries > 1:
            raise ValueError("max_retries must be 0 or 1 in Phase 1")
        if not isinstance(self.workflow_policy, SubagentOrchestrationPolicy):
            raise ValueError("workflow_policy must be SubagentOrchestrationPolicy")

    def has_tool(self, name: str) -> bool:
        """Return whether this executor handles the requested tool name."""
        tool_name = name.strip()
        return tool_name in {
            SUBAGENT_TOOL_NAME,
            SUBAGENT_WORKFLOW_TOOL_NAME,
        } or (
            tool_name == READ_SUBAGENT_RESULT_TOOL_NAME
            and self._result_store() is not None
        )

    def schema(self) -> Mapping[str, object]:
        """Return the provider-neutral schema for the subagent bridge."""
        return subagent_tool_schema(max_steps_limit=self.max_steps_limit)

    def schemas(self) -> tuple[Mapping[str, object], ...]:
        """Return all provider-neutral schemas handled by this executor."""
        return (
            self.schema(),
            subagent_workflow_tool_schema(
                max_steps_limit=self.workflow_policy.max_child_steps,
            ),
            *(
                (read_subagent_result_tool_schema(),)
                if self._result_store() is not None
                else ()
            ),
        )

    def bind_runtime(
        self,
        *,
        capability_surface: "CapabilitySurface | None",
        native_tools: "NativeToolRegistry | None",
        mcp_tools: "McpToolExecutor | None",
        tool_schemas: Sequence[Mapping[str, object]],
        selected_skill_documents: Sequence["SkillDocument"] = (),
        activation: "RuntimeActivation | None" = None,
        parent_tool_access_policy: ToolAccessPolicy | None = None,
        result_store: SubagentResultStore | None = None,
    ) -> "SubagentToolExecutor":
        """Return an executor bound to the current parent turn state."""
        binding = SubagentRuntimeBinding(
            capability_surface=capability_surface,
            native_tools=native_tools,
            mcp_tools=mcp_tools,
            tool_schemas=tuple(tool_schemas),
            selected_skill_documents=tuple(selected_skill_documents),
            activation=activation,
            parent_tool_access_policy=parent_tool_access_policy,
            default_allowed_tools=_default_allowed_tools_from_schemas(tool_schemas),
            result_store=result_store or self._result_store(),
        )
        return replace(self, runtime_binding_factory=lambda: binding)

    def execute(self, request: ModelToolRequest) -> ToolExecutionResult:
        """Run one model-requested child agent and return a tool result."""
        if not request.is_ready():
            return _failure(
                request=request,
                code="subagent_tool_request_invalid",
                message="Subagent tool request requires a tool name and tool call id.",
            )
        tool_name = request.name.strip()
        if not self.has_tool(tool_name):
            return _failure(
                request=request,
                code="subagent_tool_not_allowed",
                message=(
                    "Subagent executor can only handle "
                    f"{SUBAGENT_TOOL_NAME}, {SUBAGENT_WORKFLOW_TOOL_NAME}, "
                    f"or {READ_SUBAGENT_RESULT_TOOL_NAME} requests."
                ),
            )
        if request.argument_error.strip():
            return _failure(
                request=request,
                code="subagent_tool_arguments_invalid",
                message=(
                    f"Subagent tool arguments invalid: {request.argument_error}"
                ),
                tool_name=tool_name,
            )

        if tool_name == READ_SUBAGENT_RESULT_TOOL_NAME:
            return self._execute_read_result(request)

        if tool_name == SUBAGENT_WORKFLOW_TOOL_NAME:
            return self._execute_workflow(request)

        runtime = self._runtime()
        if not runtime.is_ready():
            return _failure(
                request=request,
                code="subagent_runtime_not_ready",
                message="Subagent runtime is not ready.",
            )

        try:
            run_request = self._run_request(request.arguments, runtime)
        except ValueError as exc:
            return _failure(
                request=request,
                code="subagent_tool_arguments_invalid",
                message=str(exc),
            )

        try:
            observed_turns = []
            result = run_subagent(
                runtime,
                run_request,
                result_observer=lambda req, res, turn: observed_turns.append(
                    (req, res, turn)
                ),
            )
        except Exception as exc:
            return _failure(
                request=request,
                code="subagent_tool_failed",
                message=f"Subagent tool failed: {exc}",
            )

        review = review_subagent_result(run_request, result)
        result_ref, persist_error = self._save_observed_result(
            observed_turns,
            run_request,
            result,
            review,
        )
        if persist_error is not None:
            return self._unpersisted_result(request, persist_error)
        retry_result = None
        retry_review = None
        retry_instruction = ""
        if (
            review.retryable
            and self.max_retries > 0
            and run_request.tool_access_mode != ToolAccessMode.WORKSPACE_WRITE
        ):
            retry_instruction = review.retry_instruction
            retry_request = _retry_request(run_request, result, review)
            if retry_request is not None:
                retry_turns = []
                retry_result = run_subagent(
                    runtime,
                    retry_request,
                    result_observer=lambda req, res, turn: retry_turns.append(
                        (req, res, turn)
                    ),
                )
                retry_result = _merge_retry_result_refs(result, retry_result)
                retry_review = review_subagent_result(retry_request, retry_result)
                result = retry_result
                review = retry_review
                result_ref, persist_error = self._save_observed_result(
                    retry_turns,
                    retry_request,
                    retry_result,
                    retry_review,
                )
                if persist_error is not None:
                    return self._unpersisted_result(request, persist_error)

        model_result = ModelToolResult(
            name=SUBAGENT_TOOL_NAME,
            request_id=request.id,
            content=_result_content(result, review, result_ref=result_ref),
            refs=_refs_with_result_ref(result.evidence_refs, result_ref),
            is_error=not result.is_success() or not review.is_accepted(),
        )
        event_kind = (
            "subagent_tool_completed"
            if result.is_success() and review.is_accepted()
            else "subagent_tool_failed"
        )
        error = result.error if not result.is_success() else None
        if error is None and not review.is_accepted():
            error = ErrorInfo(
                code=f"subagent_result_{review.status.value}",
                message=review.summary,
                refs=(f"subagent_run_id={result.run_id}", *review.refs()),
            )
        if retry_result is not None and retry_review is not None:
            retry_detail = {
                "attempts": 2,
                "instruction": retry_instruction,
                "status": retry_review.status.value,
            }
            model_result = ModelToolResult(
                name=SUBAGENT_TOOL_NAME,
                request_id=request.id,
                content=_result_content(
                    result,
                    review,
                    retry_detail,
                    result_ref=result_ref,
                ),
                refs=_refs_with_result_ref(result.evidence_refs, result_ref),
                is_error=not result.is_success() or not review.is_accepted(),
            )
        return ToolExecutionResult(
            request=request,
            model_result=model_result,
            error=error,
            events=(
                RuntimeEvent(
                    kind=event_kind,
                    summary=f"Subagent tool returned status={result.status.value}.",
                    refs=_refs_with_result_ref(result.evidence_refs, result_ref),
                ),
                RuntimeEvent(
                    kind="subagent_result_reviewed",
                    summary=review.summary,
                    refs=(
                        f"subagent_run_id={result.run_id}",
                        *review.refs(),
                        *result.evidence_refs,
                    ),
                ),
            ),
        )

    def _execute_read_result(self, request: ModelToolRequest) -> ToolExecutionResult:
        store = self._result_store()
        if store is None:
            return _failure(
                request=request,
                code="subagent_result_store_unavailable",
                message="Subagent result store is not available in this session.",
                tool_name=READ_SUBAGENT_RESULT_TOOL_NAME,
            )
        try:
            result_ref = _required_text(request.arguments, "result_ref")
            include_steps = _bool_argument(
                request.arguments.get("include_steps"),
                default=True,
                field_name="include_steps",
            )
            step_index = _optional_int_argument(
                request.arguments.get("step_index"),
                minimum=1,
                field_name="step_index",
            )
        except ValueError as exc:
            return _failure(
                request=request,
                code="subagent_result_read_arguments_invalid",
                message=str(exc),
                tool_name=READ_SUBAGENT_RESULT_TOOL_NAME,
            )
        record = store.load(result_ref)
        if record is None:
            return _failure(
                request=request,
                code="subagent_result_not_found",
                message=f"Subagent result handle was not found: {result_ref}",
                tool_name=READ_SUBAGENT_RESULT_TOOL_NAME,
            )
        payload = read_subagent_record_payload(
            record,
            include_steps=include_steps,
            step_index=step_index,
        )
        return ToolExecutionResult(
            request=request,
            model_result=ModelToolResult(
                name=READ_SUBAGENT_RESULT_TOOL_NAME,
                request_id=request.id,
                content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                refs=(f"subagent_result_ref={result_ref}",),
            ),
            events=(
                RuntimeEvent(
                    kind="subagent_result_read",
                    summary="Subagent result details read on demand.",
                    refs=(f"subagent_result_ref={result_ref}",),
                ),
            ),
        )

    def _execute_workflow(self, request: ModelToolRequest) -> ToolExecutionResult:
        runtime = self._runtime()
        if not runtime.is_ready():
            return _failure(
                request=request,
                code="subagent_runtime_not_ready",
                message="Subagent runtime is not ready.",
                tool_name=SUBAGENT_WORKFLOW_TOOL_NAME,
            )

        try:
            assignments = self._workflow_assignments(request.arguments)
            reflector_assignment = self._workflow_reflector(request.arguments)
            plan_summary = _optional_text(request.arguments.get("plan_summary")) or ""
        except ValueError as exc:
            return _failure(
                request=request,
                code="subagent_workflow_arguments_invalid",
                message=str(exc),
                tool_name=SUBAGENT_WORKFLOW_TOOL_NAME,
            )

        try:
            observed_turns = []
            result = run_subagent_orchestration(
                runtime,
                assignments,
                plan_summary=plan_summary,
                reflector_assignment=reflector_assignment,
                policy=self.workflow_policy,
                result_observer=lambda req, res, turn: observed_turns.append(
                    (req, res, turn)
                ),
            )
        except Exception as exc:
            return _failure(
                request=request,
                code="subagent_workflow_failed",
                message=f"Subagent workflow failed: {exc}",
                tool_name=SUBAGENT_WORKFLOW_TOOL_NAME,
            )

        result_refs, persist_errors = self._save_workflow_observed_results(
            observed_turns,
            result,
        )
        model_result = ModelToolResult(
            name=SUBAGENT_WORKFLOW_TOOL_NAME,
            request_id=request.id,
            content=_workflow_result_content(result, result_refs=result_refs),
            refs=(
                *_workflow_result_refs(result),
                *tuple(f"subagent_result_ref={ref}" for ref in result_refs.values()),
                *result.artifact_refs,
                *result.evidence_refs,
            ),
            is_error=not result.is_success(),
        )
        error = None
        if not result.is_success():
            error = ErrorInfo(
                code=f"subagent_workflow_{result.status.value}",
                message=result.execution_summary,
                refs=(
                    f"status={result.status.value}",
                    f"blocking_gaps={len(result.blocking_gaps)}",
                    *result.blocking_gaps,
                ),
            )
        event_kind = (
            "subagent_workflow_completed"
            if result.is_success()
            else "subagent_workflow_failed"
        )
        hook_outcome = _emit_workflow_hook(self.hook_context, result)
        hook_event = _workflow_hook_event(hook_outcome)
        return ToolExecutionResult(
            request=request,
            model_result=model_result,
            error=error,
            events=(
                RuntimeEvent(
                    kind=event_kind,
                    summary=result.execution_summary,
                    refs=(
                        f"status={result.status.value}",
                        f"child_runs={len(result.assignment_runs)}",
                        f"accepted_results={len(result.accepted_results)}",
                        f"blocking_gaps={len(result.blocking_gaps)}",
                    ),
                ),
                *tuple(
                    RuntimeEvent(
                        kind=error.code,
                        summary=error.message,
                        refs=error.refs,
                    )
                    for error in persist_errors
                ),
                *((hook_event,) if hook_event is not None else ()),
            ),
        )

    def _save_observed_result(
        self,
        observed_turns,
        request: SubagentRunRequest,
        result: SubagentRunResult,
        review: SubagentResultReview | None,
    ) -> tuple[str, ErrorInfo | None]:
        store = self._result_store()
        if store is None:
            return "", None
        turn = observed_turns[-1][2] if observed_turns else None
        try:
            record = store.save(
                request=request,
                result=result,
                turn=turn,
                review=review,
            )
        except OSError as exc:
            error = ErrorInfo(
                code="subagent_result_unpersisted",
                message=f"Subagent completed but its result could not be saved: {exc}",
                refs=(f"subagent_run_id={result.run_id}",),
            )
            return "", error
        return record.ref, None

    def _unpersisted_result(
        self,
        request: ModelToolRequest,
        error: ErrorInfo,
    ) -> ToolExecutionResult:
        model_result = ModelToolResult(
            name=SUBAGENT_TOOL_NAME,
            request_id=request.id,
            content=json.dumps(
                {
                    "status": SubagentRunStatus.FAILED.value,
                    "summary": error.message,
                    "error": {
                        "code": error.code,
                        "message": error.message,
                        "refs": list(error.refs),
                    },
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            refs=error.refs,
            is_error=True,
        )
        return ToolExecutionResult(
            request=request,
            model_result=model_result,
            error=error,
            events=(
                RuntimeEvent(
                    kind=error.code,
                    summary=error.message,
                    refs=error.refs,
                ),
            ),
        )

    def _save_workflow_observed_results(
        self,
        observed_turns,
        result: SubagentWorkflowResult,
    ) -> tuple[dict[str, str], tuple[ErrorInfo, ...]]:
        store = self._result_store()
        if store is None:
            return {}, ()
        observed_by_run_id = {
            observed_result.run_id: (observed_request, observed_result, observed_turn)
            for observed_request, observed_result, observed_turn in observed_turns
        }
        refs: dict[str, str] = {}
        errors: list[ErrorInfo] = []
        for run in result.assignment_runs:
            observed = observed_by_run_id.get(run.result.run_id)
            if observed is None:
                request = run.request
                turn = None
            else:
                request, _observed_result, turn = observed
            try:
                record = store.save(
                    request=request,
                    result=run.result,
                    turn=turn,
                    review=run.review,
                )
            except OSError as exc:
                errors.append(
                    ErrorInfo(
                        code="subagent_result_unpersisted",
                        message=(
                            "Subagent workflow child completed but its result could "
                            f"not be saved: {exc}"
                        ),
                        refs=(f"subagent_run_id={run.result.run_id}",),
                    )
                )
                continue
            refs[run.result.run_id] = record.ref
        return refs, tuple(errors)

    def _run_request(
        self,
        arguments: Mapping[str, object],
        runtime: SubagentRuntime,
    ) -> SubagentRunRequest:
        goal = _required_text(arguments, "goal")
        allowed_tools = _allowed_tools_argument(
            arguments,
            default_allowed_tools=self._default_allowed_tools(runtime),
        )
        max_steps = _int_argument(
            arguments.get("max_steps"),
            default=self.default_max_steps,
            minimum=1,
            maximum=self.max_steps_limit,
            field_name="max_steps",
        )
        activation = runtime.activation
        return SubagentRunRequest(
            goal=goal,
            input_context=_optional_text(arguments.get("input_context")) or "",
            output_contract=_optional_text(arguments.get("output_contract")),
            acceptance_criteria=_string_list(arguments.get("acceptance_criteria")),
            allowed_tools=allowed_tools,
            tool_access_mode=self._tool_access_mode(
                _optional_text(arguments.get("tool_access_mode"))
            ),
            max_steps=max_steps,
            parent_session_id=activation.session_id if activation else None,
            parent_activation_id=activation.activation_id if activation else None,
        )

    def _workflow_assignments(
        self,
        arguments: Mapping[str, object],
    ) -> tuple[SubagentAssignment, ...]:
        values = arguments.get("assignments")
        if not isinstance(values, list):
            raise ValueError("run_subagent_workflow requires assignments list.")
        assignments: list[SubagentAssignment] = []
        for index, value in enumerate(values, start=1):
            assignments.append(
                self._workflow_assignment(
                    value,
                    default_assignment_id=f"assignment_{index}",
                    stage=SubagentAssignmentStage.EXECUTE,
                )
            )
        if not assignments:
            raise ValueError("run_subagent_workflow requires at least one assignment.")
        return tuple(assignments)

    def _workflow_reflector(
        self,
        arguments: Mapping[str, object],
    ) -> SubagentAssignment | None:
        value = arguments.get("reflector")
        if value is None:
            return None
        return self._workflow_assignment(
            value,
            default_assignment_id="reflector",
            stage=SubagentAssignmentStage.REFLECT,
        )

    def _workflow_assignment(
        self,
        value: object,
        *,
        default_assignment_id: str,
        stage: SubagentAssignmentStage,
    ) -> SubagentAssignment:
        if not isinstance(value, Mapping):
            raise ValueError("workflow assignments must be objects.")
        assignment_id = (
            _optional_text(value.get("assignment_id")) or default_assignment_id
        )
        goal = _required_text(value, "goal")
        allowed_tools = _allowed_tools_argument(
            value,
            default_allowed_tools=self._default_allowed_tools(),
        )
        max_steps = _int_argument(
            value.get("max_steps"),
            default=min(6, self.workflow_policy.max_child_steps),
            minimum=1,
            maximum=self.workflow_policy.max_child_steps,
            field_name="max_steps",
        )
        return SubagentAssignment(
            assignment_id=assignment_id,
            goal=goal,
            input_context=_optional_text(value.get("input_context")) or "",
            output_contract=_optional_text(value.get("output_contract")),
            acceptance_criteria=_string_list(
                value.get("acceptance_criteria"),
                field_name="acceptance_criteria",
            ),
            allowed_tools=allowed_tools,
            tool_access_mode=self._tool_access_mode(
                _optional_text(value.get("tool_access_mode"))
            ),
            max_steps=max_steps,
            depends_on=_string_list(value.get("depends_on"), field_name="depends_on"),
            stage=stage,
        )

    def _tool_access_mode(self, requested: str | None) -> ToolAccessMode:
        if requested is None:
            return ToolAccessMode.READ_ONLY
        try:
            mode = ToolAccessMode(requested)
        except ValueError as exc:
            raise ValueError(f"Unsupported tool_access_mode: {requested}") from exc
        if (
            mode == ToolAccessMode.WORKSPACE_WRITE
            and self._parent_tool_access_mode() != ToolAccessMode.WORKSPACE_WRITE
        ):
            raise ValueError(
                "Subagent workspace_write requires parent workspace_write access."
            )
        return mode

    def _runtime(self) -> SubagentRuntime:
        runtime = self.runtime_factory()
        binding = self._runtime_binding()
        if binding is None:
            return runtime
        activation = binding.activation
        return replace(
            runtime,
            workspace=activation.workspace if activation is not None else runtime.workspace,
            profile=activation.profile if activation is not None else runtime.profile,
            capability_surface=binding.capability_surface,
            native_tools=binding.native_tools,
            mcp_tools=binding.mcp_tools,
            tool_schemas=binding.tool_schemas,
            selected_skill_documents=binding.selected_skill_documents,
            parent_tool_access_policy=binding.parent_tool_access_policy,
            activation=activation,
            context_snapshot=(
                None if activation is not None else runtime.context_snapshot
            ),
        )

    def _runtime_binding(self) -> SubagentRuntimeBinding | None:
        if self.runtime_binding_factory is None:
            return None
        return self.runtime_binding_factory()

    def _result_store(self) -> SubagentResultStore | None:
        binding = self._runtime_binding()
        if binding is not None and binding.result_store is not None:
            return binding.result_store
        return self.result_store

    def _default_allowed_tools(
        self,
        runtime: SubagentRuntime | None = None,
    ) -> tuple[str, ...]:
        binding = self._runtime_binding()
        if binding is not None:
            return binding.default_allowed_tools
        if runtime is not None:
            names = _default_allowed_tools_from_schemas(runtime.tool_schemas)
            if names:
                return names
        return self.default_allowed_tools

    def _parent_tool_access_mode(self) -> ToolAccessMode:
        binding = self._runtime_binding()
        if binding is not None and binding.parent_tool_access_policy is not None:
            return binding.parent_tool_access_policy.mode
        return self.parent_tool_access_mode


def subagent_tool_schema(max_steps_limit: int = 5) -> Mapping[str, object]:
    """Return the provider-neutral schema for explicit subagent delegation."""
    return {
        "name": SUBAGENT_TOOL_NAME,
        "description": DEFAULT_SUBAGENT_TOOL_DESCRIPTION,
        "input_schema": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": (
                        "The concrete delegated objective. Make it self-contained, "
                        "verifiable, and narrow enough for one bounded child run."
                    ),
                },
                "input_context": {
                    "type": "string",
                    "description": (
                        "Minimal context the child agent needs. Include only the "
                        "facts, paths, constraints, and evidence the child run "
                        "cannot infer on its own."
                    ),
                },
                "output_contract": {
                    "type": "string",
                    "description": (
                        "Expected shape of the child result. If omitted, Deepmate "
                        "uses a default parent-friendly contract."
                    ),
                },
                "acceptance_criteria": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Concrete criteria the child result must satisfy. "
                        "Deepmate injects them into the bounded child prompt."
                    ),
                },
                "allowed_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Tool names the child agent may use. If omitted, "
                        "Deepmate uses the current read-only default set; use [] "
                        "for a no-tool child run."
                    ),
                },
                "tool_access_mode": {
                    "type": "string",
                    "enum": [mode.value for mode in ToolAccessMode],
                    "description": (
                        "Use workspace_write only when writes are necessary and the "
                        "parent runtime already allows workspace writes."
                    ),
                },
                "max_steps": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": max_steps_limit,
                    "description": "Maximum child agent steps.",
                },
            },
            "required": ["goal"],
        },
    }


def subagent_workflow_tool_schema(max_steps_limit: int = 12) -> Mapping[str, object]:
    """Return the provider-neutral schema for bounded subagent workflows."""
    assignment_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "assignment_id": {
                "type": "string",
                "description": "Stable id used for dependency summaries.",
            },
            "goal": {
                "type": "string",
                "description": "Concrete delegated objective for this child run.",
            },
            "input_context": {
                "type": "string",
                "description": "Minimal facts, paths, constraints, and evidence.",
            },
            "output_contract": {
                "type": "string",
                "description": "Expected result shape for parent merge.",
            },
            "acceptance_criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Concrete checks this assignment must satisfy.",
            },
            "allowed_tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Tool names allowed for this assignment. If omitted, Deepmate "
                    "uses the current read-only default set; use [] for no tools."
                ),
            },
            "tool_access_mode": {
                "type": "string",
                "enum": [mode.value for mode in ToolAccessMode],
                "description": (
                    "Use workspace_write only for assignments that must write files "
                    "and only when the parent runtime allows writes."
                ),
            },
            "max_steps": {
                "type": "integer",
                "minimum": 1,
                "maximum": max_steps_limit,
                "description": "Maximum child agent steps for this assignment.",
            },
            "depends_on": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Assignment ids that must be accepted first.",
            },
        },
        "required": ["goal"],
    }
    return {
        "name": SUBAGENT_WORKFLOW_TOOL_NAME,
        "description": DEFAULT_SUBAGENT_WORKFLOW_TOOL_DESCRIPTION,
        "input_schema": {
            "type": "object",
            "properties": {
                "plan_summary": {
                    "type": "string",
                    "description": (
                        "Short parent plan that explains why these assignments "
                        "belong together."
                    ),
                },
                "assignments": {
                    "type": "array",
                    "items": assignment_schema,
                    "description": (
                        "Ordered child assignments. Deepmate enforces dependency "
                        "readiness, child budgets, review, and bounded revision."
                    ),
                },
                "reflector": {
                    **assignment_schema,
                    "description": (
                        "Optional final read-only review assignment. Include for "
                        "multi-assignment, workspace-write, or high-completeness work."
                    ),
                },
            },
            "required": ["assignments"],
        },
    }


def read_subagent_result_tool_schema() -> Mapping[str, object]:
    """Return the provider-neutral schema for subagent result retrieval."""
    return {
        "name": READ_SUBAGENT_RESULT_TOOL_NAME,
        "description": DEFAULT_READ_SUBAGENT_RESULT_DESCRIPTION,
        "input_schema": {
            "type": "object",
            "properties": {
                "result_ref": {
                    "type": "string",
                    "description": "The result_handle returned by run_subagent or run_subagent_workflow.",
                },
                "include_steps": {
                    "type": "boolean",
                    "description": "Include compact per-step child details. Defaults to true.",
                },
                "step_index": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional one-based step index to read only one child step.",
                },
            },
            "required": ["result_ref"],
            "additionalProperties": False,
        },
    }


def _result_content(
    result: SubagentRunResult,
    review: SubagentResultReview | None = None,
    retry: Mapping[str, object] | None = None,
    result_ref: str = "",
) -> str:
    payload: dict[str, object] = {
        "run_id": result.run_id,
        "status": result.status.value,
        "summary": result.summary,
    }
    if result_ref:
        payload["result_handle"] = result_ref
    if result.artifact_refs:
        payload["artifact_refs"] = list(result.artifact_refs)
    if result.evidence_refs:
        payload["evidence_refs"] = list(result.evidence_refs)
    if result.usage is not None:
        payload["usage"] = _usage_payload(result.usage)
    if result.error is not None:
        payload["error"] = {
            "code": result.error.code,
            "message": result.error.message,
            "refs": list(result.error.refs),
        }
    if review is not None:
        payload["review"] = review.to_payload()
    if retry is not None:
        payload["retry"] = dict(retry)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _workflow_result_content(
    result: SubagentWorkflowResult,
    result_refs: Mapping[str, str] | None = None,
) -> str:
    refs = result_refs or {}
    payload: dict[str, object] = {
        "status": result.status.value,
        "execution_summary": result.execution_summary,
        "accepted_results": [
            _run_result_payload(item, refs.get(item.run_id, ""))
            for item in result.accepted_results
        ],
        "blocking_gaps": list(result.blocking_gaps),
    }
    if result.plan_summary:
        payload["plan_summary"] = result.plan_summary
    if result.reflector_summary:
        payload["reflector_summary"] = result.reflector_summary
    if result.non_accepted_reviews:
        payload["non_accepted_reviews"] = [
            review.to_payload() for review in result.non_accepted_reviews
        ]
    if result.revised_assignment_ids:
        payload["revised_assignment_ids"] = list(result.revised_assignment_ids)
    if result.artifact_refs:
        payload["artifact_refs"] = list(result.artifact_refs)
    if result.evidence_refs:
        payload["evidence_refs"] = list(result.evidence_refs)
    if result.usage is not None:
        payload["usage"] = _usage_payload(result.usage)
    payload["assignment_runs"] = [
        {
            "assignment_id": run.assignment.assignment_id,
            "attempt": run.attempt,
            "stage": run.assignment.stage.value,
            "run_id": run.result.run_id,
            **(
                {"result_handle": refs[run.result.run_id]}
                if run.result.run_id in refs
                else {}
            ),
            "status": run.result.status.value,
            "review_status": run.review.status.value,
        }
        for run in result.assignment_runs
    ]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _workflow_result_refs(result: SubagentWorkflowResult) -> tuple[str, ...]:
    return (
        f"status={result.status.value}",
        f"child_runs={len(result.assignment_runs)}",
        f"accepted_results={len(result.accepted_results)}",
        f"blocking_gaps={len(result.blocking_gaps)}",
        f"revised={len(result.revised_assignment_ids)}",
        f"artifact_refs={len(result.artifact_refs)}",
        f"evidence_refs={len(result.evidence_refs)}",
    )


def _run_result_payload(
    result: SubagentRunResult,
    result_ref: str = "",
) -> Mapping[str, object]:
    payload: dict[str, object] = {
        "run_id": result.run_id,
        "status": result.status.value,
        "summary": result.summary,
    }
    if result_ref:
        payload["result_handle"] = result_ref
    if result.artifact_refs:
        payload["artifact_refs"] = list(result.artifact_refs)
    if result.evidence_refs:
        payload["evidence_refs"] = list(result.evidence_refs)
    if result.usage is not None:
        payload["usage"] = _usage_payload(result.usage)
    return payload


def _refs_with_result_ref(refs: tuple[str, ...], result_ref: str) -> tuple[str, ...]:
    if not result_ref:
        return refs
    return _unique_refs((*refs, f"subagent_result_ref={result_ref}"))


def _usage_payload(usage: TokenUsage) -> Mapping[str, int]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_hit_input_tokens": usage.cache_hit_input_tokens,
        "cache_miss_input_tokens": usage.cache_miss_input_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
    }


def _unique_refs(refs: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for ref in refs:
        text = ref.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return tuple(output)


def _failure(
    request: ModelToolRequest,
    code: str,
    message: str,
    tool_name: str = SUBAGENT_TOOL_NAME,
) -> ToolExecutionResult:
    refs = (tool_name,)
    error = ErrorInfo(code=code, message=message, refs=refs)
    request_id = _request_id(request)
    model_result = (
        ModelToolResult(
            name=tool_name,
            request_id=request_id,
            content=message,
            refs=refs,
            is_error=True,
        )
        if request_id
        else None
    )
    return ToolExecutionResult(
        request=request,
        model_result=model_result,
        error=error,
        events=(RuntimeEvent(kind=code, summary=message, refs=refs),),
    )


def _retry_request(
    request: SubagentRunRequest,
    result: SubagentRunResult,
    review: SubagentResultReview,
) -> SubagentRunRequest | None:
    if not review.retryable or not review.retry_instruction:
        return None
    output_contract = request.output_contract or ""
    pieces = [output_contract.strip(), ""]
    if output_contract.strip():
        pieces.append("Retry instruction:")
    pieces.append(review.retry_instruction)
    pieces.append(
        "Return a concise answer with explicit evidence_refs if any evidence was used."
    )
    input_context = request.input_context.strip()
    if input_context:
        input_context = (
            f"{input_context}\n\nFirst pass was incomplete. {review.retry_instruction}"
        )
    else:
        input_context = f"First pass was incomplete. {review.retry_instruction}"
    return replace(
        request,
        run_id=_retry_run_id(result.run_id),
        input_context=input_context,
        output_contract="\n".join(piece for piece in pieces if piece).strip() or None,
    )


def _merge_retry_result_refs(
    first_result: SubagentRunResult,
    retry_result: SubagentRunResult,
) -> SubagentRunResult:
    return replace(
        retry_result,
        artifact_refs=(*first_result.artifact_refs, *retry_result.artifact_refs),
        evidence_refs=(*first_result.evidence_refs, *retry_result.evidence_refs),
    )


def _retry_run_id(run_id: str) -> str:
    base = run_id.strip() or "subagent-run"
    if len(base) > 80:
        base = base[:80].rstrip(".-") or "subagent-run"
    return f"{base}-retry1"


def _emit_workflow_hook(
    hook_context: HookRuntimeContext | None,
    result: SubagentWorkflowResult,
) -> HookOutcome:
    if hook_context is None:
        return HookOutcome()
    return hook_context.emit(
        HookEnvelope(
            event_name=HookEvent.SUBAGENT_WORKFLOW_END,
            actor=HookActor.MAIN,
            payload={
                "tool_name": SUBAGENT_WORKFLOW_TOOL_NAME,
                "tool_source": "subagent",
                "status": result.status.value,
                "summary": result.execution_summary,
                "child_runs": len(result.assignment_runs),
                "accepted_results": len(result.accepted_results),
                "blocking_gaps": len(result.blocking_gaps),
                "artifact_refs": len(result.artifact_refs),
                "evidence_refs": len(result.evidence_refs),
            },
            source_refs=(
                f"status={result.status.value}",
                f"child_runs={len(result.assignment_runs)}",
                f"accepted_results={len(result.accepted_results)}",
                f"blocking_gaps={len(result.blocking_gaps)}",
                *_workflow_result_refs(result),
                *result.artifact_refs[:8],
                *result.evidence_refs[:8],
                *hook_context.trace_refs(),
            ),
        )
    )


def _workflow_hook_event(outcome: HookOutcome) -> RuntimeEvent | None:
    if not outcome.action_results and outcome.directive == HookDirective.CONTINUE:
        return None
    return RuntimeEvent(
        kind="subagent_workflow_hook_observed",
        summary="Subagent workflow hook evaluated.",
        refs=(
            f"hook_directive={outcome.directive.value}",
            *outcome.refs,
            *tuple(f"hook_warning={warning}" for warning in outcome.warnings),
        ),
    )


def _required_text(arguments: Mapping[str, object], key: str) -> str:
    text = _optional_text(arguments.get(key))
    if text is None:
        raise ValueError(f"Subagent tool argument requires non-empty {key}.")
    return text


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Subagent text arguments must be strings.")
    text = value.strip()
    return text or None


def _request_id(request: ModelToolRequest) -> str:
    if not isinstance(request.id, str):
        return ""
    return request.id.strip()


def _string_list(value: object, field_name: str = "allowed_tools") -> tuple[str, ...]:
    return _string_tuple(value, field_name=field_name)


def _allowed_tools_argument(
    arguments: Mapping[str, object],
    *,
    default_allowed_tools: tuple[str, ...],
) -> tuple[str, ...]:
    if "allowed_tools" not in arguments:
        return default_allowed_tools
    return _tool_names(_string_list(arguments.get("allowed_tools")))


def _string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError(f"{field_name} must contain only strings.")
            items.append(item)
        return tuple(items)
    raise ValueError(f"{field_name} must be a list of strings.")


def _bool_argument(value: object, *, default: bool, field_name: str) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean.")
    return value


def _optional_int_argument(
    value: object,
    *,
    minimum: int,
    field_name: str,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}.")
    return value


def _tool_names(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        name = value.strip()
        if (
            not name
            or name
            in {
                SUBAGENT_TOOL_NAME,
                SUBAGENT_WORKFLOW_TOOL_NAME,
                READ_SUBAGENT_RESULT_TOOL_NAME,
            }
            or name in seen
        ):
            continue
        seen.add(name)
        output.append(name)
    return tuple(output)


def _default_allowed_tools_from_schemas(
    schemas: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    return _tool_names(tuple(_schema_name(schema) for schema in schemas))


def _schema_name(schema: Mapping[str, object]) -> str:
    name = schema.get("name")
    if isinstance(name, str):
        return name.strip()
    if schema.get("type") == "function":
        function = schema.get("function")
        if isinstance(function, Mapping):
            function_name = function.get("name")
            if isinstance(function_name, str):
                return function_name.strip()
    return ""


def _int_argument(
    value: object,
    default: int,
    minimum: int,
    maximum: int,
    field_name: str,
) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    if value < minimum or value > maximum:
        raise ValueError(f"{field_name} must be between {minimum} and {maximum}.")
    return value
