"""Minimal child runtime for Deepmate subagents."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from deepmate.capabilities import CapabilitySurface
from deepmate.context import ContextWarning, ProfileContextSnapshot
from deepmate.domain import (
    CapabilityKind,
    ErrorInfo,
    Message,
    MessageRole,
    ProfileRef,
    RuntimeEvent,
)
from deepmate.mcp import McpToolExecutionResult, McpToolExecutor
from deepmate.providers import (
    ModelCapabilities,
    ModelProvider,
    ModelToolRequest,
    ModelToolResult,
    TokenUsage,
)
from deepmate.runtime import (
    ConversationBudgetPolicy,
    ProviderRetryPolicy,
    ToolAccessPolicy,
    ToolRepairPolicy,
    UserTurnResult,
    run_user_turn,
)
from deepmate.runtime.agent_loop import HistorySink
from deepmate.skills import SkillDocument
from deepmate.tools import NativeToolRegistry
from deepmate.trace import TraceEvent, TraceRecorder

from deepmate.subagents.types import (
    SubagentRunRequest,
    SubagentRunResult,
    SubagentRunStatus,
)

if TYPE_CHECKING:
    from deepmate.runtime import RuntimeActivation

SubagentResultObserver = Callable[[SubagentRunRequest, SubagentRunResult, UserTurnResult], None]


DEFAULT_OUTPUT_CONTRACT = (
    "Return a concise result the parent agent can merge: "
    "1) findings, 2) evidence_refs, 3) artifact_refs if any, 4) blockers if any."
)
RECURSIVE_SUBAGENT_TOOLS = frozenset(
    ("run_subagent", "run_subagent_workflow", "read_subagent_result")
)


@dataclass(frozen=True, slots=True)
class SubagentRuntime:
    """Run one bounded child agent using the existing agent loop."""

    provider: ModelProvider
    workspace: str | Path
    profile: ProfileRef
    model: str
    capability_surface: CapabilitySurface | None = None
    native_tools: NativeToolRegistry | None = None
    mcp_tools: McpToolExecutor | None = None
    tool_schemas: tuple[Mapping[str, object], ...] = field(default_factory=tuple)
    selected_skill_documents: tuple[SkillDocument, ...] = field(default_factory=tuple)
    parent_tool_access_policy: ToolAccessPolicy | None = None
    activation: "RuntimeActivation | None" = None
    context_snapshot: ProfileContextSnapshot | None = None
    conversation_budget_policy: ConversationBudgetPolicy | None = None
    provider_retry_policy: ProviderRetryPolicy | None = None
    tool_repair_policy: ToolRepairPolicy | None = None
    options: Mapping[str, object] = field(default_factory=dict)
    model_capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    trace_recorder: TraceRecorder | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace", Path(self.workspace).resolve())
        object.__setattr__(self, "model", self.model.strip())
        object.__setattr__(self, "tool_schemas", tuple(self.tool_schemas))
        object.__setattr__(
            self,
            "selected_skill_documents",
            tuple(self.selected_skill_documents),
        )
        object.__setattr__(self, "options", dict(self.options or {}))
        object.__setattr__(
            self,
            "model_capabilities",
            self.model_capabilities.normalized(),
        )
        if self.activation is not None and self.context_snapshot is not None:
            if self.activation.context_snapshot != self.context_snapshot:
                raise ValueError("activation and context_snapshot must not disagree")

    def is_ready(self) -> bool:
        """Return whether the runtime has the minimum dependencies."""
        return bool(self.model and self.profile.is_ready() and str(self.workspace).strip())

    def run(
        self,
        request: SubagentRunRequest,
        warning_sink: Callable[[ContextWarning], None] | None = None,
        history_sink: HistorySink | None = None,
        result_observer: SubagentResultObserver | None = None,
    ) -> SubagentRunResult:
        """Run one subagent request and summarize the child turn result."""
        run_id = request.run_id or uuid4().hex
        if not self.is_ready():
            return _failed_result(
                run_id=run_id,
                message="Subagent runtime is not ready.",
                refs=("subagent_runtime_not_ready",),
            )
        if not request.is_ready():
            return _failed_result(
                run_id=run_id,
                message="Subagent request requires a non-empty goal and max_steps >= 1.",
                refs=("subagent_request_invalid",),
            )

        allowed = set(request.allowed_tools) - RECURSIVE_SUBAGENT_TOOLS
        filtered_surface = _filtered_surface(self.capability_surface, allowed)
        filtered_native_tools = _filtered_native_tools(self.native_tools, allowed)
        filtered_mcp_tools = _filtered_mcp_tools(self.mcp_tools, allowed)
        filtered_tool_schemas = _filtered_tool_schemas(self.tool_schemas, allowed)

        _record_trace(
            self.trace_recorder,
            "subagent_run_started",
            "Subagent run started.",
            refs=_run_refs(run_id, request, extra=(f"allowed_tools={len(allowed)}",)),
        )
        try:
            turn = run_user_turn(
                provider=self.provider,
                workspace=self.workspace,
                profile=self.profile,
                messages=(
                    Message(
                        role=MessageRole.USER,
                        content=_child_prompt(request),
                    ),
                ),
                model=self.model,
                capability_surface=filtered_surface,
                native_tools=filtered_native_tools,
                mcp_tools=filtered_mcp_tools,
                tool_access_policy=_child_tool_access_policy(
                    self.parent_tool_access_policy,
                    request.tool_access_mode,
                ),
                tool_schemas=filtered_tool_schemas,
                selected_skill_documents=_filtered_selected_skills(
                    self.selected_skill_documents,
                    allowed,
                ),
                activation=self.activation,
                context_snapshot=self.context_snapshot,
                conversation_budget_policy=self.conversation_budget_policy,
                provider_retry_policy=self.provider_retry_policy,
                tool_repair_policy=self.tool_repair_policy,
                options=self.options,
                model_capabilities=self.model_capabilities,
                max_steps=request.max_steps,
                trace_recorder=self.trace_recorder,
                warning_sink=warning_sink,
                history_sink=history_sink,
            )
        except Exception as exc:
            error = ErrorInfo(
                code="subagent_run_failed",
                message=f"Subagent run failed: {exc}",
                refs=_run_refs(run_id, request),
            )
            _record_trace(
                self.trace_recorder,
                "subagent_run_failed",
                error.message,
                refs=error.refs,
            )
            return SubagentRunResult(
                run_id=run_id,
                status=SubagentRunStatus.FAILED,
                summary=error.message,
                evidence_refs=error.refs,
                error=error,
            )

        result = _turn_result(run_id, request, turn)
        if result_observer is not None:
            result_observer(request, result, turn)
        _record_trace(
            self.trace_recorder,
            "subagent_run_finished",
            "Subagent run finished.",
            refs=_run_refs(
                run_id,
                request,
                extra=(
                    f"status={result.status.value}",
                    f"artifact_refs={len(result.artifact_refs)}",
                    f"evidence_refs={len(result.evidence_refs)}",
                ),
            ),
        )
        return result


def run_subagent(
    runtime: SubagentRuntime,
    request: SubagentRunRequest,
    warning_sink: Callable[[ContextWarning], None] | None = None,
    history_sink: HistorySink | None = None,
    result_observer: SubagentResultObserver | None = None,
) -> SubagentRunResult:
    """Bridge one parent request into the subagent runtime."""
    return runtime.run(
        request=request,
        warning_sink=warning_sink,
        history_sink=history_sink,
        result_observer=result_observer,
    )


def _child_prompt(request: SubagentRunRequest) -> str:
    output_contract = request.output_contract or DEFAULT_OUTPUT_CONTRACT
    lines = [
        "You are a bounded child agent. Focus only on this delegated work.",
        "",
        "Goal:",
        request.goal,
        "",
        "Input context:",
        request.input_context or "(none)",
    ]
    if request.acceptance_criteria:
        lines.extend(("", "Acceptance criteria:"))
        lines.extend(f"- {item}" for item in request.acceptance_criteria)
    lines.extend(("", "Output contract:", output_contract))
    lines.extend(
        (
            "",
            "Constraints:",
            f"- Tool access mode: {request.tool_access_mode.value}.",
            "- Use only the tools visible in this run.",
            "- Do not request recursive subagent delegation.",
            "- Do not rely on unstated parent transcript or hidden parent context.",
            "- Do not modify long-term memory, skill files, or profile files.",
            "- Return concise findings for the parent agent to merge.",
            "- Prefer concrete evidence refs and artifact refs over raw logs.",
        )
    )
    return "\n".join(lines)


def _filtered_surface(
    surface: CapabilitySurface | None,
    allowed: set[str],
) -> CapabilitySurface | None:
    if surface is None:
        return None
    refs = tuple(
        ref
        for ref in surface.list_refs()
        if ref.kind != CapabilityKind.SKILL
        and ref.name in allowed
        and ref.name not in RECURSIVE_SUBAGENT_TOOLS
    )
    return CapabilitySurface(refs)


def _filtered_native_tools(
    registry: NativeToolRegistry | None,
    allowed: set[str],
) -> NativeToolRegistry | None:
    if registry is None:
        return None
    return NativeToolRegistry(
        tool
        for tool in registry.list_tools()
        if tool.name.strip() in allowed
        and tool.name.strip() not in RECURSIVE_SUBAGENT_TOOLS
    )


def _filtered_mcp_tools(
    executor: McpToolExecutor | None,
    allowed: set[str],
) -> McpToolExecutor | None:
    if executor is None:
        return None
    return _AllowedMcpToolExecutor(  # type: ignore[return-value]
        executor,
        allowed - RECURSIVE_SUBAGENT_TOOLS,
    )


def _filtered_tool_schemas(
    schemas: Iterable[Mapping[str, object]],
    allowed: set[str],
) -> tuple[Mapping[str, object], ...]:
    if not allowed:
        return ()
    filtered: list[Mapping[str, object]] = []
    for schema in schemas:
        name = _schema_name(schema)
        if name in allowed and name not in RECURSIVE_SUBAGENT_TOOLS:
            filtered.append(schema)
    return tuple(filtered)


def _filtered_selected_skills(
    skills: Iterable[SkillDocument],
    allowed: set[str],
) -> tuple[SkillDocument, ...]:
    if not allowed:
        return ()
    return tuple(
        skill
        for skill in skills
        if skill.name.strip() in allowed
        and skill.name.strip() not in RECURSIVE_SUBAGENT_TOOLS
    )


def _child_tool_access_policy(
    parent: ToolAccessPolicy | None,
    mode,
) -> ToolAccessPolicy:
    if parent is None:
        return ToolAccessPolicy(mode=mode)
    return replace(parent, mode=mode)


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


def _turn_result(
    run_id: str,
    request: SubagentRunRequest,
    turn: UserTurnResult,
) -> SubagentRunResult:
    if _turn_was_cancelled(turn):
        status = SubagentRunStatus.CANCELLED
    elif turn.reached_max_steps:
        status = SubagentRunStatus.MAX_STEPS_REACHED
    else:
        status = SubagentRunStatus.COMPLETED
    summary = _summary(turn, status)
    artifact_refs = _artifact_refs(turn)
    evidence_refs = _evidence_refs(run_id, request, turn)
    error = None
    if status == SubagentRunStatus.MAX_STEPS_REACHED:
        error = ErrorInfo(
            code="subagent_max_steps_reached",
            message="Subagent reached max_steps before a final answer.",
            refs=_run_refs(run_id, request, extra=("reached_max_steps=true",)),
        )
    return SubagentRunResult(
        run_id=run_id,
        status=status,
        summary=summary,
        artifact_refs=artifact_refs,
        evidence_refs=evidence_refs,
        error=error,
        usage=_usage(turn),
    )


def _turn_was_cancelled(turn: UserTurnResult) -> bool:
    return any(
        event.kind == "turn_interrupted" for step in turn.steps for event in step.events
    ) or any(
        error.code == "turn_interrupted" for step in turn.steps for error in step.errors
    )


def _summary(turn: UserTurnResult, status: SubagentRunStatus) -> str:
    if turn.steps:
        response = turn.final_step().response
        text = response.content.strip() or response.reasoning.strip()
        if text:
            return text
    if status == SubagentRunStatus.MAX_STEPS_REACHED:
        return "Subagent reached max_steps before a final answer."
    return "Subagent completed without a text response."


def _artifact_refs(turn: UserTurnResult) -> tuple[str, ...]:
    refs: list[str] = []
    for step in turn.steps:
        for result in step.tool_results:
            if _is_artifact_result(result):
                refs.extend(result.refs)
    return _unique_refs(refs)


def _evidence_refs(
    run_id: str,
    request: SubagentRunRequest,
    turn: UserTurnResult,
) -> tuple[str, ...]:
    refs: list[str] = list(_run_refs(run_id, request))
    for step in turn.steps:
        for result in step.tool_results:
            refs.extend(result.refs)
        for error in step.errors:
            refs.extend(error.refs)
        for event in step.events:
            refs.extend(event.refs)
    return _unique_refs(refs)


def _usage(turn: UserTurnResult) -> TokenUsage | None:
    input_tokens = 0
    output_tokens = 0
    cache_hit_input_tokens = 0
    cache_miss_input_tokens = 0
    reasoning_tokens = 0
    found = False
    for step in turn.steps:
        usage = step.response.usage
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


def _is_artifact_result(result: ModelToolResult) -> bool:
    if result.is_error:
        return False
    if result.name in {"write_text_file", "edit_text_file"}:
        return True
    return any(key in result.data for key in ("chars_written", "replacements"))


def _run_refs(
    run_id: str,
    request: SubagentRunRequest,
    extra: tuple[str, ...] = (),
) -> tuple[str, ...]:
    refs = [
        f"subagent_run_id={run_id}",
        f"max_steps={request.max_steps}",
        f"tool_access_mode={request.tool_access_mode.value}",
    ]
    if request.parent_session_id:
        refs.append(f"parent_session_id={request.parent_session_id}")
    if request.parent_activation_id:
        refs.append(f"parent_activation_id={request.parent_activation_id}")
    if request.model_purpose:
        refs.append(f"model_purpose={request.model_purpose}")
    refs.extend(extra)
    return tuple(refs)


def _failed_result(
    run_id: str,
    message: str,
    refs: tuple[str, ...],
) -> SubagentRunResult:
    error = ErrorInfo(code="subagent_run_failed", message=message, refs=refs)
    return SubagentRunResult(
        run_id=run_id,
        status=SubagentRunStatus.FAILED,
        summary=message,
        evidence_refs=refs,
        error=error,
    )


def _unique_refs(refs: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for ref in refs:
        text = ref.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return tuple(output)


def _record_trace(
    recorder: TraceRecorder | None,
    kind: str,
    summary: str,
    refs: tuple[str, ...],
) -> None:
    if recorder is None:
        return
    recorder.record(TraceEvent(kind=kind, summary=summary, refs=refs))


class _AllowedMcpToolExecutor:
    """Whitelist wrapper for MCP tools already discovered by the parent runtime."""

    def __init__(self, executor: McpToolExecutor, allowed: set[str]) -> None:
        self._executor = executor
        self._allowed = frozenset(name.strip() for name in allowed if name.strip())

    def has_tool(self, name: str) -> bool:
        tool_name = name.strip()
        return tool_name in self._allowed and self._executor.has_tool(tool_name)

    def tool_schema(self, name: str) -> Mapping[str, object] | None:
        tool_name = name.strip()
        if tool_name not in self._allowed:
            return None
        return self._executor.tool_schema(tool_name)

    def execute(self, request: ModelToolRequest) -> McpToolExecutionResult:
        tool_name = _mcp_request_name(request)
        request_id = _mcp_request_id(request)
        if tool_name not in self._allowed:
            message = (
                "MCP tool is not allowed for this subagent: "
                f"{tool_name or '<empty>'}"
            )
            refs = (tool_name,) if tool_name else ("tool_name=<empty>",)
            return McpToolExecutionResult(
                request=request,
                model_result=(
                    ModelToolResult(
                        name=tool_name or "mcp_tool",
                        request_id=request_id,
                        content=message,
                        refs=refs,
                        is_error=True,
                    )
                    if request_id
                    else None
                ),
                error=ErrorInfo(
                    code="mcp_tool_not_allowed",
                    message=message,
                    refs=refs,
                ),
                events=(
                    RuntimeEvent(
                        kind="mcp_tool_not_allowed",
                        summary=message,
                        refs=refs,
                    ),
                ),
            )
        return self._executor.execute(request)

    def close(self) -> None:
        self._executor.close()


def _mcp_request_name(request: ModelToolRequest) -> str:
    if not isinstance(request.name, str):
        return ""
    return request.name.strip()


def _mcp_request_id(request: ModelToolRequest) -> str:
    if not isinstance(request.id, str):
        return ""
    return request.id.strip()
