"""Run one user turn through bounded agent-loop steps."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Event
from time import sleep
from typing import TYPE_CHECKING

from deepmate.context import (
    ContextBuildResult,
    ContextWarning,
    ProfileContextSnapshot,
    build_profile_context_snapshot,
    build_system_context_from_snapshot,
)
from deepmate.domain import ErrorInfo, Message, MessageRole, ProfileRef, RuntimeEvent
from deepmate.foundation.tool_schema import unflatten_tool_arguments
from deepmate.mcp import McpToolExecutor
from deepmate.providers import (
    ModelCapabilities,
    ModelConversationItem,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ModelToolExchange,
    ModelToolRequest,
    ModelToolResult,
    NetworkError,
    ProviderError,
    RateLimitError,
    ServerError,
    StreamDelta,
)
from deepmate.runtime.activation import RuntimeActivation
from deepmate.runtime.conversation_budget import (
    ConversationBudgetPolicy,
    ConversationBudgetReport,
    RequestBudgetReport,
    build_conversation_budget_report,
    build_request_budget_report,
    select_conversation_window,
    TRIM_HISTORY_WINDOW_MODE,
)
from deepmate.runtime.cost_summary import build_turn_cost_cache_summary
from deepmate.runtime.diagnostics import (
    apply_post_edit_diagnostics,
    post_edit_diagnostic_events,
    post_edit_diagnostics,
)
from deepmate.runtime.model_request import build_model_request
from deepmate.providers.chat_completions import sanitize_model_request
from deepmate.runtime.prefix_cache import PrefixFingerprint, model_request_prefix_fingerprint
from deepmate.runtime.loop_guard import (
    DEFAULT_HARD_STEP_CAP,
    LoopGuardPolicy,
    LoopGuardStop,
    LoopGuardStopReason,
    build_continuation_note,
    build_hard_cap_stop,
    context_meter,
    evaluate_request_preflight,
)
from deepmate.runtime.hooks import (
    HookActor,
    HookDirective,
    HookEnvelope,
    HookEvent,
    HookOutcome,
    HookRuntimeContext,
)
from deepmate.runtime.followup import TurnFollowupBuffer
from deepmate.runtime.tool_output_compaction import ToolOutputCompactor
from deepmate.runtime.tool_executor import execute_native_tool_request
from deepmate.runtime.tool_policy import ToolAccessPolicy
from deepmate.runtime.tool_repair import (
    ToolCallRepairResult,
    ToolCallRepairState,
    ToolRepairPolicy,
    invalid_tool_request_result,
    repair_tool_arguments,
    scavenge_tool_requests_from_response,
    tool_argument_diagnostic,
)
from deepmate.tools import NativeToolRegistry
from deepmate.trace import TraceEvent, TraceRecorder

if TYPE_CHECKING:
    from deepmate.capabilities import CapabilitySurface
    from deepmate.skills import SkillDocument
    from deepmate.subagents import SubagentToolExecutor

HistorySink = Callable[[ModelConversationItem], None]
StatusSink = Callable[[str], None]
# Receives incremental text fragments as a streamed response is produced. When
# provided (and the provider supports streaming), the model step streams; when
# None, the step falls back to a single blocking completion. Kept separate from
# StatusSink so visible token output and runtime status stay distinct channels.
TokenSink = Callable[[StreamDelta], None]


class TurnCancellationToken:
    """Cooperative cancellation token checked at runtime boundaries."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    def cancelled(self) -> bool:
        return self._event.is_set()


@dataclass(frozen=True, slots=True)
class ProviderRetryPolicy:
    """Retry transient provider failures within one agent step."""

    max_attempts: int = 2
    initial_delay_seconds: float = 0.5

    def delay_before_attempt(self, next_attempt: int) -> float:
        """Return exponential backoff delay before a retry attempt."""
        if next_attempt <= 1:
            return 0.0
        return max(0.0, self.initial_delay_seconds) * (2 ** (next_attempt - 2))

    def delay_for_error(self, error: BaseException, next_attempt: int) -> float:
        """Return the retry delay, honoring provider-supplied rate-limit hints."""
        base_delay = self.delay_before_attempt(next_attempt)
        retry_after = getattr(error, "retry_after_seconds", None)
        if isinstance(retry_after, (int, float)):
            return max(base_delay, max(0.0, float(retry_after)))
        return base_delay


@dataclass(frozen=True, slots=True)
class AgentStepResult:
    """Result produced by one agent-loop step."""

    request: ModelRequest
    response: ModelResponse
    tool_results: tuple[ModelToolResult, ...] = field(default_factory=tuple)
    warnings: tuple[ContextWarning, ...] = field(default_factory=tuple)
    events: tuple[RuntimeEvent, ...] = field(default_factory=tuple)
    errors: tuple[ErrorInfo, ...] = field(default_factory=tuple)
    conversation_budget_report: ConversationBudgetReport | None = None
    request_budget_report: RequestBudgetReport | None = None
    schema_additions: tuple[Mapping[str, object], ...] = field(default_factory=tuple)
    loop_guard_stop: LoopGuardStop | None = None
    replay_tool_requests: tuple[ModelToolRequest, ...] = field(default_factory=tuple)

    def needs_followup_step(self) -> bool:
        """Return whether tool results should be sent back in another agent step."""
        return bool(self.tool_results)

    def to_tool_exchange(self) -> ModelToolExchange | None:
        """Return the replay exchange for the tool results from this step."""
        if not self.tool_results:
            return None
        return ModelToolExchange(
            assistant_content=self.response.content,
            assistant_reasoning=self.response.reasoning,
            tool_requests=self.replay_tool_requests or self.response.tool_requests,
            tool_results=self.tool_results,
        )

    def has_errors(self) -> bool:
        """Return whether this agent step observed runtime-level errors."""
        return bool(self.errors)


@dataclass(frozen=True, slots=True)
class UserTurnResult:
    """Result produced by one user turn with bounded model/tool replay."""

    steps: tuple[AgentStepResult, ...]
    conversation: tuple[ModelConversationItem, ...] = field(default_factory=tuple)
    tool_exchanges: tuple[ModelToolExchange, ...] = field(default_factory=tuple)
    reached_max_steps: bool = False
    loop_guard_stop: LoopGuardStop | None = None

    def final_step(self) -> AgentStepResult:
        """Return the last agent step produced by this user turn."""
        if not self.steps:
            raise ValueError("user turn has no agent steps")
        return self.steps[-1]

    def warnings(self) -> tuple[ContextWarning, ...]:
        """Return all context warnings observed across agent steps."""
        return tuple(warning for step in self.steps for warning in step.warnings)

    def events(self) -> tuple[RuntimeEvent, ...]:
        """Return all runtime events observed across agent steps."""
        return tuple(event for step in self.steps for event in step.events)

    def errors(self) -> tuple[ErrorInfo, ...]:
        """Return all runtime errors observed across agent steps."""
        return tuple(error for step in self.steps for error in step.errors)

    def has_errors(self) -> bool:
        """Return whether any agent step observed runtime-level errors."""
        return bool(self.errors())

    def continuation_note(self) -> str:
        """Return continuation note text for non-normal stops, if any."""
        if self.loop_guard_stop is None:
            return ""
        return self.loop_guard_stop.continuation_note.content


def run_agent_step(
    provider: ModelProvider,
    workspace: str | Path,
    profile: ProfileRef,
    messages: Iterable[Message],
    model: str,
    capability_surface: CapabilitySurface | None = None,
    native_tools: NativeToolRegistry | None = None,
    mcp_tools: McpToolExecutor | None = None,
    subagents: "SubagentToolExecutor | None" = None,
    tool_access_policy: ToolAccessPolicy | None = None,
    tool_schemas: Iterable[Mapping[str, object]] = (),
    tool_exchanges: Iterable[ModelToolExchange] = (),
    conversation: Iterable[ModelConversationItem] = (),
    turn_tail_messages: Iterable[Message] = (),
    selected_skill_documents: Iterable["SkillDocument"] = (),
    activation: RuntimeActivation | None = None,
    context_snapshot: ProfileContextSnapshot | None = None,
    system_context: ContextBuildResult | None = None,
    conversation_budget_policy: ConversationBudgetPolicy | None = None,
    provider_retry_policy: ProviderRetryPolicy | None = None,
    options: Mapping[str, object] | None = None,
    model_capabilities: ModelCapabilities | None = None,
    trace_recorder: TraceRecorder | None = None,
    step_index: int = 1,
    warning_sink: Callable[[ContextWarning], None] | None = None,
    status_sink: StatusSink | None = None,
    token_sink: TokenSink | None = None,
    tool_repair_state: ToolCallRepairState | None = None,
    tool_repair_policy: ToolRepairPolicy | None = None,
    tool_output_compactor: ToolOutputCompactor | None = None,
    loop_guard_policy: LoopGuardPolicy | None = None,
    hook_context: HookRuntimeContext | None = None,
    cancellation_token: TurnCancellationToken | None = None,
) -> AgentStepResult:
    """Run exactly one agent-loop step and execute requested tools once."""
    activation_refs = _activation_refs(activation)
    events: list[RuntimeEvent] = []
    errors: list[ErrorInfo] = []
    input_conversation = tuple(conversation)
    window_selection = select_conversation_window(
        input_conversation,
        policy=conversation_budget_policy,
    )
    request_result = build_model_request(
        workspace=workspace,
        profile=profile,
        messages=messages,
        model=model,
        capability_surface=capability_surface,
        tool_schemas=tool_schemas,
        tool_exchanges=tool_exchanges,
        conversation=window_selection.conversation,
        turn_tail_messages=turn_tail_messages,
        selected_skill_documents=selected_skill_documents,
        context_snapshot=context_snapshot,
        system_message=system_context.message if system_context else None,
        context_warnings=(
            system_context.warnings
            if system_context is not None and step_index == 1
            else ()
        ),
        options=options,
        capabilities=model_capabilities,
    )
    request_result = replace(
        request_result,
        request=sanitize_model_request(request_result.request),
    )
    budget_report = (
        window_selection.report
        if input_conversation
        else build_conversation_budget_report(
            request_result.request.conversation[1:],
            policy=conversation_budget_policy,
        )
    )
    request_budget_report = build_request_budget_report(
        request_result.request,
        policy=conversation_budget_policy,
    )
    emergency_trimmed = False
    loop_stop = evaluate_request_preflight(
        request_budget_report,
        loop_guard_policy,
    )
    emergency_policies = _emergency_trim_policies(conversation_budget_policy)
    for emergency_policy in emergency_policies:
        if loop_stop is None or not _can_emergency_trim(
            emergency_policy,
            request_result.request,
        ):
            break
        emergency_window = select_conversation_window(
            input_conversation,
            policy=emergency_policy,
        )
        request_result = build_model_request(
            workspace=workspace,
            profile=profile,
            messages=messages,
            model=model,
            capability_surface=capability_surface,
            tool_schemas=tool_schemas,
            tool_exchanges=tool_exchanges,
            conversation=emergency_window.conversation,
            turn_tail_messages=turn_tail_messages,
            selected_skill_documents=selected_skill_documents,
            context_snapshot=context_snapshot,
            system_message=system_context.message if system_context else None,
            context_warnings=(
                system_context.warnings
                if system_context is not None and step_index == 1
                else ()
            ),
            options=options,
            capabilities=model_capabilities,
        )
        request_result = replace(
            request_result,
            request=sanitize_model_request(request_result.request),
        )
        budget_report = emergency_window.report
        request_budget_report = build_request_budget_report(
            request_result.request,
            policy=emergency_policy,
        )
        loop_stop = evaluate_request_preflight(
            request_budget_report,
            loop_guard_policy,
        )
        emergency_trimmed = True
    _record_trace(
        trace_recorder,
        "conversation_budget_report",
        "Conversation history budget estimated for model request.",
        refs=(f"step={step_index}", *budget_report.trace_refs(), *activation_refs),
    )
    _record_trace(
        trace_recorder,
        "request_budget_report",
        "Full model request budget estimated.",
        refs=(
            f"step={step_index}",
            *request_budget_report.trace_refs(),
            *activation_refs,
        ),
    )
    prefix_fingerprint = model_request_prefix_fingerprint(request_result.request)
    _record_trace(
        trace_recorder,
        "model_request_prefix_fingerprint",
        "Stable model request prefix fingerprint computed.",
        refs=(
            f"step={step_index}",
            *prefix_fingerprint.trace_refs(),
            *activation_refs,
        ),
    )
    meter = context_meter(request_budget_report)
    _record_trace(
        trace_recorder,
        "context_meter",
        "Model request context pressure estimated.",
        refs=(f"step={step_index}", *meter.trace_refs(), *activation_refs),
    )
    if loop_stop is not None:
        error = ErrorInfo(
            code=f"loop_guard_{loop_stop.reason.value}",
            message=loop_stop.message,
            refs=loop_stop.trace_refs(),
        )
        event = RuntimeEvent(
            kind="loop_guard_stop",
            summary=loop_stop.message,
            refs=loop_stop.trace_refs(),
        )
        events.append(event)
        errors.append(error)
        _record_trace(
            trace_recorder,
            "loop_guard_stop",
            loop_stop.message,
            refs=(f"step={step_index}", *loop_stop.trace_refs(), *activation_refs),
        )
        _emit_status(
            status_sink,
            f"{loop_stop.context_meter.status_label() if loop_stop.context_meter else 'context exhausted'}; stopped before provider request",
        )
        return AgentStepResult(
            request=request_result.request,
            response=ModelResponse(content=_loop_guard_response_text(loop_stop)),
            warnings=request_result.warnings,
            events=tuple(events),
            errors=tuple(errors),
            conversation_budget_report=budget_report,
            request_budget_report=request_budget_report,
            loop_guard_stop=loop_stop,
        )
    if emergency_trimmed:
        event = RuntimeEvent(
            kind="context_emergency_trimmed",
            summary=(
                "Conversation history was trimmed only after the full request "
                "would have exceeded the usable context window."
            ),
            refs=budget_report.trace_refs(),
        )
        events.append(event)
        _emit_status(
            status_sink,
            (
                "context emergency trim: "
                f"dropped={budget_report.dropped_count}; "
                f"items={budget_report.selected_conversation_items}/"
                f"{budget_report.conversation_items}"
            ),
        )
        _record_trace(
            trace_recorder,
            event.kind,
            event.summary,
            refs=(f"step={step_index}", *event.refs, *activation_refs),
        )
    _record_trace(
        trace_recorder,
        "model_request_started",
        f"Model request started for step {step_index}.",
        refs=(
            f"model={model}",
            f"step={step_index}",
            f"conversation={len(request_result.request.conversation)}",
            f"tools={len(request_result.request.tool_schemas)}",
            *activation_refs,
        ),
    )
    _emit_status(status_sink, f"model request started: step={step_index}; model={model}")
    provider_before = _emit_runtime_hook(
        hook_context,
        HookEvent.PROVIDER_BEFORE_REQUEST,
        payload={
            "model": model,
            "estimated_input_tokens": request_budget_report.estimated_input_tokens,
            "tool_schema_count": len(request_result.request.tool_schemas),
            "tool_schema_estimated_tokens": (
                request_budget_report.estimated_tool_schema_tokens
            ),
            "context_epoch": (
                activation.context_epoch if activation is not None else 0
            ),
            "status": "before_request",
        },
        trace_recorder=trace_recorder,
        step_index=step_index,
        activation_refs=activation_refs,
    )
    if provider_before.directive != HookDirective.CONTINUE:
        error = ErrorInfo(
            code="provider_request_blocked_by_hook",
            message=provider_before.reason or "Provider request blocked by hook.",
            refs=_hook_outcome_refs(
                HookEvent.PROVIDER_BEFORE_REQUEST,
                provider_before,
                step_index,
                activation_refs,
            ),
        )
        events.append(
            RuntimeEvent(
                kind="provider_request_blocked_by_hook",
                summary=error.message,
                refs=error.refs,
            )
        )
        _record_trace(
            trace_recorder,
            "provider_request_blocked_by_hook",
            error.message,
            refs=error.refs,
        )
        return AgentStepResult(
            request=request_result.request,
            response=ModelResponse(content=error.message),
            warnings=request_result.warnings,
            events=tuple(events),
            errors=(error,),
            conversation_budget_report=budget_report,
            request_budget_report=request_budget_report,
        )
    for warning in request_result.warnings:
        if warning_sink is not None:
            warning_sink(warning)
        _record_trace(
            trace_recorder,
            "context_warning",
            warning.message,
            refs=(*warning.refs, *activation_refs),
        )

    try:
        response = _complete_model_span(
            provider=provider,
            request=request_result.request,
            retry_policy=provider_retry_policy,
            trace_recorder=trace_recorder,
            model=model,
            step_index=step_index,
            activation=activation,
            request_budget_report=request_budget_report,
            token_sink=token_sink,
        )
    except (NetworkError, RateLimitError, ServerError) as exc:
        _record_trace(
            trace_recorder,
            "model_request_failed",
            f"Model request failed for step {step_index}: {exc}",
            refs=(f"model={model}", f"step={step_index}", *activation_refs),
        )
        return _provider_failure_step(
            error=exc,
            model=model,
            step_index=step_index,
            activation=activation,
            trace_recorder=trace_recorder,
            request=request_result.request,
            warnings=request_result.warnings,
            conversation_budget_report=budget_report,
            request_budget_report=request_budget_report,
        )
    except ProviderError as exc:
        context_stop = _provider_context_limit_step(
            error=exc,
            request=request_result.request,
            warnings=request_result.warnings,
            conversation_budget_report=budget_report,
            request_budget_report=request_budget_report,
            trace_recorder=trace_recorder,
            step_index=step_index,
            activation=activation,
        )
        if context_stop is not None:
            return context_stop
        _record_trace(
            trace_recorder,
            "model_request_failed",
            f"Model request failed for step {step_index}: {exc}",
            refs=(
                f"model={model}",
                f"step={step_index}",
                f"error_type={type(exc).__name__}",
                *activation_refs,
            ),
        )
        return _provider_failure_step(
            error=exc,
            model=model,
            step_index=step_index,
            activation=activation,
            trace_recorder=trace_recorder,
            request=request_result.request,
            warnings=request_result.warnings,
            conversation_budget_report=budget_report,
            request_budget_report=request_budget_report,
        )
    except Exception as exc:
        _record_trace(
            trace_recorder,
            "model_request_failed",
            f"Model request failed for step {step_index}: {exc}",
            refs=(f"model={model}", f"step={step_index}", *activation_refs),
        )
        raise

    _record_trace(
        trace_recorder,
        "model_response_received",
        f"Model response received for step {step_index}.",
        refs=(
            f"model={model}",
            f"step={step_index}",
            f"tool_requests={len(response.tool_requests)}",
            *_request_usage_refs(response, request_budget_report),
            *_usage_refs(response),
            *activation_refs,
        ),
    )
    _emit_runtime_hook(
        hook_context,
        HookEvent.PROVIDER_AFTER_RESPONSE,
        payload={
            "model": model,
            "tool_requests": len(response.tool_requests),
            "status": "completed",
            "finish_reason": _safe_text(response.finish_reason),
        },
        trace_recorder=trace_recorder,
        step_index=step_index,
        activation_refs=activation_refs,
    )
    _emit_status(status_sink, _model_step_status(response, request_budget_report, step_index))
    response_events, response_errors = _response_diagnostics(response)
    events.extend(response_events)
    errors.extend(response_errors)
    finish_events, finish_errors = _finish_reason_diagnostics(response)
    events.extend(finish_events)
    errors.extend(finish_errors)
    for error in finish_errors:
        _emit_status(status_sink, f"model finish warning: {error.message}")
        _record_trace(
            trace_recorder,
            "model_finish_reason_warning",
            error.message,
            refs=(*error.refs, f"step={step_index}", *activation_refs),
        )
    tool_results_out: list[ModelToolResult] = []
    schema_additions_out: list[Mapping[str, object]] = []
    native_write_results: list[ModelToolResult] = []
    visible_tool_schemas = _schema_by_name(request_result.request.tool_schemas)
    visible_tool_names = set(visible_tool_schemas)
    repair_policy = tool_repair_policy or ToolRepairPolicy()
    repair_state = (
        repair_policy.new_state()
        if not repair_policy.enabled
        else tool_repair_state or repair_policy.new_state()
    )
    scavenge = (
        scavenge_tool_requests_from_response(
            response,
            visible_tool_names,
            step_index=step_index,
        )
        if repair_policy.enabled and repair_policy.reasoning_scavenge
        else None
    )
    if scavenge is not None:
        response = replace(response, tool_requests=scavenge.tool_requests)
        events.extend(scavenge.events)
        _emit_status(status_sink, scavenge.status)
        for event in scavenge.events:
            _record_trace(
                trace_recorder,
                event.kind,
                event.summary,
                refs=(*event.refs, f"step={step_index}", *activation_refs),
            )
    effective_tool_requests = list(response.tool_requests)
    tool_requests_changed = False
    replay_tool_requests: list[ModelToolRequest] = []
    replay_ids_seen: set[str] = set()

    for tool_index, tool_request in enumerate(response.tool_requests):
        cancel_step = _cancellation_step_result(
            cancellation_token,
            request=request_result.request,
            response=response,
            warnings=request_result.warnings,
            events=tuple(events),
            errors=tuple(errors),
            conversation_budget_report=budget_report,
            request_budget_report=request_budget_report,
            replay_tool_requests=tuple(replay_tool_requests),
            tool_results=tuple(tool_results_out),
        )
        if cancel_step is not None:
            return cancel_step
        invalid_result = invalid_tool_request_result(tool_request)
        if invalid_result is not None:
            _apply_tool_repair_result(
                invalid_result,
                events=events,
                errors=errors,
                tool_results=tool_results_out,
                trace_recorder=trace_recorder,
                status_sink=status_sink,
                step_index=step_index,
                activation_refs=activation_refs,
            )
            continue

        argument_repair = (
            repair_tool_arguments(tool_request)
            if repair_policy.enabled and repair_policy.argument_repair
            else None
        )
        if argument_repair is not None:
            tool_request = argument_repair.request
            effective_tool_requests[tool_index] = tool_request
            tool_requests_changed = True
            events.extend(argument_repair.events)
            _emit_status(status_sink, argument_repair.status)
            for event in argument_repair.events:
                _record_trace(
                    trace_recorder,
                    event.kind,
                    event.summary,
                    refs=(*event.refs, f"step={step_index}", *activation_refs),
                )

        argument_event = tool_argument_diagnostic(tool_request)
        if argument_event is not None:
            events.append(argument_event)
            _record_trace(
                trace_recorder,
                argument_event.kind,
                argument_event.summary,
                refs=(*argument_event.refs, f"step={step_index}", *activation_refs),
            )

        replay_request = _replay_tool_request(
            tool_request,
            tool_index=tool_index,
            replay_ids_seen=replay_ids_seen,
        )
        if replay_request != tool_request:
            effective_tool_requests[tool_index] = replay_request
            tool_requests_changed = True
            event = RuntimeEvent(
                kind="tool_request_replay_normalized",
                summary="Tool request name or id normalized for replay.",
                refs=(
                    f"tool={_tool_request_name(replay_request)}",
                    f"original_tool={_tool_request_name(tool_request)}",
                    f"tool_call_id={_tool_request_id(replay_request)}",
                    f"original_tool_call_id={_tool_request_id(tool_request)}",
                ),
            )
            events.append(event)
            _record_trace(
                trace_recorder,
                event.kind,
                event.summary,
                refs=(*event.refs, f"step={step_index}", *activation_refs),
            )

        repeated_result = repair_state.repeated_call_result(tool_request)
        if repeated_result is not None:
            replay_tool_requests.append(replay_request)
            _apply_tool_repair_result(
                repeated_result,
                request_id_override=replay_request.id,
                events=events,
                errors=errors,
                tool_results=tool_results_out,
                trace_recorder=trace_recorder,
                status_sink=status_sink,
                step_index=step_index,
                activation_refs=activation_refs,
            )
            continue

        replay_tool_requests.append(replay_request)

        tool_name = _tool_request_name(tool_request)
        replay_request_id = _tool_request_id(replay_request)
        tool_request_id = replay_request_id
        if subagents is not None and subagents.has_tool(tool_name):
            _record_trace(
                trace_recorder,
                "subagent_tool_requested",
                f"Subagent tool requested: {tool_name}.",
                refs=(
                    "tool_source=subagent",
                    tool_name,
                    tool_request_id,
                    f"step={step_index}",
                    *activation_refs,
                ),
            )
            if _apply_hook_tool_block(
                _emit_runtime_hook(
                    hook_context,
                    HookEvent.TOOL_BEFORE,
                    payload={
                        "tool_name": tool_name,
                        "tool_source": "subagent",
                        "actor": HookActor.MAIN.value,
                        "status": "before",
                    },
                    trace_recorder=trace_recorder,
                    step_index=step_index,
                    activation_refs=("tool_source=subagent", *activation_refs),
                ),
                hook_event=HookEvent.TOOL_BEFORE,
                tool_request=replay_request,
                events=events,
                errors=errors,
                tool_results=tool_results_out,
                trace_recorder=trace_recorder,
                step_index=step_index,
                activation_refs=("tool_source=subagent", *activation_refs),
            ):
                continue
            if _apply_hook_tool_block(
                _emit_runtime_hook(
                    hook_context,
                    HookEvent.SUBAGENT_BEFORE_RUN,
                    payload={
                        "tool_name": tool_name,
                        "tool_source": "subagent",
                        "actor": HookActor.MAIN.value,
                        "status": "before_run",
                    },
                    trace_recorder=trace_recorder,
                    step_index=step_index,
                    activation_refs=("tool_source=subagent", *activation_refs),
                ),
                hook_event=HookEvent.SUBAGENT_BEFORE_RUN,
                tool_request=replay_request,
                events=events,
                errors=errors,
                tool_results=tool_results_out,
                trace_recorder=trace_recorder,
                step_index=step_index,
                activation_refs=("tool_source=subagent", *activation_refs),
            ):
                continue
            with _tool_span(
                trace_recorder,
                tool_name=tool_name,
                tool_source="subagent",
                tool_request_id=tool_request_id,
                step_index=step_index,
                activation=activation,
                operation_name="invoke_agent",
            ) as tool_span:
                execution = subagents.execute(replay_request)
                _finish_tool_span(
                    tool_span,
                    error=execution.error,
                    model_result=execution.model_result,
                )
            events.extend(execution.events)
            if execution.error is not None:
                errors.append(execution.error)
                _record_trace(
                    trace_recorder,
                    "subagent_tool_failed",
                    execution.error.message,
                    refs=(
                        "tool_source=subagent",
                        *execution.error.refs,
                        f"step={step_index}",
                        *activation_refs,
                    ),
                )
            else:
                _record_trace(
                    trace_recorder,
                    "subagent_tool_completed",
                    f"Subagent tool completed: {tool_name}.",
                    refs=(
                        "tool_source=subagent",
                        tool_name,
                        f"step={step_index}",
                        *(
                            execution.model_result.refs
                            if execution.model_result is not None
                            else ()
                        ),
                        *activation_refs,
                    ),
                )
            if execution.model_result is not None:
                _append_governed_tool_result(
                    execution.model_result,
                    tool_source="subagent",
                    tool_results=tool_results_out,
                    events=events,
                    trace_recorder=trace_recorder,
                    status_sink=status_sink,
                    request_budget_report=request_budget_report,
                    tool_output_compactor=tool_output_compactor,
                    step_index=step_index,
                    activation_refs=activation_refs,
                )
            _emit_runtime_hook(
                hook_context,
                HookEvent.TOOL_AFTER,
                payload={
                    "tool_name": tool_name,
                    "tool_source": "subagent",
                    "status": "failed" if execution.error is not None else "completed",
                },
                trace_recorder=trace_recorder,
                step_index=step_index,
                activation_refs=("tool_source=subagent", *activation_refs),
            )
            _emit_runtime_hook(
                hook_context,
                HookEvent.SUBAGENT_AFTER_RUN,
                payload={
                    "tool_name": tool_name,
                    "tool_source": "subagent",
                    "status": "failed" if execution.error is not None else "completed",
                },
                trace_recorder=trace_recorder,
                step_index=step_index,
                activation_refs=("tool_source=subagent", *activation_refs),
            )
            continue

        if mcp_tools is not None and mcp_tools.has_tool(tool_name):
            mcp_schema = visible_tool_schemas.get(tool_name)
            if mcp_schema is None:
                schema_provider = getattr(mcp_tools, "tool_schema", None)
                if callable(schema_provider):
                    schema_candidate = schema_provider(tool_name)
                    if isinstance(schema_candidate, Mapping):
                        mcp_schema = schema_candidate
            if tool_name not in visible_tool_names:
                _record_trace(
                    trace_recorder,
                    "mcp_tool_schema_hidden",
                    f"MCP tool schema hidden but tool is registered: {tool_name}.",
                    refs=(
                        "tool_source=mcp",
                        tool_name,
                        f"step={step_index}",
                        *activation_refs,
                    ),
                )
            if _apply_hook_tool_block(
                _emit_runtime_hook(
                    hook_context,
                    HookEvent.TOOL_BEFORE,
                    payload={
                        "tool_name": tool_name,
                        "tool_source": "mcp",
                        "actor": HookActor.MAIN.value,
                        "status": "before",
                    },
                    trace_recorder=trace_recorder,
                    step_index=step_index,
                    activation_refs=("tool_source=mcp", *activation_refs),
                ),
                hook_event=HookEvent.TOOL_BEFORE,
                tool_request=replay_request,
                events=events,
                errors=errors,
                tool_results=tool_results_out,
                trace_recorder=trace_recorder,
                step_index=step_index,
                activation_refs=("tool_source=mcp", *activation_refs),
            ):
                continue
            execution_request = _mcp_execution_request(
                replay_request,
                mcp_schema,
                events=events,
                trace_recorder=trace_recorder,
                status_sink=status_sink,
                step_index=step_index,
                activation_refs=activation_refs,
            )
            _record_trace(
                trace_recorder,
                "mcp_tool_requested",
                f"MCP tool requested: {tool_name}.",
                refs=(
                    "tool_source=mcp",
                    tool_name,
                    tool_request_id,
                    f"step={step_index}",
                    *activation_refs,
                ),
            )
            _emit_status(status_sink, f"tool started: source=mcp; tool={tool_name}")
            with _tool_span(
                trace_recorder,
                tool_name=tool_name,
                tool_source="mcp",
                tool_request_id=tool_request_id,
                step_index=step_index,
                activation=activation,
            ) as tool_span:
                execution = mcp_tools.execute(execution_request)
                _finish_tool_span(
                    tool_span,
                    error=execution.error,
                    model_result=execution.model_result,
                )
            events.extend(execution.events)
            for event in execution.events:
                if event.kind in {
                    "mcp_before_hook_observed",
                    "mcp_after_hook_observed",
                    "mcp_tool_output_truncated",
                }:
                    _record_trace(
                        trace_recorder,
                        event.kind,
                        event.summary,
                        refs=(
                            "tool_source=mcp",
                            *event.refs,
                            f"step={step_index}",
                            *activation_refs,
                        ),
                    )
            if execution.error is not None:
                errors.append(execution.error)
                _record_trace(
                    trace_recorder,
                    "mcp_tool_failed",
                    execution.error.message,
                    refs=(
                        "tool_source=mcp",
                        *execution.error.refs,
                        f"step={step_index}",
                        *activation_refs,
                    ),
                )
            else:
                _record_trace(
                    trace_recorder,
                    "mcp_tool_completed",
                    f"MCP tool completed: {tool_name}.",
                    refs=(
                        "tool_source=mcp",
                        tool_name,
                        f"step={step_index}",
                        *(
                            execution.model_result.refs
                            if execution.model_result is not None
                            else ()
                        ),
                        *activation_refs,
                    ),
                )
            _emit_status(
                status_sink,
                "tool finished: source=mcp; tool={tool}; outcome={outcome}".format(
                    tool=tool_name,
                    outcome="failed" if execution.error is not None else "ok",
                ),
            )
            if execution.model_result is not None:
                _append_governed_tool_result(
                    execution.model_result,
                    tool_source="mcp",
                    tool_results=tool_results_out,
                    events=events,
                    trace_recorder=trace_recorder,
                    status_sink=status_sink,
                    request_budget_report=request_budget_report,
                    tool_output_compactor=tool_output_compactor,
                    step_index=step_index,
                    activation_refs=activation_refs,
                )
            _emit_runtime_hook(
                hook_context,
                HookEvent.TOOL_AFTER,
                payload={
                    "tool_name": tool_name,
                    "tool_source": "mcp",
                    "status": "failed" if execution.error is not None else "completed",
                },
                trace_recorder=trace_recorder,
                step_index=step_index,
                activation_refs=("tool_source=mcp", *activation_refs),
            )
            continue

        _record_trace(
            trace_recorder,
            "native_tool_requested",
            f"Native tool requested: {tool_name}.",
            refs=(
                "tool_source=native",
                tool_name,
                tool_request_id,
                f"step={step_index}",
                *activation_refs,
            ),
        )
        _emit_status(status_sink, f"tool started: source=native; tool={tool_name}")
        if native_tools is None:
            result = _missing_native_tool_registry(replay_request)
            events.extend(result.events)
            errors.append(result.error)
            tool_results_out.extend(result.tool_results)
            _record_trace(
                trace_recorder,
                "native_tool_failed",
                result.error.message,
                refs=(
                    "tool_source=native",
                    *result.error.refs,
                    f"step={step_index}",
                    *activation_refs,
                ),
            )
            continue

        native_tool = native_tools.get(tool_name)
        if native_tool is None:
            result = _native_tool_not_found_result(replay_request)
            events.extend(result.events)
            errors.append(result.error)
            tool_results_out.extend(result.tool_results)
            _record_trace(
                trace_recorder,
                "native_tool_not_found",
                result.error.message,
                refs=(
                    "tool_source=native",
                    *result.error.refs,
                    f"step={step_index}",
                    *activation_refs,
                ),
            )
            continue

        if tool_name not in visible_tool_names:
            _record_trace(
                trace_recorder,
                "native_tool_schema_hidden",
                f"Native tool schema hidden but tool is registered: {tool_name}.",
                refs=(
                    "tool_source=native",
                    tool_name,
                    f"step={step_index}",
                    *activation_refs,
                ),
            )

        if _apply_hook_tool_block(
            _emit_runtime_hook(
                hook_context,
                HookEvent.TOOL_BEFORE,
                payload={
                    "tool_name": tool_name,
                    "tool_source": "native",
                    "actor": HookActor.MAIN.value,
                    "status": "before",
                },
                trace_recorder=trace_recorder,
                step_index=step_index,
                activation_refs=("tool_source=native", *activation_refs),
            ),
            hook_event=HookEvent.TOOL_BEFORE,
            tool_request=replay_request,
            events=events,
            errors=errors,
            tool_results=tool_results_out,
            trace_recorder=trace_recorder,
            step_index=step_index,
            activation_refs=("tool_source=native", *activation_refs),
        ):
            continue
        with _tool_span(
            trace_recorder,
            tool_name=tool_name,
            tool_source="native",
            tool_request_id=tool_request_id,
            step_index=step_index,
            activation=activation,
        ) as tool_span:
            native_request = replay_request
            execution = execute_native_tool_request(
                native_request,
                native_tools,
                access_policy=tool_access_policy,
            )
            _finish_tool_span(
                tool_span,
                error=execution.error,
                model_result=execution.model_result,
                result_refs=(
                    execution.native_result.refs if execution.native_result else ()
                ),
            )
        events.extend(execution.events)
        if execution.error is not None:
            errors.append(execution.error)
            _record_trace(
                trace_recorder,
                "native_tool_failed",
                execution.error.message,
                refs=(
                    "tool_source=native",
                    *execution.error.refs,
                    f"step={step_index}",
                    *activation_refs,
                ),
            )
        else:
            _record_trace(
                trace_recorder,
                "native_tool_completed",
                f"Native tool completed: {tool_name}.",
                refs=(
                    "tool_source=native",
                    tool_name,
                    *(execution.native_result.refs if execution.native_result else ()),
                    f"step={step_index}",
                    *activation_refs,
                ),
            )
        _emit_status(
            status_sink,
            "tool finished: source=native; tool={tool}; outcome={outcome}".format(
                tool=tool_name,
                outcome="failed" if execution.error is not None else "ok",
            ),
        )
        _emit_runtime_hook(
            hook_context,
            HookEvent.TOOL_AFTER,
            payload={
                "tool_name": tool_name,
                "tool_source": "native",
                "status": "failed" if execution.error is not None else "completed",
            },
            trace_recorder=trace_recorder,
            step_index=step_index,
            activation_refs=("tool_source=native", *activation_refs),
        )
        if execution.model_result is not None:
            _append_governed_tool_result(
                execution.model_result,
                tool_source="native",
                tool_results=tool_results_out,
                events=events,
                trace_recorder=trace_recorder,
                status_sink=status_sink,
                request_budget_report=request_budget_report,
                tool_output_compactor=tool_output_compactor,
                step_index=step_index,
                activation_refs=activation_refs,
            )
            if _is_successful_write_result(execution.model_result):
                native_write_results.append(execution.model_result)
        if (
            execution.native_result is not None
            and execution.native_result.schema_additions
        ):
            schema_additions_out.extend(execution.native_result.schema_additions)
            _emit_status(
                status_sink,
                "tool schema loaded: "
                f"{tool_name} added {len(execution.native_result.schema_additions)} schema(s)",
            )
            _record_trace(
                trace_recorder,
                "tool_schema_loaded",
                f"Tool schema loaded by native tool: {tool_name}.",
                refs=(
                    "tool_source=native",
                    tool_name,
                    *(execution.native_result.refs if execution.native_result else ()),
                    f"schema_additions={len(execution.native_result.schema_additions)}",
                    f"step={step_index}",
                    *activation_refs,
                ),
            )

    if tool_requests_changed:
        response = replace(response, tool_requests=tuple(effective_tool_requests))

    if native_write_results:
        diagnostics = post_edit_diagnostics(workspace, tuple(native_write_results))
        diagnostic_events = post_edit_diagnostic_events(diagnostics)
        events.extend(diagnostic_events)
        for event in diagnostic_events:
            _record_trace(
                trace_recorder,
                event.kind,
                event.summary,
                refs=(*event.refs, f"step={step_index}", *activation_refs),
            )
        if diagnostics.checked_count() > 0 or diagnostics.has_failures():
            _emit_status(
                status_sink,
                (
                    "post-edit diagnostics: "
                    f"checked={diagnostics.checked_count()}; "
                    f"failed={diagnostics.failed_count()}; "
                    f"skipped={diagnostics.skipped_count()}"
                ),
            )
        if diagnostics.has_failures():
            _apply_diagnostics_to_write_results(tool_results_out, diagnostics)

    return AgentStepResult(
        request=request_result.request,
        response=response,
        tool_results=tuple(tool_results_out),
        warnings=request_result.warnings,
        events=tuple(events),
        errors=tuple(errors),
        conversation_budget_report=budget_report,
        request_budget_report=request_budget_report,
        schema_additions=tuple(schema_additions_out),
        replay_tool_requests=tuple(replay_tool_requests),
    )


def run_user_turn(
    provider: ModelProvider,
    workspace: str | Path,
    profile: ProfileRef,
    messages: Iterable[Message],
    model: str,
    capability_surface: CapabilitySurface | None = None,
    native_tools: NativeToolRegistry | None = None,
    mcp_tools: McpToolExecutor | None = None,
    subagents: "SubagentToolExecutor | None" = None,
    tool_access_policy: ToolAccessPolicy | None = None,
    tool_schemas: Iterable[Mapping[str, object]] = (),
    tool_exchanges: Iterable[ModelToolExchange] = (),
    conversation: Iterable[ModelConversationItem] = (),
    ephemeral_turn_tail_messages: Iterable[Message] = (),
    selected_skill_documents: Iterable["SkillDocument"] = (),
    activation: RuntimeActivation | None = None,
    context_snapshot: ProfileContextSnapshot | None = None,
    system_context: ContextBuildResult | None = None,
    conversation_budget_policy: ConversationBudgetPolicy | None = None,
    provider_retry_policy: ProviderRetryPolicy | None = None,
    options: Mapping[str, object] | None = None,
    model_capabilities: ModelCapabilities | None = None,
    max_steps: int = DEFAULT_HARD_STEP_CAP,
    trace_recorder: TraceRecorder | None = None,
    warning_sink: Callable[[ContextWarning], None] | None = None,
    history_sink: HistorySink | None = None,
    status_sink: StatusSink | None = None,
    token_sink: TokenSink | None = None,
    tool_output_compactor: ToolOutputCompactor | None = None,
    tool_repair_policy: ToolRepairPolicy | None = None,
    loop_guard_policy: LoopGuardPolicy | None = None,
    hook_context: HookRuntimeContext | None = None,
    followup_buffer: TurnFollowupBuffer | None = None,
    followup_turn_id: str | None = None,
    cancellation_token: TurnCancellationToken | None = None,
) -> UserTurnResult:
    """Run one user turn until the model stops requesting native tools."""
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    with ExitStack() as stack:
        turn_span = stack.enter_context(
            _turn_span(
                trace_recorder,
                model=model,
                max_steps=max_steps,
                activation=activation,
            )
        )
        try:
            result = _run_user_turn_inner(
                provider=provider,
                workspace=workspace,
                profile=profile,
                messages=messages,
                model=model,
                capability_surface=capability_surface,
                native_tools=native_tools,
                mcp_tools=mcp_tools,
                subagents=subagents,
                tool_access_policy=tool_access_policy,
                tool_schemas=tool_schemas,
                tool_exchanges=tool_exchanges,
                conversation=conversation,
                ephemeral_turn_tail_messages=ephemeral_turn_tail_messages,
                selected_skill_documents=selected_skill_documents,
                activation=activation,
                context_snapshot=context_snapshot,
                system_context=system_context,
                conversation_budget_policy=conversation_budget_policy,
                provider_retry_policy=provider_retry_policy,
                options=options,
                model_capabilities=model_capabilities,
                max_steps=max_steps,
                trace_recorder=trace_recorder,
                warning_sink=warning_sink,
                history_sink=history_sink,
                status_sink=status_sink,
                token_sink=token_sink,
                tool_output_compactor=tool_output_compactor,
                tool_repair_policy=tool_repair_policy,
                loop_guard_policy=loop_guard_policy,
                hook_context=hook_context,
                followup_buffer=followup_buffer,
                followup_turn_id=followup_turn_id,
                cancellation_token=cancellation_token,
            )
        except Exception:
            _span_set_status(turn_span, "ERROR")
            raise
        _finish_turn_span(turn_span, result)
        return result


def _run_user_turn_inner(
    *,
    provider: ModelProvider,
    workspace: str | Path,
    profile: ProfileRef,
    messages: Iterable[Message],
    model: str,
    capability_surface: CapabilitySurface | None = None,
    native_tools: NativeToolRegistry | None = None,
    mcp_tools: McpToolExecutor | None = None,
    subagents: "SubagentToolExecutor | None" = None,
    tool_access_policy: ToolAccessPolicy | None = None,
    tool_schemas: Iterable[Mapping[str, object]] = (),
    tool_exchanges: Iterable[ModelToolExchange] = (),
    conversation: Iterable[ModelConversationItem] = (),
    ephemeral_turn_tail_messages: Iterable[Message] = (),
    selected_skill_documents: Iterable["SkillDocument"] = (),
    activation: RuntimeActivation | None = None,
    context_snapshot: ProfileContextSnapshot | None = None,
    system_context: ContextBuildResult | None = None,
    conversation_budget_policy: ConversationBudgetPolicy | None = None,
    provider_retry_policy: ProviderRetryPolicy | None = None,
    options: Mapping[str, object] | None = None,
    model_capabilities: ModelCapabilities | None = None,
    max_steps: int = DEFAULT_HARD_STEP_CAP,
    trace_recorder: TraceRecorder | None = None,
    warning_sink: Callable[[ContextWarning], None] | None = None,
    history_sink: HistorySink | None = None,
    status_sink: StatusSink | None = None,
    token_sink: TokenSink | None = None,
    tool_output_compactor: ToolOutputCompactor | None = None,
    tool_repair_policy: ToolRepairPolicy | None = None,
    loop_guard_policy: LoopGuardPolicy | None = None,
    hook_context: HookRuntimeContext | None = None,
    followup_buffer: TurnFollowupBuffer | None = None,
    followup_turn_id: str | None = None,
    cancellation_token: TurnCancellationToken | None = None,
) -> UserTurnResult:
    body_messages = tuple(messages)
    schemas = tuple(tool_schemas)
    selected_skills = tuple(selected_skill_documents)
    exchanges = list(tool_exchanges)
    prior_conversation = tuple(conversation)
    ephemeral_tail = _ready_ephemeral_turn_tail_messages(ephemeral_turn_tail_messages)
    snapshot = _activation_snapshot(
        activation=activation,
        workspace=workspace,
        profile=profile,
        context_snapshot=context_snapshot,
    )
    history = list(_initial_conversation(prior_conversation, body_messages, exchanges))
    _append_initial_history(history_sink, prior_conversation, body_messages, exchanges)
    if system_context is None:
        system_context = build_system_context_from_snapshot(
            snapshot=snapshot,
            capability_surface=capability_surface,
            selected_skill_documents=selected_skills,
        )
    steps: list[AgentStepResult] = []
    repair_policy = tool_repair_policy or ToolRepairPolicy()
    repair_state = repair_policy.new_state()
    configured_loop_guard_policy = (loop_guard_policy or LoopGuardPolicy()).normalized()
    effective_max_steps = (
        min(max_steps, configured_loop_guard_policy.hard_step_cap)
        if configured_loop_guard_policy.enabled
        else max_steps
    )
    active_loop_guard_policy = replace(
        configured_loop_guard_policy,
        hard_step_cap=effective_max_steps,
    )
    _record_trace(
        trace_recorder,
        "user_turn_started",
        "User turn started.",
        refs=(
            f"model={model}",
            f"max_steps={effective_max_steps}",
            f"loop_guard_enabled={str(active_loop_guard_policy.enabled).lower()}",
            f"hard_step_cap={active_loop_guard_policy.hard_step_cap}",
            *repair_policy.trace_refs(),
            *_activation_refs(activation),
        ),
    )

    previous_prefix: PrefixFingerprint | None = None
    stable_prefix_count = 0
    for step_index in range(1, effective_max_steps + 1):
        cancel_result = _cancellation_turn_result(
            cancellation_token,
            steps=tuple(steps),
            history=tuple(history),
            exchanges=tuple(exchanges),
            hook_context=hook_context,
            trace_recorder=trace_recorder,
            step_index=step_index,
            activation=activation,
        )
        if cancel_result is not None:
            return cancel_result
        turn_tail_messages = _drain_turn_followups(
            followup_buffer=followup_buffer,
            followup_turn_id=followup_turn_id,
            trace_recorder=trace_recorder,
            step_index=step_index,
            activation=activation,
        )
        request_tail_messages = (*ephemeral_tail, *turn_tail_messages)
        if ephemeral_tail:
            _record_trace(
                trace_recorder,
                "ephemeral_turn_tail_injected",
                "Ephemeral turn-tail context injected outside the system prefix.",
                refs=(
                    f"step={step_index}",
                    f"messages={len(ephemeral_tail)}",
                    *_activation_refs(activation),
                ),
            )
        try:
            step = run_agent_step(
                provider=provider,
                workspace=workspace,
                profile=profile,
                messages=(),
                model=model,
                capability_surface=capability_surface,
                native_tools=native_tools,
                mcp_tools=mcp_tools,
                subagents=subagents,
                tool_access_policy=tool_access_policy,
                tool_schemas=schemas,
                tool_exchanges=(),
                conversation=tuple(history),
                turn_tail_messages=request_tail_messages,
                selected_skill_documents=selected_skills,
                activation=activation,
                context_snapshot=snapshot,
                system_context=system_context,
                conversation_budget_policy=conversation_budget_policy,
                provider_retry_policy=provider_retry_policy,
                options=options,
                model_capabilities=model_capabilities,
                trace_recorder=trace_recorder,
                step_index=step_index,
                warning_sink=warning_sink,
                status_sink=status_sink,
                token_sink=token_sink,
                tool_repair_state=repair_state,
                tool_repair_policy=repair_policy,
                tool_output_compactor=tool_output_compactor,
                loop_guard_policy=active_loop_guard_policy,
                hook_context=hook_context,
                cancellation_token=cancellation_token,
            )
        except (NetworkError, RateLimitError, ServerError) as exc:
            step = _provider_failure_step(
                error=exc,
                model=model,
                step_index=step_index,
                activation=activation,
                trace_recorder=trace_recorder,
            )
        steps.append(step)
        current_prefix = model_request_prefix_fingerprint(step.request)
        stable_prefix_count = _record_prefix_stability(
            trace_recorder,
            previous=previous_prefix,
            current=current_prefix,
            stable_count=stable_prefix_count,
            step_index=step_index,
            activation=activation,
        )
        previous_prefix = current_prefix
        if step.schema_additions:
            schemas = _merge_tool_schemas(schemas, step.schema_additions)
        _append_turn_tail_history(history, history_sink, turn_tail_messages)
        if not step.needs_followup_step():
            assistant_text = _assistant_history_text(step.response)
            if assistant_text:
                _append_history_item(
                    history,
                    history_sink,
                    ModelConversationItem.from_message(
                        Message(role=MessageRole.ASSISTANT, content=assistant_text)
                    )
                )
            _record_turn_cost_cache_summary(
                steps=tuple(steps),
                trace_recorder=trace_recorder,
                status_sink=status_sink,
                activation=activation,
            )
            _record_trace(
                trace_recorder,
                "user_turn_finished",
                "User turn finished.",
                refs=(
                    f"steps={len(steps)}",
                    f"tool_exchanges={len(exchanges)}",
                    *_activation_refs(activation),
                ),
            )
            _emit_runtime_hook(
                hook_context,
                HookEvent.AGENT_TURN_END,
                payload={
                    "status": "completed",
                    "summary": "User turn completed.",
                    "steps": len(steps),
                    "tool_exchanges": len(exchanges),
                    "errors": len(tuple(error for step in steps for error in step.errors)),
                    "events": len(tuple(event for step in steps for event in step.events)),
                },
                trace_recorder=trace_recorder,
                step_index=len(steps),
                activation_refs=_activation_refs(activation),
            )
            return UserTurnResult(
                steps=tuple(steps),
                conversation=tuple(history),
                tool_exchanges=tuple(exchanges),
                loop_guard_stop=step.loop_guard_stop,
            )

        exchange = step.to_tool_exchange()
        if exchange is None or not exchange.is_ready():
            _record_trace(
                trace_recorder,
                "tool_exchange_invalid",
                "Agent step produced an invalid tool exchange.",
                refs=(f"step={step_index}", *_activation_refs(activation)),
            )
            raise ValueError("agent step produced an invalid tool exchange")
        exchanges.append(exchange)
        _append_history_item(
            history,
            history_sink,
            ModelConversationItem.from_tool_exchange(exchange),
        )

    _record_turn_cost_cache_summary(
        steps=tuple(steps),
        trace_recorder=trace_recorder,
        status_sink=status_sink,
        activation=activation,
    )
    _record_trace(
        trace_recorder,
        "user_turn_max_steps",
        "User turn reached max agent steps before final answer.",
        refs=(
            f"steps={len(steps)}",
            f"max_steps={effective_max_steps}",
            *_activation_refs(activation),
        ),
    )
    loop_stop = (
        build_hard_cap_stop(
            step_count=len(steps),
            policy=active_loop_guard_policy,
        )
        if active_loop_guard_policy.enabled
        else None
    )
    if loop_stop is not None:
        _record_trace(
            trace_recorder,
            "loop_guard_stop",
            loop_stop.message,
            refs=(*loop_stop.trace_refs(), *_activation_refs(activation)),
        )
        _emit_status(status_sink, loop_stop.message)
    _emit_runtime_hook(
        hook_context,
        HookEvent.AGENT_TURN_END,
        payload={
            "status": "max_steps",
            "summary": "User turn reached max agent steps before final answer.",
            "steps": len(steps),
            "max_steps": effective_max_steps,
            "tool_exchanges": len(exchanges),
            "errors": len(tuple(error for step in steps for error in step.errors)),
            "events": len(tuple(event for step in steps for event in step.events)),
        },
        trace_recorder=trace_recorder,
        step_index=len(steps),
        activation_refs=_activation_refs(activation),
    )
    return UserTurnResult(
        steps=tuple(steps),
        conversation=tuple(history),
        tool_exchanges=tuple(exchanges),
        reached_max_steps=True,
        loop_guard_stop=loop_stop,
    )


def _provider_failure_step(
    *,
    error: ProviderError,
    model: str,
    step_index: int,
    activation: RuntimeActivation | None,
    trace_recorder: TraceRecorder | None,
    request: ModelRequest | None = None,
    warnings: tuple[ContextWarning, ...] = (),
    conversation_budget_report: ConversationBudgetReport | None = None,
    request_budget_report: RequestBudgetReport | None = None,
) -> AgentStepResult:
    error_type = type(error).__name__
    message = f"Model request failed after retry attempts: {error}"
    info = ErrorInfo(
        code="provider_request_failed",
        message=message,
        refs=(
            f"model={model}",
            f"step={step_index}",
            f"error_type={error_type}",
            *_activation_refs(activation),
        ),
    )
    event = RuntimeEvent(
        kind="provider_request_failed",
        summary=message,
        refs=info.refs,
    )
    _record_trace(
        trace_recorder,
        "provider_request_failed",
        message,
        refs=info.refs,
    )
    return AgentStepResult(
        request=request or ModelRequest(model=model, conversation=()),
        response=ModelResponse(content=message),
        warnings=warnings,
        events=(event,),
        errors=(info,),
        conversation_budget_report=conversation_budget_report,
        request_budget_report=request_budget_report,
    )


def _provider_context_limit_step(
    *,
    error: ProviderError,
    request: ModelRequest,
    warnings: tuple[ContextWarning, ...],
    conversation_budget_report: ConversationBudgetReport | None,
    request_budget_report: RequestBudgetReport | None,
    trace_recorder: TraceRecorder | None,
    step_index: int,
    activation: RuntimeActivation | None,
) -> AgentStepResult | None:
    if not _looks_like_context_limit_error(error):
        return None
    message = (
        "The provider rejected the model request because it exceeded the context "
        "window. Deepmate stopped the turn and saved continuation context."
    )
    note = build_continuation_note(
        reason=LoopGuardStopReason.CONTEXT_EXHAUSTED,
        progress="The provider rejected the request as too large for the context window.",
        remaining="Continue from the latest visible transcript and checkpoint context.",
        avoid="Do not resend the same oversized request without reducing context.",
        next_action=(
            "Resume with a focused prompt, reduce loaded context, or start a smaller follow-up."
        ),
    )
    meter = context_meter(request_budget_report) if request_budget_report is not None else None
    loop_stop = LoopGuardStop(
        reason=LoopGuardStopReason.CONTEXT_EXHAUSTED,
        message=message,
        continuation_note=note,
        context_meter=meter,
    )
    error_info = ErrorInfo(
        code="loop_guard_context_exhausted",
        message=message,
        refs=(
            *loop_stop.trace_refs(),
            f"provider_error={_text_preview(str(error), 220)}",
            *_activation_refs(activation),
        ),
    )
    event = RuntimeEvent(
        kind="loop_guard_stop",
        summary=message,
        refs=error_info.refs,
    )
    _record_trace(
        trace_recorder,
        "loop_guard_stop",
        message,
        refs=(f"step={step_index}", *error_info.refs),
    )
    return AgentStepResult(
        request=request,
        response=ModelResponse(content=_loop_guard_response_text(loop_stop)),
        warnings=warnings,
        events=(event,),
        errors=(error_info,),
        conversation_budget_report=conversation_budget_report,
        request_budget_report=request_budget_report,
        loop_guard_stop=loop_stop,
    )


def _looks_like_context_limit_error(error: ProviderError) -> bool:
    text = str(error).lower()
    if "http 400" not in text and "context" not in text and "token" not in text:
        return False
    markers = (
        "context length",
        "context window",
        "maximum context",
        "max context",
        "too many tokens",
        "token limit",
        "exceed",
        "exceeded",
        "too large",
    )
    return any(marker in text for marker in markers)


def _loop_guard_response_text(stop: LoopGuardStop) -> str:
    lines = [stop.message]
    if stop.context_meter is not None:
        lines.append("")
        lines.append(f"Context: {stop.context_meter.status_label()}.")
    return "\n".join(lines)


def _activation_snapshot(
    activation: RuntimeActivation | None,
    workspace: str | Path,
    profile: ProfileRef,
    context_snapshot: ProfileContextSnapshot | None,
) -> ProfileContextSnapshot:
    if activation is not None:
        if context_snapshot is not None and context_snapshot != activation.context_snapshot:
            raise ValueError("activation and context_snapshot must not disagree")
        return activation.context_snapshot
    return context_snapshot or build_profile_context_snapshot(
        workspace=workspace,
        profile=profile,
    )


def _activation_refs(activation: RuntimeActivation | None) -> tuple[str, ...]:
    if activation is None:
        return ()
    return activation.trace_refs()


def _activation_attributes(activation: RuntimeActivation | None) -> dict[str, object]:
    if activation is None:
        return {}
    return {
        "session.id": activation.session_id,
        "gen_ai.conversation.id": activation.session_id,
        "deepmate.session_id": activation.session_id,
        "deepmate.activation_id": activation.activation_id,
        "deepmate.profile": activation.profile.name,
    }


@dataclass(frozen=True, slots=True)
class _ToolUnavailableResult:
    error: ErrorInfo
    events: tuple[RuntimeEvent, ...]
    tool_results: tuple[ModelToolResult, ...]


def _response_diagnostics(response: ModelResponse) -> tuple[list[RuntimeEvent], list[ErrorInfo]]:
    if response.has_output():
        return [], []
    message = "Model response contained no content, reasoning, or tool requests."
    return (
        [RuntimeEvent(kind="model_response_empty", summary=message)],
        [ErrorInfo(code="model_response_empty", message=message)],
    )


def _finish_reason_diagnostics(
    response: ModelResponse,
) -> tuple[list[RuntimeEvent], list[ErrorInfo]]:
    reason = _safe_text(response.finish_reason)
    if reason not in {"length", "content_filter"}:
        return [], []
    message = f"Model response ended with finish_reason={reason}."
    return (
        [RuntimeEvent(kind=f"model_finish_reason_{reason}", summary=message)],
        [
            ErrorInfo(
                code=f"model_finish_reason_{reason}",
                message=message,
                refs=(f"finish_reason={reason}",),
            )
        ],
    )


def _usage_refs(response: ModelResponse) -> tuple[str, ...]:
    usage = response.usage
    if usage is None:
        return _finish_reason_ref(response)
    input_tokens = _usage_int(usage, "input_tokens")
    output_tokens = _usage_int(usage, "output_tokens")
    cache_hit_input_tokens = _usage_int(usage, "cache_hit_input_tokens")
    cache_miss_input_tokens = _usage_int(usage, "cache_miss_input_tokens")
    reasoning_tokens = _usage_int(usage, "reasoning_tokens")
    return (
        f"input_tokens={input_tokens}",
        f"output_tokens={output_tokens}",
        f"cache_hit_input_tokens={cache_hit_input_tokens}",
        f"cache_miss_input_tokens={cache_miss_input_tokens}",
        f"reasoning_tokens={reasoning_tokens}",
        *_finish_reason_ref(response),
    )


def _request_usage_refs(
    response: ModelResponse,
    report: RequestBudgetReport,
) -> tuple[str, ...]:
    usage = response.usage
    if usage is None or report.estimated_input_tokens <= 0:
        return (f"estimated_prompt_tokens={report.estimated_input_tokens}",)
    input_tokens = _usage_int(usage, "input_tokens")
    cache_hit_input_tokens = _usage_int(usage, "cache_hit_input_tokens")
    cache_miss_input_tokens = _usage_int(usage, "cache_miss_input_tokens")
    estimate_ratio = input_tokens / report.estimated_input_tokens
    cache_total = cache_hit_input_tokens + cache_miss_input_tokens
    cache_hit_ratio = (
        cache_hit_input_tokens / cache_total if cache_total > 0 else 0.0
    )
    return (
        f"estimated_prompt_tokens={report.estimated_input_tokens}",
        f"actual_prompt_tokens={input_tokens}",
        f"prompt_token_estimate_ratio={estimate_ratio:.4f}",
        f"cache_hit_ratio={cache_hit_ratio:.4f}",
        f"tool_output_ratio={report.tool_output_ratio:.4f}",
    )


def _finish_reason_ref(response: ModelResponse) -> tuple[str, ...]:
    reason = _safe_text(response.finish_reason)
    if not reason:
        return ()
    return (f"finish_reason={reason}",)


def _record_prefix_stability(
    trace_recorder: TraceRecorder | None,
    *,
    previous: PrefixFingerprint | None,
    current: PrefixFingerprint,
    stable_count: int,
    step_index: int,
    activation: RuntimeActivation | None,
) -> int:
    if previous is None:
        next_count = 1
        refs = (
            f"step={step_index}",
            "prefix_stable=baseline",
            "stable_request_count=1",
            f"current_prefix_digest={current.digest}",
            f"tool_schema_count={len(current.tool_schema_names)}",
            f"tool_schema_names={','.join(current.tool_schema_names[:40])}",
            *_activation_refs(activation),
        )
    else:
        changed_parts = _prefix_changed_parts(previous, current)
        is_stable = not changed_parts
        next_count = stable_count + 1 if is_stable else 1
        changed_text = ",".join(changed_parts)
        refs = (
            f"step={step_index}",
            f"prefix_stable={str(is_stable).lower()}",
            f"stable_request_count={next_count}",
            f"previous_prefix_digest={previous.digest}",
            f"current_prefix_digest={current.digest}",
            f"changed_parts={changed_text}",
            f"system_changed={str('system' in changed_parts).lower()}",
            f"tool_schema_changed={str('tool_schema' in changed_parts).lower()}",
            f"options_changed={str('options' in changed_parts).lower()}",
            f"tool_schema_count={len(current.tool_schema_names)}",
            f"tool_schema_names={','.join(current.tool_schema_names[:40])}",
            *_activation_refs(activation),
        )
    _record_trace(
        trace_recorder,
        "model_request_prefix_stability",
        "Model request prefix stability compared with the previous request.",
        refs=refs,
    )
    return next_count


def _prefix_changed_parts(
    previous: PrefixFingerprint,
    current: PrefixFingerprint,
) -> tuple[str, ...]:
    parts: list[str] = []
    if previous.system_digest != current.system_digest:
        parts.append("system")
    if previous.tool_schema_digest != current.tool_schema_digest:
        parts.append("tool_schema")
    if previous.options_digest != current.options_digest:
        parts.append("options")
    if previous.model != current.model:
        parts.append("model")
    return tuple(parts)


def _model_response_span_attributes(
    response: ModelResponse,
    model: str,
) -> dict[str, object]:
    attributes: dict[str, object] = {
        "gen_ai.response.model": model,
        "deepmate.model.tool_requests": len(response.tool_requests),
    }
    usage = response.usage
    if usage is not None:
        input_tokens = _usage_int(usage, "input_tokens")
        output_tokens = _usage_int(usage, "output_tokens")
        cache_hit_input_tokens = _usage_int(usage, "cache_hit_input_tokens")
        cache_miss_input_tokens = _usage_int(usage, "cache_miss_input_tokens")
        reasoning_tokens = _usage_int(usage, "reasoning_tokens")
        attributes.update(
            {
                "gen_ai.usage.input_tokens": input_tokens,
                "gen_ai.usage.output_tokens": output_tokens,
                "gen_ai.usage.cache_read.input_tokens": cache_hit_input_tokens,
                "deepmate.usage.cache_hit_input_tokens": cache_hit_input_tokens,
                "deepmate.usage.cache_miss_input_tokens": cache_miss_input_tokens,
                "deepmate.usage.reasoning_tokens": reasoning_tokens,
            }
        )
    finish_reason = _safe_text(response.finish_reason)
    if finish_reason:
        attributes["gen_ai.response.finish_reasons"] = [finish_reason]
        attributes["deepmate.finish_reason"] = finish_reason
    return attributes


def _tool_span(
    trace_recorder: TraceRecorder | None,
    *,
    tool_name: str,
    tool_source: str,
    tool_request_id: str,
    step_index: int,
    activation: RuntimeActivation | None,
    operation_name: str = "execute_tool",
):
    attributes: dict[str, object] = {
        "gen_ai.operation.name": operation_name,
        "gen_ai.tool.name": tool_name,
        "gen_ai.tool.call.id": tool_request_id,
        "deepmate.tool.name": tool_name,
        "deepmate.tool.source": tool_source,
        "deepmate.step": step_index,
    }
    attributes.update(_activation_attributes(activation))
    if not _supports_span_recorder(trace_recorder):
        return nullcontext(_NullTraceSpanScope())
    return trace_recorder.start_span(
        f"{operation_name} {tool_name}",
        kind="INTERNAL",
        attributes=attributes,
    )


def _turn_span(
    trace_recorder: TraceRecorder | None,
    *,
    model: str,
    max_steps: int,
    activation: RuntimeActivation | None,
):
    attributes: dict[str, object] = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.request.model": model,
        "deepmate.max_steps": max_steps,
    }
    attributes.update(_activation_attributes(activation))
    if not _supports_span_recorder(trace_recorder):
        return nullcontext(_NullTraceSpanScope())
    return trace_recorder.start_span(
        "deepmate turn",
        kind="INTERNAL",
        attributes=attributes,
    )


def _finish_turn_span(span: object, result: UserTurnResult) -> None:
    summary = build_turn_cost_cache_summary(result.steps)
    status = "ERROR" if result.has_errors() else "OK"
    if result.reached_max_steps:
        status = "ERROR"
        _span_set_attribute(span, "deepmate.stop_reason", "max_steps")
    if result.loop_guard_stop is not None:
        _span_set_attribute(span, "deepmate.loop_guard.reason", result.loop_guard_stop.reason.value)
    _span_set_status(span, status)
    _span_set_attribute(span, "deepmate.turn.steps", len(result.steps))
    _span_set_attribute(span, "deepmate.turn.tool_exchanges", len(result.tool_exchanges))
    _span_set_attribute(span, "gen_ai.usage.input_tokens", summary.input_tokens)
    _span_set_attribute(span, "gen_ai.usage.output_tokens", summary.output_tokens)
    _span_set_attribute(
        span,
        "gen_ai.usage.cache_read.input_tokens",
        summary.cache_hit_input_tokens,
    )
    _span_set_attribute(
        span,
        "deepmate.usage.cache_hit_input_tokens",
        summary.cache_hit_input_tokens,
    )
    _span_set_attribute(
        span,
        "deepmate.usage.cache_miss_input_tokens",
        summary.cache_miss_input_tokens,
    )
    _span_set_attribute(span, "deepmate.usage.reasoning_tokens", summary.reasoning_tokens)


def _finish_tool_span(
    span: object,
    *,
    error: ErrorInfo | None,
    model_result: ModelToolResult | None,
    result_refs: Iterable[str] = (),
) -> None:
    if error is not None:
        _span_set_status(span, "ERROR")
        _span_set_attribute(span, "error.type", error.code)
        _span_set_attribute(span, "error.message", error.message)
    elif model_result is not None and model_result.is_error:
        _span_set_status(span, "ERROR")
        _span_set_attribute(span, "error.type", "tool_result_error")
        _span_set_attribute(span, "error.message", model_result.content)
    else:
        _span_set_status(span, "OK")
    if model_result is not None:
        _span_set_attribute(span, "deepmate.tool.result_chars", len(model_result.content))
        _span_set_attribute(span, "deepmate.tool.result_refs", list(model_result.refs))
    clean_refs = [ref for ref in result_refs if ref]
    if clean_refs:
        _span_set_attribute(span, "deepmate.tool.native_refs", clean_refs)


def _span_set_attribute(span: object, key: str, value: object) -> None:
    setter = getattr(span, "set_attribute", None)
    if callable(setter):
        setter(key, value)


def _span_set_status(span: object, status: str) -> None:
    setter = getattr(span, "set_status", None)
    if callable(setter):
        setter(status)


def _supports_span_recorder(trace_recorder: TraceRecorder | None) -> bool:
    return trace_recorder is not None and callable(
        getattr(trace_recorder, "start_span", None)
    )


class _NullTraceSpanScope:
    def set_attribute(self, _key: str, _value: object) -> None:
        return

    def set_status(self, _status: str) -> None:
        return


def _assistant_history_text(response: ModelResponse) -> str:
    content = _safe_text(response.content)
    if content:
        return content
    return _safe_text(response.reasoning)


def _complete_once(
    provider: ModelProvider,
    request: ModelRequest,
    token_sink: TokenSink | None,
) -> ModelResponse:
    """Run one completion, streaming when a sink is set and supported.

    Retry safety: the provider only raises retryable connection errors before
    emitting any delta, so re-running this on a retryable failure never produces
    duplicate streamed output. A mid-stream failure surfaces as a terminal
    ProviderError, which the retry loop does not catch.
    """
    stream = getattr(provider, "complete_stream", None)
    if token_sink is not None and callable(stream):
        return stream(request, token_sink)
    return provider.complete(request)


def _complete_with_retry(
    provider: ModelProvider,
    request: ModelRequest,
    retry_policy: ProviderRetryPolicy | None,
    trace_recorder: TraceRecorder | None,
    model: str,
    step_index: int,
    activation: RuntimeActivation | None = None,
    token_sink: TokenSink | None = None,
) -> ModelResponse:
    policy = retry_policy or ProviderRetryPolicy()
    if policy.max_attempts < 1:
        raise ValueError("provider retry max_attempts must be at least 1")
    activation_refs = _activation_refs(activation)

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return _complete_once(provider, request, token_sink)
        except (NetworkError, RateLimitError, ServerError) as exc:
            if attempt >= policy.max_attempts:
                _record_trace(
                    trace_recorder,
                    "provider_retry_exhausted",
                    "Provider retry attempts exhausted.",
                    refs=(
                        f"model={model}",
                        f"step={step_index}",
                        f"attempt={attempt}",
                        f"max_attempts={policy.max_attempts}",
                        f"error_type={type(exc).__name__}",
                        *activation_refs,
                    ),
                )
                raise
            next_attempt = attempt + 1
            delay_seconds = policy.delay_for_error(exc, next_attempt)
            _record_trace(
                trace_recorder,
                "provider_retry_scheduled",
                "Provider retry scheduled after transient failure.",
                refs=(
                    f"model={model}",
                    f"step={step_index}",
                    f"attempt={attempt}",
                    f"next_attempt={next_attempt}",
                    f"max_attempts={policy.max_attempts}",
                    f"delay_seconds={delay_seconds:g}",
                    f"error_type={type(exc).__name__}",
                    *activation_refs,
                ),
            )
            if delay_seconds > 0:
                sleep(delay_seconds)
    raise RuntimeError("provider retry loop ended unexpectedly")


def _complete_model_span(
    *,
    provider: ModelProvider,
    request: ModelRequest,
    retry_policy: ProviderRetryPolicy | None,
    trace_recorder: TraceRecorder | None,
    model: str,
    step_index: int,
    activation: RuntimeActivation | None,
    request_budget_report: RequestBudgetReport,
    token_sink: TokenSink | None = None,
) -> ModelResponse:
    attributes: dict[str, object] = {
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": model,
        "deepmate.step": step_index,
        "deepmate.request.conversation_items": len(request.conversation),
        "deepmate.request.tool_schemas": len(request.tool_schemas),
        "deepmate.request.estimated_input_tokens": (
            request_budget_report.estimated_input_tokens
        ),
        "deepmate.context.pressure_ratio": request_budget_report.pressure_ratio,
        "deepmate.context.remaining_input_tokens": (
            context_meter(request_budget_report).remaining_input_tokens()
        ),
    }
    attributes.update(_activation_attributes(activation))
    if not _supports_span_recorder(trace_recorder):
        return _complete_with_retry(
            provider=provider,
            request=request,
            retry_policy=retry_policy,
            trace_recorder=None,
            model=model,
            step_index=step_index,
            activation=activation,
            token_sink=token_sink,
        )
    with trace_recorder.start_span(
        f"chat {model}",
        kind="CLIENT",
        attributes=attributes,
    ) as span:
        response = _complete_with_retry(
            provider=provider,
            request=request,
            retry_policy=retry_policy,
            trace_recorder=trace_recorder,
            model=model,
            step_index=step_index,
            activation=activation,
            token_sink=token_sink,
        )
        span.set_attributes(_model_response_span_attributes(response, model))
        return response


def _initial_conversation(
    conversation: Iterable[ModelConversationItem],
    messages: tuple[Message, ...],
    tool_exchanges: list[ModelToolExchange],
) -> tuple[ModelConversationItem, ...]:
    items = tuple(conversation)
    message_items = tuple(
        ModelConversationItem.from_message(message) for message in messages
    )
    if items:
        return (
            *items,
            *message_items,
            *(
                ModelConversationItem.from_tool_exchange(exchange)
                for exchange in tool_exchanges
            ),
        )
    return (
        *message_items,
        *(ModelConversationItem.from_tool_exchange(exchange) for exchange in tool_exchanges),
    )


def _append_initial_history(
    history_sink: HistorySink | None,
    prior_conversation: tuple[ModelConversationItem, ...],
    messages: tuple[Message, ...],
    tool_exchanges: list[ModelToolExchange],
) -> None:
    for message in messages:
        _append_history(history_sink, ModelConversationItem.from_message(message))
    for exchange in tool_exchanges:
        _append_history(history_sink, ModelConversationItem.from_tool_exchange(exchange))


def _append_history_item(
    history: list[ModelConversationItem],
    history_sink: HistorySink | None,
    item: ModelConversationItem,
) -> None:
    history.append(item)
    _append_history(history_sink, item)


def _append_history(
    history_sink: HistorySink | None,
    item: ModelConversationItem,
) -> None:
    if history_sink is not None:
        history_sink(item)


def _drain_turn_followups(
    *,
    followup_buffer: TurnFollowupBuffer | None,
    followup_turn_id: str | None,
    trace_recorder: TraceRecorder | None,
    step_index: int,
    activation: RuntimeActivation | None,
) -> tuple[Message, ...]:
    if followup_buffer is None:
        return ()
    followups = followup_buffer.drain(followup_turn_id)
    messages: list[Message] = []
    for followup in followups:
        if not followup.is_ready():
            continue
        messages.append(followup.to_message())
        _record_trace(
            trace_recorder,
            "followup_injected",
            "Running user follow-up injected before model request.",
            refs=(
                f"step={step_index}",
                f"source={followup.source}",
                *_activation_refs(activation),
            ),
    )
    return tuple(messages)


def _ready_ephemeral_turn_tail_messages(
    messages: Iterable[Message],
) -> tuple[Message, ...]:
    ready_messages = tuple(message for message in messages if message.is_ready())
    for message in ready_messages:
        if message.role == MessageRole.SYSTEM:
            raise ValueError("ephemeral turn-tail messages must not use the system role")
    return ready_messages


def _append_turn_tail_history(
    history: list[ModelConversationItem],
    history_sink: HistorySink | None,
    messages: Iterable[Message],
) -> None:
    for message in messages:
        _append_history_item(
            history,
            history_sink,
            ModelConversationItem.from_message(message),
        )


def _can_emergency_trim(
    policy: ConversationBudgetPolicy | None,
    request: ModelRequest,
) -> bool:
    normalized = (policy or ConversationBudgetPolicy()).normalized()
    minimum_context_items = 2
    return len(request.conversation) > max(minimum_context_items, normalized.protect_recent_items)


def _emergency_trim_policies(
    policy: ConversationBudgetPolicy | None,
) -> tuple[ConversationBudgetPolicy, ...]:
    normalized = (policy or ConversationBudgetPolicy()).normalized()
    usable = max(
        1,
        normalized.model_context_tokens
        - normalized.response_token_reserve
        - normalized.safety_margin_tokens,
    )
    budget = min(normalized.history_token_budget, max(1, int(usable * 0.85)))
    primary = replace(
        normalized,
        history_window_mode=TRIM_HISTORY_WINDOW_MODE,
        history_token_budget=budget,
        protect_recent_items=max(2, normalized.protect_recent_items),
    )
    aggressive = replace(
        primary,
        history_token_budget=max(1, int(usable * 0.65)),
        protect_recent_items=2,
    )
    if primary == aggressive:
        return (primary,)
    return (primary, aggressive)


def _cancellation_step_result(
    token: TurnCancellationToken | None,
    *,
    request: ModelRequest,
    response: ModelResponse,
    warnings: tuple[ContextWarning, ...],
    events: tuple[RuntimeEvent, ...],
    errors: tuple[ErrorInfo, ...],
    conversation_budget_report: ConversationBudgetReport | None,
    request_budget_report: RequestBudgetReport | None,
    replay_tool_requests: tuple[ModelToolRequest, ...],
    tool_results: tuple[ModelToolResult, ...],
) -> AgentStepResult | None:
    if token is None or not token.cancelled():
        return None
    event = RuntimeEvent(
        kind="turn_interrupted",
        summary="User interrupted the turn before the next runtime boundary.",
        refs=("interrupted=true",),
    )
    return AgentStepResult(
        request=request,
        response=response,
        tool_results=tool_results,
        warnings=warnings,
        events=(*events, event),
        errors=errors,
        conversation_budget_report=conversation_budget_report,
        request_budget_report=request_budget_report,
        replay_tool_requests=replay_tool_requests,
    )


def _cancellation_turn_result(
    token: TurnCancellationToken | None,
    *,
    steps: tuple[AgentStepResult, ...],
    history: tuple[ModelConversationItem, ...],
    exchanges: tuple[ModelToolExchange, ...],
    hook_context: HookRuntimeContext | None,
    trace_recorder: TraceRecorder | None,
    step_index: int,
    activation: RuntimeActivation | None,
) -> UserTurnResult | None:
    if token is None or not token.cancelled():
        return None
    step = _interrupted_step(step_index=step_index)
    all_steps = (*steps, step)
    _record_trace(
        trace_recorder,
        "turn_interrupted",
        "User interrupted the turn before the next model request.",
        refs=(f"step={step_index}", *_activation_refs(activation)),
    )
    _emit_runtime_hook(
        hook_context,
        HookEvent.AGENT_TURN_END,
        payload={
            "status": "interrupted",
            "summary": "User interrupted the turn.",
            "steps": len(all_steps),
            "tool_exchanges": len(exchanges),
            "errors": len(tuple(error for item in all_steps for error in item.errors)),
            "events": len(tuple(event for item in all_steps for event in item.events)),
        },
        trace_recorder=trace_recorder,
        step_index=step_index,
        activation_refs=_activation_refs(activation),
    )
    return UserTurnResult(
        steps=all_steps,
        conversation=history,
        tool_exchanges=exchanges,
    )


def _interrupted_step(*, step_index: int) -> AgentStepResult:
    message = "Turn interrupted by user before the next runtime boundary."
    event = RuntimeEvent(
        kind="turn_interrupted",
        summary=message,
        refs=(f"step={step_index}",),
    )
    error = ErrorInfo(
        code="turn_interrupted",
        message=message,
        refs=(f"step={step_index}",),
    )
    request = ModelRequest(
        model="interrupted",
        conversation=(
            ModelConversationItem.from_message(
                Message(role=MessageRole.SYSTEM, content="Turn interrupted.")
            ),
        ),
    )
    return AgentStepResult(
        request=request,
        response=ModelResponse(content=message),
        events=(event,),
        errors=(error,),
    )


def _record_trace(
    recorder: TraceRecorder | None,
    kind: str,
    summary: str,
    refs: tuple[str, ...] = (),
) -> None:
    if recorder is None:
        return
    clean_refs = tuple(ref for ref in refs if ref)
    recorder.record(TraceEvent(kind=kind, summary=summary, refs=clean_refs))


def _emit_status(sink: StatusSink | None, message: str) -> None:
    if sink is None or not message.strip():
        return
    sink(message.strip())


def _record_turn_cost_cache_summary(
    *,
    steps: tuple[AgentStepResult, ...],
    trace_recorder: TraceRecorder | None,
    status_sink: StatusSink | None,
    activation: RuntimeActivation | None,
) -> None:
    if not steps:
        return
    summary = build_turn_cost_cache_summary(steps)
    _record_trace(
        trace_recorder,
        "turn_cost_cache_summary",
        "Turn-level cost/cache summary aggregated.",
        refs=(*summary.trace_refs(), *_activation_refs(activation)),
    )
    _emit_status(status_sink, summary.status_line())


def _model_step_status(
    response: ModelResponse,
    report: RequestBudgetReport,
    step_index: int,
) -> str:
    meter = context_meter(report)
    parts = [
        f"runtime step {step_index}",
        meter.status_label(),
        f"input_pressure={report.pressure_ratio:.3f}",
        f"estimated_input_tokens={report.estimated_input_tokens}",
        f"context_remaining_input_tokens={meter.remaining_input_tokens()}",
        f"model_context_tokens={report.model_context_tokens}",
        f"tool_schema_tokens={report.estimated_tool_schema_tokens}",
        f"tool_output_ratio={report.tool_output_ratio:.3f}",
    ]
    usage = response.usage
    if usage is None:
        parts.append("usage=unreported")
    else:
        input_tokens = _usage_int(usage, "input_tokens")
        output_tokens = _usage_int(usage, "output_tokens")
        cache_hit_input_tokens = _usage_int(usage, "cache_hit_input_tokens")
        cache_miss_input_tokens = _usage_int(usage, "cache_miss_input_tokens")
        reasoning_tokens = _usage_int(usage, "reasoning_tokens")
        cache_total = cache_hit_input_tokens + cache_miss_input_tokens
        cache_ratio = (
            cache_hit_input_tokens / cache_total if cache_total > 0 else 0.0
        )
        parts.extend(
            (
                f"actual_input_tokens={input_tokens}",
                f"output_tokens={output_tokens}",
                f"cache_hit_tokens={cache_hit_input_tokens}",
                f"cache_hit_ratio={cache_ratio:.3f}",
            )
        )
        if reasoning_tokens:
            parts.append(f"reasoning_tokens={reasoning_tokens}")
    finish_reason = _safe_text(response.finish_reason)
    if finish_reason:
        parts.append(f"finish_reason={finish_reason}")
    return "; ".join(parts)


def _safe_text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _text_preview(value: str, limit: int) -> str:
    clean = " ".join(value.split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def _usage_int(usage: object, name: str) -> int:
    value = getattr(usage, name, 0)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return 0


def _append_governed_tool_result(
    result: ModelToolResult,
    *,
    tool_source: str,
    tool_results: list[ModelToolResult],
    events: list[RuntimeEvent],
    trace_recorder: TraceRecorder | None,
    status_sink: StatusSink | None,
    request_budget_report: RequestBudgetReport,
    tool_output_compactor: ToolOutputCompactor | None,
    step_index: int,
    activation_refs: tuple[str, ...],
) -> None:
    processed = (
        tool_output_compactor.process(
            result,
            tool_source=tool_source,
            request_budget_report=request_budget_report,
        )
        if tool_output_compactor is not None
        else None
    )
    if processed is None:
        tool_results.append(result)
        return
    tool_results.append(processed.result)
    events.extend(processed.events)
    for message in processed.status_messages:
        _emit_status(status_sink, message)
    for event in processed.events:
        _record_trace(
            trace_recorder,
            event.kind,
            event.summary,
            refs=(*event.refs, f"step={step_index}", *activation_refs),
        )


def _is_successful_write_result(result: ModelToolResult) -> bool:
    return (
        not result.is_error
        and result.name in {"write_text_file", "edit_text_file"}
        and isinstance(result.data.get("path"), str)
    )


def _apply_diagnostics_to_write_results(
    tool_results: list[ModelToolResult],
    diagnostics,
) -> None:
    failed_paths = {
        diagnostic.path
        for diagnostic in diagnostics.diagnostics
        if diagnostic.failed()
    }
    if not failed_paths:
        return
    for index, result in enumerate(tuple(tool_results)):
        path = result.data.get("path")
        if isinstance(path, str) and path.strip() in failed_paths:
            tool_results[index] = apply_post_edit_diagnostics(result, diagnostics)


def _apply_tool_repair_result(
    result: ToolCallRepairResult,
    *,
    request_id_override: str = "",
    events: list[RuntimeEvent],
    errors: list[ErrorInfo],
    tool_results: list[ModelToolResult],
    trace_recorder: TraceRecorder | None,
    status_sink: StatusSink | None,
    step_index: int,
    activation_refs: tuple[str, ...],
) -> None:
    events.extend(result.events)
    if result.error is not None:
        errors.append(result.error)
    if request_id_override:
        tool_results.extend(
            replace(tool_result, request_id=request_id_override)
            for tool_result in result.tool_results
        )
    else:
        tool_results.extend(result.tool_results)
    _emit_status(status_sink, result.status)
    for event in result.events:
        _record_trace(
            trace_recorder,
            event.kind,
            event.summary,
            refs=(*event.refs, f"step={step_index}", *activation_refs),
        )


def _emit_runtime_hook(
    hook_context: HookRuntimeContext | None,
    event_name: HookEvent,
    *,
    payload: Mapping[str, object],
    trace_recorder: TraceRecorder | None,
    step_index: int,
    activation_refs: tuple[str, ...],
) -> HookOutcome:
    if hook_context is None:
        return HookOutcome()
    outcome = hook_context.emit(
        HookEnvelope(
            event_name=event_name,
            actor=HookActor.MAIN,
            payload=payload,
            source_refs=(
                f"step={step_index}",
                *activation_refs,
                *hook_context.trace_refs(),
            ),
        )
    )
    if outcome.action_results or outcome.directive != HookDirective.CONTINUE:
        _record_trace(
            trace_recorder,
            "hook_event_evaluated",
            f"Hook event evaluated: {event_name.value}.",
            refs=_hook_outcome_refs(event_name, outcome, step_index, activation_refs),
        )
    return outcome


def _apply_hook_tool_block(
    outcome: HookOutcome,
    *,
    hook_event: HookEvent,
    tool_request: ModelToolRequest,
    events: list[RuntimeEvent],
    errors: list[ErrorInfo],
    tool_results: list[ModelToolResult],
    trace_recorder: TraceRecorder | None,
    step_index: int,
    activation_refs: tuple[str, ...],
) -> bool:
    if outcome.directive == HookDirective.CONTINUE:
        return False
    tool_name = _tool_request_name(tool_request)
    request_id = _tool_request_id(tool_request)
    code = (
        "tool_requires_approval_by_hook"
        if outcome.directive == HookDirective.REQUIRES_APPROVAL
        else "tool_blocked_by_hook"
    )
    message = outcome.reason or f"Tool request stopped by hook: {hook_event.value}"
    refs = _hook_outcome_refs(hook_event, outcome, step_index, activation_refs)
    error = ErrorInfo(code=code, message=message, refs=refs)
    event = RuntimeEvent(kind=code, summary=message, refs=refs)
    errors.append(error)
    events.append(event)
    if tool_name and request_id:
        tool_results.append(
            ModelToolResult(
                name=tool_name,
                request_id=request_id,
                content=message,
                refs=refs,
                is_error=True,
            )
        )
    _record_trace(trace_recorder, code, message, refs=refs)
    return True


def _replay_tool_request(
    request: ModelToolRequest,
    *,
    tool_index: int,
    replay_ids_seen: set[str],
) -> ModelToolRequest:
    """Return a replay-safe request with canonical name and unique id."""
    canonical_name = _tool_request_name(request)
    original_id = _tool_request_id(request)
    replay_id = original_id
    if replay_id in replay_ids_seen:
        suffix = 2
        candidate = f"{original_id}_{suffix}"
        while candidate in replay_ids_seen:
            suffix += 1
            candidate = f"{original_id}_{suffix}"
        replay_id = candidate
    replay_ids_seen.add(replay_id)
    if canonical_name != request.name or replay_id != request.id:
        return replace(request, name=canonical_name, id=replay_id)
    return request


def _hook_outcome_refs(
    event_name: HookEvent,
    outcome: HookOutcome,
    step_index: int,
    activation_refs: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        f"hook_event={event_name.value}",
        f"hook_directive={outcome.directive.value}",
        f"step={step_index}",
        *outcome.refs,
        *tuple(f"hook_warning={warning}" for warning in outcome.warnings),
        *activation_refs,
    )


def _missing_native_tool_registry(request: ModelToolRequest) -> _ToolUnavailableResult:
    tool_name = _tool_request_name(request)
    display_name = tool_name or "<empty>"
    request_id = _tool_request_id(request)
    message = f"Native tool registry is not available for tool request: {display_name}"
    error = ErrorInfo(
        code="native_tool_registry_missing",
        message=message,
        refs=(tool_name,) if tool_name else ("tool_name=<empty>",),
    )
    tool_results = (
        (
            ModelToolResult(
                name=display_name,
                request_id=request_id,
                content=message,
                refs=(tool_name,) if tool_name else ("tool_name=<empty>",),
                is_error=True,
            ),
        )
        if request_id
        else ()
    )
    return _ToolUnavailableResult(
        error=error,
        events=(
            RuntimeEvent(
                kind="native_tool_registry_missing",
                summary=message,
                refs=(tool_name,) if tool_name else ("tool_name=<empty>",),
            ),
        ),
        tool_results=tool_results,
    )


def _native_tool_not_found_result(request: ModelToolRequest) -> _ToolUnavailableResult:
    tool_name = _tool_request_name(request)
    display_name = tool_name or "<empty>"
    request_id = _tool_request_id(request)
    message = f"Native tool is not registered for this run: {display_name}"
    refs = (tool_name,) if tool_name else ("tool_name=<empty>",)
    error = ErrorInfo(
        code="native_tool_not_found",
        message=message,
        refs=refs,
    )
    tool_results = (
        (
            ModelToolResult(
                name=display_name,
                request_id=request_id,
                content=message,
                refs=refs,
                is_error=True,
            ),
        )
        if request_id
        else ()
    )
    return _ToolUnavailableResult(
        error=error,
        events=(
            RuntimeEvent(
                kind="native_tool_not_found",
                summary=message,
                refs=refs,
            ),
        ),
        tool_results=tool_results,
    )


def _schema_by_name(
    schemas: Iterable[Mapping[str, object]]
) -> dict[str, Mapping[str, object]]:
    indexed: dict[str, Mapping[str, object]] = {}
    for schema in schemas:
        name = _schema_name(schema)
        if name:
            indexed[name] = schema
    return indexed


def _mcp_execution_request(
    request: ModelToolRequest,
    schema: Mapping[str, object] | None,
    *,
    events: list[RuntimeEvent],
    trace_recorder: TraceRecorder | None,
    status_sink: StatusSink | None,
    step_index: int,
    activation_refs: tuple[str, ...],
) -> ModelToolRequest:
    arguments = unflatten_tool_arguments(request.arguments, schema)
    if arguments is request.arguments:
        return request
    dotted_count = sum(
        1 for key in request.arguments if isinstance(key, str) and "." in key
    )
    tool_name = _tool_request_name(request)
    event = RuntimeEvent(
        kind="mcp_tool_arguments_unflattened",
        summary=f"MCP tool arguments normalized before execution: {tool_name}.",
        refs=(
            f"tool={tool_name}",
            f"tool_call_id={_tool_request_id(request)}",
            f"dotted_arguments={dotted_count}",
        ),
    )
    events.append(event)
    _emit_status(status_sink, f"MCP tool arguments normalized: {tool_name}")
    _record_trace(
        trace_recorder,
        event.kind,
        event.summary,
        refs=("tool_source=mcp", *event.refs, f"step={step_index}", *activation_refs),
    )
    return replace(request, arguments=arguments)


def _merge_tool_schemas(
    current: Iterable[Mapping[str, object]],
    additions: Iterable[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    order: list[str] = []
    by_name: dict[str, Mapping[str, object]] = {}
    for schema in (*tuple(current), *tuple(additions)):
        name = _schema_name(schema)
        if not name:
            continue
        if name not in by_name:
            order.append(name)
        by_name[name] = schema
    return tuple(by_name[name] for name in order)


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


def _tool_request_name(request: ModelToolRequest) -> str:
    if not isinstance(request.name, str):
        return ""
    return _canonical_tool_name(request.name.strip())


def _canonical_tool_name(name: str) -> str:
    if name in {"run_command", "shell", "bash"}:
        return "run_shell_command"
    return name


def _tool_request_id(request: ModelToolRequest) -> str:
    if not isinstance(request.id, str):
        return ""
    return request.id.strip()
