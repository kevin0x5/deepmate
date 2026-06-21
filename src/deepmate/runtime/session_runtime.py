"""Session-level runtime state across user turns."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from deepmate.context import (
    ContextBuildResult,
    ContextFileChange,
    ContextWarning,
    ProfileContextSnapshot,
    build_system_context_from_snapshot,
    detect_behavior_context_changes,
)
from deepmate.domain import Message
from deepmate.mcp import McpToolExecutor
from deepmate.providers import (
    ModelCapabilities,
    ModelConversationItem,
    ModelProvider,
    ModelToolExchange,
)
from deepmate.runtime.activation import RuntimeActivation
from deepmate.runtime.agent_loop import (
    HistorySink,
    ProviderRetryPolicy,
    StatusSink,
    TokenSink,
    TurnCancellationToken,
    UserTurnResult,
    run_user_turn,
)
from deepmate.runtime.conversation_budget import ConversationBudgetPolicy
from deepmate.runtime.hooks import HookRuntimeContext
from deepmate.runtime.loop_guard import DEFAULT_HARD_STEP_CAP, LoopGuardPolicy
from deepmate.runtime.session_summary import decide_session_summary
from deepmate.runtime.tool_output_compaction import ToolOutputCompactor
from deepmate.runtime.tool_policy import ToolAccessPolicy
from deepmate.runtime.tool_repair import ToolRepairPolicy
from deepmate.runtime.followup import TurnFollowupBuffer
from deepmate.tools import NativeToolRegistry
from deepmate.trace import TraceEvent, TraceRecorder

if TYPE_CHECKING:
    from deepmate.behavior import BehaviorRuntime
    from deepmate.capabilities import CapabilitySurface
    from deepmate.skills import SkillDocument
    from deepmate.subagents import SubagentToolExecutor


@dataclass(frozen=True, slots=True)
class ContextRefreshPolicy:
    """Policy for refreshing profile context between user turns."""

    refresh_after_profile_context_changed: bool = True
    refresh_when_history_over_budget: bool = True
    refresh_when_history_trimmed: bool = True

    def should_refresh_after(self, result: UserTurnResult) -> bool:
        """Return whether a completed user turn crossed a refresh boundary."""
        reports = tuple(
            step.conversation_budget_report
            for step in result.steps
            if step.conversation_budget_report is not None
        )
        if not reports:
            return False
        if self.refresh_when_history_trimmed and any(report.trimmed for report in reports):
            return True
        if self.refresh_when_history_over_budget and any(
            report.over_budget or report.would_drop_count > 0 for report in reports
        ):
            return True
        return False


@dataclass(frozen=True, slots=True)
class SessionRuntime:
    """Runtime state for multiple user turns within one session activation."""

    activation: RuntimeActivation
    conversation: tuple[ModelConversationItem, ...] = field(default_factory=tuple)
    last_user_turn_result: UserTurnResult | None = None
    profile_context_changed: bool = False
    refresh_context_before_next_turn: bool = False
    context_refresh_reason: str = ""
    context_refresh_policy: ContextRefreshPolicy = field(
        default_factory=ContextRefreshPolicy
    )
    behavior_runtime: "BehaviorRuntime | None" = None
    # --- system-context prefix-cache fields (not part of public API) ---
    _last_context_epoch: int = 0
    _last_capability_tag: tuple[str, ...] = field(default_factory=tuple)
    _last_skill_selection_tag: tuple[str, ...] = field(default_factory=tuple)
    _cached_system_context: ContextBuildResult | None = None

    def is_ready(self) -> bool:
        """Return whether this runtime has usable session state."""
        return self.activation.is_ready()

    def mark_profile_context_changed(
        self,
        reason: str = "profile_context_changed",
    ) -> "SessionRuntime":
        """Return a runtime that knows profile context changed on disk."""
        should_refresh = (
            self.context_refresh_policy.refresh_after_profile_context_changed
            or (
                self.last_user_turn_result is not None
                and self.context_refresh_policy.should_refresh_after(
                    self.last_user_turn_result
                )
            )
        )
        return replace(
            self,
            profile_context_changed=True,
            refresh_context_before_next_turn=(
                self.refresh_context_before_next_turn or should_refresh
            ),
            context_refresh_reason=reason.strip() or self.context_refresh_reason,
        )

    def mark_profile_context_changed_on_disk(
        self,
        reason: str = "profile_context_changed_on_disk",
    ) -> "SessionRuntime":
        """Return a runtime that records profile file changes without refresh."""
        return replace(
            self,
            profile_context_changed=True,
            refresh_context_before_next_turn=self.refresh_context_before_next_turn,
            context_refresh_reason=reason.strip() or self.context_refresh_reason,
        )

    def request_context_refresh_before_next_turn(
        self,
        reason: str = "manual_refresh_requested",
    ) -> "SessionRuntime":
        """Return a runtime that will refresh context before the next user turn."""
        return replace(
            self,
            profile_context_changed=True,
            refresh_context_before_next_turn=True,
            context_refresh_reason=reason.strip() or self.context_refresh_reason,
        )

    def with_refreshed_context(
        self,
        context_snapshot: ProfileContextSnapshot | None = None,
    ) -> "SessionRuntime":
        """Return a runtime with a refreshed activation context snapshot."""
        return replace(
            self,
            activation=self.activation.refresh_context(
                context_snapshot=context_snapshot,
                pending_refresh_reason=self.context_refresh_reason
            ),
            profile_context_changed=False,
            refresh_context_before_next_turn=False,
            context_refresh_reason="",
        )

    def with_conversation(
        self,
        conversation: Iterable[ModelConversationItem],
    ) -> "SessionRuntime":
        """Return a runtime with replaced active model-facing conversation."""
        items = tuple(conversation)
        for item in items:
            if not item.is_ready():
                raise ValueError("session runtime conversation item is not ready")
        return replace(self, conversation=items)

    def with_behavior_runtime(
        self,
        behavior_runtime: "BehaviorRuntime | None",
    ) -> "SessionRuntime":
        """Return a runtime with behavior learning integration replaced."""
        return replace(self, behavior_runtime=behavior_runtime)

    def run_user_turn(
        self,
        provider: ModelProvider,
        messages: Iterable[Message],
        model: str,
        capability_surface: "CapabilitySurface | None" = None,
        native_tools: NativeToolRegistry | None = None,
        mcp_tools: McpToolExecutor | None = None,
        subagents: "SubagentToolExecutor | None" = None,
        tool_access_policy: ToolAccessPolicy | None = None,
        tool_schemas: Iterable[Mapping[str, object]] = (),
        tool_exchanges: Iterable[ModelToolExchange] = (),
        selected_skill_documents: Iterable["SkillDocument"] = (),
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
    ) -> "SessionRuntimeUserTurn":
        """Run one user turn and return the updated session runtime."""
        turn_messages = tuple(messages)
        runtime = _mark_behavior_context_changed_if_needed(self, trace_recorder)
        runtime = (
            runtime.with_refreshed_context()
            if runtime.refresh_context_before_next_turn
            else runtime
        )
        if runtime is not self:
            _record_trace(
                trace_recorder,
                "context_snapshot_refreshed",
                "Profile context snapshot refreshed before user turn.",
                refs=(
                    *runtime.activation.trace_refs(),
                    *runtime.activation.context_snapshot.trace_refs(),
                ),
            )

        _record_trace(
            trace_recorder,
            "context_invariants_checked",
            "Runtime context invariants checked before user turn.",
            refs=(
                "activation_snapshot=frozen",
                "conversation=append_only_until_summary_boundary",
                "runtime_scratch=not_profile_memory",
                *runtime.activation.trace_refs(),
            ),
        )

        # ---- prefix-cache: decide whether to build or reuse system context ----
        skill_docs = tuple(selected_skill_documents)
        schema_docs = tuple(tool_schemas)
        current_epoch = runtime.activation.context_epoch
        current_capability_tag = (
            capability_surface.surface_keys()
            if capability_surface is not None and not capability_surface.is_empty()
            else ()
        )
        current_skill_tag = tuple(_skill_cache_tag(skill) for skill in skill_docs)

        if (
            current_epoch == runtime._last_context_epoch
            and current_capability_tag == runtime._last_capability_tag
            and current_skill_tag == runtime._last_skill_selection_tag
            and runtime._cached_system_context is not None
        ):
            system_context = runtime._cached_system_context
            _record_trace(
                trace_recorder,
                "system_context_cache_hit",
                "Reusing cached system context — prefix-cache stable.",
                refs=(
                    f"context_epoch={current_epoch}",
                    f"capability_tag_n={len(current_capability_tag)}",
                    f"skill_tag_n={len(current_skill_tag)}",
                ),
            )
        else:
            system_context = build_system_context_from_snapshot(
                snapshot=runtime.activation.context_snapshot,
                capability_surface=capability_surface,
                selected_skill_documents=skill_docs,
            )
            runtime = replace(
                runtime,
                _last_context_epoch=current_epoch,
                _last_capability_tag=current_capability_tag,
                _last_skill_selection_tag=current_skill_tag,
                _cached_system_context=system_context,
            )
            _record_trace(
                trace_recorder,
                "system_context_cache_miss",
                "Built fresh system context — prefix-cache may be invalidated.",
                refs=(
                    f"context_epoch={current_epoch}",
                    f"capability_tag_n={len(current_capability_tag)}",
                    f"skill_tag_n={len(current_skill_tag)}",
                    *runtime.activation.context_snapshot.trace_refs(),
                ),
            )
            _record_trace(
                trace_recorder,
                "context_injection_policy_evaluated",
                "Context injection policy evaluated for frozen snapshot.",
                refs=(
                    f"context_epoch={current_epoch}",
                    *runtime.activation.context_snapshot.trace_refs(),
                ),
            )
        # ----------------------------------------------------------------

        behavior_tail = (
            runtime.behavior_runtime.prepare_turn_tail(
                turn_messages,
                tool_schema_names=_schema_names(schema_docs),
            )
            if runtime.behavior_runtime is not None
            else None
        )
        if behavior_tail is not None and (
            behavior_tail.messages
            or behavior_tail.matched_rules
            or behavior_tail.disabled_rules
        ):
            _record_trace(
                trace_recorder,
                "behavior_turn_context_prepared",
                "Behavior turn-tail context prepared outside the system prefix.",
                refs=(
                    *behavior_tail.refs,
                    f"matched_rules={len(behavior_tail.matched_rules)}",
                    f"disabled_rules={len(behavior_tail.disabled_rules)}",
                    *runtime.activation.trace_refs(),
                ),
            )

        bound_subagents = (
            subagents.bind_runtime(
                capability_surface=capability_surface,
                native_tools=native_tools,
                mcp_tools=mcp_tools,
                tool_schemas=schema_docs,
                selected_skill_documents=skill_docs,
                activation=runtime.activation,
                parent_tool_access_policy=tool_access_policy,
            )
            if subagents is not None
            else None
        )

        result = run_user_turn(
            provider=provider,
            workspace=runtime.activation.workspace,
            profile=runtime.activation.profile,
            messages=turn_messages,
            model=model,
            capability_surface=capability_surface,
            native_tools=native_tools,
            mcp_tools=mcp_tools,
            subagents=bound_subagents,
            tool_access_policy=tool_access_policy,
            tool_schemas=schema_docs,
            tool_exchanges=tool_exchanges,
            conversation=runtime.conversation,
            ephemeral_turn_tail_messages=(
                behavior_tail.messages if behavior_tail is not None else ()
            ),
            selected_skill_documents=skill_docs,
            activation=runtime.activation,
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
        updated = replace(
            runtime,
            conversation=result.conversation,
            last_user_turn_result=result,
            refresh_context_before_next_turn=(
                runtime.refresh_context_before_next_turn
                or runtime.context_refresh_policy.should_refresh_after(result)
                or (
                    runtime.profile_context_changed
                    and runtime.context_refresh_policy.refresh_after_profile_context_changed
                )
            ),
        )
        if updated.behavior_runtime is not None:
            learned_rules = updated.behavior_runtime.learn_after_turn(
                turn_messages,
                result,
            )
            if learned_rules:
                _record_trace(
                    trace_recorder,
                    "behavior_rules_learned",
                    "Explicit user behavior rules learned from the turn.",
                    refs=(
                        f"rules={len(learned_rules)}",
                        *(f"rule_id={rule.rule_id}" for rule in learned_rules[:8]),
                        *updated.activation.trace_refs(),
                    ),
                )
        if (
            updated.refresh_context_before_next_turn
            and not runtime.refresh_context_before_next_turn
        ):
            _record_trace(
                trace_recorder,
                "context_snapshot_refresh_pending",
                "Profile context snapshot refresh is pending for the next user turn.",
                refs=runtime.activation.trace_refs(),
            )
        _record_session_summary_decision(trace_recorder, updated, result)
        return SessionRuntimeUserTurn(runtime=updated, result=result)


@dataclass(frozen=True, slots=True)
class SessionRuntimeUserTurn:
    """One user turn result plus the updated session runtime state."""

    runtime: SessionRuntime
    result: UserTurnResult


def start_session_runtime(
    activation: RuntimeActivation,
    conversation: Iterable[ModelConversationItem] = (),
    context_refresh_policy: ContextRefreshPolicy | None = None,
    behavior_runtime: "BehaviorRuntime | None" = None,
) -> SessionRuntime:
    """Create session-level runtime state from an activation."""
    runtime = SessionRuntime(
        activation=activation,
        conversation=tuple(conversation),
        context_refresh_policy=context_refresh_policy or ContextRefreshPolicy(),
        behavior_runtime=behavior_runtime,
    )
    if not runtime.is_ready():
        raise ValueError("session runtime is not ready")
    return runtime


def _record_trace(
    recorder: TraceRecorder | None,
    kind: str,
    summary: str,
    refs: tuple[str, ...] = (),
) -> None:
    if recorder is None:
        return
    recorder.record(TraceEvent(kind=kind, summary=summary, refs=refs))


def _mark_behavior_context_changed_if_needed(
    runtime: SessionRuntime,
    trace_recorder: TraceRecorder | None,
) -> SessionRuntime:
    changes = detect_behavior_context_changes(runtime.activation.context_snapshot)
    if not changes:
        return runtime
    _record_trace(
        trace_recorder,
        "behavior_context_changed",
        "Behavior context file changed since activation snapshot.",
        refs=(
            *runtime.activation.trace_refs(),
            *_context_file_change_refs(changes),
        ),
    )
    if runtime.refresh_context_before_next_turn:
        return replace(
            runtime,
            profile_context_changed=True,
            context_refresh_reason=(
                runtime.context_refresh_reason or "behavior_context_changed"
            ),
        )
    return runtime.request_context_refresh_before_next_turn("behavior_context_changed")


def _context_file_change_refs(
    changes: tuple[ContextFileChange, ...],
) -> tuple[str, ...]:
    refs: list[str] = []
    for change in changes:
        refs.extend(change.trace_refs())
    return tuple(refs)


def _skill_cache_tag(skill: "SkillDocument") -> str:
    body = "\n".join((skill.description.strip(), skill.body.strip()))
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
    return f"{skill.name.strip()}:{digest}"


def _schema_names(schemas: Iterable[Mapping[str, object]]) -> tuple[str, ...]:
    names: list[str] = []
    for schema in schemas:
        function = schema.get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
                continue
        name = schema.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return tuple(names)


def _record_session_summary_decision(
    recorder: TraceRecorder | None,
    runtime: SessionRuntime,
    result: UserTurnResult,
) -> None:
    if recorder is None or not result.steps:
        return
    final_step = result.final_step()
    report = final_step.request_budget_report
    if report is None:
        return
    decision = decide_session_summary(
        report,
        cache_hit_ratio=_cache_hit_ratio(final_step.response.usage),
        profile_context_changed=runtime.profile_context_changed,
        history_trimmed=any(
            step.conversation_budget_report is not None
            and step.conversation_budget_report.trimmed
            for step in result.steps
        ),
    )
    _record_trace(
        recorder,
        "session_summary_policy",
        f"Session summary policy decided: {decision.action.value}.",
        refs=(
            *decision.trace_refs(),
            *runtime.activation.trace_refs(),
        ),
    )


def _cache_hit_ratio(usage: object) -> float | None:
    if usage is None:
        return None
    hit = _int_usage_field(usage, "cache_hit_input_tokens")
    miss = _int_usage_field(usage, "cache_miss_input_tokens")
    total = hit + miss
    if total <= 0:
        return None
    return hit / total


def _int_usage_field(usage: object, name: str) -> int:
    value = getattr(usage, name, 0)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return 0
