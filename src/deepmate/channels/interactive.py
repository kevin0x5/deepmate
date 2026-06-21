"""Interactive command-line mode for durable Deepmate sessions."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from deepmate.capabilities import CapabilitySurface
from deepmate.capabilities.state import CapabilityState, CapabilityStateStore
from deepmate.channels.skill_view import (
    discover_skill_cards,
    discover_workspace_skill_cards,
    format_capability_list,
    format_skill_document,
    format_skill_list,
    select_skill_documents,
    workspace_skill_catalog,
)
from deepmate.channels.checkpointing import (
    SessionCheckpointController,
    SessionCheckpointWriteRouter,
)
from deepmate.channels.session_maintenance import runtime_conversation_from_store
from deepmate.channels.remote import (
    RemoteBindingStore,
    format_remote_binding_status,
)
from deepmate.channels.session_lineage import (
    SessionLineageCommandResult,
    handle_session_lineage_command,
)
from deepmate.context import ContextWarning, ProfileContextSnapshot
from deepmate.domain import Message, MessageRole, ProfileRef
from deepmate.mcp import McpServerSpec, McpToolExecutor, format_mcp_server_list
from deepmate.preview_deploy import handle_deploy_command, is_deploy_command
from deepmate.qa import handle_qa_command
from deepmate.providers import ChatCompletionsProvider, ModelCapabilities, ModelResponse
from deepmate.runtime import (
    ConversationBudgetPolicy,
    DeliveryReview,
    DeliveryReviewStatus,
    HookLoadOptions,
    HookRuntimeContext,
    LoopGuardPolicy,
    ProviderRetryPolicy,
    SessionRuntime,
    ToolAccessPolicy,
    ToolOutputCompactor,
    ToolRepairPolicy,
    build_delivery_review_input,
    review_final_response,
    should_run_llm_delivery_review,
    start_session_runtime,
    start_runtime_activation,
)
from deepmate.runtime.hooks import (
    HookTrustStore,
    format_hook_validation,
    format_hooks_status,
    load_hook_report,
)
from deepmate.skills import SkillDocument
from deepmate.storage import SessionRecord, SessionStore, TranscriptStore
from deepmate.subagents import SubagentToolExecutor
from deepmate.tasks import TaskSessionController
from deepmate.tasks.execute import ExecuteLoopUpdate, format_execute_outcome
from deepmate.tools import COMPUTER_TOOL_NAMES, NativeToolRegistry
from deepmate.trace import TraceEvent, TraceRecorder

SessionMaintenanceHandler = Callable[
    [str, SessionRecord, TranscriptStore, SessionRuntime],
    SessionRuntime,
]
SessionEndHandler = Callable[
    [SessionRecord, TranscriptStore, SessionRuntime, str],
    None,
]
ContextSnapshotFactory = Callable[[ProfileRef], ProfileContextSnapshot]
CheckpointControllerFactory = Callable[[SessionRecord], SessionCheckpointController]
TaskMaintenanceHandler = Callable[
    [str, str, SessionRecord, SessionRuntime],
    SessionRuntime | tuple[SessionRuntime, str] | tuple[SessionRuntime, ExecuteLoopUpdate],
]

UNTITLED_SESSION_TITLE = "Untitled session"


def run_interactive_mode(
    provider: ChatCompletionsProvider,
    model: str,
    workspace: Path,
    profile: ProfileRef,
    session_store: SessionStore,
    session: SessionRecord,
    transcript: TranscriptStore,
    runtime: SessionRuntime,
    capability_surface: CapabilitySurface | None,
    native_tools: NativeToolRegistry | None,
    mcp_tools: McpToolExecutor | None,
    subagents: SubagentToolExecutor | None,
    tool_access_policy: ToolAccessPolicy,
    tool_schemas: Sequence[Mapping[str, object]],
    selected_skill_documents: Sequence[SkillDocument],
    mcp_servers: Sequence[McpServerSpec],
    conversation_budget_policy: ConversationBudgetPolicy,
    provider_retry_policy: ProviderRetryPolicy,
    options: Mapping[str, object],
    max_steps: int,
    trace_recorder: TraceRecorder,
    warning_sink: Callable[[ContextWarning], None] | None,
    model_capabilities: ModelCapabilities | None = None,
    loop_guard_policy: LoopGuardPolicy | None = None,
    status_sink: Callable[[str], None] | None = None,
    tool_output_compactor: ToolOutputCompactor | None = None,
    tool_repair_policy: ToolRepairPolicy | None = None,
    hook_context: HookRuntimeContext | None = None,
    hook_load_options: HookLoadOptions | None = None,
    data_dir: Path | None = None,
    maintenance_handler: SessionMaintenanceHandler | None = None,
    session_end_handler: SessionEndHandler | None = None,
    context_snapshot_factory: ContextSnapshotFactory | None = None,
    task_controller: TaskSessionController | None = None,
    task_maintenance_handler: TaskMaintenanceHandler | None = None,
    capability_state_store: CapabilityStateStore | None = None,
    checkpoint_controller: SessionCheckpointController | None = None,
    checkpoint_controller_factory: CheckpointControllerFactory | None = None,
    checkpoint_write_router: SessionCheckpointWriteRouter | None = None,
    initial_prompts: Sequence[str] = (),
    show_reasoning: bool = False,
) -> int:
    """Run multiple user turns in one process until the user exits."""
    _print_interactive_header(session)
    if checkpoint_controller is None and checkpoint_controller_factory is not None:
        checkpoint_controller = checkpoint_controller_factory(session)
    if checkpoint_write_router is not None:
        checkpoint_write_router.set_controller(checkpoint_controller)
    if task_controller is not None and checkpoint_write_router is not None:
        task_controller.store.set_write_checkpoint(
            checkpoint_write_router.capture_workspace_write
        )
    pending_prompts = [(prompt, prompt) for prompt in initial_prompts]
    turn_index = 0
    while True:
        if pending_prompts:
            prompt, display_prompt = pending_prompts.pop(0)
            prompt = prompt.strip()
            if prompt:
                print(f"deepmate> {display_prompt.strip() or prompt}")
        else:
            try:
                prompt = input("deepmate> ").strip()
            except EOFError:
                print()
                if turn_index > 0:
                    _handle_session_end(
                        session_end_handler,
                        session,
                        transcript,
                        runtime,
                        "eof",
                    )
                return 0
            except KeyboardInterrupt:
                print()
                if turn_index > 0:
                    _handle_session_end(
                        session_end_handler,
                        session,
                        transcript,
                        runtime,
                        "keyboard_interrupt",
                    )
                return 130

        if not prompt:
            continue
        task_turn = (
            task_controller.prepare_prompt(prompt)
            if task_controller is not None
            else None
        )
        if task_turn is not None:
            if task_turn.is_control():
                try:
                    print(task_controller.handle_control(task_turn.control))
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    print(f"error: {exc}", file=sys.stderr)
                continue
            prompt = task_turn.prompt
            task_controller.save_cursor(session_id=session.session_id)
            if context_snapshot_factory is not None:
                runtime = runtime.with_refreshed_context(
                    context_snapshot_factory(session.profile)
                )
                _record_interactive_trace(
                    trace_recorder,
                    "task_mode_context_refreshed",
                    "Task Mode context updated before task turn.",
                    session=session,
                    runtime=runtime,
                    turn_index=turn_index,
                )
        command = _handle_command(
            prompt=prompt,
            session_store=session_store,
            session=session,
            transcript=transcript,
            runtime=runtime,
            workspace=workspace,
            profile=profile,
            mcp_servers=mcp_servers,
            trace_recorder=trace_recorder,
            context_snapshot_factory=context_snapshot_factory,
            capability_state_store=capability_state_store,
            checkpoint_controller_factory=checkpoint_controller_factory,
            checkpoint_controller=checkpoint_controller,
            hook_context=hook_context,
            hook_load_options=hook_load_options,
            data_dir=data_dir,
            native_tools=native_tools,
            tool_schemas=tool_schemas,
        )
        if command.action == "exit":
            if turn_index > 0:
                _handle_session_end(
                    session_end_handler,
                    session,
                    transcript,
                    runtime,
                    "command",
                )
            return 0
        if command.session is not None:
            session = command.session
        if command.transcript is not None:
            transcript = command.transcript
        if command.runtime is not None:
            runtime = command.runtime
        if command.tool_schemas is not None:
            tool_schemas = command.tool_schemas
        if command.checkpoint_controller is not None:
            checkpoint_controller = command.checkpoint_controller
            if checkpoint_write_router is not None:
                checkpoint_write_router.set_controller(checkpoint_controller)
        if command.action == "handled":
            continue

        _close_remote_routes_for_local_turn(
            session_store=session_store,
            session=session,
            data_dir=data_dir,
            trace_recorder=trace_recorder,
            runtime=runtime,
        )
        current_turn_index = turn_index + 1
        turn_scope = (
            checkpoint_controller.start_turn(_load_latest_summary(session_store, session))
            if checkpoint_controller is not None
            else None
        )
        try:
            session = _ensure_session_title(session_store, session, prompt)
            _record_interactive_trace(
                trace_recorder,
                "interactive_turn_started",
                "Interactive user turn started.",
                session=session,
                runtime=runtime,
                turn_index=current_turn_index,
            )
            turn = runtime.run_user_turn(
                provider=provider,
                messages=(Message(role=MessageRole.USER, content=prompt),),
                model=model,
                capability_surface=capability_surface,
                native_tools=native_tools,
                mcp_tools=mcp_tools,
                subagents=subagents,
                tool_access_policy=tool_access_policy,
                tool_schemas=tool_schemas,
                selected_skill_documents=selected_skill_documents,
                conversation_budget_policy=conversation_budget_policy,
                provider_retry_policy=provider_retry_policy,
                options=options,
                model_capabilities=model_capabilities,
                max_steps=max_steps,
                loop_guard_policy=loop_guard_policy,
                trace_recorder=trace_recorder,
                warning_sink=warning_sink,
                history_sink=(
                    turn_scope.history_sink(transcript)
                    if turn_scope is not None
                    else transcript.append_item
                ),
                status_sink=status_sink,
                tool_output_compactor=tool_output_compactor,
                tool_repair_policy=tool_repair_policy,
                hook_context=hook_context,
            )
            runtime = turn.runtime
            result = turn.result
            if turn_scope is not None:
                turn_scope.mark_result(result)
            delivery_review = _review_delivery(
                user_request=prompt,
                final_response=_response_text(result.final_step().response),
                result=result,
                trace_recorder=trace_recorder,
                session=session,
                runtime=runtime,
            )
            _print_delivery_review_warning(delivery_review)
            _print_response(result.final_step().response, show_reasoning=show_reasoning)
            if result.loop_guard_stop is not None:
                print(f"stopped: {result.loop_guard_stop.message}", file=sys.stderr)
                print("hint: enter `continue` or `继续` to resume.", file=sys.stderr)
            elif result.reached_max_steps:
                print(
                    f"error: reached max_steps={max_steps} before final answer",
                    file=sys.stderr,
                )
            for error in result.errors():
                if (
                    result.loop_guard_stop is not None
                    and error.code
                    == f"loop_guard_{result.loop_guard_stop.reason.value}"
                ):
                    continue
                print(f"error: {error.message}", file=sys.stderr)
            session = session_store.touch(session.session_id)
            _record_interactive_trace(
                trace_recorder,
                "interactive_turn_finished",
                "Interactive user turn finished.",
                session=session,
                runtime=runtime,
                turn_index=current_turn_index,
            )
            turn_index = current_turn_index
            if (
                task_maintenance_handler is not None
                and task_turn is not None
                and not result.has_errors()
                and not result.reached_max_steps
            ):
                task_maintenance = task_maintenance_handler(
                    prompt,
                    _response_text(result.final_step().response),
                    session,
                    runtime,
                )
                runtime, continuation, task_status = _task_maintenance_result(
                    task_maintenance
                )
                if task_status:
                    print()
                    print(task_status)
                if continuation:
                    pending_prompts.insert(
                        0,
                        (continuation, "task/execute auto-continue"),
                    )
                if context_snapshot_factory is not None:
                    runtime = runtime.request_context_refresh_before_next_turn(
                        "task_mode_updated"
                    )
            if (
                maintenance_handler is not None
                and not result.has_errors()
                and not result.reached_max_steps
            ):
                runtime = maintenance_handler(prompt, session, transcript, runtime)
                if turn_scope is not None:
                    turn_scope.attach_summary(
                        _load_latest_summary(session_store, session)
                    )
        except KeyboardInterrupt:
            if turn_scope is not None:
                turn_scope.mark_interrupted()
            print()
            if turn_index > 0:
                _handle_session_end(
                    session_end_handler,
                    session,
                    transcript,
                    runtime,
                    "keyboard_interrupt",
                )
            return 130
        except Exception as exc:
            if turn_scope is not None:
                turn_scope.mark_failed(type(exc).__name__)
            print(f"error: {exc}", file=sys.stderr)
            trace_recorder.record(
                TraceEvent(
                    kind="interactive_turn_failed",
                    summary=f"Interactive user turn failed: {exc}",
                    refs=(
                        f"session_id={session.session_id}",
                        f"turn_index={current_turn_index}",
                        *runtime.activation.trace_refs(),
                    ),
                )
            )
        finally:
            if turn_scope is not None:
                turn_scope.close()
    return 0


def _task_maintenance_result(
    value: SessionRuntime | tuple[SessionRuntime, str] | tuple[SessionRuntime, ExecuteLoopUpdate],
) -> tuple[SessionRuntime, str, str]:
    if isinstance(value, tuple):
        runtime = value[0]
        extra = value[1] if len(value) > 1 else ""
        if isinstance(extra, ExecuteLoopUpdate):
            return (
                runtime,
                extra.continuation,
                format_execute_outcome(extra.evaluation),
            )
        return runtime, str(extra).strip(), ""
    return value, "", ""


def _handle_session_end(
    handler: SessionEndHandler | None,
    session: SessionRecord,
    transcript: TranscriptStore,
    runtime: SessionRuntime,
    reason: str,
) -> None:
    if handler is None:
        return
    handler(session, transcript, runtime, reason)


@dataclass(frozen=True, slots=True)
class _CommandResult:
    action: str
    session: SessionRecord | None = None
    transcript: TranscriptStore | None = None
    runtime: SessionRuntime | None = None
    tool_schemas: tuple[Mapping[str, object], ...] | None = None
    checkpoint_controller: SessionCheckpointController | None = None


def _handle_command(
    prompt: str,
    session_store: SessionStore,
    session: SessionRecord,
    transcript: TranscriptStore,
    runtime: SessionRuntime,
    workspace: Path,
    profile: ProfileRef,
    mcp_servers: Sequence[McpServerSpec],
    trace_recorder: TraceRecorder,
    context_snapshot_factory: ContextSnapshotFactory | None,
    capability_state_store: CapabilityStateStore | None,
    checkpoint_controller_factory: CheckpointControllerFactory | None,
    checkpoint_controller: SessionCheckpointController | None = None,
    hook_context: HookRuntimeContext | None = None,
    hook_load_options: HookLoadOptions | None = None,
    data_dir: Path | None = None,
    native_tools: NativeToolRegistry | None = None,
    tool_schemas: Sequence[Mapping[str, object]] = (),
) -> _CommandResult:
    if prompt in {"/exit", "/quit"}:
        return _CommandResult("exit")
    lineage_result = _handle_session_lineage_interactive(
        prompt=prompt,
        session_store=session_store,
        session=session,
        runtime=runtime,
        workspace=workspace,
        profile=profile,
        trace_recorder=trace_recorder,
        context_snapshot_factory=context_snapshot_factory,
        checkpoint_controller_factory=checkpoint_controller_factory,
        checkpoint_controller=checkpoint_controller,
    )
    if lineage_result is not None:
        return lineage_result
    if is_deploy_command(prompt):
        try:
            output = handle_deploy_command(
                prompt,
                workspace=workspace,
                data_dir=data_dir or session_store.directory.parent,
                owner_session_id=session.session_id,
                owner_session_workspace=session.workspace,
            )
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            print(str(exc))
            return _CommandResult("handled")
        trace_recorder.record(
            TraceEvent(
                kind="preview_deploy_command",
                summary="Preview deploy command handled.",
                refs=(
                    f"session_id={session.session_id}",
                    f"workspace={workspace}",
                    *runtime.activation.trace_refs(),
                ),
            )
        )
        print(output)
        return _CommandResult("handled")
    if prompt == "/qa" or prompt.startswith("/qa "):
        try:
            print(
                handle_qa_command(
                    prompt,
                    workspace=workspace,
                    provider=provider,
                    model=model,
                    options=options,
                )
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
        return _CommandResult("handled")
    if prompt == "/remote":
        store = RemoteBindingStore.in_data_dir(data_dir or session_store.directory.parent)
        print(format_remote_binding_status(store.list_records()))
        return _CommandResult("handled")
    if prompt == "/remote --wecom":
        store = RemoteBindingStore.in_data_dir(data_dir or session_store.directory.parent)
        record = store.bind_default_session(
            channel="wecom",
            session=session,
            bound_from="interactive",
            route_open=False,
        )
        trace_recorder.record(
            TraceEvent(
                kind="remote_session_bound",
                summary="Enterprise WeChat remote binding updated.",
                refs=(
                    f"channel={record.channel}",
                    f"remote_user_id={record.remote_user_id}",
                    f"session_id={record.session_id}",
                    f"workspace={record.workspace}",
                    *runtime.activation.trace_refs(),
                ),
            )
        )
        print("企业微信已绑定到当前 session。")
        print("remote route: closed")
        print(f"session: {session.session_id} ({session.title})")
        print(f"workspace: {session.workspace}")
        return _CommandResult("handled")
    if prompt == "/remote --open wecom":
        behavior_runtime = getattr(runtime, "behavior_runtime", None)
        if behavior_runtime is not None:
            behavior_runtime.set_computer_use(False)
        store = RemoteBindingStore.in_data_dir(data_dir or session_store.directory.parent)
        record = store.bind_default_session(
            channel="wecom",
            session=session,
            bound_from="interactive",
            route_open=True,
        )
        trace_recorder.record(
            TraceEvent(
                kind="remote_route_opened",
                summary="Enterprise WeChat remote route opened for current session.",
                refs=(
                    f"channel={record.channel}",
                    f"remote_user_id={record.remote_user_id}",
                    f"session_id={record.session_id}",
                    f"workspace={record.workspace}",
                    *runtime.activation.trace_refs(),
                ),
            )
        )
        print("企业微信已接管当前 session。")
        print("remote route: open")
        print(f"session: {session.session_id} ({session.title})")
        print(f"workspace: {session.workspace}")
        return _CommandResult("handled")
    if prompt == "/remote --close wecom":
        store = RemoteBindingStore.in_data_dir(data_dir or session_store.directory.parent)
        closed = store.close_session_routes(
            session_id=session.session_id,
            channel="wecom",
        )
        if closed:
            trace_recorder.record(
                TraceEvent(
                    kind="remote_route_closed",
                    summary="Enterprise WeChat remote route closed for current session.",
                    refs=(
                        f"channel=wecom",
                        f"session_id={session.session_id}",
                        *runtime.activation.trace_refs(),
                    ),
                )
            )
            print("企业微信已停止接管当前 session。")
        else:
            print("当前 session 没有打开的企业微信 remote route。")
        return _CommandResult("handled")
    if prompt == "/remote --unbind wecom":
        store = RemoteBindingStore.in_data_dir(data_dir or session_store.directory.parent)
        removed = store.remove_channel("wecom")
        if not removed:
            print("企业微信没有 session 绑定。")
        else:
            print(f"已解除企业微信绑定：{len(removed)} 条")
        return _CommandResult("handled")
    if prompt == "/sessions":
        _print_sessions(session_store.list_recent())
        return _CommandResult("handled")
    if prompt == "/skills":
        _print_skills(workspace, capability_state_store)
        return _CommandResult("handled")
    if prompt.startswith("/skills "):
        _handle_skill_state_command(
            prompt=prompt,
            workspace=workspace,
            state_store=capability_state_store,
            trace_recorder=trace_recorder,
            session=session,
            runtime=runtime,
        )
        return _CommandResult("handled")
    if prompt == "/mcp":
        print(format_mcp_server_list(mcp_servers))
        return _CommandResult("handled")
    if prompt == "/hooks" or prompt.startswith("/hooks "):
        _handle_hooks_command(
            prompt=prompt,
            workspace=workspace,
            data_dir=data_dir,
            hook_context=hook_context,
            hook_load_options=hook_load_options,
            trace_recorder=trace_recorder,
            session=session,
            runtime=runtime,
        )
        return _CommandResult("handled")
    if prompt == "/behavior" or prompt.startswith("/behavior "):
        _handle_behavior_command(prompt, runtime)
        return _CommandResult("handled")
    if prompt == "/computer" or prompt.startswith("/computer "):
        updated_schemas = _handle_computer_command(
            prompt,
            runtime,
            native_tools=native_tools,
            tool_schemas=tool_schemas,
        )
        return _CommandResult("handled", tool_schemas=updated_schemas)
    if prompt == "/show-skill" or prompt.startswith("/show-skill "):
        name = prompt[len("/show-skill") :].strip()
        if not name:
            print("usage: /show-skill <skill_name>", file=sys.stderr)
            return _CommandResult("handled")
        _print_skill_detail(workspace, name)
        return _CommandResult("handled")
    if prompt == "/resume":
        _print_sessions(session_store.list_recent())
        print("use /resume <session_id>")
        return _CommandResult("handled")
    if prompt.startswith("/resume "):
        return _resume_session(
            raw_session_id=prompt[len("/resume ") :].strip(),
            session_store=session_store,
            current_session=session,
            current_runtime=runtime,
            current_workspace=workspace,
            current_profile=profile,
            trace_recorder=trace_recorder,
            context_snapshot_factory=context_snapshot_factory,
            checkpoint_controller_factory=checkpoint_controller_factory,
        )
    if prompt.startswith("/title"):
        title = prompt[len("/title") :].strip()
        if not title:
            print("usage: /title <new title>", file=sys.stderr)
            return _CommandResult("handled")
        updated = session_store.rename(session.session_id, title)
        print(f"title: {updated.title}")
        return _CommandResult("handled", session=updated)
    return _CommandResult("")


def _print_interactive_header(session: SessionRecord) -> None:
    print(f"Deepmate interactive mode. Session: {session.session_id}")
    print(
        "Commands: /session, /session tree|clone|fork, /sessions, /skills, "
        "/skills heat|cool|hide|restore "
        "<name>, /show-skill <name>, /mcp, /hooks status|validate|trust|reload, "
        "/qa <goal>|run|status, /behavior, /computer on|off|status, "
        "/resume <id>, /title <title>, /exit"
    )


def _print_response(response: ModelResponse, show_reasoning: bool) -> None:
    reasoning = response.reasoning.strip()
    content = response.content.strip()
    if show_reasoning and reasoning:
        print(reasoning)
        if content:
            print()
    if content:
        print(content)
    elif reasoning and not show_reasoning:
        print(reasoning)


def _handle_behavior_command(prompt: str, runtime: SessionRuntime) -> None:
    behavior = runtime.behavior_runtime
    if behavior is None:
        print("Behavior learning is unavailable in this session.")
        return
    target = prompt[len("/behavior") :].strip().lower()
    if target in {"on", "enable", "enabled"}:
        behavior.set_interaction_learning(True)
        print("Behavior learning: on")
    elif target in {"off", "disable", "disabled"}:
        behavior.set_interaction_learning(False)
        print("Behavior learning: off")
    elif target in {"forget", "clear"}:
        disabled = behavior.rule_store.disable_matching("all")
        print(f"Behavior rules disabled: {len(disabled)}")
    elif target in {"", "status"}:
        print(behavior.status_text())
    else:
        print("usage: /behavior, /behavior on, /behavior off, /behavior forget")


def _handle_computer_command(
    prompt: str,
    runtime: SessionRuntime,
    *,
    native_tools: NativeToolRegistry | None = None,
    tool_schemas: Sequence[Mapping[str, object]] = (),
) -> tuple[Mapping[str, object], ...] | None:
    behavior = runtime.behavior_runtime
    if behavior is None:
        print("Computer Use is unavailable in this session.")
        return None
    target = prompt[len("/computer") :].strip()
    lowered = target.lower()
    if lowered in {"on", "enable", "enabled"} or lowered.startswith("on "):
        task = target[2:].strip() if lowered.startswith("on ") else ""
        behavior.set_computer_use(True, task=task)
        updated_schemas = _computer_tool_schemas(
            native_tools,
            tool_schemas,
            enabled=True,
        )
        print(
            "Computer Use: on for this session. Deepmate can use browser tools "
            "and macOS screenshot/click/type/key/open actions for the current task."
        )
        return updated_schemas
    elif lowered in {"off", "disable", "disabled"}:
        behavior.set_computer_use(False)
        updated_schemas = _computer_tool_schemas(
            native_tools,
            tool_schemas,
            enabled=False,
        )
        print("Computer Use: off")
        return updated_schemas
    elif lowered in {"learning on", "learn on"}:
        print(
            "Long-term Computer Use learning is not enabled yet. Computer Use can "
            "still run current-task actions, but Deepmate will not save desktop "
            "behavior rules until review/confirm/rollback is available."
        )
    elif lowered in {"learning off", "learn off"}:
        behavior.set_computer_learning(False)
        print("Long-term learning from Computer Use: off")
    elif lowered in {"", "status"}:
        print(behavior.status_text())
    else:
        print("usage: /computer on, /computer off, /computer status, /computer learning on|off")
    return None


def _computer_tool_schemas(
    native_tools: NativeToolRegistry | None,
    tool_schemas: Sequence[Mapping[str, object]],
    *,
    enabled: bool,
) -> tuple[Mapping[str, object], ...]:
    schemas = [
        schema
        for schema in tool_schemas
        if _schema_name(schema) not in COMPUTER_TOOL_NAMES
    ]
    if not enabled or native_tools is None:
        return tuple(schemas)
    visible = {_schema_name(schema) for schema in schemas}
    for name in COMPUTER_TOOL_NAMES:
        if name in visible:
            continue
        tool = native_tools.get(name)
        if tool is not None:
            schemas.append(tool.schema())
            visible.add(name)
    return tuple(schemas)


def _schema_name(schema: Mapping[str, object]) -> str:
    value = schema.get("name")
    return value.strip() if isinstance(value, str) else ""


def _response_text(response: ModelResponse) -> str:
    return response.content.strip() or response.reasoning.strip()


def _review_delivery(
    user_request: str,
    final_response: str,
    result,
    trace_recorder: TraceRecorder,
    session: SessionRecord,
    runtime: SessionRuntime,
) -> DeliveryReview:
    review_input = build_delivery_review_input(
        user_request=user_request,
        final_response_draft=final_response,
        tool_exchanges=result.tool_exchanges,
        errors=result.errors(),
        reached_max_steps=result.reached_max_steps,
    )
    review = review_final_response(review_input)
    trace_recorder.record(
        TraceEvent(
            kind="delivery_review_finished",
            summary=review.summary,
            refs=(
                f"session_id={session.session_id}",
                f"status={review.status.value}",
                f"issues={len(review.issues)}",
                f"llm_review_suggested={should_run_llm_delivery_review(review_input)}",
                *review.issues,
                *runtime.activation.trace_refs(),
            ),
        )
    )
    return review


def _record_capability_state_trace(
    recorder: TraceRecorder,
    kind: str,
    summary: str,
    state: CapabilityState,
    session: SessionRecord,
    runtime: SessionRuntime,
) -> None:
    recorder.record(
        TraceEvent(
            kind=kind,
            summary=summary,
            refs=(
                f"session_id={session.session_id}",
                f"capability_id={state.capability_id}",
                f"capability_kind={state.kind.value}",
                f"skill={state.name}",
                f"temperature={state.temperature.value}",
                f"exposure={state.exposure()}",
                *runtime.activation.trace_refs(),
            ),
        )
    )


def _print_delivery_review_warning(review: DeliveryReview) -> None:
    if review.status == DeliveryReviewStatus.ACCEPTED:
        return
    if review.status == DeliveryReviewStatus.BLOCKED:
        print(f"error: {review.summary}", file=sys.stderr)
        return
    if review.issues:
        print(
            "warning: delivery review needs attention: "
            + ", ".join(review.issues),
            file=sys.stderr,
        )


def _print_sessions(sessions: Sequence[SessionRecord]) -> None:
    if not sessions:
        print("No sessions found.")
        return
    print(f"{'SESSION ID: TITLE':<72}  UPDATED")
    for session in sessions:
        print(
            f"{session.session_id}: {session.title:<37}  "
            f"{session.updated_at}"
        )


def _handle_session_lineage_interactive(
    prompt: str,
    session_store: SessionStore,
    session: SessionRecord,
    runtime: SessionRuntime,
    workspace: Path,
    profile: ProfileRef,
    trace_recorder: TraceRecorder,
    context_snapshot_factory: ContextSnapshotFactory | None,
    checkpoint_controller_factory: CheckpointControllerFactory | None,
    checkpoint_controller: SessionCheckpointController | None,
) -> _CommandResult | None:
    if not _is_session_lineage_command(prompt):
        return None
    try:
        result = handle_session_lineage_command(
            prompt,
            session_store=session_store,
            session=session,
            workspace=workspace,
            profile=profile,
            turn_store=(
                checkpoint_controller.turn_store
                if checkpoint_controller is not None
                else None
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _CommandResult("handled")
    if result is None:
        return None
    if isinstance(result, str):
        print(result)
        return _CommandResult("handled")
    return _switch_to_lineage_session(
        result=result,
        source_session=session,
        session_store=session_store,
        trace_recorder=trace_recorder,
        current_runtime=runtime,
        context_snapshot_factory=context_snapshot_factory,
        checkpoint_controller_factory=checkpoint_controller_factory,
    )


def _switch_to_lineage_session(
    result: SessionLineageCommandResult,
    source_session: SessionRecord,
    session_store: SessionStore,
    trace_recorder: TraceRecorder,
    current_runtime: SessionRuntime,
    context_snapshot_factory: ContextSnapshotFactory | None,
    checkpoint_controller_factory: CheckpointControllerFactory | None,
) -> _CommandResult:
    session = result.session
    transcript = session_store.transcript_store(session)
    checkpoint_controller = (
        checkpoint_controller_factory(session)
        if checkpoint_controller_factory is not None
        else None
    )
    activation = start_runtime_activation(
        session_id=session.session_id,
        workspace=session.workspace,
        profile=session.profile,
        context_snapshot=(
            context_snapshot_factory(session.profile)
            if context_snapshot_factory is not None
            else None
        ),
    )
    runtime = start_session_runtime(
        activation,
        conversation=runtime_conversation_from_store(
            session_store,
            session,
            transcript,
            turn_checkpoint_store=(
                checkpoint_controller.turn_store
                if checkpoint_controller is not None
                else None
            ),
        ),
        behavior_runtime=(
            current_runtime.behavior_runtime.with_profile(
                workspace=session.workspace,
                profile=session.profile,
                session_id=session.session_id,
            )
            if current_runtime.behavior_runtime is not None
            else None
        ),
    )
    print(result.body)
    trace_recorder.record(
        TraceEvent(
            kind=(
                "session_branch_clone"
                if session.fork_kind == "clone"
                else "session_branch_fork"
            ),
            summary="Session lineage session created and activated.",
            refs=(
                f"source_session_id={source_session.session_id}",
                f"session_id={session.session_id}",
                f"fork_kind={session.fork_kind}",
                f"forked_from_turn_id={session.forked_from_turn_id}",
                f"forked_from_sequence={session.forked_from_sequence}",
                *current_runtime.activation.trace_refs(),
            ),
        )
    )
    return _CommandResult(
        "handled",
        session=session,
        transcript=transcript,
        runtime=runtime,
        checkpoint_controller=checkpoint_controller,
    )


def _is_session_lineage_command(prompt: str) -> bool:
    clean = prompt.strip()
    return (
        clean == "/session"
        or clean.startswith("/session ")
        or clean == "/tree"
        or clean == "/clone"
        or clean.startswith("/clone ")
        or clean == "/fork"
        or clean.startswith("/fork ")
    )


def _print_skills(
    workspace: Path,
    state_store: CapabilityStateStore | None = None,
) -> None:
    cards, warnings = discover_skill_cards(workspace)
    for warning in warnings:
        print(f"warning: {warning.message}", file=sys.stderr)
    if state_store is None:
        print(format_skill_list(cards, workspace))
        return
    workspace_cards, _workspace_warnings = discover_workspace_skill_cards(workspace)
    state_store.sync_workspace_skills(workspace_cards, workspace)
    print(format_capability_list(cards, workspace, state_store))


def _handle_skill_state_command(
    prompt: str,
    workspace: Path,
    state_store: CapabilityStateStore | None,
    trace_recorder: TraceRecorder,
    session: SessionRecord,
    runtime: SessionRuntime,
) -> None:
    if state_store is None:
        print("error: skill state store is not available", file=sys.stderr)
        return
    parts = prompt.split(maxsplit=2)
    if len(parts) != 3 or parts[1] not in {"heat", "cool", "hide", "restore"}:
        print("usage: /skills heat|cool|hide|restore <skill_name>", file=sys.stderr)
        return
    cards, warnings = discover_workspace_skill_cards(workspace)
    for warning in warnings:
        print(f"warning: {warning.message}", file=sys.stderr)
    state_store.sync_workspace_skills(cards, workspace)
    try:
        state = state_store.set_skill_state(parts[2], parts[1])
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return
    _record_capability_state_trace(
        recorder=trace_recorder,
        kind="capability_state_updated",
        summary=f"Skill state updated: {state.name}.",
        state=state,
        session=session,
        runtime=runtime,
    )
    print(
        f"skill {state.name}: temperature={state.temperature.value}, "
        f"exposure={state.exposure()}"
    )


def _handle_hooks_command(
    prompt: str,
    workspace: Path,
    data_dir: Path | None,
    hook_context: HookRuntimeContext | None,
    hook_load_options: HookLoadOptions | None,
    trace_recorder: TraceRecorder,
    session: SessionRecord,
    runtime: SessionRuntime,
) -> None:
    if data_dir is None:
        print("error: hook data_dir is not available", file=sys.stderr)
        return
    command = prompt[len("/hooks") :].strip() or "status"
    if command not in {"status", "validate", "trust", "reload"}:
        print("usage: /hooks status|validate|trust|reload", file=sys.stderr)
        return
    options = hook_load_options or HookLoadOptions()
    if command == "trust":
        try:
            record = HookTrustStore.in_data_dir(data_dir).trust_workspace(workspace)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return
        print("Workspace trusted for project hooks:")
        print(f"- workspace: {record.workspace}")
        print(f"- workspace_hash: {record.workspace_hash[:16]}")
        print(f"- trusted_at: {record.trusted_at}")
        _record_interactive_trace(
            trace_recorder,
            "hooks_workspace_trusted",
            "Workspace trusted for project hooks in interactive mode.",
            session=session,
            runtime=runtime,
            turn_index=0,
        )
        return
    try:
        report = load_hook_report(workspace, data_dir, options)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return
    if command == "validate":
        print(format_hook_validation(report, workspace))
        return
    if command == "reload":
        if hook_context is None:
            print("error: hook runtime context is not available", file=sys.stderr)
            return
        hook_context.reload_registry(report.registry)
        print(format_hooks_status(report, workspace))
        print("- reloaded: true")
        _record_interactive_trace(
            trace_recorder,
            "hooks_reloaded",
            "Interactive hook registry reloaded.",
            session=session,
            runtime=runtime,
            turn_index=0,
        )
        return
    print(format_hooks_status(report, workspace))


def _print_skill_detail(workspace: Path, name: str) -> None:
    try:
        catalog, warnings = workspace_skill_catalog(workspace)
        for warning in warnings:
            print(f"warning: {warning.message}", file=sys.stderr)
        document = select_skill_documents(
            catalog,
            (name,),
            workspace,
            command_name="/show-skill",
        )[0]
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return
    print(format_skill_document(document, workspace))


def _resume_session(
    raw_session_id: str,
    session_store: SessionStore,
    current_session: SessionRecord,
    current_runtime: SessionRuntime,
    current_workspace: Path,
    current_profile: ProfileRef,
    trace_recorder: TraceRecorder,
    context_snapshot_factory: ContextSnapshotFactory | None,
    checkpoint_controller_factory: CheckpointControllerFactory | None,
) -> _CommandResult:
    if not raw_session_id:
        print("usage: /resume <session_id>", file=sys.stderr)
        _record_resume_failed(
            trace_recorder,
            current_session=current_session,
            current_runtime=current_runtime,
            attempted_session_id="",
            reason="missing_session_id",
        )
        return _CommandResult("handled")
    try:
        session_id = session_store.resolve_id(raw_session_id)
        session = session_store.load(session_id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        _record_resume_failed(
            trace_recorder,
            current_session=current_session,
            current_runtime=current_runtime,
            attempted_session_id=raw_session_id,
            reason="not_found_or_ambiguous",
        )
        return _CommandResult("handled")
    if session.workspace.resolve() != current_workspace.resolve():
        print(
            "error: session belongs to another workspace; "
            f"start a new command with --session-id {session.session_id}",
            file=sys.stderr,
        )
        _record_resume_failed(
            trace_recorder,
            current_session=current_session,
            current_runtime=current_runtime,
            attempted_session_id=session.session_id,
            reason="workspace_mismatch",
        )
        return _CommandResult("handled")
    if session.profile != current_profile:
        print(
            "error: session uses another profile; "
            f"start a new command with --session-id {session.session_id}",
            file=sys.stderr,
        )
        _record_resume_failed(
            trace_recorder,
            current_session=current_session,
            current_runtime=current_runtime,
            attempted_session_id=session.session_id,
            reason="profile_mismatch",
        )
        return _CommandResult("handled")
    transcript = session_store.transcript_store(session)
    checkpoint_controller = (
        checkpoint_controller_factory(session)
        if checkpoint_controller_factory is not None
        else None
    )
    activation = start_runtime_activation(
        session_id=session.session_id,
        workspace=session.workspace,
        profile=session.profile,
        context_snapshot=(
            context_snapshot_factory(session.profile)
            if context_snapshot_factory is not None
            else None
        ),
    )
    runtime = start_session_runtime(
        activation,
        conversation=runtime_conversation_from_store(
            session_store,
            session,
            transcript,
            turn_checkpoint_store=(
                checkpoint_controller.turn_store
                if checkpoint_controller is not None
                else None
            ),
        ),
        behavior_runtime=(
            current_runtime.behavior_runtime.with_profile(
                workspace=session.workspace,
                profile=session.profile,
                session_id=session.session_id,
            )
            if current_runtime.behavior_runtime is not None
            else None
        ),
    )
    print(f"resumed: {session.session_id}: {session.title}")
    _record_resume_success(
        trace_recorder,
        from_session=current_session,
        to_session=session,
        runtime=runtime,
    )
    return _CommandResult(
        "handled",
        session=session,
        transcript=transcript,
        runtime=runtime,
        checkpoint_controller=checkpoint_controller,
    )


def _load_latest_summary(
    session_store: SessionStore,
    session: SessionRecord,
):
    try:
        return session_store.summary_store(session).load_latest()
    except Exception:
        return None


def _ensure_session_title(
    session_store: SessionStore,
    session: SessionRecord,
    prompt: str,
) -> SessionRecord:
    if session.title != UNTITLED_SESSION_TITLE:
        return session
    return session_store.rename(session.session_id, _title_from_prompt(prompt))


def _title_from_prompt(prompt: str, limit: int = 48) -> str:
    title = " ".join(prompt.split())
    if not title:
        return UNTITLED_SESSION_TITLE
    if len(title) <= limit:
        return title
    return title[: limit - 3].rstrip() + "..."


def _close_remote_routes_for_local_turn(
    *,
    session_store: SessionStore,
    session: SessionRecord,
    data_dir: Path | None,
    trace_recorder: TraceRecorder,
    runtime: SessionRuntime,
) -> None:
    """Release remote delivery ownership when the user resumes locally."""
    store = RemoteBindingStore.in_data_dir(data_dir or session_store.directory.parent)
    try:
        closed = store.close_session_routes(session_id=session.session_id)
    except (OSError, ValueError, json.JSONDecodeError):
        return
    if not closed:
        return
    trace_recorder.record(
        TraceEvent(
            kind="remote_route_closed",
            summary="Remote route closed because a local turn started.",
            refs=(
                f"session_id={session.session_id}",
                *(f"channel={record.channel}" for record in closed),
                *runtime.activation.trace_refs(),
            ),
        )
    )


def _record_interactive_trace(
    recorder: TraceRecorder,
    kind: str,
    summary: str,
    session: SessionRecord,
    runtime: SessionRuntime,
    turn_index: int,
) -> None:
    recorder.record(
        TraceEvent(
            kind=kind,
            summary=summary,
            refs=(
                f"session_id={session.session_id}",
                f"turn_index={turn_index}",
                *runtime.activation.trace_refs(),
            ),
        )
    )


def _record_resume_success(
    recorder: TraceRecorder,
    from_session: SessionRecord,
    to_session: SessionRecord,
    runtime: SessionRuntime,
) -> None:
    recorder.record(
        TraceEvent(
            kind="interactive_session_resumed",
            summary="Interactive session resumed.",
            refs=(
                f"from_session_id={from_session.session_id}",
                f"to_session_id={to_session.session_id}",
                *runtime.activation.trace_refs(),
            ),
        )
    )


def _record_resume_failed(
    recorder: TraceRecorder,
    current_session: SessionRecord,
    current_runtime: SessionRuntime,
    attempted_session_id: str,
    reason: str,
) -> None:
    refs = [
        f"current_session_id={current_session.session_id}",
        f"reason={reason}",
        *current_runtime.activation.trace_refs(),
    ]
    if attempted_session_id:
        refs.append(f"attempted_session_id={attempted_session_id}")
    recorder.record(
        TraceEvent(
            kind="interactive_session_resume_failed",
            summary="Interactive session resume failed.",
            refs=tuple(refs),
        )
    )
