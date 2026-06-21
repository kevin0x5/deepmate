"""Bridge between CLI-assembled runtime dependencies and the TUI app."""

from __future__ import annotations

import sys
from dataclasses import replace
from importlib.util import find_spec
from collections.abc import Mapping, Sequence
from pathlib import Path

from deepmate.capabilities import CapabilitySurface
from deepmate.capabilities.state import CapabilityStateStore
from deepmate.channels.checkpointing import (
    SessionCheckpointController,
    SessionCheckpointWriteRouter,
)
from deepmate.channels.interactive import (
    CheckpointControllerFactory,
    ContextSnapshotFactory,
    SessionEndHandler,
    SessionMaintenanceHandler,
    TaskMaintenanceHandler,
    _task_maintenance_result,
    _ensure_session_title,
    _load_latest_summary,
    _close_remote_routes_for_local_turn,
    _record_interactive_trace,
    _response_text,
    _review_delivery,
)
from deepmate.channels.tui.commands import (
    handle_tui_command,
    prepare_and_switch_to_local_model,
)
from deepmate.channels.tui.formatters import TuiMessage, friendly_error_message, result_messages
from deepmate.channels.tui.state import TuiRuntimeState, WorkspaceSwitchRequest
from deepmate.context import ContextWarning
from deepmate.domain import Message, MessageRole, ProfileRef
from deepmate.local.presets import (
    LOCAL_PROVIDER_API_KEY,
    LOCAL_PROVIDER_BASE_URL,
    LOCAL_PROVIDER_NAME,
    local_model_capabilities,
    local_model_by_runtime_name,
)
from deepmate.mcp import McpServerSpec, McpToolExecutor
from deepmate.pet.events import (
    event_for_turn_finished,
    event_for_turn_progress,
    event_for_turn_started,
    event_for_turn_waiting,
)
from deepmate.pet.state import PetStateStore
from deepmate.providers import ChatCompletionsProvider, ModelCapabilities
from deepmate.runtime import (
    ApprovalDecision,
    ConversationBudgetPolicy,
    HookLoadOptions,
    HookRuntimeContext,
    LoopGuardPolicy,
    ProviderRetryPolicy,
    SafetyDecision,
    SessionRuntime,
    ToolAccessDecision,
    ToolAccessPolicy,
    ToolOutputCompactor,
    ToolRepairPolicy,
    TurnFollowupBuffer,
)
from deepmate.runtime.delivery_review import DeliveryReviewStatus
from deepmate.skills import SkillDocument
from deepmate.storage import SessionRecord, SessionStore, TranscriptStore
from deepmate.subagents import SubagentToolExecutor
from deepmate.subagents.store import SubagentResultStore
from deepmate.tasks import TaskSessionController
from deepmate.tools import (
    BROWSER_CLICK_TOOL_NAME,
    BROWSER_CLOSE_TOOL_NAME,
    BROWSER_FILL_TOOL_NAME,
    BROWSER_OPEN_TOOL_NAME,
    BROWSER_SCREENSHOT_TOOL_NAME,
    BROWSER_SNAPSHOT_TOOL_NAME,
    BROWSER_STATUS_TOOL_NAME,
    BROWSER_WAIT_TOOL_NAME,
    COMPUTER_TOOL_NAMES,
    LSP_DEFINITION_TOOL_NAME,
    LSP_HOVER_TOOL_NAME,
    LSP_REFERENCES_TOOL_NAME,
    NativeTool,
    NativeToolRegistry,
    RENDER_TECH_DIAGRAM_TOOL_NAME,
    REVIEW_ARTIFACT_TOOL_NAME,
)
from deepmate.trace import TraceEvent, TraceRecorder


_WORKSPACE_SWITCH_REQUEST: WorkspaceSwitchRequest | None = None


def consume_workspace_switch_request() -> WorkspaceSwitchRequest | None:
    """Return and clear the latest TUI workspace switch request."""
    global _WORKSPACE_SWITCH_REQUEST
    request = _WORKSPACE_SWITCH_REQUEST
    _WORKSPACE_SWITCH_REQUEST = None
    return request


def _safe_mark_turn_result(
    state: "TuiRuntimeState",
    turn_scope,
    result,
    current_turn_index: int,
) -> None:
    if turn_scope is None:
        return
    try:
        turn_scope.mark_result(result)
    except Exception as exc:
        _record_checkpoint_warning(state, exc, current_turn_index, "mark_result")


def _safe_mark_turn_failed(
    state: "TuiRuntimeState",
    turn_scope,
    error_code: str,
    current_turn_index: int,
) -> None:
    if turn_scope is None:
        return
    try:
        turn_scope.mark_failed(error_code)
    except Exception as exc:
        _record_checkpoint_warning(state, exc, current_turn_index, "mark_failed")


def _safe_mark_turn_interrupted(
    state: "TuiRuntimeState",
    turn_scope,
    current_turn_index: int,
) -> None:
    if turn_scope is None:
        return
    try:
        turn_scope.mark_interrupted()
    except Exception as exc:
        _record_checkpoint_warning(state, exc, current_turn_index, "mark_interrupted")


def _safe_attach_turn_summary(
    state: "TuiRuntimeState",
    turn_scope,
    current_turn_index: int,
) -> None:
    if turn_scope is None:
        return
    try:
        turn_scope.attach_summary(_load_latest_summary(state.session_store, state.session))
    except Exception as exc:
        _record_checkpoint_warning(state, exc, current_turn_index, "attach_summary")


def _record_checkpoint_warning(
    state: "TuiRuntimeState",
    exc: BaseException,
    current_turn_index: int,
    operation: str,
) -> None:
    state.trace_recorder.record(
        TraceEvent(
            kind="tui_checkpoint_warning",
            summary=f"TUI progress bookkeeping failed during {operation}: {exc}",
            refs=(
                f"session_id={state.session.session_id}",
                f"turn_index={current_turn_index}",
                f"operation={operation}",
                f"error_type={type(exc).__name__}",
                *state.runtime.activation.trace_refs(),
            ),
        )
    )


def _touch_current_session(state: "TuiRuntimeState", current_turn_index: int) -> None:
    metadata_existed = state.session_store.metadata_path(state.session.session_id).exists()
    state.session = state.session_store.touch_record(state.session)
    if metadata_existed:
        return
    state.trace_recorder.record(
        TraceEvent(
            kind="tui_session_metadata_repaired",
            summary="TUI restored missing session metadata from active state.",
            refs=(
                f"session_id={state.session.session_id}",
                f"turn_index={current_turn_index}",
                *state.runtime.activation.trace_refs(),
            ),
        )
    )


def run_tui_mode(
    provider: ChatCompletionsProvider,
    provider_name: str,
    provider_api_key_env: str,
    provider_api_key_available: bool,
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
    tool_access_policy: ToolAccessPolicy | None,
    tool_schemas: Sequence[Mapping[str, object]],
    selected_skill_documents: Sequence[SkillDocument],
    mcp_servers: Sequence[McpServerSpec],
    conversation_budget_policy: ConversationBudgetPolicy | None,
    provider_retry_policy: ProviderRetryPolicy | None,
    options: Mapping[str, object],
    max_steps: int,
    trace_recorder: TraceRecorder,
    warning_sink,
    model_capabilities: ModelCapabilities | None = None,
    native_tool_factory=None,
    loop_guard_policy: LoopGuardPolicy | None = None,
    status_sink=None,
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
    approval_cache=None,
    checkpoint_controller: SessionCheckpointController | None = None,
    checkpoint_controller_factory: CheckpointControllerFactory | None = None,
    checkpoint_write_router: SessionCheckpointWriteRouter | None = None,
    pet_state_store: PetStateStore | None = None,
    refresh_skill_surface_callback=None,
    local_context_prepare_callback=None,
    behavior_runtime=None,
    initial_prompts: Sequence[str] = (),
    show_reasoning: bool = False,
    default_model: str = "",
    upgrade_model: str = "",
    remote_provider: ChatCompletionsProvider | None = None,
    remote_provider_name: str = "",
    remote_model: str = "",
    remote_default_model: str = "",
    remote_upgrade_model: str = "",
    local_provider_base_url: str = LOCAL_PROVIDER_BASE_URL,
    local_provider_api_key: str = LOCAL_PROVIDER_API_KEY,
) -> int:
    """Run Textual TUI mode, falling back to a clear install hint when unavailable."""
    state = TuiRuntimeState(
        provider=provider,
        provider_name=provider_name,
        provider_api_key_env=provider_api_key_env,
        provider_api_key_available=provider_api_key_available,
        model=model,
        default_model=default_model or model,
        upgrade_model=upgrade_model,
        workspace=workspace,
        profile=profile,
        session_store=session_store,
        session=session,
        transcript=transcript,
        runtime=runtime,
        capability_surface=capability_surface,
        native_tools=native_tools,
        native_tool_factory=native_tool_factory,
        mcp_tools=mcp_tools,
        subagents=subagents,
        tool_access_policy=tool_access_policy,
        tool_schemas=tuple(tool_schemas),
        selected_skill_documents=tuple(selected_skill_documents),
        mcp_servers=tuple(mcp_servers),
        conversation_budget_policy=conversation_budget_policy,
        provider_retry_policy=provider_retry_policy,
        options=dict(options),
        max_steps=max_steps,
        trace_recorder=trace_recorder,
        warning_sink=warning_sink,
        model_capabilities=model_capabilities
        or (
            local_model_capabilities(model)
            if local_model_by_runtime_name(model) is not None
            else ModelCapabilities()
        ),
        remote_provider=remote_provider or provider,
        remote_provider_name=remote_provider_name or provider_name,
        remote_model=remote_model or model,
        remote_default_model=remote_default_model or default_model or model,
        remote_upgrade_model=remote_upgrade_model or upgrade_model,
        remote_provider_api_key_env=provider_api_key_env,
        remote_provider_api_key_available=provider_api_key_available,
        remote_options=dict(options),
        remote_model_capabilities=(
            model_capabilities
            or (
                local_model_capabilities(model)
                if local_model_by_runtime_name(model) is not None
                else ModelCapabilities()
            )
        ),
        remote_conversation_budget_policy=conversation_budget_policy,
        local_provider=ChatCompletionsProvider(
            base_url=local_provider_base_url,
            api_key=local_provider_api_key,
        ),
        local_provider_name=LOCAL_PROVIDER_NAME,
        local_provider_base_url=local_provider_base_url,
        local_provider_api_key=local_provider_api_key,
        local_default_model="qwen3-local",
        local_upgrade_model="qwen3-coder-strong",
        loop_guard_policy=loop_guard_policy,
        status_sink=status_sink,
        tool_output_compactor=tool_output_compactor,
        tool_repair_policy=tool_repair_policy,
        hook_context=hook_context,
        hook_load_options=hook_load_options,
        data_dir=data_dir,
        maintenance_handler=maintenance_handler,
        session_end_handler=session_end_handler,
        context_snapshot_factory=context_snapshot_factory,
        task_controller=task_controller,
        task_maintenance_handler=task_maintenance_handler,
        capability_state_store=capability_state_store,
        approval_cache=approval_cache,
        checkpoint_controller=checkpoint_controller,
        checkpoint_controller_factory=checkpoint_controller_factory,
        checkpoint_write_router=checkpoint_write_router,
        pet_state_store=pet_state_store,
        refresh_skill_surface_callback=refresh_skill_surface_callback,
        local_context_prepare_callback=local_context_prepare_callback,
        behavior_runtime=behavior_runtime,
        show_reasoning=show_reasoning,
        followup_buffer=TurnFollowupBuffer(),
    )
    if state.checkpoint_controller is None and checkpoint_controller_factory is not None:
        state.checkpoint_controller = checkpoint_controller_factory(session)
    if state.checkpoint_write_router is not None:
        state.checkpoint_write_router.set_controller(state.checkpoint_controller)
    if state.task_controller is not None and state.checkpoint_write_router is not None:
        state.task_controller.store.set_write_checkpoint(
            state.checkpoint_write_router.capture_workspace_write
        )
    _install_approval_callbacks(state)
    if find_spec("textual") is None:
        print(
            "error: Textual is required for `deepmate --interactive` TUI mode. "
            "Install Deepmate project dependencies or use `--interactive-legacy`.",
            file=sys.stderr,
        )
        return 2
    from deepmate.channels.tui.app import DeepmateTuiApp, WORKSPACE_SWITCH_EXIT_CODE

    app = DeepmateTuiApp(state, initial_prompts=tuple(initial_prompts))
    app.run()
    if app.exit_code == WORKSPACE_SWITCH_EXIT_CODE:
        global _WORKSPACE_SWITCH_REQUEST
        state.workspace_switch_request = app.state.workspace_switch_request
        _WORKSPACE_SWITCH_REQUEST = state.workspace_switch_request
    return app.exit_code


def run_headless_tui_turn(
    state: TuiRuntimeState,
    prompt: str,
) -> tuple[TuiRuntimeState, tuple[TuiMessage, ...], bool]:
    """Run one TUI turn without Textual. Used by tests and the app worker."""
    _install_approval_callbacks(state)
    clean_prompt = prompt.strip()
    if not clean_prompt:
        return state, (), False
    command = handle_tui_command(clean_prompt, state)
    if command.handled:
        if command.local_prepare is not None:
            command = prepare_and_switch_to_local_model(state, command.local_prepare)
        return state, command.messages, command.exit_requested
    if state.followup_buffer is None:
        state.followup_buffer = TurnFollowupBuffer()
    if state.active_followup_turn_id is None:
        state.active_followup_turn_id = state.followup_buffer.start_turn()
    state.unconsumed_followups = ()
    state.task_continuations = ()

    task_turn = None
    if state.task_controller is not None:
        task_turn = state.task_controller.prepare_prompt(clean_prompt)
        if task_turn is not None:
            is_control = getattr(task_turn, "is_control", lambda: False)
            if is_control():
                try:
                    body = state.task_controller.handle_control(task_turn.control)
                except Exception as exc:
                    return state, (friendly_error_message(exc),), False
                return (
                    state,
                    (TuiMessage(kind="task", title="task", body=body),),
                    False,
                )
            clean_prompt = task_turn.prompt
            if not clean_prompt.strip():
                raise ValueError("task mode produced empty prompt")
            state.task_controller.save_cursor(session_id=state.session.session_id)
            if state.context_snapshot_factory is not None:
                state.runtime = state.runtime.with_refreshed_context(
                    state.context_snapshot_factory(state.session.profile)
                )
                _record_interactive_trace(
                    state.trace_recorder,
                    "task_mode_context_refreshed",
                    "Task Mode context updated before TUI task turn.",
                    session=state.session,
                    runtime=state.runtime,
                    turn_index=state.turn_index,
                )

    current_turn_index = state.turn_index + 1
    if state.approval_cache is not None:
        # Turn-scoped "Allow this time" grants must not leak into the next turn.
        state.approval_cache.reset_turn()
    turn_scope = (
        state.checkpoint_controller.start_turn(
            _load_latest_summary(state.session_store, state.session)
        )
        if state.checkpoint_controller is not None
        else None
    )
    messages: list[TuiMessage] = [
        TuiMessage(kind="user", title="you", body=clean_prompt)
    ]
    try:
        _touch_current_session(state, current_turn_index)
        _close_remote_routes_for_local_turn(
            session_store=state.session_store,
            session=state.session,
            data_dir=state.data_dir,
            trace_recorder=state.trace_recorder,
            runtime=state.runtime,
        )
        state.session = _ensure_session_title(
            state.session_store,
            state.session,
            clean_prompt,
        )
        _safe_save_pet_event(
            state.pet_state_store,
            event_for_turn_started(
                workspace=state.workspace,
                session_id=state.session.session_id,
                prompt=clean_prompt,
                title=state.session.title,
            ),
        )
        _record_interactive_trace(
            state.trace_recorder,
            "tui_turn_started",
            "TUI user turn started.",
            session=state.session,
            runtime=state.runtime,
            turn_index=current_turn_index,
        )
        current_tool_schemas = _schemas_for_prompt(state, clean_prompt)
        turn = state.runtime.run_user_turn(
            provider=state.provider,
            messages=(Message(role=MessageRole.USER, content=clean_prompt),),
            model=state.model,
            capability_surface=state.capability_surface,
            native_tools=state.native_tools,
            mcp_tools=state.mcp_tools,
            subagents=_subagents_for_state(state, current_tool_schemas),
            tool_access_policy=state.tool_access_policy,
            tool_schemas=current_tool_schemas,
            selected_skill_documents=state.selected_skill_documents,
            conversation_budget_policy=state.conversation_budget_policy,
            provider_retry_policy=state.provider_retry_policy,
            options=state.options,
            model_capabilities=state.model_capabilities,
            max_steps=state.max_steps,
            loop_guard_policy=state.loop_guard_policy,
            trace_recorder=state.trace_recorder,
            warning_sink=state.warning_sink,
            history_sink=(
                turn_scope.history_sink(state.transcript)
                if turn_scope is not None
                else state.transcript.append_item
            ),
            status_sink=_tui_status_sink(state),
            token_sink=_tui_token_sink(state),
            tool_output_compactor=state.tool_output_compactor,
            tool_repair_policy=state.tool_repair_policy,
            hook_context=state.hook_context,
            followup_buffer=state.followup_buffer,
            followup_turn_id=state.active_followup_turn_id,
            cancellation_token=state.cancellation_token,
        )
        state.runtime = turn.runtime
        result = turn.result
        _safe_mark_turn_result(state, turn_scope, result, current_turn_index)
        review = _review_delivery(
            user_request=clean_prompt,
            final_response=_response_text(result.final_step().response),
            result=result,
            trace_recorder=state.trace_recorder,
            session=state.session,
            runtime=state.runtime,
        )
        final_text = _response_text(result.final_step().response)
        failed = (
            result.has_errors()
            or result.reached_max_steps
            or review.status == DeliveryReviewStatus.BLOCKED
        )
        _safe_save_pet_event(
            state.pet_state_store,
            event_for_turn_finished(
                workspace=state.workspace,
                session_id=state.session.session_id,
                title=state.session.title,
                summary=_pet_turn_summary(result, final_text, review.summary),
                failed=failed,
            ),
        )
        if (
            review.status != DeliveryReviewStatus.ACCEPTED
            and not result.has_errors()
            and not result.reached_max_steps
        ):
            messages.append(
                TuiMessage(
                    kind="warning",
                    title="delivery review",
                    body=review.summary,
                    refs=tuple(review.issues),
                )
            )
        messages.extend(result_messages(result, show_reasoning=state.show_reasoning))
        if state.final_message_callback is not None:
            state.final_message_callback(tuple(messages))
        if _result_installed_skill(result):
            _refresh_skill_surface(state)
        _touch_current_session(state, current_turn_index)
        state.turn_index = current_turn_index
        _record_interactive_trace(
            state.trace_recorder,
            "tui_turn_finished",
            "TUI user turn finished.",
            session=state.session,
            runtime=state.runtime,
            turn_index=current_turn_index,
        )
        if (
            state.task_maintenance_handler is not None
            and task_turn is not None
            and not result.has_errors()
            and not result.reached_max_steps
        ):
            task_maintenance = state.task_maintenance_handler(
                clean_prompt,
                _response_text(result.final_step().response),
                state.session,
                state.runtime,
            )
            state.runtime, continuation, task_status = _task_maintenance_result(
                task_maintenance
            )
            if continuation:
                state.task_continuations = (
                    continuation,
                    *state.task_continuations,
                )
            if state.context_snapshot_factory is not None:
                state.runtime = state.runtime.request_context_refresh_before_next_turn(
                    "task_mode_updated"
                )
            messages.append(
                TuiMessage(
                    kind="task",
                    title="task updated",
                    body=task_status or _task_updated_body(state),
                )
            )
        if (
            state.maintenance_handler is not None
            and not result.has_errors()
            and not result.reached_max_steps
        ):
            state.runtime = state.maintenance_handler(
                clean_prompt,
                state.session,
                state.transcript,
                state.runtime,
            )
            _safe_attach_turn_summary(state, turn_scope, current_turn_index)
    except KeyboardInterrupt:
        _safe_mark_turn_interrupted(state, turn_scope, current_turn_index)
        raise
    except Exception as exc:
        _safe_mark_turn_failed(state, turn_scope, type(exc).__name__, current_turn_index)
        _safe_save_pet_event(
            state.pet_state_store,
            event_for_turn_finished(
                workspace=state.workspace,
                session_id=state.session.session_id,
                title=state.session.title,
                summary=f"{type(exc).__name__}: {exc}",
                failed=True,
            ),
        )
        state.trace_recorder.record(
            TraceEvent(
                kind="tui_turn_failed",
                summary=f"TUI user turn failed: {exc}",
                refs=(
                    f"session_id={state.session.session_id}",
                    f"turn_index={current_turn_index}",
                    *state.runtime.activation.trace_refs(),
                ),
            )
        )
        messages.append(friendly_error_message(exc))
    finally:
        if state.followup_buffer is not None:
            remaining = state.followup_buffer.finish_turn(state.active_followup_turn_id)
            state.unconsumed_followups = tuple(
                followup.text for followup in remaining if followup.text.strip()
            )
        state.active_followup_turn_id = None
        if turn_scope is not None:
            turn_scope.close()
    return state, tuple(messages), False


def _schemas_for_prompt(
    state: TuiRuntimeState,
    prompt: str,
) -> tuple[Mapping[str, object], ...]:
    schemas = list(state.tool_schemas)
    registry = state.native_tools
    if registry is None:
        return tuple(schemas)
    existing = {
        str(schema.get("name", "")).strip()
        for schema in schemas
        if isinstance(schema, Mapping)
    }
    extra_names = list(_extra_schema_names_for_prompt(prompt))
    if (
        state.behavior_runtime is not None
        and state.behavior_runtime.computer_use_enabled
    ):
        extra_names.extend(COMPUTER_TOOL_NAMES)
    if local_model_by_runtime_name(state.model) is not None:
        extra_names.extend(_extra_schema_names_for_local_prompt(prompt))
    for name in tuple(dict.fromkeys(extra_names)):
        tool = registry.get(name)
        if tool is None or name in existing:
            continue
        schemas.append(tool.schema())
        existing.add(name)
    return tuple(schemas)


def _subagents_for_state(
    state: TuiRuntimeState,
    tool_schemas: Sequence[Mapping[str, object]],
) -> SubagentToolExecutor | None:
    if state.subagents is None:
        return None
    result_store = (
        SubagentResultStore.in_data_dir(state.data_dir, state.session.session_id)
        if state.data_dir is not None
        else None
    )
    return state.subagents.bind_runtime(
        capability_surface=state.capability_surface,
        native_tools=state.native_tools,
        mcp_tools=state.mcp_tools,
        tool_schemas=tool_schemas,
        selected_skill_documents=tuple(state.selected_skill_documents),
        activation=state.runtime.activation,
        parent_tool_access_policy=state.tool_access_policy,
        result_store=result_store,
    )


def _result_installed_skill(result) -> bool:
    # install_skill_from_request is the installer the model sees by default in
    # interactive mode, so it must be included or a freshly installed skill stays
    # invisible until the next session restart.
    installer_names = {
        "install_skill",
        "install_skill_bundle",
        "install_skill_from_request",
    }
    for step in result.steps:
        for tool_result in step.tool_results:
            if (
                tool_result.name in installer_names
                and not tool_result.is_error
                and any(ref.startswith("skill=") for ref in tool_result.refs)
            ):
                return True
    return False


def _refresh_skill_surface(state: TuiRuntimeState) -> None:
    callback = state.refresh_skill_surface_callback
    if callback is not None:
        callback(state)


def _extra_schema_names_for_prompt(prompt: str) -> tuple[str, ...]:
    clean = prompt.lower()
    names: list[str] = []
    if _looks_like_shell_prompt(clean):
        names.append("run_shell_command")
    if _looks_like_browser_install_prompt(clean):
        names.append("install_browser_backend")
    return tuple(dict.fromkeys(names))


def _extra_schema_names_for_local_prompt(prompt: str) -> tuple[str, ...]:
    clean = prompt.lower()
    names: list[str] = []
    if _looks_like_lsp_prompt(clean):
        names.extend(
            (
                LSP_DEFINITION_TOOL_NAME,
                LSP_REFERENCES_TOOL_NAME,
                LSP_HOVER_TOOL_NAME,
            )
        )
    if _looks_like_document_prompt(clean):
        names.extend(("read_document", "inspect_table"))
    if _looks_like_artifact_review_prompt(clean):
        names.append(REVIEW_ARTIFACT_TOOL_NAME)
    if _looks_like_report_prompt(clean):
        names.append("render_html_report")
    if _looks_like_diagram_prompt(clean):
        names.append(RENDER_TECH_DIAGRAM_TOOL_NAME)
    if _looks_like_browser_prompt(clean):
        names.extend(
            (
                BROWSER_OPEN_TOOL_NAME,
                BROWSER_SNAPSHOT_TOOL_NAME,
                BROWSER_CLICK_TOOL_NAME,
                BROWSER_FILL_TOOL_NAME,
                BROWSER_WAIT_TOOL_NAME,
                BROWSER_SCREENSHOT_TOOL_NAME,
                BROWSER_CLOSE_TOOL_NAME,
                BROWSER_STATUS_TOOL_NAME,
            )
        )
    return tuple(dict.fromkeys(names))


def _looks_like_shell_prompt(clean: str) -> bool:
    return any(
        marker in clean
        for marker in (
            "运行命令",
            "执行命令",
            "跑一下",
            "测试",
            "验证",
            "构建",
            "打包",
            "启动",
            "安装依赖",
            "安装 cli",
            "安装cli",
            "修复",
            "命令行",
            "终端",
            "curl ",
            " bash",
            "授权",
            "允许你",
            "直接处理",
            "继续处理",
            "可以执行",
            "run command",
            "execute command",
            "shell",
            "terminal",
            "command line",
            "proceed",
            "approved",
            "pytest",
            "unittest",
            "lint",
            "build",
            "install dependencies",
            "fix",
            "verify",
            "npm ",
            "pip ",
            "pnpm ",
            "yarn ",
        )
    )


def _looks_like_browser_install_prompt(clean: str) -> bool:
    return "agent-browser" in clean or (
        any(marker in clean for marker in ("浏览器", "browser"))
        and any(marker in clean for marker in ("安装", "install", "不可用", "unavailable"))
    )


def _looks_like_lsp_prompt(clean: str) -> bool:
    return any(
        marker in clean
        for marker in (
            "定义",
            "引用",
            "调用链",
            "谁调用",
            "类型",
            "签名",
            "go to definition",
            "find references",
            "references",
            "definition",
            "hover",
        )
    )


def _looks_like_document_prompt(clean: str) -> bool:
    return any(
        marker in clean
        for marker in (
            ".docx",
            ".xlsx",
            ".pdf",
            ".csv",
            "文档",
            "表格",
            "excel",
            "spreadsheet",
            "document",
        )
    )


def _looks_like_artifact_review_prompt(clean: str) -> bool:
    return any(marker in clean for marker in ("review_artifact", "检查交付", "验收", "qa"))


def _looks_like_report_prompt(clean: str) -> bool:
    return any(
        marker in clean
        for marker in (
            "html report",
            "render_html_report",
            "报告",
            "可视化报告",
            "生成 html",
            "生成html",
        )
    )


def _looks_like_diagram_prompt(clean: str) -> bool:
    return any(
        marker in clean
        for marker in (
            "diagram",
            "render_tech_diagram",
            "架构图",
            "流程图",
            "时序图",
            "图表",
        )
    )


def _looks_like_browser_prompt(clean: str) -> bool:
    if _looks_like_browser_install_prompt(clean):
        return False
    return any(
        marker in clean
        for marker in (
            "浏览器",
            "网页",
            "页面",
            "截图",
            "点击",
            "browser",
            "web page",
            "screenshot",
            "click",
        )
    )


def end_tui_session(state: TuiRuntimeState, reason: str) -> None:
    """Run the same session-end hook used by legacy interactive mode."""
    if state.session_end_handler is not None and state.turn_index > 0:
        state.session_end_handler(state.session, state.transcript, state.runtime, reason)


def _install_approval_callbacks(state: TuiRuntimeState) -> None:
    if state.approval_callbacks_installed:
        return
    if state.tool_access_policy is not None:
        state.tool_access_policy = replace(
            state.tool_access_policy,
            defer_shell_approval_to_tool=True,
            approval_callback=lambda tool, decision: _approve_tool_access(
                state,
                tool,
                decision,
            ),
        )
    if state.approval_cache is not None:
        state.approval_cache.approval_callback = lambda decision: _approve_safety(
            state,
            decision,
        )
    state.approval_callbacks_installed = True


def _approve_tool_access(
    state: TuiRuntimeState,
    tool: NativeTool,
    decision: ToolAccessDecision,
) -> bool:
    _safe_save_pet_event(
        state.pet_state_store,
        event_for_turn_waiting(
            workspace=state.workspace,
            session_id=state.session.session_id,
            title=state.session.title,
            summary=decision.reason or f"Waiting for approval: {tool.name}",
        ),
    )
    if state.tool_approval_callback is None:
        return False
    return state.tool_approval_callback(tool, decision)


def _approve_safety(
    state: TuiRuntimeState,
    decision: SafetyDecision,
) -> ApprovalDecision:
    _safe_save_pet_event(
        state.pet_state_store,
        event_for_turn_waiting(
            workspace=state.workspace,
            session_id=state.session.session_id,
            title=state.session.title,
            summary=decision.reason or "Waiting for safety approval.",
        ),
    )
    if state.safety_approval_callback is None:
        return ApprovalDecision.DENY
    return state.safety_approval_callback(decision)


def _tui_status_sink(state: TuiRuntimeState):
    def emit(message: str) -> None:
        if (
            state.status_sink is not None
            and state.status_message_callback is None
            and state.live_status_callback is None
        ):
            state.status_sink(message)
        view = state.runtime_stats.record(message)
        _maybe_emit_pet_progress(state, view.title, view.body, view.status)
        tui_message = TuiMessage(
            kind="status",
            title=view.title,
            body=view.body,
            status=view.status,
        )
        if (
            state.live_status_callback is not None
            and view.important
            and _live_status_view(view.status)
        ):
            state.live_status_callback(tui_message)
            return
        if state.status_message_callback is not None and _main_status_view(view.status):
            state.status_message_callback(tui_message)

    return emit


def _tui_token_sink(state: TuiRuntimeState):
    """Build a token sink that forwards streamed deltas to the TUI callback.

    Returns None when no streaming callback is registered, so the turn falls
    back to a single blocking completion (headless runs, tests). The sink
    accepts the provider's StreamDelta and passes plain strings onward, keeping
    the app/state layers free of provider types.
    """
    if state.token_stream_callback is None:
        return None

    def emit(delta) -> None:
        callback = state.token_stream_callback
        if callback is None:
            return
        content = getattr(delta, "content", "") or ""
        reasoning = getattr(delta, "reasoning", "") or ""
        if content or reasoning:
            callback(content, reasoning)

    return emit


def _live_status_view(status: str) -> bool:
    """Return whether a runtime status should update the live TUI work cell."""
    return status in {
        "model",
        "tool",
        "compacted",
        "normalized",
        "schema",
        "warning",
    }


def _main_status_view(status: str) -> bool:
    """Return whether a runtime status deserves a standalone main message."""
    return status == "warning"


def _maybe_emit_pet_progress(
    state: TuiRuntimeState,
    title: str,
    body: str,
    status: str,
) -> None:
    if state.pet_state_store is None:
        return
    if status not in {"warning", "compacted", "normalized", "schema"}:
        return
    key = f"{status}:{title}:{body[:120]}"
    if key == state.pet_last_progress_key:
        return
    state.pet_last_progress_key = key
    _safe_save_pet_event(
        state.pet_state_store,
        event_for_turn_progress(
            workspace=state.workspace,
            session_id=state.session.session_id,
            title=state.session.title,
            summary=f"{title}: {body}",
        ),
    )


def _task_updated_body(state: TuiRuntimeState) -> str:
    stage = state.task_stage_label() or "Task Mode updated."
    lines = [stage]
    controller = state.task_controller
    if controller is not None and controller.active_stage is not None:
        context = controller.context()
        if context is not None:
            lines.append(f"task context: {context.estimated_tokens} estimated tokens")
            if context.rolling_summary:
                lines.append("")
                lines.append("Rolling summary")
                lines.extend(_first_bullets(context.rolling_summary, limit=5))
            if context.recent_timeline:
                lines.append("")
                lines.append("Recent timeline")
                lines.extend(_timeline_head(context.recent_timeline, limit=4))
    return "\n".join(lines)


def _first_bullets(text: str, *, limit: int) -> list[str]:
    bullets: list[str] = []
    for line in text.splitlines():
        clean = line.strip()
        if clean.startswith("- "):
            bullets.append(clean)
        elif clean:
            bullets.append(f"- {clean}")
        if len(bullets) >= limit:
            break
    return bullets


def _timeline_head(text: str, *, limit: int) -> list[str]:
    lines = []
    for line in text.splitlines():
        clean = line.strip()
        if clean:
            lines.append(clean)
        if len(lines) >= limit:
            break
    return lines


def _safe_save_pet_event(store: PetStateStore | None, event) -> None:
    if store is None:
        return
    try:
        store.save_current_state(event)
    except Exception:
        pass


def _pet_turn_summary(result, final_text: str, review_summary: str) -> str:
    if final_text.strip():
        return final_text
    errors = result.errors()
    if errors:
        return errors[0].message
    if result.reached_max_steps:
        return "Reached max steps before final answer."
    return review_summary or "Turn finished."


def approval_decision_from_text(value: str) -> ApprovalDecision:
    """Parse a TUI approval action label."""
    clean = value.strip().lower().replace("-", "_").replace(" ", "_")
    if clean in {"allow", "allow_once", "once"}:
        return ApprovalDecision.ALLOW_ONCE
    if clean in {"allow_for_session", "session"}:
        return ApprovalDecision.ALLOW_FOR_SESSION
    return ApprovalDecision.DENY
