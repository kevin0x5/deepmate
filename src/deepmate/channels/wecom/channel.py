"""WeCom remote channel dispatch and runtime bridge."""

from __future__ import annotations

import json
import threading
import time
from contextlib import nullcontext
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from deepmate.app import AppSettings
from deepmate.capabilities import CapabilitySurface
from deepmate.channels.checkpointing import SessionCheckpointController
from deepmate.channels.remote import RemoteBindingRecord, RemoteBindingStore
from deepmate.channels.session_maintenance import runtime_conversation_from_store
from deepmate.channels.wecom.client import (
    WeComClientConfig,
    WeComPayloadError,
    WeComProtocolError,
    WeComWsClient,
)
from deepmate.context import ContextWarning, build_profile_context_snapshot
from deepmate.domain import Message, MessageRole, ProfileRef
from deepmate.mcp import McpToolExecutor
from deepmate.preview_deploy import handle_deploy_command
from deepmate.providers import ChatCompletionsProvider, ModelCapabilities, ModelResponse
from deepmate.runtime import (
    ConversationBudgetPolicy,
    DEFAULT_HARD_STEP_CAP,
    LoopGuardPolicy,
    ProviderRetryPolicy,
    SessionRuntime,
    ToolAccessMode,
    ToolAccessDecision,
    ToolAccessPolicy,
    ToolOutputCompactor,
    ToolRepairPolicy,
    TurnFollowupBuffer,
    start_runtime_activation,
    start_session_runtime,
)
from deepmate.runtime.hooks import HookRuntimeContext
from deepmate.runtime.safety import ApprovalDecision, SafetyDecision, SessionApprovalCache
from deepmate.runtime.wakelock import RuntimeWakeSession, WakeConfig
from deepmate.skills import SkillDocument
from deepmate.storage import SessionRecord, SessionStore, TranscriptStore
from deepmate.subagents import SubagentToolExecutor
from deepmate.subagents.store import SubagentResultStore
from deepmate.tools import NativeTool, NativeToolRegistry
from deepmate.trace import TraceEvent, TraceRecorder

if TYPE_CHECKING:
    from deepmate.behavior import BehaviorRuntime


CHANNEL = "wecom"
WEBCOM_SEND_SLOT_TIMEOUT_SECONDS = 120.0
WEBCOM_SUBSCRIBE_ACK_TIMEOUT_SECONDS = 5.0
REMOTE_REPLY_LIMIT_BYTES = 1800
REMOTE_STREAM_LIMIT_BYTES = 20000
REMOTE_TAIL_LIMIT_BYTES = 4000
REMOTE_STATUS_LIMIT = 800
REMOTE_BINDING_REFRESH_INTERVAL_SECONDS = 15.0
WEBCOM_RECONNECT_INITIAL_DELAY_SECONDS = 1.0
WEBCOM_RECONNECT_MAX_DELAY_SECONDS = 30.0
WEBCOM_PING_INTERVAL_SECONDS = 30.0
WEBCOM_PASSIVE_REPLY_WINDOW_SECONDS = 5.0
WEBCOM_RECOVERABLE_TURN_ERRORS = (
    OSError,
    RuntimeError,
    TypeError,
    UnicodeError,
    ValueError,
    json.JSONDecodeError,
    WeComPayloadError,
    WeComProtocolError,
)
WEBCOM_SEND_ERRORS = (
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    WeComPayloadError,
    WeComProtocolError,
)
WEBCOM_PING_ERRORS = (
    OSError,
    RuntimeError,
    WeComPayloadError,
    WeComProtocolError,
)


@dataclass(frozen=True, slots=True)
class WeComInboundMessage:
    """Normalized inbound WeCom message."""

    req_id: str
    user_id: str
    chat_id: str
    chat_type: str
    content: str


@dataclass(frozen=True, slots=True)
class WeComRunDependencies:
    """Runtime dependencies used to execute WeCom turns."""

    settings: AppSettings
    provider: ChatCompletionsProvider
    model: str
    session_store: SessionStore
    trace_recorder: TraceRecorder
    provider_name: str = ""
    capability_surface: CapabilitySurface | None = None
    native_tools: NativeToolRegistry | None = None
    mcp_tools: McpToolExecutor | None = None
    subagents: SubagentToolExecutor | None = None
    tool_access_policy: ToolAccessPolicy | None = None
    tool_schemas: Sequence[Mapping[str, object]] = ()
    selected_skill_documents: Sequence[SkillDocument] = ()
    conversation_budget_policy: ConversationBudgetPolicy | None = None
    provider_retry_policy: ProviderRetryPolicy | None = None
    options: Mapping[str, object] | None = None
    model_capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    max_steps: int = DEFAULT_HARD_STEP_CAP
    loop_guard_policy: LoopGuardPolicy | None = None
    warning_sink: Callable[[ContextWarning], None] | None = None
    status_sink: Callable[[str], None] | None = None
    tool_output_compactor: ToolOutputCompactor | None = None
    tool_repair_policy: ToolRepairPolicy | None = None
    hook_context: HookRuntimeContext | None = None
    approval_cache: SessionApprovalCache | None = None
    behavior_runtime_factory: Callable[[SessionRecord], "BehaviorRuntime | None"] | None = None


@dataclass(slots=True)
class _RemoteApprovalRequest:
    tool_name: str
    reason: str
    event: threading.Event
    approved: bool = False
    created_at: float = 0.0


@dataclass(slots=True)
class _SessionState:
    binding: RemoteBindingRecord
    session: SessionRecord
    transcript: TranscriptStore
    runtime: SessionRuntime
    checkpoint_controller: SessionCheckpointController
    followup_buffer: TurnFollowupBuffer
    active_followup_turn_id: str | None = None
    is_processing: bool = False
    last_status: str = ""
    last_reply: str = ""
    tail_log: str = ""
    last_error: str = ""
    pending_approval: _RemoteApprovalRequest | None = None
    progress_started_at: float = 0.0
    last_heartbeat_at: float = 0.0
    heartbeat_index: int = 0
    heartbeat_stop_event: threading.Event | None = None
    heartbeat_thread: threading.Thread | None = None
    active_stream_id: str = ""
    active_stream_sequence: int = 0
    last_binding_refresh_at: float = 0.0
    lock: threading.RLock = field(default_factory=threading.RLock)


class WeComChannel:
    """Bridge WeCom messages to Deepmate sessions."""

    def __init__(
        self,
        deps: WeComRunDependencies,
        *,
        binding_store: RemoteBindingStore | None = None,
        sender: Callable[[dict[str, object]], None] | None = None,
        wake_factory: Callable[[str], RuntimeWakeSession] | None = None,
    ) -> None:
        self.deps = deps
        self.binding_store = binding_store or RemoteBindingStore.in_data_dir(
            deps.settings.data_dir
        )
        self.sender = sender or (lambda payload: None)
        self.wake_factory = wake_factory or self._default_wake_session
        self._states: dict[str, _SessionState] = {}
        self._lock = threading.Lock()
        self._send_rate_lock = threading.Lock()
        self._send_timestamps: list[float] = []

    def handle_payload(self, payload: Mapping[str, object]) -> None:
        if self._handle_protocol_status_payload(payload):
            return
        message = parse_wecom_message(payload)
        if message is None:
            return
        self.handle_message(message)

    def dispatch_payload(self, payload: Mapping[str, object]) -> None:
        """Handle one inbound payload without blocking the receiver on new turns."""
        if self._handle_protocol_status_payload(payload):
            return
        message = parse_wecom_message(payload)
        if message is None:
            return
        self.handle_message(message, async_turn=True)

    def handle_message(
        self,
        message: WeComInboundMessage,
        *,
        async_turn: bool = False,
    ) -> None:
        if not self._allowed_user(message.user_id):
            self._send_text(message, "Deepmate remote is not enabled for this user.")
            return
        denial_reason = self._group_denial_reason(message)
        if denial_reason:
            self._send_text(message, denial_reason)
            return
        clean = _clean_prompt(message.content)
        if not clean:
            self._send_text(message, "收到空消息，未执行。")
            return
        if self._readonly_group_command_denied(message, clean):
            self._send_text(
                message,
                "当前企业微信群策略为 readonly，只允许查询状态或执行只读任务。",
            )
            return
        if async_turn:
            worker = threading.Thread(
                target=self._handle_message_ready,
                args=(message, clean, True),
                name=f"deepmate-wecom-dispatch-{message.req_id}",
                daemon=True,
            )
            worker.start()
            return

        self._handle_message_ready(message, clean, False)

    def _handle_message_ready(
        self,
        message: WeComInboundMessage,
        clean: str,
        async_turn: bool,
    ) -> None:
        try:
            state = self._state_for_message(
                message,
                first_prompt=clean,
                open_route=not _is_remote_command(clean),
            )
        except ValueError as exc:
            self._send_text(message, str(exc))
            return
        if self._handle_remote_command(message, state, clean):
            return
        with state.lock:
            is_processing = state.is_processing
            active_followup_turn_id = state.active_followup_turn_id
        if is_processing:
            submitted = state.followup_buffer.submit(
                active_followup_turn_id,
                clean,
                source="wecom",
            )
            if submitted:
                with state.lock:
                    state.last_status = (
                        f"follow-up queued: {_compact_line(clean, 80)} "
                        f"(pending={state.followup_buffer.pending_count()})"
                    )
                self._record_tail(state, f"follow-up: {_compact_line(clean, 160)}")
                self._send_text(
                    message,
                    "已收到补充要求，会在当前任务的下一步处理。\n"
                    f"待处理补充要求：{state.followup_buffer.pending_count()} 条",
                )
                self._defer_heartbeat(state)
            else:
                self._send_text(message, "当前任务即将结束，这条消息请稍后重发。")
            return
        if async_turn:
            with state.lock:
                state.is_processing = True
                state.active_followup_turn_id = state.followup_buffer.start_turn()
            self._run_turn(message, state, clean)
            return
        self._run_turn(message, state, clean)

    def _state_for_message(
        self,
        message: WeComInboundMessage,
        *,
        first_prompt: str,
        open_route: bool = True,
    ) -> _SessionState:
        key = f"{CHANNEL}:{message.user_id}"
        with self._lock:
            binding = self.binding_store.get(CHANNEL, message.user_id)
            if binding is None:
                binding = self.binding_store.get(CHANNEL, "*")
            state = self._states.get(key)
            if (
                state is not None
                and binding is not None
                and binding.session_id == state.session.session_id
            ):
                state.binding = binding
                if open_route and not state.binding.route_open:
                    state.binding = self.binding_store.upsert(
                        state.binding.refreshed_from_session(state.session).with_route(
                            open=True
                        )
                    )
                return state
            session = None
            if binding is not None:
                try:
                    session = self.deps.session_store.load(binding.session_id)
                except (OSError, ValueError, json.JSONDecodeError):
                    session = None
            if session is None:
                session = self.deps.session_store.create(
                    workspace=self.deps.settings.workspace,
                    profile=self.deps.settings.profile_ref(),
                    title=_remote_session_title(self.deps.settings.workspace, first_prompt),
                )
                binding = self.binding_store.bind_session(
                    channel=CHANNEL,
                    remote_user_id=message.user_id,
                    session=session,
                    bound_from="remote",
                )
                self._send_text(
                    message,
                    "\n".join(
                        (
                            "企业微信接管已开启。",
                            f"当前任务：{session.title}",
                            "",
                            "后续企业微信消息会继续进入当前项目。",
                            "如果想接管电脑上已经打开的 Deepmate 窗口，请在本机窗口里执行 /remote --wecom。",
                        )
                    ),
                )
            elif session.workspace.resolve() != self.deps.settings.workspace.resolve():
                raise ValueError(
                    "当前企业微信接管的是另一个项目。\n"
                    "请在那个项目目录启动 deepmate --remote wecom，或在当前本机窗口重新执行 /remote --wecom。"
                )
            else:
                if binding is not None:
                    refreshed = binding.refreshed_from_session(session)
                    if open_route:
                        refreshed = refreshed.with_route(open=True)
                    binding = self.binding_store.upsert(refreshed)
                else:
                    binding = self.binding_store.bind_session(
                        channel=CHANNEL,
                        remote_user_id=message.user_id,
                        session=session,
                        bound_from="remote",
                    )
                    if open_route and not binding.route_open:
                        binding = self.binding_store.upsert(
                            binding.with_route(open=True)
                        )
            transcript = self.deps.session_store.transcript_store(session)
            checkpoint_controller = SessionCheckpointController.in_data_dir(
                self.deps.settings.data_dir,
                workspace=session.workspace,
                profile=session.profile.name,
                session_id=session.session_id,
            )
            activation = start_runtime_activation(
                session_id=session.session_id,
                workspace=session.workspace,
                profile=session.profile,
                context_snapshot=_profile_context_snapshot(
                    self.deps.settings,
                    session.workspace,
                    session.profile,
                    self.deps.model,
                    provider_name=self.deps.provider_name,
                ),
            )
            runtime = start_session_runtime(
                activation,
                conversation=runtime_conversation_from_store(
                    self.deps.session_store,
                    session,
                    transcript,
                    warning_sink=self.deps.status_sink,
                    turn_checkpoint_store=checkpoint_controller.turn_store,
                ),
                behavior_runtime=(
                    self.deps.behavior_runtime_factory(session)
                    if self.deps.behavior_runtime_factory is not None
                    else None
                ),
            )
            state = _SessionState(
                binding=binding,
                session=session,
                transcript=transcript,
                runtime=runtime,
                checkpoint_controller=checkpoint_controller,
                followup_buffer=TurnFollowupBuffer(),
            )
            self._states[key] = state
            return state

    def _handle_remote_command(
        self,
        message: WeComInboundMessage,
        state: _SessionState,
        clean: str,
    ) -> bool:
        command = clean.strip().lower()
        if command in {"/current", "/status"}:
            self._send_text(message, _current_status(state))
            return True
        if command == "/tail":
            with state.lock:
                tail_log = state.tail_log
                last_reply = state.last_reply
            self._send_text(message, tail_log or last_reply or "暂无远程输出。")
            return True
        if command.startswith("/deploy"):
            self._send_text(message, self._handle_remote_deploy(command, state))
            return True
        if command in {"关闭当前预览链接", "关闭预览链接", "关闭预览"}:
            self._send_text(message, self._handle_remote_deploy("/deploy stop", state))
            return True
        if command == "/pending":
            self._send_text(message, _pending_status(state))
            return True
        if command in {"/approve", "/approve once"}:
            self._resolve_remote_approval(message, state, approved=True)
            return True
        if command == "/deny":
            self._resolve_remote_approval(message, state, approved=False)
            return True
        if command == "/open":
            with state.lock:
                binding = state.binding
            opened_binding = self.binding_store.upsert(binding.with_route(open=True))
            with state.lock:
                state.binding = opened_binding
                state.last_status = "remote route open"
            self._send_text(message, "企业微信接管已开启。")
            return True
        if command == "/close":
            with state.lock:
                binding = state.binding
            closed_binding = self.binding_store.upsert(binding.with_route(open=False))
            with state.lock:
                state.binding = closed_binding
                state.last_status = "remote route closed"
            self._stop_heartbeat(state)
            with state.lock:
                pending = state.pending_approval
            if pending is not None:
                pending.approved = False
                pending.event.set()
                with state.lock:
                    if state.pending_approval is pending:
                        state.pending_approval = None
            self._send_text(message, "已关闭当前企业微信接管。")
            return True
        return False

    def _handle_remote_deploy(self, command: str, state: _SessionState) -> str:
        try:
            return handle_deploy_command(
                command,
                workspace=self.deps.settings.workspace,
                data_dir=self.deps.settings.data_dir,
                owner_session_id=state.session.session_id,
                owner_session_workspace=state.session.workspace,
            )
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            return str(exc)

    def _resolve_remote_approval(
        self,
        message: WeComInboundMessage,
        state: _SessionState,
        *,
        approved: bool,
    ) -> None:
        with state.lock:
            pending = state.pending_approval
        if pending is None:
            self._send_text(message, "当前没有等待远程审批的操作。")
            return
        binding = self._refresh_state_binding(state, force=True)
        if approved and not binding.route_open:
            pending.approved = False
            pending.event.set()
            with state.lock:
                if state.pending_approval is pending:
                    state.pending_approval = None
                state.last_status = f"approval denied because route closed: {pending.tool_name}"
            self._defer_heartbeat(state)
            self._send_text(
                message,
                _remote_route_closed_message(),
            )
            return
        pending.approved = approved
        pending.event.set()
        with state.lock:
            if state.pending_approval is pending:
                state.pending_approval = None
        self._defer_heartbeat(state)
        if approved:
            with state.lock:
                state.last_status = f"approval allowed once: {pending.tool_name}"
            self._send_text(message, "已允许这次操作。")
        else:
            with state.lock:
                state.last_status = f"approval denied: {pending.tool_name}"
            self._send_text(message, "已拒绝这次操作。")

    def _run_turn(
        self,
        message: WeComInboundMessage,
        state: _SessionState,
        prompt: str,
    ) -> None:
        with state.lock:
            if not state.is_processing:
                state.is_processing = True
            if state.active_followup_turn_id is None:
                state.active_followup_turn_id = state.followup_buffer.start_turn()
            active_followup_turn_id = state.active_followup_turn_id
        turn_scope = state.checkpoint_controller.start_turn(
            _load_latest_summary(self.deps.session_store, state.session)
        )
        with state.lock:
            state.last_error = ""
            state.last_status = "running"
            state.progress_started_at = time.monotonic()
            state.active_stream_id = _stream_id(message, state)
            state.active_stream_sequence = 0
        self._record_tail(state, f"user: {_compact_line(prompt, 160)}")
        self._send_stream(message, state, _turn_started_message(state), finish=False)
        self._start_heartbeat(message, state)
        wake_session = self.wake_factory("WeCom remote turn")
        approval_cache = self.deps.approval_cache
        approval_scope = (
            approval_cache.scoped_approval_callback(
                lambda decision: self._approve_safety_decision(
                    message,
                    state,
                    decision,
                )
            )
            if approval_cache is not None
            else nullcontext()
        )
        try:
            with approval_scope:
                wake_session.start()
                tool_access_policy = self._remote_tool_access_policy(message, state)
                turn = state.runtime.run_user_turn(
                    provider=self.deps.provider,
                    messages=(Message(role=MessageRole.USER, content=prompt),),
                    model=self.deps.model,
                    capability_surface=self.deps.capability_surface,
                    native_tools=self.deps.native_tools,
                    mcp_tools=self.deps.mcp_tools,
                    subagents=self._subagents_for_state(state, tool_access_policy),
                    tool_access_policy=tool_access_policy,
                    tool_schemas=tuple(self.deps.tool_schemas),
                    selected_skill_documents=tuple(self.deps.selected_skill_documents),
                    conversation_budget_policy=self.deps.conversation_budget_policy,
                    provider_retry_policy=self.deps.provider_retry_policy,
                    options=dict(self.deps.options or {}),
                    model_capabilities=self.deps.model_capabilities,
                    max_steps=self.deps.max_steps,
                    loop_guard_policy=self.deps.loop_guard_policy,
                    trace_recorder=self.deps.trace_recorder,
                    warning_sink=self.deps.warning_sink,
                    history_sink=turn_scope.history_sink(state.transcript),
                    status_sink=self._status_sink(message, state),
                    tool_output_compactor=self.deps.tool_output_compactor,
                    tool_repair_policy=self.deps.tool_repair_policy,
                    hook_context=self.deps.hook_context,
                    followup_buffer=state.followup_buffer,
                    followup_turn_id=active_followup_turn_id,
                )
            with state.lock:
                state.runtime = turn.runtime
            turn_scope.mark_result(turn.result)
            response = turn.result.final_step().response
            reply = _remote_result_text(turn.result, response)
            response_text = _response_text(response)
            with state.lock:
                state.last_reply = reply
                state.last_status = "completed"
            self._record_tail(state, f"assistant: {_compact_line(response_text, 1000)}")
            self._stop_heartbeat(state)
            if not self._send_stream(message, state, reply, finish=True):
                self._record_remote_send_failure(state, "final reply")
            touched = self.deps.session_store.touch(state.session.session_id)
            with state.lock:
                state.session = touched
        except WEBCOM_RECOVERABLE_TURN_ERRORS as exc:
            turn_scope.mark_failed(type(exc).__name__)
            with state.lock:
                state.last_error = f"{type(exc).__name__}: {exc}"
                state.last_status = "failed"
                state.last_reply = _remote_error_text(state, exc)
            self._stop_heartbeat(state)
            if not self._send_stream(message, state, state.last_reply, finish=True):
                self._record_remote_send_failure(state, "error reply")
            self.deps.trace_recorder.record(
                TraceEvent(
                    kind="wecom_turn_failed",
                    summary=f"WeCom remote turn failed: {exc}",
                    refs=(
                        f"session_id={state.session.session_id}",
                        f"user_id={message.user_id}",
                    ),
                )
            )
        finally:
            self._stop_heartbeat(state)
            wake_session.finish_turn()
            turn_scope.close()
            with state.lock:
                finishing_followup_turn_id = state.active_followup_turn_id
            leftovers = state.followup_buffer.finish_turn(finishing_followup_turn_id)
            if leftovers:
                with state.lock:
                    state.last_status = (
                        f"completed with {len(leftovers)} unconsumed follow-up(s)"
                    )
            with state.lock:
                state.active_followup_turn_id = None
                state.is_processing = False
                state.pending_approval = None
                state.active_stream_id = ""
                state.active_stream_sequence = 0

    def _remote_tool_access_policy(
        self,
        message: WeComInboundMessage,
        state: _SessionState,
    ) -> ToolAccessPolicy | None:
        policy = self.deps.tool_access_policy
        if policy is None:
            policy = ToolAccessPolicy()
        if self._is_readonly_group_message(message):
            policy = replace(
                policy,
                mode=ToolAccessMode.READ_ONLY,
                shell_enabled=False,
                defer_shell_approval_to_tool=False,
            )
        return replace(
            policy,
            approval_callback=lambda tool, decision: self._approve_tool(
                message,
                state,
                tool,
                decision,
            ),
        )

    def _approve_tool(
        self,
        message: WeComInboundMessage,
        state: _SessionState,
        tool: NativeTool,
        decision: ToolAccessDecision,
    ) -> bool:
        if self._is_readonly_group_message(message) and not tool.read_only:
            self._send_text(
                message,
                "远程审批已拒绝：当前企业微信群策略为 readonly，不能执行写入或 shell 工具。",
            )
            return False
        if not self._refresh_state_binding(state, force=True).route_open:
            self._send_text(
                message,
                _remote_route_closed_message(),
            )
            return False
        pending = _RemoteApprovalRequest(
            tool_name=tool.name,
            reason=decision.reason,
            event=threading.Event(),
            created_at=time.monotonic(),
        )
        with state.lock:
            state.pending_approval = pending
            state.last_status = f"approval pending: {tool.name}"
        self._send_text(
            message,
            _approval_prompt(
                tool.name,
                decision.reason,
                self.deps.settings.remote.wecom.approval_timeout_seconds,
            ),
        )
        approved = pending.event.wait(
            self.deps.settings.remote.wecom.approval_timeout_seconds
        )
        if not approved:
            with state.lock:
                if state.pending_approval is pending:
                    state.pending_approval = None
                state.last_status = f"approval timed out: {tool.name}"
            self._defer_heartbeat(state)
            self._send_text(
                message,
                "远程审批超时，已取消这次操作。",
            )
            return False
        if not pending.approved:
            return False
        if not self._refresh_state_binding(state, force=True).route_open:
            with state.lock:
                state.last_status = f"approval denied because route closed: {tool.name}"
            return False
        return True

    def _approve_safety_decision(
        self,
        message: WeComInboundMessage,
        state: _SessionState,
        decision: SafetyDecision,
    ) -> ApprovalDecision:
        if self._is_readonly_group_message(message):
            self._send_text(
                message,
                "远程审批已拒绝：当前企业微信群策略为 readonly，不能执行 shell 或高风险操作。",
            )
            return ApprovalDecision.DENY
        if not self._refresh_state_binding(state, force=True).route_open:
            self._send_text(
                message,
                _remote_route_closed_message(),
            )
            return ApprovalDecision.DENY
        name = _safety_decision_name(decision)
        pending = _RemoteApprovalRequest(
            tool_name=name,
            reason=_safety_decision_reason(decision),
            event=threading.Event(),
            created_at=time.monotonic(),
        )
        with state.lock:
            state.pending_approval = pending
            state.last_status = f"approval pending: {name}"
        self._send_text(
            message,
            _approval_prompt(
                name,
                pending.reason,
                self.deps.settings.remote.wecom.approval_timeout_seconds,
            ),
        )
        approved = pending.event.wait(
            self.deps.settings.remote.wecom.approval_timeout_seconds
        )
        if not approved:
            with state.lock:
                if state.pending_approval is pending:
                    state.pending_approval = None
                state.last_status = f"approval timed out: {name}"
            self._defer_heartbeat(state)
            self._send_text(
                message,
                "远程审批超时，已取消这次操作。",
            )
            return ApprovalDecision.DENY
        if not pending.approved:
            return ApprovalDecision.DENY
        if not self._refresh_state_binding(state, force=True).route_open:
            with state.lock:
                state.last_status = f"approval denied because route closed: {name}"
            return ApprovalDecision.DENY
        return ApprovalDecision.ALLOW_ONCE

    def _start_heartbeat(
        self,
        message: WeComInboundMessage,
        state: _SessionState,
    ) -> None:
        if not self.deps.settings.remote.wecom.progress_heartbeat:
            return
        if not self._refresh_state_binding(state, force=True).route_open:
            return
        self._stop_heartbeat(state)
        now = time.monotonic()
        stop_event = threading.Event()
        with state.lock:
            state.progress_started_at = now
            state.last_heartbeat_at = now
            state.heartbeat_index = 0
            state.heartbeat_stop_event = stop_event
        worker = threading.Thread(
            target=self._heartbeat_loop,
            args=(message, state, stop_event),
            name=f"deepmate-wecom-heartbeat-{message.req_id}",
            daemon=True,
        )
        with state.lock:
            state.heartbeat_thread = worker
        worker.start()

    def _stop_heartbeat(self, state: _SessionState) -> None:
        with state.lock:
            stop_event = state.heartbeat_stop_event
            worker = state.heartbeat_thread
            state.heartbeat_stop_event = None
            state.heartbeat_thread = None
        if stop_event is not None:
            stop_event.set()
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=1.0)

    def _defer_heartbeat(self, state: _SessionState) -> None:
        with state.lock:
            if state.heartbeat_stop_event is None:
                return
            state.last_heartbeat_at = time.monotonic()

    def _heartbeat_loop(
        self,
        message: WeComInboundMessage,
        state: _SessionState,
        stop_event: threading.Event,
    ) -> None:
        while not stop_event.wait(_heartbeat_check_interval(state)):
            with state.lock:
                is_processing = state.is_processing
                pending_approval = state.pending_approval is not None
                due = _heartbeat_due(
                    state,
                    self.deps.settings.remote.wecom.progress_intervals_seconds,
                )
            if not is_processing:
                return
            if not self._refresh_state_binding(state).route_open:
                return
            if pending_approval:
                continue
            if not due:
                continue
            self._send_stream(message, state, _remote_heartbeat_text(state), finish=False)
            self._record_tail(state, "heartbeat: remote progress sync sent")
            with state.lock:
                state.last_heartbeat_at = time.monotonic()
                state.heartbeat_index += 1

    def _refresh_state_binding(
        self,
        state: _SessionState,
        *,
        force: bool = False,
    ) -> RemoteBindingRecord:
        with state.lock:
            current = state.binding
            session_id = state.session.session_id
            now = time.monotonic()
            if (
                not force
                and state.last_binding_refresh_at
                and now - state.last_binding_refresh_at
                < REMOTE_BINDING_REFRESH_INTERVAL_SECONDS
            ):
                return current
        try:
            latest = self.binding_store.get(CHANNEL, current.remote_user_id)
            if latest is None and current.remote_user_id != "*":
                latest = self.binding_store.get(CHANNEL, "*")
        except (OSError, ValueError, json.JSONDecodeError):
            return current
        with state.lock:
            state.last_binding_refresh_at = time.monotonic()
            if latest is not None and latest.session_id == session_id:
                state.binding = latest
            return state.binding

    def _status_sink(self, message: WeComInboundMessage, state: _SessionState):
        def emit(status: str) -> None:
            if self.deps.status_sink is not None:
                self.deps.status_sink(status)
            clean_status = _compact_line(status, REMOTE_STATUS_LIMIT)
            with state.lock:
                state.last_status = clean_status
            self._record_tail(state, f"status: {clean_status}")

        return emit

    def _subagents_for_state(
        self,
        state: _SessionState,
        tool_access_policy: ToolAccessPolicy,
    ) -> SubagentToolExecutor | None:
        if self.deps.subagents is None:
            return None
        return self.deps.subagents.bind_runtime(
            capability_surface=self.deps.capability_surface,
            native_tools=self.deps.native_tools,
            mcp_tools=self.deps.mcp_tools,
            tool_schemas=tuple(self.deps.tool_schemas),
            selected_skill_documents=tuple(self.deps.selected_skill_documents),
            activation=state.runtime.activation,
            parent_tool_access_policy=tool_access_policy,
            result_store=SubagentResultStore.in_data_dir(
                self.deps.settings.data_dir,
                state.session.session_id,
            ),
        )

    def _record_tail(self, state: _SessionState, text: str) -> None:
        clean = text.strip()
        if not clean:
            return
        timestamp = datetime.now().astimezone().replace(microsecond=0).strftime("%H:%M:%S")
        entry = f"[{timestamp}] {clean}"
        with state.lock:
            if state.tail_log:
                state.tail_log = f"{state.tail_log}\n{entry}"
            else:
                state.tail_log = entry
            state.tail_log = _tail_text(state.tail_log, REMOTE_TAIL_LIMIT_BYTES)

    def _send_text(self, message: WeComInboundMessage, text: str) -> bool:
        clean = text.strip()
        if not clean:
            return True
        chunks = _split_remote_text_bytes(clean, REMOTE_REPLY_LIMIT_BYTES)
        if not chunks:
            return True
        ok = True
        for index, chunk in enumerate(chunks, start=1):
            ok = self._send_content_chunk(message, chunk) and ok
        return ok

    def _send_content_chunk(self, message: WeComInboundMessage, text: str) -> bool:
        if _looks_like_markdown(text):
            return self._send_payload(
                {
                    "cmd": "aibot_respond_msg",
                    "req_id": message.req_id,
                    "msgtype": "markdown",
                    "markdown": {"content": text},
                }
            )
        return self._send_payload(
            {
                "cmd": "aibot_respond_msg",
                "req_id": message.req_id,
                "msgtype": "text",
                "text": {"content": text},
            }
        )

    def _send_stream(
        self,
        message: WeComInboundMessage,
        state: _SessionState,
        text: str,
        *,
        finish: bool,
    ) -> bool:
        clean = text.strip()
        if not clean:
            return True
        with state.lock:
            stream_id = state.active_stream_id or _stream_id(message, state)
            state.active_stream_id = stream_id
        chunks = _split_remote_stream_text(clean, REMOTE_STREAM_LIMIT_BYTES)
        if not chunks:
            return True
        ok = True
        for index, chunk in enumerate(chunks, start=1):
            is_final_chunk = finish and index == len(chunks)
            ok = self._send_stream_chunk(
                message,
                state,
                stream_id=stream_id,
                text=chunk,
                finish=is_final_chunk,
            ) and ok
        return ok

    def _send_stream_chunk(
        self,
        message: WeComInboundMessage,
        state: _SessionState,
        *,
        stream_id: str,
        text: str,
        finish: bool,
    ) -> bool:
        with state.lock:
            state.active_stream_sequence += 1
            started_at = state.progress_started_at
        if finish and _passive_reply_window_exceeded(started_at):
            self.deps.trace_recorder.record(
                TraceEvent(
                    kind="wecom_passive_reply_window_exceeded",
                    summary=(
                        "WeCom passive reply may be rejected because the inbound "
                        "req_id reply window was exceeded."
                    ),
                    refs=(
                        f"req_id={message.req_id}",
                        f"elapsed_seconds={time.monotonic() - started_at:.3f}",
                        "reason=wecom_passive_reply_window",
                    ),
                )
            )
        return self._send_payload(
            {
                "cmd": "aibot_respond_msg",
                "req_id": message.req_id,
                "msgtype": "stream",
                "stream": {
                    "id": stream_id,
                    "content": text,
                    "finish": finish,
                },
            }
        )

    def _record_remote_send_failure(self, state: _SessionState, context: str) -> None:
        with state.lock:
            state.last_error = f"WeCom send failed while sending {context}."
            state.last_status = "reply_failed"
        self.deps.trace_recorder.record(
            TraceEvent(
                kind="wecom_reply_failed",
                summary=f"WeCom send failed while sending {context}.",
                refs=("reason=send_payload_failed",),
            )
        )

    def _send_payload(self, payload: dict[str, object]) -> bool:
        if not self._wait_for_send_slot(timeout=WEBCOM_SEND_SLOT_TIMEOUT_SECONDS):
            self.deps.trace_recorder.record(
                TraceEvent(
                    kind="wecom_send_rate_limited",
                    summary="WeCom send skipped because rate limit wait timed out.",
                    refs=(f"timeout_seconds={WEBCOM_SEND_SLOT_TIMEOUT_SECONDS:g}",),
                )
            )
            return False
        try:
            self.sender(payload)
            return True
        except WEBCOM_SEND_ERRORS as exc:
            self.deps.trace_recorder.record(
                TraceEvent(
                    kind="wecom_send_failed",
                    summary=f"WeCom send failed: {exc}",
                    refs=(f"error_type={type(exc).__name__}",),
                )
            )
            return False

    def _wait_for_send_slot(self, *, timeout: float) -> bool:
        limit = max(1, self.deps.settings.remote.wecom.max_messages_per_minute)
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            with self._send_rate_lock:
                now = time.monotonic()
                window_start = now - 60.0
                self._send_timestamps = [
                    value for value in self._send_timestamps if value >= window_start
                ]
                if len(self._send_timestamps) < limit:
                    self._send_timestamps.append(now)
                    return True
                sleep_for = max(0.05, 60.0 - (now - self._send_timestamps[0]))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(sleep_for, 60.0, remaining))

    def _allowed_user(self, user_id: str) -> bool:
        allowed = self.deps.settings.remote.wecom.allowed_users
        return "*" in allowed or user_id in allowed

    def _group_denial_reason(self, message: WeComInboundMessage) -> str:
        if not _is_group_message(message):
            return ""
        if self.deps.settings.remote.wecom.group_policy != "deny":
            return ""
        return "Deepmate remote is disabled in Enterprise WeChat group chats."

    def _readonly_group_command_denied(
        self,
        message: WeComInboundMessage,
        clean: str,
    ) -> bool:
        if not self._is_readonly_group_message(message):
            return False
        command = clean.strip().lower()
        if _is_preview_command(command):
            return True
        if not command.startswith("/"):
            return False
        return command not in {"/current", "/status", "/tail", "/pending", "/close"}

    def _is_readonly_group_message(self, message: WeComInboundMessage) -> bool:
        return (
            _is_group_message(message)
            and self.deps.settings.remote.wecom.group_policy == "readonly"
        )

    def _handle_protocol_status_payload(self, payload: Mapping[str, object]) -> bool:
        errcode = _int_field(payload, "errcode")
        if errcode is None:
            return False
        errmsg = _text(payload, "errmsg") or _text(payload, "error")
        cmd = _text(payload, "cmd") or _text(payload, "command")
        if errcode != 0:
            self.deps.trace_recorder.record(
                TraceEvent(
                    kind="wecom_protocol_error",
                    summary=f"WeCom protocol error: {errcode} {errmsg}".strip(),
                    refs=(
                        f"cmd={cmd or '-'}",
                        f"errcode={errcode}",
                        f"errmsg={_compact_line(errmsg, 300) or '-'}",
                    ),
                )
            )
        return True

    def _default_wake_session(self, reason: str) -> RuntimeWakeSession:
        wake = self.deps.settings.wake
        return RuntimeWakeSession(
            reason,
            WakeConfig(
                enabled=wake.enabled,
                post_turn_grace_minutes=wake.post_turn_grace_minutes,
            ),
        )


def run_wecom_remote_channel(deps: WeComRunDependencies) -> int:
    """Run the blocking WeCom remote channel loop with reconnects."""
    settings = deps.settings.remote.wecom
    settings.validate_ready()
    client_ref: dict[str, WeComWsClient | None] = {"client": None}

    def send(payload: dict[str, object]) -> None:
        client = client_ref.get("client")
        if client is None:
            raise RuntimeError("WeCom websocket is not connected")
        client.send_json(payload)

    channel = WeComChannel(deps, sender=send)
    reconnect_delay = WEBCOM_RECONNECT_INITIAL_DELAY_SECONDS
    while True:
        client = WeComWsClient(
            WeComClientConfig(bot_id=settings.bot_id, secret=settings.secret)
        )
        client_ref["client"] = client
        ping_stop = threading.Event()
        try:
            client.connect()
            client.subscribe()
            initial_payload = _wait_for_subscribe_ack(
                client,
                timeout=WEBCOM_SUBSCRIBE_ACK_TIMEOUT_SECONDS,
            )
            reconnect_delay = WEBCOM_RECONNECT_INITIAL_DELAY_SECONDS
            ping_thread = threading.Thread(
                target=_wecom_ping_loop,
                args=(client, ping_stop, deps.trace_recorder),
                name="deepmate-wecom-ping",
                daemon=True,
            )
            ping_thread.start()
            if initial_payload is not None:
                channel.dispatch_payload(initial_payload)
            while True:
                try:
                    payload = client.recv_json()
                except WeComPayloadError as exc:
                    deps.trace_recorder.record(
                        TraceEvent(
                            kind="wecom_recv_ignored",
                            summary=f"Ignored malformed WeCom frame: {exc}",
                            refs=(f"error_type={type(exc).__name__}",),
                        )
                    )
                    continue
                channel.dispatch_payload(payload)
        except (EOFError, OSError, RuntimeError, WeComProtocolError) as exc:
            deps.trace_recorder.record(
                TraceEvent(
                    kind="wecom_reconnect",
                    summary=f"WeCom channel reconnecting after {type(exc).__name__}: {exc}",
                    refs=(f"delay_seconds={reconnect_delay:g}",),
                )
            )
        finally:
            ping_stop.set()
            client_ref["client"] = None
            client.close()
        time.sleep(reconnect_delay)
        reconnect_delay = min(
            reconnect_delay * 2,
            WEBCOM_RECONNECT_MAX_DELAY_SECONDS,
        )


def parse_wecom_message(payload: Mapping[str, object]) -> WeComInboundMessage | None:
    cmd = _text(payload, "cmd")
    if cmd and cmd not in {"aibot_msg_callback", "message"}:
        return None
    content = _message_content(payload)
    if not content:
        return None
    user_id = (
        _text(payload, "user_id")
        or _text(payload, "from_user")
        or _text(payload, "userid")
        or _nested_text(payload, "message", "user_id")
        or _nested_text(payload, "msg", "user_id")
    )
    if not user_id:
        return None
    req_id = _text(payload, "req_id") or _text(payload, "msg_id")
    if not req_id:
        return None
    return WeComInboundMessage(
        req_id=req_id,
        user_id=user_id,
        chat_id=_text(payload, "chat_id") or _text(payload, "conversation_id"),
        chat_type=_text(payload, "chat_type") or _text(payload, "conversation_type") or "single",
        content=content,
    )


def _wait_for_subscribe_ack(
    client: WeComWsClient,
    *,
    timeout: float,
) -> dict[str, object] | None:
    payload = client.recv_json_timeout(timeout)
    if payload is None:
        return None
    cmd = _text(payload, "cmd") or _text(payload, "command")
    if cmd not in {"aibot_subscribe", "subscribe", "aibot_subscribe_ack"}:
        return payload
    success = payload.get("success")
    errcode = payload.get("errcode", payload.get("error_code"))
    status = _text(payload, "status").lower()
    if success is False or (isinstance(errcode, int) and errcode != 0):
        raise RuntimeError(f"WeCom subscribe failed: {payload}")
    if status and status not in {"ok", "success", "subscribed"}:
        raise RuntimeError(f"WeCom subscribe failed: {payload}")
    return None


def _current_status(state: _SessionState) -> str:
    with state.lock:
        pending = state.pending_approval
        is_processing = state.is_processing
        last_status = state.last_status
        pending_followups = state.followup_buffer.pending_count()
        binding = state.binding
        session = state.session
    pending_text = "有" if pending is not None else "无"
    return "\n".join(
        (
            "企业微信接管状态：",
            f"接管：{'已开启' if binding.route_open else '已关闭'}",
            f"当前任务：{session.title}",
            f"任务状态：{'运行中' if is_processing else '空闲'}",
            f"最近状态：{_user_status_text(last_status)}",
            f"待审批：{pending_text}",
            f"补充要求：{pending_followups} 条",
        )
    )


def _pending_status(state: _SessionState) -> str:
    with state.lock:
        pending = state.pending_approval
    if pending is None:
        return "当前没有等待远程审批的操作。"
    return "\n".join(
        (
            "等待远程审批：",
            f"操作：{_approval_action_summary(pending.reason, pending.tool_name)}",
            f"原因：{_compact_line(pending.reason, 500)}",
            "回复 /approve 批准一次，或 /deny 拒绝。",
        )
    )


def _profile_context_snapshot(
    settings: AppSettings,
    workspace: Path,
    profile: ProfileRef,
    model: str,
    provider_name: str = "",
):
    provider_settings = None
    if provider_name.strip():
        try:
            provider_settings = settings.provider(provider_name)
        except ValueError:
            provider_settings = None
    model_context_tokens = (
        settings.provider_context_tokens(provider_settings, model)
        if provider_settings is not None
        else settings.model_context_tokens(model)
    )
    return build_profile_context_snapshot(
        workspace=workspace,
        profile=profile,
        hot_profile_token_budget=(
            settings.context.resolved_hot_profile_token_budget(model_context_tokens)
        ),
        hot_profile_warn_tokens=(
            settings.context.hot_profile_warn_tokens(model_context_tokens)
        ),
    )


def _load_latest_summary(session_store: SessionStore, session: SessionRecord):
    try:
        return session_store.summary_store(session).load_latest()
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _response_text(response: ModelResponse) -> str:
    return response.content.strip() or response.reasoning.strip()


def _remote_result_text(result, response: ModelResponse) -> str:
    text = _response_text(response) or "任务已完成。"
    lines = ["任务已完成。", text]
    errors = tuple(
        error
        for error in result.errors()
        if not (
            result.loop_guard_stop is not None
            and error.code == f"loop_guard_{result.loop_guard_stop.reason.value}"
        )
    )
    if errors:
        lines.append("")
        lines.append("注意：执行过程中出现错误，已尽量完成。")
        for error in errors[:2]:
            lines.append(f"- {error.code}: {_compact_line(error.message, 240)}")
    if result.loop_guard_stop is not None:
        lines.append("")
        lines.append(
            "本轮已暂停："
            f"{_compact_line(result.loop_guard_stop.message, 360)}"
        )
        lines.append("可直接回复“继续”恢复执行。")
    elif result.reached_max_steps:
        lines.append("")
        lines.append("已达到本轮最大 step，可继续发送补充要求。")
    lines.append("")
    lines.append("可回复 /status 查看状态，/tail 查看最近输出。")
    return "\n".join(line for line in lines if line is not None).strip()


def _remote_error_text(state: _SessionState, exc: Exception) -> str:
    return "\n".join(
        (
            "执行失败。",
            f"原因：{_compact_line(str(exc), 600) or type(exc).__name__}",
            "可回复 /status 查看当前状态，修正后继续发送要求。",
        )
    )


def _turn_started_message(state: _SessionState) -> str:
    return "\n".join(
        (
            "收到，正在处理。",
            f"当前任务：{state.session.title}",
            "运行中可继续发送补充要求；回复 /status 查看状态，/tail 查看最近输出。",
        )
    )


def _approval_prompt(tool_name: str, reason: str, timeout_seconds: int) -> str:
    return "\n".join(
        (
            "需要远程审批。",
            f"操作：{_approval_action_summary(reason, tool_name)}",
            f"原因：{_compact_line(reason, 500)}",
            f"请在 {timeout_seconds} 秒内处理。",
            "回复 /approve 批准一次，或 /deny 拒绝。",
        )
    )


def _remote_route_closed_message() -> str:
    return "企业微信接管已关闭，请先发送 /open 或在本机重新开启远程接管。"


def _approval_action_summary(reason: str, tool_name: str) -> str:
    clean_tool = tool_name.strip()
    clean_reason = reason.strip()
    lower_tool = clean_tool.lower()
    lower_reason = clean_reason.lower()
    if "shell" in lower_tool or "run_shell" in lower_tool or "command" in lower_reason:
        command = _command_from_reason(clean_reason)
        return f"运行命令：{command}" if command else "运行一条命令"
    if "write" in lower_tool or "edit" in lower_tool or "patch" in lower_tool:
        return "修改项目文件"
    if "delete" in lower_tool or "remove" in lower_tool:
        return "删除项目文件"
    if "browser" in lower_tool:
        return "操作浏览器"
    if "deploy" in lower_tool:
        return "创建或管理预览链接"
    if "mcp" in lower_tool:
        return "调用外部连接能力"
    if clean_tool:
        return clean_tool.replace("_", " ")
    return "执行一个需要确认的操作"


def _command_from_reason(reason: str) -> str:
    command_key = "command="
    command_index = reason.lower().find(command_key)
    if command_index >= 0:
        tail = reason[command_index + len(command_key) :].strip()
        command = tail.split(",", 1)[0].rstrip(")").strip()
        if command:
            return _compact_line(command, 160)
    for marker in ("command:", "cmd:", "shell:"):
        index = reason.lower().find(marker)
        if index >= 0:
            return _compact_line(reason[index + len(marker) :].strip(), 160)
    quoted = reason.split("`")
    if len(quoted) >= 3 and quoted[1].strip():
        return _compact_line(quoted[1].strip(), 160)
    return ""


def _user_status_text(status: str) -> str:
    clean = status.strip()
    lower = clean.lower()
    if not clean:
        return "暂无"
    if lower == "running":
        return "运行中"
    if lower == "completed":
        return "已完成"
    if lower == "failed":
        return "执行失败"
    if lower == "remote route closed":
        return "接管已关闭"
    if lower == "remote route open":
        return "接管已开启"
    if lower.startswith("approval pending:"):
        return "等待你确认操作"
    if lower.startswith("approval allowed once:"):
        return "已允许一次操作"
    if lower.startswith("approval denied"):
        return "已拒绝操作"
    if lower.startswith("approval timed out:"):
        return "等待确认超时"
    if lower.startswith("follow-up queued:"):
        return "已收到补充要求"
    if lower.startswith("completed with ") and "unconsumed follow-up" in lower:
        return "已完成，仍有补充要求未处理"
    if _remote_status_is_waiting(clean):
        return "正在处理"
    return _compact_line(clean, 520)


def _safety_decision_name(decision: SafetyDecision) -> str:
    if decision.approval_key:
        return decision.approval_key
    if decision.risk_level:
        return f"{decision.risk_level.value} action"
    return "remote action"


def _safety_decision_reason(decision: SafetyDecision) -> str:
    refs = ", ".join(decision.refs[:4])
    if refs:
        return f"{decision.reason} ({refs})".strip()
    return decision.reason.strip() or "Remote approval is required."


def _remote_heartbeat_text(state: _SessionState) -> str:
    with state.lock:
        elapsed = _elapsed_text(time.monotonic() - state.progress_started_at)
        status = state.last_status.strip()
        pending_followups = state.followup_buffer.pending_count()
    if not status or status == "running" or _remote_status_is_waiting(status):
        status = "等待模型或工具返回"
    return "\n".join(
        (
            "任务仍在执行。",
            f"已运行 {elapsed}，最近状态：{_user_status_text(status)}",
            f"待处理补充要求：{pending_followups} 条",
            "可继续发送补充要求；回复 /status 查看详情，/tail 查看最近输出。",
        )
    )


def _remote_status_is_waiting(status: str) -> bool:
    clean = status.strip().lower()
    return clean.startswith("model request started:") or clean.startswith("runtime step ")


def _heartbeat_due(
    state: _SessionState,
    intervals: tuple[int, ...],
) -> bool:
    if not intervals:
        return False
    return time.monotonic() - state.last_heartbeat_at >= _heartbeat_interval(
        intervals,
        state.heartbeat_index,
    )


def _heartbeat_interval(intervals: tuple[int, ...], index: int) -> int:
    if not intervals:
        return 0
    if index < len(intervals):
        return intervals[index]
    return intervals[-1]


def _heartbeat_check_interval(state: _SessionState) -> float:
    return 1.0


def _elapsed_text(seconds: float) -> str:
    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s"
    minutes = total // 60
    if minutes < 60:
        return f"{minutes}min"
    hours = minutes // 60
    remainder = minutes % 60
    return f"{hours}h{remainder}min" if remainder else f"{hours}h"


def _remote_session_title(workspace: Path, prompt: str) -> str:
    name = workspace.name or "workspace"
    clean = " ".join(prompt.strip().split())
    if clean:
        return f"WeCom Remote - {clean[:40]}"
    return f"WeCom Remote - {name}"


def _stream_id(message: WeComInboundMessage, state: _SessionState) -> str:
    suffix = _safe_stream_id_part(message.req_id, max_chars=56)
    if suffix:
        return f"dm-{suffix}"
    fallback = _safe_stream_id_part(state.session.session_id, max_chars=56)
    return f"dm-{fallback or int(time.time() * 1000)}"


def _safe_stream_id_part(text: str, *, max_chars: int) -> str:
    clean = []
    for char in text.strip():
        if char.isalnum() or char in {"-", "_"}:
            clean.append(char)
        elif char in {":", ".", "/"}:
            clean.append("-")
    return "".join(clean).strip("-_")[:max_chars]


def _clean_prompt(content: str) -> str:
    clean = content.strip()
    if clean.startswith("@deepmate"):
        clean = clean[len("@deepmate") :].strip()
    return clean


def _is_remote_command(text: str) -> bool:
    command = text.strip().lower()
    if command.startswith(("/open ", "/deploy ")):
        return True
    if _is_preview_command(command):
        return True
    return command in {
        "/current",
        "/status",
        "/tail",
        "/pending",
        "/approve",
        "/approve once",
        "/deny",
        "/close",
        "/open",
        "/deploy",
    }


def _is_preview_command(command: str) -> bool:
    return command.startswith("/deploy") or command in {
        "关闭当前预览链接",
        "关闭预览链接",
        "关闭预览",
    }


def _tail_text(text: str, limit: int) -> str:
    clean = text.strip()
    if _utf8_len(clean) <= limit:
        return clean
    tail = _trim_utf8_prefix(clean, max(0, limit - _utf8_len("...[tail]\n")))
    return "...[tail]\n" + tail.lstrip()


def _split_remote_text_bytes(text: str, limit: int) -> list[str]:
    clean = text.strip()
    if not clean:
        return []
    limit = max(128, limit)
    if _utf8_len(clean) <= limit:
        return [clean]
    chunks: list[str] = []
    remaining = clean
    label_reserve = _utf8_len(f"[999/999]\n")
    body_limit = max(128, limit - label_reserve)
    while remaining:
        chunk = _fit_utf8_prefix(remaining, body_limit)
        if not chunk:
            break
        chunks.append(chunk.rstrip())
        remaining = remaining[len(chunk) :].lstrip()
    total = len(chunks)
    return [
        _with_chunk_label(chunk, index=index, total=total)
        for index, chunk in enumerate(chunks, start=1)
        if chunk
    ]


def _split_remote_stream_text(text: str, limit: int) -> list[str]:
    return _split_remote_text_bytes(text, limit)


def _fit_utf8_prefix(text: str, limit: int) -> str:
    if _utf8_len(text) <= limit:
        return text
    used = 0
    end = 0
    for index, char in enumerate(text):
        size = _utf8_len(char)
        if used + size > limit:
            break
        used += size
        end = index + 1
    prefix = text[:end]
    split_at = max(prefix.rfind("\n"), prefix.rfind("。"), prefix.rfind("."), prefix.rfind(" "))
    if split_at > max(40, int(len(prefix) * 0.45)):
        prefix = prefix[: split_at + 1]
    return prefix or text[:1]


def _trim_utf8_prefix(text: str, limit: int) -> str:
    if _utf8_len(text) <= limit:
        return text
    used = 0
    start = len(text)
    for index in range(len(text) - 1, -1, -1):
        size = _utf8_len(text[index])
        if used + size > limit:
            break
        used += size
        start = index
    return text[start:]


def _passive_reply_window_exceeded(started_at: float) -> bool:
    if started_at <= 0:
        return False
    return time.monotonic() - started_at > WEBCOM_PASSIVE_REPLY_WINDOW_SECONDS


def _with_chunk_label(text: str, *, index: int, total: int) -> str:
    if total <= 1:
        return text
    label = f"[{index}/{total}]\n"
    return label + text


def _looks_like_markdown(text: str) -> bool:
    clean = text.strip()
    if "```" in clean:
        return True
    for line in clean.splitlines():
        stripped = line.strip()
        if stripped.startswith(("# ", "## ", "### ", "> ")):
            return True
        if stripped.startswith(("- ", "* ", "1. ")) and len(stripped) > 2:
            return True
    return False


def _is_group_message(message: WeComInboundMessage) -> bool:
    clean = message.chat_type.strip().lower()
    return clean in {"group", "room", "chatroom", "group_chat", "multi"}


def _utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _compact_line(text: str, limit: int) -> str:
    clean = " ".join(text.strip().split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def _message_content(payload: Mapping[str, object]) -> str:
    direct = _text(payload, "content") or _text(payload, "text")
    if direct:
        return direct
    for key in ("message", "msg", "text"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            nested = _text(value, "content") or _text(value, "text")
            if nested:
                return nested
    return ""


def _text(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    return value.strip() if isinstance(value, str) else ""


def _int_field(payload: Mapping[str, object], key: str) -> int | None:
    value = payload.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _nested_text(payload: Mapping[str, object], parent: str, key: str) -> str:
    value = payload.get(parent)
    if not isinstance(value, Mapping):
        return ""
    nested = value.get(key)
    return nested.strip() if isinstance(nested, str) else ""


def _wecom_ping_loop(
    client: WeComWsClient,
    stop_event: threading.Event,
    trace_recorder: TraceRecorder,
) -> None:
    while not stop_event.wait(WEBCOM_PING_INTERVAL_SECONDS):
        try:
            client.ping()
        except WEBCOM_PING_ERRORS as exc:
            trace_recorder.record(
                TraceEvent(
                    kind="wecom_ping_failed",
                    summary=f"WeCom ping failed: {exc}",
                    refs=(f"error_type={type(exc).__name__}",),
                )
            )
            return
