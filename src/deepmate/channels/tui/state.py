"""State objects shared by the Textual TUI bridge and app."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from deepmate.capabilities import CapabilitySurface
from deepmate.capabilities.state import CapabilityStateStore
from deepmate.behavior import BehaviorRuntime
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
)
from deepmate.channels.tui.formatters import TuiMessage
from deepmate.channels.tui.status import TuiRuntimeStats
from deepmate.context import ContextWarning
from deepmate.domain import ProfileRef
from deepmate.local import (
    LOCAL_PROVIDER_API_KEY,
    LOCAL_PROVIDER_BASE_URL,
    LocalModelPreset,
    local_model_by_id,
    local_model_by_runtime_name,
    recommended_local_model,
)
from deepmate.mcp import McpServerSpec, McpToolExecutor
from deepmate.pet.state import PetStateStore
from deepmate.providers import ChatCompletionsProvider, ModelCapabilities
from deepmate.runtime import (
    ConversationBudgetPolicy,
    HookLoadOptions,
    HookRuntimeContext,
    LoopGuardPolicy,
    ProviderRetryPolicy,
    ApprovalDecision,
    SafetyDecision,
    SessionApprovalCache,
    SessionRuntime,
    ToolAccessDecision,
    ToolAccessPolicy,
    ToolOutputCompactor,
    ToolRepairPolicy,
    TurnCancellationToken,
    TurnFollowupBuffer,
)
from deepmate.skills import SkillDocument
from deepmate.storage import SessionRecord, SessionStore, TranscriptStore
from deepmate.subagents import SubagentToolExecutor
from deepmate.tasks import TaskSessionController
from deepmate.tools import NativeTool, NativeToolRegistry
from deepmate.trace import TraceRecorder

NativeToolFactory = Callable[[SessionApprovalCache | None, str], NativeToolRegistry | None]
LocalContextPrepareHandler = Callable[["TuiRuntimeState", LocalModelPreset], bool]


@dataclass(slots=True)
class TuiPromptQueue:
    """Small app-local queue for prompts submitted while a turn is running."""

    pending: list[str] = field(default_factory=list)
    paused: bool = False
    max_size: int = 50
    path: Path | None = None

    @classmethod
    def load(cls, path: str | Path, *, max_size: int = 50) -> "TuiPromptQueue":
        """Load a persisted prompt queue, ignoring malformed sidecar content."""
        resolved = Path(path)
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls(max_size=max_size, path=resolved)
        if not isinstance(payload, Mapping):
            return cls(max_size=max_size, path=resolved)
        raw_pending = payload.get("pending")
        pending = (
            [
                item.strip()
                for item in raw_pending
                if isinstance(item, str) and item.strip()
            ]
            if isinstance(raw_pending, list)
            else []
        )
        effective_max_size = max(1, max_size)
        return cls(
            pending=pending[:effective_max_size],
            paused=payload.get("paused") is True and bool(pending),
            max_size=effective_max_size,
            path=resolved,
        )

    def enqueue(self, prompt: str) -> int:
        """Queue a non-empty prompt and return the new queue length."""
        clean = prompt.strip()
        if clean and len(self.pending) < self.max_size:
            self.pending.append(clean)
            self.save()
        return len(self.pending)

    def is_full(self) -> bool:
        """Return whether the queue can accept more prompts."""
        return len(self.pending) >= self.max_size

    def pop_next(self) -> str | None:
        """Return the next queued prompt when auto-drain is allowed."""
        if self.paused or not self.pending:
            return None
        prompt = self.pending.pop(0)
        if not self.pending:
            self.paused = False
        self.save()
        return prompt

    def pause(self) -> None:
        """Pause automatic queue draining."""
        if self.pending:
            self.paused = True
            self.save()

    def resume(self) -> None:
        """Resume automatic queue draining."""
        self.paused = False
        self.save()

    def clear(self) -> int:
        """Clear queued prompts and return how many were removed."""
        count = len(self.pending)
        self.pending.clear()
        self.paused = False
        self.save()
        return count

    def footer_label(self) -> str:
        """Return a compact footer label."""
        if not self.pending:
            return ""
        label = f"queued {len(self.pending)}"
        if self.paused:
            label += " paused"
        return label

    def bind_path(self, path: str | Path | None) -> None:
        """Attach a persistence path to this queue and save current state."""
        self.path = Path(path) if path is not None else None
        self.save()

    def save(self) -> None:
        """Persist the queue sidecar when a path is configured."""
        if self.path is None:
            return
        if not self.pending:
            try:
                self.path.unlink()
            except OSError:
                pass
            return
        payload = {
            "pending": list(self.pending[: self.max_size]),
            "paused": self.paused and bool(self.pending),
            "max_size": self.max_size,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, self.path)


@dataclass(frozen=True, slots=True)
class WorkspaceSwitchRequest:
    """One request to restart the TUI in another workspace."""

    workspace: Path
    session_id: str = ""


@dataclass(frozen=True, slots=True)
class LocalModelPrepareRequest:
    """One local-model preparation request for the TUI worker."""

    preset: LocalModelPreset
    source: str = "local"
    defer_switch: bool = False


@dataclass(slots=True)
class TuiRuntimeState:
    """Mutable runtime dependencies owned by one TUI app instance."""

    provider: ChatCompletionsProvider
    provider_name: str
    provider_api_key_env: str
    provider_api_key_available: bool
    model: str
    default_model: str
    upgrade_model: str
    workspace: Path
    profile: ProfileRef
    session_store: SessionStore
    session: SessionRecord
    transcript: TranscriptStore
    runtime: SessionRuntime
    capability_surface: CapabilitySurface | None
    native_tools: NativeToolRegistry | None
    native_tool_factory: NativeToolFactory | None
    mcp_tools: McpToolExecutor | None
    subagents: SubagentToolExecutor | None
    tool_access_policy: ToolAccessPolicy | None
    tool_schemas: Sequence[Mapping[str, object]]
    selected_skill_documents: Sequence[SkillDocument]
    mcp_servers: Sequence[McpServerSpec]
    conversation_budget_policy: ConversationBudgetPolicy | None
    provider_retry_policy: ProviderRetryPolicy | None
    options: Mapping[str, object]
    max_steps: int
    trace_recorder: TraceRecorder
    warning_sink: Callable[[ContextWarning], None] | None
    model_capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    remote_provider: ChatCompletionsProvider | None = None
    remote_provider_name: str = ""
    remote_model: str = ""
    remote_default_model: str = ""
    remote_upgrade_model: str = ""
    remote_provider_api_key_env: str = ""
    remote_provider_api_key_available: bool = True
    remote_options: Mapping[str, object] = field(default_factory=dict)
    remote_model_capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    remote_conversation_budget_policy: ConversationBudgetPolicy | None = None
    local_provider: ChatCompletionsProvider | None = None
    local_provider_name: str = "local"
    local_provider_base_url: str = LOCAL_PROVIDER_BASE_URL
    local_provider_api_key: str = LOCAL_PROVIDER_API_KEY
    local_default_model: str = "qwen3-local"
    local_upgrade_model: str = "qwen3-coder-strong"
    missing_model_prompt_shown: bool = False
    pending_local_switch: LocalModelPreset | None = None
    loop_guard_policy: LoopGuardPolicy | None = None
    status_sink: Callable[[str], None] | None = None
    tool_output_compactor: ToolOutputCompactor | None = None
    tool_repair_policy: ToolRepairPolicy | None = None
    hook_context: HookRuntimeContext | None = None
    hook_load_options: HookLoadOptions | None = None
    data_dir: Path | None = None
    maintenance_handler: SessionMaintenanceHandler | None = None
    session_end_handler: SessionEndHandler | None = None
    context_snapshot_factory: ContextSnapshotFactory | None = None
    task_controller: TaskSessionController | None = None
    task_maintenance_handler: TaskMaintenanceHandler | None = None
    capability_state_store: CapabilityStateStore | None = None
    approval_cache: SessionApprovalCache | None = None
    checkpoint_controller: SessionCheckpointController | None = None
    checkpoint_controller_factory: CheckpointControllerFactory | None = None
    checkpoint_write_router: SessionCheckpointWriteRouter | None = None
    pet_state_store: PetStateStore | None = None
    show_reasoning: bool = False
    turn_index: int = 0
    tool_approval_callback: (
        Callable[[NativeTool, ToolAccessDecision], bool] | None
    ) = None
    safety_approval_callback: (
        Callable[[SafetyDecision], ApprovalDecision] | None
    ) = None
    approval_callbacks_installed: bool = False
    runtime_stats: TuiRuntimeStats = field(default_factory=TuiRuntimeStats)
    status_message_callback: Callable[[TuiMessage], None] | None = None
    live_status_callback: Callable[[TuiMessage], None] | None = None
    final_message_callback: Callable[[tuple[TuiMessage, ...]], None] | None = None
    # Receives (content_delta, reasoning_delta) per streamed fragment. Typed as
    # plain strings so state stays free of provider types; the app buffers and
    # throttles rendering. When None, the turn runs without token streaming.
    token_stream_callback: Callable[[str, str], None] | None = None
    followup_buffer: TurnFollowupBuffer | None = None
    active_followup_turn_id: str | None = None
    cancellation_token: TurnCancellationToken | None = None
    unconsumed_followups: tuple[str, ...] = ()
    task_continuations: tuple[str, ...] = ()
    pet_last_progress_key: str = ""
    refresh_skill_surface_callback: Callable[["TuiRuntimeState"], None] | None = None
    local_context_prepare_callback: LocalContextPrepareHandler | None = None
    behavior_runtime: BehaviorRuntime | None = None
    workspace_switch_request: WorkspaceSwitchRequest | None = None

    def current_local_preset(self) -> LocalModelPreset:
        """Return the active or default local preset for this session."""
        return (
            local_model_by_runtime_name(self.model)
            or local_model_by_runtime_name(self.local_default_model)
            or local_model_by_id(self.local_default_model)
            or recommended_local_model()
        )

    def task_stage_label(self) -> str:
        """Return a compact task stage label for the footer."""
        if self.task_controller is None or self.task_controller.active_stage is None:
            return ""
        return f"task:{self.task_controller.active_stage.value}"

    def prompt_queue_path(self) -> Path | None:
        """Return the session-scoped prompt queue sidecar path."""
        session_id = self.session.session_id.strip()
        if not session_id:
            return None
        root = (
            self.data_dir
            if self.data_dir is not None
            else self.session_store.directory.parent
        )
        return Path(root) / "tui" / "prompt_queues" / f"{session_id}.json"


def _int_value(value: object, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default
