"""Typed hook kernel objects.

The hook kernel is deliberately provider/tool neutral. Task 1 only loads,
validates, matches, and reports hooks; it does not connect hook actions to the
real runtime execution path yet.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path


def utc_now_iso() -> str:
    """Return an ISO timestamp for hook records."""
    return datetime.now(UTC).isoformat()


class HookLayer(StrEnum):
    """Where a hook definition came from."""

    BUILTIN = "builtin"
    MANAGED = "managed"
    USER = "user"
    PROJECT = "project"
    SESSION = "session"


HOOK_LAYER_ORDER: tuple[HookLayer, ...] = (
    HookLayer.BUILTIN,
    HookLayer.MANAGED,
    HookLayer.USER,
    HookLayer.PROJECT,
    HookLayer.SESSION,
)


class HookEvent(StrEnum):
    """Known runtime lifecycle events."""

    SESSION_START = "session.start"
    SESSION_SHUTDOWN = "session.shutdown"
    INPUT_RECEIVED = "input.received"
    ACTIVATION_BEFORE_START = "activation.before_start"
    AGENT_TURN_END = "agent.turn_end"
    CONTEXT_BEFORE_BUILD = "context.before_build"
    CONTEXT_AFTER_BUILD = "context.after_build"
    PROVIDER_BEFORE_REQUEST = "provider.before_request"
    PROVIDER_AFTER_RESPONSE = "provider.after_response"
    TOOL_BEFORE = "tool.before"
    TOOL_AFTER = "tool.after"
    WRITE_BEFORE = "write.before"
    WRITE_AFTER = "write.after"
    SHELL_BEFORE = "shell.before"
    SHELL_AFTER = "shell.after"
    MCP_BEFORE = "mcp.before"
    MCP_AFTER = "mcp.after"
    SUBAGENT_BEFORE_RUN = "subagent.before_run"
    SUBAGENT_AFTER_RUN = "subagent.after_run"
    SUBAGENT_WORKFLOW_END = "subagent.workflow_end"
    CHECKPOINT_CREATED = "checkpoint.created"
    MEMORY_PATCH_APPLIED = "memory.patch_applied"
    MAINTENANCE_BEFORE_RUN = "maintenance.before_run"
    MAINTENANCE_AFTER_RUN = "maintenance.after_run"
    EVOLUTION_CHANGE_APPLIED = "evolution.change_applied"
    SESSION_BRANCH_FORK = "session.branch_fork"
    SESSION_BRANCH_SWITCH = "session.branch_switch"
    SESSION_BRANCH_MERGE = "session.branch_merge"


class HookActor(StrEnum):
    """Runtime actor that emitted a hook event."""

    MAIN = "main"
    SUBAGENT = "subagent"
    MAINTENANCE = "maintenance"
    CLI = "cli"


class HookRunTarget(StrEnum):
    """Actor scope a hook is allowed to run on."""

    MAIN = "main"
    SUBAGENT = "subagent"
    MAINTENANCE = "maintenance"
    CLI = "cli"
    ALL = "all"


class HookActionType(StrEnum):
    """Declarative hook action names."""

    DENY = "deny"
    ASK = "ask"
    TRACE = "trace"
    CHECKPOINT = "checkpoint"
    COMPACT = "compact"
    RECORD_MEMORY_SIGNAL = "record_memory_signal"
    RECORD_EVOLUTION_SIGNAL = "record_evolution_signal"
    PATCH_TOOL_ARGS = "patch_tool_args"
    PATCH_PROVIDER_OPTIONS = "patch_provider_options"
    PATCH_TOOL_RESULT = "patch_tool_result"
    SET_STATUS = "set_status"
    NOTIFY = "notify"
    RUN_SHELL = "run_shell"
    CALL_MCP = "call_mcp"
    WORKSPACE_WRITE = "workspace_write"
    OVERRIDE_TOOL_SURFACE = "override_tool_surface"


HIGH_RISK_ACTION_TYPES: frozenset[HookActionType] = frozenset(
    {
        HookActionType.RUN_SHELL,
        HookActionType.CALL_MCP,
        HookActionType.WORKSPACE_WRITE,
        HookActionType.OVERRIDE_TOOL_SURFACE,
    }
)


class HookErrorPolicy(StrEnum):
    """How runtime should treat hook failures."""

    WARN = "warn"
    BLOCK = "block"
    SKIP = "skip"


class HookDirective(StrEnum):
    """Combined directive returned by a hook event."""

    CONTINUE = "continue"
    BLOCK = "block"
    REQUIRES_APPROVAL = "requires_approval"


class HookActionStatus(StrEnum):
    """Result status for one action."""

    APPLIED = "applied"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    REQUIRES_APPROVAL = "requires_approval"
    FAILED = "failed"


class HookDiagnosticLevel(StrEnum):
    """Diagnostic severity."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class HookLoadOptions:
    """Resolved hook loading options used by the runtime hook loader."""

    enabled: bool = True
    managed_hooks_only: bool = False
    load_project_hooks: bool = True
    load_user_hooks: bool = True
    trace_matches: bool = False
    before_timeout_ms: int = 300
    after_timeout_ms: int = 1000
    maintenance_timeout_ms: int = 5000

    def __post_init__(self) -> None:
        object.__setattr__(self, "before_timeout_ms", max(1, self.before_timeout_ms))
        object.__setattr__(self, "after_timeout_ms", max(1, self.after_timeout_ms))
        object.__setattr__(
            self,
            "maintenance_timeout_ms",
            max(1, self.maintenance_timeout_ms),
        )


@dataclass(frozen=True, slots=True)
class HookAction:
    """One declarative action in a hook definition."""

    action_type: HookActionType
    params: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HookDefinition:
    """One validated hook definition."""

    hook_id: str
    event_name: HookEvent
    layer: HookLayer
    enabled: bool = True
    description: str = ""
    run_on: HookRunTarget = HookRunTarget.MAIN
    when: Mapping[str, object] = field(default_factory=dict)
    actions: tuple[HookAction, ...] = field(default_factory=tuple)
    priority: int = 0
    on_error: HookErrorPolicy = HookErrorPolicy.WARN
    source_path: Path | None = None
    file_order: int = 0
    hook_order: int = 0

    def stable_key(self) -> str:
        """Return a stable identity key used by status and surface tags."""
        return f"{self.layer.value}:{self.hook_id}:{self.event_name.value}"


@dataclass(frozen=True, slots=True)
class HookEnvelope:
    """One event emitted into the hook kernel."""

    event_name: HookEvent
    actor: HookActor = HookActor.MAIN
    payload: Mapping[str, object] = field(default_factory=dict)
    event_id: str = ""
    recorded_at: str = field(default_factory=utc_now_iso)
    session_id: str = ""
    turn_id: str = ""
    step_id: str = ""
    task_id: str = ""
    branch_id: str = ""
    source_refs: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class HookActionResult:
    """Structured result of one hook action."""

    action_type: HookActionType
    status: HookActionStatus
    summary: str = ""
    patches: Mapping[str, object] = field(default_factory=dict)
    refs: tuple[str, ...] = field(default_factory=tuple)
    error: str = ""


@dataclass(frozen=True, slots=True)
class HookOutcome:
    """Combined result returned by HookManager.emit()."""

    directive: HookDirective = HookDirective.CONTINUE
    reason: str = ""
    patches: tuple[Mapping[str, object], ...] = field(default_factory=tuple)
    action_results: tuple[HookActionResult, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    refs: tuple[str, ...] = field(default_factory=tuple)

    def should_continue(self) -> bool:
        """Return whether the caller can continue normal execution."""
        return self.directive == HookDirective.CONTINUE


@dataclass(frozen=True, slots=True)
class HookDiagnostic:
    """Load, validation, or runtime diagnostic for hooks."""

    level: HookDiagnosticLevel
    message: str
    hook_id: str = ""
    event_name: str = ""
    source_layer: str = ""
    refs: tuple[str, ...] = field(default_factory=tuple)
    recorded_at: str = field(default_factory=utc_now_iso)

    def to_record(self) -> dict[str, object]:
        """Return a bounded JSON-compatible diagnostic record."""
        return {
            "recorded_at": self.recorded_at,
            "level": self.level.value,
            "hook_id": self.hook_id,
            "event_name": self.event_name,
            "source_layer": self.source_layer,
            "message": self.message,
            "refs": list(self.refs),
        }
