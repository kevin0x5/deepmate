"""Runtime hook kernel exports."""

from deepmate.runtime.hooks.diagnostics import (
    HookDiagnosticStore,
    format_hook_validation,
    format_hooks_status,
)
from deepmate.runtime.hooks.loader import (
    HookLoadReport,
    HookSourceFile,
    discover_hook_sources,
    load_hook_report,
)
from deepmate.runtime.hooks.manager import HookManager
from deepmate.runtime.hooks.manager import HookRuntimeContext
from deepmate.runtime.hooks.matcher import hook_matches
from deepmate.runtime.hooks.registry import (
    HookRegistry,
    builtin_hook_definitions,
)
from deepmate.runtime.hooks.signals import HookSignalRecord, HookSignalStore
from deepmate.runtime.hooks.trust import HookTrustStore, TrustedWorkspace, workspace_hash
from deepmate.runtime.hooks.types import (
    HookAction,
    HookActionResult,
    HookActionStatus,
    HookActionType,
    HookActor,
    HookDefinition,
    HookDiagnostic,
    HookDiagnosticLevel,
    HookDirective,
    HookEnvelope,
    HookErrorPolicy,
    HookEvent,
    HookLayer,
    HookLoadOptions,
    HookOutcome,
    HookRunTarget,
)

__all__ = [
    "HookAction",
    "HookActionResult",
    "HookActionStatus",
    "HookActionType",
    "HookActor",
    "HookDefinition",
    "HookDiagnostic",
    "HookDiagnosticLevel",
    "HookDiagnosticStore",
    "HookDirective",
    "HookEnvelope",
    "HookErrorPolicy",
    "HookEvent",
    "HookLayer",
    "HookLoadOptions",
    "HookLoadReport",
    "HookManager",
    "HookOutcome",
    "HookRegistry",
    "HookRunTarget",
    "HookRuntimeContext",
    "HookSignalRecord",
    "HookSignalStore",
    "HookSourceFile",
    "HookTrustStore",
    "TrustedWorkspace",
    "builtin_hook_definitions",
    "discover_hook_sources",
    "format_hook_validation",
    "format_hooks_status",
    "hook_matches",
    "load_hook_report",
    "workspace_hash",
]
