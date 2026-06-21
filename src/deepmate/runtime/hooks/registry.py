"""Hook registry and builtin hook definitions."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from deepmate.runtime.hooks.types import (
    HOOK_LAYER_ORDER,
    HookAction,
    HookActionType,
    HookDefinition,
    HookEvent,
    HookLayer,
    HookRunTarget,
)


BUILTIN_HOOK_SPECS: tuple[tuple[str, HookEvent, str], ...] = (
    (
        "context-invariants",
        HookEvent.PROVIDER_BEFORE_REQUEST,
        "Check activation and context invariant diagnostics before provider calls.",
    ),
    (
        "tool-access-gate",
        HookEvent.TOOL_BEFORE,
        "Record native, MCP, and subagent tool access policy decisions.",
    ),
    (
        "shell-safety-gate",
        HookEvent.SHELL_BEFORE,
        "Record shell safety, sandbox, and approval diagnostics.",
    ),
    (
        "mcp-schema-gate",
        HookEvent.MCP_BEFORE,
        "Record MCP schema-loaded repair diagnostics.",
    ),
    (
        "mcp-write-gate",
        HookEvent.MCP_BEFORE,
        "Record MCP read-only/write gate diagnostics.",
    ),
    (
        "workspace-write-checkpoint",
        HookEvent.WRITE_BEFORE,
        "Record workspace write checkpoint preimage diagnostics.",
    ),
    (
        "tool-output-compaction",
        HookEvent.TOOL_AFTER,
        "Record tool output compaction and retrieval-ref diagnostics.",
    ),
    (
        "tool-repair-signal",
        HookEvent.TOOL_AFTER,
        "Record malformed args, schema-not-loaded, and retry repair signals.",
    ),
    (
        "subagent-review-signal",
        HookEvent.SUBAGENT_AFTER_RUN,
        "Record child-run review and acceptance diagnostics.",
    ),
    (
        "checkpoint-memory-signal",
        HookEvent.CHECKPOINT_CREATED,
        "Record checkpoint memory patch and activity digest diagnostics.",
    ),
    (
        "maintenance-watermark",
        HookEvent.MAINTENANCE_AFTER_RUN,
        "Record memory, capability, and evolution maintenance watermarks.",
    ),
)


@dataclass(slots=True)
class HookRegistry:
    """Frozen collection of loaded hooks."""

    hooks: tuple[HookDefinition, ...] = field(default_factory=tuple)

    @classmethod
    def from_hooks(cls, hooks: Iterable[HookDefinition]) -> "HookRegistry":
        """Build a registry sorted by layer, priority, and file order."""
        return cls(tuple(sorted(hooks, key=_sort_key)))

    def hooks_for_event(self, event_name: HookEvent | str) -> tuple[HookDefinition, ...]:
        """Return enabled hooks for one event."""
        event = event_name if isinstance(event_name, HookEvent) else HookEvent(event_name)
        return tuple(
            hook for hook in self.hooks if hook.enabled and hook.event_name == event
        )

    def layer_counts(self) -> dict[str, int]:
        """Return loaded hook counts by layer."""
        counts = Counter(hook.layer.value for hook in self.hooks if hook.enabled)
        return {layer.value: counts.get(layer.value, 0) for layer in HOOK_LAYER_ORDER}

    def surface_tag(self) -> str:
        """Return a stable hash for the loaded hook surface."""
        payload = "\n".join(hook.stable_key() for hook in self.hooks if hook.enabled)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def builtin_hook_definitions() -> tuple[HookDefinition, ...]:
    """Return built-in hook definitions as non-side-effect trace hooks."""
    definitions: list[HookDefinition] = []
    for index, (hook_id, event_name, description) in enumerate(BUILTIN_HOOK_SPECS):
        definitions.append(
            HookDefinition(
                hook_id=hook_id,
                event_name=event_name,
                layer=HookLayer.BUILTIN,
                description=description,
                run_on=_builtin_run_target(event_name),
                actions=(HookAction(HookActionType.TRACE, {"summary": description}),),
                priority=1000,
                file_order=index,
            )
        )
    return tuple(definitions)


def _builtin_run_target(event_name: HookEvent) -> HookRunTarget:
    if event_name in {
        HookEvent.CHECKPOINT_CREATED,
        HookEvent.MEMORY_PATCH_APPLIED,
        HookEvent.MAINTENANCE_BEFORE_RUN,
        HookEvent.MAINTENANCE_AFTER_RUN,
    }:
        return HookRunTarget.MAINTENANCE
    return HookRunTarget.MAIN


def _sort_key(hook: HookDefinition) -> tuple[int, int, int, int, str]:
    try:
        layer_index = HOOK_LAYER_ORDER.index(hook.layer)
    except ValueError:
        layer_index = len(HOOK_LAYER_ORDER)
    return (
        layer_index,
        -hook.priority,
        hook.file_order,
        hook.hook_order,
        hook.hook_id,
    )
