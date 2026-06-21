"""Hook manager for deterministic runtime hook evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

from deepmate.runtime.hooks.matcher import hook_matches
from deepmate.runtime.hooks.registry import HookRegistry
from deepmate.runtime.hooks.signals import HookSignalStore
from deepmate.runtime.hooks.types import (
    HookAction,
    HookActionResult,
    HookActionStatus,
    HookActionType,
    HookDefinition,
    HookDiagnostic,
    HookDiagnosticLevel,
    HookDirective,
    HookEnvelope,
    HookOutcome,
)


@dataclass(slots=True)
class HookManager:
    """Evaluate loaded hooks and return deterministic outcomes.

    Safe runtime actions such as trace and signal recording are executed here.
    High-risk actions remain declarative-only until a dedicated guarded runtime
    path consumes them.
    """

    registry: HookRegistry
    signal_store: HookSignalStore | None = None
    diagnostics: list[HookDiagnostic] = field(default_factory=list)

    def emit(self, envelope: HookEnvelope) -> HookOutcome:
        """Evaluate hooks for one event and return a combined outcome."""
        action_results: list[HookActionResult] = []
        warnings: list[str] = []
        refs: list[str] = []
        directive = HookDirective.CONTINUE
        reason = ""
        for hook in self.registry.hooks_for_event(envelope.event_name):
            if not hook_matches(hook, envelope):
                continue
            refs.append(f"hook={hook.stable_key()}")
            for action in hook.actions:
                result = self._run_action(hook, action, envelope)
                action_results.append(result)
                if result.refs:
                    refs.extend(result.refs)
                if result.status == HookActionStatus.BLOCKED:
                    directive = HookDirective.BLOCK
                    reason = result.summary or f"blocked by hook {hook.hook_id}"
                    return HookOutcome(
                        directive=directive,
                        reason=reason,
                        action_results=tuple(action_results),
                        warnings=tuple(warnings),
                        refs=tuple(refs),
                    )
                if result.status == HookActionStatus.REQUIRES_APPROVAL:
                    directive = HookDirective.REQUIRES_APPROVAL
                    reason = result.summary or f"approval required by hook {hook.hook_id}"
                    return HookOutcome(
                        directive=directive,
                        reason=reason,
                        action_results=tuple(action_results),
                        warnings=tuple(warnings),
                        refs=tuple(refs),
                    )
        return HookOutcome(
            directive=directive,
            reason=reason,
            action_results=tuple(action_results),
            warnings=tuple(warnings),
            refs=tuple(refs),
        )

    def _run_action(
        self,
        hook: HookDefinition,
        action: HookAction,
        envelope: HookEnvelope,
    ) -> HookActionResult:
        action_type = action.action_type
        configured_reason = _text_param(action.params, "reason")
        if action_type == HookActionType.DENY:
            return HookActionResult(
                action_type=action_type,
                status=HookActionStatus.BLOCKED,
                summary=configured_reason or f"blocked by hook {hook.hook_id}",
                refs=(f"hook_id={hook.hook_id}",),
            )
        if action_type == HookActionType.ASK:
            return HookActionResult(
                action_type=action_type,
                status=HookActionStatus.REQUIRES_APPROVAL,
                summary=configured_reason or f"approval required by hook {hook.hook_id}",
                refs=(f"hook_id={hook.hook_id}",),
            )
        if action_type in {
            HookActionType.RECORD_MEMORY_SIGNAL,
            HookActionType.RECORD_EVOLUTION_SIGNAL,
        }:
            return self._record_signal_action(hook, action, envelope)
        if action_type != HookActionType.TRACE:
            return HookActionResult(
                action_type=action_type,
                status=HookActionStatus.SKIPPED,
                summary=(
                    f"hook action {action_type.value} matched but is not connected "
                    "to runtime side effects yet"
                ),
                refs=(f"hook_id={hook.hook_id}",),
            )
        configured_summary = _text_param(action.params, "summary")
        return HookActionResult(
            action_type=action_type,
            status=HookActionStatus.APPLIED,
            summary=(
                configured_summary
                or f"trace action matched for hook {hook.hook_id}"
            ),
            refs=(f"hook_id={hook.hook_id}",),
        )

    def _record_signal_action(
        self,
        hook: HookDefinition,
        action: HookAction,
        envelope: HookEnvelope,
    ) -> HookActionResult:
        signal_type = (
            "memory"
            if action.action_type == HookActionType.RECORD_MEMORY_SIGNAL
            else "evolution"
        )
        if self.signal_store is None:
            return HookActionResult(
                action_type=action.action_type,
                status=HookActionStatus.SKIPPED,
                summary="hook signal store is not configured",
                refs=(f"hook_id={hook.hook_id}",),
            )
        configured_summary = _text_param(action.params, "summary")
        summary = configured_summary or _text_param(envelope.payload, "summary")
        if not summary:
            summary = f"{signal_type} signal from {envelope.event_name.value}"
        signal_kind = _text_param(action.params, "signal_kind") or hook.hook_id
        refs = (
            *_refs_param(action.params, "refs"),
            *envelope.source_refs,
        )
        try:
            record = self.signal_store.append_from_envelope(
                signal_type=signal_type,
                signal_kind=signal_kind,
                summary=summary,
                refs=refs,
                hook_id=hook.hook_id,
                source_layer=hook.layer.value,
                envelope=envelope,
            )
        except Exception as exc:
            self.record_warning(f"hook signal write failed: {exc}", hook)
            return HookActionResult(
                action_type=action.action_type,
                status=HookActionStatus.FAILED,
                summary=f"hook signal write failed: {exc}",
                refs=(f"hook_id={hook.hook_id}",),
                error=str(exc),
            )
        return HookActionResult(
            action_type=action.action_type,
            status=HookActionStatus.APPLIED,
            summary=f"{signal_type} signal recorded: {record.signal_kind}",
            refs=(
                f"hook_id={hook.hook_id}",
                f"hook_signal_id={record.signal_id}",
                f"hook_signal_type={record.signal_type}",
            ),
        )

    def record_warning(
        self,
        message: str,
        hook: HookDefinition | None = None,
    ) -> None:
        """Record a runtime warning diagnostic."""
        self.diagnostics.append(
            HookDiagnostic(
                level=HookDiagnosticLevel.WARNING,
                message=message,
                hook_id=hook.hook_id if hook else "",
                event_name=hook.event_name.value if hook else "",
                source_layer=hook.layer.value if hook else "",
            )
        )


@dataclass(slots=True)
class HookRuntimeContext:
    """Stable hook manager handle plus surface refs for one runtime activation."""

    manager: HookManager
    surface_tag: str = ""

    @classmethod
    def from_registry(
        cls,
        registry: HookRegistry,
        signal_store: HookSignalStore | None = None,
    ) -> "HookRuntimeContext":
        """Build a runtime hook context from a frozen registry."""
        return cls(
            manager=HookManager(registry, signal_store=signal_store),
            surface_tag=registry.surface_tag(),
        )

    def emit(self, envelope: HookEnvelope) -> HookOutcome:
        """Emit one hook event through the contained manager."""
        return self.manager.emit(envelope)

    def reload_registry(self, registry: HookRegistry) -> None:
        """Replace the active registry while preserving shared sinks."""
        signal_store = self.manager.signal_store
        self.manager = HookManager(registry, signal_store=signal_store)
        self.surface_tag = registry.surface_tag()

    def trace_refs(self) -> tuple[str, ...]:
        """Return stable refs for runtime traces."""
        return (f"hook_surface_tag={self.surface_tag}",) if self.surface_tag else ()


def _text_param(params: object, key: str) -> str:
    if not isinstance(params, dict):
        return ""
    value = params.get(key)
    return value.strip() if isinstance(value, str) else ""


def _refs_param(params: object, key: str) -> tuple[str, ...]:
    if not isinstance(params, dict):
        return ()
    value = params.get(key)
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, (list, tuple, set)):
        return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    return ()
