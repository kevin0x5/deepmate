"""Sandboxed shell command native tool."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from deepmate.runtime.sandbox import SandboxMode, SandboxPolicy, SandboxRunner
from deepmate.runtime.hooks import (
    HookActor,
    HookDirective,
    HookEnvelope,
    HookEvent,
    HookOutcome,
    HookRuntimeContext,
)
from deepmate.runtime.safety import (
    SessionApprovalCache,
    ToolSafetyPolicy,
)
from deepmate.tools.registry import NativeTool, NativeToolResult

RUN_SHELL_COMMAND_TOOL_NAME = "run_shell_command"
DEFAULT_TIMEOUT_SECONDS = 120
MAX_TIMEOUT_SECONDS = 600


def shell_tools(
    workspace: str | Path,
    *,
    shell_enabled: bool,
    network_enabled: bool,
    env_change_enabled: bool = False,
    sandbox_mode: SandboxMode = SandboxMode.AUTO,
    approval_cache: SessionApprovalCache | None = None,
    runner: SandboxRunner | None = None,
    hook_context: HookRuntimeContext | None = None,
) -> tuple[NativeTool, ...]:
    """Return shell tool definitions for one runtime invocation."""
    root = Path(workspace).resolve()
    policy = ToolSafetyPolicy(
        workspace=root,
        shell_enabled=shell_enabled,
        network_enabled=network_enabled,
        env_change_enabled=env_change_enabled,
        approval_cache=approval_cache,
    )
    sandbox_runner = runner or SandboxRunner()
    return (
        NativeTool(
            name=RUN_SHELL_COMMAND_TOOL_NAME,
            description=(
                "Run a workspace-scoped shell command for testing, building, or "
                "small verification. Commands are policy-checked, sandboxed when "
                "available, network-off by default, and bounded by timeout."
            ),
            input_schema=_schema(),
            handler=lambda arguments: _run_shell_command(
                arguments,
                workspace=root,
                safety_policy=policy,
                sandbox_mode=sandbox_mode,
                runner=sandbox_runner,
                hook_context=hook_context,
            ),
            read_only=False,
            requires_shell=True,
        ),
    )


def _run_shell_command(
    arguments: Mapping[str, object],
    *,
    workspace: Path,
    safety_policy: ToolSafetyPolicy,
    sandbox_mode: SandboxMode,
    runner: SandboxRunner,
    hook_context: HookRuntimeContext | None = None,
) -> NativeToolResult:
    command = _text_argument(arguments, "command")
    cwd = _optional_text_argument(arguments, "cwd") or "."
    timeout_seconds = _int_argument(
        arguments,
        "timeout_seconds",
        DEFAULT_TIMEOUT_SECONDS,
        1,
        MAX_TIMEOUT_SECONDS,
    )
    network = _optional_text_argument(arguments, "network") or "off"
    decision = safety_policy.check_shell_command(command, cwd=cwd, network=network)
    before_outcome = _emit_shell_hook(
        hook_context,
        HookEvent.SHELL_BEFORE,
        payload={
            "tool_name": RUN_SHELL_COMMAND_TOOL_NAME,
            "tool_source": "native",
            "command_preview": command[:200],
            "cwd": cwd,
            "network": network.strip().lower(),
            "sandbox_mode": sandbox_mode.value,
            "risk_level": decision.risk_level.value,
            "status": "before",
            "actor": HookActor.MAIN.value,
        },
    )
    if before_outcome.directive != HookDirective.CONTINUE:
        raise ValueError(
            before_outcome.reason
            or f"Shell command stopped by hook: {before_outcome.directive.value}"
        )
    if not decision.allowed:
        raise ValueError(_decision_message(decision))
    cwd_path = Path(cwd)
    resolved_cwd = cwd_path if cwd_path.is_absolute() else workspace / cwd_path
    run_result = runner.run(
        command,
        SandboxPolicy(
            workspace=workspace,
            cwd=resolved_cwd,
            network_enabled=network.strip().lower() == "on",
            mode=sandbox_mode,
        ),
        timeout_seconds=timeout_seconds,
    )
    after_outcome = _emit_shell_hook(
        hook_context,
        HookEvent.SHELL_AFTER,
        payload={
            "tool_name": RUN_SHELL_COMMAND_TOOL_NAME,
            "tool_source": "native",
            "command_preview": command[:200],
            "cwd": cwd,
            "network": network.strip().lower(),
            "sandbox_mode": sandbox_mode.value,
            "risk_level": decision.risk_level.value,
            "status": "completed" if run_result.exit_code == 0 else "failed",
            "actor": HookActor.MAIN.value,
        },
    )
    return NativeToolResult(
        content=_shell_result_content(run_result),
        data={
            "exit_code": run_result.exit_code,
            "backend": run_result.backend,
            "sandboxed": run_result.sandboxed,
            "stdout_chars": len(run_result.stdout),
            "stderr_chars": len(run_result.stderr),
        },
        refs=(
            f"shell_exit_code={run_result.exit_code}",
            f"shell_backend={run_result.backend}",
            f"shell_sandboxed={str(run_result.sandboxed).lower()}",
            f"shell_risk={decision.risk_level.value}",
            *(
                (
                    f"hook_directive={after_outcome.directive.value}",
                    *after_outcome.refs,
                )
                if after_outcome.refs
                else ()
            ),
            *(("requires_approval=false",) if decision.approval_key else ()),
            *run_result.refs,
        ),
    )


def _shell_result_content(run_result) -> str:
    output = run_result.output_text()
    if run_result.backend == "permission-only" and not run_result.sandboxed:
        warning = (
            "Warning: OS sandbox backend is unavailable; command ran with "
            "permission-only enforcement."
        )
        return f"{warning}\n\n{output}" if output else warning
    return output


def _decision_message(decision) -> str:
    refs = ", ".join(decision.refs)
    return f"{decision.reason} {refs}".strip()


def _emit_shell_hook(
    hook_context: HookRuntimeContext | None,
    event_name: HookEvent,
    *,
    payload: Mapping[str, object],
) -> HookOutcome:
    if hook_context is None:
        return HookOutcome()
    return hook_context.emit(
        HookEnvelope(
            event_name=event_name,
            actor=HookActor.MAIN,
            payload=payload,
            source_refs=hook_context.trace_refs(),
        )
    )


def _schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to run inside the current workspace.",
            },
            "cwd": {
                "type": "string",
                "description": "Workspace-relative working directory. Defaults to '.'.",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Command timeout in seconds. Defaults to 120.",
            },
            "network": {
                "type": "string",
                "enum": ["off", "on"],
                "description": "Network mode for this command. Defaults to off.",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }


def _text_argument(arguments: Mapping[str, object], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_text_argument(arguments: Mapping[str, object], key: str) -> str:
    value = arguments.get(key)
    return value.strip() if isinstance(value, str) else ""


def _int_argument(
    arguments: Mapping[str, object],
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = arguments.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    if isinstance(value, str):
        clean = value.strip()
        try:
            value = int(clean)
        except ValueError as exc:
            raise ValueError(f"{key} must be an integer") from exc
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value
