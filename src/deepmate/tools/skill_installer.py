"""Native tools for skill source inspection and installation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from deepmate.capabilities.state import CapabilityStateStore
from deepmate.foundation import display_path
from deepmate.runtime.sandbox import SandboxMode, SandboxPolicy, SandboxRunner
from deepmate.runtime.safety import (
    SafetyRiskLevel,
    SessionApprovalCache,
    ToolSafetyPolicy,
)
from deepmate.skills import (
    InstalledSkillRecord,
    InstalledSkillManifestStore,
    format_skill_bundle_install_result,
    format_skill_inspection,
    format_skill_install_result,
    format_skill_verify_result,
    inspect_skill_source,
    install_skill_bundle,
    install_skill_source,
    verify_skill_install,
)
from deepmate.skills.install import _setup_command
from deepmate.tools.registry import NativeTool, NativeToolResult

INSPECT_SKILL_SOURCE_TOOL_NAME = "inspect_skill_source"
LOAD_SKILL_INSTALLER_TOOLS_NAME = "load_skill_installer_tools"
INSTALL_SKILL_FROM_REQUEST_TOOL_NAME = "install_skill_from_request"
INSTALL_SKILL_BUNDLE_TOOL_NAME = "install_skill_bundle"
INSTALL_SKILL_TOOL_NAME = "install_skill"
VERIFY_SKILL_INSTALL_TOOL_NAME = "verify_skill_install"
PLAN_SKILL_SETUP_TOOL_NAME = "plan_skill_setup"
RUN_SKILL_SETUP_TOOL_NAME = "run_skill_setup"
DEFAULT_SETUP_TIMEOUT_SECONDS = 120
MAX_SETUP_TIMEOUT_SECONDS = 600


def skill_installer_tools(
    workspace: str | Path,
    data_dir: str | Path,
    state_store: CapabilityStateStore,
    *,
    shell_enabled: bool = False,
    network_enabled: bool = False,
    env_change_enabled: bool = False,
    sandbox_mode: SandboxMode = SandboxMode.AUTO,
    approval_cache: SessionApprovalCache | None = None,
    runner: SandboxRunner | None = None,
) -> tuple[NativeTool, ...]:
    """Return native tools for target-oriented skill installation."""
    root = Path(workspace).resolve()
    data_root = Path(data_dir).resolve()
    manifest_store = InstalledSkillManifestStore.in_data_dir(data_root)
    safety_policy = ToolSafetyPolicy(
        workspace=root,
        shell_enabled=shell_enabled,
        network_enabled=network_enabled,
        env_change_enabled=env_change_enabled,
        approval_cache=approval_cache,
    )
    sandbox_runner = runner or SandboxRunner()
    natural_language_tool = NativeTool(
        name=INSTALL_SKILL_FROM_REQUEST_TOOL_NAME,
        description=(
            "Install a community SKILL.md bundle from a natural-language request. "
            "Use this directly when the user asks to install, import, add, or set up "
            "a skill from a URL, GitHub repo/path, ClawHub/SkillHub/OpenClaw page, "
            "local folder/archive, or copied install instruction. It performs "
            "source resolution, install, verification, and setup planning in one "
            "call; dependency setup remains approval-gated."
        ),
        input_schema=_install_from_request_schema(),
        handler=lambda arguments: _install_skill_from_request_tool(
            arguments,
            workspace=root,
            data_dir=data_root,
            state_store=state_store,
            manifest_store=manifest_store,
            safety_policy=safety_policy,
            sandbox_mode=sandbox_mode,
            runner=sandbox_runner,
        ),
        read_only=False,
    )
    concrete_tools = (
        NativeTool(
            name=INSPECT_SKILL_SOURCE_TOOL_NAME,
            description=(
                "Inspect a local path, archive, GitHub URL, remote skill page, or "
                "skill installation instruction for a standard SKILL.md bundle before "
                "installing it."
            ),
            input_schema=_inspect_schema(),
            handler=lambda arguments: _inspect_skill_source_tool(
                arguments,
                workspace=root,
            ),
            read_only=True,
            exposed_by_default=False,
        ),
        NativeTool(
            name=INSTALL_SKILL_BUNDLE_TOOL_NAME,
            description=(
                "Install a community skill from a local path, archive, GitHub URL, "
                "SkillHub/ClawHub/OpenClaw page, or Claude Code-style SKILL.md bundle. "
                "This performs inspect, install, verify, and setup planning in one "
                "step. It installs the skill files but does not execute dependency "
                "setup scripts without separate approval."
            ),
            input_schema=_install_schema(),
            handler=lambda arguments: _install_skill_bundle_tool(
                arguments,
                workspace=root,
                data_dir=data_root,
                state_store=state_store,
                manifest_store=manifest_store,
                network_enabled=network_enabled,
            ),
            read_only=False,
            exposed_by_default=False,
        ),
        NativeTool(
            name=INSTALL_SKILL_TOOL_NAME,
            description=(
                "Install a verified standard skill bundle into the workspace, preserving "
                "SKILL.md, references, scripts, assets, agents, and examples. Requires "
                "workspace write access."
            ),
            input_schema=_install_schema(),
            handler=lambda arguments: _install_skill_tool(
                arguments,
                workspace=root,
                data_dir=data_root,
                state_store=state_store,
                manifest_store=manifest_store,
                network_enabled=network_enabled,
            ),
            read_only=False,
            exposed_by_default=False,
        ),
        NativeTool(
            name=VERIFY_SKILL_INSTALL_TOOL_NAME,
            description=(
                "Verify that an installed or local skill can be discovered and its full "
                "SKILL.md can be loaded."
            ),
            input_schema=_verify_schema(),
            handler=lambda arguments: _verify_skill_tool(
                arguments,
                workspace=root,
                data_dir=data_root,
                state_store=state_store,
                manifest_store=manifest_store,
            ),
            read_only=True,
            exposed_by_default=False,
        ),
        NativeTool(
            name=PLAN_SKILL_SETUP_TOOL_NAME,
            description=(
                "Create a conservative setup plan for an installed skill without "
                "executing it. Use run_skill_setup after shell/environment "
                "permissions are enabled."
            ),
            input_schema=_plan_setup_schema(),
            handler=lambda arguments: _plan_skill_setup_tool(
                arguments,
                workspace=root,
                data_dir=data_root,
                manifest_store=manifest_store,
            ),
            read_only=True,
            exposed_by_default=False,
        ),
        NativeTool(
            name=RUN_SKILL_SETUP_TOOL_NAME,
            description=(
                "Run the recorded setup command for one installed skill inside "
                "its skill directory. The command is policy-checked, sandboxed "
                "when available, network-off by default, and updates the "
                "installed skill manifest setup status."
            ),
            input_schema=_run_setup_schema(),
            handler=lambda arguments: _run_skill_setup_tool(
                arguments,
                workspace=root,
                data_dir=data_root,
                manifest_store=manifest_store,
                safety_policy=safety_policy,
                sandbox_mode=sandbox_mode,
                runner=sandbox_runner,
            ),
            read_only=False,
            requires_shell=True,
            exposed_by_default=False,
        ),
    )
    return (
        natural_language_tool,
        NativeTool(
            name=LOAD_SKILL_INSTALLER_TOOLS_NAME,
            description=(
                "Load skill installer schemas when the user wants to inspect, install, "
                "import, verify, or set up a community SKILL.md bundle from a local "
                "path, archive, GitHub URL, SkillHub/ClawHub/OpenClaw page, or "
                "natural-language skill install request. Use this before attempting "
                "skill installation from a natural-language request."
            ),
            input_schema=_load_skill_installer_schema(),
            handler=lambda arguments: _load_skill_installer_tools(
                arguments,
                concrete_tools=concrete_tools,
            ),
            read_only=True,
        ),
        *concrete_tools,
    )


def _install_skill_from_request_tool(
    arguments: Mapping[str, object],
    *,
    workspace: Path,
    data_dir: Path,
    state_store: CapabilityStateStore,
    manifest_store: InstalledSkillManifestStore,
    safety_policy: ToolSafetyPolicy,
    sandbox_mode: SandboxMode,
    runner: SandboxRunner,
) -> NativeToolResult:
    request = _text_argument(arguments, "request")
    source = _optional_text_argument(arguments, "source") or _source_from_request(
        request,
        workspace,
    )
    if not source:
        raise ValueError(_missing_source_message(request))
    _require_network_for_remote_source(
        source, network_enabled=safety_policy.network_enabled
    )
    skill_name = _optional_text_argument(arguments, "skill_name")
    target = _optional_text_argument(arguments, "target") or _default_install_target(source)
    force = _bool_argument(arguments, "force")
    setup = _optional_text_argument(arguments, "setup") or "plan"
    network = _optional_text_argument(arguments, "network") or "off"

    install_result = install_skill_bundle(
        source,
        workspace,
        data_dir,
        state_store,
        target=target,
        skill_name=skill_name,
        force=force,
        manifest_store=manifest_store,
    )
    setup_result_content = ""
    setup_error = ""
    if install_result.setup_command and setup in {"run", "auto"}:
        try:
            setup_run = _run_skill_setup_tool(
                {
                    "name": install_result.install.skill.name,
                    "network": network,
                },
                workspace=workspace,
                data_dir=data_dir,
                manifest_store=manifest_store,
                safety_policy=safety_policy,
                sandbox_mode=sandbox_mode,
                runner=runner,
            )
            setup_result_content = setup_run.content
        except Exception as exc:
            setup_error = str(exc).strip() or "setup could not run"

    content = _natural_install_content(
        install_result,
        workspace,
        source=source,
        setup=setup,
        setup_result_content=setup_result_content,
        setup_error=setup_error,
    )
    refs = [
        f"skill={install_result.install.skill.name}",
        f"source={source}",
        f"source_kind={install_result.install.manifest_record.source_kind}",
        f"scope={install_result.install.manifest_record.target_scope or 'workspace'}",
        f"target={install_result.install.manifest_record.target_path}",
        f"verify_status={install_result.verify.status}",
        f"setup_status={install_result.setup_status}",
        f"setup_mode={setup}",
    ]
    if setup_result_content:
        refs.append("setup_run=completed")
    if setup_error:
        refs.append("setup_run=needs_approval_or_attention")
    if install_result.setup_command:
        refs.append("setup_command_present=true")
    return NativeToolResult(
        content=content,
        data={
            "request": request,
            "source": source,
            "install": install_result.to_record(),
            "setup_mode": setup,
            "setup_ran": bool(setup_result_content),
            "setup_error": setup_error,
        },
        refs=tuple(refs),
    )


def _load_skill_installer_tools(
    arguments: Mapping[str, object],
    *,
    concrete_tools: tuple[NativeTool, ...],
) -> NativeToolResult:
    reason = _optional_text_argument(arguments, "reason")
    schemas = tuple(tool.schema() for tool in concrete_tools)
    names = tuple(str(schema["name"]) for schema in schemas)
    lines = [
        "Skill installer tools loaded for the next model step.",
        f"- tools: {', '.join(names)}",
        "- use install_skill_bundle for the normal install -> verify -> setup-plan flow",
        "- use inspect_skill_source first when the source is ambiguous or user asks to review",
        "- dependency setup is not run automatically; use run_skill_setup only after approval",
    ]
    if reason:
        lines.insert(1, f"- reason: {reason}")
    return NativeToolResult(
        content="\n".join(lines),
        data={
            "tools": names,
            "schema_count": len(schemas),
            "reason": reason,
        },
        refs=(
            "skill_installer_tools_loaded=true",
            f"skill_installer_schema_count={len(schemas)}",
            *(f"skill_installer_schema={name}" for name in names),
        ),
        schema_additions=schemas,
    )


def _inspect_skill_source_tool(
    arguments: Mapping[str, object],
    *,
    workspace: Path,
) -> NativeToolResult:
    source = _text_argument(arguments, "source")
    skill_name = _optional_text_argument(arguments, "skill_name")
    result = inspect_skill_source(source, workspace, skill_name=skill_name)
    return NativeToolResult(
        content=format_skill_inspection(result, workspace),
        data=result.to_record(),
        refs=(
            f"source_kind={result.source_kind}",
            f"compatibility={result.compatibility}",
            f"candidates={len(result.candidates)}",
        ),
    )


def _install_skill_tool(
    arguments: Mapping[str, object],
    *,
    workspace: Path,
    data_dir: Path,
    state_store: CapabilityStateStore,
    manifest_store: InstalledSkillManifestStore,
    network_enabled: bool = False,
) -> NativeToolResult:
    source = _text_argument(arguments, "source")
    _require_network_for_remote_source(source, network_enabled=network_enabled)
    skill_name = _optional_text_argument(arguments, "skill_name")
    target = _optional_text_argument(arguments, "target") or _default_install_target(source)
    force = _bool_argument(arguments, "force")
    result = install_skill_source(
        source,
        workspace,
        data_dir,
        state_store,
        target=target,
        skill_name=skill_name,
        force=force,
        manifest_store=manifest_store,
    )
    return NativeToolResult(
        content=format_skill_install_result(result, workspace),
        data=result.to_record(),
        refs=(
            f"skill={result.skill.name}",
            f"scope={result.manifest_record.target_scope or 'workspace'}",
            f"target={result.manifest_record.target_path}",
            f"source_kind={result.manifest_record.source_kind}",
            f"temperature={result.state_temperature}",
        ),
    )


def _install_skill_bundle_tool(
    arguments: Mapping[str, object],
    *,
    workspace: Path,
    data_dir: Path,
    state_store: CapabilityStateStore,
    manifest_store: InstalledSkillManifestStore,
    network_enabled: bool = False,
) -> NativeToolResult:
    source = _text_argument(arguments, "source")
    _require_network_for_remote_source(source, network_enabled=network_enabled)
    skill_name = _optional_text_argument(arguments, "skill_name")
    target = _optional_text_argument(arguments, "target") or _default_install_target(source)
    force = _bool_argument(arguments, "force")
    result = install_skill_bundle(
        source,
        workspace,
        data_dir,
        state_store,
        target=target,
        skill_name=skill_name,
        force=force,
        manifest_store=manifest_store,
    )
    return NativeToolResult(
        content=format_skill_bundle_install_result(result, workspace),
        data=result.to_record(),
        refs=(
            f"skill={result.install.skill.name}",
            f"scope={result.install.manifest_record.target_scope or 'workspace'}",
            f"target={result.install.manifest_record.target_path}",
            f"source_kind={result.install.manifest_record.source_kind}",
            f"verify_status={result.verify.status}",
            f"setup_status={result.setup_status}",
            *(("setup_command_present=true",) if result.setup_command else ()),
        ),
    )


def _verify_skill_tool(
    arguments: Mapping[str, object],
    *,
    workspace: Path,
    data_dir: Path,
    state_store: CapabilityStateStore,
    manifest_store: InstalledSkillManifestStore,
) -> NativeToolResult:
    name = _text_argument(arguments, "name")
    result = verify_skill_install(
        name,
        workspace,
        data_dir,
        state_store,
        manifest_store=manifest_store,
    )
    return NativeToolResult(
        content=format_skill_verify_result(result, workspace),
        data=result.to_record(),
        refs=(
            f"skill={result.skill.name}",
            f"status={result.status}",
            f"path={result.skill_path}",
        ),
    )


def _plan_skill_setup_tool(
    arguments: Mapping[str, object],
    *,
    workspace: Path,
    data_dir: Path,
    manifest_store: InstalledSkillManifestStore,
) -> NativeToolResult:
    name = _text_argument(arguments, "name")
    record = manifest_store.get(name)
    if record is None:
        raise ValueError(f"installed skill not found: {name}")
    target = _record_target_path(workspace, data_dir, record)
    command = _setup_command(target)
    status = "not_required" if not command else "pending"
    updated = manifest_store.update_setup_status(
        record.name,
        status=status,
        command=command,
        updated_at=_now(),
    )
    content = _setup_plan_content(updated, target, command)
    return NativeToolResult(
        content=content,
        data={
            "skill": updated.name,
            "target_path": updated.target_path,
            "setup_status": updated.setup_status,
            "setup_command": updated.setup_command,
        },
        refs=(
            f"skill={updated.name}",
            f"setup_status={updated.setup_status}",
            f"target={updated.target_path}",
            *(("setup_command_present=true",) if command else ()),
        ),
    )


def _run_skill_setup_tool(
    arguments: Mapping[str, object],
    *,
    workspace: Path,
    data_dir: Path,
    manifest_store: InstalledSkillManifestStore,
    safety_policy: ToolSafetyPolicy,
    sandbox_mode: SandboxMode,
    runner: SandboxRunner,
) -> NativeToolResult:
    name = _text_argument(arguments, "name")
    timeout_seconds = _int_argument(
        arguments,
        "timeout_seconds",
        DEFAULT_SETUP_TIMEOUT_SECONDS,
        1,
        MAX_SETUP_TIMEOUT_SECONDS,
    )
    network = _optional_text_argument(arguments, "network") or "off"
    record = manifest_store.get(name)
    if record is None:
        raise ValueError(f"installed skill not found: {name}")
    target = _record_target_path(workspace, data_dir, record)
    command = record.setup_command.strip() or _setup_command(target)
    if not command:
        updated = manifest_store.update_setup_status(
            record.name,
            status="not_required",
            command="",
            updated_at=_now(),
        )
        return NativeToolResult(
            content=f"Skill setup not required: {updated.name}",
            data={
                "skill": updated.name,
                "target_path": updated.target_path,
                "setup_status": updated.setup_status,
                "setup_command": updated.setup_command,
                "exit_code": 0,
            },
            refs=(
                f"skill={updated.name}",
                "setup_status=not_required",
                f"target={updated.target_path}",
            ),
        )
    if command.lstrip().startswith("#"):
        updated = manifest_store.update_setup_status(
            record.name,
            status="pending",
            command=command,
            updated_at=_now(),
        )
        raise ValueError(
            "skill setup command requires manual inspection before execution: "
            f"{updated.setup_command}"
        )
    execution_root = _record_execution_root(workspace, data_dir, record)
    scoped_safety_policy = (
        safety_policy
        if execution_root == workspace
        else ToolSafetyPolicy(
            workspace=execution_root,
            shell_enabled=safety_policy.shell_enabled,
            network_enabled=safety_policy.network_enabled,
            env_change_enabled=safety_policy.env_change_enabled,
            approval_cache=safety_policy.approval_cache,
        )
    )
    decision = scoped_safety_policy.check_shell_command(command, cwd=target, network=network)
    if (
        not decision.allowed
        and decision.requires_approval
        and decision.risk_level == SafetyRiskLevel.MEDIUM
        and scoped_safety_policy.shell_enabled
        and (network.strip().lower() or "off") == "off"
    ):
        decision = type(decision)(
            allowed=True,
            requires_approval=False,
            requires_sandbox=decision.requires_sandbox,
            risk_level=decision.risk_level,
            approval_key=decision.approval_key,
            reason="Skill setup command allowed after shell tool approval.",
            refs=decision.refs,
        )
    if not decision.allowed:
        raise ValueError(_decision_message(decision))
    run_result = runner.run(
        command,
        SandboxPolicy(
            workspace=execution_root,
            cwd=target,
            network_enabled=network.strip().lower() == "on",
            mode=sandbox_mode,
        ),
        timeout_seconds=timeout_seconds,
    )
    status = "completed" if run_result.exit_code == 0 else "failed"
    updated = manifest_store.update_setup_status(
        record.name,
        status=status,
        command=command,
        updated_at=_now(),
    )
    return NativeToolResult(
        content=_setup_run_content(updated, target, command, run_result),
        data={
            "skill": updated.name,
            "target_path": updated.target_path,
            "setup_status": updated.setup_status,
            "setup_command": updated.setup_command,
            "exit_code": run_result.exit_code,
            "backend": run_result.backend,
            "sandboxed": run_result.sandboxed,
            "stdout_chars": len(run_result.stdout),
            "stderr_chars": len(run_result.stderr),
        },
        refs=(
            f"skill={updated.name}",
            f"setup_status={updated.setup_status}",
            f"setup_exit_code={run_result.exit_code}",
            f"setup_backend={run_result.backend}",
            f"setup_sandboxed={str(run_result.sandboxed).lower()}",
            f"target={updated.target_path}",
            *run_result.refs,
        ),
    )


def _inspect_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Skill source path, archive path, URL, GitHub URL, remote skill page, or install instruction.",
            },
            "skill_name": {
                "type": "string",
                "description": "Optional skill name when the source contains multiple SKILL.md bundles.",
            },
        },
        "required": ["source"],
        "additionalProperties": False,
    }


def _load_skill_installer_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Short reason why community skill installation tools are needed for this task.",
            },
        },
        "additionalProperties": False,
    }


def _install_from_request_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": "The user's natural-language skill install request, preserved verbatim.",
            },
            "source": {
                "type": "string",
                "description": "Optional explicit source if already known: local path, archive, URL, GitHub repo/path, ClawHub/SkillHub/OpenClaw page, or skill install instruction.",
            },
            "skill_name": {
                "type": "string",
                "description": "Optional skill name when the source contains multiple SKILL.md bundles.",
            },
            "target": {
                "type": "string",
                "description": "Install target: user/global for a cross-workspace skill, workspace for project-local skill, or a workspace-relative directory. Defaults to user for remote sources and workspace for local sources.",
            },
            "setup": {
                "type": "string",
                "enum": ["plan", "run", "auto"],
                "description": "Dependency setup mode. Defaults to plan (install and verify only). Pass run to attempt setup through the approval-gated shell policy. Set this explicitly; it is not inferred from the request text.",
            },
            "network": {
                "type": "string",
                "enum": ["off", "on"],
                "description": "Network mode for dependency setup, used only when setup is run. Defaults to off. Set this explicitly; it is not inferred from the request text.",
            },
            "force": {
                "type": "boolean",
                "description": "Overwrite an existing destination after backing it up.",
            },
        },
        "required": ["request"],
        "additionalProperties": False,
    }


def _install_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Skill source path, archive path, URL, GitHub URL, remote skill page, or install instruction.",
            },
            "skill_name": {
                "type": "string",
                "description": "Optional skill name when the source contains multiple SKILL.md bundles.",
            },
            "target": {
                "type": "string",
                "description": "Install target: user/global for a cross-workspace user skill, workspace for project-local skill, or a workspace-relative directory. Defaults to user for remote sources and workspace for local sources.",
            },
            "force": {
                "type": "boolean",
                "description": "Overwrite an existing destination after backing it up.",
            },
        },
        "required": ["source"],
        "additionalProperties": False,
    }


def _verify_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Skill name or skill directory name.",
            }
        },
        "required": ["name"],
        "additionalProperties": False,
    }


def _plan_setup_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Installed skill name or skill directory name.",
            }
        },
        "required": ["name"],
        "additionalProperties": False,
    }


def _run_setup_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Installed skill name or skill directory name.",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Setup command timeout in seconds. Defaults to 120.",
            },
            "network": {
                "type": "string",
                "enum": ["off", "on"],
                "description": "Network mode for setup. Defaults to off.",
            },
        },
        "required": ["name"],
        "additionalProperties": False,
    }


def _default_install_target(source: str) -> str:
    clean = source.strip().lower()
    if clean.startswith(("http://", "https://", "github:", "gh:")):
        return "user"
    if "github.com" in clean or "skillhub" in clean or "clawhub" in clean:
        return "user"
    return "workspace"


def _source_is_remote(source: str) -> bool:
    """Whether installing this source needs an outbound network fetch."""
    clean = source.strip().lower()
    return clean.startswith(("http://", "https://", "github:", "gh:")) or any(
        host in clean for host in ("github.com", "skillhub", "clawhub", "openclaw")
    )


def _require_network_for_remote_source(source: str, *, network_enabled: bool) -> None:
    """Block remote skill downloads unless network access is enabled."""
    if network_enabled or not _source_is_remote(source):
        return
    raise ValueError(
        "Installing this skill requires downloading from the network, which is "
        "off. Re-run with network access enabled (e.g. --allow-network), or "
        "install from a local path or archive instead. Source: " + source.strip()
    )


def _source_from_request(request: str, workspace: Path) -> str:
    clean = request.strip()
    url = _first_request_url(clean)
    if url:
        return url
    install_instruction = _install_instruction_from_request(clean)
    if install_instruction:
        return install_instruction
    path = _path_from_request(clean, workspace)
    if path:
        return path
    github_ref = _github_ref_from_request(clean)
    if github_ref:
        return github_ref
    return ""


def _first_request_url(value: str) -> str:
    match = re.search(r"https?://[^\s'\"<>）)]+", value)
    return match.group(0).rstrip(".,;") if match else ""


def _github_ref_from_request(value: str) -> str:
    """Resolve a GitHub repo only from an explicit ``gh:``/``github:`` shorthand.

    Bare ``owner/repo`` text is intentionally NOT matched: ordinary phrases like
    "and/or" or "product/roadmap" would otherwise be treated as repositories and
    trigger a network download. Full ``https://github.com/...`` URLs are handled
    earlier by ``_first_request_url``; this only covers the typed shorthand.
    """
    match = re.search(
        r"(?i)(?:gh|github):\s*"
        r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/(?:tree|blob)/[^\s'\"<>）)]+|/[^\s'\"<>）)]+)?)",
        value,
    )
    if not match:
        return ""
    candidate = match.group(1).strip().rstrip(".,;")
    owner, repo, *_rest = candidate.split("/")
    if not owner or not repo:
        return ""
    if not re.match(r"^[A-Za-z0-9_.-]+$", owner) or not re.match(
        r"^[A-Za-z0-9_.-]+$",
        repo,
    ):
        return ""
    return f"https://github.com/{candidate}"


def _install_instruction_from_request(value: str) -> str:
    match = re.search(
        r"(?i)(?:^|[\s:：])((?:skill|claude|codex)\s+install\s+[^\n\r]+)",
        value,
    )
    return " ".join(match.group(1).split()).strip() if match else ""


def _path_from_request(value: str, workspace: Path) -> str:
    candidates: list[str] = []
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", value)
    candidates.extend(quoted)
    candidates.extend(
        match.rstrip(".,;")
        for match in re.findall(
            r"(?:(?<=\s)|(?<=[:：]))((?:~|/|\.{1,2}/)[^\s'\"<>）)]+)",
            value,
        )
    )
    candidates.extend(
        token.strip(".,;")
        for token in value.split()
        if token.strip(".,;").startswith(("./", "../", "~/", "/"))
        or token.strip(".,;").endswith((".zip", ".tar", ".tar.gz", ".tgz"))
    )
    for candidate in candidates:
        path = Path(candidate).expanduser()
        resolved = path if path.is_absolute() else workspace / path
        if resolved.exists():
            return candidate
    return ""


def _missing_source_message(request: str) -> str:
    preview = " ".join(request.split())[:240]
    return (
        "I can install SKILL.md bundles from a URL, GitHub repo/path, "
        "ClawHub/SkillHub/OpenClaw page, local folder, or zip/tar archive, "
        "but this request did not include a concrete source. "
        f"Request: {preview}"
    )


def _natural_install_content(
    result,
    workspace: Path,
    *,
    source: str,
    setup: str,
    setup_result_content: str,
    setup_error: str,
) -> str:
    install = result.install
    verify = result.verify
    lines = [
        f"Skill installed: {install.skill.name}",
        "",
        f"- source: {source}",
        f"- target: {display_path(install.target_path, workspace)}",
        f"- scope: {install.manifest_record.target_scope or 'workspace'}",
        f"- verified: {verify.status}",
        "- resources: "
        f"references={verify.resources.references}, "
        f"scripts={verify.resources.scripts}, "
        f"assets={verify.resources.assets}, "
        f"agents={verify.resources.agents}, "
        f"examples={verify.resources.examples}",
    ]
    if result.setup_command:
        lines.append(f"- setup: {result.setup_status}")
        lines.append(f"- setup command: {result.setup_command}")
        if setup_result_content:
            lines.append("- dependency setup: completed")
        elif setup_error:
            lines.append("- dependency setup: needs approval or manual attention")
            lines.append(f"- setup note: {setup_error}")
        elif setup == "plan":
            lines.append(
                "- dependency setup: not run; approve setup if this skill needs dependencies"
            )
    else:
        lines.append("- setup: not required")
    lines.append("")
    if result.setup_command and not setup_result_content:
        lines.append(
            f"Next: approve running setup for {install.skill.name} if you want dependencies installed now."
        )
    else:
        lines.append("Ready: the skill is installed and discoverable.")
    warnings = tuple(dict.fromkeys((*install.warnings, *verify.warnings)))
    if warnings:
        lines.append("")
        lines.append("Notes:")
        lines.extend(f"- {warning}" for warning in warnings[:6])
    return "\n".join(lines)


def _setup_plan_content(
    record: InstalledSkillRecord,
    target: Path,
    command: str,
) -> str:
    lines = [
        f"Skill setup plan: {record.name}",
        f"- target: {record.target_path}",
        f"- setup_status: {record.setup_status or 'not_required'}",
    ]
    if command:
        lines.extend(
            (
                f"- recommended cwd: {target}",
                f"- recommended command: {command}",
                "- execution: not run automatically; use run_shell_command only after permissions are enabled",
            )
        )
    else:
        lines.append("- setup: no scripts/ setup entry found")
    if record.warnings:
        lines.append("- warnings:")
        lines.extend(f"  - {warning}" for warning in record.warnings)
    return "\n".join(lines)


def _setup_run_content(
    record: InstalledSkillRecord,
    target: Path,
    command: str,
    run_result,
) -> str:
    lines = [
        f"Skill setup run: {record.name}",
        f"- target: {record.target_path}",
        f"- cwd: {target}",
        f"- command: {command}",
        f"- setup_status: {record.setup_status}",
        f"- exit_code: {run_result.exit_code}",
        f"- sandbox_backend: {run_result.backend}",
        f"- sandboxed: {str(run_result.sandboxed).lower()}",
    ]
    if run_result.backend == "permission-only" and not run_result.sandboxed:
        lines.append(
            "Warning: OS sandbox backend is unavailable; setup ran with "
            "permission-only enforcement."
        )
    output = run_result.output_text().strip()
    if output:
        lines.extend(("", "Output:", output))
    return "\n".join(lines)


def _workspace_path(root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    path = candidate if candidate.is_absolute() else root / candidate
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"skill target must stay inside workspace: {raw_path}")
    return resolved


def _record_target_path(
    workspace: Path,
    data_dir: Path,
    record: InstalledSkillRecord,
) -> Path:
    root = _record_execution_root(workspace, data_dir, record)
    return _path_inside(root, record.target_path, label="skill target")


def _record_execution_root(
    workspace: Path,
    data_dir: Path,
    record: InstalledSkillRecord,
) -> Path:
    if (record.target_scope or "workspace") == "user":
        return data_dir.resolve()
    return workspace.resolve()


def _path_inside(root: Path, raw_path: str, *, label: str) -> Path:
    candidate = Path(raw_path)
    path = candidate if candidate.is_absolute() else root / candidate
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} must stay inside {root}: {raw_path}")
    return resolved


def _text_argument(arguments: Mapping[str, object], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_text_argument(arguments: Mapping[str, object], name: str) -> str:
    value = arguments.get(name)
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _bool_argument(arguments: Mapping[str, object], name: str) -> bool:
    value = arguments.get(name, False)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    return False


def _int_argument(
    arguments: Mapping[str, object],
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = arguments.get(name)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _decision_message(decision) -> str:
    refs = ", ".join(decision.refs)
    return f"{decision.reason} {refs}".strip()


def _now() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()
