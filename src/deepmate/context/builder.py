"""System context builder."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from deepmate.domain import (
    CapabilityKind,
    CapabilityRef,
    Message,
    MessageRole,
    ProfileRef,
)
from deepmate.foundation import estimate_text_tokens
from deepmate.context.snapshot import (
    ContextFileChange,
    ContextFileRef,
    ProfileContextSnapshot,
)
from deepmate.evolution import read_behavior_hint_documents, render_collaboration_hints

if TYPE_CHECKING:
    from deepmate.capabilities import CapabilitySurface
    from deepmate.skills import SkillDocument

WORKSPACE_RULES_FILE = "AGENTS.md"
GLOBAL_PROFILE_FILES = (
    ("identity", "identity.md"),
    ("soul", "soul.md"),
    ("user", "user.md"),
    ("memory", "memory.md"),
)
PROFILE_FILES = GLOBAL_PROFILE_FILES
PROJECT_PROFILE_FILES = (("project_memory", "memory.md"),)
OPTIONAL_PROFILE_SECTIONS = {"user", "memory"}
OPTIONAL_PROJECT_PROFILE_SECTIONS = {"project_memory"}
BEHAVIOR_CONTEXT_FILE_NAMES = frozenset(("workspace_behavior", "profile_behavior"))
CAPABILITY_SECTION = "available_capabilities"
CAPABILITY_GUIDANCE_SECTION = "capability_guidance"
SELECTED_SKILLS_SECTION = "selected_skills"
BROWSER_LOADER_TOOL_NAME = "load_browser_tools"
LSP_TOOL_NAMES = frozenset(("lsp_definition", "lsp_references", "lsp_hover"))
CAPABILITY_GROUPS = (
    (CapabilityKind.SKILL, "skills"),
    (CapabilityKind.NATIVE_TOOL, "native_tools"),
    (CapabilityKind.MCP_SERVER, "mcp_servers"),
    (CapabilityKind.MCP_TOOL, "mcp_tools"),
)


@dataclass(frozen=True, slots=True)
class ContextWarning:
    """Non-fatal context build issue kept outside the prompt."""

    code: str
    message: str
    refs: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ContextBuildResult:
    """System context plus user-visible diagnostics kept outside the prompt."""

    message: Message
    warnings: tuple[ContextWarning, ...] = field(default_factory=tuple)


def build_system_context(
    workspace: str | Path,
    profile: ProfileRef,
    capability_surface: CapabilitySurface | None = None,
    selected_skill_documents: tuple["SkillDocument", ...] = (),
) -> ContextBuildResult:
    """Build system context from workspace rules and profile files."""
    snapshot = build_profile_context_snapshot(workspace=workspace, profile=profile)
    return build_system_context_from_snapshot(
        snapshot=snapshot,
        capability_surface=capability_surface,
        selected_skill_documents=selected_skill_documents,
    )


def build_system_context_from_snapshot(
    snapshot: ProfileContextSnapshot,
    capability_surface: CapabilitySurface | None = None,
    selected_skill_documents: tuple["SkillDocument", ...] = (),
) -> ContextBuildResult:
    """Build system context from a frozen profile snapshot.

    Section order is deliberately stable-first so the provider prefix cache can
    reuse the longest possible prefix: the frozen profile/workspace sections come
    first, then the install-volatile capability surface, then the per-turn-volatile
    selected skills last. Installing a skill changes only the capability tail;
    loading a skill changes only the selected-skills tail. Keep volatile content at
    the end when adding new sections.
    """
    sections = list(snapshot.sections)
    if not sections:
        # An empty workspace (no AGENTS.md/CLAUDE.md and an empty profile) must
        # still start a session, so fall back to a minimal stable identity section
        # instead of refusing to build context.
        sections = [_default_workspace_section(snapshot.workspace)]

    capability_section = _render_capability_surface(capability_surface)
    if capability_section:
        sections.append(capability_section)
    capability_guidance = _render_capability_guidance(capability_surface)
    if capability_guidance:
        sections.append(capability_guidance)
    selected_skills_section = _render_selected_skills(selected_skill_documents)
    if selected_skills_section:
        sections.append(selected_skills_section)

    content = "\n\n".join(sections)
    message = Message(role=MessageRole.SYSTEM, content=content)
    return ContextBuildResult(message=message, warnings=snapshot.warnings)


def read_profile_context_sections(
    workspace: str | Path,
    profile: ProfileRef,
) -> tuple[tuple[str, ...], tuple[ContextWarning, ...]]:
    """Read workspace/profile markdown sections for a context snapshot."""
    sections, warnings, _, _ = _read_profile_context(
        workspace=workspace,
        profile=profile,
    )
    return sections, warnings


def detect_behavior_context_changes(
    snapshot: ProfileContextSnapshot,
) -> tuple[ContextFileChange, ...]:
    """Return behavior.md source changes relative to a frozen snapshot."""
    old_refs = {
        ref.name: ref
        for ref in snapshot.file_refs
        if ref.name in BEHAVIOR_CONTEXT_FILE_NAMES
    }
    changes: list[ContextFileChange] = []
    for document in read_behavior_hint_documents(snapshot.workspace, snapshot.profile):
        old_ref = old_refs.get(document.name)
        if old_ref is None:
            continue
        current_ref = _behavior_context_ref(document)
        if _context_file_ref_changed(old_ref, current_ref):
            changes.append(
                ContextFileChange(
                    name=old_ref.name,
                    path=old_ref.path,
                    old_status=old_ref.status,
                    new_status=current_ref.status,
                    old_sha256=old_ref.sha256,
                    new_sha256=current_ref.sha256,
                )
            )
    return tuple(changes)


def _read_profile_context(
    workspace: str | Path,
    profile: ProfileRef,
) -> tuple[tuple[str, ...], tuple[ContextWarning, ...], tuple[ContextFileRef, ...], int]:
    if not profile.is_ready():
        raise ValueError("profile must include name and uri")

    workspace_path = Path(workspace)
    project_profile_path = _resolve_profile_path(
        workspace_path,
        profile.project_uri or profile.uri,
    )
    global_profile_path = (
        _resolve_profile_path(workspace_path, profile.global_uri)
        if profile.global_uri.strip()
        else None
    )

    if global_profile_path is not None:
        context_files = [
            *(
                (name, global_profile_path / filename, True)
                for name, filename in GLOBAL_PROFILE_FILES
            ),
            ("workspace_rules", workspace_path / WORKSPACE_RULES_FILE, False),
            *(
                (name, project_profile_path / filename, True)
                for name, filename in PROJECT_PROFILE_FILES
            ),
        ]
    else:
        profile_path = _resolve_profile_path(workspace_path, profile.uri)
        context_files = [
            ("workspace_rules", workspace_path / WORKSPACE_RULES_FILE, False),
            *(
                (name, profile_path / filename, name in OPTIONAL_PROFILE_SECTIONS)
                for name, filename in PROFILE_FILES
            ),
        ]

    sections: list[str] = []
    warnings: list[ContextWarning] = []
    refs: list[ContextFileRef] = []
    hot_profile_tokens = 0
    for name, path, optional in context_files:
        section, warning, ref = _read_section(
            name,
            path,
            optional=optional,
        )
        if section:
            sections.append(section)
        if name in OPTIONAL_PROFILE_SECTIONS or name in OPTIONAL_PROJECT_PROFILE_SECTIONS:
            hot_profile_tokens += ref.estimated_tokens
        if warning:
            warnings.append(warning)
        refs.append(ref)

    behavior_documents = read_behavior_hint_documents(workspace_path, profile)
    behavior_section = render_collaboration_hints(behavior_documents)
    if behavior_section:
        sections.append(behavior_section)
    refs.extend(_behavior_context_ref(document) for document in behavior_documents)

    return tuple(sections), tuple(warnings), tuple(refs), hot_profile_tokens


def _resolve_profile_path(workspace: Path, uri: str) -> Path:
    path = Path(uri)
    return path if path.is_absolute() else workspace / path


def _default_workspace_section(workspace: Path) -> str:
    """Minimal stable identity section for a workspace with no readable rules."""
    return "\n".join(
        (
            "<workspace>",
            f"- root: {workspace}",
            "- No AGENTS.md/CLAUDE.md or profile rules were found for this workspace.",
            "- You are Deepmate, a local long-task agent. Help with the user's task "
            "using the available tools and ask for any project conventions you need.",
            "</workspace>",
        )
    )


def _behavior_context_ref(document) -> ContextFileRef:
    return ContextFileRef(
        name=document.name,
        path=document.path,
        status=document.status,
        size_bytes=document.size_bytes,
        sha256=document.sha256,
        estimated_tokens=document.estimated_tokens,
    )


def _context_file_ref_changed(old: ContextFileRef, current: ContextFileRef) -> bool:
    return old.status != current.status or old.sha256 != current.sha256


def build_profile_context_snapshot(
    workspace: str | Path,
    profile: ProfileRef,
    hot_profile_token_budget: int = 0,
    hot_profile_warn_tokens: int = 0,
    pending_refresh_reason: str = "",
    extra_sections: tuple[str, ...] = (),
) -> ProfileContextSnapshot:
    """Read workspace/profile markdown into a reusable context snapshot."""
    workspace_path = Path(workspace)
    sections, warnings, refs, hot_profile_tokens = _read_profile_context(
        workspace=workspace_path,
        profile=profile,
    )
    if not sections:
        # An empty workspace (no AGENTS.md/CLAUDE.md and an empty profile) still
        # needs a readable section so the snapshot is ready and a session can
        # start. Inject a minimal stable identity section as the only fallback.
        sections = (_default_workspace_section(workspace_path),)
    clean_extra_sections = tuple(
        section.strip() for section in extra_sections if section.strip()
    )
    if clean_extra_sections:
        sections = (*sections, *clean_extra_sections)
    warnings = (
        *warnings,
        *_hot_profile_budget_warnings(
            estimated_tokens=hot_profile_tokens,
            token_budget=hot_profile_token_budget,
            warn_tokens=hot_profile_warn_tokens,
        ),
    )
    return ProfileContextSnapshot(
        workspace=workspace_path,
        profile=profile,
        sections=sections,
        warnings=warnings,
        file_refs=refs,
        hot_profile_token_budget=max(0, hot_profile_token_budget),
        hot_profile_warn_tokens=max(0, hot_profile_warn_tokens),
        hot_profile_estimated_tokens=hot_profile_tokens,
        pending_refresh_reason=pending_refresh_reason.strip(),
    )


def _read_section(
    name: str,
    path: Path,
    optional: bool = False,
) -> tuple[str, ContextWarning | None, ContextFileRef]:
    if not path.exists():
        status = "missing_optional" if optional else "missing_required"
        ref = ContextFileRef(name=name, path=path, status=status)
        if optional:
            return "", None, ref
        warning = ContextWarning(
            code="context_file_missing",
            message=f"missing context file: {path}",
            refs=(str(path),),
        )
        return "", warning, ContextFileRef(
            name=name,
            path=path,
            status=status,
            warning_code=warning.code,
        )

    raw_content = path.read_text(encoding="utf-8")
    content = raw_content.strip()
    size_bytes = len(raw_content.encode("utf-8"))
    sha256 = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
    estimated_tokens = _estimate_context_tokens(content)
    if not content:
        status = "empty_optional" if optional else "empty_required"
        ref = ContextFileRef(
            name=name,
            path=path,
            status=status,
            size_bytes=size_bytes,
            sha256=sha256,
            estimated_tokens=0,
        )
        if optional:
            return "", None, ref
        warning = ContextWarning(
            code="context_file_empty",
            message=f"empty context file: {path}",
            refs=(str(path),),
        )
        return "", warning, ContextFileRef(
            name=name,
            path=path,
            status=status,
            size_bytes=size_bytes,
            sha256=sha256,
            estimated_tokens=0,
            warning_code=warning.code,
        )
    return (
        f"<{name}>\n{content}\n</{name}>",
        None,
        ContextFileRef(
            name=name,
            path=path,
            status="loaded",
            size_bytes=size_bytes,
            sha256=sha256,
            estimated_tokens=estimated_tokens,
        ),
    )


def _hot_profile_budget_warnings(
    estimated_tokens: int,
    token_budget: int,
    warn_tokens: int,
) -> tuple[ContextWarning, ...]:
    if token_budget <= 0 or estimated_tokens <= 0:
        return ()
    if estimated_tokens > token_budget:
        return (
            ContextWarning(
                code="hot_profile_budget_exceeded",
                message=(
                    "hot profile context exceeds budget: "
                    f"{estimated_tokens}/{token_budget} tokens"
                ),
                refs=(
                    f"hot_profile_estimated_tokens={estimated_tokens}",
                    f"hot_profile_token_budget={token_budget}",
                ),
            ),
        )
    if warn_tokens > 0 and estimated_tokens >= warn_tokens:
        return (
            ContextWarning(
                code="hot_profile_budget_warning",
                message=(
                    "hot profile context is near budget: "
                    f"{estimated_tokens}/{token_budget} tokens"
                ),
                refs=(
                    f"hot_profile_estimated_tokens={estimated_tokens}",
                    f"hot_profile_warn_tokens={warn_tokens}",
                    f"hot_profile_token_budget={token_budget}",
                ),
            ),
        )
    return ()


def _estimate_context_tokens(text: str) -> int:
    return estimate_text_tokens(text)


def _render_capability_surface(surface: CapabilitySurface | None) -> str:
    if surface is None or surface.is_empty():
        return ""

    group_sections: list[str] = []
    refs = surface.list_refs()
    for kind, group_name in CAPABILITY_GROUPS:
        group_refs = tuple(ref for ref in refs if ref.kind == kind)
        if group_refs:
            group_sections.append(_render_capability_group(group_name, group_refs))

    if not group_sections:
        return ""

    body = "\n\n".join(group_sections)
    return f"<{CAPABILITY_SECTION}>\n{body}\n</{CAPABILITY_SECTION}>"


def _render_capability_group(name: str, refs: tuple[CapabilityRef, ...]) -> str:
    lines = [f"<{name}>"]
    for ref in refs:
        clean_name = _clean_inline(ref.name)
        clean_description = _clean_inline(ref.description)
        if clean_description:
            lines.append(f"- {clean_name}: {clean_description}")
        else:
            lines.append(f"- {clean_name}")
    lines.append(f"</{name}>")
    return "\n".join(lines)


def _render_capability_guidance(surface: CapabilitySurface | None) -> str:
    if surface is None or surface.is_empty():
        return ""
    refs = surface.list_refs()
    guidance: list[str] = []
    if _has_browser_tools(refs):
        guidance.extend(
            (
                "Use the built-in browser for dynamic web pages, login-free page interaction, frontend verification, screenshots, or when static fetch/search is insufficient.",
                "When only load_browser_tools is visible, call it first to load concrete browser schemas such as browser_open and browser_snapshot for the next step.",
                "Prefer cheaper static retrieval/search or local file inspection for simple static information; do not default to browser for every web question.",
                "After navigation or DOM changes, call browser_snapshot before click/fill so element refs are current.",
                "Do not use browser tools for broad crawling, CAPTCHA bypass, credential entry, stealth automation, or DevTools-level debugging.",
            )
        )
    if _has_lsp_tools(refs):
        guidance.extend(
            (
                "For code symbol questions, prefer lsp_definition, lsp_references, or lsp_hover first; use grep/search as fallback or to verify broader text matches.",
                "LSP tools are read-only and may report unavailable when a language server is not installed; if unavailable, continue with normal workspace search.",
            )
        )
    if _has_skills(refs):
        guidance.extend(
            (
                "Deepmate supports community-style SKILL.md bundles discovered from workspace skill directories. Do not claim SKILL.md skills are unsupported or only usable in another agent platform.",
                "Visible skills are summaries only to keep context cheap. When a listed skill matches the task, call load_skill or use an explicitly selected skill before following its full instructions.",
                "A loaded SKILL.md is operational guidance for this run; related scripts, references, and assets may be inspected or executed through the normal file, shell, and approval flow.",
            )
        )
    if not guidance:
        return ""
    body = "\n".join(f"- {_clean_inline(line)}" for line in guidance)
    return f"<{CAPABILITY_GUIDANCE_SECTION}>\n{body}\n</{CAPABILITY_GUIDANCE_SECTION}>"


def _has_skills(refs: tuple[CapabilityRef, ...]) -> bool:
    return any(ref.kind == CapabilityKind.SKILL for ref in refs)


def _has_browser_tools(refs: tuple[CapabilityRef, ...]) -> bool:
    return any(
        ref.kind == CapabilityKind.NATIVE_TOOL
        and (
            ref.name.strip() == BROWSER_LOADER_TOOL_NAME
            or ref.name.strip().startswith("browser_")
        )
        for ref in refs
    )


def _has_lsp_tools(refs: tuple[CapabilityRef, ...]) -> bool:
    return any(
        ref.kind == CapabilityKind.NATIVE_TOOL
        and ref.name.strip() in LSP_TOOL_NAMES
        for ref in refs
    )


def _render_selected_skills(skills: tuple["SkillDocument", ...]) -> str:
    if not skills:
        return ""

    sections: list[str] = []
    for skill in skills:
        if not skill.is_ready():
            raise ValueError("selected skills must be ready before context build")
        sections.append(
            "\n".join(
                (
                    "<skill>",
                    f"<name>{_clean_inline(skill.name)}</name>",
                    f"<description>{_clean_inline(skill.description)}</description>",
                    "<instructions>",
                    skill.body.strip(),
                    "</instructions>",
                    "</skill>",
                )
            )
        )
    return (
        f"<{SELECTED_SKILLS_SECTION}>\n"
        + "\n\n".join(sections)
        + f"\n</{SELECTED_SKILLS_SECTION}>"
    )


def _clean_inline(value: str) -> str:
    return " ".join(value.strip().split())
