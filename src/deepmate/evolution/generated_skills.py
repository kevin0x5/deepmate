"""Generated skill lifecycle helpers.

Generated skills are ordinary community-compatible SKILL.md bundles. Deepmate's
ownership and runtime state stay in capability_state, not inside SKILL.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from deepmate.capabilities.state import (
    CapabilityAssetState,
    CapabilitySource,
    CapabilityState,
    CapabilityStateStore,
    CapabilityTemperature,
)
from deepmate.domain import CapabilityKind, ProfileRef
from deepmate.evolution.changes import (
    EvolutionChange,
    EvolutionChangeStore,
    applied_change,
    sidecar_restore_metadata,
)
from deepmate.evolution.evidence_mining import WorkflowAggregate
from deepmate.evolution.failure_patterns import FailurePatternGuard
from deepmate.foundation import normalize_name, normal_datetime, utc_isoformat
from deepmate.skills import SkillCard, load_skill_card, load_skill_document
from deepmate.storage import atomic_write_text

GENERATED_SKILLS_DIR = "generated"
SKILL_FILE_NAME = "SKILL.md"


@dataclass(frozen=True, slots=True)
class GeneratedSkillDraft:
    """One standard SKILL.md draft produced from repeated workflow evidence."""

    name: str
    description: str
    steps: tuple[str, ...]
    source_refs: tuple[str, ...]
    when_to_use: tuple[str, ...] = ()
    reference_paths: tuple[str, ...] = ()

    def is_ready(self) -> bool:
        """Return whether this draft can be rendered and validated."""
        return bool(self.name.strip() and self.description.strip() and self.steps)

    def guard_text(self) -> str:
        """Return deterministic text used by FailurePatternGuard."""
        return "\n".join(
            (
                self.name,
                self.description,
                *self.when_to_use,
                *self.steps,
                *self.source_refs,
                *self.reference_paths,
            )
        )


@dataclass(frozen=True, slots=True)
class GeneratedSkillApplyResult:
    """Result of applying a generated skill lifecycle change."""

    status: str
    reason: str
    skill_path: Path | None = None
    change: EvolutionChange | None = None
    state: CapabilityState | None = None

    def is_applied(self) -> bool:
        """Return whether the lifecycle change was applied."""
        return self.status == "applied"


def generated_skill_drafts_from_workflows(
    workflows: tuple[WorkflowAggregate, ...],
) -> tuple[GeneratedSkillDraft, ...]:
    """Turn repeated workflow aggregates into generated skill drafts."""
    drafts: list[GeneratedSkillDraft] = []
    for workflow in workflows:
        if not workflow.is_ready():
            continue
        drafts.append(
            GeneratedSkillDraft(
                name=workflow.name,
                description=workflow.description,
                steps=workflow.steps,
                source_refs=workflow.source_refs,
                when_to_use=(workflow.description,),
                reference_paths=workflow.reference_paths,
            )
        )
    return tuple(drafts)


def apply_generated_skill_draft(
    *,
    draft: GeneratedSkillDraft,
    workspace: str | Path,
    data_dir: str | Path,
    profile: ProfileRef | str,
    guard: FailurePatternGuard | None = None,
    state_store: CapabilityStateStore | None = None,
    change_store: EvolutionChangeStore | None = None,
    now: datetime | None = None,
) -> GeneratedSkillApplyResult:
    """Validate, write, register, and log one generated skill draft."""
    if not draft.is_ready():
        return GeneratedSkillApplyResult(status="rejected", reason="draft_not_ready")
    workspace_path = Path(workspace)
    data_path = Path(data_dir)
    target_path = generated_skill_path(workspace_path, draft.name)
    _require_generated_skill_path(target_path, workspace_path)
    if target_path.exists():
        return GeneratedSkillApplyResult(
            status="rejected",
            reason="generated_skill_already_exists",
            skill_path=target_path,
        )
    existing_skill_path = _existing_skill_name_path(workspace_path, draft.name)
    if existing_skill_path is not None:
        existing_ref = _workspace_relative(existing_skill_path, workspace_path)
        return GeneratedSkillApplyResult(
            status="rejected",
            reason=f"skill_name_already_exists:{existing_ref}",
            skill_path=target_path,
        )
    match = guard.check_text(draft.guard_text(), draft.source_refs) if guard else None
    if match is not None and match.blocked:
        return GeneratedSkillApplyResult(
            status="blocked",
            reason=match.reason,
            skill_path=target_path,
        )
    reference_error = _reference_error(workspace_path, draft.reference_paths)
    if reference_error:
        return GeneratedSkillApplyResult(
            status="rejected",
            reason=reference_error,
            skill_path=target_path,
        )

    state_store = state_store or CapabilityStateStore.in_data_dir(data_path, profile)
    change_store = change_store or EvolutionChangeStore.in_data_dir(data_path, profile)
    state_old_exists, state_old_content = _read_optional_text(state_store.path)
    content = render_generated_skill_markdown(draft)
    current_time = normal_datetime(now)
    timestamp = utc_isoformat(current_time)
    try:
        atomic_write_text(target_path, content)
        card = _validate_generated_skill_file(target_path, workspace_path, draft.name)
        state = state_store.record_skill_installed(
            card,
            workspace_path,
            now=current_time,
            source=CapabilitySource.GENERATED,
        )
        state_new_content = state_store.path.read_text(encoding="utf-8")
        change = applied_change(
            change_type="generated_skill_draft",
            target_path=_workspace_relative(target_path, workspace_path),
            summary=f"Created generated skill {card.name}.",
            old_content="",
            new_content=content,
            old_exists=False,
            evidence_refs=draft.source_refs,
            validation_result="passed",
            decision="auto_apply_with_validation",
            now_iso=timestamp,
            metadata=sidecar_restore_metadata(
                data_dir=data_path,
                sidecar_path=state_store.path,
                old_content=state_old_content,
                new_content=state_new_content,
                old_exists=state_old_exists,
            ),
        )
        change_store.append(change)
        return GeneratedSkillApplyResult(
            status="applied",
            reason="generated_skill_created",
            skill_path=target_path,
            change=change,
            state=state,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _restore_optional_text(state_store.path, state_old_exists, state_old_content)
        if target_path.exists():
            target_path.unlink()
        return GeneratedSkillApplyResult(
            status="rolled_back",
            reason=f"validation_failed:{exc}",
            skill_path=target_path,
        )


def apply_generated_skill_patch(
    *,
    skill_name: str,
    new_markdown: str,
    workspace: str | Path,
    data_dir: str | Path,
    profile: ProfileRef | str,
    guard: FailurePatternGuard | None = None,
    state_store: CapabilityStateStore | None = None,
    change_store: EvolutionChangeStore | None = None,
    now: datetime | None = None,
) -> GeneratedSkillApplyResult:
    """Patch only a Deepmate-generated skill and rollback on validation failure."""
    workspace_path = Path(workspace)
    data_path = Path(data_dir)
    state_store = state_store or CapabilityStateStore.in_data_dir(data_path, profile)
    target = _generated_skill_target(state_store, workspace_path, skill_name)
    if target.state is None or target.path is None:
        return GeneratedSkillApplyResult(status="rejected", reason=target.reason)
    match = guard.check_text(new_markdown, (skill_name,)) if guard else None
    if match is not None and match.blocked:
        return GeneratedSkillApplyResult(
            status="blocked",
            reason=match.reason,
            skill_path=target.path,
            state=target.state,
        )

    old_content = target.path.read_text(encoding="utf-8")
    change_store = change_store or EvolutionChangeStore.in_data_dir(data_path, profile)
    timestamp = utc_isoformat(normal_datetime(now))
    try:
        atomic_write_text(target.path, new_markdown)
        _validate_generated_skill_file(target.path, workspace_path, skill_name)
        change = applied_change(
            change_type="generated_skill_patch",
            target_path=_workspace_relative(target.path, workspace_path),
            summary=f"Patched generated skill {target.state.name}.",
            old_content=old_content,
            new_content=new_markdown,
            old_exists=True,
            evidence_refs=(f"capability_id={target.state.capability_id}",),
            validation_result="passed",
            decision="auto_apply_with_validation",
            now_iso=timestamp,
        )
        change_store.append(change)
        return GeneratedSkillApplyResult(
            status="applied",
            reason="generated_skill_patched",
            skill_path=target.path,
            change=change,
            state=target.state,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        atomic_write_text(target.path, old_content)
        return GeneratedSkillApplyResult(
            status="rolled_back",
            reason=f"validation_failed:{exc}",
            skill_path=target.path,
            state=target.state,
        )


def archive_generated_skill(
    *,
    skill_name: str,
    workspace: str | Path,
    data_dir: str | Path,
    profile: ProfileRef | str,
    state_store: CapabilityStateStore | None = None,
    change_store: EvolutionChangeStore | None = None,
    now: datetime | None = None,
) -> GeneratedSkillApplyResult:
    """Archive only a generated skill by updating capability_state."""
    workspace_path = Path(workspace)
    data_path = Path(data_dir)
    state_store = state_store or CapabilityStateStore.in_data_dir(data_path, profile)
    target = _generated_skill_target(state_store, workspace_path, skill_name)
    if target.state is None or target.path is None:
        return GeneratedSkillApplyResult(status="rejected", reason=target.reason)
    state_old_exists, state_old_content = _read_optional_text(state_store.path)
    states = state_store.load()
    current_time = normal_datetime(now)
    timestamp = utc_isoformat(current_time)
    archived = replace(
        target.state,
        hidden=True,
        asset_state=CapabilityAssetState.ARCHIVED,
        temperature=CapabilityTemperature.COLD,
        updated_at=timestamp,
    )
    states[archived.capability_id] = archived
    skill_content = target.path.read_text(encoding="utf-8")
    change_store = change_store or EvolutionChangeStore.in_data_dir(data_path, profile)
    try:
        state_store.save(states)
        state_new_content = state_store.path.read_text(encoding="utf-8")
        change = applied_change(
            change_type="generated_skill_archive",
            target_path=_workspace_relative(target.path, workspace_path),
            summary=f"Archived generated skill {archived.name}.",
            old_content=skill_content,
            new_content=skill_content,
            old_exists=True,
            evidence_refs=(
                f"capability_id={archived.capability_id}",
                "source=generated",
            ),
            validation_result="passed",
            decision="auto_apply",
            now_iso=timestamp,
            metadata=sidecar_restore_metadata(
                data_dir=data_path,
                sidecar_path=state_store.path,
                old_content=state_old_content,
                new_content=state_new_content,
                old_exists=state_old_exists,
            ),
        )
        change_store.append(change)
        return GeneratedSkillApplyResult(
            status="applied",
            reason="generated_skill_archived",
            skill_path=target.path,
            change=change,
            state=archived,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _restore_optional_text(state_store.path, state_old_exists, state_old_content)
        return GeneratedSkillApplyResult(
            status="rolled_back",
            reason=f"validation_failed:{exc}",
            skill_path=target.path,
            state=target.state,
        )


def render_generated_skill_markdown(draft: GeneratedSkillDraft) -> str:
    """Render one draft as a standard SKILL.md file."""
    name = _clean_line(draft.name)
    description = _clean_line(draft.description)
    lines = [
        "---",
        f"name: {_yaml_string(name)}",
        f"description: {_yaml_string(description)}",
        "---",
        "",
        f"# {name}",
        "",
        "## When to use",
    ]
    when_to_use = draft.when_to_use or (description,)
    lines.extend(f"- {_clean_line(item)}" for item in when_to_use if _clean_line(item))
    lines.extend(("", "## Steps"))
    lines.extend(f"- {_clean_line(step)}" for step in draft.steps if _clean_line(step))
    if draft.reference_paths:
        lines.extend(("", "## References"))
        lines.extend(
            f"- `{_clean_line(path)}`"
            for path in draft.reference_paths
            if _clean_line(path)
        )
    return "\n".join(lines).rstrip() + "\n"


def generated_skill_root(workspace: str | Path) -> Path:
    """Return the generated skill root under the workspace skill directory."""
    return Path(workspace) / "skills" / GENERATED_SKILLS_DIR


def generated_skill_path(workspace: str | Path, name: str) -> Path:
    """Return the generated SKILL.md path for a skill name."""
    return generated_skill_root(workspace) / _slugify_skill_name(name) / SKILL_FILE_NAME


@dataclass(frozen=True, slots=True)
class _GeneratedSkillTarget:
    path: Path | None
    state: CapabilityState | None
    reason: str


def _generated_skill_target(
    state_store: CapabilityStateStore,
    workspace: Path,
    skill_name: str,
) -> _GeneratedSkillTarget:
    state = state_store.skill_states_by_name().get(normalize_name(skill_name))
    if state is None:
        return _GeneratedSkillTarget(None, None, "skill_state_not_found")
    if state.kind != CapabilityKind.SKILL or state.source != CapabilitySource.GENERATED:
        return _GeneratedSkillTarget(None, state, "skill_is_not_generated")
    path = workspace / state.path_or_ref
    try:
        _require_generated_skill_path(path, workspace)
    except ValueError as exc:
        return _GeneratedSkillTarget(None, state, str(exc))
    if not path.exists():
        return _GeneratedSkillTarget(None, state, "generated_skill_file_missing")
    return _GeneratedSkillTarget(path, state, "ok")


def _validate_generated_skill_file(
    path: Path,
    workspace: Path,
    expected_name: str,
) -> SkillCard:
    _require_generated_skill_path(path, workspace)
    card = load_skill_card(path)
    document = load_skill_document(card)
    if normalize_name(document.name) != normalize_name(expected_name):
        raise ValueError("generated skill patch cannot rename the skill")
    if not document.description.strip() or not document.body.strip():
        raise ValueError("generated skill requires description and body")
    return card


def _existing_skill_name_path(
    workspace: Path,
    name: str,
    exclude_path: Path | None = None,
) -> Path | None:
    expected = normalize_name(name)
    roots = (workspace / "skills", workspace / ".claude" / "skills")
    for root in roots:
        if not root.exists():
            continue
        for skill_path in sorted(root.rglob(SKILL_FILE_NAME)):
            if (
                exclude_path is not None
                and skill_path.resolve() == exclude_path.resolve()
            ):
                continue
            try:
                card = load_skill_card(skill_path)
            except (OSError, ValueError):
                continue
            if normalize_name(card.name) == expected:
                return skill_path
    return None


def _reference_error(workspace: Path, refs: tuple[str, ...]) -> str:
    for ref in refs:
        path = workspace / ref
        try:
            path.resolve().relative_to(workspace.resolve())
        except ValueError:
            return f"reference_outside_workspace:{ref}"
        if not path.exists():
            return f"reference_missing:{ref}"
    return ""


def _require_generated_skill_path(path: Path, workspace: Path) -> None:
    root = generated_skill_root(workspace).resolve()
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError(f"generated skill path is outside generated root: {path}") from exc
    if path.name != SKILL_FILE_NAME:
        raise ValueError(f"generated skill target must be {SKILL_FILE_NAME}: {path}")


def _slugify_skill_name(name: str) -> str:
    chars: list[str] = []
    previous_dash = False
    for char in name.strip().lower():
        if char.isalnum():
            chars.append(char)
            previous_dash = False
        elif not previous_dash:
            chars.append("-")
            previous_dash = True
    slug = "".join(chars).strip("-")
    return (slug or "generated-skill")[:80].strip("-") or "generated-skill"


def _yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _clean_line(value: str) -> str:
    return " ".join(value.strip().split())


def _read_optional_text(path: Path) -> tuple[bool, str]:
    return (path.exists(), path.read_text(encoding="utf-8") if path.exists() else "")


def _restore_optional_text(path: Path, existed: bool, content: str) -> None:
    if existed:
        atomic_write_text(path, content)
        return
    if path.exists():
        path.unlink()


def _workspace_relative(path: Path, workspace: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace.resolve()))
    except ValueError:
        return str(path)
