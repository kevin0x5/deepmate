"""Skill listing and detail rendering for channel entrypoints."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from importlib import resources
from pathlib import Path

from deepmate.capabilities.state import CapabilityState, CapabilityStateStore
from deepmate.context import ContextWarning
from deepmate.foundation import (
    compact_text,
    display_path as foundation_display_path,
    normalize_name,
)
from deepmate.skills import SkillCatalog, SkillDocument, SkillCard, load_skill_document
from deepmate.skills import load_skill_card
from deepmate.skills.install import user_skill_library_root

SKILL_NAME_WIDTH = 28
SKILL_TEMPERATURE_WIDTH = 6
SKILL_EXPOSURE_WIDTH = 16
SKILL_DESCRIPTION_WIDTH = 48


def workspace_skill_catalog(
    workspace: Path,
    data_dir: Path | None = None,
) -> tuple[SkillCatalog | None, tuple[ContextWarning, ...]]:
    """Discover built-in and workspace skills with channel-friendly tolerance."""
    cards, warnings = discover_skill_cards(workspace, data_dir=data_dir)
    if not cards:
        return None, warnings
    return SkillCatalog(cards), warnings


def discover_skill_cards(
    workspace: Path,
    *,
    data_dir: Path | None = None,
) -> tuple[tuple[SkillCard, ...], tuple[ContextWarning, ...]]:
    """Return workspace/user cards followed by built-in cards.

    Workspace skills win on duplicate names because they are project-owned.
    """
    builtin_cards, builtin_warnings = discover_builtin_skill_cards()
    workspace_cards, workspace_warnings = discover_workspace_skill_cards(workspace)
    user_cards, user_warnings = discover_user_skill_cards(data_dir)
    cards: list[SkillCard] = []
    warnings: list[ContextWarning] = [
        *builtin_warnings,
        *workspace_warnings,
        *user_warnings,
    ]
    seen_names: set[str] = set()
    for card in workspace_cards:
        normalized_name = _normalize_name(card.name)
        directory_name = _normalize_name(card.path.parent.name)
        seen_names.add(normalized_name)
        seen_names.add(directory_name)
        cards.append(card)
    for card in user_cards:
        normalized_name = _normalize_name(card.name)
        directory_name = _normalize_name(card.path.parent.name)
        if normalized_name in seen_names or directory_name in seen_names:
            warnings.append(
                ContextWarning(
                    code="duplicate_skill_name",
                    message=f"user skill skipped because a workspace skill has the same name: {card.name}",
                    refs=(str(card.path),),
                )
            )
            continue
        seen_names.add(normalized_name)
        seen_names.add(directory_name)
        cards.append(card)
    for card in builtin_cards:
        normalized_name = _normalize_name(card.name)
        directory_name = _normalize_name(card.path.parent.name)
        if normalized_name in seen_names or directory_name in seen_names:
            continue
        seen_names.add(normalized_name)
        seen_names.add(directory_name)
        cards.append(card)
    return tuple(cards), tuple(warnings)


def discover_builtin_skill_cards() -> tuple[tuple[SkillCard, ...], tuple[ContextWarning, ...]]:
    """Return Deepmate built-in Work Kits as built-in skill cards."""
    skill_roots = _builtin_skill_roots()
    if not skill_roots:
        return (), ()
    cards: list[SkillCard] = []
    warnings: list[ContextWarning] = []
    for skill_root in skill_roots:
        try:
            card = load_skill_card(skill_root / "SKILL.md")
        except (OSError, ValueError) as exc:
            warnings.append(
                ContextWarning(
                    code="builtin_skill_discovery_failed",
                    message=f"built-in skill skipped: {exc}",
                    refs=(str(skill_root),),
                )
            )
            continue
        cards.append(
            replace(card, metadata={**card.metadata, "deepmate-builtin": True})
        )
    return tuple(cards), tuple(warnings)


def discover_workspace_skill_cards(
    workspace: Path,
) -> tuple[tuple[SkillCard, ...], tuple[ContextWarning, ...]]:
    """Return valid workspace skill cards and non-fatal discovery warnings."""
    skill_roots = _workspace_skill_roots(workspace)
    if not skill_roots:
        return (), ()

    cards: list[SkillCard] = []
    warnings: list[ContextWarning] = []
    seen_names: set[str] = set()
    for skill_root in skill_roots:
        for path in sorted(skill_root.rglob("SKILL.md")):
            try:
                card = load_skill_card(path)
            except (OSError, ValueError) as exc:
                warnings.append(
                    ContextWarning(
                        code="skill_discovery_failed",
                        message=f"skill skipped: {exc}",
                        refs=(str(path),),
                    )
                )
                continue
            normalized_name = _normalize_name(card.name)
            directory_name = _normalize_name(card.path.parent.name)
            if normalized_name in seen_names or directory_name in seen_names:
                warnings.append(
                    ContextWarning(
                        code="duplicate_skill_name",
                        message=f"duplicate skill skipped: {card.name} at {card.path}",
                        refs=(str(card.path),),
                    )
                )
                continue
            seen_names.add(normalized_name)
            seen_names.add(directory_name)
            cards.append(card)
    return tuple(cards), tuple(warnings)


def discover_user_skill_cards(
    data_dir: Path | None,
) -> tuple[tuple[SkillCard, ...], tuple[ContextWarning, ...]]:
    """Return user-level skill cards installed under Deepmate data dir."""
    if data_dir is None:
        return (), ()
    root = user_skill_library_root(data_dir)
    if not root.exists():
        return (), ()
    cards: list[SkillCard] = []
    warnings: list[ContextWarning] = []
    seen_names: set[str] = set()
    for path in sorted(root.rglob("SKILL.md")):
        try:
            card = load_skill_card(path)
        except (OSError, ValueError) as exc:
            warnings.append(
                ContextWarning(
                    code="user_skill_discovery_failed",
                    message=f"user skill skipped: {exc}",
                    refs=(str(path),),
                )
            )
            continue
        normalized_name = _normalize_name(card.name)
        directory_name = _normalize_name(card.path.parent.name)
        if normalized_name in seen_names or directory_name in seen_names:
            warnings.append(
                ContextWarning(
                    code="duplicate_user_skill_name",
                    message=f"duplicate user skill skipped: {card.name} at {card.path}",
                    refs=(str(card.path),),
                )
            )
            continue
        seen_names.add(normalized_name)
        seen_names.add(directory_name)
        cards.append(card)
    return tuple(cards), tuple(warnings)


def select_skill_documents(
    catalog: SkillCatalog | None,
    names: Sequence[str],
    workspace: Path,
    command_name: str = "skill",
) -> tuple[SkillDocument, ...]:
    """Load explicitly selected skill documents from a discovered catalog."""
    selected: list[SkillDocument] = []
    seen: set[str] = set()
    skill_dir_exists = bool(_workspace_skill_roots(workspace))
    for raw_name in names:
        name = raw_name.strip()
        if not name:
            raise ValueError(f"{command_name} requires a non-empty skill name")
        if catalog is None:
            if skill_dir_exists:
                raise ValueError(f"skill not found: {name}")
            raise ValueError(
                f"{command_name} {name} requires a workspace skill directory"
            )
        card = catalog.get(name)
        if card is None:
            raise ValueError(f"skill not found: {name}")
        if card.name.strip() in seen:
            continue
        seen.add(card.name.strip())
        selected.append(load_skill_document(card))
    return tuple(selected)


def format_skill_list(cards: Sequence[SkillCard], workspace: Path) -> str:
    """Render skill cards for terminal display."""
    if not cards:
        return "No skills found."

    lines = [f"{'SKILL':<{SKILL_NAME_WIDTH}}  {'DESCRIPTION':<{SKILL_DESCRIPTION_WIDTH}}  PATH"]
    for card in cards:
        lines.append(
            f"{card.name:<{SKILL_NAME_WIDTH}}  "
            f"{_preview_text(card.description, SKILL_DESCRIPTION_WIDTH):<{SKILL_DESCRIPTION_WIDTH}}  "
            f"{display_path(card.path, workspace)}"
        )
    return "\n".join(lines)


def format_capability_list(
    cards: Sequence[SkillCard],
    workspace: Path,
    state_store: CapabilityStateStore,
) -> str:
    """Render governed skill capabilities for terminal display."""
    if not cards:
        return "No capabilities found."
    states = state_store.skill_states_by_name()
    lines = [
        f"{'SKILL':<{SKILL_NAME_WIDTH}}  "
        f"{'TEMP':<{SKILL_TEMPERATURE_WIDTH}}  "
        f"{'EXPOSURE':<{SKILL_EXPOSURE_WIDTH}}  "
        f"{'DESCRIPTION':<{SKILL_DESCRIPTION_WIDTH}}  PATH"
    ]
    for card in cards:
        state = None if card.is_builtin() else states.get(_normalize_name(card.name))
        lines.append(
            f"{card.name:<{SKILL_NAME_WIDTH}}  "
            f"{_temperature_text(state):<{SKILL_TEMPERATURE_WIDTH}}  "
            f"{_exposure_text(state):<{SKILL_EXPOSURE_WIDTH}}  "
            f"{_description_preview(card, state):<{SKILL_DESCRIPTION_WIDTH}}  "
            f"{display_path(card.path, workspace)}"
        )
    return "\n".join(lines)


def format_skill_document(document: SkillDocument, workspace: Path) -> str:
    """Render one loaded skill document for terminal display."""
    return "\n".join(
        (
            f"Skill: {document.name}",
            f"Description: {document.description}",
            f"Path: {display_path(document.path, workspace)}",
            "",
            "Instructions:",
            document.body,
        )
    )


def display_path(path: Path, root: Path) -> str:
    """Return path relative to the workspace when possible."""
    return foundation_display_path(path, root)


def _preview_text(value: str, limit: int) -> str:
    return compact_text(value, limit)


def _description_preview(card: SkillCard, state: CapabilityState | None) -> str:
    if state is not None and state.exposure() == "name-only":
        return ""
    if state is not None and state.exposure() == "not-loaded":
        return ""
    return _preview_text(card.description, SKILL_DESCRIPTION_WIDTH)


def _temperature_text(state: CapabilityState | None) -> str:
    return state.temperature.value if state is not None else "hot"


def _exposure_text(state: CapabilityState | None) -> str:
    return state.exposure() if state is not None else "name+description"


def _workspace_skill_roots(workspace: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in (
            workspace / "skills",
            workspace / ".claude" / "skills",
        )
        if path.exists()
    )


def _builtin_skill_roots() -> tuple[Path, ...]:
    try:
        root = resources.files("deepmate").joinpath("builtin_skills")
    except (ModuleNotFoundError, AttributeError):
        return ()
    path = Path(str(root))
    if not path.exists():
        return ()
    return tuple(sorted(child for child in path.iterdir() if child.is_dir()))


def _normalize_name(name: str) -> str:
    return normalize_name(name)
