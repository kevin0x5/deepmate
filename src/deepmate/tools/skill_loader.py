"""Native tool for loading skill instructions on demand."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from deepmate.capabilities.state import CapabilityStateStore
from deepmate.foundation import display_path, normalize_name
from deepmate.skills import SkillCatalog, SkillDocument, load_skill_document
from deepmate.tools.registry import NativeTool, NativeToolResult

LOAD_SKILL_TOOL_NAME = "load_skill"


def skill_loader_tools(
    catalog: SkillCatalog | None,
    workspace: str | Path,
    state_store: CapabilityStateStore | None = None,
    catalog_provider: Callable[[], SkillCatalog | None] | None = None,
) -> tuple[NativeTool, ...]:
    """Return the skill progressive-disclosure native tool when skills exist."""
    if catalog_provider is None and (catalog is None or not catalog.list_cards()):
        return ()
    active_catalog_provider = catalog_provider or (lambda: catalog)
    root = Path(workspace).resolve()
    return (
        NativeTool(
            name=LOAD_SKILL_TOOL_NAME,
            description=(
                "Load the full SKILL.md instructions for a relevant listed skill. "
                "Call this before following a skill whose name or description matches "
                "the user's task."
            ),
            input_schema=_load_skill_schema(),
            handler=lambda arguments: _load_skill(
                catalog_provider=active_catalog_provider,
                workspace=root,
                state_store=state_store,
                arguments=arguments,
            ),
        ),
    )


def _load_skill(
    *,
    catalog_provider: Callable[[], SkillCatalog | None],
    workspace: Path,
    state_store: CapabilityStateStore | None,
    arguments: Mapping[str, object],
) -> NativeToolResult:
    name = _text_argument(arguments, "name")
    catalog = catalog_provider()
    if catalog is None or not catalog.list_cards():
        raise ValueError(
            "no skills are available yet. Install a SKILL.md bundle or add one "
            "under the workspace skills directory first."
        )
    card = catalog.get(name)
    if card is None:
        raise ValueError(f"skill not found: {name}")
    workspace_skill = _is_workspace_skill(card.path, workspace)
    state = _skill_state(state_store, card.name) if workspace_skill else None
    if state is not None and not state.is_exposed_by_default():
        raise ValueError(
            f"skill is not exposed by default: {card.name} "
            f"(exposure={state.exposure()})"
        )

    document = load_skill_document(card)
    updated_state = (
        state_store.record_skill_selected(document.name)
        if state_store is not None and workspace_skill
        else None
    )
    refs = [
        f"skill={document.name}",
        f"path={_display_path(document.path, workspace)}",
    ]
    if updated_state is not None:
        refs.extend(
            (
                f"capability_id={updated_state.capability_id}",
                f"temperature={updated_state.temperature.value}",
                f"exposure={updated_state.exposure()}",
                f"invocation_count={updated_state.invocation_count}",
            )
        )
    return NativeToolResult(
        content=_skill_content(document),
        data={
            "skill": document.name,
            "path": _display_path(document.path, workspace),
            "description": document.description,
        },
        refs=tuple(refs),
    )


def _load_skill_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Exact skill name from the available skills list.",
            }
        },
        "required": ["name"],
        "additionalProperties": False,
    }


def _skill_state(state_store: CapabilityStateStore | None, name: str):
    if state_store is None:
        return None
    return state_store.skill_states_by_name().get(_normalize_name(name))


def _skill_content(document: SkillDocument) -> str:
    return "\n".join(
        (
            "<skill>",
            f"<name>{document.name.strip()}</name>",
            f"<description>{document.description.strip()}</description>",
            "<instructions>",
            _render_skill_body(document),
            "</instructions>",
            "</skill>",
        )
    )


def _render_skill_body(document: SkillDocument) -> str:
    return _expand_skill_dir_placeholders(
        document.body.strip(),
        document.path.parent,
    )


def _expand_skill_dir_placeholders(body: str, skill_dir: Path) -> str:
    replacement = str(skill_dir)
    return (
        body.replace("$SKILL_DIR", replacement)
        .replace("${SKILL_DIR}", replacement)
        .replace("${SKILL_ROOT}", replacement)
    )


def _display_path(path: Path, root: Path) -> str:
    return display_path(path, root)


def _is_workspace_skill(path: Path, root: Path) -> bool:
    try:
        resolved = path.resolve()
        workspace = root.resolve()
    except OSError:
        return False
    return resolved == workspace or workspace in resolved.parents


def _text_argument(arguments: Mapping[str, object], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{LOAD_SKILL_TOOL_NAME} requires text argument: {name}")
    return value.strip()


def _normalize_name(name: str) -> str:
    return normalize_name(name)
