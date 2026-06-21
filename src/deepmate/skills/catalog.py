"""Skill discovery catalog."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from deepmate.foundation import normalize_name
from deepmate.skills.metadata import metadata_text, skill_description
from deepmate.skills.skill_file import (
    SKILL_FILE_NAME,
    read_skill_markdown,
    resolve_skill_file,
)

SKILL_DESCRIPTION_MAX_CHARS = 1_536


@dataclass(frozen=True, slots=True)
class SkillCard:
    """Minimal skill summary visible before loading the full skill body."""

    name: str
    description: str
    path: Path
    metadata: Mapping[str, object] = field(default_factory=dict)

    def is_ready(self) -> bool:
        """Return whether the skill has enough metadata to be listed."""
        return bool(
            self.name.strip()
            and self.description.strip()
            and self.path.name == SKILL_FILE_NAME
        )

    def is_model_invocable(self) -> bool:
        """Return whether the skill may appear in model-driven discovery."""
        return not _metadata_bool(self.metadata, "disable-model-invocation")

    def is_builtin(self) -> bool:
        """Return whether Deepmate owns this bundled built-in skill."""
        return _metadata_bool(self.metadata, "deepmate-builtin")


class SkillCatalog:
    """In-memory catalog of discovered skill cards."""

    def __init__(self, cards: Iterable[SkillCard] = ()) -> None:
        self._cards = tuple(cards)
        self._by_name = _index_cards(self._cards)

    @classmethod
    def from_paths(cls, paths: Iterable[str | Path]) -> "SkillCatalog":
        """Build a catalog from SKILL.md files or directories containing them."""
        return cls(load_skill_card(path) for path in _iter_skill_files(paths))

    def list_cards(self) -> tuple[SkillCard, ...]:
        """Return discovered skill cards without loading skill bodies."""
        return self._cards

    def get(self, name: str) -> SkillCard | None:
        """Return one skill card by exact normalized skill name."""
        return self._by_name.get(_normalize_name(name))


def load_skill_card(path: str | Path) -> SkillCard:
    """Load only the frontmatter metadata needed for skill discovery."""
    skill_path = resolve_skill_file(path)
    metadata, body = read_skill_markdown(skill_path)
    card = SkillCard(
        name=metadata_text(metadata, "name") or skill_path.parent.name,
        description=skill_description(
            metadata,
            body,
            max_chars=SKILL_DESCRIPTION_MAX_CHARS,
        ),
        path=skill_path,
        metadata=metadata,
    )
    if not card.is_ready():
        raise ValueError(
            f"{skill_path} must define frontmatter description or body"
        )
    return card


def _iter_skill_files(paths: Iterable[str | Path]) -> tuple[Path, ...]:
    skill_files: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            candidates = [path]
        elif (path / SKILL_FILE_NAME).is_file():
            candidates = [path / SKILL_FILE_NAME]
        else:
            candidates = sorted(path.rglob(SKILL_FILE_NAME))
        for candidate in candidates:
            skill_path = resolve_skill_file(candidate)
            key = skill_path.resolve()
            if key not in seen:
                seen.add(key)
                skill_files.append(skill_path)
    return tuple(skill_files)


def _index_cards(cards: Iterable[SkillCard]) -> dict[str, SkillCard]:
    by_name: dict[str, SkillCard] = {}
    for card in cards:
        if not card.is_ready():
            raise ValueError("SkillCard requires name, description, and a SKILL.md path")
        keys = {_normalize_name(card.name), _normalize_name(card.path.parent.name)}
        for key in keys:
            if key in by_name:
                raise ValueError(f"Duplicate skill name: {card.name}")
        for key in keys:
            by_name[key] = card
    return by_name


def _metadata_bool(metadata: Mapping[str, object], key: str) -> bool:
    value = metadata.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def _normalize_name(name: str) -> str:
    return normalize_name(name)
