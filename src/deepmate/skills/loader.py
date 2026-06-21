"""Load full skill documents on demand."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from deepmate.skills.catalog import SkillCard
from deepmate.skills.metadata import metadata_text, skill_description
from deepmate.skills.skill_file import (
    SKILL_FILE_NAME,
    read_skill_markdown,
    resolve_skill_file,
)


@dataclass(frozen=True, slots=True)
class SkillDocument:
    """Full SKILL.md document loaded after a skill card is selected."""

    name: str
    description: str
    body: str
    path: Path
    metadata: Mapping[str, object] = field(default_factory=dict)

    def is_ready(self) -> bool:
        """Return whether the loaded skill has usable instructions."""
        return bool(
            self.name.strip()
            and self.description.strip()
            and self.body.strip()
            and self.path.name == SKILL_FILE_NAME
        )


def load_skill_document(skill: SkillCard | str | Path) -> SkillDocument:
    """Load frontmatter metadata and markdown body from one SKILL.md."""
    skill_path = _skill_path(skill)
    metadata, body = read_skill_markdown(skill_path)
    document = SkillDocument(
        name=metadata_text(metadata, "name") or skill_path.parent.name,
        description=skill_description(metadata, body),
        body=body.strip(),
        path=skill_path,
        metadata=metadata,
    )
    if not document.is_ready():
        raise ValueError(f"{skill_path} must define description and body")
    return document


def _skill_path(skill: SkillCard | str | Path) -> Path:
    path = skill.path if isinstance(skill, SkillCard) else Path(skill)
    return resolve_skill_file(path)
