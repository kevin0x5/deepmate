"""Shared helpers for SKILL.md metadata display text."""

from __future__ import annotations

from collections.abc import Mapping

from deepmate.foundation import compact_text


def skill_description(
    metadata: Mapping[str, object],
    body: str,
    *,
    max_chars: int | None = None,
) -> str:
    """Return model-facing skill description from frontmatter and body."""
    description = metadata_text(metadata, "description")
    if not description:
        description = first_body_paragraph(body)
    when_to_use = metadata_text(metadata, "when_to_use")
    if when_to_use and when_to_use not in description:
        description = f"{description} {when_to_use}".strip()
    clean = " ".join(description.split())
    return compact_text(clean, max_chars) if max_chars is not None else clean


def first_body_paragraph(body: str) -> str:
    """Return the first non-heading paragraph from a skill body."""
    paragraph_lines: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            if paragraph_lines:
                break
            continue
        if stripped.startswith("#"):
            continue
        paragraph_lines.append(stripped)
    return " ".join(paragraph_lines)


def metadata_text(metadata: Mapping[str, object], key: str) -> str:
    """Return one trimmed text metadata field."""
    value = metadata.get(key, "")
    return value.strip() if isinstance(value, str) else ""
