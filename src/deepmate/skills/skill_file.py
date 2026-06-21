"""Shared SKILL.md file parsing helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

SKILL_FILE_NAME = "SKILL.md"
FRONTMATTER_BOUNDARY = "---"


def resolve_skill_file(path: str | Path) -> Path:
    """Return the SKILL.md path for a file or skill directory."""
    skill_path = Path(path)
    if skill_path.is_dir():
        skill_path = skill_path / SKILL_FILE_NAME
    if skill_path.name != SKILL_FILE_NAME:
        raise ValueError(f"Expected {SKILL_FILE_NAME}: {skill_path}")
    return skill_path


def read_skill_frontmatter(path: str | Path) -> Mapping[str, object]:
    """Read only SKILL.md frontmatter metadata."""
    metadata, _ = read_skill_markdown(path)
    return metadata


def read_skill_markdown(path: str | Path) -> tuple[Mapping[str, object], str]:
    """Read SKILL.md frontmatter metadata and markdown body."""
    skill_path = resolve_skill_file(path)
    lines = skill_path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != FRONTMATTER_BOUNDARY:
        return {}, "\n".join(lines).strip()

    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == FRONTMATTER_BOUNDARY:
            metadata = _parse_frontmatter_lines(lines[1:index])
            body = "\n".join(lines[index + 1 :]).strip()
            return metadata, body
    raise ValueError(f"{skill_path} frontmatter is missing closing ---")


def _parse_frontmatter_lines(lines: list[str]) -> Mapping[str, object]:
    metadata: dict[str, object] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            index += 1
            continue
        indent = _indent_width(line)
        if indent != 0:
            index += 1
            continue
        key, value = stripped.split(":", 1)
        clean_key = key.strip()
        clean_value = value.strip()
        child_lines, next_index = _collect_indented_block(lines, index + 1, indent)
        if _is_block_scalar(clean_value):
            metadata[clean_key] = _parse_block_scalar(
                child_lines,
                literal=clean_value.startswith("|"),
            )
            index = next_index
            continue
        if not clean_value and child_lines:
            metadata[clean_key] = _parse_nested_block(child_lines)
            index = next_index
            continue
        metadata[clean_key] = _parse_scalar(clean_value)
        index += 1
    return metadata


def _parse_nested_block(lines: list[str]) -> object:
    entries: dict[str, object] = {}
    sequence: list[object] = []
    mode: str | None = None
    index = 0
    base_indent = min(
        (_indent_width(line) for line in lines if line.strip()),
        default=0,
    )
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            index += 1
            continue
        indent = _indent_width(line)
        if indent < base_indent:
            break
        if indent > base_indent:
            index += 1
            continue
        if stripped.startswith("- "):
            mode = mode or "list"
            sequence.append(_parse_scalar(stripped[2:].strip()))
            index += 1
            continue
        if ":" not in stripped:
            index += 1
            continue
        mode = mode or "dict"
        key, value = stripped.split(":", 1)
        clean_value = value.strip()
        child_lines, next_index = _collect_indented_block(lines, index + 1, indent)
        if _is_block_scalar(clean_value):
            entries[key.strip()] = _parse_block_scalar(
                child_lines,
                literal=clean_value.startswith("|"),
            )
            index = next_index
            continue
        if not clean_value and child_lines:
            entries[key.strip()] = _parse_nested_block(child_lines)
            index = next_index
            continue
        entries[key.strip()] = _parse_scalar(clean_value)
        index += 1
    return sequence if mode == "list" else entries


def _collect_indented_block(
    lines: list[str],
    start: int,
    parent_indent: int,
) -> tuple[list[str], int]:
    block: list[str] = []
    index = start
    while index < len(lines):
        line = lines[index]
        if line.strip() and _indent_width(line) <= parent_indent:
            break
        block.append(line)
        index += 1
    return block, index


def _parse_block_scalar(lines: list[str], *, literal: bool) -> str:
    base_indent = min(
        (_indent_width(line) for line in lines if line.strip()),
        default=0,
    )
    normalized = [line[base_indent:] if len(line) >= base_indent else "" for line in lines]
    if literal:
        return "\n".join(normalized).strip()
    paragraphs = "\n".join(normalized).strip().split("\n\n")
    return "\n\n".join(" ".join(paragraph.split()) for paragraph in paragraphs).strip()


def _parse_scalar(value: str) -> object:
    clean = value.strip().strip("\"'")
    lowered = clean.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if clean.startswith("[") and clean.endswith("]"):
        inner = clean[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip("\"'") for item in inner.split(",")]
    return clean


def _is_block_scalar(value: str) -> bool:
    return value in {">", "|", ">-", "|-", ">+", "|+"}


def _indent_width(line: str) -> int:
    return len(line) - len(line.lstrip(" "))
