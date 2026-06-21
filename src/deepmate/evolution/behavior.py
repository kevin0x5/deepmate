"""Markdown behavior hint helpers.

Behavior hints are short collaboration instructions maintained by Deepmate.
They are intentionally Markdown-native so the runtime can inject them without
maintaining a separate JSON behavior profile.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from deepmate.domain import ProfileRef
from deepmate.foundation import estimate_text_tokens

BEHAVIOR_FILE = "behavior.md"
BEHAVIOR_HEADING = "# Behavior Hints"
WORKSPACE_BEHAVIOR_REF = "workspace_behavior"
PROFILE_BEHAVIOR_REF = "profile_behavior"


@dataclass(frozen=True, slots=True)
class BehaviorHintDocument:
    """One behavior.md source and the hints it contributes."""

    name: str
    path: Path
    status: str
    hints: tuple[str, ...] = ()
    size_bytes: int = 0
    sha256: str = ""
    estimated_tokens: int = 0

    def is_loaded(self) -> bool:
        """Return whether this document contributes prompt content."""
        return self.status == "loaded" and bool(self.hints)


def workspace_behavior_path(workspace: str | Path) -> Path:
    """Return the workspace-scoped behavior.md path."""
    return Path(workspace) / ".deepmate" / BEHAVIOR_FILE


def profile_behavior_path(workspace: str | Path, profile: ProfileRef | str) -> Path:
    """Return the profile-scoped behavior.md path."""
    workspace_path = Path(workspace)
    if isinstance(profile, ProfileRef):
        profile_path = Path(profile.uri)
    else:
        clean_profile = str(profile).strip() or "default"
        profile_path = Path("profiles") / clean_profile
    if not profile_path.is_absolute():
        profile_path = workspace_path / profile_path
    return profile_path / BEHAVIOR_FILE


def read_behavior_hint_documents(
    workspace: str | Path,
    profile: ProfileRef | str,
) -> tuple[BehaviorHintDocument, ...]:
    """Read workspace then profile behavior hint documents."""
    return (
        _read_behavior_hint_document(
            WORKSPACE_BEHAVIOR_REF,
            workspace_behavior_path(workspace),
        ),
        _read_behavior_hint_document(
            PROFILE_BEHAVIOR_REF,
            profile_behavior_path(workspace, profile),
        ),
    )


def render_collaboration_hints(documents: tuple[BehaviorHintDocument, ...]) -> str:
    """Render all maintained hints as one system prompt section."""
    hints: list[str] = []
    for document in documents:
        hints.extend(document.hints)
    if not hints:
        return ""
    lines = ["<collaboration_hints>"]
    lines.extend(f"- {hint}" for hint in hints)
    lines.append("</collaboration_hints>")
    return "\n".join(lines)


def extract_behavior_hints(markdown: str) -> tuple[str, ...]:
    """Extract top-level bullets under the # Behavior Hints section."""
    lines = markdown.splitlines()
    in_section = False
    current: list[str] = []
    hints: list[str] = []

    def flush_current() -> None:
        if not current:
            return
        hint = _clean_hint(" ".join(current))
        if hint:
            hints.append(hint)
        current.clear()

    for line in lines:
        stripped = line.strip()
        if _is_markdown_heading(stripped):
            if in_section:
                flush_current()
                break
            in_section = stripped.lower() == BEHAVIOR_HEADING.lower()
            continue
        if not in_section:
            continue
        if line.startswith("- "):
            flush_current()
            current.append(line[2:].strip())
            continue
        if current and (line.startswith("  ") or line.startswith("\t")) and stripped:
            current.append(stripped)
            continue
        flush_current()
    flush_current()
    return tuple(hints)


def render_behavior_hints(hints: tuple[str, ...]) -> str:
    """Render a complete Behavior Hints section."""
    clean_hints = normalize_behavior_hints(hints)
    if not clean_hints:
        return f"{BEHAVIOR_HEADING}\n"
    lines = [BEHAVIOR_HEADING, ""]
    lines.extend(f"- {hint}" for hint in clean_hints)
    return "\n".join(lines) + "\n"


def replace_behavior_hints_section(markdown: str, hints: tuple[str, ...]) -> str:
    """Return markdown with only the Behavior Hints section replaced."""
    replacement = render_behavior_hints(hints).splitlines()
    lines = markdown.splitlines()
    start = _behavior_heading_index(lines)
    if start is None:
        prefix = markdown.rstrip()
        if not prefix:
            return "\n".join(replacement) + "\n"
        return prefix + "\n\n" + "\n".join(replacement) + "\n"

    end = start + 1
    while end < len(lines):
        if _is_markdown_heading(lines[end].strip()):
            break
        end += 1
    new_lines = [*lines[:start], *replacement, *lines[end:]]
    return "\n".join(new_lines).rstrip() + "\n"


def normalize_behavior_hints(hints: tuple[str, ...]) -> tuple[str, ...]:
    """Clean and de-duplicate behavior hints while preserving order."""
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_hint in hints:
        hint = _clean_hint(raw_hint)
        key = hint.lower()
        if not hint or key in seen:
            continue
        seen.add(key)
        normalized.append(hint)
    return tuple(normalized)


def _read_behavior_hint_document(name: str, path: Path) -> BehaviorHintDocument:
    if not path.exists():
        return BehaviorHintDocument(name=name, path=path, status="missing_optional")
    raw_content = path.read_text(encoding="utf-8")
    content = raw_content.strip()
    size_bytes = len(raw_content.encode("utf-8"))
    sha256 = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
    if not content:
        return BehaviorHintDocument(
            name=name,
            path=path,
            status="empty_optional",
            size_bytes=size_bytes,
            sha256=sha256,
        )
    hints = extract_behavior_hints(raw_content)
    if not hints:
        return BehaviorHintDocument(
            name=name,
            path=path,
            status="no_behavior_hints",
            size_bytes=size_bytes,
            sha256=sha256,
            estimated_tokens=estimate_text_tokens(content),
        )
    rendered = "\n".join(f"- {hint}" for hint in hints)
    return BehaviorHintDocument(
        name=name,
        path=path,
        status="loaded",
        hints=hints,
        size_bytes=size_bytes,
        sha256=sha256,
        estimated_tokens=estimate_text_tokens(rendered),
    )


def _behavior_heading_index(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        if line.strip().lower() == BEHAVIOR_HEADING.lower():
            return index
    return None


def _is_markdown_heading(stripped_line: str) -> bool:
    if not stripped_line.startswith("#"):
        return False
    level = 0
    for char in stripped_line:
        if char != "#":
            break
        level += 1
    return 1 <= level <= 6 and (
        len(stripped_line) == level or stripped_line[level].isspace()
    )


def _clean_hint(value: str) -> str:
    return " ".join(value.strip().lstrip("-").strip().split())
