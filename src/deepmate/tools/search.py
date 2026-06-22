"""Bounded workspace search tools."""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

from deepmate.runtime.process_env import subprocess_environment
from deepmate.tools.filesystem import (
    DENIED_PATH_NAMES,
    INTERNAL_WORKSPACE_DIR_NAMES,
    _is_denied_path,
    _is_internal_workspace_dir,
    _relative_path,
    _workspace_path,
)
from deepmate.tools.registry import NativeTool, NativeToolResult

DEFAULT_MAX_MATCHES = 50
MAX_MATCHES = 500
DEFAULT_MAX_FILES = 100
MAX_FILES = 500
MAX_SEARCH_FILE_BYTES = 2_000_000


def workspace_search_tools(workspace_root: str | Path) -> tuple[NativeTool, ...]:
    """Return bounded content and file-name search tools for one workspace."""
    root = Path(workspace_root).resolve()
    if not root.is_dir():
        raise ValueError(f"workspace root must be a directory: {root}")
    return (
        NativeTool(
            name="search_content",
            description="Search text content inside the workspace with bounded results.",
            input_schema=_search_content_schema(),
            handler=lambda arguments: _search_content(root, arguments),
        ),
        NativeTool(
            name="search_files",
            description="Find workspace files or directories by glob pattern.",
            input_schema=_search_files_schema(),
            handler=lambda arguments: _search_files(root, arguments),
        ),
    )


def _search_content(root: Path, arguments: Mapping[str, object]) -> NativeToolResult:
    pattern = _text(arguments, "pattern")
    search_root = _workspace_path(root, _optional_text(arguments, "path", "."))
    if not search_root.exists():
        raise ValueError(f"path does not exist: {_relative_path(root, search_root)}")
    glob = _optional_nullable_text(arguments, "glob")
    literal = _bool(arguments, "literal", False)
    case_sensitive = _bool(arguments, "case_sensitive", False)
    max_matches = _int(
        arguments,
        "max_matches",
        DEFAULT_MAX_MATCHES,
        1,
        MAX_MATCHES,
    )

    if shutil.which("rg"):
        matches, truncated = _ripgrep_matches(
            root,
            search_root,
            pattern,
            glob=glob,
            literal=literal,
            case_sensitive=case_sensitive,
            max_matches=max_matches,
        )
        engine = "ripgrep"
    else:
        matches, truncated = _python_matches(
            root,
            search_root,
            pattern,
            glob=glob,
            literal=literal,
            case_sensitive=case_sensitive,
            max_matches=max_matches,
        )
        engine = "python"

    refs = tuple(dict.fromkeys(str(match["path"]) for match in matches))
    content = "\n".join(
        f"{match['path']}:{match['line']}:{match['column']}: {match['text']}"
        for match in matches
    )
    if not content:
        content = "No matches found."
    return NativeToolResult(
        content=content,
        data={
            "pattern": pattern,
            "path": _relative_path(root, search_root),
            "match_count": len(matches),
            "file_count": len(refs),
            "truncated": truncated,
            "engine": engine,
        },
        refs=refs,
    )


def _ripgrep_matches(
    root: Path,
    search_root: Path,
    pattern: str,
    *,
    glob: str | None,
    literal: bool,
    case_sensitive: bool,
    max_matches: int,
) -> tuple[list[Mapping[str, object]], bool]:
    command = [
        "rg",
        "--json",
        "--line-number",
        "--column",
        "--color",
        "never",
        "--max-filesize",
        f"{MAX_SEARCH_FILE_BYTES}",
        "--hidden",
    ]
    if literal:
        command.append("--fixed-strings")
    if not case_sensitive:
        command.append("--ignore-case")
    if glob:
        command.extend(("--glob", glob))
    for name in sorted(DENIED_PATH_NAMES | INTERNAL_WORKSPACE_DIR_NAMES):
        command.extend(("--glob", f"!**/{name}"))
        command.extend(("--glob", f"!**/{name}/**"))
    for suffix in (".key", ".p12", ".pem", ".pfx"):
        command.extend(("--glob", f"!**/*{suffix}"))
    command.extend(("--", pattern, str(search_root)))
    completed = subprocess.run(
        command,
        cwd=root,
        env=subprocess_environment(),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if completed.returncode not in {0, 1}:
        message = completed.stderr.strip() or "ripgrep failed"
        raise ValueError(message)

    matches: list[Mapping[str, object]] = []
    truncated = False
    for raw_line in completed.stdout.splitlines():
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "match":
            continue
        data = event.get("data")
        if not isinstance(data, Mapping):
            continue
        path_data = data.get("path")
        lines_data = data.get("lines")
        submatches = data.get("submatches")
        if not isinstance(path_data, Mapping) or not isinstance(lines_data, Mapping):
            continue
        raw_path = path_data.get("text")
        text = lines_data.get("text")
        if not isinstance(raw_path, str) or not isinstance(text, str):
            continue
        path = Path(raw_path).resolve()
        if _is_denied_path(root, path) or _has_internal_workspace_parent(root, path):
            continue
        column = 1
        if isinstance(submatches, list) and submatches:
            start = submatches[0].get("start")
            if isinstance(start, int):
                column = start + 1
        matches.append(
            {
                "path": _relative_path(root, path),
                "line": int(data.get("line_number") or 1),
                "column": column,
                "text": text.rstrip("\r\n"),
            }
        )
        if len(matches) >= max_matches:
            truncated = True
            break
    return matches, truncated


def _python_matches(
    root: Path,
    search_root: Path,
    pattern: str,
    *,
    glob: str | None,
    literal: bool,
    case_sensitive: bool,
    max_matches: int,
) -> tuple[list[Mapping[str, object]], bool]:
    flags = 0 if case_sensitive else re.IGNORECASE
    expression = re.compile(re.escape(pattern) if literal else pattern, flags)
    matches: list[Mapping[str, object]] = []
    for path in _iter_workspace_files(root, search_root):
        relative = _relative_path(root, path)
        if glob and not fnmatch.fnmatch(relative, glob):
            continue
        try:
            if path.stat().st_size > MAX_SEARCH_FILE_BYTES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "\x00" in text:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            match = expression.search(line)
            if match is None:
                continue
            matches.append(
                {
                    "path": relative,
                    "line": line_number,
                    "column": match.start() + 1,
                    "text": line,
                }
            )
            if len(matches) >= max_matches:
                return matches, True
    return matches, False


def _search_files(root: Path, arguments: Mapping[str, object]) -> NativeToolResult:
    pattern = _optional_text(arguments, "pattern", "**/*")
    search_root = _workspace_path(root, _optional_text(arguments, "path", "."))
    kind = _optional_text(arguments, "kind", "file").lower()
    if kind not in {"file", "directory", "any"}:
        raise ValueError("kind must be file, directory, or any")
    max_results = _int(arguments, "max_results", DEFAULT_MAX_FILES, 1, MAX_FILES)
    if not search_root.is_dir():
        raise ValueError(f"path is not a directory: {_relative_path(root, search_root)}")

    results: list[Mapping[str, str]] = []
    truncated = False
    for path in _iter_workspace_paths(root, search_root):
        relative_to_search = path.relative_to(search_root).as_posix()
        if not _matches_workspace_glob(relative_to_search, pattern):
            continue
        path_kind = _safe_path_kind(path)
        if path_kind is None:
            continue
        if kind != "any" and path_kind != kind:
            continue
        results.append({"path": _relative_path(root, path), "kind": path_kind})
        if len(results) >= max_results:
            truncated = True
            break
    content = "\n".join(
        f"{item['path']}/" if item["kind"] == "directory" else item["path"]
        for item in results
    )
    return NativeToolResult(
        content=content or "No files found.",
        data={
            "pattern": pattern,
            "path": _relative_path(root, search_root),
            "kind": kind,
            "result_count": len(results),
            "truncated": truncated,
        },
        refs=tuple(str(item["path"]) for item in results),
    )


def _matches_workspace_glob(relative_path: str, pattern: str) -> bool:
    patterns = (pattern, pattern[3:]) if pattern.startswith("**/") else (pattern,)
    return any(
        fnmatch.fnmatch(relative_path, candidate)
        or Path(relative_path).match(candidate)
        for candidate in patterns
    )


def _iter_workspace_files(root: Path, start: Path):
    for path in _iter_workspace_paths(root, start):
        if _safe_path_kind(path) == "file":
            yield path


def _iter_workspace_paths(root: Path, start: Path):
    for current, directories, files in os.walk(
        start,
        followlinks=False,
        onerror=lambda _error: None,
    ):
        current_path = Path(current)
        directories[:] = sorted(
            directory
            for directory in directories
            if _walkable_directory(root, current_path / directory)
        )
        for directory in directories:
            yield current_path / directory
        for filename in sorted(files):
            path = current_path / filename
            if not _safe_is_denied_path(root, path):
                yield path


def _walkable_directory(root: Path, path: Path) -> bool:
    name = path.name.lower()
    if name in DENIED_PATH_NAMES:
        return False
    try:
        if path.is_symlink():
            return False
        return not _safe_is_denied_path(root, path) and not _is_internal_workspace_dir(
            root,
            path,
        )
    except RuntimeError:
        return False


def _safe_is_denied_path(root: Path, path: Path) -> bool:
    try:
        return _is_denied_path(root, path)
    except RuntimeError:
        return True


def _has_internal_workspace_parent(root: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    return bool(relative.parts and relative.parts[0] in INTERNAL_WORKSPACE_DIR_NAMES)


def _safe_path_kind(path: Path) -> str | None:
    try:
        if path.is_symlink():
            return None
        if path.is_dir():
            return "directory"
        if path.is_file():
            return "file"
    except RuntimeError:
        return None
    return None


def _text(arguments: Mapping[str, object], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be non-empty text")
    return value.strip()


def _optional_text(arguments: Mapping[str, object], key: str, default: str) -> str:
    value = arguments.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be text")
    return value.strip() or default


def _optional_nullable_text(
    arguments: Mapping[str, object],
    key: str,
) -> str | None:
    value = arguments.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be non-empty text when provided")
    return value.strip()


def _bool(arguments: Mapping[str, object], key: str, default: bool) -> bool:
    value = arguments.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _int(
    arguments: Mapping[str, object],
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = arguments.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


def _search_content_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "Workspace-relative search root."},
            "glob": {"type": "string", "description": "Optional file glob filter."},
            "literal": {"type": "boolean"},
            "case_sensitive": {"type": "boolean"},
            "max_matches": {"type": "integer", "minimum": 1, "maximum": MAX_MATCHES},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }


def _search_files_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob such as **/*.py."},
            "path": {"type": "string", "description": "Workspace-relative search root."},
            "kind": {"type": "string", "enum": ["file", "directory", "any"]},
            "max_results": {"type": "integer", "minimum": 1, "maximum": MAX_FILES},
        },
        "additionalProperties": False,
    }
