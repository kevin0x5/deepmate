"""Workspace file and diff helpers for the TUI."""

from __future__ import annotations

import codecs
import os
import shlex
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from time import monotonic
from pathlib import Path

from deepmate.tools.filesystem import INTERNAL_WORKSPACE_DIR_NAMES, _is_denied_path


IGNORED_DIRS = {
    ".git",
    ".hg",
    ".DS_Store",
    ".skillhub-cli",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    "skillhub-cli",
    "skillhub_install",
    "var",
} | set(INTERNAL_WORKSPACE_DIR_NAMES)
IGNORED_ROOT_FILES = {
    "skillhub_install.tar.gz",
    "skillhub-latest.tar.gz",
}
COLLAPSED_DIRS = {
    "src/deepmate.egg-info",
    "build",
}
IMPORTANT_ROOT_FILES = {
    "AGENTS.md",
    "README.md",
    "README.zh-CN.md",
    "pyproject.toml",
    "package.json",
}
GIT_STATUS_CACHE_TTL_SECONDS = 2.0
GIT_STATUS_CACHE_MAX_WORKSPACES = 32
_GIT_STATUS_CACHE: dict[Path, tuple[float, dict[str, str]]] = {}


@dataclass(frozen=True, slots=True)
class TuiFileItem:
    """One workspace file visible in the TUI sidebar."""

    path: Path
    relative_path: str
    badge: str = ""
    is_dir: bool = False


@dataclass(frozen=True, slots=True)
class TuiDiffFile:
    """One changed file in the workspace diff summary."""

    path: str
    status: str
    added: int | None = None
    deleted: int | None = None


@dataclass(frozen=True, slots=True)
class TuiFilePreview:
    """One bounded slice of a workspace file for TUI preview."""

    content: str
    path: str
    start: int
    end: int
    bytes_total: int
    truncated_before: bool = False
    truncated_after: bool = False

    def rendered_content(self) -> str:
        """Return preview content with explicit pagination markers."""
        lines: list[str] = []
        if self.truncated_before:
            lines.append(
                f"[Deepmate TUI showing bytes {self.start}-{self.end} "
                f"of {self.bytes_total}; earlier content omitted]"
            )
            lines.append("")
        lines.append(self.content)
        if self.truncated_after:
            quoted_path = shlex.quote(self.path)
            lines.append("")
            lines.append(
                f"[Deepmate TUI showing bytes {self.start}-{self.end} "
                f"of {self.bytes_total}; use /open {quoted_path} "
                f"--offset {self.end} to continue]"
            )
        return "\n".join(lines)


def workspace_file_items(
    workspace: str | Path,
    *,
    expanded_dirs: Iterable[str] = (),
    limit: int = 400,
) -> tuple[TuiFileItem, ...]:
    """Return a bounded list of project files with optional git badges."""
    root = Path(workspace).resolve()
    badges = git_status_badges(root)
    expanded = frozenset(path.strip("/") for path in expanded_dirs if path.strip("/"))
    items: list[TuiFileItem] = []
    truncated = False
    for path, is_dir in _iter_workspace_entries(root, expanded_dirs=expanded):
        rel = path.relative_to(root).as_posix()
        display_rel = f"{rel}/" if is_dir else rel
        items.append(
            TuiFileItem(
                path=path,
                relative_path=display_rel,
                badge="" if is_dir else badges.get(rel, ""),
                is_dir=is_dir,
            )
        )
        if len(items) >= limit:
            truncated = True
            break
    if truncated:
        items.append(
            TuiFileItem(
                path=root,
                relative_path=f"... more files; use /files to search ({limit}+ shown)",
                badge="",
                is_dir=False,
            )
        )
    return tuple(items)


def workspace_file_matches(
    workspace: str | Path,
    query: str = "",
    *,
    limit: int = 80,
    max_scan: int = 5_000,
) -> tuple[str, ...]:
    """Return matching workspace files without depending on sidebar expansion."""
    root = Path(workspace).resolve()
    clean_query = query.strip().lower()
    matches: list[str] = []
    scanned = 0
    if limit <= 0 or max_scan <= 0:
        return ()
    for path in _iter_workspace_files(root, root):
        scanned += 1
        if scanned > max_scan:
            break
        relative = path.relative_to(root).as_posix()
        if clean_query and clean_query not in relative.lower():
            continue
        matches.append(relative)
        if len(matches) >= limit:
            break
    return tuple(matches)


def read_workspace_file(
    workspace: str | Path,
    relative_path: str,
    *,
    max_chars: int = 80_000,
) -> str:
    """Read a workspace-relative file for preview."""
    return read_workspace_file_preview(
        workspace,
        relative_path,
        max_bytes=max_chars,
    ).rendered_content()


def read_workspace_file_preview(
    workspace: str | Path,
    relative_path: str,
    *,
    offset: int = 0,
    max_bytes: int = 80_000,
) -> TuiFilePreview:
    """Read a bounded byte slice of a workspace file for preview."""
    root = Path(workspace).resolve()
    target = (root / relative_path).resolve()
    if target != root and root not in target.parents:
        raise ValueError("file preview path must stay inside the workspace")
    if _is_denied_path(root, target):
        raise ValueError("file is protected and cannot be previewed")
    if not target.is_file():
        raise ValueError(f"file not found: {relative_path}")
    safe_offset = max(0, int(offset))
    safe_max = max(1, int(max_bytes))
    with target.open("rb") as file:
        total = os.fstat(file.fileno()).st_size
        start = min(safe_offset, total)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        lookbehind = min(3, start)
        file.seek(start - lookbehind)
        decoder.decode(file.read(lookbehind))
        prefix = decoder.getstate()[0]
        if prefix and start < total and b"\x80" <= file.peek(1)[:1] < b"\xc0":
            start -= len(prefix)
            file.seek(start)
        decoder.reset()
        data = file.read(safe_max)
        content = decoder.decode(data, final=len(data) < safe_max or file.tell() >= total)
        # With a one-byte budget, finish the first character so every page advances.
        while not content and data and file.tell() < total:
            following = file.read(1)
            content = decoder.decode(following, final=not following or file.tell() >= total)
            if not following:
                break
        end = file.tell() - len(decoder.getstate()[0])
    return TuiFilePreview(
        content=content,
        path=relative_path,
        start=start,
        end=end,
        bytes_total=total,
        truncated_before=start > 0,
        truncated_after=end < total,
    )


def workspace_diff(workspace: str | Path, *, max_chars: int = 100_000) -> str:
    """Return a readable diff report for the workspace."""
    root = Path(workspace).resolve()
    status_result = _git(root, "status", "--short")
    if status_result is None:
        return "No git diff is available for this workspace."
    if status_result.returncode != 0:
        message = (status_result.stderr or status_result.stdout).strip()
        return _git_unavailable_message(message)
    status_files = _parse_status_files(status_result.stdout)
    if not status_files:
        return "No workspace diff."
    statuses = [line[:2] for line in status_result.stdout.splitlines() if len(line) >= 4]
    sections = []
    numstat: dict[str, tuple[int | None, int | None]] = {}
    for column, title, flags in ((0, "Staged", ("--cached",)), (1, "Working tree", ())):
        if not any(status[column] not in " ?!" for status in statuses):
            continue
        stats = _git(root, "diff", *flags, "--numstat", "--no-ext-diff", "--")
        if stats is not None and stats.returncode == 0:
            for path, counts in _parse_numstat(stats.stdout).items():
                previous = numstat.get(path)
                if previous is None or None in counts or None in previous:
                    numstat[path] = counts
                else:
                    numstat[path] = (previous[0] + counts[0], previous[1] + counts[1])
        result = _git(root, "diff", *flags, "--no-ext-diff", "--")
        if result is None:
            return "No git diff is available for this workspace."
        if result.returncode:
            return _git_unavailable_message((result.stderr or result.stdout).strip())
        if result.stdout.strip():
            sections.append(f"{title}\n{result.stdout.rstrip()}")
    changed_files = tuple(
        TuiDiffFile(file.path, file.status, *_numstat_for_path(numstat, file.path))
        for file in status_files
    )
    diff = "\n\n".join(sections)
    if len(diff) > max_chars:
        diff = diff[: max_chars - 35].rstrip() + "\n\n[diff truncated by Deepmate TUI]"
    return _render_diff_report(changed_files, diff)


def git_status_badges(workspace: str | Path) -> dict[str, str]:
    """Return git status badges keyed by workspace-relative path."""
    root = Path(workspace).resolve()
    now = monotonic()
    _prune_git_status_cache(now)
    cached = _GIT_STATUS_CACHE.get(root)
    if cached is not None and now - cached[0] <= GIT_STATUS_CACHE_TTL_SECONDS:
        return dict(cached[1])
    result = _git(root, "status", "--short")
    if result is None or result.returncode != 0:
        return {}
    badges: dict[str, str] = {}
    for file in _parse_status_files(result.stdout):
        badges[file.path] = "N" if file.status in {"new", "untracked"} else "M"
    _GIT_STATUS_CACHE[root] = (now, dict(badges))
    _prune_git_status_cache(now)
    return badges


def _prune_git_status_cache(now: float) -> None:
    """Keep the short-lived git status cache bounded."""
    expired = tuple(
        root
        for root, (created_at, _) in _GIT_STATUS_CACHE.items()
        if now - created_at > GIT_STATUS_CACHE_TTL_SECONDS
    )
    for root in expired:
        _GIT_STATUS_CACHE.pop(root, None)
    overflow = len(_GIT_STATUS_CACHE) - GIT_STATUS_CACHE_MAX_WORKSPACES
    if overflow <= 0:
        return
    oldest = sorted(
        _GIT_STATUS_CACHE,
        key=lambda root: _GIT_STATUS_CACHE[root][0],
    )
    for root in oldest[:overflow]:
        _GIT_STATUS_CACHE.pop(root, None)


def _git(
    root: Path,
    *args: str,
    timeout: int = 10,
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ("git", "-C", str(root), "-c", "core.quotepath=false", *args),
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _parse_status_files(output: str) -> tuple[TuiDiffFile, ...]:
    files: list[TuiDiffFile] = []
    for line in output.splitlines():
        if len(line) < 4:
            continue
        status = line[:2]
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1].strip()
        if not path:
            continue
        files.append(TuiDiffFile(path=path, status=_status_label(status)))
    return tuple(files)


def _parse_numstat(output: str) -> dict[str, tuple[int | None, int | None]]:
    values: dict[str, tuple[int | None, int | None]] = {}
    for line in output.splitlines():
        parts = line.split("\t", 2)
        if len(parts) < 3:
            continue
        added = _int_or_none(parts[0])
        deleted = _int_or_none(parts[1])
        path = _normalize_diff_path(parts[2].strip())
        if path:
            values[path] = (added, deleted)
    return values


def _numstat_for_path(
    values: dict[str, tuple[int | None, int | None]],
    path: str,
) -> tuple[int | None, int | None]:
    for candidate in _diff_path_candidates(path):
        if candidate in values:
            return values[candidate]
    return (None, None)


def _diff_path_candidates(path: str) -> tuple[str, ...]:
    clean = path.strip()
    candidates = [clean]
    if " -> " in clean:
        old, _, new = clean.partition(" -> ")
        candidates.extend((old.strip(), new.strip()))
    normalized = _normalize_diff_path(clean)
    if normalized and normalized not in candidates:
        candidates.append(normalized)
    return tuple(candidate for candidate in candidates if candidate)


def _normalize_diff_path(path: str) -> str:
    clean = path.strip()
    if " => " in clean:
        prefix, _, suffix = clean.partition(" => ")
        if "{" in prefix and "}" in suffix:
            before_base, _, before_tail = prefix.partition("{")
            after_tail, _, after_suffix = suffix.partition("}")
            return f"{before_base}{after_tail}{after_suffix}".strip()
        return suffix.strip()
    return clean


def _render_diff_report(files: tuple[TuiDiffFile, ...], diff: str) -> str:
    count = len(files)
    summary = f"{count} file{'s' if count != 1 else ''} changed"
    lines = [summary]
    if files:
        added = sum(file.added or 0 for file in files)
        deleted = sum(file.deleted or 0 for file in files)
        if count > 1 and (added or deleted):
            lines[0] += f"  ·  +{added} -{deleted}"
        for file in files[:40]:
            line = f"  {file.status}: {file.path}"
            if file.added is not None or file.deleted is not None:
                line += f" (+{file.added or 0}/-{file.deleted or 0})"
            lines.append(line)
        if len(files) > 40:
            lines.append(f"- ... +{len(files) - 40} more")
    lines.append("")
    if diff:
        lines.append(diff)
    else:
        lines.append("(No tracked raw diff. Untracked files are listed above.)")
    return "\n".join(lines).strip()


def _status_label(status: str) -> str:
    clean = status.strip()
    if "?" in status:
        return "untracked"
    if "A" in status:
        return "new"
    if "D" in status:
        return "deleted"
    if "R" in status:
        return "renamed"
    if "C" in status:
        return "copied"
    if "M" in status:
        return "modified"
    return clean or "changed"


def _int_or_none(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _git_unavailable_message(message: str) -> str:
    if not message:
        return "No git diff is available for this workspace."
    return f"No git diff is available for this workspace.\n{message}"


def _iter_workspace_entries(root: Path, *, expanded_dirs: frozenset[str]):
    yield from _iter_workspace_entries_recursive(root, root, expanded_dirs=expanded_dirs)


def _iter_workspace_files(root: Path, current: Path):
    try:
        entries = tuple(current.iterdir())
    except OSError:
        return
    directories = sorted(
        (
            path
            for path in entries
            if _is_traversable_workspace_dir(root, path)
        ),
        key=lambda path: path.name.lower(),
    )
    files = sorted(
        (
            path
            for path in entries
            if _is_visible_workspace_file(root, path)
        ),
        key=_file_path_sort_key,
    )
    for file_path in files:
        yield file_path
    for directory in directories:
        yield from _iter_workspace_files(root, directory)


def _iter_workspace_entries_recursive(
    root: Path,
    current: Path,
    *,
    expanded_dirs: frozenset[str],
):
    current_rel = "" if current == root else current.relative_to(root).as_posix()
    if _is_collapsed_path(current_rel):
        return
    try:
        entries = tuple(current.iterdir())
    except OSError:
        return
    directories = sorted(
        (
            path
            for path in entries
            if _is_traversable_workspace_dir(root, path)
        ),
        key=lambda path: path.name.lower(),
    )
    files = sorted(
        (
            path
            for path in entries
            if _is_visible_workspace_file(root, path)
        ),
        key=_file_path_sort_key,
    )
    for directory in directories:
        relative_path = directory.relative_to(root).as_posix()
        yield directory, True
        if relative_path in expanded_dirs:
            yield from _iter_workspace_entries_recursive(
                root,
                directory,
                expanded_dirs=expanded_dirs,
            )
    if current_rel and current_rel not in expanded_dirs:
        return
    for file_path in files:
        yield file_path, False


def _is_traversable_workspace_dir(root: Path, path: Path) -> bool:
    return (
        path.name not in IGNORED_DIRS
        and path.is_dir()
        and not path.is_symlink()
        and not _is_collapsed_path(path.relative_to(root).as_posix())
    )


def _is_visible_workspace_file(root: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    return (
        path.name not in IGNORED_DIRS
        and not (len(relative.parts) == 1 and path.name in IGNORED_ROOT_FILES)
        and path.is_file()
        and not path.is_symlink()
    )


def _is_collapsed_path(relative_path: str) -> bool:
    clean = relative_path.strip("/")
    return any(clean == prefix or clean.startswith(f"{prefix}/") for prefix in COLLAPSED_DIRS)


def _file_path_sort_key(path: Path) -> tuple[int, str]:
    if path.name in IMPORTANT_ROOT_FILES:
        return (0, path.name.lower())
    return (1, path.name.lower())
