"""Read-only workspace filesystem tools."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import difflib
import os
from pathlib import Path
import stat

from deepmate.storage.atomic import atomic_write_text
from deepmate.runtime.hooks import (
    HookActor,
    HookDirective,
    HookEnvelope,
    HookEvent,
    HookOutcome,
    HookRuntimeContext,
)
from deepmate.tools.registry import NativeTool, NativeToolResult

DEFAULT_MAX_CHARS = 20_000
MAX_CHARS = 100_000
MAX_WRITE_CHARS = 200_000
MAX_READ_OFFSET = 50_000_000
DEFAULT_MAX_ENTRIES = 100
MAX_ENTRIES = 500
DENIED_PATH_NAMES = frozenset(
    {
        ".aws",
        ".codebuddy",
        ".env",
        ".env.development",
        ".env.local",
        ".env.production",
        ".git",
        ".hg",
        ".ssh",
        ".svn",
        ".npmrc",
        ".pypirc",
        "__pycache__",
        "var",
    }
)
INTERNAL_WORKSPACE_DIR_NAMES = frozenset(
    {
        ".deepmate",
        "checkpoints",
        "config",
        "cron",
        "profiles",
        "qa",
        "reports",
        "task",
        "traces",
        "transcripts",
        "var",
    }
)
DENIED_SUFFIXES = (".key", ".p12", ".pem", ".pfx")
WorkspaceWriteCheckpoint = Callable[[str, Path, str], object]


def workspace_filesystem_tools(
    workspace_root: str | Path,
    include_write_tools: bool = False,
    write_checkpoint: WorkspaceWriteCheckpoint | None = None,
    hook_context: HookRuntimeContext | None = None,
) -> tuple[NativeTool, ...]:
    """Return native filesystem tools constrained to one workspace root."""
    root = Path(workspace_root).resolve()
    if not root.is_dir():
        raise ValueError(f"workspace root must be a directory: {root}")
    tools = [
        NativeTool(
            name="read_text_file",
            description="Read a UTF-8 text file inside the current workspace.",
            input_schema=_read_text_file_schema(),
            handler=lambda arguments: _read_text_file(root, arguments),
        ),
        NativeTool(
            name="list_directory",
            description="List direct children of a directory inside the current workspace.",
            input_schema=_list_directory_schema(),
            handler=lambda arguments: _list_directory(root, arguments),
        ),
    ]
    if include_write_tools:
        tools.extend(
            (
                NativeTool(
                    name="write_text_file",
                    description=(
                        "Create or overwrite a UTF-8 text file inside the current workspace."
                    ),
                    input_schema=_write_text_file_schema(),
                    handler=lambda arguments: _write_text_file(
                        root,
                        arguments,
                        write_checkpoint=write_checkpoint,
                        hook_context=hook_context,
                    ),
                    read_only=False,
                ),
                NativeTool(
                    name="edit_text_file",
                    description=(
                        "Replace one exact text fragment in a UTF-8 file inside the current workspace."
                    ),
                    input_schema=_edit_text_file_schema(),
                    handler=lambda arguments: _edit_text_file(
                        root,
                        arguments,
                        write_checkpoint=write_checkpoint,
                        hook_context=hook_context,
                    ),
                    read_only=False,
                ),
            )
        )
    return tuple(tools)


def _read_text_file(root: Path, arguments: Mapping[str, object]) -> NativeToolResult:
    path = _workspace_path(root, _text_argument(arguments, "path"))
    max_chars = _int_argument(arguments, "max_chars", DEFAULT_MAX_CHARS, 1, MAX_CHARS)
    offset = _int_argument(arguments, "offset", 0, 0, MAX_READ_OFFSET)

    # Read a bounded character window starting at `offset` so callers can page
    # through long files. Decode with replacement so non-UTF-8 content (GBK,
    # latin-1, binary noise) is readable instead of crashing the tool.
    with _open_workspace_text_file(root, path, errors="replace") as file:
        bytes_total = os.fstat(file.fileno()).st_size
        if offset:
            skipped = file.read(offset)
            if len(skipped) < offset:
                # Offset is past the end of the file; nothing left to read.
                relative_path = _relative_path(root, path)
                return NativeToolResult(
                    content="",
                    data={
                        "path": relative_path,
                        "chars_read": 0,
                        "offset": offset,
                        "offset_unit": "chars",
                        "next_offset": None,
                        "chars_total": len(skipped),
                        "chars_total_known": True,
                        "bytes_total": bytes_total,
                        "truncated": False,
                    },
                    refs=(relative_path,),
                )
        window = file.read(max_chars + 1)
    content = window[:max_chars]
    truncated = len(window) > max_chars
    relative_path = _relative_path(root, path)
    return NativeToolResult(
        content=content,
        data={
            "path": relative_path,
            "chars_read": len(content),
            "offset": offset,
            "offset_unit": "chars",
            "next_offset": offset + len(content) if truncated else None,
            "chars_total": None if truncated else offset + len(content),
            "chars_total_known": not truncated,
            "bytes_total": bytes_total,
            "truncated": truncated,
        },
        refs=(relative_path,),
    )


def _list_directory(root: Path, arguments: Mapping[str, object]) -> NativeToolResult:
    raw_path = _optional_text_argument(arguments, "path", ".")
    path = _workspace_path(root, raw_path)
    max_entries = _int_argument(
        arguments,
        "max_entries",
        DEFAULT_MAX_ENTRIES,
        1,
        MAX_ENTRIES,
    )
    if not path.is_dir():
        raise ValueError(f"path is not a directory: {path}")

    children = tuple(path.iterdir())
    allowed_children = sorted(
        (
            child
            for child in children
            if not _is_denied_path(root, child)
            and not _is_internal_workspace_dir(root, child)
        ),
        key=lambda child: (not child.is_dir(), child.name.lower()),
    )
    visible_children = allowed_children[:max_entries]
    entries = tuple(_directory_entry(root, child) for child in visible_children)
    relative_path = _relative_path(root, path)
    content = "\n".join(
        f"{entry['name']}/" if entry["kind"] == "directory" else str(entry["name"])
        for entry in entries
    )
    return NativeToolResult(
        content=content,
        data={
            "path": relative_path,
            "entries": entries,
            "filtered_entries": len(children) - len(allowed_children),
            "truncated": len(allowed_children) > len(visible_children),
        },
        refs=(relative_path,),
    )


def _write_text_file(
    root: Path,
    arguments: Mapping[str, object],
    *,
    write_checkpoint: WorkspaceWriteCheckpoint | None = None,
    hook_context: HookRuntimeContext | None = None,
) -> NativeToolResult:
    path = _workspace_path(root, _text_argument(arguments, "path"))
    content = _text_argument(arguments, "content", allow_empty=True)
    overwrite = _bool_argument(arguments, "overwrite", False)
    if len(content) > MAX_WRITE_CHARS:
        raise ValueError(f"content must be at most {MAX_WRITE_CHARS} characters")
    if path.exists() and not path.is_file():
        raise ValueError(f"path exists but is not a file: {path}")
    if path.exists() and not overwrite:
        raise ValueError("path already exists; pass overwrite=true to replace it")
    if not path.parent.is_dir():
        raise ValueError(f"parent directory does not exist: {path.parent}")

    existed = path.exists()
    before_content = _read_workspace_text_file(root, path) if existed else ""
    before_outcome = _emit_write_hook(
        hook_context,
        HookEvent.WRITE_BEFORE,
        root=root,
        path=path,
        operation="write_text_file",
        status="before",
        payload={
            "overwrite": overwrite,
            "content_size": len(content),
            "path_kind": "file",
        },
    )
    if before_outcome.directive != HookDirective.CONTINUE:
        raise ValueError(
            before_outcome.reason
            or f"Workspace write stopped by hook: {before_outcome.directive.value}"
        )
    if write_checkpoint is not None:
        write_checkpoint("write_text_file", path, content)
    _atomic_write_text(path, content)
    relative_path = _relative_path(root, path)
    after_outcome = _emit_write_hook(
        hook_context,
        HookEvent.WRITE_AFTER,
        root=root,
        path=path,
        operation="write_text_file",
        status="completed",
        payload={
            "overwrite": overwrite,
            "content_size": len(content),
            "path_kind": "file",
        },
    )
    diff = _unified_diff(
        before_content,
        content,
        fromfile=f"a/{relative_path}" if existed else "/dev/null",
        tofile=f"b/{relative_path}",
    )
    return NativeToolResult(
        content=_write_result_content("Wrote", relative_path, diff),
        data={
            "path": relative_path,
            "chars_written": len(content),
            "bytes_written": len(content.encode("utf-8")),
            "overwritten": existed,
            "diff": diff,
        },
        refs=(relative_path, *_hook_refs(after_outcome)),
    )


def _edit_text_file(
    root: Path,
    arguments: Mapping[str, object],
    *,
    write_checkpoint: WorkspaceWriteCheckpoint | None = None,
    hook_context: HookRuntimeContext | None = None,
) -> NativeToolResult:
    path = _workspace_path(root, _text_argument(arguments, "path"))
    old_text = _text_argument(arguments, "old_text", preserve=True)
    new_text = _text_argument(arguments, "new_text", allow_empty=True)
    if len(new_text) > MAX_WRITE_CHARS:
        raise ValueError(f"new_text must be at most {MAX_WRITE_CHARS} characters")
    if not path.is_file():
        raise ValueError(f"path is not a file: {path}")

    try:
        content = _read_workspace_text_file(root, path)
    except UnicodeDecodeError as exc:
        raise ValueError(
            "edit_text_file requires UTF-8 text; this file could not be decoded "
            "as UTF-8. Convert the file to UTF-8 before editing it."
        ) from exc
    matches = content.count(old_text)
    if matches == 0:
        raise ValueError("old_text was not found in file")
    if matches > 1:
        raise ValueError("old_text matched more than once; provide a unique fragment")
    updated = content.replace(old_text, new_text, 1)
    if len(updated) > MAX_WRITE_CHARS:
        raise ValueError(f"updated file must be at most {MAX_WRITE_CHARS} characters")

    before_outcome = _emit_write_hook(
        hook_context,
        HookEvent.WRITE_BEFORE,
        root=root,
        path=path,
        operation="edit_text_file",
        status="before",
        payload={
            "content_size": len(updated),
            "path_kind": "file",
        },
    )
    if before_outcome.directive != HookDirective.CONTINUE:
        raise ValueError(
            before_outcome.reason
            or f"Workspace write stopped by hook: {before_outcome.directive.value}"
        )
    if write_checkpoint is not None:
        write_checkpoint("edit_text_file", path, updated)
    _atomic_write_text(path, updated)
    relative_path = _relative_path(root, path)
    after_outcome = _emit_write_hook(
        hook_context,
        HookEvent.WRITE_AFTER,
        root=root,
        path=path,
        operation="edit_text_file",
        status="completed",
        payload={
            "content_size": len(updated),
            "path_kind": "file",
        },
    )
    diff = _unified_diff(
        content,
        updated,
        fromfile=f"a/{relative_path}",
        tofile=f"b/{relative_path}",
    )
    return NativeToolResult(
        content=_write_result_content("Edited", relative_path, diff),
        data={
            "path": relative_path,
            "replacements": 1,
            "chars_written": len(updated),
            "bytes_written": len(updated.encode("utf-8")),
            "diff": diff,
        },
        refs=(relative_path, *_hook_refs(after_outcome)),
    )


def _workspace_path(root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    path = candidate if candidate.is_absolute() else root / candidate
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path must stay inside workspace root: {raw_path}")
    path_for_denied_check = path if path == root or root in path.parents else resolved
    if _is_denied_path(root, path_for_denied_check) or _is_denied_path(root, resolved):
        raise ValueError(f"path is not allowed for read-only filesystem tools: {raw_path}")
    return resolved


def _read_workspace_text_file(root: Path, path: Path) -> str:
    with _open_workspace_text_file(root, path) as file:
        return file.read()


def _open_workspace_text_file(root: Path, path: Path, *, errors: str = "strict"):
    fd = _open_workspace_file_descriptor(root, path)
    try:
        return open(fd, "r", encoding="utf-8", errors=errors, closefd=True)
    except Exception:
        os.close(fd)
        raise


def _open_workspace_file_descriptor(root: Path, path: Path) -> int:
    flags = os.O_RDONLY
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow:
        flags |= nofollow
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"path is not a readable file: {path}") from exc
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"path is not a file: {path}")
        _validate_resolved_workspace_path(root, path)
    except Exception:
        os.close(fd)
        raise
    return fd


def _validate_resolved_workspace_path(root: Path, path: Path) -> None:
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path must stay inside workspace root: {path}")
    if _is_denied_path(root, resolved):
        raise ValueError(f"path is not allowed for read-only filesystem tools: {path}")


def _directory_entry(root: Path, path: Path) -> Mapping[str, str]:
    if path.is_dir():
        kind = "directory"
    elif path.is_file():
        kind = "file"
    else:
        kind = "other"
    return {"name": path.name, "kind": kind, "path": _relative_path(root, path)}


def _relative_path(root: Path, path: Path) -> str:
    if path == root:
        return "."
    return path.relative_to(root).as_posix()


def _is_internal_workspace_dir(root: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    return bool(
        relative.parts
        and len(relative.parts) == 1
        and path.is_dir()
        and relative.parts[0] in INTERNAL_WORKSPACE_DIR_NAMES
    )


def _unified_diff(
    before: str,
    after: str,
    *,
    fromfile: str,
    tofile: str,
) -> str:
    lines = tuple(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=fromfile,
            tofile=tofile,
            lineterm="",
        )
    )
    return "\n".join(lines) + ("\n" if lines else "")


def _write_result_content(action: str, relative_path: str, diff: str) -> str:
    lines = [f"{action} {relative_path}"]
    if diff.strip():
        lines.extend(("", "Diff:", diff.rstrip()))
    return "\n".join(lines)


def _emit_write_hook(
    hook_context: HookRuntimeContext | None,
    event_name: HookEvent,
    *,
    root: Path,
    path: Path,
    operation: str,
    status: str,
    payload: Mapping[str, object],
) -> HookOutcome:
    if hook_context is None:
        return HookOutcome()
    relative_path = _relative_path(root, path)
    return hook_context.emit(
        HookEnvelope(
            event_name=event_name,
            actor=HookActor.MAIN,
            payload={
                "tool_source": "native",
                "operation": operation,
                "path": relative_path,
                "status": status,
                "actor": HookActor.MAIN.value,
                **payload,
            },
            source_refs=(f"path={relative_path}", *hook_context.trace_refs()),
        )
    )


def _hook_refs(outcome: HookOutcome) -> tuple[str, ...]:
    if not outcome.refs:
        return ()
    return (f"hook_directive={outcome.directive.value}", *outcome.refs)


def _is_denied_path(root: Path, path: Path) -> bool:
    if path == root:
        return False
    try:
        resolved = path.resolve()
    except OSError:
        return True
    if resolved != root and root not in resolved.parents:
        return True

    requested_relative = path.relative_to(root)
    if any(_is_denied_part(part) for part in requested_relative.parts):
        return True

    if resolved == path:
        return False
    resolved_relative = resolved.relative_to(root) if resolved != root else Path()
    return any(_is_denied_part(part) for part in resolved_relative.parts)


def _is_denied_part(part: str) -> bool:
    name = part.lower()
    return (
        name in DENIED_PATH_NAMES
        or name.startswith(".env.")
        or name.endswith(DENIED_SUFFIXES)
    )


def _atomic_write_text(path: Path, content: str) -> None:
    atomic_write_text(path, content)


def _text_argument(
    arguments: Mapping[str, object],
    key: str,
    allow_empty: bool = False,
    preserve: bool = False,
) -> str:
    value = arguments.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be text")
    if not allow_empty and not value.strip():
        raise ValueError(f"{key} must be non-empty text")
    return value if allow_empty or preserve else value.strip()


def _optional_text_argument(
    arguments: Mapping[str, object],
    key: str,
    default: str,
) -> str:
    value = arguments.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be text")
    return value.strip()


def _int_argument(
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


def _bool_argument(arguments: Mapping[str, object], key: str, default: bool) -> bool:
    value = arguments.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _read_text_file_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative file path."},
            "max_chars": {"type": "integer", "minimum": 1, "maximum": MAX_CHARS},
            "offset": {
                "type": "integer",
                "minimum": 0,
                "maximum": MAX_READ_OFFSET,
                "description": (
                    "Character offset to start reading from. Use the previous "
                    "result's next_offset to page through a long file."
                ),
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }


def _list_directory_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative directory path.",
            },
            "max_entries": {"type": "integer", "minimum": 1, "maximum": MAX_ENTRIES},
        },
        "additionalProperties": False,
    }


def _write_text_file_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative file path."},
            "content": {"type": "string", "maxLength": MAX_WRITE_CHARS},
            "overwrite": {
                "type": "boolean",
                "description": "Whether to replace an existing file.",
            },
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }


def _edit_text_file_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative file path."},
            "old_text": {
                "type": "string",
                "description": "Exact existing text fragment to replace once.",
            },
            "new_text": {"type": "string", "maxLength": MAX_WRITE_CHARS},
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    }
