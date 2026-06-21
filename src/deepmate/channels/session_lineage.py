"""Shared session lineage command helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from deepmate.domain import ProfileRef
from deepmate.storage import (
    SessionLineageNode,
    SessionRecord,
    SessionStore,
    TurnCheckpointStore,
)


@dataclass(frozen=True, slots=True)
class SessionLineageCommandResult:
    """Result of a clone/fork command."""

    session: SessionRecord
    body: str


def handle_session_lineage_command(
    command: str,
    *,
    session_store: SessionStore,
    session: SessionRecord,
    workspace: Path,
    profile: ProfileRef,
    turn_store: TurnCheckpointStore | None = None,
) -> SessionLineageCommandResult | str | None:
    """Handle a /session lineage command and return output."""
    clean = command.strip()
    if clean == "/session":
        return format_session_info(session)
    if clean == "/session info":
        return format_session_info(session)
    if clean in {"/session tree", "/tree"}:
        return format_session_tree(
            session_store.lineage_tree(workspace=workspace, profile=profile),
            current_session_id=session.session_id,
            workspace=workspace,
            profile=profile,
        )
    if clean == "/session clone" or clean.startswith("/session clone "):
        title = _quoted_or_raw(clean[len("/session clone") :].strip())
        clone = session_store.clone_session(session, title=title)
        return SessionLineageCommandResult(
            session=clone,
            body=format_lineage_created(
                kind="clone",
                source=session,
                target=clone,
                copied_records=clone.forked_from_sequence,
                switched=True,
            ),
        )
    if clean == "/clone" or clean.startswith("/clone "):
        title = _quoted_or_raw(clean[len("/clone") :].strip())
        clone = session_store.clone_session(session, title=title)
        return SessionLineageCommandResult(
            session=clone,
            body=format_lineage_created(
                kind="clone",
                source=session,
                target=clone,
                copied_records=clone.forked_from_sequence,
                switched=True,
            ),
        )
    if clean.startswith("/session fork") or clean.startswith("/fork"):
        args = (
            clean[len("/session fork") :].strip()
            if clean.startswith("/session fork")
            else clean[len("/fork") :].strip()
        )
        fork_target, title = _parse_fork_args(args)
        sequence, turn_id = _resolve_fork_target(fork_target, turn_store)
        fork = session_store.fork_session_at_sequence(
            session,
            sequence,
            title=title,
            turn_id=turn_id,
        )
        return SessionLineageCommandResult(
            session=fork,
            body=format_lineage_created(
                kind="fork",
                source=session,
                target=fork,
                copied_records=fork.forked_from_sequence,
                switched=True,
            ),
        )
    if clean.startswith("/session "):
        return "usage: /session info|tree|clone [title]|fork <turn_id|--sequence N> [title]"
    return None


def format_session_info(session: SessionRecord) -> str:
    """Return current session details."""
    lines = [
        "Session",
        f"  id: {session.session_id}",
        f"  title: {session.title}",
        f"  workspace: {session.workspace}",
        f"  profile: {session.profile.name}",
        f"  updated_at: {session.updated_at}",
    ]
    lineage = _lineage_detail(session)
    if lineage:
        lines.append(f"  lineage: {lineage}")
    else:
        lines.append("  lineage: root")
    return "\n".join(lines)


def format_session_tree(
    roots: tuple[SessionLineageNode, ...],
    *,
    current_session_id: str,
    workspace: Path,
    profile: ProfileRef,
) -> str:
    """Return a user-facing session lineage tree."""
    if not roots:
        return "No sessions found."
    lines = [f"workspace: {workspace}", f"profile: {profile.name}", ""]
    for index, root in enumerate(roots):
        if index:
            lines.append("")
        _append_tree_node(
            lines,
            root,
            current_session_id=current_session_id,
            prefix="",
            is_last=True,
            is_root=True,
        )
    return "\n".join(lines)


def format_lineage_created(
    *,
    kind: str,
    source: SessionRecord,
    target: SessionRecord,
    copied_records: int,
    switched: bool,
) -> str:
    """Return clone/fork completion text."""
    title = "Created session clone" if kind == "clone" else "Created session fork"
    lines = [
        title,
        f"  new:  {_short_id(target.session_id)}  {target.title}",
        f"  from: {_short_id(source.session_id)}  {source.title}",
    ]
    if kind == "fork":
        fork_from = target.forked_from_turn_id or f"sequence={target.forked_from_sequence}"
        lines.append(
            f"  forked_from: {fork_from}, sequence={target.forked_from_sequence}"
        )
        lines.append("  workspace_restored: no")
    lines.append(f"  copied: {copied_records} transcript records")
    lines.append(f"  switched: {'yes' if switched else 'no'}")
    return "\n".join(lines)


def _append_tree_node(
    lines: list[str],
    node: SessionLineageNode,
    *,
    current_session_id: str,
    prefix: str,
    is_last: bool,
    is_root: bool,
) -> None:
    marker = "  current" if node.session.session_id == current_session_id else ""
    detail = _tree_detail(node.session)
    suffix = f"  {detail}" if detail else ""
    connector = "" if is_root else ("└─ " if is_last else "├─ ")
    lines.append(
        f"{prefix}{connector}{node.session.title}  "
        f"{_short_id(node.session.session_id)}{suffix}{marker}"
    )
    child_prefix = prefix if is_root else prefix + ("   " if is_last else "│  ")
    for index, child in enumerate(node.children):
        _append_tree_node(
            lines,
            child,
            current_session_id=current_session_id,
            prefix=child_prefix,
            is_last=index == len(node.children) - 1,
            is_root=False,
        )


def _tree_detail(session: SessionRecord) -> str:
    if session.fork_kind == "clone":
        return "clone"
    if session.fork_kind == "fork":
        if session.forked_from_turn_id:
            return f"fork from {session.forked_from_turn_id}"
        if session.forked_from_sequence:
            return f"fork from sequence {session.forked_from_sequence}"
    return ""


def _lineage_detail(session: SessionRecord) -> str:
    if not session.parent_session_id:
        return ""
    source = _short_id(session.parent_session_id)
    if session.fork_kind == "clone":
        return f"clone from {source}"
    if session.fork_kind == "fork":
        if session.forked_from_turn_id:
            return (
                f"fork from {source} / {session.forked_from_turn_id} "
                f"sequence={session.forked_from_sequence}"
            )
        return f"fork from {source} / sequence={session.forked_from_sequence}"
    return f"from {source}"


def _resolve_fork_target(
    value: str,
    turn_store: TurnCheckpointStore | None,
) -> tuple[int, str]:
    target = value.strip()
    if not target:
        raise ValueError("usage: /session fork <turn_id|--sequence N> [title]")
    if target.startswith("--sequence "):
        raw = target[len("--sequence ") :].split(maxsplit=1)[0].strip()
        return _positive_int(raw), ""
    if target.startswith("sequence="):
        return _positive_int(target[len("sequence=") :].split(maxsplit=1)[0]), ""
    first = target.split(maxsplit=1)[0]
    if first.isdigit():
        return _positive_int(first), ""
    if turn_store is None:
        raise ValueError("turn checkpoint store is not available")
    try:
        checkpoint = turn_store.require_turn(first)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"turn not found: {first}. Use /task to see available turns."
        ) from exc
    if checkpoint.last_transcript_sequence <= 0:
        raise ValueError(f"{first} has no recorded transcript sequence")
    return checkpoint.last_transcript_sequence, checkpoint.turn_id


def _parse_fork_args(args: str) -> tuple[str, str]:
    clean = args.strip()
    if not clean:
        raise ValueError("usage: /session fork <turn_id|--sequence N> [title]")
    parts = clean.split(maxsplit=1)
    if parts[0] == "--sequence":
        if len(parts) < 2:
            raise ValueError("usage: /session fork --sequence <number> [title]")
        seq_parts = parts[1].split(maxsplit=1)
        target = f"--sequence {seq_parts[0]}"
        title = _quoted_or_raw(seq_parts[1]) if len(seq_parts) > 1 else ""
        return target, title
    title = _quoted_or_raw(parts[1]) if len(parts) > 1 else ""
    return parts[0], title


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise ValueError(f"invalid sequence: {value}") from exc
    if number < 1:
        raise ValueError("sequence must be greater than 0")
    return number


def _quoted_or_raw(value: str) -> str:
    clean = value.strip()
    clean = clean.strip("\"'")
    return clean


def _short_id(value: str) -> str:
    clean = value.strip()
    return clean[:8] if len(clean) > 8 else clean
