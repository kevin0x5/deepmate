"""Session checkpoint storage for turn recovery and local rewinds."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deepmate.storage.atomic import atomic_write_json, atomic_write_text, file_lock
from deepmate.storage.session_store import TranscriptRecord

TURN_STATUS_RUNNING = "running"
TURN_STATUS_COMPLETED = "completed"
TURN_STATUS_FAILED = "failed"
TURN_STATUS_INTERRUPTED = "interrupted"
TURN_STATUS_MAX_STEPS = "max_steps"

RESUME_HINT_NORMAL = "normal"
RESUME_HINT_NO_RESPONSE = "no_response"
RESUME_HINT_AFTER_TOOL = "after_tool"
RESUME_HINT_MAX_STEPS = "max_steps"
RESUME_HINT_FAILED = "failed"
RESUME_HINT_INTERRUPTED = "interrupted"

MAX_SNAPSHOT_CHARS = 200_000
_TURN_ID_RE = re.compile(r"^turn_(\d+)$")


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class TurnCheckpointRecord:
    """Latest known state for one user turn."""

    turn_id: str
    session_id: str
    profile: str
    status: str
    resume_hint: str
    started_at: str
    completed_at: str = ""
    user_sequence: int = 0
    last_transcript_sequence: int = 0
    last_tool_exchange_sequence: int = 0
    final_assistant_sequence: int = 0
    summary_id: str = ""
    workspace_checkpoint_id: str = ""
    error_code: str = ""
    continuation_note: str = ""

    def is_ready(self) -> bool:
        """Return whether this checkpoint has enough identity to persist."""
        return bool(
            self.turn_id.strip()
            and self.session_id.strip()
            and self.profile.strip()
            and self.status.strip()
            and self.resume_hint.strip()
            and self.started_at.strip()
        )

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable checkpoint record."""
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "profile": self.profile,
            "status": self.status,
            "resume_hint": self.resume_hint,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "user_sequence": self.user_sequence,
            "last_transcript_sequence": self.last_transcript_sequence,
            "last_tool_exchange_sequence": self.last_tool_exchange_sequence,
            "final_assistant_sequence": self.final_assistant_sequence,
            "summary_id": self.summary_id,
            "workspace_checkpoint_id": self.workspace_checkpoint_id,
            "error_code": self.error_code,
            "continuation_note": self.continuation_note,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "TurnCheckpointRecord":
        """Build a checkpoint record from stored JSON."""
        return cls(
            turn_id=_text(record.get("turn_id")),
            session_id=_text(record.get("session_id")),
            profile=_text(record.get("profile")),
            status=_text(record.get("status")),
            resume_hint=_text(record.get("resume_hint")),
            started_at=_text(record.get("started_at")),
            completed_at=_text(record.get("completed_at")),
            user_sequence=_int(record.get("user_sequence")),
            last_transcript_sequence=_int(record.get("last_transcript_sequence")),
            last_tool_exchange_sequence=_int(record.get("last_tool_exchange_sequence")),
            final_assistant_sequence=_int(record.get("final_assistant_sequence")),
            summary_id=_text(record.get("summary_id")),
            workspace_checkpoint_id=_text(record.get("workspace_checkpoint_id")),
            error_code=_text(record.get("error_code")),
            continuation_note=_text(record.get("continuation_note")),
        )


class TurnCheckpointStore:
    """Persist turn-level recovery checkpoints for one session."""

    def __init__(self, root: str | Path, profile: str, session_id: str) -> None:
        self._root = Path(root).resolve()
        self._profile = _clean_segment(profile, fallback="default")
        self._session_id = _clean_segment(session_id, fallback="unknown-session")
        self._session_dir = (
            self._root / "checkpoints" / self._profile / self._session_id
        ).resolve()
        if not _is_relative_to(self._session_dir, self._root):
            raise ValueError("checkpoint store path escaped data root")
        self._turns_path = self._session_dir / "turns.jsonl"
        self._latest_path = self._session_dir / "latest.json"

    @classmethod
    def in_data_dir(
        cls,
        data_dir: str | Path,
        profile: str,
        session_id: str,
    ) -> "TurnCheckpointStore":
        """Create a turn checkpoint store rooted at Deepmate's data dir."""
        return cls(data_dir, profile=profile, session_id=session_id)

    def start_turn(self, summary_id: str = "") -> TurnCheckpointRecord:
        """Start a new user turn checkpoint."""
        with file_lock(self._turns_path):
            record = TurnCheckpointRecord(
                turn_id=_next_turn_id(self.load_turns()),
                session_id=self._session_id,
                profile=self._profile,
                status=TURN_STATUS_RUNNING,
                resume_hint=RESUME_HINT_NO_RESPONSE,
                started_at=_utc_now_iso(),
                summary_id=summary_id.strip(),
            )
            return self._write_state(record, locked=True)

    def record_transcript_item(
        self,
        turn_id: str,
        transcript_record: TranscriptRecord,
    ) -> TurnCheckpointRecord:
        """Update one turn checkpoint from an appended transcript record."""
        record = self.require_turn(turn_id)
        last_sequence = max(record.last_transcript_sequence, transcript_record.sequence)
        updates: dict[str, object] = {
            "last_transcript_sequence": last_sequence,
        }
        if transcript_record.kind == "message":
            role = str(transcript_record.payload.get("role", "")).strip()
            if role == "user":
                updates["user_sequence"] = transcript_record.sequence
                updates["resume_hint"] = RESUME_HINT_NO_RESPONSE
            elif role == "assistant":
                updates["final_assistant_sequence"] = transcript_record.sequence
                updates["resume_hint"] = RESUME_HINT_NORMAL
        elif transcript_record.kind == "tool_exchange":
            updates["last_tool_exchange_sequence"] = transcript_record.sequence
            updates["resume_hint"] = RESUME_HINT_AFTER_TOOL
        return self._write_state(_replace_record(record, updates))

    def attach_workspace_checkpoint(
        self,
        turn_id: str,
        workspace_checkpoint_id: str,
    ) -> TurnCheckpointRecord:
        """Attach a workspace checkpoint id to one turn."""
        record = self.require_turn(turn_id)
        return self._write_state(
            replace(record, workspace_checkpoint_id=workspace_checkpoint_id.strip())
        )

    def attach_summary(self, turn_id: str, summary_id: str) -> TurnCheckpointRecord:
        """Attach the latest summary id observed after a turn."""
        clean = summary_id.strip()
        if not clean:
            return self.require_turn(turn_id)
        return self._write_state(replace(self.require_turn(turn_id), summary_id=clean))

    def attach_continuation_note(
        self,
        turn_id: str,
        note: str,
    ) -> TurnCheckpointRecord:
        """Attach compact continuation context for a non-normal stop."""
        clean = note.strip()
        if not clean:
            return self.require_turn(turn_id)
        return self._write_state(
            replace(self.require_turn(turn_id), continuation_note=clean)
        )

    def complete_turn(self, turn_id: str) -> TurnCheckpointRecord:
        """Mark a turn as completed normally."""
        record = self.require_turn(turn_id)
        return self._write_state(
            replace(
                record,
                status=TURN_STATUS_COMPLETED,
                resume_hint=RESUME_HINT_NORMAL,
                completed_at=_utc_now_iso(),
                error_code="",
            )
        )

    def max_steps_turn(self, turn_id: str) -> TurnCheckpointRecord:
        """Mark a turn as stopped by max_steps."""
        record = self.require_turn(turn_id)
        return self._write_state(
            replace(
                record,
                status=TURN_STATUS_MAX_STEPS,
                resume_hint=RESUME_HINT_MAX_STEPS,
                completed_at=_utc_now_iso(),
                error_code="max_steps",
            )
        )

    def fail_turn(self, turn_id: str, error_code: str) -> TurnCheckpointRecord:
        """Mark a turn as failed."""
        record = self.require_turn(turn_id)
        return self._write_state(
            replace(
                record,
                status=TURN_STATUS_FAILED,
                resume_hint=RESUME_HINT_FAILED,
                completed_at=_utc_now_iso(),
                error_code=error_code.strip() or "failed",
            )
        )

    def interrupt_turn(
        self,
        turn_id: str,
        error_code: str = "interrupted",
    ) -> TurnCheckpointRecord:
        """Mark a turn as interrupted."""
        record = self.require_turn(turn_id)
        return self._write_state(
            replace(
                record,
                status=TURN_STATUS_INTERRUPTED,
                resume_hint=RESUME_HINT_INTERRUPTED,
                completed_at=_utc_now_iso(),
                error_code=error_code.strip() or "interrupted",
            )
        )

    def load_latest(self) -> TurnCheckpointRecord | None:
        """Load the latest checkpoint state, if present."""
        records = self._load_turn_records_in_append_order()
        if records:
            return records[-1]
        if not self._latest_path.exists():
            return None
        try:
            payload = json.loads(self._latest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        record = TurnCheckpointRecord.from_record(payload)
        return record if record.is_ready() else None

    def load_turns(self) -> tuple[TurnCheckpointRecord, ...]:
        """Load the latest state for every known turn."""
        by_turn: dict[str, TurnCheckpointRecord] = {}
        for record in self._load_turn_records_in_append_order():
            by_turn[record.turn_id] = record
        return tuple(sorted(by_turn.values(), key=lambda record: _turn_number(record.turn_id)))

    def _load_turn_records_in_append_order(self) -> tuple[TurnCheckpointRecord, ...]:
        if not self._turns_path.exists():
            return ()
        records: list[TurnCheckpointRecord] = []
        with self._turns_path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                record = TurnCheckpointRecord.from_record(payload)
                if record.is_ready():
                    records.append(record)
        return tuple(records)

    def require_turn(self, turn_id: str) -> TurnCheckpointRecord:
        """Return a turn by id or raise."""
        clean = turn_id.strip()
        for record in self.load_turns():
            if record.turn_id == clean:
                return record
        raise ValueError(f"turn checkpoint not found: {turn_id}")

    def set_latest(self, turn_id: str) -> TurnCheckpointRecord:
        """Set an existing turn as the latest checkpoint state."""
        return self._write_state(self.require_turn(turn_id))

    def _write_state(
        self,
        record: TurnCheckpointRecord,
        *,
        locked: bool = False,
    ) -> TurnCheckpointRecord:
        if record.session_id != self._session_id or record.profile != self._profile:
            raise ValueError("turn checkpoint belongs to another session")
        if not record.is_ready():
            raise ValueError("turn checkpoint record is not ready")
        if locked:
            self._write_state_unlocked(record)
        else:
            with file_lock(self._turns_path):
                self._write_state_unlocked(record)
        return record

    def _write_state_unlocked(self, record: TurnCheckpointRecord) -> None:
        self._session_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_record(), ensure_ascii=False, separators=(",", ":"))
        turns_existed = self._turns_path.exists()
        with self._turns_path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
            file.flush()
            os.fsync(file.fileno())
        if not turns_existed:
            _fsync_directory(self._turns_path.parent)
        atomic_write_json(self._latest_path, record.to_record())


@dataclass(frozen=True, slots=True)
class WorkspaceFileSnapshot:
    """Preimage for one workspace file touched by Deepmate."""

    path: str
    operation: str
    before_existed: bool
    before_size_bytes: int = 0
    before_sha256: str = ""
    before_content: str = ""
    after_sha256: str = ""
    snapshot_status: str = "captured"
    skipped_reason: str = ""

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable file snapshot."""
        return {
            "path": self.path,
            "operation": self.operation,
            "before_existed": self.before_existed,
            "before_size_bytes": self.before_size_bytes,
            "before_sha256": self.before_sha256,
            "before_content": self.before_content,
            "after_sha256": self.after_sha256,
            "snapshot_status": self.snapshot_status,
            "skipped_reason": self.skipped_reason,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "WorkspaceFileSnapshot":
        """Build a file snapshot from stored JSON."""
        return cls(
            path=_text(record.get("path")),
            operation=_text(record.get("operation")),
            before_existed=bool(record.get("before_existed")),
            before_size_bytes=_int(record.get("before_size_bytes")),
            before_sha256=_text(record.get("before_sha256")),
            before_content=_text(record.get("before_content")),
            after_sha256=_text(record.get("after_sha256")),
            snapshot_status=_text(record.get("snapshot_status")) or "captured",
            skipped_reason=_text(record.get("skipped_reason")),
        )


@dataclass(frozen=True, slots=True)
class WorkspaceCheckpointRecord:
    """Workspace file preimages captured during one turn."""

    workspace_checkpoint_id: str
    session_id: str
    profile: str
    turn_id: str
    created_at: str
    files: tuple[WorkspaceFileSnapshot, ...] = field(default_factory=tuple)

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable workspace checkpoint record."""
        return {
            "workspace_checkpoint_id": self.workspace_checkpoint_id,
            "session_id": self.session_id,
            "profile": self.profile,
            "turn_id": self.turn_id,
            "created_at": self.created_at,
            "files": [snapshot.to_record() for snapshot in self.files],
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "WorkspaceCheckpointRecord":
        """Build a workspace checkpoint record from stored JSON."""
        files = record.get("files")
        return cls(
            workspace_checkpoint_id=_text(record.get("workspace_checkpoint_id")),
            session_id=_text(record.get("session_id")),
            profile=_text(record.get("profile")),
            turn_id=_text(record.get("turn_id")),
            created_at=_text(record.get("created_at")),
            files=tuple(
                WorkspaceFileSnapshot.from_record(item)
                for item in files
                if isinstance(item, dict)
            )
            if isinstance(files, list)
            else (),
        )


@dataclass(frozen=True, slots=True)
class WorkspaceRewindAction:
    """One file action needed to rewind workspace state."""

    path: str
    action: str
    conflict: bool = False
    reason: str = ""


@dataclass(frozen=True, slots=True)
class WorkspaceRewindPlan:
    """Preview or result of a workspace rewind."""

    target_turn_id: str
    actions: tuple[WorkspaceRewindAction, ...] = field(default_factory=tuple)

    def has_conflicts(self) -> bool:
        """Return whether applying the plan would overwrite unexpected changes."""
        return any(action.conflict for action in self.actions)


class WorkspaceCheckpointStore:
    """Persist workspace preimages for Deepmate-owned file writes."""

    def __init__(self, root: str | Path, profile: str, session_id: str) -> None:
        self._root = Path(root).resolve()
        self._profile = _clean_segment(profile, fallback="default")
        self._session_id = _clean_segment(session_id, fallback="unknown-session")
        self._session_dir = (
            self._root / "checkpoints" / self._profile / self._session_id / "workspace"
        ).resolve()
        if not _is_relative_to(self._session_dir, self._root):
            raise ValueError("workspace checkpoint path escaped data root")

    @classmethod
    def in_data_dir(
        cls,
        data_dir: str | Path,
        profile: str,
        session_id: str,
    ) -> "WorkspaceCheckpointStore":
        """Create a workspace checkpoint store rooted at Deepmate's data dir."""
        return cls(data_dir, profile=profile, session_id=session_id)

    def capture_file(
        self,
        *,
        turn_id: str,
        operation: str,
        workspace: str | Path,
        path: str | Path,
        after_content: str,
    ) -> WorkspaceCheckpointRecord:
        """Capture the file preimage before a Deepmate-owned write."""
        with file_lock(self._path_for_turn(turn_id)):
            record = self.load_checkpoint(turn_id) or WorkspaceCheckpointRecord(
                workspace_checkpoint_id=f"wcp_{turn_id.strip()}",
                session_id=self._session_id,
                profile=self._profile,
                turn_id=turn_id.strip(),
                created_at=_utc_now_iso(),
            )
            relative_path = _relative_workspace_path(
                Path(workspace).resolve(),
                Path(path).resolve(),
            )
            if any(snapshot.path == relative_path for snapshot in record.files):
                return record
            snapshot = _file_snapshot(
                path=Path(path).resolve(),
                relative_path=relative_path,
                operation=operation,
                after_content=after_content,
            )
            updated = replace(record, files=(*record.files, snapshot))
            self._write_checkpoint(updated)
            return updated

    def load_checkpoint(self, turn_id: str) -> WorkspaceCheckpointRecord | None:
        """Load one workspace checkpoint by turn id."""
        path = self._path_for_turn(turn_id)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        record = WorkspaceCheckpointRecord.from_record(payload)
        if record.session_id != self._session_id or record.profile != self._profile:
            return None
        return record

    def load_checkpoints(self) -> tuple[WorkspaceCheckpointRecord, ...]:
        """Load all workspace checkpoints for this session."""
        if not self._session_dir.exists():
            return ()
        records = []
        for path in self._session_dir.glob("turn_*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            record = WorkspaceCheckpointRecord.from_record(payload)
            if record.session_id == self._session_id and record.profile == self._profile:
                records.append(record)
        return tuple(sorted(records, key=lambda record: _turn_number(record.turn_id)))

    def rewind_plan(self, target_turn_id: str, workspace: str | Path) -> WorkspaceRewindPlan:
        """Return a preview plan for rewinding changes after the target turn."""
        actions = [
            _rewind_action(snapshot, Path(workspace).resolve())
            for snapshot in _snapshots_for_rewind_plan(
                self.load_checkpoints(),
                target_turn_id,
            )
        ]
        return WorkspaceRewindPlan(target_turn_id=target_turn_id.strip(), actions=tuple(actions))

    def apply_rewind(
        self,
        target_turn_id: str,
        workspace: str | Path,
        *,
        force: bool = False,
    ) -> WorkspaceRewindPlan:
        """Apply a workspace rewind to the end of target_turn_id."""
        plan = self.rewind_plan(target_turn_id, workspace)
        if plan.has_conflicts() and not force:
            return plan
        workspace_root = Path(workspace).resolve()
        snapshots = _snapshots_for_rewind(self.load_checkpoints(), target_turn_id)
        for snapshot in snapshots:
            if snapshot.snapshot_status != "captured":
                continue
            path = (workspace_root / snapshot.path).resolve()
            if not _is_relative_to(path, workspace_root):
                continue
            if snapshot.before_existed:
                path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_text(path, snapshot.before_content)
            elif path.exists() and path.is_file():
                path.unlink()
        return plan

    def _write_checkpoint(self, record: WorkspaceCheckpointRecord) -> None:
        self._session_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self._path_for_turn(record.turn_id), record.to_record())

    def _path_for_turn(self, turn_id: str) -> Path:
        clean = turn_id.strip()
        if not _TURN_ID_RE.match(clean):
            raise ValueError(f"invalid turn id: {turn_id}")
        return self._session_dir / f"{clean}.json"


def _replace_record(
    record: TurnCheckpointRecord,
    updates: dict[str, object],
) -> TurnCheckpointRecord:
    return replace(record, **updates)


def _next_turn_id(records: tuple[TurnCheckpointRecord, ...]) -> str:
    number = max((_turn_number(record.turn_id) for record in records), default=0) + 1
    return f"turn_{number:05d}"


def _turn_number(turn_id: str) -> int:
    match = _TURN_ID_RE.match(turn_id.strip())
    return int(match.group(1)) if match else 0


def _file_snapshot(
    *,
    path: Path,
    relative_path: str,
    operation: str,
    after_content: str,
) -> WorkspaceFileSnapshot:
    after_sha = _sha256_text(after_content)
    if not path.exists():
        return WorkspaceFileSnapshot(
            path=relative_path,
            operation=operation.strip(),
            before_existed=False,
            after_sha256=after_sha,
        )
    if not path.is_file():
        before_size = _path_size(path)
        return WorkspaceFileSnapshot(
            path=relative_path,
            operation=operation.strip(),
            before_existed=True,
            before_size_bytes=before_size,
            after_sha256=after_sha,
            snapshot_status="skipped",
            skipped_reason="not_regular_file",
        )
    before_size = _path_size(path)
    if before_size > MAX_SNAPSHOT_CHARS * 4:
        return WorkspaceFileSnapshot(
            path=relative_path,
            operation=operation.strip(),
            before_existed=True,
            before_size_bytes=before_size,
            before_sha256=_sha256_file(path),
            after_sha256=after_sha,
            snapshot_status="skipped",
            skipped_reason="file_too_large",
        )
    try:
        before = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return WorkspaceFileSnapshot(
            path=relative_path,
            operation=operation.strip(),
            before_existed=True,
            before_size_bytes=before_size,
            before_sha256=_sha256_file(path),
            after_sha256=after_sha,
            snapshot_status="skipped",
            skipped_reason="non_utf8",
        )
    before_size = len(before.encode("utf-8"))
    if len(before) > MAX_SNAPSHOT_CHARS:
        return WorkspaceFileSnapshot(
            path=relative_path,
            operation=operation.strip(),
            before_existed=True,
            before_size_bytes=before_size,
            before_sha256=_sha256_text(before),
            after_sha256=after_sha,
            snapshot_status="skipped",
            skipped_reason="file_too_large",
        )
    return WorkspaceFileSnapshot(
        path=relative_path,
        operation=operation.strip(),
        before_existed=True,
        before_size_bytes=before_size,
        before_sha256=_sha256_text(before),
        before_content=before,
        after_sha256=after_sha,
    )


def _rewind_action(snapshot: WorkspaceFileSnapshot, workspace: Path) -> WorkspaceRewindAction:
    path = (workspace / snapshot.path).resolve()
    if snapshot.snapshot_status != "captured":
        return WorkspaceRewindAction(
            path=snapshot.path,
            action="skip",
            conflict=True,
            reason=snapshot.skipped_reason or "snapshot_not_captured",
        )
    current_sha = ""
    if path.exists() and path.is_file():
        try:
            current_sha = _sha256_text(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            return WorkspaceRewindAction(
                path=snapshot.path,
                action="restore" if snapshot.before_existed else "delete",
                conflict=True,
                reason="current_file_non_utf8",
            )
    expected = snapshot.after_sha256
    conflict = bool(expected and current_sha and current_sha != expected)
    action = "restore" if snapshot.before_existed else "delete"
    return WorkspaceRewindAction(
        path=snapshot.path,
        action=action,
        conflict=conflict,
        reason="current_content_differs" if conflict else "",
    )


def _snapshots_for_rewind(
    checkpoints: tuple[WorkspaceCheckpointRecord, ...],
    target_turn_id: str,
) -> tuple[WorkspaceFileSnapshot, ...]:
    target_number = _turn_number(target_turn_id)
    snapshots: list[WorkspaceFileSnapshot] = []
    seen: set[str] = set()
    for checkpoint in sorted(checkpoints, key=lambda record: _turn_number(record.turn_id)):
        if _turn_number(checkpoint.turn_id) <= target_number:
            continue
        for snapshot in checkpoint.files:
            if snapshot.path in seen:
                continue
            seen.add(snapshot.path)
            snapshots.append(snapshot)
    return tuple(snapshots)


def _snapshots_for_rewind_plan(
    checkpoints: tuple[WorkspaceCheckpointRecord, ...],
    target_turn_id: str,
) -> tuple[WorkspaceFileSnapshot, ...]:
    target_number = _turn_number(target_turn_id)
    by_path: dict[str, WorkspaceFileSnapshot] = {}
    last_after_by_path: dict[str, str] = {}
    for checkpoint in sorted(checkpoints, key=lambda record: _turn_number(record.turn_id)):
        if _turn_number(checkpoint.turn_id) <= target_number:
            continue
        for snapshot in checkpoint.files:
            by_path.setdefault(snapshot.path, snapshot)
            if snapshot.after_sha256:
                last_after_by_path[snapshot.path] = snapshot.after_sha256
    return tuple(
        replace(snapshot, after_sha256=last_after_by_path.get(snapshot.path, snapshot.after_sha256))
        for snapshot in by_path.values()
    )


def _relative_workspace_path(workspace: Path, path: Path) -> str:
    if path == workspace:
        return "."
    return path.relative_to(workspace).as_posix()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            while True:
                chunk = file.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def _path_size(path: Path) -> int:
    try:
        return max(0, int(path.stat().st_size))
    except OSError:
        return 0


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _clean_segment(value: str, *, fallback: str) -> str:
    clean = str(value).strip()
    if not clean:
        clean = fallback
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", clean).strip(".-") or fallback


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _int(value: object) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
