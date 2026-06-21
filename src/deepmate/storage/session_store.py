"""Session metadata and transcript storage."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from deepmate.domain import Message, MessageRole, ProfileRef
from deepmate.providers import (
    ModelConversationItem,
    ModelToolExchange,
    ModelToolRequest,
    ModelToolResult,
)
from deepmate.storage.atomic import atomic_write_json, atomic_write_text, file_lock
from deepmate.storage.jsonl import JsonlWriter
from deepmate.storage.tool_output_store import tool_output_ref_value


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """Durable session metadata without transcript payloads."""

    session_id: str
    title: str
    workspace: Path
    profile: ProfileRef
    created_at: str
    updated_at: str
    status: str
    transcript_path: Path
    parent_session_id: str = ""
    lineage_root_session_id: str = ""
    forked_from_turn_id: str = ""
    forked_from_sequence: int = 0
    fork_kind: str = ""
    fork_reason: str = ""
    forked_at: str = ""

    def is_ready(self) -> bool:
        """Return whether the session has enough metadata to persist."""
        return bool(
            self.session_id.strip()
            and self.title.strip()
            and str(self.workspace).strip()
            and self.profile.is_ready()
            and self.created_at.strip()
            and self.updated_at.strip()
            and self.status.strip()
            and str(self.transcript_path).strip()
        )

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable session metadata record."""
        return {
            "session_id": self.session_id,
            "title": self.title,
            "workspace": str(self.workspace),
            "profile": {
                "name": self.profile.name,
                "uri": self.profile.uri,
                "global_uri": self.profile.global_uri,
                "project_uri": self.profile.project_uri,
            },
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "transcript_path": str(self.transcript_path),
            "parent_session_id": self.parent_session_id,
            "lineage_root_session_id": self.lineage_root_session_id,
            "forked_from_turn_id": self.forked_from_turn_id,
            "forked_from_sequence": self.forked_from_sequence,
            "fork_kind": self.fork_kind,
            "fork_reason": self.fork_reason,
            "forked_at": self.forked_at,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "SessionRecord":
        """Build session metadata from a stored JSON record."""
        profile = _profile_from_record(record.get("profile"))
        return cls(
            session_id=str(record.get("session_id", "")).strip(),
            title=_session_title(record.get("title")),
            workspace=Path(str(record.get("workspace", ""))),
            profile=profile,
            created_at=str(record.get("created_at", "")).strip(),
            updated_at=str(record.get("updated_at", "")).strip(),
            status=str(record.get("status", "")).strip(),
            transcript_path=Path(str(record.get("transcript_path", ""))),
            parent_session_id=str(record.get("parent_session_id", "")).strip(),
            lineage_root_session_id=str(
                record.get("lineage_root_session_id", "")
            ).strip(),
            forked_from_turn_id=str(record.get("forked_from_turn_id", "")).strip(),
            forked_from_sequence=_int_value(record.get("forked_from_sequence")),
            fork_kind=str(record.get("fork_kind", "")).strip(),
            fork_reason=str(record.get("fork_reason", "")).strip(),
            forked_at=str(record.get("forked_at", "")).strip(),
        )


@dataclass(frozen=True, slots=True)
class SessionLineageNode:
    """One display node in a session lineage tree."""

    session: SessionRecord
    children: tuple["SessionLineageNode", ...] = ()


@dataclass(frozen=True, slots=True)
class TranscriptRecord:
    """Stored conversation item with a small persistence envelope."""

    session_id: str
    record_id: str
    recorded_at: str
    sequence: int
    kind: str
    payload: dict[str, object]

    @classmethod
    def from_item(
        cls,
        session_id: str,
        sequence: int,
        item: ModelConversationItem,
    ) -> "TranscriptRecord":
        """Build a stored record from one model-facing conversation item."""
        kind, payload = _item_payload(item)
        return cls(
            session_id=session_id.strip(),
            record_id=uuid4().hex,
            recorded_at=_utc_now_iso(),
            sequence=sequence,
            kind=kind,
            payload=payload,
        )

    def is_ready(self) -> bool:
        """Return whether the record has enough identity and content to store."""
        return bool(
            self.session_id.strip()
            and self.record_id.strip()
            and self.recorded_at.strip()
            and self.sequence > 0
            and self.kind.strip()
            and self.payload
        )

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable record."""
        return {
            "session_id": self.session_id,
            "record_id": self.record_id,
            "recorded_at": self.recorded_at,
            "sequence": self.sequence,
            "kind": self.kind,
            "payload": self.payload,
        }

    def to_item(self) -> ModelConversationItem | None:
        """Return the provider-neutral conversation item for this record."""
        return _item_from_record(self.to_record())

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "TranscriptRecord":
        """Build a transcript record from a stored JSON object."""
        payload = record.get("payload")
        return cls(
            session_id=str(record.get("session_id", "")).strip(),
            record_id=str(record.get("record_id", "")).strip(),
            recorded_at=str(record.get("recorded_at", "")).strip(),
            sequence=_int_value(record.get("sequence")),
            kind=str(record.get("kind", "")).strip(),
            payload=payload if isinstance(payload, dict) else {},
        )


@dataclass(frozen=True, slots=True)
class SessionSummaryRecord:
    """Latest compact model-facing view of older transcript items."""

    session_id: str
    summary_id: str
    created_at: str
    content: str
    covered_until_sequence: int
    covered_item_count: int
    source_item_count: int
    estimated_source_tokens: int
    source_model: str
    usage: dict[str, object] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        session_id: str,
        content: str,
        covered_until_sequence: int,
        covered_item_count: int,
        source_item_count: int,
        estimated_source_tokens: int,
        source_model: str,
        usage: Mapping[str, object] | None = None,
    ) -> "SessionSummaryRecord":
        """Create a ready summary record with generated identity fields."""
        return cls(
            session_id=session_id.strip(),
            summary_id=uuid4().hex,
            created_at=_utc_now_iso(),
            content=content.strip(),
            covered_until_sequence=covered_until_sequence,
            covered_item_count=covered_item_count,
            source_item_count=source_item_count,
            estimated_source_tokens=estimated_source_tokens,
            source_model=source_model.strip(),
            usage=dict(usage or {}),
        )

    def is_ready(self) -> bool:
        """Return whether the summary can be used as model-facing context."""
        return bool(
            self.session_id.strip()
            and self.summary_id.strip()
            and self.created_at.strip()
            and self.content.strip()
            and self.covered_until_sequence > 0
            and self.covered_item_count > 0
            and self.source_item_count > 0
            and self.estimated_source_tokens >= 0
            and self.source_model.strip()
        )

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable summary record."""
        return {
            "session_id": self.session_id,
            "summary_id": self.summary_id,
            "created_at": self.created_at,
            "content": self.content,
            "covered_until_sequence": self.covered_until_sequence,
            "covered_item_count": self.covered_item_count,
            "source_item_count": self.source_item_count,
            "estimated_source_tokens": self.estimated_source_tokens,
            "source_model": self.source_model,
            "usage": self.usage,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "SessionSummaryRecord":
        """Build a summary record from stored JSON."""
        usage = record.get("usage")
        return cls(
            session_id=str(record.get("session_id", "")).strip(),
            summary_id=str(record.get("summary_id", "")).strip(),
            created_at=str(record.get("created_at", "")).strip(),
            content=str(record.get("content", "")).strip(),
            covered_until_sequence=_int_value(record.get("covered_until_sequence")),
            covered_item_count=_int_value(record.get("covered_item_count")),
            source_item_count=_int_value(record.get("source_item_count")),
            estimated_source_tokens=_int_value(record.get("estimated_source_tokens")),
            source_model=str(record.get("source_model", "")).strip(),
            usage=dict(usage) if isinstance(usage, dict) else {},
        )


class SessionStore:
    """Create and load lightweight durable session metadata."""

    @classmethod
    def in_directory(cls, directory: str | Path) -> "SessionStore":
        """Create a session store rooted at a directory."""
        return cls(directory)

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def create(
        self,
        workspace: str | Path,
        profile: ProfileRef,
        title: str = "",
    ) -> SessionRecord:
        """Create a new active session metadata record."""
        session_id = uuid4().hex
        now = _utc_now_iso()
        record = SessionRecord(
            session_id=session_id,
            title=_session_title(title),
            workspace=Path(workspace).resolve(),
            profile=profile,
            created_at=now,
            updated_at=now,
            status="active",
            transcript_path=self.transcript_path(session_id),
        )
        if not record.is_ready():
            raise ValueError("session record is not ready")
        self._write(record)
        return record

    def list_recent(self, limit: int = 20) -> tuple[SessionRecord, ...]:
        """Return recent session metadata records ordered by updated time."""
        if limit < 1:
            return ()
        records: list[SessionRecord] = []
        if not self.directory.exists():
            return ()
        for path in self.directory.glob("*.json"):
            if path.name.endswith(".summary.json"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            record = SessionRecord.from_record(data)
            if record.is_ready():
                records.append(record)
        records.sort(key=lambda record: record.updated_at, reverse=True)
        return tuple(records[:limit])

    def list_recent_for_workspace(
        self,
        workspace: str | Path,
        limit: int = 20,
    ) -> tuple[SessionRecord, ...]:
        """Return recent sessions belonging to one resolved workspace."""
        if limit < 1:
            return ()
        resolved_workspace = Path(workspace).resolve()
        matches = [
            record
            for record in self.list_recent(limit=10_000)
            if record.workspace.resolve() == resolved_workspace
        ]
        return tuple(matches[:limit])

    def latest_for_workspace(self, workspace: str | Path) -> SessionRecord | None:
        """Return the most recently updated session for a workspace, if any."""
        recent = self.list_recent_for_workspace(workspace, limit=1)
        return recent[0] if recent else None

    def resolve_id(self, value: str) -> str:
        """Resolve a full session id or unique session id prefix."""
        clean_value = value.strip()
        if not clean_value:
            raise ValueError("session_id is required")
        if self.metadata_path(clean_value).exists():
            return clean_value
        matches = [
            record.session_id
            for record in self.list_recent(limit=10_000)
            if record.session_id.startswith(clean_value)
        ]
        if not matches:
            raise ValueError(f"session not found: {clean_value}")
        if len(matches) > 1:
            raise ValueError(f"session id prefix is ambiguous: {clean_value}")
        return matches[0]

    def rename(self, session_id: str, title: str) -> SessionRecord:
        """Update the human-readable title for a session."""
        record = self.load(session_id)
        updated = replace(
            record,
            title=_session_title(title),
            updated_at=_utc_now_iso(),
        )
        self._write(updated)
        return updated

    def touch(self, session_id: str) -> SessionRecord:
        """Update the session's activity timestamp."""
        record = self.load(session_id)
        updated = replace(record, updated_at=_utc_now_iso())
        self._write(updated)
        return updated

    def touch_record(self, record: SessionRecord) -> SessionRecord:
        """Update activity, restoring missing metadata from an active record."""
        if not record.is_ready():
            raise ValueError("session record is not ready")
        try:
            current = self.load(record.session_id)
        except ValueError:
            if self.metadata_path(record.session_id).exists():
                raise
            current = record
        updated = replace(current, updated_at=_utc_now_iso())
        self._write(updated)
        return updated

    def update_status(self, session_id: str, status: str) -> SessionRecord:
        """Update the session status."""
        clean_status = status.strip()
        if not clean_status:
            raise ValueError("status is required")
        record = self.load(session_id)
        updated = replace(record, status=clean_status, updated_at=_utc_now_iso())
        self._write(updated)
        return updated

    def clone_session(
        self,
        source: SessionRecord,
        title: str = "",
        reason: str = "",
    ) -> SessionRecord:
        """Clone the complete source session into a new linear session."""
        records = self.transcript_store(source).load_records()
        latest_sequence = max((record.sequence for record in records), default=0)
        return self._copy_session(
            source=source,
            records=records,
            title=title or f"{source.title} copy",
            fork_kind="clone",
            forked_from_sequence=latest_sequence,
            forked_from_turn_id="",
            reason=reason,
        )

    def fork_session_at_sequence(
        self,
        source: SessionRecord,
        sequence: int,
        title: str = "",
        turn_id: str = "",
        reason: str = "",
    ) -> SessionRecord:
        """Fork a source session into a new linear session up to a sequence."""
        fork_sequence = int(sequence)
        if fork_sequence < 1:
            raise ValueError("fork sequence must be greater than 0")
        records = tuple(
            record
            for record in self.transcript_store(source).load_records()
            if record.sequence <= fork_sequence
        )
        if not records:
            raise ValueError(f"no transcript records found up to sequence {fork_sequence}")
        copied_until = max(record.sequence for record in records)
        return self._copy_session(
            source=source,
            records=records,
            title=title or f"{source.title} fork",
            fork_kind="fork",
            forked_from_sequence=copied_until,
            forked_from_turn_id=turn_id,
            reason=reason,
        )

    def lineage_tree(
        self,
        workspace: str | Path | None = None,
        profile: ProfileRef | None = None,
        limit: int = 10_000,
    ) -> tuple[SessionLineageNode, ...]:
        """Return lineage roots with children grouped by parent session id."""
        records = list(self.list_recent(limit=limit))
        if workspace is not None:
            resolved_workspace = Path(workspace).resolve()
            records = [
                record
                for record in records
                if record.workspace.resolve() == resolved_workspace
            ]
        if profile is not None:
            records = [record for record in records if record.profile == profile]
        by_parent: dict[str, list[SessionRecord]] = {}
        by_id = {record.session_id: record for record in records}
        for record in records:
            parent_id = (
                record.parent_session_id
                if record.parent_session_id in by_id
                else ""
            )
            by_parent.setdefault(parent_id, []).append(record)
        for siblings in by_parent.values():
            siblings.sort(key=lambda item: (item.created_at, item.session_id))

        def build(record: SessionRecord) -> SessionLineageNode:
            return SessionLineageNode(
                session=record,
                children=tuple(build(child) for child in by_parent.get(record.session_id, ())),
            )

        roots = by_parent.get("", [])
        roots.sort(key=lambda item: (item.created_at, item.session_id))
        return tuple(build(record) for record in roots)

    def load(self, session_id: str) -> SessionRecord:
        """Load an existing session metadata record by id."""
        clean_id = session_id.strip()
        if not clean_id:
            raise ValueError("session_id is required")
        path = self.metadata_path(clean_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"unknown session: {clean_id}") from exc
        if not isinstance(data, dict):
            raise ValueError("session metadata must be a JSON object")
        record = SessionRecord.from_record(data)
        if not record.is_ready():
            raise ValueError("session record is not ready")
        return record

    def metadata_path(self, session_id: str) -> Path:
        """Return the metadata JSON path for a session id."""
        return self.directory / f"{session_id.strip()}.json"

    def transcript_path(self, session_id: str) -> Path:
        """Return the transcript JSONL path for a session id."""
        return self.directory / f"{session_id.strip()}.jsonl"

    def summary_path(self, session_id: str) -> Path:
        """Return the latest summary JSON path for a session id."""
        return self.directory / f"{session_id.strip()}.summary.json"

    def transcript_store(self, session: SessionRecord) -> "TranscriptStore":
        """Return a transcript append store for a session."""
        return TranscriptStore(session.transcript_path, session_id=session.session_id)

    def summary_store(self, session: SessionRecord) -> "SessionSummaryStore":
        """Return the summary store for a session."""
        return SessionSummaryStore(
            self.summary_path(session.session_id),
            session_id=session.session_id,
        )

    def _copy_session(
        self,
        source: SessionRecord,
        records: tuple[TranscriptRecord, ...],
        title: str,
        fork_kind: str,
        forked_from_sequence: int,
        forked_from_turn_id: str,
        reason: str,
    ) -> SessionRecord:
        session_id = uuid4().hex
        now = _utc_now_iso()
        root_id = source.lineage_root_session_id or source.session_id
        record = SessionRecord(
            session_id=session_id,
            title=_session_title(title),
            workspace=source.workspace,
            profile=source.profile,
            created_at=now,
            updated_at=now,
            status="active",
            transcript_path=self.transcript_path(session_id),
            parent_session_id=source.session_id,
            lineage_root_session_id=root_id,
            forked_from_turn_id=forked_from_turn_id.strip(),
            forked_from_sequence=max(0, int(forked_from_sequence)),
            fork_kind=fork_kind.strip(),
            fork_reason=reason.strip() or title.strip(),
            forked_at=now,
        )
        if not record.is_ready():
            raise ValueError("session record is not ready")
        self._write(record)
        _write_transcript_records(self.transcript_path(session_id), session_id, records)
        _copy_tool_output_refs(
            source_session_id=source.session_id,
            target_session_id=session_id,
            profile=source.profile.name,
            source_records=records,
            data_dir=self.directory.parent,
        )
        self._copy_summary_if_covered(
            source=source,
            target=record,
            forked_from_sequence=record.forked_from_sequence,
        )
        return record

    def _copy_summary_if_covered(
        self,
        source: SessionRecord,
        target: SessionRecord,
        forked_from_sequence: int,
    ) -> None:
        try:
            summary = self.summary_store(source).load_latest()
        except (OSError, ValueError, json.JSONDecodeError):
            return
        if summary is None:
            return
        if summary.covered_until_sequence > forked_from_sequence:
            return
        copied = replace(
            summary,
            session_id=target.session_id,
            summary_id=uuid4().hex,
            created_at=_utc_now_iso(),
        )
        self.summary_store(target).save_latest(copied)

    def _write(self, record: SessionRecord) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.metadata_path(record.session_id), record.to_record())


class TranscriptStore:
    """Append session transcript items to one JSONL file."""

    def __init__(self, path: str | Path, session_id: str | None = None) -> None:
        self.path = Path(path)
        self.session_id = (session_id or uuid4().hex).strip()
        self.writer = JsonlWriter(self.path)
        self._sequence = _existing_jsonl_max_sequence(self.path)
        if not self.session_id:
            raise ValueError("session_id is required")

    def append_item(self, item: ModelConversationItem) -> TranscriptRecord | None:
        """Persist one transcript item and return the stored record."""
        if not item.is_ready():
            raise ValueError("conversation item must be ready")
        if item.message is not None and item.message.role == MessageRole.SYSTEM:
            return None
        with file_lock(self.path):
            self._sequence = _existing_jsonl_max_sequence(self.path)
            next_sequence = self._sequence + 1
            record = TranscriptRecord.from_item(
                session_id=self.session_id,
                sequence=next_sequence,
                item=item,
            )
            if not record.is_ready():
                raise ValueError("transcript record is not ready")
            line = json.dumps(
                record.to_record(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
                file.flush()
                os.fsync(file.fileno())
            self._sequence = next_sequence
            return record

    def load_items(self) -> tuple[ModelConversationItem, ...]:
        """Load stored transcript items in sequence order."""
        return tuple(
            item
            for record in self.load_records()
            for item in (record.to_item(),)
            if item is not None
        )

    def load_items_after(self, sequence: int) -> tuple[ModelConversationItem, ...]:
        """Load transcript items with a sequence greater than the given value."""
        return tuple(
            item
            for record in self.load_records()
            if record.sequence > sequence
            for item in (record.to_item(),)
            if item is not None
        )

    def load_records(self) -> tuple[TranscriptRecord, ...]:
        """Load stored transcript records in sequence order."""
        if not self.path.exists():
            return ()
        records: list[TranscriptRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(raw, dict):
                continue
            record = TranscriptRecord.from_record(raw)
            if record.is_ready():
                records.append(record)
        records.sort(key=lambda record: record.sequence)
        return tuple(records)

    def truncate_after(self, sequence: int) -> int:
        """Rewrite transcript to keep only records up to sequence."""
        keep_until = max(0, int(sequence))
        kept = tuple(
            record for record in self.load_records() if record.sequence <= keep_until
        )
        lines = "\n".join(
            json.dumps(record.to_record(), ensure_ascii=False, separators=(",", ":"))
            for record in kept
        )
        atomic_write_text(self.path, (lines + "\n") if lines else "")
        self._sequence = _existing_jsonl_max_sequence(self.path)
        return len(kept)


class SessionSummaryStore:
    """Persist and load the latest summary for one session."""

    def __init__(self, path: str | Path, session_id: str) -> None:
        self.path = Path(path)
        self.session_id = session_id.strip()
        if not self.session_id:
            raise ValueError("session_id is required")

    def save_latest(self, record: SessionSummaryRecord) -> None:
        """Persist the latest summary record, replacing any previous summary."""
        if record.session_id != self.session_id:
            raise ValueError("summary record belongs to another session")
        if not record.is_ready():
            raise ValueError("session summary record is not ready")
        atomic_write_json(self.path, record.to_record())

    def load_latest(self) -> SessionSummaryRecord | None:
        """Load the latest summary record if one exists."""
        if not self.path.exists():
            return None
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("session summary must be a JSON object")
        record = SessionSummaryRecord.from_record(data)
        if record.session_id != self.session_id:
            raise ValueError("session summary belongs to another session")
        if not record.is_ready():
            raise ValueError("session summary record is not ready")
        return record

    def delete_latest(self) -> None:
        """Delete the latest summary record if it exists."""
        self.path.unlink(missing_ok=True)


def _item_payload(item: ModelConversationItem) -> tuple[str, dict[str, object]]:
    if item.message is not None:
        return (
            "message",
            {
                "role": item.message.role.value,
                "content": item.message.content,
            },
        )
    if item.tool_exchange is not None:
        return "tool_exchange", _tool_exchange_payload(item.tool_exchange)
    raise ValueError("conversation item requires message or tool exchange")


def _item_from_record(record: dict[str, Any]) -> ModelConversationItem | None:
    kind = str(record.get("kind", "")).strip()
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None
    if kind == "message":
        role = _message_role(payload.get("role"))
        content = str(payload.get("content", ""))
        if role is None or not content.strip():
            return None
        return ModelConversationItem.from_message(Message(role=role, content=content))
    if kind == "tool_exchange":
        exchange = _tool_exchange_from_payload(payload)
        if exchange.is_ready():
            return ModelConversationItem.from_tool_exchange(exchange)
    return None


def _tool_exchange_payload(exchange: ModelToolExchange) -> dict[str, object]:
    return {
        "assistant_content": exchange.assistant_content,
        "assistant_reasoning": exchange.assistant_reasoning,
        "tool_requests": [
            {
                "id": request.id,
                "name": request.name,
                "arguments": dict(request.arguments),
                "raw_arguments": request.raw_arguments,
            }
            for request in exchange.tool_requests
        ],
        "tool_results": [_tool_result_payload(result) for result in exchange.tool_results],
    }


def _tool_exchange_from_payload(payload: dict[str, Any]) -> ModelToolExchange:
    tool_requests = []
    raw_requests = payload.get("tool_requests")
    if isinstance(raw_requests, list):
        for item in raw_requests:
            if not isinstance(item, dict):
                continue
            arguments = item.get("arguments")
            tool_requests.append(
                ModelToolRequest(
                    id=str(item.get("id", "")).strip(),
                    name=str(item.get("name", "")).strip(),
                    arguments=arguments if isinstance(arguments, dict) else {},
                    raw_arguments=str(item.get("raw_arguments", "")),
                )
            )
    tool_results = []
    raw_results = payload.get("tool_results")
    if isinstance(raw_results, list):
        for item in raw_results:
            if isinstance(item, dict):
                tool_results.append(_tool_result_from_payload(item))
    return ModelToolExchange(
        assistant_content=str(payload.get("assistant_content", "")),
        assistant_reasoning=str(payload.get("assistant_reasoning", "")),
        tool_requests=tuple(tool_requests),
        tool_results=tuple(tool_results),
    )


def _tool_result_payload(result: ModelToolResult) -> dict[str, object]:
    return {
        "request_id": result.request_id,
        "name": result.name,
        "content": result.content,
        "data": dict(result.data),
        "refs": list(result.refs),
        "attachments": [dict(item) for item in result.attachments],
        "is_error": result.is_error,
    }


def _tool_result_from_payload(payload: dict[str, Any]) -> ModelToolResult:
    data = payload.get("data")
    refs = payload.get("refs")
    attachments = payload.get("attachments")
    return ModelToolResult(
        request_id=str(payload.get("request_id", "")),
        name=str(payload.get("name", "")),
        content=str(payload.get("content", "")),
        data=data if isinstance(data, dict) else {},
        refs=tuple(str(ref) for ref in refs) if isinstance(refs, list) else (),
        attachments=tuple(
            item for item in attachments if isinstance(item, dict)
        )
        if isinstance(attachments, list)
        else (),
        is_error=bool(payload.get("is_error", False)),
    )


def _write_transcript_records(
    path: Path,
    session_id: str,
    records: tuple[TranscriptRecord, ...],
) -> None:
    lines = "\n".join(
        json.dumps(
            replace(
                record,
                session_id=session_id,
                record_id=uuid4().hex,
                recorded_at=_utc_now_iso(),
            ).to_record(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for record in records
    )
    atomic_write_text(path, (lines + "\n") if lines else "")


def _copy_tool_output_refs(
    *,
    source_session_id: str,
    target_session_id: str,
    profile: str,
    source_records: tuple[TranscriptRecord, ...],
    data_dir: Path,
) -> None:
    refs = _tool_output_refs_from_records(source_records)
    if not refs:
        return
    source_dir = data_dir / "tool_outputs" / profile / source_session_id
    target_dir = data_dir / "tool_outputs" / profile / target_session_id
    for ref in refs:
        source_path = source_dir / f"{ref}.json"
        if not source_path.exists():
            continue
        target_path = target_dir / f"{ref}.json"
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        payload["session_id"] = target_session_id
        atomic_write_json(target_path, payload)


def _tool_output_refs_from_records(records: tuple[TranscriptRecord, ...]) -> tuple[str, ...]:
    refs: list[str] = []
    seen: set[str] = set()
    for record in records:
        payload = record.payload
        raw_results = payload.get("tool_results")
        if not isinstance(raw_results, list):
            continue
        for raw_result in raw_results:
            if not isinstance(raw_result, dict):
                continue
            raw_refs = raw_result.get("refs")
            if not isinstance(raw_refs, list):
                continue
            for raw_ref in raw_refs:
                ref = tool_output_ref_value(str(raw_ref))
                if ref and ref not in seen:
                    seen.add(ref)
                    refs.append(ref)
    return tuple(refs)


def _message_role(value: object) -> MessageRole | None:
    try:
        return MessageRole(str(value))
    except ValueError:
        return None


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def _profile_from_record(value: object) -> ProfileRef:
    if isinstance(value, dict):
        return ProfileRef(
            name=str(value.get("name", "")).strip(),
            uri=str(value.get("uri", "")).strip(),
            global_uri=str(value.get("global_uri", "")).strip(),
            project_uri=str(value.get("project_uri", "")).strip(),
        )
    return ProfileRef(name="", uri="")


def _session_title(value: object) -> str:
    title = str(value or "").strip()
    return title or "Untitled session"


def _existing_jsonl_max_sequence(path: Path) -> int:
    if not path.exists():
        return 0
    max_sequence = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        try:
            max_sequence = max(max_sequence, int(record.get("sequence", 0)))
        except (TypeError, ValueError):
            continue
    return max_sequence
