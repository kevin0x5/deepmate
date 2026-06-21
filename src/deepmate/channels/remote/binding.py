"""Durable remote channel to session bindings."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from deepmate.storage.atomic import atomic_write_json, file_lock
from deepmate.storage.session_store import SessionRecord


def _now_iso() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


@dataclass(frozen=True, slots=True)
class RemoteBindingRecord:
    """One channel/user to Deepmate session binding."""

    channel: str
    remote_user_id: str
    session_id: str
    session_title: str
    workspace: Path
    profile: str
    bound_at: str
    bound_from: str = "remote"
    route_open: bool = False
    last_opened_at: str = ""

    @classmethod
    def from_session(
        cls,
        *,
        channel: str,
        remote_user_id: str,
        session: SessionRecord,
        bound_from: str,
    ) -> "RemoteBindingRecord":
        return cls(
            channel=channel.strip(),
            remote_user_id=remote_user_id.strip(),
            session_id=session.session_id,
            session_title=session.title,
            workspace=session.workspace,
            profile=session.profile.name,
            bound_at=_now_iso(),
            bound_from=bound_from.strip() or "remote",
            route_open=bound_from == "remote",
            last_opened_at=_now_iso() if bound_from == "remote" else "",
        )

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "RemoteBindingRecord":
        return cls(
            channel=str(record.get("channel", "")).strip(),
            remote_user_id=str(record.get("remote_user_id", "")).strip(),
            session_id=str(record.get("session_id", "")).strip(),
            session_title=str(record.get("session_title", "")).strip(),
            workspace=Path(str(record.get("workspace", ""))),
            profile=str(record.get("profile", "")).strip(),
            bound_at=str(record.get("bound_at", "")).strip(),
            bound_from=str(record.get("bound_from", "remote")).strip() or "remote",
            route_open=record.get("route_open") is True,
            last_opened_at=str(record.get("last_opened_at", "")).strip(),
        )

    def is_ready(self) -> bool:
        return bool(
            self.channel
            and self.remote_user_id
            and self.session_id
            and self.session_title
            and str(self.workspace)
            and self.profile
            and self.bound_at
        )

    def to_record(self) -> dict[str, object]:
        return {
            "channel": self.channel,
            "remote_user_id": self.remote_user_id,
            "session_id": self.session_id,
            "session_title": self.session_title,
            "workspace": str(self.workspace),
            "profile": self.profile,
            "bound_at": self.bound_at,
            "bound_from": self.bound_from,
            "route_open": self.route_open,
            "last_opened_at": self.last_opened_at,
        }

    def refreshed_from_session(
        self,
        session: SessionRecord,
        *,
        bound_from: str | None = None,
    ) -> "RemoteBindingRecord":
        return replace(
            self,
            session_id=session.session_id,
            session_title=session.title,
            workspace=session.workspace,
            profile=session.profile.name,
            bound_at=_now_iso(),
            bound_from=(bound_from or self.bound_from).strip() or "remote",
        )

    def with_route(self, *, open: bool) -> "RemoteBindingRecord":
        """Return a copy with remote delivery route open or closed."""
        return replace(
            self,
            route_open=open,
            last_opened_at=_now_iso() if open else self.last_opened_at,
        )


class RemoteBindingStore:
    """Persistent remote binding store backed by one JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def in_data_dir(cls, data_dir: str | Path) -> "RemoteBindingStore":
        return cls(Path(data_dir) / "remote" / "bindings.json")

    def get(self, channel: str, remote_user_id: str) -> RemoteBindingRecord | None:
        key = _binding_key(channel, remote_user_id)
        records = self._load_records()
        return records.get(key)

    def upsert(self, record: RemoteBindingRecord) -> RemoteBindingRecord:
        if not record.is_ready():
            raise ValueError("remote binding record is not ready")
        key = _binding_key(record.channel, record.remote_user_id)
        with file_lock(self.path):
            records = self._load_records()
            records[key] = record
            self._save_records(records)
        return record

    def bind_session(
        self,
        *,
        channel: str,
        remote_user_id: str,
        session: SessionRecord,
        bound_from: str,
    ) -> RemoteBindingRecord:
        return self.upsert(
            RemoteBindingRecord.from_session(
                channel=channel,
                remote_user_id=remote_user_id,
                session=session,
                bound_from=bound_from,
            )
        )

    def bind_default_session(
        self,
        *,
        channel: str,
        session: SessionRecord,
        bound_from: str,
        route_open: bool = False,
    ) -> RemoteBindingRecord:
        """Replace one channel's bindings with a workspace-local default session."""
        record = RemoteBindingRecord.from_session(
            channel=channel,
            remote_user_id="*",
            session=session,
            bound_from=bound_from,
        )
        if record.route_open != route_open:
            record = record.with_route(open=route_open)
        with file_lock(self.path):
            records = self._load_records()
            records = {
                key: existing
                for key, existing in records.items()
                if existing.channel != record.channel
            }
            records[_binding_key(record.channel, record.remote_user_id)] = record
            self._save_records(records)
        return record

    def remove(self, channel: str, remote_user_id: str) -> RemoteBindingRecord | None:
        key = _binding_key(channel, remote_user_id)
        with file_lock(self.path):
            records = self._load_records()
            removed = records.pop(key, None)
            self._save_records(records)
        return removed

    def remove_channel(self, channel: str) -> tuple[RemoteBindingRecord, ...]:
        """Remove every binding for one remote channel."""
        clean_channel = channel.strip()
        if not clean_channel:
            raise ValueError("remote channel is required")
        with file_lock(self.path):
            records = self._load_records()
            removed = tuple(
                record for record in records.values() if record.channel == clean_channel
            )
            records = {
                key: record
                for key, record in records.items()
                if record.channel != clean_channel
            }
            self._save_records(records)
        return removed

    def close_session_routes(
        self,
        *,
        session_id: str,
        channel: str | None = None,
    ) -> tuple[RemoteBindingRecord, ...]:
        """Close open remote delivery routes for a local session."""
        clean_session_id = session_id.strip()
        if not clean_session_id:
            return ()
        clean_channel = channel.strip() if channel is not None else None
        changed: list[RemoteBindingRecord] = []
        with file_lock(self.path):
            records = self._load_records()
            for key, record in tuple(records.items()):
                if record.session_id != clean_session_id or not record.route_open:
                    continue
                if clean_channel is not None and record.channel != clean_channel:
                    continue
                closed = record.with_route(open=False)
                records[key] = closed
                changed.append(closed)
            if changed:
                self._save_records(records)
        return tuple(changed)

    def open_session_route(
        self,
        *,
        channel: str,
        remote_user_id: str,
    ) -> RemoteBindingRecord:
        """Open one existing binding as the active remote delivery route."""
        key = _binding_key(channel, remote_user_id)
        with file_lock(self.path):
            records = self._load_records()
            record = records.get(key)
            if record is None:
                raise ValueError("remote binding does not exist")
            opened = record.with_route(open=True)
            records[key] = opened
            self._save_records(records)
        return opened

    def list_records(self) -> tuple[RemoteBindingRecord, ...]:
        records = tuple(self._load_records().values())
        return tuple(sorted(records, key=lambda record: record.bound_at, reverse=True))

    def _load_records(self) -> dict[str, RemoteBindingRecord]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"remote binding store must be JSON: {self.path}") from exc
        if not isinstance(payload, dict):
            raise ValueError("remote binding store must be a JSON object")
        raw_records = payload.get("bindings", {})
        if not isinstance(raw_records, dict):
            raise ValueError("remote binding store bindings must be a JSON object")
        records: dict[str, RemoteBindingRecord] = {}
        for key, value in raw_records.items():
            if not isinstance(value, dict):
                continue
            record = RemoteBindingRecord.from_record(value)
            if record.is_ready():
                records[str(key)] = record
        return records

    def _save_records(self, records: dict[str, RemoteBindingRecord]) -> None:
        atomic_write_json(
            self.path,
            {"bindings": {key: record.to_record() for key, record in records.items()}},
        )


def format_remote_binding_status(
    records: tuple[RemoteBindingRecord, ...],
    *,
    channel: str | None = None,
) -> str:
    filtered = tuple(
        record
        for record in records
        if channel is None or record.channel == channel.strip()
    )
    if not filtered:
        return "No remote bindings."
    lines = ["Remote bindings:"]
    for record in filtered:
        lines.extend(
            (
                f"- {record.channel}:{record.remote_user_id}",
                f"  session: {record.session_id} ({record.session_title})",
                f"  workspace: {record.workspace}",
                f"  profile: {record.profile}",
                f"  route: {'open' if record.route_open else 'closed'}",
                f"  bound_at: {record.bound_at}",
            )
        )
    return "\n".join(lines)


def _binding_key(channel: str, remote_user_id: str) -> str:
    clean_channel = channel.strip()
    clean_user = remote_user_id.strip()
    if not clean_channel:
        raise ValueError("remote channel is required")
    if not clean_user:
        raise ValueError("remote user id is required")
    return f"{clean_channel}:{clean_user}"
