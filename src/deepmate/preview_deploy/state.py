"""Durable state for one host-level preview deploy."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from deepmate.storage import atomic_write_json, file_lock


def now_iso() -> str:
    """Return a compact local timestamp."""
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def expires_at_iso(ttl_seconds: int) -> str:
    """Return an expiry timestamp ttl seconds from now."""
    ttl = max(1, int(ttl_seconds))
    return (
        datetime.now().astimezone() + timedelta(seconds=ttl)
    ).replace(microsecond=0).isoformat()


def parse_datetime(value: str) -> datetime | None:
    """Parse an ISO timestamp, accepting UTC Z suffix."""
    clean = value.strip()
    if not clean:
        return None
    if clean.endswith("Z"):
        clean = clean[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(clean)
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class PreviewDeployState:
    """Persisted state for one active or recently stopped preview."""

    status: str
    owner_session_id: str = ""
    owner_session_workspace: str = ""
    target_path: str = ""
    target_kind: str = ""
    project_name: str = ""
    project_slug: str = ""
    local_url: str = ""
    lan_url: str = ""
    public_url: str = ""
    lease_id: str = ""
    provider: str = "local"
    started_at: str = ""
    expires_at: str = ""
    stopped_at: str = ""
    supervisor_pid: int = 0
    tunnel_pid: int = 0
    process_ids: tuple[int, ...] = ()
    local_service_owner: str = "external"
    wake_lock_id: str = ""
    message: str = ""

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "PreviewDeployState":
        """Build state from a JSON-compatible record."""
        return cls(
            status=str(record.get("status", "")).strip() or "unknown",
            owner_session_id=str(record.get("owner_session_id", "")).strip(),
            owner_session_workspace=str(
                record.get("owner_session_workspace", "")
            ).strip(),
            target_path=str(record.get("target_path", "")).strip(),
            target_kind=str(record.get("target_kind", "")).strip(),
            project_name=str(record.get("project_name", "")).strip(),
            project_slug=str(record.get("project_slug", "")).strip(),
            local_url=str(record.get("local_url", "")).strip(),
            lan_url=str(record.get("lan_url", "")).strip(),
            public_url=str(record.get("public_url", "")).strip(),
            lease_id=str(record.get("lease_id", "")).strip(),
            provider=str(record.get("provider", "local")).strip() or "local",
            started_at=str(record.get("started_at", "")).strip(),
            expires_at=str(record.get("expires_at", "")).strip(),
            stopped_at=str(record.get("stopped_at", "")).strip(),
            supervisor_pid=_int_value(record.get("supervisor_pid")),
            tunnel_pid=_int_value(record.get("tunnel_pid")),
            process_ids=_int_tuple(record.get("process_ids")),
            local_service_owner=str(
                record.get("local_service_owner", "external")
            ).strip()
            or "external",
            wake_lock_id=str(record.get("wake_lock_id", "")).strip(),
            message=str(record.get("message", "")).strip(),
        )

    def to_record(self) -> dict[str, object]:
        """Return a JSON-compatible record."""
        return {
            "status": self.status,
            "owner_session_id": self.owner_session_id,
            "owner_session_workspace": self.owner_session_workspace,
            "target_path": self.target_path,
            "target_kind": self.target_kind,
            "project_name": self.project_name,
            "project_slug": self.project_slug,
            "local_url": self.local_url,
            "lan_url": self.lan_url,
            "public_url": self.public_url,
            "lease_id": self.lease_id,
            "provider": self.provider,
            "started_at": self.started_at,
            "expires_at": self.expires_at,
            "stopped_at": self.stopped_at,
            "supervisor_pid": self.supervisor_pid,
            "tunnel_pid": self.tunnel_pid,
            "process_ids": list(self.process_ids),
            "local_service_owner": self.local_service_owner,
            "wake_lock_id": self.wake_lock_id,
            "message": self.message,
        }

    def is_active(self) -> bool:
        """Return whether the state represents a currently claimed preview."""
        return self.status in {"running", "unhealthy"}

    def with_status(self, status: str, *, message: str = "") -> "PreviewDeployState":
        """Return a state copy with status and optional stopped timestamp."""
        clean_status = status.strip() or "unknown"
        stopped_at = self.stopped_at
        if clean_status in {"stopped", "expired", "stale"} and not stopped_at:
            stopped_at = now_iso()
        return replace(
            self,
            status=clean_status,
            message=message.strip() if message else self.message,
            stopped_at=stopped_at,
        )


class PreviewDeployStore:
    """Small JSON store for preview deploy state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def in_data_dir(cls, data_dir: str | Path) -> "PreviewDeployStore":
        return cls(Path(data_dir) / "preview_deploy" / "state.json")

    def load(self) -> PreviewDeployState | None:
        """Load preview state if present."""
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        if not isinstance(payload, dict):
            raise ValueError(f"preview deploy state must be a JSON object: {self.path}")
        return PreviewDeployState.from_record(payload)

    def save(self, state: PreviewDeployState) -> PreviewDeployState:
        """Persist preview state with a file lock."""
        with file_lock(self.path):
            atomic_write_json(self.path, state.to_record())
        return state

    def clear(self) -> None:
        """Remove persisted state if present."""
        with file_lock(self.path):
            try:
                self.path.unlink()
            except FileNotFoundError:
                return


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float) and value.is_integer():
        return max(0, int(value))
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return 0


def _int_tuple(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    output: list[int] = []
    for item in value:
        parsed = _int_value(item)
        if parsed > 0:
            output.append(parsed)
    return tuple(output)
