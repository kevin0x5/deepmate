"""Workspace trust storage for project hooks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from deepmate.runtime.hooks.types import utc_now_iso
from deepmate.storage import atomic_write_json, file_lock

HOOK_TRUST_FILE = "trust.json"


@dataclass(frozen=True, slots=True)
class TrustedWorkspace:
    """One locally trusted workspace record."""

    workspace: str
    workspace_hash: str
    trusted_at: str
    trusted_by: str = "local-user"

    def to_record(self) -> dict[str, object]:
        """Return a JSON-compatible record."""
        return {
            "workspace": self.workspace,
            "workspace_hash": self.workspace_hash,
            "trusted_at": self.trusted_at,
            "trusted_by": self.trusted_by,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "TrustedWorkspace":
        """Build a trust record from stored JSON."""
        return cls(
            workspace=_text(record.get("workspace")),
            workspace_hash=_text(record.get("workspace_hash")),
            trusted_at=_text(record.get("trusted_at")),
            trusted_by=_text(record.get("trusted_by")) or "local-user",
        )

    def is_ready(self) -> bool:
        """Return whether this trust record is usable."""
        return bool(self.workspace and self.workspace_hash and self.trusted_at)


class HookTrustStore:
    """JSON sidecar for local workspace hook trust."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def in_data_dir(cls, data_dir: str | Path) -> "HookTrustStore":
        """Return the hook trust store under Deepmate runtime data."""
        return cls(Path(data_dir) / "hooks" / HOOK_TRUST_FILE)

    def load(self) -> dict[str, TrustedWorkspace]:
        """Load trusted workspaces keyed by workspace hash."""
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("hook trust store must be a JSON object")
        raw_items = data.get("trusted_workspaces", [])
        if not isinstance(raw_items, list):
            raise ValueError("hook trust store requires a trusted_workspaces list")
        trusted: dict[str, TrustedWorkspace] = {}
        for raw_item in raw_items:
            if not isinstance(raw_item, Mapping):
                continue
            record = TrustedWorkspace.from_record(raw_item)
            if record.is_ready():
                trusted[record.workspace_hash] = record
        return trusted

    def is_trusted(self, workspace: str | Path) -> bool:
        """Return whether one workspace is trusted for project hooks."""
        record = self.load().get(workspace_hash(workspace))
        return bool(record and _resolve_workspace(workspace) == record.workspace)

    def trust_workspace(self, workspace: str | Path) -> TrustedWorkspace:
        """Mark one workspace trusted locally."""
        target = _resolve_workspace(workspace)
        record = TrustedWorkspace(
            workspace=target,
            workspace_hash=workspace_hash(target),
            trusted_at=utc_now_iso(),
        )
        with file_lock(self.path):
            trusted = self.load()
            trusted[record.workspace_hash] = record
            atomic_write_json(
                self.path,
                {
                    "version": 1,
                    "trusted_workspaces": [
                        item.to_record()
                        for item in sorted(
                            trusted.values(),
                            key=lambda item: item.workspace,
                        )
                    ],
                },
            )
        return record


def workspace_hash(workspace: str | Path) -> str:
    """Return a hash for a resolved workspace path."""
    resolved = _resolve_workspace(workspace)
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()


def _resolve_workspace(workspace: str | Path) -> str:
    return str(Path(workspace).expanduser().resolve())


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""
