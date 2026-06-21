"""Persistent MCP usage state.

This sidecar keeps Deepmate-specific MCP activation data out of MCP server
configuration and tool schemas.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

from deepmate.domain import ProfileRef
from deepmate.foundation import non_negative_int, normal_datetime, utc_isoformat
from deepmate.mcp.spec import McpServerSpec, McpToolRef
from deepmate.storage import atomic_write_json, file_lock

MCP_STATE_FILE = "mcp_state.json"
DEFAULT_MCP_IDLE_DAYS = 7


@dataclass(frozen=True, slots=True)
class McpUsageEntry:
    """One persisted MCP server or tool usage entry."""

    key: str
    kind: str
    name: str
    server_name: str = ""
    created_at: str = ""
    updated_at: str = ""
    last_seen_at: str = ""
    last_used_at: str = ""
    invocation_count: int = 0
    load_count: int = 0

    def is_ready(self) -> bool:
        """Return whether the entry is persistable."""
        return bool(
            self.key.strip()
            and self.kind in {"server", "tool"}
            and self.name.strip()
            and self.created_at.strip()
            and self.updated_at.strip()
        )

    def is_idle(
        self,
        now: datetime | None = None,
        idle_days: int = DEFAULT_MCP_IDLE_DAYS,
    ) -> bool:
        """Return whether this entry has been unused for the idle window."""
        activity_at = _parse_datetime(self.last_used_at) or _parse_datetime(
            self.created_at
        )
        if activity_at is None:
            return False
        return _normal_datetime(now) - activity_at >= timedelta(days=max(1, idle_days))

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable record."""
        return {
            "key": self.key,
            "kind": self.kind,
            "name": self.name,
            "server_name": self.server_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_seen_at": self.last_seen_at,
            "last_used_at": self.last_used_at,
            "invocation_count": self.invocation_count,
            "load_count": self.load_count,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "McpUsageEntry":
        """Build one entry from a stored JSON record."""
        return cls(
            key=_text(record.get("key")),
            kind=_text(record.get("kind")),
            name=_text(record.get("name")),
            server_name=_text(record.get("server_name")),
            created_at=_text(record.get("created_at")),
            updated_at=_text(record.get("updated_at")),
            last_seen_at=_text(record.get("last_seen_at")),
            last_used_at=_text(record.get("last_used_at")),
            invocation_count=_non_negative_int(record.get("invocation_count")),
            load_count=_non_negative_int(record.get("load_count")),
        )


class McpUsageStateStore:
    """JSON sidecar store for profile-scoped MCP usage state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def in_data_dir(
        cls,
        data_dir: str | Path,
        profile: ProfileRef | str,
    ) -> "McpUsageStateStore":
        """Return the profile-local MCP usage state store."""
        profile_name = profile.name if isinstance(profile, ProfileRef) else str(profile)
        clean_profile = profile_name.strip() or "default"
        return cls(Path(data_dir) / "mcp" / clean_profile / MCP_STATE_FILE)

    def load(self) -> dict[str, McpUsageEntry]:
        """Load all persisted MCP usage entries keyed by entry key."""
        with file_lock(self.path):
            return self._load_unlocked()

    def _load_unlocked(self) -> dict[str, McpUsageEntry]:
        """Load all persisted MCP usage entries without acquiring a mutation lock."""
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("MCP state must be a JSON object")
        raw_items = data.get("entries", [])
        if not isinstance(raw_items, list):
            raise ValueError("MCP state requires an entries list")
        entries: dict[str, McpUsageEntry] = {}
        for raw_item in raw_items:
            if not isinstance(raw_item, Mapping):
                continue
            entry = McpUsageEntry.from_record(raw_item)
            if entry.is_ready():
                entries[entry.key] = entry
        return entries

    def save(self, entries: Mapping[str, McpUsageEntry]) -> None:
        """Atomically persist all state entries."""
        with file_lock(self.path):
            self._save_unlocked(entries)

    def _save_unlocked(self, entries: Mapping[str, McpUsageEntry]) -> None:
        """Persist all state entries while the caller holds the mutation lock."""
        ready_entries = sorted(
            (entry for entry in entries.values() if entry.is_ready()),
            key=lambda entry: entry.key,
        )
        payload = {
            "version": 1,
            "idle_days": DEFAULT_MCP_IDLE_DAYS,
            "entries": [entry.to_record() for entry in ready_entries],
        }
        atomic_write_json(self.path, payload)

    def sync_server_seen(
        self,
        server: McpServerSpec,
        now: datetime | None = None,
    ) -> McpUsageEntry:
        """Mark one configured MCP server as seen without changing usage."""
        current_time = _normal_datetime(now)
        with file_lock(self.path):
            entries = self._load_unlocked()
            key = mcp_server_key(server.name)
            entry = entries.get(key) or _new_server_entry(server.name, current_time)
            entry = replace(
                entry,
                name=server.name.strip(),
                last_seen_at=_isoformat(current_time),
                updated_at=_isoformat(current_time),
            )
            entries[key] = entry
            self._save_unlocked(entries)
        return entry

    def sync_tool_seen(
        self,
        tool: McpToolRef,
        now: datetime | None = None,
    ) -> McpUsageEntry:
        """Mark one discovered MCP tool as seen without changing usage."""
        current_time = _normal_datetime(now)
        with file_lock(self.path):
            entries = self._load_unlocked()
            key = mcp_tool_key(tool.qualified_name())
            entry = entries.get(key) or _new_tool_entry(tool, current_time)
            entry = replace(
                entry,
                name=tool.qualified_name(),
                server_name=tool.server_name.strip(),
                last_seen_at=_isoformat(current_time),
                updated_at=_isoformat(current_time),
            )
            entries[key] = entry
            self._save_unlocked(entries)
        return entry

    def sync_inventory_seen(
        self,
        server: McpServerSpec,
        tools: tuple[McpToolRef, ...],
        now: datetime | None = None,
    ) -> None:
        """Mark one server inventory as seen with a single state write."""
        current_time = _normal_datetime(now)
        timestamp = _isoformat(current_time)
        with file_lock(self.path):
            entries = self._load_unlocked()
            server_key = mcp_server_key(server.name)
            server_entry = entries.get(server_key) or _new_server_entry(
                server.name,
                current_time,
            )
            entries[server_key] = replace(
                server_entry,
                name=server.name.strip(),
                last_seen_at=timestamp,
                updated_at=timestamp,
            )
            for tool in tools:
                tool_key = mcp_tool_key(tool.qualified_name())
                tool_entry = entries.get(tool_key) or _new_tool_entry(tool, current_time)
                entries[tool_key] = replace(
                    tool_entry,
                    name=tool.qualified_name(),
                    server_name=tool.server_name.strip(),
                    last_seen_at=timestamp,
                    updated_at=timestamp,
                )
            self._save_unlocked(entries)

    def record_tool_loaded(
        self,
        tool: McpToolRef,
        now: datetime | None = None,
    ) -> McpUsageEntry:
        """Record that a tool schema was loaded for model use."""
        return self._record_tool_activity(tool, now=now, load=True, invoke=False)

    def record_tool_invoked(
        self,
        tool: McpToolRef,
        now: datetime | None = None,
    ) -> McpUsageEntry:
        """Record that a tool was actually invoked."""
        return self._record_tool_activity(tool, now=now, load=False, invoke=True)

    def server_entry(self, server_name: str) -> McpUsageEntry | None:
        """Return one server entry by server name."""
        return self.load().get(mcp_server_key(server_name))

    def tool_entry(self, qualified_name: str) -> McpUsageEntry | None:
        """Return one tool entry by qualified tool name."""
        return self.load().get(mcp_tool_key(qualified_name))

    def _record_tool_activity(
        self,
        tool: McpToolRef,
        now: datetime | None,
        load: bool,
        invoke: bool,
    ) -> McpUsageEntry:
        current_time = _normal_datetime(now)
        timestamp = _isoformat(current_time)
        with file_lock(self.path):
            entries = self._load_unlocked()
            server_key = mcp_server_key(tool.server_name)
            server_entry = entries.get(server_key) or _new_server_entry(
                tool.server_name,
                current_time,
            )
            server_entry = replace(
                server_entry,
                last_seen_at=timestamp,
                last_used_at=timestamp,
                updated_at=timestamp,
                invocation_count=(
                    server_entry.invocation_count + 1
                    if invoke
                    else server_entry.invocation_count
                ),
            )
            entries[server_key] = server_entry

            tool_key = mcp_tool_key(tool.qualified_name())
            tool_entry = entries.get(tool_key) or _new_tool_entry(tool, current_time)
            tool_entry = replace(
                tool_entry,
                last_seen_at=timestamp,
                last_used_at=timestamp if invoke else tool_entry.last_used_at,
                updated_at=timestamp,
                load_count=tool_entry.load_count + (1 if load else 0),
                invocation_count=tool_entry.invocation_count + (1 if invoke else 0),
            )
            entries[tool_key] = tool_entry
            self._save_unlocked(entries)
        return tool_entry


def mcp_server_key(server_name: str) -> str:
    """Return the stable MCP state key for one server."""
    clean_name = server_name.strip()
    if not clean_name:
        raise ValueError("MCP server name cannot be empty")
    return f"mcp_server:{clean_name}"


def mcp_tool_key(qualified_name: str) -> str:
    """Return the stable MCP state key for one qualified tool."""
    clean_name = qualified_name.strip()
    if not clean_name:
        raise ValueError("MCP tool name cannot be empty")
    return f"mcp_tool:{clean_name}"


def _new_server_entry(server_name: str, current_time: datetime) -> McpUsageEntry:
    timestamp = _isoformat(current_time)
    return McpUsageEntry(
        key=mcp_server_key(server_name),
        kind="server",
        name=server_name.strip(),
        created_at=timestamp,
        updated_at=timestamp,
        last_seen_at=timestamp,
    )


def _new_tool_entry(tool: McpToolRef, current_time: datetime) -> McpUsageEntry:
    timestamp = _isoformat(current_time)
    return McpUsageEntry(
        key=mcp_tool_key(tool.qualified_name()),
        kind="tool",
        name=tool.qualified_name(),
        server_name=tool.server_name.strip(),
        created_at=timestamp,
        updated_at=timestamp,
        last_seen_at=timestamp,
    )


def _normal_datetime(value: datetime | None = None) -> datetime:
    return normal_datetime(value)


def _parse_datetime(value: str) -> datetime | None:
    if not value.strip():
        return None
    try:
        return _normal_datetime(datetime.fromisoformat(value.strip().replace("Z", "+00:00")))
    except ValueError:
        return None


def _isoformat(value: datetime) -> str:
    return utc_isoformat(value)


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _non_negative_int(value: object) -> int:
    return non_negative_int(value)
