"""Installed skill manifest sidecar.

This file records Deepmate install metadata without modifying community
standard SKILL.md files.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from deepmate.foundation import normalize_name
from deepmate.storage import atomic_write_json, file_lock

INSTALLED_SKILLS_FILE = "installed_skills.json"


@dataclass(frozen=True, slots=True)
class InstalledSkillRecord:
    """One installed skill manifest entry."""

    name: str
    source_kind: str
    source_ref: str
    target_path: str
    installed_at: str
    target_scope: str = "workspace"
    updated_at: str = ""
    content_sha256: str = ""
    compatibility: str = ""
    setup_status: str = ""
    setup_command: str = ""
    setup_updated_at: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def is_ready(self) -> bool:
        """Return whether this record has enough identity to persist."""
        return bool(
            self.name.strip()
            and self.source_kind.strip()
            and self.source_ref.strip()
            and self.target_path.strip()
            and self.installed_at.strip()
        )

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable record."""
        return {
            "name": self.name,
            "source_kind": self.source_kind,
            "source_ref": self.source_ref,
            "target_path": self.target_path,
            "target_scope": self.target_scope,
            "installed_at": self.installed_at,
            "updated_at": self.updated_at,
            "content_sha256": self.content_sha256,
            "compatibility": self.compatibility,
            "setup_status": self.setup_status,
            "setup_command": self.setup_command,
            "setup_updated_at": self.setup_updated_at,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "InstalledSkillRecord":
        """Build one manifest record from stored JSON."""
        return cls(
            name=_text(record.get("name")),
            source_kind=_text(record.get("source_kind")),
            source_ref=_text(record.get("source_ref")),
            target_path=_text(record.get("target_path")),
            installed_at=_text(record.get("installed_at")),
            target_scope=_text(record.get("target_scope")) or "workspace",
            updated_at=_text(record.get("updated_at")),
            content_sha256=_text(record.get("content_sha256")),
            compatibility=_text(record.get("compatibility")),
            setup_status=_text(record.get("setup_status")),
            setup_command=_text(record.get("setup_command")),
            setup_updated_at=_text(record.get("setup_updated_at")),
            warnings=_string_tuple(record.get("warnings")),
        )


class InstalledSkillManifestStore:
    """Atomic JSON store for installed skill metadata."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def in_data_dir(cls, data_dir: str | Path) -> "InstalledSkillManifestStore":
        """Return the default manifest path under Deepmate runtime data."""
        return cls(Path(data_dir) / "skills" / INSTALLED_SKILLS_FILE)

    def load(self) -> dict[str, InstalledSkillRecord]:
        """Load records keyed by normalized skill name."""
        return self._load_unlocked()

    def _load_unlocked(self) -> dict[str, InstalledSkillRecord]:
        """Load records without acquiring a mutation lock."""
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("installed skill manifest must be a JSON object")
        raw_items = data.get("skills", [])
        if not isinstance(raw_items, list):
            raise ValueError("installed skill manifest requires a skills list")
        records: dict[str, InstalledSkillRecord] = {}
        for raw_item in raw_items:
            if not isinstance(raw_item, Mapping):
                continue
            record = InstalledSkillRecord.from_record(raw_item)
            if record.is_ready():
                records[_normalize_name(record.name)] = record
        return records

    def save(self, records: Mapping[str, InstalledSkillRecord]) -> None:
        """Atomically save all ready records."""
        with file_lock(self.path):
            self._save_unlocked(records)

    def _save_unlocked(self, records: Mapping[str, InstalledSkillRecord]) -> None:
        """Save all ready records while the caller holds the mutation lock."""
        ready_records = sorted(
            (record for record in records.values() if record.is_ready()),
            key=lambda record: _normalize_name(record.name),
        )
        payload = {
            "version": 1,
            "skills": [record.to_record() for record in ready_records],
        }
        atomic_write_json(self.path, payload)

    def upsert(self, record: InstalledSkillRecord) -> None:
        """Insert or replace one ready record."""
        if not record.is_ready():
            raise ValueError("installed skill record is not ready")
        with file_lock(self.path):
            records = self._load_unlocked()
            records[_normalize_name(record.name)] = record
            self._save_unlocked(records)

    def remove(self, name: str) -> InstalledSkillRecord | None:
        """Remove one record and return it if present."""
        with file_lock(self.path):
            records = self._load_unlocked()
            record = records.pop(_normalize_name(name), None)
            if record is not None:
                self._save_unlocked(records)
        return record

    def get(self, name: str) -> InstalledSkillRecord | None:
        """Return one installed skill record by normalized name."""
        return self.load().get(_normalize_name(name))

    def update_setup_status(
        self,
        name: str,
        *,
        status: str,
        command: str = "",
        updated_at: str = "",
    ) -> InstalledSkillRecord:
        """Update setup status for one installed skill."""
        with file_lock(self.path):
            records = self._load_unlocked()
            key = _normalize_name(name)
            record = records.get(key)
            if record is None:
                raise ValueError(f"installed skill not found: {name}")
            updated = InstalledSkillRecord(
                name=record.name,
                source_kind=record.source_kind,
                source_ref=record.source_ref,
                target_path=record.target_path,
                target_scope=record.target_scope,
                installed_at=record.installed_at,
                updated_at=record.updated_at,
                content_sha256=record.content_sha256,
                compatibility=record.compatibility,
                setup_status=status.strip(),
                setup_command=command.strip(),
                setup_updated_at=updated_at.strip(),
                warnings=record.warnings,
            )
            records[key] = updated
            self._save_unlocked(records)
            return updated


def _normalize_name(name: str) -> str:
    return normalize_name(name)


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
