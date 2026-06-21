"""Append-only evolution change log and rollback support."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping

from deepmate.domain import ProfileRef
from deepmate.foundation import utc_isoformat
from deepmate.storage import JsonlWriter, atomic_write_text

APPLIED_LOG_FILE = "applied_log.jsonl"


@dataclass(frozen=True, slots=True)
class EvolutionChange:
    """One applied self-evolution change."""

    change_id: str
    change_type: str
    target_path: str
    summary: str
    evidence_refs: tuple[str, ...] = ()
    decision: str = "auto_apply"
    status: str = "applied"
    old_hash: str = ""
    new_hash: str = ""
    validation_result: str = ""
    rollback_hint: str = ""
    created_at: str = ""
    applied_at: str = ""
    old_exists: bool = True
    old_content: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def is_ready(self) -> bool:
        """Return whether the record has enough identity to persist."""
        return bool(
            self.change_id.strip()
            and self.change_type.strip()
            and self.target_path.strip()
            and self.status.strip()
            and self.created_at.strip()
            and self.applied_at.strip()
        )

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable record."""
        return {
            "change_id": self.change_id,
            "change_type": self.change_type,
            "target_path": self.target_path,
            "summary": self.summary,
            "evidence_refs": list(self.evidence_refs),
            "decision": self.decision,
            "status": self.status,
            "old_hash": self.old_hash,
            "new_hash": self.new_hash,
            "validation_result": self.validation_result,
            "rollback_hint": self.rollback_hint,
            "created_at": self.created_at,
            "applied_at": self.applied_at,
            "old_exists": self.old_exists,
            "old_content": self.old_content,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "EvolutionChange":
        """Build one change record from JSON data."""
        return cls(
            change_id=_text(record.get("change_id")),
            change_type=_text(record.get("change_type")),
            target_path=_text(record.get("target_path")),
            summary=_text(record.get("summary")),
            evidence_refs=_string_tuple(record.get("evidence_refs")),
            decision=_text(record.get("decision")) or "auto_apply",
            status=_text(record.get("status")) or "applied",
            old_hash=_text(record.get("old_hash")),
            new_hash=_text(record.get("new_hash")),
            validation_result=_text(record.get("validation_result")),
            rollback_hint=_text(record.get("rollback_hint")),
            created_at=_text(record.get("created_at")),
            applied_at=_text(record.get("applied_at")),
            old_exists=bool(record.get("old_exists", True)),
            old_content=_raw_text(record.get("old_content")),
            metadata=_mapping(record.get("metadata")),
        )


@dataclass(frozen=True, slots=True)
class EvolutionRollbackResult:
    """Result of rolling back one evolution change."""

    change_id: str
    target_path: Path
    restored_hash: str
    rollback_change: EvolutionChange


class EvolutionChangeStore:
    """Append-only JSONL store for applied evolution changes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def in_data_dir(
        cls,
        data_dir: str | Path,
        profile: ProfileRef | str,
    ) -> "EvolutionChangeStore":
        """Return the profile-local evolution applied log."""
        profile_name = profile.name if isinstance(profile, ProfileRef) else str(profile)
        clean_profile = profile_name.strip() or "default"
        return cls(Path(data_dir) / "evolution" / clean_profile / APPLIED_LOG_FILE)

    def append(self, change: EvolutionChange) -> None:
        """Append one ready change record."""
        if not change.is_ready():
            raise ValueError("evolution change is incomplete")
        JsonlWriter(self.path).append(change.to_record())

    def load(self) -> tuple[EvolutionChange, ...]:
        """Load all parseable change records."""
        if not self.path.exists():
            return ()
        changes: list[EvolutionChange] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(raw_record, Mapping):
                continue
            change = EvolutionChange.from_record(raw_record)
            if change.is_ready():
                changes.append(change)
        return tuple(changes)

    def require(self, change_id: str) -> EvolutionChange:
        """Return one change by id or raise ValueError."""
        clean_id = change_id.strip()
        for change in reversed(self.load()):
            if change.change_id == clean_id:
                return change
        raise ValueError(f"unknown evolution change: {clean_id}")

    def rollback(self, change_id: str, workspace: str | Path) -> EvolutionRollbackResult:
        """Restore the old content for one applied change and append a rollback record."""
        workspace_path = Path(workspace)
        change = self.require(change_id)
        if change.status != "applied":
            raise ValueError(f"evolution change is not applied: {change.change_id}")
        data_dir = self.path.parent.parent.parent
        target_path = _resolve_target_path(workspace_path, change.target_path, data_dir)
        current_content = (
            target_path.read_text(encoding="utf-8") if target_path.exists() else ""
        )
        current_hash = content_sha256(current_content) if target_path.exists() else ""
        if current_hash != change.new_hash:
            raise ValueError(
                "cannot rollback evolution change because target content changed: "
                f"{change.target_path}"
            )
        sidecar_restores = _sidecar_restores(self.path, change)
        for sidecar in sidecar_restores:
            sidecar.verify_current()
        if change.old_exists:
            atomic_write_text(target_path, change.old_content)
            restored_hash = content_sha256(change.old_content)
        else:
            if target_path.exists():
                target_path.unlink()
            restored_hash = ""
        for sidecar in sidecar_restores:
            sidecar.restore()

        rollback_change = rollback_record(change, current_content, restored_hash)
        self.append(rollback_change)
        return EvolutionRollbackResult(
            change_id=change.change_id,
            target_path=target_path,
            restored_hash=restored_hash,
            rollback_change=rollback_change,
        )


def applied_change(
    *,
    change_type: str,
    target_path: str,
    summary: str,
    old_content: str,
    new_content: str,
    old_exists: bool,
    evidence_refs: tuple[str, ...] = (),
    validation_result: str = "passed",
    decision: str = "auto_apply",
    now_iso: str,
    metadata: Mapping[str, object] | None = None,
) -> EvolutionChange:
    """Build one applied change record."""
    return EvolutionChange(
        change_id=f"chg_{uuid.uuid4().hex[:12]}",
        change_type=change_type,
        target_path=target_path,
        summary=summary.strip(),
        evidence_refs=tuple(ref.strip() for ref in evidence_refs if ref.strip()),
        decision=decision.strip() or "auto_apply",
        status="applied",
        old_hash=content_sha256(old_content) if old_exists else "",
        new_hash=content_sha256(new_content),
        validation_result=validation_result.strip() or "passed",
        rollback_hint="Restore old_content if the target hash still matches new_hash.",
        created_at=now_iso,
        applied_at=now_iso,
        old_exists=old_exists,
        old_content=old_content if old_exists else "",
        metadata=metadata or {},
    )


def sidecar_restore_metadata(
    *,
    data_dir: str | Path,
    sidecar_path: str | Path,
    old_content: str,
    new_content: str,
    old_exists: bool,
) -> Mapping[str, object]:
    """Return metadata that lets rollback restore one data-dir sidecar file."""
    root = Path(data_dir)
    path = Path(sidecar_path)
    try:
        relative_path = str(path.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise ValueError(f"sidecar path is outside data_dir: {path}") from exc
    return {
        "sidecar_restores": [
            {
                "target_path": relative_path,
                "old_hash": content_sha256(old_content) if old_exists else "",
                "new_hash": content_sha256(new_content),
                "old_exists": old_exists,
                "old_content": old_content if old_exists else "",
            }
        ]
    }


def rollback_record(
    change: EvolutionChange,
    current_content: str,
    restored_hash: str,
) -> EvolutionChange:
    """Build the append-only record for one rollback operation."""
    now_iso = utc_isoformat(datetime.now(UTC))
    return replace(
        change,
        change_id=f"chg_{uuid.uuid4().hex[:12]}",
        change_type="rollback",
        summary=f"Rolled back evolution change {change.change_id}.",
        evidence_refs=(f"rollback_of={change.change_id}",),
        decision="manual_rollback",
        old_hash=content_sha256(current_content),
        new_hash=restored_hash,
        validation_result="passed",
        rollback_hint="Rollback record; no automatic inverse rollback.",
        created_at=now_iso,
        applied_at=now_iso,
        old_exists=True,
        old_content=current_content,
    )


def content_sha256(content: str) -> str:
    """Return the sha256 of UTF-8 text content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _resolve_target_path(
    workspace: Path,
    target_path: str,
    data_dir: Path | None = None,
) -> Path:
    raw_path = Path(target_path)
    resolved = raw_path if raw_path.is_absolute() else workspace / raw_path
    allowed_roots = (workspace,) if data_dir is None else (workspace, data_dir)
    for root in allowed_roots:
        try:
            resolved.resolve().relative_to(root.resolve())
            return resolved
        except ValueError:
            continue
    root_label = "workspace"
    if data_dir is not None:
        root_label = "workspace or evolution data_dir"
    raise ValueError(f"evolution target is outside {root_label}: {target_path}")


@dataclass(frozen=True, slots=True)
class _SidecarRestore:
    path: Path
    old_exists: bool
    old_content: str
    new_hash: str

    def verify_current(self) -> None:
        current_content = self.path.read_text(encoding="utf-8") if self.path.exists() else ""
        current_hash = content_sha256(current_content) if self.path.exists() else ""
        if current_hash != self.new_hash:
            raise ValueError(
                "cannot rollback evolution change because sidecar content changed: "
                f"{self.path}"
            )

    def restore(self) -> None:
        if self.old_exists:
            atomic_write_text(self.path, self.old_content)
            return
        if self.path.exists():
            self.path.unlink()


def _sidecar_restores(
    applied_log_path: Path,
    change: EvolutionChange,
) -> tuple[_SidecarRestore, ...]:
    raw_restores = change.metadata.get("sidecar_restores")
    if not isinstance(raw_restores, list):
        return ()
    data_dir = applied_log_path.parent.parent.parent
    restores: list[_SidecarRestore] = []
    for raw_restore in raw_restores:
        if not isinstance(raw_restore, Mapping):
            continue
        target_path = _text(raw_restore.get("target_path"))
        if not target_path:
            continue
        raw_path = data_dir / target_path
        try:
            path = raw_path.resolve().relative_to(data_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"sidecar restore is outside data_dir: {target_path}") from exc
        restores.append(
            _SidecarRestore(
                path=data_dir / path,
                old_exists=bool(raw_restore.get("old_exists", True)),
                old_content=_raw_text(raw_restore.get("old_content")),
                new_hash=_text(raw_restore.get("new_hash")),
            )
        )
    return tuple(restores)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _raw_text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
