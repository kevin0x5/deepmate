"""Data model for workspace scheduled jobs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

CRON_DIR = "cron"
CRON_JOBS_FILE = "jobs.jsonl"
CRON_OUTPUT_DIR = "cron/outputs"
APPROVAL_STATUS_APPROVED = "approved"
APPROVAL_STATUS_NEEDS_APPROVAL = "needs_approval"
JOB_STATUS_NEVER = "never"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class CronSchedule:
    """A small schedule shape that is easy to hand-edit."""

    kind: str
    time: str = ""
    weekday: str = ""
    timezone: str = ""
    interval_minutes: int = 0

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "CronSchedule":
        record = value or {}
        return cls(
            kind=str(record.get("kind", "daily")).strip() or "daily",
            time=str(record.get("time", "")).strip(),
            weekday=str(record.get("weekday", "")).strip().lower(),
            timezone=str(record.get("timezone", "")).strip(),
            interval_minutes=_int_value(record.get("interval_minutes"), 0),
        )

    def to_mapping(self) -> dict[str, Any]:
        record: dict[str, Any] = {"kind": self.kind}
        if self.time:
            record["time"] = self.time
        if self.weekday:
            record["weekday"] = self.weekday
        if self.timezone:
            record["timezone"] = self.timezone
        if self.interval_minutes:
            record["interval_minutes"] = self.interval_minutes
        return record


@dataclass(frozen=True, slots=True)
class CronJobSpec:
    """The prompt and execution bounds for a scheduled job."""

    prompt: str
    max_steps: int = 8

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "CronJobSpec":
        record = value or {}
        return cls(
            prompt=str(record.get("prompt", "")).strip(),
            max_steps=max(1, _int_value(record.get("max_steps"), 8)),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {"prompt": self.prompt, "max_steps": self.max_steps}


@dataclass(frozen=True, slots=True)
class CronOutput:
    """Where a scheduled job writes its visible result."""

    path: str = CRON_OUTPUT_DIR
    filename_template: str = "{job_id}-{date}-{time}.md"
    format: str = "markdown"

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "CronOutput":
        record = value or {}
        return cls(
            path=str(record.get("path", CRON_OUTPUT_DIR)).strip() or CRON_OUTPUT_DIR,
            filename_template=str(
                record.get("filename_template", "{job_id}-{date}-{time}.md")
            ).strip()
            or "{job_id}-{date}-{time}.md",
            format=str(record.get("format", "markdown")).strip() or "markdown",
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "filename_template": self.filename_template,
            "format": self.format,
        }


@dataclass(frozen=True, slots=True)
class CronPermissions:
    """Approved capability surface for a scheduled job."""

    read_workspace: bool = True
    write_output: bool = True
    workspace_write: bool = False
    shell: bool = False
    network: bool = False
    browser: bool = False
    computer_use: bool = False
    mcp_write: bool = False
    subagents: str = "auto"

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "CronPermissions":
        record = value or {}
        return cls(
            read_workspace=_bool_value(record.get("read_workspace"), True),
            write_output=_bool_value(record.get("write_output"), True),
            workspace_write=_bool_value(record.get("workspace_write"), False),
            shell=_bool_value(record.get("shell"), False),
            network=_bool_value(record.get("network"), False),
            browser=_bool_value(record.get("browser"), False),
            computer_use=_bool_value(record.get("computer_use"), False),
            mcp_write=_bool_value(record.get("mcp_write"), False),
            subagents=str(record.get("subagents", "auto")).strip() or "auto",
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "read_workspace": self.read_workspace,
            "write_output": self.write_output,
            "workspace_write": self.workspace_write,
            "shell": self.shell,
            "network": self.network,
            "browser": self.browser,
            "computer_use": self.computer_use,
            "mcp_write": self.mcp_write,
            "subagents": self.subagents,
        }


@dataclass(frozen=True, slots=True)
class CronState:
    """Runner-managed state for one job."""

    last_run_at: str = ""
    last_status: str = JOB_STATUS_NEVER
    next_run_at: str = ""
    last_output: str = ""
    last_error: str = ""

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "CronState":
        record = value or {}
        return cls(
            last_run_at=str(record.get("last_run_at", "")).strip(),
            last_status=str(record.get("last_status", JOB_STATUS_NEVER)).strip()
            or JOB_STATUS_NEVER,
            next_run_at=str(record.get("next_run_at", "")).strip(),
            last_output=str(record.get("last_output", "")).strip(),
            last_error=str(record.get("last_error", "")).strip(),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "last_run_at": self.last_run_at,
            "last_status": self.last_status,
            "next_run_at": self.next_run_at,
            "last_output": self.last_output,
            "last_error": self.last_error,
        }


@dataclass(frozen=True, slots=True)
class CronApproval:
    """Approval status for a hand-editable scheduled job."""

    status: str = APPROVAL_STATUS_NEEDS_APPROVAL
    digest: str = ""
    approved_at: str = ""
    reason: str = ""

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "CronApproval":
        record = value or {}
        return cls(
            status=str(record.get("status", APPROVAL_STATUS_NEEDS_APPROVAL)).strip()
            or APPROVAL_STATUS_NEEDS_APPROVAL,
            digest=str(record.get("digest", "")).strip(),
            approved_at=str(record.get("approved_at", "")).strip(),
            reason=str(record.get("reason", "")).strip(),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "digest": self.digest,
            "approved_at": self.approved_at,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CronJob:
    """One workspace scheduled job."""

    id: str
    name: str
    schedule: CronSchedule
    job: CronJobSpec
    output: CronOutput = field(default_factory=CronOutput)
    permissions: CronPermissions = field(default_factory=CronPermissions)
    state: CronState = field(default_factory=CronState)
    approval: CronApproval = field(default_factory=CronApproval)
    enabled: bool = True
    version: int = 1
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "CronJob":
        return cls(
            version=_int_value(value.get("version"), 1),
            id=str(value.get("id", "")).strip(),
            enabled=_bool_value(value.get("enabled"), True),
            name=str(value.get("name", "")).strip(),
            schedule=CronSchedule.from_mapping(_mapping(value.get("schedule"))),
            job=CronJobSpec.from_mapping(_mapping(value.get("job"))),
            output=CronOutput.from_mapping(_mapping(value.get("output"))),
            permissions=CronPermissions.from_mapping(_mapping(value.get("permissions"))),
            state=CronState.from_mapping(_mapping(value.get("state"))),
            approval=CronApproval.from_mapping(_mapping(value.get("approval"))),
            created_at=str(value.get("created_at", "")).strip(),
            updated_at=str(value.get("updated_at", "")).strip(),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "id": self.id,
            "enabled": self.enabled,
            "name": self.name,
            "schedule": self.schedule.to_mapping(),
            "job": self.job.to_mapping(),
            "output": self.output.to_mapping(),
            "permissions": self.permissions.to_mapping(),
            "state": self.state.to_mapping(),
            "approval": self.approval.to_mapping(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def approval_digest(self) -> str:
        """Return the digest for fields that require approval."""
        payload = {
            "version": self.version,
            "id": self.id,
            "schedule": self.schedule.to_mapping(),
            "job": self.job.to_mapping(),
            "output": self.output.to_mapping(),
            "permissions": self.permissions.to_mapping(),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def is_approved(self) -> bool:
        return (
            self.approval.status == APPROVAL_STATUS_APPROVED
            and self.approval.digest == self.approval_digest()
        )


def job_path(workspace: str | Path) -> Path:
    return Path(workspace) / CRON_DIR / CRON_JOBS_FILE


def now_iso() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def _mapping(value: object) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _int_value(value: object, default: int) -> int:
    try:
        if isinstance(value, bool):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool_value(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    clean = str(value).strip().lower()
    if clean in {"1", "true", "yes", "on"}:
        return True
    if clean in {"0", "false", "no", "off"}:
        return False
    return default
