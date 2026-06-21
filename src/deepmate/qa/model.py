"""Data contracts for QA Audit."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


AUDIT_STATUS_DRAFT = "draft"
AUDIT_STATUS_READY = "ready"
AUDIT_STATUS_RUNNING = "running"
AUDIT_STATUS_COMPLETED = "completed"
AUDIT_STATUS_BLOCKED = "blocked"

CASE_STATUS_PENDING = "pending"
CASE_STATUS_PASSED = "passed"
CASE_STATUS_WARNING = "warning"
CASE_STATUS_FAILED = "failed"
CASE_STATUS_BLOCKED = "blocked"


def now_iso() -> str:
    """Return an ISO timestamp for audit files."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True, slots=True)
class ProjectProfile:
    """Project shape discovered before an audit is planned."""

    project_name: str
    project_kinds: tuple[str, ...]
    surfaces: tuple[str, ...]
    package_managers: tuple[str, ...] = ()
    test_commands: tuple[str, ...] = ()
    run_commands: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, object]:
        return {
            "project_name": self.project_name,
            "project_kinds": list(self.project_kinds),
            "surfaces": list(self.surfaces),
            "package_managers": list(self.package_managers),
            "test_commands": list(self.test_commands),
            "run_commands": list(self.run_commands),
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ProjectProfile":
        return cls(
            project_name=_text(value.get("project_name")) or "workspace",
            project_kinds=_tuple_text(value.get("project_kinds")),
            surfaces=_tuple_text(value.get("surfaces")),
            package_managers=_tuple_text(value.get("package_managers")),
            test_commands=_tuple_text(value.get("test_commands")),
            run_commands=_tuple_text(value.get("run_commands")),
            evidence=_tuple_text(value.get("evidence")),
        )


@dataclass(frozen=True, slots=True)
class AuditPlan:
    """User-confirmable QA audit plan."""

    audit_id: str
    goal: str
    scope: tuple[str, ...]
    surfaces: tuple[str, ...]
    risk_model: tuple[str, ...]
    permissions: tuple[str, ...]
    project: ProjectProfile
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    status: str = AUDIT_STATUS_DRAFT

    def to_mapping(self) -> dict[str, object]:
        return {
            "audit_id": self.audit_id,
            "goal": self.goal,
            "scope": list(self.scope),
            "surfaces": list(self.surfaces),
            "risk_model": list(self.risk_model),
            "permissions": list(self.permissions),
            "project": self.project.to_mapping(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "AuditPlan":
        project_value = value.get("project")
        project = (
            ProjectProfile.from_mapping(project_value)
            if isinstance(project_value, Mapping)
            else ProjectProfile("workspace", (), ())
        )
        return cls(
            audit_id=_text(value.get("audit_id")),
            goal=_text(value.get("goal")),
            scope=_tuple_text(value.get("scope")),
            surfaces=_tuple_text(value.get("surfaces")),
            risk_model=_tuple_text(value.get("risk_model")),
            permissions=_tuple_text(value.get("permissions")),
            project=project,
            created_at=_text(value.get("created_at")),
            updated_at=_text(value.get("updated_at")),
            status=_text(value.get("status")) or AUDIT_STATUS_DRAFT,
        )


@dataclass(frozen=True, slots=True)
class AuditCase:
    """One project-adaptive QA test case."""

    case_id: str
    title: str
    surface: str
    risk_area: str
    priority: str
    persona: str
    scenario_brief: str
    preconditions: tuple[str, ...] = ()
    steps: tuple[str, ...] = ()
    expected: tuple[str, ...] = ()
    runner: str = "manual"
    tools: tuple[str, ...] = ()
    oracle: str = "evidence_review"
    evidence_required: tuple[str, ...] = ()
    cleanup: tuple[str, ...] = ()
    blocked_if: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "title": self.title,
            "surface": self.surface,
            "risk_area": self.risk_area,
            "priority": self.priority,
            "persona": self.persona,
            "scenario_brief": self.scenario_brief,
            "preconditions": list(self.preconditions),
            "steps": list(self.steps),
            "expected": list(self.expected),
            "runner": self.runner,
            "tools": list(self.tools),
            "oracle": self.oracle,
            "evidence_required": list(self.evidence_required),
            "cleanup": list(self.cleanup),
            "blocked_if": list(self.blocked_if),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "AuditCase":
        return cls(
            case_id=_text(value.get("case_id")),
            title=_text(value.get("title")),
            surface=_text(value.get("surface")),
            risk_area=_text(value.get("risk_area")),
            priority=_text(value.get("priority")) or "medium",
            persona=_text(value.get("persona")) or "target user",
            scenario_brief=_text(value.get("scenario_brief")),
            preconditions=_tuple_text(value.get("preconditions")),
            steps=_tuple_text(value.get("steps")),
            expected=_tuple_text(value.get("expected")),
            runner=_text(value.get("runner")) or "manual",
            tools=_tuple_text(value.get("tools")),
            oracle=_text(value.get("oracle")) or "evidence_review",
            evidence_required=_tuple_text(value.get("evidence_required")),
            cleanup=_tuple_text(value.get("cleanup")),
            blocked_if=_tuple_text(value.get("blocked_if")),
        )


@dataclass(frozen=True, slots=True)
class AuditCaseResult:
    """Result for one executed audit case."""

    case_id: str
    status: str
    summary: str
    evidence: tuple[str, ...] = ()
    details: str = ""

    def to_mapping(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "status": self.status,
            "summary": self.summary,
            "evidence": list(self.evidence),
            "details": self.details,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "AuditCaseResult":
        return cls(
            case_id=_text(value.get("case_id")),
            status=_text(value.get("status")) or CASE_STATUS_PENDING,
            summary=_text(value.get("summary")),
            evidence=_tuple_text(value.get("evidence")),
            details=_text(value.get("details")),
        )


@dataclass(frozen=True, slots=True)
class AuditRunResult:
    """Overall QA Audit execution result."""

    audit_id: str
    status: str
    report_markdown: str
    report_html: str
    results: tuple[AuditCaseResult, ...]
    summary: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "audit_id": self.audit_id,
            "status": self.status,
            "report_markdown": self.report_markdown,
            "report_html": self.report_html,
            "results": [item.to_mapping() for item in self.results],
            "summary": self.summary,
        }


def audit_root(workspace: str | Path) -> Path:
    return Path(workspace).resolve() / "qa" / "audits"


def json_line(value: Mapping[str, object]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _tuple_text(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if not isinstance(value, Sequence):
        return ()
    result: list[str] = []
    for item in value:
        text = _text(item)
        if text:
            result.append(text)
    return tuple(result)
