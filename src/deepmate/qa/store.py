"""Workspace storage for QA Audit artifacts."""

from __future__ import annotations

import json
import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from deepmate.qa.model import (
    AuditCase,
    AuditCaseResult,
    AuditPlan,
    audit_root,
    json_line,
    now_iso,
)
from deepmate.storage.atomic import file_lock


PLAN_FILE = "audit.plan.md"
CASES_FILE = "audit.cases.jsonl"
STATE_FILE = "audit.state.json"
RESULTS_FILE = "audit.results.jsonl"
REPORT_MD = "report.md"
REPORT_HTML = "report.html"


@dataclass(frozen=True, slots=True)
class QaCaseLoadIssue:
    """One user-editable QA case line that could not be loaded."""

    line_number: int
    message: str


@dataclass(frozen=True, slots=True)
class QaCaseLoadReport:
    """Loaded audit cases plus recoverable JSONL diagnostics."""

    cases: tuple[AuditCase, ...]
    issues: tuple[QaCaseLoadIssue, ...]


@dataclass(frozen=True, slots=True)
class QaAuditPaths:
    """Common paths for one audit."""

    root: Path
    plan: Path
    cases: Path
    state: Path
    results: Path
    evidence: Path
    report_md: Path
    report_html: Path


class QaAuditStore:
    """Read and write workspace-local QA Audit artifacts."""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).resolve()
        self.root = audit_root(self.workspace)

    def paths(self, audit_id: str) -> QaAuditPaths:
        clean = _safe_id(audit_id)
        root = self.root / clean
        return QaAuditPaths(
            root=root,
            plan=root / PLAN_FILE,
            cases=root / CASES_FILE,
            state=root / STATE_FILE,
            results=root / RESULTS_FILE,
            evidence=root / "evidence",
            report_md=root / REPORT_MD,
            report_html=root / REPORT_HTML,
        )

    def latest_audit_id(self) -> str:
        if not self.root.exists():
            raise ValueError("No QA audits found. Start one with /qa <测试目标>.")
        candidates = [path for path in self.root.iterdir() if path.is_dir()]
        if not candidates:
            raise ValueError("No QA audits found. Start one with /qa <测试目标>.")
        return max(candidates, key=_audit_sort_key).name

    def list_audits(self) -> tuple[str, ...]:
        if not self.root.exists():
            return ()
        return tuple(sorted(path.name for path in self.root.iterdir() if path.is_dir()))

    def write_audit(
        self,
        plan: AuditPlan,
        cases: Iterable[AuditCase],
        *,
        plan_markdown: str,
    ) -> QaAuditPaths:
        paths = self.paths(plan.audit_id)
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.evidence.mkdir(parents=True, exist_ok=True)
        (paths.evidence / "commands").mkdir(parents=True, exist_ok=True)
        (paths.evidence / "screenshots").mkdir(parents=True, exist_ok=True)
        (paths.evidence / "traces").mkdir(parents=True, exist_ok=True)
        (paths.evidence / "logs").mkdir(parents=True, exist_ok=True)
        (paths.evidence / "artifacts").mkdir(parents=True, exist_ok=True)
        _write_text_atomic(paths.plan, plan_markdown)
        self.write_cases(plan.audit_id, tuple(cases))
        self.write_state(plan.audit_id, plan.to_mapping())
        if not paths.results.exists():
            _write_text_atomic(paths.results, "")
        return paths

    def read_plan_state(self, audit_id: str) -> AuditPlan:
        payload = self.read_state_mapping(audit_id)
        return AuditPlan.from_mapping(payload)

    def read_state_mapping(self, audit_id: str) -> dict[str, object]:
        paths = self.paths(audit_id)
        if not paths.state.is_file():
            raise ValueError(f"QA audit state not found: {audit_id}")
        try:
            payload = json.loads(paths.state.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid QA audit state JSON for {audit_id}: line {exc.lineno} column {exc.colno}"
            ) from exc
        except OSError as exc:
            raise ValueError(f"could not read QA audit state for {audit_id}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"invalid QA audit state: {audit_id}")
        return dict(payload)

    def write_state(self, audit_id: str, state: Mapping[str, object]) -> None:
        paths = self.paths(audit_id)
        paths.root.mkdir(parents=True, exist_ok=True)
        _write_text_atomic(paths.state, json.dumps(state, ensure_ascii=False, indent=2) + "\n")

    def update_state(self, audit_id: str, **values: object) -> None:
        paths = self.paths(audit_id)
        with file_lock(paths.state):
            if not paths.state.is_file():
                raise ValueError(f"QA audit state not found: {audit_id}")
            try:
                payload = json.loads(paths.state.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid QA audit state JSON for {audit_id}: line {exc.lineno} column {exc.colno}"
                ) from exc
            except OSError as exc:
                raise ValueError(f"could not read QA audit state for {audit_id}: {exc}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"invalid QA audit state: {audit_id}")
            next_payload = dict(payload)
            next_payload.update(values)
            next_payload["updated_at"] = now_iso()
            _write_text_unlocked(
                paths.state,
                json.dumps(next_payload, ensure_ascii=False, indent=2) + "\n",
            )

    def write_cases(self, audit_id: str, cases: Iterable[AuditCase]) -> None:
        paths = self.paths(audit_id)
        content = "\n".join(json_line(case.to_mapping()) for case in cases)
        if content:
            content += "\n"
        _write_text_atomic(paths.cases, content)

    def read_cases(self, audit_id: str) -> tuple[AuditCase, ...]:
        return self.read_cases_report(audit_id).cases

    def read_cases_report(self, audit_id: str) -> QaCaseLoadReport:
        paths = self.paths(audit_id)
        if not paths.cases.exists():
            return QaCaseLoadReport(cases=(), issues=())
        cases: list[AuditCase] = []
        issues: list[QaCaseLoadIssue] = []
        for line_number, raw in enumerate(
            paths.cases.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append(QaCaseLoadIssue(line_number, f"invalid JSON: {exc.msg}"))
                continue
            if not isinstance(payload, Mapping):
                issues.append(QaCaseLoadIssue(line_number, "expected a JSON object"))
                continue
            case = AuditCase.from_mapping(payload)
            if not case.case_id or not case.title:
                issues.append(QaCaseLoadIssue(line_number, "missing case_id or title"))
                continue
            cases.append(case)
        return QaCaseLoadReport(cases=tuple(cases), issues=tuple(issues))

    def write_results(
        self,
        audit_id: str,
        results: Iterable[AuditCaseResult],
    ) -> None:
        paths = self.paths(audit_id)
        content = "\n".join(json_line(result.to_mapping()) for result in results)
        if content:
            content += "\n"
        _write_text_atomic(paths.results, content)

    def read_results(self, audit_id: str) -> tuple[AuditCaseResult, ...]:
        paths = self.paths(audit_id)
        if not paths.results.exists():
            return ()
        results: list[AuditCaseResult] = []
        for raw in paths.results.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, Mapping):
                results.append(AuditCaseResult.from_mapping(payload))
        return tuple(results)

    def write_report_markdown(self, audit_id: str, markdown: str) -> None:
        paths = self.paths(audit_id)
        _write_text_atomic(paths.report_md, markdown)

    def relative(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.workspace))
        except ValueError:
            return str(path)


def next_audit_id(workspace: str | Path, goal: str) -> str:
    """Return a stable timestamped audit id."""
    stamp = now_iso().replace(":", "").replace("+00:00", "Z")
    date = stamp[:10]
    slug = _slug(goal) or "qa-audit"
    base = f"{date}-{slug}"
    root = audit_root(workspace)
    candidate = base
    index = 2
    while (root / candidate).exists():
        candidate = f"{base}-{index}"
        index += 1
    return candidate


def _safe_id(value: str) -> str:
    clean = value.strip().replace("\\", "/").split("/")[-1]
    if not clean or clean in {".", ".."}:
        raise ValueError("invalid QA audit id")
    return clean


def _slug(value: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", value.lower())
    return "-".join(words[:6])[:60].strip("-")


def _write_text_atomic(path: Path, content: str) -> None:
    with file_lock(path):
        _write_text_unlocked(path, content)


def _write_text_unlocked(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp")
        tmp.write_text(content, encoding="utf-8")
        try:
            os.replace(tmp, path)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass


def _audit_sort_key(path: Path) -> tuple[str, float, str]:
    state = path / STATE_FILE
    timestamp = ""
    if state.is_file():
        try:
            payload = json.loads(state.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, Mapping):
            updated = payload.get("updated_at")
            created = payload.get("created_at")
            timestamp = updated if isinstance(updated, str) else ""
            if not timestamp and isinstance(created, str):
                timestamp = created
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (timestamp, mtime, path.name)
