"""QA Audit case execution."""

from __future__ import annotations

from pathlib import Path

from deepmate.qa.model import (
    CASE_STATUS_BLOCKED,
    CASE_STATUS_FAILED,
    CASE_STATUS_PASSED,
    CASE_STATUS_WARNING,
    AuditCase,
    AuditCaseResult,
)
from deepmate.qa.store import QaAuditStore
from deepmate.runtime.sandbox import SandboxMode, SandboxPolicy, SandboxRunner

MAX_COMMAND_SECONDS = 120
MAX_OUTPUT_CHARS = 40_000


def run_cases(
    store: QaAuditStore,
    audit_id: str,
    cases: tuple[AuditCase, ...],
    *,
    allow_shell: bool = True,
    allow_browser: bool = False,
    allow_computer: bool = False,
) -> tuple[AuditCaseResult, ...]:
    """Run the automatically executable part of one audit."""
    results: list[AuditCaseResult] = []
    for case in cases:
        if case.runner == "shell":
            result = _run_shell_case(store, audit_id, case, allow_shell=allow_shell)
        elif case.runner == "file":
            result = _run_file_case(store, audit_id, case)
        elif case.runner == "artifact":
            result = AuditCaseResult(
                case_id=case.case_id,
                status=CASE_STATUS_PASSED,
                summary="Report generation case is satisfied by the generated report artifacts.",
                evidence=(),
            )
        elif case.runner == "browser":
            result = _blocked_optional(
                case,
                "Browser automation was not run by this deterministic QA pass.",
                enabled=allow_browser,
            )
        elif case.runner == "computer":
            result = _blocked_optional(
                case,
                "Computer Use real interaction was not run. Grant Computer Use in an agent turn to collect screenshots and interaction evidence.",
                enabled=allow_computer,
            )
        elif case.runner == "service":
            result = AuditCaseResult(
                case_id=case.case_id,
                status=CASE_STATUS_WARNING,
                summary="Service case was planned but needs a project-specific start/health command before automatic execution.",
                evidence=(),
            )
        else:
            result = AuditCaseResult(
                case_id=case.case_id,
                status=CASE_STATUS_BLOCKED,
                summary=f"Runner is not automatic in this environment: {case.runner}",
                evidence=(),
            )
        results.append(result)
    return tuple(results)


def _run_shell_case(
    store: QaAuditStore,
    audit_id: str,
    case: AuditCase,
    *,
    allow_shell: bool,
) -> AuditCaseResult:
    if not allow_shell:
        return AuditCaseResult(
            case_id=case.case_id,
            status=CASE_STATUS_BLOCKED,
            summary="Shell execution was not allowed.",
            evidence=(),
        )
    command = case.steps[0] if case.steps else ""
    if not command:
        return AuditCaseResult(
            case_id=case.case_id,
            status=CASE_STATUS_BLOCKED,
            summary="Shell case has no command step.",
            evidence=(),
        )
    paths = store.paths(audit_id)
    output_dir = paths.evidence / "commands"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_safe_filename(case.case_id)}.txt"
    try:
        if not command.strip():
            return AuditCaseResult(
                case_id=case.case_id,
                status=CASE_STATUS_BLOCKED,
                summary="Shell case has an empty command.",
                evidence=(),
            )
        run_result = SandboxRunner().run(
            command,
            SandboxPolicy(
                workspace=store.workspace,
                cwd=store.workspace,
                network_enabled=False,
                mode=SandboxMode.AUTO,
            ),
            timeout_seconds=MAX_COMMAND_SECONDS,
        )
        output = "\n".join(
            (
                f"$ {command}",
                f"exit_code={run_result.exit_code}",
                f"sandbox_backend={run_result.backend}",
                f"sandboxed={str(run_result.sandboxed).lower()}",
                "",
                "stdout:",
                _head_tail(run_result.stdout, MAX_OUTPUT_CHARS),
                "",
                "stderr:",
                _head_tail(run_result.stderr, MAX_OUTPUT_CHARS),
            )
        )
        output_path.write_text(output, encoding="utf-8")
        relative = store.relative(output_path)
        if _sandbox_backend_failed(run_result.stderr):
            return AuditCaseResult(
                case_id=case.case_id,
                status=CASE_STATUS_BLOCKED,
                summary=f"Command could not run because the QA sandbox backend failed: {command}",
                evidence=(relative,),
                details=_head_tail(run_result.stderr, 2000),
            )
        if run_result.exit_code == 0:
            return AuditCaseResult(
                case_id=case.case_id,
                status=CASE_STATUS_PASSED,
                summary=f"Command passed: {command}",
                evidence=(relative,),
            )
        return AuditCaseResult(
            case_id=case.case_id,
            status=CASE_STATUS_FAILED,
            summary=f"Command exited with {run_result.exit_code}: {command}",
            evidence=(relative,),
            details=_head_tail(run_result.stderr or run_result.stdout, 2000),
        )
    except RuntimeError as exc:
        return AuditCaseResult(
            case_id=case.case_id,
            status=CASE_STATUS_BLOCKED,
            summary=f"Command could not run under QA sandbox policy: {exc}",
            evidence=(),
        )
    except (OSError, ValueError) as exc:
        return AuditCaseResult(
            case_id=case.case_id,
            status=CASE_STATUS_BLOCKED,
            summary=f"Command could not run: {exc}",
            evidence=(),
        )


def _run_file_case(store: QaAuditStore, audit_id: str, case: AuditCase) -> AuditCaseResult:
    paths = store.paths(audit_id)
    output_dir = paths.evidence / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_safe_filename(case.case_id)}.txt"
    evidence_lines = [
        f"case_id={case.case_id}",
        f"title={case.title}",
        f"surface={case.surface}",
        "",
        "workspace evidence:",
    ]
    for name in ("README.md", "README", "pyproject.toml", "package.json", "Cargo.toml", "go.mod"):
        path = store.workspace / name
        if path.exists():
            evidence_lines.append(f"- {name}: present")
    evidence_lines.extend(("", "scenario:", case.scenario_brief))
    output_path.write_text("\n".join(evidence_lines) + "\n", encoding="utf-8")
    return AuditCaseResult(
        case_id=case.case_id,
        status=CASE_STATUS_PASSED,
        summary="Collected project file evidence for review.",
        evidence=(store.relative(output_path),),
    )


def _blocked_optional(case: AuditCase, message: str, *, enabled: bool) -> AuditCaseResult:
    if enabled:
        return AuditCaseResult(
            case_id=case.case_id,
            status=CASE_STATUS_WARNING,
            summary="Capability was requested but no dedicated adapter is wired in this deterministic pass.",
            evidence=(),
        )
    return AuditCaseResult(
        case_id=case.case_id,
        status=CASE_STATUS_BLOCKED,
        summary=message,
        evidence=(),
    )


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value)[:80] or "case"


def _head_tail(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    head_len = max(0, limit // 2)
    tail_len = max(0, limit - head_len)
    return (
        value[:head_len]
        + f"\n... truncated {len(value) - limit} chars ...\n"
        + value[-tail_len:]
    )


def _sandbox_backend_failed(stderr: str) -> bool:
    lowered = stderr.lower()
    return "sandbox-exec:" in lowered and (
        "operation not permitted" in lowered
        or "sandbox_apply" in lowered
        or "no such file" in lowered
    )
