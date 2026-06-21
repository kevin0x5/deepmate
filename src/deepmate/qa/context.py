"""QA Audit context and agent prompt helpers."""

from __future__ import annotations

from pathlib import Path

from deepmate.qa.store import QaAuditStore


def render_qa_context_section(workspace: str | Path, audit_id: str | None = None) -> str:
    """Render bounded QA Audit context for an agent turn."""
    store = QaAuditStore(workspace)
    clean_id = audit_id.strip() if isinstance(audit_id, str) and audit_id.strip() else store.latest_audit_id()
    plan = store.read_plan_state(clean_id)
    cases = store.read_cases(clean_id)
    results = {result.case_id: result for result in store.read_results(clean_id)}
    paths = store.paths(clean_id)
    lines = [
        "<qa_audit_context>",
        f"<audit_id>{clean_id}</audit_id>",
        f"<goal>{plan.goal}</goal>",
        f"<project>{plan.project.project_name}</project>",
        f"<surfaces>{', '.join(plan.surfaces)}</surfaces>",
        f"<plan_path>{store.relative(paths.plan)}</plan_path>",
        f"<cases_path>{store.relative(paths.cases)}</cases_path>",
        f"<evidence_path>{store.relative(paths.evidence)}</evidence_path>",
        f"<report_path>{store.relative(paths.report_html)}</report_path>",
        "<guidance>",
        "- QA Audit is project-adaptive: test real project surfaces and risks, not a fixed template.",
        "- Prefer deterministic evidence first: commands, files, logs, HTTP responses, DOM, trace.",
        "- Use browser tools for Web UI when available.",
        "- Use Computer Use only after explicit user permission to validate real Web UI, TUI, CLI, or desktop visual and interaction experience.",
        "- Use subagents only as bounded workers for complex or parallel case investigation; merge results back into this audit.",
        "- Save evidence under the audit evidence directory and update report.md/report.html.",
        "</guidance>",
        "<cases>",
    ]
    ordered_cases = sorted(cases, key=lambda case: (_case_priority_rank(case.priority), _case_status_rank(results.get(case.case_id))))
    for case in ordered_cases[:20]:
        result = results.get(case.case_id)
        status = result.status if result is not None else "pending"
        lines.append(
            f"- {case.case_id} | {status} | {case.priority} | {case.surface} | "
            f"{case.title} | runner={case.runner} | oracle={case.oracle}"
        )
    if len(cases) > 20:
        lines.append(f"- ... {len(cases) - 20} more case(s) in audit.cases.jsonl")
    lines.extend(["</cases>", "</qa_audit_context>"])
    return "\n".join(lines)


def qa_execute_prompt(workspace: str | Path, audit_id: str | None = None) -> str:
    """Return an agent prompt for a QA Audit execution/review turn."""
    context = render_qa_context_section(workspace, audit_id)
    return "\n\n".join(
        (
            "Continue the active QA Audit.",
            context,
            (
                "Work through the highest-priority pending or blocked cases that can be "
                "validated with available permissions. Collect concrete evidence, avoid "
                "unsafe external actions, and keep the final report decision-oriented. "
                "If a case needs Computer Use or external credentials that are not granted, "
                "mark it blocked with the exact permission or input needed."
            ),
        )
    )


def _case_priority_rank(priority: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(priority.strip().lower(), 3)


def _case_status_rank(result) -> int:
    status = getattr(result, "status", "pending")
    return {
        "failed": 0,
        "blocked": 1,
        "warning": 2,
        "pending": 3,
        "passed": 4,
    }.get(status, 5)
