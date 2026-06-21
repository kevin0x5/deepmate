"""QA Audit report content rendering."""

from __future__ import annotations

from collections import Counter

from deepmate.qa.model import (
    CASE_STATUS_BLOCKED,
    CASE_STATUS_FAILED,
    CASE_STATUS_PASSED,
    CASE_STATUS_WARNING,
    AuditCase,
    AuditCaseResult,
    AuditPlan,
)


def render_report_markdown(
    plan: AuditPlan,
    cases: tuple[AuditCase, ...],
    results: tuple[AuditCaseResult, ...],
    *,
    relative_root: str,
) -> str:
    """Render the source Markdown for a QA Audit report."""
    result_by_case = {result.case_id: result for result in results}
    counts = Counter(result.status for result in results)
    decision = _decision(counts)
    lines = [
        f"# QA Audit Report: {plan.project.project_name}",
        "",
        "## Executive Summary",
        f"- Goal: {plan.goal}",
        f"- Decision: {decision}",
        f"- Results: {counts.get(CASE_STATUS_PASSED, 0)} passed, "
        f"{counts.get(CASE_STATUS_WARNING, 0)} warnings, "
        f"{counts.get(CASE_STATUS_FAILED, 0)} failed, "
        f"{counts.get(CASE_STATUS_BLOCKED, 0)} blocked.",
        f"- Evidence root: `{relative_root}/evidence`",
        "",
        "## Project Profile",
        f"- Project kinds: {_join(plan.project.project_kinds)}",
        f"- Surfaces: {_join(plan.surfaces)}",
        f"- Existing test commands: {_join(plan.project.test_commands) or '-'}",
        f"- Discovery evidence: {_join(plan.project.evidence) or '-'}",
        "",
        "## Audit Scope",
        *[f"- {item}" for item in plan.scope],
        "",
        "## Risk Model",
        *[f"- {item}" for item in plan.risk_model],
        "",
        "## Coverage Matrix",
        "",
        "| Surface | Cases | Result |",
        "| --- | ---: | --- |",
    ]
    for surface in _surfaces(cases):
        surface_cases = [case for case in cases if case.surface == surface]
        statuses = [
            result_by_case[case.case_id].status
            for case in surface_cases
            if case.case_id in result_by_case
        ]
        lines.append(f"| {surface} | {len(surface_cases)} | {_surface_status(statuses)} |")
    lines.extend(["", "## Case Results", ""])
    for case in cases:
        result = result_by_case.get(case.case_id)
        status = result.status if result is not None else "pending"
        summary = result.summary if result is not None else "Not run."
        lines.extend(
            [
                f"### {case.case_id} | {case.title}",
                f"- Status: {status}",
                f"- Surface: {case.surface}",
                f"- Risk area: {case.risk_area}",
                f"- Priority: {case.priority}",
                f"- Scenario: {case.scenario_brief}",
                f"- Summary: {summary}",
            ]
        )
        if result and result.evidence:
            lines.append("- Evidence:")
            lines.extend(f"  - `{ref}`" for ref in result.evidence)
        if result and result.details:
            lines.append("- Details:")
            lines.append("")
            lines.append("```text")
            lines.append(result.details[:4000])
            lines.append("```")
        lines.append("")
    lines.extend(
        [
            "## Real User Experience Notes",
            (
                "Computer Use cases are listed as blocked unless the user grants explicit "
                "real interaction permission. When enabled, Web UI, TUI, CLI, and desktop "
                "checks should include screenshots, accessibility snapshots, and interaction logs."
            ),
            "",
            "## Recommendations",
            *_recommendations(results),
            "",
            "## Residual Risks",
            "- This audit only proves the cases and evidence shown above.",
            "- Blocked or manual cases require explicit follow-up before a release decision.",
            "- AI-assisted UX findings should be tied back to screenshots or interaction logs.",
            "",
            "## Sources",
            f"- Workspace audit artifacts: `{relative_root}`",
            "- Project files and commands listed in the case evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _decision(counts: Counter[str]) -> str:
    if counts.get(CASE_STATUS_FAILED, 0):
        return "Not ready; high-confidence failures need attention."
    if counts.get(CASE_STATUS_BLOCKED, 0):
        return "Needs review; some coverage is blocked."
    if counts.get(CASE_STATUS_WARNING, 0):
        return "Conditionally ready; warnings should be triaged."
    return "Ready for the audited scope."


def _recommendations(results: tuple[AuditCaseResult, ...]) -> list[str]:
    failed = [item for item in results if item.status == CASE_STATUS_FAILED]
    blocked = [item for item in results if item.status == CASE_STATUS_BLOCKED]
    warnings = [item for item in results if item.status == CASE_STATUS_WARNING]
    lines: list[str] = []
    if failed:
        lines.append("- Fix failed cases before release or handoff.")
    if blocked:
        lines.append("- Resolve blocked coverage or explicitly accept the residual risk.")
    if warnings:
        lines.append("- Triage warnings and decide whether they affect the current release goal.")
    if not lines:
        lines.append("- Keep the generated cases as a reusable quality baseline.")
    return lines


def _surfaces(cases: tuple[AuditCase, ...]) -> tuple[str, ...]:
    seen: list[str] = []
    for case in cases:
        if case.surface not in seen:
            seen.append(case.surface)
    return tuple(seen)


def _surface_status(statuses: list[str]) -> str:
    if not statuses:
        return "pending"
    if CASE_STATUS_FAILED in statuses:
        return CASE_STATUS_FAILED
    if CASE_STATUS_BLOCKED in statuses:
        return CASE_STATUS_BLOCKED
    if CASE_STATUS_WARNING in statuses:
        return CASE_STATUS_WARNING
    return CASE_STATUS_PASSED


def _join(values: tuple[str, ...]) -> str:
    return ", ".join(values)
