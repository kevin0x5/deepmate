"""High-level QA Audit engine."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from deepmate.providers import ModelProvider
from deepmate.qa.model import (
    AUDIT_STATUS_COMPLETED,
    AUDIT_STATUS_RUNNING,
    AuditRunResult,
)
from deepmate.qa.planner import create_audit_plan, create_fallback_audit_plan
from deepmate.qa.report import render_report_markdown
from deepmate.qa.runner import run_cases
from deepmate.qa.store import QaAuditStore
from deepmate.tools import NativeToolRegistry, workspace_report_tools


def start_audit(
    goal: str,
    *,
    workspace: str | Path,
    provider: ModelProvider | None,
    model: str = "",
    options: Mapping[str, object] | None = None,
    allow_fallback: bool = False,
) -> tuple[str, str]:
    """Create audit plan/cases and return audit id plus user message."""
    store = QaAuditStore(workspace)
    if provider is None or not model.strip():
        if not allow_fallback:
            raise ValueError(
                "QA Audit planning requires an active model provider. Configure a provider key first, then run /qa <测试目标>."
            )
        plan, cases, markdown = create_fallback_audit_plan(goal, workspace=workspace)
    else:
        plan, cases, markdown = create_audit_plan(
            goal,
            workspace=workspace,
            provider=provider,
            model=model,
            options=options or {},
        )
    paths = store.write_audit(plan, cases, plan_markdown=markdown)
    relative_plan = store.relative(paths.plan)
    relative_cases = store.relative(paths.cases)
    message = "\n".join(
        (
            "QA Audit 方案已生成",
            "",
            f"目标：{plan.goal}",
            f"识别项目：{plan.project.project_name} ({', '.join(plan.project.project_kinds)})",
            f"主要 surface：{', '.join(plan.surfaces)}",
            "",
            "测试方案概述：",
            *[f"- {item}" for item in plan.scope[:6]],
            "",
            "测试大纲：",
            *[f"- {case.case_id}: {case.title}" for case in cases[:8]],
            "",
            "执行前权限清单：",
            *[f"- {item}" for item in plan.permissions],
            "",
            f"方案：{relative_plan}",
            f"用例：{relative_cases}",
            "",
            "确认方向无误后输入：/qa run",
            "需要微调细节时，先编辑 audit.cases.jsonl 再运行。",
        )
    )
    return plan.audit_id, message


def run_audit(
    audit_id: str | None,
    *,
    workspace: str | Path,
    allow_shell: bool = True,
    allow_browser: bool = False,
    allow_computer: bool = False,
) -> AuditRunResult:
    """Run one audit and write results/report artifacts."""
    store = QaAuditStore(workspace)
    clean_id = audit_id.strip() if isinstance(audit_id, str) and audit_id.strip() else store.latest_audit_id()
    plan = store.read_plan_state(clean_id)
    cases_report = store.read_cases_report(clean_id)
    if cases_report.issues:
        issues = "; ".join(f"line {issue.line_number}: {issue.message}" for issue in cases_report.issues[:5])
        raise ValueError(f"audit.cases.jsonl has invalid line(s): {issues}")
    cases = cases_report.cases
    if not cases:
        raise ValueError(f"QA audit has no runnable cases: {clean_id}")
    store.update_state(
        clean_id,
        status=AUDIT_STATUS_RUNNING,
        permissions_confirmed=True,
        permissions_confirmed_at=_now_for_state(),
        requested_permissions=list(plan.permissions),
    )
    results = run_cases(
        store,
        clean_id,
        cases,
        allow_shell=allow_shell,
        allow_browser=allow_browser,
        allow_computer=allow_computer,
    )
    store.write_results(clean_id, results)
    paths = store.paths(clean_id)
    relative_root = store.relative(paths.root)
    markdown = render_report_markdown(
        plan,
        cases,
        results,
        relative_root=relative_root,
    )
    store.write_report_markdown(clean_id, markdown)
    _render_html_report(store, paths.report_md, paths.report_html, plan.project.project_name)
    store.update_state(clean_id, status=AUDIT_STATUS_COMPLETED)
    summary = _summary(results, report_html=store.relative(paths.report_html))
    return AuditRunResult(
        audit_id=clean_id,
        status=AUDIT_STATUS_COMPLETED,
        report_markdown=store.relative(paths.report_md),
        report_html=store.relative(paths.report_html),
        results=results,
        summary=summary,
    )


def _render_html_report(store: QaAuditStore, source: Path, output: Path, project_name: str) -> None:
    registry = NativeToolRegistry(workspace_report_tools(store.workspace))
    tool = registry.get("render_html_report")
    if tool is None:
        raise ValueError("render_html_report tool is unavailable")
    tool.call(
        {
            "source_path": store.relative(source),
            "output_path": store.relative(output),
            "title": f"QA Audit Report: {project_name}",
            "theme": "graphite",
            "layout": "report",
            "overwrite": True,
        }
    )


def _summary(results, *, report_html: str) -> str:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return "\n".join(
        (
            "QA Audit 完成",
            "",
            f"报告：{report_html}",
            "",
            "结果："
            f" passed={counts.get('passed', 0)},"
            f" warning={counts.get('warning', 0)},"
            f" failed={counts.get('failed', 0)},"
            f" blocked={counts.get('blocked', 0)}",
        )
    )


def _now_for_state() -> str:
    from deepmate.qa.model import now_iso

    return now_iso()
