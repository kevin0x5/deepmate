"""User-facing QA Audit command handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from deepmate.providers import ModelProvider
from deepmate.qa.context import qa_execute_prompt
from deepmate.qa.engine import run_audit, start_audit
from deepmate.qa.store import QaAuditStore
from deepmate.tasks import TaskStore


def maybe_create_qa_audit(
    text: str,
    *,
    workspace: str | Path,
    provider: ModelProvider | None = None,
    model: str = "",
    options: Mapping[str, object] | None = None,
    allow_fallback: bool = False,
) -> str | None:
    """Return a QA audit draft when natural language clearly asks for one."""
    if not looks_like_qa_request(text):
        return None
    goal = _strip_natural_prefix(text)
    _audit_id, message = start_audit(
        goal,
        workspace=workspace,
        provider=provider,
        model=model,
        options=options,
        allow_fallback=allow_fallback,
    )
    return message


def maybe_qa_agent_prompt(text: str, *, workspace: str | Path) -> str | None:
    """Return a QA Audit execution prompt for natural-language continuation requests."""
    clean = text.strip().lower()
    if not clean or clean.startswith("/"):
        return None
    markers = (
        "继续 qa audit",
        "继续质量审计",
        "执行 qa audit",
        "运行 qa audit",
        "continue qa audit",
        "run qa audit",
    )
    if not any(marker in clean for marker in markers):
        return None
    try:
        return qa_execute_prompt(workspace)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def looks_like_qa_request(text: str) -> bool:
    """Return whether text clearly asks for a QA Audit workflow."""
    clean = text.strip().lower()
    if not clean or clean.startswith("/"):
        return False
    explicit_markers = (
        "qa audit",
        "quality audit",
        "质量审计",
    )
    action_markers = (
        "帮我",
        "做一次",
        "开始",
        "创建",
        "生成",
        "执行",
        "运行",
        "run",
        "start",
        "create",
        "generate",
    )
    natural_markers = (
        "体验验收",
        "发布前质量验收",
        "发布前验收",
    )
    if any(marker in clean for marker in explicit_markers):
        return any(marker in clean for marker in action_markers) or any(
            marker in clean for marker in natural_markers
        )
    return any(marker in clean for marker in natural_markers) and any(
        marker in clean for marker in action_markers
    )


def handle_qa_command(
    text: str,
    *,
    workspace: str | Path,
    provider: ModelProvider | None = None,
    model: str = "",
    options: Mapping[str, object] | None = None,
    allow_fallback: bool = False,
) -> str:
    """Handle one /qa command."""
    raw = text.strip()
    if raw.startswith("/qa"):
        raw = raw[len("/qa") :].strip()
    elif raw.startswith("qa "):
        raw = raw[len("qa ") :].strip()
    if not raw:
        return qa_help()
    parts = raw.split(maxsplit=1)
    action = parts[0].strip().lower()
    rest = parts[1].strip() if len(parts) > 1 else ""
    if action in {"help", "-h", "--help"}:
        return qa_help()
    if action in {"audit", "plan", "start", "new"}:
        if not rest:
            return "Usage: /qa <测试目标>"
        _audit_id, message = start_audit(
            rest,
            workspace=workspace,
            provider=provider,
            model=model,
            options=options,
            allow_fallback=allow_fallback,
        )
        return message
    if action in {"run", "execute"}:
        audit_id, options = _parse_run_args(rest)
        result = run_audit(
            audit_id,
            workspace=workspace,
            allow_shell=not options["no_shell"],
            allow_browser=False,
            allow_computer=False,
        )
        return result.summary
    if action in {"prompt", "agent-prompt"}:
        return qa_execute_prompt(workspace, rest or None)
    if action in {"status", "show"}:
        return _format_status(workspace, rest)
    if action in {"list", "ls"}:
        return _format_list(workspace)
    if action == "report":
        return _format_report(workspace, rest)
    if action == "task":
        return _create_task_plan(workspace, rest)
    _audit_id, message = start_audit(
        raw,
        workspace=workspace,
        provider=provider,
        model=model,
        options=options,
        allow_fallback=allow_fallback,
    )
    return message


def qa_help() -> str:
    return "\n".join(
        (
            "QA Audit",
            "",
            "Workflow:",
            "- /qa <测试目标>          Generate a test plan, outline, editable cases, and permission preview.",
            "- /qa run [audit_id]      Confirm the plan/permissions, execute available checks, and generate report.html.",
            "",
            "Useful views:",
            "- /qa status [audit_id]   Show current state and result counts.",
            "- /qa report [audit_id]   Show report paths.",
            "- /qa task [audit_id]     Turn findings into a Task Mode repair plan.",
            "- /qa list                List workspace audits.",
            "",
            "Artifacts are stored under qa/audits/<audit_id>/.",
        )
    )


def _parse_run_args(raw: str) -> tuple[str | None, dict[str, bool]]:
    parts = [part.strip() for part in raw.split() if part.strip()]
    audit_id: str | None = None
    options = {"no_shell": False}
    for part in parts:
        if part in {"--browser", "--computer"}:
            raise ValueError(
                "Do not pass browser/computer flags to /qa run. Those checks are planned as permissioned QA cases and reported when blocked."
            )
        elif part == "--no-shell":
            options["no_shell"] = True
        elif part.startswith("-"):
            raise ValueError(f"unknown /qa run option: {part}")
        elif audit_id is None:
            audit_id = part
        else:
            raise ValueError("usage: /qa run [audit_id] [--no-shell]")
    return audit_id, options


def _format_status(workspace: str | Path, audit_id: str) -> str:
    store = QaAuditStore(workspace)
    clean_id = audit_id.strip() or store.latest_audit_id()
    state = store.read_state_mapping(clean_id)
    plan = store.read_plan_state(clean_id)
    cases_report = store.read_cases_report(clean_id)
    results = store.read_results(clean_id)
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    lines = [
        f"QA Audit: {clean_id}",
        f"status: {plan.status}",
        f"goal: {plan.goal}",
        f"project: {plan.project.project_name}",
        f"surfaces: {', '.join(plan.surfaces)}",
        f"cases: {len(cases_report.cases)}",
        f"permissions confirmed: {'yes' if state.get('permissions_confirmed') else 'no'}",
        f"results: passed={counts.get('passed', 0)}, warning={counts.get('warning', 0)}, failed={counts.get('failed', 0)}, blocked={counts.get('blocked', 0)}",
    ]
    if not state.get("permissions_confirmed"):
        lines.append("next: review the plan, then run /qa run")
    if cases_report.issues:
        lines.append("case file issues:")
        lines.extend(f"- line {issue.line_number}: {issue.message}" for issue in cases_report.issues[:8])
    return "\n".join(lines)


def _format_list(workspace: str | Path) -> str:
    store = QaAuditStore(workspace)
    audits = store.list_audits()
    if not audits:
        return "No QA audits. Start one with /qa <测试目标>."
    lines = ["QA audits"]
    for audit_id in audits:
        try:
            plan = store.read_plan_state(audit_id)
            lines.append(f"- {audit_id}: {plan.status} - {plan.goal}")
        except (OSError, ValueError, json.JSONDecodeError):
            lines.append(f"- {audit_id}: unreadable")
    return "\n".join(lines)


def _format_report(workspace: str | Path, audit_id: str) -> str:
    store = QaAuditStore(workspace)
    clean_id = audit_id.strip() or store.latest_audit_id()
    paths = store.paths(clean_id)
    if not paths.report_html.exists():
        return f"No QA report yet for {clean_id}. Run /qa run {clean_id} first."
    return "\n".join(
        (
            f"QA Audit report: {clean_id}",
            f"- markdown: {store.relative(paths.report_md)}",
            f"- html: {store.relative(paths.report_html)}",
            f"- evidence: {store.relative(paths.evidence)}",
        )
    )


def _create_task_plan(workspace: str | Path, audit_id: str) -> str:
    store = QaAuditStore(workspace)
    clean_id = audit_id.strip() or store.latest_audit_id()
    plan = store.read_plan_state(clean_id)
    results = store.read_results(clean_id)
    paths = store.paths(clean_id)
    risky = [item for item in results if item.status in {"failed", "warning", "blocked"}]
    task_store = TaskStore(workspace)
    task_store.ensure()
    lines = [
        "# 当前任务计划",
        "",
        "## 目标",
        f"修复 QA Audit `{clean_id}` 发现的高影响问题，并保持已通过用例不回退。",
        "",
        "## 验收契约",
        f"- [ ] 复核 QA 报告：`{store.relative(paths.report_html)}`",
        "- [ ] 修复 failed / warning 中确认需要处理的问题",
        "- [ ] 对 blocked 用例明确需要的权限、环境或人工确认",
        "- [ ] 重新运行相关 QA Audit 或等价验证",
        "- [ ] 更新最终修复说明、验证结果和剩余风险",
        "",
        "## 当前方案",
        f"以 `{plan.goal}` 的 QA Audit 结果为输入，优先处理用户影响最大的失败和警告。",
        "",
        "## 执行计划",
    ]
    if risky:
        lines.extend(f"- [ ] {item.case_id}: {item.summary}" for item in risky[:12])
    else:
        lines.append("- [ ] 当前 QA Audit 没有 failed/warning/blocked；复核报告后确认是否需要优化。")
    lines.extend(
        [
            "",
            "## 验证策略",
            f"- 重新运行 `/qa run {clean_id}` 或同等测试命令。",
            "- 对真实用户体验问题，补充截图或交互证据。",
            "",
            "## 讨论与决策",
            f"- QA Audit report: `{store.relative(paths.report_html)}`",
            "",
            "## 当前进展",
            "- 已完成：从 QA Audit 生成修复任务计划",
            "- 进行中：待确认修复范围",
            "- 下一步：进入 task/execute 或直接修复最高优先级问题",
            "",
        ]
    )
    task_store.write_plan("\n".join(lines))
    return "\n".join(
        (
            "已从 QA Audit 生成修复任务计划",
            "",
            f"QA Audit：{clean_id}",
            "任务计划：task/plan.md",
            "下一步：task/execute",
        )
    )


def _strip_natural_prefix(text: str) -> str:
    clean = text.strip()
    prefixes = (
        "请做一次",
        "帮我做一次",
        "做一次",
        "run a",
        "create a",
    )
    lowered = clean.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix.lower()):
            return clean[len(prefix) :].strip()
    return clean
