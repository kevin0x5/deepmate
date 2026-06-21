"""User-facing cron command handling."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from deepmate.cron.model import (
    APPROVAL_STATUS_NEEDS_APPROVAL,
    CronApproval,
    CronJob,
    CronState,
    now_iso,
)
from deepmate.cron.planner import (
    approve_job,
    create_job_draft,
    looks_like_cron_request,
    mark_needs_approval,
    preflight_job,
)
from deepmate.cron.schedule import next_run_at
from deepmate.cron.store import CronJobLoadIssue, CronJobStore


def maybe_create_cron_draft(text: str, *, workspace: str | Path) -> str | None:
    """Return a cron draft message when natural language clearly asks for one."""
    if not looks_like_cron_request(text):
        return None
    store = CronJobStore(workspace)
    draft = create_job_draft(
        text,
        workspace=workspace,
        existing_ids={job.id for job in store.load()},
    )
    store.save(draft)
    return _format_created_draft(draft, store)


def handle_cron_command(text: str, *, workspace: str | Path) -> str:
    """Handle one /cron or CLI cron command."""
    raw = text.strip()
    if raw.startswith("/cron"):
        raw = raw[len("/cron") :].strip()
    if not raw:
        return cron_help()
    parts = raw.split(maxsplit=1)
    action = parts[0].strip().lower()
    rest = parts[1].strip() if len(parts) > 1 else ""
    store = CronJobStore(workspace)
    if action in {"help", "-h", "--help"}:
        return cron_help()
    if action in {"add", "create", "new"}:
        if not rest:
            return "Usage: /cron add <schedule and job description>"
        draft = create_job_draft(
            rest,
            workspace=workspace,
            existing_ids={job.id for job in store.load()},
        )
        store.save(draft)
        return _format_created_draft(draft, store)
    if action in {"list", "ls"}:
        report = store.load_report()
        return _with_load_issues(_format_list(report.jobs), report.issues)
    if action == "status":
        report = store.load_report()
        return _with_load_issues(_format_status(report.jobs), report.issues)
    if action == "show":
        return _format_show(store.get(rest))
    if action == "approve":
        job = store.get(rest)
        preflight_job(job, workspace=workspace)
        approved = approve_job(
            replace(
                job,
                state=replace(job.state, next_run_at=next_run_at(job.schedule)),
            )
        )
        store.save(approved)
        return _format_approved(approved, store)
    if action == "pause":
        job = store.get(rest)
        updated = replace(job, enabled=False, updated_at=now_iso())
        store.save(updated)
        return f"Cron job paused: {job.id}"
    if action == "resume":
        job = store.get(rest)
        updated = replace(
            job,
            enabled=True,
            state=replace(job.state, next_run_at=next_run_at(job.schedule)),
            updated_at=now_iso(),
        )
        store.save(updated)
        return f"Cron job resumed: {job.id}\nnext: {updated.state.next_run_at}"
    if action in {"remove", "rm", "delete"}:
        removed = store.remove(rest)
        return f"Cron job removed: {rest}" if removed else f"Cron job not found: {rest}"
    return cron_help()


def cron_help() -> str:
    return "\n".join(
        (
            "Cron jobs",
            "",
            "Commands:",
            "- /cron add <natural language schedule and job>",
            "- /cron list",
            "- /cron status",
            "- /cron show <job_id>",
            "- /cron approve <job_id>",
            "- /cron pause <job_id>",
            "- /cron resume <job_id>",
            "- /cron remove <job_id>",
            "",
            "Jobs are stored in cron/jobs.jsonl in this workspace.",
            "Run due jobs with: deepmate --cron-runner",
            "Keep polling with: deepmate --cron-runner --cron-watch",
        )
    )


def approval_current(job: CronJob) -> CronJob:
    """Return a job marked needs-approval when hand-edited risky fields changed."""
    if job.is_approved() or job.approval.status == APPROVAL_STATUS_NEEDS_APPROVAL:
        return job
    return mark_needs_approval(job, "approval digest no longer matches job fields")


def _format_created_draft(job: CronJob, store: CronJobStore) -> str:
    return "\n".join(
        (
            "准备创建定时任务",
            "",
            f"名称：{job.name}",
            f"ID：{job.id}",
            f"时间：{_schedule_text(job)}",
            f"输出：{job.output.path}/{job.output.filename_template}",
            f"下一次运行：{job.state.next_run_at}",
            "",
            "需要权限：",
            *_permission_lines(job),
            "",
            "已写入草稿："
            f" {store.path.relative_to(store.workspace)}",
            f"确认创建：/cron approve {job.id}",
            f"修改草稿：编辑 {store.path.relative_to(store.workspace)} 后再 approve",
            "运行入口：deepmate --cron-runner 或 deepmate --cron-runner --cron-watch",
        )
    )


def _format_approved(job: CronJob, store: CronJobStore) -> str:
    return "\n".join(
        (
            "定时任务已启用",
            "",
            f"名称：{job.name}",
            f"ID：{job.id}",
            f"下一次运行：{job.state.next_run_at}",
            f"配置：{store.path.relative_to(store.workspace)}",
            "运行入口：deepmate --cron-runner 或 deepmate --cron-runner --cron-watch",
        )
    )


def _format_list(jobs: tuple[CronJob, ...]) -> str:
    if not jobs:
        return "No cron jobs. Create one with /cron add <schedule and job>."
    lines = ["Cron jobs"]
    for job in jobs:
        approval = "approved" if job.is_approved() else "needs approval"
        enabled = "enabled" if job.enabled else "paused"
        lines.append(
            f"- {job.id}: {job.name} ({enabled}, {approval}) next={job.state.next_run_at or '-'}"
        )
    return "\n".join(lines)


def _format_status(jobs: tuple[CronJob, ...]) -> str:
    if not jobs:
        return "No cron jobs."
    lines = ["Cron status"]
    for job in jobs:
        approval = "approved" if job.is_approved() else "needs approval"
        lines.append(
            f"- {job.id}: {job.state.last_status}, {approval}, next={job.state.next_run_at or '-'}"
        )
        if job.state.last_output:
            lines.append(f"  output={job.state.last_output}")
        if job.state.last_error:
            lines.append(f"  error={job.state.last_error}")
    return "\n".join(lines)


def _with_load_issues(body: str, issues: tuple[CronJobLoadIssue, ...]) -> str:
    if not issues:
        return body
    lines = [body, "", "cron/jobs.jsonl has ignored line(s):"]
    for issue in issues[:8]:
        lines.append(f"- line {issue.line_number}: {issue.message}")
    hidden = len(issues) - 8
    if hidden > 0:
        lines.append(f"- ... {hidden} more issue(s)")
    lines.append("Fix the file, then rerun /cron status.")
    return "\n".join(lines)


def _format_show(job: CronJob) -> str:
    return "\n".join(
        (
            f"Cron job: {job.id}",
            f"name: {job.name}",
            f"enabled: {str(job.enabled).lower()}",
            f"schedule: {_schedule_text(job)}",
            f"prompt: {job.job.prompt}",
            f"output: {job.output.path}/{job.output.filename_template}",
            f"approval: {'approved' if job.is_approved() else 'needs approval'}",
            f"last_status: {job.state.last_status}",
            f"next_run_at: {job.state.next_run_at}",
        )
    )


def _schedule_text(job: CronJob) -> str:
    schedule = job.schedule
    if schedule.kind == "weekly":
        return f"每周 {schedule.weekday or '-'} {schedule.time or '09:00'}"
    if schedule.kind == "interval":
        return f"每隔 {schedule.interval_minutes} 分钟"
    return f"每天 {schedule.time or '09:00'}"


def _permission_lines(job: CronJob) -> tuple[str, ...]:
    permissions = job.permissions
    lines = ["- 读取当前工作区", f"- 写入输出目录：{job.output.path}"]
    if permissions.workspace_write:
        lines.append("- 修改工作区文件")
    if permissions.shell:
        lines.append("- Shell")
    if permissions.network:
        lines.append("- 网络")
    if permissions.browser:
        lines.append("- 浏览器")
    if permissions.subagents == "auto":
        lines.append("- 可自动使用 subagents 拆分复杂工作")
    if not any(
        (
            permissions.workspace_write,
            permissions.shell,
            permissions.network,
            permissions.browser,
            permissions.computer_use,
            permissions.mcp_write,
        )
    ):
        lines.append("- 无 Shell / 网络 / 任意工作区写入")
    return tuple(lines)
