"""Create scheduled job drafts from user-facing text."""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from deepmate.cron.model import (
    APPROVAL_STATUS_APPROVED,
    APPROVAL_STATUS_NEEDS_APPROVAL,
    CRON_OUTPUT_DIR,
    CronApproval,
    CronJob,
    CronJobSpec,
    CronOutput,
    CronPermissions,
    CronSchedule,
    CronState,
    now_iso,
)
from deepmate.cron.schedule import next_run_at, parse_time


def looks_like_cron_request(text: str) -> bool:
    """Return whether plain text likely asks to create a scheduled job."""
    clean = text.strip().lower()
    if not clean or clean.startswith("/"):
        return False
    schedule_markers = (
        "每天",
        "每日",
        "每周",
        "每星期",
        "每隔",
        "定时",
        "schedule",
        "every day",
        "daily",
        "weekly",
        "every week",
        "cron",
    )
    action_markers = (
        "保存",
        "生成",
        "总结",
        "报告",
        "提醒",
        "执行",
        "create",
        "save",
        "write",
        "summarize",
        "report",
    )
    return any(marker in clean for marker in schedule_markers) and any(
        marker in clean for marker in action_markers
    )


def create_job_draft(
    text: str,
    *,
    workspace: str | Path,
    existing_ids: set[str] | None = None,
) -> CronJob:
    """Create one scheduled-job draft with preflight-derived defaults."""
    clean = _clean_prompt(text)
    if _looks_one_time_request(clean):
        raise ValueError("cron jobs are recurring; use a normal prompt for one-time work")
    existing = existing_ids or set()
    schedule = _infer_schedule(clean)
    output = _infer_output(clean)
    permissions = _infer_permissions(clean)
    job_id = _unique_id(_slug(_infer_name(clean)), existing)
    created = now_iso()
    draft = CronJob(
        id=job_id,
        enabled=True,
        name=_infer_name(clean),
        schedule=schedule,
        job=CronJobSpec(prompt=clean, max_steps=8),
        output=output,
        permissions=permissions,
        state=CronState(next_run_at=next_run_at(schedule)),
        approval=CronApproval(status=APPROVAL_STATUS_NEEDS_APPROVAL),
        created_at=created,
        updated_at=created,
    )
    preflight_job(draft, workspace=workspace)
    return draft


def approve_job(job: CronJob) -> CronJob:
    """Return a job with approval for its current risky fields."""
    approved_at = now_iso()
    return replace(
        job,
        approval=CronApproval(
            status=APPROVAL_STATUS_APPROVED,
            digest=job.approval_digest(),
            approved_at=approved_at,
        ),
        updated_at=approved_at,
    )


def mark_needs_approval(job: CronJob, reason: str) -> CronJob:
    return replace(
        job,
        approval=CronApproval(
            status=APPROVAL_STATUS_NEEDS_APPROVAL,
            digest=job.approval.digest,
            approved_at=job.approval.approved_at,
            reason=reason,
        ),
        updated_at=now_iso(),
    )


def preflight_job(job: CronJob, *, workspace: str | Path) -> None:
    """Validate schedule, output path, and permission shape before approval."""
    schedule_kind = job.schedule.kind.strip().lower()
    if schedule_kind == "once":
        raise ValueError("cron jobs are recurring; use a normal prompt for one-time work")
    if schedule_kind not in {"daily", "weekly", "weekdays", "interval"}:
        raise ValueError(f"unsupported cron schedule kind: {job.schedule.kind}")
    next_run_at(job.schedule)
    if not job.job.prompt.strip():
        raise ValueError("cron job prompt is empty")
    output_dir = resolve_output_dir(workspace, job.output.path)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = output_dir / ".deepmate-cron-write-test"
    try:
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        raise ValueError(f"cron output path is not writable: {job.output.path}") from exc
    if job.permissions.workspace_write and not job.permissions.write_output:
        raise ValueError("workspace_write requires write_output")
    unsupported: list[str] = []
    if job.permissions.workspace_write:
        unsupported.append("workspace_write")
    if job.permissions.shell:
        unsupported.append("shell")
    if job.permissions.browser:
        unsupported.append("browser")
    if job.permissions.mcp_write:
        unsupported.append("mcp_write")
    if unsupported:
        joined = ", ".join(unsupported)
        raise ValueError(
            "cron jobs currently run unattended with read-only workspace access "
            f"and cron output writes only; unsupported permission(s): {joined}"
        )
    if job.permissions.computer_use:
        raise ValueError("cron jobs cannot use Computer Use")


def resolve_output_dir(workspace: str | Path, output_path: str) -> Path:
    root = Path(workspace).resolve()
    candidate = Path(output_path.strip() or CRON_OUTPUT_DIR)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise ValueError("cron output path must stay inside the workspace")
    return resolved


def _clean_prompt(text: str) -> str:
    clean = text.strip()
    if clean.startswith("/cron"):
        parts = clean.split(maxsplit=2)
        if len(parts) >= 3 and parts[1] in {"add", "create", "new"}:
            return parts[2].strip()
    return clean


def _infer_schedule(text: str) -> CronSchedule:
    timezone = datetime.now().astimezone().tzinfo
    tz_name = getattr(timezone, "key", "") or ""
    lower = text.lower()
    interval = re.search(r"(?:每隔|every)\s*(\d+)\s*(?:分钟|minute|min)", lower)
    if interval:
        return CronSchedule(
            kind="interval",
            interval_minutes=max(1, int(interval.group(1))),
            timezone=tz_name,
        )
    hour_interval = re.search(r"(?:每隔|every)\s*(\d+)\s*(?:小时|hour|hr)", lower)
    if hour_interval:
        return CronSchedule(
            kind="interval",
            interval_minutes=max(1, int(hour_interval.group(1)) * 60),
            timezone=tz_name,
        )
    if any(marker in lower for marker in ("weekday", "weekdays", "workday", "workdays", "工作日")):
        return CronSchedule(
            kind="weekdays",
            time=_time_from_text(text),
            timezone=tz_name,
        )
    weekday = _weekday_from_text(text)
    if weekday or any(marker in lower for marker in ("每周", "每星期", "weekly", "every week")):
        return CronSchedule(
            kind="weekly",
            weekday=weekday or "monday",
            time=_time_from_text(text),
            timezone=tz_name,
        )
    return CronSchedule(kind="daily", time=_time_from_text(text), timezone=tz_name)


def _time_from_text(text: str) -> str:
    clean = text.replace("：", ":")
    match = re.search(
        r"(上午|早上|下午|晚上|中午)?\s*(\d{1,2})\s*[:点]\s*(\d{1,2})?",
        clean,
    )
    if match:
        period = match.group(1) or ""
        hour = int(match.group(2))
        minute = int(match.group(3) or 0)
        if period in {"下午", "晚上"} and hour < 12:
            hour += 12
        if period == "中午" and hour < 12:
            hour += 12
        parse_time(f"{hour:02d}:{minute:02d}")
        return f"{hour:02d}:{minute:02d}"
    return "09:00"


def _weekday_from_text(text: str) -> str:
    lower = text.lower()
    for label in ("周一", "星期一", "monday", "mon"):
        if label in lower:
            return "monday"
    for label in ("周二", "星期二", "tuesday", "tue"):
        if label in lower:
            return "tuesday"
    for label in ("周三", "星期三", "wednesday", "wed"):
        if label in lower:
            return "wednesday"
    for label in ("周四", "星期四", "thursday", "thu"):
        if label in lower:
            return "thursday"
    for label in ("周五", "星期五", "friday", "fri"):
        if label in lower:
            return "friday"
    for label in ("周六", "星期六", "saturday", "sat"):
        if label in lower:
            return "saturday"
    for label in ("周日", "周天", "星期日", "星期天", "sunday", "sun"):
        if label in lower:
            return "sunday"
    return ""


def _infer_output(text: str) -> CronOutput:
    match = re.search(r"(?:保存到|放到|输出到)\s*([^\s，,。]+)", text, re.I)
    if match is None:
        match = re.search(r"(?:save to|write to)\s+([^\s，,。]+)", text, re.I)
    path = match.group(1).strip() if match else CRON_OUTPUT_DIR
    if path.endswith(".md"):
        output_dir = str(Path(path).parent)
        filename_template = Path(path).name
    else:
        output_dir = path
        filename_template = "{job_id}-{date}-{time}.md"
    return CronOutput(path=output_dir or CRON_OUTPUT_DIR, filename_template=filename_template)


def _infer_permissions(text: str) -> CronPermissions:
    lower = text.lower()
    return CronPermissions(
        read_workspace=True,
        write_output=True,
        workspace_write=any(marker in lower for marker in ("修改", "更新文件", "edit", "modify")),
        shell=any(marker in lower for marker in ("运行测试", "shell", "命令", "pytest", "npm test")),
        network=any(marker in lower for marker in ("联网", "网络", "抓取", "http", "https", "fetch", "网页", "页面")),
        browser=any(marker in lower for marker in ("浏览器", "browser")),
        computer_use=False,
        mcp_write=False,
        subagents="auto",
    )


def _looks_one_time_request(text: str) -> bool:
    lower = text.lower()
    return any(
        marker in lower
        for marker in (
            "一次性",
            "一次",
            "只运行一次",
            "只执行一次",
            "once",
            "one-time",
        )
    )


def _infer_name(text: str) -> str:
    if "日报" in text:
        return "项目日报"
    if "周报" in text:
        return "项目周报"
    if "总结" in text:
        return "定时总结"
    if "report" in text.lower():
        return "scheduled-report"
    return "scheduled-job"


def _slug(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff_-]+", "-", value.strip().lower())
    clean = clean.strip("-_")
    if not clean:
        clean = "scheduled-job"
    if any("\u4e00" <= char <= "\u9fff" for char in clean):
        clean = "cron-job"
    return clean[:48]


def _unique_id(base: str, existing: set[str]) -> str:
    clean = base or "cron-job"
    if clean not in existing:
        return clean
    index = 2
    while f"{clean}-{index}" in existing:
        index += 1
    return f"{clean}-{index}"
