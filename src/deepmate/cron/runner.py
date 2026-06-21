"""Cron job runner."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from deepmate.cron.model import (
    JOB_STATUS_BLOCKED,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    CronJob,
    CronState,
    now_iso,
)
from deepmate.cron.planner import mark_needs_approval, resolve_output_dir
from deepmate.cron.schedule import is_due, next_run_at
from deepmate.cron.store import CronJobStore


def run_due_jobs(
    *,
    workspace: str | Path,
    python_executable: str = sys.executable,
) -> str:
    """Run all due approved jobs once and return a compact report."""
    store = CronJobStore(workspace)
    jobs = store.load()
    if not jobs:
        return "No cron jobs."
    reports: list[str] = []
    for job in jobs:
        if not job.enabled:
            continue
        if not is_due(job.state.next_run_at):
            continue
        reports.append(
            _run_one(job, store=store, python_executable=python_executable)
        )
    if not reports:
        return "No cron jobs due."
    return "\n".join(reports)


def watch_due_jobs(
    *,
    workspace: str | Path,
    python_executable: str = sys.executable,
    poll_seconds: int = 60,
    report_sink=None,
    max_ticks: int | None = None,
) -> str:
    """Poll due cron jobs until interrupted.

    The optional max_ticks argument is for tests and controlled smoke checks.
    """
    interval = max(5, int(poll_seconds))
    reports: list[str] = []
    ticks = 0
    while True:
        ticks += 1
        report = run_due_jobs(
            workspace=workspace,
            python_executable=python_executable,
        )
        line = f"[{datetime.now().astimezone().replace(microsecond=0).isoformat()}] {report}"
        if report_sink is not None:
            report_sink(line)
        else:
            reports.append(line)
        if max_ticks is not None and ticks >= max_ticks:
            break
        time.sleep(interval)
    return "\n".join(reports)


def run_job_now(
    job_id: str,
    *,
    workspace: str | Path,
    python_executable: str = sys.executable,
) -> str:
    """Run one job immediately, regardless of next_run_at."""
    store = CronJobStore(workspace)
    job = store.get(job_id)
    return _run_one(job, store=store, python_executable=python_executable)


def _run_one(
    job: CronJob,
    *,
    store: CronJobStore,
    python_executable: str,
) -> str:
    started = now_iso()
    if not job.is_approved():
        updated = mark_needs_approval(job, "job changed or was never approved")
        output = _write_status_output(
            updated,
            store.workspace,
            status=JOB_STATUS_BLOCKED,
            body=(
                "# Cron job blocked\n\n"
                f"Job `{job.id}` needs approval before it can run.\n\n"
                f"Approve with `/cron approve {job.id}` or "
                f"`deepmate --cron approve {job.id}`.\n"
            ),
        )
        store.save(
            replace(
                updated,
                state=_next_state(
                    updated,
                    status=JOB_STATUS_BLOCKED,
                    started=started,
                    output=output,
                    error="needs approval",
                ),
            )
        )
        return f"{job.id}: blocked (needs approval) -> {output}"
    if _uses_unavailable_runtime_permissions(job):
        updated = mark_needs_approval(
            job,
            "job asks for permissions cron cannot run unattended",
        )
        output = _write_status_output(
            updated,
            store.workspace,
            status=JOB_STATUS_BLOCKED,
            body=(
                "# Cron job blocked\n\n"
                "This job asks for permissions that cron runner does not execute "
                "in the background yet.\n"
            ),
        )
        store.save(
            replace(
                updated,
                state=_next_state(
                    updated,
                    status=JOB_STATUS_BLOCKED,
                    started=started,
                    output=output,
                    error="unsupported background permission",
                ),
            )
        )
        return f"{job.id}: blocked (unsupported permission) -> {output}"
    prompt = _runner_prompt(job)
    command = [
        python_executable,
        "-m",
        "deepmate",
        "--workspace",
        str(store.workspace),
        "--read-only-tools",
        "--max-steps",
        str(job.job.max_steps),
        "--cron-job-run",
        prompt,
    ]
    if job.permissions.network:
        command.insert(-2, "--allow-network")
    if job.permissions.subagents == "auto":
        command.insert(-2, "--subagents")
    env = _subprocess_env()
    try:
        result = subprocess.run(
            command,
            cwd=str(store.workspace),
            env=env,
            text=True,
            capture_output=True,
            timeout=60 * 30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        output = _write_status_output(
            job,
            store.workspace,
            status=JOB_STATUS_FAILED,
            body=f"# Cron job failed\n\n{exc}\n",
        )
        store.save(
            replace(
                job,
                state=_next_state(
                    job,
                    status=JOB_STATUS_FAILED,
                    started=started,
                    output=output,
                    error=str(exc),
                ),
            )
        )
        return f"{job.id}: failed -> {output}"
    content = (result.stdout or "").strip()
    if result.stderr.strip():
        content = (content + "\n\n" if content else "") + "```text\n" + result.stderr.strip() + "\n```"
    status = JOB_STATUS_COMPLETED if result.returncode == 0 else JOB_STATUS_FAILED
    output = _write_status_output(
        job,
        store.workspace,
        status=status,
        body=_format_success_output(job, content, result.returncode),
    )
    store.save(
        replace(
            job,
            state=_next_state(
                job,
                status=status,
                started=started,
                output=output,
                error="" if result.returncode == 0 else f"exit_code={result.returncode}",
            ),
        )
    )
    return f"{job.id}: {status} -> {output}"


def _next_state(
    job: CronJob,
    *,
    status: str,
    started: str,
    output: str,
    error: str,
) -> CronState:
    return CronState(
        last_run_at=started,
        last_status=status,
        next_run_at=next_run_at(job.schedule),
        last_output=output,
        last_error=error,
    )


def _write_status_output(
    job: CronJob,
    workspace: Path,
    *,
    status: str,
    body: str,
) -> str:
    output_dir = resolve_output_dir(workspace, job.output.path)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = _render_filename(job, status=status)
    path = output_dir / file_name
    path.write_text(body.rstrip() + "\n", encoding="utf-8")
    try:
        return str(path.resolve().relative_to(workspace.resolve()))
    except ValueError:
        return str(path)


def _render_filename(job: CronJob, *, status: str) -> str:
    now = datetime.now().astimezone()
    rendered = job.output.filename_template.format(
        job_id=job.id,
        date=now.strftime("%Y-%m-%d"),
        time=now.strftime("%H%M"),
        status=status,
    )
    rendered = rendered.replace("/", "-").replace("\\", "-").strip()
    if not rendered:
        rendered = f"{job.id}-{now.strftime('%Y-%m-%d-%H%M')}-{status}.md"
    if not rendered.endswith(".md"):
        rendered += ".md"
    return rendered


def _runner_prompt(job: CronJob) -> str:
    return "\n".join(
        (
            "Scheduled Deepmate job.",
            "",
            "Follow this job prompt and produce a concise Markdown result.",
            "Do not modify workspace files except through the cron output writer.",
            "",
            f"Job ID: {job.id}",
            f"Job prompt: {job.job.prompt}",
        )
    )


def _format_success_output(job: CronJob, content: str, exit_code: int) -> str:
    title = f"# Cron job: {job.name}\n"
    meta = (
        f"\n- job_id: {job.id}"
        f"\n- status: {'completed' if exit_code == 0 else 'failed'}"
        f"\n- exit_code: {exit_code}\n\n"
    )
    return title + meta + (content.strip() or "No output.")


def _uses_unavailable_runtime_permissions(job: CronJob) -> bool:
    permissions = job.permissions
    return bool(
        permissions.workspace_write
        or permissions.shell
        or permissions.browser
        or permissions.computer_use
        or permissions.mcp_write
    )


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    src_root = str(Path(__file__).resolve().parents[2])
    paths = [path for path in env.get("PYTHONPATH", "").split(os.pathsep) if path]
    if src_root not in paths:
        env["PYTHONPATH"] = os.pathsep.join([src_root, *paths])
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("LC_ALL", "en_US.UTF-8")
    env.setdefault("LANG", "en_US.UTF-8")
    return env
