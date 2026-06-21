from __future__ import annotations

import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from deepmate.channels.cli import main
from deepmate.channels.tui.commands import handle_tui_command
from deepmate.channels.tui.state import TuiRuntimeState
from deepmate.cron.commands import handle_cron_command, maybe_create_cron_draft
from deepmate.cron.model import (
    JOB_STATUS_BLOCKED,
    CronOutput,
    CronPermissions,
    CronSchedule,
)
from deepmate.cron.planner import preflight_job
from deepmate.cron.runner import run_due_jobs, run_job_now, watch_due_jobs
from deepmate.cron.store import CronJobStore
from deepmate.domain import ProfileRef
from deepmate.runtime import (
    ToolAccessMode,
    ToolAccessPolicy,
    start_runtime_activation,
    start_session_runtime,
)
from deepmate.storage import SessionStore


class CronJobTests(unittest.TestCase):
    def test_cron_add_writes_jsonl_with_job_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)

            message = handle_cron_command(
                "/cron add 每天 9 点总结项目，保存到 reports/daily",
                workspace=workspace,
            )

            store = CronJobStore(workspace)
            jobs = store.load()
            self.assertEqual(len(jobs), 1)
            self.assertIn("准备创建定时任务", message)
            self.assertIn("/cron approve", message)
            self.assertEqual(jobs[0].job.prompt, "每天 9 点总结项目，保存到 reports/daily")
            raw = json.loads(store.path.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("job", raw)
            self.assertIn("prompt", raw["job"])
            self.assertNotIn("task", raw)
            self.assertTrue((workspace / "reports" / "daily").is_dir())

    def test_cron_add_parses_chinese_output_path_without_space(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)

            handle_cron_command(
                "/cron add 每天 9 点生成项目日报，保存到reports/daily",
                workspace=workspace,
            )

            job = CronJobStore(workspace).load()[0]
            self.assertEqual(job.output.path, "reports/daily")
            self.assertTrue((workspace / "reports" / "daily").is_dir())

    def test_cron_add_parses_weekly_and_evening_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)

            handle_cron_command(
                "/cron add 每周晚上 9 点生成项目周报，保存到 reports/weekly",
                workspace=workspace,
            )

            job = CronJobStore(workspace).load()[0]
            self.assertEqual(job.schedule.kind, "weekly")
            self.assertEqual(job.schedule.weekday, "monday")
            self.assertEqual(job.schedule.time, "21:00")
            self.assertEqual(job.output.path, "reports/weekly")

    def test_cron_add_parses_hour_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)

            handle_cron_command(
                "/cron add 每隔 2 小时总结项目状态",
                workspace=workspace,
            )

            job = CronJobStore(workspace).load()[0]
            self.assertEqual(job.schedule.kind, "interval")
            self.assertEqual(job.schedule.interval_minutes, 120)

    def test_cron_add_rejects_one_time_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)

            with self.assertRaisesRegex(ValueError, "recurring"):
                handle_cron_command(
                    "/cron add 只运行一次，生成项目日报",
                    workspace=workspace,
                )

            self.assertEqual(CronJobStore(workspace).load(), ())

    def test_cron_approve_records_digest_and_manual_edit_invalidates_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            handle_cron_command(
                "/cron add daily 09:00 summarize project save to cron/outputs",
                workspace=workspace,
            )
            store = CronJobStore(workspace)
            job = store.load()[0]

            approved_message = handle_cron_command(f"/cron approve {job.id}", workspace=workspace)
            approved = store.get(job.id)

            self.assertIn("定时任务已启用", approved_message)
            self.assertTrue(approved.is_approved())

            edited = replace(
                approved,
                job=replace(approved.job, prompt=approved.job.prompt + " with secrets"),
            )
            store.save(edited)

            self.assertFalse(store.get(job.id).is_approved())

    def test_preflight_rejects_output_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            job = CronJobStore(workspace)
            handle_cron_command("/cron add 每天 9 点总结项目", workspace=workspace)
            created = job.load()[0]
            outside = replace(created, output=CronOutput(path="../outside"))

            with self.assertRaisesRegex(ValueError, "inside the workspace"):
                preflight_job(outside, workspace=workspace)

    def test_preflight_rejects_unattended_write_shell_browser_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            handle_cron_command("/cron add 每天 9 点总结项目", workspace=workspace)
            job = CronJobStore(workspace).load()[0]
            risky = replace(
                job,
                permissions=CronPermissions(
                    read_workspace=True,
                    write_output=True,
                    workspace_write=True,
                    shell=True,
                    network=False,
                    browser=True,
                    computer_use=False,
                    mcp_write=False,
                    subagents="auto",
                ),
            )

            with self.assertRaisesRegex(ValueError, "unsupported permission"):
                preflight_job(risky, workspace=workspace)

    def test_preflight_rejects_one_time_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            handle_cron_command("/cron add 每天 9 点总结项目", workspace=workspace)
            job = CronJobStore(workspace).load()[0]
            one_time = replace(job, schedule=CronSchedule(kind="once"))

            with self.assertRaisesRegex(ValueError, "recurring"):
                preflight_job(one_time, workspace=workspace)

    def test_due_runner_blocks_unapproved_job_and_writes_workspace_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            handle_cron_command("/cron add 每隔 1 分钟总结项目", workspace=workspace)
            store = CronJobStore(workspace)
            job = store.load()[0]
            due_job = replace(
                job,
                state=replace(
                    job.state,
                    next_run_at=(datetime.now().astimezone() - timedelta(minutes=1))
                    .replace(microsecond=0)
                    .isoformat(),
                ),
            )
            store.save(due_job)

            report = run_due_jobs(workspace=workspace)
            updated = store.get(job.id)

            self.assertIn("blocked (needs approval)", report)
            self.assertEqual(updated.state.last_status, JOB_STATUS_BLOCKED)
            self.assertTrue((workspace / updated.state.last_output).exists())

    def test_run_job_now_uses_deepmate_turn_and_writes_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            handle_cron_command("/cron add 每天 9 点总结项目，需要联网抓取", workspace=workspace)
            store = CronJobStore(workspace)
            job = store.load()[0]
            handle_cron_command(f"/cron approve {job.id}", workspace=workspace)
            captured: dict[str, object] = {}

            def fake_run(command, **kwargs):
                captured["command"] = tuple(command)
                captured["cwd"] = kwargs.get("cwd")
                return subprocess.CompletedProcess(command, 0, "cron result\n", "")

            with patch("deepmate.cron.runner.subprocess.run", fake_run):
                report = run_job_now(job.id, workspace=workspace, python_executable="python3")

            updated = store.get(job.id)
            command = captured["command"]
            self.assertIsInstance(command, tuple)
            self.assertIn("--allow-network", command)
            self.assertIn("--subagents", command)
            self.assertIn("--cron-job-run", command)
            self.assertIn("completed", report)
            output = workspace / updated.state.last_output
            self.assertIn("cron result", output.read_text(encoding="utf-8"))

    def test_watch_due_jobs_can_tick_without_sleeping_when_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            lines: list[str] = []

            report = watch_due_jobs(
                workspace=workspace,
                poll_seconds=5,
                report_sink=lines.append,
                max_ticks=1,
            )

            self.assertEqual(report, "")
            self.assertEqual(len(lines), 1)
            self.assertIn("No cron jobs.", lines[0])

    def test_natural_language_creation_returns_draft_without_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)

            message = maybe_create_cron_draft(
                "每天 9 点生成项目日报，保存到 reports/daily",
                workspace=workspace,
            )

            self.assertIsNotNone(message)
            self.assertIn("准备创建定时任务", message or "")
            self.assertEqual(len(CronJobStore(workspace).load()), 1)

    def test_cli_cron_command_does_not_require_provider_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            stdout = io.StringIO()
            stderr = io.StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(
                    (
                        "--workspace",
                        str(workspace),
                        "--cron",
                        "add",
                        "每天 9 点生成项目日报，保存到 reports/daily",
                    )
                )

            self.assertEqual(code, 0, stderr.getvalue())
            self.assertIn("准备创建定时任务", stdout.getvalue())
            self.assertEqual(len(CronJobStore(workspace).load()), 1)

    def test_cli_plain_language_cron_is_intercepted_before_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            stdout = io.StringIO()
            stderr = io.StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(
                    (
                        "--workspace",
                        str(workspace),
                        "每天 9 点生成项目日报，保存到 reports/daily",
                    )
                )

            self.assertEqual(code, 0, stderr.getvalue())
            self.assertIn("准备创建定时任务", stdout.getvalue())
            self.assertEqual(len(CronJobStore(workspace).load()), 1)

    def test_cli_cron_watch_requires_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            stdout = io.StringIO()
            stderr = io.StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(("--workspace", str(workspace), "--cron-watch"))

            self.assertEqual(code, 2)
            self.assertIn("require --cron-runner", stderr.getvalue())

    def test_tui_cron_command_returns_status_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            state = _state(workspace)

            result = handle_tui_command(
                "/cron add 每天 9 点生成项目日报，保存到 reports/daily",
                state,
            )

            self.assertTrue(result.handled)
            self.assertEqual(result.messages[0].title, "/cron")
            self.assertIn("准备创建定时任务", result.messages[0].body)
            self.assertEqual(len(CronJobStore(workspace).load()), 1)

    def test_store_skips_malformed_user_edited_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            store = CronJobStore(workspace)
            store.ensure()
            store.path.write_text("{bad json\n# comment\n", encoding="utf-8")

            self.assertEqual(store.load(), ())

    def test_cron_status_reports_malformed_user_edited_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            store = CronJobStore(workspace)
            store.ensure()
            store.path.write_text("{bad json\n# keep me\n[]\n", encoding="utf-8")

            message = handle_cron_command("/cron status", workspace=workspace)

            self.assertIn("cron/jobs.jsonl has ignored line(s)", message)
            self.assertIn("line 1: invalid JSON", message)
            self.assertIn("line 3: expected a JSON object", message)

    def test_cron_save_preserves_malformed_user_edited_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            store = CronJobStore(workspace)
            store.ensure()
            store.path.write_text("{bad json\n# keep me\n", encoding="utf-8")

            handle_cron_command("/cron add 每天 9 点总结项目", workspace=workspace)

            content = store.path.read_text(encoding="utf-8")
            self.assertIn("{bad json", content)
            self.assertIn("# keep me", content)
            self.assertEqual(len(store.load()), 1)


def _state(workspace: Path) -> TuiRuntimeState:
    session_store = SessionStore.in_directory(workspace / "var" / "sessions")
    profile = ProfileRef(name="default", uri="profiles/default")
    session = session_store.create(workspace=workspace, profile=profile, title="cron test")
    runtime = start_session_runtime(
        start_runtime_activation(
            session_id=session.session_id,
            workspace=workspace,
            profile=profile,
        )
    )
    return TuiRuntimeState(
        provider=_StubProvider(),
        provider_name="stub",
        provider_api_key_env="STUB_API_KEY",
        provider_api_key_available=True,
        model="stub-main",
        default_model="stub-main",
        upgrade_model="stub-pro",
        workspace=workspace,
        profile=profile,
        session_store=session_store,
        session=session,
        transcript=session_store.transcript_store(session),
        runtime=runtime,
        capability_surface=None,
        native_tools=None,
        native_tool_factory=None,
        mcp_tools=None,
        subagents=None,
        tool_access_policy=ToolAccessPolicy(ToolAccessMode.READ_ONLY),
        tool_schemas=(),
        selected_skill_documents=(),
        mcp_servers=(),
        conversation_budget_policy=None,
        provider_retry_policy=None,
        options={},
        max_steps=2,
        trace_recorder=None,
        warning_sink=None,
        data_dir=workspace / "var",
    )


class _StubProvider:
    def complete(self, request):
        raise AssertionError("cron command test should not call provider")

    def complete_stream(self, request, on_delta):
        raise AssertionError("cron command test should not call provider")


if __name__ == "__main__":
    unittest.main()
