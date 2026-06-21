from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from deepmate.activity import ActivityEntry, ActivityStore
from deepmate.app import AppSettings, ContextSettings, ModelPurposeSettings
from deepmate.domain import Message, MessageRole, ProfileRef
from deepmate.memory import (
    curator_pending_store,
    record_curator_pending_checkpoint,
    run_daily_memory_maintenance,
)
from deepmate.providers import ModelConversationItem, ModelResponse
from deepmate.storage import SessionStore
from deepmate.trace import TraceRecorder

LOCAL_TZ = timezone(timedelta(hours=8))


class StubProvider:
    def __init__(self, content: str) -> None:
        self.content = content
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return ModelResponse(content=self.content)


class ListTraceSink:
    def __init__(self) -> None:
        self.events = []

    def write(self, event) -> None:
        self.events.append(event)


def _settings(root: Path) -> AppSettings:
    workspace = root / "workspace"
    deepmate_home = root / "home"
    global_profile_dir = deepmate_home / "profiles" / "default"
    project_profile_dir = workspace / "profiles" / "default"
    global_profile_dir.mkdir(parents=True)
    project_profile_dir.mkdir(parents=True)
    (global_profile_dir / "identity.md").write_text("# Identity\n", encoding="utf-8")
    (global_profile_dir / "soul.md").write_text("# Soul\n", encoding="utf-8")
    (global_profile_dir / "user.md").write_text("", encoding="utf-8")
    (global_profile_dir / "memory.md").write_text(
        "- 保持回答克制。\n",
        encoding="utf-8",
    )
    (project_profile_dir / "memory.md").write_text("", encoding="utf-8")
    return AppSettings(
        workspace=workspace,
        data_dir=root / "var",
        deepmate_home=deepmate_home,
        active_profile="default",
        trace_sink=root / "trace.jsonl",
        default_provider="deepseek",
        context=ContextSettings(
            response_token_reserve=0,
            safety_margin_tokens=0,
        ),
        model_purposes={
            "memory": ModelPurposeSettings(model="deepseek-v4-flash")
        },
    )


class MemoryMaintenanceTests(unittest.TestCase):
    def test_idle_day_skips_without_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = _settings(root)
            provider = StubProvider("{}")
            sink = ListTraceSink()

            result = run_daily_memory_maintenance(
                provider=provider,
                settings=settings,
                fallback_model="fallback",
                session_store=SessionStore.in_directory(root / "sessions"),
                trace_recorder=TraceRecorder(sink),
                local_date="2026-06-03",
            )

            self.assertFalse(result.ran)
            self.assertEqual(result.reason, "no_user_activity")
            self.assertEqual(len(provider.requests), 0)
            self.assertEqual(sink.events[0].kind, "memory_maintenance_skipped")

    def test_activity_day_updates_profile_and_monthly_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = _settings(root)
            store = ActivityStore(settings.data_dir / "activity" / "default")
            store.append_daily_entry(
                ActivityEntry(
                    timestamp="2026-06-03T02:00:00+08:00",
                    event="session_summary_checkpoint",
                    status="completed",
                    title="Memory checkpoint",
                    summary="确认 checkpoint 后立即写 memory，并由 maintenance 做离线整理。",
                    session_id="session-1",
                    session_title="Memory closure",
                    profile="default",
                    workspace=str(settings.workspace),
                    summary_id="summary-1",
                )
            )
            provider = StubProvider(
                (
                    '{"profile_patch":{"operations":['
                    '{"action":"write_user","content":"用户偏好把模块做闭环后再转向下一块。"}'
                    ']},'
                    '"monthly_summary":{"summary":"收敛 memory 闭环方案。",'
                    '"highlights":["确认 maintenance 是离线质量治理。"],'
                    '"next_steps":["继续 skill 和 MCP 治理。"]}}'
                )
            )
            sink = ListTraceSink()

            result = run_daily_memory_maintenance(
                provider=provider,
                settings=settings,
                fallback_model="fallback",
                session_store=SessionStore.in_directory(root / "sessions"),
                trace_recorder=TraceRecorder(sink),
                local_date="2026-06-03",
            )

            self.assertTrue(result.ran)
            self.assertEqual(result.reason, "completed")
            self.assertTrue(result.profile_changed)
            self.assertTrue(result.monthly_summary_written)
            self.assertEqual(len(provider.requests), 1)
            self.assertIn(
                "- 用户偏好把模块做闭环后再转向下一块。",
                (settings.global_profile_dir() / "user.md").read_text(encoding="utf-8"),
            )
            user_prompt = provider.requests[0].conversation[1].message.content
            self.assertIn("Current global user.md bullets:", user_prompt)
            self.assertIn("Current global memory.md bullets:", user_prompt)
            self.assertIn("Current project memory.md bullets:", user_prompt)
            monthly = store.monthly_summary_path("2026-06").read_text(encoding="utf-8")
            self.assertIn("## 2026-06-03", monthly)
            self.assertIn("收敛 memory 闭环方案。", monthly)
            self.assertIn("maintenance 是离线质量治理", monthly)
            self.assertIn(
                "memory_maintenance_completed",
                tuple(event.kind for event in sink.events),
            )

    def test_profile_patch_budget_uses_fallback_main_model_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = _settings(root)
            settings = AppSettings(
                workspace=settings.workspace,
                data_dir=settings.data_dir,
                deepmate_home=settings.deepmate_home,
                active_profile=settings.active_profile,
                trace_sink=settings.trace_sink,
                default_provider=settings.default_provider,
                context=ContextSettings(
                    hot_profile_budget_ratio=0.005,
                    hot_profile_min_tokens=1,
                    hot_profile_max_tokens=10_000,
                    response_token_reserve=0,
                    safety_margin_tokens=0,
                ),
                model_purposes=settings.model_purposes,
                model_context_windows={
                    "fallback-small": 10_000,
                    "default": 1_000_000,
                },
            )
            store = ActivityStore(settings.data_dir / "activity" / "default")
            store.append_daily_entry(
                ActivityEntry(
                    timestamp="2026-06-03T02:00:00+08:00",
                    event="session_summary_checkpoint",
                    status="completed",
                    title="Memory checkpoint",
                    summary="触发维护写入一个过长 profile patch。",
                    session_id="session-budget",
                    session_title="Memory budget",
                    profile="default",
                    workspace=str(settings.workspace),
                    summary_id="summary-budget",
                )
            )
            long_content = "用户偏好" + "非常详细" * 80
            provider = StubProvider(
                (
                    '{"profile_patch":{"operations":['
                    f'{{"action":"write_user","content":"{long_content}"}}'
                    ']},'
                    '"monthly_summary":{"summary":"预算测试。"}}'
                )
            )

            result = run_daily_memory_maintenance(
                provider=provider,
                settings=settings,
                fallback_model="fallback-small",
                session_store=SessionStore.in_directory(root / "sessions"),
                trace_recorder=TraceRecorder(ListTraceSink()),
                local_date="2026-06-03",
            )

            self.assertTrue(result.ran)
            self.assertEqual(result.reason, "completed")
            self.assertFalse(result.profile_changed)
            self.assertNotIn(
                long_content,
                (settings.global_profile_dir() / "user.md").read_text(encoding="utf-8"),
            )

    def test_activity_day_can_write_project_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = _settings(root)
            store = ActivityStore(settings.data_dir / "activity" / "default")
            store.append_daily_entry(
                ActivityEntry(
                    timestamp="2026-06-03T02:00:00+08:00",
                    event="session_summary_checkpoint",
                    status="completed",
                    title="Project convention",
                    summary="用户确认本项目统一使用 pnpm。",
                    session_id="session-project",
                    session_title="Project memory",
                    profile="default",
                    workspace=str(settings.workspace),
                    summary_id="summary-project",
                )
            )
            provider = StubProvider(
                (
                    '{"profile_patch":{"operations":['
                    '{"action":"write_project_memory","content":"本项目统一使用 pnpm。"}'
                    ']},'
                    '"monthly_summary":{"summary":"确认项目包管理器约定。"}}'
                )
            )

            result = run_daily_memory_maintenance(
                provider=provider,
                settings=settings,
                fallback_model="fallback",
                session_store=SessionStore.in_directory(root / "sessions"),
                trace_recorder=TraceRecorder(ListTraceSink()),
                local_date="2026-06-03",
            )

            self.assertTrue(result.ran)
            self.assertTrue(result.profile_changed)
            self.assertEqual(
                (settings.global_profile_dir() / "memory.md").read_text(
                    encoding="utf-8"
                ),
                "- 保持回答克制。\n",
            )
            self.assertIn(
                "- 本项目统一使用 pnpm。",
                (settings.project_profile_dir() / "memory.md").read_text(
                    encoding="utf-8"
                ),
            )

    def test_incremental_run_processes_previous_activity_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = _settings(root)
            store = ActivityStore(settings.data_dir / "activity" / "default")
            store.append_daily_entry(
                ActivityEntry(
                    timestamp="2026-06-03T10:00:00+00:00",
                    event="session_summary_checkpoint",
                    status="completed",
                    title="Memory checkpoint",
                    summary="昨天确认了 interval cursor 策略。",
                    session_id="session-1",
                    session_title="Memory interval cursor",
                    profile="default",
                    workspace=str(settings.workspace),
                    summary_id="summary-1",
                )
            )
            provider = StubProvider(
                (
                    '{"profile_patch":{"operations":[]},'
                    '"monthly_summary":{"summary":"确认 maintenance 使用 interval cursor。"}}'
                )
            )
            sink = ListTraceSink()

            result = run_daily_memory_maintenance(
                provider=provider,
                settings=settings,
                fallback_model="fallback",
                session_store=SessionStore.in_directory(root / "sessions"),
                trace_recorder=TraceRecorder(sink),
                now=datetime(2026, 6, 4, 2, 0, tzinfo=LOCAL_TZ),
            )

            self.assertTrue(result.ran)
            self.assertEqual(result.reason, "completed")
            self.assertEqual(result.date, "2026-06-03")
            self.assertEqual(result.window_end, "2026-06-04T02:00:00+08:00")
            self.assertEqual(len(provider.requests), 1)
            state = json.loads(
                (
                    settings.data_dir
                    / "memory"
                    / "default"
                    / "maintenance_state.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                state["last_successful_run_at"], "2026-06-04T02:00:00+08:00"
            )
            self.assertEqual(state["last_daily_maintenance_date"], "2026-06-03")

    def test_incremental_run_catches_up_multiple_completed_activity_days(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = _settings(root)
            state_path = (
                settings.data_dir / "memory" / "default" / "maintenance_state.json"
            )
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps(
                    {
                        "last_successful_run_at": "2026-06-01T02:00:00+08:00",
                        "last_status": "completed",
                    }
                ),
                encoding="utf-8",
            )
            store = ActivityStore(settings.data_dir / "activity" / "default")
            for local_date in ("2026-06-01", "2026-06-02", "2026-06-03"):
                store.append_daily_entry(
                    ActivityEntry(
                        timestamp=f"{local_date}T10:00:00+08:00",
                        event="session_summary_checkpoint",
                        status="completed",
                        title="Memory checkpoint",
                        summary=f"{local_date} 有用户交互。",
                        session_id=f"session-{local_date}",
                        session_title="Memory catch-up",
                        profile="default",
                        workspace=str(settings.workspace),
                        summary_id=f"summary-{local_date}",
                    )
                )
            provider = StubProvider(
                (
                    '{"profile_patch":{"operations":[]},'
                    '"monthly_summary":{"summary":"补跑维护窗口。"}}'
                )
            )
            sink = ListTraceSink()

            result = run_daily_memory_maintenance(
                provider=provider,
                settings=settings,
                fallback_model="fallback",
                session_store=SessionStore.in_directory(root / "sessions"),
                trace_recorder=TraceRecorder(sink),
                now=datetime(2026, 6, 4, 2, 0, tzinfo=LOCAL_TZ),
            )

            self.assertTrue(result.ran)
            self.assertEqual(result.reason, "completed")
            self.assertEqual(
                result.date,
                "2026-06-01,2026-06-02,2026-06-03",
            )
            self.assertEqual(len(provider.requests), 3)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(
                state["last_successful_run_at"], "2026-06-04T02:00:00+08:00"
            )
            self.assertEqual(state["last_daily_maintenance_date"], "2026-06-03")
            monthly = store.monthly_summary_path("2026-06").read_text(encoding="utf-8")
            self.assertIn("## 2026-06-01", monthly)
            self.assertIn("## 2026-06-02", monthly)
            self.assertIn("## 2026-06-03", monthly)

    def test_incremental_pending_only_window_processes_pending_and_advances_cursor(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = _settings(root)
            session_store = SessionStore.in_directory(root / "sessions")
            profile = ProfileRef(name="default", uri="profiles/default")
            session = session_store.create(
                workspace=settings.workspace,
                profile=profile,
                title="pending maintenance",
            )
            transcript = session_store.transcript_store(session)
            record = transcript.append_item(
                ModelConversationItem.from_message(
                    Message(role=MessageRole.USER, content="以后请用中文直接回答。")
                )
            )
            self.assertIsNotNone(record)
            pending = record_curator_pending_checkpoint(
                settings=settings,
                profile_name=profile.name,
                profile_uri=profile.uri,
                session_id=session.session_id,
                summary_id="summary_pending",
                source_sequences=(record.sequence,),
            )
            self.assertIsNotNone(pending)
            provider = StubProvider(
                '{"operations":[{"action":"write_user","content":"用户偏好中文直接回答。"}]}'
            )
            sink = ListTraceSink()

            result = run_daily_memory_maintenance(
                provider=provider,
                settings=settings,
                fallback_model="fallback",
                session_store=session_store,
                trace_recorder=TraceRecorder(sink),
                now=datetime(2026, 6, 4, 2, 0, tzinfo=LOCAL_TZ),
            )

            self.assertTrue(result.ran)
            self.assertEqual(result.reason, "completed")
            self.assertEqual(result.date, "")
            self.assertEqual(result.pending_processed, 1)
            self.assertEqual(result.pending_failed, 0)
            self.assertEqual(len(provider.requests), 1)
            self.assertIn(
                "- 用户偏好中文直接回答。",
                (settings.global_profile_dir() / "user.md").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                curator_pending_store(settings.data_dir, "default").load_pending(),
                (),
            )
            state = json.loads(
                (
                    settings.data_dir
                    / "memory"
                    / "default"
                    / "maintenance_state.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                state["last_successful_run_at"], "2026-06-04T02:00:00+08:00"
            )

    def test_daily_failure_does_not_consume_pending_curator_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = _settings(root)
            session_store = SessionStore.in_directory(root / "sessions")
            profile = ProfileRef(name="default", uri="profiles/default")
            session = session_store.create(
                workspace=settings.workspace,
                profile=profile,
                title="pending with daily failure",
            )
            transcript = session_store.transcript_store(session)
            record = transcript.append_item(
                ModelConversationItem.from_message(
                    Message(role=MessageRole.USER, content="以后请用中文直接回答。")
                )
            )
            self.assertIsNotNone(record)
            pending = record_curator_pending_checkpoint(
                settings=settings,
                profile_name=profile.name,
                profile_uri=profile.uri,
                session_id=session.session_id,
                summary_id="summary_pending",
                source_sequences=(record.sequence,),
            )
            self.assertIsNotNone(pending)
            store = ActivityStore(settings.data_dir / "activity" / "default")
            store.append_daily_entry(
                ActivityEntry(
                    timestamp="2026-06-03T02:00:00+08:00",
                    event="session_summary_checkpoint",
                    status="completed",
                    title="Memory checkpoint",
                    summary="这次 daily maintenance 会失败。",
                    session_id="session-daily-fail",
                    session_title="Daily failure",
                    profile="default",
                    workspace=str(settings.workspace),
                    summary_id="summary-daily-fail",
                )
            )
            provider = StubProvider("not json")

            result = run_daily_memory_maintenance(
                provider=provider,
                settings=settings,
                fallback_model="fallback",
                session_store=session_store,
                trace_recorder=TraceRecorder(ListTraceSink()),
                local_date="2026-06-03",
            )

            self.assertTrue(result.ran)
            self.assertEqual(result.reason, "failed")
            self.assertEqual(result.pending_processed, 0)
            self.assertEqual(len(provider.requests), 1)
            self.assertEqual(
                len(curator_pending_store(settings.data_dir, "default").load_pending()),
                1,
            )
            self.assertNotIn(
                "用户偏好中文直接回答。",
                (settings.global_profile_dir() / "user.md").read_text(encoding="utf-8"),
            )

    def test_incremental_empty_window_advances_cursor_without_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = _settings(root)
            provider = StubProvider("{}")
            sink = ListTraceSink()

            result = run_daily_memory_maintenance(
                provider=provider,
                settings=settings,
                fallback_model="fallback",
                session_store=SessionStore.in_directory(root / "sessions"),
                trace_recorder=TraceRecorder(sink),
                now=datetime(2026, 6, 4, 2, 0, tzinfo=LOCAL_TZ),
            )

            self.assertFalse(result.ran)
            self.assertEqual(result.reason, "no_user_activity")
            self.assertEqual(len(provider.requests), 0)
            state = json.loads(
                (
                    settings.data_dir
                    / "memory"
                    / "default"
                    / "maintenance_state.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                state["last_successful_run_at"], "2026-06-04T02:00:00+08:00"
            )
            self.assertEqual(state["last_status"], "completed")

    def test_incremental_failure_keeps_previous_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = _settings(root)
            state_path = (
                settings.data_dir / "memory" / "default" / "maintenance_state.json"
            )
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps(
                    {
                        "last_successful_run_at": "2026-06-03T02:00:00+00:00",
                        "last_status": "completed",
                    }
                ),
                encoding="utf-8",
            )
            store = ActivityStore(settings.data_dir / "activity" / "default")
            store.append_daily_entry(
                ActivityEntry(
                    timestamp="2026-06-03T10:00:00+00:00",
                    event="session_summary_checkpoint",
                    status="completed",
                    title="Memory checkpoint",
                    summary="这次模型会返回非法 JSON。",
                    session_id="session-1",
                    session_title="Memory interval cursor",
                    profile="default",
                    workspace=str(settings.workspace),
                    summary_id="summary-1",
                )
            )
            provider = StubProvider("not json")
            sink = ListTraceSink()

            result = run_daily_memory_maintenance(
                provider=provider,
                settings=settings,
                fallback_model="fallback",
                session_store=SessionStore.in_directory(root / "sessions"),
                trace_recorder=TraceRecorder(sink),
                now=datetime(2026, 6, 4, 2, 0, tzinfo=LOCAL_TZ),
            )

            self.assertTrue(result.ran)
            self.assertEqual(result.reason, "failed")
            self.assertEqual(len(provider.requests), 1)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(
                state["last_successful_run_at"], "2026-06-03T02:00:00+00:00"
            )
            self.assertEqual(state["last_status"], "failed")


if __name__ == "__main__":
    unittest.main()
