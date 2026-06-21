from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from deepmate.app import AppSettings, ContextSettings, ModelPurposeSettings
from deepmate.domain import Message, MessageRole, ProfileRef
from deepmate.memory import (
    CuratorPendingRecord,
    curate_memory_patch,
    curator_pending_store,
    record_curator_pending_checkpoint,
    run_due_curator_maintenance,
    should_run_curator,
)
from deepmate.providers import ModelConversationItem, ModelResponse
from deepmate.storage import SessionStore
from deepmate.trace import TraceRecorder


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


class MemoryCuratorTests(unittest.TestCase):
    def test_curate_memory_patch_delegates_semantic_skip_to_model(self) -> None:
        provider = StubProvider('{"operations":[{"action":"skip","reason":"low value"}]}')

        result = curate_memory_patch(
            provider=provider,
            model="deepseek-v4-flash",
            source_text="hello",
            profile_dir=Path("/does/not/matter"),
        )

        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(result.operation_count(), 1)
        self.assertEqual(result.patch.operations[0].action, "skip")

    def test_curate_memory_patch_skips_empty_source_without_model_call(self) -> None:
        provider = StubProvider('{"operations":[]}')

        result = curate_memory_patch(
            provider=provider,
            model="deepseek-v4-flash",
            source_text="  ",
            profile_dir=Path("/does/not/matter"),
        )

        self.assertEqual(len(provider.requests), 0)
        self.assertEqual(result.operation_count(), 1)
        self.assertEqual(result.patch.operations[0].reason, "empty_source")

    def test_should_run_curator_skips_empty_and_cooldown_records(self) -> None:
        now = datetime(2026, 6, 3, 2, 30, tzinfo=UTC)

        should_run, reason = should_run_curator((), now)
        self.assertFalse(should_run)
        self.assertEqual(reason, "no_pending_user_activity")

        record = CuratorPendingRecord.create(
            profile_name="default",
            profile_uri="profiles/default",
            session_id="session",
            summary_id="summary",
            source_sequences=(1,),
            created_at=datetime(2026, 6, 3, 2, 10, tzinfo=UTC).isoformat(),
        )

        should_run, reason = should_run_curator((record,), now)
        self.assertFalse(should_run)
        self.assertEqual(reason, "cooldown_pending")

    def test_should_run_curator_after_maintenance_window(self) -> None:
        record = CuratorPendingRecord.create(
            profile_name="default",
            profile_uri="profiles/default",
            session_id="session",
            summary_id="summary",
            source_sequences=(1,),
            created_at=datetime(2026, 6, 3, 1, 0, tzinfo=UTC).isoformat(),
        )

        should_run, reason = should_run_curator(
            (record,),
            datetime(2026, 6, 3, 2, 30, tzinfo=UTC),
        )

        self.assertTrue(should_run)
        self.assertEqual(reason, "maintenance_window")

    def test_run_due_curator_applies_patch_and_clears_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            global_profile_dir = root / "home" / "profiles" / "default"
            profile_dir = workspace / "profiles" / "default"
            global_profile_dir.mkdir(parents=True)
            profile_dir.mkdir(parents=True)
            (global_profile_dir / "user.md").write_text("", encoding="utf-8")
            (global_profile_dir / "memory.md").write_text("", encoding="utf-8")
            (profile_dir / "memory.md").write_text("", encoding="utf-8")
            session_store = SessionStore.in_directory(root / "sessions")
            profile = ProfileRef(
                name="default",
                uri="profiles/default",
                global_uri=str(global_profile_dir),
                project_uri="profiles/default",
            )
            session = session_store.create(
                workspace=workspace,
                profile=profile,
                title="curator test",
            )
            transcript = session_store.transcript_store(session)
            record = transcript.append_item(
                ModelConversationItem.from_message(
                    Message(role=MessageRole.USER, content="以后请用中文直接回答。")
                )
            )
            self.assertIsNotNone(record)
            settings = AppSettings(
                workspace=workspace,
                data_dir=root / "var",
                deepmate_home=root / "home",
                active_profile="default",
                trace_sink=root / "trace.jsonl",
                default_provider="deepseek",
                context=ContextSettings(
                    response_token_reserve=0,
                    safety_margin_tokens=0,
                ),
                model_purposes={
                    "memory": ModelPurposeSettings(
                        model="deepseek-v4-flash",
                        thinking="disabled",
                    )
                },
            )
            pending = record_curator_pending_checkpoint(
                settings=settings,
                profile_name=profile.name,
                profile_uri=profile.uri,
                session_id=session.session_id,
                summary_id="summary_1",
                source_sequences=(record.sequence,),
            )
            self.assertIsNotNone(pending)
            provider = StubProvider(
                '{"operations":[{"action":"write_user","content":"用户偏好中文直接回答。","confidence":0.9}]}'
            )
            sink = ListTraceSink()

            changed = run_due_curator_maintenance(
                provider=provider,
                settings=settings,
                fallback_model="fallback",
                session_store=session_store,
                trace_recorder=TraceRecorder(sink),
                now=datetime(2026, 6, 3, 3, 0, tzinfo=UTC),
                force=True,
            )

            self.assertTrue(changed)
            self.assertEqual(len(provider.requests), 1)
            self.assertIn(
                "- 用户偏好中文直接回答。",
                (global_profile_dir / "user.md").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                curator_pending_store(settings.data_dir, "default").load_pending(),
                (),
            )
            self.assertIn(
                "memory_curator_completed",
                tuple(event.kind for event in sink.events),
            )

    def test_run_due_curator_records_idle_skip_without_pending_store(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = AppSettings(
                workspace=root / "workspace",
                data_dir=root / "var",
                active_profile="default",
                trace_sink=root / "trace.jsonl",
                default_provider="deepseek",
            )
            sink = ListTraceSink()

            changed = run_due_curator_maintenance(
                provider=StubProvider("{}"),
                settings=settings,
                fallback_model="fallback",
                session_store=SessionStore.in_directory(root / "sessions"),
                trace_recorder=TraceRecorder(sink),
                now=datetime(2026, 6, 3, 3, 0, tzinfo=UTC),
            )

            self.assertFalse(changed)
            self.assertEqual(len(sink.events), 1)
            self.assertEqual(sink.events[0].kind, "memory_curator_skipped")
            self.assertIn("reason=no_pending_user_activity", sink.events[0].refs)

    def test_budget_blocked_patch_clears_pending_without_marking_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            global_profile_dir = root / "home" / "profiles" / "default"
            profile_dir = workspace / "profiles" / "default"
            global_profile_dir.mkdir(parents=True)
            profile_dir.mkdir(parents=True)
            (global_profile_dir / "user.md").write_text("", encoding="utf-8")
            (global_profile_dir / "memory.md").write_text("", encoding="utf-8")
            (profile_dir / "memory.md").write_text("", encoding="utf-8")
            session_store = SessionStore.in_directory(root / "sessions")
            profile = ProfileRef(
                name="default",
                uri="profiles/default",
                global_uri=str(global_profile_dir),
                project_uri="profiles/default",
            )
            session = session_store.create(
                workspace=workspace,
                profile=profile,
                title="curator budget test",
            )
            transcript = session_store.transcript_store(session)
            record = transcript.append_item(
                ModelConversationItem.from_message(
                    Message(role=MessageRole.USER, content="以后请保持输出非常长。")
                )
            )
            self.assertIsNotNone(record)
            settings = AppSettings(
                workspace=workspace,
                data_dir=root / "var",
                deepmate_home=root / "home",
                active_profile="default",
                trace_sink=root / "trace.jsonl",
                default_provider="deepseek",
                context=ContextSettings(
                    hot_profile_min_tokens=1,
                    hot_profile_max_tokens=1,
                    response_token_reserve=0,
                    safety_margin_tokens=0,
                ),
                model_purposes={
                    "memory": ModelPurposeSettings(model="deepseek-v4-flash")
                },
            )
            pending = record_curator_pending_checkpoint(
                settings=settings,
                profile_name=profile.name,
                profile_uri=profile.uri,
                session_id=session.session_id,
                summary_id="summary_budget",
                source_sequences=(record.sequence,),
            )
            self.assertIsNotNone(pending)
            provider = StubProvider(
                '{"operations":[{"action":"write_user","content":"用户希望所有回答都保持非常非常非常长的输出。"}]}'
            )
            sink = ListTraceSink()

            changed = run_due_curator_maintenance(
                provider=provider,
                settings=settings,
                fallback_model="fallback",
                session_store=session_store,
                trace_recorder=TraceRecorder(sink),
                now=datetime(2026, 6, 3, 3, 0, tzinfo=UTC),
                force=True,
            )

            self.assertFalse(changed)
            pending_records = curator_pending_store(
                settings.data_dir,
                "default",
            ).load_pending()
            self.assertEqual(len(pending_records), 0)
            self.assertIn(
                "memory_curator_skipped",
                tuple(event.kind for event in sink.events),
            )

    def test_curator_skips_existing_over_budget_profile_without_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            global_profile_dir = root / "home" / "profiles" / "default"
            profile_dir = workspace / "profiles" / "default"
            global_profile_dir.mkdir(parents=True)
            profile_dir.mkdir(parents=True)
            (global_profile_dir / "user.md").write_text("- " + ("偏好" * 50), encoding="utf-8")
            (global_profile_dir / "memory.md").write_text("", encoding="utf-8")
            (profile_dir / "memory.md").write_text("", encoding="utf-8")
            session_store = SessionStore.in_directory(root / "sessions")
            profile = ProfileRef(
                name="default",
                uri="profiles/default",
                global_uri=str(global_profile_dir),
                project_uri="profiles/default",
            )
            session = session_store.create(
                workspace=workspace,
                profile=profile,
                title="curator preflight budget test",
            )
            transcript = session_store.transcript_store(session)
            record = transcript.append_item(
                ModelConversationItem.from_message(
                    Message(role=MessageRole.USER, content="以后请用中文回答。")
                )
            )
            self.assertIsNotNone(record)
            settings = AppSettings(
                workspace=workspace,
                data_dir=root / "var",
                deepmate_home=root / "home",
                active_profile="default",
                trace_sink=root / "trace.jsonl",
                default_provider="deepseek",
                context=ContextSettings(
                    hot_profile_min_tokens=1,
                    hot_profile_max_tokens=1,
                    response_token_reserve=0,
                    safety_margin_tokens=0,
                ),
                model_purposes={
                    "memory": ModelPurposeSettings(model="deepseek-v4-flash")
                },
            )
            pending = record_curator_pending_checkpoint(
                settings=settings,
                profile_name=profile.name,
                profile_uri=profile.uri,
                session_id=session.session_id,
                summary_id="summary_budget",
                source_sequences=(record.sequence,),
            )
            self.assertIsNotNone(pending)
            provider = StubProvider('{"operations":[{"action":"write_user","content":"x"}]}')
            sink = ListTraceSink()

            changed = run_due_curator_maintenance(
                provider=provider,
                settings=settings,
                fallback_model="fallback",
                session_store=session_store,
                trace_recorder=TraceRecorder(sink),
                now=datetime(2026, 6, 3, 3, 0, tzinfo=UTC),
                force=True,
            )

            self.assertFalse(changed)
            self.assertEqual(len(provider.requests), 0)
            self.assertEqual(
                curator_pending_store(settings.data_dir, "default").load_pending(),
                (),
            )

    def test_run_due_curator_can_write_project_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            global_profile_dir = root / "home" / "profiles" / "default"
            project_profile_dir = workspace / "profiles" / "default"
            global_profile_dir.mkdir(parents=True)
            project_profile_dir.mkdir(parents=True)
            (global_profile_dir / "user.md").write_text("", encoding="utf-8")
            (global_profile_dir / "memory.md").write_text("", encoding="utf-8")
            (project_profile_dir / "memory.md").write_text("", encoding="utf-8")
            session_store = SessionStore.in_directory(root / "sessions")
            profile = ProfileRef(
                name="default",
                uri="profiles/default",
                global_uri=str(global_profile_dir),
                project_uri="profiles/default",
            )
            session = session_store.create(
                workspace=workspace,
                profile=profile,
                title="project memory",
            )
            transcript = session_store.transcript_store(session)
            record = transcript.append_item(
                ModelConversationItem.from_message(
                    Message(role=MessageRole.USER, content="这个项目统一使用 pnpm。")
                )
            )
            self.assertIsNotNone(record)
            settings = AppSettings(
                workspace=workspace,
                data_dir=root / "var",
                deepmate_home=root / "home",
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
            record_curator_pending_checkpoint(
                settings=settings,
                profile_name=profile.name,
                profile_uri=profile.uri,
                session_id=session.session_id,
                summary_id="summary_project",
                source_sequences=(record.sequence,),
            )
            provider = StubProvider(
                '{"operations":[{"action":"write_project_memory","content":"项目统一使用 pnpm。"}]}'
            )

            changed = run_due_curator_maintenance(
                provider=provider,
                settings=settings,
                fallback_model="fallback",
                session_store=session_store,
                trace_recorder=TraceRecorder(ListTraceSink()),
                now=datetime(2026, 6, 3, 3, 0, tzinfo=UTC),
                force=True,
            )

            self.assertTrue(changed)
            self.assertEqual(
                (global_profile_dir / "memory.md").read_text(encoding="utf-8"),
                "",
            )
            self.assertIn(
                "- 项目统一使用 pnpm。",
                (project_profile_dir / "memory.md").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
