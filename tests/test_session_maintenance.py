from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from deepmate.channels.session_maintenance import (
    _apply_memory_patch_after_checkpoint,
    _memory_source_sequences_from_transcript_records,
    _memory_source_sequences_from_summary_input,
    _memory_source_text_from_summary_input,
    run_session_maintenance,
    write_session_end_activity,
)
from deepmate.app import AppSettings, ContextSettings
from deepmate.domain import Message, MessageRole
from deepmate.memory import (
    MemoryPatch,
    MemoryPatchOperation,
    curator_pending_store,
)
from deepmate.providers import (
    ModelConversationItem,
    ModelResponse,
    ModelToolExchange,
    ModelToolRequest,
    ModelToolResult,
)
from deepmate.runtime import (
    AgentStepResult,
    ConversationBudgetPolicy,
    HookAction,
    HookActionType,
    HookDefinition,
    HookEvent,
    HookLayer,
    HookRegistry,
    HookRunTarget,
    HookRuntimeContext,
    HookSignalStore,
    SessionSummaryInput,
    SessionSummarySourceItem,
    UserTurnResult,
    start_session_runtime,
    start_runtime_activation,
)
from deepmate.storage import SessionSummaryRecord, SessionStore, ToolOutputStore
from deepmate.storage.session_store import TranscriptRecord
from deepmate.trace import TraceRecorder


class ListTraceSink:
    def __init__(self) -> None:
        self.events = []

    def write(self, event) -> None:
        self.events.append(event)


def _transcript_record(
    *,
    sequence: int,
    role: MessageRole,
    content: str,
) -> TranscriptRecord:
    return TranscriptRecord.from_item(
        session_id="session-test",
        sequence=sequence,
        item=ModelConversationItem.from_message(Message(role=role, content=content)),
    )


class SessionMaintenanceTests(unittest.TestCase):
    def test_checkpoint_memory_source_uses_user_authored_segments(self) -> None:
        summary_input = SessionSummaryInput(
            source_items=(
                SessionSummarySourceItem(
                    sequence=1,
                    item=ModelConversationItem.from_message(
                        Message(role=MessageRole.USER, content="我是小学生。")
                    ),
                ),
                SessionSummarySourceItem(
                    sequence=2,
                    item=ModelConversationItem.from_message(
                        Message(role=MessageRole.ASSISTANT, content="好的。")
                    ),
                ),
                SessionSummarySourceItem(
                    sequence=3,
                    item=ModelConversationItem.from_message(
                        Message(role=MessageRole.USER, content="以后请用中文回答。")
                    ),
                ),
            ),
        )

        source_text = _memory_source_text_from_summary_input(summary_input)

        self.assertIn("### User transcript item 1", source_text)
        self.assertIn("我是小学生。", source_text)
        self.assertIn("### User transcript item 3", source_text)
        self.assertIn("以后请用中文回答。", source_text)
        self.assertNotIn("好的。", source_text)

        self.assertEqual(
            _memory_source_sequences_from_summary_input(summary_input),
            (1, 3),
        )

    def test_session_end_records_pending_memory_without_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            profile_dir = workspace / "profiles" / "default"
            profile_dir.mkdir(parents=True)
            settings = AppSettings(
                workspace=workspace,
                data_dir=root / "var",
                deepmate_home=root / "home",
                active_profile="default",
                trace_sink=root / "trace.jsonl",
                default_provider="deepseek",
            )
            session_store = SessionStore.in_directory(settings.data_dir / "sessions")
            session = session_store.create(
                workspace=workspace,
                profile=settings.profile_ref(),
                title="short memory session",
            )
            transcript = session_store.transcript_store(session)
            transcript.append_item(
                ModelConversationItem.from_message(
                    Message(role=MessageRole.USER, content="以后请优先用中文直接回答。")
                )
            )
            transcript.append_item(
                ModelConversationItem.from_message(
                    Message(role=MessageRole.ASSISTANT, content="好的。")
                )
            )
            activation = start_runtime_activation(
                session_id=session.session_id,
                workspace=workspace,
                profile=session.profile,
            )
            runtime = start_session_runtime(activation)
            sink = ListTraceSink()

            write_session_end_activity(
                settings=settings,
                session_store=session_store,
                session=session,
                transcript=transcript,
                runtime=runtime,
                trace_recorder=TraceRecorder(sink),
                event="session_end",
                status="completed",
                summary="done",
            )
            write_session_end_activity(
                settings=settings,
                session_store=session_store,
                session=session,
                transcript=transcript,
                runtime=runtime,
                trace_recorder=TraceRecorder(sink),
                event="session_end",
                status="completed",
                summary="done",
            )

            pending = curator_pending_store(settings.data_dir, "default").load_pending()
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0].session_id, session.session_id)
            self.assertEqual(pending[0].summary_id, "session_end")
            self.assertEqual(pending[0].source_sequences, (1,))
            self.assertEqual(
                tuple(event.kind for event in sink.events).count(
                    "memory_curator_pending_recorded"
                ),
                1,
            )

    def test_session_end_memory_pending_skips_code_and_sensitive_text(self) -> None:
        records = (
            _transcript_record(
                sequence=1,
                role=MessageRole.USER,
                content="```python\nprint('hello')\n```",
            ),
            _transcript_record(
                sequence=2,
                role=MessageRole.USER,
                content="api key is sk-abcdefghijklmnopqrstuvwxyz",
            ),
            _transcript_record(
                sequence=3,
                role=MessageRole.USER,
                content="以后请把回答写得更精简。",
            ),
            _transcript_record(
                sequence=4,
                role=MessageRole.ASSISTANT,
                content="好的。",
            ),
        )

        self.assertEqual(
            _memory_source_sequences_from_transcript_records(records),
            (3,),
        )

    def test_checkpoint_memory_patch_writes_profile_without_refreshing_activation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            deepmate_home = root / "home"
            global_profile_dir = deepmate_home / "profiles" / "default"
            project_profile_dir = workspace / "profiles" / "default"
            global_profile_dir.mkdir(parents=True)
            project_profile_dir.mkdir(parents=True)
            (workspace / "AGENTS.md").write_text("# Rules\n", encoding="utf-8")
            (global_profile_dir / "identity.md").write_text(
                "# Identity\n",
                encoding="utf-8",
            )
            (global_profile_dir / "soul.md").write_text("# Soul\n", encoding="utf-8")
            (global_profile_dir / "user.md").write_text("", encoding="utf-8")
            (global_profile_dir / "memory.md").write_text("", encoding="utf-8")
            (project_profile_dir / "memory.md").write_text("", encoding="utf-8")
            settings = AppSettings(
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
            )
            session_store = SessionStore.in_directory(root / "sessions")
            session = session_store.create(
                workspace=workspace,
                profile=settings.profile_ref(),
                title="memory patch test",
            )
            activation = start_runtime_activation(
                session_id=session.session_id,
                workspace=workspace,
                profile=session.profile,
            )
            runtime = start_session_runtime(activation)
            summary_input = SessionSummaryInput(
                source_items=(
                    SessionSummarySourceItem(
                        sequence=1,
                        item=ModelConversationItem.from_message(
                            Message(role=MessageRole.USER, content="以后请用中文直接回答。")
                        ),
                    ),
                )
            )
            summary_record = SessionSummaryRecord.create(
                session_id=session.session_id,
                content="## Session Summary\n\n### User Goal\n测试 memory patch。",
                covered_until_sequence=1,
                covered_item_count=1,
                source_item_count=1,
                estimated_source_tokens=10,
                source_model="deepseek-v4-pro",
            )
            sink = ListTraceSink()
            signal_store = HookSignalStore(root / "var" / "hooks" / "signals.jsonl")

            result = _apply_memory_patch_after_checkpoint(
                settings=settings,
                session=session,
                runtime=runtime,
                summary_input=summary_input,
                summary_record=summary_record,
                memory_patch=MemoryPatch(
                    operations=(
                        MemoryPatchOperation(
                            action="write_user",
                            content="用户偏好中文直接回答。",
                        ),
                        MemoryPatchOperation(
                            action="write_project_memory",
                            content="本项目统一使用 pnpm。",
                        ),
                    )
                ),
                trace_recorder=TraceRecorder(sink),
                hook_context=HookRuntimeContext.from_registry(
                    HookRegistry.from_hooks(
                        (
                            HookDefinition(
                                hook_id="memory-patch-signal",
                                event_name=HookEvent.MEMORY_PATCH_APPLIED,
                                layer=HookLayer.SESSION,
                                run_on=HookRunTarget.MAINTENANCE,
                                actions=(
                                    HookAction(
                                        HookActionType.RECORD_MEMORY_SIGNAL,
                                        {
                                            "signal_kind": "checkpoint_memory_patch",
                                            "summary": "Checkpoint memory patch was applied.",
                                        },
                                    ),
                                ),
                            ),
                        )
                    ),
                    signal_store=signal_store,
                ),
            )
            signals = signal_store.load_recent()

            self.assertEqual(result.status, "applied")
            self.assertIn(
                "- 用户偏好中文直接回答。",
                (global_profile_dir / "user.md").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (global_profile_dir / "memory.md").read_text(encoding="utf-8"),
                "",
            )
            self.assertIn(
                "- 本项目统一使用 pnpm。",
                (project_profile_dir / "memory.md").read_text(encoding="utf-8"),
            )
            self.assertEqual(runtime.activation.context_epoch, 1)
            self.assertIn("memory_patch_applied", tuple(event.kind for event in sink.events))
            self.assertEqual(len(signals), 1)
            self.assertEqual(signals[0].hook_id, "memory-patch-signal")
            self.assertEqual(signals[0].event_name, HookEvent.MEMORY_PATCH_APPLIED.value)
            self.assertNotIn(
                "context_snapshot_refreshed",
                tuple(event.kind for event in sink.events),
            )

    def test_session_maintenance_prunes_unreferenced_tool_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            profile_dir = workspace / "profiles" / "default"
            profile_dir.mkdir(parents=True)
            (workspace / "AGENTS.md").write_text("# Rules\n", encoding="utf-8")
            settings = AppSettings(
                workspace=workspace,
                data_dir=root / "var",
                deepmate_home=root / "home",
                active_profile="default",
                trace_sink=root / "trace.jsonl",
                default_provider="deepseek",
            )
            session_store = SessionStore.in_directory(settings.data_dir / "sessions")
            session = session_store.create(
                workspace=workspace,
                profile=settings.profile_ref(),
                title="tool output cleanup",
            )
            transcript = session_store.transcript_store(session)
            output_store = ToolOutputStore.in_data_dir(
                settings.data_dir,
                session.profile.name,
                session.session_id,
            )
            kept = output_store.save(
                tool_name="run_tests",
                tool_source="native",
                content_kind="log",
                content="kept output",
                estimated_tokens=1,
                request_id="kept",
            )
            stale = output_store.save(
                tool_name="run_tests",
                tool_source="native",
                content_kind="log",
                content="stale output",
                estimated_tokens=1,
                request_id="stale",
            )
            transcript.append_item(
                ModelConversationItem.from_tool_exchange(
                    ModelToolExchange(
                        assistant_content="running tests",
                        tool_requests=(
                            ModelToolRequest(name="run_tests", id="call_1"),
                        ),
                        tool_results=(
                            ModelToolResult(
                                name="run_tests",
                                request_id="call_1",
                                content="compacted",
                                refs=(f"tool_output_ref={kept.ref}",),
                            ),
                        ),
                    )
                )
            )
            activation = start_runtime_activation(
                session_id=session.session_id,
                workspace=workspace,
                profile=session.profile,
            )
            runtime = start_session_runtime(activation)
            sink = ListTraceSink()

            updated = run_session_maintenance(
                provider=None,
                settings=settings,
                fallback_model="stub",
                prompt="",
                session_store=session_store,
                session=session,
                transcript=transcript,
                runtime=runtime,
                conversation_budget_policy=ConversationBudgetPolicy(),
                trace_recorder=TraceRecorder(sink),
            )

            self.assertIs(updated, runtime)
            self.assertIsNotNone(output_store.load(kept.ref))
            self.assertIsNone(output_store.load(stale.ref))
            self.assertIn("tool_output_pruned", tuple(event.kind for event in sink.events))

    def test_session_maintenance_records_successful_workflow_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            settings = AppSettings(
                workspace=workspace,
                data_dir=root / "var",
                deepmate_home=root / "home",
                active_profile="default",
                trace_sink=root / "trace.jsonl",
                default_provider="deepseek",
            )
            session_store = SessionStore.in_directory(settings.data_dir / "sessions")
            session = session_store.create(
                workspace=workspace,
                profile=settings.profile_ref(),
                title="workflow evidence",
            )
            transcript = session_store.transcript_store(session)
            activation = start_runtime_activation(
                session_id=session.session_id,
                workspace=workspace,
                profile=session.profile,
            )
            runtime = replace(
                start_session_runtime(activation),
                last_user_turn_result=UserTurnResult(
                    steps=(
                        AgentStepResult(
                            request=None,
                            response=ModelResponse(content="done"),
                            tool_results=(
                                ModelToolResult(
                                    name="read_text_file",
                                    request_id="call_1",
                                    content="read",
                                    refs=("path=README.md",),
                                ),
                                ModelToolResult(
                                    name="write_text_file",
                                    request_id="call_2",
                                    content="write",
                                    refs=("path=README.md",),
                                ),
                            ),
                        ),
                    ),
                ),
            )
            sink = ListTraceSink()

            updated = run_session_maintenance(
                provider=None,
                settings=settings,
                fallback_model="stub",
                prompt="",
                session_store=session_store,
                session=session,
                transcript=transcript,
                runtime=runtime,
                conversation_budget_policy=ConversationBudgetPolicy(),
                trace_recorder=TraceRecorder(sink),
            )

            self.assertIs(updated, runtime)
            workflow_events = [
                event for event in sink.events if event.kind == "workflow_success"
            ]
            self.assertEqual(len(workflow_events), 1)
            self.assertIn(
                "signature=tool workflow: read_text_file -> write_text_file",
                workflow_events[0].refs,
            )
            self.assertIn("path=README.md", workflow_events[0].refs)


if __name__ == "__main__":
    unittest.main()
