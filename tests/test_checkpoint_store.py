from __future__ import annotations

import tempfile
import threading
import unittest
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from deepmate.channels.checkpointing import (
    SessionCheckpointController,
    SessionCheckpointWriteRouter,
)
from deepmate.channels.cli import _handle_rewind_command
from deepmate.channels.session_maintenance import runtime_conversation_from_store
from deepmate.domain import Message, MessageRole, ProfileRef
from deepmate.providers import (
    ModelConversationItem,
    ModelToolExchange,
    ModelToolRequest,
    ModelToolResult,
)
from deepmate.runtime import (
    HookAction,
    HookActionType,
    HookDefinition,
    HookEvent,
    HookLayer,
    HookRegistry,
    HookRuntimeContext,
)
from deepmate.storage import (
    RESUME_HINT_AFTER_TOOL,
    RESUME_HINT_MAX_STEPS,
    RESUME_HINT_NO_RESPONSE,
    RESUME_HINT_NORMAL,
    TURN_STATUS_COMPLETED,
    SessionStore,
    TranscriptRecord,
    TurnCheckpointStore,
    WorkspaceCheckpointStore,
)
from deepmate.tasks import TaskStore
from deepmate.tools import workspace_filesystem_tools


class CheckpointStoreTests(unittest.TestCase):
    def test_turn_checkpoint_tracks_transcript_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TurnCheckpointStore.in_data_dir(tmp, "default", "session")
            turn = store.start_turn()
            self.assertEqual(turn.turn_id, "turn_00001")
            self.assertEqual(turn.resume_hint, RESUME_HINT_NO_RESPONSE)

            user = _message_record("session", 1, MessageRole.USER, "please edit")
            turn = store.record_transcript_item(turn.turn_id, user)
            self.assertEqual(turn.user_sequence, 1)
            self.assertEqual(turn.resume_hint, RESUME_HINT_NO_RESPONSE)

            exchange = TranscriptRecord.from_item(
                session_id="session",
                sequence=2,
                item=ModelConversationItem.from_tool_exchange(
                    ModelToolExchange(
                        tool_requests=(
                            ModelToolRequest(
                                id="call_1",
                                name="read_text_file",
                                arguments={"path": "README.md"},
                            ),
                        ),
                        tool_results=(
                            ModelToolResult(
                                request_id="call_1",
                                name="read_text_file",
                                content="ok",
                            ),
                        ),
                    )
                ),
            )
            turn = store.record_transcript_item(turn.turn_id, exchange)
            self.assertEqual(turn.last_tool_exchange_sequence, 2)
            self.assertEqual(turn.resume_hint, RESUME_HINT_AFTER_TOOL)

            assistant = _message_record("session", 3, MessageRole.ASSISTANT, "done")
            turn = store.record_transcript_item(turn.turn_id, assistant)
            self.assertEqual(turn.final_assistant_sequence, 3)
            self.assertEqual(turn.resume_hint, RESUME_HINT_NORMAL)

            completed = store.complete_turn(turn.turn_id)
            self.assertEqual(completed.status, TURN_STATUS_COMPLETED)
            self.assertEqual(completed.resume_hint, RESUME_HINT_NORMAL)
            self.assertEqual(store.load_latest().turn_id, turn.turn_id)

    def test_turn_checkpoint_load_latest_prefers_jsonl_truth_over_stale_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TurnCheckpointStore.in_data_dir(tmp, "default", "session")
            first = store.complete_turn(store.start_turn().turn_id)
            second = store.start_turn()
            latest_path = (
                Path(tmp)
                / "checkpoints"
                / "default"
                / "session"
                / "latest.json"
            )
            latest_path.write_text(
                json.dumps(first.to_record()),
                encoding="utf-8",
            )

            latest = store.load_latest()

            self.assertIsNotNone(latest)
            self.assertEqual(latest.turn_id, second.turn_id)
            self.assertEqual(latest.status, second.status)

    def test_workspace_checkpoint_rewinds_deepmate_owned_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            workspace = root / "workspace"
            workspace.mkdir()
            tracked = workspace / "tracked.txt"
            created = workspace / "created.txt"
            tracked.write_text("after turn one", encoding="utf-8")
            store = WorkspaceCheckpointStore.in_data_dir(data_dir, "default", "session")

            store.capture_file(
                turn_id="turn_00002",
                operation="edit_text_file",
                workspace=workspace,
                path=tracked,
                after_content="after turn two",
            )
            tracked.write_text("after turn two", encoding="utf-8")
            store.capture_file(
                turn_id="turn_00002",
                operation="write_text_file",
                workspace=workspace,
                path=created,
                after_content="new file",
            )
            created.write_text("new file", encoding="utf-8")

            plan = store.rewind_plan("turn_00001", workspace)
            self.assertFalse(plan.has_conflicts())
            self.assertEqual(
                {(action.path, action.action) for action in plan.actions},
                {("tracked.txt", "restore"), ("created.txt", "delete")},
            )

            store.apply_rewind("turn_00001", workspace)
            self.assertEqual(tracked.read_text(encoding="utf-8"), "after turn one")
            self.assertFalse(created.exists())

    def test_workspace_rewind_uses_earliest_post_target_preimage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            workspace = root / "workspace"
            workspace.mkdir()
            path = workspace / "tracked.txt"
            path.write_text("turn one", encoding="utf-8")
            store = WorkspaceCheckpointStore.in_data_dir(data_dir, "default", "session")

            store.capture_file(
                turn_id="turn_00002",
                operation="edit_text_file",
                workspace=workspace,
                path=path,
                after_content="turn two",
            )
            path.write_text("turn two", encoding="utf-8")
            store.capture_file(
                turn_id="turn_00003",
                operation="edit_text_file",
                workspace=workspace,
                path=path,
                after_content="turn three",
            )
            path.write_text("turn three", encoding="utf-8")

            store.apply_rewind("turn_00001", workspace)

            self.assertEqual(path.read_text(encoding="utf-8"), "turn one")

    def test_workspace_rewind_detects_unexpected_current_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            workspace = root / "workspace"
            workspace.mkdir()
            path = workspace / "tracked.txt"
            path.write_text("before", encoding="utf-8")
            store = WorkspaceCheckpointStore.in_data_dir(data_dir, "default", "session")
            store.capture_file(
                turn_id="turn_00002",
                operation="edit_text_file",
                workspace=workspace,
                path=path,
                after_content="after",
            )
            path.write_text("external change", encoding="utf-8")

            plan = store.rewind_plan("turn_00001", workspace)
            self.assertTrue(plan.has_conflicts())
            self.assertEqual(plan.actions[0].reason, "current_content_differs")

            store.apply_rewind("turn_00001", workspace)
            self.assertEqual(path.read_text(encoding="utf-8"), "external change")

    def test_resume_context_is_added_only_for_abnormal_latest_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session_store = SessionStore.in_directory(root / "sessions")
            workspace = root / "workspace"
            workspace.mkdir()
            session = session_store.create(
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                title="resume",
            )
            transcript = session_store.transcript_store(session)
            checkpoint_store = TurnCheckpointStore.in_data_dir(
                root / "data",
                session.profile.name,
                session.session_id,
            )
            turn = checkpoint_store.start_turn()
            record = transcript.append_item(
                ModelConversationItem.from_message(
                    Message(role=MessageRole.USER, content="continue this")
                )
            )
            checkpoint_store.record_transcript_item(turn.turn_id, record)

            conversation = runtime_conversation_from_store(
                session_store,
                session,
                transcript,
                turn_checkpoint_store=checkpoint_store,
            )
            self.assertEqual(conversation[-1].message.role, MessageRole.ASSISTANT)
            self.assertIn("did not finish normally", conversation[-1].message.content)
            self.assertIn(turn.turn_id, conversation[-1].message.content)

            checkpoint_store.complete_turn(turn.turn_id)
            conversation = runtime_conversation_from_store(
                session_store,
                session,
                transcript,
                turn_checkpoint_store=checkpoint_store,
            )
            self.assertEqual(len(conversation), 1)

    def test_resume_context_includes_continuation_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session_store = SessionStore.in_directory(root / "sessions")
            workspace = root / "workspace"
            workspace.mkdir()
            session = session_store.create(
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                title="resume",
            )
            transcript = session_store.transcript_store(session)
            checkpoint_store = TurnCheckpointStore.in_data_dir(
                root / "data",
                session.profile.name,
                session.session_id,
            )
            turn = checkpoint_store.start_turn()
            checkpoint_store.attach_continuation_note(
                turn.turn_id,
                "Stop reason: hard_step_cap\nNext action: continue safely.",
            )
            checkpoint_store.max_steps_turn(turn.turn_id)

            latest = checkpoint_store.load_latest()
            self.assertEqual(latest.resume_hint, RESUME_HINT_MAX_STEPS)
            self.assertIn("hard_step_cap", latest.continuation_note)

            conversation = runtime_conversation_from_store(
                session_store,
                session,
                transcript,
                turn_checkpoint_store=checkpoint_store,
            )

            self.assertEqual(conversation[-1].message.role, MessageRole.ASSISTANT)
            self.assertIn("Continuation note:", conversation[-1].message.content)
            self.assertIn("Next action: continue safely.", conversation[-1].message.content)

    def test_filesystem_write_checkpoint_runs_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            path = workspace / "note.txt"
            path.write_text("old", encoding="utf-8")
            observed: list[str] = []

            def checkpoint(operation: str, write_path: Path, after_content: str) -> None:
                observed.append(operation)
                observed.append(write_path.read_text(encoding="utf-8"))
                observed.append(after_content)

            tools = {
                tool.name: tool
                for tool in workspace_filesystem_tools(
                    workspace,
                    include_write_tools=True,
                    write_checkpoint=checkpoint,
                )
            }
            result = tools["write_text_file"].call(
                {"path": "note.txt", "content": "new", "overwrite": True}
            )

            self.assertIn("Wrote note.txt", result.content)
            self.assertEqual(observed, ["write_text_file", "old", "new"])
            self.assertEqual(path.read_text(encoding="utf-8"), "new")

    def test_task_store_writes_are_captured_by_workspace_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            session_store = SessionStore.in_directory(root / "var" / "sessions")
            session = session_store.create(
                workspace,
                ProfileRef(name="default", uri="profiles/default"),
                "Task checkpoint",
            )
            controller = SessionCheckpointController.in_data_dir(
                root / "var",
                workspace=workspace,
                profile=session.profile.name,
                session_id=session.session_id,
            )
            router = SessionCheckpointWriteRouter(controller)
            task_store = TaskStore(workspace)
            task_store.set_write_checkpoint(router.capture_workspace_write)
            task_store.ensure()
            old_plan = task_store.plan_path.read_text(encoding="utf-8")
            turn = controller.start_turn()

            task_store.write_plan(
                "# Current task plan\n\n"
                "## Goal\nCapture task writes.\n\n"
                "## Steps\n- [ ] Update plan.\n"
            )

            checkpoint = controller.workspace_store.load_checkpoint(turn.turn_id)
            self.assertIsNotNone(checkpoint)
            snapshots = {item.path: item for item in checkpoint.files}
            self.assertIn("task/plan.md", snapshots)
            self.assertEqual(snapshots["task/plan.md"].before_content, old_plan)
            turn.close()

    def test_task_store_writes_can_target_closed_turn_by_router_thread_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            session_store = SessionStore.in_directory(root / "var" / "sessions")
            session = session_store.create(
                workspace,
                ProfileRef(name="default", uri="profiles/default"),
                "Task maintenance checkpoint",
            )
            controller = SessionCheckpointController.in_data_dir(
                root / "var",
                workspace=workspace,
                profile=session.profile.name,
                session_id=session.session_id,
            )
            router = SessionCheckpointWriteRouter(controller)
            task_store = TaskStore(workspace)
            task_store.set_write_checkpoint(router.capture_workspace_write)
            task_store.ensure()
            turn = controller.start_turn()
            turn_id = turn.turn_id
            turn.close()

            router.set_thread_turn_id(turn_id)
            try:
                task_store.write_plan(
                    "# Current task plan\n\n"
                    "## Goal\nCapture post-turn task maintenance.\n\n"
                    "## Steps\n- [ ] Update plan.\n"
                )
            finally:
                router.clear_thread_turn_id()

            checkpoint = controller.workspace_store.load_checkpoint(turn_id)
            self.assertIsNotNone(checkpoint)
            self.assertIn("task/plan.md", {item.path for item in checkpoint.files})

    def test_write_before_hook_blocks_before_checkpoint_and_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            path = workspace / "note.txt"
            path.write_text("old", encoding="utf-8")
            observed: list[str] = []

            def checkpoint(operation: str, write_path: Path, after_content: str) -> None:
                observed.append(operation)
                observed.append(write_path.as_posix())
                observed.append(after_content)

            tools = {
                tool.name: tool
                for tool in workspace_filesystem_tools(
                    workspace,
                    include_write_tools=True,
                    write_checkpoint=checkpoint,
                    hook_context=_hook_context(
                        HookEvent.WRITE_BEFORE,
                        HookActionType.DENY,
                        when={"path_globs": ["note.txt"]},
                        params={"reason": "write blocked"},
                    ),
                )
            }

            with self.assertRaisesRegex(ValueError, "write blocked"):
                tools["write_text_file"].call(
                    {"path": "note.txt", "content": "new", "overwrite": True}
                )

            self.assertEqual(observed, [])
            self.assertEqual(path.read_text(encoding="utf-8"), "old")

    def test_write_router_switches_between_session_controllers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            workspace = root / "workspace"
            workspace.mkdir()
            path = workspace / "note.txt"
            path.write_text("start", encoding="utf-8")
            first = SessionCheckpointController.in_data_dir(
                data_dir,
                workspace=workspace,
                profile="default",
                session_id="session_one",
            )
            second = SessionCheckpointController.in_data_dir(
                data_dir,
                workspace=workspace,
                profile="default",
                session_id="session_two",
            )
            router = SessionCheckpointWriteRouter(first)

            first_turn = first.start_turn()
            router.capture_workspace_write("write_text_file", path, "one")
            first_turn.close()
            path.write_text("one", encoding="utf-8")

            router.set_controller(second)
            second_turn = second.start_turn()
            router.capture_workspace_write("write_text_file", path, "two")
            second_turn.close()

            first_checkpoint = first.workspace_store.load_checkpoint(first_turn.turn_id)
            second_checkpoint = second.workspace_store.load_checkpoint(second_turn.turn_id)
            self.assertEqual(first_checkpoint.files[0].before_content, "start")
            self.assertEqual(second_checkpoint.files[0].before_content, "one")

    def test_write_router_uses_thread_controller_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            workspace = root / "workspace"
            workspace.mkdir()
            path = workspace / "note.txt"
            path.write_text("start", encoding="utf-8")
            first = SessionCheckpointController.in_data_dir(
                data_dir,
                workspace=workspace,
                profile="default",
                session_id="session_one",
            )
            second = SessionCheckpointController.in_data_dir(
                data_dir,
                workspace=workspace,
                profile="default",
                session_id="session_two",
            )
            router = SessionCheckpointWriteRouter(second)
            first_turn = first.start_turn()
            second_turn = second.start_turn()

            def worker_write() -> None:
                router.set_thread_controller(first)
                router.capture_workspace_write("write_text_file", path, "worker")
                router.clear_thread_controller()

            thread = threading.Thread(target=worker_write)
            thread.start()
            thread.join()
            path.write_text("worker", encoding="utf-8")
            router.capture_workspace_write("write_text_file", path, "main")
            first_turn.close()
            second_turn.close()

            first_checkpoint = first.workspace_store.load_checkpoint(first_turn.turn_id)
            second_checkpoint = second.workspace_store.load_checkpoint(second_turn.turn_id)
            self.assertEqual(first_checkpoint.files[0].before_content, "start")
            self.assertEqual(second_checkpoint.files[0].before_content, "worker")

    def test_workspace_checkpoint_skips_large_preimage_but_keeps_hash_and_size(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            workspace = root / "workspace"
            workspace.mkdir()
            path = workspace / "large.txt"
            before = "x" * 900_000
            path.write_text(before, encoding="utf-8")
            store = WorkspaceCheckpointStore.in_data_dir(data_dir, "default", "session")

            record = store.capture_file(
                turn_id="turn_00002",
                operation="write_text_file",
                workspace=workspace,
                path=path,
                after_content="new",
            )
            snapshot = record.files[0]

            self.assertEqual(snapshot.snapshot_status, "skipped")
            self.assertEqual(snapshot.skipped_reason, "file_too_large")
            self.assertEqual(snapshot.before_content, "")
            self.assertEqual(snapshot.before_size_bytes, len(before.encode("utf-8")))
            self.assertTrue(snapshot.before_sha256)

            path.write_text("new", encoding="utf-8")
            plan = store.rewind_plan("turn_00001", workspace)
            self.assertEqual(plan.actions[0].action, "skip")
            self.assertTrue(plan.actions[0].conflict)
            self.assertTrue(plan.has_conflicts())
            self.assertEqual(plan.actions[0].reason, "file_too_large")

    def test_workspace_checkpoint_capture_is_locked_per_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            workspace = root / "workspace"
            workspace.mkdir()
            first = workspace / "first.txt"
            second = workspace / "second.txt"
            first.write_text("first old", encoding="utf-8")
            second.write_text("second old", encoding="utf-8")
            store = WorkspaceCheckpointStore.in_data_dir(data_dir, "default", "session")

            def capture(path: Path, after: str) -> None:
                store.capture_file(
                    turn_id="turn_00002",
                    operation="write_text_file",
                    workspace=workspace,
                    path=path,
                    after_content=after,
                )

            threads = (
                threading.Thread(target=capture, args=(first, "first new")),
                threading.Thread(target=capture, args=(second, "second new")),
            )
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            checkpoint = store.load_checkpoint("turn_00002")
            self.assertIsNotNone(checkpoint)
            self.assertEqual(
                {snapshot.path for snapshot in checkpoint.files},
                {"first.txt", "second.txt"},
            )

    def test_turn_checkpoint_start_turn_allocates_unique_ids_concurrently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TurnCheckpointStore.in_data_dir(tmp, "default", "session")
            turn_ids: list[str] = []
            lock = threading.Lock()

            def start() -> None:
                turn = store.start_turn()
                with lock:
                    turn_ids.append(turn.turn_id)

            threads = [threading.Thread(target=start) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(len(turn_ids), 8)
            self.assertEqual(len(set(turn_ids)), 8)
            self.assertEqual(
                sorted(turn_ids),
                [f"turn_{index:05d}" for index in range(1, 9)],
            )

    def test_cli_rewind_conversation_truncates_transcript_and_latest_checkpoint(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            workspace = root / "workspace"
            workspace.mkdir()
            session_store = SessionStore.in_directory(data_dir / "sessions")
            session = session_store.create(
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                title="rewind",
            )
            transcript = session_store.transcript_store(session)
            turn_store = TurnCheckpointStore.in_data_dir(
                data_dir,
                session.profile.name,
                session.session_id,
            )

            turn_one = turn_store.start_turn()
            first_user = transcript.append_item(
                _message_item(MessageRole.USER, "first")
            )
            first_assistant = transcript.append_item(
                _message_item(MessageRole.ASSISTANT, "first done")
            )
            turn_store.record_transcript_item(turn_one.turn_id, first_user)
            turn_store.record_transcript_item(turn_one.turn_id, first_assistant)
            turn_store.complete_turn(turn_one.turn_id)

            turn_two = turn_store.start_turn()
            second_user = transcript.append_item(
                _message_item(MessageRole.USER, "second")
            )
            turn_store.record_transcript_item(turn_two.turn_id, second_user)
            turn_store.complete_turn(turn_two.turn_id)

            with redirect_stdout(StringIO()):
                result = _handle_rewind_command(
                    SimpleNamespace(data_dir=data_dir),
                    session_store,
                    SimpleNamespace(
                        rewind=session.session_id,
                        rewind_to=turn_one.turn_id,
                        rewind_mode="conversation",
                        rewind_apply=True,
                        rewind_force=False,
                    ),
                )

            self.assertEqual(result, 0)
            self.assertEqual(
                [record.sequence for record in transcript.load_records()],
                [1, 2],
            )
            self.assertEqual(turn_store.load_latest().turn_id, turn_one.turn_id)


def _message_record(
    session_id: str,
    sequence: int,
    role: MessageRole,
    content: str,
) -> TranscriptRecord:
    return TranscriptRecord.from_item(
        session_id=session_id,
        sequence=sequence,
        item=ModelConversationItem.from_message(Message(role=role, content=content)),
    )


def _message_item(role: MessageRole, content: str) -> ModelConversationItem:
    return ModelConversationItem.from_message(Message(role=role, content=content))


def _hook_context(
    event_name: HookEvent,
    action_type: HookActionType,
    *,
    when: dict[str, object] | None = None,
    params: dict[str, object] | None = None,
) -> HookRuntimeContext:
    return HookRuntimeContext.from_registry(
        HookRegistry.from_hooks(
            (
                HookDefinition(
                    hook_id="test-hook",
                    event_name=event_name,
                    layer=HookLayer.SESSION,
                    when=when or {},
                    actions=(HookAction(action_type, params or {}),),
                ),
            )
        )
    )


if __name__ == "__main__":
    unittest.main()
