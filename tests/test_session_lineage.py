from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepmate.channels.session_lineage import (
    format_session_tree,
    handle_session_lineage_command,
)
from deepmate.domain import Message, MessageRole, ProfileRef
from deepmate.providers import (
    ModelConversationItem,
    ModelToolExchange,
    ModelToolRequest,
    ModelToolResult,
)
from deepmate.storage import (
    SessionStore,
    SessionSummaryRecord,
    ToolOutputStore,
    TurnCheckpointStore,
)


class SessionLineageTests(unittest.TestCase):
    def test_old_session_metadata_still_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore.in_directory(root / "sessions")
            path = store.metadata_path("session_old")
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "session_id": "session_old",
                        "title": "Old",
                        "workspace": str(root / "workspace"),
                        "profile": {"name": "default", "uri": "profiles/default"},
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "updated_at": "2026-01-01T00:00:00+00:00",
                        "status": "active",
                        "transcript_path": str(root / "sessions" / "session_old.jsonl"),
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            loaded = store.load("session_old")

            self.assertEqual(loaded.parent_session_id, "")
            self.assertEqual(loaded.fork_kind, "")
            self.assertEqual(loaded.forked_from_sequence, 0)

    def test_missing_session_metadata_reports_unknown_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore.in_directory(Path(tmp) / "sessions")

            with self.assertRaisesRegex(ValueError, "unknown session"):
                store.load("missing")

    def test_old_session_metadata_tolerates_nullable_and_string_lineage_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore.in_directory(root / "sessions")
            base = {
                "title": "Old",
                "workspace": str(root / "workspace"),
                "profile": {"name": "default", "uri": "profiles/default"},
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "active",
            }
            for session_id, value, expected in (
                ("session_none", None, 0),
                ("session_string", "5", 5),
            ):
                path = store.metadata_path(session_id)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            **base,
                            "session_id": session_id,
                            "transcript_path": str(
                                root / "sessions" / f"{session_id}.jsonl"
                            ),
                            "forked_from_sequence": value,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )

                loaded = store.load(session_id)

                self.assertEqual(loaded.forked_from_sequence, expected)

    def test_session_metadata_round_trips_two_layer_profile_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            store = SessionStore.in_directory(root / "sessions")
            profile = ProfileRef(
                name="default",
                uri="profiles/default",
                global_uri=str(root / "home" / "profiles" / "default"),
                project_uri="profiles/default",
            )

            session = store.create(workspace=workspace, profile=profile, title="Two layer")
            loaded = store.load(session.session_id)

            self.assertEqual(loaded.profile.name, "default")
            self.assertEqual(loaded.profile.uri, "profiles/default")
            self.assertEqual(
                loaded.profile.global_uri,
                str(root / "home" / "profiles" / "default"),
            )
            self.assertEqual(loaded.profile.project_uri, "profiles/default")

    def test_clone_copies_transcript_and_covered_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, session = _store_with_session(root, "Source")
            transcript = store.transcript_store(session)
            transcript.append_item(_message_item(MessageRole.USER, "one"))
            transcript.append_item(_message_item(MessageRole.ASSISTANT, "two"))
            summary = SessionSummaryRecord.create(
                session_id=session.session_id,
                content="Covered one.",
                covered_until_sequence=1,
                covered_item_count=1,
                source_item_count=1,
                estimated_source_tokens=10,
                source_model="model",
            )
            store.summary_store(session).save_latest(summary)

            clone = store.clone_session(session, title="Clone")

            self.assertEqual(clone.parent_session_id, session.session_id)
            self.assertEqual(clone.lineage_root_session_id, session.session_id)
            self.assertEqual(clone.fork_kind, "clone")
            self.assertEqual(clone.forked_from_sequence, 2)
            copied = store.transcript_store(clone).load_records()
            self.assertEqual(len(copied), 2)
            self.assertTrue(all(record.session_id == clone.session_id for record in copied))
            self.assertNotEqual(copied[0].record_id, transcript.load_records()[0].record_id)
            copied_summary = store.summary_store(clone).load_latest()
            self.assertIsNotNone(copied_summary)
            self.assertEqual(copied_summary.session_id, clone.session_id)
            self.assertNotEqual(copied_summary.summary_id, summary.summary_id)

    def test_transcript_append_fsyncs_written_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, session = _store_with_session(root, "Source")
            transcript = store.transcript_store(session)

            with patch("deepmate.storage.session_store.os.fsync") as fsync:
                record = transcript.append_item(_message_item(MessageRole.USER, "one"))

            self.assertIsNotNone(record)
            self.assertEqual(record.sequence, 1)
            fsync.assert_called_once()

    def test_transcript_tool_result_attachments_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, session = _store_with_session(root, "Source")
            transcript = store.transcript_store(session)
            transcript.append_item(
                ModelConversationItem.from_tool_exchange(
                    ModelToolExchange(
                        tool_requests=(
                            ModelToolRequest(name="computer_screenshot", id="call_1"),
                        ),
                        tool_results=(
                            ModelToolResult(
                                name="computer_screenshot",
                                request_id="call_1",
                                content="Desktop screenshot saved.",
                                attachments=(
                                    {
                                        "type": "image",
                                        "path": str(root / "screen.png"),
                                        "mime_type": "image/png",
                                    },
                                ),
                            ),
                        ),
                    )
                )
            )

            loaded = transcript.load_records()[0].to_item()

            self.assertIsNotNone(loaded.tool_exchange)
            result = loaded.tool_exchange.tool_results[0]
            self.assertEqual(result.attachments[0]["type"], "image")
            self.assertEqual(result.attachments[0]["mime_type"], "image/png")

    def test_fork_copies_until_sequence_and_skips_future_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, session = _store_with_session(root, "Source")
            transcript = store.transcript_store(session)
            transcript.append_item(_message_item(MessageRole.USER, "one"))
            transcript.append_item(_message_item(MessageRole.ASSISTANT, "two"))
            transcript.append_item(_message_item(MessageRole.USER, "three"))
            summary = SessionSummaryRecord.create(
                session_id=session.session_id,
                content="Covers future.",
                covered_until_sequence=3,
                covered_item_count=3,
                source_item_count=3,
                estimated_source_tokens=20,
                source_model="model",
            )
            store.summary_store(session).save_latest(summary)

            fork = store.fork_session_at_sequence(
                session,
                2,
                title="Fork",
                turn_id="turn_1",
            )

            self.assertEqual(fork.parent_session_id, session.session_id)
            self.assertEqual(fork.fork_kind, "fork")
            self.assertEqual(fork.forked_from_turn_id, "turn_1")
            self.assertEqual(fork.forked_from_sequence, 2)
            copied = store.transcript_store(fork).load_records()
            self.assertEqual([record.sequence for record in copied], [1, 2])
            self.assertIsNone(store.summary_store(fork).load_latest())

    def test_fork_copies_inherited_tool_output_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, session = _store_with_session(root, "Source")
            tool_outputs = ToolOutputStore.in_data_dir(
                root,
                session.profile.name,
                session.session_id,
            )
            saved = tool_outputs.save(
                tool_name="run_tests",
                tool_source="native",
                content_kind="log",
                content="raw output",
                estimated_tokens=10,
                request_id="call_1",
            )
            transcript = store.transcript_store(session)
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
                                refs=(f"tool_output_ref={saved.ref}",),
                            ),
                        ),
                    )
                )
            )

            fork = store.fork_session_at_sequence(session, 1, title="Fork")
            copied_store = ToolOutputStore.in_data_dir(
                root,
                fork.profile.name,
                fork.session_id,
            )
            copied = copied_store.load(saved.ref)

            self.assertIsNotNone(copied)
            self.assertEqual(copied.content, "raw output")
            self.assertEqual(copied.session_id, fork.session_id)

    def test_command_fork_from_turn_checkpoint_and_tree_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, session = _store_with_session(root, "Root")
            transcript = store.transcript_store(session)
            user = transcript.append_item(_message_item(MessageRole.USER, "one"))
            assistant = transcript.append_item(_message_item(MessageRole.ASSISTANT, "two"))
            self.assertIsNotNone(user)
            self.assertIsNotNone(assistant)
            turn_store = TurnCheckpointStore.in_data_dir(
                root,
                session.profile.name,
                session.session_id,
            )
            turn = turn_store.start_turn()
            turn_store.record_transcript_item(turn.turn_id, user)
            turn_store.record_transcript_item(turn.turn_id, assistant)

            result = handle_session_lineage_command(
                f"/session fork {turn.turn_id} Try again",
                session_store=store,
                session=session,
                workspace=session.workspace,
                profile=session.profile,
                turn_store=turn_store,
            )

            self.assertFalse(isinstance(result, str))
            self.assertIsNotNone(result)
            fork = result.session
            self.assertEqual(fork.forked_from_turn_id, turn.turn_id)
            self.assertIn("Created session fork", result.body)
            tree = format_session_tree(
                store.lineage_tree(workspace=session.workspace, profile=session.profile),
                current_session_id=fork.session_id,
                workspace=session.workspace,
                profile=session.profile,
            )
            self.assertIn("Root", tree)
            self.assertIn("Try again", tree)
            self.assertIn("current", tree)

    def test_command_fork_missing_turn_reports_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, session = _store_with_session(root, "Root")
            turn_store = TurnCheckpointStore.in_data_dir(
                root,
                session.profile.name,
                session.session_id,
            )

            with self.assertRaisesRegex(ValueError, "turn not found: turn-abc"):
                handle_session_lineage_command(
                    "/session fork turn-abc Try again",
                    session_store=store,
                    session=session,
                    workspace=session.workspace,
                    profile=session.profile,
                    turn_store=turn_store,
                )

    def test_clone_title_strips_unbalanced_quote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, session = _store_with_session(root, "Root")

            result = handle_session_lineage_command(
                "/session clone 'Branch title",
                session_store=store,
                session=session,
                workspace=session.workspace,
                profile=session.profile,
            )

            self.assertFalse(isinstance(result, str))
            self.assertIsNotNone(result)
            self.assertEqual(result.session.title, "Branch title")


def _store_with_session(root: Path, title: str):
    workspace = root / "workspace"
    workspace.mkdir()
    store = SessionStore.in_directory(root / "sessions")
    session = store.create(
        workspace=workspace,
        profile=ProfileRef(name="default", uri="profiles/default"),
        title=title,
    )
    return store, session


def _message_item(role: MessageRole, content: str) -> ModelConversationItem:
    return ModelConversationItem.from_message(Message(role=role, content=content))


if __name__ == "__main__":
    unittest.main()
