from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deepmate.channels.session_maintenance import (
    _summary_source_records,
    runtime_conversation_from_store,
)
from deepmate.domain import Message, MessageRole, ProfileRef
from deepmate.providers import ModelConversationItem, ModelResponse, TokenUsage
from deepmate.runtime import (
    ConversationBudgetPolicy,
    SessionSummaryInput,
    SessionSummarySourceItem,
    generate_session_summary,
    parse_checkpoint_update_response,
    session_summary_to_conversation_item,
    validate_session_summary_response,
)
from deepmate.storage import SessionStore, SessionSummaryRecord, TranscriptRecord


class StubProvider:
    def __init__(self, response: ModelResponse) -> None:
        self.response = response
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return self.response


class SessionSummaryTests(unittest.TestCase):
    def test_summary_store_roundtrip_and_uncovered_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            store = SessionStore.in_directory(Path(tmp) / "sessions")
            session = store.create(
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                title="summary test",
            )
            transcript = store.transcript_store(session)
            transcript.append_item(_message_item(MessageRole.USER, "one"))
            transcript.append_item(_message_item(MessageRole.ASSISTANT, "two"))
            transcript.append_item(_message_item(MessageRole.USER, "three"))

            record = SessionSummaryRecord.create(
                session_id=session.session_id,
                content="## Session Summary\n\nCovered one item.",
                covered_until_sequence=1,
                covered_item_count=1,
                source_item_count=1,
                estimated_source_tokens=12,
                source_model="deepseek-v4-flash",
                usage={"input_tokens": 20},
            )
            store.summary_store(session).save_latest(record)

            loaded = store.summary_store(session).load_latest()
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.summary_id, record.summary_id)
            self.assertEqual(len(transcript.load_items_after(1)), 2)
            conversation = runtime_conversation_from_store(store, session, transcript)
            self.assertEqual(len(conversation), 3)
            self.assertIn("Covered one item", conversation[0].message.content)

    def test_generate_session_summary_builds_request_and_preserves_usage(self) -> None:
        source_input = SessionSummaryInput(
            source_items=(
                SessionSummarySourceItem(
                    sequence=1,
                    item=_message_item(MessageRole.USER, "请记住这个目标"),
                ),
            )
        )
        provider = StubProvider(
            ModelResponse(
                content="## Session Summary\n\n### User Goal\n- 继续当前目标。",
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            )
        )

        summary = generate_session_summary(
            provider=provider,
            model="deepseek-v4-flash",
            summary_input=source_input,
            options={"thinking": {"type": "disabled"}},
        )

        self.assertTrue(summary.is_ready())
        self.assertEqual(summary.covered_until_sequence, 1)
        self.assertEqual(summary.usage.total_tokens(), 15)
        request = provider.requests[0]
        self.assertEqual(request.conversation[0].message.role, MessageRole.SYSTEM)
        self.assertEqual(request.conversation[1].message.role, MessageRole.USER)
        self.assertEqual(request.options["thinking"], {"type": "disabled"})
        self.assertIn(
            "Product Or Project Context",
            request.conversation[1].message.content,
        )
        self.assertIn(
            "Recent Continuation Notes",
            request.conversation[1].message.content,
        )
        self.assertIn(
            "scoped product/project context",
            request.conversation[0].message.content,
        )

    def test_checkpoint_update_rejects_none_summary_content(self) -> None:
        source_input = SessionSummaryInput(
            source_items=(
                SessionSummarySourceItem(
                    sequence=1,
                    item=_message_item(MessageRole.USER, "hello"),
                ),
            )
        )

        with self.assertRaises(ValueError):
            parse_checkpoint_update_response(
                '{"session_summary":{"content":null}}',
                "",
                source_input,
                "summary-model",
            )

    def test_summary_validation_rejects_empty_or_truncated_output(self) -> None:
        source_input = SessionSummaryInput(
            source_items=(
                SessionSummarySourceItem(
                    sequence=1,
                    item=_message_item(MessageRole.USER, "hello"),
                ),
            )
        )

        with self.assertRaises(ValueError):
            validate_session_summary_response("", "", source_input)
        with self.assertRaises(ValueError):
            validate_session_summary_response("not enough", "length", source_input)

    def test_summary_context_item_is_not_a_user_request(self) -> None:
        item = session_summary_to_conversation_item(
            "## Session Summary\n\nEarlier context.",
            summary_id="sum_1",
        )

        self.assertEqual(item.message.role, MessageRole.ASSISTANT)
        self.assertIn("not a new user request", item.message.content)
        self.assertIn("sum_1", item.message.content)

    def test_summary_source_keeps_recent_transcript_uncovered(self) -> None:
        records = tuple(
            TranscriptRecord.from_item(
                session_id="session",
                sequence=index,
                item=_message_item(MessageRole.USER, f"message {index}"),
            )
            for index in range(1, 6)
        )
        source = _summary_source_records(
            records,
            previous=None,
            policy=ConversationBudgetPolicy(protect_recent_items=2),
        )

        self.assertEqual(tuple(record.sequence for record in source), (1, 2, 3))


def _message_item(role: MessageRole, content: str) -> ModelConversationItem:
    return ModelConversationItem.from_message(Message(role=role, content=content))


if __name__ == "__main__":
    unittest.main()
