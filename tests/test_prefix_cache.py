from __future__ import annotations

import unittest

from deepmate.domain import Message, MessageRole, ProfileRef
from deepmate.providers import ModelConversationItem, ModelRequest
from deepmate.runtime import build_model_request, model_request_prefix_fingerprint


class PrefixCacheFingerprintTests(unittest.TestCase):
    def test_fingerprint_is_stable_for_equivalent_schema_ordering(self) -> None:
        first = ModelRequest(
            model="stub-model",
            conversation=(
                ModelConversationItem.from_message(
                    Message(role=MessageRole.SYSTEM, content="system context")
                ),
                ModelConversationItem.from_message(
                    Message(role=MessageRole.USER, content="hello")
                ),
            ),
            tool_schemas=(
                {
                    "name": "search",
                    "input_schema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            ),
            options={"temperature": 0, "max_tokens": 1000},
        )
        second = ModelRequest(
            model="stub-model",
            conversation=(
                ModelConversationItem.from_message(
                    Message(role=MessageRole.SYSTEM, content="system context")
                ),
                ModelConversationItem.from_message(
                    Message(role=MessageRole.USER, content="different user text")
                ),
            ),
            tool_schemas=(
                {
                    "input_schema": {
                        "properties": {"query": {"type": "string"}},
                        "type": "object",
                    },
                    "name": "search",
                },
            ),
            options={"max_tokens": 1000, "temperature": 0},
        )

        self.assertEqual(
            model_request_prefix_fingerprint(first).digest,
            model_request_prefix_fingerprint(second).digest,
        )

    def test_system_context_changes_prefix_digest(self) -> None:
        base = ModelRequest(
            model="stub-model",
            conversation=(
                ModelConversationItem.from_message(
                    Message(role=MessageRole.SYSTEM, content="system context")
                ),
            ),
        )
        changed = ModelRequest(
            model="stub-model",
            conversation=(
                ModelConversationItem.from_message(
                    Message(role=MessageRole.SYSTEM, content="changed system context")
                ),
            ),
        )

        self.assertNotEqual(
            model_request_prefix_fingerprint(base).digest,
            model_request_prefix_fingerprint(changed).digest,
        )

    def test_turn_tail_messages_do_not_change_prefix_digest(self) -> None:
        profile = ProfileRef(name="default", uri="profiles/default")
        system_message = Message(
            role=MessageRole.SYSTEM,
            content="stable system context",
        )
        first = build_model_request(
            workspace=".",
            profile=profile,
            messages=(Message(role=MessageRole.USER, content="do the work"),),
            model="stub-model",
            system_message=system_message,
            turn_tail_messages=(
                Message(role=MessageRole.USER, content="dynamic hint A"),
            ),
        ).request
        second = build_model_request(
            workspace=".",
            profile=profile,
            messages=(Message(role=MessageRole.USER, content="do the work"),),
            model="stub-model",
            system_message=system_message,
            turn_tail_messages=(
                Message(role=MessageRole.USER, content="dynamic hint B"),
            ),
        ).request

        self.assertEqual(
            [item.message.content for item in first.conversation if item.message],
            ["stable system context", "do the work", "dynamic hint A"],
        )
        self.assertEqual(
            model_request_prefix_fingerprint(first).digest,
            model_request_prefix_fingerprint(second).digest,
        )

    def test_turn_tail_messages_reject_system_role(self) -> None:
        profile = ProfileRef(name="default", uri="profiles/default")

        with self.assertRaises(ValueError):
            build_model_request(
                workspace=".",
                profile=profile,
                messages=(Message(role=MessageRole.USER, content="do the work"),),
                model="stub-model",
                system_message=Message(
                    role=MessageRole.SYSTEM,
                    content="stable system context",
                ),
                turn_tail_messages=(
                    Message(role=MessageRole.SYSTEM, content="dynamic system"),
                ),
            )


if __name__ == "__main__":
    unittest.main()
