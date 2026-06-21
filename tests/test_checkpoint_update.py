from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from deepmate.domain import Message, MessageRole
from deepmate.providers import ModelConversationItem, ModelResponse
from deepmate.runtime import (
    SessionSummaryInput,
    SessionSummarySourceItem,
    build_checkpoint_update_request,
    generate_checkpoint_update,
    parse_checkpoint_update_response,
)


class StubProvider:
    def __init__(self, responses: tuple[ModelResponse, ...]) -> None:
        self.responses = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected provider call")
        return self.responses.pop(0)


def _summary_input() -> SessionSummaryInput:
    return SessionSummaryInput(
        source_items=(
            SessionSummarySourceItem(
                sequence=1,
                item=ModelConversationItem.from_message(
                    Message(role=MessageRole.USER, content="以后请用中文直接回答。")
                ),
            ),
            SessionSummarySourceItem(
                sequence=2,
                item=ModelConversationItem.from_message(
                    Message(role=MessageRole.ASSISTANT, content="收到。")
                ),
            ),
        )
    )


class CheckpointUpdateTests(unittest.TestCase):
    def test_parse_checkpoint_update_response(self) -> None:
        payload = {
            "session_summary": {
                "content": "## Session Summary\n\n### User Goal\n继续推进 memory 闭环。"
            },
            "memory_patch": {
                "operations": [
                    {
                        "action": "write_user",
                        "content": "用户偏好中文直接回答。",
                        "confidence": 0.9,
                    },
                    {
                        "action": "write_project_memory",
                        "content": "本项目统一使用 pnpm。",
                        "confidence": 0.8,
                    }
                ]
            },
            "activity_digest": {
                "summary": "讨论了 memory 闭环。",
                "highlights": ["确定 checkpoint 立即写 memory。"],
                "next_steps": ["实现 daily maintenance。"],
            },
        }

        update = parse_checkpoint_update_response(
            content=json.dumps(payload, ensure_ascii=False),
            finish_reason="stop",
            summary_input=_summary_input(),
            model="deepseek-v4-pro",
        )

        self.assertTrue(update.is_ready())
        self.assertEqual(update.memory_operation_count(), 2)
        self.assertEqual(
            update.memory_patch.operations[0].content,
            "用户偏好中文直接回答。",
        )
        self.assertEqual(
            update.memory_patch.operations[1].action,
            "write_project_memory",
        )
        self.assertEqual(
            update.memory_patch.operations[1].content,
            "本项目统一使用 pnpm。",
        )
        self.assertIn(
            "checkpoint 立即写 memory",
            update.activity_digest.render(),
        )

    def test_build_checkpoint_update_request_reads_three_hot_profile_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            global_profile = root / "home" / "profiles" / "default"
            project_profile = root / "workspace" / "profiles" / "default"
            global_profile.mkdir(parents=True)
            project_profile.mkdir(parents=True)
            (global_profile / "user.md").write_text(
                "- 用户偏好中文直接回答。\n",
                encoding="utf-8",
            )
            (global_profile / "memory.md").write_text(
                "- 保持回答克制。\n",
                encoding="utf-8",
            )
            (project_profile / "memory.md").write_text(
                "- 本项目统一使用 pnpm。\n",
                encoding="utf-8",
            )

            request = build_checkpoint_update_request(
                model="deepseek-v4-pro",
                summary_input=_summary_input(),
                profile_dir=global_profile,
                project_profile_dir=project_profile,
            )

            user_prompt = request.conversation[1].message.content
            self.assertIn("Current global user.md bullets:", user_prompt)
            self.assertIn("用户偏好中文直接回答", user_prompt)
            self.assertIn("Current global memory.md bullets:", user_prompt)
            self.assertIn("保持回答克制", user_prompt)
            self.assertIn("Current project memory.md bullets:", user_prompt)
            self.assertIn("本项目统一使用 pnpm", user_prompt)

    def test_generate_checkpoint_update_falls_back_to_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = StubProvider(
                (
                    ModelResponse(content="not json"),
                    ModelResponse(
                        content=(
                            "## Session Summary\n\n"
                            "### User Goal\n继续推进 memory 闭环。"
                        )
                    ),
                )
            )

            update = generate_checkpoint_update(
                provider=provider,
                model="deepseek-v4-pro",
                summary_input=_summary_input(),
                profile_dir=Path(tmp),
            )

            self.assertEqual(len(provider.requests), 2)
            self.assertTrue(update.is_ready())
            self.assertEqual(update.memory_operation_count(), 0)
            self.assertIn("Session Summary", update.activity_digest.render())


if __name__ == "__main__":
    unittest.main()
