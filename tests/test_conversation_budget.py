import unittest

from deepmate.domain import Message, MessageRole
from deepmate.providers import (
    ModelConversationItem,
    ModelRequest,
    ModelToolExchange,
    ModelToolRequest,
    ModelToolResult,
)
from deepmate.runtime.conversation_budget import (
    ConversationBudgetPolicy,
    RequestBudgetReport,
    build_request_budget_report,
    build_conversation_budget_report,
    estimate_conversation_item_tokens,
    estimate_text_tokens,
)
from deepmate.runtime.session_summary import (
    SessionSummaryAction,
    SessionSummaryPolicy,
    decide_session_summary,
)


class ConversationBudgetTests(unittest.TestCase):
    def test_deepseek_aware_text_estimate(self) -> None:
        self.assertEqual(estimate_text_tokens("你好世界你好世界你好"), 6)
        self.assertEqual(estimate_text_tokens("abcdefghij"), 3)
        self.assertGreater(estimate_text_tokens('{"path":"src/deepmate"}'), 0)

    def test_unknown_conversation_item_gets_conservative_overhead(self) -> None:
        self.assertGreater(estimate_conversation_item_tokens(ModelConversationItem()), 0)

    def test_request_budget_counts_system_tools_and_tool_output(self) -> None:
        request = ModelRequest(
            model="deepseek-v4-flash",
            conversation=(
                ModelConversationItem.from_message(
                    Message(role=MessageRole.SYSTEM, content="系统提示")
                ),
                ModelConversationItem.from_message(
                    Message(role=MessageRole.USER, content="读取文件")
                ),
                ModelConversationItem.from_tool_exchange(
                    ModelToolExchange(
                        tool_requests=(
                            ModelToolRequest(
                                name="read_text_file",
                                id="call_1",
                                raw_arguments='{"path":"README.md"}',
                            ),
                        ),
                        tool_results=(
                            ModelToolResult(
                                name="read_text_file",
                                request_id="call_1",
                                content="工具输出" * 100,
                                refs=("artifact://README.md",),
                            ),
                        ),
                    )
                ),
            ),
            tool_schemas=(
                {
                    "name": "read_text_file",
                    "description": "Read a workspace text file.",
                    "input_schema": {"type": "object"},
                },
            ),
        )

        report = build_request_budget_report(request, ConversationBudgetPolicy())

        self.assertGreater(report.estimated_input_tokens, 0)
        self.assertGreater(report.estimated_system_tokens, 0)
        self.assertGreater(report.estimated_history_tokens, 0)
        self.assertGreater(report.estimated_tool_output_tokens, 0)
        self.assertGreater(report.estimated_tool_schema_tokens, 0)
        self.assertGreater(report.tool_output_ratio, 0)

    def test_small_context_window_scales_default_reserves(self) -> None:
        request = ModelRequest(
            model="small-model",
            conversation=(
                ModelConversationItem.from_message(
                    Message(role=MessageRole.SYSTEM, content="system")
                ),
                ModelConversationItem.from_message(
                    Message(role=MessageRole.USER, content="hello")
                ),
            ),
        )

        report = build_request_budget_report(
            request,
            ConversationBudgetPolicy(model_context_tokens=100_000),
        )

        self.assertEqual(
            report.response_token_reserve + report.safety_margin_tokens,
            15_000,
        )
        self.assertEqual(report.usable_input_tokens, 85_000)

    def test_history_budget_is_capped_to_usable_input_window(self) -> None:
        report = build_conversation_budget_report(
            (
                ModelConversationItem.from_message(
                    Message(role=MessageRole.USER, content="x" * 1000)
                ),
            ),
            ConversationBudgetPolicy(
                history_window_mode="trim",
                history_token_budget=1_000_000,
                model_context_tokens=1_000,
                response_token_reserve=200,
                safety_margin_tokens=100,
            ),
        )

        self.assertEqual(report.history_token_budget, 700)

    def test_session_summary_thresholds_are_explainable(self) -> None:
        self.assertEqual(
            decide_session_summary(_report(249_999)).action,
            SessionSummaryAction.SKIP,
        )
        self.assertEqual(
            decide_session_summary(_report(250_000)).action,
            SessionSummaryAction.OBSERVE,
        )
        self.assertEqual(
            decide_session_summary(_report(500_000)).action,
            SessionSummaryAction.CHECKPOINT,
        )
        self.assertEqual(
            decide_session_summary(_report(600_000)).action,
            SessionSummaryAction.CHECKPOINT,
        )
        self.assertEqual(
            decide_session_summary(_report(600_000)).reason,
            "guard_tokens",
        )
        self.assertEqual(
            decide_session_summary(_report(750_000)).action,
            SessionSummaryAction.CHECKPOINT,
        )
        self.assertEqual(
            decide_session_summary(_report(750_000)).reason,
            "emergency_tokens",
        )

    def test_session_summary_thresholds_follow_model_context_window(self) -> None:
        self.assertEqual(
            decide_session_summary(_report(24_999, model_context_tokens=100_000)).action,
            SessionSummaryAction.SKIP,
        )
        self.assertEqual(
            decide_session_summary(_report(25_000, model_context_tokens=100_000)).action,
            SessionSummaryAction.OBSERVE,
        )
        self.assertEqual(
            decide_session_summary(_report(50_000, model_context_tokens=100_000)).reason,
            "checkpoint_tokens",
        )
        self.assertEqual(
            decide_session_summary(_report(75_000, model_context_tokens=100_000)).reason,
            "emergency_tokens",
        )

    def test_session_summary_threshold_refs_report_effective_ratio(self) -> None:
        decision = decide_session_summary(
            _report(499_999),
            policy=SessionSummaryPolicy(observe_tokens=500_000, observe_ratio=0.25),
        )

        self.assertEqual(decision.action, SessionSummaryAction.SKIP)
        self.assertIn("observe_tokens=500000", decision.refs)
        self.assertIn("observe_ratio=0.5000", decision.refs)

    def test_session_summary_thresholds_are_capped_by_usable_input_window(self) -> None:
        decision = decide_session_summary(
            _report(85_000, model_context_tokens=100_000),
            policy=SessionSummaryPolicy(emergency_ratio=0.95),
        )

        self.assertIn("emergency_tokens=85000", decision.refs)
        self.assertIn("emergency_ratio=0.8500", decision.refs)

    def test_tool_output_can_trigger_checkpoint_before_500k(self) -> None:
        decision = decide_session_summary(_report(320_000, tool_output_ratio=0.55))

        self.assertEqual(decision.action, SessionSummaryAction.CHECKPOINT)
        self.assertTrue(decision.should_checkpoint())

    def test_quality_warning_uses_cache_and_profile_signals(self) -> None:
        low_cache = decide_session_summary(_report(420_000), cache_hit_ratio=0.42)
        profile_pending = decide_session_summary(
            _report(420_000),
            cache_hit_ratio=0.90,
            profile_context_changed=True,
        )

        self.assertEqual(low_cache.action, SessionSummaryAction.CHECKPOINT)
        self.assertEqual(profile_pending.action, SessionSummaryAction.OBSERVE)
        self.assertEqual(profile_pending.reason, "profile_pending_quality_warning")
        self.assertFalse(profile_pending.should_checkpoint())


def _report(
    tokens: int,
    tool_output_ratio: float = 0.0,
    model_context_tokens: int = 1_000_000,
) -> RequestBudgetReport:
    response_token_reserve = 64_000 if model_context_tokens >= 200_000 else 8_000
    safety_margin_tokens = 50_000 if model_context_tokens >= 200_000 else 7_000
    usable_input_tokens = max(
        1,
        model_context_tokens - response_token_reserve - safety_margin_tokens,
    )
    return RequestBudgetReport(
        conversation_items=1,
        tool_schema_count=0,
        estimated_input_tokens=tokens,
        estimated_system_tokens=10,
        estimated_history_tokens=max(0, tokens - 10),
        estimated_tool_output_tokens=int(tokens * tool_output_ratio),
        estimated_tool_schema_tokens=0,
        model_context_tokens=model_context_tokens,
        response_token_reserve=response_token_reserve,
        safety_margin_tokens=safety_margin_tokens,
        usable_input_tokens=usable_input_tokens,
        pressure_ratio=tokens / usable_input_tokens,
        tool_output_ratio=tool_output_ratio,
    )


if __name__ == "__main__":
    unittest.main()
