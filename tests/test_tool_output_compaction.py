from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deepmate.providers import ModelToolResult
from deepmate.runtime import ConversationBudgetPolicy, RequestBudgetReport
from deepmate.runtime.tool_output_compaction import (
    ToolOutputCompactionPolicy,
    ToolOutputCompactor,
)
from deepmate.storage import ToolOutputStore


class ToolOutputCompactionTests(unittest.TestCase):
    def test_json_minify_does_not_compact_when_headroom_is_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            compactor = _compactor(Path(tmp))
            result = ModelToolResult(
                name="api.fetch",
                request_id="call-1",
                content='{\n  "items": [\n    {"id": 1, "name": "a"}\n  ]\n}',
            )

            processed = compactor.process(
                result,
                tool_source="native",
                request_budget_report=_report(current_tokens=1_000),
            )

        self.assertIn("[tool output normalized: json_minify]", processed.result.content)
        self.assertIn('"items":[{"id":1,"name":"a"}]', processed.result.content)
        self.assertNotIn("tool_output_compacted=true", processed.result.refs)
        self.assertEqual([event.kind for event in processed.events], ["tool_output_normalized"])

    def test_repeated_lines_fold_before_compaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            compactor = _compactor(Path(tmp))
            result = ModelToolResult(
                name="pytest",
                request_id="call-1",
                content="\n".join(("warning: deprecated",) * 5),
            )

            processed = compactor.process(
                result,
                tool_source="native",
                request_budget_report=_report(current_tokens=1_000),
            )

        self.assertIn("warning: deprecated [repeated 5 times]", processed.result.content)
        self.assertIn("tool_output_normalized=true", processed.result.refs)

    def test_destructive_normalization_exposes_retrieval_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ToolOutputStore.in_data_dir(root, "default", "session-a")
            compactor = ToolOutputCompactor(
                store=store,
                policy=ConversationBudgetPolicy(
                    model_context_tokens=100_000,
                    response_token_reserve=0,
                    safety_margin_tokens=0,
                ),
            )
            result = ModelToolResult(
                name="pytest",
                request_id="call-1",
                content="\x1b[31mFAILED\x1b[0m tests/test_example.py",
            )

            processed = compactor.process(
                result,
                tool_source="native",
                request_budget_report=_report(current_tokens=1_000),
            )
            ref = _ref_from(processed.result.refs)
            loaded = store.load(ref)

        self.assertIn("FAILED tests/test_example.py", processed.result.content)
        self.assertIn("Raw output ref:", processed.result.content)
        self.assertIn(f'retrieve_tool_output(ref="{ref}")', processed.result.content)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.content, "\x1b[31mFAILED\x1b[0m tests/test_example.py")

    def test_large_output_compacts_when_normalized_output_crosses_pressure_band(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ToolOutputStore.in_data_dir(root, "default", "session-a")
            compactor = ToolOutputCompactor(
                store=store,
                policy=ConversationBudgetPolicy(
                    model_context_tokens=100_000,
                    response_token_reserve=0,
                    safety_margin_tokens=0,
                ),
            )
            content = "\n".join(
                f"FAILED tests/test_example.py::test_{index} AssertionError: value {index}"
                for index in range(4_000)
            )
            result = ModelToolResult(
                name="pytest",
                request_id="call-1",
                content=content,
            )

            processed = compactor.process(
                result,
                tool_source="native",
                request_budget_report=_report(current_tokens=49_000),
            )
            ref = _ref_from(processed.result.refs)

            self.assertIsNotNone(store.load(ref))

        self.assertIn("[tool output compacted: log]", processed.result.content)
        self.assertIn("Retrieval ref:", processed.result.content)
        self.assertIn("Suppressed:", processed.result.content)
        self.assertIn("tool_output_compacted=true", processed.result.refs)
        self.assertEqual([event.kind for event in processed.events], ["tool_output_compacted"])

    def test_single_traceback_stays_raw_when_headroom_is_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            compactor = _compactor(Path(tmp))
            content = "\n".join(
                (
                    "Traceback (most recent call last):",
                    "  File \"test.py\", line 1, in <module>",
                    "AssertionError: expected true",
                )
            )
            result = ModelToolResult(name="pytest", request_id="call-1", content=content)

            processed = compactor.process(
                result,
                tool_source="native",
                request_budget_report=_report(current_tokens=1_000),
            )

        self.assertEqual(processed.result.content, content)
        self.assertFalse(processed.events)

    def test_retrieved_tool_output_is_not_compacted_again(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            compactor = _compactor(Path(tmp))
            content = "\n".join(f"{index}: exact raw line" for index in range(4_000))
            result = ModelToolResult(
                name="retrieve_tool_output",
                request_id="call-1",
                content=content,
                refs=(
                    "tool_output_ref=out_000000000000",
                    "tool_output_retrieved_tokens=12000",
                ),
            )

            processed = compactor.process(
                result,
                tool_source="native",
                request_budget_report=_report(current_tokens=74_000),
            )

        self.assertEqual(processed.result.content, content)
        self.assertNotIn("tool_output_compacted=true", processed.result.refs)
        self.assertFalse(processed.events)

    def test_browser_snapshot_cleanup_saves_raw_output_for_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ToolOutputStore.in_data_dir(root, "default", "session-a")
            compactor = ToolOutputCompactor(
                store=store,
                policy=ConversationBudgetPolicy(
                    model_context_tokens=100_000,
                    response_token_reserve=0,
                    safety_margin_tokens=0,
                ),
            )
            snapshot = "\n".join(
                (
                    "Title: Example App",
                    "Navigation",
                    "Home",
                    "Home",
                    "@e1 button \"Log in\"",
                    "@e1 button \"Log in\"",
                    "@e2 textbox \"Email\"",
                    "Footer",
                    "Privacy Policy",
                    "Welcome to Example App",
                    "Welcome to Example App",
                    "data:image/png;base64," + "A" * 400,
                )
            )
            result = ModelToolResult(
                name="browser_snapshot",
                request_id="call-1",
                content=snapshot,
                refs=(
                    "browser_backend=agent-browser",
                    "browser_tool=browser_snapshot",
                    "browser_url=https://example.test",
                ),
            )

            processed = compactor.process(
                result,
                tool_source="native",
                request_budget_report=_report(current_tokens=1_000),
            )
            ref = _ref_from(processed.result.refs)
            loaded = store.load(ref)

        self.assertIn("[browser output normalized: snapshot]", processed.result.content)
        self.assertIn("title: Example App", processed.result.content)
        self.assertIn("url: https://example.test", processed.result.content)
        self.assertIn("@e1 button \"Log in\"", processed.result.content)
        self.assertEqual(processed.result.content.count("@e1 button \"Log in\""), 1)
        self.assertNotIn("data:image/png;base64", processed.result.content)
        self.assertIn("tool_output_normalized=true", processed.result.refs)
        self.assertIn("tool_output_kind=browser", processed.result.refs)
        self.assertEqual([event.kind for event in processed.events], ["tool_output_normalized"])
        self.assertIn("tool_output_ref=", " ".join(processed.events[0].refs))
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.content, snapshot)

    def test_large_browser_snapshot_cleanup_bounds_output_and_keeps_retrieval_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ToolOutputStore.in_data_dir(root, "default", "session-a")
            compactor = ToolOutputCompactor(
                store=store,
                policy=ConversationBudgetPolicy(
                    model_context_tokens=100_000,
                    response_token_reserve=0,
                    safety_margin_tokens=0,
                ),
            )
            snapshot = "\n".join(
                (
                    "Title: Large App",
                    *(
                        f"@e{index} button \"Action {index}\""
                        for index in range(4_000)
                    ),
                )
            )
            result = ModelToolResult(
                name="browser_snapshot",
                request_id="call-1",
                content=snapshot,
                refs=(
                    "browser_backend=agent-browser",
                    "browser_tool=browser_snapshot",
                    "browser_url=https://example.test/large",
                ),
            )

            processed = compactor.process(
                result,
                tool_source="native",
                request_budget_report=_report(current_tokens=49_000),
            )
            ref = _ref_from(processed.result.refs)
            loaded = store.load(ref)

        self.assertIn("[browser output normalized: snapshot]", processed.result.content)
        self.assertIn("browser snapshot lines omitted=3920", processed.result.content)
        self.assertIn("Raw output ref:", processed.result.content)
        self.assertIn("tool_output_normalized=true", processed.result.refs)
        self.assertIn("tool_output_kind=browser", processed.result.refs)
        self.assertNotIn("tool_output_compacted=true", processed.result.refs)
        self.assertIsNotNone(loaded)
        self.assertEqual([event.kind for event in processed.events], ["tool_output_normalized"])

    def test_browser_snapshot_still_compacts_when_cleanup_crosses_pressure_band(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ToolOutputStore.in_data_dir(root, "default", "session-a")
            compactor = ToolOutputCompactor(
                store=store,
                policy=ConversationBudgetPolicy(
                    model_context_tokens=20_000,
                    response_token_reserve=0,
                    safety_margin_tokens=0,
                ),
            )
            snapshot = "\n".join(
                (
                    "Title: Dense App",
                    *(
                        f"@e{index} button \"Action {index} with a long accessible label for testing compact pressure\""
                        for index in range(400)
                    ),
                    *(
                        f"Visible paragraph {index} with repeated product detail and long text for browser page pressure"
                        for index in range(400)
                    ),
                )
            )
            result = ModelToolResult(
                name="browser_snapshot",
                request_id="call-1",
                content=snapshot,
                refs=(
                    "browser_backend=agent-browser",
                    "browser_tool=browser_snapshot",
                    "browser_url=https://example.test/dense",
                ),
            )

            processed = compactor.process(
                result,
                tool_source="native",
                request_budget_report=RequestBudgetReport(
                    conversation_items=1,
                    tool_schema_count=0,
                    estimated_input_tokens=14_500,
                    estimated_system_tokens=0,
                    estimated_history_tokens=14_500,
                    estimated_tool_output_tokens=0,
                    estimated_tool_schema_tokens=0,
                    model_context_tokens=20_000,
                    response_token_reserve=0,
                    safety_margin_tokens=0,
                    usable_input_tokens=20_000,
                    pressure_ratio=14_500 / 20_000,
                    tool_output_ratio=0.0,
                ),
            )
            ref = _ref_from(processed.result.refs)
            loaded = store.load(ref)

        self.assertIn("[tool output compacted: browser]", processed.result.content)
        self.assertIn("Retrieval ref:", processed.result.content)
        self.assertIn("tool_output_compacted=true", processed.result.refs)
        self.assertIsNotNone(loaded)
        self.assertEqual([event.kind for event in processed.events], ["tool_output_compacted"])
        self.assertIn("browser_url=https://example.test/dense", processed.events[0].refs)

    def test_compaction_policy_can_tighten_large_noisy_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ToolOutputStore.in_data_dir(root, "default", "session-a")
            budget_policy = ConversationBudgetPolicy(
                model_context_tokens=1_000_000,
                response_token_reserve=0,
                safety_margin_tokens=0,
            )
            content = "\n".join(
                f"warning item {index} detail" for index in range(1_000)
            )
            result = ModelToolResult(
                name="pytest",
                request_id="call-1",
                content=content,
            )
            default_processed = ToolOutputCompactor(
                store=store,
                policy=budget_policy,
            ).process(
                result,
                tool_source="native",
                request_budget_report=_large_report(current_tokens=250_000),
            )
            tightened_processed = ToolOutputCompactor(
                store=store,
                policy=budget_policy,
                compaction_policy=ToolOutputCompactionPolicy(
                    medium_output_ratio=0.004,
                ),
            ).process(
                result,
                tool_source="native",
                request_budget_report=_large_report(current_tokens=250_000),
            )

        self.assertNotIn("tool_output_compacted=true", default_processed.result.refs)
        self.assertIn("tool_output_compacted=true", tightened_processed.result.refs)
        self.assertEqual(
            [event.kind for event in tightened_processed.events],
            ["tool_output_compacted"],
        )


def _compactor(root: Path) -> ToolOutputCompactor:
    return ToolOutputCompactor(
        store=ToolOutputStore.in_data_dir(root, "default", "session-a"),
        policy=ConversationBudgetPolicy(
            model_context_tokens=100_000,
            response_token_reserve=0,
            safety_margin_tokens=0,
        ),
    )


def _report(current_tokens: int) -> RequestBudgetReport:
    return RequestBudgetReport(
        conversation_items=1,
        tool_schema_count=0,
        estimated_input_tokens=current_tokens,
        estimated_system_tokens=0,
        estimated_history_tokens=current_tokens,
        estimated_tool_output_tokens=0,
        estimated_tool_schema_tokens=0,
        model_context_tokens=100_000,
        response_token_reserve=0,
        safety_margin_tokens=0,
        usable_input_tokens=100_000,
        pressure_ratio=current_tokens / 100_000,
        tool_output_ratio=0.0,
    )


def _large_report(current_tokens: int) -> RequestBudgetReport:
    return RequestBudgetReport(
        conversation_items=1,
        tool_schema_count=0,
        estimated_input_tokens=current_tokens,
        estimated_system_tokens=0,
        estimated_history_tokens=current_tokens,
        estimated_tool_output_tokens=0,
        estimated_tool_schema_tokens=0,
        model_context_tokens=1_000_000,
        response_token_reserve=0,
        safety_margin_tokens=0,
        usable_input_tokens=1_000_000,
        pressure_ratio=current_tokens / 1_000_000,
        tool_output_ratio=0.0,
    )


def _ref_from(refs: tuple[str, ...]) -> str:
    for ref in refs:
        if ref.startswith("tool_output_ref="):
            return ref.split("=", 1)[1]
    raise AssertionError("tool_output_ref missing")


if __name__ == "__main__":
    unittest.main()
