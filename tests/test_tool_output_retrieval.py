from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deepmate.storage import ToolOutputStore
from deepmate.tools import RETRIEVE_TOOL_OUTPUT_NAME, NativeToolRegistry, tool_output_tools


class ToolOutputRetrievalTests(unittest.TestCase):
    def test_retrieve_by_ref_and_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolOutputStore.in_data_dir(Path(tmp), "default", "session-a")
            record = store.save(
                tool_name="pytest",
                tool_source="native",
                content_kind="log",
                content="\n".join(
                    (
                        "line 1 ok",
                        "line 2 ok",
                        "AssertionError: expected true",
                        "line 4 traceback",
                    )
                ),
                estimated_tokens=20,
                request_id="call-1",
            )
            registry = NativeToolRegistry(tool_output_tools(store))
            tool = registry.get(RETRIEVE_TOOL_OUTPUT_NAME)
            self.assertIsNotNone(tool)

            result = tool.call({"ref": record.ref, "query": "AssertionError"})

        self.assertIn("[tool output retrieved]", result.content)
        self.assertIn("AssertionError", result.content)
        self.assertIn("matched_chunks=1", result.refs)

    def test_missing_ref_returns_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolOutputStore.in_data_dir(Path(tmp), "default", "session-a")
            registry = NativeToolRegistry(tool_output_tools(store))
            tool = registry.get(RETRIEVE_TOOL_OUTPUT_NAME)
            self.assertIsNotNone(tool)

            result = tool.call({"ref": "out_000000000000"})

        self.assertIn("tool_output_ref_not_found", result.content)
        self.assertIn("tool_output_ref_not_found", result.refs)


if __name__ == "__main__":
    unittest.main()
