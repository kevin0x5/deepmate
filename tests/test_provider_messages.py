from __future__ import annotations

import unittest

from deepmate.providers import (
    ModelResponse,
    ModelToolExchange,
    ModelToolRequest,
    ModelToolResult,
)


class ProviderMessagesTest(unittest.TestCase):
    def test_tool_request_ready_handles_non_string_name(self) -> None:
        request = ModelToolRequest(name=None)  # type: ignore[arg-type]

        self.assertFalse(request.is_ready())

    def test_tool_request_ready_requires_tool_call_id(self) -> None:
        request = ModelToolRequest(name="read_text_file")

        self.assertFalse(request.is_ready())

    def test_tool_result_ready_handles_non_string_fields(self) -> None:
        result = ModelToolResult(  # type: ignore[arg-type]
            name=None,
            content=None,
        )

        self.assertFalse(result.is_ready())

    def test_tool_result_ready_requires_request_id(self) -> None:
        result = ModelToolResult(
            name="read_text_file",
            content="ok",
        )

        self.assertFalse(result.is_ready())

    def test_tool_exchange_ready_handles_non_string_ids(self) -> None:
        exchange = ModelToolExchange(
            tool_requests=(
                ModelToolRequest(  # type: ignore[arg-type]
                    name="read_text_file",
                    id=None,
                ),
            ),
            tool_results=(
                ModelToolResult(  # type: ignore[arg-type]
                    name="read_text_file",
                    request_id=None,
                    content="ok",
                ),
            ),
        )

        self.assertFalse(exchange.is_ready())

    def test_model_response_has_output_handles_non_string_text(self) -> None:
        response = ModelResponse(  # type: ignore[arg-type]
            content=None,
            reasoning=None,
        )

        self.assertFalse(response.has_output())


if __name__ == "__main__":
    unittest.main()
