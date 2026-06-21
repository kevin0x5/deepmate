from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepmate.domain import Message, MessageRole
from deepmate.providers import (
    ChatCompletionsProvider,
    ModelCapabilities,
    ModelConversationItem,
    ModelRequest,
    ModelToolExchange,
    ModelToolRequest,
    ModelToolResult,
    NetworkError,
    ProviderError,
    RateLimitError,
)
from deepmate.providers.chat_completions import (
    MAX_MODEL_RESPONSE_BYTES,
    MAX_MODEL_STREAM_LINE_BYTES,
    _http_error,
    _post_json,
    _read_error_detail,
    _request_payload,
    sanitize_model_request,
)


class ChatCompletionsProviderTests(unittest.TestCase):
    def test_request_payload_rejects_reserved_native_mcp_name(self) -> None:
        request = _request_with_tools(
            (
                {
                    "name": "mcp__filesystem__read_file",
                    "description": "Reserved-looking native tool.",
                    "input_schema": {"type": "object"},
                },
            )
        )

        with self.assertRaisesRegex(ValueError, "reserved MCP provider namespace"):
            _request_payload(request)

    def test_request_payload_rejects_encoded_tool_name_collision(self) -> None:
        request = _request_with_tools(
            (
                {
                    "name": "server.tool",
                    "description": "MCP-style runtime tool.",
                    "input_schema": {"type": "object"},
                },
                {
                    "type": "function",
                    "function": {
                        "name": "mcp__server__tool",
                        "description": "Already encoded provider tool.",
                        "parameters": {"type": "object"},
                    },
                },
            )
        )

        with self.assertRaisesRegex(ValueError, "duplicate provider tool name"):
            _request_payload(request)

    def test_post_json_wraps_response_read_timeout_as_network_error(self) -> None:
        class SlowResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def getcode(self):
                return 200

            def read(self, _size=-1):
                raise TimeoutError("The read operation timed out")

        with patch(
            "deepmate.providers.chat_completions.urlopen",
            return_value=SlowResponse(),
        ):
            with self.assertRaises(NetworkError) as raised:
                _post_json(
                    url="https://api.example.test/chat/completions",
                    api_key="key",
                    payload={"model": "stub", "messages": []},
                    timeout=1,
                )

        self.assertIn("timed out", str(raised.exception))

    def test_post_json_rejects_oversized_response(self) -> None:
        class HugeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def getcode(self):
                return 200

            def read(self, size=-1):
                return b"{" + (b'"x":' + b'"' + (b"a" * size) + b'"')

        with patch(
            "deepmate.providers.chat_completions.urlopen",
            return_value=HugeResponse(),
        ):
            with self.assertRaisesRegex(ProviderError, "model response exceeds"):
                _post_json(
                    url="https://api.example.test/chat/completions",
                    api_key="key",
                    payload={"model": "stub", "messages": []},
                    timeout=1,
                )

    def test_http_error_redacts_secret_details(self) -> None:
        error = _http_error(
            400,
            'bad request: api_key="sk_live_secret" Authorization: Bearer abcdefghijklmnop',
        )

        text = str(error)
        self.assertIn("HTTP 400", text)
        self.assertIn("api_key=\"[redacted]\"", text)
        self.assertIn("Bearer [redacted]", text)
        self.assertNotIn("sk_live_secret", text)
        self.assertNotIn("abcdefghijklmnop", text)

    def test_http_error_redacts_short_bearer_token(self) -> None:
        error = _http_error(400, "Authorization: Bearer abc123")

        text = str(error)
        self.assertIn("Bearer [redacted]", text)
        self.assertNotIn("abc123", text)

    def test_http_error_redacts_bare_sk_api_key(self) -> None:
        error = _http_error(400, "provider echoed sk-test1234567890abcdef")

        text = str(error)
        self.assertIn("sk-[redacted]", text)
        self.assertNotIn("sk-test1234567890abcdef", text)

    def test_http_error_preserves_retry_after_for_rate_limit(self) -> None:
        error = _http_error(429, "rate limited", headers={"Retry-After": "2.5"})

        self.assertIsInstance(error, RateLimitError)
        self.assertEqual(error.retry_after_seconds, 2.5)

    def test_complete_stream_accumulates_deltas_and_tool_calls(self) -> None:
        lines = [
            'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}',
            'data: {"choices":[{"delta":{"content":"Hel"}}]}',
            'data: {"choices":[{"delta":{"content":"lo"}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1",'
            '"function":{"name":"search","arguments":"{\\"q\\":"}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"function":{"arguments":"\\"hi\\"}"}}]}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
            'data: {"usage":{"prompt_tokens":7,"completion_tokens":3}}',
            "data: [DONE]",
        ]

        class FakeStream:
            def __init__(self) -> None:
                self._lines = [f"{line}\n".encode("utf-8") for line in lines]

            def getcode(self):
                return 200

            def __iter__(self):
                return iter(self._lines)

            def close(self):
                return None

        deltas = []
        provider = ChatCompletionsProvider(base_url="https://x.test", api_key="k")
        with patch(
            "deepmate.providers.chat_completions.urlopen",
            return_value=FakeStream(),
        ):
            response = provider.complete_stream(
                _request_with_tools(()), deltas.append
            )

        self.assertEqual("".join(d.content for d in deltas), "Hello")
        self.assertEqual("".join(d.reasoning for d in deltas), "think")
        self.assertEqual(response.content, "Hello")
        self.assertEqual(response.reasoning, "think")
        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertEqual(len(response.tool_requests), 1)
        self.assertEqual(response.tool_requests[0].name, "search")
        self.assertEqual(response.tool_requests[0].arguments, {"q": "hi"})
        self.assertEqual(response.tool_requests[0].id, "c1")
        self.assertIsNotNone(response.usage)
        self.assertEqual(response.usage.input_tokens, 7)

    def test_complete_stream_accumulates_tool_calls_without_index(self) -> None:
        lines = [
            'data: {"choices":[{"delta":{"tool_calls":[{"id":"c1",'
            '"function":{"name":"search","arguments":"{\\"q\\":"}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"\\"hi\\"}"}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"id":"c2",'
            '"function":{"name":"read_text_file","arguments":"{\\"path\\":"}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"\\"README.md\\"}"}}]}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]

        class FakeStream:
            def __iter__(self):
                return iter(f"{line}\n".encode("utf-8") for line in lines)

            def getcode(self):
                return 200

            def close(self):
                return None

        provider = ChatCompletionsProvider(base_url="https://x.test", api_key="k")
        with patch(
            "deepmate.providers.chat_completions.urlopen",
            return_value=FakeStream(),
        ):
            response = provider.complete_stream(
                _request_with_tools(()),
                lambda _delta: None,
            )

        self.assertEqual(
            tuple(request.name for request in response.tool_requests),
            ("search", "read_text_file"),
        )
        self.assertEqual(response.tool_requests[0].arguments, {"q": "hi"})
        self.assertEqual(response.tool_requests[1].arguments, {"path": "README.md"})

    def test_sanitize_model_request_applies_model_capabilities(self) -> None:
        request = ModelRequest(
            model="local",
            conversation=(
                ModelConversationItem.from_message(
                    Message(role=MessageRole.SYSTEM, content="System context.")
                ),
                ModelConversationItem.from_tool_exchange(
                    ModelToolExchange(
                        assistant_reasoning="hidden",
                        tool_requests=(
                            ModelToolRequest(name="search", id="call_1"),
                        ),
                        tool_results=(
                            ModelToolResult(
                                name="search",
                                request_id="call_1",
                                content="ok",
                            ),
                        ),
                    )
                ),
                ModelConversationItem.from_tool_exchange(
                    ModelToolExchange(
                        tool_requests=(
                            ModelToolRequest(name="broken", id="call_2"),
                        ),
                        tool_results=(),
                    )
                ),
            ),
            tool_schemas=(
                {
                    "name": "search",
                    "description": "Search.",
                    "input_schema": {"type": "object"},
                },
            ),
            options={
                "thinking": {"type": "enabled"},
                "reasoning_effort": "high",
                "stream_options": {"include_usage": True},
                "max_tokens": 32,
            },
            capabilities=ModelCapabilities(
                supports_tools=False,
                supports_thinking=False,
                supports_stream_usage=False,
            ),
        )

        sanitized = sanitize_model_request(request)

        self.assertEqual(sanitized.tool_schemas, ())
        self.assertEqual(len(sanitized.conversation), 1)
        self.assertEqual(sanitized.options, {"max_tokens": 32})

    def test_request_payload_appends_image_attachment_when_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "screen.png"
            image_path.write_bytes(
                b"\x89PNG\r\n\x1a\n"
                b"\x00\x00\x00\rIHDR"
                b"\x00\x00\x00\x01"
                b"\x00\x00\x00\x01"
                b"\x08\x02\x00\x00\x00"
                b"\x00\x00\x00\x00"
            )
            request = ModelRequest(
                model="vision-model",
                conversation=(
                    ModelConversationItem.from_message(
                        Message(role=MessageRole.SYSTEM, content="System context.")
                    ),
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
                                            "path": str(image_path),
                                            "mime_type": "image/png",
                                        },
                                    ),
                                ),
                            ),
                        )
                    ),
                ),
            )

            payload = _request_payload(request)

        messages = payload["messages"]
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[2]["role"], "tool")
        self.assertEqual(messages[3]["role"], "user")
        content = messages[3]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_request_payload_omits_image_attachment_when_not_supported(self) -> None:
        request = ModelRequest(
            model="text-model",
            conversation=(
                ModelConversationItem.from_message(
                    Message(role=MessageRole.SYSTEM, content="System context.")
                ),
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
                                        "path": "/tmp/missing.png",
                                        "mime_type": "image/png",
                                    },
                                ),
                            ),
                        ),
                    )
                ),
            ),
            capabilities=ModelCapabilities(supports_image_input=False),
        )

        payload = _request_payload(request)

        self.assertEqual(len(payload["messages"]), 3)
        self.assertEqual(payload["messages"][2]["role"], "tool")

    def test_sanitize_model_request_preserves_explicit_stream_false(self) -> None:
        request = ModelRequest(
            model="local",
            conversation=(
                ModelConversationItem.from_message(
                    Message(role=MessageRole.USER, content="Hello.")
                ),
            ),
            options={"stream": False, "max_tokens": 32},
        )

        sanitized = sanitize_model_request(request)

        self.assertEqual(sanitized.options, {"stream": False, "max_tokens": 32})

    def test_read_error_detail_truncates_by_bytes(self) -> None:
        class FakeHttpError(Exception):
            def read(self, size=-1):
                return "é".encode("utf-8") * size

        detail = _read_error_detail(FakeHttpError())  # type: ignore[arg-type]

        self.assertLessEqual(len(detail.encode("utf-8")), 800)
        self.assertNotIn("\ufffd", detail)

    def test_sanitize_model_request_can_strip_assistant_reasoning_replay(self) -> None:
        request = ModelRequest(
            model="local",
            conversation=(
                ModelConversationItem.from_message(
                    Message(role=MessageRole.SYSTEM, content="System context.")
                ),
                ModelConversationItem.from_tool_exchange(
                    ModelToolExchange(
                        assistant_reasoning="hidden",
                        tool_requests=(ModelToolRequest(name="search", id="call_1"),),
                        tool_results=(
                            ModelToolResult(
                                name="search",
                                request_id="call_1",
                                content="ok",
                            ),
                        ),
                    )
                ),
            ),
            capabilities=ModelCapabilities(supports_assistant_reasoning_replay=False),
        )

        sanitized = sanitize_model_request(request)

        exchange = sanitized.conversation[1].tool_exchange
        self.assertIsNotNone(exchange)
        self.assertEqual(exchange.assistant_reasoning, "")

    def test_complete_stream_omits_stream_usage_when_capability_disabled(self) -> None:
        captured = {}

        class FakeStream:
            def __iter__(self):
                return iter((b"data: [DONE]\n",))

            def getcode(self):
                return 200

            def close(self):
                return None

        def fake_urlopen(request, timeout):
            captured["body"] = request.data.decode("utf-8")
            return FakeStream()

        provider = ChatCompletionsProvider(base_url="https://x.test", api_key="k")
        request = _request_with_tools(())
        request = ModelRequest(
            model=request.model,
            conversation=request.conversation,
            capabilities=ModelCapabilities(supports_stream_usage=False),
        )
        with patch("deepmate.providers.chat_completions.urlopen", fake_urlopen):
            provider.complete_stream(request, lambda _delta: None)

        self.assertNotIn("stream_options", captured["body"])
        self.assertIn('"stream": true', captured["body"])

    def test_complete_stream_closes_error_response(self) -> None:
        class ErrorStream:
            def __init__(self) -> None:
                self.closed = False

            def getcode(self):
                return 400

            def read(self):
                return b"bad request"

            def close(self):
                self.closed = True

        stream = ErrorStream()
        provider = ChatCompletionsProvider(base_url="https://x.test", api_key="k")
        with patch(
            "deepmate.providers.chat_completions.urlopen",
            return_value=stream,
        ):
            with self.assertRaises(Exception):
                provider.complete_stream(_request_with_tools(()), lambda _d: None)

        self.assertTrue(stream.closed)

    def test_complete_stream_raises_terminal_error_on_mid_stream_break(self) -> None:
        class BrokenStream:
            def getcode(self):
                return 200

            def __iter__(self):
                yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n'
                raise TimeoutError("connection dropped mid-stream")

            def close(self):
                return None

        provider = ChatCompletionsProvider(base_url="https://x.test", api_key="k")
        with patch(
            "deepmate.providers.chat_completions.urlopen",
            return_value=BrokenStream(),
        ):
            # Mid-stream failures are terminal (not the retryable NetworkError),
            # so the retry loop won't re-run and double the streamed output.
            with self.assertRaises(Exception) as raised:
                provider.complete_stream(_request_with_tools(()), lambda _d: None)
        self.assertNotIsInstance(raised.exception, NetworkError)
        self.assertIn("interrupted", str(raised.exception))

    def test_complete_stream_rejects_oversized_line(self) -> None:
        class HugeStream:
            def getcode(self):
                return 200

            def readline(self, size=-1):
                if size == 0:
                    return b""
                self.readline = lambda _size=-1: b""
                return b"data: " + (b"x" * MAX_MODEL_STREAM_LINE_BYTES)

            def close(self):
                return None

        provider = ChatCompletionsProvider(base_url="https://x.test", api_key="k")
        with patch(
            "deepmate.providers.chat_completions.urlopen",
            return_value=HugeStream(),
        ):
            with self.assertRaisesRegex(ProviderError, "model stream line exceeds"):
                provider.complete_stream(_request_with_tools(()), lambda _d: None)


def _request_with_tools(
    schemas: tuple[dict[str, object], ...],
) -> ModelRequest:
    return ModelRequest(
        model="stub-model",
        conversation=(
            ModelConversationItem.from_message(
                Message(role=MessageRole.SYSTEM, content="System context.")
            ),
        ),
        tool_schemas=schemas,
    )


if __name__ == "__main__":
    unittest.main()
