from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepmate.mcp import McpServerSpec, McpTransport, create_mcp_client
from deepmate.mcp.client import (
    MAX_MCP_INVALID_JSON_LINES,
    MAX_MCP_JSON_BYTES,
    _default_environment,
)


class McpHttpClientTests(unittest.TestCase):
    def test_streamable_http_client_lists_and_calls_tools(self) -> None:
        transport = _FakeHttpTransport()
        with patch("urllib.request.urlopen", transport.urlopen):
            with tempfile.TemporaryDirectory() as tmp:
                spec = McpServerSpec(
                    name="remote",
                    transport=McpTransport.STREAMABLE_HTTP,
                    url="https://mcp.example.test/mcp",
                )
                with create_mcp_client(spec, Path(tmp)) as client:
                    tools = client.list_tools()
                    result = client.call_tool("echo", {"text": "hello"})

        self.assertEqual(tuple(tool.name for tool in tools), ("echo",))
        self.assertEqual(tools[0].description, "Echo text.")
        self.assertEqual(tools[0].annotations, {"readOnlyHint": True})
        self.assertEqual(result.content, "hello")
        self.assertFalse(result.is_error)
        self.assertIn("remote", client.server_info().get("name", ""))
        self.assertIn("Remote server instructions.", client.instructions())
        self.assertEqual(
            transport.methods,
            ["initialize", "notifications/initialized", "tools/list", "tools/call"],
        )
        self.assertTrue(transport.delete_seen)
        self.assertEqual(transport.session_headers.get("tools/list"), "session-1")
        self.assertIn("text/event-stream", transport.accept_headers.get("initialize", ""))

    def test_streamable_http_client_accepts_sse_jsonrpc_response(self) -> None:
        transport = _FakeHttpTransport(sse_methods={"tools/list"})
        with patch("urllib.request.urlopen", transport.urlopen):
            with tempfile.TemporaryDirectory() as tmp:
                spec = McpServerSpec(
                    name="remote",
                    transport=McpTransport.SSE,
                    url="https://mcp.example.test/mcp",
                )
                with create_mcp_client(spec, Path(tmp)) as client:
                    tools = client.list_tools()

        self.assertEqual(tuple(tool.name for tool in tools), ("echo",))
        self.assertEqual(
            transport.methods,
            ["initialize", "notifications/initialized", "tools/list"],
        )

    def test_streamable_http_client_rejects_oversized_response(self) -> None:
        transport = _FakeHttpTransport(oversized_methods={"initialize"})
        with patch("urllib.request.urlopen", transport.urlopen):
            with tempfile.TemporaryDirectory() as tmp:
                spec = McpServerSpec(
                    name="remote",
                    transport=McpTransport.STREAMABLE_HTTP,
                    url="https://mcp.example.test/mcp",
                )
                with self.assertRaisesRegex(RuntimeError, "response payload is too large"):
                    with create_mcp_client(spec, Path(tmp)):
                        pass

    def test_streamable_http_client_rejects_oversized_request(self) -> None:
        transport = _FakeHttpTransport()
        with patch("urllib.request.urlopen", transport.urlopen):
            with tempfile.TemporaryDirectory() as tmp:
                spec = McpServerSpec(
                    name="remote",
                    transport=McpTransport.STREAMABLE_HTTP,
                    url="https://mcp.example.test/mcp",
                )
                with create_mcp_client(spec, Path(tmp)) as client:
                    with self.assertRaisesRegex(RuntimeError, "request payload is too large"):
                        client.call_tool("echo", {"text": "x" * MAX_MCP_JSON_BYTES})

        self.assertNotIn("tools/call", transport.methods)

    def test_streamable_http_client_enforces_request_rate_limit(self) -> None:
        transport = _FakeHttpTransport()
        with (
            patch("urllib.request.urlopen", transport.urlopen),
            patch("deepmate.mcp.client.MAX_MCP_REQUESTS_PER_MINUTE", 2),
            tempfile.TemporaryDirectory() as tmp,
        ):
            spec = McpServerSpec(
                name="remote",
                transport=McpTransport.STREAMABLE_HTTP,
                url="https://mcp.example.test/mcp",
            )
            with create_mcp_client(spec, Path(tmp)) as client:
                with self.assertRaisesRegex(RuntimeError, "rate limit exceeded"):
                    client.list_tools()

    def test_stdio_client_rejects_oversized_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = Path(tmp) / "mcp_server.py"
            server.write_text(
                (
                    "import sys\n"
                    "for line in sys.stdin:\n"
                    "    sys.stdout.write('x' * "
                    f"{MAX_MCP_JSON_BYTES + 1}"
                    " + '\\n')\n"
                    "    sys.stdout.flush()\n"
                ),
                encoding="utf-8",
            )
            spec = McpServerSpec(
                name="local",
                transport=McpTransport.STDIO,
                command="python3",
                args=(str(server),),
            )

            with self.assertRaisesRegex(RuntimeError, "response payload is too large"):
                with create_mcp_client(spec, Path(tmp)):
                    pass

    def test_stdio_client_rejects_repeated_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = Path(tmp) / "mcp_server.py"
            server.write_text(
                (
                    "import sys\n"
                    "for line in sys.stdin:\n"
                    f"    for _ in range({MAX_MCP_INVALID_JSON_LINES + 1}):\n"
                    "        sys.stdout.write('not-json\\n')\n"
                    "        sys.stdout.flush()\n"
                ),
                encoding="utf-8",
            )
            spec = McpServerSpec(
                name="local",
                transport=McpTransport.STDIO,
                command="python3",
                args=(str(server),),
            )

            with (
                patch("deepmate.mcp.client.MAX_MCP_INVALID_JSON_LINES", 0),
                self.assertRaisesRegex(RuntimeError, "too many invalid JSON"),
            ):
                with create_mcp_client(spec, Path(tmp)):
                    pass

    def test_default_environment_inherits_locale(self) -> None:
        with patch.dict("os.environ", {"LANG": "en_US.UTF-8", "LC_ALL": "C.UTF-8"}):
            env = _default_environment()

        self.assertEqual(env["LANG"], "en_US.UTF-8")
        self.assertEqual(env["LC_ALL"], "C.UTF-8")


class _FakeHttpTransport:
    def __init__(
        self,
        *,
        sse_methods: set[str] | None = None,
        oversized_methods: set[str] | None = None,
    ) -> None:
        self.sse_methods = sse_methods or set()
        self.oversized_methods = oversized_methods or set()
        self.methods: list[str] = []
        self.accept_headers: dict[str, str] = {}
        self.session_headers: dict[str, str] = {}
        self.delete_seen = False

    def urlopen(self, request, timeout):
        if request.get_method() == "DELETE":
            self.delete_seen = True
            return _FakeHttpResponse(204, b"", {})
        payload = json.loads(request.data.decode("utf-8"))
        method = payload.get("method", "")
        if isinstance(method, str):
            self.methods.append(method)
            self.accept_headers[method] = request.headers.get("Accept", "")
            self.session_headers[method] = request.headers.get("Mcp-session-id", "")
        if "id" not in payload:
            return _FakeHttpResponse(202, b"", {})
        response = self.response_for(payload)
        headers = {}
        if method in self.oversized_methods:
            raw = b"x" * 1_100_000
            headers["Content-Type"] = "application/json"
        elif method in self.sse_methods:
            raw = (
                "event: message\n"
                f"data: {json.dumps(response, separators=(',', ':'))}\n\n"
            ).encode("utf-8")
            headers["Content-Type"] = "text/event-stream"
        else:
            raw = json.dumps(response, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if method == "initialize":
            headers["Mcp-Session-Id"] = "session-1"
        return _FakeHttpResponse(200, raw, headers)

    def response_for(self, payload: dict[str, object]) -> dict[str, object]:
        request_id = payload["id"]
        method = payload.get("method")
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "serverInfo": {"name": "remote-test", "version": "1.0"},
                    "capabilities": {"tools": {}},
                    "instructions": "Remote server instructions.",
                },
            }
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echo text.",
                            "inputSchema": {"type": "object"},
                            "annotations": {"readOnlyHint": True},
                        }
                    ]
                },
            }
        if method == "tools/call":
            params = payload.get("params")
            arguments = params.get("arguments", {}) if isinstance(params, dict) else {}
            text = arguments.get("text", "") if isinstance(arguments, dict) else ""
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": str(text)}]},
            }
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"unknown method: {method}"},
        }


class _FakeHttpResponse:
    def __init__(self, status: int, body: bytes, headers: dict[str, str]) -> None:
        self.status = status
        self._body = body
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self._body)
        return self._body[:size]

    def close(self) -> None:
        return None


if __name__ == "__main__":
    unittest.main()
