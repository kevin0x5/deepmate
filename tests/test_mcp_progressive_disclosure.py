from __future__ import annotations

import json
import io
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from deepmate.capabilities import from_mcp_tool_catalog
from deepmate.domain import (
    CapabilityKind,
    Message,
    MessageRole,
    ProfileRef,
    RuntimeEvent,
)
from deepmate.mcp import (
    McpCallResult,
    McpServerInventory,
    McpServerSpec,
    McpToolCatalog,
    McpToolExecutionResult,
    McpToolRef,
    McpTransport,
    McpUsageStateStore,
    format_mcp_catalog_status,
)
from deepmate.mcp.client import McpClientError, McpStdioClient
from deepmate.providers import ModelResponse, ModelToolRequest, ModelToolResult
from deepmate.runtime import run_user_turn
from deepmate.tools import LOAD_MCP_TOOL_NAME, NativeToolRegistry, mcp_loader_tools
from deepmate.trace import TraceRecorder


class _PagedMcpClient(McpStdioClient):
    def __init__(self) -> None:
        super().__init__(
            McpServerSpec(
                name="filesystem",
                transport=McpTransport.STDIO,
                command="server",
            ),
            "/workspace",
        )
        self.requests = []

    def _request(self, method, params):
        self.requests.append((method, dict(params)))
        if len(self.requests) == 1:
            return {
                "result": {
                    "tools": [
                        {
                            "name": "read_text_file",
                            "title": "Read text",
                            "description": "Read a file.",
                            "inputSchema": {"type": "object"},
                            "outputSchema": {"type": "object"},
                            "annotations": {"readOnlyHint": True},
                            "_meta": {"source": "test"},
                        }
                    ],
                    "nextCursor": "page-2",
                }
            }
        return {
            "result": {
                "tools": [
                    {
                        "name": "list_directory",
                        "annotations": {"readOnlyHint": True},
                    }
                ]
            }
        }


class _FakeMcpProcess:
    def __init__(self, *, stdin=None, stdout=None) -> None:
        self.stdin = stdin if stdin is not None else io.StringIO()
        self.stdout = stdout if stdout is not None else io.StringIO()
        self.stderr = io.StringIO()
        self.returncode = None

    def poll(self):
        return self.returncode


class _BoundaryMcpClient(McpStdioClient):
    def __init__(self) -> None:
        super().__init__(
            McpServerSpec(
                name="filesystem",
                transport=McpTransport.STDIO,
                command="server",
            ),
            "/workspace",
        )
        self._process = _FakeMcpProcess()

    def _request_timeout_seconds(self) -> float:
        return 0.01


class _StubProvider:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("stub provider received too many requests")
        return self.responses.pop(0)


class _FakeMcpExecutor:
    def __init__(self) -> None:
        self.calls: list[ModelToolRequest] = []

    def has_tool(self, name: str) -> bool:
        return name.strip() == "filesystem.read_text_file"

    def tool_schema(self, name: str):
        if not self.has_tool(name):
            return None
        return _read_tool().schema()

    def execute(self, request: ModelToolRequest) -> McpToolExecutionResult:
        self.calls.append(request)
        return McpToolExecutionResult(
            request=request,
            model_result=ModelToolResult(
                name=request.name,
                request_id=request.id,
                content="file content",
                refs=("mcp_tool=filesystem.read_text_file",),
            ),
        )


class _InspectMetadataFakeMcpExecutor:
    def __init__(self) -> None:
        self.calls: list[ModelToolRequest] = []

    def has_tool(self, name: str) -> bool:
        return name.strip() == "filesystem.inspect_file_metadata"

    def tool_schema(self, name: str):
        if not self.has_tool(name):
            return None
        return _inspect_metadata_tool().schema()

    def execute(self, request: ModelToolRequest) -> McpToolExecutionResult:
        self.calls.append(request)
        return McpToolExecutionResult(
            request=request,
            model_result=ModelToolResult(
                name=request.name,
                request_id=request.id,
                content="metadata",
                refs=("mcp_tool=filesystem.inspect_file_metadata",),
            ),
        )


class _TruncatingFakeMcpExecutor(_FakeMcpExecutor):
    def execute(self, request: ModelToolRequest) -> McpToolExecutionResult:
        self.calls.append(request)
        return McpToolExecutionResult(
            request=request,
            model_result=ModelToolResult(
                name=request.name,
                request_id=request.id,
                content="[Deepmate MCP output truncated:]",
                refs=(
                    "mcp_tool=filesystem.read_text_file",
                    "mcp_output_truncated=true",
                ),
            ),
            events=(
                RuntimeEvent(
                    kind="mcp_tool_output_truncated",
                    summary="MCP tool output truncated: filesystem.read_text_file.",
                    refs=(
                        "mcp_tool=filesystem.read_text_file",
                        "mcp_output_truncated=true",
                    ),
                ),
            ),
        )


class _TraceSink:
    def __init__(self) -> None:
        self.events = []

    def write(self, event) -> None:
        self.events.append(event)


class McpProgressiveDisclosureTests(unittest.TestCase):
    def test_stdio_client_rejects_oversized_request_payload(self) -> None:
        client = _BoundaryMcpClient()
        with self.assertRaisesRegex(McpClientError, "request payload is too large"):
            client._write({"jsonrpc": "2.0", "params": {"text": "x" * 1_100_000}})

    def test_stdio_client_rejects_oversized_response_payload(self) -> None:
        client = _BoundaryMcpClient()
        client._process = _FakeMcpProcess(stdout=io.StringIO("x" * 1_100_000))
        with patch("deepmate.mcp.client._wait_readable", return_value=True):
            with self.assertRaisesRegex(McpClientError, "response payload is too large"):
                client._read()

    def test_stdio_client_rejects_stdout_eof_before_process_exit(self) -> None:
        client = _BoundaryMcpClient()
        client._process = _FakeMcpProcess(stdout=io.StringIO(""))
        with patch("deepmate.mcp.client._wait_readable", return_value=True):
            with self.assertRaisesRegex(McpClientError, "stdout reached EOF"):
                client._read()

    def test_stdio_client_enforces_request_rate_limit(self) -> None:
        client = _BoundaryMcpClient()
        client._request_timestamps = [100.0] * 600
        with patch("deepmate.mcp.client.time.monotonic", return_value=120.0):
            with self.assertRaisesRegex(McpClientError, "rate limit exceeded"):
                client._write({"jsonrpc": "2.0", "method": "ping"})

    def test_mutating_tool_name_overrides_read_only_hint(self) -> None:
        tool = McpToolRef(
            server_name="filesystem",
            name="write_file",
            description="Write a file.",
            annotations={"readOnlyHint": True},
        )

        self.assertFalse(tool.is_read_only())

    def test_risky_mcp_verbs_override_read_only_hint(self) -> None:
        risky_names = (
            "save_file",
            "send_message",
            "deploy_service",
            "install_package",
            "push_branch",
        )

        for name in risky_names:
            with self.subTest(name=name):
                tool = McpToolRef(
                    server_name="remote",
                    name=name,
                    description="Server claims this is read only.",
                    annotations={"readOnlyHint": True},
                )
                self.assertFalse(tool.is_read_only())

    def test_mcp_usage_float_counts_are_loaded_as_non_negative_ints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = McpUsageStateStore.in_data_dir(Path(tmp) / "var", "default")
            store.path.parent.mkdir(parents=True, exist_ok=True)
            store.path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "entries": [
                            {
                                "key": "tool:filesystem.read_text_file",
                                "kind": "tool",
                                "name": "filesystem.read_text_file",
                                "server_name": "filesystem",
                                "created_at": "2026-06-01T00:00:00+00:00",
                                "updated_at": "2026-06-01T00:00:00+00:00",
                                "invocation_count": 3.0,
                                "load_count": 2.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            entry = store.load()["tool:filesystem.read_text_file"]

            self.assertEqual(entry.invocation_count, 3)
            self.assertEqual(entry.load_count, 2)

    def test_mcp_usage_accepts_z_suffix_datetimes_for_idle_tiers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = McpUsageStateStore.in_data_dir(root / "var", "default")
            store.path.parent.mkdir(parents=True, exist_ok=True)
            store.path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "entries": [
                            {
                                "key": "server:filesystem",
                                "kind": "server",
                                "name": "filesystem",
                                "server_name": "filesystem",
                                "created_at": "2026-06-01T00:00:00Z",
                                "updated_at": "2026-06-01T00:00:00Z",
                                "last_seen_at": "2026-06-01T00:00:00Z",
                                "last_used_at": "2026-06-01T00:00:00Z",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            entry = store.load()["server:filesystem"]

        self.assertTrue(entry.is_idle(now=datetime(2026, 6, 8, 1, tzinfo=UTC)))

    def test_stdio_list_tools_follows_pagination_and_keeps_metadata(self) -> None:
        client = _PagedMcpClient()

        tools = client.list_tools()

        self.assertEqual(
            client.requests,
            [
                ("tools/list", {}),
                ("tools/list", {"cursor": "page-2"}),
            ],
        )
        self.assertEqual(
            [tool.name for tool in tools],
            ["read_text_file", "list_directory"],
        )
        self.assertEqual(tools[0].title, "Read text")
        self.assertEqual(tools[0].output_schema, {"type": "object"})
        self.assertEqual(tools[0].meta, {"source": "test"})

    def test_stdio_request_matches_string_jsonrpc_id_and_counts_skips(self) -> None:
        client = _BoundaryMcpClient()
        messages = iter(
            (
                {"jsonrpc": "2.0", "method": "notifications/message"},
                {"jsonrpc": "2.0", "id": 999, "result": None},
                {"jsonrpc": "2.0", "id": "1", "result": {"ok": True}},
            )
        )

        with (
            patch.object(client, "_write"),
            patch.object(client, "_read", side_effect=lambda: next(messages)),
        ):
            response = client._request("tools/list", {})

        self.assertEqual(response["result"], {"ok": True})
        self.assertIn("mcp_notifications_skipped=1", client.diagnostic_refs())
        self.assertIn("mcp_unmatched_responses_skipped=1", client.diagnostic_refs())

    def test_mcp_catalog_keeps_idle_server_tool_names_after_seven_days(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_store = McpUsageStateStore.in_data_dir(root / "var", "default")
            server = _server()
            tool = _read_tool()
            start = datetime(2026, 6, 1, tzinfo=UTC)
            state_store.sync_server_seen(server, now=start)
            catalog = McpToolCatalog(
                inventories=(McpServerInventory(server=server, tools=(tool,)),),
                state_store=state_store,
                now=start + timedelta(days=8),
            )

            surface = from_mcp_tool_catalog(catalog, model_context_tokens=1_000_000)

        refs = surface.list_refs()
        self.assertEqual(
            [(ref.kind, ref.name) for ref in refs],
            [
                (CapabilityKind.MCP_SERVER, "filesystem"),
                (CapabilityKind.MCP_TOOL, "filesystem.read_text_file"),
            ],
        )
        self.assertIn("tier=idle", refs[0].description)
        self.assertIn("7 days", refs[0].description)
        self.assertEqual(refs[1].description, "")

    def test_load_mcp_tool_adds_schema_for_followup_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            state_store = McpUsageStateStore.in_data_dir(workspace / "var", "default")
            server = _server()
            tool = _read_tool()
            catalog = McpToolCatalog(
                inventories=(McpServerInventory(server=server, tools=(tool,)),),
                state_store=state_store,
            )
            registry = NativeToolRegistry(mcp_loader_tools(catalog))
            mcp_executor = _FakeMcpExecutor()
            provider = _StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name=LOAD_MCP_TOOL_NAME,
                                arguments={"name": "filesystem.read_text_file"},
                                id="call_1",
                            ),
                        )
                    ),
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="filesystem.read_text_file",
                                arguments={"path": "README.md"},
                                id="call_2",
                            ),
                        )
                    ),
                    ModelResponse(content="done"),
                ]
            )

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="Read README."),),
                model="stub-model",
                native_tools=registry,
                mcp_tools=mcp_executor,  # type: ignore[arg-type]
                tool_schemas=registry.schemas(),
                max_steps=3,
            )
            loaded_state = state_store.tool_entry("filesystem.read_text_file")

        self.assertFalse(result.has_errors())
        self.assertEqual(result.final_step().response.content, "done")
        self.assertEqual(
            [call.name for call in mcp_executor.calls],
            ["filesystem.read_text_file"],
        )
        self.assertEqual(
            tuple(schema["name"] for schema in provider.requests[0].tool_schemas),
            ("search_mcp_tools", "load_mcp_tool"),
        )
        self.assertIn(
            "filesystem.read_text_file",
            tuple(schema["name"] for schema in provider.requests[1].tool_schemas),
        )
        self.assertIsNotNone(loaded_state)
        self.assertEqual(loaded_state.load_count, 1)

    def test_registered_mcp_tool_runs_even_when_schema_is_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            state_store = McpUsageStateStore.in_data_dir(workspace / "var", "default")
            server = _server()
            tool = _read_tool()
            catalog = McpToolCatalog(
                inventories=(McpServerInventory(server=server, tools=(tool,)),),
                state_store=state_store,
            )
            registry = NativeToolRegistry(mcp_loader_tools(catalog))
            mcp_executor = _FakeMcpExecutor()
            trace_sink = _TraceSink()
            provider = _StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="filesystem.read_text_file",
                                arguments={"path": "README.md"},
                                id="call_1",
                            ),
                        )
                    ),
                    ModelResponse(content="done"),
                ]
            )

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="Read README."),),
                model="stub-model",
                native_tools=registry,
                mcp_tools=mcp_executor,  # type: ignore[arg-type]
                tool_schemas=registry.schemas(),
                max_steps=2,
                trace_recorder=TraceRecorder(trace_sink),
            )

        self.assertFalse(result.has_errors())
        self.assertEqual(
            [call.name for call in mcp_executor.calls],
            ["filesystem.read_text_file"],
        )
        self.assertEqual(result.final_step().response.content, "done")
        self.assertIn("mcp_tool_schema_hidden", [event.kind for event in trace_sink.events])

    def test_loaded_complex_mcp_schema_is_flattened_and_arguments_are_unflattened(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            state_store = McpUsageStateStore.in_data_dir(workspace / "var", "default")
            server = _server()
            tool = _inspect_metadata_tool()
            catalog = McpToolCatalog(
                inventories=(McpServerInventory(server=server, tools=(tool,)),),
                state_store=state_store,
            )
            registry = NativeToolRegistry(mcp_loader_tools(catalog))
            mcp_executor = _InspectMetadataFakeMcpExecutor()
            provider = _StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name=LOAD_MCP_TOOL_NAME,
                                arguments={"name": "filesystem.inspect_file_metadata"},
                                id="call_1",
                            ),
                        )
                    ),
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="filesystem.inspect_file_metadata",
                                arguments={
                                    "path": "README.md",
                                    "options.permissions.mode": "0644",
                                    "options.overwrite": True,
                                },
                                id="call_2",
                            ),
                        )
                    ),
                    ModelResponse(content="done"),
                ]
            )

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="Configure README."),),
                model="stub-model",
                native_tools=registry,
                mcp_tools=mcp_executor,  # type: ignore[arg-type]
                tool_schemas=registry.schemas(),
                max_steps=3,
            )

        self.assertFalse(result.has_errors())
        loaded_schema = next(
            schema
            for schema in provider.requests[1].tool_schemas
            if schema["name"] == "filesystem.inspect_file_metadata"
        )
        self.assertIn("schema_flattened=true", result.steps[0].tool_results[0].refs)
        self.assertIn("options.permissions.mode", loaded_schema["input_schema"]["properties"])
        self.assertIn("options.overwrite", loaded_schema["input_schema"]["properties"])
        self.assertEqual(
            mcp_executor.calls[0].arguments,
            {
                "path": "README.md",
                "options": {
                    "permissions": {"mode": "0644"},
                    "overwrite": True,
                },
            },
        )
        self.assertIn(
            "mcp_tool_arguments_unflattened",
            [event.kind for event in result.events()],
        )

    def test_mcp_output_truncation_event_is_recorded_to_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            mcp_executor = _TruncatingFakeMcpExecutor()
            trace_sink = _TraceSink()
            provider = _StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="filesystem.read_text_file",
                                arguments={"path": "README.md"},
                                id="call_1",
                            ),
                        )
                    ),
                    ModelResponse(content="done"),
                ]
            )

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="Read README."),),
                model="stub-model",
                mcp_tools=mcp_executor,  # type: ignore[arg-type]
                tool_schemas=(
                    {
                        "name": "filesystem.read_text_file",
                        "description": "Read a file.",
                        "input_schema": {"type": "object"},
                    },
                ),
                max_steps=2,
                trace_recorder=TraceRecorder(trace_sink),
            )

        self.assertFalse(result.has_errors())
        trace_events = tuple(event.kind for event in trace_sink.events)
        self.assertIn("mcp_tool_output_truncated", trace_events)
        truncation_event = next(
            event
            for event in trace_sink.events
            if event.kind == "mcp_tool_output_truncated"
        )
        self.assertIn("tool_source=mcp", truncation_event.refs)
        self.assertIn("mcp_output_truncated=true", truncation_event.refs)

    def test_mcp_catalog_status_reports_tier_and_schema_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_store = McpUsageStateStore.in_data_dir(root / "var", "default")
            server = _server()
            tool = _read_tool()
            state_store.record_tool_loaded(tool)
            catalog = McpToolCatalog(
                inventories=(McpServerInventory(server=server, tools=(tool,)),),
                state_store=state_store,
            )

            status = format_mcp_catalog_status(catalog, model_context_tokens=1_000_000)

        self.assertIn("MCP status:", status)
        self.assertIn("filesystem: transport=stdio", status)
        self.assertIn("read_only_tools=1", status)
        self.assertIn("filesystem.read_text_file", status)
        self.assertIn("previously-loaded", status)


def _server() -> McpServerSpec:
    return McpServerSpec(
        name="filesystem",
        transport=McpTransport.STDIO,
        command="server",
        description="Filesystem access.",
    )


def _read_tool() -> McpToolRef:
    return McpToolRef(
        server_name="filesystem",
        name="read_text_file",
        description="Read a file.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        annotations={"readOnlyHint": True},
    )


def _inspect_metadata_tool() -> McpToolRef:
    return McpToolRef(
        server_name="filesystem",
        name="inspect_file_metadata",
        description="Inspect file metadata.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "options": {
                    "type": "object",
                    "properties": {
                        "permissions": {
                            "type": "object",
                            "properties": {
                                "mode": {"type": "string"},
                            },
                            "required": ["mode"],
                        },
                        "overwrite": {"type": "boolean"},
                    },
                    "required": ["permissions"],
                },
            },
            "required": ["path", "options"],
            "additionalProperties": False,
        },
        annotations={"readOnlyHint": True},
    )


if __name__ == "__main__":
    unittest.main()
