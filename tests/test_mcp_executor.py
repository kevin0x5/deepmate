import unittest
from pathlib import Path
from unittest.mock import patch

from deepmate.channels.cli import _format_mcp_tool_line
from deepmate.mcp.client import McpCallResult
from deepmate.mcp.executor import McpToolExecutor
from deepmate.mcp.output_policy import McpOutputPolicy
from deepmate.mcp.spec import McpServerSpec, McpToolRef, McpTransport
from deepmate.providers import ModelToolRequest
from deepmate.runtime import (
    HookAction,
    HookActionType,
    HookDefinition,
    HookEvent,
    HookLayer,
    HookRegistry,
    HookRuntimeContext,
)


class _FakeMcpClient:
    instances = []

    def __init__(self, server, workspace):
        self.server = server
        self.workspace = Path(workspace)
        self.calls = []
        self.closed = False
        _FakeMcpClient.instances.append(self)

    def connect(self):
        return None

    def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, dict(arguments)))
        return McpCallResult(
            content=f"called {tool_name}",
            data={"content": [], "ok": True},
        )

    def close(self):
        self.closed = True

    def diagnostic_refs(self):
        return ("mcp_notifications_skipped=1",)


class _FailingCloseMcpClient(_FakeMcpClient):
    def call_tool(self, tool_name, arguments):
        raise ValueError("tool failure")

    def close(self):
        super().close()
        raise RuntimeError("close failure")


class _ConnectFailMcpClient(_FakeMcpClient):
    def connect(self):
        raise OSError("connect failed")


class _LargeContentMcpClient(_FakeMcpClient):
    def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, dict(arguments)))
        return McpCallResult(
            content=("H" * 800) + ("M" * 800) + ("T" * 800),
            data={"content": [], "ok": True},
        )


class _LargeErrorMcpClient(_FakeMcpClient):
    def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, dict(arguments)))
        return McpCallResult(
            content=("S" * 1000) + ("E" * 1000),
            data={"content": [], "ok": False},
            is_error=True,
        )


class _LargeDataMcpClient(_FakeMcpClient):
    def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, dict(arguments)))
        return McpCallResult(
            content="",
            data={
                "content": [],
                "items": ["x" * 200 for _ in range(80)],
                "nextCursor": "next-page",
            },
        )


class _FlakyMcpClient(_FakeMcpClient):
    def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, dict(arguments)))
        if len(_FakeMcpClient.instances) == 1:
            raise OSError("transport failure")
        return McpCallResult(
            content=f"recovered {tool_name}",
            data={"content": [], "ok": True},
        )


class _ValueErrorMcpClient(_FakeMcpClient):
    def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, dict(arguments)))
        raise ValueError("configuration failure")


def _fake_client_factory(client_class):
    def factory(server, workspace):
        return client_class(server, workspace)

    return factory


class McpToolExecutorTests(unittest.TestCase):
    def setUp(self):
        _FakeMcpClient.instances = []

    def test_output_policy_budget_scales_with_context_window(self):
        self.assertEqual(McpOutputPolicy(1_000_000).output_token_budget(), 25_000)
        self.assertEqual(McpOutputPolicy(400_000).output_token_budget(), 10_000)
        self.assertEqual(McpOutputPolicy(128_000).output_token_budget(), 4_000)

    def test_reuses_stdio_client_until_closed(self):
        server = McpServerSpec(
            name="filesystem",
            transport=McpTransport.STDIO,
            command="mcp-server-filesystem",
        )
        tool = McpToolRef(
            server_name="filesystem",
            name="read_text_file",
            annotations={"readOnlyHint": True},
        )
        executor = McpToolExecutor((server,), (tool,), "/workspace")

        with patch(
            "deepmate.mcp.executor.create_mcp_client",
            _fake_client_factory(_FakeMcpClient),
        ):
            first = executor.execute(
                ModelToolRequest(
                    name="filesystem.read_text_file",
                    arguments={"path": "a.txt"},
                    id="call-1",
                )
            )
            second = executor.execute(
                ModelToolRequest(
                    name="filesystem.read_text_file",
                    arguments={"path": "b.txt"},
                    id="call-2",
                )
            )
            executor.close()

        self.assertTrue(first.is_success())
        self.assertTrue(second.is_success())
        self.assertIsNotNone(first.model_result)
        self.assertEqual(len(_FakeMcpClient.instances), 1)
        self.assertEqual(
            _FakeMcpClient.instances[0].calls,
            [
                ("read_text_file", {"path": "a.txt"}),
                ("read_text_file", {"path": "b.txt"}),
            ],
        )
        self.assertIn("mcp_tool=filesystem.read_text_file", first.model_result.refs)
        self.assertIn("mcp_notifications_skipped=1", first.model_result.refs)
        self.assertTrue(_FakeMcpClient.instances[0].closed)

    def test_closes_new_client_when_connect_fails(self):
        server = McpServerSpec(
            name="filesystem",
            transport=McpTransport.STDIO,
            command="mcp-server-filesystem",
        )
        tool = McpToolRef(
            server_name="filesystem",
            name="read_text_file",
            annotations={"readOnlyHint": True},
        )
        executor = McpToolExecutor((server,), (tool,), "/workspace")

        with patch(
            "deepmate.mcp.executor.create_mcp_client",
            _fake_client_factory(_ConnectFailMcpClient),
        ):
            result = executor.execute(
                ModelToolRequest(
                    name="filesystem.read_text_file",
                    arguments={"path": "a.txt"},
                    id="call-1",
                )
            )

        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.code, "mcp_tool_failed")
        self.assertEqual(len(_ConnectFailMcpClient.instances), 2)
        self.assertTrue(all(client.closed for client in _ConnectFailMcpClient.instances))
        self.assertEqual(executor._clients, {})

    def test_rejects_non_read_only_tool_without_starting_client(self):
        server = McpServerSpec(
            name="filesystem",
            transport=McpTransport.STDIO,
            command="mcp-server-filesystem",
        )
        tool = McpToolRef(server_name="filesystem", name="write_file")
        executor = McpToolExecutor((server,), (tool,), "/workspace")

        with patch(
            "deepmate.mcp.executor.create_mcp_client",
            _fake_client_factory(_FakeMcpClient),
        ):
            result = executor.execute(
                ModelToolRequest(
                    name="filesystem.write_file",
                    arguments={"path": "a.txt", "content": "x"},
                    id="call-1",
                )
            )

        self.assertFalse(result.is_success())
        self.assertEqual(result.error.code if result.error else "", "mcp_tool_not_read_only")
        self.assertEqual(_FakeMcpClient.instances, [])

    def test_allows_non_read_only_tool_when_write_gate_enabled(self):
        server = McpServerSpec(
            name="filesystem",
            transport=McpTransport.STDIO,
            command="mcp-server-filesystem",
        )
        tool = McpToolRef(server_name="filesystem", name="write_file")
        executor = McpToolExecutor(
            (server,),
            (tool,),
            "/workspace",
            allow_write_tools=True,
        )

        with patch(
            "deepmate.mcp.executor.create_mcp_client",
            _fake_client_factory(_FakeMcpClient),
        ):
            result = executor.execute(
                ModelToolRequest(
                    name="filesystem.write_file",
                    arguments={"path": "a.txt", "content": "x"},
                    id="call-1",
                )
            )

        self.assertTrue(result.is_success())
        self.assertEqual(
            _FakeMcpClient.instances[0].calls,
            [("write_file", {"path": "a.txt", "content": "x"})],
        )

    def test_mcp_before_hook_blocks_before_client_starts(self):
        server = McpServerSpec(
            name="filesystem",
            transport=McpTransport.STDIO,
            command="mcp-server-filesystem",
        )
        tool = McpToolRef(
            server_name="filesystem",
            name="read_text_file",
            annotations={"readOnlyHint": True},
        )
        executor = McpToolExecutor(
            (server,),
            (tool,),
            "/workspace",
            hook_context=_hook_context(
                HookEvent.MCP_BEFORE,
                HookActionType.DENY,
                when={"tool_names": ["filesystem.read_text_file"]},
                params={"reason": "mcp blocked"},
            ),
        )

        with patch(
            "deepmate.mcp.executor.create_mcp_client",
            _fake_client_factory(_FakeMcpClient),
        ):
            result = executor.execute(
                ModelToolRequest(
                    name="filesystem.read_text_file",
                    arguments={"path": "a.txt"},
                    id="call-1",
                )
            )

        self.assertFalse(result.is_success())
        self.assertEqual(result.error.code if result.error else "", "mcp_tool_blocked_by_hook")
        self.assertEqual(_FakeMcpClient.instances, [])
        self.assertIn("hook_event=mcp.before", result.error.refs if result.error else ())

    def test_mcp_hooks_are_reported_on_success(self):
        server = McpServerSpec(
            name="filesystem",
            transport=McpTransport.STDIO,
            command="mcp-server-filesystem",
        )
        tool = McpToolRef(
            server_name="filesystem",
            name="read_text_file",
            annotations={"readOnlyHint": True},
        )
        executor = McpToolExecutor(
            (server,),
            (tool,),
            "/workspace",
            hook_context=HookRuntimeContext.from_registry(
                HookRegistry.from_hooks(
                    (
                        HookDefinition(
                            hook_id="trace-before",
                            event_name=HookEvent.MCP_BEFORE,
                            layer=HookLayer.SESSION,
                            actions=(HookAction(HookActionType.TRACE),),
                        ),
                        HookDefinition(
                            hook_id="trace-after",
                            event_name=HookEvent.MCP_AFTER,
                            layer=HookLayer.SESSION,
                            actions=(HookAction(HookActionType.TRACE),),
                        ),
                    )
                )
            ),
        )

        with patch(
            "deepmate.mcp.executor.create_mcp_client",
            _fake_client_factory(_FakeMcpClient),
        ):
            result = executor.execute(
                ModelToolRequest(
                    name="filesystem.read_text_file",
                    arguments={"path": "a.txt"},
                    id="call-1",
                )
            )

        self.assertTrue(result.is_success())
        self.assertIn("hook_id=trace-before", result.model_result.refs)
        self.assertIn("hook_id=trace-after", result.model_result.refs)
        self.assertIn(
            "mcp_before_hook_observed",
            tuple(event.kind for event in result.events),
        )
        self.assertIn(
            "mcp_after_hook_observed",
            tuple(event.kind for event in result.events),
        )

    def test_close_failure_does_not_hide_tool_failure(self):
        server = McpServerSpec(
            name="filesystem",
            transport=McpTransport.STDIO,
            command="mcp-server-filesystem",
        )
        tool = McpToolRef(
            server_name="filesystem",
            name="read_text_file",
            annotations={"readOnlyHint": True},
        )
        executor = McpToolExecutor((server,), (tool,), "/workspace")

        with patch(
            "deepmate.mcp.executor.create_mcp_client",
            _fake_client_factory(_FailingCloseMcpClient),
        ):
            result = executor.execute(
                ModelToolRequest(
                    name="filesystem.read_text_file",
                    arguments={"path": "a.txt"},
                    id="call-1",
                )
            )

        self.assertEqual(result.error.code if result.error else "", "mcp_tool_failed")
        self.assertIn("tool failure", result.error.message if result.error else "")

    def test_output_policy_keeps_small_output_and_records_tool_meta(self):
        server = McpServerSpec(
            name="filesystem",
            transport=McpTransport.STDIO,
            command="mcp-server-filesystem",
        )
        tool = McpToolRef(
            server_name="filesystem",
            name="read_text_file",
            annotations={"readOnlyHint": True},
            meta={"anthropic/maxResultSizeChars": "1234"},
        )
        executor = McpToolExecutor(
            (server,),
            (tool,),
            "/workspace",
            output_policy=_small_output_policy(),
        )

        with patch(
            "deepmate.mcp.executor.create_mcp_client",
            _fake_client_factory(_FakeMcpClient),
        ):
            result = executor.execute(
                ModelToolRequest(
                    name="filesystem.read_text_file",
                    arguments={"path": "a.txt"},
                    id="call-1",
                )
            )

        self.assertTrue(result.is_success())
        self.assertEqual(result.model_result.content, "called read_text_file")
        self.assertEqual(result.model_result.data, {"ok": True})
        self.assertIn("mcp_tool_max_result_size_chars=1234", result.model_result.refs)
        self.assertNotIn("mcp_output_truncated=true", result.model_result.refs)

    def test_output_policy_truncates_large_text_with_marker_and_refs(self):
        server = McpServerSpec(
            name="filesystem",
            transport=McpTransport.STDIO,
            command="mcp-server-filesystem",
        )
        tool = McpToolRef(
            server_name="filesystem",
            name="read_text_file",
            annotations={"readOnlyHint": True},
        )
        executor = McpToolExecutor(
            (server,),
            (tool,),
            "/workspace",
            output_policy=_small_output_policy(),
        )

        with patch(
            "deepmate.mcp.executor.create_mcp_client",
            _fake_client_factory(_LargeContentMcpClient),
        ):
            result = executor.execute(
                ModelToolRequest(
                    name="filesystem.read_text_file",
                    arguments={"path": "a.txt"},
                    id="call-1",
                )
            )

        self.assertTrue(result.is_success())
        self.assertIn("[Deepmate MCP output truncated:", result.model_result.content)
        self.assertLess(len(result.model_result.content), 2400)
        self.assertIn("mcp_output_truncated=true", result.model_result.refs)
        self.assertIn(
            "mcp_tool_output_truncated",
            tuple(event.kind for event in result.events),
        )

    def test_output_policy_preserves_more_tail_for_error_output(self):
        server = McpServerSpec(
            name="filesystem",
            transport=McpTransport.STDIO,
            command="mcp-server-filesystem",
        )
        tool = McpToolRef(
            server_name="filesystem",
            name="read_text_file",
            annotations={"readOnlyHint": True},
        )
        executor = McpToolExecutor(
            (server,),
            (tool,),
            "/workspace",
            output_policy=_small_output_policy(),
        )

        with patch(
            "deepmate.mcp.executor.create_mcp_client",
            _fake_client_factory(_LargeErrorMcpClient),
        ):
            result = executor.execute(
                ModelToolRequest(
                    name="filesystem.read_text_file",
                    arguments={"path": "a.txt"},
                    id="call-1",
                )
            )

        self.assertEqual(
            result.error.code if result.error else "",
            "mcp_tool_returned_error",
        )
        self.assertGreater(
            result.model_result.content.count("E"),
            result.model_result.content.count("S"),
        )
        self.assertIn("mcp_output_truncated=true", result.model_result.refs)

    def test_output_policy_omits_large_structured_data_without_content(self):
        server = McpServerSpec(
            name="search",
            transport=McpTransport.STDIO,
            command="mcp-server-search",
        )
        tool = McpToolRef(
            server_name="search",
            name="query",
            annotations={"readOnlyHint": True},
        )
        executor = McpToolExecutor(
            (server,),
            (tool,),
            "/workspace",
            output_policy=_small_output_policy(),
        )

        with patch(
            "deepmate.mcp.executor.create_mcp_client",
            _fake_client_factory(_LargeDataMcpClient),
        ):
            result = executor.execute(
                ModelToolRequest(
                    name="search.query",
                    arguments={"q": "deepmate"},
                    id="call-1",
                )
            )

        self.assertTrue(result.is_success())
        self.assertEqual(result.model_result.content, "")
        self.assertTrue(result.model_result.data["mcp_output_truncated"])
        self.assertTrue(result.model_result.data["data_omitted"])
        self.assertEqual(
            result.model_result.data["data_keys"],
            ["items", "nextCursor"],
        )
        self.assertIn("mcp_structured_data_omitted=true", result.model_result.refs)

    def test_cli_mcp_tool_line_shows_missing_read_only_hint_and_description(self):
        tool = McpToolRef(server_name="filesystem", name="stat_file")

        self.assertEqual(
            _format_mcp_tool_line(tool),
            (
                "- filesystem.stat_file [read-only-hint-missing] - "
                "MCP tool filesystem.stat_file. [description fallback]"
            ),
        )

    def test_retries_once_after_transport_failure(self):
        server = McpServerSpec(
            name="filesystem",
            transport=McpTransport.STDIO,
            command="mcp-server-filesystem",
        )
        tool = McpToolRef(
            server_name="filesystem",
            name="read_text_file",
            annotations={"readOnlyHint": True},
        )
        executor = McpToolExecutor((server,), (tool,), "/workspace")

        with patch(
            "deepmate.mcp.executor.create_mcp_client",
            _fake_client_factory(_FlakyMcpClient),
        ):
            result = executor.execute(
                ModelToolRequest(
                    name="filesystem.read_text_file",
                    arguments={"path": "a.txt"},
                    id="call-1",
                )
            )

        self.assertTrue(result.is_success())
        self.assertEqual(len(_FakeMcpClient.instances), 2)
        self.assertEqual(result.model_result.content, "recovered read_text_file")
        self.assertIn("mcp_retry_count=1", result.model_result.refs)

    def test_does_not_retry_value_error_from_client(self):
        server = McpServerSpec(
            name="filesystem",
            transport=McpTransport.STDIO,
            command="mcp-server-filesystem",
        )
        tool = McpToolRef(
            server_name="filesystem",
            name="read_text_file",
            annotations={"readOnlyHint": True},
        )
        executor = McpToolExecutor((server,), (tool,), "/workspace")

        with patch(
            "deepmate.mcp.executor.create_mcp_client",
            _fake_client_factory(_ValueErrorMcpClient),
        ):
            result = executor.execute(
                ModelToolRequest(
                    name="filesystem.read_text_file",
                    arguments={"path": "a.txt"},
                    id="call-1",
                )
            )

        self.assertEqual(result.error.code if result.error else "", "mcp_tool_failed")
        self.assertEqual(len(_FakeMcpClient.instances), 1)

    def test_connect_failure_does_not_cache_half_initialized_client(self):
        server = McpServerSpec(
            name="filesystem",
            transport=McpTransport.STDIO,
            command="mcp-server-filesystem",
        )
        tool = McpToolRef(
            server_name="filesystem",
            name="read_text_file",
            annotations={"readOnlyHint": True},
        )
        executor = McpToolExecutor((server,), (tool,), "/workspace")

        with patch(
            "deepmate.mcp.executor.create_mcp_client",
            _fake_client_factory(_ConnectFailMcpClient),
        ):
            result = executor.execute(
                ModelToolRequest(
                    name="filesystem.read_text_file",
                    arguments={"path": "a.txt"},
                    id="call-1",
                )
            )

        self.assertEqual(result.error.code if result.error else "", "mcp_tool_failed")
        self.assertEqual(executor._clients, {})


def _small_output_policy() -> McpOutputPolicy:
    return McpOutputPolicy(
        model_context_tokens=10_000,
        max_output_ratio=0.01,
        min_output_tokens=100,
        max_output_tokens=120,
    )


def _hook_context(
    event_name: HookEvent,
    action_type: HookActionType,
    *,
    when: dict[str, object] | None = None,
    params: dict[str, object] | None = None,
) -> HookRuntimeContext:
    return HookRuntimeContext.from_registry(
        HookRegistry.from_hooks(
            (
                HookDefinition(
                    hook_id="test-hook",
                    event_name=event_name,
                    layer=HookLayer.SESSION,
                    when=when or {},
                    actions=(HookAction(action_type, params or {}),),
                ),
            )
        )
    )


if __name__ == "__main__":
    unittest.main()
