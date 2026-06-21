from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepmate.domain import Message, MessageRole, ProfileRef
from deepmate.mcp import McpToolExecutor
from deepmate.providers import (
    ModelConversationItem,
    ModelResponse,
    ModelToolExchange,
    ModelToolRequest,
    ModelToolResult,
    NetworkError,
    ProviderError,
    RateLimitError,
    TokenUsage,
)
from deepmate.runtime import (
    ConversationBudgetPolicy,
    HookAction,
    HookActionType,
    HookDefinition,
    HookEvent,
    HookLayer,
    HookRegistry,
    HookRuntimeContext,
    HookSignalStore,
    LoopGuardPolicy,
    LoopGuardStopReason,
    ProviderRetryPolicy,
    ToolOutputCompactor,
    ToolRepairPolicy,
    ToolAccessMode,
    ToolAccessPolicy,
    execute_native_tool_request,
    run_user_turn,
)
from deepmate.storage import ToolOutputStore
from deepmate.tools import NativeTool, NativeToolRegistry, NativeToolResult, shell_tools
from deepmate.trace import TraceRecorder
from deepmate.trace.schema import TraceSpan


class StubProvider:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("stub provider received too many requests")
        return self.responses.pop(0)


class FailingProvider:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def complete(self, _request):
        self.calls += 1
        raise self.error


class ToolExecutionTests(unittest.TestCase):
    def test_native_tool_runtime_error_returns_structured_failure(self) -> None:
        registry = NativeToolRegistry(
            (
                NativeTool(
                    name="fail_tool",
                    description="Fail.",
                    input_schema={"type": "object"},
                    handler=lambda arguments: (_ for _ in ()).throw(
                        RuntimeError("boom")
                    ),
                ),
            )
        )

        result = execute_native_tool_request(
            ModelToolRequest(name="fail_tool", id="call_1", arguments={}),
            registry,
            ToolAccessPolicy(ToolAccessMode.READ_ONLY),
        )

        self.assertEqual(result.error.code if result.error else "", "native_tool_failed")
        self.assertIn("boom", result.error.message if result.error else "")

    def test_native_tool_base_exception_is_not_swallowed(self) -> None:
        registry = NativeToolRegistry(
            (
                NativeTool(
                    name="exit_tool",
                    description="Exit.",
                    input_schema={"type": "object"},
                    handler=lambda arguments: (_ for _ in ()).throw(
                        KeyboardInterrupt()
                    ),
                ),
            )
        )

        with self.assertRaises(KeyboardInterrupt):
            execute_native_tool_request(
                ModelToolRequest(name="exit_tool", id="call_1", arguments={}),
                registry,
                ToolAccessPolicy(ToolAccessMode.READ_ONLY),
            )

    def test_invalid_tool_call_does_not_break_valid_tool_exchange(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            registry = NativeToolRegistry(
                (
                    NativeTool(
                        name="write_marker",
                        description="Write a marker file.",
                        input_schema={"type": "object"},
                        handler=lambda arguments: (
                            (workspace / "marker.txt").write_text(
                                str(arguments.get("content", "")),
                                encoding="utf-8",
                            )
                            and NativeToolResult(content="written")
                        ),
                        read_only=False,
                    ),
                )
            )
            provider = StubProvider(
                [
                    ModelResponse(
                        content="writing",
                        tool_requests=(
                            ModelToolRequest(
                                name="write_marker",
                                id="call_1",
                                arguments={"content": "ok"},
                            ),
                            ModelToolRequest(
                                name="write_marker",
                                id="",
                                arguments={"content": "bad"},
                            ),
                        ),
                    ),
                    ModelResponse(content="done"),
                ]
            )

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="write"),),
                model="stub-model",
                native_tools=registry,
                tool_schemas=registry.schemas(),
                tool_access_policy=ToolAccessPolicy(ToolAccessMode.WORKSPACE_WRITE),
                max_steps=2,
            )

            self.assertEqual((workspace / "marker.txt").read_text(encoding="utf-8"), "ok")
            self.assertEqual(result.final_step().response.content, "done")
            self.assertEqual(len(result.tool_exchanges), 1)
            self.assertEqual(
                tuple(request.id for request in result.tool_exchanges[0].tool_requests),
                ("call_1",),
            )
            self.assertIn("tool_request_invalid", [event.kind for event in result.events()])

    def test_provider_retry_exhaustion_returns_structured_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            provider = FailingProvider(NetworkError("temporary outage"))
            trace_sink = _TraceSink()

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="hello"),),
                model="stub-model",
                provider_retry_policy=ProviderRetryPolicy(
                    max_attempts=2,
                    initial_delay_seconds=0,
                ),
                trace_recorder=TraceRecorder(trace_sink),
                max_steps=1,
            )

        self.assertEqual(provider.calls, 2)
        self.assertTrue(result.has_errors())
        self.assertEqual(result.errors()[0].code, "provider_request_failed")
        self.assertIn("temporary outage", result.final_step().response.content)
        self.assertEqual(result.conversation[-1].message.role, MessageRole.ASSISTANT)
        self.assertIn("temporary outage", result.conversation[-1].message.content)
        self.assertIn("provider_retry_exhausted", [event.kind for event in trace_sink.events])
        self.assertIn("provider_request_failed", [event.kind for event in trace_sink.events])

    def test_provider_retry_honors_rate_limit_retry_after(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            provider = FailingProvider(RateLimitError("rate limited", retry_after_seconds=1.25))
            sleeps = []

            with patch("deepmate.runtime.agent_loop.sleep", sleeps.append):
                result = run_user_turn(
                    provider=provider,
                    workspace=workspace,
                    profile=ProfileRef(name="default", uri="profiles/default"),
                    messages=(Message(role=MessageRole.USER, content="hello"),),
                    model="stub-model",
                    provider_retry_policy=ProviderRetryPolicy(
                        max_attempts=2,
                        initial_delay_seconds=0.1,
                    ),
                    max_steps=1,
                )

        self.assertTrue(result.has_errors())
        self.assertEqual(provider.calls, 2)
        self.assertEqual(sleeps, [1.25])

    def test_max_steps_preserves_last_assistant_text_with_tool_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")

            def handle(arguments):
                return NativeToolResult(content=str(arguments.get("text", "")))

            registry = NativeToolRegistry(
                (
                    NativeTool(
                        name="echo",
                        description="Echo arguments.",
                        input_schema={"type": "object"},
                        handler=handle,
                    ),
                )
            )
            provider = StubProvider(
                [
                    ModelResponse(
                        content="I will inspect this before continuing.",
                        tool_requests=(
                            ModelToolRequest(
                                name="echo",
                                arguments={"text": "hello"},
                                id="call_1",
                            ),
                        ),
                    )
                ]
            )

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="Use echo."),),
                model="stub-model",
                native_tools=registry,
                tool_schemas=registry.schemas(),
                max_steps=1,
            )

        self.assertTrue(result.reached_max_steps)
        self.assertIsNotNone(result.loop_guard_stop)
        self.assertEqual(result.loop_guard_stop.reason, LoopGuardStopReason.HARD_STEP_CAP)
        self.assertIn("Stop reason: hard_step_cap", result.continuation_note())
        self.assertEqual(
            result.final_step().response.content,
            "I will inspect this before continuing.",
        )
        self.assertIsNotNone(result.conversation[-1].tool_exchange)
        self.assertEqual(
            result.conversation[-1].tool_exchange.assistant_content,
            "I will inspect this before continuing.",
        )
        assistant_messages = [
            item.message.content
            for item in result.conversation
            if item.message is not None and item.message.role == MessageRole.ASSISTANT
        ]
        self.assertNotIn("I will inspect this before continuing.", assistant_messages)

    def test_loop_guard_hard_step_cap_limits_turn_below_max_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            registry = NativeToolRegistry(
                (
                    NativeTool(
                        name="echo",
                        description="Echo arguments.",
                        input_schema={"type": "object"},
                        handler=lambda arguments: NativeToolResult(
                            content=str(arguments.get("text", ""))
                        ),
                    ),
                )
            )
            provider = StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="echo",
                                arguments={"text": "hello"},
                                id="call_1",
                            ),
                        ),
                    ),
                    ModelResponse(content="should not be requested"),
                ]
            )

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="Use echo."),),
                model="stub-model",
                native_tools=registry,
                tool_schemas=registry.schemas(),
                loop_guard_policy=LoopGuardPolicy(enabled=True, hard_step_cap=1),
                max_steps=5,
            )

        self.assertEqual(len(provider.requests), 1)
        self.assertTrue(result.reached_max_steps)
        self.assertIsNotNone(result.loop_guard_stop)
        self.assertEqual(result.loop_guard_stop.reason, LoopGuardStopReason.HARD_STEP_CAP)
        self.assertEqual(len(result.tool_exchanges), 1)

    def test_context_preflight_stops_before_provider_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            provider = StubProvider([])

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="x" * 2000),),
                model="stub-model",
                conversation_budget_policy=ConversationBudgetPolicy(
                    model_context_tokens=64,
                    response_token_reserve=0,
                    safety_margin_tokens=0,
                ),
                loop_guard_policy=LoopGuardPolicy(enabled=True, hard_step_cap=100),
                max_steps=100,
            )

        self.assertEqual(provider.requests, [])
        self.assertFalse(result.reached_max_steps)
        self.assertIsNotNone(result.loop_guard_stop)
        self.assertEqual(
            result.loop_guard_stop.reason,
            LoopGuardStopReason.CONTEXT_EXHAUSTED,
        )
        self.assertEqual(
            result.errors()[0].code,
            "loop_guard_context_exhausted",
        )
        self.assertIn("Stop reason: context_exhausted", result.continuation_note())
        self.assertIn("context window", result.final_step().response.content)

    def test_trim_mode_emergency_reduces_protected_recent_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            provider = StubProvider([ModelResponse(content="done")])
            prior = tuple(
                ModelConversationItem.from_message(
                    Message(role=MessageRole.USER, content=f"{index}: " + ("x" * 200))
                )
                for index in range(50)
            )

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="continue"),),
                model="stub-model",
                conversation=prior,
                conversation_budget_policy=ConversationBudgetPolicy(
                    history_window_mode="trim",
                    history_token_budget=800,
                    protect_recent_items=40,
                    model_context_tokens=500,
                    response_token_reserve=0,
                    safety_margin_tokens=0,
                ),
                loop_guard_policy=LoopGuardPolicy(enabled=True),
                max_steps=1,
            )

        self.assertFalse(result.has_errors())
        self.assertEqual(result.final_step().response.content, "done")
        self.assertEqual(len(provider.requests), 1)
        self.assertLess(len(provider.requests[0].conversation), 42)
        self.assertTrue(result.steps[0].conversation_budget_report.trimmed)

    def test_warn_mode_emergency_trims_instead_of_bricking_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            provider = StubProvider([ModelResponse(content="done")])
            prior = tuple(
                ModelConversationItem.from_message(
                    Message(role=MessageRole.USER, content=f"{index}: " + ("x" * 200))
                )
                for index in range(50)
            )

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="continue"),),
                model="stub-model",
                conversation=prior,
                conversation_budget_policy=ConversationBudgetPolicy(
                    history_window_mode="warn",
                    history_token_budget=100_000,
                    protect_recent_items=40,
                    model_context_tokens=500,
                    response_token_reserve=0,
                    safety_margin_tokens=0,
                ),
                loop_guard_policy=LoopGuardPolicy(enabled=True),
                max_steps=1,
            )

        self.assertFalse(result.has_errors())
        self.assertEqual(result.final_step().response.content, "done")
        self.assertEqual(len(provider.requests), 1)
        self.assertLess(len(provider.requests[0].conversation), 42)
        self.assertTrue(result.steps[0].conversation_budget_report.trimmed)
        self.assertTrue(
            any(event.kind == "context_emergency_trimmed" for event in result.events())
        )

    def test_provider_context_length_error_returns_loop_guard_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            provider = FailingProvider(
                ProviderError("model request failed with HTTP 400: context length exceeded")
            )

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="hello"),),
                model="stub-model",
                max_steps=1,
            )

        self.assertEqual(provider.calls, 1)
        self.assertTrue(result.has_errors())
        self.assertEqual(result.errors()[0].code, "loop_guard_context_exhausted")
        self.assertIsNotNone(result.loop_guard_stop)
        self.assertEqual(
            result.loop_guard_stop.reason,
            LoopGuardStopReason.CONTEXT_EXHAUSTED,
        )
        self.assertIn("context window", result.final_step().response.content)

    def test_generic_provider_error_returns_structured_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            provider = FailingProvider(ProviderError("provider rejected request"))

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="hello"),),
                model="stub-model",
                max_steps=1,
            )

        self.assertEqual(provider.calls, 1)
        self.assertTrue(result.has_errors())
        self.assertEqual(result.errors()[0].code, "provider_request_failed")
        self.assertIn("provider rejected request", result.final_step().response.content)

    def test_provider_before_request_hook_can_block_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            provider = StubProvider([ModelResponse(content="should not run")])
            trace_sink = _TraceSink()

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="hello"),),
                model="stub-model",
                hook_context=_hook_context(
                    HookEvent.PROVIDER_BEFORE_REQUEST,
                    HookActionType.DENY,
                    hook_id="block-provider",
                    params={"reason": "provider blocked for test"},
                ),
                trace_recorder=TraceRecorder(trace_sink),
                max_steps=1,
            )

        self.assertEqual(provider.requests, [])
        self.assertTrue(result.has_errors())
        self.assertEqual(result.errors()[0].code, "provider_request_blocked_by_hook")
        self.assertEqual(result.final_step().response.content, "provider blocked for test")
        self.assertIn("hook_event_evaluated", [event.kind for event in trace_sink.events])

    def test_tool_before_hook_blocks_native_tool_without_calling_handler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            calls: list[dict[str, object]] = []

            def handle(arguments):
                calls.append(dict(arguments))
                return NativeToolResult(content="should not run")

            registry = NativeToolRegistry(
                (
                    NativeTool(
                        name="echo",
                        description="Echo arguments.",
                        input_schema={"type": "object"},
                        handler=handle,
                    ),
                )
            )
            provider = StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="echo",
                                arguments={"text": "hello"},
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
                messages=(Message(role=MessageRole.USER, content="Use echo."),),
                model="stub-model",
                native_tools=registry,
                tool_schemas=registry.schemas(),
                hook_context=_hook_context(
                    HookEvent.TOOL_BEFORE,
                    HookActionType.DENY,
                    when={"tool_names": ["echo"]},
                    params={"reason": "echo blocked"},
                ),
                max_steps=2,
            )

        self.assertEqual(calls, [])
        self.assertTrue(result.has_errors())
        self.assertEqual(result.errors()[0].code, "tool_blocked_by_hook")
        self.assertEqual(result.steps[0].tool_results[0].content, "echo blocked")
        self.assertTrue(result.steps[0].tool_results[0].is_error)

    def test_tool_trace_hook_records_native_lifecycle_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")

            def handle(_arguments):
                return NativeToolResult(content="ok")

            registry = NativeToolRegistry(
                (
                    NativeTool(
                        name="echo",
                        description="Echo arguments.",
                        input_schema={"type": "object"},
                        handler=handle,
                    ),
                )
            )
            provider = StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(name="echo", id="call_1"),
                        )
                    ),
                    ModelResponse(content="done"),
                ]
            )
            trace_sink = _TraceSink()
            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="Use echo."),),
                model="stub-model",
                native_tools=registry,
                tool_schemas=registry.schemas(),
                hook_context=_hook_context(
                    HookEvent.TOOL_BEFORE,
                    HookActionType.TRACE,
                    hook_id="trace-echo",
                    when={"tool_names": ["echo"]},
                ),
                trace_recorder=TraceRecorder(trace_sink),
                max_steps=2,
            )

        self.assertFalse(result.has_errors())
        hook_events = [event for event in trace_sink.events if event.kind == "hook_event_evaluated"]
        self.assertTrue(hook_events)
        refs = hook_events[0].refs
        self.assertIn("hook_event=tool.before", refs)
        self.assertIn("hook_directive=continue", refs)
        self.assertIn("hook_id=trace-echo", refs)

    def test_agent_turn_end_hook_records_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            store = HookSignalStore(workspace / "var" / "hooks" / "signals.jsonl")
            provider = StubProvider([ModelResponse(content="done")])

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="hello"),),
                model="stub-model",
                hook_context=HookRuntimeContext.from_registry(
                    HookRegistry.from_hooks(
                        (
                            HookDefinition(
                                hook_id="turn-end-signal",
                                event_name=HookEvent.AGENT_TURN_END,
                                layer=HookLayer.SESSION,
                                actions=(
                                    HookAction(
                                        HookActionType.RECORD_EVOLUTION_SIGNAL,
                                        {
                                            "signal_kind": "turn_completed",
                                            "summary": "Turn end signal.",
                                        },
                                    ),
                                ),
                            ),
                        )
                    ),
                    signal_store=store,
                ),
                max_steps=1,
            )
            signals = store.load_recent()

        self.assertFalse(result.has_errors())
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].hook_id, "turn-end-signal")
        self.assertEqual(signals[0].event_name, HookEvent.AGENT_TURN_END.value)

    def test_agent_loop_records_malformed_tool_request_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            provider = StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(  # type: ignore[arg-type]
                                name=None,
                                id="call_bad",
                            ),
                        )
                    )
                ]
            )
            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="Use a tool."),),
                model="stub-model",
                max_steps=1,
            )

        self.assertTrue(result.has_errors())
        self.assertEqual(result.errors()[0].code, "tool_request_invalid")

    def test_agent_loop_rejects_tool_request_without_id_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            (workspace / "README.md").write_text("Read me.", encoding="utf-8")
            provider = StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="read_text_file",
                                arguments={"path": "README.md"},
                            ),
                        )
                    )
                ]
            )
            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="Use a tool."),),
                model="stub-model",
                native_tools=NativeToolRegistry(),
                max_steps=1,
            )

        self.assertTrue(result.has_errors())
        self.assertEqual(result.errors()[0].code, "tool_request_invalid")
        self.assertIn("tool call id", result.errors()[0].message)

    def test_registered_native_tool_runs_even_when_schema_is_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            calls: list[dict[str, object]] = []

            def handle(arguments):
                calls.append(dict(arguments))
                return NativeToolResult(content="hidden ok")

            registry = NativeToolRegistry(
                (
                    NativeTool(
                        name="hidden_tool",
                        description="Hidden but registered.",
                        input_schema={"type": "object"},
                        handler=handle,
                        exposed_by_default=False,
                    ),
                )
            )
            provider = StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="hidden_tool",
                                id="call_1",
                                arguments={"value": "ok"},
                            ),
                        )
                    ),
                    ModelResponse(content="done"),
                ]
            )
            trace_sink = _TraceSink()

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="Use hidden tool."),),
                model="stub-model",
                native_tools=registry,
                tool_schemas=(),
                trace_recorder=TraceRecorder(trace_sink),
                max_steps=2,
            )

        self.assertFalse(result.has_errors())
        self.assertEqual(calls, [{"value": "ok"}])
        self.assertEqual(result.final_step().response.content, "done")
        self.assertIn("native_tool_schema_hidden", [event.kind for event in trace_sink.events])

    def test_run_command_alias_dispatches_to_shell_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            registry = NativeToolRegistry(
                shell_tools(
                    workspace,
                    shell_enabled=True,
                    network_enabled=False,
                )
            )
            provider = StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="run_command",
                                id="call_1",
                                arguments={"command": "pwd"},
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
                messages=(Message(role=MessageRole.USER, content="Run pwd."),),
                model="stub-model",
                native_tools=registry,
                tool_schemas=registry.schemas(),
                tool_access_policy=ToolAccessPolicy(shell_enabled=True),
                max_steps=2,
            )

        self.assertFalse(result.has_errors())
        self.assertEqual(result.steps[0].tool_results[0].name, "run_shell_command")
        self.assertEqual(
            result.tool_exchanges[0].tool_requests[0].name,
            "run_shell_command",
        )
        self.assertEqual(result.final_step().response.content, "done")

    def test_duplicate_tool_call_ids_are_normalized_for_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            calls: list[dict[str, object]] = []

            def handle(arguments):
                calls.append(dict(arguments))
                return NativeToolResult(content=str(arguments.get("text", "")))

            registry = NativeToolRegistry(
                (
                    NativeTool(
                        name="echo",
                        description="Echo arguments.",
                        input_schema={"type": "object"},
                        handler=handle,
                    ),
                )
            )
            provider = StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="echo",
                                arguments={"text": "first"},
                                id="call_1",
                            ),
                            ModelToolRequest(
                                name="echo",
                                arguments={"text": "second"},
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
                messages=(Message(role=MessageRole.USER, content="Use echo twice."),),
                model="stub-model",
                native_tools=registry,
                tool_schemas=registry.schemas(),
                max_steps=2,
            )

        self.assertFalse(result.has_errors())
        self.assertEqual(calls, [{"text": "first"}, {"text": "second"}])
        exchange = result.tool_exchanges[0]
        self.assertTrue(exchange.is_ready())
        self.assertEqual(
            tuple(request.id for request in exchange.tool_requests),
            ("call_1", "call_1_2"),
        )
        self.assertEqual(
            tuple(tool_result.request_id for tool_result in exchange.tool_results),
            ("call_1", "call_1_2"),
        )
        self.assertIn("tool_request_replay_normalized", [event.kind for event in result.events()])

    def test_tool_exchanges_append_after_existing_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            prior = (
                ModelConversationItem.from_message(
                    Message(role=MessageRole.USER, content="Earlier prompt.")
                ),
                ModelConversationItem.from_message(
                    Message(role=MessageRole.ASSISTANT, content="Earlier answer.")
                ),
            )
            replay = ModelToolExchange(
                assistant_content="I used echo earlier.",
                tool_requests=(
                    ModelToolRequest(
                        name="echo",
                        arguments={"text": "old"},
                        id="call_old",
                    ),
                ),
                tool_results=(
                    ModelToolResult(
                        name="echo",
                        request_id="call_old",
                        content="old",
                    ),
                ),
            )
            history_items: list[ModelConversationItem] = []
            provider = StubProvider([ModelResponse(content="done")])

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="Continue."),),
                model="stub-model",
                conversation=prior,
                tool_exchanges=(replay,),
                history_sink=history_items.append,
                max_steps=1,
            )

        self.assertFalse(result.has_errors())
        self.assertEqual(len(provider.requests), 1)
        request_items = provider.requests[0].conversation
        message_contents = [
            item.message.content for item in request_items if item.message is not None
        ]
        self.assertIn("Earlier prompt.", message_contents)
        self.assertIn("Earlier answer.", message_contents)
        self.assertIn("Continue.", message_contents)
        replay_items = [item for item in request_items if item.tool_exchange is not None]
        self.assertEqual(len(replay_items), 1)
        self.assertEqual(replay_items[0].tool_exchange.tool_results[0].content, "old")
        self.assertIsNotNone(result.conversation[-1].message)
        self.assertEqual(result.conversation[-1].message.content, "done")
        self.assertEqual(len(history_items), 3)
        self.assertEqual(history_items[0].message.content, "Continue.")
        self.assertIsNotNone(history_items[1].tool_exchange)
        self.assertEqual(history_items[2].message.content, "done")

    def test_native_tool_invalid_name_does_not_crash(self) -> None:
        result = execute_native_tool_request(
            ModelToolRequest(name=None),  # type: ignore[arg-type]
            NativeToolRegistry(),
        )

        self.assertEqual(result.error.code, "tool_request_invalid")
        self.assertIsNone(result.model_result)

    def test_mcp_tool_invalid_name_does_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executor = McpToolExecutor(servers=(), tools=(), workspace=Path(tmp))
            result = executor.execute(
                ModelToolRequest(name=None)  # type: ignore[arg-type]
            )

        self.assertEqual(result.error.code, "mcp_tool_request_invalid")
        self.assertIsNone(result.model_result)

    def test_repeated_tool_call_is_suppressed_after_repair_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            calls: list[dict[str, object]] = []

            def handle(arguments):
                calls.append(dict(arguments))
                return NativeToolResult(content="same result")

            registry = NativeToolRegistry(
                (
                    NativeTool(
                        name="echo",
                        description="Echo arguments.",
                        input_schema={
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                        },
                        handler=handle,
                    ),
                )
            )
            provider = StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="echo",
                                arguments={"text": "again"},
                                id="call_1",
                            ),
                        )
                    ),
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="echo",
                                arguments={"text": "again"},
                                id="call_2",
                            ),
                        )
                    ),
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="echo",
                                arguments={"text": "again"},
                                id="call_3",
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
                messages=(Message(role=MessageRole.USER, content="Use echo."),),
                model="stub-model",
                native_tools=registry,
                tool_schemas=registry.schemas(),
                max_steps=4,
            )

        self.assertFalse(result.has_errors())
        self.assertEqual(len(calls), 2)
        self.assertEqual(result.final_step().response.content, "done")
        suppressed = result.steps[2].tool_results[0]
        self.assertTrue(suppressed.is_error)
        self.assertIn("Repeated tool call suppressed", suppressed.content)

    def test_truncated_tool_arguments_are_repaired_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            calls: list[dict[str, object]] = []
            statuses: list[str] = []

            def handle(arguments):
                calls.append(dict(arguments))
                return NativeToolResult(content="ok")

            registry = NativeToolRegistry(
                (
                    NativeTool(
                        name="echo",
                        description="Echo arguments.",
                        input_schema={
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                        handler=handle,
                    ),
                )
            )
            provider = StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="echo",
                                arguments={},
                                raw_arguments='{"text":"hello"',
                                argument_error="JSONDecodeError: truncated",
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
                messages=(Message(role=MessageRole.USER, content="Use echo."),),
                model="stub-model",
                native_tools=registry,
                tool_schemas=registry.schemas(),
                max_steps=2,
                status_sink=statuses.append,
            )

        self.assertFalse(result.has_errors())
        self.assertEqual(calls, [{"text": "hello"}])
        self.assertIn("tool_arguments_repaired", [event.kind for event in result.events()])
        self.assertTrue(any("tool arguments repaired: echo" in status for status in statuses))
        replay_exchange = provider.requests[1].conversation[-1].tool_exchange
        self.assertIsNotNone(replay_exchange)
        self.assertEqual(replay_exchange.tool_requests[0].raw_arguments, '{"text":"hello"}')

    def test_reasoning_tool_call_json_is_scavenged_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            calls: list[dict[str, object]] = []
            statuses: list[str] = []

            def handle(arguments):
                calls.append(dict(arguments))
                return NativeToolResult(content="ok")

            registry = NativeToolRegistry(
                (
                    NativeTool(
                        name="echo",
                        description="Echo arguments.",
                        input_schema={
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                        handler=handle,
                    ),
                )
            )
            provider = StubProvider(
                [
                    ModelResponse(
                        reasoning=(
                            '{"name":"echo","arguments":{"text":"from reasoning"}}'
                        )
                    ),
                    ModelResponse(content="done"),
                ]
            )

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="Use echo."),),
                model="stub-model",
                native_tools=registry,
                tool_schemas=registry.schemas(),
                max_steps=2,
                status_sink=statuses.append,
            )

        self.assertFalse(result.has_errors())
        self.assertEqual(calls, [{"text": "from reasoning"}])
        self.assertIn("tool_call_scavenged", [event.kind for event in result.events()])
        self.assertTrue(any("tool call scavenged: echo" in status for status in statuses))
        replay_exchange = provider.requests[1].conversation[-1].tool_exchange
        self.assertIsNotNone(replay_exchange)
        self.assertEqual(replay_exchange.tool_requests[0].id, "scavenged_1_1")
        self.assertEqual(
            replay_exchange.tool_requests[0].arguments,
            {"text": "from reasoning"},
        )

    def test_tool_repair_policy_can_disable_reasoning_scavenge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            calls: list[dict[str, object]] = []

            def handle(arguments):
                calls.append(dict(arguments))
                return NativeToolResult(content="ok")

            registry = NativeToolRegistry(
                (
                    NativeTool(
                        name="echo",
                        description="Echo arguments.",
                        input_schema={
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                        },
                        handler=handle,
                    ),
                )
            )
            provider = StubProvider(
                [
                    ModelResponse(
                        reasoning=(
                            '{"name":"echo","arguments":{"text":"from reasoning"}}'
                        )
                    ),
                ]
            )

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="Use echo."),),
                model="stub-model",
                native_tools=registry,
                tool_schemas=registry.schemas(),
                tool_repair_policy=ToolRepairPolicy(reasoning_scavenge=False),
                max_steps=1,
            )

        self.assertFalse(result.has_errors())
        self.assertEqual(calls, [])
        self.assertNotIn("tool_call_scavenged", [event.kind for event in result.events()])

    def test_plain_json_in_assistant_text_is_not_scavenged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            calls: list[dict[str, object]] = []

            def handle(arguments):
                calls.append(dict(arguments))
                return NativeToolResult(content="should not run")

            registry = NativeToolRegistry(
                (
                    NativeTool(
                        name="echo",
                        description="Echo arguments.",
                        input_schema={"type": "object"},
                        handler=handle,
                    ),
                )
            )
            provider = StubProvider(
                [
                    ModelResponse(
                        content=(
                            'Example JSON: {"name":"echo","arguments":{"text":"demo"}}'
                        )
                    )
                ]
            )

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="Explain JSON."),),
                model="stub-model",
                native_tools=registry,
                tool_schemas=registry.schemas(),
                max_steps=1,
            )

        self.assertFalse(result.has_errors())
        self.assertEqual(calls, [])
        self.assertNotIn("tool_call_scavenged", [event.kind for event in result.events()])

    def test_reasoning_example_json_is_not_scavenged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            calls: list[dict[str, object]] = []

            def handle(arguments):
                calls.append(dict(arguments))
                return NativeToolResult(content="should not run")

            registry = NativeToolRegistry(
                (
                    NativeTool(
                        name="echo",
                        description="Echo arguments.",
                        input_schema={"type": "object"},
                        handler=handle,
                    ),
                )
            )
            provider = StubProvider(
                [
                    ModelResponse(
                        reasoning=(
                            'I might call {"name":"echo","arguments":{"text":"demo"}} '
                            "if this were a real tool call."
                        )
                    )
                ]
            )

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="Explain JSON."),),
                model="stub-model",
                native_tools=registry,
                tool_schemas=registry.schemas(),
                max_steps=1,
            )

        self.assertFalse(result.has_errors())
        self.assertEqual(calls, [])
        self.assertNotIn("tool_call_scavenged", [event.kind for event in result.events()])

    def test_tagged_assistant_tool_call_json_is_scavenged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            calls: list[dict[str, object]] = []

            def handle(arguments):
                calls.append(dict(arguments))
                return NativeToolResult(content="ok")

            registry = NativeToolRegistry(
                (
                    NativeTool(
                        name="echo",
                        description="Echo arguments.",
                        input_schema={"type": "object"},
                        handler=handle,
                    ),
                )
            )
            provider = StubProvider(
                [
                    ModelResponse(
                        content=(
                            '<tool_call>{"name":"echo","arguments":{"text":"tagged"}}</tool_call>'
                        )
                    ),
                    ModelResponse(content="done"),
                ]
            )

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="Use echo."),),
                model="stub-model",
                native_tools=registry,
                tool_schemas=registry.schemas(),
                max_steps=2,
            )

        self.assertFalse(result.has_errors())
        self.assertEqual(calls, [{"text": "tagged"}])
        self.assertIn("tool_call_scavenged", [event.kind for event in result.events()])

    def test_schema_additions_replace_existing_schema_for_followup_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            trace_sink = _TraceSink()

            def load_schema(_arguments):
                return NativeToolResult(
                    content="schema loaded",
                    schema_additions=(
                        {
                            "name": "echo",
                            "description": "Updated echo schema.",
                            "input_schema": {
                                "type": "object",
                                "properties": {"message": {"type": "string"}},
                            },
                        },
                    ),
                )

            registry = NativeToolRegistry(
                (
                    NativeTool(
                        name="load_echo",
                        description="Load echo schema.",
                        input_schema={"type": "object"},
                        handler=load_schema,
                    ),
                    NativeTool(
                        name="echo",
                        description="Old echo schema.",
                        input_schema={"type": "object"},
                        handler=lambda _arguments: NativeToolResult(content="ok"),
                        exposed_by_default=False,
                    ),
                )
            )
            provider = StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(name="load_echo", id="call_1"),
                        )
                    ),
                    ModelResponse(content="done"),
                ]
            )

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="Load schema."),),
                model="stub-model",
                native_tools=registry,
                tool_schemas=(
                    {
                        "name": "echo",
                        "description": "Old echo schema.",
                        "input_schema": {"type": "object"},
                    },
                    *registry.schemas(),
                ),
                trace_recorder=TraceRecorder(trace_sink),
                max_steps=2,
            )

        self.assertFalse(result.has_errors())
        schemas = {schema["name"]: schema for schema in provider.requests[1].tool_schemas}
        self.assertEqual(schemas["echo"]["description"], "Updated echo schema.")
        self.assertEqual(
            tuple(schema["name"] for schema in provider.requests[1].tool_schemas).count("echo"),
            1,
        )
        stability_events = [
            event
            for event in trace_sink.events
            if event.kind == "model_request_prefix_stability"
        ]
        self.assertEqual(len(stability_events), 2)
        self.assertIn("prefix_stable=baseline", stability_events[0].refs)
        self.assertIn("tool_schema_changed=true", stability_events[1].refs)
        self.assertIn("changed_parts=tool_schema", stability_events[1].refs)

    def test_similar_search_tool_calls_are_suppressed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            calls: list[dict[str, object]] = []

            def handle(arguments):
                calls.append(dict(arguments))
                return NativeToolResult(content="No matches.")

            registry = NativeToolRegistry(
                (
                    NativeTool(
                        name="search_docs",
                        description="Search docs.",
                        input_schema={
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                        handler=handle,
                    ),
                )
            )
            provider = StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="search_docs",
                                arguments={"query": "Auth flow"},
                                id="call_1",
                            ),
                        )
                    ),
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="search_docs",
                                arguments={"query": "auth   flow!"},
                                id="call_2",
                            ),
                        )
                    ),
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="search_docs",
                                arguments={"query": "auth-flow"},
                                id="call_3",
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
                messages=(Message(role=MessageRole.USER, content="Search docs."),),
                model="stub-model",
                native_tools=registry,
                tool_schemas=registry.schemas(),
                max_steps=4,
            )

        self.assertFalse(result.has_errors())
        self.assertEqual(len(calls), 2)
        suppressed = result.steps[2].tool_results[0]
        self.assertTrue(suppressed.is_error)
        self.assertIn("repeat_kind=similar", suppressed.refs)
        self.assertIn("tool_call_similar_suppressed", [event.kind for event in result.events()])

    def test_runtime_status_sink_reports_usage_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            statuses: list[str] = []
            provider = StubProvider(
                [
                    ModelResponse(
                        content="ok",
                        usage=TokenUsage(
                            input_tokens=100,
                            output_tokens=20,
                            cache_hit_input_tokens=80,
                            cache_miss_input_tokens=20,
                        ),
                    )
                ]
            )

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="hello"),),
                model="stub-model",
                max_steps=1,
                status_sink=statuses.append,
            )

        self.assertFalse(result.has_errors())
        self.assertTrue(statuses)
        usage_status = "\n".join(statuses)
        self.assertIn("actual_input_tokens=100", usage_status)
        self.assertIn("cache_hit_ratio=0.800", usage_status)

    def test_agent_loop_compacts_large_native_tool_output_before_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            statuses: list[str] = []
            trace_sink = _TraceSink()

            def handle(_arguments):
                return NativeToolResult(
                    content="\n".join(
                        f"FAILED tests/test_example.py::test_{index} AssertionError"
                        for index in range(4_000)
                    )
                )

            registry = NativeToolRegistry(
                (
                    NativeTool(
                        name="run_tests",
                        description="Run tests.",
                        input_schema={"type": "object"},
                        handler=handle,
                    ),
                )
            )
            provider = StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(name="run_tests", id="call-1"),
                        )
                    ),
                    ModelResponse(content="done"),
                ]
            )
            compactor = ToolOutputCompactor(
                store=ToolOutputStore.in_data_dir(workspace / "var", "default", "s1"),
                policy=ConversationBudgetPolicy(
                    model_context_tokens=100_000,
                    response_token_reserve=0,
                    safety_margin_tokens=0,
                ),
            )

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="Run tests."),),
                model="stub-model",
                native_tools=registry,
                tool_schemas=registry.schemas(),
                conversation_budget_policy=ConversationBudgetPolicy(
                    model_context_tokens=100_000,
                    response_token_reserve=0,
                    safety_margin_tokens=0,
                ),
                tool_output_compactor=compactor,
                trace_recorder=TraceRecorder(trace_sink),
                status_sink=statuses.append,
                max_steps=2,
            )

        self.assertFalse(result.has_errors())
        compacted = result.steps[0].tool_results[0]
        self.assertIn("[tool output compacted: log]", compacted.content)
        self.assertIn("tool_output_compacted=true", compacted.refs)
        replay_exchange = provider.requests[1].conversation[-1].tool_exchange
        self.assertIsNotNone(replay_exchange)
        self.assertIn("[tool output compacted: log]", replay_exchange.tool_results[0].content)
        self.assertIn("tool_output_compacted", [event.kind for event in trace_sink.events])
        self.assertTrue(any("tool output compacted" in status for status in statuses))

    def test_post_edit_diagnostics_replays_failure_and_allows_self_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")

            def write_file(arguments):
                path = workspace / str(arguments["path"])
                path.write_text(str(arguments["content"]), encoding="utf-8")
                return NativeToolResult(
                    content=f"Wrote {arguments['path']}",
                    data={"path": str(arguments["path"])},
                    refs=(str(arguments["path"]),),
                )

            registry = NativeToolRegistry(
                (
                    NativeTool(
                        name="write_text_file",
                        description="Write a file.",
                        input_schema={"type": "object"},
                        handler=write_file,
                        read_only=False,
                    ),
                )
            )
            statuses: list[str] = []
            trace_sink = _TraceSink()
            provider = StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="write_text_file",
                                id="call_1",
                                arguments={
                                    "path": "bad.py",
                                    "content": "def broken(:\n    pass\n",
                                },
                            ),
                        )
                    ),
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="write_text_file",
                                id="call_2",
                                arguments={
                                    "path": "bad.py",
                                    "content": "def fixed():\n    return 'ok'\n",
                                },
                            ),
                        )
                    ),
                    ModelResponse(content="fixed"),
                ]
            )

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="Write Python."),),
                model="stub-model",
                native_tools=registry,
                tool_schemas=registry.schemas(),
                tool_access_policy=ToolAccessPolicy(ToolAccessMode.WORKSPACE_WRITE),
                trace_recorder=TraceRecorder(trace_sink),
                status_sink=statuses.append,
                max_steps=3,
            )

        self.assertFalse(result.has_errors())
        self.assertEqual(result.final_step().response.content, "fixed")
        first_result = result.steps[0].tool_results[0]
        self.assertTrue(first_result.is_error)
        self.assertIn("Post-edit diagnostics found issues", first_result.content)
        replay_exchange = provider.requests[1].conversation[-1].tool_exchange
        self.assertIsNotNone(replay_exchange)
        self.assertTrue(replay_exchange.tool_results[0].is_error)
        self.assertIn("post_edit_diagnostics_failed", [event.kind for event in trace_sink.events])
        self.assertIn("post_edit_diagnostics_passed", [event.kind for event in trace_sink.events])
        self.assertTrue(any("post-edit diagnostics" in status for status in statuses))

    def test_turn_cost_cache_summary_is_emitted_once_per_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            statuses: list[str] = []
            trace_sink = _TraceSink()
            provider = StubProvider(
                [
                    ModelResponse(
                        content="ok",
                        usage=TokenUsage(
                            input_tokens=100,
                            output_tokens=20,
                            cache_hit_input_tokens=75,
                            cache_miss_input_tokens=25,
                            reasoning_tokens=5,
                        ),
                    )
                ]
            )

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="hello"),),
                model="stub-model",
                trace_recorder=TraceRecorder(trace_sink),
                status_sink=statuses.append,
                max_steps=1,
            )

        self.assertFalse(result.has_errors())
        self.assertTrue(
            any("turn cost/cache summary" in status for status in statuses)
        )
        event = next(
            event for event in trace_sink.events if event.kind == "turn_cost_cache_summary"
        )
        self.assertIn("input_tokens=100", event.refs)
        self.assertIn("cache_hit_ratio=0.7500", event.refs)

    def test_prefix_fingerprint_stays_stable_across_tool_replay_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            trace_sink = _TraceSink()
            registry = NativeToolRegistry(
                (
                    NativeTool(
                        name="lookup",
                        description="Lookup a value.",
                        input_schema={"type": "object"},
                        handler=lambda _arguments: NativeToolResult(content="value"),
                    ),
                )
            )
            provider = StubProvider(
                [
                    ModelResponse(
                        tool_requests=(ModelToolRequest(name="lookup", id="call-1"),)
                    ),
                    ModelResponse(content="done"),
                ]
            )

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="lookup"),),
                model="stub-model",
                native_tools=registry,
                tool_schemas=registry.schemas(),
                trace_recorder=TraceRecorder(trace_sink),
                max_steps=2,
            )

        self.assertFalse(result.has_errors())
        events = [
            event
            for event in trace_sink.events
            if event.kind == "model_request_prefix_fingerprint"
        ]
        self.assertEqual(len(events), 2)
        digests = {_ref_value(event.refs, "prefix_digest") for event in events}
        self.assertEqual(len(digests), 1)
        self.assertIn("tool_schema_count=1", events[0].refs)
        stability_events = [
            event
            for event in trace_sink.events
            if event.kind == "model_request_prefix_stability"
        ]
        self.assertEqual(len(stability_events), 2)
        self.assertIn("prefix_stable=baseline", stability_events[0].refs)
        self.assertIn("prefix_stable=true", stability_events[1].refs)
        self.assertIn("stable_request_count=2", stability_events[1].refs)

    def test_runtime_records_turn_model_and_tool_spans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")

            def handle(_arguments):
                return NativeToolResult(content="ok", refs=("result_ref=ok",))

            registry = NativeToolRegistry(
                (
                    NativeTool(
                        name="echo",
                        description="Echo arguments.",
                        input_schema={"type": "object"},
                        handler=handle,
                    ),
                )
            )
            provider = StubProvider(
                [
                    ModelResponse(
                        tool_requests=(ModelToolRequest(name="echo", id="call_1"),),
                        usage=TokenUsage(
                            input_tokens=100,
                            output_tokens=12,
                            cache_hit_input_tokens=80,
                            cache_miss_input_tokens=20,
                        ),
                        finish_reason="tool_calls",
                    ),
                    ModelResponse(
                        content="done",
                        usage=TokenUsage(input_tokens=120, output_tokens=8),
                        finish_reason="stop",
                    ),
                ]
            )
            trace_sink = _TraceSink()

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="Use echo."),),
                model="stub-model",
                native_tools=registry,
                tool_schemas=registry.schemas(),
                trace_recorder=TraceRecorder(trace_sink),
                max_steps=2,
            )

        self.assertFalse(result.has_errors())
        spans = [record for record in trace_sink.events if isinstance(record, TraceSpan)]
        names = [span.name for span in spans]
        self.assertIn("deepmate turn", names)
        self.assertEqual(names.count("chat stub-model"), 2)
        self.assertIn("execute_tool echo", names)

        turn_span = next(span for span in spans if span.name == "deepmate turn")
        self.assertEqual(turn_span.status, "OK")
        self.assertEqual(turn_span.attributes["gen_ai.operation.name"], "invoke_agent")
        self.assertEqual(turn_span.attributes["gen_ai.usage.input_tokens"], 220)
        self.assertEqual(turn_span.attributes["deepmate.usage.cache_miss_input_tokens"], 20)

        model_span = next(span for span in spans if span.name == "chat stub-model")
        self.assertEqual(model_span.kind, "CLIENT")
        self.assertEqual(model_span.attributes["gen_ai.operation.name"], "chat")
        self.assertEqual(model_span.attributes["gen_ai.usage.input_tokens"], 100)
        self.assertEqual(
            model_span.attributes["deepmate.usage.cache_miss_input_tokens"],
            20,
        )

        tool_span = next(span for span in spans if span.name == "execute_tool echo")
        self.assertEqual(tool_span.status, "OK")
        self.assertEqual(tool_span.attributes["gen_ai.operation.name"], "execute_tool")
        self.assertEqual(tool_span.attributes["gen_ai.tool.name"], "echo")
        self.assertEqual(tool_span.attributes["deepmate.tool.result_refs"], ["result_ref=ok"])


class _TraceSink:
    def __init__(self) -> None:
        self.events = []

    def write(self, event) -> None:
        self.events.append(event)


def _ref_value(refs: tuple[str, ...], key: str) -> str:
    prefix = f"{key}="
    for ref in refs:
        if ref.startswith(prefix):
            return ref.split("=", 1)[1]
    raise AssertionError(f"{key} missing from refs")


def _hook_context(
    event_name: HookEvent,
    action_type: HookActionType,
    *,
    hook_id: str = "test-hook",
    when: dict[str, object] | None = None,
    params: dict[str, object] | None = None,
) -> HookRuntimeContext:
    return HookRuntimeContext.from_registry(
        HookRegistry.from_hooks(
            (
                HookDefinition(
                    hook_id=hook_id,
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
