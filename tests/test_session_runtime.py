from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deepmate.capabilities import CapabilitySurface
from deepmate.domain import CapabilityKind, CapabilityRef
from deepmate.domain import Message, MessageRole, ProfileRef
from deepmate.providers import (
    ModelCapabilities,
    ModelResponse,
    ModelToolRequest,
    StreamDelta,
)
from deepmate.runtime import (
    ConversationBudgetPolicy,
    TurnFollowupBuffer,
    start_runtime_activation,
    start_session_runtime,
)
from deepmate.skills import SkillDocument
from deepmate.subagents import (
    SubagentRuntime,
    SubagentToolExecutor,
    subagent_tool_schema,
)
from deepmate.tools import NativeTool, NativeToolRegistry, NativeToolResult
from deepmate.trace import TraceRecorder


class StubProvider:
    def __init__(self, response: ModelResponse | list[ModelResponse]) -> None:
        self.responses = response if isinstance(response, list) else [response]
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("stub provider received too many requests")
        return self.responses.pop(0)


class StreamingStubProvider:
    """Stub provider that emits content deltas before returning the response.

    Mirrors the real provider contract: complete_stream feeds fragments to
    on_delta and returns the assembled ModelResponse, so it exercises the full
    streaming pipeline (agent_loop -> session_runtime -> token_sink).
    """

    def __init__(self, fragments: list[str]) -> None:
        self.fragments = fragments
        self.complete_calls = 0
        self.stream_calls = 0

    def complete(self, request):
        self.complete_calls += 1
        return ModelResponse(content="".join(self.fragments))

    def complete_stream(self, request, on_delta):
        self.stream_calls += 1
        for fragment in self.fragments:
            on_delta(StreamDelta(content=fragment))
        return ModelResponse(content="".join(self.fragments))


class SessionRuntimeTests(unittest.TestCase):
    def test_activation_snapshot_stays_frozen_without_refresh_request(self) -> None:
        with _workspace() as workspace:
            profile = ProfileRef(name="default", uri="profiles/default")
            activation = start_runtime_activation(
                session_id="session_1",
                workspace=workspace,
                profile=profile,
            )
            runtime = start_session_runtime(activation)
            (workspace / "profiles" / "default" / "memory.md").write_text(
                "- 用户喜欢北京。\n",
                encoding="utf-8",
            )
            provider = StubProvider(ModelResponse(content="ok"))
            trace_sink = _TraceSink()

            result = runtime.run_user_turn(
                provider=provider,
                messages=(Message(role=MessageRole.USER, content="继续"),),
                model="stub-model",
                trace_recorder=TraceRecorder(trace_sink),
            )

            self.assertEqual(result.runtime.activation.context_epoch, 1)
            system_message = provider.requests[0].conversation[0].message
            self.assertIsNotNone(system_message)
            self.assertNotIn("用户喜欢北京", system_message.content)
            self.assertIn(
                "context_invariants_checked",
                tuple(event.kind for event in trace_sink.events),
            )

    def test_streaming_provider_pipes_deltas_to_token_sink(self) -> None:
        with _workspace() as workspace:
            profile = ProfileRef(name="default", uri="profiles/default")
            activation = start_runtime_activation(
                session_id="session_1",
                workspace=workspace,
                profile=profile,
            )
            runtime = start_session_runtime(activation)
            provider = StreamingStubProvider(["Hel", "lo ", "world"])
            deltas = []

            result = runtime.run_user_turn(
                provider=provider,
                messages=(Message(role=MessageRole.USER, content="hi"),),
                model="stub-model",
                token_sink=deltas.append,
            )

            # The whole pipeline routed deltas to the sink in order...
            self.assertEqual([d.content for d in deltas], ["Hel", "lo ", "world"])
            # ...streaming was used (not the blocking path)...
            self.assertEqual(provider.stream_calls, 1)
            self.assertEqual(provider.complete_calls, 0)
            # ...and the assembled response is identical to non-streaming.
            self.assertEqual(
                result.runtime.last_user_turn_result.final_step().response.content,
                "Hello world",
            )

    def test_no_token_sink_uses_blocking_completion(self) -> None:
        with _workspace() as workspace:
            profile = ProfileRef(name="default", uri="profiles/default")
            activation = start_runtime_activation(
                session_id="session_1",
                workspace=workspace,
                profile=profile,
            )
            runtime = start_session_runtime(activation)
            provider = StreamingStubProvider(["a", "b"])

            runtime.run_user_turn(
                provider=provider,
                messages=(Message(role=MessageRole.USER, content="hi"),),
                model="stub-model",
            )

            # Without a sink, the loop must not stream.
            self.assertEqual(provider.stream_calls, 0)
            self.assertEqual(provider.complete_calls, 1)

    def test_model_capabilities_sanitize_request_before_budget_and_provider(self) -> None:
        with _workspace() as workspace:
            profile = ProfileRef(name="default", uri="profiles/default")
            activation = start_runtime_activation(
                session_id="session_1",
                workspace=workspace,
                profile=profile,
            )
            runtime = start_session_runtime(activation)
            provider = StubProvider(ModelResponse(content="ok"))

            result = runtime.run_user_turn(
                provider=provider,
                messages=(Message(role=MessageRole.USER, content="hello"),),
                model="local-model",
                tool_schemas=(
                    {
                        "name": "search",
                        "description": "Search.",
                        "input_schema": {"type": "object"},
                    },
                ),
                options={
                    "thinking": {"type": "enabled"},
                    "stream_options": {"include_usage": True},
                    "max_tokens": 64,
                },
                model_capabilities=ModelCapabilities(
                    supports_tools=False,
                    supports_thinking=False,
                    supports_stream_usage=False,
                ),
            )

            request = provider.requests[0]
            self.assertEqual(request.tool_schemas, ())
            self.assertEqual(request.options, {"max_tokens": 64})
            report = result.runtime.last_user_turn_result.final_step().request_budget_report
            self.assertIsNotNone(report)
            self.assertEqual(report.tool_schema_count, 0)

    def test_profile_context_change_refreshes_before_next_user_turn(self) -> None:
        with _workspace() as workspace:
            profile = ProfileRef(name="default", uri="profiles/default")
            activation = start_runtime_activation(
                session_id="session_1",
                workspace=workspace,
                profile=profile,
            )
            runtime = start_session_runtime(activation)

            (workspace / "profiles" / "default" / "memory.md").write_text(
                "- 用户喜欢北京。\n",
                encoding="utf-8",
            )
            runtime = runtime.mark_profile_context_changed()

            self.assertTrue(runtime.refresh_context_before_next_turn)

            provider = StubProvider(ModelResponse(content="ok"))
            result = runtime.run_user_turn(
                provider=provider,
                messages=(Message(role=MessageRole.USER, content="继续"),),
                model="stub-model",
            )

            self.assertEqual(result.runtime.activation.context_epoch, 2)
            self.assertFalse(result.runtime.profile_context_changed)
            self.assertFalse(result.runtime.refresh_context_before_next_turn)
            system_message = provider.requests[0].conversation[0].message
            self.assertIsNotNone(system_message)
            self.assertIn("用户喜欢北京", system_message.content)

    def test_subagent_runtime_uses_refreshed_activation_after_context_change(self) -> None:
        with _workspace() as workspace:
            profile = ProfileRef(name="default", uri="profiles/default")
            memory_path = workspace / "profiles" / "default" / "memory.md"
            memory_path.write_text("- stale child context marker.\n", encoding="utf-8")
            activation = start_runtime_activation(
                session_id="session_1",
                workspace=workspace,
                profile=profile,
            )
            runtime = start_session_runtime(activation)
            memory_path.write_text(
                "- refreshed child context marker.\n",
                encoding="utf-8",
            )
            runtime = runtime.mark_profile_context_changed()
            provider = StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="run_subagent",
                                id="call_child",
                                arguments={
                                    "goal": "Check the refreshed project memory.",
                                    "max_steps": 1,
                                },
                            ),
                        )
                    ),
                    ModelResponse(content="child saw refreshed memory"),
                    ModelResponse(content="parent merged child"),
                ]
            )

            def runtime_factory() -> SubagentRuntime:
                return SubagentRuntime(
                    provider=provider,
                    workspace=workspace,
                    profile=profile,
                    model="stub-model",
                    activation=activation,
                )

            result = runtime.run_user_turn(
                provider=provider,
                messages=(Message(role=MessageRole.USER, content="Delegate it."),),
                model="stub-model",
                subagents=SubagentToolExecutor(runtime_factory=runtime_factory),
                tool_schemas=(subagent_tool_schema(),),
                max_steps=2,
            )

            self.assertEqual(result.runtime.activation.context_epoch, 2)
            self.assertEqual(len(provider.requests), 3)
            child_system = provider.requests[1].conversation[0].message
            self.assertIsNotNone(child_system)
            self.assertIn("refreshed child context marker", child_system.content)
            self.assertNotIn("stale child context marker", child_system.content)
            self.assertEqual(
                result.runtime.last_user_turn_result.final_step().response.content,
                "parent merged child",
            )

    def test_behavior_context_change_refreshes_before_next_user_turn(self) -> None:
        with _workspace() as workspace:
            profile = ProfileRef(name="default", uri="profiles/default")
            activation = start_runtime_activation(
                session_id="session_1",
                workspace=workspace,
                profile=profile,
            )
            runtime = start_session_runtime(activation)
            behavior_path = workspace / ".deepmate" / "behavior.md"
            behavior_path.parent.mkdir()
            behavior_path.write_text(
                "# Behavior Hints\n\n- Prefer compact closure.\n",
                encoding="utf-8",
            )
            provider = StubProvider(ModelResponse(content="ok"))
            trace_sink = _TraceSink()

            result = runtime.run_user_turn(
                provider=provider,
                messages=(Message(role=MessageRole.USER, content="继续"),),
                model="stub-model",
                trace_recorder=TraceRecorder(trace_sink),
            )

            self.assertEqual(result.runtime.activation.context_epoch, 2)
            self.assertFalse(result.runtime.profile_context_changed)
            self.assertFalse(result.runtime.refresh_context_before_next_turn)
            system_message = provider.requests[0].conversation[0].message
            self.assertIsNotNone(system_message)
            self.assertIn("Prefer compact closure.", system_message.content)
            event_kinds = tuple(event.kind for event in trace_sink.events)
            self.assertIn("behavior_context_changed", event_kinds)
            self.assertIn("context_snapshot_refreshed", event_kinds)

    def test_history_budget_pressure_requests_next_context_refresh(self) -> None:
        with _workspace() as workspace:
            profile = ProfileRef(name="default", uri="profiles/default")
            activation = start_runtime_activation(
                session_id="session_1",
                workspace=workspace,
                profile=profile,
            )
            runtime = start_session_runtime(activation)
            provider = StubProvider(ModelResponse(content="ok"))

            result = runtime.run_user_turn(
                provider=provider,
                messages=(Message(role=MessageRole.USER, content="x" * 2000),),
                model="stub-model",
                conversation_budget_policy=ConversationBudgetPolicy(
                    history_token_budget=1,
                    protect_recent_items=0,
                ),
            )

            self.assertFalse(result.runtime.profile_context_changed)
            self.assertTrue(result.runtime.refresh_context_before_next_turn)

    def test_skill_body_change_invalidates_system_context_cache(self) -> None:
        with _workspace() as workspace:
            skill_path = workspace / "skills" / "writer" / "SKILL.md"
            skill_path.parent.mkdir(parents=True)
            first_skill = SkillDocument(
                name="writer",
                description="Write concise updates.",
                body="Use the first checklist.",
                path=skill_path,
            )
            second_skill = SkillDocument(
                name="writer",
                description="Write concise updates.",
                body="Use the changed checklist.",
                path=skill_path,
            )
            profile = ProfileRef(name="default", uri="profiles/default")
            activation = start_runtime_activation(
                session_id="session_1",
                workspace=workspace,
                profile=profile,
            )
            runtime = start_session_runtime(activation)
            provider = StubProvider(
                [
                    ModelResponse(content="first"),
                    ModelResponse(content="second"),
                    ModelResponse(content="third"),
                ]
            )
            trace_sink = _TraceSink()

            first = runtime.run_user_turn(
                provider=provider,
                messages=(Message(role=MessageRole.USER, content="first"),),
                model="stub-model",
                selected_skill_documents=(first_skill,),
                trace_recorder=TraceRecorder(trace_sink),
            )
            second = first.runtime.run_user_turn(
                provider=provider,
                messages=(Message(role=MessageRole.USER, content="second"),),
                model="stub-model",
                selected_skill_documents=(first_skill,),
                trace_recorder=TraceRecorder(trace_sink),
            )
            second.runtime.run_user_turn(
                provider=provider,
                messages=(Message(role=MessageRole.USER, content="third"),),
                model="stub-model",
                selected_skill_documents=(second_skill,),
                trace_recorder=TraceRecorder(trace_sink),
            )

        cache_events = [
            event.kind
            for event in trace_sink.events
            if event.kind in {"system_context_cache_hit", "system_context_cache_miss"}
        ]
        self.assertEqual(
            cache_events,
            [
                "system_context_cache_miss",
                "system_context_cache_hit",
                "system_context_cache_miss",
            ],
        )

    def test_capability_order_change_invalidates_system_context_cache(self) -> None:
        with _workspace() as workspace:
            profile = ProfileRef(name="default", uri="profiles/default")
            activation = start_runtime_activation(
                session_id="session_1",
                workspace=workspace,
                profile=profile,
            )
            runtime = start_session_runtime(activation)
            provider = StubProvider(
                [
                    ModelResponse(content="first"),
                    ModelResponse(content="second"),
                ]
            )
            trace_sink = _TraceSink()
            read_ref = CapabilityRef(
                kind=CapabilityKind.NATIVE_TOOL,
                name="read_text_file",
                description="Read a workspace file.",
            )
            search_ref = CapabilityRef(
                kind=CapabilityKind.NATIVE_TOOL,
                name="search_workspace",
                description="Search workspace files.",
            )
            first_surface = CapabilitySurface((read_ref, search_ref))
            second_surface = CapabilitySurface((search_ref, read_ref))

            first = runtime.run_user_turn(
                provider=provider,
                messages=(Message(role=MessageRole.USER, content="first"),),
                model="stub-model",
                capability_surface=first_surface,
                trace_recorder=TraceRecorder(trace_sink),
            )
            first.runtime.run_user_turn(
                provider=provider,
                messages=(Message(role=MessageRole.USER, content="second"),),
                model="stub-model",
                capability_surface=second_surface,
                trace_recorder=TraceRecorder(trace_sink),
            )

        cache_events = [
            event.kind
            for event in trace_sink.events
            if event.kind in {"system_context_cache_hit", "system_context_cache_miss"}
        ]
        self.assertEqual(
            cache_events,
            [
                "system_context_cache_miss",
                "system_context_cache_miss",
            ],
        )

    def test_running_followup_injects_before_next_model_request(self) -> None:
        with _workspace() as workspace:
            profile = ProfileRef(name="default", uri="profiles/default")
            activation = start_runtime_activation(
                session_id="session_1",
                workspace=workspace,
                profile=profile,
            )
            runtime = start_session_runtime(activation)
            provider = StubProvider(
                [
                    ModelResponse(
                        content="I will use a tool.",
                        tool_requests=(
                            ModelToolRequest(
                                name="echo",
                                arguments={"text": "hello"},
                                id="call_1",
                            ),
                        ),
                    ),
                    ModelResponse(content="done"),
                ]
            )
            buffer = TurnFollowupBuffer()
            turn_id = buffer.start_turn()
            history_items = []

            def handle_tool(arguments):
                self.assertTrue(
                    buffer.submit(
                        turn_id,
                        "不要改 public API。",
                        source="test",
                    )
                )
                return NativeToolResult(content=str(arguments.get("text", "")))

            registry = NativeToolRegistry(
                (
                    NativeTool(
                        name="echo",
                        description="Echo arguments.",
                        input_schema={"type": "object"},
                        handler=handle_tool,
                    ),
                )
            )

            result = runtime.run_user_turn(
                provider=provider,
                messages=(Message(role=MessageRole.USER, content="Use echo."),),
                model="stub-model",
                native_tools=registry,
                tool_schemas=registry.schemas(),
                followup_buffer=buffer,
                followup_turn_id=turn_id,
                history_sink=history_items.append,
            )

            self.assertEqual(len(provider.requests), 2)
            second_request_text = "\n".join(
                item.message.content
                for item in provider.requests[1].conversation
                if item.message is not None
            )
            self.assertIn("User follow-up while this turn was running", second_request_text)
            self.assertIn("不要改 public API", second_request_text)
            self.assertIn("不要改 public API", "\n".join(
                item.message.content
                for item in history_items
                if item.message is not None
            ))
            self.assertEqual(buffer.pending_count(), 0)
            self.assertEqual(result.runtime.last_user_turn_result.final_step().response.content, "done")

    def test_running_followup_keeps_system_prefix_stable_across_steps(self) -> None:
        with _workspace() as workspace:
            profile = ProfileRef(name="default", uri="profiles/default")
            activation = start_runtime_activation(
                session_id="session_1",
                workspace=workspace,
                profile=profile,
            )
            runtime = start_session_runtime(activation)
            provider = StubProvider(
                [
                    ModelResponse(
                        content="I will use a tool.",
                        tool_requests=(
                            ModelToolRequest(
                                name="echo",
                                arguments={"text": "hello"},
                                id="call_1",
                            ),
                        ),
                    ),
                    ModelResponse(content="done"),
                ]
            )
            buffer = TurnFollowupBuffer()
            turn_id = buffer.start_turn()

            def handle_tool(arguments):
                self.assertTrue(
                    buffer.submit(
                        turn_id,
                        "Use the compatibility path.",
                        source="test",
                    )
                )
                return NativeToolResult(content=str(arguments.get("text", "")))

            registry = NativeToolRegistry(
                (
                    NativeTool(
                        name="echo",
                        description="Echo arguments.",
                        input_schema={"type": "object"},
                        handler=handle_tool,
                    ),
                )
            )

            runtime.run_user_turn(
                provider=provider,
                messages=(Message(role=MessageRole.USER, content="Use echo."),),
                model="stub-model",
                native_tools=registry,
                tool_schemas=registry.schemas(),
                followup_buffer=buffer,
                followup_turn_id=turn_id,
            )

            self.assertEqual(len(provider.requests), 2)
            first_system = provider.requests[0].conversation[0].message
            second_system = provider.requests[1].conversation[0].message
            self.assertIsNotNone(first_system)
            self.assertIsNotNone(second_system)
            self.assertEqual(first_system.content, second_system.content)
            second_messages = [
                item.message.content
                for item in provider.requests[1].conversation
                if item.message is not None
            ]
            self.assertEqual(
                second_messages[-1],
                "User follow-up while this turn was running:\n"
                "Use the compatibility path.",
            )

    def test_running_followup_can_inject_before_first_model_request(self) -> None:
        with _workspace() as workspace:
            profile = ProfileRef(name="default", uri="profiles/default")
            activation = start_runtime_activation(
                session_id="session_1",
                workspace=workspace,
                profile=profile,
            )
            runtime = start_session_runtime(activation)
            provider = StubProvider(ModelResponse(content="done"))
            buffer = TurnFollowupBuffer()
            turn_id = buffer.start_turn()
            self.assertTrue(
                buffer.submit(
                    turn_id,
                    "补充一下，优先保持兼容。",
                    source="test",
                )
            )
            history_items = []

            result = runtime.run_user_turn(
                provider=provider,
                messages=(Message(role=MessageRole.USER, content="Do it."),),
                model="stub-model",
                followup_buffer=buffer,
                followup_turn_id=turn_id,
                history_sink=history_items.append,
            )

            request_text = "\n".join(
                item.message.content
                for item in provider.requests[0].conversation
                if item.message is not None
            )
            self.assertIn("User follow-up while this turn was running", request_text)
            self.assertIn("优先保持兼容", request_text)
            self.assertIn("优先保持兼容", "\n".join(
                item.message.content
                for item in history_items
                if item.message is not None
            ))
            self.assertEqual(buffer.pending_count(), 0)
            self.assertEqual(result.runtime.last_user_turn_result.final_step().response.content, "done")

    def test_running_followup_ignores_stale_turn_id(self) -> None:
        with _workspace() as workspace:
            profile = ProfileRef(name="default", uri="profiles/default")
            activation = start_runtime_activation(
                session_id="session_1",
                workspace=workspace,
                profile=profile,
            )
            runtime = start_session_runtime(activation)
            provider = StubProvider(ModelResponse(content="done"))
            buffer = TurnFollowupBuffer()
            active_turn_id = buffer.start_turn()
            self.assertFalse(
                buffer.submit(
                    "stale-turn",
                    "这条不该进入当前请求。",
                    source="test",
                )
            )

            result = runtime.run_user_turn(
                provider=provider,
                messages=(Message(role=MessageRole.USER, content="Do it."),),
                model="stub-model",
                followup_buffer=buffer,
                followup_turn_id=active_turn_id,
            )

            request_text = "\n".join(
                item.message.content
                for item in provider.requests[0].conversation
                if item.message is not None
            )
            self.assertNotIn("这条不该进入当前请求", request_text)
            self.assertEqual(buffer.pending_count(), 0)
            self.assertEqual(result.runtime.last_user_turn_result.final_step().response.content, "done")

    def test_running_followup_finish_returns_unconsumed_items(self) -> None:
        buffer = TurnFollowupBuffer()
        turn_id = buffer.start_turn()

        self.assertTrue(buffer.submit(turn_id, "no next model step", source="test"))
        remaining = buffer.finish_turn(turn_id)

        self.assertEqual(tuple(item.text for item in remaining), ("no next model step",))
        self.assertEqual(buffer.pending_count(), 0)

    def test_running_followup_finish_requires_active_turn_id(self) -> None:
        buffer = TurnFollowupBuffer()
        turn_id = buffer.start_turn()

        self.assertTrue(buffer.submit(turn_id, "keep pending", source="test"))
        remaining = buffer.finish_turn(None)

        self.assertEqual(remaining, ())
        self.assertEqual(buffer.pending_count(), 1)
        self.assertEqual(buffer.finish_turn(turn_id)[0].text, "keep pending")


class _workspace:
    def __enter__(self) -> Path:
        self._temp = tempfile.TemporaryDirectory()
        root = Path(self._temp.name)
        profile_dir = root / "profiles" / "default"
        profile_dir.mkdir(parents=True)
        (root / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
        (profile_dir / "identity.md").write_text("Identity.", encoding="utf-8")
        (profile_dir / "soul.md").write_text("Style.", encoding="utf-8")
        return root

    def __exit__(self, exc_type, exc, tb) -> None:
        self._temp.cleanup()


class _TraceSink:
    def __init__(self) -> None:
        self.events = []

    def write(self, event) -> None:
        self.events.append(event)


if __name__ == "__main__":
    unittest.main()
