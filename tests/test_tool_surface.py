from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deepmate.capabilities import (
    CapabilitySurface,
    CapabilityTemperature,
    combine_surfaces,
    from_mcp_tool_refs,
    from_native_tool_schemas,
    from_skill_cards,
)
from deepmate.capabilities.state import CapabilityState
from deepmate.domain import CapabilityKind, Message, MessageRole, ProfileRef
from deepmate.domain import CapabilityRef
from deepmate.mcp import McpToolExecutionResult, McpToolRef
from deepmate.providers import ModelResponse, ModelToolRequest, ModelToolResult
from deepmate.runtime import run_user_turn
from deepmate.skills import SkillCard
from deepmate.subagents import subagent_tool_schema
from deepmate.tools import NativeTool, NativeToolRegistry, NativeToolResult
from deepmate.trace import TraceRecorder


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

    def execute(self, request: ModelToolRequest) -> McpToolExecutionResult:
        self.calls.append(request)
        return McpToolExecutionResult(
            request=request,
            model_result=ModelToolResult(
                name=request.name,
                request_id=request.id,
                content="mcp file content",
                refs=("mcp_tool=filesystem.read_text_file",),
            ),
        )


class _TraceSink:
    def __init__(self) -> None:
        self.events = []

    def write(self, event) -> None:
        self.events.append(event)


class ToolSurfaceTests(unittest.TestCase):
    def test_surface_keeps_native_and_mcp_refs_provider_neutral(self) -> None:
        native = NativeTool(
            name="read_text_file",
            description="Read a text file.",
            input_schema={"type": "object"},
            handler=lambda _arguments: NativeToolResult(content=""),
        )
        mcp_tool = McpToolRef(
            server_name="filesystem",
            name="stat_file",
            annotations={"readOnlyHint": True},
        )

        surface = combine_surfaces(
            (
                from_native_tool_schemas((native.schema(),)),
                from_mcp_tool_refs((mcp_tool,)),
            )
        )

        refs = surface.list_refs()
        self.assertEqual(
            [(ref.kind, ref.name) for ref in refs],
            [
                (CapabilityKind.NATIVE_TOOL, "read_text_file"),
                (CapabilityKind.MCP_TOOL, "filesystem.stat_file"),
            ],
        )
        self.assertEqual(refs[1].description, "MCP tool filesystem.stat_file.")
        self.assertEqual(
            surface.surface_keys(),
            (
                "mcp_tool:filesystem.stat_file:MCP tool filesystem.stat_file.",
                "native_tool:read_text_file:Read a text file.",
            ),
        )

    def test_skill_temperature_filters_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hot_path = root / "skills" / "hot" / "SKILL.md"
            warm_path = root / "skills" / "warm" / "SKILL.md"
            cold_path = root / "skills" / "cold" / "SKILL.md"
            hot = SkillCard("hot", "Hot skill.", hot_path)
            warm = SkillCard("warm", "Warm skill.", warm_path)
            cold = SkillCard("cold", "Cold skill.", cold_path)
            states = {
                "warm": CapabilityState(
                    capability_id="skill:workspace:warm",
                    kind=CapabilityKind.SKILL,
                    name="warm",
                    path_or_ref=str(warm_path),
                    temperature=CapabilityTemperature.WARM,
                    created_at="2026-06-01T00:00:00+00:00",
                    updated_at="2026-06-01T00:00:00+00:00",
                ),
                "cold": CapabilityState(
                    capability_id="skill:workspace:cold",
                    kind=CapabilityKind.SKILL,
                    name="cold",
                    path_or_ref=str(cold_path),
                    temperature=CapabilityTemperature.COLD,
                    created_at="2026-06-01T00:00:00+00:00",
                    updated_at="2026-06-01T00:00:00+00:00",
                ),
            }

            surface = from_skill_cards((hot, warm, cold), states)

        refs = surface.list_refs()
        self.assertEqual([ref.name for ref in refs], ["hot", "warm"])
        self.assertEqual(refs[0].description, "Hot skill.")
        self.assertEqual(refs[1].description, "")

    def test_builtin_skill_ignores_stale_workspace_temperature_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            card = SkillCard(
                "prd",
                "Built-in PRD work kit.",
                root / "builtin_skills" / "prd" / "SKILL.md",
                metadata={"deepmate-builtin": True},
            )
            stale_state = CapabilityState(
                capability_id="skill:workspace:prd",
                kind=CapabilityKind.SKILL,
                name="prd",
                path_or_ref="skills/prd/SKILL.md",
                temperature=CapabilityTemperature.COLD,
                created_at="2026-06-01T00:00:00+00:00",
                updated_at="2026-06-01T00:00:00+00:00",
            )

            surface = from_skill_cards((card,), {"prd": stale_state})

        self.assertEqual(
            [(ref.name, ref.description) for ref in surface.list_refs()],
            [("prd", "Built-in PRD work kit.")],
        )

    def test_mcp_tool_surface_allows_name_only_refs(self) -> None:
        surface = CapabilitySurface(
            refs=(
                CapabilityRef(
                    kind=CapabilityKind.MCP_TOOL,
                    name="github.search_issues",
                    description="",
                ),
            )
        )

        refs = surface.list_refs()
        self.assertEqual(refs[0].name, "github.search_issues")
        self.assertEqual(refs[0].description, "")

    def test_runtime_dispatches_mcp_tool_without_putting_subagent_in_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "AGENTS.md").write_text(
                "# Workspace Rules\n\n- Use available tools when needed.\n",
                encoding="utf-8",
            )
            mcp_tool = McpToolRef(
                server_name="filesystem",
                name="read_text_file",
                description="Read a file through MCP.",
                annotations={"readOnlyHint": True},
            )
            surface = from_mcp_tool_refs((mcp_tool,))
            mcp_executor = _FakeMcpExecutor()
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
            trace_sink = _TraceSink()

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="Read README."),),
                model="stub-model",
                capability_surface=surface,
                mcp_tools=mcp_executor,  # type: ignore[arg-type]
                tool_schemas=(mcp_tool.schema(), subagent_tool_schema()),
                max_steps=2,
                trace_recorder=TraceRecorder(trace_sink),
            )

        self.assertFalse(result.has_errors())
        self.assertEqual(
            [call.name for call in mcp_executor.calls],
            ["filesystem.read_text_file"],
        )
        first_request = provider.requests[0]
        self.assertEqual(
            tuple(schema["name"] for schema in first_request.tool_schemas),
            ("filesystem.read_text_file", "run_subagent"),
        )
        system_prompt = first_request.conversation[0].message.content
        self.assertIn("filesystem.read_text_file", system_prompt)
        self.assertNotIn("run_subagent", system_prompt)
        self.assertEqual(result.final_step().response.content, "done")
        completed = [
            event for event in trace_sink.events if event.kind == "mcp_tool_completed"
        ][0]
        self.assertIn("tool_source=mcp", completed.refs)
        self.assertIn("mcp_tool=filesystem.read_text_file", completed.refs)


if __name__ == "__main__":
    unittest.main()
