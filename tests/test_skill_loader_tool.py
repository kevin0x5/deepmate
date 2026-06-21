from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deepmate.capabilities import (
    CapabilityStateStore,
    CapabilityTemperature,
    from_skill_cards,
)
from deepmate.channels.cli import _attach_skill_loader_tools
from deepmate.channels.skill_view import workspace_skill_catalog
from deepmate.domain import Message, MessageRole, ProfileRef
from deepmate.providers import ModelResponse, ModelToolRequest
from deepmate.runtime import (
    ToolAccessMode,
    ToolAccessPolicy,
    execute_native_tool_request,
    run_user_turn,
)
from deepmate.skills import SkillCatalog
from deepmate.tools import (
    INSTALL_SKILL_TOOL_NAME,
    LOAD_SKILL_TOOL_NAME,
    NativeTool,
    NativeToolRegistry,
    NativeToolResult,
    skill_installer_tools,
    skill_loader_tools,
)


class _StubProvider:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("stub provider received too many requests")
        return self.responses.pop(0)


class SkillLoaderToolTests(unittest.TestCase):
    def test_cli_registry_attaches_loader_without_filesystem_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace_with_skill(Path(tmp), "writer")
            catalog = SkillCatalog.from_paths((workspace / "skills",))
            store = CapabilityStateStore.in_data_dir(workspace / "var", "default")
            store.sync_workspace_skills(catalog.list_cards(), workspace)

            registry = _attach_skill_loader_tools(
                native_tools=None,
                skill_catalog=catalog,
                workspace=workspace,
                capability_state_store=store,
            )

            self.assertIsNotNone(registry)
            self.assertEqual(
                tuple(schema["name"] for schema in registry.schemas()),
                (LOAD_SKILL_TOOL_NAME,),
            )

    def test_cli_registry_preserves_existing_native_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace_with_skill(Path(tmp), "writer")
            catalog = SkillCatalog.from_paths((workspace / "skills",))
            base_registry = NativeToolRegistry(
                (
                    NativeTool(
                        name="read_text_file",
                        description="Read a file.",
                        input_schema={"type": "object"},
                        handler=lambda _arguments: NativeToolResult(content="ok"),
                    ),
                )
            )

            registry = _attach_skill_loader_tools(
                native_tools=base_registry,
                skill_catalog=catalog,
                workspace=workspace,
            )

            self.assertEqual(
                tuple(schema["name"] for schema in registry.schemas()),
                ("read_text_file", LOAD_SKILL_TOOL_NAME),
            )

    def test_load_skill_returns_full_instructions_and_records_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace_with_skill(Path(tmp), "writer")
            catalog = SkillCatalog.from_paths((workspace / "skills",))
            store = CapabilityStateStore.in_data_dir(workspace / "var", "default")
            store.sync_workspace_skills(catalog.list_cards(), workspace)
            registry = NativeToolRegistry(
                skill_loader_tools(catalog, workspace, store),
            )

            result = execute_native_tool_request(
                ModelToolRequest(
                    name=LOAD_SKILL_TOOL_NAME,
                    arguments={"name": "writer"},
                    id="call_1",
                ),
                registry,
            )

            self.assertIsNone(result.error)
            self.assertIsNotNone(result.model_result)
            self.assertIn("<name>writer</name>", result.model_result.content)
            self.assertIn("Follow the writer checklist.", result.model_result.content)
            state = store.skill_states_by_name()["writer"]
            self.assertEqual(state.temperature, CapabilityTemperature.HOT)
            self.assertEqual(state.invocation_count, 1)

    def test_dynamic_loader_sees_skill_installed_after_tool_registration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            store = CapabilityStateStore.in_data_dir(workspace / "var", "default")

            registry = NativeToolRegistry(
                skill_loader_tools(
                    None,
                    workspace,
                    store,
                    catalog_provider=lambda: workspace_skill_catalog(workspace)[0],
                ),
            )
            _add_workspace_skill(workspace, "writer")

            result = execute_native_tool_request(
                ModelToolRequest(
                    name=LOAD_SKILL_TOOL_NAME,
                    arguments={"name": "writer"},
                    id="call_1",
                ),
                registry,
            )

            self.assertIsNone(result.error)
            self.assertIsNotNone(result.model_result)
            self.assertIn("<name>writer</name>", result.model_result.content)
            self.assertIn("Follow the writer checklist.", result.model_result.content)
            state = store.skill_states_by_name()["writer"]
            self.assertEqual(state.temperature, CapabilityTemperature.HOT)
            self.assertEqual(state.invocation_count, 1)

    def test_agent_loop_can_install_then_load_skill_in_same_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            source = root / "source-skill"
            data_dir = workspace / "var"
            workspace.mkdir()
            _write_skill(source, "writer")
            store = CapabilityStateStore.in_data_dir(data_dir, "default")
            registry = NativeToolRegistry(
                (
                    *skill_installer_tools(workspace, data_dir, store),
                    *skill_loader_tools(
                        None,
                        workspace,
                        store,
                        catalog_provider=lambda: workspace_skill_catalog(
                            workspace,
                            data_dir=data_dir,
                        )[0],
                    ),
                )
            )
            provider = _StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name=INSTALL_SKILL_TOOL_NAME,
                                arguments={"source": str(source)},
                                id="call_install",
                            ),
                        )
                    ),
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name=LOAD_SKILL_TOOL_NAME,
                                arguments={"name": "writer"},
                                id="call_load",
                            ),
                        )
                    ),
                    ModelResponse(content="used installed writer skill"),
                ]
            )

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="Install and use writer."),),
                model="stub-model",
                native_tools=registry,
                tool_access_policy=ToolAccessPolicy(ToolAccessMode.WORKSPACE_WRITE),
                tool_schemas=registry.schemas(include_hidden=True),
                max_steps=3,
            )

            self.assertFalse(result.has_errors())
            self.assertEqual(result.final_step().response.content, "used installed writer skill")
            self.assertTrue((workspace / "skills" / "writer" / "SKILL.md").is_file())
            loaded = result.steps[1].tool_results[0]
            self.assertEqual(loaded.name, LOAD_SKILL_TOOL_NAME)
            self.assertIn("Follow the writer checklist.", loaded.content)

    def test_load_skill_expands_skill_dir_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace_with_skill(Path(tmp), "writer")
            skill_path = workspace / "skills" / "writer" / "SKILL.md"
            skill_path.write_text(
                "\n".join(
                    (
                        "---",
                        "name: writer",
                        "description: Draft concise product updates.",
                        "---",
                        "Read ${SKILL_DIR}/references/style.md when needed.",
                    )
                ),
                encoding="utf-8",
            )
            catalog = SkillCatalog.from_paths((workspace / "skills",))
            registry = NativeToolRegistry(skill_loader_tools(catalog, workspace))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name=LOAD_SKILL_TOOL_NAME,
                    arguments={"name": "writer"},
                    id="call_1",
                ),
                registry,
            )

        self.assertIsNone(result.error)
        self.assertIsNotNone(result.model_result)
        self.assertIn(str(skill_path.parent), result.model_result.content)
        self.assertNotIn("${SKILL_DIR}", result.model_result.content)

    def test_load_skill_rejects_cold_skill_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace_with_skill(Path(tmp), "writer")
            catalog = SkillCatalog.from_paths((workspace / "skills",))
            store = CapabilityStateStore.in_data_dir(workspace / "var", "default")
            store.sync_workspace_skills(catalog.list_cards(), workspace)
            store.set_skill_state("writer", "cool")
            store.set_skill_state("writer", "cool")
            registry = NativeToolRegistry(
                skill_loader_tools(catalog, workspace, store),
            )

            result = execute_native_tool_request(
                ModelToolRequest(
                    name=LOAD_SKILL_TOOL_NAME,
                    arguments={"name": "writer"},
                    id="call_1",
                ),
                registry,
            )

            self.assertIsNotNone(result.error)
            self.assertEqual(result.error.code, "native_tool_failed")
            self.assertIn("not exposed by default", result.error.message)
            state = store.skill_states_by_name()["writer"]
            self.assertEqual(state.temperature, CapabilityTemperature.COLD)
            self.assertEqual(state.invocation_count, 0)

    def test_load_builtin_work_kit_does_not_create_workspace_skill_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            catalog, warnings = workspace_skill_catalog(workspace)
            store = CapabilityStateStore.in_data_dir(workspace / "var", "default")
            registry = NativeToolRegistry(skill_loader_tools(catalog, workspace, store))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name=LOAD_SKILL_TOOL_NAME,
                    arguments={"name": "research-brief"},
                    id="call_1",
                ),
                registry,
            )

            self.assertEqual(warnings, ())
            self.assertIsNone(result.error)
            self.assertIsNotNone(result.model_result)
            self.assertIn("<name>research-brief</name>", result.model_result.content)
            self.assertEqual(store.skill_states_by_name(), {})

    def test_agent_loop_replays_loaded_skill_before_followup_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace_with_skill(Path(tmp), "writer")
            catalog = SkillCatalog.from_paths((workspace / "skills",))
            store = CapabilityStateStore.in_data_dir(workspace / "var", "default")
            store.sync_workspace_skills(catalog.list_cards(), workspace)
            registry = NativeToolRegistry(
                skill_loader_tools(catalog, workspace, store),
            )
            provider = _StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name=LOAD_SKILL_TOOL_NAME,
                                arguments={"name": "writer"},
                                id="call_1",
                            ),
                        )
                    ),
                    ModelResponse(content="used writer skill"),
                ]
            )

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="Draft the update."),),
                model="stub-model",
                capability_surface=from_skill_cards(
                    catalog.list_cards(),
                    store.skill_states_by_name(),
                ),
                native_tools=registry,
                tool_schemas=registry.schemas(),
                max_steps=2,
            )

            self.assertFalse(result.has_errors())
            self.assertEqual(result.final_step().response.content, "used writer skill")
            self.assertEqual(
                tuple(schema["name"] for schema in provider.requests[0].tool_schemas),
                (LOAD_SKILL_TOOL_NAME,),
            )
            followup_exchange = provider.requests[1].conversation[-1].tool_exchange
            self.assertIsNotNone(followup_exchange)
            self.assertIn(
                "Follow the writer checklist.",
                followup_exchange.tool_results[0].content,
            )
            state = store.skill_states_by_name()["writer"]
            self.assertEqual(state.invocation_count, 1)


def _workspace_with_skill(root: Path, name: str) -> Path:
    workspace = root / "workspace"
    _add_workspace_skill(workspace, name)
    (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
    return workspace


def _add_workspace_skill(workspace: Path, name: str) -> Path:
    skill_dir = workspace / "skills" / name
    _write_skill(skill_dir, name)
    return skill_dir


def _write_skill(skill_dir: Path, name: str) -> Path:
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            (
                "---",
                f"name: {name}",
                "description: Draft concise product updates.",
                "---",
                "Follow the writer checklist.",
            )
        ),
        encoding="utf-8",
    )
    return skill_dir


if __name__ == "__main__":
    unittest.main()
