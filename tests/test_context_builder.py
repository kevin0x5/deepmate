from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deepmate.capabilities import CapabilitySurface, from_native_tool_schemas
from deepmate.context import (
    build_profile_context_snapshot,
    build_system_context_from_snapshot,
    detect_behavior_context_changes,
)
from deepmate.domain import CapabilityKind, CapabilityRef, ProfileRef
from deepmate.foundation import estimate_text_tokens
from deepmate.skills import SkillDocument
from deepmate.tools import NativeTool, NativeToolResult


class ContextBuilderTests(unittest.TestCase):
    def test_two_layer_profile_reads_stable_global_prefix_before_project_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            global_profile = root / "home" / "profiles" / "default"
            workspace_a = root / "workspace-a"
            workspace_b = root / "workspace-b"
            for workspace in (workspace_a, workspace_b):
                (workspace / "profiles" / "default").mkdir(parents=True)
            global_profile.mkdir(parents=True)
            (global_profile / "identity.md").write_text("Global identity.", encoding="utf-8")
            (global_profile / "soul.md").write_text("Global soul.", encoding="utf-8")
            (global_profile / "user.md").write_text("- Global user.", encoding="utf-8")
            (global_profile / "memory.md").write_text("- Global memory.", encoding="utf-8")
            (workspace_a / "AGENTS.md").write_text("Workspace A rules.", encoding="utf-8")
            (workspace_b / "AGENTS.md").write_text("Workspace B rules.", encoding="utf-8")
            (workspace_a / "profiles" / "default" / "memory.md").write_text(
                "- Project A memory.",
                encoding="utf-8",
            )
            (workspace_b / "profiles" / "default" / "memory.md").write_text(
                "- Project B memory.",
                encoding="utf-8",
            )
            profile = ProfileRef(
                name="default",
                uri="profiles/default",
                global_uri=str(global_profile),
                project_uri="profiles/default",
            )

            snapshot_a = build_profile_context_snapshot(workspace=workspace_a, profile=profile)
            snapshot_b = build_profile_context_snapshot(workspace=workspace_b, profile=profile)
            content_a = build_system_context_from_snapshot(snapshot_a).message.content
            content_b = build_system_context_from_snapshot(snapshot_b).message.content

            self.assertLess(content_a.index("<identity>"), content_a.index("<workspace_rules>"))
            self.assertLess(content_a.index("<workspace_rules>"), content_a.index("<project_memory>"))
            self.assertIn("- Project A memory.", content_a)
            self.assertIn("- Project B memory.", content_b)
            stable_prefix_a = content_a[: content_a.index("<workspace_rules>")]
            stable_prefix_b = content_b[: content_b.index("<workspace_rules>")]
            self.assertEqual(stable_prefix_a, stable_prefix_b)
            refs = {ref.name: ref for ref in snapshot_a.file_refs}
            self.assertEqual(refs["project_memory"].status, "loaded")

    def test_explicit_missing_global_profile_does_not_fall_back_to_project_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            project_profile = workspace / "profiles" / "default"
            project_profile.mkdir(parents=True)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            (project_profile / "identity.md").write_text(
                "Project identity should not load.",
                encoding="utf-8",
            )
            (project_profile / "soul.md").write_text(
                "Project soul should not load.",
                encoding="utf-8",
            )
            (project_profile / "user.md").write_text(
                "Project user should not load.",
                encoding="utf-8",
            )
            (project_profile / "memory.md").write_text(
                "- Project memory loads.",
                encoding="utf-8",
            )
            profile = ProfileRef(
                name="default",
                uri="profiles/default",
                global_uri=str(workspace / "missing-home" / "profiles" / "default"),
                project_uri="profiles/default",
            )

            snapshot = build_profile_context_snapshot(workspace=workspace, profile=profile)
            content = build_system_context_from_snapshot(snapshot).message.content
            refs = {ref.name: ref for ref in snapshot.file_refs}

            self.assertNotIn("Project identity should not load.", content)
            self.assertNotIn("Project soul should not load.", content)
            self.assertNotIn("Project user should not load.", content)
            self.assertIn("- Project memory loads.", content)
            self.assertEqual(refs["identity"].status, "missing_optional")
            self.assertEqual(refs["project_memory"].status, "loaded")

    def test_empty_workspace_snapshot_is_ready_with_default_section(self) -> None:
        # A workspace with no AGENTS.md/CLAUDE.md and an empty profile must still
        # produce a ready snapshot so a session can start instead of crashing.
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            profile = ProfileRef(name="default", uri="profiles/default")

            snapshot = build_profile_context_snapshot(
                workspace=workspace,
                profile=profile,
            )

            self.assertTrue(snapshot.sections)
            self.assertTrue(snapshot.is_ready())
            result = build_system_context_from_snapshot(snapshot)
            self.assertIn(str(workspace), result.message.content)

    def test_profile_snapshot_records_file_refs_and_hot_profile_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            profile = ProfileRef(name="default", uri="profiles/default")
            profile_dir = workspace / "profiles" / "default"
            profile_dir.mkdir(parents=True)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            (profile_dir / "identity.md").write_text("Identity.", encoding="utf-8")
            (profile_dir / "soul.md").write_text("Soul.", encoding="utf-8")
            (profile_dir / "user.md").write_text(
                "- 用户偏好中文。\n",
                encoding="utf-8",
            )
            (profile_dir / "memory.md").write_text("", encoding="utf-8")

            snapshot = build_profile_context_snapshot(
                workspace=workspace,
                profile=profile,
                hot_profile_token_budget=2,
                hot_profile_warn_tokens=1,
                pending_refresh_reason="memory_curator_applied",
            )

            refs = {ref.name: ref for ref in snapshot.file_refs}
            self.assertEqual(refs["user"].status, "loaded")
            self.assertEqual(
                refs["user"].estimated_tokens,
                estimate_text_tokens("- 用户偏好中文。"),
            )
            self.assertEqual(refs["memory"].status, "empty_optional")
            self.assertTrue(refs["user"].sha256)
            self.assertIn("user", snapshot.loaded_section_names())
            self.assertNotIn("memory", snapshot.loaded_section_names())
            self.assertEqual(snapshot.hot_profile_token_budget, 2)
            self.assertEqual(snapshot.pending_refresh_reason, "memory_curator_applied")
            self.assertTrue(
                any(
                    warning.code == "hot_profile_budget_exceeded"
                    for warning in snapshot.warnings
                )
            )
            self.assertIn(
                "pending_refresh_reason=memory_curator_applied",
                snapshot.trace_refs(),
            )

    def test_behavior_hints_are_injected_after_profile_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            profile = ProfileRef(name="default", uri="profiles/default")
            profile_dir = workspace / "profiles" / "default"
            profile_dir.mkdir(parents=True)
            (workspace / ".deepmate").mkdir()
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            (profile_dir / "identity.md").write_text("Identity.", encoding="utf-8")
            (profile_dir / "soul.md").write_text("Soul.", encoding="utf-8")
            (profile_dir / "user.md").write_text("", encoding="utf-8")
            (profile_dir / "memory.md").write_text("", encoding="utf-8")
            (profile_dir / "behavior.md").write_text(
                "# Behavior Hints\n\n- Global hint.\n",
                encoding="utf-8",
            )
            (workspace / ".deepmate" / "behavior.md").write_text(
                "# Behavior Hints\n\n- Workspace hint.\n",
                encoding="utf-8",
            )

            snapshot = build_profile_context_snapshot(
                workspace=workspace,
                profile=profile,
                hot_profile_token_budget=1,
                hot_profile_warn_tokens=1,
            )
            result = build_system_context_from_snapshot(snapshot)
            content = result.message.content

            self.assertIn("<collaboration_hints>", content)
            self.assertLess(content.index("- Workspace hint."), content.index("- Global hint."))
            self.assertTrue(content.rstrip().endswith("</collaboration_hints>"))
            self.assertFalse(
                any(
                    warning.code == "hot_profile_budget_exceeded"
                    for warning in snapshot.warnings
                )
            )
            refs = {ref.name: ref for ref in snapshot.file_refs}
            self.assertEqual(refs["workspace_behavior"].status, "loaded")
            self.assertEqual(refs["profile_behavior"].status, "loaded")

    def test_missing_behavior_files_do_not_inject_prompt_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            profile = ProfileRef(name="default", uri="profiles/default")
            profile_dir = workspace / "profiles" / "default"
            profile_dir.mkdir(parents=True)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            (profile_dir / "identity.md").write_text("Identity.", encoding="utf-8")
            (profile_dir / "soul.md").write_text("Soul.", encoding="utf-8")
            (profile_dir / "user.md").write_text("", encoding="utf-8")
            (profile_dir / "memory.md").write_text("", encoding="utf-8")

            snapshot = build_profile_context_snapshot(workspace=workspace, profile=profile)
            result = build_system_context_from_snapshot(snapshot)

            self.assertNotIn("<collaboration_hints>", result.message.content)
            refs = {ref.name: ref for ref in snapshot.file_refs}
            self.assertEqual(refs["workspace_behavior"].status, "missing_optional")
            self.assertEqual(refs["profile_behavior"].status, "missing_optional")

    def test_skill_surface_does_not_inject_full_skill_body_until_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            profile = ProfileRef(name="default", uri="profiles/default")
            profile_dir = workspace / "profiles" / "default"
            profile_dir.mkdir(parents=True)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            (profile_dir / "identity.md").write_text("Identity.", encoding="utf-8")
            (profile_dir / "soul.md").write_text("Soul.", encoding="utf-8")
            skill_path = workspace / "skills" / "prd" / "SKILL.md"
            skill_path.parent.mkdir(parents=True)
            surface = CapabilitySurface(
                refs=(
                    CapabilityRef(
                        kind=CapabilityKind.SKILL,
                        name="prd",
                        description="Draft a focused PRD.",
                    ),
                )
            )
            skill = SkillDocument(
                name="prd",
                description="Draft a focused PRD.",
                body="FULL PRD BODY SHOULD LOAD ONLY WHEN SELECTED.",
                path=skill_path,
            )
            snapshot = build_profile_context_snapshot(workspace=workspace, profile=profile)

            surface_only = build_system_context_from_snapshot(
                snapshot,
                capability_surface=surface,
            ).message.content
            selected = build_system_context_from_snapshot(
                snapshot,
                capability_surface=surface,
                selected_skill_documents=(skill,),
            ).message.content

        self.assertIn("prd: Draft a focused PRD.", surface_only)
        self.assertNotIn("FULL PRD BODY", surface_only)
        self.assertIn("Deepmate supports community-style SKILL.md bundles", surface_only)
        self.assertIn("call load_skill", surface_only)
        self.assertIn("FULL PRD BODY SHOULD LOAD ONLY WHEN SELECTED.", selected)

    def test_browser_capability_guidance_is_injected_only_when_browser_tools_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            profile = ProfileRef(name="default", uri="profiles/default")
            profile_dir = workspace / "profiles" / "default"
            profile_dir.mkdir(parents=True)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            (profile_dir / "identity.md").write_text("Identity.", encoding="utf-8")
            (profile_dir / "soul.md").write_text("Soul.", encoding="utf-8")
            (profile_dir / "user.md").write_text("", encoding="utf-8")
            (profile_dir / "memory.md").write_text("", encoding="utf-8")
            snapshot = build_profile_context_snapshot(workspace=workspace, profile=profile)
            browser_surface = from_native_tool_schemas(
                (
                    NativeTool(
                        name="browser_open",
                        description="Open a URL.",
                        input_schema={"type": "object"},
                        handler=lambda _arguments: NativeToolResult(),
                    ).schema(),
                )
            )
            filesystem_surface = from_native_tool_schemas(
                (
                    NativeTool(
                        name="read_text_file",
                        description="Read a file.",
                        input_schema={"type": "object"},
                        handler=lambda _arguments: NativeToolResult(),
                    ).schema(),
                )
            )

            browser_context = build_system_context_from_snapshot(
                snapshot,
                capability_surface=browser_surface,
            ).message.content
            file_context = build_system_context_from_snapshot(
                snapshot,
                capability_surface=filesystem_surface,
            ).message.content

        self.assertIn("<capability_guidance>", browser_context)
        self.assertIn("Use the built-in browser for dynamic web pages", browser_context)
        self.assertIn("call it first to load concrete browser schemas", browser_context)
        self.assertIn("Prefer cheaper static retrieval/search", browser_context)
        self.assertIn("browser_snapshot before click/fill", browser_context)
        self.assertIn("Do not use browser tools for broad crawling", browser_context)
        self.assertNotIn("<capability_guidance>", file_context)

    def test_lsp_capability_guidance_is_injected_only_when_lsp_tools_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            profile = ProfileRef(name="default", uri="profiles/default")
            profile_dir = workspace / "profiles" / "default"
            profile_dir.mkdir(parents=True)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            (profile_dir / "identity.md").write_text("Identity.", encoding="utf-8")
            (profile_dir / "soul.md").write_text("Soul.", encoding="utf-8")
            (profile_dir / "user.md").write_text("", encoding="utf-8")
            (profile_dir / "memory.md").write_text("", encoding="utf-8")
            snapshot = build_profile_context_snapshot(workspace=workspace, profile=profile)
            lsp_surface = from_native_tool_schemas(
                (
                    NativeTool(
                        name="lsp_definition",
                        description="Find symbol definitions.",
                        input_schema={"type": "object"},
                        handler=lambda _arguments: NativeToolResult(),
                    ).schema(),
                )
            )
            filesystem_surface = from_native_tool_schemas(
                (
                    NativeTool(
                        name="read_text_file",
                        description="Read a workspace file.",
                        input_schema={"type": "object"},
                        handler=lambda _arguments: NativeToolResult(),
                    ).schema(),
                )
            )

            lsp_context = build_system_context_from_snapshot(
                snapshot,
                capability_surface=lsp_surface,
            ).message.content
            file_context = build_system_context_from_snapshot(
                snapshot,
                capability_surface=filesystem_surface,
            ).message.content

        self.assertIn("<capability_guidance>", lsp_context)
        self.assertIn("prefer lsp_definition, lsp_references, or lsp_hover", lsp_context)
        self.assertIn("use grep/search as fallback", lsp_context)
        self.assertNotIn("<capability_guidance>", file_context)

    def test_detect_behavior_context_changes_from_snapshot_refs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            profile = ProfileRef(name="default", uri="profiles/default")
            profile_dir = workspace / "profiles" / "default"
            profile_dir.mkdir(parents=True)
            behavior_dir = workspace / ".deepmate"
            behavior_dir.mkdir()
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            (profile_dir / "identity.md").write_text("Identity.", encoding="utf-8")
            (profile_dir / "soul.md").write_text("Soul.", encoding="utf-8")
            (profile_dir / "user.md").write_text("", encoding="utf-8")
            (profile_dir / "memory.md").write_text("", encoding="utf-8")
            behavior_path = behavior_dir / "behavior.md"
            behavior_path.write_text(
                "# Behavior Hints\n\n- Original hint.\n",
                encoding="utf-8",
            )
            snapshot = build_profile_context_snapshot(workspace=workspace, profile=profile)

            self.assertEqual(detect_behavior_context_changes(snapshot), ())

            behavior_path.write_text(
                "# Behavior Hints\n\n- Updated hint.\n",
                encoding="utf-8",
            )

            changes = detect_behavior_context_changes(snapshot)

            self.assertEqual(len(changes), 1)
            self.assertEqual(changes[0].name, "workspace_behavior")
            self.assertEqual(changes[0].old_status, "loaded")
            self.assertEqual(changes[0].new_status, "loaded")
            self.assertNotEqual(changes[0].old_sha256, changes[0].new_sha256)


if __name__ == "__main__":
    unittest.main()
