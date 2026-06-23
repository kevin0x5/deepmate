from __future__ import annotations

import os
import tempfile
import tarfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from deepmate.capabilities import (
    CapabilitySource,
    CapabilityStateStore,
    CapabilityTemperature,
)
from deepmate.channels.skill_view import discover_skill_cards, discover_workspace_skill_cards
from deepmate.providers import ModelToolRequest
from deepmate.runtime import (
    SessionApprovalCache,
    ToolAccessMode,
    ToolAccessPolicy,
    execute_native_tool_request,
)
from deepmate.runtime.sandbox import SandboxMode, SandboxRunResult
from deepmate.skills import (
    InstalledSkillManifestStore,
    install_skill_bundle,
    inspect_skill_source,
    install_skill_source,
    uninstall_skill,
    verify_skill_install,
)
from deepmate.skills.install import _download_to, _extract_archive, _fetch_text
from deepmate.tools import (
    INSTALL_SKILL_FROM_REQUEST_TOOL_NAME,
    INSTALL_SKILL_BUNDLE_TOOL_NAME,
    INSTALL_SKILL_TOOL_NAME,
    NativeToolRegistry,
    LOAD_SKILL_INSTALLER_TOOLS_NAME,
    PLAN_SKILL_SETUP_TOOL_NAME,
    RUN_SKILL_SETUP_TOOL_NAME,
    skill_installer_tools,
)


class SkillInstallTests(unittest.TestCase):
    def test_install_local_bundle_preserves_resources_manifest_and_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            data_dir = workspace / "var"
            source = root / "source" / "self-improving-agent"
            _write_skill_bundle(source)
            store = CapabilityStateStore.in_data_dir(data_dir, "default")

            result = install_skill_source(source, workspace, data_dir, store)

            target = workspace / "skills" / "self-improving-agent"
            self.assertEqual(result.skill.name, "self-improving-agent")
            self.assertTrue((target / "SKILL.md").is_file())
            self.assertTrue((target / "references" / "guide.md").is_file())
            self.assertTrue((target / "scripts" / "setup.py").is_file())
            self.assertTrue((target / "assets" / "icon.txt").is_file())
            self.assertTrue((target / "agents" / "openai.yaml").is_file())

            manifest = InstalledSkillManifestStore.in_data_dir(data_dir).get(
                "self-improving-agent"
            )
            self.assertIsNotNone(manifest)
            self.assertEqual(manifest.source_kind, "local_dir")
            self.assertEqual(manifest.setup_status, "pending")

            state = store.skill_states_by_name()["self-improving-agent"]
            self.assertEqual(state.source, CapabilitySource.IMPORTED)
            self.assertEqual(state.temperature, CapabilityTemperature.HOT)

            cards, warnings = discover_workspace_skill_cards(workspace)
            self.assertEqual(warnings, ())
            self.assertEqual(tuple(card.name for card in cards), ("self-improving-agent",))

            verified = verify_skill_install(
                "self-improving-agent",
                workspace,
                data_dir,
                store,
            )
            self.assertEqual(verified.status, "ok")
            self.assertEqual(verified.resources.references, 1)

    def test_install_rolls_back_copied_skill_when_manifest_write_fails(self) -> None:
        class FailingManifestStore:
            def get(self, _name):
                return None

            def upsert(self, _record):
                raise OSError("manifest write failed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            data_dir = workspace / "var"
            source = root / "source" / "writer"
            workspace.mkdir()
            _write_skill_bundle(source, name="writer")
            store = CapabilityStateStore.in_data_dir(data_dir, "default")

            with self.assertRaisesRegex(OSError, "manifest write failed"):
                install_skill_source(
                    source,
                    workspace,
                    data_dir,
                    store,
                    manifest_store=FailingManifestStore(),
                )

            self.assertFalse((workspace / "skills" / "writer").exists())

    def test_install_zip_bundle_and_reject_existing_destination_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            data_dir = workspace / "var"
            source = root / "source" / "reviewer"
            _write_skill_bundle(source, name="reviewer")
            archive = root / "reviewer.zip"
            _zip_dir(source, archive)
            store = CapabilityStateStore.in_data_dir(data_dir, "default")

            install_skill_source(archive, workspace, data_dir, store)

            with self.assertRaisesRegex(ValueError, "already exists"):
                install_skill_source(archive, workspace, data_dir, store)

    def test_install_local_bundle_rejects_symlinks_before_copying(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            data_dir = workspace / "var"
            source = root / "source" / "writer"
            _write_skill_bundle(source, name="writer")
            outside = root / "private-notes.txt"
            outside.write_text("secret", encoding="utf-8")
            try:
                (source / "references" / "private-notes.txt").symlink_to(outside)
            except (NotImplementedError, OSError):
                self.skipTest("symlink creation is not available on this filesystem")
            store = CapabilityStateStore.in_data_dir(data_dir, "default")

            with self.assertRaisesRegex(ValueError, "unsupported link"):
                install_skill_source(source, workspace, data_dir, store)

            self.assertFalse((workspace / "skills" / "writer").exists())

    def test_inspect_skill_install_command_requires_environment_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"

            result = inspect_skill_source(
                "skill install self-improving-agent",
                workspace,
            )

            self.assertFalse(result.is_installable())
            self.assertEqual(result.source_kind, "install_instruction")
            self.assertTrue(result.approval_required)
            self.assertIn("requires_approval", result.compatibility)

    def test_fetch_text_uses_github_token_for_github_api(self) -> None:
        captured = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self, _size=-1):
                return b"{}"

        def fake_urlopen(request, timeout):
            captured.append(request)
            return Response()

        with (
            patch.dict(os.environ, {"GITHUB_TOKEN": "gh_test"}),
            patch("urllib.request.urlopen", fake_urlopen),
            patch("deepmate.skills.install._validate_public_url"),
        ):
            self.assertEqual(_fetch_text("https://api.github.com/repos/a/b"), "{}")

        self.assertEqual(captured[0].get_header("Authorization"), "Bearer gh_test")

    def test_fetch_text_rejects_oversized_response(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self, size=-1):
                return b"x" * (size + 1)

        with (
            patch("urllib.request.urlopen", return_value=Response()),
            patch("deepmate.skills.install._validate_public_url"),
        ):
            with self.assertRaisesRegex(OSError, "remote skill response exceeds"):
                _fetch_text("https://api.github.com/repos/a/b")

    def test_download_to_rejects_oversized_response(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self, size=-1):
                return b"x" * (size + 1)

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("urllib.request.urlopen", return_value=Response()),
                patch("deepmate.skills.install._validate_public_url"),
            ):
                with self.assertRaisesRegex(OSError, "remote skill response exceeds"):
                    _download_to(Path(tmp), "https://example.test/skill.zip", "skill.zip")

    def test_download_to_rejects_local_network_urls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "local network URLs"):
                _download_to(Path(tmp), "http://localhost:8000/skill.zip", "skill.zip")

    def test_extract_archive_rejects_oversized_member(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "large.zip"
            with zipfile.ZipFile(archive, "w") as zip_file:
                zip_file.writestr("SKILL.md", "x" * 8)

            with patch("deepmate.skills.install.MAX_SKILL_ARCHIVE_MEMBER_BYTES", 4):
                with self.assertRaisesRegex(ValueError, "archive member is too large"):
                    _extract_archive(archive, root / "out")

    def test_install_remote_skill_page_embedded_skill_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            data_dir = workspace / "var"
            workspace.mkdir()
            store = CapabilityStateStore.in_data_dir(data_dir, "default")
            html = """
            <html><body>
              <nav>SKILL.md Skill Card Files Versions Download</nav>
              <main>
                <h1>Ontology</h1>
                <p>Creates and maintains ontology-driven project knowledge.</p>
                <h2>When to use</h2>
                <p>Use this skill when the user asks to model concepts.</p>
              </main>
            </body></html>
            """

            with patch("deepmate.skills.install._fetch_text", return_value=html):
                result = install_skill_source(
                    "https://clawhub.ai/oswalpalash/ontology",
                    workspace,
                    data_dir,
                    store,
                    target="user",
                )

            target = data_dir / "skills" / "library" / "ontology"

            self.assertEqual(result.skill.name, "ontology")
            self.assertEqual(result.manifest_record.source_kind, "remote_skill_page")
            self.assertTrue((target / "SKILL.md").is_file())
            self.assertIn(
                "remote skill page converted to SKILL.md",
                result.warnings,
            )

    def test_install_skill_bundle_installs_verifies_and_plans_setup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            data_dir = workspace / "var"
            source = root / "source" / "writer"
            _write_skill_bundle(source, name="writer")
            store = CapabilityStateStore.in_data_dir(data_dir, "default")

            result = install_skill_bundle(source, workspace, data_dir, store)
            manifest = InstalledSkillManifestStore.in_data_dir(data_dir).get("writer")

        self.assertEqual(result.install.skill.name, "writer")
        self.assertEqual(result.verify.status, "ok")
        self.assertEqual(result.setup_status, "approval_required")
        self.assertEqual(result.setup_command, "python3 scripts/setup.py")
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest.setup_status, "approval_required")
        self.assertEqual(manifest.setup_command, "python3 scripts/setup.py")

    def test_native_install_skill_bundle_tool_returns_end_to_end_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            data_dir = workspace / "var"
            source = root / "source" / "writer"
            _write_skill_bundle(source, name="writer")
            store = CapabilityStateStore.in_data_dir(data_dir, "default")
            registry = NativeToolRegistry(skill_installer_tools(workspace, data_dir, store))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name=INSTALL_SKILL_BUNDLE_TOOL_NAME,
                    id="call_1",
                    arguments={"source": str(source)},
                ),
                registry,
                ToolAccessPolicy(ToolAccessMode.WORKSPACE_WRITE),
            )

        self.assertIsNone(result.error)
        self.assertIsNotNone(result.model_result)
        self.assertIn("Skill installed: writer", result.model_result.content)
        self.assertIn("- verified: ok", result.model_result.content)
        self.assertIn("- setup_status: approval_required", result.model_result.content)
        self.assertIn("dependency setup is pending approval", result.model_result.content)
        self.assertIn(
            "Setup command detected. Call run_skill_setup(name='writer')",
            result.model_result.content,
        )
        self.assertIn("setup_command_present=true", result.model_result.refs)

    def test_skill_installer_exposes_loader_and_loads_concrete_schemas_on_demand(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            data_dir = workspace / "var"
            store = CapabilityStateStore.in_data_dir(data_dir, "default")
            registry = NativeToolRegistry(skill_installer_tools(workspace, data_dir, store))

            visible = {schema["name"] for schema in registry.schemas()}
            registered = {tool.name for tool in registry.list_tools()}
            result = execute_native_tool_request(
                ModelToolRequest(
                    name=LOAD_SKILL_INSTALLER_TOOLS_NAME,
                    id="call_1",
                    arguments={"reason": "install a community skill"},
                ),
                registry,
                ToolAccessPolicy(ToolAccessMode.READ_ONLY),
            )

        self.assertEqual(
            visible,
            {INSTALL_SKILL_FROM_REQUEST_TOOL_NAME, LOAD_SKILL_INSTALLER_TOOLS_NAME},
        )
        self.assertIn(INSTALL_SKILL_BUNDLE_TOOL_NAME, registered)
        self.assertIn(INSTALL_SKILL_FROM_REQUEST_TOOL_NAME, registered)
        self.assertIn(RUN_SKILL_SETUP_TOOL_NAME, registered)
        self.assertIsNone(result.error)
        self.assertIsNotNone(result.native_result)
        loaded = {schema["name"] for schema in result.native_result.schema_additions}
        self.assertIn(INSTALL_SKILL_BUNDLE_TOOL_NAME, loaded)
        self.assertIn(INSTALL_SKILL_TOOL_NAME, loaded)
        self.assertIn(PLAN_SKILL_SETUP_TOOL_NAME, loaded)
        self.assertIn(RUN_SKILL_SETUP_TOOL_NAME, loaded)

    def test_natural_language_skill_install_request_installs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            data_dir = workspace / "var"
            source = root / "source" / "writer"
            _write_skill_bundle(source, name="writer")
            store = CapabilityStateStore.in_data_dir(data_dir, "default")
            registry = NativeToolRegistry(skill_installer_tools(workspace, data_dir, store))

            request = ModelToolRequest(
                name=INSTALL_SKILL_FROM_REQUEST_TOOL_NAME,
                id="call_1",
                arguments={
                    "request": f"帮我安装这个 skill：{source}",
                },
            )
            denied = execute_native_tool_request(request, registry)
            allowed = execute_native_tool_request(
                request,
                registry,
                ToolAccessPolicy(ToolAccessMode.WORKSPACE_WRITE),
            )

        self.assertIsNotNone(denied.error)
        self.assertEqual(denied.error.code, "native_tool_denied")
        self.assertIsNone(allowed.error)
        self.assertIsNotNone(allowed.model_result)
        self.assertIn("Skill installed: writer", allowed.model_result.content)
        self.assertIn("- verified: ok", allowed.model_result.content)
        self.assertIn("- dependency setup: not run", allowed.model_result.content)
        self.assertIn("setup_command_present=true", allowed.model_result.refs)

    def test_natural_language_skill_install_request_accepts_github_shorthand(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            data_dir = workspace / "var"
            store = CapabilityStateStore.in_data_dir(data_dir, "default")
            registry = NativeToolRegistry(
                skill_installer_tools(workspace, data_dir, store, network_enabled=True)
            )

            captured: list[str] = []

            def fake_install(source, *args, **kwargs):
                captured.append(str(source))
                skill_root = Path(tmp) / "installed"
                _write_skill_bundle(skill_root, name="repo-review")
                return install_skill_bundle(skill_root, *args, **kwargs)

            with patch(
                "deepmate.tools.skill_installer.install_skill_bundle",
                side_effect=fake_install,
            ):
                result = execute_native_tool_request(
                    ModelToolRequest(
                        name=INSTALL_SKILL_FROM_REQUEST_TOOL_NAME,
                        id="call_1",
                        arguments={
                            "request": "安装 gh:example/repo-review 这个 skill",
                        },
                    ),
                    registry,
                    ToolAccessPolicy(ToolAccessMode.WORKSPACE_WRITE),
                )

        self.assertIsNone(result.error)
        self.assertEqual(captured, ["https://github.com/example/repo-review"])
        self.assertIn("Skill installed: repo-review", result.model_result.content)

    def test_natural_language_skill_install_ignores_bare_slash_phrase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            data_dir = workspace / "var"
            store = CapabilityStateStore.in_data_dir(data_dir, "default")
            registry = NativeToolRegistry(skill_installer_tools(workspace, data_dir, store))

            captured: list[str] = []

            def fake_install(source, *args, **kwargs):
                captured.append(str(source))
                raise AssertionError("bare a/b must not trigger an install")

            with patch(
                "deepmate.tools.skill_installer.install_skill_bundle",
                side_effect=fake_install,
            ):
                result = execute_native_tool_request(
                    ModelToolRequest(
                        name=INSTALL_SKILL_FROM_REQUEST_TOOL_NAME,
                        id="call_1",
                        arguments={"request": "帮我装一个 product/roadmap 技能"},
                    ),
                    registry,
                    ToolAccessPolicy(ToolAccessMode.WORKSPACE_WRITE),
                )

        # No concrete source -> the tool errors out and never downloads.
        self.assertEqual(captured, [])
        self.assertIsNotNone(result.error)

    def test_remote_skill_install_blocked_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            data_dir = workspace / "var"
            store = CapabilityStateStore.in_data_dir(data_dir, "default")
            registry = NativeToolRegistry(
                skill_installer_tools(workspace, data_dir, store, network_enabled=False)
            )

            def fake_install(source, *args, **kwargs):
                raise AssertionError("remote install must be blocked when network is off")

            with patch(
                "deepmate.tools.skill_installer.install_skill_bundle",
                side_effect=fake_install,
            ):
                result = execute_native_tool_request(
                    ModelToolRequest(
                        name=INSTALL_SKILL_FROM_REQUEST_TOOL_NAME,
                        id="call_1",
                        arguments={"request": "安装 gh:example/repo-review"},
                    ),
                    registry,
                    ToolAccessPolicy(ToolAccessMode.WORKSPACE_WRITE),
                )

            self.assertIsNotNone(result.error)
            self.assertIn("network", str(result.error).lower())

    def test_natural_language_skill_install_request_requires_concrete_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            data_dir = workspace / "var"
            store = CapabilityStateStore.in_data_dir(data_dir, "default")
            registry = NativeToolRegistry(skill_installer_tools(workspace, data_dir, store))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name=INSTALL_SKILL_FROM_REQUEST_TOOL_NAME,
                    id="call_1",
                    arguments={"request": "帮我安装一个写作 skill"},
                ),
                registry,
                ToolAccessPolicy(ToolAccessMode.WORKSPACE_WRITE),
            )

        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.code, "native_tool_failed")
        self.assertIn("did not include a concrete source", result.error.message)

    def test_archive_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as file:
                file.writestr("../SKILL.md", "---\nname: bad\ndescription: Bad.\n---\nBody.")

            with self.assertRaisesRegex(ValueError, "escapes destination"):
                inspect_skill_source(archive, workspace)

    def test_tar_archive_links_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            archive = root / "bad.tar"
            with tarfile.open(archive, "w") as file:
                info = tarfile.TarInfo("skill/link")
                info.type = tarfile.SYMTYPE
                info.linkname = "/tmp/outside"
                file.addfile(info)

            with self.assertRaisesRegex(ValueError, "unsupported link"):
                inspect_skill_source(archive, workspace)

    def test_tar_archive_special_files_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            archive = root / "bad-special.tar"
            with tarfile.open(archive, "w") as file:
                info = tarfile.TarInfo("skill/fifo")
                info.type = tarfile.FIFOTYPE
                file.addfile(info)

            with self.assertRaisesRegex(ValueError, "unsupported special file"):
                inspect_skill_source(archive, workspace)

    def test_zip_archive_links_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            archive = root / "bad.zip"
            info = zipfile.ZipInfo("skill/link")
            info.external_attr = 0o120777 << 16
            with zipfile.ZipFile(archive, "w") as file:
                file.writestr(info, "SKILL.md")

            with self.assertRaisesRegex(ValueError, "unsupported link"):
                inspect_skill_source(archive, workspace)

    def test_native_install_tool_writes_with_workspace_policy_and_returns_summary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            data_dir = workspace / "var"
            source = root / "source" / "writer"
            _write_skill_bundle(source, name="writer")
            store = CapabilityStateStore.in_data_dir(data_dir, "default")
            registry = NativeToolRegistry(
                skill_installer_tools(workspace, data_dir, store)
            )
            request = ModelToolRequest(
                name=INSTALL_SKILL_TOOL_NAME,
                id="call_1",
                arguments={"source": str(source)},
            )

            denied = execute_native_tool_request(request, registry)
            self.assertIsNotNone(denied.error)
            self.assertEqual(denied.error.code, "native_tool_denied")

            allowed = execute_native_tool_request(
                request,
                registry,
                ToolAccessPolicy(ToolAccessMode.WORKSPACE_WRITE),
            )

            self.assertIsNone(allowed.error)
            self.assertIsNotNone(allowed.model_result)
            self.assertIn("skill installed: writer", allowed.model_result.content)
            self.assertIn("- scope: workspace", allowed.model_result.content)
            self.assertIn("dependency setup still needs review", allowed.model_result.content)
            self.assertIn("run plan_skill_setup", allowed.model_result.content)
            self.assertNotIn("<instructions>", allowed.model_result.content)

    def test_native_install_tool_accepts_string_bool_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            data_dir = workspace / "var"
            source = root / "source" / "writer"
            _write_skill_bundle(source, name="writer")
            store = CapabilityStateStore.in_data_dir(data_dir, "default")
            install_skill_source(source, workspace, data_dir, store)
            registry = NativeToolRegistry(
                skill_installer_tools(workspace, data_dir, store)
            )
            request = ModelToolRequest(
                name=INSTALL_SKILL_TOOL_NAME,
                id="call_1",
                arguments={"source": str(source), "force": "true"},
            )

            result = execute_native_tool_request(
                request,
                registry,
                ToolAccessPolicy(ToolAccessMode.WORKSPACE_WRITE),
            )

            self.assertIsNone(result.error)
            self.assertIsNotNone(result.model_result)
            self.assertIn("skill reinstalled: writer", result.model_result.content)
            self.assertIn("- resources:", result.model_result.content)

    def test_user_scope_skill_is_discovered_across_workspaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace_a = root / "workspace-a"
            workspace_b = root / "workspace-b"
            data_dir = root / "data"
            source = root / "source" / "writer"
            workspace_a.mkdir()
            workspace_b.mkdir()
            _write_skill_bundle(source, name="writer")
            store = CapabilityStateStore.in_data_dir(data_dir, "default")

            result = install_skill_source(
                source,
                workspace_a,
                data_dir,
                store,
                target="user",
            )
            cards, warnings = discover_skill_cards(workspace_b, data_dir=data_dir)
            verified = verify_skill_install("writer", workspace_b, data_dir, store)

        self.assertFalse(warnings)
        self.assertEqual(result.manifest_record.target_scope, "user")
        self.assertTrue(any(card.name == "writer" for card in cards))
        self.assertEqual(verified.status, "ok")
        self.assertIn("skills/library/writer", verified.manifest_record.target_path)

    def test_workspace_skill_shadowing_user_skill_reports_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            data_dir = root / "data"
            user_source = root / "source" / "writer"
            workspace_skill = workspace / "skills" / "writer"
            workspace.mkdir()
            _write_skill_bundle(user_source, name="writer")
            _write_skill_bundle(workspace_skill, name="writer")
            store = CapabilityStateStore.in_data_dir(data_dir, "default")
            install_skill_source(
                user_source,
                workspace,
                data_dir,
                store,
                target="user",
            )

            cards, warnings = discover_skill_cards(workspace, data_dir=data_dir)

        self.assertEqual(sum(1 for card in cards if card.name == "writer"), 1)
        self.assertTrue(any(warning.code == "duplicate_skill_name" for warning in warnings))
        self.assertTrue(any("user skill skipped" in warning.message for warning in warnings))

    def test_plan_skill_setup_marks_pending_without_running_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            data_dir = workspace / "var"
            source = root / "source" / "writer"
            _write_skill_bundle(source, name="writer")
            store = CapabilityStateStore.in_data_dir(data_dir, "default")
            install_skill_source(source, workspace, data_dir, store)
            registry = NativeToolRegistry(
                skill_installer_tools(workspace, data_dir, store)
            )

            result = execute_native_tool_request(
                ModelToolRequest(
                    name=PLAN_SKILL_SETUP_TOOL_NAME,
                    id="call_1",
                    arguments={"name": "writer"},
                ),
                registry,
            )
            manifest = InstalledSkillManifestStore.in_data_dir(data_dir).get("writer")

        self.assertIsNone(result.error)
        self.assertIn("recommended command: python3 scripts/setup.py", result.model_result.content)
        self.assertEqual(manifest.setup_status, "pending")
        self.assertEqual(manifest.setup_command, "python3 scripts/setup.py")

    def test_run_skill_setup_updates_manifest_on_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            data_dir = workspace / "var"
            source = root / "source" / "writer"
            _write_skill_bundle(source, name="writer")
            (source / "scripts" / "setup.py").write_text(
                "from pathlib import Path\n"
                "Path('setup-marker.txt').write_text('ok', encoding='utf-8')\n"
                "print('setup ok')\n",
                encoding="utf-8",
            )
            store = CapabilityStateStore.in_data_dir(data_dir, "default")
            install_skill_source(source, workspace, data_dir, store)
            registry = NativeToolRegistry(
                skill_installer_tools(
                    workspace,
                    data_dir,
                    store,
                    shell_enabled=True,
                    sandbox_mode=SandboxMode.OFF,
                )
            )

            result = execute_native_tool_request(
                ModelToolRequest(
                    name=RUN_SKILL_SETUP_TOOL_NAME,
                    id="call_1",
                    arguments={"name": "writer"},
                ),
                registry,
                ToolAccessPolicy(ToolAccessMode.READ_ONLY, shell_enabled=True),
            )
            manifest = InstalledSkillManifestStore.in_data_dir(data_dir).get("writer")
            marker = workspace / "skills" / "writer" / "setup-marker.txt"
            marker_exists = marker.is_file()

        self.assertIsNone(result.error)
        self.assertIn("setup_status: completed", result.model_result.content)
        self.assertTrue(marker_exists)
        self.assertEqual(manifest.setup_status, "completed")
        self.assertEqual(manifest.setup_command, "python3 scripts/setup.py")
        self.assertTrue(manifest.setup_updated_at)

    def test_run_user_scope_skill_setup_uses_user_skill_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace_a = root / "workspace-a"
            workspace_b = root / "workspace-b"
            data_dir = root / "data"
            source = root / "source" / "writer"
            workspace_a.mkdir()
            workspace_b.mkdir()
            _write_skill_bundle(source, name="writer")
            (source / "scripts" / "setup.py").write_text(
                "from pathlib import Path\n"
                "Path('setup-marker.txt').write_text(str(Path.cwd()), encoding='utf-8')\n"
                "print('setup ok')\n",
                encoding="utf-8",
            )
            store = CapabilityStateStore.in_data_dir(data_dir, "default")
            install_skill_source(source, workspace_a, data_dir, store, target="user")
            registry = NativeToolRegistry(
                skill_installer_tools(
                    workspace_b,
                    data_dir,
                    store,
                    shell_enabled=True,
                    sandbox_mode=SandboxMode.OFF,
                )
            )

            result = execute_native_tool_request(
                ModelToolRequest(
                    name=RUN_SKILL_SETUP_TOOL_NAME,
                    id="call_1",
                    arguments={"name": "writer"},
                ),
                registry,
                ToolAccessPolicy(ToolAccessMode.READ_ONLY, shell_enabled=True),
            )
            manifest = InstalledSkillManifestStore.in_data_dir(data_dir).get("writer")
            user_marker = data_dir / "skills" / "library" / "writer" / "setup-marker.txt"
            marker_exists = user_marker.is_file()
            marker_content = (
                user_marker.read_text(encoding="utf-8") if marker_exists else ""
            )
            workspace_b_user_skill_exists = (
                workspace_b / "skills" / "library" / "writer"
            ).exists()

        self.assertIsNone(result.error)
        self.assertTrue(marker_exists)
        self.assertIn("skills/library/writer", marker_content)
        self.assertFalse(workspace_b_user_skill_exists)
        self.assertEqual(manifest.setup_status, "completed")

    def test_run_skill_setup_updates_manifest_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            data_dir = workspace / "var"
            source = root / "source" / "writer"
            _write_skill_bundle(source, name="writer")
            (source / "scripts" / "setup.py").write_text(
                "import sys\nprint('setup failed')\nsys.exit(3)\n",
                encoding="utf-8",
            )
            store = CapabilityStateStore.in_data_dir(data_dir, "default")
            install_skill_source(source, workspace, data_dir, store)
            registry = NativeToolRegistry(
                skill_installer_tools(
                    workspace,
                    data_dir,
                    store,
                    shell_enabled=True,
                    sandbox_mode=SandboxMode.OFF,
                )
            )

            result = execute_native_tool_request(
                ModelToolRequest(
                    name=RUN_SKILL_SETUP_TOOL_NAME,
                    id="call_1",
                    arguments={"name": "writer"},
                ),
                registry,
                ToolAccessPolicy(ToolAccessMode.READ_ONLY, shell_enabled=True),
            )
            manifest = InstalledSkillManifestStore.in_data_dir(data_dir).get("writer")

        self.assertIsNone(result.error)
        self.assertIn("setup_status: failed", result.model_result.content)
        self.assertEqual(manifest.setup_status, "failed")

    def test_run_skill_setup_warns_when_permission_only_backend_is_used(self) -> None:
        class PermissionOnlyRunner:
            def run(self, command, policy, *, timeout_seconds):
                return SandboxRunResult(
                    stdout="setup ok\n",
                    stderr="",
                    exit_code=0,
                    backend="permission-only",
                    sandboxed=False,
                    refs=("sandbox_backend=permission-only", "sandboxed=false"),
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            data_dir = workspace / "var"
            source = root / "source" / "writer"
            _write_skill_bundle(source, name="writer")
            store = CapabilityStateStore.in_data_dir(data_dir, "default")
            install_skill_source(source, workspace, data_dir, store)
            cache = SessionApprovalCache()
            cache.allow_for_session("capability:shell-network")
            registry = NativeToolRegistry(
                skill_installer_tools(
                    workspace,
                    data_dir,
                    store,
                    shell_enabled=True,
                    network_enabled=True,
                    sandbox_mode=SandboxMode.AUTO,
                    approval_cache=cache,
                    runner=PermissionOnlyRunner(),
                )
            )

            result = execute_native_tool_request(
                ModelToolRequest(
                    name=RUN_SKILL_SETUP_TOOL_NAME,
                    id="call_1",
                    arguments={"name": "writer", "network": "on"},
                ),
                registry,
                ToolAccessPolicy(ToolAccessMode.READ_ONLY, shell_enabled=True),
            )

        self.assertIsNone(result.error)
        self.assertIn("permission-only enforcement", result.model_result.content)
        self.assertIn("setup_sandboxed=false", result.model_result.refs)

    def test_run_skill_setup_requires_shell_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            data_dir = workspace / "var"
            source = root / "source" / "writer"
            _write_skill_bundle(source, name="writer")
            store = CapabilityStateStore.in_data_dir(data_dir, "default")
            install_skill_source(source, workspace, data_dir, store)
            registry = NativeToolRegistry(
                skill_installer_tools(
                    workspace,
                    data_dir,
                    store,
                    shell_enabled=True,
                    sandbox_mode=SandboxMode.OFF,
                )
            )

            result = execute_native_tool_request(
                ModelToolRequest(
                    name=RUN_SKILL_SETUP_TOOL_NAME,
                    id="call_1",
                    arguments={"name": "writer"},
                ),
                registry,
                ToolAccessPolicy(ToolAccessMode.WORKSPACE_WRITE, shell_enabled=False),
            )

        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.code, "native_tool_denied")
        self.assertIn("Approve shell access", result.error.message)
        self.assertNotIn("approved", result.error.message.lower())

    def test_run_skill_setup_safety_denial_does_not_claim_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            data_dir = workspace / "var"
            source = root / "source" / "writer"
            _write_skill_bundle(source, name="writer")
            store = CapabilityStateStore.in_data_dir(data_dir, "default")
            install_skill_source(source, workspace, data_dir, store)
            registry = NativeToolRegistry(
                skill_installer_tools(
                    workspace,
                    data_dir,
                    store,
                    shell_enabled=True,
                    network_enabled=False,
                    sandbox_mode=SandboxMode.OFF,
                )
            )

            result = execute_native_tool_request(
                ModelToolRequest(
                    name=RUN_SKILL_SETUP_TOOL_NAME,
                    id="call_1",
                    arguments={"name": "writer", "network": "on"},
                ),
                registry,
                ToolAccessPolicy(ToolAccessMode.READ_ONLY, shell_enabled=True),
            )

        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.code, "native_tool_failed")
        self.assertIn("--allow-network", result.error.message)
        self.assertNotIn("approved", result.error.message.lower())

    def test_run_skill_setup_is_hidden_but_registered_without_shell_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            data_dir = workspace / "var"
            store = CapabilityStateStore.in_data_dir(data_dir, "default")

            registry = NativeToolRegistry(skill_installer_tools(workspace, data_dir, store))
            visible = {schema["name"] for schema in registry.schemas()}
            registered = {tool.name for tool in registry.list_tools()}

        self.assertNotIn(RUN_SKILL_SETUP_TOOL_NAME, visible)
        self.assertIn(LOAD_SKILL_INSTALLER_TOOLS_NAME, visible)
        self.assertIn(INSTALL_SKILL_FROM_REQUEST_TOOL_NAME, visible)
        self.assertIn(RUN_SKILL_SETUP_TOOL_NAME, registered)

    def test_uninstall_imported_skill_deletes_bundle_and_archives_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            data_dir = workspace / "var"
            source = root / "source" / "writer"
            _write_skill_bundle(source, name="writer")
            store = CapabilityStateStore.in_data_dir(data_dir, "default")
            install_skill_source(source, workspace, data_dir, store)

            result = uninstall_skill("writer", workspace, data_dir, store)

            self.assertEqual(result.status, "uninstalled")
            self.assertTrue(result.deleted_files)
            self.assertFalse((workspace / "skills" / "writer").exists())
            self.assertIsNone(InstalledSkillManifestStore.in_data_dir(data_dir).get("writer"))
            state = store.load()["skill:workspace:writer"]
            self.assertTrue(state.hidden)
            self.assertEqual(state.asset_state.value, "archived")


def _write_skill_bundle(path: Path, *, name: str = "self-improving-agent") -> None:
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(
        "\n".join(
            (
                "---",
                f"name: {name}",
                "description: Improve agent behavior from repeated workflow signals.",
                "allowed-tools: [read, search]",
                "---",
                "Follow the skill checklist.",
                "Read ${SKILL_DIR}/references/guide.md when needed.",
            )
        ),
        encoding="utf-8",
    )
    (path / "references").mkdir()
    (path / "references" / "guide.md").write_text("Reference guide.", encoding="utf-8")
    (path / "scripts").mkdir()
    (path / "scripts" / "setup.py").write_text("print('setup')\n", encoding="utf-8")
    (path / "assets").mkdir()
    (path / "assets" / "icon.txt").write_text("icon", encoding="utf-8")
    (path / "agents").mkdir()
    (path / "agents" / "openai.yaml").write_text("agent: test\n", encoding="utf-8")
    (path / "examples").mkdir()
    (path / "examples" / "example.md").write_text("Example.", encoding="utf-8")


def _zip_dir(source: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w") as file:
        for path in source.rglob("*"):
            if path.is_file():
                file.write(path, path.relative_to(source.parent))


if __name__ == "__main__":
    unittest.main()
