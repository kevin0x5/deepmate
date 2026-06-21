from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from deepmate.app import AppSettings, ProviderSettings
from deepmate.channels.cli import (
    _build_parser,
    _conversation_budget_policy,
    _default_model_context_tokens,
    _install_memory_maintenance_schedule,
    _finish_runtime_validation,
    _load_or_create_session,
    main,
    _resolve_chat_model,
    _runtime_validation_errors,
    _runtime_validation_prompt,
)
from deepmate.domain import Message, MessageRole, ProfileRef
from deepmate.evolution import apply_behavior_hint_change
from deepmate.local import LocalModelInstallResult
from deepmate.local.ollama import LocalModelStatus
from deepmate.local.presets import local_model_by_id
from deepmate.providers import (
    ModelConversationItem,
    ModelRequest,
    ModelResponse,
)
from deepmate.runtime import AgentStepResult, UserTurnResult
from deepmate.storage import SessionStore
from deepmate.trace import JsonlTraceSink, TraceEvent, TraceRecorder


class _StubMaintenanceProvider:
    def __init__(self, content: str) -> None:
        self.content = content
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return ModelResponse(content=self.content)


class _FakeLocalRuntime:
    def __init__(
        self,
        *,
        status: LocalModelStatus | None = None,
        prepare_result: LocalModelInstallResult | None = None,
        prepared=None,
    ) -> None:
        self._status = status or LocalModelStatus(
            available=False,
            installed=False,
            running=False,
        )
        self._prepare_result = prepare_result
        self._prepared = prepared
        self.prepared_calls: list[bool] = []
        self.prepare_calls = []

    def status(self) -> LocalModelStatus:
        return self._status

    def prepared_model(self, *, start_server: bool = True):
        self.prepared_calls.append(start_server)
        return self._prepared

    def prepare_model(self, preset, *, progress=None, state_store=None):
        self.prepare_calls.append((preset, state_store))
        if progress is not None:
            progress(type("Progress", (), {"message": "checking local model"})())
        if self._prepare_result is not None:
            return self._prepare_result
        return LocalModelInstallResult(ok=False, preset=preset, message="not ready")


class RuntimeValidationTests(unittest.TestCase):
    def test_default_model_context_tokens_does_not_require_provider_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = AppSettings(
                workspace=Path(tmp),
                data_dir=Path(tmp) / "var",
                active_profile="default",
                trace_sink=Path(tmp) / "var" / "traces" / "trace.jsonl",
                default_provider="missing",
                providers={},
                model_context_windows={"default": 123_456},
            )

            self.assertEqual(_default_model_context_tokens(settings), 123_456)

    def test_memory_maintenance_flags_parse_without_prompt(self) -> None:
        args = _build_parser().parse_args(
            (
                "--run-memory-maintenance",
                "--memory-maintenance-date",
                "2026-06-03",
                "--force-memory-maintenance",
            )
        )

        self.assertTrue(args.run_memory_maintenance)
        self.assertEqual(args.memory_maintenance_date, "2026-06-03")
        self.assertTrue(args.force_memory_maintenance)

        install_args = _build_parser().parse_args(
            ("--install-memory-maintenance-schedule",)
        )

        self.assertTrue(install_args.install_memory_maintenance_schedule)

        mcp_args = _build_parser().parse_args(("--mcp-status",))

        self.assertTrue(mcp_args.mcp_status)

        sandbox_status_args = _build_parser().parse_args(("--sandbox-status",))

        self.assertTrue(sandbox_status_args.sandbox_status)

        sandbox_args = _build_parser().parse_args(
            (
                "--shell",
                "--allow-network",
                "--allow-env-change",
                "--sandbox",
                "require",
                "--mcp-write",
                "check",
            )
        )

        self.assertTrue(sandbox_args.shell)
        self.assertTrue(sandbox_args.allow_network)
        self.assertTrue(sandbox_args.allow_env_change)
        self.assertEqual(sandbox_args.sandbox, "require")
        self.assertTrue(sandbox_args.mcp_write)

        interactive_legacy_args = _build_parser().parse_args(("--interactive-legacy",))

        self.assertTrue(interactive_legacy_args.interactive_legacy)

        evolution_args = _build_parser().parse_args(("--run-evolution-maintenance",))

        self.assertTrue(evolution_args.run_evolution_maintenance)

        list_evolution_args = _build_parser().parse_args(("--list-evolution-changes",))

        self.assertTrue(list_evolution_args.list_evolution_changes)

        rollback_args = _build_parser().parse_args(
            ("--rollback-evolution-change", "chg_example")
        )

        self.assertEqual(rollback_args.rollback_evolution_change, "chg_example")

        local_status_args = _build_parser().parse_args(("--local-status",))

        self.assertTrue(local_status_args.local_status)

        local_prepare_args = _build_parser().parse_args(
            ("--prepare-local-model", "qwen3-local")
        )

        self.assertEqual(local_prepare_args.prepare_local_model, "qwen3-local")

    def test_upgrade_model_flag_selects_provider_upgrade_model(self) -> None:
        provider = ProviderSettings(
            name="stub",
            base_url="https://example.test",
            default_model="stub-flash",
            upgrade_model="stub-pro",
        )

        upgrade_args = _build_parser().parse_args(("--upgrade-model", "hello"))
        explicit_args = _build_parser().parse_args(
            ("--upgrade-model", "--model", "manual-model", "hello")
        )

        self.assertEqual(_resolve_chat_model(upgrade_args, provider), "stub-pro")
        self.assertEqual(_resolve_chat_model(explicit_args, provider), "manual-model")

    def test_upgrade_model_flag_requires_provider_upgrade_model(self) -> None:
        provider = ProviderSettings(
            name="stub",
            base_url="https://example.test",
            default_model="stub-flash",
        )
        args = _build_parser().parse_args(("--upgrade-model", "hello"))

        with self.assertRaises(ValueError):
            _resolve_chat_model(args, provider)

    def test_provider_model_field_is_primary_chat_model(self) -> None:
        provider = ProviderSettings(
            name="custom",
            base_url="https://example.test",
            default_model="legacy-flash",
            model="qwen-coder-plus",
            context_window=128_000,
        )
        args = _build_parser().parse_args(("hello",))

        self.assertEqual(_resolve_chat_model(args, provider), "qwen-coder-plus")

    def test_conversation_budget_uses_provider_context_window(self) -> None:
        provider = ProviderSettings(
            name="custom",
            base_url="https://example.test",
            default_model="qwen-coder-plus",
            model="qwen-coder-plus",
            context_window=128_000,
        )
        settings = AppSettings(
            workspace=Path("/workspace"),
            data_dir=Path("/workspace/var"),
            active_profile="default",
            trace_sink=Path("/workspace/var/traces/trace.jsonl"),
            default_provider="custom",
        )

        policy = _conversation_budget_policy(
            settings,
            "qwen-coder-plus",
            provider_settings=provider,
        )

        self.assertEqual(policy.model_context_tokens, 128_000)
        self.assertEqual(policy.response_token_reserve, 8_192)
        self.assertEqual(policy.safety_margin_tokens, 6_400)
        self.assertEqual(policy.history_token_budget, 85_056)

    def test_cli_help_groups_common_flags(self) -> None:
        help_text = _build_parser().format_help()

        self.assertIn("Core:", help_text)
        self.assertIn("Tool Access:", help_text)
        self.assertIn("Sessions And Recovery:", help_text)
        self.assertIn("Remote:", help_text)
        self.assertIn("--interactive", help_text)
        self.assertIn("--setup-key", help_text)
        self.assertIn("--setup-wecom", help_text)
        self.assertIn("--workspace-write", help_text)
        self.assertIn("--rewind", help_text)

    def test_setup_key_saves_provider_key_to_local_private_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    (
                        "--workspace",
                        str(workspace),
                        "--setup-key",
                        "test-secret",
                    )
                )

            secret_path = workspace / "var" / "secrets" / "providers.env"
            secret_exists = secret_path.is_file()
            secret_content = secret_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertIn("Model API key saved locally.", stdout.getvalue())
        self.assertTrue(secret_exists)
        self.assertIn('DEEPSEEK_API_KEY="test-secret"', secret_content)

    def test_setup_wecom_saves_remote_settings_to_local_private_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    (
                        "--workspace",
                        str(workspace),
                        "--setup-wecom",
                        "bot-id",
                        "bot-secret",
                        "alice,bob",
                        "full",
                    )
                )

            secret_path = workspace / "var" / "secrets" / "providers.env"
            secret_exists = secret_path.is_file()
            secret_content = secret_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertIn("Enterprise WeChat remote saved locally.", stdout.getvalue())
        self.assertTrue(secret_exists)
        self.assertIn('DEEPMATE_WECOM_BOT_ID="bot-id"', secret_content)
        self.assertIn('DEEPMATE_WECOM_SECRET="bot-secret"', secret_content)
        self.assertIn('DEEPMATE_WECOM_ALLOWED_USERS="alice,bob"', secret_content)
        self.assertIn('DEEPMATE_WECOM_GROUP_POLICY="full"', secret_content)

    def test_remote_validate_missing_wecom_config_prints_setup_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            stdout = io.StringIO()
            stderr = io.StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main(
                    (
                        "--workspace",
                        str(workspace),
                        "--remote-validate",
                        "--remote",
                        "wecom",
                    )
                )

            initialized_config = (workspace / "config" / "deepmate.yaml").is_file()

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        error = stderr.getvalue()
        self.assertIn("remote.wecom.bot_id is required", error)
        self.assertIn("deepmate --setup-wecom <bot-id> <secret>", error)
        self.assertTrue(initialized_config)
        self.assertIn("deepmate --remote wecom", error)

    def test_interactive_start_resumes_latest_workspace_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            settings = AppSettings(
                workspace=workspace,
                data_dir=root / "var",
                active_profile="default",
                trace_sink=root / "var" / "trace.jsonl",
                default_provider="deepseek",
            )
            store = SessionStore(root / "sessions")
            older = store.create(
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                title="older",
            )
            newer = store.create(
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                title="newer",
            )
            store.touch(older.session_id)

            with redirect_stdout(io.StringIO()):
                resumed = _load_or_create_session(
                    store,
                    settings,
                    profile_name=None,
                    session_id=None,
                    title=None,
                    prompt="",
                    interactive=True,
                )

            self.assertEqual(resumed.session_id, older.session_id)
            self.assertNotEqual(resumed.session_id, newer.session_id)
            self.assertEqual(len(store.list_recent_for_workspace(workspace)), 2)

    def test_memory_maintenance_schedule_uses_absolute_pythonpath(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            fake_home = root / "home"
            settings = AppSettings(
                workspace=workspace,
                data_dir=workspace / "var",
                active_profile="default",
                trace_sink=workspace / "var" / "trace.jsonl",
                default_provider="deepseek",
            )

            with patch("deepmate.channels.cli.sys.platform", "darwin"), patch(
                "deepmate.channels.cli.Path.home",
                return_value=fake_home,
            ):
                path = _install_memory_maintenance_schedule(settings)

            payload = path.read_text(encoding="utf-8")
            self.assertIn("PYTHONPATH=/", payload)
            self.assertNotIn("PYTHONPATH=src", payload)
            self.assertIn("--run-memory-maintenance", payload)

    def test_sandbox_status_cli_does_not_require_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    (
                        "--workspace",
                        str(workspace),
                        "--sandbox-status",
                    )
                )

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("Sandbox status:", output)
        self.assertIn("- backend:", output)
        self.assertIn("- network_default: off", output)

    def test_doctor_cli_does_not_require_provider_or_optional_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            stdout = io.StringIO()
            stderr = io.StringIO()
            runtime = _FakeLocalRuntime(
                status=LocalModelStatus(
                    available=False,
                    installed=False,
                    running=False,
                    message="Ollama is not installed.",
                )
            )

            with (
                patch("deepmate.channels.cli._local_runtime", return_value=runtime),
                patch("deepmate.channels.cli.electron_pet_command", return_value=None),
                patch(
                    "deepmate.channels.cli.AgentBrowserBackend.is_available",
                    return_value=False,
                ),
                patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(("--workspace", str(workspace), "--doctor"))

        self.assertEqual(exit_code, 0, stderr.getvalue())
        output = stdout.getvalue()
        self.assertIn("Deepmate doctor", output)
        self.assertIn("Base install:", output)
        self.assertIn("model_key: missing", output)
        self.assertIn("desktop_pet: optional missing", output)
        self.assertIn("browser_backend: optional missing", output)
        self.assertIn("local_model_runtime: optional missing", output)
        self.assertIn("does not block normal CLI/TUI use", output)
        self.assertIn("pet_ui/node_modules/ is generated by npm", output)

    def test_local_status_cli_does_not_require_provider_key_or_start_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            stdout = io.StringIO()
            runtime = _FakeLocalRuntime(
                status=LocalModelStatus(
                    available=False,
                    installed=True,
                    running=False,
                    message="Ollama is installed but not running.",
                )
            )

            with (
                patch("deepmate.channels.cli._local_runtime", return_value=runtime),
                redirect_stdout(stdout),
                redirect_stderr(io.StringIO()),
            ):
                exit_code = main(("--workspace", str(workspace), "--local-status"))

        self.assertEqual(exit_code, 0)
        text = stdout.getvalue()
        self.assertIn("Local model status:", text)
        self.assertIn("ollama_installed: true", text)
        self.assertIn("prepared_model: none", text)
        self.assertEqual(runtime.prepared_calls, [False])

    def test_prepare_local_model_cli_uses_recommended_model_and_reports_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            stdout = io.StringIO()
            stderr = io.StringIO()
            preset = local_model_by_id("qwen3-local")
            self.assertIsNotNone(preset)
            runtime = _FakeLocalRuntime(
                prepare_result=LocalModelInstallResult(
                    ok=True,
                    preset=preset,
                    message="local ready",
                )
            )

            with (
                patch("deepmate.channels.cli.recommended_local_model", return_value=preset),
                patch("deepmate.channels.cli._local_runtime", return_value=runtime),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(("--workspace", str(workspace), "--prepare-local-model"))

        self.assertEqual(exit_code, 0)
        self.assertIn("local ready", stdout.getvalue())
        self.assertEqual([call[0].id for call in runtime.prepare_calls], ["qwen3-local"])
        self.assertTrue(stderr.getvalue().strip())

    def test_noninteractive_local_provider_requires_prepared_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            config_dir = workspace / "config"
            config_dir.mkdir(parents=True)
            (config_dir / "deepmate.yaml").write_text(
                "\n".join(
                    (
                        "runtime:",
                        "  data_dir: var",
                        "provider:",
                        "  default: local",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            (config_dir / "providers.yaml").write_text(
                "\n".join(
                    (
                        "providers:",
                        "  local:",
                        "    base_url: http://127.0.0.1:11434/v1",
                        "    model: qwen3:4b",
                        "    api_key_required: false",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            stderr = io.StringIO()
            runtime = _FakeLocalRuntime()

            with (
                patch("deepmate.channels.cli._local_runtime", return_value=runtime),
                redirect_stdout(io.StringIO()),
                redirect_stderr(stderr),
            ):
                exit_code = main(("--workspace", str(workspace), "hello"))

        self.assertEqual(exit_code, 1)
        self.assertIn("Local model is not ready.", stderr.getvalue())
        self.assertIn("--prepare-local-model", stderr.getvalue())
        self.assertEqual(runtime.prepared_calls, [True])

    def test_evolution_maintenance_cli_does_not_require_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            workspace.mkdir(exist_ok=True)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    (
                        "--workspace",
                        str(workspace),
                        "--run-evolution-maintenance",
                    )
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("evolution maintenance:", stdout.getvalue())

    def test_list_evolution_changes_cli_does_not_require_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            workspace.mkdir(exist_ok=True)
            apply_behavior_hint_change(
                workspace=workspace,
                data_dir=workspace / "var",
                profile=ProfileRef(name="default", uri="profiles/default"),
                hints=("Prefer closed-loop execution.",),
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    (
                        "--workspace",
                        str(workspace),
                        "--list-evolution-changes",
                    )
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("Evolution changes", stdout.getvalue())
            self.assertIn("behavior_patch", stdout.getvalue())

    def test_evolution_rollback_cli_restores_behavior_file_without_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            profile = ProfileRef(name="default", uri="profiles/default")
            behavior_path = workspace / ".deepmate" / "behavior.md"
            behavior_path.parent.mkdir(parents=True)
            original = "# Behavior Hints\n\n- Original hint.\n"
            behavior_path.write_text(original, encoding="utf-8")
            change = apply_behavior_hint_change(
                workspace=workspace,
                data_dir=workspace / "var",
                profile=profile,
                hints=("New hint.",),
            )
            assert change is not None
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    (
                        "--workspace",
                        str(workspace),
                        "--rollback-evolution-change",
                        change.change_id,
                    )
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(behavior_path.read_text(encoding="utf-8"), original)
            self.assertIn("evolution rollback:", stdout.getvalue())

    def test_memory_maintenance_cli_runs_capability_maintenance_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            profile_dir = workspace / "profiles" / "default"
            skill_dir = workspace / "skills" / "demo"
            config_dir = workspace / "config"
            profile_dir.mkdir(parents=True)
            skill_dir.mkdir(parents=True)
            config_dir.mkdir(parents=True)
            (profile_dir / "identity.md").write_text("# Identity\n", encoding="utf-8")
            (profile_dir / "soul.md").write_text("# Soul\n", encoding="utf-8")
            (profile_dir / "user.md").write_text("", encoding="utf-8")
            (profile_dir / "memory.md").write_text("", encoding="utf-8")
            (skill_dir / "SKILL.md").write_text(
                "---\nname: demo\ndescription: Demo skill.\n---\nBody.",
                encoding="utf-8",
            )
            (config_dir / "deepmate.yaml").write_text(
                "\n".join(
                    (
                        "runtime:",
                        "  data_dir: var",
                        "  active_profile: default",
                        "trace:",
                        "  sink: var/traces/trace.jsonl",
                        "provider:",
                        "  default: stub",
                        "models:",
                        "  memory:",
                        "    model: stub-memory",
                    )
                ),
                encoding="utf-8",
            )
            (config_dir / "providers.yaml").write_text(
                "\n".join(
                    (
                        "providers:",
                        "  stub:",
                        "    base_url: http://example.test",
                        "    default_model: stub-main",
                        "    api_key_env: STUB_API_KEY",
                    )
                ),
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with (
                patch.dict(os.environ, {"STUB_API_KEY": "test-key"}),
                patch(
                    "deepmate.channels.cli.ChatCompletionsProvider",
                    lambda base_url, api_key: _StubMaintenanceProvider("{}"),
                ),
                redirect_stdout(stdout),
            ):
                exit_code = main(
                    (
                        "--workspace",
                        str(workspace),
                        "--run-memory-maintenance",
                    )
                )

            output = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertIn("memory maintenance:", output)
            self.assertIn("capability maintenance:", output)
            self.assertIn("evolution maintenance:", output)
            state_path = (
                workspace
                / "var"
                / "capabilities"
                / "default"
                / "capability_index.json"
            )
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["capabilities"][0]["name"], "demo")
            self.assertEqual(state["capabilities"][0]["temperature"], "hot")

    def test_runtime_validation_prompt_mentions_requested_checks(self) -> None:
        args = argparse.Namespace(
            read_only_tools=True,
            workspace_write=False,
            subagents=True,
            mcp=True,
            shell=True,
        )

        prompt = _runtime_validation_prompt(args)

        self.assertIn("list_directory", prompt)
        self.assertIn("run_subagent", prompt)
        self.assertIn("max_steps=2", prompt)
        self.assertIn("Do not give the subagent any tools", prompt)
        self.assertIn("MCP check", prompt)
        self.assertIn("run_shell_command", prompt)
        self.assertIn("Use command `pwd`", prompt)
        self.assertIn("deepmate runtime ok", prompt)

    def test_runtime_validation_prompt_requires_write_when_enabled(self) -> None:
        args = argparse.Namespace(
            read_only_tools=False,
            workspace_write=True,
            subagents=False,
            mcp=False,
            shell=False,
        )

        prompt = _runtime_validation_prompt(args)

        self.assertIn("write_text_file", prompt)
        self.assertIn("runtime_validation_write_test.txt", prompt)
        self.assertIn("overwrite=true", prompt)
        self.assertNotIn("Do not write files", prompt)

    def test_finish_runtime_validation_writes_reportable_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            store = SessionStore.in_directory(root / "sessions")
            session = store.create(
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                title="Runtime validation",
            )
            transcript = store.transcript_store(session)
            transcript.append_item(
                ModelConversationItem.from_message(
                    Message(role=MessageRole.USER, content="validate runtime")
                )
            )
            transcript.append_item(
                ModelConversationItem.from_message(
                    Message(role=MessageRole.ASSISTANT, content="deepmate runtime ok")
                )
            )
            trace_path = root / "trace.jsonl"
            trace_recorder = TraceRecorder(JsonlTraceSink(trace_path))
            trace_recorder.record(
                TraceEvent(
                    kind="model_response_received",
                    summary="Model response received for step 1.",
                    refs=(
                        f"session_id={session.session_id}",
                        "model=deepseek-v4-flash",
                        "step=1",
                        "input_tokens=12",
                        "output_tokens=4",
                    ),
                )
            )
            trace_recorder.record(
                TraceEvent(
                    kind="native_tool_completed",
                    summary="Native tool completed: list_directory.",
                    refs=(
                        "tool_source=native",
                        "list_directory",
                        f"session_id={session.session_id}",
                    ),
                )
            )
            trace_recorder.record(
                TraceEvent(
                    kind="activity_daily_note_written",
                    summary="Activity daily note written.",
                    refs=(
                        "event=runtime_validation_end",
                        f"session_id={session.session_id}",
                    ),
                )
            )
            request = ModelRequest(
                model="deepseek-v4-flash",
                conversation=(
                    ModelConversationItem.from_message(
                        Message(role=MessageRole.SYSTEM, content="system")
                    ),
                    ModelConversationItem.from_message(
                        Message(role=MessageRole.USER, content="validate runtime")
                    ),
                ),
            )
            response = ModelResponse(content="deepmate runtime ok")
            result = UserTurnResult(
                steps=(
                    AgentStepResult(
                        request=request,
                        response=response,
                        tool_results=(),
                    ),
                ),
                conversation=(
                    ModelConversationItem.from_message(
                        Message(role=MessageRole.USER, content="validate runtime")
                    ),
                    ModelConversationItem.from_message(
                        Message(role=MessageRole.ASSISTANT, content="deepmate runtime ok")
                    ),
                ),
            )

            with redirect_stdout(io.StringIO()):
                exit_code = _finish_runtime_validation(
                    result=result,
                    session_store=store,
                    session=session,
                    transcript=transcript,
                    trace_path=trace_path,
                    trace_recorder=trace_recorder,
                    model="deepseek-v4-flash",
                    require_native_tool=True,
                    require_workspace_write=False,
                    require_shell=False,
                    require_subagent=False,
                    require_activity_note=True,
                    mcp_tool_count=0,
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("runtime_validation_finished", trace_path.read_text(encoding="utf-8"))
            self.assertEqual(len(transcript.load_records()), 2)
            self.assertEqual(
                _runtime_validation_errors(
                    result=result,
                    transcript_record_count=2,
                    trace_records=(
                        {
                            "kind": "model_response_received",
                            "refs": ["session_id=abc"],
                        },
                        {
                            "kind": "native_tool_completed",
                            "refs": ["tool_source=native", "list_directory"],
                        },
                        {
                            "kind": "activity_daily_note_written",
                            "refs": ["event=runtime_validation_end"],
                        },
                    ),
                    require_native_tool=True,
                    require_workspace_write=False,
                    require_shell=False,
                    require_subagent=False,
                    require_activity_note=True,
                ),
                (),
            )

    def test_runtime_validation_detects_missing_write_and_activity(self) -> None:
        request = ModelRequest(
            model="deepseek-v4-flash",
            conversation=(
                ModelConversationItem.from_message(
                    Message(role=MessageRole.SYSTEM, content="system")
                ),
            ),
        )
        result = UserTurnResult(
            steps=(
                AgentStepResult(
                    request=request,
                    response=ModelResponse(content="deepmate runtime ok"),
                    tool_results=(),
                ),
            ),
            conversation=(),
        )

        errors = _runtime_validation_errors(
            result=result,
            transcript_record_count=1,
            trace_records=(
                {"kind": "model_response_received", "refs": ["session_id=abc"]},
                {
                    "kind": "native_tool_completed",
                    "refs": ["tool_source=native", "list_directory"],
                },
            ),
            require_native_tool=True,
            require_workspace_write=True,
            require_shell=False,
            require_subagent=False,
            require_activity_note=True,
        )

        self.assertIn("workspace write tool call was not completed", errors)
        self.assertIn("activity daily note was not written", errors)


if __name__ == "__main__":
    unittest.main()
