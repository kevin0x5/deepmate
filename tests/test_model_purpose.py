import unittest
import os
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from deepmate.app import (
    AppSettings,
    ContextSettings,
    ModelPurposeSettings,
    load_settings,
    provider_api_key,
    provider_secret_path,
    resolve_model_purpose,
    save_provider_api_key,
    save_wecom_remote_settings,
)
from deepmate.local import LOCAL_PROVIDER_NAME


class ModelPurposeTests(unittest.TestCase):
    def test_resolves_configured_model_and_options(self):
        settings = AppSettings(
            workspace=Path("/workspace"),
            data_dir=Path("/workspace/var"),
            active_profile="default",
            trace_sink=Path("/workspace/var/traces/trace.jsonl"),
            default_provider="deepseek",
            model_purposes={
                "memory": ModelPurposeSettings(
                    model="deepseek-v4-flash",
                    thinking="disabled",
                    temperature=0.1,
                    max_tokens=1200,
                )
            },
        )

        config = resolve_model_purpose(
            settings,
            "memory",
            fallback_model="deepseek-v4-pro",
            option_overrides={"max_tokens": 800},
        )

        self.assertEqual(config.model, "deepseek-v4-flash")
        self.assertEqual(
            config.options,
            {
                "thinking": {"type": "disabled"},
                "temperature": 0.1,
                "max_tokens": 800,
            },
        )

    def test_falls_back_to_current_chat_model(self):
        settings = AppSettings(
            workspace=Path("/workspace"),
            data_dir=Path("/workspace/var"),
            active_profile="default",
            trace_sink=Path("/workspace/var/traces/trace.jsonl"),
            default_provider="deepseek",
        )

        config = resolve_model_purpose(
            settings,
            "summary",
            fallback_model="deepseek-v4-flash",
        )

        self.assertEqual(config.model, "deepseek-v4-flash")
        self.assertEqual(config.options, {})

    def test_model_context_window_and_history_budget_are_resolved_together(self):
        settings = AppSettings(
            workspace=Path("/workspace"),
            data_dir=Path("/workspace/var"),
            active_profile="default",
            trace_sink=Path("/workspace/var/traces/trace.jsonl"),
            default_provider="deepseek",
            context=ContextSettings(
                history_token_budget_ratio=0.5,
                response_token_reserve=8_000,
                safety_margin_tokens=7_000,
            ),
            model_context_windows={
                "default": 1_000_000,
                "small-model": 128_000,
            },
        )

        window = settings.model_context_tokens("small-model")

        self.assertEqual(window, 128_000)
        self.assertEqual(settings.context.resolved_history_token_budget(window), 56_500)

    def test_loads_custom_provider_single_model_context_and_capabilities(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config"
            config.mkdir()
            (config / "deepmate.yaml").write_text(
                "\n".join(
                    (
                        "provider:",
                        "  default: custom",
                    )
                ),
                encoding="utf-8",
            )
            (config / "providers.yaml").write_text(
                "\n".join(
                    (
                        "providers:",
                        "  custom:",
                        "    base_url: https://api.example.com/v1",
                        "    model: qwen-coder-plus",
                        "    api_key_env: CUSTOM_API_KEY",
                        "    context_window: 128000",
                    )
                ),
                encoding="utf-8",
            )

            settings = load_settings(root)

        provider = settings.provider("custom")
        self.assertEqual(provider.primary_model(), "qwen-coder-plus")
        self.assertEqual(provider.default_model, "qwen-coder-plus")
        self.assertEqual(provider.context_window, 128_000)
        self.assertEqual(settings.provider_context_tokens(provider), 128_000)
        self.assertTrue(provider.capabilities.supports_tools)
        self.assertFalse(provider.capabilities.supports_thinking)
        self.assertFalse(provider.capabilities.supports_stream_usage)
        self.assertFalse(provider.capabilities.supports_assistant_reasoning_replay)

    def test_custom_provider_missing_context_window_fails_loudly(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config"
            config.mkdir()
            (config / "deepmate.yaml").write_text(
                "provider:\n  default: custom\n",
                encoding="utf-8",
            )
            (config / "providers.yaml").write_text(
                "\n".join(
                    (
                        "providers:",
                        "  custom:",
                        "    base_url: https://api.example.com/v1",
                        "    model: qwen-coder-plus",
                        "    api_key_env: CUSTOM_API_KEY",
                    )
                ),
                encoding="utf-8",
            )

            settings = load_settings(root)

        with self.assertRaisesRegex(ValueError, "custom.*context_window"):
            settings.provider_context_tokens(settings.provider("custom"))

    def test_custom_provider_legacy_default_context_window_is_compatible(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config"
            config.mkdir()
            (config / "deepmate.yaml").write_text(
                "\n".join(
                    (
                        "provider:",
                        "  default: custom",
                        "model_context_windows:",
                        "  default: 64000",
                    )
                ),
                encoding="utf-8",
            )
            (config / "providers.yaml").write_text(
                "\n".join(
                    (
                        "providers:",
                        "  custom:",
                        "    base_url: https://api.example.com/v1",
                        "    default_model: legacy-model",
                    )
                ),
                encoding="utf-8",
            )

            settings = load_settings(root)

        provider = settings.provider("custom")
        self.assertEqual(provider.primary_model(), "legacy-model")
        self.assertEqual(settings.provider_context_tokens(provider), 64_000)

    def test_custom_provider_internal_purposes_follow_fallback_model(self):
        settings = AppSettings(
            workspace=Path("/workspace"),
            data_dir=Path("/workspace/var"),
            active_profile="default",
            trace_sink=Path("/workspace/var/traces/trace.jsonl"),
            default_provider="custom",
            model_purposes={
                "memory": ModelPurposeSettings(model="deepseek-v4-flash")
            },
        )

        config = resolve_model_purpose(
            settings,
            "memory",
            fallback_model="custom-model",
            provider="custom",
        )
        deepseek_config = resolve_model_purpose(
            settings,
            "memory",
            fallback_model="deepseek-v4-pro",
            provider="deepseek",
        )

        self.assertEqual(config.model, "custom-model")
        self.assertEqual(deepseek_config.model, "deepseek-v4-flash")

    def test_loads_otlp_trace_export_settings(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config"
            config.mkdir()
            (config / "deepmate.yaml").write_text(
                "\n".join(
                    (
                        "observability:",
                        "  otlp:",
                        "    endpoint: https://cloud.langfuse.com/api/public/otel",
                        "    headers: Authorization=Basic ${LANGFUSE_AUTH},X-Env=test",
                        "    service_name: deepmate-dev",
                        "    service_version: 0.1.0",
                    )
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"LANGFUSE_AUTH": "secret-token"}):
                settings = load_settings(root)

        self.assertEqual(
            settings.otlp_traces.endpoint,
            "https://cloud.langfuse.com/api/public/otel",
        )
        self.assertEqual(
            settings.otlp_traces.headers,
            (("Authorization", "Basic secret-token"), ("X-Env", "test")),
        )
        self.assertEqual(settings.otlp_traces.service_name, "deepmate-dev")
        self.assertEqual(settings.otlp_traces.service_version, "0.1.0")

    def test_load_settings_initializes_default_workspace_files(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            deepmate_home = root / "home"

            with patch.dict(os.environ, {"DEEPMATE_HOME": str(deepmate_home)}):
                settings = load_settings(root)

            self.assertEqual(settings.workspace, root.resolve())
            self.assertTrue((root / "config" / "deepmate.yaml").exists())
            self.assertTrue((root / "config" / "providers.yaml").exists())
            self.assertTrue(
                (deepmate_home / "profiles" / "default" / "identity.md").exists()
            )
            self.assertTrue(
                (deepmate_home / "profiles" / "default" / "soul.md").exists()
            )
            self.assertTrue(
                (deepmate_home / "profiles" / "default" / "user.md").exists()
            )
            self.assertTrue(
                (deepmate_home / "profiles" / "default" / "memory.md").exists()
            )
            self.assertFalse((root / "profiles" / "default" / "identity.md").exists())
            self.assertFalse((root / "profiles" / "default" / "soul.md").exists())
            self.assertFalse((root / "profiles" / "default" / "user.md").exists())
            self.assertTrue((root / "profiles" / "default" / "memory.md").exists())
            self.assertIn(LOCAL_PROVIDER_NAME, settings.providers)
            self.assertFalse(settings.provider("local").api_key_required)
            self.assertEqual(settings.provider("local").api_key(), "ollama")
            self.assertEqual(settings.model_context_tokens("qwen3-local"), 24_576)
            self.assertEqual(settings.model_context_tokens("qwen3:4b"), 24_576)
            self.assertEqual(
                (root / "profiles" / "default" / "memory.md").read_text(
                    encoding="utf-8"
                ),
                "",
            )
            self.assertEqual(settings.profile_ref().uri, "profiles/default")
            self.assertEqual(
                settings.profile_ref().global_uri,
                str((deepmate_home / "profiles" / "default").resolve()),
            )
            self.assertEqual(settings.profile_ref().project_uri, "profiles/default")

    def test_local_provider_is_available_for_existing_workspace_configs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config"
            config.mkdir()
            (config / "deepmate.yaml").write_text(
                "\n".join(
                    (
                        "provider:",
                        "  default: deepseek",
                        "model_context_windows:",
                        "  default: 1000000",
                    )
                ),
                encoding="utf-8",
            )
            (config / "providers.yaml").write_text(
                "\n".join(
                    (
                        "providers:",
                        "  deepseek:",
                        "    base_url: https://api.deepseek.com",
                        "    default_model: deepseek-v4-flash",
                    )
                ),
                encoding="utf-8",
            )

            settings = load_settings(root)

            self.assertIn("local", settings.providers)
            self.assertEqual(settings.provider("local").default_model, "qwen3-local")
            self.assertEqual(settings.provider("local").upgrade_model, "qwen3-coder-strong")
            self.assertEqual(settings.model_context_tokens("qwen3-local-balanced"), 32_768)
            self.assertEqual(settings.model_context_tokens("qwen3:8b"), 32_768)

    def test_model_context_window_accepts_quoted_model_names_with_colons(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config"
            config.mkdir()
            (config / "deepmate.yaml").write_text(
                "\n".join(
                    (
                        "model_context_windows:",
                        '  "qwen3:4b": 12345',
                    )
                ),
                encoding="utf-8",
            )
            (config / "providers.yaml").write_text(
                "providers:\n",
                encoding="utf-8",
            )

            settings = load_settings(root)

            self.assertEqual(settings.model_context_tokens("qwen3:4b"), 12_345)

    def test_load_settings_does_not_overwrite_existing_workspace_config(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "config"
            config_dir.mkdir()
            existing = "runtime:\n  active_profile: custom\n"
            (config_dir / "deepmate.yaml").write_text(existing, encoding="utf-8")

            settings = load_settings(root)

            self.assertEqual(settings.active_profile, "custom")
            self.assertEqual(
                (config_dir / "deepmate.yaml").read_text(encoding="utf-8"),
                existing,
            )

    def test_provider_api_key_reads_local_private_store_after_env(self):
        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "var"
            path = save_provider_api_key(data_dir, "STUB_API_KEY", "local-secret")

            self.assertEqual(path, provider_secret_path(data_dir))
            self.assertEqual(provider_api_key("STUB_API_KEY", data_dir), "local-secret")

            with patch.dict(os.environ, {"STUB_API_KEY": "env-secret"}):
                self.assertEqual(
                    provider_api_key("STUB_API_KEY", data_dir),
                    "env-secret",
                )

    def test_wecom_settings_read_local_private_store_after_env(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_wecom_remote_settings(
                root / "var",
                bot_id="bot-from-store",
                secret="secret-from-store",
                allowed_users="alice,bob",
                group_policy="full",
            )

            with patch.dict(
                os.environ,
                {
                    "DEEPMATE_WECOM_BOT_ID": "",
                    "DEEPMATE_WECOM_SECRET": "",
                    "DEEPMATE_WECOM_ALLOWED_USERS": "",
                    "DEEPMATE_WECOM_GROUP_POLICY": "",
                },
                clear=False,
            ):
                settings = load_settings(root)

        self.assertEqual(settings.remote.wecom.bot_id, "bot-from-store")
        self.assertEqual(settings.remote.wecom.secret, "secret-from-store")
        self.assertEqual(settings.remote.wecom.allowed_users, ("alice", "bob"))
        self.assertEqual(settings.remote.wecom.group_policy, "full")

    def test_loads_mcp_args_from_multiline_yaml_list(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "config"
            config_dir.mkdir()
            (config_dir / "deepmate.yaml").write_text(
                "\n".join(
                    (
                        "mcp_servers:",
                        "  filesystem:",
                        "    command: npx",
                        "    args:",
                        "      - -y",
                        "      - '@modelcontextprotocol/server-filesystem'",
                        "      - .",
                    )
                ),
                encoding="utf-8",
            )

            settings = load_settings(root)

            self.assertEqual(len(settings.mcp_servers), 1)
            self.assertEqual(
                settings.mcp_servers[0].args,
                ("-y", "@modelcontextprotocol/server-filesystem", "."),
            )

    def test_history_budget_is_capped_by_usable_input_window(self):
        context = ContextSettings(history_token_budget_ratio=1.2)

        self.assertEqual(context.usable_input_tokens(1_000_000), 886_000)
        self.assertEqual(context.resolved_history_token_budget(1_000_000), 886_000)

        context = ContextSettings(history_token_budget=950_000)

        self.assertEqual(context.resolved_history_token_budget(1_000_000), 886_000)

    def test_small_context_window_keeps_usable_input_room(self):
        context = ContextSettings()

        self.assertEqual(context.resolved_response_token_reserve(100_000), 6_400)
        self.assertEqual(context.resolved_safety_margin_tokens(100_000), 5_000)
        self.assertEqual(context.usable_input_tokens(100_000), 88_600)
        self.assertEqual(context.resolved_history_token_budget(100_000), 66_450)

    def test_zero_history_budget_ratio_resolves_to_minimum_budget(self):
        context = ContextSettings(history_token_budget_ratio=0)

        self.assertEqual(context.resolved_history_token_budget(1_000_000), 1)

    def test_hot_profile_budget_uses_model_window_ratio_with_guards(self):
        context = ContextSettings()

        self.assertEqual(
            context.resolved_hot_profile_token_budget(1_000_000),
            5_000,
        )
        self.assertEqual(context.hot_profile_warn_tokens(1_000_000), 4_000)

        small = ContextSettings(
            response_token_reserve=0,
            safety_margin_tokens=0,
        )
        self.assertEqual(small.resolved_hot_profile_token_budget(10_000), 800)

        large = ContextSettings()
        self.assertEqual(large.resolved_hot_profile_token_budget(4_000_000), 6_000)

    def test_loads_hot_profile_budget_settings(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            config_dir.mkdir()
            (config_dir / "deepmate.yaml").write_text(
                "\n".join(
                    (
                        "context:",
                        "  hot_profile_budget_ratio: 0.01",
                        "  hot_profile_warn_ratio: 0.7",
                        "  hot_profile_min_tokens: 500",
                        "  hot_profile_max_tokens: 7000",
                    )
                ),
                encoding="utf-8",
            )

            settings = load_settings(root)

        self.assertEqual(settings.context.hot_profile_budget_ratio, 0.01)
        self.assertEqual(settings.context.hot_profile_warn_ratio, 0.7)
        self.assertEqual(settings.context.hot_profile_min_tokens, 500)
        self.assertEqual(settings.context.hot_profile_max_tokens, 7000)

    def test_loads_subagent_budget_settings(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            config_dir.mkdir()
            (config_dir / "deepmate.yaml").write_text(
                "\n".join(
                    (
                        "subagents:",
                        "  max_child_runs: 5",
                        "  max_workspace_write_child_runs: 2",
                        "  max_revise_attempts: 2",
                        "  max_child_steps: 10",
                        "  revise_step_extension: 3",
                    )
                ),
                encoding="utf-8",
            )

            settings = load_settings(root)

        self.assertEqual(settings.subagents.max_child_runs, 5)
        self.assertEqual(settings.subagents.max_workspace_write_child_runs, 2)
        self.assertEqual(settings.subagents.max_revise_attempts, 2)
        self.assertEqual(settings.subagents.max_child_steps, 10)
        self.assertEqual(settings.subagents.revise_step_extension, 3)

    def test_loads_runtime_tool_repair_and_output_settings(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            config_dir.mkdir()
            (config_dir / "deepmate.yaml").write_text(
                "\n".join(
                    (
                        "runtime:",
                        "  loop_guard:",
                        "    enabled: false",
                        "    hard_step_cap: 77",
                        "  tool_repair:",
                        "    enabled: true",
                        "    reasoning_scavenge: false",
                        "    argument_repair: true",
                        "    max_identical_tool_calls: 3",
                        "    max_similar_tool_calls: 0",
                        "  tool_output:",
                        "    compaction_enabled: false",
                        "    lossless_normalization: true",
                        "    small_output_ratio: 0.004",
                        "    medium_output_ratio: 0.02",
                        "    huge_output_ratio: 0.05",
                        "    compact_target_ratio: 0.006",
                    )
                ),
                encoding="utf-8",
            )

            settings = load_settings(root)

        self.assertTrue(settings.tool_repair.enabled)
        self.assertFalse(settings.loop_guard.enabled)
        self.assertEqual(settings.loop_guard.hard_step_cap, 77)
        self.assertFalse(settings.tool_repair.reasoning_scavenge)
        self.assertTrue(settings.tool_repair.argument_repair)
        self.assertEqual(settings.tool_repair.max_identical_tool_calls, 3)
        self.assertEqual(settings.tool_repair.max_similar_tool_calls, 0)
        self.assertFalse(settings.tool_output.compaction_enabled)
        self.assertTrue(settings.tool_output.lossless_normalization)
        self.assertEqual(settings.tool_output.small_output_ratio, 0.004)
        self.assertEqual(settings.tool_output.medium_output_ratio, 0.02)
        self.assertEqual(settings.tool_output.huge_output_ratio, 0.05)
        self.assertEqual(settings.tool_output.compact_target_ratio, 0.006)

    def test_invalid_model_context_window_fails_loudly(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            config_dir.mkdir()
            (config_dir / "deepmate.yaml").write_text(
                "\n".join(
                    (
                        "model_context_windows:",
                        "  default: 1000000",
                        "  bad-model: one-million",
                    )
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "model_context_windows.bad-model",
            ):
                load_settings(root)

    def test_loads_remote_wecom_settings_and_expands_secret_env(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            config_dir.mkdir()
            (config_dir / "deepmate.yaml").write_text(
                "\n".join(
                    (
                        "remote:",
                        "  wecom:",
                        "    enabled: true",
                        "    bot_id: ww_test",
                        "    secret: ${WECOM_TEST_SECRET}",
                        "    allowed_users: alice,bob",
                        "    progress_heartbeat: true",
                        "    progress_intervals: 5m,10m,15m,30m,1h",
                        "runtime:",
                        "  wake:",
                        "    post_turn_grace: 90m",
                    )
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"WECOM_TEST_SECRET": "secret-value"}):
                settings = load_settings(root)

        self.assertTrue(settings.remote.wecom.enabled)
        self.assertEqual(settings.remote.wecom.bot_id, "ww_test")
        self.assertEqual(settings.remote.wecom.secret, "secret-value")
        self.assertEqual(settings.remote.wecom.allowed_users, ("alice", "bob"))
        self.assertTrue(settings.remote.wecom.progress_heartbeat)
        self.assertEqual(
            settings.remote.wecom.progress_intervals_seconds,
            (5 * 60, 10 * 60, 15 * 60, 30 * 60, 60 * 60),
        )
        self.assertEqual(settings.wake.post_turn_grace_minutes, 90)

    def test_loads_wecom_settings_from_environment_defaults(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch.dict(
                os.environ,
                {
                    "DEEPMATE_WECOM_BOT_ID": "ww_env",
                    "DEEPMATE_WECOM_SECRET": "secret-env",
                    "DEEPMATE_WECOM_ALLOWED_USERS": "alice,bob",
                    "DEEPMATE_WECOM_GROUP_POLICY": "full",
                },
            ):
                settings = load_settings(root)

        self.assertEqual(settings.remote.wecom.bot_id, "ww_env")
        self.assertEqual(settings.remote.wecom.secret, "secret-env")
        self.assertEqual(settings.remote.wecom.allowed_users, ("alice", "bob"))
        self.assertEqual(settings.remote.wecom.group_policy, "full")

    def test_rejects_wake_grace_longer_than_24h(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            config_dir.mkdir()
            (config_dir / "deepmate.yaml").write_text(
                "\n".join(
                    (
                        "runtime:",
                        "  wake:",
                        "    post_turn_grace: 25h",
                    )
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "24h or less"):
                load_settings(root)

    def test_loads_wake_grace_hour_duration(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            config_dir.mkdir()
            (config_dir / "deepmate.yaml").write_text(
                "\n".join(
                    (
                        "runtime:",
                        "  wake:",
                        "    post_turn_grace: 2h",
                    )
                ),
                encoding="utf-8",
            )

            settings = load_settings(root)

        self.assertEqual(settings.wake.post_turn_grace_minutes, 120)


if __name__ == "__main__":
    unittest.main()
