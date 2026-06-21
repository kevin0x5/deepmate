from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from deepmate.channels.cli import main
from deepmate.runtime.hooks import (
    HookActor,
    HookAction,
    HookActionStatus,
    HookActionType,
    HookDefinition,
    HookDirective,
    HookEnvelope,
    HookEvent,
    HookLayer,
    HookLoadOptions,
    HookManager,
    HookRegistry,
    HookRunTarget,
    HookRuntimeContext,
    HookSignalStore,
    HookTrustStore,
    builtin_hook_definitions,
    hook_matches,
    load_hook_report,
)


class HookKernelTests(unittest.TestCase):
    def test_loads_trusted_project_yaml_and_matches_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            workspace = root / "workspace"
            data_dir = workspace / "var"
            project_hooks = workspace / ".deepmate" / "hooks"
            project_hooks.mkdir(parents=True)
            (project_hooks / "protect.yaml").write_text(
                "\n".join(
                    (
                        "version: 1",
                        "hooks:",
                        "  - id: protect-sensitive-writes",
                        "    on: write.before",
                        "    when:",
                        "      path_globs:",
                        "        - '**/.env'",
                        "    actions:",
                        "      - type: deny",
                        "        reason: Sensitive path.",
                    )
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"HOME": str(home)}, clear=False):
                HookTrustStore.in_data_dir(data_dir).trust_workspace(workspace)
                report = load_hook_report(workspace, data_dir)

            self.assertFalse(report.has_errors())
            self.assertTrue(report.workspace_trusted)
            self.assertEqual(report.loaded_counts()["project"], 1)
            hook = next(
                hook
                for hook in report.registry.hooks
                if hook.hook_id == "protect-sensitive-writes"
            )
            self.assertTrue(
                hook_matches(
                    hook,
                    HookEnvelope(
                        event_name=HookEvent.WRITE_BEFORE,
                        actor=HookActor.MAIN,
                        payload={"path": "config/.env"},
                    ),
                )
            )
            outcome = HookManager(report.registry).emit(
                HookEnvelope(
                    event_name=HookEvent.WRITE_BEFORE,
                    actor=HookActor.MAIN,
                    payload={"path": "config/.env"},
                )
            )
            self.assertEqual(outcome.directive, HookDirective.BLOCK)
            self.assertTrue(
                any(
                    result.status == HookActionStatus.BLOCKED
                    for result in outcome.action_results
                )
            )

    def test_builtin_maintenance_hook_runs_on_maintenance_actor_only(self) -> None:
        registry = HookRegistry.from_hooks(builtin_hook_definitions())

        maintenance = HookManager(registry).emit(
            HookEnvelope(
                event_name=HookEvent.MAINTENANCE_AFTER_RUN,
                actor=HookActor.MAINTENANCE,
                payload={"summary": "done"},
            )
        )
        main = HookManager(registry).emit(
            HookEnvelope(
                event_name=HookEvent.MAINTENANCE_AFTER_RUN,
                actor=HookActor.MAIN,
                payload={"summary": "done"},
            )
        )

        self.assertEqual(maintenance.action_results[0].status, HookActionStatus.APPLIED)
        self.assertEqual(main.action_results, ())

    def test_untrusted_project_hook_is_skipped_but_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            project_hooks = workspace / ".deepmate" / "hooks"
            project_hooks.mkdir(parents=True)
            (project_hooks / "project.json").write_text(
                '{"version":1,"hooks":[{"id":"project-hook","on":"tool.before","actions":[{"type":"trace"}]}]}',
                encoding="utf-8",
            )
            report = load_hook_report(workspace, workspace / "var")

            self.assertFalse(report.workspace_trusted)
            self.assertEqual(report.loaded_counts()["project"], 0)
            self.assertEqual(report.skipped_counts.get("untrusted_project"), 1)
            self.assertTrue(any("untrusted_project" in item.message for item in report.diagnostics))

    def test_managed_only_skips_user_project_and_session_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            workspace = root / "workspace"
            data_dir = workspace / "var"
            for directory in (
                home / ".deepmate" / "hooks" / "user",
                workspace / ".deepmate" / "hooks",
            ):
                directory.mkdir(parents=True)
                (directory / "hook.json").write_text(
                    '{"version":1,"hooks":[{"id":"skip-me","on":"tool.before","actions":[{"type":"trace"}]}]}',
                    encoding="utf-8",
                )
            session_hook = root / "session.json"
            session_hook.write_text(
                '{"version":1,"hooks":[{"id":"session-hook","on":"tool.before","actions":[{"type":"trace"}]}]}',
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"HOME": str(home)}, clear=False):
                HookTrustStore.in_data_dir(data_dir).trust_workspace(workspace)
                report = load_hook_report(
                    workspace,
                    data_dir,
                    HookLoadOptions(managed_hooks_only=True),
                    session_hook_paths=(session_hook,),
                )

            self.assertEqual(report.loaded_counts()["user"], 0)
            self.assertEqual(report.loaded_counts()["project"], 0)
            self.assertEqual(report.loaded_counts()["session"], 0)
            self.assertEqual(report.skipped_counts.get("managed_only"), 3)

    def test_rejects_invalid_event_path_glob_and_high_risk_project_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            data_dir = workspace / "var"
            project_hooks = workspace / ".deepmate" / "hooks"
            project_hooks.mkdir(parents=True)
            (project_hooks / "bad.json").write_text(
                """
{
  "version": 1,
  "hooks": [
    {
      "id": "bad",
      "on": "tool.before",
      "when": {"path_globs": ["/absolute"]},
      "actions": [{"type": "run_shell", "command": "echo bad"}]
    },
    {
      "id": "bad-event",
      "on": "tool.beforeish",
      "actions": [{"type": "trace"}]
    }
  ]
}
""".strip(),
                encoding="utf-8",
            )
            HookTrustStore.in_data_dir(data_dir).trust_workspace(workspace)
            report = load_hook_report(workspace, data_dir)

            self.assertTrue(report.has_errors())
            messages = "\n".join(item.message for item in report.diagnostics)
            self.assertIn("path glob must be relative", messages)
            self.assertIn("high-risk action run_shell", messages)
            self.assertIn("known on event", messages)

    def test_unconnected_hook_action_warns_and_is_not_stored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            data_dir = workspace / "var"
            project_hooks = workspace / ".deepmate" / "hooks"
            project_hooks.mkdir(parents=True)
            (project_hooks / "compact.json").write_text(
                """
{
  "version": 1,
  "hooks": [
    {
      "id": "compact-soon",
      "on": "tool.after",
      "actions": [{"type": "compact", "target": "tool_result"}]
    }
  ]
}
""".strip(),
                encoding="utf-8",
            )
            HookTrustStore.in_data_dir(data_dir).trust_workspace(workspace)

            report = load_hook_report(workspace, data_dir)

            self.assertFalse(report.has_errors())
            messages = "\n".join(item.message for item in report.diagnostics)
            self.assertIn("not connected to runtime side effects yet", messages)
            hook = next(hook for hook in report.registry.hooks if hook.hook_id == "compact-soon")
            self.assertEqual(hook.actions, ())

    def test_cli_hooks_status_validate_and_trust_do_not_need_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir(parents=True)
            (workspace / "config").mkdir()
            (workspace / "config" / "deepmate.yaml").write_text(
                "\n".join(
                    (
                        "runtime:",
                        "  data_dir: var",
                        "provider:",
                        "  default: missing",
                    )
                ),
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status_code = main(("--workspace", str(workspace), "--hooks-status"))
            self.assertEqual(status_code, 0)
            self.assertIn("Hooks status:", stdout.getvalue())

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                validate_code = main(("--workspace", str(workspace), "--validate-hooks"))
            self.assertEqual(validate_code, 0)
            self.assertIn("Hook validation:", stdout.getvalue())

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                trust_code = main(("--workspace", str(workspace), "--trust-workspace"))
            self.assertEqual(trust_code, 0)
            self.assertIn("Workspace trusted for project hooks:", stdout.getvalue())

    def test_settings_parse_hook_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "config").mkdir()
            (workspace / "config" / "deepmate.yaml").write_text(
                "\n".join(
                    (
                        "hooks:",
                        "  enabled: false",
                        "  managed_hooks_only: true",
                        "  load_project_hooks: false",
                        "  load_user_hooks: false",
                        "  trace_matches: true",
                        "  before_timeout_ms: 10",
                        "  after_timeout_ms: 20",
                        "  maintenance_timeout_ms: 30",
                    )
                ),
                encoding="utf-8",
            )
            from deepmate.app import load_settings

            settings = load_settings(workspace)

            self.assertFalse(settings.hooks.enabled)
            self.assertTrue(settings.hooks.managed_hooks_only)
            self.assertFalse(settings.hooks.load_project_hooks)
            self.assertFalse(settings.hooks.load_user_hooks)
            self.assertTrue(settings.hooks.trace_matches)
            self.assertEqual(settings.hooks.before_timeout_ms, 10)
            self.assertEqual(settings.hooks.after_timeout_ms, 20)
            self.assertEqual(settings.hooks.maintenance_timeout_ms, 30)

    def test_manager_uses_configured_reason_and_skips_unwired_actions(self) -> None:
        registry = HookRegistry.from_hooks(
            (
                HookDefinition(
                    hook_id="deny-shell",
                    event_name=HookEvent.SHELL_BEFORE,
                    layer=HookLayer.SESSION,
                    actions=(
                        HookAction(
                            HookActionType.DENY,
                            {"reason": "custom deny reason"},
                        ),
                    ),
                ),
                HookDefinition(
                    hook_id="future-patch",
                    event_name=HookEvent.PROVIDER_AFTER_RESPONSE,
                    layer=HookLayer.SESSION,
                    actions=(HookAction(HookActionType.PATCH_TOOL_RESULT),),
                ),
            )
        )

        blocked = HookManager(registry).emit(
            HookEnvelope(
                event_name=HookEvent.SHELL_BEFORE,
                actor=HookActor.MAIN,
            )
        )
        skipped = HookManager(registry).emit(
            HookEnvelope(
                event_name=HookEvent.PROVIDER_AFTER_RESPONSE,
                actor=HookActor.MAIN,
            )
        )

        self.assertEqual(blocked.directive, HookDirective.BLOCK)
        self.assertEqual(blocked.reason, "custom deny reason")
        self.assertEqual(skipped.directive, HookDirective.CONTINUE)
        self.assertEqual(skipped.action_results[0].status, HookActionStatus.SKIPPED)
        self.assertIn("not connected", skipped.action_results[0].summary)

    def test_signal_store_bounds_and_loads_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HookSignalStore(Path(tmp) / "signals.jsonl")
            record = store.append(
                signal_type="memory",
                signal_kind="user_correction",
                summary="x" * 700,
                refs=tuple(f"ref-{index}-" + "y" * 400 for index in range(30)),
                hook_id="memory-hook",
                event_name="agent.turn_end",
                source_layer="project",
                source_actor="main",
                session_id="session_1",
            )

            loaded = store.load_recent()

        self.assertTrue(record.is_ready())
        self.assertLessEqual(len(record.summary), 500)
        self.assertEqual(len(record.refs), 20)
        self.assertLessEqual(max(len(ref) for ref in record.refs), 240)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].signal_type, "memory")
        self.assertEqual(loaded[0].hook_id, "memory-hook")

    def test_record_signal_action_writes_when_store_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HookSignalStore(Path(tmp) / "signals.jsonl")
            context = HookRuntimeContext.from_registry(
                HookRegistry.from_hooks(
                    (
                        HookDefinition(
                            hook_id="turn-signal",
                            event_name=HookEvent.AGENT_TURN_END,
                            layer=HookLayer.SESSION,
                            actions=(
                                HookAction(
                                    HookActionType.RECORD_EVOLUTION_SIGNAL,
                                    {
                                        "signal_kind": "turn_outcome",
                                        "summary": "Turn completed with useful evidence.",
                                        "refs": ["ref=from_action"],
                                    },
                                ),
                            ),
                        ),
                    )
                ),
                signal_store=store,
            )

            outcome = context.emit(
                HookEnvelope(
                    event_name=HookEvent.AGENT_TURN_END,
                    actor=HookActor.MAIN,
                    payload={"status": "completed"},
                    session_id="session_1",
                    source_refs=("ref=from_envelope",),
                )
            )
            loaded = store.load_recent()

        self.assertEqual(outcome.directive, HookDirective.CONTINUE)
        self.assertEqual(outcome.action_results[0].status, HookActionStatus.APPLIED)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].signal_type, "evolution")
        self.assertEqual(loaded[0].signal_kind, "turn_outcome")
        self.assertEqual(loaded[0].session_id, "session_1")
        self.assertIn("ref=from_action", loaded[0].refs)
        self.assertIn("ref=from_envelope", loaded[0].refs)

    def test_record_signal_action_skips_without_store(self) -> None:
        context = HookRuntimeContext.from_registry(
            HookRegistry.from_hooks(
                (
                    HookDefinition(
                        hook_id="memory-signal",
                        event_name=HookEvent.CHECKPOINT_CREATED,
                        layer=HookLayer.SESSION,
                        run_on=HookRunTarget.MAINTENANCE,
                        actions=(HookAction(HookActionType.RECORD_MEMORY_SIGNAL),),
                    ),
                )
            )
        )

        outcome = context.emit(
            HookEnvelope(
                event_name=HookEvent.CHECKPOINT_CREATED,
                actor=HookActor.MAINTENANCE,
                payload={"summary": "checkpoint completed"},
            )
        )

        self.assertEqual(outcome.action_results[0].status, HookActionStatus.SKIPPED)
        self.assertIn("signal store is not configured", outcome.action_results[0].summary)

    def test_trusted_project_signal_hook_writes_and_untrusted_does_not_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            data_dir = workspace / "var"
            hook_dir = workspace / ".deepmate" / "hooks"
            hook_dir.mkdir(parents=True)
            (hook_dir / "signals.yaml").write_text(
                "\n".join(
                    (
                        "version: 1",
                        "hooks:",
                        "  - id: project-turn-signal",
                        "    on: agent.turn_end",
                        "    actions:",
                        "      - type: record_evolution_signal",
                        "        signal_kind: workflow_hint",
                        "        summary: Project hook captured a turn signal.",
                    )
                ),
                encoding="utf-8",
            )
            store = HookSignalStore(data_dir / "hooks" / "signals.jsonl")

            untrusted = load_hook_report(workspace, data_dir)
            HookRuntimeContext.from_registry(
                untrusted.registry,
                signal_store=store,
            ).emit(HookEnvelope(event_name=HookEvent.AGENT_TURN_END))
            HookTrustStore.in_data_dir(data_dir).trust_workspace(workspace)
            trusted = load_hook_report(workspace, data_dir)
            HookRuntimeContext.from_registry(
                trusted.registry,
                signal_store=store,
            ).emit(
                HookEnvelope(
                    event_name=HookEvent.AGENT_TURN_END,
                    payload={"status": "completed"},
                )
            )

            loaded = store.load_recent()

        self.assertEqual(untrusted.loaded_counts()["project"], 0)
        self.assertEqual(trusted.loaded_counts()["project"], 1)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].hook_id, "project-turn-signal")

    def test_runtime_context_reload_registry_preserves_signal_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HookSignalStore(Path(tmp) / "signals.jsonl")
            context = HookRuntimeContext.from_registry(
                HookRegistry.from_hooks(()),
                signal_store=store,
            )
            context.reload_registry(
                HookRegistry.from_hooks(
                    (
                        HookDefinition(
                            hook_id="reloaded-signal",
                            event_name=HookEvent.AGENT_TURN_END,
                            layer=HookLayer.SESSION,
                            actions=(
                                HookAction(
                                    HookActionType.RECORD_EVOLUTION_SIGNAL,
                                    {"summary": "Reloaded hook is active."},
                                ),
                            ),
                        ),
                    )
                )
            )

            context.emit(HookEnvelope(event_name=HookEvent.AGENT_TURN_END))
            loaded = store.load_recent()

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].hook_id, "reloaded-signal")


if __name__ == "__main__":
    unittest.main()
