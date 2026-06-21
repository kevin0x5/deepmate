from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from deepmate.channels.interactive import _handle_hooks_command
from deepmate.domain import ProfileRef
from deepmate.runtime import (
    HookEnvelope,
    HookEvent,
    HookLoadOptions,
    HookRuntimeContext,
    HookSignalStore,
    start_runtime_activation,
    start_session_runtime,
)
from deepmate.runtime.hooks import HookRegistry
from deepmate.storage import SessionStore
from deepmate.trace import TraceRecorder


class HookInteractiveCommandTests(unittest.TestCase):
    def test_hooks_trust_reload_updates_context_and_project_signal_hook_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            data_dir = workspace / "var"
            profile_dir = workspace / "profiles" / "default"
            hook_dir = workspace / ".deepmate" / "hooks"
            profile_dir.mkdir(parents=True)
            hook_dir.mkdir(parents=True)
            (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
            (profile_dir / "identity.md").write_text("Identity.", encoding="utf-8")
            (profile_dir / "soul.md").write_text("Style.", encoding="utf-8")
            (hook_dir / "turn.yaml").write_text(
                "\n".join(
                    (
                        "version: 1",
                        "hooks:",
                        "  - id: interactive-project-signal",
                        "    on: agent.turn_end",
                        "    actions:",
                        "      - type: record_evolution_signal",
                        "        signal_kind: interactive_reload",
                        "        summary: Project hook loaded after trust and reload.",
                    )
                ),
                encoding="utf-8",
            )
            session_store = SessionStore.in_directory(data_dir / "sessions")
            profile = ProfileRef(name="default", uri="profiles/default")
            session = session_store.create(
                workspace=workspace,
                profile=profile,
                title="hooks interactive",
            )
            runtime = start_session_runtime(
                start_runtime_activation(
                    session_id=session.session_id,
                    workspace=workspace,
                    profile=profile,
                )
            )
            context = HookRuntimeContext.from_registry(
                HookRegistry.from_hooks(()),
                signal_store=HookSignalStore.in_data_dir(data_dir),
            )
            trace_recorder = TraceRecorder(_TraceSink())

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                _handle_hooks_command(
                    prompt="/hooks trust",
                    workspace=workspace,
                    data_dir=data_dir,
                    hook_context=context,
                    hook_load_options=HookLoadOptions(),
                    trace_recorder=trace_recorder,
                    session=session,
                    runtime=runtime,
                )
                _handle_hooks_command(
                    prompt="/hooks reload",
                    workspace=workspace,
                    data_dir=data_dir,
                    hook_context=context,
                    hook_load_options=HookLoadOptions(),
                    trace_recorder=trace_recorder,
                    session=session,
                    runtime=runtime,
                )

            context.emit(HookEnvelope(event_name=HookEvent.AGENT_TURN_END))
            signals = HookSignalStore.in_data_dir(data_dir).load_recent()

        self.assertIn("Workspace trusted for project hooks:", stdout.getvalue())
        self.assertIn("- reloaded: true", stdout.getvalue())
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].hook_id, "interactive-project-signal")


class _TraceSink:
    def __init__(self) -> None:
        self.events = []

    def write(self, event) -> None:
        self.events.append(event)


if __name__ == "__main__":
    unittest.main()
