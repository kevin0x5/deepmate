from __future__ import annotations

import io
import os
import signal
import subprocess
import sys
import tempfile
import unittest
import asyncio
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from time import monotonic
from unittest.mock import patch

from rich.markdown import Markdown as RichMarkdown
from rich.padding import Padding
from rich.table import Table
from rich.text import Text
from textual import events
from textual.geometry import Offset
from textual.selection import Selection

from deepmate.behavior import behavior_runtime_for_session
from deepmate.channels.cli import (
    _argv_with_workspace,
    _default_tool_schemas_for_model,
    _hide_native_tool_schemas,
    _schemas_with_local_prompt_extras,
    main,
)
from deepmate.capabilities import CapabilityStateStore
from deepmate.channels.checkpointing import SessionCheckpointController
from deepmate.channels.remote import RemoteBindingStore
from deepmate.channels.tui.app import (
    DeepmateTuiApp,
    WORKSPACE_SWITCH_EXIT_CODE,
    _command_hint_name,
    _directory_input_path,
    _file_nav_label,
    _find_in_content_preview,
    _has_terminal_error,
    _is_immediate_command,
    _approval_body,
    _approval_diff_renderable,
    _approval_input_result,
    _approval_result_body,
    _compact_workspace_label,
    _context_window_color,
    _insert_prompt_text,
    _messages_from_transcript_items,
    _OpenTab,
    _pet_start_command,
    _PromptTextArea,
    _prompt_too_long_message,
    _preview_tab_content,
    _readable_markdown,
    _render_message,
    _session_button_id,
    _session_id_from_button_id,
    _sessions_preview,
    _SelectableRichLog,
    _short_session_title,
    _TurnRun,
    _turn_anchor_id,
    _transcript_has_items,
    _welcome_splash,
)
from deepmate.channels.tui.bridge import (
    _result_installed_skill,
    _tui_status_sink,
    run_tui_mode,
    run_headless_tui_turn,
)
from deepmate.channels.tui.commands import (
    apply_local_model_prepare_result,
    handle_tui_command,
)
from deepmate.channels.tui.formatters import (
    TuiMessage,
    friendly_error_message,
    tool_exchange_messages,
)
from deepmate.channels.tui.files import (
    GIT_STATUS_CACHE_MAX_WORKSPACES,
    _GIT_STATUS_CACHE,
    git_status_badges,
    read_workspace_file,
    read_workspace_file_preview,
    workspace_file_matches,
    workspace_diff,
    workspace_file_items,
)
from deepmate.channels.tui.formatters import result_messages
from deepmate.channels.tui.status import TuiRuntimeStats, status_view
from deepmate.channels.tui.state import (
    LocalModelPrepareRequest,
    TuiPromptQueue,
    TuiRuntimeState,
    WorkspaceSwitchRequest,
)
from deepmate.context import build_profile_context_snapshot
from deepmate.domain import ErrorInfo, Message, MessageRole, ProfileRef, RuntimeEvent
from deepmate.local import (
    LocalModelInstallResult,
    LocalModelStateStore,
    local_model_by_id,
    local_model_by_runtime_name,
)
from deepmate.providers import (
    ModelCapabilities,
    ModelConversationItem,
    ModelResponse,
    ModelToolResult,
    ModelToolExchange,
    ModelToolRequest,
    NetworkError,
)
from deepmate.qa import handle_qa_command
from deepmate.pet.state import PetStateStore
from deepmate.runtime import (
    AgentStepResult,
    ApprovalDecision,
    ContinuationNote,
    ConversationBudgetPolicy,
    LoopGuardStop,
    LoopGuardStopReason,
    ToolAccessDecision,
    ToolAccessMode,
    ToolAccessPolicy,
    ProviderRetryPolicy,
    TurnFollowupBuffer,
    UserTurnResult,
    start_runtime_activation,
    start_session_runtime,
)
from deepmate.runtime.sandbox import SandboxMode
from deepmate.runtime.safety import SessionApprovalCache, ToolSafetyPolicy
from deepmate.storage import SessionStore
from deepmate.tasks import (
    TaskSessionController,
    TaskStage,
    apply_task_update_result,
    generate_task_update,
    render_task_context_section,
)
from deepmate.tasks.execute import (
    ExecuteDecision,
    ExecuteEvaluation,
    ExecuteLoopUpdate,
    continuation_prompt,
)
from deepmate.tools import (
    AgentBrowserBackend,
    BrowserCommandResult,
    INSTALL_BROWSER_BACKEND_TOOL_NAME,
    INSTALL_SKILL_FROM_REQUEST_TOOL_NAME,
    INSTALL_SKILL_BUNDLE_TOOL_NAME,
    INSTALL_SKILL_TOOL_NAME,
    LOAD_SKILL_INSTALLER_TOOLS_NAME,
    LOAD_BROWSER_TOOLS_NAME,
    NativeTool,
    NativeToolRegistry,
    NativeToolResult,
    RUN_SHELL_COMMAND_TOOL_NAME,
    browser_loader_tools,
    shell_tools,
    skill_installer_tools,
    workspace_artifact_tools,
    workspace_diagram_tools,
    workspace_document_tools,
    workspace_filesystem_tools,
    workspace_lsp_tools,
    workspace_report_tools,
    workspace_search_tools,
)


class TuiChannelTests(unittest.TestCase):
    def test_result_installed_skill_detects_install_from_request(self) -> None:
        # The model-facing installer in interactive mode is
        # install_skill_from_request; a successful call must trigger a refresh.
        for tool_name in (
            "install_skill",
            "install_skill_bundle",
            "install_skill_from_request",
        ):
            result = UserTurnResult(
                steps=(
                    AgentStepResult(
                        request=None,
                        response=ModelResponse(content="done"),
                        tool_results=(
                            _tool_result(
                                name=tool_name,
                                request_id="call_1",
                                content="Skill installed: repo-review",
                                refs=("skill=repo-review",),
                            ),
                        ),
                    ),
                ),
            )
            self.assertTrue(_result_installed_skill(result), tool_name)

    def test_result_installed_skill_ignores_errors_and_other_tools(self) -> None:
        failed = UserTurnResult(
            steps=(
                AgentStepResult(
                    request=None,
                    response=ModelResponse(content="done"),
                    tool_results=(
                        _tool_result(
                            name="install_skill_from_request",
                            request_id="call_1",
                            content="boom",
                            refs=("skill=repo-review",),
                            is_error=True,
                        ),
                        _tool_result(
                            name="read_text_file",
                            request_id="call_2",
                            content="ok",
                        ),
                    ),
                ),
            ),
        )
        self.assertFalse(_result_installed_skill(failed))

    def test_result_messages_render_tool_cards_and_final_answer(self) -> None:
        result = UserTurnResult(
            steps=(
                AgentStepResult(
                    request=None,
                    response=ModelResponse(
                        reasoning="Need to inspect files.",
                        tool_requests=(
                            ModelToolRequest(
                                name="browser_open",
                                id="call_1",
                                arguments={"url": "https://example.test"},
                            ),
                        ),
                    ),
                    tool_results=(
                        _tool_result(
                            name="browser_open",
                            request_id="call_1",
                            content="Opened page.",
                        ),
                    ),
                ),
                AgentStepResult(
                    request=None,
                    response=ModelResponse(content="Done."),
                ),
            ),
        )

        messages = result_messages(result)

        self.assertEqual(messages[0].kind, "tool browser")
        self.assertEqual(messages[0].status, "ok")
        self.assertIn("Opened page.", messages[0].body)
        self.assertNotIn("url=https://example.test", messages[0].body)
        self.assertEqual(messages[0].preview, "Opened page.")
        self.assertEqual(messages[-1].kind, "assistant")
        self.assertEqual(messages[-1].body, "Done.")

        with_reasoning = result_messages(result, show_reasoning=True)
        self.assertEqual(with_reasoning[0].kind, "thinking")

    def test_result_messages_surface_compacted_tool_output_refs(self) -> None:
        result = UserTurnResult(
            steps=(
                AgentStepResult(
                    request=None,
                    response=ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="run_shell_command",
                                id="call_1",
                                arguments={"command": "pytest"},
                            ),
                        ),
                    ),
                    tool_results=(
                        _tool_result(
                            name="run_shell_command",
                            request_id="call_1",
                            content="compact view",
                            refs=(
                                "tool_output_compacted=true",
                                "tool_output_ref=tool-output:abc",
                                "original_estimated_tokens=9000",
                                "compacted_estimated_tokens=800",
                            ),
                        ),
                    ),
                ),
                AgentStepResult(request=None, response=ModelResponse(content="Done.")),
            ),
        )

        messages = result_messages(result)

        self.assertEqual(messages[0].kind, "tool shell")
        self.assertEqual(messages[0].status, "compacted")
        self.assertNotIn("command=pytest", messages[0].body)
        self.assertIn("Raw output handle: tool-output:abc", messages[0].body)
        self.assertIn("Output folded for context", messages[0].body)
        self.assertIn("compact view", messages[0].preview)
        self.assertNotIn("original=9000", messages[0].body)

    def test_result_messages_suppress_redundant_tool_completed_event(self) -> None:
        result = UserTurnResult(
            steps=(
                AgentStepResult(
                    request=None,
                    response=ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="browser_status",
                                id="call_1",
                                arguments={},
                            ),
                        ),
                    ),
                    tool_results=(
                        _tool_result(
                            name="browser_status",
                            request_id="call_1",
                            content="Browser backend is not available.",
                        ),
                    ),
                    events=(
                        RuntimeEvent(
                            kind="native_tool_completed",
                            summary="Native tool completed: browser_status.",
                            refs=("browser_status",),
                        ),
                    ),
                ),
            ),
        )

        messages = result_messages(result)

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].kind, "tool browser")

    def test_result_messages_fold_long_tool_output_but_keep_detail(self) -> None:
        long_output = "line 1\n" + ("0123456789" * 80)
        result = UserTurnResult(
            steps=(
                AgentStepResult(
                    request=None,
                    response=ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="run_shell_command",
                                id="call_1",
                                arguments={"command": "pytest -vv"},
                            ),
                        ),
                    ),
                    tool_results=(
                        _tool_result(
                            name="run_shell_command",
                            request_id="call_1",
                            content=long_output,
                        ),
                    ),
                ),
                AgentStepResult(request=None, response=ModelResponse(content="Done.")),
            ),
        )

        messages = result_messages(result)

        self.assertEqual(messages[0].kind, "tool shell")
        self.assertIn("Completed.", messages[0].body)
        self.assertNotIn("01234567890123456789", messages[0].body)
        self.assertIn("01234567890123456789", messages[0].preview)

    def test_tool_error_body_stays_compact_with_detail_escape_hatch(self) -> None:
        exchange = ModelToolExchange(
            tool_requests=(
                ModelToolRequest(
                    name="run_shell_command",
                    id="call_1",
                    arguments={"command": "pytest"},
                ),
            ),
            tool_results=(
                _tool_result(
                    name="run_shell_command",
                    request_id="call_1",
                    content="error line " * 120,
                    is_error=True,
                ),
            ),
        )

        messages = tool_exchange_messages(exchange)

        self.assertEqual(messages[0].status, "error")
        self.assertLess(len(messages[0].body), 260)
        self.assertIn("Use /detail to view full output.", messages[0].body)
        self.assertIn("error line error line", messages[0].preview)

    def test_result_messages_do_not_duplicate_native_tool_failure_errors(self) -> None:
        result = UserTurnResult(
            steps=(
                AgentStepResult(
                    request=None,
                    response=ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="read_text_file",
                                id="call_1",
                                arguments={"path": "missing.py"},
                            ),
                        ),
                    ),
                    tool_results=(
                        _tool_result(
                            name="read_text_file",
                            request_id="call_1",
                            content="Native tool failed: read_text_file: path is not a file: missing.py",
                            is_error=True,
                        ),
                    ),
                    errors=(
                        ErrorInfo(
                            code="native_tool_failed",
                            message=(
                                "Native tool failed: read_text_file: "
                                "path is not a file: missing.py"
                            ),
                        ),
                    ),
                ),
            ),
        )

        messages = result_messages(result)

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].kind, "tool read")
        self.assertNotEqual(messages[0].kind, "error")

    def test_result_messages_fold_short_noisy_tool_failures_when_final_answer_exists(self) -> None:
        result = UserTurnResult(
            steps=(
                AgentStepResult(
                    request=None,
                    response=ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="install_skill",
                                id="call_1",
                                arguments={"source": "missing"},
                            ),
                            ModelToolRequest(
                                name="load_skill",
                                id="call_2",
                                arguments={"name": "writer"},
                            ),
                        ),
                    ),
                    tool_results=(
                        _tool_result(
                            name="install_skill",
                            request_id="call_1",
                            content="skill candidate not found: writer",
                            is_error=True,
                        ),
                        _tool_result(
                            name="load_skill",
                            request_id="call_2",
                            content="skill not found: writer",
                            is_error=True,
                        ),
                    ),
                ),
                AgentStepResult(request=None, response=ModelResponse(content="Skill installed.")),
            ),
        )

        messages = result_messages(result)

        self.assertEqual(messages[0].kind, "tool summary")
        self.assertIn("Ran 2 tools.", messages[0].body)
        self.assertIn("final answer was produced", messages[0].body)
        self.assertNotIn("First failures", messages[0].body)
        self.assertEqual(messages[-1].kind, "assistant")
        self.assertEqual(
            [
                message.kind
                for message in messages
                if message.kind.startswith("tool ") and message.kind != "tool summary"
            ],
            [],
        )

    def test_result_messages_clean_session_file_errors(self) -> None:
        result = UserTurnResult(
            steps=(
                AgentStepResult(
                    request=None,
                    response=ModelResponse(content=""),
                    errors=(
                        ErrorInfo(
                            code="runtime_failed",
                            message=(
                                "[Errno 2] No such file or directory: "
                                "'/Users/me/project/var/sessions/abc.json'"
                            ),
                        ),
                    ),
                ),
            ),
        )

        messages = result_messages(result)

        self.assertEqual(messages[0].title, "runtime_failed")
        self.assertIn("current session state", messages[0].body)
        self.assertNotIn("/Users/me", messages[0].body)
        self.assertNotIn("abc.json", messages[0].body)

    def test_result_messages_collapse_provider_timeout_failure(self) -> None:
        failure = (
            "Model request failed after retry attempts: "
            "model request timed out while waiting for response data"
        )
        result = UserTurnResult(
            steps=(
                AgentStepResult(
                    request=None,
                    response=ModelResponse(content=failure),
                    events=(
                        RuntimeEvent(
                            kind="provider_request_failed",
                            summary=failure,
                        ),
                    ),
                    errors=(
                        ErrorInfo(
                            code="provider_request_failed",
                            message=failure,
                        ),
                    ),
                ),
            ),
        )

        messages = result_messages(result)

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].kind, "error")
        self.assertEqual(messages[0].title, "provider_request_failed")
        self.assertIn("model connection timed out", messages[0].body.lower())
        self.assertNotIn("Model request failed after retry attempts", messages[0].body)

    def test_result_messages_fold_intermediate_tool_failures_when_final_answer_exists(self) -> None:
        steps = tuple(
            AgentStepResult(
                request=None,
                response=ModelResponse(
                    tool_requests=(
                        ModelToolRequest(
                            name="run_shell_command",
                            id=f"call_{index}",
                            arguments={"command": "install"},
                        ),
                    ),
                ),
                tool_results=(
                    _tool_result(
                        name="run_shell_command",
                        request_id=f"call_{index}",
                        content=(
                            "Remote script piped directly to shell is not allowed"
                            if index < 4
                            else "installed ok"
                        ),
                        is_error=index < 4,
                    ),
                ),
            )
            for index in range(9)
        ) + (AgentStepResult(request=None, response=ModelResponse(content="已完成。")),)
        result = UserTurnResult(steps=steps)

        messages = result_messages(result)
        summary = messages[0]

        self.assertEqual(summary.kind, "tool summary")
        self.assertIn("final answer was produced", summary.body)
        self.assertNotIn("First failures", summary.body)
        self.assertEqual(messages[-1].kind, "assistant")

    def test_result_messages_summarize_noisy_tool_runs(self) -> None:
        steps = tuple(
            AgentStepResult(
                request=None,
                response=ModelResponse(
                    tool_requests=(
                        ModelToolRequest(
                            name="web_fetch" if index % 2 else "install_skill",
                            id=f"call_{index}",
                            arguments={"url": f"https://example.test/{index}"},
                        ),
                    ),
                ),
                tool_results=(
                    _tool_result(
                        name="web_fetch" if index % 2 else "install_skill",
                        request_id=f"call_{index}",
                        content=(
                            "Native tool failed: web_fetch: HTTP Error 404: Not Found"
                            if index % 2
                            else "Native tool failed: install_skill: skill candidate not found: ontology"
                        ),
                        is_error=True,
                    ),
                ),
            )
            for index in range(12)
        )
        result = UserTurnResult(steps=steps, reached_max_steps=True)

        messages = result_messages(result)
        tool_messages = [
            message
            for message in messages
            if message.kind.startswith("tool ") and message.kind != "tool summary"
        ]
        summary = messages[0]

        self.assertEqual(tool_messages, [])
        self.assertEqual(summary.kind, "tool summary")
        self.assertIn("Ran 12 tools.", summary.body)
        self.assertIn("12 failed.", summary.body)
        self.assertIn("install_skill x6", summary.body)
        self.assertIn("Use /detail", summary.body)
        self.assertIn("skill candidate not found: ontology", summary.body)
        self.assertIn("web_fetch [error]", summary.preview)
        self.assertLess(len(messages), 5)

    def test_reasoning_only_response_is_not_final_answer(self) -> None:
        result = UserTurnResult(
            steps=(AgentStepResult(request=None, response=ModelResponse(reasoning="private reasoning")),)
        )

        messages = result_messages(result, show_reasoning=False)
        with_reasoning = result_messages(result, show_reasoning=True)

        self.assertEqual(messages[0].kind, "warning")
        self.assertIn("no final answer", messages[0].title)
        self.assertEqual(with_reasoning[0].kind, "thinking")
        self.assertNotEqual(with_reasoning[0].kind, "assistant")

    def test_result_messages_surface_loop_guard_stop_as_pause(self) -> None:
        stop = LoopGuardStop(
            reason=LoopGuardStopReason.HARD_STEP_CAP,
            message="Deepmate reached the safety step cap.",
            continuation_note=ContinuationNote(
                stop_reason=LoopGuardStopReason.HARD_STEP_CAP,
                content="Stop reason: hard_step_cap",
            ),
        )
        result = UserTurnResult(
            steps=(AgentStepResult(request=None, response=ModelResponse(content="Partial.")),),
            reached_max_steps=True,
            loop_guard_stop=stop,
        )

        messages = result_messages(result)

        self.assertEqual(messages[-1].kind, "warning")
        self.assertEqual(messages[-1].title, "turn paused")
        self.assertIn("safety step cap", messages[-1].body)
        self.assertIn("continue", messages[-1].body)

    def test_friendly_error_message_hides_checkpoint_internals(self) -> None:
        message = friendly_error_message(ValueError("turn checkpoint not found: turn_00002"))

        self.assertEqual(message.kind, "error")
        self.assertEqual(message.title, "session state")
        self.assertNotIn("checkpoint", message.body.lower())
        self.assertNotIn("turn_00002", message.body)
        self.assertNotIn("ValueError", message.title)
        self.assertIn("new message", message.body)

    def test_workspace_file_items_and_preview_are_workspace_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "src").mkdir()
            (workspace / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
            (workspace / "var").mkdir()
            (workspace / "var" / "trace.jsonl").write_text("{}", encoding="utf-8")

            items = workspace_file_items(workspace)
            expanded_items = workspace_file_items(workspace, expanded_dirs=("src",))
            preview = read_workspace_file(workspace, "src/app.py")

            self.assertEqual(
                tuple(item.relative_path for item in items),
                ("src/",),
            )
            self.assertEqual(
                tuple(item.relative_path for item in expanded_items),
                ("src/", "src/app.py"),
            )
            self.assertTrue(items[0].is_dir)
            self.assertIn("print('ok')", preview)
            with self.assertRaises(ValueError):
                read_workspace_file(workspace, "../outside.txt")

    def test_workspace_file_preview_supports_offset_continuation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "log.txt").write_text("0123456789abcdef", encoding="utf-8")

            first = read_workspace_file_preview(workspace, "log.txt", max_bytes=6)
            second = read_workspace_file_preview(
                workspace,
                "log.txt",
                offset=first.end,
                max_bytes=6,
            )

            self.assertEqual(first.content, "012345")
            self.assertTrue(first.truncated_after)
            self.assertIn("/open log.txt --offset 6", first.rendered_content())
            self.assertEqual(second.content, "6789ab")
            self.assertTrue(second.truncated_before)
            self.assertTrue(second.truncated_after)

    def test_workspace_file_items_marks_truncated_sidebar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            for index in range(5):
                (workspace / f"file{index}.txt").write_text("x", encoding="utf-8")

            items = workspace_file_items(workspace, limit=3)
            paths = tuple(item.relative_path for item in items)

            self.assertEqual(len(items), 4)
            self.assertTrue(paths[-1].startswith("... more files; use /files"))

    def test_workspace_file_preview_blocks_protected_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / ".env").write_text("SECRET=abc", encoding="utf-8")
            (workspace / "id_rsa.pem").write_text("-----BEGIN-----", encoding="utf-8")
            (workspace / ".ssh").mkdir()
            (workspace / ".ssh" / "config").write_text("Host x", encoding="utf-8")
            (workspace / "readme.md").write_text("# ok", encoding="utf-8")

            for protected in (".env", "id_rsa.pem", ".ssh/config"):
                with self.assertRaises(ValueError):
                    read_workspace_file_preview(workspace, protected)

            # Non-sensitive files are still previewable.
            self.assertEqual(
                read_workspace_file_preview(workspace, "readme.md").content,
                "# ok",
            )

    def test_workspace_file_items_prune_ignored_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "src").mkdir()
            (workspace / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
            (workspace / "node_modules" / "pkg").mkdir(parents=True)
            (workspace / "skillhub-cli" / "bin").mkdir(parents=True)
            (workspace / "skillhub_install").mkdir()
            (workspace / "skillhub_install.tar.gz").write_text("archive", encoding="utf-8")
            (workspace / "node_modules" / "pkg" / "index.js").write_text(
                "module.exports = 1\n",
                encoding="utf-8",
            )
            (workspace / "tests").mkdir()
            (workspace / "tests" / "test_app.py").write_text("def test_ok(): pass\n", encoding="utf-8")

            items = workspace_file_items(workspace)
            expanded_items = workspace_file_items(workspace, expanded_dirs=("src", "tests"))

            self.assertEqual(
                tuple(item.relative_path for item in items),
                ("src/", "tests/"),
            )
            self.assertNotIn("skillhub-cli/", tuple(item.relative_path for item in items))
            self.assertNotIn("skillhub_install/", tuple(item.relative_path for item in items))
            self.assertNotIn("skillhub_install.tar.gz", tuple(item.relative_path for item in items))
            self.assertEqual(
                tuple(item.relative_path for item in expanded_items),
                ("src/", "src/app.py", "tests/", "tests/test_app.py"),
            )

    def test_workspace_file_items_stay_project_oriented(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "README.md").write_text("# Project\n", encoding="utf-8")
            (workspace / "src").mkdir()
            (workspace / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
            internal = workspace / "src" / "deepmate" / "channels" / "tui"
            internal.mkdir(parents=True)
            (internal / "app.py").write_text("internal\n", encoding="utf-8")
            deep = workspace / "packages" / "web" / "src"
            deep.mkdir(parents=True)
            (deep / "component.tsx").write_text("export {}\n", encoding="utf-8")

            items = workspace_file_items(workspace)
            expanded_src_items = workspace_file_items(workspace, expanded_dirs=("src",))
            paths = tuple(item.relative_path for item in items)
            expanded_src_paths = tuple(item.relative_path for item in expanded_src_items)

            self.assertIn("README.md", paths)
            self.assertIn("src/", paths)
            self.assertNotIn("src/app.py", paths)
            self.assertIn("src/app.py", expanded_src_paths)
            self.assertIn("src/deepmate/", expanded_src_paths)
            self.assertNotIn("src/deepmate/channels/tui/app.py", expanded_src_paths)
            self.assertNotIn("packages/web/src/component.tsx", expanded_src_paths)

    def test_file_nav_label_styles_git_status_badge(self) -> None:
        label = _file_nav_label("deepmate.html", "N", False)

        self.assertEqual(label.plain, "· deepmate.html N")
        self.assertTrue(
            any(
                label.plain[span.start : span.end] == " N"
                and "cdbb7a" in str(span.style)
                for span in label.spans
            )
        )

    def test_git_status_badges_use_short_ttl_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _GIT_STATUS_CACHE.clear()
            calls = []

            def fake_run(*args, **kwargs):
                calls.append(args)

                class Result:
                    returncode = 0
                    stdout = "?? src/app.py\n"
                    stderr = ""

                return Result()

            with patch("deepmate.channels.tui.files.subprocess.run", fake_run):
                first = git_status_badges(workspace)
                second = git_status_badges(workspace)

            self.assertEqual(first, {"src/app.py": "N"})
            self.assertEqual(second, {"src/app.py": "N"})
            self.assertEqual(len(calls), 1)

    def test_git_status_cache_is_bounded_by_workspace_count(self) -> None:
        _GIT_STATUS_CACHE.clear()
        with tempfile.TemporaryDirectory() as tmp:
            roots = [Path(tmp) / f"workspace_{index}" for index in range(GIT_STATUS_CACHE_MAX_WORKSPACES + 2)]
            for root in roots:
                root.mkdir()

            class Result:
                returncode = 0
                stdout = ""
                stderr = ""

            with patch("deepmate.channels.tui.files.subprocess.run", lambda *args, **kwargs: Result()):
                for root in roots:
                    git_status_badges(root)

            self.assertLessEqual(len(_GIT_STATUS_CACHE), GIT_STATUS_CACHE_MAX_WORKSPACES)

    def test_prompt_queue_tracks_pending_and_pause_state(self) -> None:
        queue = TuiPromptQueue()

        self.assertEqual(queue.enqueue("first"), 1)
        self.assertEqual(queue.enqueue("second"), 2)
        self.assertIn("queued 2", queue.footer_label())
        self.assertEqual(queue.pop_next(), "first")
        queue.pause()

        self.assertIsNone(queue.pop_next())
        self.assertIn("paused", queue.footer_label())
        queue.resume()
        self.assertEqual(queue.pop_next(), "second")
        self.assertEqual(queue.clear(), 0)

    def test_prompt_queue_has_capacity_limit(self) -> None:
        queue = TuiPromptQueue(max_size=2)

        self.assertEqual(queue.enqueue("first"), 1)
        self.assertEqual(queue.enqueue("second"), 2)
        self.assertTrue(queue.is_full())
        self.assertEqual(queue.enqueue("third"), 2)
        self.assertEqual(queue.pending, ["first", "second"])

    def test_prompt_queue_persists_pending_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queue.json"
            queue = TuiPromptQueue(path=path)

            queue.enqueue("first")
            queue.enqueue("second")
            queue.pause()
            loaded = TuiPromptQueue.load(path)

            self.assertEqual(loaded.pending, ["first", "second"])
            self.assertTrue(loaded.paused)
            loaded.resume()
            self.assertEqual(loaded.pop_next(), "first")
            self.assertTrue(path.exists())
            self.assertEqual(loaded.pop_next(), "second")
            self.assertFalse(path.exists())

    def test_tui_runtime_state_prompt_queue_path_is_session_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            state.data_dir = workspace / "var"

            path = state.prompt_queue_path()

            self.assertEqual(
                path,
                workspace / "var" / "tui" / "prompt_queues" / f"{state.session.session_id}.json",
            )

    def test_tui_immediate_commands_do_not_need_prompt_queue(self) -> None:
        self.assertTrue(_is_immediate_command("/commands"))
        self.assertTrue(_is_immediate_command("/status"))
        self.assertTrue(_is_immediate_command("/task"))
        self.assertTrue(_is_immediate_command("/diff"))
        self.assertTrue(_is_immediate_command("/detail"))
        self.assertTrue(_is_immediate_command("/close-tab"))
        self.assertTrue(_is_immediate_command("/undo-clear"))
        self.assertFalse(_is_immediate_command("/copy"))
        self.assertFalse(_is_immediate_command("/copy main"))
        self.assertTrue(_is_immediate_command("/open src/app.py"))
        self.assertTrue(_is_immediate_command("/find Todo"))
        self.assertTrue(_is_immediate_command("/files app"))
        self.assertTrue(_is_immediate_command("@app"))
        self.assertTrue(_is_immediate_command("/preview"))
        self.assertTrue(_is_immediate_command("/hide-preview"))
        self.assertTrue(_is_immediate_command("/search deepmate"))
        self.assertTrue(_is_immediate_command("/pet"))
        self.assertTrue(_is_immediate_command("/pet select cat"))
        self.assertTrue(_is_immediate_command("/model"))
        self.assertTrue(_is_immediate_command("/model upgrade"))
        self.assertTrue(_is_immediate_command("/skills"))
        self.assertTrue(_is_immediate_command("/mcp"))
        self.assertTrue(_is_immediate_command("/remote --wecom"))
        self.assertTrue(_is_immediate_command("/hooks status"))
        self.assertTrue(_is_immediate_command("/title New title"))
        self.assertTrue(_is_immediate_command("/rewind turn_00001"))
        self.assertTrue(_is_immediate_command("/deploy status"))
        self.assertFalse(_is_immediate_command("ordinary prompt"))

    def test_tui_command_hint_name_strips_placeholders(self) -> None:
        self.assertEqual(_command_hint_name("/open <path> [--offset N] - Open file"), "/open")
        self.assertEqual(_command_hint_name("/resume <id> - Resume a session."), "/resume")
        self.assertEqual(_command_hint_name("/tree, /clone, /fork - Short aliases"), "/tree")

    def test_tui_model_command_switches_future_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            provider = _StubProvider(
                [
                    ModelResponse(content="pro"),
                    ModelResponse(content="flash"),
                ]
            )
            state = _state(workspace, provider=provider)

            upgraded = handle_tui_command("/model upgrade", state)
            state, _messages, _exit = run_headless_tui_turn(state, "first")
            restored = handle_tui_command("/model default", state)
            state, _messages, _exit = run_headless_tui_turn(state, "second")

            self.assertTrue(upgraded.handled)
            self.assertTrue(restored.handled)
            self.assertEqual(provider.requests[0].model, "stub-pro")
            self.assertEqual(provider.requests[1].model, "stub-main")

    def test_tui_model_local_uses_configured_default_preset_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            state.local_default_model = "qwen3-local"

            result = handle_tui_command("/model local", state)

            self.assertTrue(result.handled)
            self.assertIsNotNone(result.local_prepare)
            self.assertEqual(result.local_prepare.preset.id, "qwen3-local")

    def test_tui_local_command_returns_background_prepare_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            preset = local_model_by_id("qwen3-local")
            self.assertIsNotNone(preset)

            result = handle_tui_command("/local qwen3-local", state)

            self.assertTrue(result.handled)
            self.assertIsNotNone(result.local_prepare)
            self.assertEqual(result.local_prepare.preset.id, "qwen3-local")
            self.assertEqual(state.provider_name, "stub")

    def test_headless_tui_local_command_switches_provider_and_context_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            local_provider = _StubProvider([ModelResponse(content="local")])
            state.local_provider = local_provider
            preset = local_model_by_id("qwen3-local")
            self.assertIsNotNone(preset)

            with patch(
                "deepmate.channels.tui.commands.OllamaLocalRuntime.prepare_model",
                return_value=LocalModelInstallResult(
                    ok=True,
                    preset=preset,
                    message="ready",
                ),
            ):
                state, messages, exit_requested = run_headless_tui_turn(
                    state,
                    "/local qwen3-local",
                )

            self.assertFalse(exit_requested)
            self.assertTrue(messages)
            self.assertEqual(state.provider_name, "local")
            self.assertIs(state.provider, local_provider)
            self.assertEqual(state.model, "qwen3:4b")
            self.assertIsNotNone(state.conversation_budget_policy)
            self.assertEqual(
                state.conversation_budget_policy.model_context_tokens,
                preset.effective_context_tokens,
            )
            self.assertEqual(state.options["max_tokens"], preset.max_tokens)

    def test_run_tui_mode_uses_configured_local_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            base = _state(workspace)
            created = []

            def fake_provider(base_url, api_key):
                created.append((base_url, api_key))
                return _StubProvider([ModelResponse(content="unused")])

            with (
                patch("deepmate.channels.tui.bridge.ChatCompletionsProvider", fake_provider),
                patch("deepmate.channels.tui.bridge.find_spec", lambda _name: None),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                exit_code = run_tui_mode(
                    provider=base.provider,
                    provider_name=base.provider_name,
                    provider_api_key_env=base.provider_api_key_env,
                    provider_api_key_available=base.provider_api_key_available,
                    model=base.model,
                    default_model=base.default_model,
                    upgrade_model=base.upgrade_model,
                    workspace=base.workspace,
                    profile=base.profile,
                    session_store=base.session_store,
                    session=base.session,
                    transcript=base.transcript,
                    runtime=base.runtime,
                    capability_surface=base.capability_surface,
                    native_tools=base.native_tools,
                    mcp_tools=base.mcp_tools,
                    subagents=base.subagents,
                    tool_access_policy=base.tool_access_policy,
                    tool_schemas=base.tool_schemas,
                    selected_skill_documents=base.selected_skill_documents,
                    mcp_servers=base.mcp_servers,
                    conversation_budget_policy=base.conversation_budget_policy,
                    provider_retry_policy=base.provider_retry_policy,
                    options=base.options,
                    max_steps=base.max_steps,
                    trace_recorder=base.trace_recorder,
                    warning_sink=base.warning_sink,
                    data_dir=base.data_dir,
                    local_provider_base_url="http://127.0.0.1:11555/v1",
                    local_provider_api_key="ollama",
                )

            self.assertEqual(exit_code, 2)
            self.assertEqual(created, [("http://127.0.0.1:11555/v1", "ollama")])

    def test_tui_local_switch_runs_context_prepare_callback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            local_provider = _StubProvider([ModelResponse(content="local")])
            state.local_provider = local_provider
            preset = local_model_by_id("qwen3-local")
            self.assertIsNotNone(preset)
            calls = []
            state.local_context_prepare_callback = lambda current, selected: calls.append(
                (current.session.session_id, selected.id)
            ) or True

            with patch(
                "deepmate.channels.tui.commands.OllamaLocalRuntime.prepare_model",
                return_value=LocalModelInstallResult(
                    ok=True,
                    preset=preset,
                    message="ready",
                ),
            ):
                state, messages, exit_requested = run_headless_tui_turn(
                    state,
                    "/local qwen3-local",
                )

            self.assertFalse(exit_requested)
            self.assertEqual(calls, [(state.session.session_id, "qwen3-local")])
            self.assertEqual(state.provider_name, "local")
            self.assertIn("已自动整理上下文", "\n".join(message.body for message in messages))

    def test_tui_local_prepare_result_rebuilds_provider_for_result_base_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            preset = local_model_by_id("qwen3-local")
            self.assertIsNotNone(preset)
            created = []

            def fake_provider(base_url, api_key):
                created.append((base_url, api_key))
                return _StubProvider([ModelResponse(content="local")])

            with patch(
                "deepmate.channels.tui.commands.ChatCompletionsProvider",
                fake_provider,
            ):
                result = apply_local_model_prepare_result(
                    state,
                    LocalModelPrepareRequest(preset=preset, source="/local"),
                    LocalModelInstallResult(
                        ok=True,
                        preset=preset,
                        message="ready",
                        provider_base_url="http://127.0.0.1:11555/v1",
                    ),
                )

            self.assertTrue(result.handled)
            self.assertEqual(state.provider_name, "local")
            self.assertEqual(state.local_provider_base_url, "http://127.0.0.1:11555/v1")
            self.assertEqual(created, [("http://127.0.0.1:11555/v1", "ollama")])
            self.assertIs(state.provider, state.local_provider)

    def test_tui_app_local_prepare_starts_background_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            app = DeepmateTuiApp(state)
            workers = []
            messages = []
            app._append_messages = lambda incoming: messages.extend(incoming)
            app._start_live_work = lambda _body="": None
            app._refresh_footer = lambda: None
            app.run_worker = lambda work, *, thread=False: workers.append((work, thread))

            handled = app._handle_immediate_command("/local qwen3-local")

            self.assertTrue(handled)
            self.assertTrue(workers)
            self.assertTrue(workers[0][1])
            self.assertTrue(app._local_prepare_running)
            self.assertTrue(any(message.title == "/local" for message in messages))

    def test_tui_prompt_without_available_model_is_queued_with_choice_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            provider = _StubProvider([ModelResponse(content="should not run")])
            state = _state(workspace, provider=provider)
            state.provider_api_key_available = False
            app = DeepmateTuiApp(state)
            messages = []
            app._write = lambda message: messages.append(message)
            app._refresh_footer = lambda: None

            app._submit_prompt("帮我看一下项目")

            self.assertEqual(app._prompt_queue.pending, ["帮我看一下项目"])
            self.assertEqual(provider.requests, [])
            self.assertTrue(state.missing_model_prompt_shown)
            self.assertTrue(any("/setup-key" in message.body for message in messages))
            self.assertTrue(any("/local" in message.body for message in messages))

    def test_tui_local_status_includes_last_prepare_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            data_dir = workspace / "var"
            state = _state(workspace, data_dir=data_dir)
            LocalModelStateStore(data_dir).record(
                model_id="qwen3-local",
                stage="failed",
                message="本地模型上次没有准备完成。输入 /local 后会继续。",
                status="failed",
                failure_kind="download_failed",
            )

            with patch("deepmate.channels.tui.commands.OllamaLocalRuntime.status") as status:
                status.return_value.running = False
                status.return_value.installed = True
                status.return_value.version = ""
                status.return_value.message = ""
                result = handle_tui_command("/local status", state)

            self.assertTrue(result.handled)
            self.assertIn("上次准备", result.messages[0].body)
            self.assertIn("上次本地模型没有准备完成", result.messages[0].body)

    def test_tui_local_prepare_success_runs_queued_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            local_provider = _StubProvider([ModelResponse(content="local answer")])
            state.local_provider = local_provider
            preset = local_model_by_id("qwen3-local")
            self.assertIsNotNone(preset)
            app = DeepmateTuiApp(state)
            messages = []
            submitted = []
            app._append_messages = lambda incoming: messages.extend(incoming)
            app._write = lambda message: messages.append(message)
            app._submit_prompt = lambda prompt: submitted.append(prompt)
            app._refresh_footer = lambda: None
            app._clear_live_work = lambda: None
            app._prompt_queue.enqueue("继续刚才的问题")
            app._prompt_queue.pause()

            app._finish_local_prepare_worker(
                LocalModelPrepareRequest(preset=preset, source="/local"),
                LocalModelInstallResult(ok=True, preset=preset, message="ready"),
            )

            self.assertEqual(state.provider_name, "local")
            self.assertIs(state.provider, local_provider)
            self.assertEqual(submitted, ["继续刚才的问题"])
            self.assertFalse(app._prompt_queue.pending)

    def test_tui_local_prepare_exception_resumes_paused_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            app = DeepmateTuiApp(state)
            messages = []
            app._write = messages.append
            app._refresh_footer = lambda: None
            app._clear_live_work = lambda: None
            app._prompt_queue.enqueue("继续刚才的问题")
            app._prompt_queue.pause()
            app._local_prepare_running = True
            app._running_turn = True

            app._handle_local_prepare_exception(RuntimeError("ollama failed"))

            self.assertFalse(app._local_prepare_running)
            self.assertFalse(app._running_turn)
            self.assertFalse(app._prompt_queue.paused)
            self.assertEqual(app._prompt_queue.pending, ["继续刚才的问题"])
            self.assertEqual(messages[-1].title, "/local")
            self.assertIn("已保留排队请求", messages[-1].body)

    def test_tui_model_remote_restores_cloud_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            remote_provider = _StubProvider([ModelResponse(content="remote")])
            state = _state(workspace, provider=remote_provider)
            state.remote_provider = remote_provider
            state.remote_provider_name = "stub"
            state.remote_model = "stub-main"
            state.remote_default_model = "stub-main"
            state.remote_upgrade_model = "stub-pro"
            state.remote_provider_api_key_env = "STUB_REMOTE_KEY"
            state.remote_provider_api_key_available = False
            state.remote_options = {"temperature": 0}
            state.remote_model_capabilities = ModelCapabilities(supports_tools=False)
            state.provider_name = "local"
            state.provider_api_key_env = "DEEPMATE_LOCAL_API_KEY"
            state.provider_api_key_available = True
            state.model = "qwen3-local"
            state.options = {"max_tokens": 4096}
            state.model_capabilities = ModelCapabilities(supports_stream_usage=False)
            state.conversation_budget_policy = ConversationBudgetPolicy(
                model_context_tokens=24_576,
                response_token_reserve=4_096,
                safety_margin_tokens=2_048,
            )

            result = handle_tui_command("/model remote", state)

            self.assertTrue(result.handled)
            self.assertEqual(state.provider_name, "stub")
            self.assertIs(state.provider, remote_provider)
            self.assertEqual(state.model, "stub-main")
            self.assertEqual(state.provider_api_key_env, "STUB_REMOTE_KEY")
            self.assertFalse(state.provider_api_key_available)
            self.assertEqual(state.options, {"temperature": 0})
            self.assertFalse(state.model_capabilities.supports_tools)
            self.assertIsNone(state.conversation_budget_policy)

    def test_tui_local_upgrade_uses_real_ollama_model_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            state.provider_name = "local"
            state.upgrade_model = "qwen3-coder-strong"

            result = handle_tui_command("/model upgrade", state)

            self.assertTrue(result.handled)
            self.assertEqual(state.model, "qwen3-coder:30b")

    def test_tui_enter_submits_exact_command_when_hints_are_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            submitted = []

            class Event:
                key = "enter"
                stopped = False
                prevented = False

                def stop(self) -> None:
                    self.stopped = True

                def prevent_default(self) -> None:
                    self.prevented = True

            event = Event()
            prompt = _PromptTextArea("/resume", id="prompt-input")
            event.widget = prompt
            app._safe_focused = lambda: prompt
            app.query_one = lambda selector, _widget_type=None: prompt
            app._command_hint_matches = (("/resume", "Resume a session."),)
            app._submit_prompt_editor = lambda: submitted.append("submitted")

            app.on_key(event)

            self.assertTrue(event.prevented)
            self.assertTrue(event.stopped)
            self.assertEqual(submitted, ["submitted"])

    def test_tui_clear_command_mentions_context_is_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            result = handle_tui_command("/clear", _state(workspace))

            self.assertTrue(result.handled)
            self.assertIn("transcript and context are unchanged", result.messages[0].body)

    def test_tui_render_message_escapes_markup(self) -> None:
        status_rendered = _render_message(
            TuiMessage(
                kind="status",
                title="[not-style]",
                body="Use [path] and [/path] literally.",
            )
        )
        assistant_rendered = _render_message(
            TuiMessage(
                kind="assistant",
                title="[not-style]",
                body="Use [path] and [/path] literally.",
            )
        )

        self.assertIn("\\[not-style]", status_rendered)
        self.assertIn("\\[path]", status_rendered)
        self.assertIn("\\[/path]", status_rendered)
        self.assertIsInstance(assistant_rendered, Padding)
        self.assertIsInstance(assistant_rendered.renderable, RichMarkdown)

    def test_tui_render_message_distinguishes_user_and_agent(self) -> None:
        user = _render_message(TuiMessage(kind="user", title="you", body="Fix it."))
        assistant = _render_message(
            TuiMessage(kind="assistant", title="assistant", body="Done.")
        )

        self.assertIsInstance(user, Padding)
        self.assertEqual((user.top, user.right, user.bottom, user.left), (1, 0, 1, 0))
        table = user.renderable
        self.assertIsInstance(table, Table)
        self.assertTrue(table.expand)
        self.assertEqual(str(table.style), "on #2c2c2c")
        self.assertEqual(str(table.rows[0].style), "on #2c2c2c")
        self.assertEqual(table.columns[0]._cells[0].plain, "\n› Fix it.\n")
        self.assertNotIn("You", table.columns[0]._cells[0].plain)
        self.assertIsInstance(assistant, Padding)
        self.assertIsInstance(assistant.renderable, RichMarkdown)
        self.assertEqual(assistant.renderable.markup, "Done.")

    def test_tui_assistant_message_uses_rich_markdown(self) -> None:
        rendered = _render_message(
            TuiMessage(
                kind="assistant",
                title="assistant",
                body="# Plan\n\n- read `src/app.py`\n\n```python\nprint('ok')\n```",
            )
        )

        self.assertIsInstance(rendered, Padding)
        self.assertIsInstance(rendered.renderable, RichMarkdown)
        self.assertIn("```python", rendered.renderable.markup)

    def test_tui_thinking_message_does_not_show_collapsed_jargon(self) -> None:
        rendered = _render_message(TuiMessage(kind="thinking", title="thinking", body="checking"))

        self.assertIn("∴ Thinking", rendered)
        self.assertNotIn("collapsed", rendered)

    def test_tui_tool_message_uses_family_hierarchy(self) -> None:
        rendered = _render_message(
            TuiMessage(
                kind="tool shell",
                title="run_shell_command",
                body="Completed.",
                status="ok",
            )
        )

        self.assertIn("Action: command", rendered)
        self.assertIn("run_shell_command", rendered)
        self.assertNotIn("tool browser", rendered)

    def test_tui_permission_summary_is_not_rendered_as_status(self) -> None:
        rendered = _render_message(
            TuiMessage(
                kind="permissions",
                title="permissions",
                body="Allowed (session): shell execution",
            )
        )

        self.assertIn("Permissions", rendered)
        self.assertIn("Allowed (session): shell execution", rendered)
        self.assertNotIn("Status", rendered)
        self.assertNotIn("approval", rendered.lower())

    def test_tui_live_status_renders_without_runtime_header(self) -> None:
        rendered = _render_message(
            TuiMessage(
                kind="status",
                title="runtime status",
                body="calling tool: search",
                status="live",
            )
        )

        self.assertIn("calling tool: search", rendered)
        self.assertNotIn("runtime status", rendered)
        self.assertNotIn("Status", rendered)
        self.assertNotIn("live", rendered)

    def test_tui_welcome_message_preserves_internal_markup(self) -> None:
        body = _welcome_splash(
            workspace="/tmp/[demo]",
            session_id="session-abcdef",
            provider_name="stub",
            api_key_env="STUB_API_KEY",
            api_key_available=True,
        )
        rendered = _render_message(TuiMessage(kind="welcome", title="new session", body=body))

        self.assertIn("[bold #e0e0e0]Deepmate[/]", rendered)
        self.assertIn("[#8fb7bd] /\\_/\\  [/]", rendered)
        self.assertIn("( o o )", rendered)
        self.assertIn("[#d0b66b]=  ^  =  [/]", rendered)
        self.assertIn("/commands", rendered)
        self.assertIn("/status", rendered)
        self.assertIn("/task", rendered)
        self.assertNotIn("Approvals:", rendered)
        self.assertNotIn("/trust", rendered)
        self.assertNotIn("/trust off", rendered)
        self.assertNotIn("/diff", rendered)
        self.assertIn("\\[demo]", rendered)
        self.assertNotIn("/tmp/[demo]", rendered)

    def test_tui_welcome_shortcuts_fit_with_file_tree_open(self) -> None:
        body = _welcome_splash(
            workspace="/tmp/demo",
            session_id="session-abcdef",
            provider_name="stub",
            api_key_env="STUB_API_KEY",
            api_key_available=True,
        )

        shortcut_lines = [
            Text.from_markup(line).plain
            for line in body.splitlines()
            if "/commands" in line and "/status" in line and "/task" in line
        ]

        self.assertEqual(len(shortcut_lines), 1)
        shortcut_text = " ".join(shortcut_lines[0].split())
        self.assertIn("try /commands /status /task files", shortcut_text)
        self.assertLessEqual(len(shortcut_text.split("try ", 1)[1]), 36)

    def test_tui_welcome_message_uses_current_provider_key_status(self) -> None:
        body = _welcome_splash(
            workspace="/tmp/demo",
            session_id="session-abcdef",
            provider_name="custom",
            api_key_env="CUSTOM_API_KEY",
            api_key_available=False,
        )
        rendered = _render_message(TuiMessage(kind="welcome", title="new session", body=body))

        self.assertIn("CUSTOM_API_KEY", rendered)
        self.assertIn("provider custom", rendered)
        self.assertIn("/setup-key", rendered)
        self.assertNotIn("DEEPSEEK_API_KEY", rendered)

    def test_tui_welcome_workspace_label_stays_compact(self) -> None:
        label = _compact_workspace_label("/home/user/workspace/deepmate_副本", limit=28)

        self.assertEqual(label, ".../deepmate_副本")
        self.assertLessEqual(len(label), 28)

    def test_tui_prompt_too_long_message_is_actionable(self) -> None:
        message = _prompt_too_long_message(50_001)

        self.assertEqual(message.kind, "error")
        self.assertIn("Input is 50001 characters", message.body)

    def test_tui_css_does_not_reintroduce_bright_blue_selection(self) -> None:
        css = DeepmateTuiApp.CSS.lower()

        self.assertNotIn("#000080", css)
        self.assertNotIn("#0000ff", css)
        self.assertNotIn("#7aa2f7", css)
        self.assertNotIn("#336699", css)
        self.assertNotIn("#003333", css)
        self.assertNotIn("$primary", css)
        self.assertNotIn("$block-cursor-background", css)
        self.assertIn("screen > .screen--selection", css)
        self.assertIn("#4a5658", css)
        self.assertIn("#file-tree:focus > listitem.-highlight", css)
        self.assertIn("#file-tree > listitem.-highlight label", css)
        self.assertIn("background: #4a5658 !important", css)
        self.assertIn(".session-tab-active", css)

    def test_tui_css_matches_input_row_to_user_message_height(self) -> None:
        css = DeepmateTuiApp.CSS.lower()

        self.assertIn(
            "#input-row {\n        width: 1fr;\n        min-width: 0;\n        height: auto;",
            css,
        )
        self.assertIn("max-height: 10;", css)
        self.assertIn("padding: 0 1;", css)
        self.assertIn("overflow-x: hidden", css)
        self.assertIn("overflow-y: auto", css)
        self.assertIn("#input-gap {\n        height: 1;", css)
        self.assertIn("#prompt-glyph {\n        width: 2;\n        height: 3;", css)
        self.assertIn("#prompt-input {\n        width: 1fr;\n        min-width: 0;\n        height: auto;", css)
        self.assertIn("max-height: 8;", css)
        self.assertIn("margin: 1 0;", css)
        self.assertIn(".text-area--cursor", css)
        self.assertIn("#sidebar {\n        width: 36;", css)
        self.assertIn("border-right: solid #464646;", css)
        self.assertIn(".session-tab:hover", css)
        self.assertIn("background: #3f3f3f !important", css)
        self.assertIn("#title-bar {\n        background: #333333;", css)
        self.assertIn("background: #2c2c2c;", css)
        self.assertIn("border-left: solid #8fb7bd;", css)
        self.assertIn("#content-markdown {\n        height: 1fr;", css)
        self.assertIn("overflow-y: auto", css)
        self.assertNotIn("border-bottom: solid #2d3b43", css)

    def test_tui_sidebar_defaults_open_with_close_hint(self) -> None:
        self.assertTrue(DeepmateTuiApp.sidebar_visible._default)
        css = DeepmateTuiApp.CSS
        self.assertIn("#sidebar-hint {", css)
        self.assertIn("background: #2f2f2f;", css)

    def test_tui_app_mounts_with_stylesheet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))

            async def run() -> None:
                async with app.run_test() as pilot:
                    await pilot.pause()

            asyncio.run(run())

    def test_tui_queue_pauses_only_on_terminal_error(self) -> None:
        self.assertFalse(
            _has_terminal_error(
                (
                    _message("error", "tool failed"),
                    _message("assistant", "recovered"),
                )
            )
        )
        self.assertTrue(
            _has_terminal_error(
                (
                    _message("assistant", "partial"),
                    _message("error", "max steps"),
                )
            )
        )

    def test_tui_new_session_starts_untitled_for_auto_naming(self) -> None:
        # A New+ session must start as "Untitled session" so the first prompt
        # auto-renames it (like the first session), not a fixed "new session".
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))

            from deepmate.channels.interactive import _ensure_session_title

            async def run() -> None:
                async with app.run_test():
                    app._create_new_session()
                    self.assertEqual(app.state.session.title, "Untitled session")
                    renamed = _ensure_session_title(
                        app.state.session_store,
                        app.state.session,
                        "重构登录模块",
                    )
                    self.assertEqual(renamed.title, "重构登录模块")

            asyncio.run(run())

    def test_tui_trust_command_relaxes_and_restores_gating(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            app.state.approval_cache = SessionApprovalCache()

            async def run() -> None:
                async with app.run_test():
                    base_mode = app.state.tool_access_policy.mode

                    self.assertTrue(app._handle_trust_command("/trust"))
                    self.assertTrue(app._trusted)
                    self.assertEqual(
                        app.state.tool_access_policy.mode,
                        ToolAccessMode.WORKSPACE_WRITE,
                    )
                    self.assertTrue(
                        app.state.approval_cache.is_allowed("capability:shell")
                    )
                    self.assertIn("auto-approve", app._status_label())

                    self.assertTrue(app._handle_trust_command("/trust off"))
                    self.assertFalse(app._trusted)
                    self.assertEqual(app.state.tool_access_policy.mode, base_mode)
                    self.assertNotIn("auto-approve", app._status_label())

            asyncio.run(run())

    def test_tui_new_session_button_creates_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            calls = []

            class Event:
                widget = type("WidgetRef", (), {"id": "new-session"})()
                x = 0

            app._create_new_session = lambda: calls.append("created")

            app.on_click(Event())

            self.assertEqual(calls, ["created"])

    def test_tui_new_session_is_immediate_while_another_session_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))

            async def run() -> None:
                async with app.run_test() as pilot:
                    old_session_id = app.state.session.session_id
                    app._running_turn = True
                    app._active_turn = _TurnRun(
                        session_id=old_session_id,
                        started_at=monotonic(),
                    )
                    app._session_running.add(old_session_id)
                    app.on_input_submitted(
                        type(
                            "Submitted",
                            (),
                            {
                                "value": "/new",
                                "input": type("InputRef", (), {"value": "/new"})(),
                            },
                        )()
                    )
                    await pilot.pause()

                    self.assertNotEqual(app.state.session.session_id, old_session_id)
                    self.assertIn(old_session_id, app._session_running)
                    self.assertEqual(app._prompt_queue.pending, [])
                    self.assertNotIn("/new", app._prompt_queue.pending)

            asyncio.run(run())

    def test_tui_switch_session_allowed_while_other_session_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))

            async def run() -> None:
                async with app.run_test() as pilot:
                    old_session_id = app.state.session.session_id
                    app._running_turn = True
                    app._active_turn = _TurnRun(
                        session_id=old_session_id,
                        started_at=monotonic(),
                    )
                    app._session_running.add(old_session_id)
                    app._create_new_session()
                    new_session_id = app.state.session.session_id
                    app._switch_session(old_session_id)
                    await pilot.pause()

                    self.assertEqual(app.state.session.session_id, old_session_id)
                    self.assertNotEqual(new_session_id, old_session_id)
                    self.assertIn(old_session_id, app._session_running)

            asyncio.run(run())

    def test_tui_new_session_prompt_runs_while_previous_session_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))

            async def run() -> None:
                async with app.run_test() as pilot:
                    messages = []
                    worker_calls = []
                    app._write = messages.append
                    app._render_active_tab = lambda: None
                    app.run_worker = lambda *args, **kwargs: worker_calls.append(
                        (args, kwargs)
                    )

                    old_session_id = app.state.session.session_id
                    old_turn = _TurnRun(
                        session_id=old_session_id,
                        started_at=monotonic(),
                    )
                    app._running_turn = True
                    app._active_turn = old_turn
                    app._session_turns[old_session_id] = old_turn
                    app._session_running.add(old_session_id)

                    app._create_new_session()
                    await pilot.pause()
                    new_session_id = app.state.session.session_id
                    app._submit_prompt("介绍下你自己")

                    self.assertNotEqual(new_session_id, old_session_id)
                    self.assertIn(old_session_id, app._session_running)
                    self.assertIn(new_session_id, app._session_running)
                    self.assertEqual(app._prompt_queue.pending, [])
                    self.assertEqual(len(worker_calls), 1)
                    self.assertEqual(messages[-1].kind, "user")
                    self.assertEqual(messages[-1].body, "介绍下你自己")
                    self.assertIn("running", app._status_label())

            asyncio.run(run())

    def test_tui_finishing_old_session_does_not_clear_current_session_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))

            async def run() -> None:
                async with app.run_test() as pilot:
                    old_session_id = app.state.session.session_id
                    app._create_new_session()
                    await pilot.pause()
                    new_session_id = app.state.session.session_id
                    new_turn = _TurnRun(
                        session_id=new_session_id,
                        started_at=monotonic(),
                    )
                    old_turn = _TurnRun(
                        session_id=old_session_id,
                        started_at=monotonic(),
                    )
                    app._running_turn = True
                    app._active_turn = new_turn
                    app._session_turns[old_session_id] = old_turn
                    app._session_turns[new_session_id] = new_turn
                    app._session_running.update({old_session_id, new_session_id})
                    app._maybe_start_next_queued_prompt = lambda: None
                    app._route_checkpoint_writes_to_current_session = lambda: None

                    app._mark_turn_idle(old_session_id)

                    self.assertTrue(app._running_turn)
                    self.assertIn(new_session_id, app._session_running)
                    self.assertNotIn(old_session_id, app._session_running)
                    self.assertIsNotNone(app._active_turn)
                    self.assertEqual(app._active_turn.session_id, new_session_id)

            asyncio.run(run())

    def test_tui_background_session_messages_survive_switch(self) -> None:
        # A backgrounded session keeps its full message history (including errors
        # that never reach the transcript), and switching back restores it from
        # memory rather than rebuilding from disk.
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))

            async def run() -> None:
                async with app.run_test():
                    session_a = app.state.session.session_id
                    app._append_background_message(
                        session_a,
                        TuiMessage(kind="assistant", title="a", body="answer A"),
                    )
                    app._create_new_session()
                    session_b = app.state.session.session_id
                    self.assertNotEqual(session_a, session_b)

                    # A's buffer is independent of B's.
                    self.assertTrue(
                        any(m.body == "answer A" for m in app._session_messages[session_a])
                    )
                    self.assertNotIn(
                        "answer A",
                        [m.body for m in app._main_messages],
                    )

                    # A fails in the background while viewing B; the error is kept.
                    app._handle_worker_exception(RuntimeError("boom A"), session_a)
                    self.assertTrue(
                        any(m.kind == "error" for m in app._session_messages[session_a])
                    )
                    self.assertIn(session_a, app._session_has_updates)

                    # Switching back to A restores the error and the answer.
                    app._switch_session(session_a)
                    kinds = {m.kind for m in app._main_messages}
                    bodies = [m.body for m in app._main_messages]
                    self.assertIn("error", kinds)
                    self.assertIn("answer A", bodies)

            asyncio.run(run())

    def test_tui_per_session_stats_isolate_context_window_footer(self) -> None:
        # Two concurrent sessions must not clobber each other's context-window
        # numbers; the footer always reflects the currently-viewed session.
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))

            async def run() -> None:
                async with app.run_test():
                    session_a = app.state.session.session_id
                    app._stats_for_session(session_a).record(
                        "runtime step 1; context ok (12% used); input_pressure=0.12; "
                        "context_remaining_input_tokens=880000"
                    )
                    app._create_new_session()
                    session_b = app.state.session.session_id
                    app._stats_for_session(session_b).record(
                        "runtime step 1; context ok (47% used); input_pressure=0.47; "
                        "context_remaining_input_tokens=500000"
                    )

                    self.assertIn("500k left", app._context_window_label())
                    app._switch_session(session_a)
                    self.assertIn("880k left", app._context_window_label())
                    self.assertIsNot(
                        app._stats_for_session(session_a),
                        app._stats_for_session(session_b),
                    )

            asyncio.run(run())

    def test_tui_typed_approval_does_not_cross_resolve_other_session(self) -> None:
        # Typing "allow" while viewing one session must not resolve an approval
        # raised by a different (background) session.
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))

            from threading import Event

            from deepmate.channels.tui.app import _PendingApproval

            async def run() -> None:
                async with app.run_test():
                    session_a = app.state.session.session_id
                    app._create_new_session()
                    session_b = app.state.session.session_id

                    pending = _PendingApproval(
                        title="Tool approval",
                        body="A wants shell",
                        event=Event(),
                        session_id=session_a,
                    )
                    app._pending_approval = pending
                    app._session_running.update({session_a, session_b})
                    turn_b = _TurnRun(session_id=session_b, started_at=monotonic())
                    app._session_turns[session_b] = turn_b
                    app._active_turn = turn_b
                    app._running_turn = True

                    # Viewing B, type "allow" -> A's approval must stay pending.
                    app._submit_prompt("allow")
                    self.assertIs(app._pending_approval, pending)
                    self.assertFalse(pending.event.is_set())

                    # Switch to A, then "allow" resolves it.
                    app._switch_session(session_a)
                    app._submit_prompt("allow")
                    self.assertTrue(pending.event.is_set())
                    self.assertEqual(pending.result, "once")

            asyncio.run(run())

    def test_tui_session_tool_approval_does_not_leak_across_sessions(self) -> None:
        # "Allow for session" granted in one session must not auto-approve the
        # same tool in another session.
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))

            async def run() -> None:
                async with app.run_test():
                    session_a = app.state.session.session_id
                    app._create_new_session()
                    session_b = app.state.session.session_id

                    app._tool_session_approvals_for_session(session_a).add(
                        "run_shell_command"
                    )

                    self.assertIn(
                        "run_shell_command",
                        app._tool_session_approvals_for_session(session_a),
                    )
                    self.assertNotIn(
                        "run_shell_command",
                        app._tool_session_approvals_for_session(session_b),
                    )

            asyncio.run(run())

    def test_tui_tool_session_approval_uses_approval_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            approvals = iter(("session",))
            app._request_approval = lambda *_args, **_kwargs: next(approvals)
            tool = NativeTool(
                name="computer_click",
                description="Click",
                input_schema={"type": "object"},
                handler=lambda _arguments: None,
                read_only=False,
            )

            click_decision = ToolAccessDecision(
                allowed=False,
                reason="Computer action requires approval.",
                requires_approval=True,
                refs=("approval_key=computer:click",),
            )
            input_decision = ToolAccessDecision(
                allowed=False,
                reason="Computer action requires approval.",
                requires_approval=True,
                refs=("approval_key=computer:input",),
            )

            self.assertTrue(app._tool_approval(tool, click_decision))
            self.assertTrue(app._tool_approval(tool, click_decision))
            with self.assertRaises(StopIteration):
                self.assertTrue(app._tool_approval(tool, input_decision))

    def test_tui_safety_approval_cache_does_not_leak_across_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            state.approval_cache = SessionApprovalCache()
            app = DeepmateTuiApp(state)

            async def run() -> None:
                async with app.run_test():
                    session_a = app.state.session.session_id
                    app._create_new_session()
                    session_b = app.state.session.session_id

                    cache_a = app._approval_cache_for_session(session_a)
                    cache_b = app._approval_cache_for_session(session_b)
                    self.assertIsNot(cache_a, cache_b)
                    cache_a.allow_for_session("capability:shell")

                    self.assertTrue(cache_a.is_allowed("capability:shell"))
                    self.assertFalse(cache_b.is_allowed("capability:shell"))

            asyncio.run(run())

    def test_tui_worker_state_uses_session_scoped_approval_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            state.approval_cache = SessionApprovalCache()
            built_for_sessions: list[str] = []

            def native_tool_factory(cache, session_id):
                built_for_sessions.append(session_id)
                return NativeToolRegistry(())

            state.native_tool_factory = native_tool_factory
            app = DeepmateTuiApp(state)

            async def run() -> None:
                async with app.run_test():
                    session_a = app.state.session.session_id
                    worker_a = app._state_for_worker_turn(session_a)
                    self.assertEqual(built_for_sessions, [session_a])
                    app._create_new_session()
                    session_b = app.state.session.session_id
                    built_for_sessions.clear()
                    worker_b = app._state_for_worker_turn(session_b)

                    self.assertIsNot(worker_a.approval_cache, worker_b.approval_cache)
                    self.assertEqual(built_for_sessions, [session_b])
                    with self.assertRaises(RuntimeError):
                        app._state_for_worker_turn(session_a)

            asyncio.run(run())

    def test_tui_directory_input_requests_workspace_switch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            target = Path(tmp) / "other project"
            target.mkdir()
            app = DeepmateTuiApp(_state(workspace))
            switched = []
            app._request_workspace_switch = lambda target, session_id="": switched.append(
                (target, session_id)
            )

            handled = app._handle_workspace_input(f"'{target}'")

            self.assertTrue(handled)
            self.assertEqual(switched, [(target.resolve(), "")])

    def test_tui_workspace_command_accepts_relative_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            target = workspace / "app"
            target.mkdir()
            app = DeepmateTuiApp(_state(workspace))
            switched = []
            app._request_workspace_switch = lambda target, session_id="": switched.append(
                (target, session_id)
            )

            handled = app._handle_workspace_input("/workspace app")

            self.assertTrue(handled)
            self.assertEqual(switched, [(target.resolve(), "")])

    def test_tui_workspace_command_reports_missing_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            messages = []
            app._write = messages.append

            handled = app._handle_workspace_input("/workspace /definitely/missing")

            self.assertTrue(handled)
            self.assertEqual(messages[-1].kind, "error")
            self.assertIn("Folder not found", messages[-1].body)

    def test_tui_workspace_switch_request_exits_for_full_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            target = Path(tmp) / "target"
            target.mkdir()
            state = _state(workspace)
            app = DeepmateTuiApp(state)
            messages = []
            exits = []
            app._write = messages.append
            app.exit = lambda: exits.append(True)

            with patch("deepmate.channels.tui.app.end_tui_session") as end_session:
                app._request_workspace_switch(target)

            self.assertEqual(app.exit_code, WORKSPACE_SWITCH_EXIT_CODE)
            self.assertEqual(
                state.workspace_switch_request,
                WorkspaceSwitchRequest(workspace=target.resolve()),
            )
            self.assertEqual(exits, [True])
            end_session.assert_called_once()
            self.assertIn("Opening workspace", messages[-1].body)

    def test_tui_workspace_switch_stops_pet_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            target = Path(tmp) / "target"
            target.mkdir()
            state = _state(workspace)
            app = DeepmateTuiApp(state)
            app._write = lambda message: None
            app.exit = lambda: None

            class Process:
                pid = 12345
                returncode = None
                signals: list[int] = []
                wait_calls = 0

                def poll(self):
                    return self.returncode

                def wait(self, timeout=None):
                    self.wait_calls += 1
                    if self.wait_calls == 1:
                        raise subprocess.TimeoutExpired("pet", timeout or 0)
                    self.returncode = -9
                    return self.returncode

            process = Process()
            app._pet_process = process

            with patch("deepmate.channels.tui.app.end_tui_session"), patch(
                "deepmate.channels.tui.app.os.killpg",
            ) as killpg:
                app._request_workspace_switch(target)

            self.assertIsNone(app._pet_process)
            self.assertEqual(killpg.call_count, 2)

    def test_tui_workspace_switch_resumes_latest_session_for_target_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            target = Path(tmp) / "target"
            target.mkdir()
            state = _state(workspace)
            latest = state.session_store.create(
                workspace=target,
                profile=state.profile,
                title="target work",
            )
            app = DeepmateTuiApp(state)
            messages = []
            app._write = messages.append
            app.exit = lambda: None

            with patch("deepmate.channels.tui.app.end_tui_session"):
                app._request_workspace_switch(target)

            self.assertEqual(
                state.workspace_switch_request,
                WorkspaceSwitchRequest(
                    workspace=target.resolve(),
                    session_id=latest.session_id,
                ),
            )
            self.assertIn("Resuming session", messages[-1].body)

    def test_tui_cross_workspace_resume_requests_full_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            target = Path(tmp) / "target"
            target.mkdir()
            state = _state(workspace)
            target_session = state.session_store.create(
                workspace=target,
                profile=state.profile,
                title="target work",
            )
            app = DeepmateTuiApp(state)
            switched = []
            app._request_workspace_switch = lambda target_workspace, session_id="": switched.append(
                (target_workspace, session_id)
            )

            handled = app._handle_session_browser_command(
                f"/resume {target_session.session_id[:8]}"
            )

            self.assertTrue(handled)
            self.assertEqual(switched, [(target.resolve(), target_session.session_id)])

    def test_tui_same_workspace_resume_switches_without_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            target_session = state.session_store.create(
                workspace=workspace,
                profile=state.profile,
                title="same workspace",
            )
            app = DeepmateTuiApp(state)
            switched = []
            app._switch_session = switched.append

            handled = app._handle_session_browser_command(
                f"/resume {target_session.session_id[:8]}"
            )

            self.assertTrue(handled)
            self.assertEqual(switched, [target_session.session_id])

    def test_tui_sessions_preview_groups_by_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            other = Path(tmp) / "other"
            other.mkdir()
            state = _state(workspace)
            current = state.session
            remote = state.session_store.create(
                workspace=other,
                profile=state.profile,
                title="other workspace",
            )

            preview = _sessions_preview(
                state.session_store.list_recent(limit=10_000),
                current_workspace=workspace,
                current_session_id=current.session_id,
            )

            self.assertIn(f"{workspace.name} (current workspace)", preview)
            self.assertIn(str(workspace.resolve()), preview)
            self.assertIn(str(other.resolve()), preview)
            self.assertIn(current.session_id[:8], preview)
            self.assertIn(remote.session_id[:8], preview)
            self.assertIn("Use /resume", preview)

    def test_directory_input_path_supports_terminal_drag_forms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "other project"
            target.mkdir()

            self.assertEqual(_directory_input_path(f"'{target}'"), target.resolve())
            self.assertEqual(
                _directory_input_path(target.as_uri()),
                target.resolve(),
            )
            self.assertIsNone(_directory_input_path(f"open {target}"))

    def test_cli_workspace_restart_args_replace_workspace_and_drop_prompt(self) -> None:
        target = Path("/tmp/target")

        args = _argv_with_workspace(
            (
                "--workspace",
                "/tmp/old",
                "--interactive",
                "--provider",
                "deepseek",
                "--model",
                "m",
                "--skill",
                "reviewer",
                "--temperature",
                "0.2",
                "--session-id",
                "old-session",
                "initial prompt",
            ),
            target,
            session_id="target-session",
        )

        self.assertEqual(args[:2], ["--workspace", str(target)])
        self.assertIn("--interactive", args)
        self.assertIn("--provider", args)
        self.assertIn("deepseek", args)
        self.assertIn("--model", args)
        self.assertIn("m", args)
        self.assertIn("--skill", args)
        self.assertIn("reviewer", args)
        self.assertIn("--temperature", args)
        self.assertIn("0.2", args)
        self.assertIn("--session-id", args)
        self.assertIn("target-session", args)
        self.assertNotIn("old-session", args)
        self.assertNotIn("initial prompt", args)

    def test_tui_copy_to_clipboard_prefers_system_clipboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            copied = []

            with patch("deepmate.channels.tui.app._copy_to_clipboard", copied.append):
                app.copy_to_clipboard("selected text")

            self.assertEqual(copied, ["selected text"])
            self.assertEqual(app._clipboard, "selected text")

    def test_tui_copy_to_clipboard_falls_back_to_textual_driver(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            fallback = []

            with patch(
                "deepmate.channels.tui.app._copy_to_clipboard",
                side_effect=OSError("no clipboard command"),
            ):
                with patch.object(
                    DeepmateTuiApp.__mro__[1],
                    "copy_to_clipboard",
                    lambda _self, text: fallback.append(text),
                ):
                    app.copy_to_clipboard("selected text")

            self.assertEqual(fallback, ["selected text"])
            self.assertEqual(app._clipboard, "selected text")

    def test_tui_write_tool_summary_shows_compact_diff_in_main_chat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            target = workspace / "notes.md"
            tool = next(
                item
                for item in workspace_filesystem_tools(workspace, include_write_tools=True)
                if item.name == "write_text_file"
            )
            result = tool.call(
                {
                    "path": "notes.md",
                    "content": "# Notes\n- done\n",
                    "overwrite": True,
                }
            )
            exchange = ModelToolExchange(
                tool_requests=(
                    ModelToolRequest(
                        name="write_text_file",
                        id="call_1",
                        arguments={
                            "path": "notes.md",
                            "content": "# Notes\n- done\n",
                            "overwrite": True,
                        },
                    ),
                ),
                tool_results=(
                    ModelToolResult(
                        name="write_text_file",
                        request_id="call_1",
                        content=result.content,
                        data=dict(result.data),
                        refs=tuple(result.refs),
                    ),
                ),
            )

            messages = tool_exchange_messages(exchange)

            self.assertTrue(target.is_file())
            self.assertIn("✓ Wrote notes.md", messages[0].body)
            self.assertIn("Full diff is available in /detail.", messages[0].body)
            self.assertIn("```diff", messages[0].body)
            self.assertIn("+# Notes", messages[0].body)
            self.assertIn("+# Notes", messages[0].preview)

    def test_tui_approval_button_resolves_pending_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            messages = []
            focused = []
            app._write = messages.append
            app._refresh_footer = lambda: None
            app._render_approval_panel = lambda: None

            class Prompt:
                def focus(self) -> None:
                    focused.append(True)

            app.query_one = lambda selector, _widget_type=None: Prompt()
            pending = app._pending_approval = _pending_approval("Tool approval")

            class Event:
                static = type("StaticRef", (), {"id": "approval-session"})()
                stopped = False

                def stop(self) -> None:
                    self.stopped = True

            event = Event()

            app.on_static_clicked(event)

            self.assertTrue(event.stopped)
            self.assertEqual(pending.result, "session")
            self.assertIsNone(app._pending_approval)
            self.assertTrue(pending.event.is_set())
            self.assertEqual(messages[-1].kind, "permissions")
            self.assertEqual(messages[-1].title, "permissions")
            self.assertEqual(messages[-1].status, "")
            self.assertIn("Allowed (session): Tool approval", messages[-1].body)
            self.assertEqual(focused, [True])

    def test_tui_pending_approval_uses_panel_without_chat_log_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            messages = []
            renders = []
            app._write = messages.append
            app._refresh_footer = lambda: None
            app._render_approval_panel = lambda: renders.append("panel")

            pending = _pending_approval("Tool approval")
            app._show_pending_approval(pending)

            self.assertIs(app._pending_approval, pending)
            self.assertEqual(renders, ["panel"])
            self.assertEqual(messages, [])
            self.assertIn("approval · waiting for your choice", app._status_label())

    def test_tui_pending_approvals_are_queued_fifo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            messages = []
            renders = []
            app._write = messages.append
            app._refresh_footer = lambda: None
            app._render_approval_panel = lambda: renders.append(
                app._pending_approval.title if app._pending_approval else "none"
            )

            class Prompt:
                def focus(self) -> None:
                    pass

            app.query_one = lambda selector, _widget_type=None: Prompt()
            first = _pending_approval("First approval")
            second = _pending_approval("Second approval")

            app._show_pending_approval(first)
            app._show_pending_approval(second)

            self.assertIs(app._pending_approval, first)
            self.assertEqual(app._approval_queue, [second])
            self.assertFalse(first.event.is_set())
            self.assertFalse(second.event.is_set())
            self.assertIn("1 queued", app._status_label())

            self.assertTrue(app._resolve_pending_approval("once"))

            self.assertEqual(first.result, "once")
            self.assertTrue(first.event.is_set())
            self.assertIs(app._pending_approval, second)
            self.assertEqual(app._approval_queue, [])
            self.assertFalse(second.event.is_set())
            self.assertEqual(renders, ["First approval", "Second approval"])

            self.assertTrue(app._resolve_pending_approval("deny"))

            self.assertEqual(second.result, "deny")
            self.assertTrue(second.event.is_set())
            self.assertIsNone(app._pending_approval)
            self.assertEqual(app._last_queue_pause_reason, "approval denied")
            self.assertEqual(messages[-1].kind, "permissions")
            self.assertEqual(messages[-1].title, "permissions")
            self.assertEqual(messages[-1].status, "")
            self.assertIn("Denied: Second approval", messages[-1].body)

    def test_tui_approval_results_are_grouped_by_permission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            app._refresh_footer = lambda: None
            app._render_approval_panel = lambda: None
            app._write = lambda message: app._append_main_message(message)

            class Prompt:
                def focus(self) -> None:
                    pass

            app.query_one = lambda selector, _widget_type=None: Prompt()
            approvals = (
                _pending_approval("Safety approval"),
                _pending_approval("Safety approval"),
                _pending_approval("Safety approval"),
            )
            approvals[0].subject = "shell execution"
            approvals[0].refs = ("approval_key=capability:shell",)
            approvals[1].subject = "this shell command"
            approvals[1].refs = ("command=python3 -m pip install skillhub",)
            approvals[2].subject = "shell network access"
            approvals[2].refs = ("approval_key=capability:shell-network",)

            for pending in approvals:
                app._pending_approval = pending
                app._resolve_pending_approval("session")

            permission_messages = [
                message
                for message in app._main_messages
                if message.kind == "permissions"
            ]
            self.assertEqual(len(permission_messages), 1)
            self.assertEqual(permission_messages[0].status, "")
            body = permission_messages[0].body
            self.assertIn("Allowed (session): shell execution", body)
            self.assertIn(
                "Allowed (session): shell command `python3 -m pip install skillhub`",
                body,
            )
            self.assertIn("Allowed (session): shell network access", body)
            self.assertNotIn("Deepmate will continue automatically", body)

    def test_tui_deny_pending_approval_releases_queued_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            app._refresh_footer = lambda: None
            app._render_approval_panel = lambda: None
            first = app._pending_approval = _pending_approval("First approval")
            second = _pending_approval("Second approval")
            app._approval_queue.append(second)

            app._deny_pending_approval()

            self.assertIsNone(app._pending_approval)
            self.assertEqual(app._approval_queue, [])
            self.assertEqual(first.result, "deny")
            self.assertEqual(second.result, "deny")
            self.assertTrue(first.event.is_set())
            self.assertTrue(second.event.is_set())
            self.assertEqual(app._last_queue_pause_reason, "approval denied")

    def test_tui_approval_body_includes_action_refs(self) -> None:
        body = _approval_body(
            "Needs workspace write.",
            (
                "tool=write_text_file",
                "path=demo.html",
                "content_chars=14",
                "content_preview=<h1>Hello</h1>",
            ),
            fallback="tool=write_text_file",
        )

        self.assertIn("Deepmate wants to create or overwrite a file.", body)
        self.assertIn("Location: demo.html", body)
        self.assertIn("Size: 14 chars", body)
        self.assertIn("Content summary: new file content", body)
        self.assertIn("Preview hidden", body)
        self.assertNotIn("<h1>Hello</h1>", body)
        self.assertIn("Allow once: applies to this request only", body)

    def test_tui_edit_approval_body_hides_large_html_diff(self) -> None:
        body = _approval_body(
            "Needs workspace write.",
            (
                "tool=edit_text_file",
                "path=report.html",
                "old_text_chars=4000",
                "new_text_chars=5200",
                "old_text_preview=<div><table><tr><td>old</td></tr></table></div>",
                "new_text_preview=<div><table><tr><td>new</td></tr></table></div>",
            ),
            fallback="tool=edit_text_file",
        )

        self.assertIn("Deepmate wants to modify a file.", body)
        self.assertIn("Location: report.html", body)
        self.assertIn("Change summary:", body)
        self.assertIn("Preview hidden", body)
        self.assertNotIn("<table>", body)
        self.assertNotIn("<td>new</td>", body)

    def test_tui_approval_diff_renderable_colors_changed_lines(self) -> None:
        diff = _approval_diff_renderable(
            {
                "old_text_preview": "def add(a, b):\n    return a + b\n",
                "new_text_preview": "def add(a, b):\n    # sum\n    return a + b\n",
            }
        )
        self.assertIsNotNone(diff)
        self.assertIn("# sum", diff.plain)
        added_span = next(
            (
                span
                for span in diff.spans
                if "# sum" in diff.plain[span.start:span.end]
            ),
            None,
        )
        self.assertIsNotNone(added_span)
        self.assertEqual(added_span.style, "green")

    def test_tui_approval_diff_renderable_skips_when_nothing_to_show(self) -> None:
        # Identical text, markup content, and empty refs all yield no diff.
        self.assertIsNone(
            _approval_diff_renderable(
                {"old_text_preview": "same", "new_text_preview": "same"}
            )
        )
        self.assertIsNone(
            _approval_diff_renderable(
                {
                    "old_text_preview": "<div>a</div>",
                    "new_text_preview": "<div>b</div>",
                }
            )
        )
        self.assertIsNone(_approval_diff_renderable({}))

    def test_tui_leave_without_widget_does_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))

            class Event:
                pass

            app.on_leave(Event())

    def test_tui_approval_result_includes_subject_and_refs(self) -> None:
        from threading import Event

        from deepmate.channels.tui.app import _PendingApproval

        body = _approval_result_body(
            _PendingApproval(
                title="Safety approval",
                body="needs approval",
                event=Event(),
                result="session",
                subject="shell execution",
                refs=("command=python3 setup.py", "network=off"),
            )
        )

        self.assertIn(
            "Permission allowed for this session for shell execution.",
            body,
        )
        self.assertIn("Deepmate will continue automatically.", body)
        self.assertNotIn("command=python3 setup.py", body)
        self.assertNotIn("network=off", body)

    def test_tui_approval_accepts_explicit_grant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            messages = []
            app._write = messages.append
            app._refresh_footer = lambda: None
            app._render_approval_panel = lambda: None

            class Prompt:
                def focus(self) -> None:
                    pass

            app.query_one = lambda selector, _widget_type=None: Prompt()
            app._running_turn = True
            pending = app._pending_approval = _pending_approval("Tool approval")

            app._submit_prompt("允许")

            self.assertIsNone(app._pending_approval)
            self.assertTrue(pending.event.is_set())
            self.assertEqual(pending.result, "once")
            self.assertEqual(app._prompt_queue.pending, [])
            self.assertEqual(messages[-1].kind, "permissions")
            self.assertEqual(messages[-1].title, "permissions")
            self.assertEqual(messages[-1].status, "")
            self.assertIn("Allowed (once): Tool approval", messages[-1].body)

    def test_tui_approval_ignores_free_chat_while_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            messages = []
            app._write = messages.append
            app._refresh_footer = lambda: None
            app._render_approval_panel = lambda: None
            app._running_turn = True
            pending = app._pending_approval = _pending_approval("Tool approval")

            # A line that merely contains "继续" must not auto-approve.
            app._submit_prompt("我们继续聊别的吧")

            self.assertIs(app._pending_approval, pending)
            self.assertFalse(pending.event.is_set())

    def test_tui_tool_allow_once_applies_to_same_tool_for_current_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            approvals = iter(("once",))
            app._request_approval = lambda *_args, **_kwargs: next(approvals)
            tool = next(
                item
                for item in workspace_filesystem_tools(workspace, include_write_tools=True)
                if item.name == "write_text_file"
            )
            decision = ToolAccessDecision(
                allowed=False,
                reason="Native tool requires workspace write access: write_text_file",
                requires_approval=True,
            )

            self.assertTrue(app._tool_approval(tool, decision))
            self.assertTrue(app._tool_approval(tool, decision))
            with self.assertRaises(StopIteration):
                next(approvals)

    def test_tui_approval_input_result_parses_common_phrases(self) -> None:
        # Explicit signals are honored (incl. common CN approval phrases).
        self.assertEqual(_approval_input_result("允许"), "once")
        self.assertEqual(_approval_input_result("approve"), "once")
        self.assertEqual(_approval_input_result("总是允许"), "session")
        self.assertEqual(_approval_input_result("拒绝"), "deny")
        self.assertEqual(_approval_input_result("给你写入权限啊，不是给了吗"), "once")
        # Free chat that merely contains a keyword must NOT be a decision.
        self.assertIsNone(_approval_input_result("我们继续聊别的"))
        self.assertIsNone(_approval_input_result("不要继续"))
        self.assertIsNone(_approval_input_result("不要授权"))
        self.assertIsNone(_approval_input_result("补充一下文案"))

    def test_tui_ctrl_c_copies_screen_selected_chat_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            copied = []

            async def run() -> None:
                async with app.run_test(size=(80, 16)) as pilot:
                    app.copy_to_clipboard = copied.append
                    chat = app.query_one("#chat", _SelectableRichLog)
                    chat.clear()
                    chat.write("hello selectable world", scroll_end=False)
                    chat.write("second selectable line", scroll_end=False)
                    chat.text_select_all()

                    await pilot.press("ctrl+c")

            asyncio.run(run())

            self.assertEqual(
                copied,
                ["hello selectable world\nsecond selectable line"],
            )

    def test_tui_mouse_selected_chat_text_copies_with_ctrl_c(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            copied = []

            async def run() -> None:
                async with app.run_test(size=(80, 16)) as pilot:
                    app.copy_to_clipboard = copied.append
                    chat = app.query_one("#chat", _SelectableRichLog)
                    chat.clear()
                    chat.write("hello selectable world", scroll_end=False)
                    chat.write("second selectable line", scroll_end=False)

                    await pilot.mouse_down("#chat", offset=(1, 0))
                    await pilot.hover("#chat", offset=(24, 0))
                    await pilot.mouse_up("#chat", offset=(24, 0))

                    self.assertEqual(
                        app.screen.get_selected_text(),
                        "hello selectable world",
                    )

                    await pilot.press("ctrl+c")

            asyncio.run(run())

            self.assertEqual(copied, ["hello selectable world"])

    def test_tui_mouse_selected_content_tab_text_copies_with_ctrl_c(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            copied = []

            async def run() -> None:
                async with app.run_test(size=(80, 16)) as pilot:
                    app.copy_to_clipboard = copied.append
                    app._open_content_tab(
                        "notes.txt",
                        "alpha beta\ngamma delta\n",
                        render_mode="plain",
                    )
                    await pilot.pause()

                    content = app.query_one("#content-markdown", _SelectableRichLog)
                    self.assertTrue(content.display)

                    await pilot.mouse_down("#content-markdown", offset=(1, 1))
                    await pilot.hover("#content-markdown", offset=(12, 1))
                    await pilot.mouse_up("#content-markdown", offset=(12, 1))

                    self.assertEqual(app.screen.get_selected_text(), "alpha beta")

                    await pilot.press("ctrl+c")

            asyncio.run(run())

            self.assertEqual(copied, ["alpha beta"])

    def test_tui_app_ctrl_c_binding_copies_screen_selection_only(self) -> None:
        keys = {
            getattr(binding, "key", "")
            for binding in DeepmateTuiApp.BINDINGS
            if getattr(binding, "action", "") == "copy_screen_selection"
        }

        self.assertIn("ctrl+c,super+c", keys)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            copied = []

            async def run() -> None:
                async with app.run_test(size=(80, 16)) as pilot:
                    app.copy_to_clipboard = copied.append
                    chat = app.query_one("#chat", _SelectableRichLog)
                    chat.clear()
                    chat.write("hello selectable world", scroll_end=False)
                    chat.text_select_all()

                    app.query_one("#prompt-input", _PromptTextArea).focus()
                    await pilot.press("ctrl+c")

            asyncio.run(run())

            self.assertEqual(copied, ["hello selectable world"])

    def test_tui_ctrl_c_copies_widget_selection_when_screen_text_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            copied = []

            async def run() -> None:
                async with app.run_test(size=(80, 16)) as pilot:
                    app.copy_to_clipboard = copied.append
                    chat = app.query_one("#chat", _SelectableRichLog)
                    chat.clear()
                    chat.write("hello selectable world", scroll_end=False)
                    app.screen.selections = {
                        chat: Selection(Offset(0, 0), Offset(5, 0))
                    }
                    app.screen.get_selected_text = lambda: None

                    await pilot.press("ctrl+c")

            asyncio.run(run())

            self.assertEqual(copied, ["hello"])

    def test_tui_on_key_ctrl_c_copies_selection_before_binding_bell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            copied = []

            async def run() -> None:
                async with app.run_test(size=(80, 16)):
                    app.copy_to_clipboard = copied.append
                    chat = app.query_one("#chat", _SelectableRichLog)
                    chat.clear()
                    chat.write("hello selectable world", scroll_end=False)
                    chat.text_select_all()
                    event = events.Key("ctrl+c", None)

                    app.on_key(event)

                    self.assertTrue(event._no_default_action)
                    self.assertTrue(event._stop_propagation)

            asyncio.run(run())

            self.assertEqual(copied, ["hello selectable world"])

    def test_tui_ctrl_c_uses_rich_log_selection_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            copied = []

            async def run() -> None:
                async with app.run_test(size=(80, 16)):
                    app.copy_to_clipboard = copied.append
                    chat = app.query_one("#chat", _SelectableRichLog)
                    chat._selected_text_cache = "cached selection"
                    app.screen.clear_selection()

                    event = events.Key("ctrl+c", None)
                    app.on_key(event)

                    self.assertTrue(event._no_default_action)
                    self.assertTrue(event._stop_propagation)

            asyncio.run(run())

            self.assertEqual(copied, ["cached selection"])

    def test_tui_ctrl_c_copies_prompt_selection_when_no_screen_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            copied = []

            async def run() -> None:
                async with app.run_test(size=(80, 16)) as pilot:
                    app.copy_to_clipboard = copied.append
                    prompt = app.query_one("#prompt-input", _PromptTextArea)
                    prompt.value = "copy this prompt"
                    prompt.focus()
                    prompt.selection = type(prompt.selection)((0, 0), (0, 9))

                    await pilot.press("ctrl+c")

            asyncio.run(run())

            self.assertEqual(copied, ["copy this"])

    def test_tui_ctrl_c_without_selection_does_not_bell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            bell_calls = []

            async def run() -> None:
                async with app.run_test(size=(80, 16)) as pilot:
                    app.bell = lambda: bell_calls.append(True)

                    await pilot.press("ctrl+c")

            asyncio.run(run())

            self.assertEqual(bell_calls, [])

    def test_tui_selectable_rich_log_extracts_rendered_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))

            async def run() -> None:
                async with app.run_test(size=(80, 16)):
                    log = app.query_one("#chat", _SelectableRichLog)
                    log.clear()
                    log.write("alpha beta", scroll_end=False)
                    log.write("gamma delta", scroll_end=False)

                    selected = log.get_selection(Selection(Offset(6, 0), Offset(5, 1)))

                    self.assertEqual(selected, ("beta\ngamma", "\n"))

            asyncio.run(run())

    def test_tui_ctrl_c_without_selection_does_not_interrupt_or_quit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            quit_calls = []
            app.action_quit = lambda: quit_calls.append(True)

            async def run() -> None:
                async with app.run_test() as pilot:
                    app._running_turn = True
                    await pilot.press("ctrl+c")

            asyncio.run(run())

            self.assertFalse(app._interrupted)
            self.assertEqual(quit_calls, [])

    def test_tui_escape_interrupts_running_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            messages = []
            app._write = messages.append
            app._refresh_footer = lambda: None
            app._render_approval_panel = lambda: None
            app._running_turn = True

            app.action_interrupt_or_cancel()

            self.assertTrue(app._interrupted)
            self.assertEqual(messages[-1].title, "interrupt")

    def test_tui_escape_denies_pending_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            pending = _pending_approval("Tool approval")
            pending.session_id = app.state.session.session_id
            app._pending_approval = pending
            app._write = lambda _message: None
            app._refresh_footer = lambda: None
            app._render_approval_panel = lambda: None

            app.action_interrupt_or_cancel()

            self.assertEqual(pending.result, "deny")
            self.assertTrue(pending.event.is_set())
            self.assertIsNone(app._pending_approval)

    def test_tui_quit_stops_desktop_pet_process(self) -> None:
        class FakePetProcess:
            pid = 12345

            def __init__(self) -> None:
                self.wait_calls = 0

            def poll(self):
                return None

            def wait(self, timeout=None):
                self.wait_calls += 1
                return 0

        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            process = FakePetProcess()
            app._pet_process = process
            signals = []
            app._deny_pending_approval = lambda: None
            app.exit = lambda: None
            app._signal_pet_process = lambda proc, sig: signals.append((proc, sig))

            app.action_quit()

            self.assertIsNone(app._pet_process)
            self.assertEqual(signals, [(process, signal.SIGTERM)])
            self.assertEqual(process.wait_calls, 1)

    def test_tui_paste_action_keeps_multiline_text_in_prompt_editor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            hints = []

            class Prompt:
                id = "prompt-input"
                value = "ask  now"
                cursor_position = 4
                selection = None
                focused = False

                def focus(self) -> None:
                    self.focused = True

            prompt = Prompt()
            app.query_one = lambda selector, _widget_type=None: prompt
            app._update_command_hints = hints.append

            with patch("deepmate.channels.tui.app._read_clipboard", return_value="me\nabout this"):
                app.action_paste_clipboard()

            self.assertEqual(prompt.value, "ask me\nabout this now")
            self.assertFalse(app._compose_mode)
            self.assertEqual(app._compose_lines, [])
            self.assertEqual(hints[-1], "ask me\nabout this now")

    def test_tui_paste_deduplicates_event_and_ctrl_v_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            hints = []

            class Prompt:
                id = "prompt-input"
                value = ""
                cursor_position = 0
                selection = None
                focused = False

                def focus(self) -> None:
                    self.focused = True

            prompt = Prompt()
            app.query_one = lambda selector, _widget_type=None: prompt
            app._update_command_hints = hints.append

            with patch("deepmate.channels.tui.app.monotonic", return_value=10.0):
                app._paste_into_prompt("hello")
            with patch("deepmate.channels.tui.app._read_clipboard", return_value="hello"):
                with patch("deepmate.channels.tui.app.monotonic", return_value=10.1):
                    app.action_paste_clipboard()

            self.assertEqual(prompt.value, "hello")
            self.assertEqual(hints, ["hello"])

            with patch("deepmate.channels.tui.app.monotonic", return_value=10.6):
                app._paste_into_prompt("hello")

            self.assertEqual(prompt.value, "hellohello")

    def test_tui_prompt_history_recalls_previous_prompt_and_restores_draft(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            app._render_active_tab = lambda: None
            app._write = lambda message: app._append_main_message(message)
            app.run_worker = lambda *_args, **_kwargs: None
            app._submit_prompt("first prompt")
            app._session_running.clear()
            app._session_turns.clear()
            app._running_turn = False
            app._active_turn = None
            app._submit_prompt("second prompt")

            class Prompt:
                value = "draft"
                cursor_position = 0

            prompt = Prompt()
            app.query_one = lambda selector, _widget_type=None: prompt
            app._update_command_hints = lambda _value: None

            self.assertTrue(app._handle_prompt_history_key("up"))
            self.assertEqual(prompt.value, "second prompt")
            self.assertTrue(app._handle_prompt_history_key("up"))
            self.assertEqual(prompt.value, "first prompt")
            self.assertTrue(app._handle_prompt_history_key("down"))
            self.assertEqual(prompt.value, "second prompt")
            self.assertTrue(app._handle_prompt_history_key("down"))
            self.assertEqual(prompt.value, "draft")

    def test_tui_clear_preserves_tabs_and_can_undo_display_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            app._open_tabs["README.md"] = _OpenTab("README.md", "# Demo\n")
            app._active_tab = "README.md"
            app._main_messages = [TuiMessage(kind="assistant", title="assistant", body="Done")]
            app._render_active_tab = lambda: None
            app._refresh_content_tabs = lambda: None

            class Chat:
                def clear(self) -> None:
                    pass

            app.query_one = lambda selector, _widget_type=None: Chat()
            app._clear_chat_display()

            self.assertIn("README.md", app._open_tabs)
            self.assertEqual(app._main_messages[-1].title, "clear")

            app._undo_clear_chat_display()

            self.assertEqual(app._main_messages[0].body, "Done")

    def test_tui_restore_draft_restores_last_unsent_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            messages = []
            hints = []
            app._write = messages.append
            app._update_command_hints = hints.append

            class Prompt:
                value = ""
                cursor_position = 0
                focused = False

                def focus(self) -> None:
                    self.focused = True

            prompt = Prompt()
            app.query_one = lambda selector, _widget_type=None: prompt
            app._remember_unsent_draft("draft line 1\nline 2")

            app._handle_prompt_submission("/restore-draft", "/restore-draft")

            self.assertEqual(prompt.value, "draft line 1\nline 2")
            self.assertTrue(prompt.focused)
            self.assertEqual(hints[-1], "draft line 1\nline 2")
            self.assertEqual(messages[-1].title, "draft")
            self.assertIn("Restored", messages[-1].body)

    def test_tui_submitted_prompt_can_be_restored_as_draft(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            messages = []
            hints = []
            app._write = messages.append
            app._update_command_hints = hints.append
            app._render_active_tab = lambda: None
            app.run_worker = lambda *_args, **_kwargs: None

            class Prompt:
                value = ""
                cursor_position = 0
                focused = False

                def focus(self) -> None:
                    self.focused = True

            prompt = Prompt()
            app.query_one = lambda selector, _widget_type=None: prompt

            app._handle_prompt_submission("final prompt", "final prompt")
            app._handle_prompt_submission("/restore-draft", "/restore-draft")

            self.assertEqual(prompt.value, "final prompt")
            self.assertTrue(prompt.focused)
            self.assertEqual(hints[-1], "final prompt")

    def test_tui_control_characters_are_cleaned_from_prompts_and_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            app._render_active_tab = lambda: None
            app._write = lambda message: app._append_main_message(message)
            app.run_worker = lambda *_args, **_kwargs: None

            app._submit_prompt("hello\x1b[2Jworld\x00")

            self.assertEqual(app._last_submitted_prompt_by_session[app.state.session.session_id], "helloworld")
            rendered = _render_message(
                TuiMessage(kind="status", title="status", body="ok\x1b[2Jbad\x00")
            )
            self.assertIn("okbad", rendered)
            self.assertNotIn("\x1b", rendered)

    def test_tui_qa_continue_keeps_visible_prompt_and_sends_audit_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            handle_qa_command("/qa 发布前验收", workspace=workspace, allow_fallback=True)
            app = DeepmateTuiApp(_state(workspace))
            messages = []
            worker_prompts = []

            def fake_worker(turn_state, prompt, *, session_id, started_at):
                worker_prompts.append(prompt)

            app._write = lambda message: messages.append(message)
            app._render_active_tab = lambda: None
            app._run_prompt_worker = fake_worker
            app.run_worker = lambda work, *, thread=False: work()

            app._submit_prompt("继续 QA Audit")

            self.assertEqual(messages[0].kind, "user")
            self.assertEqual(messages[0].body, "继续 QA Audit")
            self.assertEqual(
                app._last_submitted_prompt_by_session[app.state.session.session_id],
                "继续 QA Audit",
            )
            self.assertEqual(
                app._session_prompt_history[app.state.session.session_id][-1],
                "继续 QA Audit",
            )
            self.assertEqual(len(worker_prompts), 1)
            self.assertIn("<qa_audit_context>", worker_prompts[0])
            self.assertIn("Continue the active QA Audit.", worker_prompts[0])

    def test_tui_insert_prompt_text_respects_max_length(self) -> None:
        class Prompt:
            value = "abcd"
            cursor_position = 2
            selection = None

        prompt = Prompt()

        inserted = _insert_prompt_text(prompt, "XYZ", max_chars=6)

        self.assertTrue(inserted)
        self.assertEqual(prompt.value, "abXYcd")
        self.assertEqual(prompt.cursor_position, 4)

    def test_tui_insert_prompt_text_replaces_selection(self) -> None:
        class SelectionRef:
            start = 2
            end = 5
            is_empty = False

        class Prompt:
            value = "abcdef"
            cursor_position = 4
            selection = SelectionRef()

        prompt = Prompt()

        inserted = _insert_prompt_text(prompt, "XY", max_chars=20)

        self.assertTrue(inserted)
        self.assertEqual(prompt.value, "abXYf")
        self.assertEqual(prompt.cursor_position, 4)

    def test_tui_prompt_text_area_supports_multiline_prompt_insert(self) -> None:
        prompt = _PromptTextArea("hello", id="prompt-input")
        prompt.cursor_position = 5

        inserted = _insert_prompt_text(prompt, "\nworld", max_chars=20)

        self.assertTrue(inserted)
        self.assertEqual(prompt.value, "hello\nworld")
        self.assertEqual(prompt.cursor_position, len("hello\nworld"))

    def test_tui_running_followup_routes_to_active_turn_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            app = DeepmateTuiApp(state)
            messages = []
            app._write = messages.append
            app._refresh_footer = lambda: None
            app._running_turn = True
            state.followup_buffer = TurnFollowupBuffer()
            state.active_followup_turn_id = state.followup_buffer.start_turn()

            handled = app._route_running_prompt("/followup 不要改 public API")

            self.assertTrue(handled)
            self.assertEqual(state.followup_buffer.pending_count(), 1)
            self.assertEqual(messages[-1].kind, "followup")
            self.assertEqual(messages[-1].title, "follow-up")
            self.assertEqual(messages[-1].body, "不要改 public API")

    def test_tui_running_plain_input_adds_to_current_turn_before_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            app = DeepmateTuiApp(state)
            messages = []
            app._write = messages.append
            app._refresh_footer = lambda: None
            app._running_turn = True
            state.followup_buffer = TurnFollowupBuffer()
            state.active_followup_turn_id = state.followup_buffer.start_turn()
            app._session_turns[state.session.session_id] = _TurnRun(
                session_id=state.session.session_id,
                started_at=monotonic(),
                followup_buffer=state.followup_buffer,
                followup_turn_id=state.active_followup_turn_id,
            )

            handled = app._route_running_prompt("普通追问默认追加当前轮")

            self.assertTrue(handled)
            self.assertEqual(state.followup_buffer.pending_count(), 1)
            self.assertEqual(app._prompt_queue.pending, [])
            self.assertEqual(messages[-1].kind, "followup")
            self.assertEqual(messages[-1].body, "普通追问默认追加当前轮")

    def test_tui_running_plain_input_queues_after_answer_is_visible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            app = DeepmateTuiApp(state)
            messages = []
            app._write = messages.append
            app._refresh_footer = lambda: None
            app._running_turn = True
            state.followup_buffer = TurnFollowupBuffer()
            state.active_followup_turn_id = state.followup_buffer.start_turn()
            app._session_turns[state.session.session_id] = _TurnRun(
                session_id=state.session.session_id,
                started_at=monotonic(),
                followup_buffer=state.followup_buffer,
                followup_turn_id=state.active_followup_turn_id,
                answer_visible=True,
            )

            handled = app._route_running_prompt("这是下一轮问题")

            self.assertTrue(handled)
            self.assertEqual(state.followup_buffer.pending_count(), 0)
            self.assertEqual(app._prompt_queue.pending, ["这是下一轮问题"])
            self.assertEqual(messages[-1].title, "queued")
            self.assertIn("finishing the previous turn", messages[-1].body)
            self.assertIn("这是下一轮问题", messages[-1].body)

    def test_tui_running_queue_command_bypasses_current_turn_followup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            app = DeepmateTuiApp(state)
            messages = []
            app._write = messages.append
            app._refresh_footer = lambda: None
            app._running_turn = True
            state.followup_buffer = TurnFollowupBuffer()
            state.active_followup_turn_id = state.followup_buffer.start_turn()

            handled = app._route_running_prompt("/queue 等当前结束后处理")

            self.assertTrue(handled)
            self.assertEqual(app._prompt_queue.pending, ["等当前结束后处理"])
            self.assertEqual(state.followup_buffer.pending_count(), 0)
            self.assertEqual(messages[-1].title, "queued")
            self.assertIn("等当前结束后处理", messages[-1].body)

    def test_tui_running_immediate_tab_command_does_not_become_followup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            app = DeepmateTuiApp(state)
            messages = []
            app._write = messages.append
            app._refresh_footer = lambda: None
            app._show_detail = lambda title, content: messages.append(
                TuiMessage(kind="file", title=title, body=content)
            )
            app._running_turn = True
            state.followup_buffer = TurnFollowupBuffer()
            state.active_followup_turn_id = state.followup_buffer.start_turn()
            app._session_turns[state.session.session_id] = _TurnRun(
                session_id=state.session.session_id,
                started_at=monotonic(),
                followup_buffer=state.followup_buffer,
                followup_turn_id=state.active_followup_turn_id,
            )
            app._current_tab_title = "detail"
            app._current_tab_content = "cached content"

            app._submit_prompt("/detail")

            self.assertEqual(state.followup_buffer.pending_count(), 0)
            self.assertEqual(app._prompt_queue.pending, [])
            self.assertEqual(messages[0].title, "detail")
            self.assertEqual(messages[0].body, "cached content")
            self.assertEqual(messages[-1].title, "tab")

    def test_tui_running_followup_falls_back_to_queue_when_turn_is_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            app = DeepmateTuiApp(state)
            messages = []
            app._write = messages.append
            app._refresh_footer = lambda: None
            app._running_turn = True
            state.followup_buffer = TurnFollowupBuffer()
            state.active_followup_turn_id = "closed"

            handled = app._route_running_prompt("/followup 补充到下一轮")

            self.assertTrue(handled)
            self.assertEqual(app._prompt_queue.pending, ["补充到下一轮"])
            self.assertEqual(messages[-1].title, "queued")
            self.assertIn("补充到下一轮", messages[-1].body)

    def test_tui_approval_pending_followup_adds_context_to_current_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            app = DeepmateTuiApp(state)
            messages = []
            app._write = messages.append
            app._refresh_footer = lambda: None
            app._running_turn = True
            app._pending_approval = object()
            state.followup_buffer = TurnFollowupBuffer()
            state.active_followup_turn_id = state.followup_buffer.start_turn()

            app._submit_prompt("/followup 等待审批后再处理")

            self.assertEqual(app._prompt_queue.pending, [])
            self.assertEqual(state.followup_buffer.pending_count(), 1)
            self.assertEqual(messages[-1].kind, "followup")
            self.assertEqual(messages[-1].title, "follow-up")
            self.assertEqual(messages[-1].body, "等待审批后再处理")

    def test_tui_approval_pending_plain_prompt_does_not_auto_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            app = DeepmateTuiApp(state)
            messages = []
            app._write = messages.append
            app._refresh_footer = lambda: None
            app._running_turn = True
            app._pending_approval = object()

            app._submit_prompt("fix this bug")

            self.assertEqual(app._prompt_queue.pending, [])
            self.assertEqual(messages[-1].title, "approval pending")
            self.assertIn("Choose allow or deny first", messages[-1].body)
            self.assertIn("/queue <text>", messages[-1].body)

    def test_tui_enqueue_prompt_reports_full_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            app = DeepmateTuiApp(state)
            app._prompt_queue = TuiPromptQueue(max_size=1)
            messages = []
            app._write = messages.append
            app._refresh_footer = lambda: None

            self.assertTrue(app._enqueue_prompt("first"))
            self.assertFalse(app._enqueue_prompt("second"))

            self.assertEqual(app._prompt_queue.pending, ["first"])
            self.assertEqual(messages[-1].title, "queue full")

    def test_tui_queue_paused_message_includes_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            messages = []
            app._write = messages.append
            app._refresh_footer = lambda: None
            app._clear_live_work = lambda *_args, **_kwargs: None
            app._prompt_queue.enqueue("next task")
            app._turn_failed = True

            app._mark_turn_idle()

            self.assertEqual(messages[-1].title, "queue paused")
            self.assertIn("Denied or failed operation paused", messages[-1].body)
            self.assertIn("Reason: turn failed.", messages[-1].body)
            self.assertIn("/resume-queue", messages[-1].body)

    def test_tui_empty_session_detection_and_footer_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            app = DeepmateTuiApp(state)

            self.assertFalse(_transcript_has_items(state))
            self.assertEqual(app._status_label(), "")
            self.assertNotIn("F1 Help", app._status_label())
            messages = []
            app._write = messages.append
            app._write_start_message()
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[-1].kind, "welcome")
            self.assertIn("Deepmate", messages[-1].body)
            self.assertIn("/commands", messages[-1].body)
            app.sidebar_visible = False
            self.assertEqual(app._status_label(), "")
            app._running_turn = True
            self.assertEqual(
                app._status_label(),
                "stub-main  │  ⠋ running  │  Esc interrupt",
            )
            app._running_turn = False
            state.behavior_runtime = behavior_runtime_for_session(
                data_dir=workspace / "var",
                workspace=workspace,
                profile=state.profile,
                session_id=state.session.session_id,
            )
            state.behavior_runtime.set_computer_use(True)
            self.assertEqual(app._status_label(), "stub-main  │  computer on")
            state.behavior_runtime.set_computer_use(False)

            state.transcript.append_item(
                ModelConversationItem(message=Message(role=MessageRole.USER, content="hi"))
            )
            self.assertTrue(_transcript_has_items(state))
            messages.clear()
            app._write_start_message()
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[-1].kind, "welcome")
            self.assertIn("Deepmate", messages[-1].body)

    def test_workspace_diff_renders_summary_and_raw_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            calls = []

            def fake_run(args, **kwargs):
                calls.append(args)

                class Result:
                    returncode = 0
                    stderr = ""

                    def __init__(self, stdout: str) -> None:
                        self.stdout = stdout

                if "--short" in args:
                    return Result(" M src/app.py\n?? notes.md\n")
                if "--numstat" in args:
                    return Result("2\t1\tsrc/app.py\n")
                return Result("diff --git a/src/app.py b/src/app.py\n+print('ok')\n")

            with patch("deepmate.channels.tui.files.subprocess.run", fake_run):
                report = workspace_diff(workspace)

            self.assertIn("changed files: 2", report)
            self.assertIn("modified: src/app.py (+2/-1)", report)
            self.assertIn("untracked: notes.md", report)
            self.assertIn("Raw diff", report)
            self.assertIn("+print('ok')", report)

    def test_workspace_diff_handles_renamed_files_numstat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)

            def fake_run(args, **kwargs):
                class Result:
                    returncode = 0
                    stderr = ""

                    def __init__(self, stdout: str) -> None:
                        self.stdout = stdout

                if "--short" in args:
                    return Result("R  old name.py -> src/new name.py\n")
                if "--numstat" in args:
                    return Result("4\t1\told name.py => src/new name.py\n")
                return Result("diff --git a/src/new name.py b/src/new name.py\n")

            with patch("deepmate.channels.tui.files.subprocess.run", fake_run):
                report = workspace_diff(workspace)

            self.assertIn("renamed: src/new name.py (+4/-1)", report)

    def test_tui_task_command_renders_task_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            controller = TaskSessionController(workspace)
            controller.enable(TaskStage.PLAN)
            state.task_controller = controller

            result = handle_tui_command("/task", state)

            self.assertTrue(result.handled)
            self.assertEqual(result.messages[0].kind, "task")
            self.assertIn("stage: plan", result.messages[0].body)
            self.assertNotIn("<task_context>", result.messages[0].body)
            self.assertIn("Task Mode", result.messages[0].preview)
            self.assertIn("task/plan.md", result.messages[0].preview)

    def test_tui_commands_open_command_palette_tab(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)

            result = handle_tui_command("/commands", state)

            self.assertTrue(result.handled)
            self.assertEqual(result.messages[0].title, "command palette")
            self.assertIn("content tab", result.messages[0].body)
            self.assertIn("/diff - Open workspace diff", result.messages[0].preview)
            self.assertIn("/pet - Show desktop pet status", result.messages[0].preview)
            self.assertIn("/close-tab - Close the current content tab", result.messages[0].preview)
            self.assertNotIn("/help", result.messages[0].preview)
            self.assertNotIn("/?", result.messages[0].preview)
            self.assertIn("task/plan", result.messages[0].preview)

    def test_tui_pet_commands_update_profile_and_request_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace, data_dir=workspace / "var")
            state.pet_state_store = PetStateStore.in_data_dir(workspace / "var")

            selected = handle_tui_command("/pet select cat", state)
            learning = handle_tui_command("/pet learning standard", state)
            bubble = handle_tui_command("/pet bubble frugal", state)
            started = handle_tui_command("/pet on", state)
            status = handle_tui_command("/pet", state)

            profile = state.pet_state_store.load_profile()
            self.assertTrue(selected.handled)
            self.assertTrue(learning.handled)
            self.assertTrue(bubble.handled)
            self.assertEqual(profile.pet_id, "cat-lazy")
            self.assertEqual(profile.learning_mode, "standard")
            self.assertEqual(profile.bubble_generation, "frugal")
            self.assertNotIn("sources", state.pet_state_store.load_learning_state())
            self.assertIn("no external learning sources", learning.messages[0].body)
            self.assertIn("pet_start_requested=true", started.messages[0].refs)
            self.assertIn("Desktop pet", status.messages[0].preview)

    def test_tui_command_palette_filters_and_completes_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))

            class Panel:
                display = False
                body = ""

                def update(self, body: str) -> None:
                    self.body = body

            class Prompt:
                value = "/m"
                cursor_position = 0

            panel = Panel()
            prompt = Prompt()

            def fake_query(selector, _widget_type=None):
                if selector == "#command-hints":
                    return panel
                if selector == "#prompt-input":
                    return prompt
                raise AssertionError(selector)

            app.query_one = fake_query

            app._update_command_hints("/m")

            self.assertTrue(panel.display)
            self.assertIn("/mcp", panel.body)
            self.assertIn("›", panel.body)

            app._command_hint_index = 1
            app._complete_selected_command()

            self.assertTrue(prompt.value.startswith("/"))
            self.assertEqual(prompt.cursor_position, len(prompt.value))

    def test_tui_command_palette_arrow_keys_move_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            app._command_hint_matches = (("/model", "choose model"), ("/mcp", "list mcp"))
            app._command_hint_index = 0
            renders = []
            app._render_command_hints = lambda: renders.append(app._command_hint_index)

            class Event:
                key = "down"
                stopped = False
                prevented = False

                def prevent_default(self) -> None:
                    self.prevented = True

                def stop(self) -> None:
                    self.stopped = True

            event = Event()

            app.on_key(event)

            self.assertTrue(event.prevented)
            self.assertTrue(event.stopped)
            self.assertEqual(app._command_hint_index, 1)
            self.assertEqual(renders, [1])

    def test_tui_command_palette_enter_does_not_swallow_exact_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            app._command_hint_matches = (("/resume-queue", "resume queue"),)
            app._command_hint_index = 0

            class Prompt:
                value = "/resume"
                id = "prompt-input"

            class Event:
                key = "enter"
                stopped = False
                prevented = False

                def prevent_default(self) -> None:
                    self.prevented = True

                def stop(self) -> None:
                    self.stopped = True

            event = Event()
            prompt = Prompt()
            app.query_one = lambda selector, _widget_type=None: prompt
            submitted = []
            app._submit_prompt_editor = lambda: submitted.append(prompt.value)

            app._handle_prompt_editor_enter()

            self.assertEqual(prompt.value, "/resume")
            self.assertEqual(submitted, ["/resume"])

    def test_tui_command_completion_inserts_command_not_usage_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))

            class Panel:
                display = False
                body = ""

                def update(self, value: str) -> None:
                    self.body = value

            panel = Panel()

            class Prompt:
                value = "/op"
                cursor_position = 3

            prompt = Prompt()

            def fake_query(selector, _widget_type=None):
                if selector == "#command-hints":
                    return panel
                if selector == "#prompt-input":
                    return prompt
                raise AssertionError(selector)

            app.query_one = fake_query
            app._update_command_hints("/op")

            app._complete_selected_command()

            self.assertEqual(prompt.value, "/open ")
            self.assertNotIn("<path>", prompt.value)

    def test_tui_open_diff_and_status_use_tab_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            (workspace / "src").mkdir()
            (workspace / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
            state = _state(workspace)
            state.runtime_stats.record(
                "runtime step 1; input_pressure=0.125; estimated_input_tokens=1000; "
                "context_remaining_input_tokens=7000; actual_input_tokens=900; "
                "output_tokens=80; cache_hit_ratio=0.500"
            )

            opened = handle_tui_command("/open src/app.py", state)
            status = handle_tui_command("/status", state)
            diff = handle_tui_command("/diff", state)

            self.assertTrue(opened.handled)
            self.assertEqual(opened.messages[0].kind, "file")
            self.assertIn("Opened src/app.py", opened.messages[0].body)
            self.assertIn("print('ok')", opened.messages[0].preview)
            self.assertTrue(status.handled)
            self.assertIn("content tab", status.messages[0].body)
            self.assertIn("Capabilities", status.messages[0].preview)
            self.assertIn("tools: unavailable", status.messages[0].preview)
            self.assertIn("input pressure: 12%", status.messages[0].preview)
            self.assertIn("context remaining input tokens: 7000", status.messages[0].preview)
            self.assertTrue(diff.handled)
            self.assertEqual(diff.messages[0].kind, "diff")
            self.assertTrue(diff.messages[0].preview)
            self.assertIn("No git diff is available", diff.messages[0].preview)

    def test_tui_open_command_supports_offset_continuation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            (workspace / "log.txt").write_text("0123456789abcdef", encoding="utf-8")
            state = _state(workspace)

            opened = handle_tui_command("/open log.txt --offset 6 --limit 4", state)

            self.assertTrue(opened.handled)
            self.assertIn("bytes 6-10 of 16", opened.messages[0].body)
            self.assertIn("6789", opened.messages[0].preview)
            self.assertIn("earlier content omitted", opened.messages[0].preview)
            self.assertIn("/open log.txt --offset 10", opened.messages[0].preview)

    def test_tui_status_reports_capability_visibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            registry = NativeToolRegistry(
                (
                    *workspace_filesystem_tools(workspace, include_write_tools=True),
                    *shell_tools(workspace, shell_enabled=False, network_enabled=False),
                )
            )
            state = _state(workspace)
            state.native_tools = _hide_native_tool_schemas(
                registry,
                (RUN_SHELL_COMMAND_TOOL_NAME,),
            )
            state.tool_schemas = state.native_tools.schemas()

            status = handle_tui_command("/status", state)
            app = DeepmateTuiApp(state)

            self.assertIn("workspace read: enabled", status.messages[0].preview)
            self.assertIn(
                "workspace write: available with approval",
                status.messages[0].preview,
            )
            self.assertIn(
                "shell: available on explicit shell request with approval",
                status.messages[0].preview,
            )
            self.assertEqual(app._status_label(), "")

    def test_tui_file_preview_uses_auto_render_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            opened = []
            app._open_content_tab = lambda title, content, *, render_mode="auto": opened.append(
                (title, content, render_mode)
            )

            app._append_messages(
                (TuiMessage(kind="file", title="src/app.py", body="Opened.", preview="print('ok')\n"),)
            )

            self.assertEqual(opened, [("src/app.py", "print('ok')\n", "auto")])
            self.assertEqual(app._main_messages[-1].title, "src/app.py")

    def test_tui_append_messages_clears_live_work_without_completion_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            app._write = lambda message: app._main_messages.append(message)
            app._render_active_tab = lambda: None
            app._turn_started_at = monotonic()
            app._start_live_work("Preparing context...")

            app._append_messages(
                (
                    TuiMessage(
                        kind="tool shell",
                        title="run_shell_command",
                        body="Completed.",
                        status="ok",
                        preview="full output",
                    ),
                    TuiMessage(kind="assistant", title="assistant", body="Done."),
                ),
                started_at=app._turn_started_at,
                finish_turn=True,
            )

            self.assertFalse(
                any(message.status == "live" for message in app._main_messages)
            )
            self.assertFalse(
                any(
                    message.title == "done" or message.status == "summary"
                    for message in app._main_messages
                )
            )
            self.assertEqual(app._current_tab_title, "run_shell_command")
            self.assertEqual(app._current_tab_content, "full output")
            self.assertEqual(app._main_messages[-1].kind, "assistant")

    def test_tui_verbose_command_toggles_reasoning_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            app._write = lambda message: app._main_messages.append(message)
            self.assertFalse(app._show_reasoning_stream)

            self.assertTrue(app._handle_verbose_command("/verbose"))
            self.assertTrue(app._show_reasoning_stream)
            self.assertTrue(app._handle_verbose_command("/verbose"))
            self.assertFalse(app._show_reasoning_stream)

            # Explicit on/off is idempotent.
            app._handle_verbose_command("/verbose on")
            self.assertTrue(app._show_reasoning_stream)
            app._handle_verbose_command("/verbose on")
            self.assertTrue(app._show_reasoning_stream)
            app._handle_verbose_command("/verbose off")
            self.assertFalse(app._show_reasoning_stream)

            # Non-verbose prompts are not intercepted.
            self.assertFalse(app._handle_verbose_command("/status"))

    def test_tui_token_stream_flush_renders_and_throttles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            app._render_active_tab = lambda: None
            app._safe_refresh_footer_throttled = lambda: None
            sid = app.state.session.session_id
            # Simulate turn start pre-creating the buffer.
            app._stream_pending[sid] = []
            app._stream_content[sid] = ""
            app._stream_reasoning[sid] = ""

            # Two worker fragments arrive before a single flush tick.
            app._stream_tokens_from_worker("Hel", "")
            app._stream_tokens_from_worker("lo", "")
            self.assertEqual(len(app._stream_pending[sid]), 2)

            app._flush_token_stream()

            # One flush drains both fragments into the live cell.
            self.assertEqual(app._stream_content[sid], "Hello")
            self.assertEqual(app._stream_pending[sid], [])
            self.assertEqual(app._live_status_text, "Hello")
            live = [m for m in app._main_messages if m.status == "live"]
            self.assertEqual(len(live), 1)
            self.assertEqual(live[0].body, "Hello")

            # A no-op flush with an empty buffer leaves the text intact.
            app._flush_token_stream()
            self.assertEqual(app._live_status_text, "Hello")

    def test_tui_token_stream_cleared_on_turn_idle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            app._render_active_tab = lambda: None
            sid = app.state.session.session_id
            app._stream_pending[sid] = [("hi", "")]
            app._stream_content[sid] = "hi"
            app._stream_reasoning[sid] = ""

            app._mark_turn_idle(sid)

            self.assertNotIn(sid, app._stream_pending)
            self.assertNotIn(sid, app._stream_content)
            self.assertNotIn(sid, app._stream_reasoning)

    def test_tui_final_callback_does_not_clear_live_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            app = DeepmateTuiApp(state)
            app._render_active_tab = lambda: None
            app._refresh_content_tabs = lambda: None
            app._turn_started_at = monotonic()
            app._start_live_work("Preparing context...")
            app._session_turns[state.session.session_id] = _TurnRun(
                session_id=state.session.session_id,
                started_at=app._turn_started_at,
            )

            app._append_final_messages_for_session(
                state.session.session_id,
                (TuiMessage(kind="assistant", title="assistant", body="Done."),),
            )

            self.assertTrue(
                any(message.status == "live" for message in app._main_messages)
            )
            self.assertFalse(any(message.title == "done" for message in app._main_messages))
            self.assertEqual(app._main_messages[-2].kind, "assistant")

    def test_tui_turn_result_inserts_after_own_prompt_when_next_prompt_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            app = DeepmateTuiApp(state)
            app._render_active_tab = lambda: None
            app._refresh_content_tabs = lambda: None
            app._clear_live_work = lambda *_args, **_kwargs: None
            started_at = monotonic()
            anchor_id = _turn_anchor_id(state.session.session_id, started_at)
            app._main_messages = [
                TuiMessage(
                    kind="user",
                    title="you",
                    body="问题1",
                    refs=(f"turn_anchor={anchor_id}",),
                ),
                TuiMessage(kind="user", title="you", body="问题2"),
            ]
            app._session_turns[state.session.session_id] = _TurnRun(
                session_id=state.session.session_id,
                started_at=started_at,
                anchor_id=anchor_id,
            )

            app._append_messages(
                (
                    TuiMessage(kind="user", title="you", body="问题1"),
                    TuiMessage(kind="assistant", title="assistant", body="答复1"),
                ),
                session_id=state.session.session_id,
            )

            self.assertEqual(
                [(message.kind, message.body) for message in app._main_messages],
                [
                    ("user", "问题1"),
                    ("assistant", "答复1"),
                    ("user", "问题2"),
                ],
            )

    def test_tui_final_callback_deduplicates_worker_completion_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            app = DeepmateTuiApp(state)
            app._render_active_tab = lambda: None
            app._refresh_content_tabs = lambda: None
            app._clear_live_work = lambda *_args, **_kwargs: None
            started_at = monotonic()
            anchor_id = _turn_anchor_id(state.session.session_id, started_at)
            app._main_messages = [
                TuiMessage(
                    kind="user",
                    title="you",
                    body="问题1",
                    refs=(f"turn_anchor={anchor_id}",),
                )
            ]
            app._session_turns[state.session.session_id] = _TurnRun(
                session_id=state.session.session_id,
                started_at=started_at,
                anchor_id=anchor_id,
            )
            messages = (
                TuiMessage(kind="user", title="you", body="问题1"),
                TuiMessage(kind="assistant", title="assistant", body="答复1"),
            )

            app._append_final_messages_for_session(state.session.session_id, messages)
            app._append_messages(messages, session_id=state.session.session_id)

            self.assertEqual(
                [(message.kind, message.body) for message in app._main_messages],
                [
                    ("user", "问题1"),
                    ("assistant", "答复1"),
                ],
            )

    def test_tui_final_callback_keeps_later_maintenance_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            app = DeepmateTuiApp(state)
            app._render_active_tab = lambda: None
            app._refresh_content_tabs = lambda: None
            app._clear_live_work = lambda *_args, **_kwargs: None
            started_at = monotonic()
            anchor_id = _turn_anchor_id(state.session.session_id, started_at)
            app._main_messages = [
                TuiMessage(
                    kind="user",
                    title="you",
                    body="问题1",
                    refs=(f"turn_anchor={anchor_id}",),
                )
            ]
            app._session_turns[state.session.session_id] = _TurnRun(
                session_id=state.session.session_id,
                started_at=started_at,
                anchor_id=anchor_id,
            )
            final_messages = (
                TuiMessage(kind="user", title="you", body="问题1"),
                TuiMessage(kind="assistant", title="assistant", body="答复1"),
            )
            completion_messages = (
                *final_messages,
                TuiMessage(kind="task", title="task updated", body="Plan updated."),
            )

            app._append_final_messages_for_session(state.session.session_id, final_messages)
            app._append_messages(
                completion_messages,
                session_id=state.session.session_id,
            )

            self.assertEqual(
                [(message.kind, message.title, message.body) for message in app._main_messages],
                [
                    ("user", "you", "问题1"),
                    ("assistant", "assistant", "答复1"),
                    ("task", "task updated", "Plan updated."),
                ],
            )

    def test_tui_completion_callback_before_final_callback_deduplicates_by_message_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            app = DeepmateTuiApp(state)
            app._render_active_tab = lambda: None
            app._refresh_content_tabs = lambda: None
            app._clear_live_work = lambda *_args, **_kwargs: None
            started_at = monotonic()
            anchor_id = _turn_anchor_id(state.session.session_id, started_at)
            app._main_messages = [
                TuiMessage(
                    kind="user",
                    title="you",
                    body="问题1",
                    refs=(f"turn_anchor={anchor_id}",),
                )
            ]
            app._session_turns[state.session.session_id] = _TurnRun(
                session_id=state.session.session_id,
                started_at=started_at,
                anchor_id=anchor_id,
            )
            final_messages = (
                TuiMessage(kind="user", title="you", body="问题1"),
                TuiMessage(kind="assistant", title="assistant", body="答复1"),
            )
            completion_messages = (
                *final_messages,
                TuiMessage(kind="task", title="task updated", body="Plan updated."),
            )

            app._append_messages(
                completion_messages,
                session_id=state.session.session_id,
            )
            app._append_final_messages_for_session(state.session.session_id, final_messages)

            self.assertTrue(app._session_turns[state.session.session_id].answer_visible)
            self.assertEqual(
                [(message.kind, message.title, message.body) for message in app._main_messages],
                [
                    ("user", "you", "问题1"),
                    ("assistant", "assistant", "答复1"),
                    ("task", "task updated", "Plan updated."),
                ],
            )

    def test_tui_active_content_scroll_targets_main_or_tab(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            calls = []

            class ScrollTarget:
                def __init__(self, name: str) -> None:
                    self.name = name

                def scroll_page_down(self, *, animate: bool) -> None:
                    calls.append((self.name, "down", animate))

                def scroll_page_up(self, *, animate: bool) -> None:
                    calls.append((self.name, "up", animate))

                def scroll_home(self, *, animate: bool) -> None:
                    calls.append((self.name, "home", animate))

                def scroll_end(self, *, animate: bool) -> None:
                    calls.append((self.name, "end", animate))

            chat = ScrollTarget("chat")
            tab = ScrollTarget("tab")

            def fake_query(selector, _widget_type=None):
                if selector == "#chat":
                    return chat
                if selector == "#content-markdown":
                    return tab
                raise AssertionError(selector)

            app.query_one = fake_query
            app._active_tab = "main"
            app._scroll_active_content("page_down")
            app._active_tab = "README.md"
            app._scroll_active_content("page_up")

            self.assertEqual(calls, [("chat", "down", False), ("tab", "up", False)])

    def test_tui_mouse_wheel_scrolls_active_file_tab(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            calls = []

            class ScrollTarget:
                def scroll_down(self, *, animate: bool) -> None:
                    calls.append(("down", animate))

                def scroll_up(self, *, animate: bool) -> None:
                    calls.append(("up", animate))

            class Widget:
                id = "content-markdown"

            class Event:
                widget = Widget()
                prevented = False
                stopped = False

                def prevent_default(self) -> None:
                    self.prevented = True

                def stop(self) -> None:
                    self.stopped = True

            target = ScrollTarget()
            app.query_one = lambda selector, _widget_type=None: target
            app._active_tab = "README.md"
            down = Event()
            up = Event()

            self.assertTrue(app._route_active_content_scroll_event(down, "scroll_down"))
            self.assertTrue(app._route_active_content_scroll_event(up, "scroll_up"))

            self.assertEqual(calls, [("down", False), ("up", False)])
            self.assertTrue(down.prevented)
            self.assertTrue(down.stopped)
            self.assertTrue(up.prevented)
            self.assertTrue(up.stopped)

    def test_tui_pet_start_failure_reports_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            messages = []
            app._write = messages.append

            class Stderr:
                def read(self, limit: int) -> str:
                    return "Desktop pet frontend is not installed"

            class Process:
                returncode = 2
                stderr = Stderr()

                def poll(self):
                    return self.returncode

            app._pet_process = Process()

            app._report_pet_start_result()

            self.assertEqual(messages[-1].kind, "error")
            self.assertIn("Desktop pet frontend is not installed", messages[-1].body)

    def test_tui_pet_start_command_uses_deepmate_pet_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "var"
            command = _pet_start_command(data_dir)

            self.assertIsNotNone(command)
            self.assertEqual(command, [sys.executable, "-m", "deepmate", "--pet"])

    def test_tui_diff_reports_empty_workspace_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            with patch(
                "deepmate.channels.tui.commands.workspace_diff",
                return_value="No workspace diff.",
            ):
                diff = handle_tui_command("/diff", state)

            self.assertTrue(diff.handled)
            self.assertEqual(diff.messages[0].body, "No workspace changes detected.")
            self.assertEqual(diff.messages[0].preview, "No workspace diff.")

    def test_workspace_file_matches_searches_beyond_sidebar_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            deep = workspace / "pkg" / "feature"
            deep.mkdir(parents=True)
            (deep / "target.py").write_text("print('ok')\n", encoding="utf-8")

            matches = workspace_file_matches(workspace, "target", limit=5)

            self.assertIn("pkg/feature/target.py", matches)

    def test_tui_file_reference_command_opens_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            (workspace / "src").mkdir()
            (workspace / "tests").mkdir()
            (workspace / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
            (workspace / "tests" / "test_app.py").write_text("def test_ok(): pass\n", encoding="utf-8")
            (workspace / "README.md").write_text("# Demo\n", encoding="utf-8")
            state = _state(workspace)
            app = DeepmateTuiApp(state)
            messages = []
            details = []
            app._write = messages.append
            app._show_detail = lambda title, content: details.append((title, content))

            handled = app._handle_file_reference_command("@app")

            self.assertTrue(handled)
            self.assertEqual(messages[-1].title, "files")
            self.assertIn("@src/app.py", details[-1][1])
            self.assertIn("@tests/test_app.py", details[-1][1])

    def test_tui_content_tabs_track_active_file_and_main(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            app = DeepmateTuiApp(state)
            renders = []
            app._render_active_tab = lambda: renders.append(app._active_tab)

            app._open_content_tab("src/app.py", "print('ok')\n")

            self.assertEqual(app._active_tab, "src/app.py")
            self.assertIn("● app.py", app._content_tabs_label())
            self.assertTrue(app._content_tabs_label().startswith("  main"))

            app._activate_content_tab("main")

            self.assertEqual(app._active_tab, "main")
            self.assertIn("● main", app._content_tabs_label())
            self.assertIn("src/app.py", app._open_tabs)
            self.assertGreaterEqual(len(renders), 2)

            app._active_tab = "src/app.py"
            app._handle_content_tab_click(0)

            self.assertEqual(app._active_tab, "main")

            app._active_tab = "main"
            file_x = app._content_tabs_label().index("app.py")
            app._handle_content_tab_click(file_x)

            self.assertEqual(app._active_tab, "src/app.py")

    def test_tui_content_tabs_keep_main_and_opened_files_visible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            app._render_active_tab = lambda: None
            app._open_content_tab("src/app.py", "print('ok')\n")
            app._open_content_tab("README.md", "# Readme\n")

            label = app._content_tabs_label()

            self.assertIn("main", label)
            self.assertIn("app.py", label)
            self.assertIn("README.md", label)
            self.assertIn("app.py ×", label)
            self.assertIn("README.md ×", label)
            self.assertIn(" ｜ ", label)

            app._hovered_content_tab = "src/app.py"
            self.assertIn("app.py ×", app._content_tabs_label())
            self.assertIn("README.md ×", app._content_tabs_label())

    def test_tui_content_tabs_hide_when_sidebar_closed_and_no_files_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))

            self.assertTrue(app._content_tabs_visible())
            app.sidebar_visible = False
            self.assertFalse(app._content_tabs_visible())

            app._open_tabs["src/app.py"] = _OpenTab("src/app.py", "print('ok')\n")
            self.assertTrue(app._content_tabs_visible())
            self.assertIn("main", app._content_tabs_label())
            self.assertIn("app.py", app._content_tabs_label())

    def test_tui_content_tab_close_marker_closes_file_tab(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            app._render_active_tab = lambda: None
            app._open_content_tab("src/app.py", "print('ok')\n")
            app._hovered_content_tab = "src/app.py"

            label = app._content_tabs_label()
            close_x = label.index("×") + 1
            app._handle_content_tab_click(close_x)

            self.assertEqual(app._active_tab, "main")
            self.assertEqual(app._hovered_content_tab, "")
            self.assertNotIn("src/app.py", app._open_tabs)

    def test_tui_content_tab_close_marker_accounts_for_widget_padding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            app._render_active_tab = lambda: None
            app._open_content_tab("src/app.py", "print('ok')\n")

            event_x = app._content_tabs_label().index("×") + 1
            app._handle_content_tab_click(event_x)

            self.assertEqual(app._active_tab, "main")
            self.assertNotIn("src/app.py", app._open_tabs)

    def test_tui_content_tab_close_marker_does_not_require_hovered_tab(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            app._render_active_tab = lambda: None
            app._open_content_tab("src/app.py", "print('ok')\n")
            app._hovered_content_tab = ""

            close_x = app._content_tabs_label().index("×") + 1
            app._handle_content_tab_click(close_x)

            self.assertEqual(app._active_tab, "main")
            self.assertNotIn("src/app.py", app._open_tabs)

    def test_tui_close_content_tab_uses_safe_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            safe_renders = []
            app._safe_render_active_tab = lambda: safe_renders.append(True)
            app._open_tabs["src/app.py"] = _OpenTab("src/app.py", "print('ok')\n")
            app._active_tab = "src/app.py"

            self.assertTrue(app._close_content_tab("src/app.py"))

            self.assertEqual(safe_renders, [True])
            self.assertEqual(app._active_tab, "main")

    def test_tui_content_tab_hover_controls_close_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            refreshed = []
            app._refresh_content_tabs = lambda: refreshed.append(app._hovered_content_tab)
            app._render_active_tab = lambda: None
            app._open_content_tab("src/app.py", "print('ok')\n")

            file_x = app._content_tabs_label().index("app.py")
            app._set_hovered_content_tab(app._content_tab_at_x(file_x))

            self.assertEqual(app._hovered_content_tab, "src/app.py")
            self.assertIn("app.py ×", app._content_tabs_label())
            self.assertEqual(refreshed[-1], "src/app.py")

            app._set_hovered_content_tab("")

            self.assertEqual(app._hovered_content_tab, "")
            self.assertIn("×", app._content_tabs_label())

    def test_tui_content_tab_click_handles_wide_char_titles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            app._render_active_tab = lambda: None
            app._open_content_tab("文档.md", "# ok\n")

            label = app._content_tabs_label()
            click_x = label.index("文档")
            # Textual click x is in terminal cells, not Python code points.
            cell_x = len(label[:click_x]) + 1
            self.assertEqual(app._content_tab_at_x(cell_x), "文档.md")

    def test_tui_prompt_from_file_tab_returns_to_main(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            app._render_active_tab = lambda: None
            app._open_content_tab("src/app.py", "print('ok')\n")
            app._active_tab = "src/app.py"
            activations = []
            app._activate_content_tab = lambda target: activations.append(target)
            app._write = lambda _message: None
            app.run_worker = lambda *_args, **_kwargs: None

            app._submit_prompt("hello")

            self.assertEqual(activations, ["main"])

    def test_tui_submit_prompt_shows_user_prompt_and_live_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            messages = []
            def record_write(message):
                messages.append(message)
                app._append_main_message(message)

            app._write = record_write
            app._render_active_tab = lambda: None
            app.run_worker = lambda *_args, **_kwargs: None

            app._submit_prompt("写一个 web 页面")

            self.assertEqual(messages[0].kind, "user")
            self.assertEqual(messages[0].body, "写一个 web 页面")
            self.assertEqual(len(messages), 1)
            self.assertEqual(app._main_messages[-1].status, "live")
            self.assertIn("Preparing context", app._main_messages[-1].body)
            self.assertIsNotNone(app._pending_status_message)
            self.assertIn("Preparing context", app._pending_status_message.body)
            self.assertIn("running  │  Esc interrupt", app._status_label())

    def test_tui_live_work_updates_single_message_and_clears(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            app._write = lambda message: app._main_messages.append(message)
            renders = []
            app._render_active_tab = lambda: renders.append("render")
            app._running_turn = True

            app._start_live_work("Preparing context...")
            self.assertIsNotNone(app._pending_status_message)
            app._update_live_status(TuiMessage(kind="status", title="Deepmate", body="正在请求模型", status="model"))
            app._update_live_status(TuiMessage(kind="status", title="Deepmate", body="正在调用工具：search", status="tool"))
            app._update_live_status(TuiMessage(kind="status", title="Deepmate", body="working on", status="model"))

            self.assertEqual(len(app._main_messages), 1)
            self.assertEqual(app._main_messages[0].status, "live")
            self.assertEqual(app._main_messages[0].body, "正在调用工具：search")
            self.assertEqual(app._pending_status_message.body, "正在调用工具：search")
            self.assertIn("running  │  Esc interrupt", app._status_label())
            self.assertNotIn("正在调用工具：search", app._status_label())

            app._clear_live_work()

            self.assertIsNone(app._pending_status_message)
            self.assertEqual(app._main_messages, [])

    def test_tui_worker_exception_clears_live_work_and_hides_internal_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            app._render_active_tab = lambda: None
            app._refresh_footer = lambda: None
            app._write = lambda message: app._append_main_message(message)
            app._running_turn = True
            app._start_live_work("working on")

            app._handle_worker_exception(ValueError("turn checkpoint not found: turn_00002"))

            self.assertIsNone(app._pending_status_message)
            self.assertFalse(any(message.status == "live" for message in app._main_messages))
            self.assertEqual(app._main_messages[-1].kind, "error")
            self.assertEqual(app._main_messages[-1].title, "session state")
            self.assertNotIn("checkpoint", app._main_messages[-1].body.lower())
            self.assertNotIn("turn_00002", app._main_messages[-1].body)

    def test_tui_safe_call_from_thread_ignores_callbacks_after_quit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            called = []
            app._exiting = True

            result = app._safe_call_from_thread(lambda: called.append(True))

            self.assertFalse(result)
            self.assertEqual(called, [])

    def test_tui_request_approval_denies_when_app_is_exiting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            app._exiting = True

            result = app._request_approval("Tool approval", "Allow read?")

            self.assertEqual(result, "deny")

    def test_tui_live_work_stays_after_new_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))

            app._append_main_message(TuiMessage(kind="user", title="you", body="do it"))
            app._append_main_message(
                TuiMessage(
                    kind="status",
                    title="runtime status",
                    body="working on",
                    status="live",
                )
            )
            app._append_main_message(
                TuiMessage(
                    kind="approval",
                    title="Safety approval",
                    body="Needs approval.",
                    status="waiting",
                )
            )

            self.assertEqual([message.kind for message in app._main_messages], ["user", "approval", "status"])
            self.assertEqual(app._main_messages[-1].status, "live")

            app._append_main_message(
                TuiMessage(
                    kind="status",
                    title="runtime status",
                    body="still working",
                    status="live",
                )
            )

            self.assertEqual([message.kind for message in app._main_messages], ["user", "approval", "status"])
            self.assertEqual(app._main_messages[-1].body, "still working")

    def test_tui_session_tabs_keep_order_and_route_clicks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            app = DeepmateTuiApp(state)
            first_id = state.session.session_id
            second = state.session_store.create(
                workspace=workspace,
                profile=state.profile,
                title="second",
            )
            switched = []
            app._remember_session_tab(second.session_id, second.title)
            app._remember_session_tab(first_id, "renamed first")
            app._switch_session = switched.append

            self.assertEqual(app._session_tabs[0], (first_id, "renamed first"))
            self.assertEqual(app._session_tabs[1], (second.session_id, "second"))

            class ClickEvent:
                widget = type(
                    "WidgetRef",
                    (),
                    {"id": _session_button_id(second.session_id)},
                )()
                x = 0

            app.on_click(ClickEvent())

            self.assertEqual(switched, [second.session_id])

    def test_tui_session_spacer_does_not_parse_as_session_id(self) -> None:
        self.assertEqual(_session_id_from_button_id("session-spacer"), "")
        self.assertEqual(_session_id_from_button_id(_session_button_id("abc123")), "abc123")

    def test_tui_session_tabs_keep_active_session_visible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            app = DeepmateTuiApp(state)
            active_id = state.session.session_id
            for index in range(6):
                session = state.session_store.create(
                    workspace=workspace,
                    profile=state.profile,
                    title=f"session {index}",
                )
                app._remember_session_tab(session.session_id, session.title)

            visible_ids = [session_id for session_id, _ in app._visible_session_tabs()]

            self.assertIn(active_id, visible_ids)
            self.assertLessEqual(len(visible_ids), 4)

    def test_tui_restore_main_messages_from_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            state.transcript.append_item(
                ModelConversationItem.from_message(
                    Message(role=MessageRole.SYSTEM, content="hidden")
                )
            )
            state.transcript.append_item(
                ModelConversationItem.from_message(
                    Message(role=MessageRole.USER, content="old question")
                )
            )
            state.transcript.append_item(
                ModelConversationItem.from_message(
                    Message(role=MessageRole.ASSISTANT, content="old answer")
                )
            )
            app = DeepmateTuiApp(state)

            app._restore_main_messages_from_transcript()

            self.assertEqual([message.kind for message in app._main_messages], ["user", "assistant"])
            self.assertEqual(app._main_messages[0].body, "old question")

    def test_tui_transcript_restore_is_bounded(self) -> None:
        items = tuple(
            ModelConversationItem.from_message(
                Message(role=MessageRole.USER, content=f"message {index}")
            )
            for index in range(5)
        )

        messages = _messages_from_transcript_items(items, limit=2)

        self.assertEqual(messages[0].kind, "status")
        self.assertIn("3 older", messages[0].body)
        self.assertEqual([message.body for message in messages[1:]], ["message 3", "message 4"])

    def test_tui_session_title_truncation_keeps_active_more_readable(self) -> None:
        title = "a very long implementation review session"

        self.assertEqual(_short_session_title(title, active=False), "a very long imple…")
        self.assertEqual(_short_session_title(title, active=True), "a very long implementati…")

    def test_tui_compose_mode_collects_multiline_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            app = DeepmateTuiApp(state)
            messages = []
            details = []
            submitted = []
            app._write = messages.append
            app._show_detail = lambda title, content: details.append((title, content))
            app._close_active_tab = lambda: True
            app._refresh_footer = lambda: None
            app._submit_prompt = submitted.append

            self.assertTrue(app._handle_compose_input("/compose"))
            self.assertTrue(app._handle_compose_input("Line one"))
            self.assertTrue(app._handle_compose_input("  indented line"))
            self.assertTrue(app._handle_compose_input("/send"))

            self.assertEqual(submitted, ["Line one\n  indented line"])
            self.assertFalse(app._compose_mode)
            self.assertIn("Line one", details[-1][1])

    def test_tui_session_clone_uses_lineage_handler_and_switches_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            data_dir = workspace / "var"
            state = _state(workspace, data_dir=data_dir)
            state.checkpoint_controller_factory = lambda session: (
                SessionCheckpointController.in_data_dir(
                    data_dir,
                    workspace=session.workspace,
                    profile=session.profile.name,
                    session_id=session.session_id,
                )
            )
            source_id = state.session.session_id

            result = handle_tui_command("/session clone 'Branch title", state)

            self.assertTrue(result.handled)
            self.assertEqual(result.messages[0].title, "/session clone 'Branch title")
            self.assertIn("Created session clone", result.messages[0].body)
            self.assertNotEqual(state.session.session_id, source_id)
            self.assertEqual(state.session.title, "Branch title")
            self.assertEqual(state.session.parent_session_id, source_id)
            self.assertEqual(state.session.fork_kind, "clone")
            self.assertIsNotNone(state.checkpoint_controller)

    def test_tui_runtime_stats_parse_footer_and_detail(self) -> None:
        stats = TuiRuntimeStats()

        view = stats.record(
            "runtime step 2; input_pressure=0.250; estimated_input_tokens=2000; "
            "context_remaining_input_tokens=8000; model_context_tokens=10000; "
            "actual_input_tokens=1800; "
            "output_tokens=120; cache_hit_ratio=0.750; tool_schema_tokens=300; "
            "tool_output_ratio=0.100"
        )
        stats.record(
            "tool output compacted: source=native; tool=run_shell_command; "
            "kind=log; original_tokens=6000; compacted_tokens=900; ref=tool-output:1"
        )

        self.assertEqual(view.title, "step 2")
        self.assertIn("input pressure: 25%", view.body)
        self.assertIn("step 2", stats.footer_summary())
        self.assertEqual(stats.context_window_summary(), "ctx 20% · 8k left")
        self.assertNotIn("cache 75%", stats.footer_summary())
        self.assertIn("compacted run_shell_command 6000->900", stats.footer_summary())
        self.assertIn("context remaining input tokens: 8000", stats.detail_text())
        self.assertIn("model context tokens: 10000", stats.detail_text())
        self.assertIn("tool output ratio: 10%", stats.detail_text())

    def test_tui_context_usage_ratio_and_color_thresholds(self) -> None:
        stats = TuiRuntimeStats()
        self.assertIsNone(stats.context_usage_ratio())
        self.assertIsNone(_context_window_color(None))

        stats.estimated_input_tokens = 2000
        stats.model_context_tokens = 10000
        self.assertAlmostEqual(stats.context_usage_ratio(), 0.20)
        self.assertIsNone(_context_window_color(0.20))
        self.assertEqual(_context_window_color(0.70), "#cdbb7a")
        self.assertEqual(_context_window_color(0.90), "#c98787")

    def test_tui_cache_summary_label(self) -> None:
        stats = TuiRuntimeStats()
        self.assertEqual(stats.cache_summary(), "")
        stats.cache_hit_ratio = 0.0
        self.assertEqual(stats.cache_summary(), "")
        stats.cache_hit_ratio = 0.983
        self.assertEqual(stats.cache_summary(), "cache 98%")

    def test_tui_status_view_uses_short_step_title(self) -> None:
        view = status_view("runtime step 1;input_pressure=0.5")

        self.assertEqual(view.title, "step 1")
        self.assertIn("input pressure: 50%", view.body)

    def test_tui_status_view_renders_tool_finished_transition(self) -> None:
        ok = status_view("tool finished: source=native; tool=read_file; outcome=ok")
        self.assertEqual(ok.status, "tool")
        self.assertIn("read_file", ok.body)
        self.assertIn("✓", ok.body)

        failed = status_view("tool finished: source=mcp; tool=search; outcome=failed")
        self.assertEqual(failed.status, "tool")
        self.assertIn("✗", failed.body)
        self.assertIn("failed", failed.body)

    def test_tui_status_sink_routes_progress_to_live_callback_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            live_messages = []
            main_messages = []
            state.live_status_callback = live_messages.append
            state.status_message_callback = main_messages.append

            sink = _tui_status_sink(state)
            sink("model request started: step=1; model=stub-main")
            sink("runtime step 1; input_pressure=0.250; estimated_input_tokens=2000")

            self.assertEqual(len(live_messages), 1)
            self.assertEqual(live_messages[0].title, "Deepmate")
            self.assertEqual(live_messages[0].body, "working on")
            self.assertNotIn("stub-main", live_messages[0].body)
            self.assertNotIn("step", live_messages[0].body.lower())
            self.assertEqual(main_messages, [])
            self.assertEqual(state.runtime_stats.context_window_summary(), "ctx 25% used")

    def test_tui_runtime_context_summary_avoids_false_zero(self) -> None:
        stats = TuiRuntimeStats()
        stats.context_remaining_input_tokens = 883_700

        self.assertEqual(stats.context_window_summary(), "ctx 883.7k left")

        stats.input_pressure = 0
        self.assertEqual(stats.context_window_summary(), "ctx 883.7k left")

        stats.input_pressure = 0.003
        self.assertEqual(stats.context_window_summary(), "ctx 883.7k left")

    def test_tui_markdown_is_rendered_for_readability(self) -> None:
        rendered = _readable_markdown("# Title\n\n1. **Do this**\n- `path`")

        self.assertIn("Title", rendered)
        self.assertIn("1. Do this", rendered)
        self.assertIn("• path", rendered)
        self.assertNotIn("**", rendered)

    def test_tui_file_tabs_render_by_file_type(self) -> None:
        markdown = _preview_tab_content("README.md", "# Title\n\nBody")
        python = _preview_tab_content("src/app.py", "print('ok')\n")
        gitignore = _preview_tab_content(".gitignore", "*.pyc\n# cache\n")
        fenced = _preview_tab_content("notes.txt", "```bad\n")

        self.assertTrue(markdown.startswith("# Title"))
        self.assertTrue(python.startswith("```python\n"))
        self.assertIn("print('ok')", python)
        self.assertTrue(gitignore.startswith("```text\n"))
        self.assertIn("*.pyc", gitignore)
        self.assertIn("``\\`bad", fenced)

    def test_tui_find_in_content_preview_reports_matching_lines(self) -> None:
        preview, total = _find_in_content_preview(
            "one\nTODO first\nthree\ntodo second",
            "todo",
            title="notes.md",
        )

        self.assertEqual(total, 2)
        self.assertIn("Matches for 'todo' in notes.md", preview)
        self.assertIn("2: TODO first", preview)
        self.assertIn("4: todo second", preview)

    def test_tui_find_command_searches_active_content_tab(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            app = DeepmateTuiApp(_state(workspace))
            messages = []
            app._write = messages.append
            app._open_tabs["notes.md"] = type(
                "Tab",
                (),
                {
                    "title": "notes.md",
                    "content": "intro\nTODO fix this\nend\n",
                    "render_mode": "markdown",
                },
            )()
            app._active_tab = "notes.md"
            app._show_detail = lambda title, content: (
                setattr(app, "_current_tab_title", title),
                setattr(app, "_current_tab_content", content),
            )

            self.assertTrue(app._handle_find_command("/find todo"))

            self.assertEqual(messages[-1].title, "/find")
            self.assertIn("Found 1 matching", messages[-1].body)
            self.assertIn("2: TODO fix this", messages[-1].preview)
            self.assertEqual(app._current_tab_title, "find: todo")

    def test_tui_search_command_requires_network_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)

            result = handle_tui_command("/search deepmate", state)

            self.assertTrue(result.handled)
            self.assertEqual(result.messages[0].kind, "warning")
            self.assertIn("web_search/web_fetch are registered", result.messages[0].body)
            self.assertNotIn("--allow-network", result.messages[0].body)

    def test_tui_search_command_uses_web_search_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            calls = []

            def search(arguments):
                calls.append(dict(arguments))
                return NativeToolResult(
                    content="1. Deepmate\nhttps://example.test\nAgent workbench",
                    refs=("https://example.test",),
                )

            state = _state(workspace)
            state.native_tools = NativeToolRegistry(
                (
                    NativeTool(
                        name="web_search",
                        description="Search web.",
                        input_schema={"type": "object"},
                        handler=search,
                    ),
                )
            )

            result = handle_tui_command("/search deepmate tui", state)

            self.assertTrue(result.handled)
            self.assertEqual(calls[0]["query"], "deepmate tui")
            self.assertEqual(result.messages[0].kind, "file")
            self.assertIn("Search results opened", result.messages[0].body)
            self.assertIn("https://example.test", result.messages[0].preview)

    def test_tui_context_window_label_uses_runtime_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)
            app = DeepmateTuiApp(state)

            self.assertEqual(app._context_window_label(), "ctx --")
            state.runtime_stats.record(
                "runtime step 1; input_pressure=0.420; estimated_input_tokens=82000; "
                "context_remaining_input_tokens=118000; model_context_tokens=200000"
            )

            self.assertEqual(app._context_window_label(), "ctx 41% · 118k left")

    def test_headless_tui_turn_runs_real_runtime_and_task_maintenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            provider = _StubProvider(
                [
                    ModelResponse(content="plan answer"),
                    ModelResponse(content=_task_update_json()),
                ]
            )
            state = _state(workspace, provider=provider)
            controller = TaskSessionController(workspace)
            state.task_controller = controller

            def context_snapshot_factory(profile_ref):
                return build_profile_context_snapshot(
                    workspace=workspace,
                    profile=profile_ref,
                    extra_sections=(render_task_context_section(controller.context()),),
                )

            def task_maintenance(prompt, final_text, current_session, current_runtime):
                stage = controller.active_stage
                self.assertEqual(stage, TaskStage.PLAN)
                update = generate_task_update(
                    provider,
                    model="stub-memory",
                    stage=stage,
                    documents=controller.store.read_documents(),
                    user_prompt=prompt,
                    final_answer=final_text,
                )
                apply_task_update_result(controller.store, update, stage=stage)
                controller.finish_turn(stage)
                return current_runtime

            state.context_snapshot_factory = context_snapshot_factory
            state.task_maintenance_handler = task_maintenance

            updated, messages, exit_requested = run_headless_tui_turn(
                state,
                "task/plan Discuss TUI.",
            )

            self.assertFalse(exit_requested)
            self.assertEqual(updated.task_controller.active_stage, TaskStage.PLAN)
            self.assertIn("plan answer", "\n".join(message.body for message in messages))
            self.assertIn(
                "Updated by TUI maintenance",
                (workspace / "task" / "plan.md").read_text(encoding="utf-8"),
            )
            main_requests = [
                request for request in provider.requests if request.model == "stub-main"
            ]
            self.assertEqual(len(main_requests), 1)
            self.assertEqual(
                main_requests[0].conversation[0].message.role,
                MessageRole.SYSTEM,
            )
            self.assertIn("<task_context>", main_requests[0].conversation[0].message.content)

    def test_headless_tui_task_execute_continue_keeps_internal_prompt_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            provider = _StubProvider([ModelResponse(content="execute answer")])
            state = _state(workspace, provider=provider)
            controller = TaskSessionController(workspace)
            controller.store.ensure()
            controller.store.write_plan(_execution_plan_md())
            state.task_controller = controller

            evaluation = ExecuteEvaluation(
                decision=ExecuteDecision.CONTINUE,
                reason="verification still missing",
                next_instruction="run focused checks",
            )

            def task_maintenance(prompt, final_text, current_session, current_runtime):
                return (
                    current_runtime,
                    ExecuteLoopUpdate(
                        evaluation=evaluation,
                        turns=1,
                        continuation=continuation_prompt(evaluation),
                    ),
                )

            state.task_maintenance_handler = task_maintenance

            updated, messages, exit_requested = run_headless_tui_turn(
                state,
                "task/execute",
            )

            self.assertFalse(exit_requested)
            bodies = "\n".join(message.body for message in messages)
            self.assertIn("task/execute continuing", bodies)
            self.assertIn("verification still missing", bodies)
            self.assertEqual(
                updated.task_continuations,
                (continuation_prompt(evaluation),),
            )
            self.assertEqual(updated.unconsumed_followups, ())
            self.assertNotIn("Continue task/execute from task/plan.md", bodies)

    def test_tui_streaming_token_sink_receives_deltas_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            provider = _StreamingStubProvider(["Hel", "lo ", "there"])
            state = _state(workspace, provider=provider)
            received = []
            # Mirror how the TUI registers a per-session streaming callback.
            state.token_stream_callback = (
                lambda content, reasoning: received.append((content, reasoning))
            )

            updated, messages, exit_requested = run_headless_tui_turn(state, "hi")

            self.assertFalse(exit_requested)
            # Deltas flowed through provider -> agent_loop -> bridge token sink.
            self.assertEqual([c for c, _ in received], ["Hel", "lo ", "there"])
            self.assertEqual(provider.stream_calls, 1)
            # Final assistant message still carries the full assembled text.
            self.assertIn("Hello there", "\n".join(m.body for m in messages))

    def test_headless_tui_plain_turn_does_not_activate_task_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace, provider=_StubProvider([ModelResponse(content="ok")]))
            state.task_controller = TaskSessionController(workspace)

            updated, messages, exit_requested = run_headless_tui_turn(
                state,
                "ordinary conversation",
            )

            self.assertFalse(exit_requested)
            self.assertIsNone(updated.task_controller.active_stage)
            self.assertFalse((workspace / "task").exists())
            self.assertIn("ok", "\n".join(message.body for message in messages))

    def test_headless_tui_rejects_empty_task_mode_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace)

            class EmptyTaskController:
                def prepare_prompt(self, prompt: str):
                    return type("TaskTurn", (), {"prompt": "   "})()

                def save_cursor(self, *, session_id: str) -> None:
                    raise AssertionError("empty task prompt should not save cursor")

            state.task_controller = EmptyTaskController()

            with self.assertRaisesRegex(ValueError, "task mode produced empty prompt"):
                run_headless_tui_turn(state, "continue task")

    def test_headless_tui_turn_does_not_create_checkpoint_controller(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace, provider=_StubProvider([ModelResponse(content="ok")]))

            def factory(session):
                raise AssertionError("headless turn should not create checkpoint controller")

            state.checkpoint_controller_factory = factory

            updated, messages, exit_requested = run_headless_tui_turn(state, "hello")

            self.assertFalse(exit_requested)
            self.assertIsNone(updated.checkpoint_controller)
            self.assertIn("ok", "\n".join(message.body for message in messages))

    def test_headless_tui_provider_timeout_returns_structured_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            provider = _FailingProvider(
                NetworkError("model request timed out while waiting for response data")
            )
            state = _state(workspace, provider=provider)
            state.provider_retry_policy = ProviderRetryPolicy(
                max_attempts=2,
                initial_delay_seconds=0,
            )

            updated, messages, exit_requested = run_headless_tui_turn(state, "hello")

            rendered = "\n".join(
                f"{message.title}\n{message.body}\n{message.preview}"
                for message in messages
            )
            self.assertFalse(exit_requested)
            self.assertEqual(provider.calls, 2)
            self.assertEqual(updated.turn_index, 1)
            self.assertIn("provider_request_failed", [message.title for message in messages])
            self.assertIn("model connection timed out", rendered.lower())
            self.assertNotIn("something went wrong", rendered.lower())
            self.assertFalse(any(message.kind == "assistant" for message in messages))

    def test_headless_tui_repairs_missing_current_session_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace, provider=_StubProvider([ModelResponse(content="ok")]))
            state.session = replace(state.session, title="Untitled session")
            metadata_path = state.session_store.metadata_path(state.session.session_id)
            metadata_path.unlink()

            updated, messages, exit_requested = run_headless_tui_turn(
                state,
                "hello from recovered session",
            )

            rendered = "\n".join(
                f"{message.title}\n{message.body}\n{message.preview}"
                for message in messages
            )
            self.assertFalse(exit_requested)
            self.assertTrue(metadata_path.is_file())
            self.assertEqual(
                updated.session_store.load(updated.session.session_id).session_id,
                updated.session.session_id,
            )
            self.assertNotIn("FileNotFoundError", rendered)
            self.assertNotIn("/sessions/", rendered)
            self.assertIn("ok", rendered)
            self.assertNotEqual(updated.session.title, "Untitled session")
            self.assertTrue(
                any(
                    event.kind == "tui_session_metadata_repaired"
                    for event in updated.trace_recorder.events
                )
            )

    def test_headless_tui_local_turn_closes_open_remote_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            state = _state(workspace, provider=_StubProvider([ModelResponse(content="ok")]))
            store = RemoteBindingStore.in_data_dir(workspace / "var")
            store.bind_default_session(
                channel="wecom",
                session=state.session,
                bound_from="interactive",
                route_open=True,
            )

            updated, messages, exit_requested = run_headless_tui_turn(
                state,
                "work locally",
            )

            self.assertFalse(exit_requested)
            self.assertIn("ok", "\n".join(message.body for message in messages))
            self.assertFalse(store.get("wecom", "*").route_open)
            self.assertTrue(
                any(event.kind == "remote_route_closed" for event in updated.trace_recorder.events)
            )

    def test_cli_interactive_uses_tui_and_reports_missing_textual(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            _write_cli_config(workspace)
            provider = _StubProvider([ModelResponse(content="unused")])
            stdout = io.StringIO()
            stderr = io.StringIO()

            with (
                patch.dict(os.environ, {"STUB_API_KEY": "test-key"}),
                patch(
                    "deepmate.channels.cli.ChatCompletionsProvider",
                    lambda base_url, api_key: provider,
                ),
                patch("deepmate.channels.tui.bridge.find_spec", lambda name: None),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(("--workspace", str(workspace), "--interactive"))

            self.assertEqual(exit_code, 2)
            self.assertIn("Textual is required", stderr.getvalue())
            self.assertIn("--interactive-legacy", stderr.getvalue())

    def test_cli_interactive_defaults_to_workspace_write_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            _write_cli_config(workspace)
            provider = _StubProvider([ModelResponse(content="unused")])
            captured = {}

            def fake_run_tui_mode(**kwargs):
                captured.update(kwargs)
                return 0

            with (
                patch.dict(os.environ, {"STUB_API_KEY": "test-key"}),
                patch(
                    "deepmate.channels.cli.ChatCompletionsProvider",
                    lambda base_url, api_key: provider,
                ),
                patch("deepmate.channels.cli.run_tui_mode", fake_run_tui_mode),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                exit_code = main(("--workspace", str(workspace), "--interactive"))

            self.assertEqual(exit_code, 0)
            self.assertIsNotNone(captured.get("tool_access_policy"))
            self.assertEqual(
                captured["tool_access_policy"].mode,
                ToolAccessMode.WORKSPACE_WRITE,
            )

    def test_cli_interactive_passes_configured_local_provider_to_tui(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            config_dir = workspace / "config"
            config_dir.mkdir(parents=True)
            (config_dir / "deepmate.yaml").write_text(
                "\n".join(
                    (
                        "runtime:",
                        "  data_dir: var",
                        "provider:",
                        "  default: deepseek",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            (config_dir / "providers.yaml").write_text(
                "\n".join(
                    (
                        "providers:",
                        "  deepseek:",
                        "    base_url: https://api.deepseek.com",
                        "    default_model: deepseek-v4-flash",
                        "  local:",
                        "    base_url: http://127.0.0.1:11555/v1",
                        "    model: qwen3-local",
                        "    api_key_required: false",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            captured = {}
            provider = _StubProvider([ModelResponse(content="unused")])

            def fake_run_tui_mode(**kwargs):
                captured.update(kwargs)
                return 0

            with (
                patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}),
                patch(
                    "deepmate.channels.cli.ChatCompletionsProvider",
                    lambda base_url, api_key: provider,
                ),
                patch("deepmate.channels.cli.run_tui_mode", fake_run_tui_mode),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                exit_code = main(("--workspace", str(workspace), "--interactive"))

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                captured["local_provider_base_url"],
                "http://127.0.0.1:11555/v1",
            )
            self.assertEqual(captured["local_provider_api_key"], "ollama")

    def test_cli_interactive_without_remote_key_waits_for_user_choice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            config_dir = workspace / "config"
            config_dir.mkdir(parents=True)
            (config_dir / "deepmate.yaml").write_text(
                "\n".join(
                    (
                        "runtime:",
                        "  data_dir: var",
                        "provider:",
                        "  default: deepseek",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            (config_dir / "providers.yaml").write_text(
                "\n".join(
                    (
                        "providers:",
                        "  deepseek:",
                        "    base_url: https://api.deepseek.com",
                        "    default_model: deepseek-v4-flash",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            captured = {}
            created = []

            def fake_provider(base_url, api_key):
                created.append((base_url, api_key))
                return _StubProvider([ModelResponse(content="unused")])

            def fake_run_tui_mode(**kwargs):
                captured.update(kwargs)
                return 0

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("deepmate.channels.cli.ChatCompletionsProvider", fake_provider),
                patch("deepmate.channels.cli.run_tui_mode", fake_run_tui_mode),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                exit_code = main(("--workspace", str(workspace), "--interactive"))

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["provider_name"], "deepseek")
            self.assertFalse(captured["provider_api_key_available"])
            self.assertEqual(captured["model"], "deepseek-v4-flash")
            self.assertEqual(captured["remote_provider_name"], "deepseek")
            self.assertFalse(any(api_key == "ollama" for _base_url, api_key in created))

    def test_cli_without_arguments_opens_tui_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            _write_cli_config(workspace)
            captured = {}

            def fake_run_tui_mode(**kwargs):
                captured.update(kwargs)
                return 0

            with (
                patch.dict(os.environ, {"STUB_API_KEY": "test-key"}),
                patch("deepmate.channels.cli.run_tui_mode", fake_run_tui_mode),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                exit_code = main(("--workspace", str(workspace)))

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["workspace"], workspace.resolve())
            self.assertEqual(captured["initial_prompts"], ())

    def test_setup_key_command_saves_key_for_current_tui_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            data_dir = workspace / "var"
            state = _state(workspace, data_dir=data_dir)
            state.provider_api_key_available = False

            result = handle_tui_command("/setup-key test-secret", state)

            self.assertTrue(result.handled)
            self.assertTrue(state.provider_api_key_available)
            self.assertIn("Model API key saved locally.", result.messages[0].body)
            self.assertIn(
                'STUB_API_KEY="test-secret"',
                (data_dir / "secrets" / "providers.env").read_text(encoding="utf-8"),
            )

    def test_safety_approval_callback_does_not_default_allow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = SessionApprovalCache()
            policy = ToolSafetyPolicy(
                workspace=Path(tmp),
                shell_enabled=True,
                network_enabled=True,
                approval_cache=cache,
            )

            denied = policy.check_shell_command("python3 -c 'print(1)'", network="on")
            self.assertFalse(denied.allowed)
            cache.approval_callback = lambda decision: ApprovalDecision.ALLOW_ONCE
            allowed = policy.check_shell_command("python3 -c 'print(1)'", network="on")
            self.assertTrue(allowed.allowed)

    def test_interactive_default_shell_tool_can_be_hidden_but_registered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            registry = NativeToolRegistry(
                (
                    *workspace_filesystem_tools(workspace, include_write_tools=True),
                    *shell_tools(workspace, shell_enabled=False, network_enabled=False),
                )
            )
            hidden = _hide_native_tool_schemas(
                registry,
                (RUN_SHELL_COMMAND_TOOL_NAME,),
            )

            visible = tuple(schema["name"] for schema in hidden.schemas())
            all_names = tuple(schema["name"] for schema in hidden.schemas(include_hidden=True))

        self.assertIn("read_text_file", visible)
        self.assertIn("list_directory", visible)
        self.assertIn("write_text_file", visible)
        self.assertNotIn(RUN_SHELL_COMMAND_TOOL_NAME, visible)
        self.assertIn(RUN_SHELL_COMMAND_TOOL_NAME, all_names)

    def test_local_model_default_tool_schemas_hide_low_frequency_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            registry = NativeToolRegistry(
                (
                    *workspace_filesystem_tools(workspace, include_write_tools=True),
                    *workspace_search_tools(workspace),
                    *workspace_lsp_tools(workspace, server_resolver=lambda _path: None),
                    *workspace_document_tools(workspace),
                    *workspace_artifact_tools(workspace),
                    *workspace_report_tools(workspace),
                    *workspace_diagram_tools(workspace),
                    *shell_tools(workspace, shell_enabled=False, network_enabled=False),
                )
            )

            visible = {
                str(schema.get("name", ""))
                for schema in _default_tool_schemas_for_model(registry, "qwen3:4b")
            }

        self.assertIn("read_text_file", visible)
        self.assertIn("write_text_file", visible)
        self.assertIn("search_files", visible)
        self.assertIn(RUN_SHELL_COMMAND_TOOL_NAME, visible)
        self.assertNotIn("lsp_references", visible)
        self.assertNotIn("read_document", visible)
        self.assertNotIn("render_html_report", visible)
        self.assertNotIn("render_tech_diagram", visible)

    def test_local_model_prompt_can_reveal_hidden_tool_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            registry = NativeToolRegistry(
                (
                    *workspace_filesystem_tools(workspace, include_write_tools=True),
                    *workspace_lsp_tools(workspace, server_resolver=lambda _path: None),
                    *workspace_document_tools(workspace),
                )
            )
            base = _default_tool_schemas_for_model(registry, "qwen3:4b")

            selected = {
                str(schema.get("name", ""))
                for schema in _schemas_with_local_prompt_extras(
                    base,
                    registry,
                    "帮我看这个函数的定义和引用，再读一下需求文档",
                )
            }

        self.assertIn("lsp_definition", selected)
        self.assertIn("lsp_references", selected)
        self.assertIn("lsp_hover", selected)
        self.assertIn("read_document", selected)
        self.assertIn("inspect_table", selected)

    def test_headless_tui_write_prompt_exposes_and_executes_write_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            provider = _StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="write_text_file",
                                id="call_1",
                                arguments={
                                    "path": "notes/todo.md",
                                    "content": "# Todo\n- ship deepmate\n",
                                },
                            ),
                        )
                    ),
                    ModelResponse(content="已创建 notes/todo.md。"),
                ]
            )
            (workspace / "notes").mkdir()
            registry = NativeToolRegistry(
                workspace_filesystem_tools(workspace, include_write_tools=True)
            )
            approvals = []
            state = _state(workspace, provider=provider)
            # Write tools are visible by default now (no prompt-keyword gating);
            # the approval policy is the actual gate.
            state.native_tools = registry
            state.tool_schemas = registry.schemas()
            state.tool_access_policy = ToolAccessPolicy(ToolAccessMode.READ_ONLY)
            state.tool_approval_callback = lambda tool, decision: approvals.append(
                (tool.name, decision.reason)
            ) or True

            updated, messages, exit_requested = run_headless_tui_turn(
                state,
                "请创建文件 notes/todo.md，写入一个 todo 列表",
            )

            self.assertFalse(exit_requested)
            first_request_tools = {
                str(schema.get("name", "")) for schema in provider.requests[0].tool_schemas
            }
            self.assertIn("read_text_file", first_request_tools)
            self.assertIn("write_text_file", first_request_tools)
            self.assertIn(
                (
                    "write_text_file",
                    "Native tool requires workspace write access: write_text_file",
                ),
                approvals,
            )
            self.assertEqual(
                (workspace / "notes" / "todo.md").read_text(encoding="utf-8"),
                "# Todo\n- ship deepmate\n",
            )
            self.assertIn("已创建", "\n".join(message.body for message in messages))
            self.assertEqual(updated.turn_index, 1)

    def test_headless_tui_plain_prompt_keeps_write_schema_visible_but_approval_gated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            provider = _StubProvider([ModelResponse(content="ok")])
            registry = NativeToolRegistry(
                workspace_filesystem_tools(workspace, include_write_tools=True)
            )
            state = _state(workspace, provider=provider)
            state.native_tools = registry
            state.tool_schemas = registry.schemas()

            run_headless_tui_turn(state, "总结一下当前项目结构")

        first_request_tools = {
            str(schema.get("name", "")) for schema in provider.requests[0].tool_schemas
        }
        self.assertIn("read_text_file", first_request_tools)
        self.assertIn("write_text_file", first_request_tools)
        self.assertIn("edit_text_file", first_request_tools)

    def test_headless_tui_skill_install_prompt_exposes_and_installs_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = _workspace(root)
            data_dir = workspace / "var"
            source = root / "source" / "reviewer"
            _write_skill_bundle(source, name="reviewer")
            provider = _StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name=LOAD_SKILL_INSTALLER_TOOLS_NAME,
                                id="call_1",
                                arguments={"reason": "install a community skill"},
                            ),
                        )
                    ),
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name=INSTALL_SKILL_TOOL_NAME,
                                id="call_2",
                                arguments={"source": str(source), "target": "workspace"},
                            ),
                        )
                    ),
                    ModelResponse(content="reviewer skill 已安装。"),
                ]
            )
            store = CapabilityStateStore.in_data_dir(data_dir, "default")
            registry = NativeToolRegistry(skill_installer_tools(workspace, data_dir, store))
            approvals = []
            state = _state(workspace, provider=provider, data_dir=data_dir)
            state.native_tools = registry
            state.tool_schemas = registry.schemas()
            state.capability_state_store = store
            state.tool_access_policy = ToolAccessPolicy(ToolAccessMode.READ_ONLY)
            state.tool_approval_callback = lambda tool, decision: approvals.append(
                tool.name
            ) or True
            state.max_steps = 3

            updated, messages, exit_requested = run_headless_tui_turn(
                state,
                f"请安装 skill {source}",
            )

            first_request_tools = {
                str(schema.get("name", "")) for schema in provider.requests[0].tool_schemas
            }
            second_request_tools = {
                str(schema.get("name", "")) for schema in provider.requests[1].tool_schemas
            }
            self.assertFalse(exit_requested)
            self.assertEqual(
                first_request_tools,
                {INSTALL_SKILL_FROM_REQUEST_TOOL_NAME, LOAD_SKILL_INSTALLER_TOOLS_NAME},
            )
            self.assertIn(INSTALL_SKILL_BUNDLE_TOOL_NAME, second_request_tools)
            self.assertIn(INSTALL_SKILL_TOOL_NAME, second_request_tools)
            self.assertNotIn("inspect_skill_source", first_request_tools)
            self.assertIn("verify_skill_install", second_request_tools)
            self.assertIn("plan_skill_setup", second_request_tools)
            self.assertIn("run_skill_setup", second_request_tools)
            self.assertIn(INSTALL_SKILL_TOOL_NAME, approvals)
            self.assertTrue((workspace / "skills" / "reviewer" / "SKILL.md").is_file())
            self.assertIn("已安装", "\n".join(message.body for message in messages))
            self.assertNotIn("safety step cap", "\n".join(message.body for message in messages))
            self.assertEqual(updated.turn_index, 1)

    def test_headless_tui_skill_setup_prompt_exposes_setup_recovery_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            provider = _StubProvider([ModelResponse(content="ok")])
            store = CapabilityStateStore.in_data_dir(workspace / "var", "default")
            registry = NativeToolRegistry(
                (
                    *skill_installer_tools(workspace, workspace / "var", store),
                    *(
                        replace(tool, exposed_by_default=False)
                        for tool in shell_tools(
                            workspace,
                            shell_enabled=True,
                            network_enabled=False,
                            sandbox_mode=SandboxMode.OFF,
                            approval_cache=SessionApprovalCache(),
                        )
                    ),
                )
            )
            state = _state(workspace, provider=provider, data_dir=workspace / "var")
            state.native_tools = registry
            state.tool_schemas = registry.schemas()

            run_headless_tui_turn(state, "帮我安装 ontology skill，如果需要就运行 setup 安装命令")

        first_request_tools = {
            str(schema.get("name", "")) for schema in provider.requests[0].tool_schemas
        }
        self.assertEqual(
            first_request_tools,
            {INSTALL_SKILL_FROM_REQUEST_TOOL_NAME, LOAD_SKILL_INSTALLER_TOOLS_NAME},
        )
        self.assertNotIn(INSTALL_SKILL_TOOL_NAME, first_request_tools)
        self.assertNotIn(INSTALL_SKILL_BUNDLE_TOOL_NAME, first_request_tools)
        self.assertNotIn("plan_skill_setup", first_request_tools)
        self.assertNotIn("run_skill_setup", first_request_tools)
        self.assertNotIn(RUN_SHELL_COMMAND_TOOL_NAME, first_request_tools)

    def test_headless_tui_browser_loader_exposes_install_recovery_and_executes_after_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            commands = []

            def runner(args, *_args, **_kwargs):
                commands.append(tuple(args))
                return BrowserCommandResult(exit_code=0, stdout="ok\n")

            def which(command: str) -> str | None:
                if command == "npm":
                    return "/usr/local/bin/npm"
                if command == "agent-browser" and commands:
                    return "/usr/local/bin/agent-browser"
                return None

            backend = AgentBrowserBackend(
                workspace,
                session_name="test-session",
                runner=runner,
                which=which,
            )
            approval_cache = SessionApprovalCache()
            registry = NativeToolRegistry(
                browser_loader_tools(backend, approval_cache=approval_cache)
            )
            provider = _StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name=LOAD_BROWSER_TOOLS_NAME,
                                id="call_1",
                                arguments={"reason": "search the web"},
                            ),
                        )
                    ),
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name=INSTALL_BROWSER_BACKEND_TOOL_NAME,
                                id="call_2",
                                arguments={"timeout_seconds": 120},
                            ),
                        )
                    ),
                    ModelResponse(content="浏览器后端已安装。"),
                ]
            )
            approvals = []
            state = _state(workspace, provider=provider)
            state.native_tools = registry
            state.tool_schemas = registry.schemas()
            state.tool_access_policy = ToolAccessPolicy(
                ToolAccessMode.READ_ONLY,
                shell_enabled=True,
            )
            state.approval_cache = approval_cache
            state.max_steps = 3
            state.safety_approval_callback = lambda decision: approvals.append(
                decision.approval_key
            ) or ApprovalDecision.ALLOW_ONCE

            updated, messages, exit_requested = run_headless_tui_turn(
                state,
                "需要使用浏览器搜索网页信息",
            )

        first_request_tools = {
            str(schema.get("name", "")) for schema in provider.requests[0].tool_schemas
        }
        second_request_tools = {
            str(schema.get("name", "")) for schema in provider.requests[1].tool_schemas
        }
        self.assertFalse(exit_requested)
        self.assertEqual(first_request_tools, {LOAD_BROWSER_TOOLS_NAME})
        self.assertIn(INSTALL_BROWSER_BACKEND_TOOL_NAME, second_request_tools)
        self.assertIn("capability:shell", approvals)
        self.assertIn(("npm", "install", "-g", "agent-browser"), commands)
        self.assertIn(("agent-browser", "install"), commands)
        self.assertIn("浏览器后端已安装", "\n".join(message.body for message in messages))
        self.assertEqual(updated.turn_index, 1)

    def test_headless_tui_shell_prompt_exposes_shell_schema_and_requests_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            approval_cache = SessionApprovalCache()
            provider = _StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name=RUN_SHELL_COMMAND_TOOL_NAME,
                                id="call_1",
                                arguments={"command": "printf shell-ok"},
                            ),
                        )
                    ),
                    ModelResponse(content="shell-ok"),
                ]
            )
            registry = NativeToolRegistry(
                shell_tools(
                    workspace,
                    shell_enabled=False,
                    network_enabled=False,
                    sandbox_mode=SandboxMode.OFF,
                    approval_cache=approval_cache,
                )
            )
            tool_approvals = []
            safety_approvals = []
            state = _state(workspace, provider=provider)
            # Shell schema is visible by default now; the safety policy is the gate.
            state.native_tools = registry
            state.tool_schemas = registry.schemas()
            state.tool_access_policy = ToolAccessPolicy(
                ToolAccessMode.READ_ONLY,
                shell_enabled=False,
            )
            state.approval_cache = approval_cache
            state.tool_approval_callback = lambda tool, decision: tool_approvals.append(
                (tool.name, decision.reason)
            ) or True
            state.safety_approval_callback = lambda decision: safety_approvals.append(
                (decision.approval_key, decision.refs)
            ) or ApprovalDecision.ALLOW_ONCE

            updated, messages, exit_requested = run_headless_tui_turn(
                state,
                "我授权你直接处理，在终端执行命令 printf shell-ok",
            )

        first_request_tools = {
            str(schema.get("name", "")) for schema in provider.requests[0].tool_schemas
        }
        self.assertFalse(exit_requested)
        self.assertIn(RUN_SHELL_COMMAND_TOOL_NAME, first_request_tools)
        self.assertEqual(tool_approvals, [])
        self.assertEqual(safety_approvals[0][0], "capability:shell")
        self.assertIn("command=printf shell-ok", safety_approvals[0][1])
        self.assertIn("shell-ok", "\n".join(message.body for message in messages))
        self.assertEqual(updated.turn_index, 1)

    def test_headless_tui_plain_prompt_keeps_shell_schema_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            provider = _StubProvider([ModelResponse(content="ok")])
            registry = NativeToolRegistry(
                replace(tool, exposed_by_default=False)
                for tool in shell_tools(
                    workspace,
                    shell_enabled=True,
                    network_enabled=False,
                    sandbox_mode=SandboxMode.OFF,
                    approval_cache=SessionApprovalCache(),
                )
            )
            state = _state(workspace, provider=provider)
            state.native_tools = registry
            state.tool_schemas = registry.schemas()
            state.tool_access_policy = ToolAccessPolicy(
                ToolAccessMode.READ_ONLY,
                shell_enabled=False,
            )

            run_headless_tui_turn(state, "总结一下当前项目")

        first_request_tools = {
            str(schema.get("name", "")) for schema in provider.requests[0].tool_schemas
        }
        self.assertNotIn(RUN_SHELL_COMMAND_TOOL_NAME, first_request_tools)


def _tool_result(
    name: str,
    request_id: str,
    content: str,
    refs: tuple[str, ...] = (),
    is_error: bool = False,
):
    from deepmate.providers import ModelToolResult

    return ModelToolResult(
        name=name,
        request_id=request_id,
        content=content,
        refs=refs,
        is_error=is_error,
    )


def _message(kind: str, body: str):
    from deepmate.channels.tui.formatters import TuiMessage

    return TuiMessage(kind=kind, title=kind, body=body)


def _pending_approval(title: str):
    from threading import Event

    from deepmate.channels.tui.app import _PendingApproval

    return _PendingApproval(title=title, body="needs approval", event=Event())


def _state(
    workspace: Path,
    provider=None,
    *,
    data_dir: Path | None = None,
) -> TuiRuntimeState:
    session_store = SessionStore.in_directory(workspace / "var" / "sessions")
    profile = ProfileRef(name="default", uri="profiles/default")
    session = session_store.create(workspace=workspace, profile=profile, title="tui test")
    runtime = start_session_runtime(
        start_runtime_activation(
            session_id=session.session_id,
            workspace=workspace,
            profile=profile,
        )
    )
    return TuiRuntimeState(
        provider=provider or _StubProvider([ModelResponse(content="ok")]),
        provider_name="stub",
        provider_api_key_env="STUB_API_KEY",
        provider_api_key_available=True,
        model="stub-main",
        default_model="stub-main",
        upgrade_model="stub-pro",
        workspace=workspace,
        profile=profile,
        session_store=session_store,
        session=session,
        transcript=session_store.transcript_store(session),
        runtime=runtime,
        capability_surface=None,
        native_tools=None,
        native_tool_factory=None,
        mcp_tools=None,
        subagents=None,
        tool_access_policy=ToolAccessPolicy(ToolAccessMode.READ_ONLY),
        tool_schemas=(),
        selected_skill_documents=(),
        mcp_servers=(),
        conversation_budget_policy=None,
        provider_retry_policy=None,
        options={},
        max_steps=2,
        trace_recorder=_TraceRecorder(),
        warning_sink=None,
        data_dir=data_dir,
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


class _StreamingStubProvider:
    """Streaming-capable stub: feeds content deltas, then returns the response."""

    def __init__(self, fragments: list[str]) -> None:
        self.fragments = list(fragments)
        self.stream_calls = 0

    def complete(self, request):
        return ModelResponse(content="".join(self.fragments))

    def complete_stream(self, request, on_delta):
        from deepmate.providers import StreamDelta

        self.stream_calls += 1
        for fragment in self.fragments:
            on_delta(StreamDelta(content=fragment))
        return ModelResponse(content="".join(self.fragments))


class _FailingProvider:
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.calls = 0
        self.requests = []

    def complete(self, request):
        self.calls += 1
        self.requests.append(request)
        raise self.error


class _TraceRecorder:
    def __init__(self) -> None:
        self.events = []

    def record(self, event) -> None:
        self.events.append(event)

    def record_span(self, span) -> None:
        self.events.append(span)


def _workspace(root: Path) -> Path:
    workspace = root / "workspace"
    profile_dir = workspace / "profiles" / "default"
    profile_dir.mkdir(parents=True)
    (workspace / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
    (profile_dir / "identity.md").write_text("Identity.", encoding="utf-8")
    (profile_dir / "soul.md").write_text("Soul.", encoding="utf-8")
    (profile_dir / "user.md").write_text("", encoding="utf-8")
    (profile_dir / "memory.md").write_text("", encoding="utf-8")
    return workspace


def _write_cli_config(workspace: Path) -> None:
    config_dir = workspace / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "deepmate.yaml").write_text(
        "\n".join(
            (
                "runtime:",
                "  data_dir: var",
                "  active_profile: default",
                "providers:",
                "  default: stub",
                "  items:",
                "    stub:",
                "      base_url: https://stub.local/v1",
                "      api_key_env: STUB_API_KEY",
                "      default_model: stub-main",
                "models:",
                "  purposes:",
                "    main:",
                "      model: stub-main",
                "    memory:",
                "      model: stub-memory",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def _write_skill_bundle(path: Path, *, name: str = "reviewer") -> None:
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(
        "\n".join(
            (
                "---",
                f"name: {name}",
                "description: Review implementation quality.",
                "allowed-tools: [read, search]",
                "---",
                "Review the current implementation carefully.",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "references").mkdir()
    (path / "references" / "guide.md").write_text("Reference guide.\n", encoding="utf-8")


def _task_update_json() -> str:
    return (
        "{"
        '"plan_md":"# Current task plan\\n\\nUpdated by TUI maintenance.\\n",'
        '"rolling_summary":["Goal","Done","Current","Decision","Next"],'
        '"timeline_entry":"### 2026-06-07 | TUI\\n- Updated by TUI maintenance.",'
        '"achievement_title":"",'
        '"achievement_md":""'
        "}"
    )


def _execution_plan_md() -> str:
    return (
        "# Current task plan\n\n"
        "## Goal\nKeep executing.\n\n"
        "## Acceptance Contract\n"
        "- [ ] Complete the requested task.\n"
        "- [ ] Verify the result.\n\n"
        "## Execution Plan\n- [ ] Continue.\n\n"
        "## Verification Strategy\n- Run focused checks.\n"
    )


if __name__ == "__main__":
    unittest.main()
