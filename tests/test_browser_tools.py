from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from deepmate.channels.cli import _attach_browser_tools
from deepmate.channels.cli import _attach_tool_output_tools
from deepmate.channels.cli import main
from deepmate.capabilities import from_native_tool_schemas
from deepmate.domain import Message, MessageRole, ProfileRef
from deepmate.providers import ModelToolRequest
from deepmate.providers import ModelResponse
from deepmate.runtime import ToolAccessPolicy, execute_native_tool_request, run_user_turn
from deepmate.tools import (
    BROWSER_CLICK_TOOL_NAME,
    BROWSER_CLOSE_TOOL_NAME,
    BROWSER_FILL_TOOL_NAME,
    BROWSER_OPEN_TOOL_NAME,
    BROWSER_SCREENSHOT_TOOL_NAME,
    BROWSER_SNAPSHOT_TOOL_NAME,
    BROWSER_STATUS_TOOL_NAME,
    BROWSER_WAIT_TOOL_NAME,
    INSTALL_BROWSER_BACKEND_TOOL_NAME,
    LOAD_BROWSER_TOOLS_NAME,
    AgentBrowserBackend,
    BrowserCommandResult,
    NativeToolRegistry,
    RETRIEVE_TOOL_OUTPUT_NAME,
    browser_tools,
    format_browser_install_result,
    format_browser_validation_result,
    install_browser_backend,
    validate_browser_backend,
)
from deepmate.storage import SessionStore, ToolOutputStore


class BrowserToolTests(unittest.TestCase):
    def test_cli_default_exposes_only_browser_loader_in_first_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            _write_cli_config(workspace)
            provider = _StubProvider([ModelResponse(content="browser loader ready")])
            stdout = io.StringIO()
            stderr = io.StringIO()

            with (
                patch.dict(os.environ, {"STUB_API_KEY": "test-key"}),
                patch(
                    "deepmate.channels.cli.ChatCompletionsProvider",
                    lambda base_url, api_key: provider,
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    (
                        "--workspace",
                        str(workspace),
                        "Check a dynamic page.",
                    )
                )

        self.assertEqual(exit_code, 0, stderr.getvalue())
        self.assertIn("browser loader ready", stdout.getvalue())
        request = provider.requests[0]
        schema_names = tuple(schema["name"] for schema in request.tool_schemas)
        self.assertIn(LOAD_BROWSER_TOOLS_NAME, schema_names)
        system_prompt = request.conversation[0].message.content
        self.assertIn("<capability_guidance>", system_prompt)
        self.assertIn("load_browser_tools", system_prompt)
        self.assertNotIn(BROWSER_OPEN_TOOL_NAME, schema_names)
        self.assertNotIn(RETRIEVE_TOOL_OUTPUT_NAME, schema_names)

    def test_cli_exposes_retrieval_schema_when_session_has_saved_tool_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            _write_cli_config(workspace)
            session_store = SessionStore.in_directory(workspace / "var" / "sessions")
            session = session_store.create(
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                title="Has compacted output",
            )
            store = ToolOutputStore.in_data_dir(
                workspace / "var",
                "default",
                session.session_id,
            )
            store.save(
                tool_name="search_content",
                tool_source="native",
                content_kind="plain",
                content="raw compacted output",
                estimated_tokens=10,
                request_id="call_1",
            )
            provider = _StubProvider([ModelResponse(content="ready")])
            stdout = io.StringIO()
            stderr = io.StringIO()

            with (
                patch.dict(os.environ, {"STUB_API_KEY": "test-key"}),
                patch(
                    "deepmate.channels.cli.ChatCompletionsProvider",
                    lambda base_url, api_key: provider,
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    (
                        "--workspace",
                        str(workspace),
                        "--session-id",
                        session.session_id,
                        "Continue from saved output.",
                    )
                )

        self.assertEqual(exit_code, 0, stderr.getvalue())
        schema_names = tuple(schema["name"] for schema in provider.requests[0].tool_schemas)
        self.assertIn(RETRIEVE_TOOL_OUTPUT_NAME, schema_names)

    def test_cli_browser_flag_preloads_concrete_browser_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            _write_cli_config(workspace)
            provider = _StubProvider([ModelResponse(content="browser tools ready")])
            stdout = io.StringIO()
            stderr = io.StringIO()

            with (
                patch.dict(os.environ, {"STUB_API_KEY": "test-key"}),
                patch(
                    "deepmate.channels.cli.ChatCompletionsProvider",
                    lambda base_url, api_key: provider,
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    (
                        "--workspace",
                        str(workspace),
                        "--browser",
                        "Check a dynamic page.",
                    )
                )

        self.assertEqual(exit_code, 0, stderr.getvalue())
        names = tuple(schema["name"] for schema in provider.requests[0].tool_schemas)
        self.assertIn(BROWSER_OPEN_TOOL_NAME, names)
        self.assertIn(BROWSER_STATUS_TOOL_NAME, names)
        self.assertNotIn(LOAD_BROWSER_TOOLS_NAME, names)

    def test_cli_browser_loader_to_missing_backend_diagnostic_is_self_recovering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            _write_cli_config(workspace)
            provider = _StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name=LOAD_BROWSER_TOOLS_NAME,
                                id="call_1",
                                arguments={"reason": "Need dynamic page inspection."},
                            ),
                        )
                    ),
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name=BROWSER_STATUS_TOOL_NAME,
                                id="call_2",
                                arguments={},
                            ),
                        )
                    ),
                    ModelResponse(
                        content=(
                            "The built-in browser backend is not installed, so I cannot "
                            "inspect the page with browser tools in this run."
                        )
                    ),
                ]
            )
            stdout = io.StringIO()
            stderr = io.StringIO()

            with (
                patch.dict(os.environ, {"STUB_API_KEY": "test-key"}),
                patch(
                    "deepmate.channels.cli.ChatCompletionsProvider",
                    lambda base_url, api_key: provider,
                ),
                patch("deepmate.tools.browser.shutil.which", lambda _command: None),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    (
                        "--workspace",
                        str(workspace),
                        "Use the browser to inspect a dynamic page.",
                    )
                )

        self.assertEqual(exit_code, 0, stderr.getvalue())
        self.assertIn("built-in browser backend is not installed", stdout.getvalue())
        first_step_names = tuple(
            schema["name"] for schema in provider.requests[0].tool_schemas
        )
        self.assertIn(LOAD_BROWSER_TOOLS_NAME, first_step_names)
        self.assertNotIn(BROWSER_STATUS_TOOL_NAME, first_step_names)
        second_step_names = tuple(
            schema["name"] for schema in provider.requests[1].tool_schemas
        )
        self.assertIn(BROWSER_STATUS_TOOL_NAME, second_step_names)
        self.assertIn(RETRIEVE_TOOL_OUTPUT_NAME, second_step_names)
        diagnostic_exchange = provider.requests[2].conversation[-1].tool_exchange
        self.assertIsNotNone(diagnostic_exchange)
        self.assertIn(
            "Browser backend is not available",
            diagnostic_exchange.tool_results[0].content,
        )
        self.assertIn(
            "browser_available=false",
            diagnostic_exchange.tool_results[0].refs,
        )

    def test_validate_browser_backend_runs_local_smoke_with_fake_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = _FakeRunner(write_png=True)
            backend = AgentBrowserBackend(
                workspace,
                session_name="validate-browser",
                runner=runner.run,
                which=lambda _command: "/usr/local/bin/agent-browser",
            )

            result = validate_browser_backend(workspace, backend=backend)
            local_page_exists = result.local_page.exists()
            screenshot_path = result.screenshot_path
            screenshot_exists = (
                screenshot_path.exists() if screenshot_path is not None else False
            )
            local_page_text = result.local_page.read_text(encoding="utf-8")

        self.assertTrue(result.ok())
        self.assertEqual(result.reason, "completed")
        self.assertEqual(
            tuple(step.name for step in result.steps),
            (
                BROWSER_STATUS_TOOL_NAME,
                BROWSER_OPEN_TOOL_NAME,
                BROWSER_SNAPSHOT_TOOL_NAME,
                BROWSER_SCREENSHOT_TOOL_NAME,
                BROWSER_CLOSE_TOOL_NAME,
            ),
        )
        self.assertTrue(local_page_exists)
        self.assertIsNotNone(screenshot_path)
        self.assertTrue(screenshot_exists)
        self.assertIn("Deepmate Browser Validation", local_page_text)
        self.assertIn("screenshot saved", result.steps[3].summary)
        commands = tuple(call.argv[3] for call in runner.calls)
        self.assertEqual(commands, ("session", "open", "snapshot", "screenshot", "close"))
        report = format_browser_validation_result(result)
        self.assertIn("Browser validation", report)
        self.assertIn("status: ok", report)
        self.assertIn("browser_open: ok", report)

    def test_validate_browser_backend_reports_missing_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            backend = AgentBrowserBackend(
                workspace,
                session_name="validate-browser",
                runner=_FakeRunner().run,
                which=lambda _command: None,
            )

            result = validate_browser_backend(workspace, backend=backend)

        self.assertFalse(result.ok())
        self.assertEqual(result.reason, "backend_unavailable")
        self.assertEqual(tuple(step.name for step in result.steps), (BROWSER_STATUS_TOOL_NAME,))
        self.assertIn("backend unavailable", result.steps[0].summary)
        report = format_browser_validation_result(result)
        self.assertIn("status: failed", report)
        self.assertIn("npm install -g agent-browser", report)
        self.assertIn("normal Deepmate runs continue without it", report)

    def test_validate_browser_cli_does_not_require_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            stdout = io.StringIO()
            stderr = io.StringIO()

            with (
                patch("deepmate.tools.browser.shutil.which", lambda _command: None),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    (
                        "--workspace",
                        str(workspace),
                        "--validate-browser",
                    )
                )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stderr.getvalue(), "")
        output = stdout.getvalue()
        self.assertIn("Browser validation", output)
        self.assertIn("status: failed", output)
        self.assertIn("backend_unavailable", output)
        self.assertIn("npm install -g agent-browser", output)
        self.assertIn("~/.agent-browser is writable", output)
        self.assertIn("deepmate --install-browser", output)

    def test_install_browser_backend_runs_npm_then_setup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            calls = []
            installed = {"agent-browser": False}

            def which(command: str) -> str | None:
                if command == "npm":
                    return "/usr/local/bin/npm"
                if command == "agent-browser" and installed["agent-browser"]:
                    return "/usr/local/bin/agent-browser"
                return None

            def runner(argv, cwd, timeout_seconds):
                calls.append(tuple(argv))
                if tuple(argv) == ("npm", "install", "-g", "agent-browser"):
                    installed["agent-browser"] = True
                    return BrowserCommandResult(exit_code=0, stdout="installed\n")
                if tuple(argv) == ("agent-browser", "install"):
                    return BrowserCommandResult(exit_code=0, stdout="ready\n")
                return BrowserCommandResult(exit_code=1, stderr="unexpected\n")

            result = install_browser_backend(workspace, runner=runner, which=which)

        self.assertTrue(result.ok())
        self.assertEqual(
            calls,
            [
                ("npm", "install", "-g", "agent-browser"),
                ("agent-browser", "install"),
            ],
        )
        report = format_browser_install_result(result)
        self.assertIn("Browser installer", report)
        self.assertIn("deepmate --validate-browser", report)

    def test_install_browser_cli_does_not_require_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            stdout = io.StringIO()
            stderr = io.StringIO()
            result = install_browser_backend(
                workspace,
                runner=lambda _argv, _cwd, _timeout: BrowserCommandResult(
                    exit_code=0,
                    stdout="ready\n",
                ),
                which=lambda command: (
                    "/usr/local/bin/agent-browser"
                    if command == "agent-browser"
                    else "/usr/local/bin/npm"
                    if command == "npm"
                    else None
                ),
            )

            with (
                patch(
                    "deepmate.channels.cli.install_browser_backend",
                    lambda _workspace: result,
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    (
                        "--workspace",
                        str(workspace),
                        "--install-browser",
                    )
                )

        self.assertEqual(exit_code, 0, stderr.getvalue())
        self.assertIn("Browser installer", stdout.getvalue())

    def test_cli_attach_browser_tools_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = AgentBrowserBackend(
                Path(tmp),
                session_name="test-session",
                runner=_FakeRunner().run,
                which=lambda _command: "/usr/local/bin/agent-browser",
            )

            registry = _attach_browser_tools(native_tools=None, backend=backend)

        names = tuple(schema["name"] for schema in registry.schemas())
        self.assertEqual(names, (LOAD_BROWSER_TOOLS_NAME,))
        all_names = tuple(schema["name"] for schema in registry.schemas(include_hidden=True))
        self.assertIn(BROWSER_OPEN_TOOL_NAME, all_names)
        self.assertIn(BROWSER_STATUS_TOOL_NAME, all_names)
        self.assertIn(INSTALL_BROWSER_BACKEND_TOOL_NAME, all_names)
        self.assertNotIn(INSTALL_BROWSER_BACKEND_TOOL_NAME, names)

    def test_cli_attach_browser_tools_can_preload_concrete_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = AgentBrowserBackend(
                Path(tmp),
                session_name="test-session",
                runner=_FakeRunner().run,
                which=lambda _command: "/usr/local/bin/agent-browser",
            )

            registry = _attach_browser_tools(
                native_tools=None,
                backend=backend,
                preload=True,
            )

        names = tuple(schema["name"] for schema in registry.schemas())
        self.assertIn(BROWSER_OPEN_TOOL_NAME, names)
        self.assertIn(BROWSER_STATUS_TOOL_NAME, names)
        self.assertNotIn(LOAD_BROWSER_TOOLS_NAME, names)

    def test_load_browser_tools_adds_install_recovery_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = AgentBrowserBackend(
                Path(tmp),
                session_name="test-session",
                runner=_FakeRunner().run,
                which=lambda _command: "/usr/local/bin/agent-browser",
            )
            registry = _attach_browser_tools(native_tools=None, backend=backend)
            loader = registry.get(LOAD_BROWSER_TOOLS_NAME)

            result = loader.call({"reason": "inspect a page"})

        names = tuple(schema["name"] for schema in result.schema_additions)
        self.assertIn(BROWSER_OPEN_TOOL_NAME, names)
        self.assertIn(INSTALL_BROWSER_BACKEND_TOOL_NAME, names)

    def test_agent_request_includes_browser_guidance_only_when_browser_is_exposed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            backend = AgentBrowserBackend(
                workspace,
                session_name="test-session",
                runner=_FakeRunner().run,
                which=lambda _command: "/usr/local/bin/agent-browser",
            )
            browser_registry = _attach_browser_tools(native_tools=None, backend=backend)
            browser_provider = _StubProvider([ModelResponse(content="done")])
            no_browser_provider = _StubProvider([ModelResponse(content="done")])

            run_user_turn(
                provider=browser_provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="Check the page."),),
                model="stub-model",
                capability_surface=from_native_tool_schemas(browser_registry.schemas()),
                native_tools=browser_registry,
                tool_schemas=browser_registry.schemas(),
                max_steps=1,
            )
            run_user_turn(
                provider=no_browser_provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="Check the page."),),
                model="stub-model",
                capability_surface=None,
                native_tools=None,
                tool_schemas=(),
                max_steps=1,
            )

        browser_request = browser_provider.requests[0]
        no_browser_request = no_browser_provider.requests[0]
        browser_context = browser_request.conversation[0].message.content
        no_browser_context = no_browser_request.conversation[0].message.content
        self.assertEqual(
            tuple(schema["name"] for schema in browser_request.tool_schemas),
            (LOAD_BROWSER_TOOLS_NAME,),
        )
        self.assertIn("<capability_guidance>", browser_context)
        self.assertIn("Use the built-in browser for dynamic web pages", browser_context)
        self.assertIn("call it first to load concrete browser schemas", browser_context)
        self.assertNotIn(BROWSER_OPEN_TOOL_NAME, tuple(schema["name"] for schema in no_browser_request.tool_schemas))
        self.assertNotIn("<capability_guidance>", no_browser_context)
        self.assertNotIn("browser_open", no_browser_context)

    def test_load_browser_tools_adds_concrete_schemas_for_followup_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            runner = _FakeRunner()
            backend = AgentBrowserBackend(
                workspace,
                session_name="test-session",
                runner=runner.run,
                which=lambda _command: "/usr/local/bin/agent-browser",
            )
            registry = _attach_browser_tools(native_tools=None, backend=backend)
            provider = _StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name=LOAD_BROWSER_TOOLS_NAME,
                                id="call_1",
                                arguments={"reason": "Need to inspect a dynamic page."},
                            ),
                        )
                    ),
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name=BROWSER_OPEN_TOOL_NAME,
                                id="call_2",
                                arguments={"url": "https://example.test"},
                            ),
                        )
                    ),
                    ModelResponse(content="done"),
                ]
            )

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="Check this page."),),
                model="stub-model",
                capability_surface=from_native_tool_schemas(registry.schemas()),
                native_tools=registry,
                tool_schemas=registry.schemas(),
                max_steps=3,
            )

        self.assertFalse(result.has_errors())
        self.assertEqual(result.final_step().response.content, "done")
        self.assertEqual(
            tuple(schema["name"] for schema in provider.requests[0].tool_schemas),
            (LOAD_BROWSER_TOOLS_NAME,),
        )
        self.assertIn(
            BROWSER_OPEN_TOOL_NAME,
            tuple(schema["name"] for schema in provider.requests[1].tool_schemas),
        )
        self.assertEqual(
            [call.argv[-2:] for call in runner.calls],
            [("open", "https://example.test")],
        )

    def test_load_browser_tools_can_add_retrieval_schema_without_initial_exposure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            backend = AgentBrowserBackend(
                workspace,
                session_name="test-session",
                runner=_FakeRunner().run,
                which=lambda _command: "/usr/local/bin/agent-browser",
            )
            store = ToolOutputStore.in_data_dir(
                workspace / "var",
                "default",
                "session-1",
            )
            retrieval_registry = _attach_tool_output_tools(
                native_tools=None,
                store=store,
                enabled=True,
            )
            self.assertIsNotNone(retrieval_registry)
            registry = _attach_browser_tools(
                native_tools=None,
                backend=backend,
                extra_schema_loader=lambda: retrieval_registry.schemas(),
            )
            registry = _attach_tool_output_tools(
                native_tools=registry,
                store=store,
                enabled=True,
                exposed_by_default=False,
            )
            provider = _StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name=LOAD_BROWSER_TOOLS_NAME,
                                id="call_1",
                                arguments={},
                            ),
                        )
                    ),
                    ModelResponse(content="done"),
                ]
            )

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="Check this page."),),
                model="stub-model",
                capability_surface=from_native_tool_schemas(registry.schemas()),
                native_tools=registry,
                tool_schemas=registry.schemas(),
                max_steps=2,
            )

        self.assertFalse(result.has_errors())
        self.assertEqual(
            tuple(schema["name"] for schema in provider.requests[0].tool_schemas),
            (LOAD_BROWSER_TOOLS_NAME,),
        )
        second_step_names = tuple(
            schema["name"] for schema in provider.requests[1].tool_schemas
        )
        self.assertIn(BROWSER_OPEN_TOOL_NAME, second_step_names)
        self.assertIn(RETRIEVE_TOOL_OUTPUT_NAME, second_step_names)

    def test_direct_hidden_browser_tool_call_executes_registered_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _workspace(Path(tmp))
            runner = _FakeRunner()
            backend = AgentBrowserBackend(
                workspace,
                session_name="test-session",
                runner=runner.run,
                which=lambda _command: "/usr/local/bin/agent-browser",
            )
            registry = _attach_browser_tools(native_tools=None, backend=backend)
            provider = _StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name=BROWSER_OPEN_TOOL_NAME,
                                id="call_1",
                                arguments={"url": "https://example.test"},
                            ),
                        )
                    ),
                    ModelResponse(content="done"),
                ]
            )

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=ProfileRef(name="default", uri="profiles/default"),
                messages=(Message(role=MessageRole.USER, content="Check this page."),),
                model="stub-model",
                capability_surface=from_native_tool_schemas(registry.schemas()),
                native_tools=registry,
                tool_schemas=registry.schemas(),
                max_steps=2,
            )

        self.assertFalse(result.has_errors())
        self.assertEqual(result.final_step().response.content, "done")
        self.assertEqual(
            [call.argv[-2:] for call in runner.calls],
            [("open", "https://example.test")],
        )

    def test_browser_tools_are_read_only_and_expose_expected_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = AgentBrowserBackend(
                Path(tmp),
                session_name="test-session",
                runner=_FakeRunner().run,
                which=lambda _command: "/usr/local/bin/agent-browser",
            )
            registry = NativeToolRegistry(browser_tools(backend))

        names = tuple(schema["name"] for schema in registry.schemas())
        self.assertEqual(
            names,
            (
                BROWSER_OPEN_TOOL_NAME,
                BROWSER_SNAPSHOT_TOOL_NAME,
                BROWSER_CLICK_TOOL_NAME,
                BROWSER_FILL_TOOL_NAME,
                BROWSER_WAIT_TOOL_NAME,
                BROWSER_SCREENSHOT_TOOL_NAME,
                BROWSER_CLOSE_TOOL_NAME,
                BROWSER_STATUS_TOOL_NAME,
            ),
        )
        self.assertTrue(all(tool.read_only for tool in registry.list_tools()))

    def test_missing_backend_returns_install_diagnostic_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = AgentBrowserBackend(
                Path(tmp),
                runner=_FakeRunner().run,
                which=lambda _command: None,
            )
            registry = NativeToolRegistry(browser_tools(backend))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name=BROWSER_STATUS_TOOL_NAME,
                    id="call_1",
                    arguments={},
                ),
                registry,
                ToolAccessPolicy(),
            )

        self.assertIsNone(result.error)
        self.assertIn("Browser backend is not available", result.model_result.content)
        self.assertIn("npm install -g agent-browser", result.model_result.content)
        self.assertIn("browser_available=false", result.model_result.refs)

    def test_browser_open_builds_agent_browser_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = _FakeRunner()
            backend = AgentBrowserBackend(
                Path(tmp),
                session_name="session one",
                runner=runner.run,
                which=lambda _command: "/usr/local/bin/agent-browser",
            )
            registry = NativeToolRegistry(browser_tools(backend))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name=BROWSER_OPEN_TOOL_NAME,
                    id="call_1",
                    arguments={
                        "url": "https://example.com",
                        "timeout_seconds": 12,
                    },
                ),
                registry,
                ToolAccessPolicy(),
            )

        self.assertIsNone(result.error)
        self.assertEqual(
            runner.calls[0].argv,
            (
                "agent-browser",
                "--session",
                "session-one",
                "open",
                "https://example.com",
            ),
        )
        self.assertEqual(runner.calls[0].timeout_seconds, 12)
        self.assertIn("browser_url=https://example.com", result.model_result.refs)

    def test_browser_snapshot_click_fill_wait_and_close_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = _FakeRunner()
            backend = AgentBrowserBackend(
                Path(tmp),
                session_name="abc",
                runner=runner.run,
                which=lambda _command: "/usr/local/bin/agent-browser",
            )
            registry = NativeToolRegistry(browser_tools(backend))

            for request in (
                ModelToolRequest(
                    name=BROWSER_SNAPSHOT_TOOL_NAME,
                    id="call_snapshot",
                    arguments={"interactive_only": True},
                ),
                ModelToolRequest(
                    name=BROWSER_CLICK_TOOL_NAME,
                    id="call_click",
                    arguments={"selector": "@e1"},
                ),
                ModelToolRequest(
                    name=BROWSER_FILL_TOOL_NAME,
                    id="call_fill",
                    arguments={"selector": "@e2", "text": "hello"},
                ),
                ModelToolRequest(
                    name=BROWSER_WAIT_TOOL_NAME,
                    id="call_wait",
                    arguments={"load_state": "networkidle"},
                ),
                ModelToolRequest(
                    name=BROWSER_CLOSE_TOOL_NAME,
                    id="call_close",
                    arguments={"all": True},
                ),
            ):
                result = execute_native_tool_request(request, registry, ToolAccessPolicy())
                self.assertIsNone(result.error)

        command_tails = tuple(call.argv[3:] for call in runner.calls)
        self.assertEqual(
            command_tails,
            (
                ("snapshot", "-i"),
                ("click", "@e1"),
                ("fill", "@e2", "hello"),
                ("wait", "--load", "networkidle"),
                ("close", "--all"),
            ),
        )

    def test_browser_screenshot_writes_only_inside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            runner = _FakeRunner(write_png=True)
            backend = AgentBrowserBackend(
                workspace,
                session_name="abc",
                runner=runner.run,
                which=lambda _command: "/usr/local/bin/agent-browser",
            )
            registry = NativeToolRegistry(browser_tools(backend))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name=BROWSER_SCREENSHOT_TOOL_NAME,
                    id="call_1",
                    arguments={
                        "path": "shots/page.png",
                        "full_page": True,
                        "annotate": True,
                    },
                ),
                registry,
                ToolAccessPolicy(),
            )

        self.assertIsNone(result.error)
        self.assertEqual(
            runner.calls[0].argv,
            (
                "agent-browser",
                "--session",
                "abc",
                "screenshot",
                "--full",
                "--annotate",
                str(workspace / "shots" / "page.png"),
            ),
        )
        self.assertEqual(result.model_result.data["path"], "shots/page.png")
        self.assertGreater(result.model_result.data["bytes"], 0)
        self.assertEqual(result.model_result.data["image_format"], "png")
        self.assertEqual(result.model_result.data["width"], 2)
        self.assertEqual(result.model_result.data["height"], 3)
        self.assertIn("Browser screenshot saved.", result.model_result.content)
        self.assertIn("- dimensions: 2x3", result.model_result.content)
        self.assertIn("backend_output: omitted", result.model_result.content)
        self.assertNotIn("data:image", result.model_result.content)
        self.assertIn("browser_output=shots/page.png", result.model_result.refs)

    def test_browser_screenshot_reports_bmp_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            runner = _FakeRunner(screenshot_bytes=_bmp_header(width=7, height=5))
            backend = AgentBrowserBackend(
                workspace,
                session_name="abc",
                runner=runner.run,
                which=lambda _command: "/usr/local/bin/agent-browser",
            )
            registry = NativeToolRegistry(browser_tools(backend))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name=BROWSER_SCREENSHOT_TOOL_NAME,
                    id="call_1",
                    arguments={"path": "shots/page.bmp"},
                ),
                registry,
                ToolAccessPolicy(),
            )

        self.assertIsNone(result.error)
        self.assertEqual(result.model_result.data["image_format"], "bmp")
        self.assertEqual(result.model_result.data["width"], 7)
        self.assertEqual(result.model_result.data["height"], 5)
        self.assertIn("- dimensions: 7x5", result.model_result.content)

    def test_browser_screenshot_reports_webp_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            runner = _FakeRunner(screenshot_bytes=_webp_vp8x_header(width=11, height=9))
            backend = AgentBrowserBackend(
                workspace,
                session_name="abc",
                runner=runner.run,
                which=lambda _command: "/usr/local/bin/agent-browser",
            )
            registry = NativeToolRegistry(browser_tools(backend))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name=BROWSER_SCREENSHOT_TOOL_NAME,
                    id="call_1",
                    arguments={"path": "shots/page.webp"},
                ),
                registry,
                ToolAccessPolicy(),
            )

        self.assertIsNone(result.error)
        self.assertEqual(result.model_result.data["image_format"], "webp")
        self.assertEqual(result.model_result.data["width"], 11)
        self.assertEqual(result.model_result.data["height"], 9)
        self.assertIn("- dimensions: 11x9", result.model_result.content)

    def test_browser_screenshot_reports_unknown_image_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            runner = _FakeRunner(screenshot_bytes=b"not an image")
            backend = AgentBrowserBackend(
                workspace,
                session_name="abc",
                runner=runner.run,
                which=lambda _command: "/usr/local/bin/agent-browser",
            )
            registry = NativeToolRegistry(browser_tools(backend))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name=BROWSER_SCREENSHOT_TOOL_NAME,
                    id="call_1",
                    arguments={"path": "shots/page.bin"},
                ),
                registry,
                ToolAccessPolicy(),
            )

        self.assertIsNone(result.error)
        self.assertEqual(result.model_result.data["image_format"], "unknown")
        self.assertEqual(result.model_result.data["width"], 0)
        self.assertEqual(result.model_result.data["height"], 0)
        self.assertIn("- image_format: unknown", result.model_result.content)
        self.assertNotIn("- dimensions:", result.model_result.content)

    def test_browser_screenshot_rejects_paths_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = AgentBrowserBackend(
                Path(tmp),
                runner=_FakeRunner().run,
                which=lambda _command: "/usr/local/bin/agent-browser",
            )
            registry = NativeToolRegistry(browser_tools(backend))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name=BROWSER_SCREENSHOT_TOOL_NAME,
                    id="call_1",
                    arguments={"path": "../outside.png"},
                ),
                registry,
                ToolAccessPolicy(),
            )

        self.assertEqual(result.error.code, "native_tool_failed")
        self.assertIn("inside workspace", result.error.message)

    def test_browser_wait_accepts_only_one_wait_condition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = AgentBrowserBackend(
                Path(tmp),
                runner=_FakeRunner().run,
                which=lambda _command: "/usr/local/bin/agent-browser",
            )
            registry = NativeToolRegistry(browser_tools(backend))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name=BROWSER_WAIT_TOOL_NAME,
                    id="call_1",
                    arguments={"selector": "#ready", "text": "Ready"},
                ),
                registry,
                ToolAccessPolicy(),
            )

        self.assertEqual(result.error.code, "native_tool_failed")
        self.assertIn("only one", result.error.message)

    def test_browser_wait_selector_uses_explicit_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = _FakeRunner()
            backend = AgentBrowserBackend(
                Path(tmp),
                session_name="abc",
                runner=runner.run,
                which=lambda _command: "/usr/local/bin/agent-browser",
            )
            registry = NativeToolRegistry(browser_tools(backend))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name=BROWSER_WAIT_TOOL_NAME,
                    id="call_1",
                    arguments={"selector": "#ready"},
                ),
                registry,
                ToolAccessPolicy(),
            )

        self.assertIsNone(result.error)
        self.assertEqual(runner.calls[0].argv[3:], ("wait", "--selector", "#ready"))


class _FakeRunner:
    def __init__(
        self,
        *,
        write_png: bool = False,
        screenshot_bytes: bytes | None = None,
    ) -> None:
        self.calls: list[_FakeCall] = []
        self.write_png = write_png
        self.screenshot_bytes = screenshot_bytes

    def run(
        self,
        argv,
        cwd: Path,
        timeout_seconds: int,
    ) -> BrowserCommandResult:
        self.calls.append(_FakeCall(tuple(argv), cwd, timeout_seconds))
        if (self.write_png or self.screenshot_bytes is not None) and "screenshot" in argv:
            Path(argv[-1]).parent.mkdir(parents=True, exist_ok=True)
            Path(argv[-1]).write_bytes(
                self.screenshot_bytes or _png_header(width=2, height=3)
            )
            return BrowserCommandResult(
                exit_code=0,
                stdout="data:image/png;base64," + "A" * 400,
            )
        return BrowserCommandResult(exit_code=0, stdout=f"ok: {' '.join(argv)}")


class _FakeCall:
    def __init__(self, argv: tuple[str, ...], cwd: Path, timeout_seconds: int) -> None:
        self.argv = argv
        self.cwd = cwd
        self.timeout_seconds = timeout_seconds


def _png_header(*, width: int, height: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x02\x00\x00\x00"
    )


def _bmp_header(*, width: int, height: int) -> bytes:
    dib_size = 40
    return (
        b"BM"
        + b"\x00" * 12
        + dib_size.to_bytes(4, "little")
        + width.to_bytes(4, "little", signed=True)
        + height.to_bytes(4, "little", signed=True)
        + b"\x01\x00\x18\x00"
    )


def _webp_vp8x_header(*, width: int, height: int) -> bytes:
    return (
        b"RIFF"
        + (22).to_bytes(4, "little")
        + b"WEBP"
        + b"VP8X"
        + (10).to_bytes(4, "little")
        + b"\x00\x00\x00\x00"
        + (width - 1).to_bytes(3, "little")
        + (height - 1).to_bytes(3, "little")
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
                "trace:",
                "  sink: var/traces/trace.jsonl",
                "provider:",
                "  default: stub",
                "  retry:",
                "    max_attempts: 1",
                "models:",
                "  main:",
                "    model: stub-main",
                "    max_tokens: 512",
                "model_context_windows:",
                "  default: 100000",
                "  stub-main: 100000",
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


if __name__ == "__main__":
    unittest.main()
