"""Built-in browser native tools backed by agent-browser."""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

from deepmate.runtime.process_env import subprocess_environment
from deepmate.runtime.safety import SessionApprovalCache, ToolSafetyPolicy
from deepmate.tools.registry import NativeTool, NativeToolRegistry, NativeToolResult

BROWSER_OPEN_TOOL_NAME = "browser_open"
BROWSER_SNAPSHOT_TOOL_NAME = "browser_snapshot"
BROWSER_CLICK_TOOL_NAME = "browser_click"
BROWSER_FILL_TOOL_NAME = "browser_fill"
BROWSER_WAIT_TOOL_NAME = "browser_wait"
BROWSER_SCREENSHOT_TOOL_NAME = "browser_screenshot"
BROWSER_CLOSE_TOOL_NAME = "browser_close"
BROWSER_STATUS_TOOL_NAME = "browser_status"
LOAD_BROWSER_TOOLS_NAME = "load_browser_tools"
INSTALL_BROWSER_BACKEND_TOOL_NAME = "install_browser_backend"

DEFAULT_BROWSER_TIMEOUT_SECONDS = 30
STATUS_TIMEOUT_SECONDS = 5
MAX_BROWSER_TIMEOUT_SECONDS = 120
DEFAULT_SCREENSHOT_DIR = Path("var/browser/screenshots")
BROWSER_VALIDATION_DIR = Path("var/browser/validation")
BROWSER_VALIDATION_PAGE = "deepmate-browser-validation.html"
BROWSER_VALIDATION_SCREENSHOT = "deepmate-browser-validation.png"
BROWSER_INSTALL_HINT = (
    "Install the optional browser backend with `npm install -g agent-browser`, "
    "then run `agent-browser install` once."
)


@dataclass(frozen=True, slots=True)
class BrowserCommandResult:
    """Result returned by one browser backend command."""

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    def output_text(self) -> str:
        """Return stdout/stderr as a single model-visible diagnostic."""
        parts = []
        if self.stdout.strip():
            parts.append(self.stdout.rstrip())
        if self.stderr.strip():
            parts.append(self.stderr.rstrip())
        if self.timed_out:
            parts.append("Browser command timed out.")
        return "\n".join(parts).strip()


@dataclass(frozen=True, slots=True)
class BrowserValidationStep:
    """One step in the local browser backend validation flow."""

    name: str
    ok: bool
    summary: str
    content: str = ""
    refs: tuple[str, ...] = field(default_factory=tuple)
    data: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BrowserValidationResult:
    """Result of validating the optional browser backend without a model call."""

    status: str
    reason: str
    workspace: Path
    backend: str
    executable: str
    local_page: Path
    screenshot_path: Path | None
    steps: tuple[BrowserValidationStep, ...]
    install_hint: str = BROWSER_INSTALL_HINT

    def ok(self) -> bool:
        """Return whether all required validation steps succeeded."""
        return self.status == "ok"


@dataclass(frozen=True, slots=True)
class BrowserInstallStep:
    """One step in the optional browser backend installer."""

    name: str
    ok: bool
    summary: str
    exit_code: int | None = None
    timed_out: bool = False


@dataclass(frozen=True, slots=True)
class BrowserInstallResult:
    """Result of installing or setting up the optional browser backend."""

    status: str
    reason: str
    workspace: Path
    executable: str
    steps: tuple[BrowserInstallStep, ...]
    install_hint: str = BROWSER_INSTALL_HINT

    def ok(self) -> bool:
        """Return whether the browser backend installer completed."""
        return self.status == "ok"


BrowserCommandRunner = Callable[[Sequence[str], Path, int], BrowserCommandResult]
BrowserToolLoader = Callable[[], Sequence[NativeTool]]
SchemaLoader = Callable[[], Iterable[Mapping[str, object]]]


class AgentBrowserBackend:
    """Small manager for the optional agent-browser CLI backend."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        command: str = "agent-browser",
        session_name: str | None = None,
        runner: BrowserCommandRunner | None = None,
        which: Callable[[str], str | None] | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.command = command.strip() or "agent-browser"
        self.session_name = _safe_session_name(session_name)
        self._runner = runner or _subprocess_runner
        self._which = which or shutil.which
        self.current_url = ""
        self.current_title = ""
        self.last_error = ""
        self.started = False
        self.closed = False
        self._lock = RLock()

    def executable_path(self) -> str:
        """Return the resolved executable path, if the backend is installed."""
        return self._which(self.command) or ""

    def is_available(self) -> bool:
        """Return whether the configured agent-browser binary is discoverable."""
        return bool(self.executable_path())

    def status(self) -> NativeToolResult:
        """Return a readable status result without failing the agent turn."""
        executable = self.executable_path()
        if not executable:
            return self._unavailable_result(BROWSER_STATUS_TOOL_NAME, ())
        health = self._run_raw(("session",), timeout_seconds=STATUS_TIMEOUT_SECONDS)
        if health.exit_code != 0 or health.timed_out:
            with self._lock:
                self.last_error = _command_error_text(health)
        with self._lock:
            current_url = self.current_url
            current_title = self.current_title
            started = self.started
            closed = self.closed
            last_error = self.last_error
        content = "\n".join(
            part
            for part in (
                "Browser backend: agent-browser",
                f"Executable: {executable}",
                f"Session: {self.session_name or '(default)'}",
                f"Current URL: {current_url or '(unknown)'}",
                f"Current title: {current_title or '(unknown)'}",
                f"Started: {str(started).lower()}",
                f"Closed: {str(closed).lower()}",
                f"Last error: {last_error or '(none)'}",
                _format_backend_output(health),
            )
            if part
        )
        return NativeToolResult(
            content=content,
            data={
                "backend": "agent-browser",
                "available": True,
                "executable": executable,
                "session_name": self.session_name,
                "current_url": current_url,
                "current_title": current_title,
                "started": started,
                "closed": closed,
                "last_error": last_error,
                "health_exit_code": health.exit_code,
            },
            refs=self._refs(
                BROWSER_STATUS_TOOL_NAME,
                command=("session",),
                exit_code=health.exit_code,
                timed_out=health.timed_out,
            ),
        )

    def close(self) -> None:
        """Close the managed browser session if this run opened it."""
        with self._lock:
            should_close = self.started and not self.closed
        if not should_close or not self.is_available():
            return
        try:
            self._run_raw(("close",), timeout_seconds=STATUS_TIMEOUT_SECONDS)
        except Exception:
            pass
        finally:
            with self._lock:
                self.started = False
                self.closed = True

    def run_tool(
        self,
        tool_name: str,
        command: Sequence[str],
        *,
        timeout_seconds: int,
        output_path: Path | None = None,
        current_url: str = "",
    ) -> NativeToolResult:
        """Run a browser command and return a native tool result."""
        if not self.is_available():
            return self._unavailable_result(tool_name, command)
        result = self._run_raw(command, timeout_seconds=timeout_seconds)
        if result.exit_code != 0 or result.timed_out:
            with self._lock:
                self.last_error = _command_error_text(result)
        elif tool_name == BROWSER_OPEN_TOOL_NAME and current_url:
            with self._lock:
                self.current_url = current_url
                self.current_title = _extract_browser_title(result.output_text())
                self.started = True
                self.closed = False
                self.last_error = ""
        elif tool_name == BROWSER_CLOSE_TOOL_NAME:
            with self._lock:
                self.started = False
                self.closed = True
                self.last_error = ""
        elif tool_name != BROWSER_STATUS_TOOL_NAME:
            title = _extract_browser_title(result.output_text())
            with self._lock:
                self.started = True
                self.closed = False
                if title:
                    self.current_title = title
                self.last_error = ""
        with self._lock:
            current_url_snapshot = self.current_url
            last_error_snapshot = self.last_error

        content = _tool_visible_content(
            tool_name,
            result=result,
            workspace=self.workspace,
            output_path=output_path,
        )
        data: dict[str, object] = {
            "backend": "agent-browser",
            "available": True,
            "tool": tool_name,
            "command": _command_display(command),
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "stdout_chars": len(result.stdout),
            "stderr_chars": len(result.stderr),
            "session_name": self.session_name,
            "current_url": current_url_snapshot,
            "last_error": last_error_snapshot,
        }
        if output_path is not None:
            screenshot_info = _screenshot_info(self.workspace, output_path)
            data["path"] = _display_path(self.workspace, output_path)
            data.update(screenshot_info)
        return NativeToolResult(
            content=content,
            data=data,
            refs=self._refs(
                tool_name,
                command=command,
                exit_code=result.exit_code,
                timed_out=result.timed_out,
                output_path=output_path,
            ),
        )

    def _run_raw(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: int,
    ) -> BrowserCommandResult:
        argv = [self.command]
        if self.session_name:
            argv.extend(("--session", self.session_name))
        argv.extend(str(part) for part in command)
        return self._runner(tuple(argv), self.workspace, timeout_seconds)

    def _unavailable_result(
        self,
        tool_name: str,
        command: Sequence[str],
    ) -> NativeToolResult:
        with self._lock:
            self.last_error = "agent-browser executable was not found on PATH"
            last_error = self.last_error
        content = (
            "Browser backend is not available.\n"
            "- backend: agent-browser\n"
            "- reason: executable was not found on PATH\n"
            "- recovery: call install_browser_backend; Deepmate will ask before changing the environment\n"
            "- manual install: npm install -g agent-browser && agent-browser install"
        )
        return NativeToolResult(
            content=content,
            data={
                "backend": "agent-browser",
                "available": False,
                "tool": tool_name,
                "command": _command_display(command),
                "session_name": self.session_name,
                "last_error": last_error,
                "recovery_tool": INSTALL_BROWSER_BACKEND_TOOL_NAME,
            },
            refs=(
                *self._refs(
                    tool_name,
                    command=command,
                    exit_code=None,
                    timed_out=False,
                    available=False,
                ),
                f"recovery={INSTALL_BROWSER_BACKEND_TOOL_NAME}",
            ),
        )

    def _refs(
        self,
        tool_name: str,
        *,
        command: Sequence[str],
        exit_code: int | None,
        timed_out: bool,
        output_path: Path | None = None,
        available: bool = True,
    ) -> tuple[str, ...]:
        refs = [
            "browser_backend=agent-browser",
            f"browser_available={str(available).lower()}",
            f"browser_tool={tool_name}",
            f"browser_command={_command_display(command)}",
            f"browser_session={self.session_name or 'default'}",
        ]
        if exit_code is not None:
            refs.append(f"browser_exit_code={exit_code}")
        if timed_out:
            refs.append("browser_timed_out=true")
        with self._lock:
            current_url = self.current_url
        if current_url:
            refs.append(f"browser_url={current_url}")
        if output_path is not None:
            refs.append(f"browser_output={_display_path(self.workspace, output_path)}")
        return tuple(refs)


def browser_tools(backend: AgentBrowserBackend) -> tuple[NativeTool, ...]:
    """Return minimal browser tools for one runtime invocation."""
    return (
        NativeTool(
            name=BROWSER_OPEN_TOOL_NAME,
            description=(
                "Open a URL in the built-in browser backend for web research, "
                "frontend verification, or dynamic page interaction. Use "
                "browser_snapshot after navigation to get refs before clicking."
            ),
            input_schema=_browser_open_schema(),
            handler=lambda arguments: _browser_open(backend, arguments),
        ),
        NativeTool(
            name=BROWSER_SNAPSHOT_TOOL_NAME,
            description=(
                "Read the current page accessibility snapshot. Re-run this after "
                "navigation or DOM changes because element refs can change."
            ),
            input_schema=_browser_snapshot_schema(),
            handler=lambda arguments: _browser_snapshot(backend, arguments),
        ),
        NativeTool(
            name=BROWSER_CLICK_TOOL_NAME,
            description=(
                "Click an element in the current browser page by snapshot ref "
                "such as @e2, CSS selector, or supported agent-browser selector."
            ),
            input_schema=_browser_click_schema(),
            handler=lambda arguments: _browser_click(backend, arguments),
        ),
        NativeTool(
            name=BROWSER_FILL_TOOL_NAME,
            description=(
                "Clear and fill an input element in the current browser page by "
                "snapshot ref, CSS selector, or supported agent-browser selector."
            ),
            input_schema=_browser_fill_schema(),
            handler=lambda arguments: _browser_fill(backend, arguments),
        ),
        NativeTool(
            name=BROWSER_WAIT_TOOL_NAME,
            description=(
                "Wait for browser page readiness, an element, text, URL pattern, "
                "or a short millisecond delay."
            ),
            input_schema=_browser_wait_schema(),
            handler=lambda arguments: _browser_wait(backend, arguments),
        ),
        NativeTool(
            name=BROWSER_SCREENSHOT_TOOL_NAME,
            description=(
                "Save a screenshot of the current browser page inside the "
                "workspace and return a path reference, not image bytes."
            ),
            input_schema=_browser_screenshot_schema(),
            handler=lambda arguments: _browser_screenshot(backend, arguments),
        ),
        NativeTool(
            name=BROWSER_CLOSE_TOOL_NAME,
            description="Close the browser session opened for this Deepmate run.",
            input_schema=_browser_close_schema(),
            handler=lambda arguments: _browser_close(backend, arguments),
        ),
        NativeTool(
            name=BROWSER_STATUS_TOOL_NAME,
            description=(
                "Show whether the built-in browser backend is installed, healthy, "
                "and what page state Deepmate currently knows."
            ),
            input_schema=_empty_schema(),
            handler=lambda _arguments: backend.status(),
        ),
    )


def browser_loader_tools(
    backend: AgentBrowserBackend,
    *,
    load_tools: BrowserToolLoader | None = None,
    extra_schema_loader: SchemaLoader | None = None,
    approval_cache: SessionApprovalCache | None = None,
) -> tuple[NativeTool, ...]:
    """Return a tiny entrypoint that loads full browser schemas on demand."""
    return (
        NativeTool(
            name=LOAD_BROWSER_TOOLS_NAME,
            description=(
                "Load concrete built-in browser tool schemas when a task needs "
                "dynamic web pages, login-free page interaction, frontend "
                "verification, screenshots, or static retrieval is insufficient."
            ),
            input_schema=_load_browser_tools_schema(),
            handler=lambda arguments: _load_browser_tools(
                backend,
                arguments,
                load_tools=load_tools,
                extra_schema_loader=extra_schema_loader,
            ),
        ),
        NativeTool(
            name=INSTALL_BROWSER_BACKEND_TOOL_NAME,
            description=(
                "Install and initialize the optional agent-browser backend after "
                "browser_status or browser_open reports that it is unavailable. "
                "This changes the user's environment and requires approval."
            ),
            input_schema=_install_browser_backend_tool_input_schema(),
            handler=lambda arguments: _install_browser_backend_tool(
                backend,
                arguments,
                approval_cache=approval_cache,
            ),
            read_only=False,
            requires_shell=True,
            exposed_by_default=False,
        ),
    )


def validate_browser_backend(
    workspace: str | Path,
    *,
    backend: AgentBrowserBackend | None = None,
    timeout_seconds: int = DEFAULT_BROWSER_TIMEOUT_SECONDS,
) -> BrowserValidationResult:
    """Run a local, no-network smoke check for the optional browser backend."""
    root = Path(workspace).resolve()
    validation_dir = root / BROWSER_VALIDATION_DIR
    validation_dir.mkdir(parents=True, exist_ok=True)
    local_page = validation_dir / BROWSER_VALIDATION_PAGE
    screenshot_path = validation_dir / BROWSER_VALIDATION_SCREENSHOT
    local_page.write_text(_browser_validation_html(), encoding="utf-8")
    if screenshot_path.exists() and screenshot_path.is_file():
        try:
            screenshot_path.unlink()
        except OSError:
            pass

    managed_backend = backend or AgentBrowserBackend(
        root,
        session_name=f"browser-validation-{uuid4().hex[:8]}",
    )
    registry = NativeToolRegistry(browser_tools(managed_backend))
    steps: list[BrowserValidationStep] = []
    executable = managed_backend.executable_path()

    status_result = _run_validation_step(
        registry,
        BROWSER_STATUS_TOOL_NAME,
        {},
        _status_step_ok,
    )
    steps.append(status_result)
    executable = _text_data(status_result.data, "executable") or executable
    if not status_result.ok:
        return BrowserValidationResult(
            status="failed",
            reason=(
                "backend_unavailable"
                if status_result.data.get("available") is False
                else "backend_status_failed"
            ),
            workspace=root,
            backend="agent-browser",
            executable=executable,
            local_page=local_page,
            screenshot_path=None,
            steps=tuple(steps),
        )

    try:
        page_url = local_page.as_uri()
        for name, arguments, ok_checker in (
            (
                BROWSER_OPEN_TOOL_NAME,
                {"url": page_url, "timeout_seconds": timeout_seconds},
                _command_step_ok,
            ),
            (
                BROWSER_SNAPSHOT_TOOL_NAME,
                {"interactive_only": False, "timeout_seconds": timeout_seconds},
                _command_step_ok,
            ),
            (
                BROWSER_SCREENSHOT_TOOL_NAME,
                {
                    "path": _display_path(root, screenshot_path),
                    "timeout_seconds": timeout_seconds,
                },
                _screenshot_step_ok,
            ),
            (
                BROWSER_CLOSE_TOOL_NAME,
                {"timeout_seconds": STATUS_TIMEOUT_SECONDS},
                _command_step_ok,
            ),
        ):
            step = _run_validation_step(registry, name, arguments, ok_checker)
            steps.append(step)
            if not step.ok:
                return BrowserValidationResult(
                    status="failed",
                    reason=f"{name}_failed",
                    workspace=root,
                    backend="agent-browser",
                    executable=executable,
                    local_page=local_page,
                    screenshot_path=screenshot_path,
                    steps=tuple(steps),
                )
    finally:
        managed_backend.close()

    return BrowserValidationResult(
        status="ok",
        reason="completed",
        workspace=root,
        backend="agent-browser",
        executable=executable,
        local_page=local_page,
        screenshot_path=screenshot_path,
        steps=tuple(steps),
    )


def install_browser_backend(
    workspace: str | Path,
    *,
    runner: BrowserCommandRunner | None = None,
    which: Callable[[str], str | None] | None = None,
    timeout_seconds: int = 300,
) -> BrowserInstallResult:
    """Install and initialize the optional agent-browser backend."""
    root = Path(workspace).resolve()
    command_runner = runner or _subprocess_runner
    find = which or shutil.which
    steps: list[BrowserInstallStep] = []
    executable = find("agent-browser") or ""

    if not executable:
        npm = find("npm") or ""
        if not npm:
            return BrowserInstallResult(
                status="failed",
                reason="npm_unavailable",
                workspace=root,
                executable="",
                steps=(
                    BrowserInstallStep(
                        name="find npm",
                        ok=False,
                        summary="npm was not found on PATH.",
                    ),
                ),
            )
        step = _install_step(
            "npm install -g agent-browser",
            command_runner(("npm", "install", "-g", "agent-browser"), root, timeout_seconds),
        )
        steps.append(step)
        if not step.ok:
            return BrowserInstallResult(
                status="failed",
                reason="npm_install_failed",
                workspace=root,
                executable="",
                steps=tuple(steps),
            )
        executable = find("agent-browser") or "agent-browser"
    else:
        steps.append(
            BrowserInstallStep(
                name="find agent-browser",
                ok=True,
                summary=f"found {executable}",
            )
        )

    setup_step = _install_step(
        "agent-browser install",
        command_runner(("agent-browser", "install"), root, timeout_seconds),
    )
    steps.append(setup_step)
    if not setup_step.ok:
        return BrowserInstallResult(
            status="failed",
            reason="agent_browser_install_failed",
            workspace=root,
            executable=executable,
            steps=tuple(steps),
        )
    return BrowserInstallResult(
        status="ok",
        reason="completed",
        workspace=root,
        executable=executable,
        steps=tuple(steps),
    )


def format_browser_install_result(result: BrowserInstallResult) -> str:
    """Return a concise CLI report for browser backend installation."""
    lines = [
        "Browser installer",
        f"  status: {result.status}",
        f"  reason: {result.reason}",
        f"  executable: {result.executable or '(not found)'}",
        "  steps:",
    ]
    for step in result.steps:
        status = "ok" if step.ok else "failed"
        suffix = ""
        if step.exit_code is not None:
            suffix = f" (exit {step.exit_code})"
        if step.timed_out:
            suffix += " (timed out)"
        lines.append(f"    - {step.name}: {status}{suffix} - {step.summary}")
    if not result.ok():
        lines.extend(
            (
                "  install:",
                "    npm install -g agent-browser",
                "    agent-browser install",
            )
        )
    else:
        lines.append("  next: run `deepmate --validate-browser` to verify the backend.")
    return "\n".join(lines)


def _install_step(name: str, result: BrowserCommandResult) -> BrowserInstallStep:
    output = result.output_text()
    if result.exit_code == 0 and not result.timed_out:
        return BrowserInstallStep(
            name=name,
            ok=True,
            summary=_first_line(output) or "completed",
            exit_code=result.exit_code,
            timed_out=result.timed_out,
        )
    return BrowserInstallStep(
        name=name,
        ok=False,
        summary=_first_line(output) or "command failed",
        exit_code=result.exit_code,
        timed_out=result.timed_out,
    )


def _first_line(text: str) -> str:
    for line in text.splitlines():
        clean = line.strip()
        if clean:
            return clean
    return ""


def format_browser_validation_result(result: BrowserValidationResult) -> str:
    """Return a concise CLI report for browser backend validation."""
    lines = [
        "Browser validation",
        f"  status: {result.status}",
        f"  reason: {result.reason}",
        f"  backend: {result.backend}",
        f"  executable: {result.executable or '(not found)'}",
        f"  local_page: {_display_path(result.workspace, result.local_page)}",
    ]
    if result.screenshot_path is not None:
        lines.append(
            f"  screenshot: {_display_path(result.workspace, result.screenshot_path)}"
        )
    lines.append("  steps:")
    for step in result.steps:
        status = "ok" if step.ok else "failed"
        lines.append(f"    - {step.name}: {status} - {step.summary}")
    if not result.ok():
        lines.extend(
            (
                "  install:",
                "    npm install -g agent-browser",
                "    agent-browser install",
                "  repair:",
                "    - if the backend exists but cannot open, check that ~/.agent-browser is writable",
                "    - remove stale socket files under ~/.agent-browser, then rerun validation",
                "    - reinstall with `deepmate --install-browser` if setup files are missing",
                "  note: browser is optional; normal Deepmate runs continue without it.",
            )
        )
    return "\n".join(lines)


def _load_browser_tools(
    backend: AgentBrowserBackend,
    arguments: Mapping[str, object],
    *,
    load_tools: BrowserToolLoader | None,
    extra_schema_loader: SchemaLoader | None,
) -> NativeToolResult:
    reason = _optional_text_argument(arguments, "reason")
    tools = tuple(load_tools() if load_tools is not None else browser_tools(backend))
    extra_schemas = (
        tuple(extra_schema_loader()) if extra_schema_loader is not None else ()
    )
    schemas = (
        *tuple(tool.schema() for tool in tools),
        _install_browser_backend_tool_schema(),
        *extra_schemas,
    )
    names = tuple(str(schema["name"]) for schema in schemas)
    lines = [
        "Browser tools loaded for the next model step.",
        f"- tools: {', '.join(names)}",
        "- use browser_open before browser_snapshot when no page is open",
        "- call browser_snapshot after navigation or DOM changes before click/fill",
        "- do not use browser for CAPTCHA bypass, credential entry, stealth automation, or broad crawling",
    ]
    if reason:
        lines.insert(1, f"- reason: {reason}")
    return NativeToolResult(
        content="\n".join(lines),
        data={
            "backend": "agent-browser",
            "tools": names,
            "schema_count": len(schemas),
            "reason": reason,
        },
        refs=(
            "browser_backend=agent-browser",
            "browser_tools_loaded=true",
            f"browser_schema_count={len(schemas)}",
            *(f"browser_schema={name}" for name in names),
        ),
        schema_additions=schemas,
    )


def _browser_open(
    backend: AgentBrowserBackend,
    arguments: Mapping[str, object],
) -> NativeToolResult:
    url = _text_argument(arguments, "url")
    timeout_seconds = _timeout_argument(arguments)
    return backend.run_tool(
        BROWSER_OPEN_TOOL_NAME,
        ("open", url),
        timeout_seconds=timeout_seconds,
        current_url=url,
    )


def _install_browser_backend_tool(
    backend: AgentBrowserBackend,
    arguments: Mapping[str, object],
    *,
    approval_cache: SessionApprovalCache | None,
) -> NativeToolResult:
    timeout_seconds = _int_argument(arguments, "timeout_seconds", 300, 30, 600)
    decision = ToolSafetyPolicy(
        workspace=backend.workspace,
        shell_enabled=False,
        network_enabled=False,
        env_change_enabled=False,
        approval_cache=approval_cache,
    ).check_shell_command(
        "npm install -g agent-browser && agent-browser install",
        cwd=backend.workspace,
        network="on",
    )
    if not decision.allowed:
        raise ValueError(_decision_message(decision))
    result = install_browser_backend(
        backend.workspace,
        runner=backend._runner,
        which=backend._which,
        timeout_seconds=timeout_seconds,
    )
    return NativeToolResult(
        content=format_browser_install_result(result),
        data={
            "backend": "agent-browser",
            "status": result.status,
            "reason": result.reason,
            "executable": result.executable,
            "step_count": len(result.steps),
        },
        refs=(
            "browser_backend=agent-browser",
            f"browser_install_status={result.status}",
            f"browser_install_reason={result.reason}",
            *(("browser_available=true",) if result.ok() else ("browser_available=false",)),
        ),
    )


def _decision_message(decision) -> str:
    refs = ", ".join(decision.refs)
    return f"{decision.reason} {refs}".strip()


def _browser_snapshot(
    backend: AgentBrowserBackend,
    arguments: Mapping[str, object],
) -> NativeToolResult:
    timeout_seconds = _timeout_argument(arguments)
    interactive_only = _bool_argument(arguments, "interactive_only", True)
    command = ("snapshot", "-i") if interactive_only else ("snapshot",)
    return backend.run_tool(
        BROWSER_SNAPSHOT_TOOL_NAME,
        command,
        timeout_seconds=timeout_seconds,
    )


def _browser_click(
    backend: AgentBrowserBackend,
    arguments: Mapping[str, object],
) -> NativeToolResult:
    selector = _selector_argument(arguments)
    timeout_seconds = _timeout_argument(arguments)
    return backend.run_tool(
        BROWSER_CLICK_TOOL_NAME,
        ("click", selector),
        timeout_seconds=timeout_seconds,
    )


def _browser_fill(
    backend: AgentBrowserBackend,
    arguments: Mapping[str, object],
) -> NativeToolResult:
    selector = _selector_argument(arguments)
    text = _text_argument(arguments, "text", allow_empty=True)
    timeout_seconds = _timeout_argument(arguments)
    return backend.run_tool(
        BROWSER_FILL_TOOL_NAME,
        ("fill", selector, text),
        timeout_seconds=timeout_seconds,
    )


def _browser_wait(
    backend: AgentBrowserBackend,
    arguments: Mapping[str, object],
) -> NativeToolResult:
    timeout_seconds = _timeout_argument(arguments)
    command = ("wait", *_wait_arguments(arguments))
    return backend.run_tool(
        BROWSER_WAIT_TOOL_NAME,
        command,
        timeout_seconds=timeout_seconds,
    )


def _browser_screenshot(
    backend: AgentBrowserBackend,
    arguments: Mapping[str, object],
) -> NativeToolResult:
    timeout_seconds = _timeout_argument(arguments, default=60)
    output_path = _screenshot_path(backend.workspace, arguments)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["screenshot"]
    if _bool_argument(arguments, "full_page", False):
        command.append("--full")
    if _bool_argument(arguments, "annotate", False):
        command.append("--annotate")
    command.append(str(output_path))
    return backend.run_tool(
        BROWSER_SCREENSHOT_TOOL_NAME,
        tuple(command),
        timeout_seconds=timeout_seconds,
        output_path=output_path,
    )


def _browser_close(
    backend: AgentBrowserBackend,
    arguments: Mapping[str, object],
) -> NativeToolResult:
    timeout_seconds = _timeout_argument(arguments, default=STATUS_TIMEOUT_SECONDS)
    close_all = _bool_argument(arguments, "all", False)
    command = ("close", "--all") if close_all else ("close",)
    return backend.run_tool(
        BROWSER_CLOSE_TOOL_NAME,
        command,
        timeout_seconds=timeout_seconds,
    )


def _load_browser_tools_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Short reason why browser tools are needed for this task.",
            }
        },
        "additionalProperties": False,
    }


def _install_browser_backend_tool_schema() -> Mapping[str, object]:
    return {
        "name": INSTALL_BROWSER_BACKEND_TOOL_NAME,
        "description": (
            "Install and initialize the optional agent-browser backend after "
            "browser_status or browser_open reports that it is unavailable. "
            "This changes the user's environment and requires approval."
        ),
        "input_schema": _install_browser_backend_tool_input_schema(),
    }


def _install_browser_backend_tool_input_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "timeout_seconds": {
                "type": "integer",
                "description": "Install timeout in seconds. Defaults to 300, max 600.",
                "minimum": 30,
                "maximum": 600,
            },
        },
        "additionalProperties": False,
    }


def _subprocess_runner(
    argv: Sequence[str],
    cwd: Path,
    timeout_seconds: int,
) -> BrowserCommandResult:
    try:
        completed = subprocess.run(
            list(argv),
            cwd=str(cwd),
            env=subprocess_environment(),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except OSError as exc:
        return BrowserCommandResult(
            exit_code=127,
            stderr=f"Failed to run browser backend: {exc}",
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return BrowserCommandResult(
            exit_code=124,
            stdout=stdout,
            stderr=stderr,
            timed_out=True,
        )
    return BrowserCommandResult(
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        timed_out=False,
    )


def _run_validation_step(
    registry: NativeToolRegistry,
    tool_name: str,
    arguments: Mapping[str, object],
    ok_checker: Callable[[NativeToolResult], bool],
) -> BrowserValidationStep:
    tool = registry.get(tool_name)
    if tool is None:
        return BrowserValidationStep(
            name=tool_name,
            ok=False,
            summary="tool is not registered",
        )
    try:
        result = tool.call(arguments)
    except (OSError, ValueError) as exc:
        return BrowserValidationStep(
            name=tool_name,
            ok=False,
            summary=str(exc),
        )
    ok = ok_checker(result)
    return BrowserValidationStep(
        name=tool_name,
        ok=ok,
        summary=_validation_step_summary(result),
        content=result.content,
        refs=result.refs,
        data=result.data,
    )


def _status_step_ok(result: NativeToolResult) -> bool:
    return (
        result.data.get("available") is True
        and _int_data(result.data, "health_exit_code") == 0
    )


def _command_step_ok(result: NativeToolResult) -> bool:
    return (
        result.data.get("available") is True
        and _int_data(result.data, "exit_code") == 0
        and result.data.get("timed_out") is False
    )


def _screenshot_step_ok(result: NativeToolResult) -> bool:
    return _command_step_ok(result) and _int_data(result.data, "bytes") > 0


def _validation_step_summary(result: NativeToolResult) -> str:
    if result.data.get("available") is False:
        return "backend unavailable"
    last_error = _text_data(result.data, "last_error")
    if last_error and last_error != "(none)":
        return _short_line(last_error)
    exit_code = result.data.get("exit_code", result.data.get("health_exit_code"))
    timed_out = result.data.get("timed_out")
    if isinstance(exit_code, int):
        if timed_out is True:
            return f"timed out with exit_code={exit_code}"
        if exit_code != 0:
            return f"exit_code={exit_code}"
    if "bytes" in result.data:
        image_format = _text_data(result.data, "image_format") or "unknown"
        byte_count = _int_data(result.data, "bytes")
        return f"screenshot saved ({image_format}, {byte_count} bytes)"
    content = _short_line(result.content)
    return content or "completed"


def _browser_validation_html() -> str:
    return "\n".join(
        (
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8">',
            "  <title>Deepmate Browser Validation</title>",
            "  <style>",
            "    body { font-family: system-ui, sans-serif; margin: 32px; }",
            "    button { padding: 8px 12px; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <main>",
            "    <h1>Deepmate Browser Validation</h1>",
            "    <p>This local page validates the built-in browser backend without network access.</p>",
            '    <button id="ready">Ready</button>',
            "  </main>",
            "</body>",
            "</html>",
            "",
        )
    )


def _text_data(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    return value.strip() if isinstance(value, str) else ""


def _int_data(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool):
        return 0
    return value if isinstance(value, int) else 0


def _short_line(value: str, *, limit: int = 120) -> str:
    text = " ".join(value.strip().split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _wait_arguments(arguments: Mapping[str, object]) -> tuple[str, ...]:
    selector = _optional_text_argument(arguments, "selector")
    text = _optional_text_argument(arguments, "text")
    url = _optional_text_argument(arguments, "url")
    load_state = _optional_text_argument(arguments, "load_state")
    milliseconds = _optional_int_argument(arguments, "milliseconds")
    selected = [
        bool(selector),
        bool(text),
        bool(url),
        bool(load_state),
        milliseconds is not None,
    ]
    if sum(1 for value in selected if value) > 1:
        raise ValueError(
            "browser_wait accepts only one of selector, text, url, "
            "load_state, or milliseconds"
        )
    if selector:
        return ("--selector", selector)
    if text:
        return ("--text", text)
    if url:
        return ("--url", url)
    if milliseconds is not None:
        if milliseconds < 0 or milliseconds > 60_000:
            raise ValueError("milliseconds must be between 0 and 60000")
        return (str(milliseconds),)
    state = load_state or "networkidle"
    if state not in {"load", "domcontentloaded", "networkidle"}:
        raise ValueError("load_state must be load, domcontentloaded, or networkidle")
    return ("--load", state)


def _screenshot_path(root: Path, arguments: Mapping[str, object]) -> Path:
    raw = _optional_text_argument(arguments, "path")
    if not raw:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        raw = f"{DEFAULT_SCREENSHOT_DIR.as_posix()}/{stamp}-{uuid4().hex[:8]}.png"
    candidate = Path(raw)
    path = candidate if candidate.is_absolute() else root / candidate
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"screenshot path must stay inside workspace: {raw}")
    if resolved.exists() and not resolved.is_file():
        raise ValueError(f"screenshot path exists but is not a file: {resolved}")
    return resolved


def _selector_argument(arguments: Mapping[str, object]) -> str:
    return _text_argument(arguments, "selector")


def _text_argument(
    arguments: Mapping[str, object],
    key: str,
    *,
    allow_empty: bool = False,
) -> str:
    if key not in arguments:
        raise ValueError(f"{key} is required")
    value = arguments.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    text = value.strip()
    if not allow_empty and not text:
        raise ValueError(f"{key} must be a non-empty string")
    return text


def _optional_text_argument(arguments: Mapping[str, object], key: str) -> str:
    value = arguments.get(key)
    return value.strip() if isinstance(value, str) else ""


def _timeout_argument(
    arguments: Mapping[str, object],
    *,
    default: int = DEFAULT_BROWSER_TIMEOUT_SECONDS,
) -> int:
    return _int_argument(
        arguments,
        "timeout_seconds",
        default,
        1,
        MAX_BROWSER_TIMEOUT_SECONDS,
    )


def _optional_int_argument(arguments: Mapping[str, object], key: str) -> int | None:
    value = arguments.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _int_argument(
    arguments: Mapping[str, object],
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = arguments.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


def _bool_argument(
    arguments: Mapping[str, object],
    key: str,
    default: bool,
) -> bool:
    value = arguments.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        clean = value.strip().lower()
        if clean in {"true", "1", "yes", "on"}:
            return True
        if clean in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{key} must be a boolean")


def _safe_session_name(value: str | None) -> str:
    if not value:
        return ""
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value)
    safe = "-".join(part for part in safe.split("-") if part)
    return safe[:80]


def _format_backend_output(result: BrowserCommandResult) -> str:
    output = result.output_text()
    if result.exit_code == 0 and not result.timed_out:
        return output
    status = (
        f"Browser command failed with exit_code={result.exit_code}"
        if not result.timed_out
        else f"Browser command timed out with exit_code={result.exit_code}"
    )
    return "\n".join(part for part in (status, output) if part)


def _extract_browser_title(text: str) -> str:
    if not text.strip():
        return ""
    for pattern in (
        r"(?im)^\s*title\s*[:=]\s*(.+?)\s*$",
        r"(?is)<title[^>]*>\s*(.*?)\s*</title>",
    ):
        match = re.search(pattern, text)
        if match:
            return " ".join(match.group(1).split())[:160]
    return ""


def _command_error_text(result: BrowserCommandResult) -> str:
    return _format_backend_output(result) or f"exit_code={result.exit_code}"


def _tool_visible_content(
    tool_name: str,
    *,
    result: BrowserCommandResult,
    workspace: Path,
    output_path: Path | None,
) -> str:
    if (
        tool_name == BROWSER_SCREENSHOT_TOOL_NAME
        and output_path is not None
        and result.exit_code == 0
        and not result.timed_out
    ):
        return _screenshot_visible_content(workspace, output_path, result)
    return _format_backend_output(result) or _default_success_message(
        tool_name,
        workspace=workspace,
        output_path=output_path,
    )


def _screenshot_visible_content(
    workspace: Path,
    output_path: Path,
    result: BrowserCommandResult,
) -> str:
    info = _screenshot_info(workspace, output_path)
    lines = [
        "Browser screenshot saved.",
        f"- path: {info['path']}",
        f"- bytes: {info['bytes']}",
    ]
    if info.get("image_format"):
        lines.append(f"- image_format: {info['image_format']}")
    if info.get("width") and info.get("height"):
        lines.append(f"- dimensions: {info['width']}x{info['height']}")
    output = result.output_text()
    if output:
        lines.append(f"- backend_output_chars: {len(output)}")
        lines.append("- backend_output: omitted from screenshot result")
    return "\n".join(lines)


def _screenshot_info(workspace: Path, output_path: Path) -> dict[str, object]:
    info: dict[str, object] = {
        "path": _display_path(workspace, output_path),
        "bytes": 0,
        "image_format": "",
        "width": 0,
        "height": 0,
    }
    try:
        info["bytes"] = output_path.stat().st_size
        with output_path.open("rb") as file:
            data = file.read(1_048_576)
    except OSError:
        return info
    image_format, width, height = _image_metadata(data)
    info["image_format"] = image_format
    info["width"] = width
    info["height"] = height
    return info


def _image_metadata(data: bytes) -> tuple[str, int, int]:
    if len(data) >= 24 and data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ("png", int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big"))
    if len(data) >= 10 and data.startswith(b"GIF"):
        return ("gif", int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little"))
    jpeg_size = _jpeg_size(data)
    if jpeg_size is not None:
        return ("jpeg", jpeg_size[0], jpeg_size[1])
    bmp_size = _bmp_size(data)
    if bmp_size is not None:
        return ("bmp", bmp_size[0], bmp_size[1])
    webp_size = _webp_size(data)
    if webp_size is not None:
        return ("webp", webp_size[0], webp_size[1])
    if _looks_like_webp(data):
        return ("webp", 0, 0)
    return ("unknown", 0, 0)


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    if len(data) < 4 or not data.startswith(b"\xff\xd8"):
        return None
    index = 2
    while index + 9 < len(data):
        marker_index = data.find(b"\xff", index)
        if marker_index < 0:
            return None
        if marker_index + 9 >= len(data):
            return None
        if marker_index > index:
            index = marker_index
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        index += 2
        if marker in {0xD8, 0xD9}:
            continue
        if index + 2 > len(data):
            return None
        length = int.from_bytes(data[index : index + 2], "big")
        if length < 2 or index + length > len(data):
            return None
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if length >= 7:
                height = int.from_bytes(data[index + 3 : index + 5], "big")
                width = int.from_bytes(data[index + 5 : index + 7], "big")
                return (width, height)
            return None
        index += length
    return None


def _bmp_size(data: bytes) -> tuple[int, int] | None:
    if len(data) < 26 or not data.startswith(b"BM"):
        return None
    dib_size = int.from_bytes(data[14:18], "little")
    if dib_size < 12:
        return None
    if dib_size == 12 and len(data) >= 26:
        width = int.from_bytes(data[18:20], "little")
        height = int.from_bytes(data[20:22], "little")
        return (width, height)
    if len(data) >= 26:
        width = int.from_bytes(data[18:22], "little", signed=True)
        height = int.from_bytes(data[22:26], "little", signed=True)
        return (abs(width), abs(height))
    return None


def _webp_size(data: bytes) -> tuple[int, int] | None:
    if not _looks_like_webp(data) or len(data) < 30:
        return None
    chunk_type = data[12:16]
    if chunk_type == b"VP8X" and len(data) >= 30:
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return (width, height)
    if chunk_type == b"VP8 " and len(data) >= 30 and data[23:26] == b"\x9d\x01\x2a":
        width = int.from_bytes(data[26:28], "little") & 0x3FFF
        height = int.from_bytes(data[28:30], "little") & 0x3FFF
        return (width, height)
    if chunk_type == b"VP8L" and len(data) >= 25 and data[20] == 0x2F:
        bits = int.from_bytes(data[21:25], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return (width, height)
    return None


def _looks_like_webp(data: bytes) -> bool:
    return len(data) >= 16 and data.startswith(b"RIFF") and data[8:12] == b"WEBP"


def _default_success_message(
    tool_name: str,
    *,
    workspace: Path | None = None,
    output_path: Path | None = None,
) -> str:
    if output_path is not None:
        visible = _display_path(workspace or output_path.parent, output_path)
        return f"Browser screenshot saved to {visible}"
    return f"Browser tool completed: {tool_name}"


def _command_display(command: Sequence[str]) -> str:
    return " ".join(str(part) for part in command)


def _display_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _browser_open_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to open, for example https://example.com.",
            },
            "timeout_seconds": _timeout_schema(),
        },
        "required": ["url"],
        "additionalProperties": False,
    }


def _browser_snapshot_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "interactive_only": {
                "type": "boolean",
                "description": (
                    "Return interactive refs only. Defaults to true for compact "
                    "agent-friendly snapshots."
                ),
            },
            "timeout_seconds": _timeout_schema(),
        },
        "additionalProperties": False,
    }


def _browser_click_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "selector": {
                "type": "string",
                "description": "Snapshot ref such as @e2, CSS selector, or locator.",
            },
            "timeout_seconds": _timeout_schema(),
        },
        "required": ["selector"],
        "additionalProperties": False,
    }


def _browser_fill_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "selector": {
                "type": "string",
                "description": "Snapshot ref such as @e3, CSS selector, or locator.",
            },
            "text": {
                "type": "string",
                "description": "Text to fill into the target element.",
            },
            "timeout_seconds": _timeout_schema(),
        },
        "required": ["selector", "text"],
        "additionalProperties": False,
    }


def _browser_wait_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "selector": {
                "type": "string",
                "description": "Element ref, CSS selector, or locator to wait for.",
            },
            "text": {
                "type": "string",
                "description": "Visible text substring to wait for.",
            },
            "url": {
                "type": "string",
                "description": "URL glob pattern to wait for, for example **/dashboard.",
            },
            "load_state": {
                "type": "string",
                "enum": ["load", "domcontentloaded", "networkidle"],
                "description": "Page load state to wait for. Defaults to networkidle.",
            },
            "milliseconds": {
                "type": "integer",
                "description": "Short delay in milliseconds, max 60000.",
            },
            "timeout_seconds": _timeout_schema(),
        },
        "additionalProperties": False,
    }


def _browser_screenshot_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Workspace-relative output path. Defaults to "
                    "var/browser/screenshots/<timestamp>.png."
                ),
            },
            "full_page": {
                "type": "boolean",
                "description": "Capture the full page instead of viewport only.",
            },
            "annotate": {
                "type": "boolean",
                "description": "Overlay element labels that correspond to snapshot refs.",
            },
            "timeout_seconds": _timeout_schema(),
        },
        "additionalProperties": False,
    }


def _browser_close_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "all": {
                "type": "boolean",
                "description": "Close all active agent-browser sessions. Defaults to false.",
            },
            "timeout_seconds": _timeout_schema(),
        },
        "additionalProperties": False,
    }


def _empty_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }


def _timeout_schema() -> Mapping[str, object]:
    return {
        "type": "integer",
        "description": (
            f"Backend command timeout in seconds. Defaults to "
            f"{DEFAULT_BROWSER_TIMEOUT_SECONDS}, max {MAX_BROWSER_TIMEOUT_SECONDS}."
        ),
    }
