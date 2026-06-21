"""Minimal MCP clients for tool discovery and read-only calls."""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TextIO

from deepmate.mcp.spec import McpServerSpec, McpToolRef, McpTransport

MCP_PROTOCOL_VERSION = "2025-11-25"
MCP_REQUEST_TIMEOUT_SECONDS = 30.0
MAX_MCP_JSON_BYTES = 1_000_000
MAX_MCP_REQUESTS_PER_MINUTE = 600
MAX_MCP_INVALID_JSON_LINES = 20
DEFAULT_INHERITED_ENV_VARS = (
    (
        "APPDATA",
        "HOMEDRIVE",
        "HOMEPATH",
        "LOCALAPPDATA",
        "PATH",
        "PROCESSOR_ARCHITECTURE",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "USERNAME",
        "USERPROFILE",
        "PROGRAMFILES",
    )
    if sys.platform == "win32"
    else (
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LOGNAME",
        "PATH",
        "SHELL",
        "TERM",
        "USER",
    )
)


class McpClientError(RuntimeError):
    """Raised when MCP communication fails."""


@dataclass(frozen=True, slots=True)
class McpCallResult:
    """Result returned by an MCP tool call."""

    content: str = ""
    data: Mapping[str, object] = field(default_factory=dict)
    is_error: bool = False

    def has_output(self) -> bool:
        """Return whether the call produced model-facing output."""
        return bool(self.content.strip() or self.data or self.is_error)


class McpClientSession(Protocol):
    """Shared MCP client interface used by discovery and execution."""

    def __enter__(self) -> "McpClientSession": ...

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None: ...

    def connect(self) -> None: ...

    def list_tools(self) -> tuple[McpToolRef, ...]: ...

    def server_info(self) -> Mapping[str, object]: ...

    def server_capabilities(self) -> Mapping[str, object]: ...

    def instructions(self) -> str: ...

    def call_tool(
        self,
        tool_name: str,
        arguments: Mapping[str, object],
    ) -> McpCallResult: ...

    def close(self) -> None: ...

    def diagnostic_refs(self) -> tuple[str, ...]: ...


class McpStdioClient:
    """Synchronous stdio MCP client for one server process."""

    def __init__(self, spec: McpServerSpec, workspace: str | Path) -> None:
        if spec.transport != McpTransport.STDIO:
            raise ValueError("McpStdioClient only supports stdio servers")
        if not spec.command.strip():
            raise ValueError("stdio MCP server requires command")
        self._spec = spec
        self._workspace = Path(workspace)
        self._process: subprocess.Popen[str] | None = None
        self._next_id = 1
        self._skipped_notifications = 0
        self._skipped_unmatched_responses = 0
        self._server_info: Mapping[str, object] = {}
        self._server_capabilities: Mapping[str, object] = {}
        self._instructions = ""
        self._request_timestamps: list[float] = []

    def __enter__(self) -> "McpStdioClient":
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def connect(self) -> None:
        """Start the server process and initialize the MCP session."""
        if self._process is not None:
            return
        process = subprocess.Popen(
            [self._spec.command, *self._spec.args],
            cwd=self._cwd(),
            env=self._env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        self._process = process
        try:
            response = self._request(
                "initialize",
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "deepmate", "version": "0.1.0"},
                },
            )
            if "result" not in response:
                raise McpClientError(f"MCP initialize failed: {response}")
            result = _mapping(response.get("result"))
            self._server_info = _mapping(result.get("serverInfo"))
            self._server_capabilities = _mapping(result.get("capabilities"))
            instructions = result.get("instructions")
            self._instructions = instructions.strip() if isinstance(instructions, str) else ""
            self._notify("notifications/initialized", {})
        except Exception:
            self.close()
            raise

    def list_tools(self) -> tuple[McpToolRef, ...]:
        """Return tools discovered from the server."""
        refs: list[McpToolRef] = []
        cursor = ""
        while True:
            params = {"cursor": cursor} if cursor else {}
            response = self._request("tools/list", params)
            result = _mapping(response.get("result"))
            tools = result.get("tools")
            if not isinstance(tools, list):
                raise McpClientError("MCP tools/list response missing tools list")
            refs.extend(self._tool_refs_from_items(tools))
            next_cursor = result.get("nextCursor")
            if not isinstance(next_cursor, str) or not next_cursor.strip():
                break
            cursor = next_cursor.strip()
        return tuple(refs)

    def server_info(self) -> Mapping[str, object]:
        """Return serverInfo from initialize, if provided."""
        return dict(self._server_info)

    def server_capabilities(self) -> Mapping[str, object]:
        """Return server capabilities from initialize, if provided."""
        return dict(self._server_capabilities)

    def instructions(self) -> str:
        """Return server instructions from initialize, if provided."""
        return self._instructions

    def call_tool(
        self,
        tool_name: str,
        arguments: Mapping[str, object],
    ) -> McpCallResult:
        """Call one MCP tool by server-local name."""
        if not tool_name.strip():
            raise ValueError("MCP tool name is required")
        response = self._request(
            "tools/call",
            {"name": tool_name.strip(), "arguments": dict(arguments)},
        )
        result = _mapping(response.get("result"))
        return McpCallResult(
            content=_content_text(result.get("content")),
            data=result,
            is_error=result.get("isError") is True,
        )

    def close(self) -> None:
        """Terminate the server process."""
        process = self._process
        if process is None:
            return
        self._process = None
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.stdout is not None:
            try:
                process.stdout.close()
            except OSError:
                pass
        if process.stderr is not None:
            try:
                process.stderr.close()
            except OSError:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def diagnostic_refs(self) -> tuple[str, ...]:
        """Return lightweight communication diagnostics for trace refs."""
        refs: list[str] = []
        if self._skipped_notifications:
            refs.append(f"mcp_notifications_skipped={self._skipped_notifications}")
        if self._skipped_unmatched_responses:
            refs.append(
                "mcp_unmatched_responses_skipped="
                f"{self._skipped_unmatched_responses}"
            )
        return tuple(refs)

    def _request(self, method: str, params: Mapping[str, object]) -> Mapping[str, object]:
        request_id = self._next_id
        self._next_id += 1
        self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while True:
            response = self._read()
            if not _jsonrpc_id_matches(response.get("id"), request_id):
                if "id" not in response:
                    self._skipped_notifications += 1
                else:
                    self._skipped_unmatched_responses += 1
                continue
            error = response.get("error")
            if error:
                raise McpClientError(f"MCP request failed: {error}")
            return response

    def _notify(self, method: str, params: Mapping[str, object]) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def _write(self, payload: Mapping[str, object]) -> None:
        process = self._ready_process()
        stdin = process.stdin
        if stdin is None:
            raise McpClientError("MCP process stdin is not available")
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        _check_mcp_payload_size(text.encode("utf-8"), "request")
        self._enforce_request_rate()
        stdin.write(text + "\n")
        stdin.flush()

    def _enforce_request_rate(self) -> None:
        _enforce_mcp_request_rate(self._request_timestamps)

    def _read(self) -> Mapping[str, object]:
        process = self._ready_process()
        stdout = process.stdout
        if stdout is None:
            raise McpClientError("MCP process stdout is not available")
        invalid_json_lines = 0
        while True:
            if not _wait_readable(stdout, self._request_timeout_seconds()):
                raise McpClientError(
                    f"MCP server did not respond within "
                    f"{self._request_timeout_seconds():g} seconds"
                )
            line = stdout.readline(MAX_MCP_JSON_BYTES + 1)
            if line:
                _check_mcp_payload_size(line.encode("utf-8"), "response")
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    invalid_json_lines += 1
                    if invalid_json_lines > MAX_MCP_INVALID_JSON_LINES:
                        raise McpClientError(
                            "MCP server returned too many invalid JSON lines"
                        )
                    continue
                if isinstance(parsed, Mapping):
                    return parsed
                invalid_json_lines += 1
                if invalid_json_lines > MAX_MCP_INVALID_JSON_LINES:
                    raise McpClientError(
                        "MCP server returned too many non-object responses"
                    )
                continue
            if stdout.closed:
                raise McpClientError("MCP server stdout closed without a response")
            if process.poll() is not None:
                raise McpClientError(_process_exit_message(process))
            raise McpClientError("MCP server stdout reached EOF without exiting")

    def _ready_process(self) -> subprocess.Popen[str]:
        process = self._process
        if process is None:
            raise McpClientError("MCP process is not connected")
        if process.poll() is not None:
            raise McpClientError(_process_exit_message(process))
        return process

    def _cwd(self) -> str:
        if not self._spec.cwd.strip():
            return str(self._workspace)
        path = Path(self._spec.cwd)
        if not path.is_absolute():
            path = self._workspace / path
        return str(path)

    def _env(self) -> Mapping[str, str]:
        env = _default_environment()
        env.update({key: value for key, value in self._spec.env.items() if key.strip()})
        return env

    def _request_timeout_seconds(self) -> float:
        timeout = self._spec.startup_timeout_seconds
        if timeout is None:
            return MCP_REQUEST_TIMEOUT_SECONDS
        return max(0.1, timeout)

    def _tool_refs_from_items(self, tools: list[object]) -> list[McpToolRef]:
        refs: list[McpToolRef] = []
        for item in tools:
            if not isinstance(item, Mapping):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            title = item.get("title")
            description = item.get("description")
            input_schema = item.get("inputSchema")
            output_schema = item.get("outputSchema")
            annotations = item.get("annotations")
            meta = item.get("_meta")
            refs.append(
                McpToolRef(
                    server_name=self._spec.name,
                    name=name.strip(),
                    title=title.strip() if isinstance(title, str) else "",
                    description=description if isinstance(description, str) else "",
                    input_schema=_mapping(input_schema),
                    output_schema=_mapping(output_schema),
                    annotations=_mapping(annotations),
                    meta=_mapping(meta),
                )
            )
        return refs


def _check_mcp_payload_size(payload: bytes, kind: str) -> None:
    if len(payload) > MAX_MCP_JSON_BYTES:
        if kind == "request":
            raise McpClientError("MCP request payload is too large")
        raise McpClientError("MCP response payload is too large")


def _enforce_mcp_request_rate(timestamps: list[float]) -> None:
    now = time.monotonic()
    window_start = now - 60.0
    timestamps[:] = [value for value in timestamps if value >= window_start]
    if len(timestamps) >= MAX_MCP_REQUESTS_PER_MINUTE:
        raise McpClientError("MCP request rate limit exceeded")
    timestamps.append(now)


class McpStreamableHttpClient:
    """Synchronous Streamable HTTP MCP client for URL-based servers."""

    def __init__(self, spec: McpServerSpec, workspace: str | Path) -> None:
        if spec.transport == McpTransport.STDIO:
            raise ValueError("McpStreamableHttpClient requires a URL transport")
        if not spec.url.strip():
            raise ValueError("URL MCP server requires url")
        self._spec = spec
        self._workspace = Path(workspace)
        self._next_id = 1
        self._connected = False
        self._session_id = ""
        self._skipped_notifications = 0
        self._skipped_unmatched_responses = 0
        self._server_info: Mapping[str, object] = {}
        self._server_capabilities: Mapping[str, object] = {}
        self._instructions = ""
        self._request_timestamps: list[float] = []

    def __enter__(self) -> "McpStreamableHttpClient":
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def connect(self) -> None:
        """Initialize the HTTP MCP session."""
        if self._connected:
            return
        response = self._request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "deepmate", "version": "0.1.0"},
            },
        )
        if "result" not in response:
            raise McpClientError(f"MCP initialize failed: {response}")
        result = _mapping(response.get("result"))
        self._server_info = _mapping(result.get("serverInfo"))
        self._server_capabilities = _mapping(result.get("capabilities"))
        instructions = result.get("instructions")
        self._instructions = instructions.strip() if isinstance(instructions, str) else ""
        self._notify("notifications/initialized", {})
        self._connected = True

    def list_tools(self) -> tuple[McpToolRef, ...]:
        """Return tools discovered from the server."""
        refs: list[McpToolRef] = []
        cursor = ""
        while True:
            params = {"cursor": cursor} if cursor else {}
            response = self._request("tools/list", params)
            result = _mapping(response.get("result"))
            tools = result.get("tools")
            if not isinstance(tools, list):
                raise McpClientError("MCP tools/list response missing tools list")
            refs.extend(self._tool_refs_from_items(tools))
            next_cursor = result.get("nextCursor")
            if not isinstance(next_cursor, str) or not next_cursor.strip():
                break
            cursor = next_cursor.strip()
        return tuple(refs)

    def server_info(self) -> Mapping[str, object]:
        """Return serverInfo from initialize, if provided."""
        return dict(self._server_info)

    def server_capabilities(self) -> Mapping[str, object]:
        """Return server capabilities from initialize, if provided."""
        return dict(self._server_capabilities)

    def instructions(self) -> str:
        """Return server instructions from initialize, if provided."""
        return self._instructions

    def call_tool(
        self,
        tool_name: str,
        arguments: Mapping[str, object],
    ) -> McpCallResult:
        """Call one MCP tool by server-local name."""
        if not tool_name.strip():
            raise ValueError("MCP tool name is required")
        response = self._request(
            "tools/call",
            {"name": tool_name.strip(), "arguments": dict(arguments)},
        )
        result = _mapping(response.get("result"))
        return McpCallResult(
            content=_content_text(result.get("content")),
            data=result,
            is_error=result.get("isError") is True,
        )

    def close(self) -> None:
        """Terminate the HTTP session when the server supports it."""
        if self._session_id:
            try:
                request = urllib.request.Request(
                    self._spec.url.strip(),
                    headers=self._headers(include_content_type=False),
                    method="DELETE",
                )
                urllib.request.urlopen(
                    request,
                    timeout=self._request_timeout_seconds(),
                ).close()
            except (OSError, urllib.error.URLError):
                pass
        self._connected = False
        self._session_id = ""

    def diagnostic_refs(self) -> tuple[str, ...]:
        """Return lightweight communication diagnostics for trace refs."""
        refs = [f"mcp_transport={self._spec.transport.value}"]
        if self._skipped_notifications:
            refs.append(f"mcp_notifications_skipped={self._skipped_notifications}")
        if self._skipped_unmatched_responses:
            refs.append(
                "mcp_unmatched_responses_skipped="
                f"{self._skipped_unmatched_responses}"
            )
        return tuple(refs)

    def _request(self, method: str, params: Mapping[str, object]) -> Mapping[str, object]:
        request_id = self._next_id
        self._next_id += 1
        responses = self._send_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            },
            expect_response=True,
        )
        return self._matching_response(responses, request_id)

    def _notify(self, method: str, params: Mapping[str, object]) -> None:
        self._send_jsonrpc(
            {"jsonrpc": "2.0", "method": method, "params": params},
            expect_response=False,
        )

    def _send_jsonrpc(
        self,
        payload: Mapping[str, object],
        *,
        expect_response: bool,
    ) -> tuple[Mapping[str, object], ...]:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        _check_mcp_payload_size(body, "request")
        self._enforce_request_rate()
        request = urllib.request.Request(
            self._spec.url.strip(),
            data=body,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self._request_timeout_seconds(),
            ) as response:
                session_id = response.headers.get("Mcp-Session-Id")
                if isinstance(session_id, str) and session_id.strip():
                    self._session_id = session_id.strip()
                raw = response.read(MAX_MCP_JSON_BYTES + 1)
                _check_mcp_payload_size(raw, "response")
                status = getattr(response, "status", 200)
                content_type = response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            detail = _http_error_detail(exc)
            raise McpClientError(
                f"MCP HTTP request failed with status {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise McpClientError(f"MCP HTTP request failed: {exc}") from exc

        if status in {202, 204} or not raw:
            if expect_response:
                raise McpClientError("MCP HTTP request returned no JSON-RPC response")
            return ()
        return _http_response_messages(raw, content_type)

    def _enforce_request_rate(self) -> None:
        _enforce_mcp_request_rate(self._request_timestamps)

    def _matching_response(
        self,
        messages: tuple[Mapping[str, object], ...],
        request_id: int,
    ) -> Mapping[str, object]:
        for response in messages:
            if not _jsonrpc_id_matches(response.get("id"), request_id):
                if "id" not in response:
                    self._skipped_notifications += 1
                else:
                    self._skipped_unmatched_responses += 1
                continue
            error = response.get("error")
            if error:
                raise McpClientError(f"MCP request failed: {error}")
            return response
        raise McpClientError(f"MCP HTTP response missing id {request_id}")

    def _headers(self, *, include_content_type: bool = True) -> Mapping[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
        }
        if include_content_type:
            headers["Content-Type"] = "application/json"
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        token_env_var = self._spec.bearer_token_env_var.strip()
        if token_env_var:
            token = os.environ.get(token_env_var, "").strip()
            if not token:
                raise McpClientError(
                    f"MCP bearer token env var is not set: {token_env_var}"
                )
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _request_timeout_seconds(self) -> float:
        timeout = self._spec.startup_timeout_seconds
        if timeout is None:
            return MCP_REQUEST_TIMEOUT_SECONDS
        return max(0.1, timeout)

    def _tool_refs_from_items(self, tools: list[object]) -> list[McpToolRef]:
        refs: list[McpToolRef] = []
        for item in tools:
            if not isinstance(item, Mapping):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            title = item.get("title")
            description = item.get("description")
            input_schema = item.get("inputSchema")
            output_schema = item.get("outputSchema")
            annotations = item.get("annotations")
            meta = item.get("_meta")
            refs.append(
                McpToolRef(
                    server_name=self._spec.name,
                    name=name.strip(),
                    title=title.strip() if isinstance(title, str) else "",
                    description=description if isinstance(description, str) else "",
                    input_schema=_mapping(input_schema),
                    output_schema=_mapping(output_schema),
                    annotations=_mapping(annotations),
                    meta=_mapping(meta),
                )
            )
        return refs


def create_mcp_client(
    server: McpServerSpec,
    workspace: str | Path,
) -> McpClientSession:
    """Create a client for the configured MCP transport."""
    if server.transport == McpTransport.STDIO:
        return McpStdioClient(server, workspace)
    if server.transport in {
        McpTransport.HTTP,
        McpTransport.SSE,
        McpTransport.STREAMABLE_HTTP,
    }:
        return McpStreamableHttpClient(server, workspace)
    raise ValueError(f"unsupported MCP transport: {server.transport}")


def discover_mcp_tools(
    server: McpServerSpec,
    workspace: str | Path,
) -> tuple[McpToolRef, ...]:
    """Discover all tools from one MCP server."""
    with create_mcp_client(server, workspace) as client:
        return client.list_tools()


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _content_text(value: object) -> str:
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        text = item.get("text")
        if item.get("type") == "text" and isinstance(text, str):
            parts.append(text)
        else:
            parts.append(_non_text_content_summary(item))
    return "\n".join(part for part in parts if part.strip())


def _non_text_content_summary(item: Mapping[str, object]) -> str:
    content_type = item.get("type")
    mime_type = item.get("mimeType") or item.get("mime_type")
    parts = ["[MCP non-text content"]
    if isinstance(content_type, str) and content_type.strip():
        parts.append(f"type={content_type.strip()}")
    if isinstance(mime_type, str) and mime_type.strip():
        parts.append(f"mime={mime_type.strip()}")
    parts.append("omitted from text replay]")
    return " ".join(parts)


def _process_exit_message(process: subprocess.Popen[str]) -> str:
    stderr = _read_stderr(process.stderr)
    suffix = f": {stderr}" if stderr else ""
    return f"MCP process exited with code {process.returncode}{suffix}"


def _read_stderr(stderr: TextIO | None) -> str:
    if stderr is None:
        return ""
    try:
        return stderr.read(801).strip()[:800]
    except OSError:
        return ""


def _wait_readable(stream: TextIO, timeout_seconds: float) -> bool:
    if sys.platform == "win32":
        return _wait_windows_pipe_readable(stream, timeout_seconds)
    try:
        ready, _, _ = select.select([stream], [], [], timeout_seconds)
    except (OSError, ValueError) as exc:
        raise McpClientError(f"MCP stdout wait failed: {exc}") from exc
    return bool(ready)


def _wait_windows_pipe_readable(stream: TextIO, timeout_seconds: float) -> bool:
    import ctypes
    import msvcrt
    import time

    try:
        handle = msvcrt.get_osfhandle(stream.fileno())
    except (AttributeError, OSError) as exc:
        raise McpClientError(f"MCP stdout wait failed: {exc}") from exc

    available = ctypes.c_ulong(0)
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        ok = ctypes.windll.kernel32.PeekNamedPipe(
            handle,
            None,
            0,
            None,
            ctypes.byref(available),
            None,
        )
        if not ok:
            raise McpClientError("MCP stdout wait failed: PeekNamedPipe failed")
        if available.value > 0:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.05, remaining))


def _default_environment() -> dict[str, str]:
    env: dict[str, str] = {}
    for key in DEFAULT_INHERITED_ENV_VARS:
        value = os.environ.get(key)
        if value is None or value.startswith("()"):
            continue
        env[key] = value
    return env


def _http_response_messages(
    raw: bytes,
    content_type: str,
) -> tuple[Mapping[str, object], ...]:
    text = raw.decode("utf-8", errors="replace")
    if "text/event-stream" in content_type.lower() or text.lstrip().startswith(
        ("event:", "data:", ":")
    ):
        return _sse_json_messages(text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise McpClientError("MCP HTTP response is not valid JSON") from exc
    return _json_messages(parsed)


def _sse_json_messages(text: str) -> tuple[Mapping[str, object], ...]:
    messages: list[Mapping[str, object]] = []
    data_lines: list[str] = []

    def flush() -> None:
        if not data_lines:
            return
        data = "\n".join(data_lines).strip()
        data_lines.clear()
        if not data or data == "[DONE]":
            return
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            return
        messages.extend(_json_messages(parsed))

    for line in text.splitlines():
        if not line:
            flush()
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if not separator:
            continue
        if value.startswith(" "):
            value = value[1:]
        if field == "data":
            data_lines.append(value)
    flush()
    if not messages:
        raise McpClientError("MCP SSE response did not contain JSON-RPC data")
    return tuple(messages)


def _json_messages(value: object) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, Mapping):
        return (dict(value),)
    if isinstance(value, list):
        return tuple(dict(item) for item in value if isinstance(item, Mapping))
    raise McpClientError("MCP response must be a JSON-RPC object or batch")


def _jsonrpc_id_matches(response_id: object, request_id: int) -> bool:
    if response_id == request_id:
        return True
    if isinstance(response_id, str):
        return response_id == str(request_id)
    return False


def _http_error_detail(error: urllib.error.HTTPError) -> str:
    try:
        detail = error.read(801).decode("utf-8", errors="replace").strip()
    except OSError:
        detail = ""
    return detail[:800] if detail else str(error.reason)
