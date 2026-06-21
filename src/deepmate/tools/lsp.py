"""Read-only Language Server Protocol tools."""

from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from deepmate.runtime.process_env import subprocess_environment
from deepmate.tools.filesystem import _relative_path, _workspace_path
from deepmate.tools.registry import NativeTool, NativeToolResult

LSP_DEFINITION_TOOL_NAME = "lsp_definition"
LSP_REFERENCES_TOOL_NAME = "lsp_references"
LSP_HOVER_TOOL_NAME = "lsp_hover"

DEFAULT_TIMEOUT_SECONDS = 8.0
MAX_TIMEOUT_SECONDS = 20.0
MAX_REFERENCES = 100
MAX_HOVER_CHARS = 4000
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_LSP_PAYLOAD_BYTES = 4 * 1024 * 1024
MAX_LSP_HEADER_BYTES = 64 * 1024


def workspace_lsp_tools(
    workspace_root: str | Path,
    *,
    server_resolver: Callable[[Path], Sequence[str] | None] | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[NativeTool, ...]:
    """Return read-only LSP definition/reference/hover tools for one workspace."""
    root = Path(workspace_root).resolve()
    if not root.is_dir():
        raise ValueError(f"workspace root must be a directory: {root}")
    client = _LspToolClient(
        root=root,
        server_resolver=server_resolver or _server_command_for_file,
        timeout_seconds=timeout_seconds,
    )
    return (
        NativeTool(
            name=LSP_DEFINITION_TOOL_NAME,
            description=(
                "Find the semantic definition for a symbol at a workspace file "
                "position using the language server when available."
            ),
            input_schema=_position_schema(),
            handler=lambda arguments: client.definition(arguments),
        ),
        NativeTool(
            name=LSP_REFERENCES_TOOL_NAME,
            description=(
                "Find semantic references for a symbol at a workspace file "
                "position using the language server when available."
            ),
            input_schema={
                **_position_schema(),
                "properties": {
                    **_position_schema()["properties"],
                    "include_declaration": {
                        "type": "boolean",
                        "description": "Whether to include the symbol declaration.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum references to return.",
                        "minimum": 1,
                        "maximum": MAX_REFERENCES,
                    },
                },
            },
            handler=lambda arguments: client.references(arguments),
        ),
        NativeTool(
            name=LSP_HOVER_TOOL_NAME,
            description=(
                "Return semantic type, signature, or documentation for a "
                "workspace file position using the language server when available."
            ),
            input_schema=_position_schema(),
            handler=lambda arguments: client.hover(arguments),
        ),
    )


class _LspToolClient:
    def __init__(
        self,
        *,
        root: Path,
        server_resolver: Callable[[Path], Sequence[str] | None],
        timeout_seconds: float,
    ) -> None:
        self._root = root
        self._server_resolver = server_resolver
        self._timeout_seconds = _bounded_float(
            timeout_seconds,
            DEFAULT_TIMEOUT_SECONDS,
            0.5,
            MAX_TIMEOUT_SECONDS,
        )

    def definition(self, arguments: Mapping[str, object]) -> NativeToolResult:
        request = self._position_request(arguments)
        result = self._request(
            request,
            "textDocument/definition",
            {
                "textDocument": {"uri": request.uri},
                "position": request.position_payload(),
            },
        )
        locations = _locations_from_lsp(result.payload)
        return _location_result(
            self._root,
            locations,
            empty_message="No LSP definition found.",
            unavailable=result.unavailable,
            operation="definition",
        )

    def references(self, arguments: Mapping[str, object]) -> NativeToolResult:
        request = self._position_request(arguments)
        max_results = _int(arguments, "max_results", 50, 1, MAX_REFERENCES)
        result = self._request(
            request,
            "textDocument/references",
            {
                "textDocument": {"uri": request.uri},
                "position": request.position_payload(),
                "context": {
                    "includeDeclaration": _bool(
                        arguments,
                        "include_declaration",
                        True,
                    )
                },
            },
        )
        locations = _locations_from_lsp(result.payload)
        truncated = len(locations) > max_results
        return _location_result(
            self._root,
            locations[:max_results],
            empty_message="No LSP references found.",
            unavailable=result.unavailable,
            operation="references",
            truncated=truncated,
            total_count=len(locations),
        )

    def hover(self, arguments: Mapping[str, object]) -> NativeToolResult:
        request = self._position_request(arguments)
        result = self._request(
            request,
            "textDocument/hover",
            {
                "textDocument": {"uri": request.uri},
                "position": request.position_payload(),
            },
        )
        if result.unavailable:
            return _unavailable_result(result.unavailable, operation="hover")
        content = _hover_text(result.payload)
        if not content:
            return NativeToolResult(
                content="No LSP hover information found.",
                data={
                    "operation": "hover",
                    "available": True,
                    "result_count": 0,
                },
            )
        truncated = len(content) > MAX_HOVER_CHARS
        rendered = content[:MAX_HOVER_CHARS].rstrip()
        if truncated:
            rendered = f"{rendered}\n... truncated ..."
        return NativeToolResult(
            content=rendered,
            data={
                "operation": "hover",
                "available": True,
                "truncated": truncated,
            },
            refs=(_relative_path(self._root, request.path),),
        )

    def _position_request(self, arguments: Mapping[str, object]) -> "_PositionRequest":
        path = _workspace_path(self._root, _required_text(arguments, "file"))
        if not path.is_file():
            raise ValueError(f"file does not exist: {_relative_path(self._root, path)}")
        line = _int(arguments, "line", 1, 1, 10_000_000)
        column = _int(arguments, "column", 1, 1, 10_000_000)
        return _PositionRequest(
            path=path,
            uri=path.as_uri(),
            line=line,
            column=column,
        )

    def _request(
        self,
        request: "_PositionRequest",
        method: str,
        params: Mapping[str, object],
    ) -> "_LspRequestResult":
        if request.path.stat().st_size > MAX_SOURCE_BYTES:
            return _LspRequestResult(
                unavailable="Source file is too large for on-demand LSP."
            )
        command = self._server_resolver(request.path)
        if not command:
            return _LspRequestResult(
                unavailable=f"No supported language server found for {request.path.suffix or 'file'}."
            )
        deadline = time.monotonic() + self._timeout_seconds
        process = None
        try:
            process = subprocess.Popen(
                list(command),
                cwd=str(self._root),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=False,
                env=subprocess_environment(),
            )
            client = _JsonRpcClient(process, deadline)
            initialize = client.request(
                "initialize",
                {
                    "processId": None,
                    "rootUri": self._root.as_uri(),
                    "capabilities": {},
                },
            )
            if initialize.get("error"):
                return _LspRequestResult(unavailable="Language server initialize failed.")
            client.notify("initialized", {})
            text = request.path.read_text(encoding="utf-8", errors="replace")
            client.notify(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": request.uri,
                        "languageId": _language_id(request.path),
                        "version": 1,
                        "text": text,
                    }
                },
            )
            response = client.request(method, params)
            if response.get("error"):
                return _LspRequestResult(unavailable=_error_message(response["error"]))
            client.notify(
                "textDocument/didClose",
                {"textDocument": {"uri": request.uri}},
            )
            try:
                client.request("shutdown", {})
                client.notify("exit", {})
            except (OSError, TimeoutError, json.JSONDecodeError):
                pass
            return _LspRequestResult(payload=response.get("result"))
        except (
            OSError,
            TimeoutError,
            ValueError,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ):
            return _LspRequestResult(unavailable="Language server request timed out or failed.")
        finally:
            if process is not None:
                _close_lsp_process_pipes(process)
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=0.5)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        process.kill()
                    except OSError:
                        pass


def _close_lsp_process_pipes(process: subprocess.Popen) -> None:
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is None:
            continue
        try:
            stream.close()
        except OSError:
            pass


class _JsonRpcClient:
    def __init__(self, process: subprocess.Popen, deadline: float) -> None:
        self._process = process
        self._deadline = deadline
        self._next_id = 1
        self._read_buffer = b""
        self.skipped_notifications = 0
        self.skipped_unmatched_responses = 0

    def request(self, method: str, params: Mapping[str, object]) -> Mapping[str, object]:
        request_id = self._next_id
        self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while True:
            message = self._read_message()
            if _jsonrpc_id_matches(message.get("id"), request_id):
                return message
            if "id" not in message:
                self.skipped_notifications += 1
            else:
                self.skipped_unmatched_responses += 1

    def notify(self, method: str, params: Mapping[str, object]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _send(self, payload: Mapping[str, object]) -> None:
        if time.monotonic() > self._deadline:
            raise TimeoutError
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        if len(body) > MAX_LSP_PAYLOAD_BYTES:
            raise OSError("language server request payload is too large")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        if self._process.stdin is None:
            raise OSError("language server stdin is unavailable")
        self._process.stdin.write(header + body)
        self._process.stdin.flush()

    def _read_message(self) -> Mapping[str, object]:
        if self._process.stdout is None:
            raise OSError("language server stdout is unavailable")
        length = 0
        while True:
            line = self._read_line()
            if not line:
                raise OSError("language server closed stdout")
            if line in (b"\r\n", b"\n"):
                break
            key, _, value = line.decode("ascii", errors="replace").partition(":")
            if key.lower() == "content-length":
                length = int(value.strip())
        if length <= 0:
            raise OSError("language server sent empty payload")
        if length > MAX_LSP_PAYLOAD_BYTES:
            raise OSError("language server payload is too large")
        body = self._read_bytes(length)
        if len(body) != length:
            raise OSError("language server payload was truncated")
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("language server payload must be an object")
        return payload

    def _read_line(self) -> bytes:
        while b"\n" not in self._read_buffer:
            self._read_more()
            if len(self._read_buffer) > MAX_LSP_HEADER_BYTES:
                raise OSError("language server header is too large")
        line, _, rest = self._read_buffer.partition(b"\n")
        self._read_buffer = rest
        return line + b"\n"

    def _read_bytes(self, length: int) -> bytes:
        while len(self._read_buffer) < length:
            self._read_more()
        body = self._read_buffer[:length]
        self._read_buffer = self._read_buffer[length:]
        return body

    def _read_more(self) -> None:
        if self._process.stdout is None:
            raise OSError("language server stdout is unavailable")
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        readable, _, _ = select.select([self._process.stdout], [], [], remaining)
        if not readable:
            raise TimeoutError
        chunk = os.read(self._process.stdout.fileno(), 4096)
        if not chunk:
            raise OSError("language server closed stdout")
        self._read_buffer += chunk
        if len(self._read_buffer) > MAX_LSP_PAYLOAD_BYTES + MAX_LSP_HEADER_BYTES:
            raise OSError("language server payload is too large")


class _PositionRequest:
    def __init__(self, *, path: Path, uri: str, line: int, column: int) -> None:
        self.path = path
        self.uri = uri
        self.line = line
        self.column = column

    def position_payload(self) -> Mapping[str, int]:
        return {"line": self.line - 1, "character": self.column - 1}


class _LspRequestResult:
    def __init__(self, payload: object = None, unavailable: str = "") -> None:
        self.payload = payload
        self.unavailable = unavailable


def _server_command_for_file(path: Path) -> Sequence[str] | None:
    suffix = path.suffix.lower()
    if suffix == ".py":
        for command in ("pyright-langserver", "pylsp"):
            executable = shutil.which(command)
            if executable:
                if command == "pyright-langserver":
                    return (executable, "--stdio")
                return (executable,)
        return None
    if suffix in {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}:
        executable = shutil.which("typescript-language-server")
        if executable:
            return (executable, "--stdio")
        return None
    if suffix == ".go":
        executable = shutil.which("gopls")
        if executable:
            return (executable,)
        return None
    return None


def _language_id(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix in {".ts", ".tsx"}:
        return "typescriptreact" if suffix == ".tsx" else "typescript"
    if suffix in {".js", ".jsx", ".mjs", ".cjs"}:
        return "javascriptreact" if suffix == ".jsx" else "javascript"
    if suffix == ".go":
        return "go"
    return "plaintext"


def _locations_from_lsp(payload: object) -> tuple[Mapping[str, object], ...]:
    if payload is None:
        return ()
    if isinstance(payload, Mapping):
        if "targetUri" in payload or "uri" in payload:
            return (payload,)
        return ()
    if isinstance(payload, list):
        return tuple(item for item in payload if isinstance(item, Mapping))
    return ()


def _jsonrpc_id_matches(response_id: object, request_id: int) -> bool:
    if response_id == request_id:
        return True
    if isinstance(response_id, str):
        return response_id == str(request_id)
    return False


def _location_result(
    root: Path,
    locations: Sequence[Mapping[str, object]],
    *,
    empty_message: str,
    unavailable: str,
    operation: str,
    truncated: bool = False,
    total_count: int | None = None,
) -> NativeToolResult:
    if unavailable:
        return _unavailable_result(unavailable, operation=operation)
    rendered = tuple(
        location
        for location in (_render_location(root, location) for location in locations)
        if location
    )
    if not rendered:
        return NativeToolResult(
            content=empty_message,
            data={
                "operation": operation,
                "available": True,
                "result_count": 0,
                "truncated": False,
            },
        )
    lines = [item["text"] for item in rendered]
    if truncated:
        lines.append("... truncated ...")
    refs = tuple(dict.fromkeys(str(item["path"]) for item in rendered))
    return NativeToolResult(
        content="\n".join(lines),
        data={
            "operation": operation,
            "available": True,
            "result_count": total_count if total_count is not None else len(rendered),
            "returned_count": len(rendered),
            "truncated": truncated,
        },
        refs=refs,
    )


def _render_location(root: Path, location: Mapping[str, object]) -> Mapping[str, object] | None:
    uri = _text_value(location.get("targetUri")) or _text_value(location.get("uri"))
    range_payload = location.get("targetSelectionRange") or location.get("range")
    if not uri or not isinstance(range_payload, Mapping):
        return None
    start = range_payload.get("start")
    if not isinstance(start, Mapping):
        return None
    path = _path_from_uri(uri)
    if path is None:
        return None
    try:
        relative = _relative_path(root, path.resolve())
    except ValueError:
        return None
    line = _int_value(start.get("line"), 0) + 1
    column = _int_value(start.get("character"), 0) + 1
    return {
        "path": relative,
        "line": line,
        "column": column,
        "text": f"{relative}:{line}:{column}",
    }


def _path_from_uri(uri: str) -> Path | None:
    prefix = "file://"
    if not uri.startswith(prefix):
        return None
    from urllib.parse import unquote, urlparse

    parsed = urlparse(uri)
    return Path(unquote(parsed.path))


def _hover_text(payload: object) -> str:
    if not isinstance(payload, Mapping):
        return ""
    return _marked_string_text(payload.get("contents")).strip()


def _marked_string_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        raw = value.get("value")
        return raw if isinstance(raw, str) else ""
    if isinstance(value, list):
        return "\n\n".join(
            text for text in (_marked_string_text(item).strip() for item in value) if text
        )
    return ""


def _unavailable_result(reason: str, *, operation: str) -> NativeToolResult:
    return NativeToolResult(
        content=f"LSP unavailable: {reason}",
        data={
            "operation": operation,
            "available": False,
            "reason": reason,
        },
        refs=("lsp_available=false",),
    )


def _position_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "file": {
                "type": "string",
                "description": "Workspace-relative source file path.",
            },
            "line": {
                "type": "integer",
                "description": "1-based line number.",
                "minimum": 1,
            },
            "column": {
                "type": "integer",
                "description": "1-based column number.",
                "minimum": 1,
            },
        },
        "required": ["file", "line", "column"],
        "additionalProperties": False,
    }


def _required_text(arguments: Mapping[str, object], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")
    return value.strip()


def _text_value(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _bool(arguments: Mapping[str, object], key: str, default: bool) -> bool:
    value = arguments.get(key, default)
    return value if isinstance(value, bool) else default


def _int(
    arguments: Mapping[str, object],
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    return min(max(_int_value(arguments.get(key), default), minimum), maximum)


def _int_value(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _bounded_float(value: float, default: float, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return min(max(number, minimum), maximum)


def _error_message(error: object) -> str:
    if isinstance(error, Mapping):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    return "Language server returned an error."
