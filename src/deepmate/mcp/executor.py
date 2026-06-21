"""Execute discovered read-only MCP tool requests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deepmate.domain import ErrorInfo, RuntimeEvent
from deepmate.mcp.client import McpClientError, McpClientSession, create_mcp_client
from deepmate.mcp.output_policy import McpOutputPolicy
from deepmate.mcp.spec import McpServerSpec, McpToolRef
from deepmate.mcp.state import McpUsageStateStore
from deepmate.providers import ModelToolRequest, ModelToolResult

HOOK_DIRECTIVE_CONTINUE = "continue"
HOOK_DIRECTIVE_REQUIRES_APPROVAL = "requires_approval"
HOOK_EVENT_MCP_BEFORE = "mcp.before"
HOOK_EVENT_MCP_AFTER = "mcp.after"


@dataclass(frozen=True, slots=True)
class McpToolExecutionResult:
    """Runtime result produced by one MCP tool execution attempt."""

    request: ModelToolRequest
    model_result: ModelToolResult | None = None
    error: ErrorInfo | None = None
    events: tuple[RuntimeEvent, ...] = field(default_factory=tuple)

    def is_success(self) -> bool:
        """Return whether the tool request completed without runtime error."""
        return self.error is None and self.model_result is not None


@dataclass(frozen=True, slots=True)
class _HookOutcomeView:
    directive: str = HOOK_DIRECTIVE_CONTINUE
    reason: str = ""
    refs: tuple[str, ...] = field(default_factory=tuple)


class McpToolExecutor:
    """Executor for discovered read-only MCP tools."""

    def __init__(
        self,
        servers: Iterable[McpServerSpec],
        tools: Iterable[McpToolRef],
        workspace: str | Path,
        usage_state_store: McpUsageStateStore | None = None,
        output_policy: McpOutputPolicy | None = None,
        allow_write_tools: bool = False,
        hook_context: Any = None,
    ) -> None:
        self._servers = {server.name.strip(): server for server in servers}
        self._tools = {tool.qualified_name(): tool for tool in tools}
        self._workspace = Path(workspace)
        self._clients: dict[str, McpClientSession] = {}
        self._usage_state_store = usage_state_store
        self._output_policy = output_policy
        self._allow_write_tools = allow_write_tools
        self._hook_context = hook_context

    def __enter__(self) -> "McpToolExecutor":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def has_tool(self, name: str) -> bool:
        """Return whether a qualified MCP tool is known."""
        return name.strip() in self._tools

    def tool_schema(self, name: str) -> Mapping[str, object] | None:
        """Return the schema for a known qualified MCP tool."""
        tool = self._tools.get(name.strip())
        if tool is None:
            return None
        return tool.schema()

    def execute(self, request: ModelToolRequest) -> McpToolExecutionResult:
        """Execute one qualified, read-only MCP tool request."""
        tool_name = _request_name(request)
        request_id = _request_id(request)
        if not tool_name:
            return _failure(
                request=request,
                code="mcp_tool_request_invalid",
                message="MCP tool request requires a tool name and tool call id.",
                event_kind="mcp_tool_request_invalid",
            )
        if not request_id:
            return _failure(
                request=request,
                code="mcp_tool_request_invalid",
                message="MCP tool request requires a tool call id.",
                event_kind="mcp_tool_request_invalid",
                refs=(tool_name, "tool_call_id=<empty>"),
            )
        if _request_argument_error(request):
            return _failure(
                request=request,
                code="mcp_tool_arguments_invalid",
                message=(
                    f"MCP tool arguments invalid for {tool_name}: "
                    f"{_request_argument_error(request)}"
                ),
                event_kind="mcp_tool_arguments_invalid",
                refs=(tool_name,),
            )
        tool = self._tools.get(tool_name)
        if tool is None:
            return _failure(
                request=request,
                code="mcp_tool_not_found",
                message=f"MCP tool not found: {tool_name}",
                event_kind="mcp_tool_not_found",
                refs=(tool_name,),
            )
        server = self._servers.get(tool.server_name)
        if not tool.is_read_only() and not self._allow_write_tools:
            return _failure(
                request=request,
                code="mcp_tool_not_read_only",
                message=(
                    f"MCP tool is not marked read-only: {tool_name}. "
                    "Enable MCP write access before calling it."
                ),
                event_kind="mcp_tool_not_read_only",
                refs=(tool_name,),
            )
        before_outcome = _emit_mcp_hook(
            self._hook_context,
            HOOK_EVENT_MCP_BEFORE,
            tool=tool,
            status="before",
            payload={
                "read_only": tool.is_read_only(),
                "schema_loaded": True,
                "transport": server.transport.value if server is not None else "",
                "allow_write_tools": self._allow_write_tools,
            },
        )
        if before_outcome.directive != HOOK_DIRECTIVE_CONTINUE:
            failure_code = (
                "mcp_tool_requires_approval_by_hook"
                if before_outcome.directive == HOOK_DIRECTIVE_REQUIRES_APPROVAL
                else "mcp_tool_blocked_by_hook"
            )
            return _failure(
                request=request,
                code=failure_code,
                message=(
                    before_outcome.reason
                    or f"MCP tool stopped by hook: {before_outcome.directive}"
                ),
                event_kind=failure_code,
                refs=(
                    f"hook_event={HOOK_EVENT_MCP_BEFORE}",
                    f"hook_directive={before_outcome.directive}",
                    *before_outcome.refs,
                ),
            )
        if server is None:
            return _failure(
                request=request,
                code="mcp_server_not_found",
                message=f"MCP server not found for tool: {tool_name}",
                event_kind="mcp_server_not_found",
                refs=(tool_name, tool.server_name),
            )

        retry_count = 0
        try:
            for attempt in range(2):
                try:
                    client = self._client_for(server)
                    result = client.call_tool(tool.name, request.arguments)
                    diagnostic_refs = client.diagnostic_refs()
                    break
                except (OSError, McpClientError):
                    self._drop_client(server.name)
                    if attempt == 0:
                        retry_count = 1
                        continue
                    raise
            else:  # pragma: no cover - the loop always breaks or raises.
                raise McpClientError("MCP tool failed without a response")
            if self._usage_state_store is not None:
                self._usage_state_store.record_tool_invoked(tool)
        except (OSError, McpClientError, ValueError) as exc:
            return _failure(
                request=request,
                code="mcp_tool_failed",
                message=f"MCP tool failed: {tool_name}: {exc}",
                event_kind="mcp_tool_failed",
                refs=(tool_name, tool.server_name),
            )

        policy_result = (
            self._output_policy.apply(tool, result)
            if self._output_policy is not None
            else None
        )
        content = policy_result.content if policy_result is not None else result.content
        data = (
            policy_result.data
            if policy_result is not None
            else _result_data(result.data)
        )
        refs = _tool_refs(
            tool,
            (
                *diagnostic_refs,
                *(("mcp_retry_count=1",) if retry_count else ()),
                *(policy_result.refs if policy_result is not None else ()),
                *_hook_refs(before_outcome),
            ),
        )
        after_outcome = _emit_mcp_hook(
            self._hook_context,
            HOOK_EVENT_MCP_AFTER,
            tool=tool,
            status="failed" if result.is_error else "completed",
            payload={
                "read_only": tool.is_read_only(),
                "schema_loaded": True,
                "allow_write_tools": self._allow_write_tools,
            },
        )
        if after_outcome.refs:
            refs = (*refs, *_hook_refs(after_outcome))
        model_result = ModelToolResult(
            name=tool_name,
            request_id=request_id,
            content=content,
            data=data,
            refs=refs,
            is_error=result.is_error,
        )
        events = []
        if policy_result is not None and policy_result.truncated:
            events.append(
                RuntimeEvent(
                    kind="mcp_tool_output_truncated",
                    summary=f"MCP tool output truncated: {tool_name}.",
                    refs=refs,
                )
            )
        if before_outcome.refs:
            events.append(
                RuntimeEvent(
                    kind="mcp_before_hook_observed",
                    summary=f"MCP before hook observed: {tool_name}.",
                    refs=_hook_refs(before_outcome),
                )
            )
        events.append(
            RuntimeEvent(
                kind="mcp_tool_completed",
                summary=f"MCP tool completed: {tool_name}.",
                refs=refs,
            )
        )
        if after_outcome.refs:
            events.append(
                RuntimeEvent(
                    kind="mcp_after_hook_observed",
                    summary=f"MCP after hook observed: {tool_name}.",
                    refs=_hook_refs(after_outcome),
                )
            )
        return McpToolExecutionResult(
            request=request,
            model_result=model_result,
            error=(
                ErrorInfo(
                    code="mcp_tool_returned_error",
                    message=f"MCP tool returned an error: {tool_name}",
                    refs=refs,
                )
                if result.is_error
                else None
            ),
            events=tuple(events),
        )

    def close(self) -> None:
        """Close all live MCP stdio clients owned by this executor."""
        clients = tuple(self._clients.values())
        self._clients.clear()
        for client in clients:
            try:
                client.close()
            except Exception:
                # Cleanup must not hide the tool/provider error that caused it.
                pass

    def _client_for(self, server: McpServerSpec) -> McpClientSession:
        server_name = server.name.strip()
        client = self._clients.get(server_name)
        if client is None:
            client = create_mcp_client(server, self._workspace)
            try:
                client.connect()
            except Exception:
                try:
                    client.close()
                except Exception:
                    pass
                raise
            self._clients[server_name] = client
        else:
            client.connect()
        return client

    def _drop_client(self, server_name: str) -> None:
        client = self._clients.pop(server_name.strip(), None)
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


def _result_data(value: Mapping[str, object]) -> Mapping[str, object]:
    if not value:
        return {}
    return {key: data for key, data in value.items() if key != "content"}


def _tool_refs(
    tool: McpToolRef,
    diagnostic_refs: tuple[str, ...] = (),
) -> tuple[str, ...]:
    return (
        f"mcp_tool={tool.qualified_name()}",
        f"mcp_server={tool.server_name}",
        f"mcp_server_tool={tool.name}",
        *diagnostic_refs,
    )


def _emit_mcp_hook(
    hook_context: Any,
    event_name: str,
    *,
    tool: McpToolRef,
    status: str,
    payload: Mapping[str, object],
) -> _HookOutcomeView:
    if hook_context is None:
        return _HookOutcomeView()
    from deepmate.runtime.hooks.types import HookActor, HookEnvelope, HookEvent

    outcome = hook_context.emit(
        HookEnvelope(
            event_name=HookEvent(event_name),
            actor=HookActor.MAIN,
            payload={
                "tool_name": tool.qualified_name(),
                "tool_source": "mcp",
                "server_name": tool.server_name,
                "qualified_name": tool.qualified_name(),
                "status": status,
                "actor": HookActor.MAIN.value,
                **payload,
            },
            source_refs=(
                f"mcp_tool={tool.qualified_name()}",
                f"mcp_server={tool.server_name}",
                *hook_context.trace_refs(),
            ),
        )
    )
    return _HookOutcomeView(
        directive=outcome.directive.value,
        reason=outcome.reason,
        refs=outcome.refs,
    )


def _hook_refs(outcome: _HookOutcomeView) -> tuple[str, ...]:
    if not outcome.refs:
        return ()
    return (f"hook_directive={outcome.directive}", *outcome.refs)


def _failure(
    request: ModelToolRequest,
    code: str,
    message: str,
    event_kind: str,
    refs: tuple[str, ...] = (),
) -> McpToolExecutionResult:
    error = ErrorInfo(code=code, message=message, refs=refs)
    model_result = None
    tool_name = _request_name(request)
    request_id = _request_id(request)
    if tool_name and request_id:
        model_result = ModelToolResult(
            name=tool_name,
            request_id=request_id,
            content=message,
            refs=refs,
            is_error=True,
        )
    return McpToolExecutionResult(
        request=request,
        model_result=model_result,
        error=error,
        events=(RuntimeEvent(kind=event_kind, summary=message, refs=refs),),
    )


def _request_name(request: ModelToolRequest) -> str:
    if not isinstance(request.name, str):
        return ""
    return request.name.strip()


def _request_argument_error(request: ModelToolRequest) -> str:
    if not isinstance(request.argument_error, str):
        return ""
    return request.argument_error.strip()


def _request_id(request: ModelToolRequest) -> str:
    if not isinstance(request.id, str):
        return ""
    return request.id.strip()
