"""Execute native tool requests from model responses."""

from __future__ import annotations

from dataclasses import dataclass, field

from deepmate.domain import ErrorInfo, RuntimeEvent
from deepmate.providers import ModelToolRequest, ModelToolResult
from deepmate.runtime.tool_policy import ToolAccessPolicy
from deepmate.tools import NativeToolRegistry, NativeToolResult

NATIVE_TOOL_RECOVERABLE_ERRORS = (
    OSError,
    RuntimeError,
    TypeError,
    UnicodeError,
    ValueError,
)


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """Runtime result produced by one native tool execution attempt."""

    request: ModelToolRequest
    model_result: ModelToolResult | None = None
    native_result: NativeToolResult | None = None
    error: ErrorInfo | None = None
    events: tuple[RuntimeEvent, ...] = field(default_factory=tuple)

    def is_success(self) -> bool:
        """Return whether the tool request completed without runtime error."""
        return self.error is None and self.native_result is not None


def execute_native_tool_request(
    request: ModelToolRequest,
    registry: NativeToolRegistry,
    access_policy: ToolAccessPolicy | None = None,
) -> ToolExecutionResult:
    """Execute one model-requested native tool through the registry."""
    if not request.is_ready():
        return _failure(
            request=request,
            code="tool_request_invalid",
            message="Tool request requires a tool name and tool call id.",
            event_kind="native_tool_request_invalid",
        )

    tool_name = _request_name(request)
    argument_error = _request_argument_error(request)
    if argument_error:
        return _failure(
            request=request,
            code="tool_arguments_invalid",
            message=(
                f"Tool arguments invalid for {tool_name}: {argument_error}"
            ),
            event_kind="native_tool_arguments_invalid",
            refs=(tool_name,),
        )

    tool = registry.get(tool_name)
    if tool is None:
        return _failure(
            request=request,
            code="native_tool_not_found",
            message=f"Native tool not found: {tool_name}",
            event_kind="native_tool_not_found",
            refs=(tool_name,),
        )

    policy = access_policy or ToolAccessPolicy()
    decision = policy.check_native_tool(tool, request.arguments)
    if not decision.allowed:
        reason = decision.reason or f"Native tool is not allowed: {tool_name}"
        return _failure(
            request=request,
            code="native_tool_denied",
            message=reason,
            event_kind="native_tool_denied",
            refs=(
                tool_name,
                f"access_mode={policy.mode.value}",
                f"requires_approval={decision.requires_approval}",
                *decision.refs,
            ),
        )

    try:
        native_result = tool.call(request.arguments)
    except NATIVE_TOOL_RECOVERABLE_ERRORS as exc:
        return _failure(
            request=request,
            code="native_tool_failed",
            message=f"Native tool failed: {tool_name}: {exc}",
            event_kind="native_tool_failed",
            refs=(tool_name,),
        )

    model_result = ModelToolResult(
        name=tool_name,
        request_id=request.id,
        content=native_result.content,
        data=native_result.data,
        refs=native_result.refs,
        attachments=native_result.attachments,
    )
    return ToolExecutionResult(
        request=request,
        model_result=model_result,
        native_result=native_result,
        events=(
            RuntimeEvent(
                kind="native_tool_completed",
                summary=f"Native tool completed: {tool_name}.",
                refs=(tool_name, *native_result.refs),
            ),
        ),
    )


def _failure(
    request: ModelToolRequest,
    code: str,
    message: str,
    event_kind: str,
    refs: tuple[str, ...] = (),
) -> ToolExecutionResult:
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
    return ToolExecutionResult(
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
