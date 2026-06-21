"""Tool access policy for runtime execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from deepmate.tools import NativeTool

_COMPUTER_ACTION_TOOLS = frozenset(
    (
        "computer_click",
        "computer_type",
        "computer_key",
        "computer_open",
    )
)
_COMPUTER_DANGEROUS_KEYS = frozenset(
    (
        "return",
        "enter",
        "space",
        "delete",
        "backspace",
    )
)
_COMPUTER_PRIVATE_TARGET_PREFIXES = (
    "/users/",
    "/library/",
    "/system/",
    "file://",
    "~",
)
_SENSITIVE_INPUT_MARKERS = (
    "password",
    "passcode",
    "secret",
    "token",
    "api key",
    "apikey",
    "access key",
    "验证码",
    "captcha",
    "otp",
    "credit card",
    "card number",
    "ssn",
)


class ToolAccessMode(StrEnum):
    """Coarse tool access modes for one runtime invocation."""

    READ_ONLY = "read_only"
    WORKSPACE_WRITE = "workspace_write"


@dataclass(frozen=True, slots=True)
class ToolAccessDecision:
    """Decision returned before a tool execution attempt."""

    allowed: bool
    reason: str = ""
    requires_approval: bool = False
    refs: tuple[str, ...] = ()


ToolApprovalCallback = Callable[[NativeTool, ToolAccessDecision], bool]


@dataclass(frozen=True, slots=True)
class ToolAccessPolicy:
    """Small policy gate for native tool execution."""

    mode: ToolAccessMode = ToolAccessMode.READ_ONLY
    shell_enabled: bool = False
    approval_callback: ToolApprovalCallback | None = None
    defer_shell_approval_to_tool: bool = False

    def check_native_tool(
        self,
        tool: NativeTool,
        arguments: Mapping[str, object] | None = None,
    ) -> ToolAccessDecision:
        """Return whether a native tool may execute in this invocation."""
        if tool.read_only:
            return ToolAccessDecision(allowed=True)
        if tool.name.strip() in _COMPUTER_ACTION_TOOLS:
            action_key, risk = _computer_action_scope(tool.name.strip(), arguments or {})
            decision = ToolAccessDecision(
                allowed=False,
                reason=(
                    f"Computer action requires approval: {tool.name.strip()} ({risk}). "
                    "Approve only if this visible GUI action matches the current task."
                ),
                requires_approval=True,
                refs=(
                    f"approval_key={action_key}",
                    f"risk={risk}",
                    *_tool_argument_refs(tool, arguments or {}),
                ),
            )
            return self._maybe_approve(tool, decision)
        refs = _tool_argument_refs(tool, arguments or {})
        if tool.requires_shell:
            if self.defer_shell_approval_to_tool:
                return ToolAccessDecision(allowed=True)
            if self.shell_enabled:
                return ToolAccessDecision(allowed=True)
            decision = ToolAccessDecision(
                allowed=False,
                reason=(
                    f"Native tool requires shell access: {tool.name.strip()}. "
                    "Approve shell access to continue this turn."
                ),
                requires_approval=True,
                refs=refs,
            )
            return self._maybe_approve(tool, decision)
        if self.mode == ToolAccessMode.WORKSPACE_WRITE:
            return ToolAccessDecision(allowed=True)
        decision = ToolAccessDecision(
            allowed=False,
            reason=(
                f"Native tool requires workspace write access: {tool.name.strip()}"
            ),
            requires_approval=True,
            refs=refs,
        )

        return self._maybe_approve(tool, decision)

    def _maybe_approve(
        self,
        tool: NativeTool,
        decision: ToolAccessDecision,
    ) -> ToolAccessDecision:
        if (
            self.approval_callback is not None
            and decision.requires_approval
            and self.approval_callback(tool, decision)
        ):
            return ToolAccessDecision(allowed=True, reason="Approved for this turn.")
        return decision


def _tool_argument_refs(
    tool: NativeTool,
    arguments: Mapping[str, object],
) -> tuple[str, ...]:
    refs: list[str] = [f"tool={tool.name.strip()}"]
    for key in ("path", "source", "target", "name", "cwd", "network"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            refs.append(f"{key}={_preview(value)}")
    command = arguments.get("command")
    if isinstance(command, str) and command.strip():
        refs.append(f"command={_preview(command)}")
    content = arguments.get("content")
    if isinstance(content, str):
        refs.extend(
            (
                f"content_chars={len(content)}",
                f"content_preview={_multiline_preview(content)}",
            )
        )
    old_text = arguments.get("old_text")
    if isinstance(old_text, str):
        refs.extend(
            (
                f"old_text_chars={len(old_text)}",
                f"old_text_preview={_multiline_preview(old_text)}",
            )
        )
    new_text = arguments.get("new_text")
    if isinstance(new_text, str):
        refs.extend(
            (
                f"new_text_chars={len(new_text)}",
                f"new_text_preview={_multiline_preview(new_text)}",
            )
        )
    overwrite = arguments.get("overwrite")
    if isinstance(overwrite, bool):
        refs.append(f"overwrite={str(overwrite).lower()}")
    return tuple(refs)


def _computer_action_scope(tool_name: str, arguments: Mapping[str, object]) -> tuple[str, str]:
    if tool_name == "computer_type":
        text = _argument_text(arguments, "text").lower()
        if any(marker in text for marker in _SENSITIVE_INPUT_MARKERS):
            return "computer:sensitive_input", "sensitive_input"
        return "computer:input", "computer_action"
    if tool_name == "computer_key":
        key = _argument_text(arguments, "key").lower()
        modifiers = _argument_text(arguments, "modifiers").lower()
        modifier_values = arguments.get("modifiers")
        if isinstance(modifier_values, (list, tuple)):
            modifiers = "+".join(str(item).lower() for item in modifier_values)
        if key in _COMPUTER_DANGEROUS_KEYS or "command" in modifiers or "cmd" in modifiers:
            return "computer:external_commit", "external_commit"
        return "computer:keyboard", "computer_action"
    if tool_name == "computer_open":
        target = _argument_text(arguments, "target").lower()
        if target.startswith(_COMPUTER_PRIVATE_TARGET_PREFIXES):
            return "computer:local_private_access", "local_private_access"
        return "computer:open", "computer_action"
    if tool_name == "computer_click":
        return "computer:click", "computer_action"
    return "computer:action", "computer_action"


def _argument_text(arguments: Mapping[str, object], name: str) -> str:
    value = arguments.get(name)
    return value.strip() if isinstance(value, str) else ""


def _preview(value: str, limit: int = 180) -> str:
    clean = " ".join(value.strip().split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 1)] + "…"


def _multiline_preview(value: str, limit: int = 4000) -> str:
    clean = value.strip()
    if not clean:
        return "<empty>"
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 1)].rstrip() + "…"
