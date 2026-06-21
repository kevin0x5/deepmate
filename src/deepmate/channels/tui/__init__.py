"""Textual TUI channel for Deepmate."""

from deepmate.channels.tui.bridge import (
    consume_workspace_switch_request,
    run_headless_tui_turn,
    run_tui_mode,
)
from deepmate.channels.tui.commands import handle_tui_command
from deepmate.channels.tui.formatters import TuiMessage, result_messages
from deepmate.channels.tui.state import TuiRuntimeState, WorkspaceSwitchRequest

__all__ = [
    "TuiMessage",
    "TuiRuntimeState",
    "WorkspaceSwitchRequest",
    "handle_tui_command",
    "result_messages",
    "consume_workspace_switch_request",
    "run_headless_tui_turn",
    "run_tui_mode",
]
