"""Textual app for Deepmate interactive mode."""

from __future__ import annotations

import os
import shlex
import signal
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
import subprocess
import sys
from threading import Event
from time import monotonic
from urllib.parse import unquote, urlparse

from rich.cells import cell_len
from rich.console import Group
from rich.markup import escape
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events
from textual.app import ScreenStackError
from textual.binding import Binding
from textual.css.query import NoMatches
from textual.strip import Strip

from deepmate.channels.tui.bridge import (
    approval_decision_from_text,
    end_tui_session,
    run_headless_tui_turn,
)
from deepmate.channels.session_maintenance import runtime_conversation_from_store
from deepmate.channels.tui.commands import (
    apply_local_model_prepare_result,
    command_suggestions,
    handle_tui_command,
    refresh_computer_tool_surface,
)
from deepmate.channels.tui.files import (
    read_workspace_file_preview,
    workspace_file_items,
    workspace_file_matches,
)
from deepmate.channels.tui.formatters import (
    TuiMessage,
    friendly_error_message,
    tool_exchange_messages,
)
from deepmate.local import (
    LOCAL_PROVIDER_NAME,
    LocalModelInstallResult,
    LocalModelProgress,
    LocalModelStateStore,
    OllamaLocalRuntime,
    ollama_api_url_from_provider_base_url,
)
from deepmate.pet.setup import PetSetupResult, prepare_pet_runtime
from deepmate.channels.tui.render import (
    clean_input_text as _clean_input_text,
    compact_workspace_label as _compact_workspace_label,
    content_tab_display_label as _content_tab_display_label,
    preview_tab_content as _preview_tab_content,
    preview_text as _preview_text,
    readable_markdown as _readable_markdown,
    render_message as _render_message,
    sessions_preview as _sessions_preview,
    should_expand_message as _should_expand_message,
    short_session_title as _short_session_title,
    single_line as _single_line,
    welcome_splash as _welcome_splash,
    workspace_nav_title as _workspace_nav_title,
)
from deepmate.domain import MessageRole
from deepmate.channels.tui.state import (
    LocalModelPrepareRequest,
    TuiPromptQueue,
    TuiRuntimeState,
    WorkspaceSwitchRequest,
)
from deepmate.channels.tui.status import TuiRuntimeStats
from deepmate.cron import maybe_create_cron_draft
from deepmate.qa import maybe_qa_agent_prompt
from deepmate.runtime import (
    ApprovalDecision,
    SafetyDecision,
    SessionApprovalCache,
    ToolAccessDecision,
    ToolAccessMode,
    TurnCancellationToken,
    TurnFollowupBuffer,
    start_runtime_activation,
    start_session_runtime,
)
from deepmate.storage import TurnCheckpointStore, WorkspaceCheckpointStore
from deepmate.tools import NativeTool, NativeToolRegistry

from textual.app import App, ComposeResult, ScreenStackError
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    Input,
    Label,
    ListItem,
    ListView,
    Markdown,
    RichLog,
    Static,
    TextArea,
)


MAX_PROMPT_CHARS = 20_000
WORKSPACE_SWITCH_EXIT_CODE = 75
APPROVAL_WAIT_TIMEOUT_SECONDS = 30 * 60
FOOTER_REFRESH_MIN_INTERVAL_SECONDS = 0.15
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
PASTE_DEDUPE_SECONDS = 0.4
PROMPT_HISTORY_LIMIT = 100
CONTENT_TAB_SEPARATOR = "｜"
CONTENT_TAB_PREFIX = ""
CONTENT_TAB_LEFT_PADDING_CELLS = 1
SESSION_RESTORE_MESSAGE_LIMIT = 80
APPROVAL_PREVIEW_MAX_LINES = 6
PET_START_CHECK_SECONDS = 0.8
PET_START_ERROR_LIMIT = 900


def _command_hint_name(suggestion: str) -> str:
    """Return the literal command prefix from one command suggestion line."""
    label = suggestion.split(" - ", 1)[0].strip()
    if "," in label:
        label = label.split(",", 1)[0].strip()
    parts = label.split()
    if len(parts) <= 1:
        return label
    literal = [parts[0]]
    for part in parts[1:]:
        if (
            part.startswith(("<", "["))
            or part.endswith((">", "]"))
            or "|" in part
        ):
            break
        literal.append(part)
    return " ".join(literal)


_COMMAND_HINTS = tuple(
    (_command_hint_name(item), item.split(" - ", 1)[1] if " - " in item else "")
    for item in command_suggestions()
)


@dataclass(slots=True)
class _PendingApproval:
    """One approval request waiting on the UI thread."""

    title: str
    body: str
    event: Event
    result: str = "deny"
    subtitle: str = ""
    subject: str = ""
    refs: tuple[str, ...] = ()
    session_id: str = ""


@dataclass(slots=True)
class _OpenTab:
    """One content tab in the main workbench."""

    title: str
    content: str
    render_mode: str = "auto"


@dataclass(slots=True)
class _TurnRun:
    """One worker turn bound to the session that started it."""

    session_id: str
    started_at: float
    anchor_id: str = ""
    followup_buffer: TurnFollowupBuffer | None = None
    followup_turn_id: str | None = None
    cancellation_token: TurnCancellationToken | None = None
    answer_visible: bool = False
    delivered_message_keys: set[str] = field(default_factory=set)
    computer_use_enabled_at_start: bool = False
    failed: bool = False
    interrupted: bool = False


@dataclass(slots=True)
class _SessionContentTabs:
    """Workbench tabs owned by one TUI session."""

    open_tabs: dict[str, _OpenTab] = field(default_factory=dict)
    active_tab: str = "main"
    hovered_tab: str = ""
    current_title: str = ""
    current_content: str = ""


class _PromptTextArea(TextArea):
    """Multiline prompt editor with paste behavior tuned for agent prompts."""

    @property
    def value(self) -> str:
        """Compatibility with the old Input-backed tests and helpers."""
        return self.text

    @value.setter
    def value(self, text: str) -> None:
        self.load_text(str(text))
        self.move_cursor(_text_area_end_location(self.text))

    @property
    def cursor_position(self) -> int:
        return _text_area_offset(self.text, self.cursor_location)

    @cursor_position.setter
    def cursor_position(self, offset: int) -> None:
        self.move_cursor(_text_area_location_for_offset(self.text, int(offset)))

    def _on_key(self, event: events.Key) -> None:
        if event.key == "shift+enter":
            return
        if event.key != "enter":
            return
        event.prevent_default()
        event.stop()
        submit = getattr(self.app, "_handle_prompt_editor_enter", None)
        if callable(submit):
            submit()

    def _on_paste(self, event: events.Paste) -> None:
        if event.text:
            try:
                app = self.app
            except Exception:
                app = None
            paste_into_prompt = getattr(app, "_paste_into_prompt", None)
            if callable(paste_into_prompt):
                paste_into_prompt(event.text)
            else:
                _insert_prompt_text(self, event.text, max_chars=MAX_PROMPT_CHARS)
        event.prevent_default()
        event.stop()


class _SelectableRichLog(RichLog):
    """RichLog variant that exposes its rendered lines to Textual selection copy."""

    _selected_text_cache = ""

    def get_selection(self, selection) -> tuple[str, str] | None:
        text = "\n".join(line.text.rstrip() for line in self.lines)
        return selection.extract(text), "\n"

    def selection_updated(self, selection) -> None:
        if selection is not None:
            self._selected_text_cache = _selection_text_from_widget(self, selection)
        else:
            self._selected_text_cache = ""
        self._line_cache.clear()
        self.refresh()

    def render_line(self, y: int) -> Strip:
        scroll_x, scroll_y = self.scroll_offset
        content_y = scroll_y + y
        line = self._render_line(content_y, scroll_x, self.scrollable_content_region.width)
        line = line.apply_style(self.rich_style)
        selection = self.text_selection
        if selection is not None:
            line = _style_strip_selection(
                line,
                selection.get_span(content_y),
                self.screen.get_component_rich_style("screen--selection"),
            )
        return line.apply_offsets(scroll_x, content_y)


def _style_strip_selection(
    strip: Strip,
    span: tuple[int, int] | None,
    selection_style: Style,
) -> Strip:
    if span is None:
        return strip
    start, end = span
    if end == -1:
        end = len(strip.text)
    if end <= start:
        return strip
    segments: list[Segment] = []
    position = 0
    for segment in strip:
        text, style, control = segment
        if control:
            segments.append(segment)
            continue
        segment_start = position
        segment_end = position + len(text)
        if segment_end <= start or segment_start >= end:
            segments.append(segment)
        else:
            left_chars = max(0, start - segment_start)
            right_chars = max(0, segment_end - end)
            selected_chars = len(text) - left_chars - right_chars
            before_text = text[:left_chars]
            selected_text = text[left_chars : left_chars + selected_chars]
            after_text = text[left_chars + selected_chars :]
            if before_text:
                segments.append(Segment(before_text, style, control))
            if selected_text:
                selected_style = (
                    style + selection_style
                    if style
                    else selection_style
                )
                segments.append(Segment(selected_text, selected_style, control))
            if after_text:
                segments.append(Segment(after_text, style, control))
        position = segment_end
    return Strip(segments, strip.cell_length)


def _selection_parts_from_widget(widget: object, selection: object) -> list[str]:
    get_selection = getattr(widget, "get_selection", None)
    if not callable(get_selection):
        return []
    try:
        selected = get_selection(selection)
    except Exception:
        return []
    if selected is None:
        return []
    if isinstance(selected, tuple):
        return [str(piece) for piece in selected]
    return [str(selected)]


def _selection_text_from_widget(widget: object, selection: object) -> str:
    return "".join(_selection_parts_from_widget(widget, selection)).rstrip("\n")


def _selected_text_from_screen_selections(screen: object) -> str:
    selections = getattr(screen, "selections", None)
    if not selections:
        return ""
    pieces: list[str] = []
    for widget, selection in selections.items():
        if not getattr(widget, "is_attached", False):
            continue
        pieces.extend(_selection_parts_from_widget(widget, selection))
    return "".join(pieces).rstrip("\n")


class DeepmateTuiApp(App):
    """A lightweight TUI workbench over the existing Deepmate runtime."""

    CSS = """
    * {
        scrollbar-background: #333333;
        scrollbar-color: #4a4a4a;
        scrollbar-color-hover: #5a5a5a;
        scrollbar-color-active: #6a6a6a;
        scrollbar-corner-color: #333333;
    }
    Screen {
        background: #333333;
        color: #d8d8d8;
        border: none;
        padding: 0;
        margin: 0;
        overflow: hidden;
    }
    Screen > .screen--selection {
        background: #4a5658;
        color: #f0f0f0;
    }
    DeepmateTuiApp {
        background: #333333;
        border: none;
        padding: 0;
        margin: 0;
    }
    #title-bar {
        background: #333333;
        color: #8fb7bd;
        height: 1;
        padding: 0 2;
        margin: 0;
    }
    #session-row, #status-bar {
        background: #333333;
        color: #8a8a8a;
        height: 1;
        margin: 0;
    }
    #session-tabs {
        width: auto;
        background: #333333;
        color: #c8c8c8;
    }
    #session-spacer {
        width: 1fr;
        background: #333333;
    }
    #context-window {
        width: 28;
        padding: 0 1;
        color: #8a8a8a;
        content-align: right middle;
    }
    #body {
        height: 1fr;
        background: #333333;
        border: none;
        padding: 0;
        margin: 0;
    }
    #sidebar {
        width: 36;
        background: #2f2f2f;
        border-right: solid #464646;
    }
    #sidebar-title {
        height: 1;
        color: #a7a7a7;
        padding: 0 2;
        text-style: bold;
    }
    #sidebar-hint {
        height: 1;
        color: #8fb7bd;
        background: #2f2f2f;
        padding: 0 2;
        border-top: solid #464646;
    }
    #content {
        width: 1fr;
        min-width: 0;
        height: 1fr;
    }
    #workbench {
        height: 1fr;
        background: #333333;
    }
    #main-column {
        width: 1fr;
        min-width: 0;
        background: #333333;
    }
    #content-tabs {
        height: 1;
        background: #333333;
        color: #8a8a8a;
        padding: 0 1;
        text-style: bold;
    }
    #chat {
        height: 1fr;
        padding: 0 1;
        background: #333333;
    }
    #content-markdown {
        height: 1fr;
        padding: 0 2;
        background: #333333;
        color: #d8d8d8;
        overflow-y: auto;
        scrollbar-background: #333333;
        scrollbar-color: #4a4a4a;
        scrollbar-color-hover: #5a5a5a;
        scrollbar-color-active: #6a6a6a;
        scrollbar-size-vertical: 1;
        display: none;
    }
    Markdown {
        background: #333333;
        color: #d8d8d8;
        overflow-y: auto;
    }
    MarkdownBlock {
        background: #333333;
        color: #d8d8d8;
    }
    MarkdownH1, MarkdownH2, MarkdownH3 {
        color: #8fb7bd;
    }
    MarkdownCode {
        background: #2b2b2b;
        color: #e0e0e0;
        border: solid #464646;
    }
    MarkdownFence {
        background: #2b2b2b;
        color: #e0e0e0;
        border: solid #464646;
    }
    MarkdownBlockQuote {
        color: #b7b7b7;
        background: #2b2b2b;
        border-left: solid #5a5a5a;
    }
    #command-hints {
        height: auto;
        max-height: 8;
        background: #2b2b2b;
        color: #b7b7b7;
        padding: 0 2;
        display: none;
    }
    #approval-panel {
        height: auto;
        max-height: 11;
        background: #2b2b2b;
        color: #e0e0e0;
        border-top: solid #4a4a4a;
        padding: 0 2;
        display: none;
    }
    #approval-title {
        height: auto;
        color: #8fbe8f;
        text-style: bold;
    }
    #approval-body {
        height: auto;
        max-height: 6;
        overflow-y: hidden;
        color: #d8d8d8;
    }
    #approval-actions {
        height: 1;
        background: #2b2b2b;
    }
    .approval-action {
        width: auto;
        min-width: 12;
        height: 1;
        margin: 0 2 0 0;
        padding: 0 1;
        background: #3a3a3a;
        color: #e0e0e0;
        content-align: center middle;
    }
    .approval-action:hover,
    .approval-action:focus {
        background: #4a5658;
        color: #ffffff;
    }
    #approval-once {
        background: #314137;
        color: #bfe2bf;
    }
    #approval-session {
        background: #453d2b;
        color: #ead48a;
    }
    #approval-deny {
        background: #463232;
        color: #e3a0a0;
    }
    #input-gap {
        height: 1;
        background: #333333;
    }
    #input-row {
        width: 1fr;
        min-width: 0;
        height: auto;
        max-height: 10;
        background: #2c2c2c;
        padding: 0 1;
        border-left: solid #2c2c2c;
        overflow-x: hidden;
        overflow-y: auto;
    }
    #input-row:focus-within {
        border-left: solid #8fb7bd;
    }
    #prompt-glyph {
        width: 2;
        height: 3;
        color: #8fb7bd;
        content-align: center middle;
    }
    #new-session {
        width: 9;
        min-width: 9;
        height: 1;
        padding: 0;
        margin: 0;
        background: #333333;
        color: #8fb7bd;
        content-align: center middle;
    }
    #new-session:hover {
        background: #3f3f3f !important;
        color: #f0f0f0 !important;
    }
    #new-session:focus {
        background: #3f3f3f !important;
        color: #f0f0f0 !important;
    }
    #prompt-input {
        width: 1fr;
        min-width: 0;
        height: auto;
        min-height: 1;
        max-height: 8;
        margin: 1 0;
        background: #2c2c2c;
        color: #c8c8c8;
        border: none;
        padding: 0 1;
        overflow-x: hidden;
        overflow-y: auto;
        scrollbar-size-vertical: 1;
    }
    #prompt-input:focus {
        background: #2c2c2c;
        color: #d8d8d8;
        border: none;
        background-tint: transparent;
    }
    #prompt-input > .text-area--cursor {
        background: #8fb7bd;
        color: #202020;
    }
    #prompt-input > .text-area--selection {
        background: #4a5658;
        color: #f0f0f0;
    }
    #prompt-input > .text-area--placeholder {
        color: #777777;
    }
    RichLog {
        background: #333333;
        color: #d8d8d8;
        border: none;
        scrollbar-background: #333333;
        scrollbar-color: #4a4a4a;
        scrollbar-color-hover: #5a5a5a;
        scrollbar-color-active: #6a6a6a;
        scrollbar-size-vertical: 1;
    }
    RichLog:focus {
        border: none;
        background: #333333;
        background-tint: transparent;
    }
    #file-tree {
        height: 1fr;
        background: #2f2f2f;
        color: #d8d8d8;
        border: none;
        scrollbar-background: #2f2f2f;
        scrollbar-color: #4a4a4a;
        scrollbar-color-hover: #5a5a5a;
        scrollbar-color-active: #6a6a6a;
        scrollbar-size-vertical: 1;
    }
    #file-tree:focus {
        background: #2f2f2f;
        background-tint: transparent;
    }
    #file-tree > ListItem {
        background: #2f2f2f !important;
        color: #d8d8d8 !important;
    }
    #file-tree > ListItem Label {
        background: transparent !important;
        color: #d8d8d8 !important;
    }
    #file-tree > ListItem.-highlight,
    #file-tree > ListItem.-highlight.-hovered,
    #file-tree:focus > ListItem.-highlight,
    #file-tree:focus > ListItem.-highlight.-hovered {
        background: #4a5658 !important;
        color: #f0f0f0 !important;
        text-style: none !important;
    }
    #file-tree > ListItem.-highlight Label,
    #file-tree:focus > ListItem.-highlight Label,
    #file-tree > ListItem.-hovered Label,
    #file-tree:focus > ListItem.-hovered Label,
    #file-tree > ListItem.-highlight.-hovered Label,
    #file-tree:focus > ListItem.-highlight.-hovered Label {
        background: #4a5658 !important;
        color: #f0f0f0 !important;
        text-style: none !important;
    }
    #file-tree > ListItem.-hovered {
        background: #3f3f3f !important;
        color: #f0f0f0 !important;
    }
    .session-tab {
        width: auto;
        min-width: 0;
        height: 1;
        padding: 0 1;
        margin: 0;
        border: none !important;
        background: #333333 !important;
        color: #c8c8c8 !important;
        text-style: bold !important;
        content-align: left middle;
    }
    .session-tab:hover,
    .session-tab:focus {
        background: #3f3f3f !important;
        color: #f0f0f0 !important;
    }
    .session-tab-active {
        background: #333333 !important;
        color: #f0f0f0 !important;
        border: none !important;
    }
    .dim {
        color: #8a8a8a;
    }
    .accent {
        color: #8fb7bd;
    }
    .green {
        color: #8fbe8f;
    }
    .amber {
        color: #d0b66b;
    }
    .red {
        color: #d98a8a;
    }
    .magenta {
        color: #a993bd;
    }
    """

    BINDINGS = [
        ("ctrl+d", "quit", "Quit"),
        Binding("ctrl+c,super+c", "copy_screen_selection", "Copy", show=False, priority=True),
        Binding("escape", "interrupt_or_cancel", "Interrupt", priority=True),
        ("ctrl+v", "paste_clipboard", "Paste"),
        ("ctrl+b", "toggle_sidebar", "Sidebar"),
        ("ctrl+w", "close_active_tab_with_feedback", "Close tab"),
        ("pageup", "scroll_active_page_up", "Scroll up"),
        ("pagedown", "scroll_active_page_down", "Scroll down"),
        ("home", "scroll_active_home", "Top"),
        ("end", "scroll_active_end", "Bottom"),
        ("f1", "help", "Help"),
    ]

    exit_code = 0
    # Default the file-tree sidebar to open so the workspace is visible on first
    # screen; the sidebar footer shows the Ctrl+B shortcut to collapse it.
    # compose() builds the widget unconditionally, so on_mount() syncs the
    # initial display to this flag.
    sidebar_visible = reactive(True)

    def __init__(
        self,
        state: TuiRuntimeState,
        *,
        initial_prompts: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self.state = state
        self.initial_prompts = list(initial_prompts)
        self._exiting = False
        self._running_turn = False
        self._local_prepare_running = False
        self._pet_setup_running = False
        self._local_ready_checked = False
        self._pending_approval: _PendingApproval | None = None
        self._approval_queue: list[_PendingApproval] = []
        self._approval_caches_by_session: dict[str, SessionApprovalCache] = {}
        self._tool_turn_approvals: set[str] = set()
        self._tool_turn_approvals_by_session: dict[str, set[str]] = {}
        self._tool_session_approvals: dict[str, set[str]] = {}
        self._approval_result_lines_by_session: dict[str, list[str]] = {}
        self._last_approval_result_by_session: dict[str, str] = {}
        self._file_items: tuple[str, ...] = ()
        self._file_item_is_dir: tuple[bool, ...] = ()
        self._expanded_dirs: set[str] = set()
        self._session_tabs: list[tuple[str, str]] = [
            (state.session.session_id, state.session.title)
        ]
        self._open_tabs: dict[str, _OpenTab] = {}
        self._active_tab = "main"
        self._hovered_content_tab = ""
        self._current_tab_title = ""
        self._current_tab_content = ""
        self._session_content_tabs: dict[str, _SessionContentTabs] = {}
        queue_path = state.prompt_queue_path()
        self._prompt_queue = (
            TuiPromptQueue.load(queue_path)
            if queue_path is not None
            else TuiPromptQueue()
        )
        self._last_queue_pause_reason = ""
        self._active_turn: _TurnRun | None = None
        self._session_turns: dict[str, _TurnRun] = {}
        self._session_running: set[str] = set()
        self._session_has_updates: set[str] = set()
        self._session_live_status: dict[str, str] = {}
        # Token streaming: workers append (content, reasoning) fragments to the
        # per-session pending list (GIL-atomic, no per-token call_from_thread);
        # a set_interval timer on the UI thread drains and renders, throttling
        # high-frequency tokens into ~60ms batches. Accumulated text per session
        # backs the live cell and is cleared when the turn ends.
        self._stream_pending: dict[str, list[tuple[str, str]]] = {}
        self._stream_content: dict[str, str] = {}
        self._stream_reasoning: dict[str, str] = {}
        self._show_reasoning_stream = False
        self._turn_failed = False
        self._interrupted = False
        self._turn_started_at: float | None = None
        self._last_footer_refresh_at = 0.0
        self._spinner_frame = 0
        self._last_file_tree_signature: tuple[tuple[str, str], ...] = ()
        self._compose_mode = False
        self._compose_lines: list[str] = []
        # Per-session chat buffers so a backgrounded session keeps its full
        # message history (incl. errors/warnings that never reach the transcript)
        # and switching back restores from memory rather than rebuilding from disk.
        self._session_messages: dict[str, list[TuiMessage]] = {}
        self._session_prompt_history: dict[str, list[str]] = {}
        self._session_prompt_history_index: dict[str, int | None] = {}
        self._session_prompt_history_draft: dict[str, str] = {}
        self._session_unsent_drafts: dict[str, str] = {}
        self._session_clear_snapshots: dict[
            str,
            list[TuiMessage],
        ] = {}
        self._last_submitted_prompt_by_session: dict[str, str] = {}
        # Per-session runtime stats so two concurrent turns don't clobber each
        # other's context-window numbers in the footer (each worker records into
        # its own session's stats; the footer reads the current session's).
        self._session_stats: dict[str, TuiRuntimeStats] = {
            state.session.session_id: state.runtime_stats
        }
        self._command_hint_matches: tuple[tuple[str, str], ...] = ()
        self._command_hint_index = 0
        self._pet_process: subprocess.Popen | None = None
        self._live_status_text = ""
        self._live_message_active = False
        self._chat_follow_tail = True
        self._last_prompt_paste_key = ""
        self._last_prompt_paste_at = 0.0
        self._spinner_timer = None
        self._stream_timer = None
        # Per-session /trust state. A trust decision must not leak between open
        # sessions because each session may be working on a different task.
        self._trusted_sessions: set[str] = set()
        # Remember the launch-time access mode so /trust off can restore it.
        self._base_access_mode = (
            state.tool_access_policy.mode
            if state.tool_access_policy is not None
            else None
        )
        self._save_content_tabs_for_session(state.session.session_id)
        if state.approval_cache is not None:
            self._approval_caches_by_session[state.session.session_id] = state.approval_cache

    def _current_session_id(self) -> str:
        return self.state.session.session_id

    @property
    def _trusted(self) -> bool:
        return self.state.session.session_id in self._trusted_sessions

    @property
    def _main_messages(self) -> list[TuiMessage]:
        return self._session_messages.setdefault(self._current_session_id(), [])

    @_main_messages.setter
    def _main_messages(self, value: list[TuiMessage]) -> None:
        self._session_messages[self._current_session_id()] = list(value)

    def _messages_for_session(self, session_id: str) -> list[TuiMessage]:
        return self._session_messages.setdefault(session_id, [])

    def _forget_session_buffers(self, session_id: str) -> None:
        self._session_messages.pop(session_id, None)

    def compose(self) -> ComposeResult:
        yield Static("deepmate", id="title-bar")
        with Horizontal(id="session-row"):
            with Horizontal(id="session-tabs"):
                for session_id, title in self._visible_session_tabs():
                    yield Static(
                        self._session_tab_text(session_id, title),
                        id=_session_button_id(session_id),
                        classes=self._session_tab_classes(session_id),
                    )
            yield Static("｜ New +", id="new-session")
            yield Static("", id="session-spacer")
            yield Static(self._context_window_label(), id="context-window")
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Static(_workspace_nav_title(self.state.workspace), id="sidebar-title")
                yield ListView(id="file-tree")
                yield Static("Ctrl+B 关闭侧栏", id="sidebar-hint")
            with Vertical(id="content"):
                with Horizontal(id="workbench"):
                    with Vertical(id="main-column"):
                        yield Static(self._content_tabs_label(), id="content-tabs")
                        yield _SelectableRichLog(id="chat", wrap=True, highlight=False, markup=True)
                        yield _SelectableRichLog(id="content-markdown", wrap=True, highlight=False, markup=False)
                yield Static("", id="command-hints")
                with Vertical(id="approval-panel"):
                    yield Static("", id="approval-title")
                    yield Static("", id="approval-body")
                    with Horizontal(id="approval-actions"):
                        yield Static("Allow once", id="approval-once", classes="approval-action")
                        yield Static("Always allow", id="approval-session", classes="approval-action")
                        yield Static("Deny", id="approval-deny", classes="approval-action")
                yield Static("", id="input-gap")
                with Horizontal(id="input-row"):
                    yield Static(">", id="prompt-glyph")
                    yield _PromptTextArea(
                        "",
                        placeholder="Type a task or /command...",
                        id="prompt-input",
                    )
        yield Static(self._status_label(), id="status-bar")

    def on_mount(self) -> None:
        self.state.tool_approval_callback = self._tool_approval
        self.state.safety_approval_callback = self._safety_approval
        self.state.status_message_callback = self._write_from_worker
        self.state.live_status_callback = self._update_live_status_from_worker
        self.state.final_message_callback = self._append_final_messages_from_worker
        self.state.token_stream_callback = self._stream_tokens_from_worker
        self._refresh_file_tree()
        self.query_one("#sidebar").display = self.sidebar_visible
        self._refresh_content_tabs()
        self._refresh_footer()
        self._write_start_message()
        self.query_one("#prompt-input", TextArea).focus()
        self._spinner_timer = self.set_interval(0.1, self._tick_spinner)
        self._stream_timer = self.set_interval(0.06, self._flush_token_stream)
        should_prepare_local = self.state.provider_name == LOCAL_PROVIDER_NAME
        self._maybe_use_prepared_local_model()
        if should_prepare_local:
            self._start_local_prepare(
                LocalModelPrepareRequest(
                    preset=self.state.current_local_preset(),
                    source="/local",
                )
            )
        for prompt in self.initial_prompts:
            if self._local_prepare_running:
                self._enqueue_prompt(prompt, prefix="本地模型准备好后会继续执行。")
            else:
                self._submit_prompt(prompt)

    def _maybe_use_prepared_local_model(self) -> None:
        """Use an already prepared local model when no cloud model is available."""
        if self._local_ready_checked:
            return
        self._local_ready_checked = True
        if self.state.provider_api_key_available:
            return
        if self.state.provider_name == LOCAL_PROVIDER_NAME:
            return
        self.run_worker(self._detect_prepared_local_model_worker, thread=True)

    def _detect_prepared_local_model_worker(self) -> None:
        try:
            preset = OllamaLocalRuntime(
                api_url=ollama_api_url_from_provider_base_url(
                    self.state.local_provider_base_url
                )
            ).prepared_model()
        except Exception:
            preset = None
        if preset is not None:
            self._safe_call_from_thread(self._use_prepared_local_model, preset)

    def _use_prepared_local_model(self, preset) -> None:
        if preset is None:
            return
        result = apply_local_model_prepare_result(
            self.state,
            LocalModelPrepareRequest(preset=preset, source="/local"),
            LocalModelInstallResult(
                ok=True,
                preset=preset,
                message=f"{preset.label} 已就绪。",
                provider_base_url=self.state.local_provider_base_url,
            ),
        )
        if result.messages:
            self._append_messages(result.messages)
        self._prompt_queue.resume()
        self._maybe_start_next_queued_prompt()

    def _tick_spinner(self) -> None:
        """Advance the working-indicator animation while a turn is running."""
        if not self._current_session_is_running():
            return
        self._spinner_frame = (self._spinner_frame + 1) % len(SPINNER_FRAMES)
        try:
            self.query_one("#status-bar", Static).update(self._status_label())
        except (NoMatches, ScreenStackError):
            return

    def _submit_prompt_editor(self) -> None:
        try:
            prompt_widget = self.query_one("#prompt-input", TextArea)
        except (NoMatches, ScreenStackError):
            return
        raw_value = _clean_input_text(_prompt_text(prompt_widget))
        value = raw_value.strip()
        if raw_value.strip():
            self._remember_unsent_draft(raw_value)
        _set_prompt_text(prompt_widget, "")
        self._handle_prompt_submission(raw_value, value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Compatibility path for tests and older Input-backed builds."""
        raw_value = _clean_input_text(getattr(event, "value", ""))
        value = raw_value.strip()
        input_widget = getattr(event, "input", None)
        if input_widget is not None:
            try:
                input_widget.value = ""
            except AttributeError:
                pass
        if raw_value.strip():
            self._remember_unsent_draft(raw_value)
        self._handle_prompt_submission(raw_value, value)

    def _handle_prompt_submission(self, raw_value: str, value: str) -> None:
        self._update_command_hints("")
        if value and not value.startswith("/"):
            self._remember_unsent_draft(raw_value)
        if value in {"/exit", "/quit"}:
            self.action_quit()
            return
        if value in {"/new", "/new-session"}:
            self._create_new_session()
            return
        if self._handle_compose_input(raw_value):
            return
        if self._handle_approval_input(value):
            return
        if self._handle_trust_command(value):
            return
        if value == "/clear":
            self._clear_chat_display()
            return
        if value == "/undo-clear":
            self._undo_clear_chat_display()
            return
        if value == "/restore-draft":
            self._restore_unsent_draft()
            return
        if value == "/queue":
            self._write(TuiMessage(kind="status", title="queue", body=self._queue_status_body()))
            return
        if value == "/approvals":
            self._show_approval_history()
            return
        if value.startswith("/queue "):
            prompt = value[len("/queue ") :].strip()
            self._enqueue_prompt(prompt)
            return
        if value == "/clear-queue":
            count = self._prompt_queue.clear()
            self._last_queue_pause_reason = ""
            self._refresh_footer()
            self._write(TuiMessage(kind="status", title="queue", body=f"Cleared {count} queued prompt(s)."))
            return
        if value == "/resume-queue":
            self._prompt_queue.resume()
            self._last_queue_pause_reason = ""
            self._refresh_footer()
            self._write(TuiMessage(kind="status", title="queue", body=self._queue_status_body()))
            self._maybe_start_next_queued_prompt()
            return
        if self._handle_session_browser_command(value):
            return
        if self._handle_file_reference_command(value):
            return
        if self._handle_workspace_input(raw_value):
            return
        if value == "/followup":
            body = (
                "Usage: /followup <text>"
                if self._current_session_is_running()
                else "No running turn. Usage while running: /followup <text>"
            )
            self._write(
                TuiMessage(
                    kind="status",
                    title="follow-up",
                    body=body,
                )
            )
            return
        idle_followup = _strip_command_arg(value, ("/followup ",))
        if idle_followup is not None and not self._current_session_is_running():
            self._submit_prompt(idle_followup)
            return
        if self._handle_preview_command(value):
            return
        if self._handle_find_command(value):
            return
        if self._handle_rewind_command(value):
            return
        if self._handle_immediate_command(value):
            return
        if self._handle_cron_natural_language(value):
            return
        self._submit_prompt(value)

    def _handle_prompt_editor_enter(self) -> None:
        if self._command_hint_matches:
            current = _prompt_text(self.query_one("#prompt-input", TextArea)).strip()
            selected = self._selected_command_hint()
            if _command_accepts_immediate_enter(current):
                self._submit_prompt_editor()
                return
            if (
                selected
                and current not in {selected, selected + " "}
            ):
                self._complete_selected_command()
                current = _prompt_text(self.query_one("#prompt-input", TextArea)).strip()
                if _command_accepts_immediate_enter(current):
                    self._submit_prompt_editor()
                    return
                return
        self._submit_prompt_editor()

    def _toggle_file_dir(self, relative_path: str) -> None:
        clean = relative_path.strip("/")
        if not clean:
            return
        if clean in self._expanded_dirs:
            self._expanded_dirs.remove(clean)
        else:
            self._expanded_dirs.add(clean)
        self._last_file_tree_signature = ()
        self._refresh_file_tree()

    def on_static_clicked(self, event: Static.Clicked) -> None:
        widget_id = event.static.id
        approval_action = _approval_action_from_widget_id(widget_id or "")
        if approval_action is not None:
            event.stop()
            self._resolve_pending_approval(approval_action)
            return
        session_id = _session_id_from_button_id(widget_id or "")
        if session_id:
            self._switch_session(session_id)
            return
        if widget_id == "content-tabs":
            event.stop()
            self._handle_content_tab_click(getattr(event, "x", 0))

    def on_click(self, event: events.Click) -> None:
        widget_id = getattr(event.widget, "id", "") or ""
        if widget_id == "new-session":
            self._create_new_session()
            return
        approval_action = _approval_action_from_widget_id(widget_id)
        if approval_action is not None:
            event.stop()
            self._resolve_pending_approval(approval_action)
            return
        session_id = _session_id_from_button_id(widget_id)
        if session_id:
            self._switch_session(session_id)
            return
        if widget_id == "content-tabs":
            event.stop()
            self._handle_content_tab_click(event.x)

    def on_mouse_move(self, event: events.MouseMove) -> None:
        widget_id = getattr(event.widget, "id", "") or ""
        if widget_id != "content-tabs":
            return
        event.stop()
        self._set_hovered_content_tab(self._content_tab_at_x(event.x))

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        if self._route_active_content_scroll_event(event, "scroll_down"):
            return
        if self._active_tab == "main":
            try:
                self._chat_follow_tail = _is_scroll_at_end(self.query_one("#chat", RichLog))
            except Exception:
                self._chat_follow_tail = True

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        if self._route_active_content_scroll_event(event, "scroll_up"):
            return
        if self._active_tab == "main":
            self._chat_follow_tail = False

    def on_leave(self, event: events.Leave) -> None:
        control = getattr(event, "control", None)
        widget_id = getattr(control, "id", "") or ""
        if widget_id == "content-tabs":
            self._set_hovered_content_tab("")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "prompt-input":
            return
        text = _prompt_text(event.text_area)
        self._remember_unsent_draft(text)
        self._update_command_hints(text)

    def on_paste(self, event: events.Paste) -> None:
        if not event.text:
            return
        focused = self._safe_focused()
        if isinstance(focused, TextArea) and focused.id == "prompt-input":
            return
        event.prevent_default()
        event.stop()
        self._paste_into_prompt(event.text)

    def on_key(self, event: events.Key) -> None:
        if event.key in {"ctrl+c", "super+c"}:
            self._copy_selected_text_if_available()
            event.prevent_default()
            event.stop()
            return
        focused = self._safe_focused()
        focused_prompt = isinstance(focused, TextArea) and focused.id == "prompt-input"
        if self._command_hint_matches and event.key == "escape":
            event.prevent_default()
            event.stop()
            self._hide_command_hints()
            return
        if focused_prompt and event.key == "shift+enter":
            return
        if focused_prompt and event.key == "enter":
            event.prevent_default()
            event.stop()
            self._handle_prompt_editor_enter()
            return
        if event.key in {"up", "down"}:
            if self._command_hint_matches:
                event.prevent_default()
                event.stop()
                delta = -1 if event.key == "up" else 1
                self._command_hint_index = (
                    self._command_hint_index + delta
                ) % len(self._command_hint_matches)
                self._render_command_hints()
                return
            if self._handle_prompt_history_key(event.key):
                event.prevent_default()
                event.stop()
                return
        if self._command_hint_matches and event.key in {"ctrl+up", "ctrl+down"}:
            event.prevent_default()
            event.stop()
            delta = -1 if event.key == "ctrl+up" else 1
            self._command_hint_index = (
                self._command_hint_index + delta
            ) % len(self._command_hint_matches)
            self._render_command_hints()
            return
        if self._command_hint_matches and event.key == "tab":
            event.prevent_default()
            event.stop()
            self._complete_selected_command()

    def _safe_focused(self):
        try:
            return self.focused
        except ScreenStackError:
            return None

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.list_view.id != "file-tree":
            return
        index = getattr(event.list_view, "index", -1)
        if index is None or index < 0 or index >= len(self._file_items):
            return
        relative = self._file_items[index]
        if self._file_item_is_dir[index]:
            self._toggle_file_dir(relative)
            return
        if relative.startswith("... more files;"):
            self._handle_file_reference_command("/files")
            return
        try:
            preview = read_workspace_file_preview(self.state.workspace, relative)
        except (OSError, ValueError) as exc:
            self._write(TuiMessage(kind="error", title=relative, body=str(exc)))
            return
        self._open_content_tab(relative, preview.rendered_content())
        self._write(
            TuiMessage(
                kind="file",
                title="file",
                body=(
                    f"Opened {relative} bytes {preview.start}-{preview.end} "
                    f"of {preview.bytes_total}."
                ),
            )
        )

    def action_quit(self) -> None:
        self._exiting = True
        self._stop_timers()
        self._deny_pending_approval()
        self._stop_pet_process()
        end_tui_session(self.state, "command")
        self.exit_code = 0
        self.exit()

    def _stop_timers(self) -> None:
        for timer in (self._spinner_timer, self._stream_timer):
            stop = getattr(timer, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    pass
        self._spinner_timer = None
        self._stream_timer = None

    def _safe_call_from_thread(self, callback, *args, **kwargs) -> bool:
        if self._exiting:
            return False
        try:
            self.call_from_thread(callback, *args, **kwargs)
        except RuntimeError:
            return False
        return True

    def _request_workspace_switch(self, workspace: Path, session_id: str = "") -> None:
        if self._running_turn:
            self._write(
                TuiMessage(
                    kind="warning",
                    title="workspace",
                    body=(
                        "A turn is running. Wait for it to finish or interrupt it before "
                        "switching workspace."
                    ),
                )
            )
            return
        if self._pending_approval is not None or self._approval_queue:
            self._write(
                TuiMessage(
                    kind="warning",
                    title="workspace",
                    body="Resolve the pending approval before switching workspace.",
                )
            )
            return
        current = self.state.workspace.resolve()
        target = workspace.resolve()
        if target == current:
            if session_id.strip():
                self._switch_session(session_id)
                return
            self._write(
                TuiMessage(
                    kind="status",
                    title="workspace",
                    body=f"Already using workspace: {target}",
                )
            )
            return
        target_session_id = session_id.strip()
        if not target_session_id:
            target_session = self.state.session_store.latest_for_workspace(target)
            target_session_id = target_session.session_id if target_session else ""
        self._write(
            TuiMessage(
                kind="status",
                title="workspace",
                body=(
                    f"Opening workspace: {target}\n"
                    + (
                        f"Resuming session: {target_session_id}\n"
                        if target_session_id
                        else "Starting a new session for that workspace.\n"
                    )
                    +
                    "Deepmate will restart the TUI so file tools, skills, browser state, "
                    "and checkpoints all point at the new workspace."
                ),
            )
        )
        self._stop_pet_process()
        end_tui_session(self.state, "workspace-switch")
        self.state.workspace_switch_request = WorkspaceSwitchRequest(
            workspace=target,
            session_id=target_session_id,
        )
        self.exit_code = WORKSPACE_SWITCH_EXIT_CODE
        self.exit()

    def action_interrupt_or_cancel(self) -> None:
        if self._pending_approval_is_current_session():
            self._resolve_pending_approval("deny")
            return
        if self._active_tab != "main":
            self._close_active_tab_with_feedback()
            return
        if self._active_turn_is_current_session():
            self._interrupt_current_turn()
            return
        if self._session_running:
            self._interrupt_background_turns()
            return
        self._hide_command_hints()

    def action_interrupt_or_quit(self) -> None:
        self.action_interrupt_or_cancel()

    def _interrupt_current_turn(self) -> None:
        self._deny_pending_approval()
        self._interrupted = True
        active_turn = self._turn_for_current_session()
        if active_turn is not None:
            active_turn.interrupted = True
            if active_turn.cancellation_token is not None:
                active_turn.cancellation_token.cancel()
        self._pause_prompt_queue("turn interrupted")
        self._refresh_footer()
        self._write(
            TuiMessage(
                kind="warning",
                title="interrupt",
                body="Turn will stop at the next runtime boundary.",
            )
        )

    def _interrupt_background_turns(self) -> None:
        interrupted = 0
        for session_id in tuple(self._session_running):
            turn = self._session_turns.get(session_id)
            if turn is not None:
                turn.interrupted = True
                if turn.cancellation_token is not None:
                    turn.cancellation_token.cancel()
                interrupted += 1
        if not interrupted:
            return
        self._pause_prompt_queue("turn interrupted")
        self._refresh_footer()
        self._write(
            TuiMessage(
                kind="warning",
                title="interrupt",
                body=(
                    f"Interrupting {interrupted} background session(s); "
                    "each will stop at its next runtime boundary."
                ),
            )
        )

    def action_toggle_sidebar(self) -> None:
        self.sidebar_visible = not self.sidebar_visible
        sidebar = self.query_one("#sidebar")
        sidebar.display = self.sidebar_visible
        self._refresh_content_tabs()
        self._refresh_footer()

    def action_help(self) -> None:
        self._handle_immediate_command("/commands")

    def action_show_approvals(self) -> None:
        self._show_approval_history()

    def action_paste_clipboard(self) -> None:
        try:
            text = _read_clipboard()
        except OSError as exc:
            self._write(
                TuiMessage(
                    kind="error",
                    title="paste",
                    body=f"Could not read clipboard: {exc}",
                )
            )
            return
        self._paste_into_prompt(text)

    def action_copy_screen_selection(self) -> None:
        self._copy_selected_text_if_available()

    def _copy_selected_text_if_available(self) -> bool:
        selected = self._selected_text_for_copy()
        if selected:
            self.copy_to_clipboard(selected)
            return True
        focused = self.focused
        if isinstance(focused, TextArea) and focused.selected_text:
            self.copy_to_clipboard(focused.selected_text)
            return True
        return False

    def _selected_text_for_copy(self) -> str:
        try:
            selected = self.screen.get_selected_text()
        except Exception:
            selected = None
        if isinstance(selected, str) and selected:
            return selected
        selected = _selected_text_from_screen_selections(self.screen)
        if selected:
            return selected
        for widget_id in ("chat", "content-markdown"):
            try:
                widget = self.query_one(f"#{widget_id}", _SelectableRichLog)
            except Exception:
                continue
            if widget.display and widget._selected_text_cache:
                return widget._selected_text_cache
        return ""

    def copy_to_clipboard(self, text: str) -> None:
        self._clipboard = text
        try:
            _copy_to_clipboard(text)
        except OSError:
            super().copy_to_clipboard(text)

    def action_scroll_active_page_up(self) -> None:
        self._scroll_active_content("page_up")

    def action_scroll_active_page_down(self) -> None:
        self._scroll_active_content("page_down")

    def action_scroll_active_home(self) -> None:
        self._scroll_active_content("home")

    def action_scroll_active_end(self) -> None:
        self._scroll_active_content("end")

    def _submit_prompt(
        self,
        prompt: str,
        *,
        display_prompt: str | None = None,
        remember_prompt: bool = True,
    ) -> None:
        prompt = _clean_input_text(prompt).strip()
        if not prompt:
            return
        if not _prompt_length_ok(prompt):
            self._write(_prompt_too_long_message(len(prompt)))
            return
        if self._needs_model_choice_before_prompt(prompt):
            self._queue_prompt_until_model_ready(prompt)
            return
        if self._current_session_is_running():
            if self._pending_approval_is_current_session():
                approval_result = _approval_input_result(prompt)
                if approval_result is not None:
                    self._resolve_pending_approval(approval_result)
                    return
                if prompt == "/followup":
                    self._write(
                        TuiMessage(
                            kind="warning",
                            title="follow-up",
                            body="Approval is waiting. Add text after /followup or use /queue.",
                        )
                    )
                    return
                forced_followup = _strip_command_arg(prompt, ("/followup ",))
                if forced_followup is not None:
                    self._submit_followup(forced_followup, forced=True)
                    return
                queued_prompt = _strip_command_arg(prompt, ("/queue ",))
                if queued_prompt is not None:
                    self._enqueue_prompt(
                        queued_prompt,
                        kind="warning",
                        title="approval pending",
                        prefix="Approval is waiting. Queued for after the current turn.",
                    )
                    return
                self._write(
                    TuiMessage(
                        kind="warning",
                        title="approval pending",
                        body=(
                            "Approval is waiting. Choose allow or deny first. "
                            "Use /followup <text> to add context to this turn, "
                            "or /queue <text> to run it after the current turn."
                        ),
                    )
                )
                return
            if self._route_running_prompt(prompt):
                return
            self._enqueue_prompt(prompt, prefix="It will run after the current turn.")
            return
        if self._pending_approval is not None and self._is_session_management_command(prompt):
            if self._handle_session_management_command(prompt):
                return
        visible_prompt = (display_prompt or prompt).strip()
        qa_prompt = maybe_qa_agent_prompt(prompt, workspace=self.state.workspace)
        if qa_prompt is not None:
            prompt = qa_prompt
        if remember_prompt:
            self._remember_prompt_history(visible_prompt or prompt)
        self._last_submitted_prompt_by_session[
            self.state.session.session_id
        ] = visible_prompt or prompt
        self._running_turn = True
        self._turn_failed = False
        self._interrupted = False
        self._turn_started_at = monotonic()
        turn_session_id = self.state.session.session_id
        turn_started_at = self._turn_started_at
        turn_anchor_id = _turn_anchor_id(turn_session_id, turn_started_at)
        if self.state.followup_buffer is None:
            self.state.followup_buffer = TurnFollowupBuffer()
        self.state.active_followup_turn_id = self.state.followup_buffer.start_turn()
        self.state.cancellation_token = TurnCancellationToken()
        self.state.unconsumed_followups = ()
        turn_run = _TurnRun(
            session_id=turn_session_id,
            started_at=turn_started_at,
            anchor_id=turn_anchor_id,
            followup_buffer=self.state.followup_buffer,
            followup_turn_id=self.state.active_followup_turn_id,
            cancellation_token=self.state.cancellation_token,
            computer_use_enabled_at_start=(
                self.state.behavior_runtime.computer_use_enabled
                if self.state.behavior_runtime is not None
                else False
            ),
        )
        self._active_turn = turn_run
        self._session_turns[turn_session_id] = turn_run
        self._session_running.add(turn_session_id)
        # Pre-create the stream buffer on the UI thread so the worker only ever
        # appends to an existing list (never mutates the dict), keeping the
        # flush timer's iteration race-free without a lock.
        self._stream_pending[turn_session_id] = []
        self._stream_content[turn_session_id] = ""
        self._stream_reasoning[turn_session_id] = ""
        self._session_has_updates.discard(turn_session_id)
        self._tool_turn_approvals = set()
        self._tool_turn_approvals_by_session[turn_session_id] = self._tool_turn_approvals
        if self._active_tab != "main":
            self._activate_content_tab("main")
        self._write(
            TuiMessage(
                kind="user",
                title="you",
                body=visible_prompt or prompt,
                refs=(f"turn_anchor={turn_anchor_id}",),
            )
        )
        self._start_live_work("Preparing context...")
        self._route_checkpoint_writes_to_current_session()
        turn_state = self._state_for_worker_turn(turn_session_id)
        self.run_worker(
            lambda: self._run_prompt_worker(
                turn_state,
                prompt,
                session_id=turn_session_id,
                started_at=turn_started_at,
            ),
            thread=True,
        )

    def _needs_model_choice_before_prompt(self, prompt: str) -> bool:
        if prompt.startswith("/"):
            return False
        if self.state.provider_name == LOCAL_PROVIDER_NAME:
            return False
        return not self.state.provider_api_key_available

    def _queue_prompt_until_model_ready(self, prompt: str) -> None:
        queued = self._enqueue_prompt(
            prompt,
            kind="warning",
            title="model",
            prefix="还没有可用模型，我会先保留这次请求。",
        )
        if not queued:
            return
        if self.state.missing_model_prompt_shown:
            return
        self.state.missing_model_prompt_shown = True
        self._write(
            TuiMessage(
                kind="status",
                title="model",
                body=(
                    "还没有可用的云端模型。可以输入 /setup-key 配置云端模型，"
                    "或输入 /local 准备本地模型。"
                ),
            )
        )

    def _run_prompt_worker(
        self,
        state: TuiRuntimeState,
        prompt: str,
        *,
        session_id: str,
        started_at: float | None,
    ) -> None:
        try:
            self._set_worker_checkpoint_controller(state)
            updated_state, messages, exit_requested = run_headless_tui_turn(state, prompt)
            self._safe_call_from_thread(
                self._finish_prompt_worker,
                updated_state,
                messages,
                exit_requested,
                session_id,
                started_at,
            )
        except Exception as exc:
            self._safe_call_from_thread(self._handle_worker_exception, exc, session_id)
        finally:
            self._clear_worker_checkpoint_controller(state)

    def _start_local_prepare(self, request: LocalModelPrepareRequest) -> None:
        if self._local_prepare_running:
            self._write(
                TuiMessage(
                    kind="status",
                    title=request.source,
                    body="本地模型正在准备中，完成后会自动切换。",
                )
            )
            return
        if self._current_session_is_running() and not request.defer_switch:
            request = replace(request, defer_switch=True)
        self._local_prepare_running = True
        self._running_turn = True
        self._start_live_work(
            f"正在准备本地模型 · {request.preset.short_label}"
        )
        self._refresh_footer()
        self.run_worker(
            lambda: self._run_local_prepare_worker(request),
            thread=True,
        )

    def _run_local_prepare_worker(self, request: LocalModelPrepareRequest) -> None:
        try:
            result = OllamaLocalRuntime(
                api_url=ollama_api_url_from_provider_base_url(
                    self.state.local_provider_base_url
                )
            ).prepare_model(
                request.preset,
                progress=lambda progress: self._safe_call_from_thread(
                    self._update_local_prepare_progress,
                    progress,
                ),
                state_store=LocalModelStateStore(self.state.data_dir),
            )
            self._safe_call_from_thread(self._finish_local_prepare_worker, request, result)
        except Exception as exc:
            try:
                LocalModelStateStore(self.state.data_dir).record(
                    model_id=request.preset.id,
                    stage="failed",
                    message="本地模型暂时没有准备成功，稍后可以再次输入 /local。",
                    status="failed",
                    failure_kind="prepare_failed",
                )
            except Exception:
                pass
            self._safe_call_from_thread(self._handle_local_prepare_exception, exc)

    def _update_local_prepare_progress(self, progress: LocalModelProgress) -> None:
        self._start_live_work(progress.message)
        self._refresh_footer_throttled()

    def _finish_local_prepare_worker(self, request, install_result) -> None:
        if install_result.ok and request.defer_switch and not self._session_running:
            request = replace(request, defer_switch=False)
        result = apply_local_model_prepare_result(
            self.state,
            request,
            install_result,
        )
        self._local_prepare_running = False
        self._running_turn = bool(self._session_running)
        self._clear_live_work()
        if result.messages:
            self._append_messages(result.messages)
        if install_result.ok:
            self._prompt_queue.resume()
            self._apply_pending_local_switch()
        self._refresh_footer()
        self._maybe_start_next_queued_prompt()

    def _handle_local_prepare_exception(self, exc: BaseException) -> None:
        self._local_prepare_running = False
        self._running_turn = bool(self._session_running)
        self._clear_live_work()
        self._prompt_queue.resume()
        self._write(
            TuiMessage(
                kind="warning",
                title="/local",
                body=(
                    "本地模型暂时没有准备成功，已继续使用当前模型。"
                    "已保留排队请求；配置云端 key 或稍后再次输入 /local 后可以继续。"
                ),
            )
        )
        self._refresh_footer()

    def _set_worker_checkpoint_controller(self, state: TuiRuntimeState) -> None:
        if state.checkpoint_write_router is not None:
            state.checkpoint_write_router.set_thread_controller(
                state.checkpoint_controller
            )

    def _clear_worker_checkpoint_controller(self, state: TuiRuntimeState) -> None:
        if state.checkpoint_write_router is not None:
            state.checkpoint_write_router.clear_thread_controller()

    def _finish_prompt_worker(
        self,
        state: TuiRuntimeState,
        messages: Iterable[TuiMessage],
        exit_requested: bool,
        session_id: str,
        started_at: float | None,
    ) -> None:
        if self._exiting:
            return
        materialized = tuple(messages)
        current = session_id == self.state.session.session_id
        if current:
            self.state = state
            self.state.status_message_callback = self._write_from_worker
            self.state.live_status_callback = self._update_live_status_from_worker
            self.state.final_message_callback = self._append_final_messages_from_worker
            self.state.token_stream_callback = self._stream_tokens_from_worker
            self.state.tool_approval_callback = self._tool_approval
            self.state.safety_approval_callback = self._safety_approval
            if state.unconsumed_followups:
                self._enqueue_unconsumed_followups(state.unconsumed_followups)
                self.state.unconsumed_followups = ()
            if state.task_continuations:
                self.state.task_continuations = state.task_continuations
            self._refresh_file_tree()
        if _has_terminal_error(materialized):
            self._mark_turn_failed(session_id)
        self._append_messages(
            materialized,
            started_at=started_at,
            session_id=session_id,
            finish_turn=True,
        )
        self._mark_turn_idle(session_id)
        self._refresh_footer()
        if exit_requested and current:
            self.action_quit()

    def _append_final_messages_from_worker(self, messages: tuple[TuiMessage, ...]) -> None:
        self._append_final_messages_from_worker_for_session(
            self.state.session.session_id,
            messages,
        )

    def _append_final_messages_from_worker_for_session(
        self,
        session_id: str,
        messages: Iterable[TuiMessage],
    ) -> None:
        if self._exiting:
            return
        materialized = tuple(messages)
        self._safe_call_from_thread(
            self._append_final_messages_for_session,
            session_id,
            materialized,
        )
        self._safe_call_from_thread(self._refresh_footer_throttled)

    def _append_final_messages_for_session(
        self,
        session_id: str,
        messages: tuple[TuiMessage, ...],
    ) -> None:
        self._append_messages(messages, session_id=session_id)

    def _mark_turn_failed(self, session_id: str) -> None:
        active_turn = self._session_turns.get(session_id)
        if active_turn is not None:
            active_turn.failed = True
        elif session_id == self.state.session.session_id:
            self._turn_failed = True

    def _handle_worker_exception(self, exc: BaseException, session_id: str = "") -> None:
        if self._exiting:
            return
        session_id = session_id or self.state.session.session_id
        self._mark_turn_failed(session_id)
        if session_id == self.state.session.session_id:
            self._clear_live_work()
            self._write(friendly_error_message(exc))
        else:
            self._append_background_message(session_id, friendly_error_message(exc))
        self._mark_turn_idle(session_id)

    def _mark_turn_idle(self, session_id: str = "") -> None:
        session_id = session_id or (
            self._active_turn.session_id if self._active_turn is not None else ""
        )
        if not session_id:
            session_id = self.state.session.session_id
        active_turn = self._session_turns.pop(session_id, None)
        failed = False
        interrupted = False
        if active_turn is not None and active_turn.session_id == session_id:
            failed = active_turn.failed
            interrupted = active_turn.interrupted
            if self._active_turn is active_turn:
                self._active_turn = self._turn_for_current_session()
        elif session_id == self.state.session.session_id:
            failed = self._turn_failed
            interrupted = self._interrupted
        if session_id:
            self._session_running.discard(session_id)
            self._session_live_status.pop(session_id, None)
            self._stream_pending.pop(session_id, None)
            self._stream_content.pop(session_id, None)
            self._stream_reasoning.pop(session_id, None)
            self._tool_turn_approvals_by_session.pop(session_id, None)
        self._running_turn = bool(self._session_running)
        if session_id == self.state.session.session_id:
            self._turn_failed = False
            self._interrupted = False
            self.state.cancellation_token = None
        if active_turn is not None and active_turn.session_id == session_id:
            self._turn_started_at = None
        if not self._current_session_is_running():
            self._route_checkpoint_writes_to_current_session()
        current = session_id == self.state.session.session_id
        if current and (failed or interrupted):
            self._clear_live_work()
        elif current:
            self._clear_live_work()
        self._close_computer_use_after_turn(session_id, active_turn)
        if current and (failed or interrupted):
            self._pause_prompt_queue(
                "turn failed" if failed else "turn interrupted"
            )
        self._refresh_footer()
        if current and self._prompt_queue.paused and self._prompt_queue.pending:
            reason = (
                f" Reason: {self._last_queue_pause_reason}."
                if self._last_queue_pause_reason
                else ""
            )
            self._write(
                TuiMessage(
                    kind="warning",
                    title="queue paused",
                    body=(
                        f"Denied or failed operation paused "
                        f"{len(self._prompt_queue.pending)} queued prompt(s)."
                        f"{reason} "
                        "Use /resume-queue to continue or /clear-queue to discard."
                    ),
                )
            )
            return
        if current:
            self._apply_pending_local_switch()
            if self._maybe_start_task_continuation():
                return
            self._maybe_start_next_queued_prompt()
        else:
            self._maybe_start_next_queued_prompt()

    def _apply_pending_local_switch(self) -> None:
        preset = self.state.pending_local_switch
        if preset is None or self._current_session_is_running():
            return
        result = apply_local_model_prepare_result(
            self.state,
            LocalModelPrepareRequest(preset=preset, source="/local"),
            LocalModelInstallResult(
                ok=True,
                preset=preset,
                message=f"{preset.label} 已就绪。",
                provider_base_url=self.state.local_provider_base_url,
            ),
        )
        self.state.pending_local_switch = None
        if result.messages:
            self._append_messages(result.messages)
        self._refresh_footer()

    def _append_messages(
        self,
        messages: Iterable[TuiMessage],
        *,
        started_at: float | None = None,
        session_id: str = "",
        finish_turn: bool = False,
    ) -> None:
        materialized = tuple(messages)
        target_session_id = session_id or self.state.session.session_id
        if finish_turn and target_session_id == self.state.session.session_id:
            self._clear_live_work()
        elif finish_turn:
            self._session_live_status.pop(target_session_id, None)
        self._deliver_turn_messages(target_session_id, materialized)

    def _deliver_turn_messages(
        self,
        session_id: str,
        messages: Iterable[TuiMessage],
    ) -> None:
        materialized = tuple(messages)
        active_turn = self._session_turns.get(session_id)
        display_messages = _undelivered_messages(active_turn, materialized)
        if active_turn is not None and _contains_visible_answer(materialized):
            active_turn.answer_visible = True
        if not display_messages:
            return
        self._insert_messages_for_session(
            session_id,
            display_messages,
            anchor_id=active_turn.anchor_id if active_turn is not None else "",
        )

    def _insert_messages_for_session(
        self,
        session_id: str,
        messages: Iterable[TuiMessage],
        *,
        anchor_id: str = "",
    ) -> None:
        materialized = tuple(messages)
        if not materialized:
            return
        current = session_id == self.state.session.session_id
        buffer = self._messages_for_session(session_id)
        insert_at = _turn_result_insert_index(buffer, anchor_id)
        for message in materialized:
            if message.kind == "user":
                continue
            if current and message.kind == "file" and message.preview:
                self._current_tab_title = message.title.strip()
                self._current_tab_content = message.preview
                self._open_content_tab(
                    self._current_tab_title or "file",
                    message.preview,
                    render_mode="auto",
                )
            elif current and message.kind in {"diff", "status", "task"} and message.preview:
                self._show_detail(message.title, message.preview)
            elif current and message.kind.startswith("tool ") and message.preview:
                self._current_tab_title = message.title.strip()
                self._current_tab_content = message.preview
            buffer.insert(insert_at, message)
            insert_at += 1
        if len(buffer) > 300:
            del buffer[: len(buffer) - 300]
        if current:
            if self._active_tab == "main":
                self._safe_render_active_tab()
            else:
                self._safe_refresh_content_tabs()
        elif any(
            message.kind != "status" or message.status in {"warning", "error"}
            for message in materialized
        ):
            self._session_has_updates.add(session_id)

    def _enqueue_unconsumed_followups(self, followups: Iterable[str]) -> None:
        for followup in followups:
            self._enqueue_prompt(
                followup,
                prefix="Current turn ended before this follow-up was consumed.",
            )

    def _handle_workspace_input(self, raw_value: str) -> bool:
        command_path = _strip_command_arg(raw_value.strip(), ("/workspace ", "/cd "))
        explicit = command_path is not None
        if command_path is None and raw_value.strip() in {"/workspace", "/cd"}:
            self._write(
                TuiMessage(
                    kind="status",
                    title="workspace",
                    body=(
                        "Usage: /workspace <folder>\n"
                        "You can also drag a folder into the terminal input and press Enter."
                    ),
                )
            )
            return True
        candidate = command_path if command_path is not None else raw_value
        workspace = (
            _directory_input_path(candidate, base=self.state.workspace, allow_bare=True)
            if explicit
            else _directory_input_path(candidate)
        )
        if workspace is None:
            if explicit:
                self._write(
                    TuiMessage(
                        kind="error",
                        title="workspace",
                        body=f"Folder not found: {candidate.strip()}",
                    )
                )
                return True
            return False
        self._request_workspace_switch(workspace)
        return True

    def _handle_session_browser_command(self, value: str) -> bool:
        clean = value.strip()
        if clean in {"/sessions", "/resume"}:
            preview = _sessions_preview(
                self.state.session_store.list_recent(limit=10_000),
                current_workspace=self.state.workspace,
                current_session_id=self.state.session.session_id,
            )
            self._show_detail("sessions", preview)
            self._write(
                TuiMessage(
                    kind="status",
                    title="sessions",
                    body="Sessions opened in a content tab.",
                )
            )
            return True
        raw_session_id = _strip_command_arg(clean, ("/resume ",))
        if raw_session_id is None:
            return False
        try:
            session_id = self.state.session_store.resolve_id(raw_session_id)
            session = self.state.session_store.load(session_id)
        except (OSError, ValueError) as exc:
            self._write(TuiMessage(kind="error", title="/resume", body=str(exc)))
            return True
        if session.workspace.resolve() != self.state.workspace.resolve():
            self._request_workspace_switch(session.workspace, session.session_id)
            return True
        self._switch_session(session.session_id)
        return True

    def _is_session_management_command(self, value: str) -> bool:
        clean = value.strip()
        return (
            clean in {"/new", "/new-session", "/sessions", "/resume"}
            or clean.startswith("/resume ")
        )

    def _handle_session_management_command(self, value: str) -> bool:
        clean = value.strip()
        if clean in {"/new", "/new-session"}:
            self._create_new_session()
            return True
        return self._handle_session_browser_command(clean)

    def _handle_immediate_command(self, prompt: str) -> bool:
        if prompt.strip().startswith("/task "):
            return False
        if not _is_immediate_command(prompt):
            return False
        if self._handle_verbose_command(prompt):
            return True
        previous_session_id = self.state.session.session_id
        result = handle_tui_command(prompt, self.state)
        if not result.handled:
            return False
        if prompt.strip().startswith("/remote --open "):
            self._disable_computer_use_for_remote_route()
        if self.state.session.session_id != previous_session_id:
            self._load_prompt_queue_for_current_session()
            if self.state.session.session_id not in self._session_messages:
                self._restore_main_messages_from_transcript()
        if result.messages:
            self._append_messages(result.messages)
            self._handle_pet_start_request(result.messages)
            self._handle_pet_setup_request(result.messages)
        if result.local_prepare is not None:
            self._start_local_prepare(result.local_prepare)
        if (
            prompt.strip().startswith("/setup-key ")
            and self.state.provider_api_key_available
        ):
            self._prompt_queue.resume()
            self._maybe_start_next_queued_prompt()
        if result.exit_requested:
            self.action_quit()
        return True

    def _handle_cron_natural_language(self, prompt: str) -> bool:
        try:
            body = maybe_create_cron_draft(prompt, workspace=self.state.workspace)
        except (OSError, ValueError) as exc:
            self._write(TuiMessage(kind="error", title="/cron", body=str(exc)))
            return True
        if body is None:
            return False
        self._remember_prompt_history(prompt)
        self._write(TuiMessage(kind="status", title="/cron", body=body))
        return True

    def _disable_computer_use_for_remote_route(self) -> None:
        runtime = self.state.behavior_runtime
        if runtime is None or not runtime.computer_use_enabled:
            return
        runtime.set_computer_use(False)
        refresh_computer_tool_surface(self.state)

    def _close_computer_use_after_turn(
        self,
        session_id: str,
        active_turn: _TurnRun | None,
    ) -> None:
        if active_turn is None or not active_turn.computer_use_enabled_at_start:
            return
        if session_id != self.state.session.session_id:
            return
        runtime = self.state.behavior_runtime
        if runtime is None or not runtime.computer_use_enabled:
            return
        runtime.set_computer_use(False)
        refresh_computer_tool_surface(self.state)

    def _handle_verbose_command(self, prompt: str) -> bool:
        """Toggle live reasoning streaming. Returns True if handled.

        `/verbose` flips the flag; `/verbose on|off` sets it explicitly. The
        flag is UI-local (it gates how streamed deltas render) so it is handled
        here rather than in the session-level command handler.
        """
        parts = prompt.strip().split()
        if not parts or parts[0] != "/verbose":
            return False
        argument = parts[1].lower() if len(parts) > 1 else ""
        if argument == "on":
            self._show_reasoning_stream = True
        elif argument == "off":
            self._show_reasoning_stream = False
        else:
            self._show_reasoning_stream = not self._show_reasoning_stream
        status = "on" if self._show_reasoning_stream else "off"
        self._write(
            TuiMessage(
                kind="status",
                title="verbose",
                body=f"Live reasoning streaming is {status}.",
                status="summary",
            )
        )
        return True

    def _write(self, message: TuiMessage) -> None:
        rerender = self._append_main_message(message)
        if self._active_tab != "main":
            self._refresh_content_tabs()
            return
        if rerender:
            self._render_active_tab()
            return
        log = self.query_one("#chat", RichLog)
        rendered = _render_message(message)
        log.write(rendered, expand=_should_expand_message(message.kind))
        self._maybe_scroll_chat_end(log)

    def _append_main_message(self, message: TuiMessage) -> bool:
        """Append a chat message, keeping live work status pinned last."""
        removed_permission_summary = False
        if _is_permission_summary_message(message):
            before_count = len(self._main_messages)
            self._main_messages = [
                existing
                for existing in self._main_messages
                if not _is_permission_summary_message(existing)
            ]
            removed_permission_summary = len(self._main_messages) != before_count
        if message.status == "live":
            self._main_messages = [
                existing for existing in self._main_messages if existing.status != "live"
            ]
            self._main_messages.append(message)
            self._live_message_active = True
            self._trim_main_messages()
            return True
        live_index = next(
            (
                index
                for index, existing in enumerate(self._main_messages)
                if existing.status == "live"
            ),
            None,
        )
        if live_index is None:
            self._main_messages.append(message)
            self._trim_main_messages()
            return removed_permission_summary
        self._main_messages.insert(live_index, message)
        self._trim_main_messages()
        return True

    def _trim_main_messages(self) -> None:
        if len(self._main_messages) > 300:
            del self._main_messages[: len(self._main_messages) - 300]

    def _append_background_message(self, session_id: str, message: TuiMessage) -> None:
        """Store a backgrounded session's message in its own buffer.

        Live placeholders are not persisted (live status is tracked separately in
        _session_live_status); everything else is appended so switching back shows
        the full turn, including errors/warnings that never reach the transcript.
        """
        if message.status == "live":
            return
        buffer = self._messages_for_session(session_id)
        if _is_permission_summary_message(message):
            buffer[:] = [
                existing
                for existing in buffer
                if not _is_permission_summary_message(existing)
            ]
        buffer.append(message)
        if len(buffer) > 300:
            del buffer[: len(buffer) - 300]
        if message.kind not in {"status"} or message.status in {"warning", "error"}:
            self._session_has_updates.add(session_id)

    def _start_live_work(self, body: str = "Preparing context...") -> None:
        self._live_status_text = body.strip() or "working on"
        self._upsert_live_message()
        self._safe_refresh_footer_throttled()

    def _update_live_status(self, message: TuiMessage) -> None:
        if not self._current_session_is_running():
            return
        body = message.body.strip() or message.title.strip()
        if not body:
            return
        if _is_generic_live_status(body) and _is_specific_live_status(self._live_status_text):
            return
        self._live_status_text = body
        self._session_live_status[self.state.session.session_id] = body
        self._upsert_live_message()
        self._safe_refresh_footer_throttled()

    def _upsert_live_message(self) -> None:
        message = self._pending_status_message
        if message is None:
            return
        self._append_main_message(message)
        if self._active_tab == "main":
            self._render_active_tab()

    def _clear_live_work(self, final_summary: str = "") -> None:
        had_live = bool(self._live_status_text.strip())
        self._live_status_text = ""
        before_count = len(self._main_messages)
        self._main_messages = [
            message for message in self._main_messages if message.status != "live"
        ]
        removed_live = len(self._main_messages) != before_count or self._live_message_active
        self._live_message_active = False
        if final_summary.strip():
            self._main_messages.append(
                TuiMessage(
                    kind="status",
                    title="done",
                    body=final_summary.strip(),
                    status="summary",
                )
            )
        if (had_live or removed_live or final_summary.strip()) and self._active_tab == "main":
            self._render_active_tab()

    @property
    def _pending_status_message(self) -> TuiMessage | None:
        if not self._live_status_text.strip():
            return None
        return TuiMessage(
            kind="status",
            title="runtime status",
            body=self._live_status_text.strip(),
            status="live",
        )

    @_pending_status_message.setter
    def _pending_status_message(self, value: TuiMessage | None) -> None:
        self._live_status_text = value.body.strip() if value is not None else ""

    def _update_live_status_from_worker(self, message: TuiMessage) -> None:
        self._safe_call_from_thread(self._update_live_status, message)
        self._safe_call_from_thread(self._refresh_footer_throttled)

    def _stream_tokens_from_worker(self, content: str, reasoning: str) -> None:
        """Buffer a streamed fragment for the active session (worker thread).

        Appends only to a pre-created list, so no dict mutation and no
        call_from_thread per token; the flush timer renders on the UI thread.
        """
        self._stream_tokens_from_worker_for_session(
            self.state.session.session_id, content, reasoning
        )

    def _stream_tokens_from_worker_for_session(
        self,
        session_id: str,
        content: str,
        reasoning: str,
    ) -> None:
        pending = self._stream_pending.get(session_id)
        if pending is not None:
            pending.append((content, reasoning))

    def _flush_token_stream(self) -> None:
        """Drain buffered stream fragments and render the active session's text.

        Runs on the UI thread via set_interval. Background sessions accumulate
        their text (shown on switch) but only the active session re-renders, so
        token bursts cost one render per ~60ms tick instead of one per token.
        """
        active_id = self.state.session.session_id
        for session_id, pending in self._stream_pending.items():
            count = len(pending)
            if count == 0:
                continue
            # Snapshot then delete exactly what we read; concurrent worker
            # appends land beyond `count` and are picked up next tick.
            fragments = pending[:count]
            del pending[:count]
            content_delta = "".join(part[0] for part in fragments)
            reasoning_delta = "".join(part[1] for part in fragments)
            if content_delta:
                self._stream_content[session_id] = (
                    self._stream_content.get(session_id, "") + content_delta
                )
            if reasoning_delta:
                self._stream_reasoning[session_id] = (
                    self._stream_reasoning.get(session_id, "") + reasoning_delta
                )
            text = self._streamed_live_text(session_id)
            if session_id == active_id:
                if text:
                    self._live_status_text = text
                    self._session_live_status[active_id] = text
                    self._upsert_live_message()
            elif text:
                self._session_live_status[session_id] = text
                self._session_has_updates.add(session_id)

    def _streamed_live_text(self, session_id: str) -> str:
        """Compose the live preview text from streamed content and reasoning."""
        content = self._stream_content.get(session_id, "")
        reasoning = self._stream_reasoning.get(session_id, "")
        if self._show_reasoning_stream and reasoning and not content:
            return f"💭 {reasoning}"
        return content

    def _write_from_worker(self, message: TuiMessage) -> None:
        self._safe_call_from_thread(self._write, message)
        self._safe_call_from_thread(self._refresh_footer_throttled)

    def _update_live_status_from_worker_for_session(
        self,
        session_id: str,
        message: TuiMessage,
    ) -> None:
        self._safe_call_from_thread(self._update_live_status_for_session, session_id, message)
        self._safe_call_from_thread(self._refresh_footer_throttled)

    def _write_from_worker_for_session(
        self,
        session_id: str,
        message: TuiMessage,
    ) -> None:
        self._safe_call_from_thread(self._write_for_session, session_id, message)
        self._safe_call_from_thread(self._refresh_footer_throttled)

    def _update_live_status_for_session(
        self,
        session_id: str,
        message: TuiMessage,
    ) -> None:
        body = message.body.strip() or message.title.strip()
        if not body:
            return
        if _is_generic_live_status(body) and _is_specific_live_status(
            self._session_live_status.get(session_id, "")
        ):
            return
        self._session_live_status[session_id] = body
        if session_id == self.state.session.session_id:
            self._update_live_status(message)
        else:
            self._session_has_updates.add(session_id)

    def _write_for_session(self, session_id: str, message: TuiMessage) -> None:
        if session_id == self.state.session.session_id:
            self._write(message)
            return
        self._append_background_message(session_id, message)

    def _show_detail(self, title: str, content: str) -> None:
        self._current_tab_title = title.strip()
        self._current_tab_content = content
        self._open_content_tab(
            self._current_tab_title or "detail",
            content,
            render_mode="markdown",
        )

    def _save_content_tabs_for_session(self, session_id: str) -> None:
        key = session_id or self.state.session.session_id
        self._session_content_tabs[key] = _SessionContentTabs(
            open_tabs=dict(self._open_tabs),
            active_tab=self._active_tab,
            hovered_tab=self._hovered_content_tab,
            current_title=self._current_tab_title,
            current_content=self._current_tab_content,
        )

    def _restore_content_tabs_for_session(self, session_id: str) -> None:
        tabs = self._session_content_tabs.get(session_id)
        if tabs is None:
            self._open_tabs = {}
            self._active_tab = "main"
            self._hovered_content_tab = ""
            self._current_tab_title = ""
            self._current_tab_content = ""
            return
        self._open_tabs = dict(tabs.open_tabs)
        self._active_tab = (
            tabs.active_tab
            if tabs.active_tab == "main" or tabs.active_tab in self._open_tabs
            else "main"
        )
        self._hovered_content_tab = tabs.hovered_tab
        self._current_tab_title = tabs.current_title
        self._current_tab_content = tabs.current_content

    def _handle_trust_command(self, value: str) -> bool:
        clean = value.strip().lower()
        if clean not in {"/trust", "/trust on", "/trust off"}:
            return False
        if clean == "/trust off":
            self._set_trusted(False)
            self._write(
                TuiMessage(
                    kind="status",
                    title="trust",
                    body=(
                        "Auto-approve turned off. Writes and shell commands will ask "
                        "for approval again."
                    ),
                )
            )
            return True
        self._set_trusted(True)
        self._write(
            TuiMessage(
                kind="warning",
                title="trust",
                body=(
                    "Auto-approve on for this session: workspace writes, shell "
                    "commands, and network egress may run without prompting. "
                    "Sensitive paths and hard-denied commands are still blocked. "
                    "Use /trust off to revert."
                ),
            )
        )
        return True

    def _set_trusted(self, trusted: bool) -> None:
        session_id = self.state.session.session_id
        if trusted:
            self._trusted_sessions.add(session_id)
        else:
            self._trusted_sessions.discard(session_id)
        self._apply_trust_to_current_session()
        self._refresh_footer()

    def _apply_trust_to_current_session(self) -> None:
        """Relax (or restore) write/shell gating for the current session's state."""
        self._bind_current_session_approval_cache()
        policy = self.state.tool_access_policy
        if policy is not None:
            target_mode = (
                ToolAccessMode.WORKSPACE_WRITE
                if self._trusted
                else (self._base_access_mode or policy.mode)
            )
            if policy.mode != target_mode:
                self.state.tool_access_policy = replace(policy, mode=target_mode)
        if self._trusted and self.state.approval_cache is not None:
            for key in (
                "capability:shell",
                "capability:shell-network",
                "capability:network",
            ):
                self.state.approval_cache.allow_for_session(key)

    def _bind_current_session_approval_cache(self) -> None:
        cache = self._approval_cache_for_session(self.state.session.session_id)
        if cache is not self.state.approval_cache:
            self.state.approval_cache = cache
        self.state.native_tools = self._native_tools_for_approval_cache(
            cache,
            self.state.session.session_id,
        )
        self.state.approval_callbacks_installed = False

    def _clear_chat_display(self) -> None:
        session_id = self.state.session.session_id
        self._session_clear_snapshots[session_id] = list(self._main_messages)
        try:
            self.query_one("#chat", RichLog).clear()
        except (NoMatches, ScreenStackError):
            pass
        self._main_messages.clear()
        self._render_active_tab()
        self._write(
            TuiMessage(
                kind="status",
                title="clear",
                body=(
                    "Screen chat cleared; transcript and context are unchanged. "
                    "Open tabs were kept. Use /undo-clear to restore the display."
                ),
            )
        )

    def _undo_clear_chat_display(self) -> None:
        snapshot = self._session_clear_snapshots.pop(
            self.state.session.session_id,
            None,
        )
        if snapshot is None:
            self._write(
                TuiMessage(
                    kind="status",
                    title="undo-clear",
                    body="No cleared chat display to restore.",
                )
            )
            return
        self._main_messages = snapshot
        self._render_active_tab()
        self._write(
            TuiMessage(
                kind="status",
                title="undo-clear",
                body="Restored the cleared chat display.",
            )
        )

    def _remember_unsent_draft(self, text: str) -> None:
        clean = _clean_input_text(text).strip()
        if clean and not clean.startswith("/restore-draft"):
            self._session_unsent_drafts[self.state.session.session_id] = clean

    def _restore_unsent_draft(self) -> None:
        draft = self._session_unsent_drafts.get(self.state.session.session_id, "")
        if not draft:
            self._write(
                TuiMessage(
                    kind="status",
                    title="draft",
                    body="No prompt draft is available to restore.",
                )
            )
            return
        try:
            input_widget = self.query_one("#prompt-input", TextArea)
        except (NoMatches, ScreenStackError):
            self._write(
                TuiMessage(
                    kind="warning",
                    title="draft",
                    body="Prompt editor is not available right now.",
                )
            )
            return
        _set_prompt_text(input_widget, draft)
        input_widget.focus()
        self._update_command_hints(draft)
        self._write(
            TuiMessage(
                kind="status",
                title="draft",
                body="Restored the last prompt draft to the input box.",
            )
        )

    def _remember_prompt_history(self, prompt: str) -> None:
        clean = prompt.strip()
        if not clean or clean.startswith("/"):
            return
        session_id = self.state.session.session_id
        history = self._session_prompt_history.setdefault(session_id, [])
        if history and history[-1] == clean:
            self._session_prompt_history_index[session_id] = None
            self._session_prompt_history_draft.pop(session_id, None)
            return
        history.append(clean)
        if len(history) > PROMPT_HISTORY_LIMIT:
            del history[: len(history) - PROMPT_HISTORY_LIMIT]
        self._session_prompt_history_index[session_id] = None
        self._session_prompt_history_draft.pop(session_id, None)

    def _handle_prompt_history_key(self, key: str) -> bool:
        session_id = self.state.session.session_id
        history = self._session_prompt_history.get(session_id, [])
        if not history:
            return False
        try:
            input_widget = self.query_one("#prompt-input", TextArea)
        except (NoMatches, ScreenStackError):
            return False
        index = self._session_prompt_history_index.get(session_id)
        if index is None:
            self._session_prompt_history_draft[session_id] = _prompt_text(input_widget)
            index = len(history)
        if key == "up":
            index = max(0, index - 1)
            value = history[index]
        else:
            index += 1
            if index >= len(history):
                index = len(history)
                value = self._session_prompt_history_draft.pop(session_id, "")
                self._session_prompt_history_index[session_id] = None
                _set_prompt_text(input_widget, value)
                self._update_command_hints(value)
                return True
            value = history[index]
        self._session_prompt_history_index[session_id] = index
        _set_prompt_text(input_widget, value)
        self._update_command_hints(value)
        return True

    def _paste_into_prompt(self, text: str) -> None:
        if not text:
            return
        paste_key = _paste_dedupe_key(text)
        now = monotonic()
        if (
            paste_key
            and paste_key == self._last_prompt_paste_key
            and now - self._last_prompt_paste_at < PASTE_DEDUPE_SECONDS
        ):
            return
        self._last_prompt_paste_key = paste_key
        self._last_prompt_paste_at = now
        input_widget = self.query_one("#prompt-input", TextArea)
        inserted = _insert_prompt_text(
            input_widget,
            text,
            max_chars=MAX_PROMPT_CHARS,
        )
        if not inserted:
            self._write(
                TuiMessage(
                    kind="warning",
                    title="paste",
                    body="Prompt is already at the maximum supported length.",
                )
            )
            return
        input_widget.focus()
        self._update_command_hints(_prompt_text(input_widget))

    def _append_multiline_paste_to_compose(self, text: str) -> None:
        clean = _clean_input_text(text).strip("\n")
        if not clean:
            return
        if not self._compose_mode:
            self._compose_mode = True
            self._compose_lines = []
        self._compose_lines.extend(clean.splitlines())
        draft = "\n".join(self._compose_lines)
        if len(draft) > MAX_PROMPT_CHARS:
            draft = draft[:MAX_PROMPT_CHARS]
            self._compose_lines = draft.splitlines()
        self._show_detail("compose draft", "\n".join(self._compose_lines))
        self._refresh_footer()
        self._write(
            TuiMessage(
                kind="status",
                title="compose",
                body=(
                    f"Pasted {len(clean.splitlines())} line(s) into compose draft. "
                    "Review it in the content tab, then use /send or /cancel-compose."
                ),
            )
        )

    def _open_content_tab(self, title: str, content: str, *, render_mode: str = "auto") -> None:
        clean_title = title.strip() or "detail"
        self._open_tabs[clean_title] = _OpenTab(
            title=clean_title,
            content=content,
            render_mode=render_mode.strip() or "auto",
        )
        self._active_tab = clean_title
        self._save_content_tabs_for_session(self.state.session.session_id)
        self._render_active_tab()

    def _render_active_tab(self) -> None:
        self._refresh_content_tabs()
        log = self.query_one("#chat", RichLog)
        markdown = self.query_one("#content-markdown", _SelectableRichLog)
        if self._active_tab == "main":
            markdown.display = False
            log.display = True
            log.clear()
            for message in self._main_messages:
                log.write(
                    _render_message(message),
                    expand=_should_expand_message(message.kind),
                )
            self._maybe_scroll_chat_end(log)
            return
        log.display = False
        markdown.display = True
        tab = self._open_tabs.get(self._active_tab)
        if tab is not None:
            markdown.clear()
            markdown.write(
                _preview_tab_content(tab.title, tab.content, mode=tab.render_mode),
                scroll_end=False,
            )
            markdown.scroll_home(animate=False)

    def _content_tabs_label(self) -> str:
        pieces = [
            _content_tab_display_label(
                name,
                active=name == self._active_tab,
                hovered=name == self._hovered_content_tab,
            )
            for name in self._content_tab_names()
        ]
        return CONTENT_TAB_PREFIX + f" {CONTENT_TAB_SEPARATOR} ".join(pieces)

    def _content_tab_names(self) -> list[str]:
        return ["main", *self._open_tabs.keys()]

    def _content_tabs_visible(self) -> bool:
        return self.sidebar_visible or bool(self._open_tabs)

    def _refresh_content_tabs(self) -> None:
        tabs = self.query_one("#content-tabs", Static)
        if not self._content_tabs_visible():
            tabs.update("")
            tabs.display = False
            return
        tabs.display = True
        tabs.update(self._content_tabs_label())

    def _set_hovered_content_tab(self, target: str) -> None:
        clean = target if target in self._content_tab_names() else ""
        if clean == self._hovered_content_tab:
            return
        self._hovered_content_tab = clean
        self._save_content_tabs_for_session(self.state.session.session_id)
        self._refresh_content_tabs()

    def _content_tab_at_x(self, x: int) -> str:
        local_x = _content_tab_local_x(x)
        tabs = self._content_tab_names()
        labels = [
            _content_tab_display_label(
                name,
                active=name == self._active_tab,
                hovered=name == self._hovered_content_tab,
            )
            for name in tabs
        ]
        index = _tab_index_from_x(local_x, labels)
        if index < 0 or index >= len(tabs):
            return ""
        return tabs[index]

    def _handle_content_tab_click(self, x: int) -> None:
        local_x = _content_tab_local_x(x)
        tabs = self._content_tab_names()
        labels = [
            _content_tab_display_label(
                name,
                active=name == self._active_tab,
                hovered=name == self._hovered_content_tab,
            )
            for name in tabs
        ]
        index = _tab_index_from_x(local_x, labels)
        if index < 0 or index >= len(tabs):
            return
        target = tabs[index]
        if target != "main" and _tab_click_hits_close(local_x, labels, index):
            self._close_content_tab(target)
            return
        self._activate_content_tab(target)

    def _activate_content_tab(self, target: str) -> None:
        if target != "main" and target not in self._open_tabs:
            return
        if target == self._active_tab:
            return
        self._active_tab = target
        self._save_content_tabs_for_session(self.state.session.session_id)
        self._render_active_tab()

    def _safe_render_active_tab(self) -> None:
        try:
            self._render_active_tab()
        except (NoMatches, ScreenStackError):
            return

    def _safe_refresh_content_tabs(self) -> None:
        try:
            self._refresh_content_tabs()
        except (NoMatches, ScreenStackError):
            return

    def _close_active_tab(self) -> bool:
        if self._active_tab == "main":
            return False
        return self._close_content_tab(self._active_tab)

    def _close_active_tab_with_feedback(self) -> None:
        closed = self._close_active_tab()
        body = "Closed current tab." if closed else "Already on main tab."
        self._write(TuiMessage(kind="status", title="tab", body=body))

    def _close_content_tab(self, target: str) -> bool:
        if target == "main" or target not in self._open_tabs:
            return False
        self._open_tabs.pop(target, None)
        if self._hovered_content_tab == target:
            self._hovered_content_tab = ""
        self._active_tab = "main"
        self._save_content_tabs_for_session(self.state.session.session_id)
        self._safe_render_active_tab()
        return True

    def _active_tab_content(self) -> str:
        if self._active_tab == "main":
            return ""
        tab = self._open_tabs.get(self._active_tab)
        if tab is not None:
            return tab.content
        return self._current_tab_content

    def _scroll_active_content(self, action: str) -> None:
        target = (
            self.query_one("#chat", RichLog)
            if self._active_tab == "main"
            else self.query_one("#content-markdown", _SelectableRichLog)
        )
        if action == "page_up":
            self._chat_follow_tail = False
            target.scroll_page_up(animate=False)
        elif action == "page_down":
            target.scroll_page_down(animate=False)
            if self._active_tab == "main":
                self._chat_follow_tail = _is_scroll_at_end(target)
        elif action == "home":
            self._chat_follow_tail = False
            target.scroll_home(animate=False)
        elif action == "end":
            if self._active_tab == "main":
                self._chat_follow_tail = True
            target.scroll_end(animate=False)

    def _maybe_scroll_chat_end(self, log: object) -> None:
        if self._chat_follow_tail:
            try:
                log.scroll_end(animate=False, immediate=True)
            except TypeError:
                log.scroll_end(animate=False)

    def _route_active_content_scroll_event(self, event: object, action: str) -> bool:
        widget_id = getattr(getattr(event, "widget", None), "id", "") or ""
        if self._active_tab == "main":
            return False
        if widget_id not in {"content-markdown", "main-column", "workbench", "content"}:
            return False
        try:
            if action == "scroll_down":
                self.query_one("#content-markdown", _SelectableRichLog).scroll_down(animate=False)
            elif action == "scroll_up":
                self.query_one("#content-markdown", _SelectableRichLog).scroll_up(animate=False)
            else:
                return False
        except Exception:
            return False
        prevent_default = getattr(event, "prevent_default", None)
        if callable(prevent_default):
            prevent_default()
        stop = getattr(event, "stop", None)
        if callable(stop):
            stop()
        return True

    def _handle_pet_start_request(self, messages: Iterable[TuiMessage]) -> None:
        if not any("pet_start_requested=true" in message.refs for message in messages):
            return
        if self._pet_process is not None and self._pet_process.poll() is None:
            self._write(
                TuiMessage(
                    kind="status",
                    title="desktop pet",
                    body=(
                        "Desktop pet window is already running from this TUI. "
                        "If you cannot see it, check the right side of the desktop "
                        "or Mission Control."
                    ),
                )
            )
            return
        try:
            command = _pet_start_command(self.state.data_dir)
            if command is None:
                self._write(
                    TuiMessage(
                        kind="error",
                        title="desktop pet",
                        body=_pet_frontend_missing_message(),
                    )
                )
                return
            self._pet_process = subprocess.Popen(
                command,
                cwd=str(self.state.workspace),
                env=_pet_subprocess_env(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            self._write(
                TuiMessage(
                    kind="error",
                    title="desktop pet",
                    body=f"Could not start desktop pet window: {exc}",
                )
            )
            return
        self.set_timer(PET_START_CHECK_SECONDS, self._report_pet_start_result)

    def _handle_pet_setup_request(self, messages: Iterable[TuiMessage]) -> None:
        if not any("pet_setup_requested=true" in message.refs for message in messages):
            return
        if self._pet_setup_running:
            self._write(
                TuiMessage(
                    kind="status",
                    title="desktop pet",
                    body="Desktop pet setup is already running.",
                )
            )
            return
        data_dir = self.state.data_dir
        if data_dir is None:
            self._write(
                TuiMessage(
                    kind="error",
                    title="desktop pet",
                    body="Desktop pet setup needs a Deepmate data directory.",
                )
            )
            return
        approval_key = "capability:pet-runtime-setup"
        cache = self.state.approval_cache
        allowed = (
            cache is not None
            and (cache.is_allowed(approval_key) or cache.consume_once(approval_key))
        )
        if not allowed:
            result = self._request_approval(
                "Pet setup approval",
                "\n".join(
                    (
                        "Install the optional Electron runtime for the desktop pet.",
                        f"Location: {data_dir / 'pet' / 'ui_runtime'}",
                        "Network: npm registry and Electron download host.",
                        f"Scope: {approval_key}",
                        "Allow once: applies to this setup run only.",
                        "Allow for session: remembered until Deepmate exits.",
                    )
                ),
                subtitle="network",
                subject="desktop pet runtime setup",
                refs=(f"approval_key={approval_key}", "network=on"),
                session_id=self.state.session.session_id,
            )
            if result == "deny":
                self._write(
                    TuiMessage(
                        kind="warning",
                        title="desktop pet",
                        body="Desktop pet setup was denied.",
                    )
                )
                return
            if cache is not None:
                if result == "session":
                    cache.allow_for_session(approval_key)
                elif result == "once":
                    cache.allow_once(approval_key)
        self._start_pet_setup(data_dir)

    def _start_pet_setup(self, data_dir: Path) -> None:
        self._pet_setup_running = True
        self._running_turn = True
        self._start_live_work("正在安装桌面宠物运行时")
        self._refresh_footer()
        self.run_worker(
            lambda: self._run_pet_setup_worker(data_dir),
            thread=True,
        )

    def _run_pet_setup_worker(self, data_dir: Path) -> None:
        try:
            result = prepare_pet_runtime(
                data_dir,
                progress=lambda message: self._safe_call_from_thread(
                    self._update_pet_setup_progress,
                    message,
                ),
            )
        except Exception as exc:
            result = PetSetupResult(
                ok=False,
                ui_dir=data_dir / "pet" / "ui_runtime",
                message=f"Desktop pet setup failed: {exc}",
            )
        self._safe_call_from_thread(self._finish_pet_setup_worker, result)

    def _update_pet_setup_progress(self, message: str) -> None:
        self._start_live_work(message)
        self._refresh_footer_throttled()

    def _finish_pet_setup_worker(self, result: PetSetupResult) -> None:
        self._pet_setup_running = False
        self._running_turn = bool(self._session_running)
        self._clear_live_work()
        if result.ok:
            body = f"{result.message}\nRuntime directory: {result.ui_dir}"
            kind = "status"
        else:
            details = []
            if result.stderr:
                details.append(f"stderr:\n{result.stderr}")
            if result.stdout:
                details.append(f"stdout:\n{result.stdout}")
            body = f"{result.message}\nRuntime directory: {result.ui_dir}"
            if details:
                body += "\n\n" + "\n\n".join(details)
            kind = "error"
        self._write(TuiMessage(kind=kind, title="desktop pet", body=body))
        self._refresh_footer()
        self._maybe_start_next_queued_prompt()

    def _report_pet_start_result(self) -> None:
        process = self._pet_process
        if process is None:
            return
        if process.poll() is None:
            self._write(
                TuiMessage(
                    kind="status",
                    title="desktop pet",
                    body=(
                        "Desktop pet window is running. It appears as a small "
                        "always-on-top desktop companion near the upper-right area."
                    ),
                )
            )
            return
        stderr = ""
        if process.stderr is not None:
            try:
                stderr = process.stderr.read(PET_START_ERROR_LIMIT)
            except OSError:
                stderr = ""
        self._write(
            TuiMessage(
                kind="error",
                title="desktop pet",
                body=(
                    f"Desktop pet exited immediately with code {process.returncode}."
                    + (f"\n{stderr.strip()}" if stderr.strip() else "")
                ),
            )
        )

    def _stop_pet_process(self) -> None:
        process = self._pet_process
        self._pet_process = None
        if process is None:
            return
        poll = getattr(process, "poll", None)
        if callable(poll) and poll() is not None:
            return
        self._signal_pet_process(process, signal.SIGTERM)
        wait = getattr(process, "wait", None)
        if callable(wait):
            try:
                wait(timeout=2)
                return
            except subprocess.TimeoutExpired:
                pass
            except OSError:
                return
        self._signal_pet_process(process, signal.SIGKILL)
        if callable(wait):
            try:
                wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                return

    def _signal_pet_process(self, process: object, sig: int) -> None:
        pid = getattr(process, "pid", None)
        if isinstance(pid, int) and pid > 0:
            try:
                os.killpg(pid, sig)
                return
            except ProcessLookupError:
                return
            except OSError:
                pass
        method_name = "terminate" if sig == signal.SIGTERM else "kill"
        method = getattr(process, method_name, None)
        if callable(method):
            try:
                method()
            except OSError:
                return

    def _refresh_file_tree(self) -> None:
        items = workspace_file_items(
            self.state.workspace,
            expanded_dirs=tuple(self._expanded_dirs),
            limit=200,
        )
        signature = tuple((item.relative_path, item.badge) for item in items)
        if signature == self._last_file_tree_signature:
            return
        self._last_file_tree_signature = signature
        try:
            tree = self.query_one("#file-tree", ListView)
        except (NoMatches, ScreenStackError):
            return
        tree.clear()
        self._file_items = tuple(item.relative_path for item in items)
        self._file_item_is_dir = tuple(item.is_dir for item in items)
        for item in items:
            tree.append(
                ListItem(
                    Label(
                        _file_nav_label(
                            item.relative_path,
                            item.badge,
                            item.is_dir,
                            expanded=item.relative_path.rstrip("/") in self._expanded_dirs,
                        )
                    )
                )
            )

    def _refresh_footer(self) -> None:
        try:
            self._refresh_session_tabs()
            self.query_one("#context-window", Static).update(self._context_window_label())
            status_bar = self.query_one("#status-bar", Static)
            label = self._status_label()
            status_bar.update(label)
            status_bar.display = bool(label)
        except (NoMatches, ScreenStackError):
            return

    def _refresh_footer_throttled(self) -> None:
        now = monotonic()
        if now - self._last_footer_refresh_at < FOOTER_REFRESH_MIN_INTERVAL_SECONDS:
            return
        self._last_footer_refresh_at = now
        self._refresh_footer()

    def _safe_refresh_footer_throttled(self) -> None:
        try:
            self._refresh_footer_throttled()
        except Exception:
            return

    def _visible_session_tabs(self) -> list[tuple[str, str]]:
        active_id = self.state.session.session_id
        tabs = list(self._session_tabs)
        if len(tabs) <= 4:
            return tabs
        active_index = next(
            (index for index, (session_id, _) in enumerate(tabs) if session_id == active_id),
            -1,
        )
        if active_index < 0:
            return [*tabs[-3:], (active_id, self.state.session.title)]
        start = min(max(0, active_index - 1), max(0, len(tabs) - 4))
        return tabs[start : start + 4]

    def _session_tab_text(self, session_id: str, title: str) -> str:
        active = session_id == self.state.session.session_id
        if active:
            dot = "●"
        elif session_id in self._session_running:
            dot = "…"
        elif session_id in self._session_has_updates:
            dot = "•"
        else:
            dot = "·"
        clean_title = _short_session_title(title, active=active)
        suffix = session_id[:8]
        if active and len(self._session_tabs) > 4:
            current = next(
                (
                    index
                    for index, (known_id, _) in enumerate(self._session_tabs, start=1)
                    if known_id == session_id
                ),
                0,
            )
            if current:
                suffix = f"{suffix} {current}/{len(self._session_tabs)}"
        return f"{dot} {clean_title} · {suffix}"

    def _session_tab_classes(self, session_id: str) -> str:
        base = "session-tab"
        if session_id == self.state.session.session_id:
            return f"{base} session-tab-active"
        return base

    def _refresh_session_tabs(self) -> None:
        self._remember_session_tab(self.state.session.session_id, self.state.session.title)
        tabs = self.query_one("#session-tabs", Horizontal)
        visible_ids = {
            _session_button_id(session_id)
            for session_id, _ in self._visible_session_tabs()
        }
        for child in tabs.children:
            if isinstance(child, Static):
                child.display = child.id in visible_ids
        for session_id, title in self._visible_session_tabs():
            button_id = _session_button_id(session_id)
            try:
                button = self.query(f"#{button_id}").first()
            except NoMatches:
                button = None
            if isinstance(button, Static):
                button.update(self._session_tab_text(session_id, title))
                button.set_classes(self._session_tab_classes(session_id))
                button.display = True
                continue
            tabs.mount(
                Static(
                    self._session_tab_text(session_id, title),
                    id=button_id,
                    classes=self._session_tab_classes(session_id),
                )
            )

    def _context_window_label(self) -> str:
        stats = self._stats_for_session(self.state.session.session_id)
        summary = stats.context_window_summary()
        ratio = stats.context_usage_ratio()
        color = _context_window_color(ratio)
        if color is None:
            return summary
        return f"[{color}]{summary}[/]"

    def _status_label(self) -> str:
        stage = self.state.task_stage_label()
        left = _footer_left_label(self.state)
        if self._trusted:
            left = f"{left}  │  auto-approve"
        if (
            self.state.behavior_runtime is not None
            and self.state.behavior_runtime.computer_use_enabled
            and "computer on" not in left
        ):
            left = f"{left}  │  computer on"
        show_left = self._trusted or (
            self.state.behavior_runtime is not None
            and self.state.behavior_runtime.computer_use_enabled
        )
        if self._pending_approval_is_current_session():
            center = "approval · waiting for your choice"
            queue_count = self._approval_queue_count_for_current_session()
            if queue_count:
                center += f" · {queue_count} queued"
            right = "Esc deny"
        elif self._current_session_is_running():
            center = f"{SPINNER_FRAMES[self._spinner_frame]} running"
            right = "Esc interrupt"
        elif self._other_session_running_label():
            center = self._other_session_running_label()
            right = "new prompts queue"
        elif self._compose_mode:
            center = f"compose · {len(self._compose_lines)} lines"
            right = "/send"
        elif self._prompt_queue.footer_label():
            center = self._prompt_queue.footer_label()
            right = "/queue"
        elif stage:
            center = stage
            right = "/task"
        else:
            return left if show_left else ""
        return f"{left}  │  {center}  │  {right}"

    def _write_start_message(self) -> None:
        self._write(
            TuiMessage(
                kind="welcome",
                title="new session",
                body=_welcome_splash(
                    workspace=self.state.workspace,
                    session_id=self.state.session.session_id,
                    provider_name=self.state.provider_name,
                    api_key_env=self.state.provider_api_key_env,
                    api_key_available=self.state.provider_api_key_available,
                ),
            )
        )

    def _handle_preview_command(self, value: str) -> bool:
        clean = value.strip()
        if clean in {"/hide-preview", "/close-preview", "/close-tab"}:
            self._close_active_tab_with_feedback()
            return True
        if clean not in {"/preview", "/detail"}:
            return False
        if not self._current_tab_content.strip():
            self._write(
                TuiMessage(
                    kind="status",
                    title="tab",
                    body=(
                        "No content tab yet. Use /open, /diff, /status, /task, "
                        "or run a tool and then use /detail."
                    ),
                )
            )
            return True
        self._show_detail(self._current_tab_title, self._current_tab_content)
        self._write(TuiMessage(kind="status", title="tab", body="Content tab shown."))
        return True

    def _handle_find_command(self, value: str) -> bool:
        clean = value.strip()
        if clean == "/find":
            self._write(
                TuiMessage(
                    kind="status",
                    title="/find",
                    body="Usage: /find <keyword>",
                )
            )
            return True
        if not clean.startswith("/find "):
            return False
        query = clean[len("/find ") :].strip()
        if not query:
            self._write(
                TuiMessage(
                    kind="status",
                    title="/find",
                    body="Usage: /find <keyword>",
                )
            )
            return True
        content = self._active_tab_content()
        if not content.strip():
            self._write(
                TuiMessage(
                    kind="warning",
                    title="/find",
                    body="No active content to search. Use /open <path> or /detail first.",
                )
            )
            return True
        title = self._active_tab if self._active_tab != "main" else "main"
        preview, total = _find_in_content_preview(content, query, title=title)
        if total == 0:
            self._write(
                TuiMessage(
                    kind="status",
                    title="/find",
                    body=f"No matches for {query!r} in {title}.",
                )
            )
            return True
        self._show_detail(f"find: {query}", preview)
        self._write(
            TuiMessage(
                kind="file",
                title="/find",
                body=f"Found {total} matching line(s) for {query!r}.",
                preview=preview,
            )
        )
        return True

    def _handle_rewind_command(self, value: str) -> bool:
        clean = value.strip()
        if clean not in {"/rewind", "/undo"} and not clean.startswith(("/rewind ", "/undo ")):
            return False
        try:
            args = shlex.split(clean)
        except ValueError as exc:
            self._write(TuiMessage(kind="error", title="rewind", body=str(exc)))
            return True
        if not args:
            return True
        if self._current_session_is_running() and any(arg in {"--apply", "-y"} for arg in args):
            self._write(
                TuiMessage(
                    kind="warning",
                    title="rewind",
                    body="Wait for the current turn to finish before applying a rewind.",
                )
            )
            return True
        try:
            message, detail = self._build_or_apply_rewind(args)
        except (OSError, ValueError) as exc:
            self._write(TuiMessage(kind="error", title="rewind", body=str(exc)))
            return True
        self._write(message)
        if detail:
            self._show_detail("rewind", detail)
        return True

    def _build_or_apply_rewind(self, args: list[str]) -> tuple[TuiMessage, str]:
        mode = "workspace"
        apply = False
        force = False
        target_turn_id = ""
        index = 1
        while index < len(args):
            arg = args[index]
            if arg in {"--apply", "-y"}:
                apply = True
            elif arg == "--force":
                force = True
            elif arg == "--mode":
                index += 1
                if index >= len(args):
                    raise ValueError("usage: /rewind [TURN_ID] [--mode workspace|conversation|both] [--apply] [--force]")
                mode = args[index].strip().lower()
            elif arg.startswith("--mode="):
                mode = arg.partition("=")[2].strip().lower()
            elif not target_turn_id:
                target_turn_id = arg
            else:
                raise ValueError(f"unknown rewind argument: {arg}")
            index += 1
        if mode not in {"workspace", "conversation", "both"}:
            raise ValueError("rewind mode must be workspace, conversation, or both")
        data_dir = self.state.data_dir or self.state.session_store.directory.parent
        turn_store = TurnCheckpointStore.in_data_dir(
            data_dir,
            self.state.session.profile.name,
            self.state.session.session_id,
        )
        turns = turn_store.load_turns()
        if not target_turn_id:
            return _rewind_turn_list_message(turns), ""
        target = turn_store.require_turn(target_turn_id)
        workspace_store = WorkspaceCheckpointStore.in_data_dir(
            data_dir,
            self.state.session.profile.name,
            self.state.session.session_id,
        )
        workspace_plan = (
            workspace_store.rewind_plan(target.turn_id, self.state.session.workspace)
            if mode in {"workspace", "both"}
            else None
        )
        conversation_plan = (
            _tui_conversation_rewind_plan(self.state.session_store, self.state.session, target)
            if mode in {"conversation", "both"}
            else None
        )
        detail = _format_tui_rewind_detail(
            target_turn_id=target.turn_id,
            mode=mode,
            apply=apply,
            force=force,
            workspace_plan=workspace_plan,
            conversation_plan=conversation_plan,
        )
        if not apply:
            return (
                TuiMessage(
                    kind="status",
                    title="rewind preview",
                    body=(
                        _rewind_preview_summary(workspace_plan, conversation_plan)
                        + f"\n\nUse `/rewind {target.turn_id} --apply` to apply."
                    ),
                ),
                detail,
            )
        if workspace_plan is not None and workspace_plan.has_conflicts() and not force:
            return (
                TuiMessage(
                    kind="warning",
                    title="rewind blocked",
                    body=(
                        "Workspace rewind has conflicts. Review the detail tab, then "
                        "rerun with `--force` only if you want to overwrite current files."
                    ),
                ),
                detail,
            )
        if workspace_plan is not None:
            workspace_store.apply_rewind(
                target.turn_id,
                self.state.session.workspace,
                force=force,
            )
        if conversation_plan is not None:
            _apply_tui_conversation_rewind(
                self.state.session_store,
                self.state.session,
                turn_store,
                target,
            )
            self._restore_main_messages_from_transcript()
            self._safe_render_active_tab()
        self.state.session_store.touch(self.state.session.session_id)
        self._refresh_file_tree()
        return (
            TuiMessage(
                kind="status",
                title="rewind applied",
                body=f"Rewind applied to {target.turn_id} ({mode}).",
            ),
            detail,
        )

    def _handle_compose_input(self, value: str) -> bool:
        clean = value.strip()
        if clean == "/compose":
            self._compose_mode = True
            self._compose_lines = []
            self._refresh_footer()
            self._show_detail("compose draft", "")
            self._write(
                TuiMessage(
                    kind="status",
                    title="compose",
                    body="Compose mode started. Enter lines, then /send. Use /cancel-compose to discard.",
                )
            )
            return True
        if not self._compose_mode:
            return False
        if clean == "/cancel-compose":
            count = len(self._compose_lines)
            self._compose_mode = False
            self._compose_lines = []
            self._refresh_footer()
            self._show_detail("compose draft", "")
            self._write(
                TuiMessage(
                    kind="status",
                    title="compose",
                    body=f"Discarded {count} draft line(s).",
                )
            )
            return True
        if clean == "/send":
            prompt = "\n".join(self._compose_lines).strip()
            self._compose_mode = False
            self._compose_lines = []
            self._refresh_footer()
            self._close_active_tab()
            if not prompt:
                self._write(
                    TuiMessage(
                        kind="status",
                        title="compose",
                        body="Empty draft was not sent.",
                    )
                )
                return True
            self._submit_prompt(prompt)
            return True
        self._compose_lines.append(value.rstrip())
        draft = "\n".join(self._compose_lines)
        self._show_detail("compose draft", draft)
        self._refresh_footer()
        self._write(
            TuiMessage(
                kind="status",
                title="compose",
                body=f"Added line {len(self._compose_lines)}. Use /send when ready.",
            )
        )
        return True

    def _handle_file_reference_command(self, value: str) -> bool:
        clean = value.strip()
        if not (clean == "/files" or clean.startswith("/files ") or clean.startswith("@")):
            return False
        query = ""
        if clean.startswith("/files "):
            query = clean[len("/files ") :].strip()
        elif clean.startswith("@"):
            query = clean[1:].strip()
        matches = workspace_file_matches(self.state.workspace, query, limit=12)
        if not matches:
            body = "No matching workspace files."
            if query:
                body += f" query={query}"
            self._write(TuiMessage(kind="status", title="files", body=body))
            return True
        lines = ["Matching workspace files"]
        for index, path in enumerate(matches, start=1):
            lines.append(f"{index}. @{path}")
        lines.append("")
        lines.append("Use the @path in your prompt, or /open <path> to open it.")
        self._show_detail("file references", "\n".join(lines))
        self._write(
            TuiMessage(
                kind="file",
                title="files",
                body=f"Found {len(matches)} matching file(s).",
                preview="\n".join(lines),
            )
        )
        return True

    def _maybe_start_next_queued_prompt(self) -> None:
        if (
            self._current_session_is_running()
            or self._pending_approval is not None
            or self._approval_queue
        ):
            return
        prompt = self._prompt_queue.pop_next()
        if prompt is None:
            self._refresh_footer()
            return
        self._refresh_footer()
        self._write(
            TuiMessage(
                kind="status",
                title="queue",
                body=f"Running queued prompt. {len(self._prompt_queue.pending)} remaining.",
            )
        )
        self._submit_prompt(prompt)

    def _maybe_start_task_continuation(self) -> bool:
        if (
            self._current_session_is_running()
            or self._pending_approval is not None
            or self._approval_queue
            or not self.state.task_continuations
        ):
            return False
        continuation = self.state.task_continuations[0]
        self.state.task_continuations = tuple(self.state.task_continuations[1:])
        self._write(
            TuiMessage(
                kind="task",
                title="task/execute",
                body="Continuing task/execute automatically.",
            )
        )
        self._submit_prompt(
            continuation,
            display_prompt="task/execute auto-continue",
            remember_prompt=False,
        )
        return True

    def _route_running_prompt(self, prompt: str) -> bool:
        clean = prompt.strip()
        if clean.startswith("/") and clean != "/followup" and not clean.startswith(("/followup ", "/queue ", "/task ")):
            if clean == "/approvals":
                self._show_approval_history()
                return True
            if (
                self._handle_preview_command(clean)
                or self._handle_find_command(clean)
                or self._handle_rewind_command(clean)
                or self._handle_immediate_command(clean)
            ):
                return True
            self._write(
                TuiMessage(
                    kind="warning",
                    title="command",
                    body=(
                        f"Unknown TUI command while a turn is running: {clean}. "
                        "Use /followup <text> to send text to the running turn, "
                        "or /queue <text> to run it after this turn."
                    ),
                )
            )
            return True
        if clean == "/followup":
            self._write(
                TuiMessage(
                    kind="status",
                    title="follow-up",
                    body="Usage: /followup <text>",
                )
            )
            return True
        forced = _strip_command_arg(clean, ("/followup ",))
        if forced is not None:
            self._submit_followup(forced, forced=True)
            return True
        queued = _strip_command_arg(clean, ("/queue ",))
        if queued is not None:
            self._enqueue_prompt(queued)
            return True
        active_turn = self._turn_for_current_session()
        if active_turn is not None and not active_turn.answer_visible:
            self._submit_followup(clean, forced=False)
            return True
        self._enqueue_prompt(
            clean,
            prefix="Deepmate is finishing the previous turn.",
        )
        return True

    def _submit_followup(self, text: str, *, forced: bool) -> None:
        clean = text.strip()
        if not clean:
            return
        active_turn = self._turn_for_current_session()
        if active_turn is not None and active_turn.answer_visible:
            self._enqueue_prompt(
                clean,
                prefix="The previous answer is already visible.",
            )
            return
        followup_buffer = (
            active_turn.followup_buffer
            if active_turn is not None
            else self.state.followup_buffer
        )
        followup_turn_id = (
            active_turn.followup_turn_id
            if active_turn is not None
            else self.state.active_followup_turn_id
        )
        submitted = (
            followup_buffer is not None
            and followup_buffer.submit(
                followup_turn_id,
                clean,
                source="forced" if forced else "input",
            )
        )
        if not submitted:
            self._enqueue_prompt(
                clean,
                prefix="Current turn cannot accept follow-ups now.",
            )
            return
        self._refresh_footer()
        self._write(
            TuiMessage(
                kind="followup",
                title="follow-up",
                body=clean,
            )
        )

    def _enqueue_prompt(
        self,
        prompt: str,
        *,
        kind: str = "status",
        title: str = "queued",
        prefix: str = "",
    ) -> bool:
        clean = prompt.strip()
        if not clean:
            self._write(TuiMessage(kind="status", title="queue", body="Empty prompt was not queued."))
            return False
        if not _prompt_length_ok(clean):
            self._write(_prompt_too_long_message(len(clean)))
            return False
        if self._prompt_queue.is_full():
            self._write(
                TuiMessage(
                    kind="warning",
                    title="queue full",
                    body=(
                        f"Prompt queue already has {len(self._prompt_queue.pending)} item(s). "
                        "Use /clear-queue or /resume-queue before adding more."
                    ),
                )
            )
            return False
        size = self._prompt_queue.enqueue(clean)
        self._refresh_footer()
        heading = f"Message queued for the next turn. Queue position: {size}."
        if prefix:
            heading = f"{prefix} {heading}"
        body = _pending_prompt_body(clean, heading=heading)
        self._write(TuiMessage(kind=kind, title=title, body=body))
        return True

    def _load_prompt_queue_for_current_session(self) -> None:
        path = self.state.prompt_queue_path()
        self._prompt_queue = (
            TuiPromptQueue.load(path) if path is not None else TuiPromptQueue()
        )
        self._refresh_footer()

    def _current_session_is_running(self) -> bool:
        if self._local_prepare_running:
            return True
        if self.state.session.session_id in self._session_running:
            return True
        return (
            self._running_turn
            and self._active_turn is None
            and not self._session_running
        )

    def _active_turn_is_current_session(self) -> bool:
        return self._current_session_is_running()

    def _turn_for_current_session(self) -> _TurnRun | None:
        return self._session_turns.get(self.state.session.session_id)

    def _other_session_running_label(self) -> str:
        running = [
            session_id
            for session_id in self._session_running
            if session_id != self.state.session.session_id
        ]
        if not running:
            return ""
        session_id = running[0]
        title = next(
            (
                known_title
                for known_id, known_title in self._session_tabs
                if known_id == session_id
            ),
            "",
        )
        label = _short_session_title(title or session_id[:8], active=False)
        return f"{label} running"

    def _stats_for_session(self, session_id: str) -> TuiRuntimeStats:
        return self._session_stats.setdefault(session_id, TuiRuntimeStats())

    def _state_for_worker_turn(self, session_id: str) -> TuiRuntimeState:
        # Worker state is a snapshot for the session that started the turn.
        # Session switches rebuild self.state.runtime/transcript/checkpoint
        # before new work starts; this assertion keeps that isolation contract
        # explicit until the remaining TUI state is moved into SessionUiState.
        if session_id != self.state.session.session_id:
            raise RuntimeError("worker turn state must be built while its session is active")
        approval_cache = self._approval_cache_for_session(session_id)
        native_tools = self._native_tools_for_approval_cache(approval_cache, session_id)
        state = replace(
            self.state,
            runtime_stats=self._stats_for_session(session_id),
            native_tools=native_tools,
            tool_access_policy=(
                replace(self.state.tool_access_policy)
                if self.state.tool_access_policy is not None
                else None
            ),
            approval_cache=approval_cache,
            approval_callbacks_installed=False,
            status_message_callback=(
                lambda message: self._write_from_worker_for_session(
                    session_id,
                    message,
                )
            ),
            live_status_callback=(
                lambda message: self._update_live_status_from_worker_for_session(
                    session_id,
                    message,
                )
            ),
            final_message_callback=(
                lambda messages: self._append_final_messages_from_worker_for_session(
                    session_id,
                    messages,
                )
            ),
            token_stream_callback=(
                lambda content, reasoning: self._stream_tokens_from_worker_for_session(
                    session_id,
                    content,
                    reasoning,
                )
            ),
            tool_approval_callback=(
                lambda tool, decision: self._tool_approval_for_session(
                    session_id,
                    tool,
                    decision,
                )
            ),
            safety_approval_callback=(
                lambda decision: self._safety_approval_for_session(
                    session_id,
                    decision,
                )
            ),
        )
        return state

    def _approval_cache_for_session(self, session_id: str) -> SessionApprovalCache | None:
        if self.state.approval_cache is None:
            return None
        key = session_id or self.state.session.session_id
        cache = self._approval_caches_by_session.get(key)
        if cache is None:
            cache = SessionApprovalCache()
            self._approval_caches_by_session[key] = cache
        cache.approval_callback = (
            lambda decision, session_key=key: self._safety_approval_for_session(
                session_key,
                decision,
            )
        )
        return cache

    def _native_tools_for_approval_cache(
        self,
        approval_cache: SessionApprovalCache | None,
        session_id: str = "",
    ) -> NativeToolRegistry | None:
        factory = self.state.native_tool_factory
        if factory is None:
            return self.state.native_tools
        return factory(approval_cache, session_id or self.state.session.session_id)

    def _route_checkpoint_writes_to_current_session(self) -> None:
        if self.state.checkpoint_write_router is not None:
            self.state.checkpoint_write_router.set_controller(
                self.state.checkpoint_controller
            )

    def _activate_session_record(self, session) -> None:
        self._save_content_tabs_for_session(self.state.session.session_id)
        self.state.session = session
        self.state.runtime_stats = self._stats_for_session(session.session_id)
        self.state.transcript = self.state.session_store.transcript_store(session)
        if self.state.checkpoint_controller_factory is not None:
            self.state.checkpoint_controller = self.state.checkpoint_controller_factory(session)
        activation = start_runtime_activation(
            session_id=session.session_id,
            workspace=session.workspace,
            profile=session.profile,
            context_snapshot=(
                self.state.context_snapshot_factory(session.profile)
                if self.state.context_snapshot_factory is not None
                else self.state.runtime.activation.context_snapshot
            ),
        )
        self.state.runtime = start_session_runtime(
            activation,
            conversation=runtime_conversation_from_store(
                self.state.session_store,
                session,
                self.state.transcript,
                turn_checkpoint_store=(
                    self.state.checkpoint_controller.turn_store
                    if self.state.checkpoint_controller is not None
                    else None
                ),
            ),
            behavior_runtime=(
                self.state.behavior_runtime.with_profile(
                    workspace=session.workspace,
                    profile=session.profile,
                    session_id=session.session_id,
                )
                if self.state.behavior_runtime is not None
                else None
            ),
        )
        self.state.behavior_runtime = self.state.runtime.behavior_runtime
        refresh_computer_tool_surface(self.state)
        self.state.turn_index = 0
        active_turn = (
            self._session_turns.get(session.session_id)
        )
        self._active_turn = active_turn
        self._tool_turn_approvals = self._tool_turn_approvals_by_session.setdefault(
            session.session_id,
            set(),
        )
        if active_turn is not None:
            self.state.followup_buffer = active_turn.followup_buffer
            self.state.active_followup_turn_id = active_turn.followup_turn_id
            self.state.cancellation_token = active_turn.cancellation_token
        else:
            self.state.followup_buffer = TurnFollowupBuffer()
            self.state.active_followup_turn_id = None
            self.state.cancellation_token = None
        self.state.unconsumed_followups = ()
        self.state.status_message_callback = self._write_from_worker
        self.state.live_status_callback = self._update_live_status_from_worker
        self.state.tool_approval_callback = self._tool_approval
        self.state.safety_approval_callback = self._safety_approval
        self.state.approval_callbacks_installed = False
        self._bind_current_session_approval_cache()
        if not self._running_turn:
            self._route_checkpoint_writes_to_current_session()
        self._apply_trust_to_current_session()
        self._load_prompt_queue_for_current_session()
        self._remember_session_tab(session.session_id, session.title)
        self._live_status_text = ""
        self._live_message_active = False
        self._restore_content_tabs_for_session(session.session_id)
        self.query_one("#chat", RichLog).clear()
        # _main_messages now resolves to this session's own buffer. Only rebuild
        # from the transcript on first visit; otherwise keep the in-memory history
        # (which includes errors/warnings/status that never reach the transcript).
        if session.session_id not in self._session_messages:
            self._restore_main_messages_from_transcript()
        if session.session_id in self._session_running:
            self._live_status_text = self._session_live_status.get(
                session.session_id,
                "Working on this session...",
            )
            self._upsert_live_message()
        self._session_has_updates.discard(session.session_id)
        self._promote_current_session_approval()
        self._render_approval_panel()
        self._render_active_tab()
        if not self._main_messages:
            self._write_start_message()
        self._refresh_file_tree()
        self._refresh_footer()

    def _create_new_session(self) -> None:
        # Leave the title unset ("Untitled session") so the first prompt renames
        # it via _ensure_session_title, exactly like the first session. Passing a
        # concrete title here would suppress that auto-naming.
        session = self.state.session_store.create(
            workspace=self.state.workspace,
            profile=self.state.profile,
        )
        self._expanded_dirs.clear()
        self._activate_session_record(session)

    def _remember_session_tab(self, session_id: str, title: str) -> None:
        clean_id = session_id.strip()
        if not clean_id:
            return
        clean_title = title.strip() or "Untitled session"
        for index, (known_id, _) in enumerate(self._session_tabs):
            if known_id == clean_id:
                self._session_tabs[index] = (clean_id, clean_title)
                return
        self._session_tabs.append((clean_id, clean_title))
        if len(self._session_tabs) > 20:
            active_id = self.state.session.session_id
            recent = self._session_tabs[-20:]
            if active_id and all(session_id != active_id for session_id, _ in recent):
                active_tab = next(
                    (
                        (session_id, known_title)
                        for session_id, known_title in self._session_tabs
                        if session_id == active_id
                    ),
                    None,
                )
                if active_tab is not None:
                    recent = [*recent[-19:], active_tab]
            self._session_tabs = recent

    def _switch_session(self, session_id: str) -> None:
        try:
            session = self.state.session_store.load(session_id)
        except (OSError, ValueError) as exc:
            self._write(TuiMessage(kind="error", title="session", body=str(exc)))
            return
        if session.workspace.resolve() != self.state.workspace.resolve():
            self._request_workspace_switch(session.workspace, session.session_id)
            return
        self._activate_session_record(session)

    def _restore_main_messages_from_transcript(self) -> None:
        messages = _messages_from_transcript_items(
            self.state.transcript.load_items(),
            limit=SESSION_RESTORE_MESSAGE_LIMIT,
        )
        self._main_messages = list(messages)

    def _update_command_hints(self, value: str) -> None:
        panel = self.query_one("#command-hints", Static)
        clean = value.strip()
        if not clean.startswith("/"):
            panel.display = False
            panel.update("")
            self._command_hint_matches = ()
            self._command_hint_index = 0
            return
        matches = tuple(
            hint for hint in _COMMAND_HINTS if hint[0].startswith(clean) or clean == "/"
        )[:8]
        if not matches:
            panel.display = False
            panel.update("")
            self._command_hint_matches = ()
            self._command_hint_index = 0
            return
        previous = self._selected_command_hint()
        self._command_hint_matches = matches
        if previous:
            self._command_hint_index = next(
                (
                    index
                    for index, (command, _) in enumerate(matches)
                    if command == previous
                ),
                0,
            )
        else:
            self._command_hint_index = 0
        self._render_command_hints()

    def _hide_command_hints(self) -> None:
        self._command_hint_matches = ()
        self._command_hint_index = 0
        try:
            panel = self.query_one("#command-hints", Static)
        except NoMatches:
            return
        panel.display = False
        panel.update("")

    def _render_command_hints(self) -> None:
        panel = self.query_one("#command-hints", Static)
        if not self._command_hint_matches:
            panel.display = False
            panel.update("")
            return
        lines = []
        for index, (command, description) in enumerate(self._command_hint_matches):
            selected = index == self._command_hint_index
            command_style = "bold #7fd6df" if selected else "#cad7dc"
            description_style = "#93aab3" if selected else "#687d86"
            prefix = "› " if selected else "  "
            lines.append(
                f"{prefix}[{command_style}]{escape(command)}[/]  "
                f"[{description_style}]{escape(description)}[/]"
            )
        panel.update("\n".join(lines))
        panel.display = True

    def _selected_command_hint(self) -> str:
        if not self._command_hint_matches:
            return ""
        index = max(0, min(self._command_hint_index, len(self._command_hint_matches) - 1))
        return self._command_hint_matches[index][0]

    def _complete_selected_command(self) -> None:
        selected = self._selected_command_hint()
        if not selected:
            return
        input_widget = self.query_one("#prompt-input", TextArea)
        needs_space = selected not in {
            "/commands",
            "/help",
            "/?",
            "/status",
            "/task",
            "/diff",
            "/find",
            "/compose",
            "/send",
            "/cancel-compose",
            "/restore-draft",
            "/queue",
            "/resume-queue",
            "/clear-queue",
            "/approvals",
            "/clear",
            "/approvals",
            "/session",
            "/session tree",
            "/tree",
            "/remote",
            "/sessions",
            "/skills",
            "/mcp",
            "/exit",
            "/quit",
        }
        value = selected + (" " if needs_space else "")
        _set_prompt_text(input_widget, value)
        self._update_command_hints(value)

    def _queue_status_body(self) -> str:
        if not self._prompt_queue.pending:
            return "No queued prompts."
        lines = [
            f"status: {'paused' if self._prompt_queue.paused else 'active'}",
            f"pending: {len(self._prompt_queue.pending)}",
        ]
        if self._last_queue_pause_reason and self._prompt_queue.paused:
            lines.append(f"reason: {self._last_queue_pause_reason}")
        for index, prompt in enumerate(self._prompt_queue.pending[:5], start=1):
            lines.append(f"{index}. {_single_line(prompt, limit=120)}")
        if len(self._prompt_queue.pending) > 5:
            lines.append(f"... +{len(self._prompt_queue.pending) - 5} more")
        return "\n".join(lines)

    def _show_approval_history(self) -> None:
        lines = self._approval_result_lines_by_session.get(
            self.state.session.session_id,
            [],
        )
        body = (
            "\n".join(lines)
            if lines
            else "No approvals have been resolved in this session."
        )
        self._show_detail("approvals", body)
        self._write(
            TuiMessage(
                kind="status",
                title="/approvals",
                body="Approval history opened in a content tab.",
            )
        )

    def _pause_prompt_queue(self, reason: str) -> None:
        self._last_queue_pause_reason = reason.strip()
        self._prompt_queue.pause()

    def _tool_approval(
        self,
        tool: NativeTool,
        decision: ToolAccessDecision,
    ) -> bool:
        return self._tool_approval_for_session(
            self.state.session.session_id,
            tool,
            decision,
        )

    def _tool_approval_for_session(
        self,
        session_id: str,
        tool: NativeTool,
        decision: ToolAccessDecision,
    ) -> bool:
        approval_key = _tool_approval_key(tool, decision)
        turn_approvals = self._tool_turn_approvals_for_session(session_id)
        if approval_key in turn_approvals:
            return True
        session_approvals = self._tool_session_approvals_for_session(session_id)
        if approval_key in session_approvals:
            return True
        body = _approval_body(
            decision.reason,
            decision.refs,
            fallback=f"tool={tool.name}",
        )
        result = self._request_approval(
            "Tool approval",
            body,
            subtitle=tool.name,
            subject=f"tool {tool.name}",
            refs=decision.refs,
            session_id=session_id,
        )
        if result == "session":
            session_approvals.add(approval_key)
            return True
        if result == "once":
            turn_approvals.add(approval_key)
            return True
        return False

    def _tool_session_approvals_for_session(self, session_id: str) -> set[str]:
        key = session_id or self.state.session.session_id
        return self._tool_session_approvals.setdefault(key, set())

    def _tool_turn_approvals_for_session(self, session_id: str) -> set[str]:
        if session_id:
            if session_id == self.state.session.session_id:
                return self._tool_turn_approvals
            return self._tool_turn_approvals_by_session.setdefault(session_id, set())
        return self._tool_turn_approvals

    def _safety_approval(self, decision: SafetyDecision) -> ApprovalDecision:
        return self._safety_approval_for_session(
            self.state.session.session_id,
            decision,
        )

    def _safety_approval_for_session(
        self,
        session_id: str,
        decision: SafetyDecision,
    ) -> ApprovalDecision:
        body = _approval_body(
            decision.reason,
            decision.refs,
            fallback=decision.approval_key or decision.risk_level.value,
        )
        return approval_decision_from_text(
            self._request_approval(
                "Safety approval",
                body,
                subtitle=decision.risk_level.value,
                subject=_safety_approval_subject(decision),
                refs=_safety_approval_refs(decision),
                session_id=session_id,
            )
        )

    def _request_approval(
        self,
        title: str,
        body: str,
        *,
        subtitle: str = "",
        subject: str = "",
        refs: tuple[str, ...] = (),
        session_id: str = "",
    ) -> str:
        pending = _PendingApproval(
            title=title,
            body=body,
            event=Event(),
            subtitle=subtitle,
            subject=subject,
            refs=refs,
            session_id=session_id,
        )
        if not self._safe_call_from_thread(self._show_pending_approval, pending):
            pending.result = "deny"
            pending.event.set()
            return pending.result
        if not pending.event.wait(APPROVAL_WAIT_TIMEOUT_SECONDS):
            pending.result = "deny"
            self._safe_call_from_thread(self._expire_pending_approval, pending)
        return pending.result

    def _expire_pending_approval(self, pending: _PendingApproval) -> None:
        if self._pending_approval is pending:
            self._pending_approval = None
            self._show_next_pending_approval()
        else:
            self._approval_queue = [
                queued for queued in self._approval_queue if queued is not pending
            ]
        self._write(
            TuiMessage(
                kind="warning",
                title="approval",
                body="Approval request timed out and was denied.",
            )
        )
        self._render_approval_panel()
        self._refresh_footer()

    def _show_pending_approval(self, pending: _PendingApproval) -> None:
        if self._pending_approval is not None:
            self._approval_queue.append(pending)
            self._refresh_footer()
            return
        self._pending_approval = pending
        self._render_approval_panel()
        self._refresh_footer()

    def _show_next_pending_approval(self) -> None:
        if self._pending_approval is not None:
            return
        current_session_id = self.state.session.session_id
        for index, queued in enumerate(self._approval_queue):
            pending_session_id = getattr(queued, "session_id", "")
            if not pending_session_id or pending_session_id == current_session_id:
                self._pending_approval = self._approval_queue.pop(index)
                break
        self._render_approval_panel()
        self._refresh_footer()

    def _promote_current_session_approval(self) -> None:
        pending = self._pending_approval
        if pending is None or self._pending_approval_is_current_session():
            return
        current_session_id = self.state.session.session_id
        for index, queued in enumerate(self._approval_queue):
            queued_session_id = getattr(queued, "session_id", "")
            if queued_session_id and queued_session_id != current_session_id:
                continue
            self._approval_queue[index] = pending
            self._pending_approval = queued
            return

    def _handle_approval_input(self, value: str) -> bool:
        pending = self._pending_approval
        if pending is None:
            return False
        result = _approval_input_result(value)
        if result is None:
            return False
        return self._resolve_pending_approval(result)

    def _pending_approval_is_current_session(self) -> bool:
        """Whether the head approval was raised by the currently-viewed session.

        Typed allow/deny must only resolve approvals belonging to the session the
        user is looking at, so a decision typed while viewing one session can't
        silently approve another session's tool/shell request.
        """
        pending = self._pending_approval
        if pending is None:
            return False
        session_id = getattr(pending, "session_id", "")
        if not session_id:
            return True
        return session_id == self.state.session.session_id

    def _resolve_pending_approval(self, result: str) -> bool:
        pending = self._pending_approval
        if pending is None:
            return False
        clean = result.strip().lower()
        if clean not in {"once", "session", "deny"}:
            return False
        if not self._pending_approval_is_current_session():
            self._write(
                TuiMessage(
                    kind="warning",
                    title="approval",
                    body=(
                        "This approval belongs to another session. Switch to that "
                        "session to allow or deny it."
                    ),
                )
            )
            return True
        self._pending_approval = None
        pending.result = clean
        pending.event.set()
        if pending.result == "deny":
            self._pause_prompt_queue("approval denied")
        self._show_next_pending_approval()
        try:
            self.query_one("#prompt-input", TextArea).focus()
        except (NoMatches, ScreenStackError):
            pass
        self._record_approval_result(pending)
        return True

    def _record_approval_result(self, pending: _PendingApproval) -> None:
        session_id = pending.session_id or self.state.session.session_id
        lines = self._approval_result_lines_by_session.setdefault(session_id, [])
        line = _approval_result_line(pending)
        if line and line not in lines:
            lines.append(line)
        if len(lines) > 8:
            del lines[: len(lines) - 8]
        body = line or _approval_result_body(pending)
        self._last_approval_result_by_session[session_id] = body
        message = TuiMessage(
            kind="permissions",
            title="permissions",
            body=body,
        )
        if session_id == self.state.session.session_id:
            self._write(message)
        else:
            self._append_background_message(session_id, message)

    def _deny_pending_approval(self) -> None:
        pending_items: list[_PendingApproval] = []
        if self._pending_approval is not None:
            pending_items.append(self._pending_approval)
        pending_items.extend(self._approval_queue)
        if not pending_items:
            return
        self._pending_approval = None
        self._approval_queue.clear()
        for pending in pending_items:
            pending.result = "deny"
            pending.event.set()
        self._pause_prompt_queue("approval denied")
        self._render_approval_panel()
        self._refresh_footer()

    def _render_approval_panel(self) -> None:
        try:
            panel = self.query_one("#approval-panel", Vertical)
            title = self.query_one("#approval-title", Static)
            body = self.query_one("#approval-body", Static)
        except NoMatches:
            return
        pending = self._pending_approval
        if pending is None or not self._pending_approval_is_current_session():
            panel.display = False
            title.update("")
            body.update("")
            return
        panel.display = True
        label = pending.title
        if pending.subtitle:
            label = f"{label} · {pending.subtitle}"
        session_queue_count = self._approval_queue_count_for_current_session()
        if session_queue_count:
            label = f"{label} · {session_queue_count} queued"
        if (
            pending.session_id
            and pending.session_id != self.state.session.session_id
        ):
            label = f"{label} · from session {pending.session_id[:8]}"
        title.update(label)
        diff = _approval_diff_renderable(_approval_ref_map(pending.refs))
        if diff is None:
            body.update(escape(pending.body))
        else:
            body.update(Group(Text.from_markup(escape(pending.body)), Text(""), diff))

    def _approval_queue_count_for_current_session(self) -> int:
        current_session_id = self.state.session.session_id
        return sum(
            1
            for pending in self._approval_queue
            if not pending.session_id or pending.session_id == current_session_id
        )


def _approval_body(
    reason: str,
    refs: tuple[str, ...],
    *,
    fallback: str,
) -> str:
    ref_values = _approval_ref_map(refs)
    tool = ref_values.get("tool") or fallback.removeprefix("tool=").strip()
    path = ref_values.get("path") or ref_values.get("target") or ref_values.get("source")
    action = _approval_action_summary(reason, tool)
    lines = [action]
    if path:
        lines.append(f"Location: {path}")
    elif tool:
        lines.append(f"Action type: {_approval_tool_label(tool)}")
    if ref_values.get("command"):
        lines.append(f"Command: {_clip_middle(ref_values['command'], 140)}")
    size = (
        ref_values.get("content_chars")
        or ref_values.get("new_text_chars")
        or ref_values.get("old_text_chars")
    )
    if size:
        lines.append(f"Size: {size} chars")
    risk = ref_values.get("risk", "").strip()
    if risk:
        lines.append(f"Risk: {risk.replace('_', ' ')}")
    approval_key = ref_values.get("approval_key", "").strip()
    if approval_key:
        lines.append(f"Scope: {approval_key}")
    preview_lines = _approval_content_preview_lines(ref_values)
    if preview_lines:
        lines.extend(preview_lines)
    if not path and not tool and fallback.strip():
        lines.append(f"Scope: {fallback.strip()}")
    if not lines:
        lines = [reason.strip() or "This action needs your confirmation."]
    if tool == "run_shell_command" or fallback in {
        "capability:shell",
        "capability:shell-network",
        "capability:network",
        "capability:env_change",
        "capability:environment",
    }:
        lines.append(
            "Command permissions are confirmed per risk level; "
            "network or environment changes may prompt again."
        )
    lines.append("Allow once: applies to this request only.")
    lines.append("Allow for session: remembered until Deepmate exits.")
    return "\n".join(lines)


def _approval_content_preview_lines(ref_values: dict[str, str]) -> list[str]:
    """Return a compact write/edit summary for the approval panel."""
    old_text = ref_values.get("old_text_preview", "")
    new_text = ref_values.get("new_text_preview", "")
    content = ref_values.get("content_preview", "")

    if old_text or new_text:
        stat = _approval_diff_stat(old_text, new_text)
        lines = [f"Change summary: {stat}" if stat else "Change summary: text edit"]
        if _looks_like_markup(old_text) or _looks_like_markup(new_text):
            lines.append("Preview hidden: view HTML/XML content in the file.")
        else:
            snippet = _approval_text_snippet(new_text or old_text)
            if snippet:
                lines.append(f"Snippet: {snippet}")
        return lines
    if content:
        lines = ["Content summary: new file content"]
        if _looks_like_markup(content):
            lines.append("Preview hidden: open the file to view HTML/XML content.")
        else:
            snippet = _approval_text_snippet(content)
            if snippet:
                lines.append(f"Snippet: {snippet}")
        return lines
    return []


def _approval_diff_stat(old_text: str, new_text: str) -> str:
    import difflib

    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    added = 0
    removed = 0
    for line in difflib.unified_diff(old_lines, new_lines, lineterm="", n=2):
        if line.startswith(("---", "+++")):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    if not added and not removed:
        return ""
    return f"+{added} -{removed}"


# Cap on rendered diff rows so a large edit can't flood the approval panel; the
# preview text is already truncated upstream (content_chars), so this is a
# secondary guard on line count.
APPROVAL_DIFF_MAX_ROWS = 20


def _approval_diff_renderable(ref_values: dict[str, str]) -> Text | None:
    """Build a line-by-line colored +/- diff from write/edit preview refs.

    Returns None when there is nothing useful to show (no edit, identical text,
    or markup content we deliberately hide). The previews are truncated upstream,
    so the diff is preview-level and is annotated as such when clipped.
    """
    old_text = ref_values.get("old_text_preview", "")
    new_text = ref_values.get("new_text_preview", "")
    if not old_text and not new_text:
        return None
    if _looks_like_markup(old_text) or _looks_like_markup(new_text):
        return None

    import difflib

    diff = Text(overflow="fold")
    rows = 0
    clipped = False
    for line in difflib.unified_diff(
        old_text.splitlines(), new_text.splitlines(), lineterm="", n=1
    ):
        if line.startswith(("---", "+++", "@@")):
            continue
        if rows >= APPROVAL_DIFF_MAX_ROWS:
            clipped = True
            break
        if line.startswith("+"):
            diff.append(line + "\n", style="green")
        elif line.startswith("-"):
            diff.append(line + "\n", style="red")
        else:
            diff.append(line + "\n", style="#687d86")
        rows += 1
    if rows == 0:
        return None
    if clipped:
        diff.append("… preview truncated; see the file for the full change.", style="#687d86 italic")
    return diff


def _approval_text_snippet(value: str) -> str:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if not lines:
        return ""
    snippet = " ".join(lines[:APPROVAL_PREVIEW_MAX_LINES])
    return _clip_middle(snippet, 180)


def _looks_like_markup(value: str) -> bool:
    clean = value.lower()
    return any(marker in clean for marker in ("<html", "<body", "<div", "<table", "</"))


def _approval_ref_map(refs: tuple[str, ...]) -> dict[str, str]:
    values: dict[str, str] = {}
    for ref in refs:
        if not isinstance(ref, str) or "=" not in ref:
            continue
        key, _, value = ref.partition("=")
        values[key.strip()] = value.strip()
    return values


def _tool_approval_key(tool: NativeTool, decision: ToolAccessDecision) -> str:
    ref_key = _approval_ref_map(decision.refs).get("approval_key", "").strip()
    if ref_key:
        return ref_key
    clean_tool = tool.name.strip()
    if clean_tool == "run_shell_command":
        return "capability:tool-shell"
    if clean_tool.startswith("computer_"):
        return f"capability:{clean_tool}"
    if clean_tool in {"write_text_file", "edit_text_file"}:
        return f"capability:{clean_tool}"
    if clean_tool.startswith("browser_") or clean_tool == "load_browser_tools":
        return "capability:browser"
    if clean_tool.startswith("mcp.") or clean_tool in {"search_mcp_tools", "load_mcp_tool"}:
        return "capability:mcp"
    return clean_tool


def _approval_action_summary(reason: str, tool: str) -> str:
    clean_reason = " ".join(reason.strip().split())
    clean_tool = tool.strip()
    if clean_tool == "write_text_file":
        return "Deepmate wants to create or overwrite a file."
    if clean_tool == "edit_text_file":
        return "Deepmate wants to modify a file."
    if clean_tool == "run_shell_command":
        return "Deepmate wants to run a command."
    if clean_tool.startswith("browser_") or clean_tool == "load_browser_tools":
        return "Deepmate wants to control the browser."
    if clean_tool.startswith("computer_"):
        return "Deepmate wants to control the computer."
    if clean_tool:
        return f"Deepmate wants to run: {_approval_tool_label(clean_tool)}."
    return clean_reason or "This action needs your confirmation."


def _approval_tool_label(tool: str) -> str:
    clean = tool.strip()
    if clean == "write_text_file":
        return "write file"
    if clean == "edit_text_file":
        return "edit file"
    if clean == "run_shell_command":
        return "run command"
    if clean.startswith("browser_") or clean == "load_browser_tools":
        return "browser action"
    if clean.startswith("computer_"):
        return "computer action"
    if clean.startswith("mcp.") or clean in {"search_mcp_tools", "load_mcp_tool"}:
        return "external connection"
    if clean.startswith("skill") or "skill" in clean:
        return "skill install or load"
    return clean.replace("_", " ")


def _clip_middle(value: str, limit: int) -> str:
    clean = " ".join(value.split())
    if len(clean) <= limit:
        return clean
    keep = max(0, (limit - 3) // 2)
    return clean[:keep].rstrip() + "..." + clean[-keep:].lstrip()


def _approval_result_label(value: str) -> str:
    return {
        "once": "allowed for this request",
        "session": "allowed for this session",
        "deny": "denied",
    }.get(value, value)


def _approval_action_from_widget_id(widget_id: str) -> str | None:
    return {
        "approval-once": "once",
        "approval-session": "session",
        "approval-deny": "deny",
    }.get(widget_id.strip())


def _approval_result_body(pending: _PendingApproval) -> str:
    subject = (
        pending.subject.strip()
        or pending.subtitle.strip()
        or pending.title.strip()
        or "this action"
    )
    label = _approval_result_label(pending.result)
    if pending.result == "deny":
        return f"Permission {label} for {subject}."
    return f"Permission {label} for {subject}. Deepmate will continue automatically."


def _approval_result_line(pending: _PendingApproval) -> str:
    subject = _approval_result_subject(pending)
    if pending.result == "deny":
        return f"Denied: {subject}"
    scope = "session" if pending.result == "session" else "once"
    return f"Allowed ({scope}): {subject}"


def _is_permission_summary_message(message: TuiMessage) -> bool:
    return message.kind == "permissions" or message.status == "approval summary"


def _approval_result_subject(pending: _PendingApproval) -> str:
    refs = _approval_ref_map(pending.refs)
    command = refs.get("command", "").strip()
    if command:
        return f"shell command `{_clip_middle(command, 96)}`"
    approval_key = refs.get("approval_key", "").strip()
    if approval_key:
        return _approval_key_subject(approval_key)
    subject = (
        pending.subject.strip()
        or pending.subtitle.strip()
        or pending.title.strip()
        or "this action"
    )
    return subject


def _approval_key_subject(key: str) -> str:
    if key == "capability:shell":
        return "shell execution"
    if key == "capability:shell-medium":
        return "shell command outside the safe read-only set"
    if key == "capability:shell-high":
        return "higher-risk shell command"
    if key in {"capability:shell-network", "capability:network"}:
        return "shell network access"
    if key in {"capability:env_change", "capability:environment"}:
        return "environment changes"
    if key.startswith("shell:"):
        return "this shell command"
    return key


def _safety_approval_subject(decision: SafetyDecision) -> str:
    key = decision.approval_key.strip()
    if key == "capability:shell":
        return "shell execution"
    if key == "capability:shell-medium":
        return "shell command outside the safe read-only set"
    if key == "capability:shell-high":
        return "higher-risk shell command"
    if key in {"capability:shell-network", "capability:network"}:
        return "shell network access"
    if key in {"capability:env_change", "capability:environment"}:
        return "environment changes"
    if key.startswith("shell:"):
        return "this shell command"
    if key:
        return key
    reason = decision.reason.strip()
    if reason:
        return reason
    return f"{decision.risk_level.value} risk action"


def _safety_approval_refs(decision: SafetyDecision) -> tuple[str, ...]:
    key = decision.approval_key.strip()
    refs = tuple(decision.refs)
    if key and not any(ref.startswith("approval_key=") for ref in refs):
        return (f"approval_key={key}", *refs)
    return refs


def _approval_input_result(value: str) -> str | None:
    """Map an input line to an approval decision, or None if it is not a decision.

    Only explicit signals count. Earlier versions matched loose substrings
    (e.g. any line containing "继续"/"proceed"), which silently approved ordinary
    chat like "我们继续聊别的" — both an approval-bypass risk and a confusing UX.
    Short natural-language grants that explicitly mention permission/approval
    are accepted; anything else falls through so the caller can prompt the user
    to choose explicitly.
    """
    clean = " ".join(value.strip().lower().split())
    if not clean:
        return None
    normalized = clean.replace("-", "_").replace(" ", "_")
    if normalized in {"/deny", "deny", "denied", "no", "reject", "拒绝", "不允许"}:
        return "deny"
    if normalized in {
        "/approve_session",
        "/approve-session",
        "approve_session",
        "always_allow",
        "allow_session",
        "allow_for_session",
        "session",
        "本次都允许",
        "总是允许",
        "一直允许",
    }:
        return "session"
    if normalized in {
        "/approve",
        "/approve_once",
        "approve",
        "approve_once",
        "allow",
        "once",
        "yes",
        "ok",
        "允许",
        "批准",
        "同意",
    }:
        return "once"
    if _looks_like_explicit_approval_grant(clean):
        return "once"
    return None


def _looks_like_explicit_approval_grant(clean: str) -> bool:
    if not clean or len(clean) > 80:
        return False
    deny_markers = (
        "不允许",
        "拒绝",
        "不要",
        "别",
        "deny",
        "reject",
        "not allowed",
    )
    if any(marker in clean for marker in deny_markers):
        return False
    grant_markers = (
        "给你权限",
        "给你写入权限",
        "给权限",
        "授权",
        "批准",
        "允许",
        "同意",
        "approve",
        "permission",
    )
    return any(marker in clean for marker in grant_markers)


def _pending_prompt_body(prompt: str, *, heading: str) -> str:
    clean = prompt.strip()
    preview = _preview_text(clean, limit=900)
    lines = [heading.strip()]
    if preview:
        lines.append(f"↳ {preview}")
    return "\n".join(lines)


def _turn_anchor_id(session_id: str, started_at: float | None) -> str:
    stamp = "0" if started_at is None else f"{started_at:.9f}"
    return f"{session_id}:{stamp}"


def _turn_result_insert_index(messages: list[TuiMessage], anchor_id: str) -> int:
    if not anchor_id:
        return _before_trailing_live_message(messages)
    for index, message in enumerate(messages):
        if _message_has_turn_anchor(message, anchor_id):
            return _after_anchor_turn_messages(messages, index + 1)
    return _before_trailing_live_message(messages)


def _after_anchor_turn_messages(messages: list[TuiMessage], index: int) -> int:
    while index < len(messages):
        message = messages[index]
        if message.kind == "user":
            break
        if message.status == "live":
            break
        index += 1
    return index


def _before_trailing_live_message(messages: list[TuiMessage]) -> int:
    if messages and messages[-1].status == "live":
        return len(messages) - 1
    return len(messages)


def _message_has_turn_anchor(message: TuiMessage, anchor_id: str) -> bool:
    expected = f"turn_anchor={anchor_id}"
    return expected in message.refs


def _messages_from_transcript_items(
    items: Iterable[object],
    *,
    limit: int,
) -> tuple[TuiMessage, ...]:
    messages: list[TuiMessage] = []
    for item in items:
        message = getattr(item, "message", None)
        if message is not None:
            role = getattr(message, "role", None)
            content = str(getattr(message, "content", "")).strip()
            if not content or role == MessageRole.SYSTEM:
                continue
            if role == MessageRole.USER:
                messages.append(TuiMessage(kind="user", title="you", body=content))
            elif role == MessageRole.ASSISTANT:
                messages.append(TuiMessage(kind="assistant", title="assistant", body=content))
            else:
                messages.append(TuiMessage(kind="status", title=str(role), body=content))
            continue
        exchange = getattr(item, "tool_exchange", None)
        if exchange is not None:
            messages.extend(tool_exchange_messages(exchange))
    if limit > 0 and len(messages) > limit:
        omitted = len(messages) - limit
        return (
            TuiMessage(
                kind="status",
                title="session history",
                body=f"Showing the latest {limit} restored message(s); {omitted} older item(s) omitted.",
            ),
            *messages[-limit:],
        )
    return tuple(messages)


def _copy_to_clipboard(text: str) -> None:
    commands = (
        ("pbcopy",),
        ("wl-copy",),
        ("xclip", "-selection", "clipboard"),
        ("xsel", "--clipboard", "--input"),
    )
    for command in commands:
        try:
            subprocess.run(
                command,
                input=text,
                text=True,
                check=True,
                timeout=3,
            )
            return
        except FileNotFoundError:
            continue
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise OSError(str(exc)) from exc
    raise OSError("clipboard command is not available on this platform")


def _read_clipboard() -> str:
    commands = (
        ("pbpaste",),
        ("wl-paste", "--no-newline"),
        ("xclip", "-selection", "clipboard", "-o"),
        ("xsel", "--clipboard", "--output"),
    )
    for command in commands:
        try:
            result = subprocess.run(
                command,
                text=True,
                check=True,
                capture_output=True,
                timeout=3,
            )
        except FileNotFoundError:
            continue
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise OSError(str(exc)) from exc
        return result.stdout
    raise OSError("clipboard command is not available on this platform")


def _insert_prompt_text(input_widget: object, text: str, *, max_chars: int) -> bool:
    current = _prompt_text(input_widget)
    pasted = _clean_input_text(text)
    selection = getattr(input_widget, "selection", None)
    start = end = int(getattr(input_widget, "cursor_position", len(current)) or 0)
    if selection is not None and not getattr(selection, "is_empty", True):
        start = int(getattr(selection, "start", start))
        end = int(getattr(selection, "end", end))
    start, end = sorted((max(0, start), min(len(current), end)))
    remaining = max_chars - (len(current) - (end - start))
    if remaining <= 0:
        return False
    inserted = pasted[:remaining]
    if not inserted:
        return False
    _set_prompt_text(input_widget, f"{current[:start]}{inserted}{current[end:]}")
    try:
        input_widget.cursor_position = start + len(inserted)
    except AttributeError:
        pass
    return True


def _prompt_text(input_widget: object) -> str:
    if isinstance(input_widget, TextArea):
        return input_widget.text
    return str(getattr(input_widget, "value", "") or "")


def _set_prompt_text(input_widget: object, text: str) -> None:
    clean = str(text)
    if isinstance(input_widget, TextArea):
        input_widget.load_text(clean)
        input_widget.move_cursor(_text_area_end_location(clean))
        return
    setattr(input_widget, "value", clean)
    try:
        setattr(input_widget, "cursor_position", len(clean))
    except AttributeError:
        pass


def _text_area_end_location(text: str) -> tuple[int, int]:
    lines = str(text).splitlines()
    if not lines:
        return (0, 0)
    if str(text).endswith("\n"):
        return (len(lines), 0)
    return (len(lines) - 1, len(lines[-1]))


def _text_area_offset(text: str, location: tuple[int, int]) -> int:
    row, column = location
    lines = str(text).splitlines(keepends=True)
    if not lines:
        return 0
    row = max(0, min(int(row), len(lines) - 1))
    offset = sum(len(line) for line in lines[:row])
    return offset + max(0, min(int(column), len(lines[row].rstrip("\n\r"))))


def _text_area_location_for_offset(text: str, offset: int) -> tuple[int, int]:
    remaining = max(0, int(offset))
    lines = str(text).splitlines(keepends=True)
    if not lines:
        return (0, 0)
    for row, line in enumerate(lines):
        line_length = len(line)
        if remaining <= line_length:
            return (row, min(remaining, len(line.rstrip("\n\r"))))
        remaining -= line_length
    return _text_area_end_location(text)


def _paste_dedupe_key(text: str) -> str:
    return _clean_input_text(text)


def _pet_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    src_root = str(Path(__file__).resolve().parents[3])
    existing = env.get("PYTHONPATH", "")
    paths = [path for path in existing.split(os.pathsep) if path]
    if src_root not in paths:
        env["PYTHONPATH"] = os.pathsep.join([src_root, *paths])
    return env


def _pet_start_command(data_dir: Path | None) -> list[str] | None:
    if data_dir is None:
        return None
    return [sys.executable, "-m", "deepmate", "--pet"]


def _pet_frontend_missing_message() -> str:
    return (
        "Desktop pet needs an Electron runtime before it can open. "
        "Normal Deepmate work continues without it. Run `/pet setup` for the "
        "one-time setup options, or set DEEPMATE_PET_ELECTRON to an existing "
        "Electron binary, then retry `/pet on`."
    )


def _file_nav_label(
    relative_path: str,
    badge: str,
    is_dir: bool,
    *,
    expanded: bool = False,
) -> Text:
    clean = relative_path.strip()
    depth = clean.strip("/").count("/")
    indent = "  " * depth
    name = clean.rstrip("/").rsplit("/", 1)[-1]
    label = Text()
    if indent:
        label.append(indent)
    if is_dir:
        marker = "▾" if expanded else "▸"
        label.append(f"{marker} {name}/")
        return label
    icon = _file_icon(clean)
    label.append(f"{icon} {name}")
    clean_badge = badge.strip()
    if clean_badge:
        label.append(f" {clean_badge}", style=_file_badge_style(clean_badge))
    return label


def _file_badge_style(badge: str) -> str:
    if badge == "N":
        return "#cdbb7a bold"
    if badge == "M":
        return "#8dbd9a bold"
    return "#8aa6b1 bold"


def _file_icon(relative_path: str) -> str:
    suffix = relative_path.rsplit(".", 1)[-1].lower() if "." in relative_path else ""
    if suffix == "py":
        return "▷"
    if suffix in {"md", "mdx"}:
        return "◇"
    if suffix in {"yaml", "yml", "toml", "json"}:
        return "◇"
    return "·"


def _tab_index_from_x(x: int, labels: Iterable[str]) -> int:
    position = max(0, int(x))
    cursor = 0
    separator_width = cell_len(f" {CONTENT_TAB_SEPARATOR} ")
    for index, label in enumerate(labels):
        if index:
            cursor += separator_width
        width = cell_len(label)
        if cursor <= position < cursor + width:
            return index
        cursor += width
    return -1


def _content_tab_local_x(x: int) -> int:
    return max(
        0,
        int(x) - cell_len(CONTENT_TAB_PREFIX) - CONTENT_TAB_LEFT_PADDING_CELLS,
    )


def _tab_click_hits_close(x: int, labels: Iterable[str], index: int) -> bool:
    if index <= 0:
        return False
    position = max(0, int(x))
    cursor = 0
    separator_width = cell_len(f" {CONTENT_TAB_SEPARATOR} ")
    label_tuple = tuple(labels)
    for current, label in enumerate(label_tuple):
        if current:
            cursor += separator_width
        if current == index:
            width = cell_len(label)
            close_start = cursor + max(0, width - 2)
            return close_start <= position < cursor + width
        cursor += cell_len(label)
    return False


def _session_button_id(session_id: str) -> str:
    return "session-tab-" + "".join(
        char if char.isalnum() or char in {"-", "_"} else "-"
        for char in session_id.strip()
    )


def _session_id_from_button_id(button_id: str) -> str:
    prefix = "session-tab-"
    if not button_id.startswith(prefix):
        return ""
    return button_id[len(prefix) :]


def _find_in_content_preview(
    content: str,
    query: str,
    *,
    title: str,
    limit: int = 24,
) -> tuple[str, int]:
    needle = query.strip().lower()
    if not needle:
        return "", 0
    lines = content.splitlines()
    matches: list[tuple[int, str]] = []
    for line_number, line in enumerate(lines, start=1):
        if needle in line.lower():
            matches.append((line_number, line))
    total = len(matches)
    if not matches:
        return "", 0
    rendered = [f"Matches for {query!r} in {title}", ""]
    for line_number, line in matches[:limit]:
        rendered.append(f"{line_number}: {_single_line(line, limit=180)}")
    hidden = total - limit
    if hidden > 0:
        rendered.append(f"... +{hidden} more matching line(s)")
    return "\n".join(rendered), total


def _is_immediate_command(prompt: str) -> bool:
    clean = prompt.strip()
    if clean in _EXACT_IMMEDIATE_COMMANDS:
        return True
    return (
        clean.startswith("/open ")
        or clean.startswith("/find ")
        or clean.startswith("/search ")
        or clean.startswith("/files ")
        or clean.startswith("/workspace ")
        or clean.startswith("/cd ")
        or clean.startswith("/resume ")
        or clean.startswith("/pet ")
        or clean.startswith("/local ")
        or clean.startswith("/behavior ")
        or clean.startswith("/computer ")
        or clean.startswith("/model ")
        or clean.startswith("/skills ")
        or clean.startswith("/show-skill ")
        or clean.startswith("/remote ")
        or clean.startswith("/hooks ")
        or clean.startswith("/cron ")
        or clean.startswith("/title ")
        or clean.startswith("/rewind ")
        or clean.startswith("/undo ")
        or clean.startswith("/deploy")
        or clean.startswith("/verbose ")
        or clean.startswith("@")
        or clean == "/session"
        or clean.startswith("/session ")
        or clean in {"/tree", "/clone", "/fork"}
        or clean in {"/behavior", "/computer"}
        or clean.startswith("/clone ")
        or clean.startswith("/fork ")
    )


def _is_exact_command(prompt: str) -> bool:
    clean = prompt.strip()
    if not clean:
        return False
    commands = {command for command, _ in _COMMAND_HINTS}
    return clean in commands or clean in _EXACT_IMMEDIATE_COMMANDS


def _command_accepts_immediate_enter(prompt: str) -> bool:
    clean = prompt.strip()
    return clean in _EXACT_IMMEDIATE_COMMANDS or clean in {
        "/commands",
        "/help",
        "/?",
        "/status",
        "/task",
        "/diff",
        "/detail",
        "/preview",
        "/close-tab",
        "/close-preview",
        "/hide-preview",
        "/files",
        "/new",
        "/new-session",
        "/undo-clear",
        "/verbose",
        "/restore-draft",
        "/pet",
        "/local",
        "/model",
        "/sessions",
        "/remote",
        "/hooks",
        "/cron",
        "/skills",
        "/mcp",
        "/session",
        "/session tree",
        "/tree",
        "/clone",
        "/fork",
        "/exit",
        "/quit",
    }


def _is_generic_live_status(value: str) -> bool:
    clean = " ".join(value.strip().lower().split())
    return clean in {"working on", "preparing context..."}


def _is_scroll_at_end(widget: object, *, tolerance: int = 1) -> bool:
    try:
        scroll_y = int(getattr(widget, "scroll_y", 0) or 0)
        max_scroll_y = int(getattr(widget, "max_scroll_y", scroll_y) or 0)
    except (TypeError, ValueError):
        return True
    return scroll_y >= max_scroll_y - tolerance


def _is_specific_live_status(value: str) -> bool:
    return bool(value.strip()) and not _is_generic_live_status(value)


_EXACT_IMMEDIATE_COMMANDS = {
    "/commands",
    "/help",
    "/?",
    "/status",
    "/task",
    "/diff",
    "/detail",
    "/preview",
    "/close-tab",
    "/close-preview",
    "/hide-preview",
    "/files",
    "/find",
    "/new",
    "/new-session",
    "/undo-clear",
    "/verbose",
    "/restore-draft",
    "/pet",
    "/local",
    "/model",
    "/sessions",
    "/resume",
    "/rewind",
    "/undo",
    "/skills",
    "/mcp",
    "/remote",
    "/hooks",
        "/cron",
        "/approvals",
        "/exit",
        "/quit",
    }


def _directory_input_path(
    raw_value: str,
    *,
    base: Path | None = None,
    allow_bare: bool = False,
) -> Path | None:
    clean = raw_value.strip()
    if not clean:
        return None
    candidate_text = _single_path_token(clean)
    if candidate_text is None:
        return None
    if candidate_text.startswith("file://"):
        parsed = urlparse(candidate_text)
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
            return None
        candidate_text = unquote(parsed.path)
    if not allow_bare and not candidate_text.startswith(("~", "/", ".")):
        return None
    candidate = Path(candidate_text).expanduser()
    if base is not None and not candidate.is_absolute():
        candidate = base / candidate
    try:
        return candidate.resolve() if candidate.is_dir() else None
    except OSError:
        return None


def _single_path_token(clean: str) -> str | None:
    try:
        parts = shlex.split(clean)
    except ValueError:
        parts = [clean.strip("'\"")]
    if len(parts) != 1:
        return None
    token = parts[0].strip()
    return token or None


def _footer_left_label(state: TuiRuntimeState) -> str:
    summary = _capability_status_summary(state)
    cache = state.runtime_stats.cache_summary()
    parts = [state.model]
    if summary:
        parts.append(summary)
    if cache:
        parts.append(cache)
    return "  │  ".join(parts)


def _context_window_color(ratio: float | None) -> str | None:
    """Return a footer color for the context-window gauge by fill level."""
    if ratio is None:
        return None
    if ratio >= 0.85:
        return "#c98787"
    if ratio >= 0.60:
        return "#cdbb7a"
    return None


def _capability_status_summary(state: TuiRuntimeState) -> str:
    registry = state.native_tools
    if registry is None:
        return ""
    names = {tool.name for tool in registry.list_tools()}
    visible = {
        str(schema.get("name", "")).strip()
        for schema in state.tool_schemas
        if isinstance(schema, dict) or hasattr(schema, "get")
    }
    parts = []
    if "computer_status" in visible:
        parts.append("computer on")
    return " · ".join(part for part in parts if part)


def _write_capability_status(
    state: TuiRuntimeState,
    names: set[str],
    visible: set[str],
) -> str:
    if not {"write_text_file", "edit_text_file"} & names:
        return "edit off"
    if not {"write_text_file", "edit_text_file"} & visible:
        return "edit when needed"
    if (
        state.tool_access_policy is not None
        and state.tool_access_policy.mode != ToolAccessMode.WORKSPACE_WRITE
    ):
        return "confirm edits"
    return "can edit"


def _shell_capability_status(names: set[str], visible: set[str]) -> str:
    if "run_shell_command" not in names:
        return "commands off"
    if "run_shell_command" not in visible:
        return "commands when needed"
    return "confirm commands"


def _prompt_length_ok(prompt: str) -> bool:
    return len(prompt) <= MAX_PROMPT_CHARS


def _prompt_too_long_message(length: int) -> TuiMessage:
    return TuiMessage(
        kind="error",
        title="input too long",
        body=(
            f"Input is {length} characters; TUI accepts up to {MAX_PROMPT_CHARS} "
            "characters per prompt."
        ),
    )


def _undelivered_messages(
    active_turn: _TurnRun | None,
    messages: tuple[TuiMessage, ...],
) -> tuple[TuiMessage, ...]:
    """Return non-user messages that have not been inserted for this turn yet."""
    if active_turn is None:
        return tuple(message for message in messages if message.kind != "user")
    delivered: list[TuiMessage] = []
    seen_in_batch: dict[tuple[str, str, str, str, tuple[str, ...], str], int] = {}
    for message in messages:
        if message.kind == "user":
            continue
        fingerprint = _message_fingerprint(message)
        occurrence = seen_in_batch.get(fingerprint, 0)
        seen_in_batch[fingerprint] = occurrence + 1
        key = _message_delivery_key(fingerprint, occurrence)
        if key in active_turn.delivered_message_keys:
            continue
        active_turn.delivered_message_keys.add(key)
        delivered.append(message)
    return tuple(delivered)


def _contains_visible_answer(messages: tuple[TuiMessage, ...]) -> bool:
    return any(
        message.kind == "assistant" and message.body.strip()
        for message in messages
    )


def _message_fingerprint(
    message: TuiMessage,
) -> tuple[str, str, str, str, tuple[str, ...], str]:
    return (
        message.kind,
        message.title,
        message.body,
        message.status,
        tuple(message.refs),
        message.preview,
    )


def _message_delivery_key(
    fingerprint: tuple[str, str, str, str, tuple[str, ...], str],
    occurrence: int,
) -> str:
    kind, title, body, status, refs, preview = fingerprint
    return "\x1f".join(
        (
            str(occurrence),
            kind,
            title,
            body,
            status,
            "\x1e".join(refs),
            preview,
        )
    )


def _rewind_turn_list_message(turns) -> TuiMessage:
    if not turns:
        return TuiMessage(
            kind="status",
            title="rewind",
            body="No checkpoints found for this session yet.",
        )
    lines = ["Available checkpoints:"]
    for turn in turns[-12:]:
        lines.append(
            f"- {turn.turn_id}: status={turn.status}, resume={turn.resume_hint}, "
            f"last_sequence={turn.last_transcript_sequence}"
        )
    lines.append("")
    lines.append("Usage: /rewind <turn_id> [--mode workspace|conversation|both] [--apply] [--force]")
    return TuiMessage(kind="status", title="rewind", body="\n".join(lines))


def _tui_conversation_rewind_plan(session_store, session, target) -> dict[str, object]:
    transcript = session_store.transcript_store(session)
    records = transcript.load_records()
    keep_sequence = target.last_transcript_sequence
    remove_count = sum(1 for record in records if record.sequence > keep_sequence)
    summary = session_store.summary_store(session).load_latest()
    summary_action = "none"
    if summary is not None:
        summary_action = (
            "delete"
            if summary.covered_until_sequence > keep_sequence
            else "keep"
        )
    return {
        "keep_sequence": keep_sequence,
        "current_records": len(records),
        "remove_records": remove_count,
        "summary_action": summary_action,
        "summary_id": summary.summary_id if summary is not None else "",
    }


def _apply_tui_conversation_rewind(session_store, session, turn_store, target) -> None:
    plan = _tui_conversation_rewind_plan(session_store, session, target)
    transcript = session_store.transcript_store(session)
    transcript.truncate_after(int(plan["keep_sequence"]))
    if plan["summary_action"] == "delete":
        session_store.summary_store(session).delete_latest()
    turn_store.set_latest(target.turn_id)


def _rewind_preview_summary(workspace_plan, conversation_plan) -> str:
    parts: list[str] = []
    if workspace_plan is not None:
        conflicts = sum(1 for action in workspace_plan.actions if action.conflict)
        parts.append(
            f"workspace actions={len(workspace_plan.actions)}"
            + (f", conflicts={conflicts}" if conflicts else "")
        )
    if conversation_plan is not None:
        parts.append(f"conversation remove_records={conversation_plan['remove_records']}")
    return "Preview: " + ("; ".join(parts) if parts else "no changes")


def _format_tui_rewind_detail(
    *,
    target_turn_id: str,
    mode: str,
    apply: bool,
    force: bool,
    workspace_plan,
    conversation_plan,
) -> str:
    lines = [
        "# Checkpoint rewind",
        "",
        f"- target_turn: {target_turn_id}",
        f"- mode: {mode}",
        f"- action: {'apply' if apply else 'preview'}",
    ]
    if force:
        lines.append("- force: true")
    if workspace_plan is not None:
        lines.extend(["", "## Workspace"])
        if not workspace_plan.actions:
            lines.append("- No workspace file changes after target turn.")
        for action in workspace_plan.actions:
            suffix = " conflict" if action.conflict else ""
            reason = f" ({action.reason})" if action.reason else ""
            lines.append(f"- {action.action}: `{action.path}`{suffix}{reason}")
    if conversation_plan is not None:
        lines.extend(
            [
                "",
                "## Conversation",
                f"- keep_until_sequence: {conversation_plan['keep_sequence']}",
                f"- current_records: {conversation_plan['current_records']}",
                f"- remove_records: {conversation_plan['remove_records']}",
                f"- summary: {conversation_plan['summary_action']}",
            ]
        )
    return "\n".join(lines)


def _strip_command_arg(text: str, prefixes: tuple[str, ...]) -> str | None:
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return None


def _has_terminal_error(messages: Iterable[TuiMessage]) -> bool:
    visible = [
        message
        for message in messages
        if message.kind not in {"user", "status", "task"}
    ]
    if not visible:
        return False
    return visible[-1].kind == "error"


def _transcript_has_items(state: TuiRuntimeState) -> bool:
    try:
        return bool(state.transcript.load_items())
    except OSError:
        return False
