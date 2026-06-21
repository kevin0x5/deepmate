"""Pure rendering helpers for the Textual TUI."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

from rich.markup import escape
from rich.markdown import Markdown as RichMarkdown
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from deepmate.channels.tui.formatters import TuiMessage

_MESSAGE_STYLES = {
    "user": "bold #8fb7bd",
    "assistant": "#c8c8c8",
    "followup": "#8fbe8f",
    "thinking": "#8a8a8a",
    "tool shell": "#8fb7bd",
    "tool browser": "#8fb7bd",
    "tool mcp": "#8fb7bd",
    "tool skill": "#8fb7bd",
    "tool subagent": "#8fb7bd",
    "tool read": "#8fb7bd",
    "tool write": "#8fb7bd",
    "tool tool": "#8fb7bd",
    "task": "#8fbe8f",
    "diff": "#8a8a8a",
    "file": "#8a8a8a",
    "approval": "#d0b66b",
    "permissions": "#8fbe8f",
    "warning": "#d0b66b",
    "error": "#d98a8a",
    "status": "#8a8a8a",
    "welcome": "#c8c8c8",
}

_MESSAGE_LABELS = {
    "user": "You",
    "assistant": "Deepmate",
    "followup": "Added",
    "thinking": "Thinking",
    "approval": "Approval",
    "permissions": "Permissions",
    "warning": "Warning",
    "error": "Error",
    "status": "Status",
    "task": "Task",
    "diff": "Diff",
    "file": "File",
    "welcome": "Deepmate",
}

_MESSAGE_PREFIXES = {
    "user": "▌",
    "assistant": "▎",
    "followup": "↳",
    "thinking": "∴",
    "approval": "▶",
    "permissions": "✓",
    "warning": "·",
    "error": "✗",
    "status": "·",
    "task": "▶",
    "diff": "↪",
    "file": "↪",
    "welcome": "",
}


def render_message(message: TuiMessage):
    """Return a Rich-renderable chat item for one TUI message."""
    if message.kind == "welcome":
        return message.body.strip()
    if message.status == "live":
        body = readable_markdown(message.body.strip())
        if not body:
            return ""
        style = _MESSAGE_STYLES.get("status", "#8a8a8a")
        return f"[{style}]{escape('· ' + body)}[/]"
    title = message.title.strip()
    body = message.body.strip()
    style = _MESSAGE_STYLES.get(message.kind, "#d8d8d8")
    status = f" - {message.status}" if message.status else ""
    label = _MESSAGE_LABELS.get(message.kind, title or message.kind)
    if message.kind.startswith("tool "):
        family = message.kind.removeprefix("tool ").strip() or "tool"
        label = f"Action: {_tool_family_label(family)}"
        if message.status == "error":
            style = _MESSAGE_STYLES["error"]
        elif message.status == "approval required":
            style = _MESSAGE_STYLES["approval"]
        elif message.status in {"compacted", "pending"}:
            style = _MESSAGE_STYLES["status"]
    prefix = _MESSAGE_PREFIXES.get(message.kind, "╭─")
    header_text = f"{prefix} {label}{status}" if prefix != "·" else f"· {label}{status}"
    if (
        title
        and title.lower() != label.lower()
        and title.lower() != message.kind.lower()
    ):
        header_text += f" · {title}"
    header = f"[{style}]{escape(header_text)}[/]"
    if not body:
        return header
    if message.kind == "user":
        return _user_prompt_block(readable_markdown(body))
    if message.kind == "assistant":
        return _assistant_markdown_block(body)
    if message.kind == "followup":
        return _followup_block(body)
    rendered_body = escape(_indent_message_body(readable_markdown(body), message.kind))
    return f"{header}\n{rendered_body}"


def _assistant_markdown_block(body: str) -> Padding:
    return Padding(
        RichMarkdown(body.strip(), code_theme="ansi_dark", hyperlinks=False),
        (0, 0, 1, 0),
        expand=True,
    )


def _user_prompt_block(body: str) -> Padding:
    prompt = Text()
    lines = body.strip().splitlines() or [""]
    if len(lines) == 1:
        prompt.append("\n")
    for index, line in enumerate(lines):
        if index:
            prompt.append("\n")
        prefix = "› " if index == 0 else "  "
        prompt.append(prefix + line, style="#e0e0e0")
    if len(lines) == 1:
        prompt.append("\n")
    table = Table.grid(expand=True)
    table.add_column(ratio=1)
    table.style = "on #2c2c2c"
    table.add_row(prompt, style="on #2c2c2c")
    return Padding(table, (1, 0, 1, 0), expand=True)


def _followup_block(body: str) -> Padding:
    clean = readable_markdown(body)
    text = Text()
    lines = clean.strip().splitlines() or [""]
    text.append("↳ added to current turn", style="#8fbe8f")
    for line in lines:
        text.append("\n  " + line, style="#c8c8c8")
    return Padding(text, (0, 0, 1, 2), expand=True)


def should_expand_message(kind: str) -> bool:
    """Return whether a message should expand to fill the chat width."""
    return kind in {"user", "assistant"}


def _indent_message_body(text: str, kind: str) -> str:
    clean = text.strip()
    if not clean:
        return ""
    if kind in {"status", "permissions"}:
        return clean
    return "\n".join(f"│ {line}" if line else "│" for line in clean.splitlines())


def _tool_family_label(value: str) -> str:
    clean = value.strip()
    return {
        "shell": "command",
        "browser": "browser",
        "mcp": "external connection",
        "skill": "skill",
        "subagent": "subtask",
        "read": "file read",
        "write": "file edit",
        "tool": "work",
        "run_shell_command": "command",
    }.get(clean, clean.replace("_", " ") or "work")


def preview_text(content: str, *, limit: int = 12_000) -> str:
    """Return a short terminal-readable text preview."""
    clean = content.strip()
    if not clean:
        return "(empty)"
    if len(clean) <= limit:
        return readable_markdown(clean)
    return readable_markdown(
        clean[: limit - 40].rstrip() + "\n\n[detail truncated by Deepmate TUI]"
    )


def preview_tab_content(title: str, content: str, *, mode: str = "auto") -> str:
    """Return Markdown content for one workbench content tab."""
    clean_mode = mode.strip().lower()
    if clean_mode == "markdown" or (clean_mode == "auto" and _is_markdown_path(title)):
        return _preview_markdown(content)
    language = _code_fence_language(title)
    return _preview_code_block(content, language=language)


def _preview_markdown(content: str, *, limit: int = 80_000) -> str:
    clean = content.strip()
    if not clean:
        return "_empty_"
    if len(clean) <= limit:
        return clean
    return clean[: limit - 54].rstrip() + "\n\n> Content truncated by Deepmate TUI."


def _preview_code_block(content: str, *, language: str = "", limit: int = 80_000) -> str:
    clean = content.rstrip("\n")
    if not clean:
        return "```text\n(empty)\n```"
    truncated = False
    if len(clean) > limit:
        clean = clean[: limit - 42].rstrip()
        truncated = True
    safe = clean.replace("```", "``\\`")
    fence = f"```{language}" if language else "```"
    if truncated:
        safe += "\n\n[content truncated by Deepmate TUI]"
    return f"{fence}\n{safe}\n```"


def _is_markdown_path(path: str) -> bool:
    suffix = _path_suffix(path)
    return suffix in {"md", "mdx", "markdown"}


def _path_suffix(path: str) -> str:
    name = path.strip().rstrip("/").rsplit("/", 1)[-1].lower()
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[-1]


def _code_fence_language(path: str) -> str:
    suffix = _path_suffix(path)
    return {
        "py": "python",
        "js": "javascript",
        "jsx": "jsx",
        "ts": "typescript",
        "tsx": "tsx",
        "json": "json",
        "jsonl": "json",
        "toml": "toml",
        "yaml": "yaml",
        "yml": "yaml",
        "html": "html",
        "css": "css",
        "sh": "bash",
        "bash": "bash",
        "zsh": "bash",
        "sql": "sql",
        "xml": "xml",
        "svg": "xml",
        "txt": "text",
    }.get(suffix, "text")


def readable_markdown(text: str) -> str:
    """Return a terminal-readable approximation of common Markdown."""
    lines: list[str] = []
    in_fence = False
    for raw_line in clean_input_text(text).splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            lines.append("    " + line)
            continue
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            if heading:
                lines.append(heading)
            continue
        if stripped.startswith(("- ", "* ")):
            lines.append("• " + _strip_inline_markdown(stripped[2:].strip()))
            continue
        if len(stripped) > 2 and stripped[0].isdigit() and ". " in stripped[:5]:
            number, _, rest = stripped.partition(". ")
            lines.append(f"{number}. {_strip_inline_markdown(rest.strip())}")
            continue
        lines.append(_strip_inline_markdown(line))
    return "\n".join(lines).strip()


def _strip_inline_markdown(text: str) -> str:
    value = re.sub(r"(?<!\*)\*\*([^*\n]+)\*\*(?!\*)", r"\1", text)
    value = re.sub(r"(?<!_)__([^_\n]+)__(?!_)", r"\1", value)
    value = re.sub(r"`([^`\\\s]+)`", r"\1", value)
    return value


def clean_input_text(text: str) -> str:
    """Strip terminal control characters while preserving normal whitespace."""
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    cleaned: list[str] = []
    index = 0
    while index < len(normalized):
        char = normalized[index]
        code = ord(char)
        if char == "\x1b":
            index += 1
            if index < len(normalized) and normalized[index] == "[":
                index += 1
                while index < len(normalized):
                    final = normalized[index]
                    index += 1
                    if "@" <= final <= "~":
                        break
                continue
            if index < len(normalized) and normalized[index] == "]":
                index += 1
                while index < len(normalized):
                    final = normalized[index]
                    index += 1
                    if final == "\x07":
                        break
                    if final == "\x1b" and index < len(normalized) and normalized[index] == "\\":
                        index += 1
                        break
                continue
            continue
        if char in {"\n", "\t"}:
            cleaned.append(char)
        elif code < 32 or code == 127:
            cleaned.append(" ")
        else:
            cleaned.append(char)
        index += 1
    return "".join(cleaned)


def welcome_splash(
    *,
    workspace: object,
    session_id: str,
    provider_name: str = "",
    api_key_env: str = "",
    api_key_available: bool = True,
) -> str:
    """Return the first-run welcome text for an empty TUI session."""
    clean_workspace = escape(compact_workspace_label(str(workspace), limit=28))
    short_session = (session_id or "new-session").strip()
    if short_session.startswith("session-"):
        short_session = short_session[len("session-") :] or short_session
    clean_session = escape(short_session[:8] or "new")
    lines = [
        "[#4a4a4a]······························[/]   [bold #e0e0e0]Deepmate[/] [#8a8a8a]local agent workspace[/]",
        "[#4a4a4a]······[/][#8fb7bd] /\\_/\\  [/][#4a4a4a]················[/]   [#8a8a8a]New session is ready[/] [#8a8a8a]·[/] [#e0e0e0]Let‘s work on[/]",
        "[#4a4a4a]····[/][#e0e0e0]  ( o o )  [/][#4a4a4a]···············[/]   [#8a8a8a]workspace[/] [#e0e0e0]"
        + clean_workspace
        + "[/]",
        "[#4a4a4a]······[/][#d0b66b]=  ^  =  [/][#4a4a4a]···············[/]   [#8a8a8a]session[/] [#e0e0e0]"
        + clean_session
        + "[/]",
        "[#4a4a4a]······························[/]   [#8a8a8a]try[/] [#8fb7bd]/commands[/] [#8fb7bd]/status[/] [#8fb7bd]/task[/]  [#8a8a8a]files[/] [#8fb7bd][/]",
        "[#4a4a4a]·······································································[/]",
    ]
    missing_key = missing_api_key_env(
        api_key_env=api_key_env,
        api_key_available=api_key_available,
    )
    if missing_key:
        provider_label = (
            f" for provider {provider_name.strip()}" if provider_name.strip() else ""
        )
        lines.extend(
            (
                "",
                f"[#d98a8a]Model key needed:[/] [#8a8a8a]paste it with[/] [#8fb7bd]/setup-key <api-key>[/][#8a8a8a]{escape(provider_label)}. Advanced env name:[/] [#e0e0e0]{escape(missing_key)}[/]",
            )
        )
    lines.extend(
        (
            "",
            "[#8a8a8a]Try asking:[/]",
            "[#8fb7bd]  • Summarize this project structure and main modules[/]",
            "[#8fb7bd]  • Find and fix a bug, then run tests[/]",
            "[#8fb7bd]  • Explain the code of a function[/]",
            "[#8fb7bd]  • Generate a code sample for a function[/]",
        )
    )
    return "\n".join(lines)


def missing_api_key_env(
    *,
    api_key_env: str = "",
    api_key_available: bool = True,
) -> str:
    """Return the missing API key env name for the welcome splash, if any."""
    clean = api_key_env.strip() or "DEEPSEEK_API_KEY"
    if api_key_available or os.environ.get(clean, "").strip():
        return ""
    return clean


def compact_workspace_label(workspace: str, *, limit: int = 42) -> str:
    """Shorten a workspace path for narrow TUI labels."""
    clean = workspace.strip() or "."
    if len(clean) <= limit:
        return clean
    marker = "/"
    if marker in clean:
        tail = clean.rsplit(marker, 1)[-1]
        if tail and len(tail) + 4 <= limit:
            return ".../" + tail
    return "..." + clean[-(limit - 3) :]


def short_session_title(title: str, *, active: bool, limit: int = 18) -> str:
    """Return a compact session tab title."""
    clean = title.strip() or "Untitled"
    active_limit = max(limit, 25) if active else limit
    if len(clean) <= active_limit:
        return clean
    return clean[: active_limit - 1].rstrip() + "…"


def content_tab_display_label(
    tab_name: str,
    *,
    active: bool,
    hovered: bool = False,
) -> str:
    """Return one workbench tab label."""
    _ = hovered
    title = "main" if tab_name == "main" else _short_tab_title(tab_name)
    label = _tab_label(title, active=active)
    if tab_name != "main":
        label += " ×"
    return label


def workspace_nav_title(workspace: Path) -> str:
    """Return the sidebar workspace title."""
    name = workspace.name.strip() or str(workspace)
    return single_line(name, limit=24)


def sessions_preview(
    sessions,
    *,
    current_workspace: Path,
    current_session_id: str,
    limit_per_workspace: int = 12,
) -> str:
    """Render the /sessions content tab grouped by workspace."""
    materialized = tuple(sessions)
    if not materialized:
        return "No sessions found."
    current = current_workspace.resolve()
    grouped: dict[Path, list[object]] = {}
    for session in materialized:
        grouped.setdefault(session.workspace.resolve(), []).append(session)
    ordered_workspaces = sorted(
        grouped,
        key=lambda workspace: (
            workspace != current,
            -_timestamp_sort_value(
                max(session.updated_at for session in grouped[workspace])
            ),
        ),
    )
    lines = ["Sessions", ""]
    for workspace in ordered_workspaces:
        workspace_sessions = sorted(
            grouped[workspace],
            key=lambda session: session.updated_at,
            reverse=True,
        )
        workspace_name = workspace.name or str(workspace)
        current_marker = " (current workspace)" if workspace == current else ""
        lines.append(f"{workspace_name}{current_marker}")
        lines.append(f"  {workspace}")
        for session in workspace_sessions[:limit_per_workspace]:
            marker = "*" if session.session_id == current_session_id else "-"
            title = single_line(session.title, limit=42)
            lines.append(
                f"  {marker} {session.session_id[:8]}  {title}  {session.updated_at}"
            )
        hidden = len(workspace_sessions) - limit_per_workspace
        if hidden > 0:
            lines.append(f"  ... {hidden} more")
        lines.append("")
    lines.append("Use /resume <session-id-prefix> to switch sessions.")
    return "\n".join(lines).rstrip()


def single_line(text: str, *, limit: int) -> str:
    """Compact arbitrary text to one line."""
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def _tab_label(title: str, *, active: bool) -> str:
    marker = "●" if active else " "
    return f"{marker} {title}"


def _short_tab_title(title: str, *, limit: int = 28) -> str:
    clean = title.strip() or "tab"
    name = clean.rstrip("/").rsplit("/", 1)[-1]
    if len(name) <= limit:
        return name
    return name[: limit - 1].rstrip() + "…"


def _timestamp_sort_value(value: str) -> float:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0
