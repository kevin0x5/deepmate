"""Format runtime results into compact TUI view models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json

from deepmate.providers import ModelResponse, ModelToolExchange, ModelToolResult
from deepmate.runtime import UserTurnResult

TOOL_MESSAGE_SUMMARY_THRESHOLD = 8

_SUPPRESSED_EVENT_KINDS = {
    "browser_backend_unavailable",
    "mcp_tool_schema_not_loaded",
    "mcp_tool_completed",
    "native_tool_completed",
    "native_tool_schema_not_loaded",
    "native_tool_schema_hidden",
    "mcp_tool_schema_hidden",
    "tool_output_compacted",
    "tool_output_normalized",
}

_SUPPRESSED_ERROR_CODES = {
    "browser_backend_unavailable",
    "mcp_tool_schema_not_loaded",
    "native_tool_schema_not_loaded",
    "native_tool_failed",
}


@dataclass(frozen=True, slots=True)
class TuiMessage:
    """One rendered chat item."""

    kind: str
    title: str
    body: str = ""
    status: str = ""
    refs: tuple[str, ...] = ()
    preview: str = ""


def friendly_error_message(exc: BaseException) -> TuiMessage:
    """Return a user-facing TUI error without leaking internal names."""
    text = str(exc).strip()
    lowered = text.lower()
    if _looks_like_missing_model_key(text):
        return TuiMessage(
            kind="error",
            title="model setup",
            body=_missing_model_key_text(text),
        )
    if "turn checkpoint not found" in lowered:
        return TuiMessage(
            kind="error",
            title="session state",
            body=(
                "Deepmate could not find the saved progress for this request. "
                "Send a new message to continue."
            ),
        )
    if "checkpoint" in lowered:
        return TuiMessage(
            kind="error",
            title="session state",
            body=(
                "Deepmate could not update the saved progress for this request. "
                "You can continue with a new message."
            ),
        )
    if _looks_like_model_timeout(text):
        return TuiMessage(
            kind="error",
            title="model connection timed out",
            body=(
                "The model connection timed out while Deepmate was waiting for "
                "response data. You can retry the request or resume the queue "
                "when the network/provider is stable."
            ),
        )
    if _looks_like_missing_session_state(text):
        return TuiMessage(
            kind="error",
            title="session state",
            body=(
                "Deepmate could not find the current session state. "
                "Create a new session or switch to another session to continue."
            ),
        )
    return TuiMessage(
        kind="error",
        title="something went wrong",
        body=text or "The request stopped unexpectedly.",
    )


def friendly_error_text(text: str) -> str:
    """Return user-facing error text without internal filesystem details."""
    clean = str(text).strip()
    lowered = clean.lower()
    normalized = clean.replace("\\", "/")
    if _looks_like_missing_model_key(clean):
        return _missing_model_key_text(clean)
    if _looks_like_missing_session_state(clean):
        return (
            "Deepmate could not find the current session state. "
            "Create a new session or switch to another session to continue."
        )
    if "turn checkpoint not found" in lowered:
        return "Deepmate could not find the saved progress for this request."
    if "checkpoint" in lowered:
        return "Deepmate could not update the saved progress for this request."
    if _looks_like_model_timeout(clean):
        return (
            "The model connection timed out while Deepmate was waiting for "
            "response data. You can retry the request or resume the queue when "
            "the network/provider is stable."
        )
    return clean


def _looks_like_model_timeout(text: str) -> bool:
    lowered = str(text).strip().lower()
    return "timed out" in lowered and (
        "model request" in lowered
        or "response data" in lowered
        or "read operation" in lowered
    )


def _looks_like_missing_model_key(text: str) -> bool:
    lowered = str(text).strip().lower()
    return (
        "deepmate needs a model api key" in lowered
        or "model api key saved locally" in lowered
        or "/setup-key" in lowered
    )


def _missing_model_key_text(text: str) -> str:
    clean = str(text).strip()
    if clean:
        return clean
    return "Deepmate needs a model API key. Paste it with /setup-key <api-key>."


def _looks_like_missing_session_state(text: str) -> bool:
    clean = str(text).strip()
    lowered = clean.lower()
    normalized = clean.replace("\\", "/")
    return "/sessions/" in normalized and (
        "filenotfounderror" in lowered
        or "no such file or directory" in lowered
        or "errno 2" in lowered
    )


def result_messages(
    result: UserTurnResult,
    *,
    show_reasoning: bool = False,
) -> tuple[TuiMessage, ...]:
    """Return user-turn result messages in display order."""
    messages: list[TuiMessage] = []
    tool_messages: list[TuiMessage] = []
    detail_lines: list[str] = []
    final_candidate = ""
    provider_failure_codes = {
        error.code for step in result.steps for error in step.errors
    }
    for step in result.steps:
        reasoning = (step.response.reasoning or "").strip()
        if reasoning and show_reasoning:
            messages.append(
                TuiMessage(
                    kind="thinking",
                    title="thinking",
                    body=reasoning,
                    status="full",
                )
            )
        exchange = step.to_tool_exchange()
        if exchange is not None:
            step_tool_messages = tool_exchange_messages(exchange)
            tool_messages.extend(step_tool_messages)
            detail_lines.extend(_tool_detail_lines(step_tool_messages))
        for event in step.events:
            if event.kind in _SUPPRESSED_EVENT_KINDS:
                continue
            if (
                event.kind == "provider_request_failed"
                and "provider_request_failed" in provider_failure_codes
            ):
                continue
            messages.append(
                TuiMessage(
                    kind=_event_kind(event.kind),
                    title=event.kind,
                    body=event.summary,
                    refs=tuple(event.refs),
                )
            )
        for error in step.errors:
            if error.code in _SUPPRESSED_ERROR_CODES:
                continue
            body = friendly_error_text(error.message)
            messages.append(
                TuiMessage(
                    kind="error",
                    title=error.code,
                    body=body,
                    refs=tuple(error.refs),
                )
            )
    final = response_text(result.final_step().response)
    final_candidate = final
    # Fold noisy tool cards into a summary once there's a final answer, but only
    # when there are several of them — a single tool card is worth showing in full
    # (its compacted/diff detail is the useful part).
    fold_tools_for_final = (
        bool(final_candidate)
        and len(tool_messages) > 1
        and _has_noisy_tool_messages(tool_messages)
    )
    if len(tool_messages) > TOOL_MESSAGE_SUMMARY_THRESHOLD or fold_tools_for_final:
        messages.append(
            _tool_run_summary(
                tool_messages,
                "\n\n".join(detail_lines),
                final_present=bool(final_candidate),
            )
        )
    else:
        messages.extend(tool_messages)
    if final and not _final_duplicates_terminal_error(final, messages):
        messages.append(TuiMessage(kind="assistant", title="assistant", body=final))
    elif (result.final_step().response.reasoning or "").strip() and not show_reasoning:
        messages.append(
            TuiMessage(
                kind="warning",
                title="no final answer",
                body=(
                    "The model returned reasoning but no final answer. "
                    "Run with reasoning display enabled or ask it to continue."
                ),
            )
        )
    if result.loop_guard_stop is not None:
        messages.append(
            TuiMessage(
                kind="warning",
                title="turn paused",
                body=(
                    f"{result.loop_guard_stop.message}\n\n"
                    "Enter `continue` or `继续` to resume with the saved checkpoint."
                ),
            )
        )
    elif result.reached_max_steps:
        messages.append(
            TuiMessage(
                kind="error",
                title="max steps",
                body="Reached max_steps before a final answer.",
            )
        )
    return tuple(messages)


def _final_duplicates_terminal_error(final: str, messages: list[TuiMessage]) -> bool:
    if not messages:
        return False
    last = messages[-1]
    if last.kind != "error":
        return False
    if (
        last.title == "provider_request_failed"
        and final.startswith("Model request failed after retry attempts:")
    ):
        return True
    clean_final = " ".join(final.split())
    clean_error = " ".join(last.body.split())
    if not clean_final or not clean_error:
        return False
    return clean_final == clean_error or clean_final.endswith(clean_error)


def _has_noisy_tool_messages(messages: list[TuiMessage]) -> bool:
    return any(
        message.status in {"error", "approval required"}
        or message.title
        in {
            "run_shell_command",
            "load_skill",
            "install_skill",
            "install_skill_bundle",
            "install_skill_from_request",
        }
        for message in messages
    )


def _tool_run_summary(
    messages: tuple[TuiMessage, ...] | list[TuiMessage],
    preview: str,
    *,
    final_present: bool = False,
) -> TuiMessage:
    total = len(messages)
    errors = sum(1 for message in messages if message.status in {"error", "approval required"})
    compacted = sum(1 for message in messages if message.status == "compacted")
    successes = sum(1 for message in messages if message.status in {"ok", "compacted"})
    counts: dict[str, int] = {}
    first_errors: list[str] = []
    for message in messages:
        counts[message.title] = counts.get(message.title, 0) + 1
        if message.status in {"error", "approval required"} and len(first_errors) < 3:
            first_errors.append(_first_meaningful_line(message.body, limit=180) or message.title)
    lines = [f"Ran {total} tools."]
    if final_present:
        lines.append("Intermediate tool output was folded because a final answer was produced.")
        if errors:
            lines.append(f"{errors} intermediate attempts failed; details are in /detail.")
    elif errors:
        lines.append(f"{errors} failed.")
    if compacted:
        lines.append(f"{compacted} outputs were folded.")
    top_tools = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5]
    if top_tools:
        lines.append(
            "Most used: "
            + ", ".join(f"{name} x{count}" if count > 1 else name for name, count in top_tools)
        )
    if first_errors and not final_present:
        lines.append("First failures:")
        lines.extend(f"- {line}" for line in first_errors)
    lines.append("Use /detail to inspect the full tool log.")
    return TuiMessage(
        kind="tool summary",
        title="tool activity",
        body="\n".join(lines),
        status="summary",
        preview=preview,
    )


def _tool_detail_lines(messages: tuple[TuiMessage, ...]) -> tuple[str, ...]:
    lines: list[str] = []
    for message in messages:
        header = f"{message.title} [{message.status or 'status'}]"
        lines.append(header)
        body = message.body.strip()
        preview = message.preview.strip()
        if body:
            lines.append(body)
        if preview and preview != body:
            lines.append("detail:")
            lines.append(_preview_detail(preview, 20_000))
    return tuple(lines)


def tool_exchange_messages(exchange: ModelToolExchange) -> tuple[TuiMessage, ...]:
    """Return compact messages for one assistant/tool exchange."""
    messages: list[TuiMessage] = []
    results_by_id = {result.request_id: result for result in exchange.tool_results}
    for request in exchange.tool_requests:
        result = results_by_id.get(request.id)
        messages.append(
            TuiMessage(
                kind=f"tool {tool_source(request.name)}",
                title=request.name,
                body=_tool_summary(request.name, request.arguments, result),
                status=_tool_status(result),
                refs=tuple(result.refs) if result is not None else (),
                preview=_tool_preview(result),
            )
        )
    return tuple(messages)


def response_text(response: ModelResponse) -> str:
    """Return the user-visible response text."""
    return (response.content or "").strip()


def tool_source(name: str) -> str:
    """Return the rough tool family for display color/icon mapping."""
    clean = name.strip()
    if clean.startswith("browser_") or clean == "load_browser_tools":
        return "browser"
    if clean.startswith("mcp.") or clean in {"search_mcp_tools", "load_mcp_tool"}:
        return "mcp"
    if clean in {
        "load_skill",
        "install_skill",
        "install_skill_bundle",
        "inspect_skill_source",
        "verify_skill_install",
        "plan_skill_setup",
        "run_skill_setup",
    }:
        return "skill"
    if clean.startswith("run_subagent"):
        return "subagent"
    if clean == "run_shell_command":
        return "shell"
    if any(marker in clean for marker in ("write", "patch", "edit")):
        return "write"
    if any(marker in clean for marker in ("read", "list", "search", "status")):
        return "read"
    return "tool"


def _tool_summary(
    name: str,
    arguments: Mapping[str, object],
    result: ModelToolResult | None,
) -> str:
    lines: list[str] = []
    if result is None:
        arg_summary = _argument_summary(arguments)
        return f"Requested{': ' + arg_summary if arg_summary else '.'}"
    refs = _ref_map(result.refs)
    status = _tool_status(result)
    if status == "ok":
        lines.append(_success_summary(name, arguments, result))
        if _has_detail(result):
            lines.append("Use /detail to view the latest tool output.")
    elif status == "compacted":
        lines.append(_success_summary(name, arguments, result))
        lines.append("Output folded for context. Use /detail to view available detail.")
    elif status == "approval required":
        lines.append(_tool_error_summary(result.content, fallback="Approval is required before this can run."))
    else:
        lines.append(_tool_error_summary(result.content, fallback="Tool failed."))
    if refs.get("tool_output_ref"):
        lines.append(f"Raw output handle: {refs['tool_output_ref']}")
    return "\n".join(lines)


def _success_summary(
    name: str,
    arguments: Mapping[str, object],
    result: ModelToolResult,
) -> str:
    if name in {"write_text_file", "edit_text_file"}:
        return _write_success_summary(name, result)
    if name == "load_browser_tools":
        count = result.data.get("schema_count")
        return (
            f"Browser tools loaded ({count})."
            if isinstance(count, int)
            else "Browser tools loaded."
        )
    arg_summary = _argument_summary(arguments)
    content = _first_meaningful_line(result.content, limit=160)
    if content:
        return f"Completed.\n{content}"
    if arg_summary:
        return f"Completed: {arg_summary}"
    return "Completed."


def _write_success_summary(name: str, result: ModelToolResult) -> str:
    path = result.data.get("path")
    label = "Edited" if name == "edit_text_file" else "Wrote"
    diff = result.data.get("diff")
    stat = _diff_line_stat(diff)
    target = path if isinstance(path, str) and path.strip() else "file"
    head = f"✓ {label} {target}" + (f" ({stat})" if stat else "") + "."
    if not isinstance(diff, str) or not diff.strip():
        return head
    return f"{head}\n{_inline_diff_summary(diff)}\nFull diff is available in /detail."


def _diff_line_stat(diff: object) -> str:
    if not isinstance(diff, str) or not diff.strip():
        return ""
    added = sum(
        1 for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    removed = sum(
        1 for line in diff.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )
    if not added and not removed:
        return ""
    return f"+{added} -{removed}"


def _inline_diff_summary(diff: str, *, limit: int = 20) -> str:
    lines = diff.strip().splitlines()
    if not lines:
        return "Diff is available in /detail."
    visible = lines[:limit]
    body = "\n".join(visible)
    hidden = len(lines) - len(visible)
    suffix = f"\n... +{hidden} more diff line(s)" if hidden > 0 else ""
    return f"```diff\n{body}{suffix}\n```"


def _tool_preview(result: ModelToolResult | None) -> str:
    if result is None:
        return ""
    compacted = _compacted_ref(result.refs)
    prefix = "Compacted tool output\n\n" if compacted else ""
    content = result.content.strip()
    if content:
        return prefix + _preview_detail(content, 80_000)
    if result.data:
        return prefix + _preview_detail(
            json.dumps(result.data, ensure_ascii=False, indent=2, sort_keys=True),
            80_000,
        )
    if result.refs:
        return prefix + ", ".join(result.refs[:4])
    return ""


def _tool_status(result: ModelToolResult | None) -> str:
    if result is None:
        return "pending"
    if result.is_error:
        if _requires_approval(result):
            return "approval required"
        return "error"
    if _compacted_ref(result.refs):
        return "compacted"
    return "ok"


def _has_detail(result: ModelToolResult) -> bool:
    return bool(result.content.strip() or result.data or result.refs)


def _requires_approval(result: ModelToolResult) -> bool:
    text = " ".join((result.content, *result.refs)).lower()
    return "requires approval" in text or "requires_approval=true" in text


def _compacted_ref(refs: tuple[str, ...]) -> bool:
    return any(
        ref.startswith("tool_output_ref=")
        or ref.startswith("tool_output_id=")
        or ref == "compacted=true"
        for ref in refs
    )


def _argument_summary(arguments: Mapping[str, object]) -> str:
    if not arguments:
        return ""
    preferred = []
    for key in (
        "path",
        "file_path",
        "relative_path",
        "command",
        "query",
        "url",
        "name",
        "tool_name",
        "assignment_id",
    ):
        if key in arguments:
            preferred.append(f"{key}={_short_value(arguments[key])}")
    if preferred:
        return ", ".join(preferred[:4])
    items = tuple(arguments.items())[:4]
    return ", ".join(f"{key}={_short_value(value)}" for key, value in items)


def _short_value(value: object, *, limit: int = 120) -> str:
    if isinstance(value, str):
        text = " ".join(value.split())
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _ref_map(refs: tuple[str, ...]) -> dict[str, str]:
    values: dict[str, str] = {}
    for ref in refs:
        if "=" not in ref:
            continue
        key, value = ref.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _event_kind(kind: str) -> str:
    clean = kind.strip().lower()
    explicit_error = {
        "model_request_failed",
        "tool_exchange_invalid",
        "tool_call_failed",
        "tool_denied",
        "safety_denied",
        "loop_guard_stop",
    }
    explicit_approval = {
        "tool_approval_requested",
        "safety_approval_requested",
    }
    if clean in explicit_error or clean.endswith("_failed") or clean.endswith(".failed"):
        return "error"
    if clean in explicit_approval:
        return "approval"
    return "status"


def _tool_error_summary(text: str, *, fallback: str) -> str:
    clean = " ".join(friendly_error_text(text).split())
    if not clean:
        return fallback
    if "Remote script piped directly to shell is not allowed" in clean:
        return (
            "Deepmate blocked a remote install script from being piped directly into "
            "the shell. Download/install steps must be reviewed before execution."
        )
    if len(clean) <= 240:
        return clean
    return _preview(clean, 200) + "\nUse /detail to view full output."


def _preview(text: str, limit: int) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def _preview_detail(text: str, limit: int) -> str:
    clean = text.strip()
    if len(clean) <= limit:
        return clean
    return clean[: limit - 80].rstrip() + "\n\n... output truncated in TUI detail ..."


def _first_meaningful_line(text: str, *, limit: int) -> str:
    for line in text.splitlines():
        clean = " ".join(line.strip().split())
        if clean:
            return _preview(clean, limit)
    return ""
