"""Runtime status parsing for the TUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock


@dataclass(frozen=True, slots=True)
class TuiStatusView:
    """User-facing view of one runtime status message."""

    title: str
    body: str
    status: str = ""
    important: bool = True


@dataclass(slots=True)
class TuiRuntimeStats:
    """Latest runtime status signals surfaced in the TUI footer."""

    last_status: str = ""
    step_index: int = 0
    input_pressure: float | None = None
    estimated_input_tokens: int | None = None
    context_remaining_input_tokens: int | None = None
    model_context_tokens: int | None = None
    actual_input_tokens: int | None = None
    output_tokens: int | None = None
    cache_hit_ratio: float | None = None
    tool_schema_tokens: int | None = None
    tool_output_ratio: float | None = None
    finish_reason: str = ""
    last_tool_event: str = ""
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def record(self, message: str) -> TuiStatusView:
        """Record one raw runtime status message and return its TUI view."""
        clean = " ".join(message.strip().split())
        view = status_view(clean)
        with self._lock:
            self.last_status = clean
            self._update_from_message(clean)
        return view

    def footer_summary(self) -> str:
        """Return a compact footer summary."""
        with self._lock:
            step_index = self.step_index
            last_tool_event = self.last_tool_event
        parts: list[str] = []
        if step_index:
            parts.append(f"step {step_index}")
        if last_tool_event:
            parts.append(last_tool_event)
        return " | ".join(parts)

    def context_window_summary(self) -> str:
        """Return the persistent context window summary for the session bar."""
        with self._lock:
            input_pressure = self.input_pressure
            estimated = self.estimated_input_tokens
            remaining = self.context_remaining_input_tokens
            model_context = self.model_context_tokens
        if model_context is not None and model_context > 0 and estimated is not None:
            used = max(0, estimated)
            remaining_context = max(0, model_context - used)
            return (
                f"ctx {_percent(used / model_context)} · "
                f"{_compact_context_tokens(remaining_context)} left"
            )
        if remaining is not None:
            return f"ctx {_compact_context_tokens(remaining)} left"
        if input_pressure is None or input_pressure <= 0:
            return "ctx --"
        return f"ctx {_percent(input_pressure)} used"

    def context_usage_ratio(self) -> float | None:
        """Return how full the context window is (0.0–1.0), or None if unknown."""
        with self._lock:
            input_pressure = self.input_pressure
            estimated = self.estimated_input_tokens
            model_context = self.model_context_tokens
        if model_context is not None and model_context > 0 and estimated is not None:
            return max(0.0, min(1.0, max(0, estimated) / model_context))
        if input_pressure is not None and input_pressure > 0:
            return max(0.0, min(1.0, input_pressure))
        return None

    def cache_summary(self) -> str:
        """Return a compact cache-hit label for the footer, or '' if unknown."""
        with self._lock:
            ratio = self.cache_hit_ratio
        if ratio is None or ratio <= 0:
            return ""
        return f"cache {_percent(ratio)}"

    def detail_text(self) -> str:
        """Return a readable status report for `/status`."""
        with self._lock:
            last_status = self.last_status
            step_index = self.step_index
            input_pressure = self.input_pressure
            estimated_input_tokens = self.estimated_input_tokens
            context_remaining_input_tokens = self.context_remaining_input_tokens
            model_context_tokens = self.model_context_tokens
            actual_input_tokens = self.actual_input_tokens
            output_tokens = self.output_tokens
            cache_hit_ratio = self.cache_hit_ratio
            tool_schema_tokens = self.tool_schema_tokens
            tool_output_ratio = self.tool_output_ratio
            finish_reason = self.finish_reason
            last_tool_event = self.last_tool_event
        lines = ["Runtime status"]
        if last_status:
            lines.append(f"- latest: {last_status}")
        if step_index:
            lines.append(f"- current step: {step_index}")
        if input_pressure is not None:
            lines.append(f"- input pressure: {_percent(input_pressure)}")
        if estimated_input_tokens is not None:
            lines.append(f"- estimated input tokens: {estimated_input_tokens}")
        if context_remaining_input_tokens is not None:
            lines.append(
                f"- context remaining input tokens: {context_remaining_input_tokens}"
            )
        if model_context_tokens is not None:
            lines.append(f"- model context tokens: {model_context_tokens}")
        if actual_input_tokens is not None:
            lines.append(f"- actual input tokens: {actual_input_tokens}")
        if output_tokens is not None:
            lines.append(f"- output tokens: {output_tokens}")
        if cache_hit_ratio is not None:
            lines.append(f"- cache hit ratio: {_percent(cache_hit_ratio)}")
        if tool_schema_tokens is not None:
            lines.append(f"- tool schema tokens: {tool_schema_tokens}")
        if tool_output_ratio is not None:
            lines.append(f"- tool output ratio: {_percent(tool_output_ratio)}")
        if finish_reason:
            lines.append(f"- finish reason: {finish_reason}")
        if last_tool_event:
            lines.append(f"- latest tool event: {last_tool_event}")
        if len(lines) == 1:
            lines.append("- no runtime status has been reported yet")
        return "\n".join(lines)

    def _update_from_message(self, message: str) -> None:
        if message.startswith("runtime step "):
            head, _, tail = message.partition(";")
            self.step_index = _trailing_int(head) or self.step_index
            values = _status_values(tail)
            self.input_pressure = _float_value(values, "input_pressure", self.input_pressure)
            self.estimated_input_tokens = _int_value(
                values,
                "estimated_input_tokens",
                self.estimated_input_tokens,
            )
            self.context_remaining_input_tokens = _int_value(
                values,
                "context_remaining_input_tokens",
                self.context_remaining_input_tokens,
            )
            self.model_context_tokens = _int_value(
                values,
                "model_context_tokens",
                self.model_context_tokens,
            )
            self.actual_input_tokens = _int_value(
                values,
                "actual_input_tokens",
                self.actual_input_tokens,
            )
            self.output_tokens = _int_value(values, "output_tokens", self.output_tokens)
            self.cache_hit_ratio = _float_value(
                values,
                "cache_hit_ratio",
                self.cache_hit_ratio,
            )
            self.tool_schema_tokens = _int_value(
                values,
                "tool_schema_tokens",
                self.tool_schema_tokens,
            )
            self.tool_output_ratio = _float_value(
                values,
                "tool_output_ratio",
                self.tool_output_ratio,
            )
            self.finish_reason = values.get("finish_reason", self.finish_reason)
            return
        if message.startswith("tool output compacted:"):
            values = _status_values(message.partition(":")[2])
            tool = values.get("tool", "tool")
            original = values.get("original_tokens", "")
            compacted = values.get("compacted_tokens", "")
            self.last_tool_event = (
                f"compacted {tool}"
                if not original or not compacted
                else f"compacted {tool} {original}->{compacted}"
            )
            return
        if message.startswith("tool output normalized:"):
            values = _status_values(message.partition(":")[2])
            tool = values.get("tool", "tool")
            self.last_tool_event = f"normalized {tool}"
            return
        if message.startswith("tool schema loaded:"):
            self.last_tool_event = "schema loaded"


def status_view(message: str) -> TuiStatusView:
    """Convert one raw status string into a compact display card."""
    clean = " ".join(message.strip().split())
    if clean.startswith("runtime step "):
        head, _, tail = clean.partition(";")
        values = _status_values(tail)
        body = _runtime_step_body(values)
        return TuiStatusView(title=_step_title(head), body=body, status="model", important=False)
    if clean.startswith("model request started:"):
        body = "working on"
        return TuiStatusView(title="Deepmate", body=body, status="model")
    if clean.startswith("tool started:"):
        values = _status_values(clean.partition(":")[2])
        tool = values.get("tool", "tool")
        source = values.get("source", "")
        body = f"calling tool: {tool}"
        if source:
            body += f" · {source}"
        return TuiStatusView(title="Deepmate", body=body, status="tool")
    if clean.startswith("tool finished:"):
        values = _status_values(clean.partition(":")[2])
        tool = values.get("tool", "tool")
        outcome = values.get("outcome", "ok")
        marker = "✓" if outcome != "failed" else "✗"
        return TuiStatusView(
            title="Deepmate",
            body=f"{marker} {tool} {outcome}",
            status="tool",
        )
    if clean.startswith("tool output compacted:"):
        values = _status_values(clean.partition(":")[2])
        return TuiStatusView(
            title="tool output compacted",
            body=_tool_output_body(values, fallback=clean),
            status="compacted",
        )
    if clean.startswith("tool output normalized:"):
        values = _status_values(clean.partition(":")[2])
        return TuiStatusView(
            title="tool output normalized",
            body=_tool_output_body(values, fallback=clean),
            status="normalized",
        )
    if clean.startswith("tool schema loaded:"):
        return TuiStatusView(
            title="tool schema loaded",
            body=clean,
            status="schema",
            important=False,
        )
    if "repair" in clean.lower() or "warning" in clean.lower():
        return TuiStatusView(title="runtime", body=clean, status="warning")
    return TuiStatusView(title="runtime", body=clean, status="", important=False)


def _runtime_step_body(values: dict[str, str]) -> str:
    parts: list[str] = []
    if "input_pressure" in values:
        parts.append(f"input pressure: {_percent_text(values['input_pressure'])}")
    if "estimated_input_tokens" in values:
        parts.append(f"estimated input: {values['estimated_input_tokens']} tokens")
    if "actual_input_tokens" in values:
        actual = f"actual input: {values['actual_input_tokens']} tokens"
        if "output_tokens" in values:
            actual += f", output: {values['output_tokens']}"
        parts.append(actual)
    elif values.get("usage") == "unreported":
        parts.append("provider usage: unreported")
    if "cache_hit_ratio" in values:
        parts.append(f"cache hit: {_percent_text(values['cache_hit_ratio'])}")
    if "tool_schema_tokens" in values:
        parts.append(f"tool schema tokens: {values['tool_schema_tokens']}")
    if "tool_output_ratio" in values:
        parts.append(f"tool output ratio: {_percent_text(values['tool_output_ratio'])}")
    if "finish_reason" in values:
        parts.append(f"finish reason: {values['finish_reason']}")
    return "\n".join(f"- {part}" for part in parts) or "model step completed"


def _step_title(value: str) -> str:
    step = _trailing_int(value)
    return f"step {step}" if step else "step"


def _tool_output_body(values: dict[str, str], *, fallback: str) -> str:
    if not values:
        return fallback
    parts = []
    source = values.get("source", "")
    tool = values.get("tool", "")
    kind = values.get("kind", "")
    if source or tool:
        parts.append(" / ".join(part for part in (source, tool, kind) if part))
    original = values.get("original_tokens", "")
    normalized = values.get("normalized_tokens", "")
    compacted = values.get("compacted_tokens", "")
    if original or normalized or compacted:
        pieces = []
        if original:
            pieces.append(f"original={original}")
        if normalized:
            pieces.append(f"normalized={normalized}")
        if compacted:
            pieces.append(f"compacted={compacted}")
        parts.append("tokens: " + ", ".join(pieces))
    ratio = values.get("compression_ratio") or values.get("normalization")
    if ratio:
        parts.append(f"ratio/detail: {ratio}")
    ref = values.get("ref")
    if ref:
        parts.append(f"retrieval ref: {ref}")
    return "\n".join(f"- {part}" for part in parts) or fallback


def _status_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_part in text.split(";"):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            values[part] = "true"
            continue
        key, value = part.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _int_value(values: dict[str, str], key: str, fallback: int | None) -> int | None:
    try:
        return int(values[key])
    except (KeyError, TypeError, ValueError):
        return fallback


def _float_value(
    values: dict[str, str],
    key: str,
    fallback: float | None,
) -> float | None:
    try:
        return float(values[key])
    except (KeyError, TypeError, ValueError):
        return fallback


def _trailing_int(text: str) -> int | None:
    tail = text.strip().split()[-1:]
    if not tail:
        return None
    try:
        return int(tail[0])
    except ValueError:
        return None


def _percent(value: float) -> str:
    percent = value * 100
    if 0 < percent < 1:
        return "<1%"
    return f"{percent:.0f}%"


def _percent_text(value: str) -> str:
    try:
        return _percent(float(value))
    except ValueError:
        return value


def _compact_int(value: int) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def _compact_context_tokens(value: int) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000:
        scaled = value / 1_000_000
        return f"{scaled:.0f}m" if value % 1_000_000 == 0 else f"{scaled:.1f}m"
    if absolute >= 1_000:
        scaled = value / 1_000
        return f"{scaled:.0f}k" if value % 1_000 == 0 else f"{scaled:.1f}k"
    return str(value)
