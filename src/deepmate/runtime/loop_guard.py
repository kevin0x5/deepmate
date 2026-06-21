"""Lightweight guardrails for long-running agent turns."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from deepmate.runtime.conversation_budget import RequestBudgetReport

DEFAULT_HARD_STEP_CAP = 100


class LoopGuardStopReason(StrEnum):
    """Stable reasons for stopping a user turn before normal completion."""

    CONTEXT_EXHAUSTED = "context_exhausted"
    HARD_STEP_CAP = "hard_step_cap"


@dataclass(frozen=True, slots=True)
class LoopGuardPolicy:
    """Minimal long-task safety policy.

    The step cap is a last-resort fuse, not the normal task budget.
    """

    enabled: bool = True
    hard_step_cap: int = DEFAULT_HARD_STEP_CAP

    def normalized(self) -> "LoopGuardPolicy":
        """Return a policy with safe minimum values."""
        return LoopGuardPolicy(
            enabled=self.enabled,
            hard_step_cap=max(1, self.hard_step_cap),
        )


@dataclass(frozen=True, slots=True)
class ContextMeter:
    """User-facing context pressure summary."""

    estimated_input_tokens: int
    usable_input_tokens: int
    pressure_ratio: float

    def used_percent(self) -> int:
        """Return rounded context usage percentage."""
        return max(0, int(round(max(0.0, self.pressure_ratio) * 100)))

    def remaining_input_tokens(self) -> int:
        """Return rough remaining input tokens before the usable window is full."""
        return max(0, self.usable_input_tokens - self.estimated_input_tokens)

    def status_label(self) -> str:
        """Return a compact user-facing label."""
        percent = self.used_percent()
        if self.pressure_ratio >= 1.0:
            return f"context exhausted ({percent}% used)"
        if self.pressure_ratio >= 0.95:
            return f"context critical ({percent}% used)"
        if self.pressure_ratio >= 0.75:
            return f"context tight ({percent}% used)"
        if self.pressure_ratio >= 0.50:
            return f"context high ({percent}% used)"
        return f"context ok ({percent}% used)"

    def trace_refs(self) -> tuple[str, ...]:
        """Return compact refs for trace events."""
        return (
            f"context_used_percent={self.used_percent()}",
            f"context_remaining_input_tokens={self.remaining_input_tokens()}",
            f"context_pressure_ratio={self.pressure_ratio:.4f}",
        )


@dataclass(frozen=True, slots=True)
class ContinuationNote:
    """A compact recovery note for a non-normal turn stop."""

    stop_reason: LoopGuardStopReason
    content: str

    def is_ready(self) -> bool:
        """Return whether the note contains useful continuation text."""
        return bool(self.content.strip())

    def trace_refs(self) -> tuple[str, ...]:
        """Return compact refs for trace events."""
        return (
            f"stop_reason={self.stop_reason.value}",
            f"continuation_note_chars={len(self.content)}",
        )


@dataclass(frozen=True, slots=True)
class LoopGuardStop:
    """Structured stop produced by loop guard."""

    reason: LoopGuardStopReason
    message: str
    continuation_note: ContinuationNote
    context_meter: ContextMeter | None = None

    def is_ready(self) -> bool:
        """Return whether this stop can be surfaced and persisted."""
        return bool(self.reason.value and self.message.strip())

    def trace_refs(self) -> tuple[str, ...]:
        """Return compact refs for trace events."""
        refs: tuple[str, ...] = (
            f"loop_guard_stop_reason={self.reason.value}",
            *self.continuation_note.trace_refs(),
        )
        if self.context_meter is not None:
            refs = (*refs, *self.context_meter.trace_refs())
        return refs


def context_meter(report: RequestBudgetReport) -> ContextMeter:
    """Return the user-facing pressure summary for one model request."""
    return ContextMeter(
        estimated_input_tokens=report.estimated_input_tokens,
        usable_input_tokens=report.usable_input_tokens,
        pressure_ratio=report.pressure_ratio,
    )


def evaluate_request_preflight(
    report: RequestBudgetReport,
    policy: LoopGuardPolicy | None = None,
) -> LoopGuardStop | None:
    """Return a stop when a provider request would exceed the usable context."""
    active_policy = (policy or LoopGuardPolicy()).normalized()
    if not active_policy.enabled:
        return None
    meter = context_meter(report)
    if report.estimated_input_tokens <= report.usable_input_tokens:
        return None
    message = (
        "The next model request would exceed the usable context window. "
        "Deepmate stopped before calling the provider and saved continuation "
        "context for the next turn."
    )
    note = build_continuation_note(
        reason=LoopGuardStopReason.CONTEXT_EXHAUSTED,
        progress="The turn stopped before the next model request was sent.",
        remaining="Continue from the latest visible transcript and checkpoint context.",
        avoid="Do not resend the same oversized request without reducing context.",
        next_action=(
            "Resume with the saved continuation context, or start a focused follow-up "
            "if this session has become too large."
        ),
    )
    return LoopGuardStop(
        reason=LoopGuardStopReason.CONTEXT_EXHAUSTED,
        message=message,
        continuation_note=note,
        context_meter=meter,
    )


def build_hard_cap_stop(
    *,
    step_count: int,
    policy: LoopGuardPolicy | None = None,
) -> LoopGuardStop:
    """Return a structured stop for the hard step fuse."""
    active_policy = (policy or LoopGuardPolicy()).normalized()
    message = (
        f"Deepmate reached the safety step cap ({step_count}/"
        f"{active_policy.hard_step_cap}) before a final answer. Current progress "
        "was saved so the task can be continued."
    )
    note = build_continuation_note(
        reason=LoopGuardStopReason.HARD_STEP_CAP,
        progress=f"The previous turn reached {step_count} agent step(s).",
        remaining="Continue from the last successful model/tool result.",
        avoid="Avoid repeating identical failed or suppressed tool calls.",
        next_action="Summarize the current state briefly, then take the next useful step.",
    )
    return LoopGuardStop(
        reason=LoopGuardStopReason.HARD_STEP_CAP,
        message=message,
        continuation_note=note,
    )


def build_continuation_note(
    *,
    reason: LoopGuardStopReason,
    progress: str,
    remaining: str,
    avoid: str,
    next_action: str,
) -> ContinuationNote:
    """Build a small continuation note for checkpoint resume context."""
    content = "\n".join(
        (
            f"Stop reason: {reason.value}",
            f"Progress: {_clean(progress)}",
            f"Remaining: {_clean(remaining)}",
            f"Avoid: {_clean(avoid)}",
            f"Next action: {_clean(next_action)}",
        )
    )
    return ContinuationNote(stop_reason=reason, content=content)


def is_explicit_continue_prompt(prompt: object) -> bool:
    """Return whether a user prompt is an explicit request to resume."""
    if not isinstance(prompt, str):
        return False
    normalized = " ".join(prompt.strip().lower().split())
    return normalized in {
        "继续",
        "继续执行",
        "继续这个任务",
        "接着做",
        "continue",
        "go on",
    }


def _clean(value: str) -> str:
    clean = " ".join(str(value).split())
    return clean or "(not available)"
