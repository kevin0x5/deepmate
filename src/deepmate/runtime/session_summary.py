"""Policy and helpers for session history summary checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from deepmate.domain import Message, MessageRole
from deepmate.providers import (
    ModelConversationItem,
    ModelProvider,
    ModelRequest,
    ModelToolExchange,
    TokenUsage,
)
from deepmate.runtime.conversation_budget import (
    DEFAULT_MODEL_CONTEXT_TOKENS,
    RequestBudgetReport,
    estimate_conversation_item_tokens,
    estimate_text_tokens,
)

DEFAULT_OBSERVE_RATIO = 0.25
DEFAULT_QUALITY_WARNING_RATIO = 0.40
DEFAULT_CHECKPOINT_RATIO = 0.50
DEFAULT_GUARD_RATIO = 0.60
DEFAULT_EMERGENCY_RATIO = 0.75
DEFAULT_LOW_CACHE_HIT_RATIO = 0.60
DEFAULT_TOOL_OUTPUT_CHECKPOINT_RATIO = 0.35
DEFAULT_TOOL_OUTPUT_HIGH_RATIO = 0.50
DEFAULT_SUMMARY_TARGET_TOKENS = 4_000
DEFAULT_LONG_SOURCE_TOKENS = 64_000
DEFAULT_MIN_LONG_SOURCE_SUMMARY_TOKENS = 256
SUMMARY_TOOL_CONTENT_PREVIEW_CHARS = 600
SUMMARY_TOOL_DATA_PREVIEW_CHARS = 300
SUMMARY_TOOL_ARGUMENT_PREVIEW_CHARS = 500
SUMMARY_TOOL_ARGUMENT_TEXT_PREVIEW_CHARS = 120
SUMMARY_TOOL_ARGUMENT_REDACT_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "content",
        "html",
        "input",
        "markdown",
        "new_text",
        "old_text",
        "password",
        "secret",
        "text",
        "token",
    }
)

SUMMARY_SYSTEM_PROMPT = """You compact an earlier part of one Deepmate session.

Return only a concise Markdown summary. Do not use JSON.

Preserve information needed for the next turns:
- user's current goal and explicit constraints
- scoped product/project context such as positioning, target users, phase
  boundaries, design decisions, reasons for changes, and rejected alternatives
- important decisions and assumptions
- files, tools, commands, artifacts, and evidence refs
- errors, blockers, unfinished checks, and open questions
- verified facts and checks separately from unverified assumptions
- concrete next actions needed to continue the work
- recent continuation notes needed to continue work

Do not preserve:
- greetings or repeated small talk
- large raw tool outputs
- secrets, credentials, tokens, private keys, payment data, or full addresses
- unverified guesses as facts

When preserving product or project context, keep the product, project, task, or
session scope explicit. Do not rewrite scoped product decisions as global
user/profile memory.

The summary is context for the same session, not a new user request.
"""


class SessionSummaryAction(StrEnum):
    """Decision action for session summary checkpointing."""

    SKIP = "skip"
    OBSERVE = "observe"
    CHECKPOINT = "checkpoint"


@dataclass(frozen=True, slots=True)
class SessionSummaryPolicy:
    """Thresholds for lightweight session summary checkpoint decisions.

    The *_tokens fields are explicit absolute overrides. When they are unset,
    thresholds are derived from the current model context window via *_ratio.
    """

    enabled: bool = True
    observe_tokens: int | None = None
    quality_warning_tokens: int | None = None
    checkpoint_tokens: int | None = None
    guard_tokens: int | None = None
    emergency_tokens: int | None = None
    observe_ratio: float = DEFAULT_OBSERVE_RATIO
    quality_warning_ratio: float = DEFAULT_QUALITY_WARNING_RATIO
    checkpoint_ratio: float = DEFAULT_CHECKPOINT_RATIO
    guard_ratio: float = DEFAULT_GUARD_RATIO
    emergency_ratio: float = DEFAULT_EMERGENCY_RATIO
    low_cache_hit_ratio: float = DEFAULT_LOW_CACHE_HIT_RATIO
    tool_output_checkpoint_ratio: float = DEFAULT_TOOL_OUTPUT_CHECKPOINT_RATIO
    tool_output_high_ratio: float = DEFAULT_TOOL_OUTPUT_HIGH_RATIO

    def normalized(
        self,
        model_context_tokens: int = DEFAULT_MODEL_CONTEXT_TOKENS,
        usable_input_tokens: int | None = None,
    ) -> "SessionSummaryPolicy":
        """Return a policy with resolved monotonic thresholds and sane ratios."""
        context_tokens = max(1, model_context_tokens)
        usable_tokens = max(1, usable_input_tokens) if usable_input_tokens else None
        observe = _threshold_tokens(
            self.observe_tokens,
            self.observe_ratio,
            context_tokens,
            usable_tokens,
        )
        quality = max(
            observe,
            _threshold_tokens(
                self.quality_warning_tokens,
                self.quality_warning_ratio,
                context_tokens,
                usable_tokens,
            ),
        )
        checkpoint = max(
            quality,
            _threshold_tokens(
                self.checkpoint_tokens,
                self.checkpoint_ratio,
                context_tokens,
                usable_tokens,
            ),
        )
        guard = max(
            checkpoint,
            _threshold_tokens(
                self.guard_tokens,
                self.guard_ratio,
                context_tokens,
                usable_tokens,
            ),
        )
        emergency = max(
            guard,
            _threshold_tokens(
                self.emergency_tokens,
                self.emergency_ratio,
                context_tokens,
                usable_tokens,
            ),
        )
        tool_checkpoint_ratio = _clamped_ratio(self.tool_output_checkpoint_ratio)
        tool_high_ratio = max(
            tool_checkpoint_ratio,
            _clamped_ratio(self.tool_output_high_ratio),
        )
        return SessionSummaryPolicy(
            enabled=self.enabled,
            observe_tokens=observe,
            quality_warning_tokens=quality,
            checkpoint_tokens=checkpoint,
            guard_tokens=guard,
            emergency_tokens=emergency,
            observe_ratio=_effective_context_ratio(observe, context_tokens),
            quality_warning_ratio=_effective_context_ratio(quality, context_tokens),
            checkpoint_ratio=_effective_context_ratio(checkpoint, context_tokens),
            guard_ratio=_effective_context_ratio(guard, context_tokens),
            emergency_ratio=_effective_context_ratio(emergency, context_tokens),
            low_cache_hit_ratio=_clamped_ratio(self.low_cache_hit_ratio),
            tool_output_checkpoint_ratio=tool_checkpoint_ratio,
            tool_output_high_ratio=tool_high_ratio,
        )


@dataclass(frozen=True, slots=True)
class SessionSummaryDecision:
    """Decision produced after inspecting request pressure and runtime signals."""

    action: SessionSummaryAction
    reason: str
    refs: tuple[str, ...] = field(default_factory=tuple)

    def should_checkpoint(self) -> bool:
        """Return whether the caller should run a summary checkpoint."""
        return self.action == SessionSummaryAction.CHECKPOINT

    def trace_refs(self) -> tuple[str, ...]:
        """Return compact refs suitable for trace events."""
        return (
            f"summary_action={self.action.value}",
            f"summary_reason={self.reason}",
            f"summary_should_checkpoint={str(self.should_checkpoint()).lower()}",
            *self.refs,
        )


@dataclass(frozen=True, slots=True)
class SessionSummarySourceItem:
    """One transcript item selected as source material for a summary."""

    sequence: int
    item: ModelConversationItem

    def is_ready(self) -> bool:
        """Return whether the source item can be summarized."""
        return self.sequence > 0 and self.item.is_ready()


@dataclass(frozen=True, slots=True)
class SessionSummaryInput:
    """Provider-neutral input used to create the next session summary."""

    source_items: tuple[SessionSummarySourceItem, ...]
    previous_summary: str = ""
    previous_covered_item_count: int = 0
    target_tokens: int = DEFAULT_SUMMARY_TARGET_TOKENS

    def is_ready(self) -> bool:
        """Return whether there is new source material to summarize."""
        return bool(
            self.source_items and all(item.is_ready() for item in self.source_items)
        )

    def covered_until_sequence(self) -> int:
        """Return the transcript sequence covered by the resulting summary."""
        if not self.source_items:
            return 0
        return max(item.sequence for item in self.source_items)

    def source_item_count(self) -> int:
        """Return how many new transcript items are summarized."""
        return len(self.source_items)

    def covered_item_count(self) -> int:
        """Return total raw transcript items covered by the resulting summary."""
        return self.previous_covered_item_count + self.source_item_count()

    def estimated_source_tokens(self) -> int:
        """Return estimated tokens for the new source segment."""
        return sum(
            estimate_conversation_item_tokens(item.item) for item in self.source_items
        )


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """Generated summary plus source coverage metadata."""

    content: str
    covered_until_sequence: int
    covered_item_count: int
    source_item_count: int
    estimated_source_tokens: int
    source_model: str
    usage: TokenUsage | None = None

    def is_ready(self) -> bool:
        """Return whether the summary can replace older model-facing history."""
        return bool(
            self.content.strip()
            and self.covered_until_sequence > 0
            and self.covered_item_count > 0
            and self.source_item_count > 0
            and self.estimated_source_tokens >= 0
            and self.source_model.strip()
        )


def decide_session_summary(
    request_budget: RequestBudgetReport,
    policy: SessionSummaryPolicy | None = None,
    cache_hit_ratio: float | None = None,
    profile_context_changed: bool = False,
    history_trimmed: bool = False,
) -> SessionSummaryDecision:
    """Return a deterministic session-summary checkpoint decision."""
    normalized_policy = (policy or SessionSummaryPolicy()).normalized(
        model_context_tokens=request_budget.model_context_tokens,
        usable_input_tokens=request_budget.usable_input_tokens,
    )
    if not normalized_policy.enabled:
        return _decision(
            SessionSummaryAction.SKIP,
            "disabled",
            request_budget,
            normalized_policy,
            cache_hit_ratio,
            profile_context_changed,
            history_trimmed,
        )

    tokens = request_budget.estimated_input_tokens
    tool_ratio = request_budget.tool_output_ratio
    low_cache = (
        cache_hit_ratio is not None
        and cache_hit_ratio < normalized_policy.low_cache_hit_ratio
    )

    if history_trimmed:
        return _decision(
            SessionSummaryAction.CHECKPOINT,
            "history_trimmed",
            request_budget,
            normalized_policy,
            cache_hit_ratio,
            profile_context_changed,
            history_trimmed,
        )
    if tokens >= normalized_policy.emergency_tokens:
        return _decision(
            SessionSummaryAction.CHECKPOINT,
            "emergency_tokens",
            request_budget,
            normalized_policy,
            cache_hit_ratio,
            profile_context_changed,
            history_trimmed,
        )
    if tokens >= normalized_policy.guard_tokens:
        return _decision(
            SessionSummaryAction.CHECKPOINT,
            "guard_tokens",
            request_budget,
            normalized_policy,
            cache_hit_ratio,
            profile_context_changed,
            history_trimmed,
        )
    if tokens >= normalized_policy.checkpoint_tokens:
        return _decision(
            SessionSummaryAction.CHECKPOINT,
            "checkpoint_tokens",
            request_budget,
            normalized_policy,
            cache_hit_ratio,
            profile_context_changed,
            history_trimmed,
        )
    if tool_ratio >= normalized_policy.tool_output_high_ratio:
        return _decision(
            SessionSummaryAction.CHECKPOINT,
            "tool_output_high",
            request_budget,
            normalized_policy,
            cache_hit_ratio,
            profile_context_changed,
            history_trimmed,
        )
    if tokens >= normalized_policy.quality_warning_tokens:
        if tool_ratio >= normalized_policy.tool_output_checkpoint_ratio:
            return _decision(
                SessionSummaryAction.CHECKPOINT,
                "tool_output_checkpoint",
                request_budget,
                normalized_policy,
                cache_hit_ratio,
                profile_context_changed,
                history_trimmed,
            )
        if low_cache:
            return _decision(
                SessionSummaryAction.CHECKPOINT,
                "low_cache_hit_ratio",
                request_budget,
                normalized_policy,
                cache_hit_ratio,
                profile_context_changed,
                history_trimmed,
            )
        if profile_context_changed:
            return _decision(
                SessionSummaryAction.OBSERVE,
                "profile_pending_quality_warning",
                request_budget,
                normalized_policy,
                cache_hit_ratio,
                profile_context_changed,
                history_trimmed,
            )
        return _decision(
            SessionSummaryAction.OBSERVE,
            "quality_warning_tokens",
            request_budget,
            normalized_policy,
            cache_hit_ratio,
            profile_context_changed,
            history_trimmed,
        )
    if tokens >= normalized_policy.observe_tokens:
        return _decision(
            SessionSummaryAction.OBSERVE,
            "observe_tokens",
            request_budget,
            normalized_policy,
            cache_hit_ratio,
            profile_context_changed,
            history_trimmed,
        )
    return _decision(
        SessionSummaryAction.SKIP,
        "below_observe_tokens",
        request_budget,
        normalized_policy,
        cache_hit_ratio,
        profile_context_changed,
        history_trimmed,
    )


def _decision(
    action: SessionSummaryAction,
    reason: str,
    request_budget: RequestBudgetReport,
    policy: SessionSummaryPolicy,
    cache_hit_ratio: float | None,
    profile_context_changed: bool,
    history_trimmed: bool,
) -> SessionSummaryDecision:
    refs = (
        f"estimated_input_tokens={request_budget.estimated_input_tokens}",
        f"tool_output_ratio={request_budget.tool_output_ratio:.4f}",
        f"cache_hit_ratio={_ratio_ref(cache_hit_ratio)}",
        f"profile_context_changed={str(profile_context_changed).lower()}",
        f"history_trimmed={str(history_trimmed).lower()}",
        f"observe_tokens={policy.observe_tokens}",
        f"quality_warning_tokens={policy.quality_warning_tokens}",
        f"checkpoint_tokens={policy.checkpoint_tokens}",
        f"guard_tokens={policy.guard_tokens}",
        f"emergency_tokens={policy.emergency_tokens}",
        f"observe_ratio={policy.observe_ratio:.4f}",
        f"quality_warning_ratio={policy.quality_warning_ratio:.4f}",
        f"checkpoint_ratio={policy.checkpoint_ratio:.4f}",
        f"guard_ratio={policy.guard_ratio:.4f}",
        f"emergency_ratio={policy.emergency_ratio:.4f}",
        f"model_context_tokens={request_budget.model_context_tokens}",
        f"usable_input_tokens={request_budget.usable_input_tokens}",
    )
    return SessionSummaryDecision(action=action, reason=reason, refs=refs)


def generate_session_summary(
    provider: ModelProvider,
    model: str,
    summary_input: SessionSummaryInput,
    options: dict[str, object] | None = None,
) -> SessionSummary:
    """Generate and validate a compact summary from selected source items."""
    request = build_session_summary_request(
        model=model,
        summary_input=summary_input,
        options=options,
    )
    response = provider.complete(request)
    validate_session_summary_response(
        response.content,
        response.finish_reason,
        summary_input,
    )
    content = _text(response.content)
    summary = SessionSummary(
        content=content,
        covered_until_sequence=summary_input.covered_until_sequence(),
        covered_item_count=summary_input.covered_item_count(),
        source_item_count=summary_input.source_item_count(),
        estimated_source_tokens=summary_input.estimated_source_tokens(),
        source_model=_text(model),
        usage=response.usage,
    )
    if not summary.is_ready():
        raise ValueError("session summary is not ready")
    return summary


def build_session_summary_request(
    model: str,
    summary_input: SessionSummaryInput,
    options: dict[str, object] | None = None,
) -> ModelRequest:
    """Build the provider request used for summary generation."""
    clean_model = _text(model)
    if not clean_model:
        raise ValueError("summary model is required")
    if not summary_input.is_ready():
        raise ValueError("session summary input requires source items")
    request = ModelRequest(
        model=clean_model,
        conversation=(
            ModelConversationItem.from_message(
                Message(role=MessageRole.SYSTEM, content=SUMMARY_SYSTEM_PROMPT)
            ),
            ModelConversationItem.from_message(
                Message(
                    role=MessageRole.USER,
                    content=_summary_user_prompt(summary_input),
                )
            ),
        ),
        options={
            "temperature": 0,
            "max_tokens": 5_000,
            **dict(options or {}),
        },
    )
    if not request.is_ready():
        raise ValueError("session summary request is not ready")
    return request


def validate_session_summary_response(
    content: str,
    finish_reason: str,
    summary_input: SessionSummaryInput,
) -> None:
    """Raise when a generated summary is unsafe to persist."""
    if _text(finish_reason) == "length":
        raise ValueError("session summary response was truncated")
    if not _text(content):
        raise ValueError("session summary response is empty")
    if (
        summary_input.estimated_source_tokens() >= DEFAULT_LONG_SOURCE_TOKENS
        and estimate_text_tokens(content) < DEFAULT_MIN_LONG_SOURCE_SUMMARY_TOKENS
    ):
        raise ValueError("session summary is too short for a long source segment")


def session_summary_to_conversation_item(
    content: str,
    summary_id: str = "",
) -> ModelConversationItem:
    """Return the synthetic model-facing item for an active session summary."""
    clean_content = _text(content)
    if not clean_content:
        raise ValueError("session summary content is required")
    prefix = (
        "The following is a Deepmate-generated summary of earlier conversation "
        "in this session. It is context, not a new user request."
    )
    if summary_id.strip():
        prefix += f"\nSummary id: {summary_id.strip()}."
    return ModelConversationItem.from_message(
        Message(
            role=MessageRole.ASSISTANT,
            content=f"{prefix}\n\n{clean_content}",
        )
    )


def _clamped_ratio(value: float) -> float:
    return min(1.0, max(0.0, value))


def _effective_context_ratio(tokens: int, model_context_tokens: int) -> float:
    return _clamped_ratio(max(1, tokens) / max(1, model_context_tokens))


def _threshold_tokens(
    explicit_tokens: int | None,
    ratio: float,
    model_context_tokens: int,
    usable_input_tokens: int | None,
) -> int:
    if explicit_tokens is not None and explicit_tokens > 0:
        tokens = max(1, explicit_tokens)
    else:
        tokens = max(1, int(model_context_tokens * _clamped_ratio(ratio)))
    if usable_input_tokens is not None:
        tokens = min(tokens, usable_input_tokens)
    return max(1, tokens)


def _ratio_ref(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{_clamped_ratio(value):.4f}"


def _summary_user_prompt(summary_input: SessionSummaryInput) -> str:
    lines = [
        "Create the next session summary from the source segment below.",
        f"Target length: about {max(1, summary_input.target_tokens)} tokens.",
        "",
        "Required Markdown headings:",
        "## Session Summary",
        "### User Goal",
        "### Product Or Project Context",
        "### Current State",
        "### Decisions And Constraints",
        "### Files, Tools, And Artifacts",
        "### Evidence And References",
        "### Verified And Unverified State",
        "### Open Questions Or Blockers",
        "### Next Actions",
        "### Recent Continuation Notes",
    ]
    if summary_input.previous_summary.strip():
        lines.extend(
            (
                "",
                "Previous summary to preserve and update:",
                summary_input.previous_summary.strip(),
            )
        )
    lines.extend(
        (
            "",
            "New source segment:",
            _render_source_items(summary_input.source_items),
        )
    )
    return "\n".join(lines).strip()


def _render_source_items(items: tuple[SessionSummarySourceItem, ...]) -> str:
    sections: list[str] = []
    for source in items:
        sections.append(_render_source_item(source))
    return "\n\n".join(sections)


def _render_source_item(source: SessionSummarySourceItem) -> str:
    item = source.item
    if item.message is not None:
        return (
            f"### Transcript item {source.sequence}: {item.message.role.value}\n"
            f"{_text(item.message.content)}"
        )
    if item.tool_exchange is not None:
        return _render_tool_exchange(source.sequence, item.tool_exchange)
    return f"### Transcript item {source.sequence}: empty"


def _render_tool_exchange(sequence: int, exchange: ModelToolExchange) -> str:
    lines = [f"### Transcript item {sequence}: tool_exchange"]
    assistant_content = _text(exchange.assistant_content)
    assistant_reasoning = _text(exchange.assistant_reasoning)
    if assistant_content:
        lines.extend(("", "Assistant content:", assistant_content))
    if assistant_reasoning:
        lines.extend(("", "Assistant reasoning:", assistant_reasoning))
    for request in _iter_values(exchange.tool_requests):
        lines.extend(
            (
                "",
                f"Tool request: {_text(getattr(request, 'name', ''))}",
                f"id: {_text(getattr(request, 'id', ''))}",
                f"arguments: {_compact_tool_arguments(request)}",
            )
        )
    for result in _iter_values(exchange.tool_results):
        result_text = _text(getattr(result, "content", ""))
        data = getattr(result, "data", {}) or {}
        refs = _iter_values(getattr(result, "refs", ()))
        result_preview = _compact_tool_result_text(result_text)
        if not result_text and data:
            result_preview = _compact_tool_data_preview(data)
        if not result_preview and data:
            result_preview = _compact_tool_data_preview(data)
        lines.extend(
            (
                "",
                f"Tool result: {_text(getattr(result, 'name', ''))}",
                f"request_id: {_text(getattr(result, 'request_id', ''))}",
                f"is_error: {str(bool(getattr(result, 'is_error', False))).lower()}",
                f"refs: {', '.join(str(ref).strip() for ref in refs if str(ref).strip()) or '-'}",
                "content_summary:",
                result_preview or "(empty)",
            )
        )
    return "\n".join(lines)


def _compact_tool_result_text(text: str) -> str:
    clean = " ".join(text.split()).strip()
    if not clean:
        return ""
    if len(clean) <= SUMMARY_TOOL_CONTENT_PREVIEW_CHARS:
        return clean
    return (
        clean[:SUMMARY_TOOL_CONTENT_PREVIEW_CHARS].rstrip()
        + f"... [truncated {len(clean) - SUMMARY_TOOL_CONTENT_PREVIEW_CHARS} chars]"
    )


def _compact_tool_data_preview(data: object) -> str:
    if not isinstance(data, dict) or not data:
        return ""
    preview = " ".join(str(dict(data)).split()).strip()
    if len(preview) <= SUMMARY_TOOL_DATA_PREVIEW_CHARS:
        return preview
    return (
        preview[:SUMMARY_TOOL_DATA_PREVIEW_CHARS].rstrip()
        + f"... [truncated {len(preview) - SUMMARY_TOOL_DATA_PREVIEW_CHARS} chars]"
    )


def _compact_tool_arguments(request: object) -> str:
    arguments = getattr(request, "arguments", {}) or {}
    if isinstance(arguments, Mapping) and arguments:
        return _compact_argument_payload(arguments)
    raw = _text(getattr(request, "raw_arguments", ""))
    if raw:
        return _compact_argument_text(raw)
    return "{}"


def _compact_argument_payload(arguments: Mapping[object, object]) -> str:
    safe: dict[str, object] = {}
    for key, value in sorted(arguments.items(), key=lambda item: str(item[0])):
        clean_key = str(key).strip()
        if clean_key:
            safe[clean_key] = _compact_argument_value(clean_key, value)
    clean = " ".join(str(safe).split()).strip()
    if not clean:
        return "{}"
    if len(clean) <= SUMMARY_TOOL_ARGUMENT_PREVIEW_CHARS:
        return clean
    return (
        clean[:SUMMARY_TOOL_ARGUMENT_PREVIEW_CHARS].rstrip()
        + f"... [truncated {len(clean) - SUMMARY_TOOL_ARGUMENT_PREVIEW_CHARS} chars]"
    )


def _compact_argument_value(key: str, value: object) -> object:
    clean_key = key.strip().lower()
    if isinstance(value, str):
        clean_value = " ".join(value.split()).strip()
        if clean_key in SUMMARY_TOOL_ARGUMENT_REDACT_KEYS:
            return f"<omitted {len(value)} chars>"
        if len(clean_value) > SUMMARY_TOOL_ARGUMENT_TEXT_PREVIEW_CHARS:
            return (
                clean_value[:SUMMARY_TOOL_ARGUMENT_TEXT_PREVIEW_CHARS].rstrip()
                + f"... [truncated {len(clean_value) - SUMMARY_TOOL_ARGUMENT_TEXT_PREVIEW_CHARS} chars]"
            )
        return clean_value
    if isinstance(value, dict):
        return {
            str(child_key): _compact_argument_value(str(child_key), child_value)
            for child_key, child_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_compact_argument_value(key, child) for child in value[:8]]
    return value


def _compact_argument_text(text: str) -> str:
    clean = " ".join(text.split()).strip()
    if not clean:
        return "{}"
    for marker in SUMMARY_TOOL_ARGUMENT_REDACT_KEYS:
        if f'"{marker}"' in clean or f"'{marker}'" in clean:
            return f"<omitted raw arguments {len(text)} chars>"
    if len(clean) <= SUMMARY_TOOL_ARGUMENT_PREVIEW_CHARS:
        return clean
    return (
        clean[:SUMMARY_TOOL_ARGUMENT_PREVIEW_CHARS].rstrip()
        + f"... [truncated {len(clean) - SUMMARY_TOOL_ARGUMENT_PREVIEW_CHARS} chars]"
    )


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _iter_values(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return ()
