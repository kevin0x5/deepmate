"""Estimate model-facing conversation and request budget usage."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass

from deepmate.domain import MessageRole
from deepmate.foundation import estimate_text_tokens
from deepmate.providers import (
    ModelConversationItem,
    ModelRequest,
    ModelToolExchange,
    ModelToolResult,
)

DEFAULT_HISTORY_TOKEN_BUDGET = 750_000
DEFAULT_PROTECT_RECENT_ITEMS = 40
DEFAULT_HISTORY_WINDOW_MODE = "warn"
DEFAULT_MODEL_CONTEXT_TOKENS = 1_000_000
DEFAULT_RESPONSE_TOKEN_RESERVE = 64_000
DEFAULT_SAFETY_MARGIN_TOKENS = 50_000
FALLBACK_RESERVED_CONTEXT_RATIO = 0.15
MESSAGE_OVERHEAD_TOKENS = 4
TOOL_SCHEMA_OVERHEAD_TOKENS = 8
TRIM_HISTORY_WINDOW_MODE = "trim"


@dataclass(frozen=True, slots=True)
class ConversationBudgetPolicy:
    """Budget policy for model-facing conversation history."""

    history_token_budget: int = DEFAULT_HISTORY_TOKEN_BUDGET
    protect_recent_items: int = DEFAULT_PROTECT_RECENT_ITEMS
    history_window_mode: str = DEFAULT_HISTORY_WINDOW_MODE
    model_context_tokens: int = DEFAULT_MODEL_CONTEXT_TOKENS
    response_token_reserve: int = DEFAULT_RESPONSE_TOKEN_RESERVE
    safety_margin_tokens: int = DEFAULT_SAFETY_MARGIN_TOKENS

    def normalized(self) -> "ConversationBudgetPolicy":
        """Return a policy with safe minimum values."""
        mode = self.history_window_mode.strip().lower() or DEFAULT_HISTORY_WINDOW_MODE
        if mode not in {DEFAULT_HISTORY_WINDOW_MODE, TRIM_HISTORY_WINDOW_MODE}:
            mode = DEFAULT_HISTORY_WINDOW_MODE
        model_context_tokens = max(1, self.model_context_tokens)
        response_token_reserve, safety_margin_tokens = _effective_context_reserves(
            model_context_tokens,
            max(0, self.response_token_reserve),
            max(0, self.safety_margin_tokens),
        )
        usable_input_tokens = max(
            1,
            model_context_tokens - response_token_reserve - safety_margin_tokens,
        )
        return ConversationBudgetPolicy(
            history_token_budget=min(max(1, self.history_token_budget), usable_input_tokens),
            protect_recent_items=max(0, self.protect_recent_items),
            history_window_mode=mode,
            model_context_tokens=model_context_tokens,
            response_token_reserve=response_token_reserve,
            safety_margin_tokens=safety_margin_tokens,
        )


@dataclass(frozen=True, slots=True)
class ConversationBudgetReport:
    """Estimated history budget usage for one model request."""

    conversation_items: int
    selected_conversation_items: int
    estimated_history_tokens: int
    selected_history_tokens: int
    history_token_budget: int
    protect_recent_items: int
    history_window_mode: str
    would_drop_count: int
    dropped_count: int
    tool_exchange_count: int
    over_budget: bool
    trimmed: bool

    def trace_refs(self) -> tuple[str, ...]:
        """Return compact refs suitable for trace events."""
        return (
            f"conversation_items={self.conversation_items}",
            f"selected_conversation_items={self.selected_conversation_items}",
            f"estimated_history_tokens={self.estimated_history_tokens}",
            f"selected_history_tokens={self.selected_history_tokens}",
            f"history_token_budget={self.history_token_budget}",
            f"protect_recent_items={self.protect_recent_items}",
            f"history_window_mode={self.history_window_mode}",
            f"would_drop_count={self.would_drop_count}",
            f"dropped_count={self.dropped_count}",
            f"tool_exchange_count={self.tool_exchange_count}",
            f"over_budget={str(self.over_budget).lower()}",
            f"trimmed={str(self.trimmed).lower()}",
        )


@dataclass(frozen=True, slots=True)
class ConversationWindowSelection:
    """Conversation history selected for one model request."""

    conversation: tuple[ModelConversationItem, ...]
    report: ConversationBudgetReport


@dataclass(frozen=True, slots=True)
class RequestBudgetReport:
    """Estimated full model request input budget usage."""

    conversation_items: int
    tool_schema_count: int
    estimated_input_tokens: int
    estimated_system_tokens: int
    estimated_history_tokens: int
    estimated_tool_output_tokens: int
    estimated_tool_schema_tokens: int
    model_context_tokens: int
    response_token_reserve: int
    safety_margin_tokens: int
    usable_input_tokens: int
    pressure_ratio: float
    tool_output_ratio: float

    def trace_refs(self) -> tuple[str, ...]:
        """Return compact refs suitable for trace events."""
        return (
            f"conversation_items={self.conversation_items}",
            f"tool_schema_count={self.tool_schema_count}",
            f"estimated_input_tokens={self.estimated_input_tokens}",
            f"estimated_system_tokens={self.estimated_system_tokens}",
            f"estimated_history_tokens={self.estimated_history_tokens}",
            f"estimated_tool_output_tokens={self.estimated_tool_output_tokens}",
            f"estimated_tool_schema_tokens={self.estimated_tool_schema_tokens}",
            f"model_context_tokens={self.model_context_tokens}",
            f"response_token_reserve={self.response_token_reserve}",
            f"safety_margin_tokens={self.safety_margin_tokens}",
            f"usable_input_tokens={self.usable_input_tokens}",
            f"pressure_ratio={self.pressure_ratio:.4f}",
            f"tool_output_ratio={self.tool_output_ratio:.4f}",
        )


def build_conversation_budget_report(
    conversation: Iterable[ModelConversationItem],
    policy: ConversationBudgetPolicy | None = None,
) -> ConversationBudgetReport:
    """Estimate history budget usage without trimming conversation items."""
    return select_conversation_window(conversation, policy).report


def build_request_budget_report(
    request: ModelRequest,
    policy: ConversationBudgetPolicy | None = None,
) -> RequestBudgetReport:
    """Estimate full provider request input usage."""
    normalized_policy = (policy or ConversationBudgetPolicy()).normalized()
    conversation_tokens = tuple(
        estimate_conversation_item_tokens(item) for item in request.conversation
    )
    system_tokens = _system_tokens(request.conversation, conversation_tokens)
    history_tokens = max(0, sum(conversation_tokens) - system_tokens)
    tool_output_tokens = sum(
        estimate_conversation_item_tool_output_tokens(item)
        for item in request.conversation
    )
    tool_schema_tokens = sum(
        estimate_tool_schema_tokens(schema) for schema in request.tool_schemas
    )
    estimated_input_tokens = sum(conversation_tokens) + tool_schema_tokens
    usable_input_tokens = max(
        1,
        normalized_policy.model_context_tokens
        - normalized_policy.response_token_reserve
        - normalized_policy.safety_margin_tokens,
    )
    pressure_ratio = estimated_input_tokens / usable_input_tokens
    tool_output_ratio = tool_output_tokens / history_tokens if history_tokens else 0.0
    return RequestBudgetReport(
        conversation_items=len(request.conversation),
        tool_schema_count=len(request.tool_schemas),
        estimated_input_tokens=estimated_input_tokens,
        estimated_system_tokens=system_tokens,
        estimated_history_tokens=history_tokens,
        estimated_tool_output_tokens=tool_output_tokens,
        estimated_tool_schema_tokens=tool_schema_tokens,
        model_context_tokens=normalized_policy.model_context_tokens,
        response_token_reserve=normalized_policy.response_token_reserve,
        safety_margin_tokens=normalized_policy.safety_margin_tokens,
        usable_input_tokens=usable_input_tokens,
        pressure_ratio=pressure_ratio,
        tool_output_ratio=tool_output_ratio,
    )


def select_conversation_window(
    conversation: Iterable[ModelConversationItem],
    policy: ConversationBudgetPolicy | None = None,
) -> ConversationWindowSelection:
    """Select the model-facing history window for one request."""
    normalized_policy = (policy or ConversationBudgetPolicy()).normalized()
    items = tuple(conversation)
    item_tokens = tuple(estimate_conversation_item_tokens(item) for item in items)
    selected = _selected_items(items, item_tokens, normalized_policy)
    estimated_tokens = sum(item_tokens)
    selected_start = len(items) - len(selected)
    selected_tokens = sum(item_tokens[selected_start:])
    budget_suffix = _budget_suffix(items, item_tokens, normalized_policy)
    would_drop_count = max(0, len(items) - len(budget_suffix))
    should_trim = normalized_policy.history_window_mode == TRIM_HISTORY_WINDOW_MODE
    output_items = selected if should_trim else items
    dropped_count = len(items) - len(output_items)
    report = ConversationBudgetReport(
        conversation_items=len(items),
        selected_conversation_items=len(output_items),
        estimated_history_tokens=estimated_tokens,
        selected_history_tokens=selected_tokens if should_trim else estimated_tokens,
        history_token_budget=normalized_policy.history_token_budget,
        protect_recent_items=normalized_policy.protect_recent_items,
        history_window_mode=normalized_policy.history_window_mode,
        would_drop_count=would_drop_count,
        dropped_count=dropped_count,
        tool_exchange_count=sum(
            1 for item in output_items if item.tool_exchange is not None
        ),
        over_budget=estimated_tokens > normalized_policy.history_token_budget,
        trimmed=dropped_count > 0,
    )
    return ConversationWindowSelection(conversation=output_items, report=report)


def estimate_conversation_item_tokens(item: ModelConversationItem) -> int:
    """Return a rough token estimate for one conversation item."""
    if item.message is not None:
        return MESSAGE_OVERHEAD_TOKENS + estimate_text_tokens(item.message.content)
    if item.tool_exchange is not None:
        return MESSAGE_OVERHEAD_TOKENS + _estimate_tool_exchange_tokens(
            item.tool_exchange
        )
    return MESSAGE_OVERHEAD_TOKENS


def estimate_conversation_item_tool_output_tokens(
    item: ModelConversationItem,
) -> int:
    """Return estimated tokens from tool result payloads in one item."""
    if item.tool_exchange is None:
        return 0
    return sum(
        _estimate_tool_result_tokens(result)
        for result in item.tool_exchange.tool_results
    )


def estimate_tool_schema_tokens(schema: object) -> int:
    """Return a rough token estimate for one tool schema."""
    return TOOL_SCHEMA_OVERHEAD_TOKENS + estimate_text_tokens(
        _json_for_estimate(schema)
    )


def _effective_context_reserves(
    model_context_tokens: int,
    response_token_reserve: int,
    safety_margin_tokens: int,
) -> tuple[int, int]:
    total = response_token_reserve + safety_margin_tokens
    if total < model_context_tokens:
        return response_token_reserve, safety_margin_tokens
    if total <= 0:
        return 0, 0
    target_total = min(
        max(0, model_context_tokens - 1),
        max(1, int(model_context_tokens * FALLBACK_RESERVED_CONTEXT_RATIO)),
    )
    response = int(target_total * (response_token_reserve / total))
    safety = target_total - response
    return response, safety


def _selected_items(
    items: tuple[ModelConversationItem, ...],
    item_tokens: tuple[int, ...],
    policy: ConversationBudgetPolicy,
) -> tuple[ModelConversationItem, ...]:
    if policy.history_window_mode != TRIM_HISTORY_WINDOW_MODE:
        return items
    return _budget_suffix(items, item_tokens, policy)


def _budget_suffix(
    items: tuple[ModelConversationItem, ...],
    item_tokens: tuple[int, ...],
    policy: ConversationBudgetPolicy,
) -> tuple[ModelConversationItem, ...]:
    used = 0
    keep_start = len(items)
    protected_start = max(0, len(items) - policy.protect_recent_items)
    for index in reversed(range(len(items))):
        item_token_count = item_tokens[index]
        is_protected = index >= protected_start
        if not is_protected and used + item_token_count > policy.history_token_budget:
            break
        used += item_token_count
        keep_start = index
    return items[keep_start:]


def _estimate_tool_exchange_tokens(exchange: ModelToolExchange) -> int:
    total = estimate_text_tokens(exchange.assistant_content)
    total += estimate_text_tokens(exchange.assistant_reasoning)
    for request in exchange.tool_requests:
        total += estimate_text_tokens(request.name)
        total += estimate_text_tokens(request.raw_arguments)
        if request.arguments:
            total += estimate_text_tokens(_json_for_estimate(request.arguments))
    for result in exchange.tool_results:
        total += _estimate_tool_result_tokens(result)
    return total


def _estimate_tool_result_tokens(result: ModelToolResult) -> int:
    total = estimate_text_tokens(result.name)
    total += estimate_text_tokens(result.content)
    if result.data:
        total += estimate_text_tokens(_json_for_estimate(result.data))
    if result.refs:
        total += estimate_text_tokens(" ".join(result.refs))
    return total


def _system_tokens(
    conversation: tuple[ModelConversationItem, ...],
    conversation_tokens: tuple[int, ...],
) -> int:
    if not conversation or not conversation_tokens:
        return 0
    first = conversation[0]
    if first.message is None:
        return 0
    if first.message.role != MessageRole.SYSTEM:
        return 0
    return conversation_tokens[0]


def _json_for_estimate(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return str(value)
