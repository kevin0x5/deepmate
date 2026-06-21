"""Model-facing output budget policy for MCP tool results."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field

from deepmate.foundation import estimate_text_tokens
from deepmate.mcp.client import McpCallResult
from deepmate.mcp.spec import McpToolRef

DEFAULT_MCP_OUTPUT_RATIO = 0.025
DEFAULT_MIN_MCP_OUTPUT_TOKENS = 4_000
DEFAULT_MAX_MCP_OUTPUT_TOKENS = 25_000


@dataclass(frozen=True, slots=True)
class McpOutputPolicyResult:
    """MCP tool result after model-facing output budgeting."""

    content: str = ""
    data: Mapping[str, object] = field(default_factory=dict)
    refs: tuple[str, ...] = field(default_factory=tuple)
    truncated: bool = False
    structured_data_omitted: bool = False


@dataclass(frozen=True, slots=True)
class McpOutputPolicy:
    """Bound MCP result payloads before they enter model-visible history."""

    model_context_tokens: int
    max_output_ratio: float = DEFAULT_MCP_OUTPUT_RATIO
    min_output_tokens: int = DEFAULT_MIN_MCP_OUTPUT_TOKENS
    max_output_tokens: int = DEFAULT_MAX_MCP_OUTPUT_TOKENS

    def output_token_budget(self) -> int:
        """Return the resolved model-facing MCP output token budget."""
        context_tokens = max(1, self.model_context_tokens)
        raw_budget = max(
            1,
            int(context_tokens * _bounded_ratio(self.max_output_ratio)),
        )
        max_tokens = max(1, self.max_output_tokens)
        min_tokens = min(max_tokens, max(1, self.min_output_tokens))
        return min(max_tokens, max(min_tokens, raw_budget))

    def apply(self, tool: McpToolRef, result: McpCallResult) -> McpOutputPolicyResult:
        """Return an MCP call result bounded for model replay and storage."""
        budget_tokens = self.output_token_budget()
        refs: list[str] = []
        max_result_size = _mcp_max_result_size_chars(tool.meta)
        if max_result_size is None:
            max_result_size = _mcp_max_result_size_chars(
                _result_meta(result.data)
            )
        if max_result_size is not None:
            refs.append(f"mcp_tool_max_result_size_chars={max_result_size}")

        content_result = _bounded_content(
            result.content,
            budget_tokens=budget_tokens,
            tail_biased=result.is_error,
        )
        refs.extend(content_result.refs)

        data = _model_data(result.data)
        data_result = _bounded_data(
            data,
            budget_tokens=budget_tokens,
            has_content=bool(content_result.content.strip()),
        )
        refs.extend(data_result.refs)

        return McpOutputPolicyResult(
            content=content_result.content,
            data=data_result.data,
            refs=tuple(refs),
            truncated=content_result.truncated or data_result.omitted,
            structured_data_omitted=data_result.omitted,
        )


@dataclass(frozen=True, slots=True)
class _ContentBudgetResult:
    content: str
    refs: tuple[str, ...] = field(default_factory=tuple)
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class _DataBudgetResult:
    data: Mapping[str, object]
    refs: tuple[str, ...] = field(default_factory=tuple)
    omitted: bool = False


def _bounded_content(
    content: str,
    *,
    budget_tokens: int,
    tail_biased: bool,
) -> _ContentBudgetResult:
    text = content if isinstance(content, str) else ""
    estimated_tokens = estimate_text_tokens(text)
    if estimated_tokens <= budget_tokens:
        return _ContentBudgetResult(content=text)

    truncated = _truncate_middle(
        text,
        budget_tokens=budget_tokens,
        head_ratio=0.30 if tail_biased else 0.60,
    )
    returned_tokens = estimate_text_tokens(truncated.content)
    return _ContentBudgetResult(
        content=truncated.content,
        refs=(
            "mcp_output_truncated=true",
            f"mcp_output_original_chars={len(text)}",
            f"mcp_output_returned_chars={len(truncated.content)}",
            f"mcp_output_budget_tokens={budget_tokens}",
            f"mcp_output_estimated_tokens={estimated_tokens}",
            f"mcp_output_returned_estimated_tokens={returned_tokens}",
            f"mcp_output_omitted_middle_chars={truncated.omitted_chars}",
        ),
        truncated=True,
    )


def _bounded_data(
    data: Mapping[str, object],
    *,
    budget_tokens: int,
    has_content: bool,
) -> _DataBudgetResult:
    if not data:
        return _DataBudgetResult(data={})

    payload = _json_for_estimate(data)
    estimated_tokens = estimate_text_tokens(payload)
    if estimated_tokens <= budget_tokens:
        return _DataBudgetResult(data=data)

    keys = tuple(str(key) for key in data.keys())
    reason = (
        "structured MCP data omitted because content already carries the result"
        if has_content
        else "structured MCP data exceeded output budget"
    )
    summary: dict[str, object] = {
        "mcp_output_truncated": True,
        "data_omitted": True,
        "data_keys": list(keys),
        "reason": reason,
    }
    return _DataBudgetResult(
        data=summary,
        refs=(
            "mcp_output_truncated=true",
            "mcp_structured_data_omitted=true",
            f"mcp_structured_data_original_chars={len(payload)}",
            f"mcp_output_budget_tokens={budget_tokens}",
            f"mcp_structured_data_estimated_tokens={estimated_tokens}",
            f"mcp_structured_data_keys={','.join(keys[:12])}",
        ),
        omitted=True,
    )


@dataclass(frozen=True, slots=True)
class _TruncatedText:
    content: str
    omitted_chars: int


def _truncate_middle(
    text: str,
    *,
    budget_tokens: int,
    head_ratio: float,
) -> _TruncatedText:
    estimated_tokens = estimate_text_tokens(text)
    if estimated_tokens <= budget_tokens:
        return _TruncatedText(content=text, omitted_chars=0)
    marker_template = (
        "\n\n[Deepmate MCP output truncated:\n"
        f"original_estimated_tokens={estimated_tokens}\n"
        "returned_estimated_tokens={returned_tokens}\n"
        "omitted_middle_chars={omitted_chars}\n"
        "reason=mcp_output_budget_exceeded]\n\n"
    )
    marker_budget = estimate_text_tokens(
        marker_template.format(returned_tokens=0, omitted_chars=0)
    )
    keep_budget = max(1, budget_tokens - marker_budget)
    keep_chars = max(1, int(len(text) * keep_budget / max(1, estimated_tokens)))
    head_ratio = min(0.9, max(0.1, head_ratio))
    candidate = text
    omitted_chars = 0

    for _ in range(8):
        head_chars = max(0, int(keep_chars * head_ratio))
        tail_chars = max(0, keep_chars - head_chars)
        head = text[:head_chars]
        tail = text[len(text) - tail_chars :] if tail_chars else ""
        omitted_chars = max(0, len(text) - len(head) - len(tail))
        marker = marker_template.format(
            returned_tokens=0,
            omitted_chars=omitted_chars,
        )
        candidate = f"{head}{marker}{tail}"
        returned_tokens = estimate_text_tokens(candidate)
        marker = marker_template.format(
            returned_tokens=returned_tokens,
            omitted_chars=omitted_chars,
        )
        candidate = f"{head}{marker}{tail}"
        if estimate_text_tokens(candidate) <= budget_tokens or keep_chars <= 1:
            return _TruncatedText(content=candidate, omitted_chars=omitted_chars)
        keep_chars = max(1, int(keep_chars * 0.85))

    return _TruncatedText(content=candidate, omitted_chars=omitted_chars)


def _model_data(value: Mapping[str, object]) -> Mapping[str, object]:
    if not value:
        return {}
    return {key: data for key, data in value.items() if key != "content"}


def _result_meta(value: Mapping[str, object]) -> Mapping[str, object]:
    meta = value.get("_meta") if isinstance(value, Mapping) else None
    return dict(meta) if isinstance(meta, Mapping) else {}


def _mcp_max_result_size_chars(meta: Mapping[str, object]) -> int | None:
    value = meta.get("anthropic/maxResultSizeChars")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _bounded_ratio(value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return DEFAULT_MCP_OUTPUT_RATIO
    return min(1.0, max(0.0, number))


def _json_for_estimate(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return str(value)
