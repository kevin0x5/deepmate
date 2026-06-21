"""Turn-level cost/cache summary for runtime observability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepmate.runtime.agent_loop import AgentStepResult


@dataclass(frozen=True, slots=True)
class TurnCostCacheSummary:
    """Aggregated model usage and request-budget pressure for one user turn."""

    steps: int = 0
    model_response_events: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_hit_input_tokens: int = 0
    cache_miss_input_tokens: int = 0
    estimated_input_tokens: int = 0
    estimated_tool_schema_tokens: int = 0
    estimated_tool_output_tokens: int = 0
    max_input_pressure: float = 0.0
    max_tool_output_ratio: float = 0.0

    def cache_hit_ratio(self) -> float | None:
        """Return provider-reported cache hit ratio when available."""
        total = self.cache_hit_input_tokens + self.cache_miss_input_tokens
        if total <= 0:
            return None
        return self.cache_hit_input_tokens / total

    def trace_refs(self) -> tuple[str, ...]:
        """Return compact refs for trace records."""
        refs = [
            f"steps={self.steps}",
            f"model_response_events={self.model_response_events}",
            f"input_tokens={self.input_tokens}",
            f"output_tokens={self.output_tokens}",
            f"reasoning_tokens={self.reasoning_tokens}",
            f"cache_hit_input_tokens={self.cache_hit_input_tokens}",
            f"cache_miss_input_tokens={self.cache_miss_input_tokens}",
            f"estimated_input_tokens={self.estimated_input_tokens}",
            f"estimated_tool_schema_tokens={self.estimated_tool_schema_tokens}",
            f"estimated_tool_output_tokens={self.estimated_tool_output_tokens}",
            f"max_input_pressure={self.max_input_pressure:.4f}",
            f"max_tool_output_ratio={self.max_tool_output_ratio:.4f}",
        ]
        ratio = self.cache_hit_ratio()
        if ratio is not None:
            refs.append(f"cache_hit_ratio={ratio:.4f}")
        return tuple(refs)

    def status_line(self) -> str:
        """Return a compact user-facing runtime status line."""
        cache_ratio = self.cache_hit_ratio()
        cache_part = (
            f"; cache_hit_ratio={cache_ratio:.0%}"
            if cache_ratio is not None
            else "; cache=unreported"
        )
        usage_part = (
            f"input={self.input_tokens}; output={self.output_tokens}; "
            f"reasoning={self.reasoning_tokens}"
            if self.model_response_events
            else "usage=unreported"
        )
        return (
            "turn cost/cache summary: "
            f"steps={self.steps}; {usage_part}{cache_part}; "
            f"max_input_pressure={self.max_input_pressure:.3f}; "
            f"tool_output_ratio={self.max_tool_output_ratio:.3f}"
        )


def build_turn_cost_cache_summary(
    steps: tuple[AgentStepResult, ...],
) -> TurnCostCacheSummary:
    """Aggregate model usage and request budget reports across a turn."""
    model_response_events = 0
    input_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0
    cache_hit_input_tokens = 0
    cache_miss_input_tokens = 0
    estimated_input_tokens = 0
    estimated_tool_schema_tokens = 0
    estimated_tool_output_tokens = 0
    max_input_pressure = 0.0
    max_tool_output_ratio = 0.0

    for step in steps:
        usage = step.response.usage
        if usage is not None:
            model_response_events += 1
            input_tokens += usage.input_tokens
            output_tokens += usage.output_tokens
            reasoning_tokens += usage.reasoning_tokens
            cache_hit_input_tokens += usage.cache_hit_input_tokens
            cache_miss_input_tokens += usage.cache_miss_input_tokens
        report = step.request_budget_report
        if report is not None:
            estimated_input_tokens += report.estimated_input_tokens
            estimated_tool_schema_tokens += report.estimated_tool_schema_tokens
            estimated_tool_output_tokens += report.estimated_tool_output_tokens
            max_input_pressure = max(max_input_pressure, report.pressure_ratio)
            max_tool_output_ratio = max(max_tool_output_ratio, report.tool_output_ratio)

    return TurnCostCacheSummary(
        steps=len(steps),
        model_response_events=model_response_events,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cache_hit_input_tokens=cache_hit_input_tokens,
        cache_miss_input_tokens=cache_miss_input_tokens,
        estimated_input_tokens=estimated_input_tokens,
        estimated_tool_schema_tokens=estimated_tool_schema_tokens,
        estimated_tool_output_tokens=estimated_tool_output_tokens,
        max_input_pressure=max_input_pressure,
        max_tool_output_ratio=max_tool_output_ratio,
    )
