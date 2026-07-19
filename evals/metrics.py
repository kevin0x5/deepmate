"""Metric calculations and comparison gates for Deepmate evaluations."""

from __future__ import annotations

import resource
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping

from deepmate.runtime import UserTurnResult

from evals.model import (
    EvalCase,
    EvalCaseResult,
    EvalComparison,
    EvalMetrics,
    EvalRunResult,
)


HORIZON_ORDER = ("L0", "L1", "L2", "L3", "L4")


def trial_metrics(
    case: EvalCase,
    result: UserTurnResult,
    *,
    elapsed_seconds: float,
    schema_tokens: int,
    pricing: Mapping[str, object] | None = None,
) -> EvalMetrics:
    """Calculate raw runtime metrics without using grader outcomes."""
    input_tokens = 0
    output_tokens = 0
    cache_hit_tokens = 0
    cache_miss_tokens = 0
    reasoning_tokens = 0
    for step in result.steps:
        usage = step.response.usage
        if usage is None:
            continue
        input_tokens += usage.input_tokens
        output_tokens += usage.output_tokens
        cache_hit_tokens += usage.cache_hit_input_tokens
        cache_miss_tokens += usage.cache_miss_input_tokens
        reasoning_tokens += usage.reasoning_tokens
    cache_total = cache_hit_tokens + cache_miss_tokens
    pressure_values = [
        step.request_budget_report.pressure_ratio
        for step in result.steps
        if step.request_budget_report is not None
    ]
    raw_peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_memory_kib = raw_peak_memory / 1024 if sys.platform == "darwin" else raw_peak_memory
    price = dict(pricing or {})
    priced = all(
        key in price
        for key in ("input_per_million", "output_per_million", "source", "effective_at")
    )
    cost = None
    if priced:
        input_cost = input_tokens * float(price["input_per_million"])
        if cache_total and "cache_hit_input_per_million" in price:
            input_cost = (
                cache_hit_tokens * float(price["cache_hit_input_per_million"])
                + cache_miss_tokens * float(price["input_per_million"])
            )
        cost = (
            input_cost + output_tokens * float(price["output_per_million"])
        ) / 1_000_000
    values: dict[str, object] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cache_hit_tokens": cache_hit_tokens,
        "cache_miss_tokens": cache_miss_tokens,
        "cache_hit_ratio": cache_hit_tokens / cache_total if cache_total else None,
        "tool_schema_tokens": schema_tokens,
        "tool_calls": sum(
            len(exchange.tool_requests) for exchange in result.tool_exchanges
        ),
        "tool_errors": sum(
            result_item.is_error
            for exchange in result.tool_exchanges
            for result_item in exchange.tool_results
        ),
        "steps": len(result.steps),
        "reached_max_steps": result.reached_max_steps,
        "runtime_errors": len(result.errors()),
        "latency_seconds": max(0.0, elapsed_seconds),
        "token_per_second": (
            output_tokens / elapsed_seconds if elapsed_seconds > 0 else None
        ),
        "process_peak_memory_kib": peak_memory_kib,
        "context_pressure": max(pressure_values, default=0.0),
        "cost": cost,
        "horizon": _case_horizon(case),
    }
    return EvalMetrics(values=values, priced=priced)


def aggregate_metrics(results: Iterable[EvalCaseResult]) -> EvalMetrics:
    """Aggregate case outcomes into reliability, safety, cache, and cost metrics."""
    items = tuple(results)
    if not items:
        return EvalMetrics(values={"trials": 0}, priced=False)
    grouped: dict[str, list[EvalCaseResult]] = defaultdict(list)
    for item in items:
        grouped[item.trial.case_id].append(item)
    for group in grouped.values():
        group.sort(key=lambda result: result.trial.index)

    total = len(items)
    successes = sum(item.passed for item in items)
    safety_successes = sum(item.safety_passed for item in items)
    first_attempts = [group[0] for group in grouped.values()]
    pass_at_1 = sum(item.passed for item in first_attempts) / len(first_attempts)
    values: dict[str, object] = {
        "cases": len(grouped),
        "trials": total,
        "task_success": successes / total,
        "safety_gate": safety_successes == total,
        "pass@1": pass_at_1,
        "pass^3": _pass_power(grouped, 3),
        "pass^5": _pass_power(grouped, 5),
        "recovery_success_rate": _tagged_success(items, "recovery"),
        "governed_tool_success": _tagged_success(items, "tool-governance"),
        "context_fidelity": _tagged_grade_score(items, "context"),
        "compression_continuity": _tagged_grade_score(items, "compression"),
        "evolution_net_gain": _evolution_net_gain(items),
        "reliable_task_horizon": _reliable_horizon(items),
    }
    numeric_keys = (
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cache_hit_tokens",
        "cache_miss_tokens",
        "tool_schema_tokens",
        "tool_calls",
        "tool_errors",
        "latency_seconds",
        "process_peak_memory_kib",
    )
    for key in numeric_keys:
        values[key] = sum(_float_metric(item, key) for item in items)
    cache_total = float(values["cache_hit_tokens"]) + float(values["cache_miss_tokens"])
    values["cache_hit_ratio"] = (
        float(values["cache_hit_tokens"]) / cache_total if cache_total else None
    )
    latencies = sorted(_float_metric(item, "latency_seconds") for item in items)
    values["p95_latency_seconds"] = _percentile(latencies, 0.95)
    priced = all(item.metrics.priced for item in items)
    if priced:
        total_cost = sum(_float_metric(item, "cost") for item in items)
        values["total_cost"] = total_cost
        values["cost_per_reliable_success"] = (
            total_cost / successes if successes else None
        )
    else:
        values["total_cost"] = None
        values["cost_per_reliable_success"] = None
        values["cost_status"] = "unpriced"
    return EvalMetrics(values=values, priced=priced)


def compare_runs(baseline: EvalRunResult, candidate: EvalRunResult) -> EvalComparison:
    """Compare only equivalent runs and emit explicit performance warnings."""
    if not baseline.valid or not candidate.valid:
        return EvalComparison(
            baseline.manifest.run_id,
            candidate.manifest.run_id,
            valid=False,
            invalid_reason="one or both runs are invalid",
        )
    if baseline.manifest.comparison_key() != candidate.manifest.comparison_key():
        return EvalComparison(
            baseline.manifest.run_id,
            candidate.manifest.run_id,
            valid=False,
            invalid_reason=(
                "case set, fixture, policy, provider/model, judge, permissions, "
                "environment, or pricing provenance differs"
            ),
        )
    baseline_values = baseline.metrics.values
    candidate_values = candidate.metrics.values
    keys = (
        "task_success",
        "pass@1",
        "pass^3",
        "pass^5",
        "cache_hit_ratio",
        "p95_latency_seconds",
        "cost_per_reliable_success",
        "evolution_net_gain",
        "context_fidelity",
        "compression_continuity",
    )
    deltas: dict[str, object] = {}
    for key in keys:
        before = _optional_number(baseline_values.get(key))
        after = _optional_number(candidate_values.get(key))
        delta = None if before is None or after is None else after - before
        if (
            delta is not None
            and key in {"context_fidelity", "compression_continuity"}
            and abs(delta) < 0.3
        ):
            delta = 0.0
        deltas[key] = delta
    warnings: list[str] = []
    cost_growth = _relative_growth(
        baseline_values.get("cost_per_reliable_success"),
        candidate_values.get("cost_per_reliable_success"),
    )
    cache_drop = _relative_drop(
        baseline_values.get("cache_hit_ratio"),
        candidate_values.get("cache_hit_ratio"),
    )
    latency_growth = _relative_growth(
        baseline_values.get("p95_latency_seconds"),
        candidate_values.get("p95_latency_seconds"),
    )
    if cost_growth is not None and cost_growth > 0.10:
        warnings.append(f"cost per reliable success increased by {cost_growth:.1%}")
    if cache_drop is not None and cache_drop > 0.10:
        warnings.append(f"cache hit ratio decreased by {cache_drop:.1%}")
    if latency_growth is not None and latency_growth > 0.20:
        warnings.append(f"P95 latency increased by {latency_growth:.1%}")
    if baseline.safety_gate and not candidate.safety_gate:
        warnings.append("candidate failed the non-compensable safety gate")
    return EvalComparison(
        baseline.manifest.run_id,
        candidate.manifest.run_id,
        valid=True,
        deltas=deltas,
        warnings=tuple(warnings),
    )


def _pass_power(groups: Mapping[str, list[EvalCaseResult]], k: int) -> float | None:
    eligible = [items for items in groups.values() if len(items) >= k]
    if not eligible:
        return None
    return sum(all(item.passed for item in items[:k]) for items in eligible) / len(eligible)


def _tagged_success(items: tuple[EvalCaseResult, ...], tag: str) -> float | None:
    selected = [
        item
        for item in items
        if tag in tuple(str(value) for value in item.observations.get("tags", ()))
    ]
    if not selected:
        return None
    return sum(item.passed for item in selected) / len(selected)


def _tagged_grade_score(items: tuple[EvalCaseResult, ...], tag: str) -> float | None:
    selected = [
        grade.score
        for item in items
        if tag in tuple(str(value) for value in item.observations.get("tags", ()))
        for grade in item.grades
        if not grade.hard
    ]
    return sum(selected) / len(selected) if selected else None


def _evolution_net_gain(items: tuple[EvalCaseResult, ...]) -> float | None:
    selected = [
        item
        for item in items
        if "evolution" in tuple(str(value) for value in item.observations.get("tags", ()))
    ]
    if (
        not selected
        or not all(item.safety_passed for item in selected)
        or not all(
            all(
                key in item.observations
                for key in ("measured_pre_score", "measured_post_score", "measured_negative_transfer")
            )
            for item in selected
        )
    ):
        return None
    gains = [
        _float_value(item.observations.get("measured_post_score"))
        - _float_value(item.observations.get("measured_pre_score"))
        - _float_value(item.observations.get("measured_negative_transfer"))
        for item in selected
    ]
    return sum(gains) / len(gains)


def _reliable_horizon(items: tuple[EvalCaseResult, ...]) -> str | None:
    by_level: dict[str, list[EvalCaseResult]] = defaultdict(list)
    for item in items:
        level = str(item.metrics.values.get("horizon", ""))
        if level in HORIZON_ORDER:
            by_level[level].append(item)
    reliable: str | None = None
    for level in HORIZON_ORDER:
        level_items = by_level.get(level, [])
        if level_items and all(item.passed for item in level_items):
            reliable = level
        elif level_items:
            break
    return reliable


def _case_horizon(case: EvalCase) -> str | None:
    for tag in case.tags:
        if tag.startswith("horizon:"):
            level = tag.partition(":")[2].upper()
            return level if level in HORIZON_ORDER else None
    return None


def _float_metric(item: EvalCaseResult, key: str) -> float:
    return _float_value(item.metrics.values.get(key))


def _float_value(value: object) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _optional_number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _relative_growth(before: object, after: object) -> float | None:
    first = _optional_number(before)
    second = _optional_number(after)
    if first is None or second is None or first <= 0:
        return None
    return (second - first) / first


def _relative_drop(before: object, after: object) -> float | None:
    first = _optional_number(before)
    second = _optional_number(after)
    if first is None or second is None or first <= 0:
        return None
    return (first - second) / first


def _percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    index = min(len(values) - 1, max(0, round((len(values) - 1) * ratio)))
    return values[index]
