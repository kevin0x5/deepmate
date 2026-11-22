"""Provider usage objects."""

from __future__ import annotations

from dataclasses import dataclass


def reported_cache_hit_ratio(hits: int, misses: int) -> float | None:
    """Return a ratio over inputs with cache accounting."""
    total = hits + misses
    return hits / total if total > 0 else None


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token usage reported by a model provider."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit_input_tokens: int = 0
    cache_miss_input_tokens: int = 0
    reasoning_tokens: int = 0
    totals_reported: bool = True

    def total_tokens(self) -> int:
        """Return normalized input plus output tokens."""
        return self.input_tokens + self.output_tokens
