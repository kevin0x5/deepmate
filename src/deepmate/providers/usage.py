"""Provider usage objects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token usage reported by a model provider."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit_input_tokens: int = 0
    cache_miss_input_tokens: int = 0
    reasoning_tokens: int = 0

    def total_tokens(self) -> int:
        """Return normalized input plus output tokens."""
        return self.input_tokens + self.output_tokens
