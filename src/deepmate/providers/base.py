"""Provider protocols."""

from __future__ import annotations

from typing import Protocol

from deepmate.providers.messages import ModelRequest, ModelResponse


class ModelProvider(Protocol):
    """Minimal interface implemented by model provider adapters."""

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return one model response for the given request."""
        ...
