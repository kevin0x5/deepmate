"""Shared JSON response helpers for Task Mode."""

from __future__ import annotations

import re
from collections.abc import Mapping


def task_request_options(
    options: Mapping[str, object] | None, *, max_tokens: int
) -> dict[str, object]:
    """Honor the model-purpose settings within the budget for this JSON response."""
    resolved = {"temperature": 0, **(options or {})}
    limit = resolved.get("max_tokens")
    if type(limit) is int and limit > 0:
        max_tokens = min(limit, max_tokens)
    return {**resolved, "max_tokens": max_tokens}


def contract_status_items(value: object) -> tuple[str, ...]:
    """Read an evaluator checklist from a JSON array."""
    if not isinstance(value, list):
        return ()
    return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())


def strip_fenced_json(content: str) -> str:
    """Return JSON payload text from a raw or fenced model response."""
    stripped = content.strip()
    if not stripped.startswith("```"):
        fenced = re.search(r"```(?:json)?\s*", stripped, flags=re.IGNORECASE)
        if fenced:
            end = stripped.rfind("```")
            if end >= 0:
                return stripped[fenced.end() : end].strip()
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```"):
        if lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return stripped
