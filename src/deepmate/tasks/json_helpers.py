"""Shared JSON response helpers for Task Mode."""

from __future__ import annotations

import re


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
