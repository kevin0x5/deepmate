"""Shared text normalization and rough token estimation helpers."""

from __future__ import annotations


def normalize_name(name: str) -> str:
    """Return a stable case-insensitive display-name key."""
    return " ".join(name.strip().lower().split())


def compact_text(value: str, limit: int) -> str:
    """Collapse whitespace and truncate text to a display-safe limit."""
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def non_negative_int(value: object) -> int:
    """Return a non-negative integer parsed from persisted JSON-like data."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    return 0


def estimate_text_tokens(text: str) -> int:
    """Return a rough model-facing token estimate.

    This is intentionally tokenizer-agnostic. It is used for budget heuristics,
    not provider billing or exact context-window accounting.
    """
    cleaned = text.strip()
    if not cleaned:
        return 0
    token_tenths = 0
    for char in cleaned:
        if char.isspace():
            token_tenths += 1
            continue
        codepoint = ord(char)
        if _is_cjk_codepoint(codepoint):
            token_tenths += 6
        elif char.isascii() and (char.isalnum() or char == "_"):
            token_tenths += 3
        elif char.isascii():
            token_tenths += 5
        else:
            token_tenths += 8
    estimated = (token_tenths + 9) // 10
    return max(1, estimated)


def _is_cjk_codepoint(codepoint: int) -> bool:
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2A6DF
        or 0x2A700 <= codepoint <= 0x2B73F
        or 0x2B740 <= codepoint <= 0x2B81F
        or 0x2B820 <= codepoint <= 0x2CEAF
        or 0x30000 <= codepoint <= 0x3134F
    )
