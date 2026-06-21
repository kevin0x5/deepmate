"""Shared path display helpers."""

from __future__ import annotations

from pathlib import Path


def display_path(path: str | Path, root: str | Path) -> str:
    """Return a path relative to root when possible."""
    target = Path(path)
    base = Path(root)
    try:
        return str(target.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(target)
