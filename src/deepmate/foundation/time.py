"""Shared datetime normalization helpers."""

from __future__ import annotations

from datetime import UTC, datetime


def normal_datetime(value: datetime | None = None) -> datetime:
    """Return a timezone-aware UTC datetime."""
    current = value or datetime.now(UTC)
    if current.tzinfo is None:
        return current.replace(tzinfo=UTC)
    return current.astimezone(UTC)


def utc_isoformat(value: datetime) -> str:
    """Return a second-precision UTC ISO timestamp."""
    return normal_datetime(value).replace(microsecond=0).isoformat()
