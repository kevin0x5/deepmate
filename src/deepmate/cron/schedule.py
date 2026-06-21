"""Schedule parsing and due-time helpers for cron jobs."""

from __future__ import annotations

from datetime import datetime, timedelta, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from deepmate.cron.model import CronSchedule

WEEKDAYS = {
    "mon": 0,
    "monday": 0,
    "周一": 0,
    "星期一": 0,
    "tue": 1,
    "tuesday": 1,
    "周二": 1,
    "星期二": 1,
    "wed": 2,
    "wednesday": 2,
    "周三": 2,
    "星期三": 2,
    "thu": 3,
    "thursday": 3,
    "周四": 3,
    "星期四": 3,
    "fri": 4,
    "friday": 4,
    "周五": 4,
    "星期五": 4,
    "sat": 5,
    "saturday": 5,
    "周六": 5,
    "星期六": 5,
    "sun": 6,
    "sunday": 6,
    "周日": 6,
    "周天": 6,
    "星期日": 6,
    "星期天": 6,
}


def next_run_at(schedule: CronSchedule, *, now: datetime | None = None) -> str:
    """Return the next run timestamp for a simple schedule."""
    current = now or datetime.now().astimezone()
    tz = _timezone(schedule.timezone, current)
    current = current.astimezone(tz)
    kind = schedule.kind.strip().lower()
    if kind == "interval":
        minutes = max(1, schedule.interval_minutes)
        return (current + timedelta(minutes=minutes)).replace(microsecond=0).isoformat()
    hour, minute = parse_time(schedule.time or "09:00")
    candidate = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if kind == "weekly":
        weekday = WEEKDAYS.get(schedule.weekday.strip().lower(), current.weekday())
        days = (weekday - current.weekday()) % 7
        candidate = candidate + timedelta(days=days)
        if candidate <= current:
            candidate = candidate + timedelta(days=7)
        return candidate.isoformat()
    if candidate <= current:
        candidate = candidate + timedelta(days=1)
    return candidate.isoformat()


def is_due(next_run: str, *, now: datetime | None = None) -> bool:
    """Return whether a next_run timestamp is due."""
    if not next_run.strip():
        return True
    current = now or datetime.now().astimezone()
    try:
        due_at = datetime.fromisoformat(next_run)
    except ValueError:
        return True
    if due_at.tzinfo is None:
        due_at = due_at.astimezone()
    return due_at <= current


def parse_time(value: str) -> tuple[int, int]:
    """Parse HH:MM, HH点, or natural-ish hour text."""
    clean = value.strip().lower().replace("：", ":")
    if "点" in clean and ":" not in clean:
        hour = int(clean.split("点", 1)[0])
        return _bounded_time(hour, 0)
    if ":" in clean:
        hour_text, minute_text = clean.split(":", 1)
        return _bounded_time(int(hour_text), int(minute_text[:2] or "0"))
    return _bounded_time(int(clean), 0)


def _bounded_time(hour: int, minute: int) -> tuple[int, int]:
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError("schedule time must be within 00:00-23:59")
    return hour, minute


def _timezone(value: str, now: datetime) -> tzinfo:
    clean = value.strip()
    if not clean:
        current_tz = now.tzinfo
        if current_tz is not None:
            return current_tz
        return ZoneInfo("UTC")
    try:
        return ZoneInfo(clean)
    except ZoneInfoNotFoundError:
        return ZoneInfo("UTC")
