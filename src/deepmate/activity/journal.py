"""Daily activity journal for human-readable session handoff."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ActivityEntry:
    """One activity checkpoint written to a daily note."""

    timestamp: str
    event: str
    status: str
    title: str
    summary: str
    session_id: str
    session_title: str
    profile: str
    workspace: str
    summary_id: str = ""
    covered_until_sequence: int = 0
    transcript_path: str = ""
    session_summary_path: str = ""
    trace_path: str = ""
    refs: tuple[str, ...] = field(default_factory=tuple)

    def is_ready(self) -> bool:
        """Return whether the entry has enough metadata to be useful."""
        return bool(
            _date_from_timestamp(self.timestamp)
            and self.event.strip()
            and self.status.strip()
            and self.title.strip()
            and self.session_id.strip()
            and self.profile.strip()
        )

    def local_date(self) -> str:
        """Return the YYYY-MM-DD date part used for the daily note filename."""
        date = _date_from_timestamp(self.timestamp)
        if not date:
            raise ValueError("activity timestamp must start with YYYY-MM-DD")
        return date


class ActivityStore:
    """Append human-readable activity entries under one profile activity root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def daily_path(self, local_date: str) -> Path:
        """Return the daily note path for YYYY-MM-DD."""
        date = _validated_date(local_date)
        return self.root / "daily" / f"{date}.md"

    def monthly_summary_path(self, month: str) -> Path:
        """Return the future monthly summary path for YYYY-MM."""
        clean_month = _validated_month(month)
        return self.root / "summaries" / f"{clean_month}.md"

    def list_daily_dates(self, month: str) -> tuple[str, ...]:
        """Return YYYY-MM-DD dates that have daily notes for a month."""
        clean_month = _validated_month(month)
        directory = self.root / "daily"
        if not directory.exists():
            return ()
        dates = [
            path.stem
            for path in directory.glob(f"{clean_month}-*.md")
            if _date_from_timestamp(path.stem)
        ]
        return tuple(sorted(dates))

    def append_daily_entry(self, entry: ActivityEntry) -> Path:
        """Append one entry to the matching daily note and return its path."""
        if not entry.is_ready():
            raise ValueError("activity entry is not ready")
        path = self.daily_path(entry.local_date())
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        with path.open("a", encoding="utf-8") as file:
            if not existing.strip():
                file.write(f"# Activity Note - {entry.local_date()}\n")
            elif not existing.endswith("\n"):
                file.write("\n")
            file.write("\n")
            file.write(_render_entry(entry))
        return path

    def upsert_monthly_summary_entry(
        self,
        local_date: str,
        summary: str,
        highlights: tuple[str, ...] = (),
        next_steps: tuple[str, ...] = (),
        refs: tuple[str, ...] = (),
    ) -> Path:
        """Create or replace one date section in the monthly summary."""
        date = _validated_date(local_date)
        month = date[:7]
        path = self.monthly_summary_path(month)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        section = _render_monthly_summary_entry(
            local_date=date,
            summary=summary,
            highlights=highlights,
            next_steps=next_steps,
            refs=refs,
        )
        content = _upsert_markdown_section(
            existing=existing,
            title=f"# Activity Summary - {month}",
            section_heading=f"## {date}",
            section=section,
        )
        path.write_text(content, encoding="utf-8")
        return path


def preview_activity_text(text: str, limit: int = 500) -> str:
    """Return a compact single-line preview for activity notes."""
    value = " ".join(text.split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _render_entry(entry: ActivityEntry) -> str:
    lines = [
        f"## {_time_from_timestamp(entry.timestamp)} - {entry.title.strip()}",
        "",
        f"- event: {entry.event.strip()}",
        f"- status: {entry.status.strip()}",
        f"- session_id: {entry.session_id.strip()}",
        f"- session_title: {preview_activity_text(entry.session_title, 160)}",
        f"- profile: {entry.profile.strip()}",
    ]
    if entry.workspace.strip():
        lines.append(f"- workspace: {entry.workspace.strip()}")
    summary = preview_activity_text(entry.summary, 700)
    if summary:
        lines.append(f"- summary: {summary}")
    if entry.summary_id.strip():
        lines.append(f"- summary_id: {entry.summary_id.strip()}")
    if entry.covered_until_sequence > 0:
        lines.append(f"- covered_until_sequence: {entry.covered_until_sequence}")
    sources = _source_lines(entry)
    if sources:
        lines.append("- sources:")
        lines.extend(f"  - {source}" for source in sources)
    refs = tuple(ref.strip() for ref in entry.refs if ref.strip())
    if refs:
        lines.append("- refs:")
        lines.extend(f"  - {ref}" for ref in refs)
    lines.append("")
    return "\n".join(lines)


def _render_monthly_summary_entry(
    local_date: str,
    summary: str,
    highlights: tuple[str, ...],
    next_steps: tuple[str, ...],
    refs: tuple[str, ...],
) -> str:
    lines = [f"## {local_date}", ""]
    clean_summary = preview_activity_text(summary, 900)
    if clean_summary:
        lines.append(f"- summary: {clean_summary}")
    clean_highlights = tuple(value.strip() for value in highlights if value.strip())
    if clean_highlights:
        lines.append("- highlights:")
        lines.extend(f"  - {preview_activity_text(value, 240)}" for value in clean_highlights)
    clean_next_steps = tuple(value.strip() for value in next_steps if value.strip())
    if clean_next_steps:
        lines.append("- next_steps:")
        lines.extend(f"  - {preview_activity_text(value, 240)}" for value in clean_next_steps)
    lines.append(f"- daily_note: daily/{local_date}.md")
    clean_refs = tuple(value.strip() for value in refs if value.strip())
    if clean_refs:
        lines.append("- refs:")
        lines.extend(f"  - {preview_activity_text(value, 240)}" for value in clean_refs)
    lines.append("")
    return "\n".join(lines)


def _upsert_markdown_section(
    existing: str,
    title: str,
    section_heading: str,
    section: str,
) -> str:
    clean_existing = existing.strip()
    if not clean_existing:
        return f"{title}\n\n{section.strip()}\n"
    lines = clean_existing.splitlines()
    if lines[0].strip() != title:
        lines.insert(0, title)
        lines.insert(1, "")
    start = None
    for index, line in enumerate(lines):
        if line.strip() == section_heading:
            start = index
            break
    if start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(section.strip().splitlines())
        return "\n".join(lines).rstrip() + "\n"
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("## "):
            end = index
            break
    updated = lines[:start] + section.strip().splitlines() + lines[end:]
    return "\n".join(updated).rstrip() + "\n"


def _source_lines(entry: ActivityEntry) -> tuple[str, ...]:
    sources: list[str] = []
    if entry.session_summary_path.strip():
        sources.append(f"session_summary: {entry.session_summary_path.strip()}")
    if entry.transcript_path.strip():
        sources.append(f"transcript: {entry.transcript_path.strip()}")
    if entry.trace_path.strip():
        sources.append(f"trace: {entry.trace_path.strip()}")
    return tuple(sources)


def _date_from_timestamp(timestamp: str) -> str:
    value = timestamp.strip()
    if len(value) < 10:
        return ""
    date = value[:10]
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return ""
    return date


def _time_from_timestamp(timestamp: str) -> str:
    value = timestamp.strip()
    if "T" not in value:
        return value
    return value.split("T", 1)[1]


def _validated_date(value: str) -> str:
    date = value.strip()
    if not _date_from_timestamp(date):
        raise ValueError("date must use YYYY-MM-DD")
    return date[:10]


def _validated_month(value: str) -> str:
    month = value.strip()
    if len(month) != 7:
        raise ValueError("month must use YYYY-MM")
    try:
        datetime.strptime(month, "%Y-%m")
    except ValueError as exc:
        raise ValueError("month must use YYYY-MM") from exc
    return month
