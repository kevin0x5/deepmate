"""Render project Task Mode context."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from xml.sax.saxutils import escape

from deepmate.foundation import estimate_text_tokens

TASK_CONTEXT_TOKEN_BUDGET = 6000
PLAN_TOKEN_BUDGET = 2500


class TaskStage(StrEnum):
    """Supported single-task project stages."""

    PLAN = "plan"
    EXECUTE = "execute"
    CHECKPOINT = "checkpoint"

    @classmethod
    def parse(cls, value: str) -> "TaskStage | None":
        """Return a stage from user input, if the value is a stage command."""
        clean = value.strip().lower()
        for stage in cls:
            if clean == stage.value:
                return stage
        return None


@dataclass(frozen=True, slots=True)
class TaskDocuments:
    """Markdown files that form the project-level task memory."""

    plan: str = ""
    evolution: str = ""
    achievements: tuple[tuple[Path, str], ...] = field(default_factory=tuple)

    def has_content(self) -> bool:
        """Return whether any task document has useful content."""
        return bool(
            self.plan.strip()
            or self.evolution.strip()
            or any(content.strip() for _, content in self.achievements)
        )


@dataclass(frozen=True, slots=True)
class TaskContext:
    """Bounded task context for system prompt injection."""

    stage: TaskStage
    plan: str
    rolling_summary: str
    recent_timeline: str
    recent_achievements: str
    estimated_tokens: int

    def has_content(self) -> bool:
        """Return whether the rendered task context carries project memory."""
        return bool(
            self.plan.strip()
            or self.rolling_summary.strip()
            or self.recent_timeline.strip()
            or self.recent_achievements.strip()
        )


def build_task_context(
    documents: TaskDocuments,
    stage: TaskStage,
    *,
    timeline_limit: int = 5,
    achievement_limit: int = 3,
) -> TaskContext:
    """Build a bounded Task Mode context from project task markdown files."""
    plan = _bounded_text(documents.plan.strip(), token_budget=PLAN_TOKEN_BUDGET)
    rolling_summary = extract_rolling_summary(documents.evolution)
    recent_timeline = extract_recent_timeline(documents.evolution, limit=timeline_limit)
    recent_achievements = render_recent_achievements(
        documents.achievements,
        limit=achievement_limit,
    )
    plan, rolling_summary, recent_timeline, recent_achievements = _fit_task_parts(
        plan,
        rolling_summary,
        recent_timeline,
        recent_achievements,
        token_budget=TASK_CONTEXT_TOKEN_BUDGET,
    )
    content = "\n\n".join(
        part
        for part in (plan, rolling_summary, recent_timeline, recent_achievements)
        if part.strip()
    )
    return TaskContext(
        stage=stage,
        plan=plan,
        rolling_summary=rolling_summary,
        recent_timeline=recent_timeline,
        recent_achievements=recent_achievements,
        estimated_tokens=estimate_text_tokens(content),
    )


def render_task_context_section(context: TaskContext | None) -> str:
    """Render Task Mode context for system prompt injection."""
    if context is None or not context.has_content():
        return ""
    lines = [
        "<task_context>",
        f"<stage>{context.stage.value}</stage>",
    ]
    if context.plan:
        lines.extend(("<current_plan>", _xml_text(context.plan), "</current_plan>"))
    if context.rolling_summary:
        lines.extend(
            ("<rolling_summary>", _xml_text(context.rolling_summary), "</rolling_summary>")
        )
    if context.recent_timeline:
        lines.extend(
            (
                "<recent_timeline>",
                _xml_text(context.recent_timeline),
                "</recent_timeline>",
            )
        )
    if context.recent_achievements:
        lines.extend(
            (
                "<recent_achievements>",
                _xml_text(context.recent_achievements),
                "</recent_achievements>",
            )
        )
    lines.extend(
        (
            "<guidance>",
            "- task/plan.md is the task contract and current working plan.",
            "- The plan must keep an explicit goal, acceptance contract, execution plan, and verification strategy.",
            "- task/execute follows task/plan.md and must not silently expand scope beyond it.",
            "- Durable task history belongs in task/evolution.md and task/achievements/.",
            "- Do not write project-specific task decisions into user.md or memory.md.",
            "- Add evolution entries only for goal changes, key milestones, decision changes, blockers, or direction corrections.",
            "- task/achievements/ records completed work or manual checkpoints; it is not a user-facing stage.",
            "</guidance>",
            "</task_context>",
        )
    )
    return "\n".join(lines)


def extract_rolling_summary(content: str) -> str:
    """Extract the fixed Rolling Summary section from evolution.md."""
    return _section_body(content, "## Rolling Summary").strip()


def extract_recent_timeline(content: str, *, limit: int = 5) -> str:
    """Extract recent timeline entries from evolution.md."""
    timeline = _section_body(content, "## Timeline")
    entries = _timeline_entries(timeline)
    if limit < 1:
        return ""
    return "\n\n".join(entries[-limit:]).strip()


def render_recent_achievements(
    achievements: tuple[tuple[Path, str], ...],
    *,
    limit: int = 3,
) -> str:
    """Render compact summaries for recent achievement files."""
    if limit < 1:
        return ""
    summaries: list[str] = []
    for path, content in achievements[-limit:]:
        title = _first_heading(content) or path.stem
        done = _first_bullets(_section_body(content, "## 本轮完成"), limit=2)
        decisions = _first_bullets(_section_body(content, "## 关键决策"), limit=2)
        details = [*done, *decisions]
        if details:
            summaries.append(
                "\n".join((f"- {title}", *(f"  - {item}" for item in details[:3])))
            )
        else:
            summaries.append(f"- {title}")
    return "\n".join(summaries).strip()


def _section_body(content: str, heading: str) -> str:
    lines = content.splitlines()
    start = -1
    target = heading.strip()
    for index, line in enumerate(lines):
        if line.strip() == target:
            start = index + 1
            break
    if start < 0:
        return ""
    body: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("## ") and body:
            break
        if stripped.startswith("## ") and not body:
            break
        body.append(line)
    return "\n".join(body).strip()


def _timeline_entries(content: str) -> tuple[str, ...]:
    entries: list[list[str]] = []
    current: list[str] = []
    for line in content.splitlines():
        if line.strip().startswith("### "):
            if current:
                entries.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        entries.append(current)
    return tuple("\n".join(entry).strip() for entry in entries if entry)


def _first_heading(content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return ""


def _first_bullets(content: str, *, limit: int) -> tuple[str, ...]:
    bullets: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            bullets.append(stripped[2:].strip())
            if len(bullets) >= limit:
                break
    return tuple(bullet for bullet in bullets if bullet)


def _fit_task_parts(
    plan: str,
    rolling_summary: str,
    recent_timeline: str,
    recent_achievements: str,
    *,
    token_budget: int,
) -> tuple[str, str, str, str]:
    parts = [plan, rolling_summary, recent_timeline, recent_achievements]
    if estimate_text_tokens("\n\n".join(part for part in parts if part.strip())) <= token_budget:
        return plan, rolling_summary, recent_timeline, recent_achievements
    recent_achievements = _bounded_text(recent_achievements, token_budget=1000)
    parts = [plan, rolling_summary, recent_timeline, recent_achievements]
    if estimate_text_tokens("\n\n".join(part for part in parts if part.strip())) <= token_budget:
        return plan, rolling_summary, recent_timeline, recent_achievements
    entries = _timeline_entries(recent_timeline)
    while (
        len(entries) > 1
        and estimate_text_tokens("\n\n".join(part for part in parts if part.strip()))
        > token_budget
    ):
        entries = entries[1:]
        recent_timeline = "\n\n".join(entries).strip()
        parts = [plan, rolling_summary, recent_timeline, recent_achievements]
    if estimate_text_tokens("\n\n".join(part for part in parts if part.strip())) > token_budget:
        recent_timeline = _bounded_text(recent_timeline, token_budget=1000)
    parts = [plan, rolling_summary, recent_timeline, recent_achievements]
    if estimate_text_tokens("\n\n".join(part for part in parts if part.strip())) > token_budget:
        plan = _bounded_text(plan, token_budget=1200)
    return plan, rolling_summary, recent_timeline, recent_achievements


def _bounded_text(text: str, *, token_budget: int) -> str:
    clean = text.strip()
    if not clean or estimate_text_tokens(clean) <= token_budget:
        return clean
    target_chars = max(200, token_budget * 4)
    head = clean[: int(target_chars * 0.7)].rstrip()
    tail = clean[-int(target_chars * 0.25) :].lstrip()
    omitted = max(0, len(clean) - len(head) - len(tail))
    return (
        f"{head}\n\n...[truncated task context: {omitted} chars omitted]...\n\n{tail}"
    ).strip()


def _xml_text(text: str) -> str:
    return escape(text, {'"': "&quot;"})
