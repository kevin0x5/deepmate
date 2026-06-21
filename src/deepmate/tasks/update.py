"""Model-assisted Task Mode maintenance."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

from deepmate.domain import Message, MessageRole
from deepmate.providers import ModelConversationItem, ModelProvider, ModelRequest
from deepmate.tasks.json_helpers import strip_fenced_json
from deepmate.tasks.render import (
    TaskDocuments,
    TaskStage,
    extract_rolling_summary,
    extract_recent_timeline,
)
from deepmate.tasks.store import (
    TaskStore,
    execution_contract_gaps,
    validate_real_plan_content,
)

MAX_TIMELINE_ENTRIES = 200
LOW_SIGNAL_EXECUTE_FINAL_ANSWER_CHARS = 160
TASK_PROGRESS_FILE_RE = re.compile(
    r"\b\S+\.(?:py|ts|tsx|js|jsx|md|json|yaml|yml)\b",
    re.IGNORECASE,
)
TASK_PROGRESS_WORK_VERBS = (
    "implemented",
    "updated",
    "changed",
    "fixed",
    "resolved",
    "added",
    "removed",
    "renamed",
    "refactored",
    "tested",
    "verified",
    "wrote",
    "created",
    "deleted",
    "modified",
)
TASK_PROGRESS_CJK_WORK_VERBS = (
    "实现",
    "修复",
    "更新",
    "修改",
    "新增",
    "删除",
    "重构",
    "验证",
    "测试",
)
TASK_PROGRESS_STRONG_SIGNALS = (
    "tests passed",
    "test passed",
    "all tests passed",
    "tests failed",
    "test failed",
    "blocked",
    "unblocked",
    "next step",
    "todo",
    "commit",
    "pull request",
    "pr ready",
)
TASK_PROGRESS_CJK_STRONG_SIGNALS = (
    "验证通过",
    "测试通过",
    "测试失败",
    "失败",
    "阻塞",
    "解除阻塞",
    "下一步",
    "待办",
    "已完成",
)

TASK_UPDATE_SYSTEM_PROMPT = """You maintain Deepmate Task Mode files for a single project.

Output one JSON object only, without markdown fences.

Fields:
- plan_md: string. Updated task/plan.md. Keep it as the current working plan, not permanent history.
- rolling_summary: array of 5 short bullet strings for evolution.md Rolling Summary. Empty array means keep existing.
- timeline_entry: string. One markdown timeline entry starting with "### YYYY-MM-DD | ...", or empty.
- achievement_title: string. Short title for a new achievement file, required only when achievement_required=yes.
- achievement_md: string. Full markdown achievement file content, required only when achievement_required=yes.

Rules:
- Durable project history belongs in evolution.md and achievements/, not in plan.md.
- Add timeline_entry only for goal changes, key milestones, decision changes, blockers, or direction corrections.
- Do not add timeline entries for routine tool calls, ordinary tests, file reads, or implementation details that do not affect future direction.
- Keep Rolling Summary exactly 5 bullets: long-term goal, completed stages, current stage, key decisions, next step.
- When achievement_required=yes, always produce achievement_title, achievement_md, timeline_entry, and rolling_summary.
- For stage=plan, focus on clarifying/updating the task contract: goal, acceptance contract, scope, execution plan, verification strategy, risks, and decisions.
- For stage=execute, update acceptance progress, execution progress, verification records, blockers, and next step in plan_md.
- For stage=checkpoint, create a checkpoint achievement without implying the whole task is complete unless the evidence says it is.
"""


@dataclass(frozen=True, slots=True)
class TaskUpdateResult:
    """Parsed model output for one Task Mode maintenance pass."""

    plan_md: str = ""
    rolling_summary: tuple[str, ...] = ()
    timeline_entry: str = ""
    achievement_title: str = ""
    achievement_md: str = ""
    fallback_reason: str = ""

    def has_changes(self) -> bool:
        """Return whether this update contains any file change."""
        return bool(
            self.plan_md.strip()
            or self.rolling_summary
            or self.timeline_entry.strip()
            or self.achievement_md.strip()
        )


def generate_task_update(
    provider: ModelProvider,
    *,
    model: str,
    stage: TaskStage,
    documents: TaskDocuments,
    user_prompt: str,
    final_answer: str,
    achievement_required: bool = False,
) -> TaskUpdateResult:
    """Generate a Task Mode file update from the completed user turn."""
    request = ModelRequest(
        model=model,
        conversation=(
            ModelConversationItem.from_message(
                Message(role=MessageRole.SYSTEM, content=TASK_UPDATE_SYSTEM_PROMPT)
            ),
            ModelConversationItem.from_message(
                Message(
                    role=MessageRole.USER,
                    content=_task_update_user_prompt(
                        stage=stage,
                        documents=documents,
                        user_prompt=user_prompt,
                        final_answer=final_answer,
                        achievement_required=achievement_required,
                    ),
                )
            ),
        ),
        options={"temperature": 0, "max_tokens": 1400},
    )
    response = provider.complete(request)
    try:
        result = parse_task_update_response(response.content)
    except ValueError as exc:
        if achievement_required:
            return fallback_achievement_update(
                stage=stage,
                user_prompt=user_prompt,
                final_answer=final_answer,
                reason=str(exc),
            )
        return TaskUpdateResult(fallback_reason=str(exc))
    if achievement_required and not result.achievement_md.strip():
        fallback = fallback_achievement_update(
            stage=stage,
            user_prompt=user_prompt,
            final_answer=final_answer,
            reason="achievement_md missing",
        )
        return TaskUpdateResult(
            plan_md=result.plan_md,
            rolling_summary=result.rolling_summary,
            timeline_entry=result.timeline_entry or fallback.timeline_entry,
            achievement_title=fallback.achievement_title,
            achievement_md=fallback.achievement_md,
            fallback_reason=fallback.fallback_reason,
        )
    return result


def parse_task_update_response(content: str) -> TaskUpdateResult:
    """Parse a model JSON response for Task Mode maintenance."""
    payload = _parse_json_object(content)
    rolling = payload.get("rolling_summary")
    if not isinstance(rolling, list):
        rolling_summary: tuple[str, ...] = ()
    else:
        rolling_summary = tuple(
            str(item).strip().lstrip("- ").strip()
            for item in rolling[:5]
            if str(item).strip()
        )
    return TaskUpdateResult(
        plan_md=str(payload.get("plan_md", "")).strip(),
        rolling_summary=rolling_summary,
        timeline_entry=str(payload.get("timeline_entry", "")).strip(),
        achievement_title=str(payload.get("achievement_title", "")).strip(),
        achievement_md=str(payload.get("achievement_md", "")).strip(),
    )


def apply_task_update_result(
    store: TaskStore,
    result: TaskUpdateResult,
    *,
    stage: TaskStage,
    documents: TaskDocuments | None = None,
    allow_achievement: bool = True,
) -> tuple[Path, ...]:
    """Apply a parsed Task Mode update and return changed paths."""
    changed: list[Path] = []
    if result.plan_md.strip():
        if documents is None or result.plan_md.strip() != documents.plan.strip():
            validate_real_plan_content(result.plan_md)
            if stage == TaskStage.EXECUTE and execution_contract_gaps(result.plan_md):
                raise ValueError(
                    "task/execute update would remove required contract sections"
                )
            store.write_plan(result.plan_md)
            changed.append(store.plan_path)
    if result.rolling_summary or result.timeline_entry.strip():
        changed_evolution = store.update_evolution(
            lambda current: update_evolution_markdown(
                current,
                rolling_summary=result.rolling_summary,
                timeline_entry=result.timeline_entry,
            )
        )
        if changed_evolution:
            changed.append(store.evolution_path)
    if allow_achievement and result.achievement_md.strip():
        path = store.append_achievement(
            result.achievement_title or _achievement_title(result.achievement_md),
            result.achievement_md,
        )
        changed.append(path)
    return tuple(changed)


def should_run_task_update(
    stage: TaskStage,
    *,
    user_prompt: str,
    final_answer: str,
) -> bool:
    """Return whether Task Mode should spend a model call on maintenance.

    Planning and checkpointing are explicit task-document transitions, so they
    always update. Execute is a contract-driven loop; every successful execute
    turn updates progress and verification evidence for the evaluator.
    """
    if stage in {TaskStage.PLAN, TaskStage.EXECUTE, TaskStage.CHECKPOINT}:
        return True
    return False


def update_evolution_markdown(
    content: str,
    *,
    rolling_summary: tuple[str, ...] = (),
    timeline_entry: str = "",
) -> str:
    """Update Rolling Summary in place and append one Timeline entry."""
    current = content.strip()
    if not current:
        current = "# 任务演化链\n\n## Rolling Summary\n\n## Timeline"
    if rolling_summary:
        current = _replace_rolling_summary(current, rolling_summary)
    if timeline_entry.strip():
        current = _append_timeline_entry(current, timeline_entry)
    return current.strip() + "\n"


def _contains_progress_signal(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False
    has_file_ref = TASK_PROGRESS_FILE_RE.search(normalized) is not None
    has_work_verb = _contains_ascii_progress_marker(
        normalized,
        TASK_PROGRESS_WORK_VERBS,
    ) or _contains_cjk_progress_marker(normalized, TASK_PROGRESS_CJK_WORK_VERBS)
    if has_file_ref and has_work_verb:
        return True
    return _contains_ascii_progress_marker(
        normalized,
        TASK_PROGRESS_STRONG_SIGNALS,
    ) or _contains_cjk_progress_marker(normalized, TASK_PROGRESS_CJK_STRONG_SIGNALS)


def _contains_ascii_progress_marker(text: str, markers: tuple[str, ...]) -> bool:
    for marker in markers:
        pattern = r"(?<![a-z0-9_])" + re.escape(marker) + r"(?![a-z0-9_])"
        if re.search(pattern, text):
            return True
    return False


def _contains_cjk_progress_marker(text: str, markers: tuple[str, ...]) -> bool:
    for marker in markers:
        if marker not in text:
            continue
        pattern = re.escape(marker)
        if marker in {"更新"}:
            pattern += r"(?!日志|记录|说明|历史|文档)"
        if marker in {"修改"}:
            pattern += r"(?!记录|说明|历史)"
        if marker in {"测试"}:
            pattern += r"(?!计划|环境|策略)"
        if marker in {"验证"}:
            pattern += r"(?!计划|方式|策略)"
        if re.search(pattern, text):
            return True
    return False


def fallback_achievement_update(
    *,
    stage: TaskStage,
    user_prompt: str,
    final_answer: str,
    reason: str,
) -> TaskUpdateResult:
    """Create a deterministic achievement if the model response is unusable."""
    title = _short_title(user_prompt) or "task-checkpoint"
    answer = final_answer.strip() or "See the session transcript for details."
    achievement = "\n".join(
        (
            f"# Task checkpoint: {title}",
            "",
            "## 本轮完成",
            f"- {answer[:500]}",
            "",
            "## 关键决策",
            "- See task/evolution.md and the session transcript for detailed decisions.",
            "",
            "## 产物",
            "- See the workspace diff and transcript for concrete artifacts.",
            "",
            "## 下一步",
            "- Continue from task/plan.md.",
        )
    )
    return TaskUpdateResult(
        timeline_entry=(
            f"### {_today()} | task checkpoint\n"
            f"- 本轮生成 checkpoint achievement：{title}。\n"
            f"- task update 使用 fallback：{reason}。"
        ),
        achievement_title=title,
        achievement_md=achievement,
        fallback_reason=reason,
    )


def _task_update_user_prompt(
    *,
    stage: TaskStage,
    documents: TaskDocuments,
    user_prompt: str,
    final_answer: str,
    achievement_required: bool = False,
) -> str:
    plan = _xml_text(_bounded_prompt_text(documents.plan.strip(), max_chars=12_000))
    rolling = _xml_text(extract_rolling_summary(documents.evolution))
    timeline = _xml_text(extract_recent_timeline(documents.evolution, limit=5))
    return "\n\n".join(
        (
            f"Stage: {stage.value}",
            f"Achievement required: {'yes' if achievement_required else 'no'}",
            "<current_plan>\n" + plan + "\n</current_plan>",
            "<rolling_summary>\n" + rolling + "\n</rolling_summary>",
            "<recent_timeline>\n" + timeline + "\n</recent_timeline>",
            f"<user_prompt>\n{_xml_text(user_prompt.strip())}\n</user_prompt>",
            "<assistant_final_answer>\n"
            + _xml_text(final_answer.strip())
            + "\n</assistant_final_answer>",
        )
    )


def _parse_json_object(content: str) -> dict[str, object]:
    stripped = strip_fenced_json(content).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"task update response must be JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("task update response must be a JSON object")
    return payload



def _replace_rolling_summary(content: str, bullets: tuple[str, ...]) -> str:
    summary = "\n".join(f"- {bullet}" for bullet in bullets[:5])
    replacement = f"## Rolling Summary\n{summary}"
    lines = content.splitlines()
    start = _heading_index(lines, "## Rolling Summary")
    if start < 0:
        return content.rstrip() + "\n\n" + replacement
    end = _next_h2_index(lines, start + 1)
    new_lines = [*lines[:start], *replacement.splitlines()]
    if end >= 0:
        new_lines.extend(lines[end:])
    return "\n".join(new_lines)


def _append_timeline_entry(content: str, entry: str) -> str:
    clean_entry = entry.strip()
    if not clean_entry.startswith("### "):
        clean_entry = f"### {_today()} | 任务演化\n{clean_entry}"
    if _heading_index(content.splitlines(), "## Timeline") < 0:
        content = content.rstrip() + "\n\n## Timeline\n"
    appended = content.rstrip() + "\n\n" + clean_entry
    return _trim_timeline_entries(appended, limit=MAX_TIMELINE_ENTRIES)


def _heading_index(lines: list[str], heading: str) -> int:
    for index, line in enumerate(lines):
        if line.strip() == heading:
            return index
    return -1


def _next_h2_index(lines: list[str], start: int) -> int:
    for index in range(start, len(lines)):
        if lines[index].strip().startswith("## "):
            return index
    return -1


def _trim_timeline_entries(content: str, *, limit: int) -> str:
    lines = content.splitlines()
    timeline_index = _heading_index(lines, "## Timeline")
    if timeline_index < 0:
        return content
    next_section = _next_h2_index(lines, timeline_index + 1)
    if next_section < 0:
        before = lines[: timeline_index + 1]
        timeline_lines = lines[timeline_index + 1 :]
        after: list[str] = []
    else:
        before = lines[: timeline_index + 1]
        timeline_lines = lines[timeline_index + 1 : next_section]
        after = lines[next_section:]
    entries = _timeline_entries("\n".join(timeline_lines))
    if len(entries) <= limit:
        return content
    trimmed_count = _timeline_trimmed_count(timeline_lines) + len(entries) - limit
    trimmed = entries[-limit:]
    marker = f"<!-- trimmed {trimmed_count} older timeline entries -->"
    rebuilt = [*before, marker, "", *"\n\n".join(trimmed).splitlines()]
    if after:
        rebuilt.extend(["", *after])
    return "\n".join(rebuilt)


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


def _timeline_trimmed_count(lines: list[str]) -> int:
    for line in lines:
        match = re.search(r"trimmed\s+(\d+)\s+older timeline entries", line)
        if match:
            return int(match.group(1))
    return 0


def _achievement_title(content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].replace("阶段达成：", "").strip()
    return "task-achievement"


def _short_title(value: str) -> str:
    words = " ".join(value.strip().split())
    if not words:
        return ""
    return words[:40].strip()


def _today() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d")


def _xml_text(text: str) -> str:
    return escape(text, {'"': "&quot;"})


def _bounded_prompt_text(text: str, *, max_chars: int) -> str:
    clean = text.strip()
    if len(clean) <= max_chars:
        return clean
    head = clean[: int(max_chars * 0.7)].rstrip()
    tail = clean[-int(max_chars * 0.25) :].lstrip()
    omitted = max(0, len(clean) - len(head) - len(tail))
    return f"{head}\n\n...[truncated task document: {omitted} chars omitted]...\n\n{tail}"
