"""Project-level Task Mode storage."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections.abc import Callable

from deepmate.foundation.time import normal_datetime, utc_isoformat
from deepmate.storage.atomic import atomic_write_json, atomic_write_text, file_lock
from deepmate.tasks.json_helpers import contract_status_items
from deepmate.tasks.render import TaskDocuments, TaskStage, build_task_context

TASK_DIR = Path("task")
ACHIEVEMENTS_DIR = "achievements"
LOCAL_STATE_PATH = Path(".deepmate/task_mode.json")
PLAN_FILE = "plan.md"
EVOLUTION_FILE = "evolution.md"
WRITE_CHECKPOINT = Callable[[str, Path, str], None]


class DegenerateTaskPlanError(ValueError):
    """Raised when an update would replace task/plan.md with unusable content."""


EXECUTION_CONTRACT_SECTIONS = (
    ("goal", ("## 大目标", "## 目标", "## Goal")),
    ("acceptance_contract", ("## 验收契约", "## 验收标准", "## Acceptance Contract")),
    ("execution_plan", ("## 执行步骤", "## 执行计划", "## Steps", "## Execution Plan")),
    ("verification_strategy", ("## 验证策略", "## Verification Strategy")),
)


@dataclass(frozen=True, slots=True)
class TaskModeState:
    """Local Task Mode runtime cursor for single-turn CLI sessions."""

    stage: TaskStage
    updated_at: str
    last_session_id: str = ""
    execute_status: str = ""
    execute_turns: int = 0
    last_reason: str = ""
    next_instruction: str = ""
    contract_status: tuple[str, ...] = ()

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable local state record."""
        return {
            "stage": self.stage.value,
            "updated_at": self.updated_at,
            "last_session_id": self.last_session_id,
            "execute_status": self.execute_status,
            "execute_turns": self.execute_turns,
            "last_reason": self.last_reason,
            "next_instruction": self.next_instruction,
            "contract_status": list(self.contract_status),
        }

    @classmethod
    def from_record(cls, record: object) -> "TaskModeState | None":
        """Read local state from a decoded JSON object."""
        if not isinstance(record, dict):
            return None
        stage = TaskStage.parse(str(record.get("stage", "")))
        if stage is None:
            return None
        status = str(record.get("execute_status") or "").strip()
        instruction = str(record.get("next_instruction") or "").strip()
        return cls(
            stage=stage,
            updated_at=str(record.get("updated_at", "")).strip(),
            last_session_id=str(record.get("last_session_id", "")).strip(),
            execute_status=status,
            execute_turns=_int_record_value(record.get("execute_turns")),
            last_reason=str(record.get("last_reason", "")).strip(),
            next_instruction="" if status == "achieved" else instruction,
            contract_status=contract_status_items(record.get("contract_status")),
        )


class TaskStore:
    """Read and write the project-level single task line."""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).resolve()
        self._write_checkpoint: WRITE_CHECKPOINT | None = None

    def set_write_checkpoint(self, callback: WRITE_CHECKPOINT | None) -> None:
        """Set an optional checkpoint callback for Task Mode file writes."""
        self._write_checkpoint = callback

    @property
    def task_dir(self) -> Path:
        """Return the project task directory."""
        return self.workspace / TASK_DIR

    @property
    def plan_path(self) -> Path:
        """Return task/plan.md."""
        return self.task_dir / PLAN_FILE

    @property
    def evolution_path(self) -> Path:
        """Return task/evolution.md."""
        return self.task_dir / EVOLUTION_FILE

    @property
    def achievements_dir(self) -> Path:
        """Return task/achievements/."""
        return self.task_dir / ACHIEVEMENTS_DIR

    @property
    def state_path(self) -> Path:
        """Return the local stage cursor path."""
        return self.workspace / LOCAL_STATE_PATH

    def ensure(self) -> None:
        """Create the task directory and default markdown files."""
        with file_lock(self.task_dir / ".task-store"):
            self.achievements_dir.mkdir(parents=True, exist_ok=True)
            if not self.plan_path.exists():
                self._write_task_text(
                    self.plan_path,
                    default_plan_markdown(),
                    operation="task.ensure.plan",
                )
            if not self.evolution_path.exists():
                self._write_task_text(
                    self.evolution_path,
                    default_evolution_markdown(),
                    operation="task.ensure.evolution",
                )

    def read_documents(self) -> TaskDocuments:
        """Read project task markdown documents."""
        return TaskDocuments(
            plan=_read_text(self.plan_path),
            evolution=_read_text(self.evolution_path),
            achievements=self.read_achievements(),
        )

    def read_achievements(self) -> tuple[tuple[Path, str], ...]:
        """Read achievement markdown files ordered by file name."""
        if not self.achievements_dir.exists():
            return ()
        items: list[tuple[Path, str]] = []
        for path in sorted(self.achievements_dir.glob("*.md")):
            content = _read_text(path)
            if content.strip():
                items.append((path, content))
        return tuple(items)

    def read_state(self) -> TaskModeState | None:
        """Read the local task stage cursor."""
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        return TaskModeState.from_record(data)

    def resolve_stage(self, requested: TaskStage | None) -> TaskStage:
        """Resolve the current stage from explicit input or the local cursor."""
        if requested is not None:
            return requested
        state = self.read_state()
        if state is not None:
            return state.stage
        return TaskStage.PLAN

    def save_state(
        self,
        stage: TaskStage,
        *,
        session_id: str = "",
        now: datetime | None = None,
        execute_status: str = "",
        execute_turns: int | None = None,
        last_reason: str | None = None,
        next_instruction: str | None = None,
        contract_status: tuple[str, ...] | None = None,
    ) -> TaskModeState:
        """Persist the local task stage cursor."""
        with file_lock(self.state_path):
            previous = self.read_state()
            if stage == TaskStage.CHECKPOINT:
                stage = previous.stage if previous is not None else TaskStage.PLAN
            if stage != TaskStage.PLAN and not has_real_user_plan(
                _read_text(self.plan_path)
            ):
                raise ValueError(
                    "Task Mode needs a real task/plan.md before persisting execute state"
                )
            current = utc_isoformat(normal_datetime(now))
            prior = previous if stage == TaskStage.EXECUTE else None
            status = execute_status.strip() or (prior.execute_status if prior else "")
            instruction = (
                next_instruction.strip() if next_instruction is not None
                else (prior.next_instruction if prior else "")
            )
            state = TaskModeState(
                stage=stage,
                updated_at=current,
                last_session_id=session_id.strip(),
                execute_status=status,
                execute_turns=(
                    max(0, execute_turns)
                    if execute_turns is not None
                    else (prior.execute_turns if prior else 0)
                ),
                last_reason=(
                    last_reason.strip()
                    if last_reason is not None
                    else (prior.last_reason if prior else "")
                ),
                next_instruction="" if status == "achieved" else instruction,
                contract_status=(
                    contract_status if contract_status is not None
                    else (prior.contract_status if prior else ())
                ),
            )
            atomic_write_json(self.state_path, state.to_record())
        return state

    def clear_state(self) -> bool:
        """Delete the local runtime state file."""
        with file_lock(self.state_path):
            try:
                self.state_path.unlink()
            except FileNotFoundError:
                return False
        return True

    def context_for_stage(self, stage: TaskStage):
        """Return bounded task context for prompt injection."""
        self.ensure()
        return build_task_context(self.read_documents(), stage)

    def write_plan(self, content: str) -> None:
        """Write task/plan.md."""
        clean = content.strip()
        if not _looks_like_real_plan(clean):
            raise DegenerateTaskPlanError(
                "refusing to replace task/plan.md with a degenerate plan"
            )
        with file_lock(self.task_dir / ".task-store"):
            self._write_task_text(
                self.plan_path,
                _ensure_trailing_newline(clean),
                operation="task.write_plan",
            )

    def write_evolution(self, content: str) -> None:
        """Write task/evolution.md."""
        with file_lock(self.task_dir / ".task-store"):
            self._write_task_text(
                self.evolution_path,
                _ensure_trailing_newline(
                    content.strip() or default_evolution_markdown()
                ),
                operation="task.write_evolution",
            )

    def append_achievement(self, title: str, content: str) -> Path:
        """Create one achievement markdown file and return its path."""
        clean = content.strip()
        digest = _content_digest(clean)
        with file_lock(self.task_dir / ".task-store"):
            self.achievements_dir.mkdir(parents=True, exist_ok=True)
            for path in sorted(self.achievements_dir.glob("*.md")):
                if _content_digest(_read_text(path).strip()) == digest:
                    return path
            slug = _slug(title) or "task-achievement"
            stamp = datetime.now().strftime("%Y-%m-%d")
            base = self.achievements_dir / f"{stamp}-{slug}.md"
            path = _unique_path(base)
            self._write_task_text(
                path,
                _ensure_trailing_newline(clean),
                operation="task.append_achievement",
            )
            return path

    def update_evolution(
        self,
        updater: Callable[[str], str],
    ) -> bool:
        """Atomically read, update, and write task/evolution.md."""
        with file_lock(self.task_dir / ".task-store"):
            before = _read_text(self.evolution_path)
            after = updater(before)
            if after.strip() == before.strip():
                return False
            self._write_task_text(
                self.evolution_path,
                _ensure_trailing_newline(
                    after.strip() or default_evolution_markdown()
                ),
                operation="task.write_evolution",
            )
            return True

    def apply_update(
        self,
        *,
        plan_content: str = "",
        evolution_updater: Callable[[str], str] | None = None,
        achievement_title: str = "",
        achievement_content: str = "",
    ) -> tuple[Path, ...]:
        """Apply one multi-file Task Mode update with rollback on failure."""
        clean_plan = plan_content.strip()
        if clean_plan and not _looks_like_real_plan(clean_plan):
            raise DegenerateTaskPlanError(
                "refusing to replace task/plan.md with a degenerate plan"
            )
        clean_achievement = achievement_content.strip()
        with file_lock(self.task_dir / ".task-store"):
            mutations: list[tuple[Path, str, str, bool, str]] = []
            changed: list[Path] = []

            old_plan = _read_text(self.plan_path)
            if clean_plan and clean_plan != old_plan.strip():
                mutations.append(
                    (
                        self.plan_path,
                        _ensure_trailing_newline(clean_plan),
                        "task.write_plan",
                        self.plan_path.exists(),
                        old_plan,
                    )
                )
                changed.append(self.plan_path)

            old_evolution = _read_text(self.evolution_path)
            if evolution_updater is not None:
                updated_evolution = evolution_updater(old_evolution).strip()
                if updated_evolution != old_evolution.strip():
                    mutations.append(
                        (
                            self.evolution_path,
                            _ensure_trailing_newline(
                                updated_evolution or default_evolution_markdown()
                            ),
                            "task.write_evolution",
                            self.evolution_path.exists(),
                            old_evolution,
                        )
                    )
                    changed.append(self.evolution_path)

            if clean_achievement:
                digest = _content_digest(clean_achievement)
                existing = next(
                    (
                        path
                        for path in sorted(self.achievements_dir.glob("*.md"))
                        if _content_digest(_read_text(path).strip()) == digest
                    ),
                    None,
                )
                if existing is not None:
                    changed.append(existing)
                else:
                    self.achievements_dir.mkdir(parents=True, exist_ok=True)
                    slug = _slug(achievement_title) or "task-achievement"
                    stamp = datetime.now().strftime("%Y-%m-%d")
                    achievement_path = _unique_path(
                        self.achievements_dir / f"{stamp}-{slug}.md"
                    )
                    mutations.append(
                        (
                            achievement_path,
                            _ensure_trailing_newline(clean_achievement),
                            "task.append_achievement",
                            False,
                            "",
                        )
                    )
                    changed.append(achievement_path)

            applied: list[tuple[Path, bool, str]] = []
            try:
                for path, content, operation, old_exists, old_content in mutations:
                    self._write_task_text(path, content, operation=operation)
                    applied.append((path, old_exists, old_content))
            except Exception as exc:
                rollback_error: OSError | None = None
                for path, old_exists, old_content in reversed(applied):
                    try:
                        if old_exists:
                            atomic_write_text(path, old_content)
                        else:
                            path.unlink(missing_ok=True)
                    except OSError as restore_exc:
                        rollback_error = rollback_error or restore_exc
                if rollback_error is not None:
                    raise RuntimeError(
                        "Task Mode update failed and rollback was incomplete: "
                        f"{rollback_error}"
                    ) from exc
                raise
            return tuple(changed)

    def _write_task_text(self, path: Path, content: str, *, operation: str) -> None:
        if self._write_checkpoint is not None:
            self._write_checkpoint(operation, path, content)
        atomic_write_text(path, content)


def default_plan_markdown() -> str:
    """Return the default task contract template."""
    return """# 当前任务计划

## 大目标
说明本任务要达成的最终结果。

## 验收契约
- [ ] 可验证的完成条件 1
- [ ] 可验证的完成条件 2

## 范围与非目标
- 范围：
- 非目标：

## 当前方案
- 当前采用的方案、边界和关键取舍。

## 执行步骤
- [ ] 步骤 1
- [ ] 步骤 2

## 验证策略
- 需要运行的测试、检查、人工验证或替代证据。

## 风险与待确认问题
- 风险：
- 待确认：

## 讨论与决策
- 影响后续执行的用户确认、方案取舍和排除项。

## 当前进展
- 已完成：
- 进行中：
- 下一步：
"""


def default_evolution_markdown() -> str:
    """Return the default task evolution template."""
    return """# 任务演化链

## Rolling Summary
- 长期目标：
- 已完成阶段：
- 当前阶段：
- 关键决策：
- 下一步：

## Timeline
"""


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _int_record_value(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, str) and value.strip().isdigit():
        return max(0, int(value.strip()))
    return 0


def _ensure_trailing_newline(content: str) -> str:
    return content if content.endswith("\n") else content + "\n"


def _slug(value: str) -> str:
    text = value.strip().lower()
    chars: list[str] = []
    previous_dash = False
    for char in text:
        if char.isalnum() or "\u4e00" <= char <= "\u9fff":
            chars.append(char)
            previous_dash = False
        else:
            if not previous_dash:
                chars.append("-")
                previous_dash = True
    return "".join(chars).strip("-")[:80].strip("-")


def _looks_like_real_plan(content: str) -> bool:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    headings = [line for line in lines if line.startswith("#")]
    if not headings:
        return False
    if content.strip() in {"# Plan", "# Current Plan", "# 当前任务计划"}:
        return False
    if len(content.strip()) < 24:
        return False
    return True


def has_real_user_plan(content: str) -> bool:
    """Return whether content is a non-template usable task plan."""
    clean = content.strip()
    if not _looks_like_real_plan(clean):
        return False
    return clean != default_plan_markdown().strip()


def validate_real_plan_content(content: str) -> None:
    """Raise when content is not a usable task plan."""
    if not _looks_like_real_plan(content.strip()):
        raise DegenerateTaskPlanError(
            "refusing to replace task/plan.md with a degenerate plan"
        )


def execution_contract_gaps(content: str) -> tuple[str, ...]:
    """Return missing task contract sections required by task/execute."""
    clean = content.strip()
    if not has_real_user_plan(clean):
        return ("real task/plan.md",)
    gaps: list[str] = []
    normalized_lines = {line.strip() for line in clean.splitlines()}
    for label, headings in EXECUTION_CONTRACT_SECTIONS:
        if not any(heading in normalized_lines for heading in headings):
            gaps.append(label)
    return tuple(gaps)


def _content_digest(content: str) -> str:
    return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 1000):
        candidate = path.with_name(f"{stem}-{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise ValueError(f"too many achievement files with base name: {path.name}")
