"""In-process Task Mode state for CLI and interactive sessions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from deepmate.tasks.command import (
    TASK_CLEAR,
    TASK_STATUS,
    default_task_prompt,
    parse_task_prompt_command,
    persisted_task_stage,
)
from deepmate.tasks.execute import execute_start_prompt
from deepmate.tasks.render import TaskContext, TaskStage
from deepmate.tasks.store import TaskStore, execution_contract_gaps, has_real_user_plan


@dataclass(frozen=True, slots=True)
class TaskTurn:
    """Task Mode metadata for one user turn."""

    stage: TaskStage
    prompt: str
    control: str = ""

    def is_control(self) -> bool:
        """Return whether this turn is handled without a model call."""
        return bool(self.control)


class TaskSessionController:
    """Track the current Task Mode stage inside one process."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        enabled: bool = False,
        initial_stage: TaskStage | None = None,
    ) -> None:
        self.store = TaskStore(workspace)
        self._stage: TaskStage | None = None
        self._cursor_stage: TaskStage | None = None
        if enabled:
            self.enable(initial_stage)

    @property
    def active_stage(self) -> TaskStage | None:
        """Return the active task stage, if Task Mode is active."""
        return self._stage

    def enable(self, stage: TaskStage | None = None) -> TaskStage:
        """Enable Task Mode and return the resolved current stage."""
        self.store.ensure()
        self._cursor_stage = self.store.resolve_stage(None)
        resolved = self.store.resolve_stage(stage)
        documents = self.store.read_documents()
        if resolved == TaskStage.EXECUTE:
            gaps = execution_contract_gaps(documents.plan)
            if gaps:
                raise ValueError(
                    "Task Mode needs task/plan.md with goal, acceptance contract, "
                    "execution plan, and verification strategy before task/execute. "
                    f"Missing: {', '.join(gaps)}."
                )
        if resolved == TaskStage.CHECKPOINT and not _has_user_plan(documents.plan):
            raise ValueError(
                "Task Mode needs a real task/plan.md before task/checkpoint. "
                "Run task/plan first or edit task/plan.md."
            )
        self._stage = resolved
        return self._stage

    def prepare_prompt(self, prompt: str) -> TaskTurn | None:
        """Return task metadata for a prompt, activating on `task/<stage>`."""
        command = parse_task_prompt_command(prompt)
        if command is not None:
            if command.is_control():
                return TaskTurn(
                    stage=self._stage or self.store.resolve_stage(None),
                    prompt=command.prompt,
                    control=command.control,
                )
            if command.stage is None:
                return None
            self.enable(command.stage)
            prepared = (
                execute_start_prompt(command.prompt)
                if command.stage == TaskStage.EXECUTE
                else command.prompt or default_task_prompt(command.stage)
            )
            return TaskTurn(stage=command.stage, prompt=prepared)
        if self._stage is None:
            return None
        prepared = (
            execute_start_prompt(prompt)
            if self._stage == TaskStage.EXECUTE
            else prompt.strip() or default_task_prompt(self._stage)
        )
        return TaskTurn(stage=self._stage, prompt=prepared)

    def save_cursor(self, *, session_id: str = "") -> None:
        """Persist the local stage cursor for the active task stage."""
        if self._stage is None:
            return
        state = self.store.save_state(
            persisted_task_stage(self._stage, self._cursor_stage),
            session_id=session_id,
            execute_status="active" if self._stage == TaskStage.EXECUTE else "",
        )
        self._cursor_stage = state.stage

    def finish_turn(self, stage: TaskStage) -> None:
        """Update in-memory stage after a task-mode turn finishes."""
        if stage == TaskStage.CHECKPOINT and self._stage == TaskStage.CHECKPOINT:
            self._stage = self._cursor_stage or TaskStage.EXECUTE

    def handle_control(self, control: str) -> str:
        """Handle a local Task Mode command and return user-facing text."""
        clean = control.strip().lower()
        if clean == TASK_STATUS:
            return self.format_status()
        if clean == TASK_CLEAR:
            removed = self.store.clear_state()
            self._stage = None
            self._cursor_stage = None
            return (
                "Task Mode local runtime state cleared. task/ documents were kept."
                if removed
                else "Task Mode local runtime state was already clear. task/ documents were kept."
            )
        raise ValueError(f"unsupported task control command: {control}")

    def format_status(self) -> str:
        """Return a compact Task Mode status summary."""
        state = self.store.read_state()
        documents = self.store.read_documents()
        active = self._stage or (state.stage if state is not None else None)
        gaps = execution_contract_gaps(documents.plan)
        lines = ["Task Mode status", ""]
        lines.append(f"stage: {active.value if active is not None else 'inactive'}")
        if state is not None:
            if state.execute_status:
                lines.append(f"execute_status: {state.execute_status}")
            if state.execute_turns:
                lines.append(f"execute_turns: {state.execute_turns}")
            if state.last_reason:
                lines.append(f"last_reason: {state.last_reason}")
            if state.next_instruction:
                lines.append(f"next: {state.next_instruction}")
        lines.extend(
            (
                "",
                "documents:",
                f"- task/plan.md: {'ready' if not gaps else 'needs ' + ', '.join(gaps)}",
                f"- task/evolution.md: {'present' if documents.evolution.strip() else 'empty'}",
                f"- achievements: {len(documents.achievements)}",
            )
        )
        return "\n".join(lines)

    def context(self) -> TaskContext | None:
        """Return context for the active task stage."""
        if self._stage is None:
            return None
        return self.store.context_for_stage(self._stage)


def _has_user_plan(plan: str) -> bool:
    return has_real_user_plan(plan)
