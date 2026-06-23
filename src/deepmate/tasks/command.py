"""User-facing Task Mode command parsing."""

from __future__ import annotations

from dataclasses import dataclass

from deepmate.tasks.render import TaskStage


TASK_STATUS = "status"
TASK_CLEAR = "clear"


@dataclass(frozen=True, slots=True)
class TaskPromptCommand:
    """Parsed `task/<stage>` command embedded in a user prompt."""

    stage: TaskStage | None
    prompt: str
    control: str = ""

    def is_control(self) -> bool:
        """Return whether this command should be handled locally."""
        return bool(self.control)


def parse_task_prompt_command(prompt: str) -> TaskPromptCommand | None:
    """Parse Task Mode commands from a prompt."""
    text = _normalize_task_prompt(prompt)
    lowered = text.lower()
    for control in (TASK_STATUS, TASK_CLEAR):
        prefix = f"task/{control}"
        if lowered == prefix:
            return TaskPromptCommand(stage=None, prompt="", control=control)
        if lowered.startswith(prefix + " "):
            return TaskPromptCommand(
                stage=None,
                prompt=text[len(prefix) :].strip(),
                control=control,
            )
    for stage in TaskStage:
        prefix = f"task/{stage.value}"
        if lowered == prefix:
            return TaskPromptCommand(stage=stage, prompt="")
        if lowered.startswith(prefix + " "):
            return TaskPromptCommand(stage=stage, prompt=text[len(prefix) :].strip())
    return None


def _normalize_task_prompt(prompt: str) -> str:
    text = prompt.strip()
    if not text.startswith("/task"):
        return text
    parts = text.split(maxsplit=2)
    if not parts or parts[0] != "/task":
        return text
    if len(parts) == 1:
        return "task/status"
    raw_stage = parts[1].strip().lower()
    remainder = parts[2].strip() if len(parts) >= 3 else ""
    aliases = {
        "--plan": "plan",
        "plan": "plan",
        "--execute": "execute",
        "execute": "execute",
        "--checkpoint": "checkpoint",
        "checkpoint": "checkpoint",
        "--status": "status",
        "status": "status",
        "--clear": "clear",
        "clear": "clear",
    }
    stage = aliases.get(raw_stage)
    if stage is None:
        return text
    return f"task/{stage}{' ' + remainder if remainder else ''}"


def default_task_prompt(stage: TaskStage) -> str:
    """Return a useful default prompt when the user only switches task stage."""
    if stage == TaskStage.PLAN:
        return (
            "Continue planning the current project task. Clarify the goal, "
            "acceptance contract, scope, execution plan, verification strategy, "
            "risks, and open decisions in task/plan.md."
        )
    if stage == TaskStage.EXECUTE:
        return (
            "Execute task/plan.md. Work against its acceptance contract, verify "
            "with evidence, and continue until the contract is satisfied or a "
            "clear blocker is reached."
        )
    return "Create a checkpoint achievement for the current project task work."


def persisted_task_stage(stage: TaskStage, previous: TaskStage | None) -> TaskStage:
    """Return the cursor stage that should persist after a task-mode turn."""
    if stage == TaskStage.CHECKPOINT:
        return previous if previous is not None else TaskStage.PLAN
    return stage
