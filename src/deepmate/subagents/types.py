"""Request and result objects for Deepmate subagent runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from deepmate.domain import ErrorInfo
from deepmate.providers import TokenUsage
from deepmate.runtime.tool_policy import ToolAccessMode


class SubagentRunStatus(StrEnum):
    """Stable status values for one subagent run."""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    MAX_STEPS_REACHED = "max_steps_reached"


@dataclass(frozen=True, slots=True)
class SubagentRunRequest:
    """Minimal request used to start one child runtime."""

    run_id: str | None = None
    goal: str = ""
    input_context: str = ""
    output_contract: str | None = None
    acceptance_criteria: tuple[str, ...] = field(default_factory=tuple)
    allowed_tools: tuple[str, ...] = field(default_factory=tuple)
    tool_access_mode: ToolAccessMode = ToolAccessMode.READ_ONLY
    model_purpose: str | None = None
    max_steps: int = 3
    parent_session_id: str | None = None
    parent_activation_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _optional_text(self.run_id))
        object.__setattr__(self, "goal", self.goal.strip())
        object.__setattr__(self, "input_context", self.input_context.strip())
        object.__setattr__(
            self,
            "output_contract",
            _optional_text(self.output_contract),
        )
        object.__setattr__(
            self,
            "acceptance_criteria",
            _unique_texts(self.acceptance_criteria),
        )
        object.__setattr__(
            self,
            "allowed_tools",
            _unique_texts(self.allowed_tools),
        )
        object.__setattr__(
            self,
            "model_purpose",
            _optional_text(self.model_purpose),
        )
        object.__setattr__(
            self,
            "parent_session_id",
            _optional_text(self.parent_session_id),
        )
        object.__setattr__(
            self,
            "parent_activation_id",
            _optional_text(self.parent_activation_id),
        )
        if not isinstance(self.tool_access_mode, ToolAccessMode):
            object.__setattr__(
                self,
                "tool_access_mode",
                ToolAccessMode(str(self.tool_access_mode)),
            )

    def is_ready(self) -> bool:
        """Return whether the request has enough detail to run."""
        return bool(self.goal and self.max_steps >= 1)


@dataclass(frozen=True, slots=True)
class SubagentRunResult:
    """Structured result returned after one subagent run."""

    run_id: str
    status: SubagentRunStatus
    summary: str
    artifact_refs: tuple[str, ...] = field(default_factory=tuple)
    evidence_refs: tuple[str, ...] = field(default_factory=tuple)
    error: ErrorInfo | None = None
    usage: TokenUsage | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", self.run_id.strip())
        object.__setattr__(self, "summary", self.summary.strip())
        object.__setattr__(self, "artifact_refs", _unique_texts(self.artifact_refs))
        object.__setattr__(self, "evidence_refs", _unique_texts(self.evidence_refs))
        if not isinstance(self.status, SubagentRunStatus):
            object.__setattr__(self, "status", SubagentRunStatus(str(self.status)))

    def is_ready(self) -> bool:
        """Return whether the result has the minimum fields to be consumed."""
        return bool(self.run_id and self.summary)

    def is_success(self) -> bool:
        """Return whether the child runtime completed without an error."""
        return self.status == SubagentRunStatus.COMPLETED and self.error is None


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _unique_texts(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        text = value.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return tuple(normalized)
