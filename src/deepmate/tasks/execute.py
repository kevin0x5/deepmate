"""Execute-stage goal loop helpers for Task Mode."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum

from deepmate.domain import Message, MessageRole
from deepmate.providers import (
    ModelConversationItem,
    ModelProvider,
    ModelRequest,
    ModelToolResult,
)
from deepmate.runtime.agent_loop import UserTurnResult
from deepmate.tasks.json_helpers import strip_fenced_json
from deepmate.tasks.render import TaskDocuments, extract_recent_timeline
from deepmate.tasks.store import TaskModeState

DEFAULT_EXECUTE_TURN_BUDGET = 6


class ExecuteDecision(StrEnum):
    """Evaluator decisions for one execute turn."""

    ACHIEVED = "achieved"
    CONTINUE = "continue"
    BLOCKED = "blocked"
    BUDGET_LIMITED = "budget_limited"


@dataclass(frozen=True, slots=True)
class ExecuteEvidence:
    """Evidence collected after one execute turn."""

    user_prompt: str
    final_answer: str
    tool_summaries: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    reached_max_steps: bool = False


@dataclass(frozen=True, slots=True)
class ExecuteEvaluation:
    """Result of evaluating execute progress against task/plan.md."""

    decision: ExecuteDecision
    reason: str = ""
    next_instruction: str = ""
    contract_status: tuple[str, ...] = ()

    def should_continue(self) -> bool:
        """Return whether execute should start another automatic turn."""
        return self.decision == ExecuteDecision.CONTINUE


@dataclass(frozen=True, slots=True)
class ExecuteLoopUpdate:
    """Post-turn execute-loop update."""

    evaluation: ExecuteEvaluation
    turns: int
    continuation: str = ""

    def should_continue(self) -> bool:
        """Return whether another automatic execute turn should be queued."""
        return bool(self.continuation)


EXECUTE_EVALUATOR_PROMPT = """You evaluate Deepmate task/execute progress.

Deepmate Task Mode has a planning phase and an execution phase. During execute,
the agent must follow task/plan.md. The plan is the task contract: goal,
acceptance contract, scope, execution plan, and verification strategy.

Return one JSON object only, without markdown fences:
{
  "decision": "achieved" | "continue" | "blocked" | "budget_limited",
  "reason": "short reason based only on supplied evidence",
  "next_instruction": "one concrete next instruction for the executor",
  "contract_status": ["short status bullets"]
}

Rules:
- Do not run tools or invent evidence.
- achieved requires credible evidence that every acceptance-contract item is done.
- continue when work can proceed without user input.
- blocked when missing information, permission, external state, or user decision prevents progress.
- budget_limited only when the supplied runtime budget is exhausted.
- Subagents are only an optional execution tool. If subagent results appear, judge them as evidence for the same task contract; do not treat them as separate tasks.
"""


def evidence_from_result(
    *,
    user_prompt: str,
    result: UserTurnResult,
    final_answer: str,
) -> ExecuteEvidence:
    """Build execute evidence from one runtime result."""
    tool_summaries: list[str] = []
    for step in result.steps:
        for tool_result in step.tool_results:
            tool_summaries.append(_tool_summary(tool_result))
    return ExecuteEvidence(
        user_prompt=user_prompt,
        final_answer=final_answer,
        tool_summaries=tuple(tool_summaries[:20]),
        errors=tuple(error.message for error in result.errors()[:10]),
        reached_max_steps=result.reached_max_steps,
    )


def evaluate_execute_progress(
    provider: ModelProvider | None,
    *,
    model: str,
    documents: TaskDocuments,
    evidence: ExecuteEvidence,
    state: TaskModeState | None = None,
    turn_budget: int = DEFAULT_EXECUTE_TURN_BUDGET,
) -> ExecuteEvaluation:
    """Evaluate one execute turn against the task contract."""
    turns = (state.execute_turns if state is not None else 0) + 1
    if evidence.reached_max_steps:
        return ExecuteEvaluation(
            decision=ExecuteDecision.BLOCKED,
            reason="The runtime reached max_steps before a normal final answer.",
            next_instruction="Resume execute after addressing the max_steps stop.",
        )
    if turns >= max(1, turn_budget):
        return ExecuteEvaluation(
            decision=ExecuteDecision.BUDGET_LIMITED,
            reason=f"Execute turn budget reached: {turns}/{turn_budget}.",
            next_instruction="Review remaining acceptance items and resume task/execute if needed.",
        )
    if provider is None:
        return heuristic_execute_evaluation(documents, evidence)
    request = ModelRequest(
        model=model,
        conversation=(
            ModelConversationItem.from_message(
                Message(role=MessageRole.SYSTEM, content=EXECUTE_EVALUATOR_PROMPT)
            ),
            ModelConversationItem.from_message(
                Message(
                    role=MessageRole.USER,
                    content=_evaluation_user_prompt(
                        documents=documents,
                        evidence=evidence,
                        turns=turns,
                        turn_budget=turn_budget,
                    ),
                )
            ),
        ),
        options={"temperature": 0, "max_tokens": 700},
    )
    response = provider.complete(request)
    return parse_execute_evaluation(response.content)


def heuristic_execute_evaluation(
    documents: TaskDocuments,
    evidence: ExecuteEvidence,
) -> ExecuteEvaluation:
    """Conservative deterministic fallback when no evaluator is available."""
    text = "\n".join((documents.plan, evidence.final_answer)).lower()
    blocked_markers = (
        "blocked",
        "阻塞",
        "需要用户",
        "need user",
        "cannot proceed",
        "无法继续",
    )
    if any(marker in text for marker in blocked_markers):
        return ExecuteEvaluation(
            decision=ExecuteDecision.BLOCKED,
            reason="The latest evidence reports a blocker or required user decision.",
            next_instruction="Ask the user for the missing decision or input.",
        )
    if _all_acceptance_items_checked(documents.plan) and _has_verification_evidence(
        text
    ):
        return ExecuteEvaluation(
            decision=ExecuteDecision.ACHIEVED,
            reason="All visible acceptance items are checked and verification evidence is present.",
            next_instruction="Create the final achievement and close execute.",
        )
    return ExecuteEvaluation(
        decision=ExecuteDecision.CONTINUE,
        reason="The acceptance contract is not fully evidenced yet.",
        next_instruction="Continue the next unchecked acceptance or verification item from task/plan.md.",
    )


def parse_execute_evaluation(content: str) -> ExecuteEvaluation:
    """Parse evaluator JSON output."""
    payload = _parse_json_object(content)
    decision = ExecuteDecision.CONTINUE
    raw_decision = str(payload.get("decision", "")).strip().lower()
    for candidate in ExecuteDecision:
        if raw_decision == candidate.value:
            decision = candidate
            break
    raw_status = payload.get("contract_status")
    contract_status = (
        tuple(str(item).strip() for item in raw_status if str(item).strip())
        if isinstance(raw_status, list)
        else ()
    )
    return ExecuteEvaluation(
        decision=decision,
        reason=str(payload.get("reason", "")).strip(),
        next_instruction=str(payload.get("next_instruction", "")).strip(),
        contract_status=contract_status,
    )


def continuation_prompt(evaluation: ExecuteEvaluation) -> str:
    """Return the next execute prompt after a continue decision."""
    reason = evaluation.reason.strip() or "The acceptance contract is not done yet."
    instruction = (
        evaluation.next_instruction.strip()
        or "Continue the next unchecked acceptance item in task/plan.md."
    )
    return (
        "Continue task/execute from task/plan.md.\n\n"
        f"Evaluator reason: {reason}\n"
        f"Next instruction: {instruction}\n\n"
        "Stay within the task contract. Use tools, tests, and subagents only when "
        "they help satisfy the existing acceptance contract. If subagents are useful, "
        "treat them as execution helpers and fold their results back into the same "
        "verification evidence."
    )


def loop_update_from_evaluation(
    evaluation: ExecuteEvaluation,
    *,
    turns: int,
) -> ExecuteLoopUpdate:
    """Build a loop update and continuation prompt from an evaluation."""
    return ExecuteLoopUpdate(
        evaluation=evaluation,
        turns=turns,
        continuation=(
            continuation_prompt(evaluation)
            if evaluation.decision == ExecuteDecision.CONTINUE
            else ""
        ),
    )


def execute_start_prompt(user_prompt: str) -> str:
    """Return the first execute prompt given optional user text."""
    extra = user_prompt.strip()
    if extra.startswith(("Start task/execute", "Continue task/execute")):
        return extra
    base = (
        "Start task/execute from task/plan.md. Follow the goal, acceptance "
        "contract, scope, execution plan, and verification strategy. Work in "
        "small verified steps, update evidence, and stop only when the contract "
        "is achieved or a real blocker appears. Use subagents as optional tools "
        "for complex inspection, implementation, or verification; their outputs "
        "must be folded back into the same task evidence."
    )
    return base if not extra else f"{base}\n\nUser execution note:\n{extra}"


def format_execute_outcome(evaluation: ExecuteEvaluation) -> str:
    """Return a compact user-facing execute outcome."""
    title = {
        ExecuteDecision.ACHIEVED: "task/execute complete",
        ExecuteDecision.CONTINUE: "task/execute continuing",
        ExecuteDecision.BLOCKED: "task/execute blocked",
        ExecuteDecision.BUDGET_LIMITED: "task/execute budget limited",
    }[evaluation.decision]
    lines = [title]
    if evaluation.reason:
        lines.extend(("", f"reason: {evaluation.reason}"))
    if evaluation.next_instruction:
        lines.append(f"next: {evaluation.next_instruction}")
    if evaluation.contract_status:
        lines.extend(("", "contract status:")
        )
        lines.extend(f"- {item}" for item in evaluation.contract_status[:8])
    return "\n".join(lines)


def _evaluation_user_prompt(
    *,
    documents: TaskDocuments,
    evidence: ExecuteEvidence,
    turns: int,
    turn_budget: int,
) -> str:
    return "\n\n".join(
        (
            f"Execute turns: {turns}/{turn_budget}",
            "<task_plan>\n" + _bounded(documents.plan, 14_000) + "\n</task_plan>",
            "<recent_timeline>\n"
            + _bounded(extract_recent_timeline(documents.evolution, limit=5), 3000)
            + "\n</recent_timeline>",
            "<user_prompt>\n" + evidence.user_prompt.strip() + "\n</user_prompt>",
            "<assistant_final_answer>\n"
            + _bounded(evidence.final_answer, 6000)
            + "\n</assistant_final_answer>",
            "<tool_evidence>\n"
            + "\n".join(f"- {item}" for item in evidence.tool_summaries)
            + "\n</tool_evidence>",
            "<errors>\n"
            + "\n".join(f"- {item}" for item in evidence.errors)
            + "\n</errors>",
        )
    )


def _tool_summary(result: ModelToolResult) -> str:
    marker = "error" if result.is_error else "ok"
    refs = ", ".join(result.refs[:4])
    content = " ".join(result.content.split())[:240]
    parts = [f"{result.name}: {marker}"]
    if content:
        parts.append(content)
    if refs:
        parts.append(f"refs={refs}")
    return " | ".join(parts)


def _all_acceptance_items_checked(plan: str) -> bool:
    in_section = False
    found = False
    for line in plan.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            in_section = stripped in {"## 验收契约", "## 验收标准", "## Acceptance Contract"}
            continue
        if not in_section:
            continue
        if stripped.startswith("- ["):
            found = True
            if not stripped.lower().startswith("- [x]"):
                return False
    return found


def _has_verification_evidence(text: str) -> bool:
    markers = (
        "tests passed",
        "test passed",
        "验证通过",
        "测试通过",
        "passed",
        "已验证",
    )
    return any(marker in text for marker in markers)


def _parse_json_object(content: str) -> dict[str, object]:
    stripped = strip_fenced_json(content).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"execute evaluator response must be JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("execute evaluator response must be a JSON object")
    return payload



def _bounded(text: str, max_chars: int) -> str:
    clean = text.strip()
    if len(clean) <= max_chars:
        return clean
    head = clean[: int(max_chars * 0.7)].rstrip()
    tail = clean[-int(max_chars * 0.25) :].lstrip()
    omitted = max(0, len(clean) - len(head) - len(tail))
    return f"{head}\n\n...[truncated: {omitted} chars omitted]...\n\n{tail}"
