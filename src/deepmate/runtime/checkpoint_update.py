"""Structured checkpoint updates for summary, memory, and activity."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from deepmate.domain import Message, MessageRole
from deepmate.memory.manager import MemoryPatch, MemoryPatchOperation
from deepmate.providers import (
    ModelConversationItem,
    ModelProvider,
    ModelRequest,
    TokenUsage,
)
from deepmate.runtime.session_summary import (
    SessionSummary,
    SessionSummaryInput,
    _render_source_items,
    generate_session_summary,
    validate_session_summary_response,
)

CHECKPOINT_UPDATE_SYSTEM_PROMPT = """You write one Deepmate checkpoint update.

Return only one JSON object. Do not use markdown fences.

Output schema:
{
  "session_summary": {
    "content": "Markdown summary for continuing the same session"
  },
  "activity_digest": {
    "summary": "Short human-readable daily note summary",
    "highlights": ["..."],
    "next_steps": ["..."]
  }
}

Session summary rules:
- Preserve the user's current goal, explicit constraints, scoped product/project
  context, decisions, rejected alternatives, files, tools, evidence refs,
  blockers, and continuation notes.
- Distinguish verified facts/checks from unverified assumptions.
- Preserve concrete next actions needed to continue work.
- Keep product/project/task scope explicit. Do not rewrite scoped decisions as
  global profile memory.
- Do not preserve secrets, credentials, tokens, private keys, payment data,
  full addresses, or unverified guesses as facts.

Activity digest rules:
- Summarize what happened for a daily activity note.
- Include scoped project/task decisions and next steps here when they are useful
  but not global profile memory.
- Keep it concise; the full transcript and trace remain the source of truth.
"""

MEMORY_PATCH_SYSTEM_PROMPT = """You write one Deepmate hot memory patch.

Return only one JSON object. Do not use markdown fences.

Output schema:
{
  "operations": [
    {
      "action": "write_user|write_memory|write_project_memory|replace|remove|demote_to_warm|skip",
      "target": "user|memory|project_memory",
      "content": "...",
      "replace_ref": "...",
      "reason": "...",
      "confidence": 0.0
    }
  ]
}

Rules:
- Hot profile memory is global user.md + global memory.md + project memory.md.
  It is injected into future activations, so keep it small and stable.
- Use write_user only for durable user profile facts or long-term interaction
  preferences.
- Use write_memory only for cross-session, cross-task working principles.
- Use write_project_memory for current-project facts, constraints, conventions,
  decisions, or recurring instructions that should survive future sessions.
- Use replace/remove only when an existing bullet is clearly stale or wrong.
- Use demote_to_warm for task-, session-, date-, file-, test-, bug-, or
  implementation-specific context that is too narrow even for project memory.
- Use skip for unsafe, sensitive, speculative, or low-value content.
- Memory patch decisions must come only from user-authored source or explicit
  corrections. The source below intentionally excludes assistant and tool output.
- Keep memory content as one short bullet fact without a leading dash.
"""


@dataclass(frozen=True, slots=True)
class ActivityDigest:
    """Human-readable digest for activity daily/monthly notes."""

    summary: str = ""
    highlights: tuple[str, ...] = field(default_factory=tuple)
    next_steps: tuple[str, ...] = field(default_factory=tuple)

    def render(self, fallback: str = "") -> str:
        """Return compact Markdown-ish text suitable for an activity entry."""
        lines: list[str] = []
        if self.summary.strip():
            lines.append(self.summary.strip())
        for label, values in (
            ("Highlights", self.highlights),
            ("Next steps", self.next_steps),
        ):
            clean_values = tuple(value.strip() for value in values if value.strip())
            if not clean_values:
                continue
            lines.append(f"{label}:")
            lines.extend(f"- {value}" for value in clean_values)
        rendered = "\n".join(lines).strip()
        return rendered or fallback.strip()


@dataclass(frozen=True, slots=True)
class CheckpointUpdate:
    """One structured checkpoint result."""

    session_summary: SessionSummary
    memory_patch: MemoryPatch = field(default_factory=MemoryPatch)
    activity_digest: ActivityDigest = field(default_factory=ActivityDigest)

    def is_ready(self) -> bool:
        """Return whether this update can be persisted."""
        return self.session_summary.is_ready()

    def memory_operation_count(self) -> int:
        """Return how many memory patch operations were proposed."""
        return len(self.memory_patch.operations)


def generate_checkpoint_update(
    provider: ModelProvider,
    model: str,
    summary_input: SessionSummaryInput,
    profile_dir: str | Path,
    project_profile_dir: str | Path | None = None,
    options: dict[str, object] | None = None,
) -> CheckpointUpdate:
    """Generate a structured checkpoint update with summary fallback."""
    try:
        request = build_checkpoint_update_request(
            model=model,
            summary_input=summary_input,
            options=options,
        )
        response = provider.complete(request)
        update = parse_checkpoint_update_response(
            content=response.content,
            finish_reason=response.finish_reason,
            summary_input=summary_input,
            model=model,
            usage=response.usage,
        )
        if not update.is_ready():
            raise ValueError("checkpoint update is not ready")
    except Exception:
        summary = generate_session_summary(
            provider=provider,
            model=model,
            summary_input=summary_input,
            options=options,
        )
        update = CheckpointUpdate(
            session_summary=summary,
            activity_digest=ActivityDigest(summary=summary.content),
        )
    memory_patch = generate_memory_patch(
        provider=provider,
        model=model,
        summary_input=summary_input,
        profile_dir=profile_dir,
        project_profile_dir=project_profile_dir,
        options=options,
    )
    return CheckpointUpdate(
        session_summary=update.session_summary,
        memory_patch=memory_patch,
        activity_digest=update.activity_digest,
    )


def build_checkpoint_update_request(
    model: str,
    summary_input: SessionSummaryInput,
    options: dict[str, object] | None = None,
) -> ModelRequest:
    """Build the provider request used for structured checkpoint updates."""
    clean_model = _text(model)
    if not clean_model:
        raise ValueError("checkpoint update model is required")
    if not summary_input.is_ready():
        raise ValueError("checkpoint update input requires source items")
    request = ModelRequest(
        model=clean_model,
        conversation=(
            ModelConversationItem.from_message(
                Message(
                    role=MessageRole.SYSTEM,
                    content=CHECKPOINT_UPDATE_SYSTEM_PROMPT,
                )
            ),
            ModelConversationItem.from_message(
                Message(
                    role=MessageRole.USER,
                    content=_checkpoint_update_user_prompt(
                        summary_input,
                    ),
                )
            ),
        ),
        options={
            "temperature": 0,
            "max_tokens": 6_000,
            **dict(options or {}),
        },
    )
    if not request.is_ready():
        raise ValueError("checkpoint update request is not ready")
    return request


def generate_memory_patch(
    provider: ModelProvider,
    model: str,
    summary_input: SessionSummaryInput,
    profile_dir: str | Path,
    project_profile_dir: str | Path | None = None,
    options: dict[str, object] | None = None,
) -> MemoryPatch:
    """Generate a hot memory patch from user-authored source only."""
    request = build_memory_patch_request(
        model=model,
        summary_input=summary_input,
        profile_dir=profile_dir,
        project_profile_dir=project_profile_dir,
        options=options,
    )
    if request is None:
        return MemoryPatch()
    try:
        response = provider.complete(request)
    except Exception:
        return MemoryPatch()
    try:
        return parse_memory_patch_response(response.content)
    except Exception:
        return MemoryPatch()


def build_memory_patch_request(
    model: str,
    summary_input: SessionSummaryInput,
    profile_dir: str | Path,
    project_profile_dir: str | Path | None = None,
    options: dict[str, object] | None = None,
) -> ModelRequest | None:
    """Build the provider request used for user-sourced memory patching."""
    clean_model = _text(model)
    if not clean_model:
        raise ValueError("memory patch model is required")
    if not summary_input.is_ready():
        raise ValueError("memory patch input requires source items")
    user_source = _render_user_source_items(summary_input)
    if not user_source:
        return None
    profile_path = Path(profile_dir)
    project_profile_path = (
        Path(project_profile_dir) if project_profile_dir is not None else profile_path
    )
    request = ModelRequest(
        model=clean_model,
        conversation=(
            ModelConversationItem.from_message(
                Message(role=MessageRole.SYSTEM, content=MEMORY_PATCH_SYSTEM_PROMPT)
            ),
            ModelConversationItem.from_message(
                Message(
                    role=MessageRole.USER,
                    content=_memory_patch_user_prompt(
                        user_source=user_source,
                        profile_path=profile_path,
                        project_profile_path=project_profile_path,
                    ),
                )
            ),
        ),
        options={
            "temperature": 0,
            "max_tokens": 1_600,
            **dict(options or {}),
        },
    )
    if not request.is_ready():
        raise ValueError("memory patch request is not ready")
    return request


def parse_memory_patch_response(content: str) -> MemoryPatch:
    """Parse a memory-patch-only model response."""
    return _parse_memory_patch(_parse_json_object(content))


def parse_checkpoint_update_response(
    content: str,
    finish_reason: str,
    summary_input: SessionSummaryInput,
    model: str,
    usage: TokenUsage | None = None,
) -> CheckpointUpdate:
    """Parse and validate a model checkpoint update response."""
    payload = _parse_json_object(content)
    summary_payload = payload.get("session_summary")
    if not isinstance(summary_payload, Mapping):
        raise ValueError("checkpoint update missing session_summary")
    summary_content = _text(summary_payload.get("content"))
    validate_session_summary_response(summary_content, finish_reason, summary_input)
    summary = SessionSummary(
        content=summary_content,
        covered_until_sequence=summary_input.covered_until_sequence(),
        covered_item_count=summary_input.covered_item_count(),
        source_item_count=summary_input.source_item_count(),
        estimated_source_tokens=summary_input.estimated_source_tokens(),
        source_model=_text(model),
        usage=usage,
    )
    if not summary.is_ready():
        raise ValueError("checkpoint update summary is not ready")
    return CheckpointUpdate(
        session_summary=summary,
        memory_patch=MemoryPatch(),
        activity_digest=_parse_activity_digest(payload.get("activity_digest")),
    )


def _checkpoint_update_user_prompt(
    summary_input: SessionSummaryInput,
) -> str:
    lines = [
        "Create the next checkpoint update from the source segment below.",
        f"Target session summary length: about {max(1, summary_input.target_tokens)} tokens.",
        "",
        "Required Markdown headings inside session_summary.content:",
        "## Session Summary",
        "### User Goal",
        "### Product Or Project Context",
        "### Current State",
        "### Decisions And Constraints",
        "### Files, Tools, And Artifacts",
        "### Evidence And References",
        "### Verified And Unverified State",
        "### Open Questions Or Blockers",
        "### Next Actions",
        "### Recent Continuation Notes",
    ]
    if summary_input.previous_summary.strip():
        lines.extend(
            (
                "",
                "Previous summary to preserve and update:",
                summary_input.previous_summary.strip(),
            )
        )
    lines.extend(
        (
            "",
            "New source segment:",
            _render_source_items(summary_input.source_items),
        )
    )
    return "\n".join(lines).strip()


def _memory_patch_user_prompt(
    *,
    user_source: str,
    profile_path: Path,
    project_profile_path: Path,
) -> str:
    return "\n".join(
        (
            "Create a memory patch from user-authored source only.",
            "",
            "Current global user.md bullets:",
            _bullet_block(profile_path / "user.md"),
            "",
            "Current global memory.md bullets:",
            _bullet_block(profile_path / "memory.md"),
            "",
            "Current project memory.md bullets:",
            _bullet_block(project_profile_path / "memory.md"),
            "",
            "User-authored source:",
            user_source,
        )
    ).strip()


def _render_user_source_items(summary_input: SessionSummaryInput) -> str:
    sections: list[str] = []
    for source in summary_input.source_items:
        item = source.item
        if item.message is None or item.message.role != MessageRole.USER:
            continue
        content = item.message.content.strip()
        if not content:
            continue
        sections.append(f"### User transcript item {source.sequence}\n{content}")
    return "\n\n".join(sections).strip()


def _parse_memory_patch(value: object) -> MemoryPatch:
    if not isinstance(value, Mapping):
        return MemoryPatch()
    raw_operations = value.get("operations")
    if not isinstance(raw_operations, list):
        return MemoryPatch()
    operations: list[MemoryPatchOperation] = []
    for item in raw_operations:
        if not isinstance(item, Mapping):
            continue
        operations.append(
            MemoryPatchOperation(
                action=str(item.get("action", "")).strip().lower(),
                target=str(item.get("target", "")).strip().lower(),
                content=str(item.get("content", "")).strip(),
                replace_ref=str(item.get("replace_ref", "")).strip(),
                reason=str(item.get("reason", "")).strip(),
                confidence=_float_optional(item.get("confidence")),
            )
        )
    return MemoryPatch(operations=tuple(operations))


def _parse_activity_digest(value: object) -> ActivityDigest:
    if not isinstance(value, Mapping):
        return ActivityDigest()
    return ActivityDigest(
        summary=str(value.get("summary", "")).strip(),
        highlights=_string_tuple(value.get("highlights")),
        next_steps=_string_tuple(value.get("next_steps")),
    )


def _parse_json_object(text: str) -> Mapping[str, object]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = _strip_fenced_json(cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"checkpoint update response must be JSON: {exc.msg}") from exc
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, Mapping):
        raise ValueError("checkpoint update response must be a JSON object")
    return parsed


def _strip_fenced_json(text: str) -> str:
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```"):
        if lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return text


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _bullet_block(path: Path) -> str:
    if not path.exists():
        return "(empty)"
    bullets = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            bullets.append(stripped)
    return "\n".join(bullets) if bullets else "(empty)"


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _float_optional(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None
