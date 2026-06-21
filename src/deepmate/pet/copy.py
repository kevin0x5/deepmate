"""Short pet bubble copy generation with deterministic fallback."""

from __future__ import annotations

from dataclasses import dataclass

from deepmate.domain import Message, MessageRole
from deepmate.pet.events import PetEvent
from deepmate.pet.state import PetProfile
from deepmate.providers import ModelConversationItem, ModelProvider, ModelRequest


@dataclass(frozen=True, slots=True)
class PetCopyResult:
    """One generated or fallback pet bubble."""

    text: str
    source: str = "fallback"


def generate_pet_copy(
    event: PetEvent,
    profile: PetProfile,
    provider: ModelProvider | None = None,
    model: str = "",
    max_chars: int = 90,
) -> PetCopyResult:
    """Generate one short pet bubble, falling back to deterministic copy."""
    fallback = fallback_pet_copy(event, profile, max_chars=max_chars)
    if provider is None or profile.bubble_generation == "frugal" or not model.strip():
        return PetCopyResult(fallback, "fallback")
    try:
        response = provider.complete(_copy_request(event, profile, model, max_chars))
    except Exception:
        return PetCopyResult(fallback, "fallback")
    clean = bounded_pet_text(response.content, max_chars)
    return PetCopyResult(clean or fallback, "llm" if clean else "fallback")


def fallback_pet_copy(
    event: PetEvent,
    profile: PetProfile,
    max_chars: int = 90,
) -> str:
    """Return deterministic copy for one event and pet style."""
    summary = event.summary.strip()
    style = profile.style.strip().lower()
    if event.kind == "task.achievement":
        text = _by_style(
            style,
            dog=f"Stage saved. {summary}",
            cat=f"Stage saved. {summary}",
            squirrel=f"Milestone stored: {summary}",
            penguin=f"I saved the stage result. {summary}",
        )
    elif event.kind == "task.completed":
        text = _by_style(
            style,
            dog=f"Done. {summary}",
            cat=f"Done. {summary}",
            squirrel=f"Done. Found the result: {summary}",
            penguin=f"I think it is done. {summary}",
        )
    elif event.kind in {"task.failed", "task.waiting"}:
        text = _by_style(
            style,
            dog=f"This needs a look. {summary}",
            cat=f"Something needs attention. {summary}",
            squirrel=f"This path is blocked. {summary}",
            penguin=f"I ran into a problem. {summary}",
        )
    elif event.kind.startswith("learning."):
        text = _by_style(
            style,
            dog=f"I found something that may help: {summary}",
            cat=f"Found something mildly relevant: {summary}",
            squirrel=f"I spotted a useful link: {summary}",
            penguin=f"This may be related: {summary}",
        )
    elif event.kind.startswith("care."):
        text = _by_style(
            style,
            dog=summary,
            cat=summary,
            squirrel=f"Small pause reminder: {summary}",
            penguin=f"Maybe pause for a moment. {summary}",
        )
    else:
        text = _by_style(
            style,
            dog=f"I am on it. {summary}",
            cat=f"Still working. {summary}",
            squirrel=f"Progress update: {summary}",
            penguin=f"I am still working. {summary}",
        )
    return bounded_pet_text(text, max_chars)


def _copy_request(
    event: PetEvent,
    profile: PetProfile,
    model: str,
    max_chars: int,
) -> ModelRequest:
    system = Message(
        role=MessageRole.SYSTEM,
        content=(
            "You write one short desktop pet bubble for Deepmate. "
            "Use zh-CN if the input is Chinese, otherwise match the input language. "
            "Do not mention session ids, token counts, hidden context, or screen reading. "
            f"Return at most {max_chars} characters."
        ),
    )
    user = Message(
        role=MessageRole.USER,
        content=(
            f"pet={profile.species}; style={profile.style}; "
            f"event={event.kind}; state={event.state.value}; "
            f"title={event.current_work_title}; summary={event.summary}"
        ),
    )
    return ModelRequest(
        model=model.strip(),
        conversation=(
            ModelConversationItem.from_message(system),
            ModelConversationItem.from_message(user),
        ),
        options={"max_tokens": 80},
    )


def _by_style(
    style: str,
    *,
    dog: str,
    cat: str,
    squirrel: str,
    penguin: str,
) -> str:
    if "lazy" in style or "cat" in style:
        return cat
    if "lively" in style or "squirrel" in style:
        return squirrel
    if "naive" in style or "penguin" in style:
        return penguin
    return dog


def bounded_pet_text(value: str, limit: int) -> str:
    """Return one whitespace-normalized pet bubble within a character budget."""
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _bounded(value: str, limit: int) -> str:
    return bounded_pet_text(value, limit)
