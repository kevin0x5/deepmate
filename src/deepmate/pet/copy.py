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
            dog=f"已保存阶段成果：{summary}",
            cat=f"阶段成果收好啦：{summary}",
            squirrel=f"这个阶段记下来了：{summary}",
            penguin=f"阶段成果已归档：{summary}",
        )
    elif event.kind == "task.completed":
        text = _by_style(
            style,
            dog=f"完成啦：{summary}",
            cat=f"处理好了：{summary}",
            squirrel=f"搞定：{summary}",
            penguin=f"已经完成：{summary}",
        )
    elif event.kind in {"task.failed", "task.waiting"}:
        text = _by_style(
            style,
            dog=f"这里需要你看一下：{summary}",
            cat=f"有个地方需要确认：{summary}",
            squirrel=f"这一步先卡住了：{summary}",
            penguin=f"遇到一个问题：{summary}",
        )
    elif event.kind.startswith("learning."):
        text = _by_style(
            style,
            dog=f"这条可能有用：{summary}",
            cat=f"看到一条相关内容：{summary}",
            squirrel=f"顺手标了一条资料：{summary}",
            penguin=f"这条可以参考：{summary}",
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
            dog=f"正在处理：{summary}",
            cat=f"还在看：{summary}",
            squirrel=f"进度更新：{summary}",
            penguin=f"还在处理中：{summary}",
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
            "Write naturally and warmly. Do not introduce the pet, its implementation, "
            "frameworks, session ids, token counts, hidden context, or screen reading. "
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
