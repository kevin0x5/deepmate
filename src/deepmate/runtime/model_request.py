"""Build provider-neutral model requests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from deepmate.context import (
    ContextWarning,
    ProfileContextSnapshot,
    build_system_context,
    build_system_context_from_snapshot,
)
from deepmate.domain import Message, MessageRole, ProfileRef
from deepmate.providers import (
    ModelCapabilities,
    ModelConversationItem,
    ModelRequest,
    ModelToolExchange,
)

if TYPE_CHECKING:
    from deepmate.capabilities import CapabilitySurface
    from deepmate.skills import SkillDocument


@dataclass(frozen=True, slots=True)
class ModelRequestBuildResult:
    """Model request plus non-fatal context diagnostics."""

    request: ModelRequest
    warnings: tuple[ContextWarning, ...] = field(default_factory=tuple)


def build_model_request(
    workspace: str | Path,
    profile: ProfileRef,
    messages: Iterable[Message],
    model: str,
    capability_surface: CapabilitySurface | None = None,
    tool_schemas: Iterable[Mapping[str, object]] = (),
    tool_exchanges: Iterable[ModelToolExchange] = (),
    conversation: Iterable[ModelConversationItem] = (),
    turn_tail_messages: Iterable[Message] = (),
    selected_skill_documents: Iterable["SkillDocument"] = (),
    context_snapshot: ProfileContextSnapshot | None = None,
    system_message: Message | None = None,
    context_warnings: Iterable[ContextWarning] = (),
    options: Mapping[str, object] | None = None,
    capabilities: ModelCapabilities | None = None,
) -> ModelRequestBuildResult:
    """Build a model request from context, conversation messages, and tools."""
    selected_skills = tuple(selected_skill_documents)
    if system_message is not None:
        if not system_message.is_ready() or system_message.role != MessageRole.SYSTEM:
            raise ValueError("system_message must be a ready system message")
        context_message = system_message
        warnings = tuple(context_warnings)
    elif context_snapshot is None:
        context_result = build_system_context(
            workspace=workspace,
            profile=profile,
            capability_surface=capability_surface,
            selected_skill_documents=selected_skills,
        )
        context_message = context_result.message
        warnings = context_result.warnings
    else:
        context_result = build_system_context_from_snapshot(
            snapshot=context_snapshot,
            capability_surface=capability_surface,
            selected_skill_documents=selected_skills,
        )
        context_message = context_result.message
        warnings = context_result.warnings
    body_messages = _ready_messages(messages)
    tail_messages = _ready_turn_tail_messages(turn_tail_messages)
    ready_tool_exchanges = _ready_tool_exchanges(tool_exchanges)
    conversation_items = _ready_conversation(
        conversation,
        fallback_messages=body_messages,
        fallback_tool_exchanges=ready_tool_exchanges,
    )
    clean_model = model.strip() if isinstance(model, str) else ""
    request = ModelRequest(
        model=clean_model,
        tool_schemas=tuple(tool_schemas),
        conversation=(
            ModelConversationItem.from_message(context_message),
            *conversation_items,
            *(ModelConversationItem.from_message(message) for message in tail_messages),
        ),
        options=dict(options or {}),
        capabilities=(capabilities or ModelCapabilities()).normalized(),
    )
    if not request.is_ready():
        raise ValueError("model request requires a model and system context")
    return ModelRequestBuildResult(
        request=request,
        warnings=warnings,
    )


def _ready_messages(messages: Iterable[Message]) -> tuple[Message, ...]:
    ready_messages = tuple(messages)
    for message in ready_messages:
        if not message.is_ready():
            raise ValueError("model request messages must not be empty")
    return ready_messages


def _ready_turn_tail_messages(messages: Iterable[Message]) -> tuple[Message, ...]:
    ready_messages = _ready_messages(messages)
    for message in ready_messages:
        if message.role == MessageRole.SYSTEM:
            raise ValueError("turn-tail messages must not use the system role")
    return ready_messages


def _ready_tool_exchanges(
    tool_exchanges: Iterable[ModelToolExchange],
) -> tuple[ModelToolExchange, ...]:
    ready_exchanges = tuple(tool_exchanges)
    for exchange in ready_exchanges:
        if not exchange.is_ready():
            raise ValueError("model request tool exchanges must be ready")
    return ready_exchanges


def _ready_conversation(
    conversation: Iterable[ModelConversationItem],
    fallback_messages: tuple[Message, ...],
    fallback_tool_exchanges: tuple[ModelToolExchange, ...],
) -> tuple[ModelConversationItem, ...]:
    ready_items = tuple(conversation)
    if not ready_items:
        return (
            *(ModelConversationItem.from_message(message) for message in fallback_messages),
            *(
                ModelConversationItem.from_tool_exchange(exchange)
                for exchange in fallback_tool_exchanges
            ),
        )
    for item in ready_items:
        if not item.is_ready():
            raise ValueError("model request conversation items must be ready")
    return ready_items
