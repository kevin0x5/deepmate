"""Per-turn visible capability surface."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from deepmate.domain import CapabilityKind, CapabilityRef
from deepmate.foundation import normalize_name
from deepmate.mcp import McpToolCatalog, McpToolRef
from deepmate.skills import SkillCard
from deepmate.capabilities.state import CapabilityState, CapabilityTemperature

CAPABILITY_RENDER_KIND_ORDER = (
    CapabilityKind.SKILL,
    CapabilityKind.NATIVE_TOOL,
    CapabilityKind.MCP_SERVER,
    CapabilityKind.MCP_TOOL,
)


@dataclass(frozen=True, slots=True)
class CapabilitySurface:
    """Derived view of capabilities visible to the model for one turn."""

    refs: tuple[CapabilityRef, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _validate_refs(self.refs)

    def is_empty(self) -> bool:
        """Return whether no capability is visible in this surface."""
        return not self.refs

    def list_refs(self) -> tuple[CapabilityRef, ...]:
        """Return visible capability references in render order."""
        return self.refs

    def surface_keys(self) -> tuple[str, ...]:
        """Return stable keys for visible capabilities (sorted for deterministic caching)."""
        return tuple(sorted(ref.surface_key() for ref in self.refs))

    def render_order_keys(self) -> tuple[str, ...]:
        """Return stable keys in the same order used by system-context rendering."""
        ordered: list[str] = []
        for kind in CAPABILITY_RENDER_KIND_ORDER:
            ordered.extend(
                ref.surface_key()
                for ref in self.refs
                if ref.kind == kind
            )
        return tuple(ordered)


def from_skill_cards(
    cards: Iterable[SkillCard],
    states_by_name: Mapping[str, CapabilityState] | None = None,
) -> CapabilitySurface:
    """Build a surface from discovered skill cards."""
    states = states_by_name or {}
    refs = tuple(
        ref
        for card in cards
        for ref in (_skill_ref(card, states.get(_normalize_name(card.name))),)
        if ref is not None
    )
    return CapabilitySurface(refs)


def from_native_tool_schemas(schemas: Iterable[Mapping[str, object]]) -> CapabilitySurface:
    """Build a surface from native tool schema references."""
    return CapabilitySurface(tuple(_native_tool_ref(schema) for schema in schemas))


def from_mcp_tool_refs(tools: Iterable[McpToolRef]) -> CapabilitySurface:
    """Build a surface from discovered MCP tool references."""
    return CapabilitySurface(tuple(_mcp_tool_ref(tool) for tool in tools))


def from_mcp_tool_catalog(
    catalog: McpToolCatalog,
    model_context_tokens: int,
) -> CapabilitySurface:
    """Build a compact MCP surface from a progressive-disclosure catalog."""
    return CapabilitySurface(catalog.capability_refs(model_context_tokens))


def combine_surfaces(surfaces: Iterable[CapabilitySurface]) -> CapabilitySurface:
    """Combine surfaces while preserving their input order."""
    refs: list[CapabilityRef] = []
    for surface in surfaces:
        refs.extend(surface.list_refs())
    return CapabilitySurface(tuple(refs))


def _skill_ref(
    card: SkillCard,
    state: CapabilityState | None = None,
) -> CapabilityRef | None:
    if not card.is_ready():
        raise ValueError("SkillCard must be ready before entering CapabilitySurface")
    if not card.is_model_invocable():
        return None
    if card.is_builtin():
        state = None
    if state is not None and not state.is_exposed_by_default():
        return None
    description = card.description.strip()
    if state is not None and state.temperature == CapabilityTemperature.WARM:
        description = ""
    return CapabilityRef(
        kind=CapabilityKind.SKILL,
        name=card.name.strip(),
        description=description,
    )


def _native_tool_ref(schema: Mapping[str, object]) -> CapabilityRef:
    return CapabilityRef(
        kind=CapabilityKind.NATIVE_TOOL,
        name=_required_text(schema, "name"),
        description=_required_text(schema, "description"),
    )


def _mcp_tool_ref(tool: McpToolRef) -> CapabilityRef:
    if not tool.is_ready():
        raise ValueError("McpToolRef must be ready before entering CapabilitySurface")
    return CapabilityRef(
        kind=CapabilityKind.MCP_TOOL,
        name=tool.qualified_name(),
        description=tool.display_description(),
    )


def _required_text(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key, "")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Capability schema requires text field: {key}")
    return value.strip()


def _validate_refs(refs: Iterable[CapabilityRef]) -> None:
    seen_keys: set[str] = set()
    for ref in refs:
        if not ref.is_ready():
            raise ValueError("CapabilitySurface requires ready CapabilityRef values")
        key = f"{ref.kind.value}:{ref.name.strip()}"
        if key in seen_keys:
            raise ValueError(f"Duplicate capability surface key: {key}")
        seen_keys.add(key)


def _normalize_name(name: str) -> str:
    return normalize_name(name)
