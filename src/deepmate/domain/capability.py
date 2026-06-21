"""Capability domain objects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CapabilityKind(StrEnum):
    """Supported sources for exposed agent capabilities."""

    SKILL = "skill"
    NATIVE_TOOL = "native_tool"
    MCP_SERVER = "mcp_server"
    MCP_TOOL = "mcp_tool"


@dataclass(frozen=True, slots=True)
class CapabilityRef:
    """Reference to a surfaced capability, not its implementation."""

    kind: CapabilityKind
    name: str
    description: str

    def is_ready(self) -> bool:
        """Return whether the capability is clear enough to expose."""
        if self.kind in {CapabilityKind.SKILL, CapabilityKind.MCP_TOOL}:
            return bool(self.name.strip())
        return bool(self.name.strip() and self.description.strip())

    def surface_key(self) -> str:
        """Return the stable key used when rendering a capability surface."""
        description_tag = (
            "name-only" if not self.description.strip() else self.description.strip()
        )
        return f"{self.kind.value}:{self.name.strip()}:{description_tag}"
