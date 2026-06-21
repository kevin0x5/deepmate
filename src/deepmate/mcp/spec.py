"""MCP server and discovered tool specifications."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class McpTransport(StrEnum):
    """Supported MCP server transport kinds."""

    STDIO = "stdio"
    HTTP = "http"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable_http"


class McpExposure(StrEnum):
    """Default visibility level for an MCP server."""

    ALWAYS_ON = "always_on"
    SEARCHABLE = "searchable"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class McpServerSpec:
    """Minimal MCP server configuration without a live client session."""

    name: str
    transport: McpTransport
    command: str = ""
    args: tuple[str, ...] = field(default_factory=tuple)
    url: str = ""
    cwd: str = ""
    env: Mapping[str, str] = field(default_factory=dict)
    bearer_token_env_var: str = ""
    description: str = ""
    exposure: McpExposure = McpExposure.SEARCHABLE
    startup_timeout_seconds: float | None = None

    def is_ready(self) -> bool:
        """Return whether the server has the minimum connection target."""
        if not self.name.strip():
            return False
        if self.transport == McpTransport.STDIO:
            return bool(self.command.strip())
        if self.transport in {
            McpTransport.HTTP,
            McpTransport.SSE,
            McpTransport.STREAMABLE_HTTP,
        }:
            return bool(self.url.strip())
        return False

    def has_description(self) -> bool:
        """Return whether the server has discovery text for future surfaces."""
        return bool(self.description.strip())


@dataclass(frozen=True, slots=True)
class McpToolRef:
    """Reference to a tool discovered from one MCP server."""

    server_name: str
    name: str
    title: str = ""
    description: str = ""
    input_schema: Mapping[str, object] = field(default_factory=dict)
    output_schema: Mapping[str, object] = field(default_factory=dict)
    annotations: Mapping[str, object] = field(default_factory=dict)
    meta: Mapping[str, object] = field(default_factory=dict)

    def is_ready(self) -> bool:
        """Return whether the tool can be addressed unambiguously."""
        return bool(self.server_name.strip() and self.name.strip())

    def qualified_name(self) -> str:
        """Return the stable MCP tool name used outside the MCP client."""
        return f"{self.server_name.strip()}.{self.name.strip()}"

    def has_description(self) -> bool:
        """Return whether the tool has model-facing discovery text."""
        return bool(self.description.strip())

    def display_description(self) -> str:
        """Return model-facing discovery text, with a safe fallback."""
        if self.description.strip():
            return self.description.strip()
        if self.title.strip():
            return self.title.strip()
        return f"MCP tool {self.qualified_name()}."

    def is_read_only(self) -> bool:
        """Return whether the MCP server marks this tool as read-only."""
        return (
            self.annotations.get("readOnlyHint") is True
            and not _looks_like_mutating_tool(self.name)
            and not _looks_like_mutating_tool(self.title)
            and not _looks_like_mutating_tool(self.description)
        )

    def schema(self) -> Mapping[str, object]:
        """Return a provider-neutral tool schema for model requests."""
        return {
            "name": self.qualified_name(),
            "description": self.display_description(),
            "input_schema": self.input_schema,
        }


def _looks_like_mutating_tool(value: str) -> bool:
    text = f" {value.strip().lower().replace('_', ' ').replace('-', ' ')} "
    if not text.strip():
        return False
    markers = (
        " write ",
        " save ",
        " edit ",
        " update ",
        " upsert ",
        " set ",
        " configure ",
        " config ",
        " create ",
        " add ",
        " delete ",
        " remove ",
        " clear ",
        " reset ",
        " rename ",
        " move ",
        " copy ",
        " upload ",
        " download ",
        " send ",
        " publish ",
        " post ",
        " put ",
        " patch ",
        " execute ",
        " run ",
        " start ",
        " stop ",
        " restart ",
        " deploy ",
        " release ",
        " shell ",
        " command ",
        " mutate ",
        " insert ",
        " append ",
        " replace ",
        " merge ",
        " commit ",
        " push ",
        " install ",
        " uninstall ",
    )
    return any(marker in text for marker in markers)
