"""MCP tool inventory and progressive-disclosure helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from deepmate.domain import CapabilityKind, CapabilityRef
from deepmate.foundation import compact_text
from deepmate.mcp.client import create_mcp_client
from deepmate.mcp.spec import McpExposure, McpServerSpec, McpToolRef
from deepmate.mcp.state import DEFAULT_MCP_IDLE_DAYS, McpUsageStateStore

MCP_SCHEMA_PRELOAD_RATIO = 0.01
MCP_COMPACT_SURFACE_RATIO = 0.002


@dataclass(frozen=True, slots=True)
class McpServerInventory:
    """Discovered MCP server metadata and tools for one run."""

    server: McpServerSpec
    tools: tuple[McpToolRef, ...] = field(default_factory=tuple)
    server_info: Mapping[str, object] = field(default_factory=dict)
    instructions: str = ""
    capabilities: Mapping[str, object] = field(default_factory=dict)

    def read_only_tools(self) -> tuple[McpToolRef, ...]:
        """Return read-only tools discovered from this server."""
        return tuple(tool for tool in self.tools if tool.is_read_only())


@dataclass(frozen=True, slots=True)
class McpToolCatalog:
    """Run-local MCP inventory with sidecar-backed exposure decisions."""

    inventories: tuple[McpServerInventory, ...] = field(default_factory=tuple)
    state_store: McpUsageStateStore | None = None
    idle_days: int = DEFAULT_MCP_IDLE_DAYS
    now: datetime | None = None

    def is_empty(self) -> bool:
        """Return whether no MCP tools were discovered."""
        return not any(inventory.tools for inventory in self.inventories)

    def all_tools(self) -> tuple[McpToolRef, ...]:
        """Return all discovered tools."""
        return tuple(tool for inventory in self.inventories for tool in inventory.tools)

    def read_only_tools(self) -> tuple[McpToolRef, ...]:
        """Return all discovered read-only tools."""
        return tuple(
            tool
            for inventory in self.inventories
            for tool in inventory.read_only_tools()
        )

    def get_tool(self, qualified_name: str) -> McpToolRef | None:
        """Return one read-only tool by qualified name."""
        clean_name = qualified_name.strip()
        for tool in self.read_only_tools():
            if tool.qualified_name() == clean_name:
                return tool
        return None

    def search(
        self,
        query: str = "",
        server_name: str = "",
        limit: int = 10,
    ) -> tuple[McpToolRef, ...]:
        """Return compact tool candidates matching query and optional server."""
        clean_query = query.strip().lower()
        clean_server = server_name.strip()
        matches: list[McpToolRef] = []
        for tool in self.read_only_tools():
            if clean_server and tool.server_name != clean_server:
                continue
            haystack = " ".join(
                (
                    tool.qualified_name(),
                    tool.title,
                    tool.description,
                )
            ).lower()
            if clean_query and clean_query not in haystack:
                continue
            matches.append(tool)
        return tuple(matches[: max(1, limit)])

    def default_schema_tools(self, model_context_tokens: int) -> tuple[McpToolRef, ...]:
        """Return MCP tools whose full schema should be preloaded by default."""
        budget = max(1, int(max(1, model_context_tokens) * MCP_SCHEMA_PRELOAD_RATIO))
        selected: list[McpToolRef] = []
        used = 0
        for tool in self._schema_preload_candidates():
            cost = _estimate_tool_schema_tokens(tool.schema())
            if used + cost > budget and selected:
                continue
            if used + cost > budget:
                continue
            selected.append(tool)
            used += cost
        return tuple(selected)

    def capability_refs(self, model_context_tokens: int) -> tuple[CapabilityRef, ...]:
        """Return compact MCP refs for the model-facing capability surface."""
        refs: list[CapabilityRef] = []
        tool_budget = max(
            1,
            int(max(1, model_context_tokens) * MCP_COMPACT_SURFACE_RATIO),
        )
        tool_budget_used = 0
        for inventory in self.inventories:
            tier = self.server_tier(inventory.server)
            if inventory.server.exposure == McpExposure.MANUAL:
                continue
            refs.append(self._server_ref(inventory, tier))
            for tool in inventory.read_only_tools():
                ref = _tool_name_ref(tool) if tier == "idle" else _tool_ref(tool)
                cost = _compact_ref_cost(ref)
                if tool_budget_used + cost > tool_budget:
                    break
                refs.append(ref)
                tool_budget_used += cost
        return tuple(refs)

    def server_tier(self, server: McpServerSpec) -> str:
        """Return active or idle for default MCP context injection."""
        if server.exposure == McpExposure.ALWAYS_ON:
            return "active"
        if self.state_store is None:
            return "active"
        entry = self.state_store.server_entry(server.name)
        if entry is None:
            return "active"
        if entry.is_idle(_normal_datetime(self.now), idle_days=self.idle_days):
            return "idle"
        return "active"

    def loaded_schema_tool_names(self) -> tuple[str, ...]:
        """Return MCP tools whose schema has been loaded at least once."""
        if self.state_store is None:
            return ()
        entries = self.state_store.load()
        loaded = [
            entry.name
            for entry in entries.values()
            if entry.kind == "tool" and entry.load_count > 0 and entry.name.strip()
        ]
        return tuple(sorted(set(loaded)))

    def _schema_preload_candidates(self) -> tuple[McpToolRef, ...]:
        candidates: list[McpToolRef] = []
        seen: set[str] = set()
        for inventory in self.inventories:
            if inventory.server.exposure == McpExposure.ALWAYS_ON:
                for tool in inventory.read_only_tools():
                    if tool.qualified_name() not in seen:
                        candidates.append(tool)
                        seen.add(tool.qualified_name())
        candidates.extend(self._recent_read_only_tools(seen))
        return tuple(candidates)

    def _recent_read_only_tools(self, seen: set[str]) -> tuple[McpToolRef, ...]:
        if self.state_store is None:
            return ()
        entries = self.state_store.load()
        tools_by_name = {tool.qualified_name(): tool for tool in self.read_only_tools()}
        recent: list[tuple[str, McpToolRef]] = []
        for qualified_name, tool in tools_by_name.items():
            if qualified_name in seen:
                continue
            entry = entries.get(f"mcp_tool:{qualified_name}")
            if entry is None or not entry.last_used_at.strip():
                continue
            recent.append((entry.last_used_at, tool))
        return tuple(tool for _, tool in sorted(recent, reverse=True))

    def _server_ref(self, inventory: McpServerInventory, tier: str) -> CapabilityRef:
        server = inventory.server
        read_only_count = len(inventory.read_only_tools())
        total_count = len(inventory.tools)
        description_parts = [
            _server_description(inventory),
            f"tier={tier}",
            f"tools={total_count}",
            f"read_only_tools={read_only_count}",
        ]
        if tier == "idle":
            description_parts.append(
                "Default context is compact because the server has not been used "
                f"for {max(1, self.idle_days)} days; search MCP tools when relevant."
            )
        elif inventory.instructions.strip():
            description_parts.append(compact_text(inventory.instructions, 360))
        return CapabilityRef(
            kind=CapabilityKind.MCP_SERVER,
            name=server.name.strip(),
            description=" ".join(part for part in description_parts if part.strip()),
        )


def discover_mcp_catalog(
    servers: Iterable[McpServerSpec],
    workspace: str | Path,
    state_store: McpUsageStateStore | None = None,
    now: datetime | None = None,
) -> McpToolCatalog:
    """Discover MCP server inventories for one run."""
    current_time = _normal_datetime(now)
    inventories: list[McpServerInventory] = []
    for server in servers:
        with create_mcp_client(server, workspace) as client:
            tools = client.list_tools()
            inventories.append(
                McpServerInventory(
                    server=server,
                    tools=tools,
                    server_info=client.server_info(),
                    instructions=client.instructions(),
                    capabilities=client.server_capabilities(),
                )
            )
        if state_store is not None:
            state_store.sync_inventory_seen(server, tools, now=current_time)
    return McpToolCatalog(
        inventories=tuple(inventories),
        state_store=state_store,
        now=current_time,
    )


def format_mcp_catalog_status(
    catalog: McpToolCatalog,
    model_context_tokens: int,
) -> str:
    """Return a readable MCP status report for one discovered catalog."""
    if not catalog.inventories:
        return "No MCP servers discovered."

    preloaded = {
        tool.qualified_name()
        for tool in catalog.default_schema_tools(model_context_tokens)
    }
    loaded = set(catalog.loaded_schema_tool_names())
    lines = ["MCP status:"]
    for inventory in catalog.inventories:
        server = inventory.server
        tier = catalog.server_tier(server)
        read_only_tools = inventory.read_only_tools()
        total_tools = len(inventory.tools)
        lines.append(
            "- "
            f"{server.name}: transport={server.transport.value}, "
            f"exposure={server.exposure.value}, tier={tier}, "
            f"tools={total_tools}, read_only_tools={len(read_only_tools)}, "
            f"default_schema_preload={_preload_count(inventory, preloaded)}"
        )
        if inventory.instructions.strip():
            lines.append(f"  instructions: {compact_text(inventory.instructions, 160)}")
        if not inventory.tools:
            lines.append("  tools: none")
            continue
        for tool in inventory.tools:
            qualified = tool.qualified_name()
            if tool.is_read_only():
                access = "read-only"
            elif "readOnlyHint" not in tool.annotations:
                access = "not-executable: readOnlyHint missing"
            else:
                access = "not-executable: not read-only"
            schema_state = "preloaded" if qualified in preloaded else "on-demand"
            if qualified in loaded:
                schema_state = f"{schema_state}, previously-loaded"
            lines.append(
                "  - "
                f"{qualified}: {access}, schema={schema_state} - "
                f"{compact_text(tool.display_description(), 180)}"
            )
    return "\n".join(lines)


def _preload_count(inventory: McpServerInventory, preloaded: set[str]) -> int:
    return sum(
        1
        for tool in inventory.read_only_tools()
        if tool.qualified_name() in preloaded
    )


def _tool_ref(tool: McpToolRef) -> CapabilityRef:
    return CapabilityRef(
        kind=CapabilityKind.MCP_TOOL,
        name=tool.qualified_name(),
        description=tool.display_description(),
    )


def _tool_name_ref(tool: McpToolRef) -> CapabilityRef:
    return CapabilityRef(
        kind=CapabilityKind.MCP_TOOL,
        name=tool.qualified_name(),
        description="",
    )


def _server_description(inventory: McpServerInventory) -> str:
    if inventory.server.description.strip():
        return inventory.server.description.strip()
    name = inventory.server_info.get("name")
    version = inventory.server_info.get("version")
    if isinstance(name, str) and name.strip():
        if isinstance(version, str) and version.strip():
            return f"{name.strip()} {version.strip()} MCP server."
        return f"{name.strip()} MCP server."
    return f"MCP server {inventory.server.name.strip()}."


def _compact_ref_cost(ref: CapabilityRef) -> int:
    text = f"{ref.kind.value} {ref.name} {ref.description}"
    return max(1, len(text) // 4)


def _estimate_tool_schema_tokens(schema: Mapping[str, object]) -> int:
    try:
        payload = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        payload = str(schema)
    return max(1, len(payload) // 4)


def _normal_datetime(value: datetime | None = None) -> datetime:
    current = value or datetime.now(UTC)
    if current.tzinfo is None:
        return current.replace(tzinfo=UTC)
    return current.astimezone(UTC)
