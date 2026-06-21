"""Native tools for MCP progressive disclosure."""

from __future__ import annotations

from collections.abc import Mapping

from deepmate.foundation.tool_schema import (
    flatten_tool_schema_if_complex,
    tool_schema_is_flattened,
)
from deepmate.mcp import McpToolCatalog
from deepmate.tools.registry import NativeTool, NativeToolResult

SEARCH_MCP_TOOLS_NAME = "search_mcp_tools"
LOAD_MCP_TOOL_NAME = "load_mcp_tool"


def mcp_loader_tools(catalog: McpToolCatalog | None) -> tuple[NativeTool, ...]:
    """Return MCP progressive-disclosure native tools when MCP tools exist."""
    if catalog is None or not catalog.read_only_tools():
        return ()
    return (
        NativeTool(
            name=SEARCH_MCP_TOOLS_NAME,
            description=(
                "Search available MCP tools by task, server, or keyword. Use this "
                "when an MCP server looks relevant but the exact tool schema is not "
                "loaded yet."
            ),
            input_schema=_search_schema(),
            handler=lambda arguments: _search_mcp_tools(catalog, arguments),
        ),
        NativeTool(
            name=LOAD_MCP_TOOL_NAME,
            description=(
                "Load one exact read-only MCP tool schema for the next model step. "
                "Call this before invoking an MCP tool whose schema is not currently "
                "available."
            ),
            input_schema=_load_schema(),
            handler=lambda arguments: _load_mcp_tool(catalog, arguments),
        ),
    )


def _search_mcp_tools(
    catalog: McpToolCatalog,
    arguments: Mapping[str, object],
) -> NativeToolResult:
    query = _optional_text_argument(arguments, "query")
    server = _optional_text_argument(arguments, "server")
    limit = _limit_argument(arguments.get("limit"))
    tools = catalog.search(query=query, server_name=server, limit=limit)
    if not tools:
        return NativeToolResult(
            content="No matching read-only MCP tools were found.",
            refs=("mcp_tools=0",),
        )
    lines = [
        "Matching read-only MCP tools:",
        *(
            f"- {tool.qualified_name()}: {tool.display_description()}"
            for tool in tools
        ),
        "Call load_mcp_tool with an exact tool name before invoking one.",
    ]
    return NativeToolResult(
        content="\n".join(lines),
        data={"tools": [tool.qualified_name() for tool in tools]},
        refs=(
            f"mcp_tools={len(tools)}",
            *(f"mcp_tool={tool.qualified_name()}" for tool in tools),
        ),
    )


def _load_mcp_tool(
    catalog: McpToolCatalog,
    arguments: Mapping[str, object],
) -> NativeToolResult:
    name = _required_text_argument(arguments, "name")
    tool = catalog.get_tool(name)
    if tool is None:
        raise ValueError(f"read-only MCP tool not found: {name}")
    if catalog.state_store is not None:
        catalog.state_store.record_tool_loaded(tool)
    schema = flatten_tool_schema_if_complex(tool.schema())
    flattened = tool_schema_is_flattened(schema)
    return NativeToolResult(
        content=(
            f"MCP tool schema loaded for next step: {tool.qualified_name()}. "
            "Invoke that exact tool name with arguments matching its schema."
            + (
                " Nested fields may be exposed as dotted argument names; Deepmate "
                "will translate them back before calling the MCP server."
                if flattened
                else ""
            )
        ),
        data={
            "tool": tool.qualified_name(),
            "server": tool.server_name,
            "description": tool.display_description(),
            "schema_flattened": flattened,
        },
        refs=(
            f"mcp_tool={tool.qualified_name()}",
            f"mcp_server={tool.server_name}",
            "schema_loaded=true",
            *(("schema_flattened=true",) if flattened else ()),
        ),
        schema_additions=(schema,),
    )


def _search_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keyword or task phrase to match against MCP tool names and descriptions.",
            },
            "server": {
                "type": "string",
                "description": "Optional MCP server name to restrict the search.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum matches to return. Defaults to 10.",
                "minimum": 1,
                "maximum": 20,
            },
        },
        "additionalProperties": False,
    }


def _load_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Exact qualified MCP tool name, for example filesystem.read_text_file.",
            }
        },
        "required": ["name"],
        "additionalProperties": False,
    }


def _required_text_argument(arguments: Mapping[str, object], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{LOAD_MCP_TOOL_NAME} requires text argument: {name}")
    return value.strip()


def _optional_text_argument(arguments: Mapping[str, object], name: str) -> str:
    value = arguments.get(name)
    return value.strip() if isinstance(value, str) else ""


def _limit_argument(value: object) -> int:
    if isinstance(value, int):
        return min(20, max(1, value))
    return 10
