"""MCP configuration objects for Deepmate."""

from deepmate.mcp.client import (
    McpCallResult,
    McpClientError,
    McpClientSession,
    McpStdioClient,
    McpStreamableHttpClient,
    create_mcp_client,
    discover_mcp_tools,
)
from deepmate.mcp.catalog import (
    MCP_COMPACT_SURFACE_RATIO,
    MCP_SCHEMA_PRELOAD_RATIO,
    McpServerInventory,
    McpToolCatalog,
    discover_mcp_catalog,
    format_mcp_catalog_status,
)
from deepmate.mcp.discovery import (
    format_mcp_server_list,
    mcp_server_specs_from_mapping,
)
from deepmate.mcp.executor import McpToolExecutionResult, McpToolExecutor
from deepmate.mcp.output_policy import (
    DEFAULT_MAX_MCP_OUTPUT_TOKENS,
    DEFAULT_MCP_OUTPUT_RATIO,
    DEFAULT_MIN_MCP_OUTPUT_TOKENS,
    McpOutputPolicy,
    McpOutputPolicyResult,
)
from deepmate.mcp.spec import McpExposure, McpServerSpec, McpToolRef, McpTransport
from deepmate.mcp.state import DEFAULT_MCP_IDLE_DAYS, McpUsageEntry, McpUsageStateStore

__all__ = [
    "DEFAULT_MCP_IDLE_DAYS",
    "DEFAULT_MAX_MCP_OUTPUT_TOKENS",
    "DEFAULT_MCP_OUTPUT_RATIO",
    "DEFAULT_MIN_MCP_OUTPUT_TOKENS",
    "MCP_COMPACT_SURFACE_RATIO",
    "MCP_SCHEMA_PRELOAD_RATIO",
    "McpCallResult",
    "McpClientError",
    "McpClientSession",
    "McpExposure",
    "McpOutputPolicy",
    "McpOutputPolicyResult",
    "McpServerSpec",
    "McpServerInventory",
    "McpStdioClient",
    "McpStreamableHttpClient",
    "McpToolCatalog",
    "McpToolExecutionResult",
    "McpToolExecutor",
    "McpToolRef",
    "McpTransport",
    "McpUsageEntry",
    "McpUsageStateStore",
    "create_mcp_client",
    "discover_mcp_catalog",
    "discover_mcp_tools",
    "format_mcp_catalog_status",
    "format_mcp_server_list",
    "mcp_server_specs_from_mapping",
]
