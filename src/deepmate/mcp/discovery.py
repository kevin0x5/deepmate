"""Read-only MCP server discovery helpers."""

from __future__ import annotations

import json
from ast import literal_eval
from collections.abc import Mapping, Sequence

from deepmate.mcp.spec import McpExposure, McpServerSpec, McpTransport


def mcp_server_specs_from_mapping(
    values: Mapping[tuple[str, ...], str],
    root: tuple[str, ...] = ("mcp_servers",),
) -> tuple[McpServerSpec, ...]:
    """Build MCP server specs from a nested config mapping."""
    server_names = sorted(
        path[len(root)]
        for path in values
        if len(path) > len(root) and path[: len(root)] == root
    )
    specs: list[McpServerSpec] = []
    seen: set[str] = set()
    for name in server_names:
        if name in seen:
            continue
        seen.add(name)
        spec = _server_spec(name, values, (*root, name))
        if _is_enabled(values.get((*root, name, "enabled"))) and spec.is_ready():
            specs.append(spec)
    return tuple(specs)


def format_mcp_server_list(servers: Sequence[McpServerSpec]) -> str:
    """Return a readable list of configured MCP servers."""
    if not servers:
        return "No MCP servers configured."

    lines = ["MCP servers:"]
    for server in servers:
        target = server.command if server.transport == McpTransport.STDIO else server.url
        detail = f"{server.transport.value} {target}".strip()
        if server.description.strip():
            detail = f"{detail} - {server.description.strip()}"
        lines.append(f"- {server.name}: {detail}")
    return "\n".join(lines)


def _server_spec(
    name: str,
    values: Mapping[tuple[str, ...], str],
    prefix: tuple[str, ...],
) -> McpServerSpec:
    command = values.get((*prefix, "command"), "").strip()
    url = values.get((*prefix, "url"), "").strip()
    transport = _transport(values.get((*prefix, "transport")), command, url)
    return McpServerSpec(
        name=name.strip(),
        transport=transport,
        command=command,
        args=_args(values.get((*prefix, "args"), "")),
        url=url,
        cwd=values.get((*prefix, "cwd"), "").strip(),
        env=_env(values, (*prefix, "env")),
        bearer_token_env_var=values.get((*prefix, "bearer_token_env_var"), "").strip(),
        description=values.get((*prefix, "description"), "").strip(),
        exposure=_exposure(values.get((*prefix, "exposure"))),
        startup_timeout_seconds=_float_optional(
            values.get((*prefix, "startup_timeout_sec"))
            or values.get((*prefix, "startup_timeout_seconds"))
        ),
    )


def _transport(value: str | None, command: str, url: str) -> McpTransport:
    if value and value.strip():
        normalized = value.strip().replace("-", "_")
        return McpTransport(normalized)
    if command:
        return McpTransport.STDIO
    if url:
        return McpTransport.STREAMABLE_HTTP
    return McpTransport.STDIO


def _args(value: str) -> tuple[str, ...]:
    cleaned = value.strip()
    if not cleaned:
        return ()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        parsed = _parse_inline_array(cleaned)
        if not isinstance(parsed, list):
            raise ValueError(f"MCP args must be a list: {value}")
        return tuple(_arg_text(part) for part in parsed if _arg_text(part))
    return (cleaned,)


def _parse_inline_array(value: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return literal_eval(value)


def _arg_text(value: object) -> str:
    return str(value).strip()


def _env(
    values: Mapping[tuple[str, ...], str],
    prefix: tuple[str, ...],
) -> Mapping[str, str]:
    return {
        path[len(prefix)]: value
        for path, value in values.items()
        if len(path) == len(prefix) + 1 and path[: len(prefix)] == prefix
    }


def _is_enabled(value: str | None) -> bool:
    if value is None or not value.strip():
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _exposure(value: str | None) -> McpExposure:
    if not value or not value.strip():
        return McpExposure.SEARCHABLE
    return McpExposure(value.strip())


def _float_optional(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        return float(value.strip())
    except ValueError as exc:
        raise ValueError(f"invalid MCP timeout value: {value}") from exc
