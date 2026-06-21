"""Stable request-prefix fingerprints for cache observability."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256

from deepmate.domain import MessageRole
from deepmate.providers import ModelRequest


@dataclass(frozen=True, slots=True)
class PrefixFingerprint:
    """Digest summary for the stable prefix portion of one model request."""

    digest: str
    system_digest: str
    tool_schema_digest: str
    options_digest: str
    model: str
    tool_schema_names: tuple[str, ...]
    conversation_prefix_items: int = 1

    def trace_refs(self) -> tuple[str, ...]:
        """Return compact trace refs for runtime observability."""
        names = ",".join(self.tool_schema_names[:40])
        if len(self.tool_schema_names) > 40:
            names = f"{names},..."
        return (
            f"prefix_digest={self.digest}",
            f"system_digest={self.system_digest}",
            f"tool_schema_digest={self.tool_schema_digest}",
            f"tool_schema_count={len(self.tool_schema_names)}",
            f"tool_schema_names={names}",
            f"options_digest={self.options_digest}",
            f"model={self.model}",
            f"conversation_prefix_items={self.conversation_prefix_items}",
        )


def model_request_prefix_fingerprint(request: ModelRequest) -> PrefixFingerprint:
    """Return a deterministic fingerprint for request prefix cache analysis."""
    system_payload = _system_payload(request)
    tool_schema_payload = tuple(_json_safe(schema) for schema in request.tool_schemas)
    options_payload = _json_safe(request.options)
    model = request.model.strip()
    system_digest = _digest(system_payload)
    tool_schema_digest = _digest(tool_schema_payload)
    options_digest = _digest(options_payload)
    digest = _digest(
        {
            "model": model,
            "system_digest": system_digest,
            "tool_schema_digest": tool_schema_digest,
            "options_digest": options_digest,
        }
    )
    return PrefixFingerprint(
        digest=digest,
        system_digest=system_digest,
        tool_schema_digest=tool_schema_digest,
        options_digest=options_digest,
        model=model,
        tool_schema_names=tuple(
            _tool_schema_name(schema) for schema in request.tool_schemas
        ),
    )


def _system_payload(request: ModelRequest) -> Mapping[str, object]:
    if not request.conversation:
        return {"role": "", "content": ""}
    item = request.conversation[0]
    message = item.message
    if message is None:
        return {"role": "", "content": ""}
    role = (
        message.role.value
        if isinstance(message.role, MessageRole)
        else str(message.role)
    )
    return {"role": role, "content": message.content}


def _tool_schema_name(schema: Mapping[str, object]) -> str:
    function = schema.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    name = schema.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return "<unnamed>"


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    return str(value)


def _digest(value: object) -> str:
    payload = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()[:16]
