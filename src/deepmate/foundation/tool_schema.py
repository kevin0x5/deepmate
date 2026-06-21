"""Provider-neutral tool schema normalization helpers."""

from __future__ import annotations

from collections.abc import Mapping

DEEPMATE_SCHEMA_META_KEY = "_deepmate"
MCP_SCHEMA_FLATTENED_FLAG = "mcp_schema_flattened"

_MAX_UNFLATTEN_DEPTH = 12


def flatten_tool_schema_if_complex(schema: Mapping[str, object]) -> Mapping[str, object]:
    """Return a model-facing schema with complex nested input fields flattened."""
    input_schema = schema.get("input_schema")
    if not isinstance(input_schema, Mapping):
        return schema
    if not _should_flatten_input_schema(input_schema):
        return schema
    flat_input_schema = _flatten_input_schema(input_schema)
    if flat_input_schema == input_schema:
        return schema

    metadata = dict(_metadata(schema))
    metadata[MCP_SCHEMA_FLATTENED_FLAG] = True
    return {
        **dict(schema),
        "input_schema": flat_input_schema,
        DEEPMATE_SCHEMA_META_KEY: metadata,
    }


def tool_schema_is_flattened(schema: Mapping[str, object] | None) -> bool:
    """Return whether a tool schema was flattened by Deepmate."""
    if schema is None:
        return False
    return _metadata(schema).get(MCP_SCHEMA_FLATTENED_FLAG) is True


def unflatten_tool_arguments(
    arguments: Mapping[str, object],
    schema: Mapping[str, object] | None,
) -> Mapping[str, object]:
    """Translate dotted model-facing arguments back to nested tool arguments."""
    if not tool_schema_is_flattened(schema):
        return arguments
    normalized: dict[str, object] = {}
    changed = False
    dotted_items: list[tuple[tuple[str, ...], object]] = []
    for key, value in arguments.items():
        if not isinstance(key, str) or "." not in key:
            normalized[key] = value
            continue
        parts = tuple(part.strip() for part in key.split(".") if part.strip())
        if len(parts) < 2 or len(parts) > _MAX_UNFLATTEN_DEPTH:
            normalized[key] = value
            continue
        dotted_items.append((parts, value))
        changed = True
    for parts, value in dotted_items:
        _set_nested_value(normalized, parts, value)
    return normalized if changed else arguments


def _should_flatten_input_schema(schema: Mapping[str, object]) -> bool:
    if not _has_nested_object_properties(schema):
        return False
    return _object_depth(schema) > 2 or _leaf_property_count(schema) > 10


def _flatten_input_schema(schema: Mapping[str, object]) -> Mapping[str, object]:
    properties = _properties(schema)
    if not properties:
        return schema
    flat_properties, flat_required = _flatten_properties(schema)
    if not flat_properties:
        return schema

    flattened = {
        key: value
        for key, value in dict(schema).items()
        if key not in {"type", "properties", "required", "additionalProperties"}
    }
    flattened["type"] = "object"
    flattened["properties"] = flat_properties
    if flat_required:
        flattened["required"] = flat_required
    flattened["additionalProperties"] = False
    return flattened


def _flatten_properties(
    schema: Mapping[str, object],
    prefix: tuple[str, ...] = (),
    ancestors_required: bool = True,
) -> tuple[dict[str, object], list[str]]:
    flat_properties: dict[str, object] = {}
    flat_required: list[str] = []
    required_names = _required_names(schema)
    for name, child_schema in _properties(schema).items():
        path = (*prefix, name)
        child_required = ancestors_required and name in required_names
        if isinstance(child_schema, Mapping) and _properties(child_schema):
            child_properties, child_required_names = _flatten_properties(
                child_schema,
                prefix=path,
                ancestors_required=child_required,
            )
            flat_properties.update(child_properties)
            flat_required.extend(child_required_names)
            continue

        flat_name = ".".join(path)
        flat_properties[flat_name] = _flattened_leaf_schema(child_schema, path)
        if child_required:
            flat_required.append(flat_name)
    return flat_properties, flat_required


def _flattened_leaf_schema(value: object, path: tuple[str, ...]) -> object:
    if not isinstance(value, Mapping):
        return value
    schema = dict(value)
    if len(path) <= 1:
        return schema
    description = schema.get("description")
    prefix = f"Nested field: {'.'.join(path)}."
    if isinstance(description, str) and description.strip():
        schema["description"] = f"{prefix} {description.strip()}"
    else:
        schema["description"] = prefix
    return schema


def _has_nested_object_properties(schema: Mapping[str, object]) -> bool:
    for child_schema in _properties(schema).values():
        if isinstance(child_schema, Mapping) and _properties(child_schema):
            return True
    return False


def _object_depth(schema: object) -> int:
    if not isinstance(schema, Mapping):
        return 1
    properties = _properties(schema)
    if not properties:
        return 1
    return 1 + max(_object_depth(child) for child in properties.values())


def _leaf_property_count(schema: object) -> int:
    if not isinstance(schema, Mapping):
        return 1
    properties = _properties(schema)
    if not properties:
        return 1
    return sum(_leaf_property_count(child) for child in properties.values())


def _properties(schema: Mapping[str, object]) -> Mapping[str, object]:
    properties = schema.get("properties")
    return properties if isinstance(properties, Mapping) else {}


def _required_names(schema: Mapping[str, object]) -> set[str]:
    required = schema.get("required")
    if not isinstance(required, list):
        return set()
    return {item.strip() for item in required if isinstance(item, str) and item.strip()}


def _metadata(schema: Mapping[str, object]) -> Mapping[str, object]:
    metadata = schema.get(DEEPMATE_SCHEMA_META_KEY)
    return metadata if isinstance(metadata, Mapping) else {}


def _set_nested_value(target: dict[str, object], path: tuple[str, ...], value: object) -> None:
    current = target
    for part in path[:-1]:
        existing = current.get(part)
        if not isinstance(existing, dict):
            existing = {}
            current[part] = existing
        current = existing
    current[path[-1]] = value
