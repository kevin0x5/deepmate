"""Deterministic hook condition matching."""

from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import PurePosixPath
from typing import Any

from deepmate.runtime.hooks.types import HookActor, HookDefinition, HookEnvelope


SUPPORTED_WHEN_KEYS: frozenset[str] = frozenset(
    {
        "tool_names",
        "tool_sources",
        "server_names",
        "path_globs",
        "changed_globs",
        "risk_levels",
        "network",
        "actors",
        "session_reasons",
        "status",
        "status_in",
        "text_contains",
    }
)


def hook_matches(hook: HookDefinition, envelope: HookEnvelope) -> bool:
    """Return whether one hook should run for the envelope."""
    if hook.event_name != envelope.event_name:
        return False
    if not _run_target_matches(hook.run_on.value, envelope.actor):
        return False
    for key, expected in hook.when.items():
        if not _condition_matches(key, expected, envelope):
            return False
    return True


def _run_target_matches(run_on: str, actor: HookActor) -> bool:
    return run_on == "all" or run_on == actor.value


def _condition_matches(key: str, expected: object, envelope: HookEnvelope) -> bool:
    payload = envelope.payload
    if key == "actors":
        return _text(payload.get("actor") or envelope.actor.value) in _string_values(
            expected
        )
    if key == "tool_names":
        return _text(payload.get("tool_name")) in _string_values(expected)
    if key == "tool_sources":
        return _text(payload.get("tool_source")) in _string_values(expected)
    if key == "server_names":
        return _text(payload.get("server_name")) in _string_values(expected)
    if key == "risk_levels":
        return _text(payload.get("risk_level")) in _string_values(expected)
    if key == "network":
        return _text(payload.get("network")) == _single_text(expected)
    if key == "session_reasons":
        return _text(payload.get("session_reason")) in _string_values(expected)
    if key in {"status", "status_in"}:
        return _text(payload.get("status")) in _string_values(expected)
    if key == "path_globs":
        return _path_matches(payload.get("path"), expected)
    if key == "changed_globs":
        changed = _string_values(payload.get("changed_paths"))
        globs = _string_values(expected)
        return any(_glob_matches(path, pattern) for path in changed for pattern in globs)
    if key == "text_contains":
        text = _text(payload.get("text") or payload.get("summary"))
        needles = _string_values(expected)
        return any(needle and needle in text for needle in needles)
    return False


def _path_matches(value: object, expected: object) -> bool:
    path = _text(value)
    if not path:
        return False
    return any(_glob_matches(path, pattern) for pattern in _string_values(expected))


def _glob_matches(path: str, pattern: str) -> bool:
    normalized_path = str(PurePosixPath(path.replace("\\", "/")))
    normalized_pattern = str(PurePosixPath(pattern.replace("\\", "/")))
    return fnmatchcase(normalized_path, normalized_pattern)


def _string_values(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, (list, tuple, set)):
        return tuple(_text(item) for item in value if _text(item))
    return tuple()


def _single_text(value: object) -> str:
    values = _string_values(value)
    return values[0] if values else _text(value)


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""
