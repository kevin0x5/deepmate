"""Load and validate declarative hook packs."""

from __future__ import annotations

import ast
import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from deepmate.runtime.hooks.matcher import SUPPORTED_WHEN_KEYS
from deepmate.runtime.hooks.registry import HookRegistry, builtin_hook_definitions
from deepmate.runtime.hooks.trust import HookTrustStore
from deepmate.runtime.hooks.types import (
    HIGH_RISK_ACTION_TYPES,
    HOOK_LAYER_ORDER,
    HookAction,
    HookActionType,
    HookDefinition,
    HookDiagnostic,
    HookDiagnosticLevel,
    HookErrorPolicy,
    HookEvent,
    HookLayer,
    HookLoadOptions,
    HookRunTarget,
)

SUPPORTED_HOOK_FILE_EXTENSIONS = frozenset({".json", ".yaml", ".yml"})
CONNECTED_ACTION_TYPES: frozenset[HookActionType] = frozenset(
    {
        HookActionType.DENY,
        HookActionType.ASK,
        HookActionType.TRACE,
        HookActionType.RECORD_MEMORY_SIGNAL,
        HookActionType.RECORD_EVOLUTION_SIGNAL,
    }
)

_ACTION_ALLOWED_KEYS: dict[HookActionType, frozenset[str]] = {
    HookActionType.DENY: frozenset({"type", "reason", "summary"}),
    HookActionType.ASK: frozenset({"type", "reason", "summary"}),
    HookActionType.TRACE: frozenset({"type", "summary", "refs"}),
    HookActionType.CHECKPOINT: frozenset({"type", "reason", "summary"}),
    HookActionType.COMPACT: frozenset(
        {"type", "target", "reason", "summary", "budget_tokens"}
    ),
    HookActionType.RECORD_MEMORY_SIGNAL: frozenset(
        {"type", "signal_kind", "summary", "refs"}
    ),
    HookActionType.RECORD_EVOLUTION_SIGNAL: frozenset(
        {"type", "signal_kind", "summary", "refs"}
    ),
    HookActionType.PATCH_TOOL_ARGS: frozenset(
        {
            "type",
            "set",
            "timeout_ms",
            "network",
            "output_budget_tokens",
            "flattened_args",
            "reason",
        }
    ),
    HookActionType.PATCH_PROVIDER_OPTIONS: frozenset(
        {
            "type",
            "max_tokens",
            "temperature",
            "reasoning_effort",
            "metadata",
            "instruction_suffix",
            "reason",
        }
    ),
    HookActionType.PATCH_TOOL_RESULT: frozenset(
        {"type", "redact", "refs", "labels", "summary", "reason"}
    ),
    HookActionType.SET_STATUS: frozenset({"type", "status", "summary"}),
    HookActionType.NOTIFY: frozenset({"type", "message", "level"}),
    HookActionType.RUN_SHELL: frozenset({"type", "command", "timeout_ms", "reason"}),
    HookActionType.CALL_MCP: frozenset(
        {"type", "server_name", "tool_name", "arguments", "reason"}
    ),
    HookActionType.WORKSPACE_WRITE: frozenset(
        {"type", "path", "content", "reason"}
    ),
    HookActionType.OVERRIDE_TOOL_SURFACE: frozenset(
        {"type", "mode", "tools", "reason"}
    ),
}


@dataclass(frozen=True, slots=True)
class HookSourceFile:
    """One hook pack file discovered in a source layer."""

    layer: HookLayer
    path: Path
    trusted: bool = True
    skipped_reason: str = ""


@dataclass(frozen=True, slots=True)
class HookLoadReport:
    """Result of loading and validating all configured hook sources."""

    registry: HookRegistry
    options: HookLoadOptions = field(default_factory=HookLoadOptions)
    diagnostics: tuple[HookDiagnostic, ...] = field(default_factory=tuple)
    discovered_counts: Mapping[str, int] = field(default_factory=dict)
    skipped_counts: Mapping[str, int] = field(default_factory=dict)
    workspace_trusted: bool = False
    trust_path: Path | None = None

    def loaded_counts(self) -> dict[str, int]:
        """Return loaded hook counts by layer."""
        return self.registry.layer_counts()

    def has_errors(self) -> bool:
        """Return whether validation found an error."""
        return any(
            diagnostic.level == HookDiagnosticLevel.ERROR
            for diagnostic in self.diagnostics
        )


def load_hook_report(
    workspace: str | Path,
    data_dir: str | Path,
    options: HookLoadOptions | None = None,
    session_hook_paths: Iterable[str | Path] = (),
) -> HookLoadReport:
    """Load builtin and configured hook packs into a registry."""
    resolved_options = options or HookLoadOptions()
    workspace_path = Path(workspace).resolve()
    trust_store = HookTrustStore.in_data_dir(data_dir)
    workspace_trusted = trust_store.is_trusted(workspace_path)
    diagnostics: list[HookDiagnostic] = []
    skipped_counts: Counter[str] = Counter()
    hooks: list[HookDefinition] = list(builtin_hook_definitions())
    sources = discover_hook_sources(
        workspace_path,
        resolved_options,
        workspace_trusted=workspace_trusted,
        session_hook_paths=session_hook_paths,
    )
    discovered_counts = Counter(source.layer.value for source in sources)

    if not resolved_options.enabled:
        skipped = sum(
            1 for source in sources if source.layer is not HookLayer.BUILTIN
        )
        if skipped:
            skipped_counts["disabled_config"] += skipped
            diagnostics.append(
                HookDiagnostic(
                    level=HookDiagnosticLevel.INFO,
                    message=f"configurable hooks disabled; skipped {skipped} hook file(s)",
                    source_layer="config",
                )
            )
        return HookLoadReport(
            registry=HookRegistry.from_hooks(hooks),
            options=resolved_options,
            diagnostics=tuple(diagnostics),
            discovered_counts=_count_map(discovered_counts),
            skipped_counts=_count_map(skipped_counts),
            workspace_trusted=workspace_trusted,
            trust_path=trust_store.path,
        )

    for file_order, source in enumerate(sources):
        if source.skipped_reason:
            skipped_counts[source.skipped_reason] += 1
            diagnostics.append(
                HookDiagnostic(
                    level=HookDiagnosticLevel.WARNING,
                    message=f"skipped hook file: {source.skipped_reason}",
                    source_layer=source.layer.value,
                    refs=(f"path={source.path}",),
                )
            )
            continue
        try:
            loaded = _load_hook_file(source, file_order)
        except ValueError as exc:
            skipped_counts["invalid"] += 1
            diagnostics.append(
                HookDiagnostic(
                    level=HookDiagnosticLevel.ERROR,
                    message=str(exc),
                    source_layer=source.layer.value,
                    refs=(f"path={source.path}",),
                )
            )
            continue
        for hook, hook_diagnostics in loaded:
            diagnostics.extend(hook_diagnostics)
            if any(
                diagnostic.level == HookDiagnosticLevel.ERROR
                for diagnostic in hook_diagnostics
            ):
                skipped_counts["invalid"] += 1
                continue
            if not hook.enabled:
                skipped_counts["disabled"] += 1
                continue
            hooks.append(hook)

    return HookLoadReport(
        registry=HookRegistry.from_hooks(hooks),
        options=resolved_options,
        diagnostics=tuple(diagnostics),
        discovered_counts=_count_map(discovered_counts),
        skipped_counts=_count_map(skipped_counts),
        workspace_trusted=workspace_trusted,
        trust_path=trust_store.path,
    )


def discover_hook_sources(
    workspace: Path,
    options: HookLoadOptions,
    *,
    workspace_trusted: bool,
    session_hook_paths: Iterable[str | Path] = (),
) -> tuple[HookSourceFile, ...]:
    """Discover hook source files while preserving skipped-layer diagnostics."""
    sources: list[HookSourceFile] = []
    home = _deepmate_home()
    sources.extend(
        HookSourceFile(HookLayer.MANAGED, path)
        for path in _hook_files(home / "hooks" / "managed")
    )
    user_files = _hook_files(home / "hooks" / "user")
    project_files = _hook_files(workspace / ".deepmate" / "hooks")
    session_files = tuple(
        Path(path).expanduser().resolve()
        for path in session_hook_paths
        if Path(path).suffix.lower() in SUPPORTED_HOOK_FILE_EXTENSIONS
    )

    if options.managed_hooks_only:
        sources.extend(
            HookSourceFile(HookLayer.USER, path, skipped_reason="managed_only")
            for path in user_files
        )
        sources.extend(
            HookSourceFile(HookLayer.PROJECT, path, skipped_reason="managed_only")
            for path in project_files
        )
        sources.extend(
            HookSourceFile(HookLayer.SESSION, path, skipped_reason="managed_only")
            for path in session_files
        )
        return tuple(sources)

    if options.load_user_hooks:
        sources.extend(HookSourceFile(HookLayer.USER, path) for path in user_files)
    else:
        sources.extend(
            HookSourceFile(HookLayer.USER, path, skipped_reason="user_hooks_disabled")
            for path in user_files
        )

    if options.load_project_hooks:
        skip_reason = "" if workspace_trusted else "untrusted_project"
        sources.extend(
            HookSourceFile(HookLayer.PROJECT, path, skipped_reason=skip_reason)
            for path in project_files
        )
    else:
        sources.extend(
            HookSourceFile(
                HookLayer.PROJECT,
                path,
                skipped_reason="project_hooks_disabled",
            )
            for path in project_files
        )

    sources.extend(HookSourceFile(HookLayer.SESSION, path) for path in session_files)
    return tuple(sources)


def _load_hook_file(
    source: HookSourceFile,
    file_order: int,
) -> tuple[tuple[HookDefinition, tuple[HookDiagnostic, ...]], ...]:
    raw = _read_hook_pack(source.path)
    if not isinstance(raw, Mapping):
        raise ValueError("hook pack must be an object with version and hooks")
    version = raw.get("version")
    if str(version).strip() != "1":
        raise ValueError(f"unsupported hook pack version: {version}")
    raw_hooks = raw.get("hooks")
    if not isinstance(raw_hooks, list):
        raise ValueError("hook pack requires a hooks list")

    loaded: list[tuple[HookDefinition, tuple[HookDiagnostic, ...]]] = []
    for hook_order, raw_hook in enumerate(raw_hooks):
        if not isinstance(raw_hook, Mapping):
            loaded.append(
                (
                    _invalid_placeholder(source, file_order, hook_order),
                    (
                        HookDiagnostic(
                            level=HookDiagnosticLevel.ERROR,
                            message="hook entry must be an object",
                            source_layer=source.layer.value,
                            refs=(f"path={source.path}",),
                        ),
                    ),
                )
            )
            continue
        loaded.append(
            _parse_hook_definition(source, raw_hook, file_order, hook_order)
        )
    return tuple(loaded)


def _parse_hook_definition(
    source: HookSourceFile,
    raw_hook: Mapping[str, object],
    file_order: int,
    hook_order: int,
) -> tuple[HookDefinition, tuple[HookDiagnostic, ...]]:
    diagnostics: list[HookDiagnostic] = []
    hook_id = _required_text(raw_hook, "id")
    if not hook_id:
        hook_id = f"invalid-{file_order}-{hook_order}"
        diagnostics.append(_error("hook requires non-empty id", source, hook_id))

    event_name = _parse_enum(
        HookEvent,
        raw_hook.get("on"),
        source,
        hook_id,
        "hook requires a known on event",
        diagnostics,
        fallback=HookEvent.AGENT_TURN_END,
    )
    run_on = _parse_enum(
        HookRunTarget,
        raw_hook.get("run_on") or _default_run_target(event_name).value,
        source,
        hook_id,
        "hook has invalid run_on value",
        diagnostics,
        fallback=HookRunTarget.MAIN,
    )
    on_error = _parse_enum(
        HookErrorPolicy,
        raw_hook.get("on_error") or HookErrorPolicy.WARN.value,
        source,
        hook_id,
        "hook has invalid on_error value",
        diagnostics,
        fallback=HookErrorPolicy.WARN,
    )
    when = raw_hook.get("when") or {}
    if not isinstance(when, Mapping):
        diagnostics.append(_error("hook when must be an object", source, hook_id))
        when = {}
    else:
        diagnostics.extend(_validate_when(source, hook_id, when))
    actions = raw_hook.get("actions")
    parsed_actions: tuple[HookAction, ...] = ()
    if not isinstance(actions, list) or not actions:
        diagnostics.append(_error("hook requires a non-empty actions list", source, hook_id))
    else:
        parsed_actions, action_diagnostics = _parse_actions(
            source,
            hook_id,
            actions,
        )
        diagnostics.extend(action_diagnostics)

    hook = HookDefinition(
        hook_id=hook_id,
        event_name=event_name,
        layer=source.layer,
        enabled=_bool_value(raw_hook.get("enabled"), True),
        description=_text(raw_hook.get("description")),
        run_on=run_on,
        when=dict(when),
        actions=parsed_actions,
        priority=_int_value(raw_hook.get("priority"), 0),
        on_error=on_error,
        source_path=source.path,
        file_order=file_order,
        hook_order=hook_order,
    )
    return hook, tuple(diagnostics)


def _default_run_target(event_name: HookEvent) -> HookRunTarget:
    if event_name in {
        HookEvent.CHECKPOINT_CREATED,
        HookEvent.MEMORY_PATCH_APPLIED,
        HookEvent.MAINTENANCE_BEFORE_RUN,
        HookEvent.MAINTENANCE_AFTER_RUN,
    }:
        return HookRunTarget.MAINTENANCE
    return HookRunTarget.MAIN


def _parse_actions(
    source: HookSourceFile,
    hook_id: str,
    actions: list[object],
) -> tuple[tuple[HookAction, ...], tuple[HookDiagnostic, ...]]:
    parsed: list[HookAction] = []
    diagnostics: list[HookDiagnostic] = []
    for raw_action in actions:
        if not isinstance(raw_action, Mapping):
            diagnostics.append(_error("hook action must be an object", source, hook_id))
            continue
        action_type = _parse_action_type(source, hook_id, raw_action, diagnostics)
        if action_type is None:
            continue
        if (
            action_type in HIGH_RISK_ACTION_TYPES
            and source.layer not in {HookLayer.BUILTIN, HookLayer.MANAGED}
        ):
            diagnostics.append(
                _error(
                    f"high-risk action {action_type.value} is only allowed in builtin or managed hooks",
                    source,
                    hook_id,
                )
            )
            continue
        if action_type not in CONNECTED_ACTION_TYPES:
            diagnostics.append(
                _warning(
                    f"hook action {action_type.value} is not connected to runtime side effects yet",
                    source,
                    hook_id,
                )
            )
            continue
        allowed_keys = _ACTION_ALLOWED_KEYS[action_type]
        unknown_keys = sorted(str(key) for key in raw_action if str(key) not in allowed_keys)
        if unknown_keys:
            diagnostics.append(
                _error(
                    f"action {action_type.value} has unsupported field(s): {', '.join(unknown_keys)}",
                    source,
                    hook_id,
                )
            )
        parsed.append(
            HookAction(
                action_type=action_type,
                params={str(key): value for key, value in raw_action.items() if key != "type"},
            )
        )
    return tuple(parsed), tuple(diagnostics)


def _parse_action_type(
    source: HookSourceFile,
    hook_id: str,
    raw_action: Mapping[str, object],
    diagnostics: list[HookDiagnostic],
) -> HookActionType | None:
    value = _text(raw_action.get("type"))
    try:
        return HookActionType(value)
    except ValueError:
        diagnostics.append(
            _error(f"unknown hook action type: {value or '<empty>'}", source, hook_id)
        )
        return None


def _validate_when(
    source: HookSourceFile,
    hook_id: str,
    when: Mapping[str, object],
) -> tuple[HookDiagnostic, ...]:
    diagnostics: list[HookDiagnostic] = []
    for key, value in when.items():
        clean_key = str(key).strip()
        if clean_key not in SUPPORTED_WHEN_KEYS:
            diagnostics.append(_error(f"unsupported when condition: {clean_key}", source, hook_id))
            continue
        if clean_key in {"path_globs", "changed_globs"}:
            for pattern in _string_values(value):
                if not _is_safe_relative_glob(pattern):
                    diagnostics.append(
                        _error(
                            f"path glob must be relative and stay inside workspace: {pattern}",
                            source,
                            hook_id,
                        )
                    )
    return tuple(diagnostics)


def _read_hook_pack(path: Path) -> object:
    suffix = path.suffix.lower()
    content = path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(content)
    if suffix in {".yaml", ".yml"}:
        return _parse_yaml_subset(content, path)
    raise ValueError(f"unsupported hook file extension: {path.suffix}")


@dataclass(frozen=True, slots=True)
class _YamlLine:
    indent: int
    content: str
    line_no: int


def _parse_yaml_subset(content: str, path: Path) -> object:
    lines = _yaml_lines(content, path)
    if not lines:
        return {}
    result, index = _parse_yaml_block(lines, 0, lines[0].indent, path)
    if index != len(lines):
        raise ValueError(f"invalid YAML structure in {path}: line {lines[index].line_no}")
    return result


def _yaml_lines(content: str, path: Path) -> tuple[_YamlLine, ...]:
    lines: list[_YamlLine] = []
    for line_no, raw_line in enumerate(content.splitlines(), start=1):
        if raw_line.lstrip(" ") != raw_line.lstrip():
            raise ValueError(f"tabs are not supported in hook YAML: {path}:{line_no}")
        cleaned = _strip_unquoted_comment(raw_line).rstrip()
        if not cleaned.strip():
            continue
        indent = len(cleaned) - len(cleaned.lstrip(" "))
        lines.append(_YamlLine(indent=indent, content=cleaned.strip(), line_no=line_no))
    return tuple(lines)


def _parse_yaml_block(
    lines: tuple[_YamlLine, ...],
    index: int,
    indent: int,
    path: Path,
) -> tuple[object, int]:
    if index >= len(lines):
        return {}, index
    if lines[index].indent != indent:
        raise ValueError(f"invalid YAML indentation in {path}: line {lines[index].line_no}")
    if lines[index].content.startswith("- "):
        return _parse_yaml_list(lines, index, indent, path)
    return _parse_yaml_mapping(lines, index, indent, path)


def _parse_yaml_mapping(
    lines: tuple[_YamlLine, ...],
    index: int,
    indent: int,
    path: Path,
) -> tuple[dict[str, object], int]:
    mapping: dict[str, object] = {}
    while index < len(lines):
        line = lines[index]
        if line.indent < indent:
            break
        if line.indent > indent:
            raise ValueError(f"invalid YAML indentation in {path}: line {line.line_no}")
        if line.content.startswith("- "):
            break
        key, separator, raw_value = line.content.partition(":")
        if not separator or not key.strip():
            raise ValueError(f"invalid YAML mapping line in {path}: line {line.line_no}")
        index += 1
        clean_value = raw_value.strip()
        if clean_value:
            mapping[key.strip()] = _parse_scalar(clean_value)
            continue
        if index < len(lines) and lines[index].indent > indent:
            child, index = _parse_yaml_block(lines, index, lines[index].indent, path)
            mapping[key.strip()] = child
        else:
            mapping[key.strip()] = {}
    return mapping, index


def _parse_yaml_list(
    lines: tuple[_YamlLine, ...],
    index: int,
    indent: int,
    path: Path,
) -> tuple[list[object], int]:
    items: list[object] = []
    while index < len(lines):
        line = lines[index]
        if line.indent < indent:
            break
        if line.indent > indent:
            raise ValueError(f"invalid YAML indentation in {path}: line {line.line_no}")
        if not line.content.startswith("- "):
            break
        item_text = line.content[2:].strip()
        index += 1
        if not item_text:
            if index < len(lines) and lines[index].indent > indent:
                item, index = _parse_yaml_block(lines, index, lines[index].indent, path)
            else:
                item = ""
            items.append(item)
            continue
        if _looks_like_mapping_pair(item_text):
            key, _, raw_value = item_text.partition(":")
            item_map: dict[str, object] = {}
            clean_value = raw_value.strip()
            if clean_value:
                item_map[key.strip()] = _parse_scalar(clean_value)
            elif index < len(lines) and lines[index].indent > indent:
                child, index = _parse_yaml_block(lines, index, lines[index].indent, path)
                item_map[key.strip()] = child
            else:
                item_map[key.strip()] = {}
            if index < len(lines) and lines[index].indent > indent:
                child, index = _parse_yaml_block(lines, index, lines[index].indent, path)
                if not isinstance(child, Mapping):
                    raise ValueError(
                        f"list mapping continuation must be a mapping in {path}: line {line.line_no}"
                    )
                item_map.update({str(child_key): child_value for child_key, child_value in child.items()})
            items.append(item_map)
        else:
            items.append(_parse_scalar(item_text))
    return items, index


def _parse_scalar(value: str) -> object:
    clean = value.strip()
    if clean in {"true", "True"}:
        return True
    if clean in {"false", "False"}:
        return False
    if clean in {"null", "Null", "none", "None", "~"}:
        return None
    if len(clean) >= 2 and clean[0] == clean[-1] and clean[0] in {"'", '"'}:
        return clean[1:-1]
    if clean.startswith("[") and clean.endswith("]"):
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(clean)
            except (SyntaxError, ValueError):
                return [item.strip().strip("'\"") for item in clean[1:-1].split(",") if item.strip()]
            if isinstance(parsed, list):
                return parsed
    try:
        return int(clean)
    except ValueError:
        return clean


def _strip_unquoted_comment(line: str) -> str:
    quote = ""
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote:
            escaped = True
            continue
        if char in {"'", '"'}:
            if quote == char:
                quote = ""
            elif not quote:
                quote = char
            continue
        if char == "#" and not quote:
            return line[:index]
    return line


def _looks_like_mapping_pair(text: str) -> bool:
    key, separator, _ = text.partition(":")
    return bool(separator and key.strip() and " " not in key.strip())


def _hook_files(directory: Path) -> tuple[Path, ...]:
    if not directory.exists() or not directory.is_dir():
        return ()
    return tuple(
        sorted(
            (
                path.resolve()
                for path in directory.iterdir()
                if path.is_file()
                and path.suffix.lower() in SUPPORTED_HOOK_FILE_EXTENSIONS
            ),
            key=lambda path: path.name,
        )
    )


def _deepmate_home() -> Path:
    raw_home = os.environ.get("DEEPMATE_HOME")
    if raw_home:
        return Path(raw_home).expanduser().resolve()
    return Path.home() / ".deepmate"


def _parse_enum(
    enum_type,
    raw_value: object,
    source: HookSourceFile,
    hook_id: str,
    message: str,
    diagnostics: list[HookDiagnostic],
    *,
    fallback,
):
    value = _text(raw_value)
    try:
        return enum_type(value)
    except ValueError:
        diagnostics.append(_error(f"{message}: {value or '<empty>'}", source, hook_id))
        return fallback


def _invalid_placeholder(
    source: HookSourceFile,
    file_order: int,
    hook_order: int,
) -> HookDefinition:
    return HookDefinition(
        hook_id=f"invalid-{file_order}-{hook_order}",
        event_name=HookEvent.AGENT_TURN_END,
        layer=source.layer,
        enabled=False,
        source_path=source.path,
        file_order=file_order,
        hook_order=hook_order,
    )


def _error(message: str, source: HookSourceFile, hook_id: str) -> HookDiagnostic:
    return HookDiagnostic(
        level=HookDiagnosticLevel.ERROR,
        message=message,
        hook_id=hook_id,
        source_layer=source.layer.value,
        refs=(f"path={source.path}",),
    )


def _warning(message: str, source: HookSourceFile, hook_id: str) -> HookDiagnostic:
    return HookDiagnostic(
        level=HookDiagnosticLevel.WARNING,
        message=message,
        hook_id=hook_id,
        source_layer=source.layer.value,
        refs=(f"path={source.path}",),
    )


def _required_text(mapping: Mapping[str, object], key: str) -> str:
    return _text(mapping.get(key))


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _bool_value(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        clean = value.strip().lower()
        if clean in {"true", "yes", "1", "on"}:
            return True
        if clean in {"false", "no", "0", "off"}:
            return False
    return default


def _int_value(value: object, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _string_values(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, (list, tuple, set)):
        return tuple(_text(item) for item in value if _text(item))
    return ()


def _is_safe_relative_glob(pattern: str) -> bool:
    clean = pattern.strip().replace("\\", "/")
    parts = PurePosixPath(clean).parts
    if not clean or clean.startswith(("/", "~")) or (parts and ":" in parts[0]):
        return False
    return ".." not in parts and "~" not in parts


def _count_map(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}
