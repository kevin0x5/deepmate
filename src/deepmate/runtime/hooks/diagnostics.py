"""Hook diagnostics and CLI report formatting."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from deepmate.foundation import display_path
from deepmate.runtime.hooks.loader import HookLoadReport
from deepmate.runtime.hooks.types import (
    HOOK_LAYER_ORDER,
    HookDiagnostic,
    HookDiagnosticLevel,
)
from deepmate.storage import JsonlWriter

HOOK_DIAGNOSTICS_FILE = "diagnostics.jsonl"


class HookDiagnosticStore:
    """Append-only bounded-by-caller diagnostic sink for hook summaries."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def in_data_dir(cls, data_dir: str | Path) -> "HookDiagnosticStore":
        """Return the default hook diagnostics sink."""
        return cls(Path(data_dir) / "hooks" / HOOK_DIAGNOSTICS_FILE)

    def append(self, diagnostic: HookDiagnostic) -> None:
        """Append one sanitized diagnostic record."""
        JsonlWriter(self.path).append(diagnostic.to_record())

    def append_many(self, diagnostics: tuple[HookDiagnostic, ...]) -> None:
        """Append diagnostics in order."""
        for diagnostic in diagnostics:
            self.append(diagnostic)


def format_hooks_status(report: HookLoadReport, workspace: str | Path) -> str:
    """Return a human-readable hook load status report."""
    root = Path(workspace)
    loaded = report.loaded_counts()
    discovered = _layer_count_line(report.discovered_counts)
    skipped = _count_line(report.skipped_counts) or "none"
    lines = [
        "Hooks status:",
        f"- enabled: {str(report.options.enabled).lower()}",
        f"- managed_only: {str(report.options.managed_hooks_only).lower()}",
        f"- load_user_hooks: {str(report.options.load_user_hooks).lower()}",
        f"- load_project_hooks: {str(report.options.load_project_hooks).lower()}",
        f"- workspace_trusted: {str(report.workspace_trusted).lower()}",
        f"- trust_store: {_format_path(report.trust_path, root)}",
        f"- loaded: {_layer_count_line(loaded)}",
        f"- discovered_files: {discovered}",
        f"- skipped: {skipped}",
        f"- hook_surface_tag: {report.registry.surface_tag()}",
    ]
    if report.diagnostics:
        lines.append("- diagnostics:")
        for diagnostic in report.diagnostics[:20]:
            lines.append(f"  - {_format_diagnostic(diagnostic, root)}")
        if len(report.diagnostics) > 20:
            lines.append(f"  - ... {len(report.diagnostics) - 20} more")
    else:
        lines.append("- diagnostics: none")
    return "\n".join(lines)


def format_hook_validation(report: HookLoadReport, workspace: str | Path) -> str:
    """Return a static validation report."""
    root = Path(workspace)
    errors = tuple(
        diagnostic
        for diagnostic in report.diagnostics
        if diagnostic.level == HookDiagnosticLevel.ERROR
    )
    warnings = tuple(
        diagnostic
        for diagnostic in report.diagnostics
        if diagnostic.level == HookDiagnosticLevel.WARNING
    )
    lines = [
        "Hook validation:",
        f"- status: {'failed' if errors else 'passed'}",
        f"- enabled: {str(report.options.enabled).lower()}",
        f"- managed_only: {str(report.options.managed_hooks_only).lower()}",
        f"- loaded: {_layer_count_line(report.loaded_counts())}",
        f"- skipped: {_count_line(report.skipped_counts) or 'none'}",
        f"- errors: {len(errors)}",
        f"- warnings: {len(warnings)}",
    ]
    for diagnostic in (*errors, *warnings):
        lines.append(f"  - {_format_diagnostic(diagnostic, root)}")
    return "\n".join(lines)


def _layer_count_line(counts: dict[str, int] | object) -> str:
    if not isinstance(counts, dict):
        return " ".join(f"{layer.value}=0" for layer in HOOK_LAYER_ORDER)
    return " ".join(f"{layer.value}={int(counts.get(layer.value, 0))}" for layer in HOOK_LAYER_ORDER)


def _count_line(counts: object) -> str:
    if not isinstance(counts, dict) or not counts:
        return ""
    counter = Counter({str(key): int(value) for key, value in counts.items()})
    return " ".join(f"{key}={counter[key]}" for key in sorted(counter))


def _format_diagnostic(diagnostic: HookDiagnostic, workspace: Path) -> str:
    prefix = diagnostic.level.value
    context = []
    if diagnostic.source_layer:
        context.append(f"layer={diagnostic.source_layer}")
    if diagnostic.hook_id:
        context.append(f"hook={diagnostic.hook_id}")
    if diagnostic.event_name:
        context.append(f"event={diagnostic.event_name}")
    refs = tuple(_sanitize_ref(ref, workspace) for ref in diagnostic.refs[:3])
    suffix_parts = []
    if context:
        suffix_parts.append(", ".join(context))
    if refs:
        suffix_parts.append(", ".join(refs))
    suffix = f" ({'; '.join(suffix_parts)})" if suffix_parts else ""
    return f"{prefix}: {diagnostic.message}{suffix}"


def _sanitize_ref(ref: str, workspace: Path) -> str:
    if not ref.startswith("path="):
        return ref
    raw_path = Path(ref[len("path=") :])
    return f"path={_format_path(raw_path, workspace)}"


def _format_path(path: Path | None, workspace: Path) -> str:
    if path is None:
        return ""
    return display_path(path, workspace)
