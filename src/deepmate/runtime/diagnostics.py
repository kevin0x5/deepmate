"""Deterministic post-edit diagnostics for workspace writes."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:  # pragma: no cover - Python 3.11+ stdlib path.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older interpreters.
    tomllib = None  # type: ignore[assignment]

from deepmate.domain import RuntimeEvent
from deepmate.providers import ModelToolResult
from deepmate.runtime.process_env import subprocess_environment

WRITE_TOOL_NAMES = frozenset({"write_text_file", "edit_text_file"})
DEFAULT_DIAGNOSTIC_TIMEOUT_SECONDS = 5.0
MAX_DIAGNOSTIC_OUTPUT_CHARS = 8_000


@dataclass(frozen=True, slots=True)
class PostEditDiagnostic:
    """One file-local diagnostic result after a write."""

    path: str
    kind: str
    status: str
    message: str = ""
    refs: tuple[str, ...] = field(default_factory=tuple)

    def failed(self) -> bool:
        """Return whether this diagnostic found a concrete issue."""
        return self.status == "failed"

    def skipped(self) -> bool:
        """Return whether this file has no deterministic local checker."""
        return self.status == "skipped"


@dataclass(frozen=True, slots=True)
class PostEditDiagnosticReport:
    """Aggregated deterministic diagnostics for one agent step."""

    diagnostics: tuple[PostEditDiagnostic, ...] = field(default_factory=tuple)

    def has_failures(self) -> bool:
        """Return whether any checker failed."""
        return any(diagnostic.failed() for diagnostic in self.diagnostics)

    def checked_count(self) -> int:
        """Return the number of concrete checks that ran."""
        return sum(1 for diagnostic in self.diagnostics if not diagnostic.skipped())

    def failed_count(self) -> int:
        """Return the number of failed checks."""
        return sum(1 for diagnostic in self.diagnostics if diagnostic.failed())

    def skipped_count(self) -> int:
        """Return the number of skipped paths."""
        return sum(1 for diagnostic in self.diagnostics if diagnostic.skipped())

    def trace_refs(self) -> tuple[str, ...]:
        """Return compact refs for runtime trace/status surfaces."""
        return (
            f"post_edit_diagnostics={len(self.diagnostics)}",
            f"checked={self.checked_count()}",
            f"failed={self.failed_count()}",
            f"skipped={self.skipped_count()}",
            *tuple(
                f"{diagnostic.kind}:{diagnostic.status}:{diagnostic.path}"
                for diagnostic in self.diagnostics[:8]
            ),
        )

    def failure_content(self) -> str:
        """Return model-visible repair guidance for failed diagnostics."""
        failures = tuple(diagnostic for diagnostic in self.diagnostics if diagnostic.failed())
        if not failures:
            return ""
        lines = [
            "Post-edit diagnostics found issues in files changed by the previous tool call.",
            "Fix these issues before delivering the final answer.",
        ]
        for diagnostic in failures:
            lines.append("")
            lines.append(
                f"- {diagnostic.path} ({diagnostic.kind}): {diagnostic.message.strip()}"
            )
        return "\n".join(lines).strip()


def post_edit_diagnostics(
    workspace: str | Path,
    tool_results: tuple[ModelToolResult, ...],
    *,
    timeout_seconds: float = DEFAULT_DIAGNOSTIC_TIMEOUT_SECONDS,
) -> PostEditDiagnosticReport:
    """Run deterministic file-local checks for successful workspace writes."""
    root = Path(workspace).resolve()
    paths = _changed_paths(root, tool_results)
    diagnostics = tuple(
        _diagnose_path(root, relative_path, timeout_seconds=timeout_seconds)
        for relative_path in paths
    )
    return PostEditDiagnosticReport(diagnostics=diagnostics)


def post_edit_diagnostic_events(
    report: PostEditDiagnosticReport,
) -> tuple[RuntimeEvent, ...]:
    """Return traceable runtime events for a diagnostics report."""
    if not report.diagnostics:
        return ()
    kind = (
        "post_edit_diagnostics_failed"
        if report.has_failures()
        else "post_edit_diagnostics_passed"
    )
    summary = (
        "Post-edit diagnostics found issues."
        if report.has_failures()
        else "Post-edit diagnostics completed."
    )
    return (
        RuntimeEvent(
            kind=kind,
            summary=summary,
            refs=report.trace_refs(),
        ),
    )


def apply_post_edit_diagnostics(
    result: ModelToolResult,
    report: PostEditDiagnosticReport,
) -> ModelToolResult:
    """Attach diagnostic failure details to the write result that triggered them."""
    content = report.failure_content()
    if not content:
        return result
    refs = _merge_refs(
        result.refs,
        (
            "post_edit_diagnostics=failed",
            f"post_edit_diagnostics_failed={report.failed_count()}",
        ),
    )
    data = dict(result.data)
    data["post_edit_diagnostics"] = [
        {
            "path": diagnostic.path,
            "kind": diagnostic.kind,
            "status": diagnostic.status,
            "message": diagnostic.message,
            "refs": list(diagnostic.refs),
        }
        for diagnostic in report.diagnostics
    ]
    return ModelToolResult(
        name=result.name,
        request_id=result.request_id,
        content="\n\n".join(part for part in (result.content, content) if part.strip()),
        data=data,
        refs=refs,
        is_error=True,
    )


def _changed_paths(
    root: Path,
    tool_results: tuple[ModelToolResult, ...],
) -> tuple[str, ...]:
    paths: list[str] = []
    seen: set[str] = set()
    for result in tool_results:
        if result.is_error or result.name not in WRITE_TOOL_NAMES:
            continue
        path_value = result.data.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            continue
        path = _safe_relative_path(root, path_value)
        if path is None or path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return tuple(paths)


def _safe_relative_path(root: Path, value: str) -> str | None:
    raw = value.strip()
    candidate = Path(raw)
    path = candidate if candidate.is_absolute() else root / candidate
    try:
        resolved = path.resolve()
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    if resolved == root:
        return None
    return resolved.relative_to(root).as_posix()


def _diagnose_path(
    root: Path,
    relative_path: str,
    *,
    timeout_seconds: float,
) -> PostEditDiagnostic:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return PostEditDiagnostic(
            path=relative_path,
            kind="path",
            status="failed",
            message="Path escaped the workspace root.",
        )
    if not path.is_file():
        return PostEditDiagnostic(
            path=relative_path,
            kind="file",
            status="failed",
            message="Changed path is not a file after write.",
        )
    suffix = path.suffix.lower()
    if suffix == ".py":
        return _diagnose_python(path, relative_path, timeout_seconds=timeout_seconds)
    if suffix == ".json":
        return _diagnose_json(path, relative_path)
    if suffix == ".toml":
        return _diagnose_toml(path, relative_path)
    return PostEditDiagnostic(
        path=relative_path,
        kind="unsupported",
        status="skipped",
        message="No deterministic file-local checker is available.",
    )


def _diagnose_python(
    path: Path,
    relative_path: str,
    *,
    timeout_seconds: float,
) -> PostEditDiagnostic:
    try:
        completed = subprocess.run(
            [
                sys.executable or "python3",
                "-c",
                (
                    "import py_compile, sys, tempfile\n"
                    "with tempfile.NamedTemporaryFile(suffix='.pyc') as file:\n"
                    "    py_compile.compile(sys.argv[1], cfile=file.name, doraise=True)\n"
                ),
                str(path),
            ],
            cwd=str(path.parent),
            env=subprocess_environment(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=max(0.1, timeout_seconds),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return PostEditDiagnostic(
            path=relative_path,
            kind="python_compile",
            status="failed",
            message=f"py_compile timed out after {timeout_seconds:g}s.",
            refs=("checker=py_compile", "timeout=true"),
        )
    except OSError as exc:
        return PostEditDiagnostic(
            path=relative_path,
            kind="python_compile",
            status="failed",
            message=f"Could not run py_compile: {exc}",
            refs=("checker=py_compile",),
        )
    output = _bounded_output(completed.stderr or completed.stdout)
    if completed.returncode != 0:
        return PostEditDiagnostic(
            path=relative_path,
            kind="python_compile",
            status="failed",
            message=output or f"py_compile exited with code {completed.returncode}.",
            refs=("checker=py_compile", f"returncode={completed.returncode}"),
        )
    return PostEditDiagnostic(
        path=relative_path,
        kind="python_compile",
        status="passed",
        refs=("checker=py_compile",),
    )


def _diagnose_json(path: Path, relative_path: str) -> PostEditDiagnostic:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return PostEditDiagnostic(
            path=relative_path,
            kind="json_parse",
            status="failed",
            message=str(exc),
            refs=("checker=json",),
        )
    return PostEditDiagnostic(
        path=relative_path,
        kind="json_parse",
        status="passed",
        refs=("checker=json",),
    )


def _diagnose_toml(path: Path, relative_path: str) -> PostEditDiagnostic:
    if tomllib is None:
        return PostEditDiagnostic(
            path=relative_path,
            kind="toml_parse",
            status="skipped",
            message="tomllib is not available in this Python runtime.",
        )
    try:
        tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return PostEditDiagnostic(
            path=relative_path,
            kind="toml_parse",
            status="failed",
            message=str(exc),
            refs=("checker=toml",),
        )
    return PostEditDiagnostic(
        path=relative_path,
        kind="toml_parse",
        status="passed",
        refs=("checker=toml",),
    )


def _bounded_output(value: str) -> str:
    text = value.strip()
    if len(text) <= MAX_DIAGNOSTIC_OUTPUT_CHARS:
        return text
    return (
        text[:MAX_DIAGNOSTIC_OUTPUT_CHARS]
        + f"\n...[truncated {len(text) - MAX_DIAGNOSTIC_OUTPUT_CHARS} chars]"
    )


def _merge_refs(existing: tuple[str, ...], extra: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    refs: list[str] = []
    for ref in (*existing, *extra):
        text = ref.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        refs.append(text)
    return tuple(refs)
