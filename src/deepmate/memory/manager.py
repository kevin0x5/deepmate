"""Apply curated memory patches to profile markdown files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from deepmate.foundation import estimate_text_tokens
from deepmate.memory.extractor import MemoryExtractionResult
from deepmate.storage import atomic_write_text, file_lock

USER_MEMORY_FILE = "user.md"
LONG_TERM_MEMORY_FILE = "memory.md"

PATCH_WRITE_USER = "write_user"
PATCH_WRITE_MEMORY = "write_memory"
PATCH_WRITE_PROJECT_MEMORY = "write_project_memory"
PATCH_REPLACE = "replace"
PATCH_REMOVE = "remove"
PATCH_DEMOTE_TO_WARM = "demote_to_warm"
PATCH_SKIP = "skip"
PATCH_ACTIONS = {
    PATCH_WRITE_USER,
    PATCH_WRITE_MEMORY,
    PATCH_WRITE_PROJECT_MEMORY,
    PATCH_REPLACE,
    PATCH_REMOVE,
    PATCH_DEMOTE_TO_WARM,
    PATCH_SKIP,
}
PATCH_TARGETS = {"user", "memory", "project_memory"}


@dataclass(frozen=True, slots=True)
class MemoryPatchOperation:
    """One Curator-selected hot memory edit."""

    action: str
    target: str = ""
    content: str = ""
    replace_ref: str = ""
    reason: str = ""
    confidence: float | None = None

    def normalized(self) -> "MemoryPatchOperation":
        """Return an operation with compact strings."""
        action = self.action.strip().lower()
        target = self.target.strip().lower()
        content = _clean_bullet(self.content)
        replace_ref = _clean_bullet(self.replace_ref)
        reason = " ".join(self.reason.split()).strip()
        confidence = self.confidence
        if confidence is not None:
            try:
                confidence = max(0.0, min(1.0, float(confidence)))
            except (TypeError, ValueError):
                confidence = None
        if action == PATCH_WRITE_USER:
            target = "user"
        elif action == PATCH_WRITE_MEMORY:
            target = "memory"
        elif action == PATCH_WRITE_PROJECT_MEMORY:
            target = "project_memory"
        return MemoryPatchOperation(
            action=action,
            target=target,
            content=content,
            replace_ref=replace_ref,
            reason=reason,
            confidence=confidence,
        )


@dataclass(frozen=True, slots=True)
class MemoryPatch:
    """Curator output to be applied by the manager."""

    operations: tuple[MemoryPatchOperation, ...] = field(default_factory=tuple)

    def normalized(self) -> "MemoryPatch":
        """Return a patch with normalized operations."""
        return MemoryPatch(
            operations=tuple(operation.normalized() for operation in self.operations)
        )


@dataclass(frozen=True, slots=True)
class MemoryPatchApplyResult:
    """Summary of patch application without exposing profile contents."""

    applied_operations: tuple[MemoryPatchOperation, ...] = field(default_factory=tuple)
    skipped: tuple[str, ...] = field(default_factory=tuple)
    budget_blocked: tuple[str, ...] = field(default_factory=tuple)

    def changed(self) -> bool:
        """Return whether any profile markdown file was changed."""
        return bool(self.applied_operations)

    def summary_refs(self) -> tuple[str, ...]:
        """Return compact refs suitable for trace or CLI diagnostics."""
        return (
            f"applied={len(self.applied_operations)}",
            f"skipped={len(self.skipped)}",
            f"budget_blocked={len(self.budget_blocked)}",
        )


@dataclass(frozen=True, slots=True)
class MemoryApplyResult:
    """Compatibility summary for applying extraction results."""

    user_added: tuple[str, ...] = field(default_factory=tuple)
    memory_added: tuple[str, ...] = field(default_factory=tuple)
    user_replaced: tuple[str, ...] = field(default_factory=tuple)
    memory_replaced: tuple[str, ...] = field(default_factory=tuple)
    skipped: tuple[str, ...] = field(default_factory=tuple)
    budget_blocked: tuple[str, ...] = field(default_factory=tuple)

    def changed(self) -> bool:
        """Return whether any profile markdown file was changed."""
        return bool(
            self.user_added
            or self.memory_added
            or self.user_replaced
            or self.memory_replaced
        )

    def summary_refs(self) -> tuple[str, ...]:
        """Return compact refs suitable for trace or CLI diagnostics."""
        return (
            f"user_added={len(self.user_added)}",
            f"memory_added={len(self.memory_added)}",
            f"user_replaced={len(self.user_replaced)}",
            f"memory_replaced={len(self.memory_replaced)}",
            f"skipped={len(self.skipped)}",
            f"budget_blocked={len(self.budget_blocked)}",
        )


def memory_patch_from_extraction(extraction: MemoryExtractionResult) -> MemoryPatch:
    """Convert extractor output to a Curator-style patch."""
    operations: list[MemoryPatchOperation] = []
    operations.extend(
        MemoryPatchOperation(action=PATCH_WRITE_USER, content=fact.content)
        for fact in extraction.user_facts
        if fact.is_ready()
    )
    operations.extend(
        MemoryPatchOperation(action=PATCH_WRITE_MEMORY, content=fact.content)
        for fact in extraction.memory_facts
        if fact.is_ready()
    )
    operations.extend(
        MemoryPatchOperation(
            action=PATCH_DEMOTE_TO_WARM,
            content=note.content,
            reason=note.reason,
        )
        for note in extraction.session_only
        if note.is_ready()
    )
    operations.extend(
        MemoryPatchOperation(action=PATCH_SKIP, content=note.content, reason=note.reason)
        for note in extraction.rejected
        if note.is_ready()
    )
    return MemoryPatch(operations=tuple(operations))


def apply_memory_extraction(
    profile_dir: str | Path,
    extraction: MemoryExtractionResult,
    hot_profile_token_budget: int = 0,
    project_profile_dir: str | Path | None = None,
) -> MemoryApplyResult:
    """Write extracted long-term candidates through the patch applier."""
    patch_result = apply_memory_patch(
        profile_dir,
        memory_patch_from_extraction(extraction),
        hot_profile_token_budget=hot_profile_token_budget,
        project_profile_dir=project_profile_dir,
    )
    user_added = tuple(
        operation.content
        for operation in patch_result.applied_operations
        if operation.action == PATCH_WRITE_USER
    )
    memory_added = tuple(
        operation.content
        for operation in patch_result.applied_operations
        if operation.action == PATCH_WRITE_MEMORY
    )
    user_replaced = tuple(
        operation.content
        for operation in patch_result.applied_operations
        if operation.action == PATCH_REPLACE and operation.target == "user"
    )
    memory_replaced = tuple(
        operation.content
        for operation in patch_result.applied_operations
        if operation.action == PATCH_REPLACE and operation.target == "memory"
    )
    return MemoryApplyResult(
        user_added=user_added,
        memory_added=memory_added,
        user_replaced=user_replaced,
        memory_replaced=memory_replaced,
        skipped=patch_result.skipped,
        budget_blocked=patch_result.budget_blocked,
    )


def apply_memory_patch(
    profile_dir: str | Path,
    patch: MemoryPatch,
    hot_profile_token_budget: int = 0,
    project_profile_dir: str | Path | None = None,
) -> MemoryPatchApplyResult:
    """Apply a Curator patch with schema, budget, and atomic-write guards."""
    profile_path = Path(profile_dir)
    project_profile_path = (
        Path(project_profile_dir) if project_profile_dir is not None else None
    )
    lock_paths = [profile_path / "profile_memory.md"]
    if project_profile_path is not None and project_profile_path != profile_path:
        lock_paths.append(project_profile_path / "profile_memory.md")
    sorted_lock_paths = tuple(
        sorted(
            {path.resolve(strict=False) for path in lock_paths},
            key=lambda path: str(path),
        )
    )
    return _apply_memory_patch_with_locks(
        sorted_lock_paths,
        profile_path,
        patch,
        hot_profile_token_budget,
        project_profile_path,
    )


def _apply_memory_patch_with_locks(
    lock_paths: tuple[Path, ...],
    profile_path: Path,
    patch: MemoryPatch,
    hot_profile_token_budget: int,
    project_profile_path: Path | None,
) -> MemoryPatchApplyResult:
    if not lock_paths:
        return _apply_memory_patch_unlocked(
            profile_path,
            patch,
            hot_profile_token_budget,
            project_profile_path,
        )
    first, *rest = lock_paths
    with file_lock(first):
        return _apply_memory_patch_with_locks(
            tuple(rest),
            profile_path,
            patch,
            hot_profile_token_budget,
            project_profile_path,
        )


def _apply_memory_patch_unlocked(
    profile_path: Path,
    patch: MemoryPatch,
    hot_profile_token_budget: int = 0,
    project_profile_path: Path | None = None,
) -> MemoryPatchApplyResult:
    project_path = project_profile_path or profile_path
    documents = {
        "user": _read_document(profile_path / USER_MEMORY_FILE),
        "memory": _read_document(profile_path / LONG_TERM_MEMORY_FILE),
    }
    if project_path != profile_path:
        documents["project_memory"] = _read_document(project_path / LONG_TERM_MEMORY_FILE)
    lines = {target: list(document.bullets) for target, document in documents.items()}
    applied: list[MemoryPatchOperation] = []
    skipped: list[str] = []
    patch_writes: set[tuple[str, str]] = set()

    for raw_operation in patch.normalized().operations:
        operation = raw_operation.normalized()
        reason = _operation_schema_error(operation)
        if reason:
            skipped.append(reason)
            continue
        if operation.action in {PATCH_SKIP, PATCH_DEMOTE_TO_WARM}:
            skipped.append(operation.action)
            continue
        target_lines = lines[operation.target]
        if operation.action in {
            PATCH_WRITE_USER,
            PATCH_WRITE_MEMORY,
            PATCH_WRITE_PROJECT_MEMORY,
        }:
            write_key = (operation.target, operation.content)
            if write_key in patch_writes:
                skipped.append("duplicate_content")
                continue
            patch_writes.add(write_key)
            target_lines.append(operation.content)
            applied.append(operation)
            continue
        if operation.action == PATCH_REPLACE:
            index = _exact_bullet_index(target_lines, operation.replace_ref)
            if index is None:
                skipped.append("replace_ref_not_found")
                continue
            target_lines[index] = operation.content
            applied.append(operation)
            continue
        if operation.action == PATCH_REMOVE:
            index = _exact_bullet_index(target_lines, operation.replace_ref)
            if index is None:
                skipped.append("remove_ref_not_found")
                continue
            del target_lines[index]
            applied.append(operation)

    if not applied:
        return MemoryPatchApplyResult(skipped=tuple(skipped))

    estimated_tokens = _estimate_hot_profile_tokens(lines)
    if hot_profile_token_budget > 0 and estimated_tokens > hot_profile_token_budget:
        return MemoryPatchApplyResult(
            skipped=tuple(skipped),
            budget_blocked=(
                f"hot_profile_budget_exceeded:{estimated_tokens}/{hot_profile_token_budget}",
            ),
        )

    written_paths: set[Path] = set()
    for target in ("user", "memory", "project_memory"):
        document = documents.get(target)
        if document is None:
            continue
        if document.path in written_paths:
            continue
        updated = tuple(lines[target])
        if updated != document.bullets:
            _write_document(document.path, document, updated)
        written_paths.add(document.path)

    return MemoryPatchApplyResult(
        applied_operations=tuple(applied),
        skipped=tuple(skipped),
    )


@dataclass(frozen=True, slots=True)
class _MemoryDocument:
    path: Path
    raw_lines: tuple[str, ...]
    bullets: tuple[str, ...]
    bullet_indexes: tuple[int, ...]


def _operation_schema_error(operation: MemoryPatchOperation) -> str:
    if operation.action not in PATCH_ACTIONS:
        return "invalid_action"
    if operation.action in {
        PATCH_WRITE_USER,
        PATCH_WRITE_MEMORY,
        PATCH_WRITE_PROJECT_MEMORY,
    } and not operation.content:
        return "empty_content"
    if operation.action in {PATCH_REPLACE, PATCH_REMOVE}:
        if operation.target not in PATCH_TARGETS:
            return "invalid_target"
        if not operation.replace_ref:
            return "missing_replace_ref"
        if operation.action == PATCH_REPLACE and not operation.content:
            return "empty_content"
    return ""


def _read_document(path: Path) -> _MemoryDocument:
    if not path.exists():
        return _MemoryDocument(path=path, raw_lines=(), bullets=(), bullet_indexes=())
    raw_lines = tuple(path.read_text(encoding="utf-8").splitlines())
    lines: list[str] = []
    indexes: list[int] = []
    for index, raw_line in enumerate(raw_lines):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("- "):
            content = line[2:].strip()
            if content:
                lines.append(content)
                indexes.append(index)
    return _MemoryDocument(
        path=path,
        raw_lines=raw_lines,
        bullets=tuple(line for line in lines if line),
        bullet_indexes=tuple(indexes),
    )


def _write_document(path: Path, document: _MemoryDocument, bullets: tuple[str, ...]) -> None:
    if not document.raw_lines:
        _write_bullets(path, bullets)
        return

    output: list[str] = []
    bullet_indexes = set(document.bullet_indexes)
    bullet_index = 0
    for index, raw_line in enumerate(document.raw_lines):
        if index not in bullet_indexes:
            output.append(raw_line)
            continue
        if bullet_index < len(bullets):
            output.append(f"- {bullets[bullet_index]}")
            bullet_index += 1
    new_bullets = bullets[bullet_index:]
    if new_bullets:
        if output and output[-1].strip():
            output.append("")
        output.extend(f"- {line}" for line in new_bullets)
    atomic_write_text(path, "\n".join(output).rstrip() + "\n")


def _write_bullets(path: Path, bullets: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, _render_bullets(bullets))


def _render_bullets(bullets: tuple[str, ...]) -> str:
    if not bullets:
        return ""
    return "\n".join(f"- {line}" for line in bullets).rstrip() + "\n"


def _clean_bullet(content: str) -> str:
    line = " ".join(str(content).split()).strip()
    if line.startswith("- "):
        line = line[2:].strip()
    return line


def _exact_bullet_index(lines: list[str], value: str) -> int | None:
    for index, line in enumerate(lines):
        if line == value:
            return index
    return None


def _estimate_hot_profile_tokens(lines: dict[str, list[str]]) -> int:
    text = "\n".join(
        f"- {line}"
        for target in ("user", "memory", "project_memory")
        for line in lines.get(target, [])
        if line.strip()
    )
    return estimate_text_tokens(text)
