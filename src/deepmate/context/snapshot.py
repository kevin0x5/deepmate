"""Frozen profile context snapshot objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from deepmate.domain import ProfileRef

if TYPE_CHECKING:
    from deepmate.context.builder import ContextWarning


@dataclass(frozen=True, slots=True)
class ContextFileRef:
    """Metadata for one context source file kept outside the prompt."""

    name: str
    path: Path
    status: str
    size_bytes: int = 0
    sha256: str = ""
    estimated_tokens: int = 0
    warning_code: str = ""

    def is_loaded(self) -> bool:
        """Return whether this file contributed prompt content."""
        return self.status == "loaded"

    def trace_refs(self) -> tuple[str, ...]:
        """Return compact trace refs without exposing file content."""
        refs = (
            f"context_file.{self.name}.status={self.status}",
            f"context_file.{self.name}.size_bytes={self.size_bytes}",
            f"context_file.{self.name}.estimated_tokens={self.estimated_tokens}",
        )
        if self.sha256:
            refs = (*refs, f"context_file.{self.name}.sha256={self.sha256[:12]}")
        if self.warning_code:
            refs = (*refs, f"context_file.{self.name}.warning={self.warning_code}")
        return refs


@dataclass(frozen=True, slots=True)
class ContextFileChange:
    """One context source file changed since a snapshot was created."""

    name: str
    path: Path
    old_status: str
    new_status: str
    old_sha256: str = ""
    new_sha256: str = ""

    def trace_refs(self) -> tuple[str, ...]:
        """Return compact trace refs without exposing file content."""
        refs = (
            f"context_file_change.{self.name}.old_status={self.old_status}",
            f"context_file_change.{self.name}.new_status={self.new_status}",
        )
        if self.old_sha256:
            refs = (*refs, f"context_file_change.{self.name}.old_sha256={self.old_sha256[:12]}")
        if self.new_sha256:
            refs = (*refs, f"context_file_change.{self.name}.new_sha256={self.new_sha256[:12]}")
        return refs


@dataclass(frozen=True, slots=True)
class ProfileContextSnapshot:
    """Workspace/profile context frozen for one activation."""

    workspace: Path
    profile: ProfileRef
    sections: tuple[str, ...]
    warnings: tuple["ContextWarning", ...] = field(default_factory=tuple)
    file_refs: tuple[ContextFileRef, ...] = field(default_factory=tuple)
    hot_profile_token_budget: int = 0
    hot_profile_warn_tokens: int = 0
    hot_profile_estimated_tokens: int = 0
    pending_refresh_reason: str = ""

    def is_ready(self) -> bool:
        """Return whether the snapshot has readable context sections."""
        return bool(self.profile.is_ready() and self.sections)

    def loaded_section_names(self) -> tuple[str, ...]:
        """Return names of context files that contributed prompt content."""
        return tuple(ref.name for ref in self.file_refs if ref.is_loaded())

    def trace_refs(self) -> tuple[str, ...]:
        """Return compact refs suitable for context trace events."""
        refs = (
            f"loaded_sections={','.join(self.loaded_section_names())}",
            f"hot_profile_estimated_tokens={self.hot_profile_estimated_tokens}",
            f"hot_profile_token_budget={self.hot_profile_token_budget}",
            f"hot_profile_warn_tokens={self.hot_profile_warn_tokens}",
        )
        if self.pending_refresh_reason:
            refs = (*refs, f"pending_refresh_reason={self.pending_refresh_reason}")
        for file_ref in self.file_refs:
            refs = (*refs, *file_ref.trace_refs())
        return refs
