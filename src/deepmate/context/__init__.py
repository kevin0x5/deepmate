"""Context assembly helpers for Deepmate."""

from deepmate.context.builder import (
    ContextBuildResult,
    ContextWarning,
    build_profile_context_snapshot,
    build_system_context,
    build_system_context_from_snapshot,
    detect_behavior_context_changes,
)
from deepmate.context.snapshot import (
    ContextFileChange,
    ContextFileRef,
    ProfileContextSnapshot,
)

__all__ = [
    "ContextFileChange",
    "ContextFileRef",
    "ContextBuildResult",
    "ContextWarning",
    "ProfileContextSnapshot",
    "build_profile_context_snapshot",
    "build_system_context",
    "build_system_context_from_snapshot",
    "detect_behavior_context_changes",
]
