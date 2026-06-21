"""Runtime activation context for one session opening."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from deepmate.context import ProfileContextSnapshot, build_profile_context_snapshot
from deepmate.domain import ProfileRef


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class RuntimeActivation:
    """Runtime context for one process/opening over a session."""

    activation_id: str
    session_id: str
    started_at: str
    workspace: Path
    profile: ProfileRef
    context_snapshot: ProfileContextSnapshot
    context_epoch: int = 1

    def is_ready(self) -> bool:
        """Return whether the activation has usable runtime context."""
        return bool(
            self.activation_id.strip()
            and self.session_id.strip()
            and self.started_at.strip()
            and str(self.workspace).strip()
            and self.profile.is_ready()
            and self.context_snapshot.is_ready()
            and self.context_epoch > 0
        )

    def trace_refs(self) -> tuple[str, ...]:
        """Return compact refs suitable for trace events."""
        return (
            f"activation_id={self.activation_id}",
            f"session_id={self.session_id}",
            f"context_epoch={self.context_epoch}",
        )

    def refresh_context(
        self,
        context_snapshot: ProfileContextSnapshot | None = None,
        pending_refresh_reason: str = "",
    ) -> "RuntimeActivation":
        """Return a new activation with a refreshed profile context snapshot."""
        snapshot = context_snapshot or build_profile_context_snapshot(
            workspace=self.workspace,
            profile=self.profile,
            hot_profile_token_budget=(
                self.context_snapshot.hot_profile_token_budget
            ),
            hot_profile_warn_tokens=self.context_snapshot.hot_profile_warn_tokens,
            pending_refresh_reason=pending_refresh_reason,
        )
        refreshed = replace(
            self,
            context_snapshot=snapshot,
            context_epoch=self.context_epoch + 1,
        )
        if not refreshed.is_ready():
            raise ValueError("refreshed runtime activation is not ready")
        return refreshed


def start_runtime_activation(
    session_id: str,
    workspace: str | Path,
    profile: ProfileRef,
    context_snapshot: ProfileContextSnapshot | None = None,
    context_epoch: int = 1,
) -> RuntimeActivation:
    """Create a runtime activation and freeze profile context if needed."""
    clean_session_id = session_id.strip()
    if not clean_session_id:
        raise ValueError("session_id is required")
    if context_epoch < 1:
        raise ValueError("context_epoch must be at least 1")
    if isinstance(workspace, str) and not workspace.strip():
        raise ValueError("workspace is required")
    workspace_path = Path(workspace)
    snapshot = context_snapshot or build_profile_context_snapshot(
        workspace=workspace_path,
        profile=profile,
    )
    activation = RuntimeActivation(
        activation_id=uuid4().hex,
        session_id=clean_session_id,
        started_at=_utc_now_iso(),
        workspace=workspace_path,
        profile=profile,
        context_snapshot=snapshot,
        context_epoch=context_epoch,
    )
    if not activation.is_ready():
        raise ValueError("runtime activation is not ready")
    return activation


def refresh_runtime_activation_context(
    activation: RuntimeActivation,
    context_snapshot: ProfileContextSnapshot | None = None,
    pending_refresh_reason: str = "",
) -> RuntimeActivation:
    """Return an activation whose profile context snapshot has been refreshed."""
    return activation.refresh_context(
        context_snapshot=context_snapshot,
        pending_refresh_reason=pending_refresh_reason,
    )
