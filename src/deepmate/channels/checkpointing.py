"""Channel helpers for session checkpoint wiring."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from deepmate.domain import Message, MessageRole
from deepmate.providers import ModelConversationItem
from deepmate.storage import (
    RESUME_HINT_AFTER_TOOL,
    RESUME_HINT_FAILED,
    RESUME_HINT_INTERRUPTED,
    RESUME_HINT_MAX_STEPS,
    RESUME_HINT_NO_RESPONSE,
    SessionSummaryRecord,
    TranscriptStore,
    TurnCheckpointRecord,
    TurnCheckpointStore,
    WorkspaceCheckpointStore,
)

if TYPE_CHECKING:
    from deepmate.runtime import UserTurnResult


class SessionCheckpointController:
    """Coordinate turn and workspace checkpoints for one active session."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        turn_store: TurnCheckpointStore,
        workspace_store: WorkspaceCheckpointStore,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.turn_store = turn_store
        self.workspace_store = workspace_store
        self._current_turn_id = ""

    @classmethod
    def in_data_dir(
        cls,
        data_dir: str | Path,
        *,
        workspace: str | Path,
        profile: str,
        session_id: str,
    ) -> "SessionCheckpointController":
        """Create a checkpoint controller rooted at Deepmate's data dir."""
        return cls(
            workspace=workspace,
            turn_store=TurnCheckpointStore.in_data_dir(data_dir, profile, session_id),
            workspace_store=WorkspaceCheckpointStore.in_data_dir(
                data_dir,
                profile,
                session_id,
            ),
        )

    def start_turn(self, summary: SessionSummaryRecord | None = None) -> "TurnScope":
        """Start a checkpointed user turn scope."""
        record = self.turn_store.start_turn(
            summary_id=summary.summary_id if summary is not None else ""
        )
        self._current_turn_id = record.turn_id
        return TurnScope(controller=self, turn_id=record.turn_id)

    def capture_workspace_write(
        self,
        operation: str,
        path: Path,
        after_content: str,
    ) -> None:
        """Capture a file preimage for the active turn, if any."""
        if not self._current_turn_id:
            return
        record = self.workspace_store.capture_file(
            turn_id=self._current_turn_id,
            operation=operation,
            workspace=self.workspace,
            path=path,
            after_content=after_content,
        )
        self.turn_store.attach_workspace_checkpoint(
            self._current_turn_id,
            record.workspace_checkpoint_id,
        )

    def capture_workspace_write_for_turn(
        self,
        turn_id: str,
        operation: str,
        path: Path,
        after_content: str,
    ) -> None:
        """Capture a file preimage for an explicit turn id."""
        clean_turn_id = turn_id.strip()
        if not clean_turn_id:
            return
        record = self.workspace_store.capture_file(
            turn_id=clean_turn_id,
            operation=operation,
            workspace=self.workspace,
            path=path,
            after_content=after_content,
        )
        self.turn_store.attach_workspace_checkpoint(
            clean_turn_id,
            record.workspace_checkpoint_id,
        )

    def clear_current_turn(self, turn_id: str) -> None:
        """Clear the active turn marker if it matches the provided turn."""
        if self._current_turn_id == turn_id:
            self._current_turn_id = ""


class SessionCheckpointWriteRouter:
    """Route workspace write checkpoints to the current session controller."""

    def __init__(
        self,
        controller: SessionCheckpointController | None = None,
    ) -> None:
        self._controller = controller
        self._local = threading.local()

    def set_controller(
        self,
        controller: SessionCheckpointController | None,
    ) -> None:
        """Update the active checkpoint controller."""
        self._controller = controller

    def set_thread_controller(
        self,
        controller: SessionCheckpointController | None,
    ) -> None:
        """Update the checkpoint controller for the current worker thread."""
        self._local.controller = controller

    def clear_thread_controller(self) -> None:
        """Clear the checkpoint controller override for the current worker thread."""
        try:
            del self._local.controller
        except AttributeError:
            pass

    def set_thread_turn_id(self, turn_id: str) -> None:
        """Route current-thread writes to an explicit checkpoint turn id."""
        self._local.turn_id = turn_id.strip()

    def clear_thread_turn_id(self) -> None:
        """Clear the current-thread explicit checkpoint turn id."""
        try:
            del self._local.turn_id
        except AttributeError:
            pass

    def capture_workspace_write(
        self,
        operation: str,
        path: Path,
        after_content: str,
    ) -> None:
        """Capture a workspace write through the active controller, if any."""
        controller = getattr(self._local, "controller", self._controller)
        if controller is None:
            return
        turn_id = getattr(self._local, "turn_id", "")
        if turn_id:
            controller.capture_workspace_write_for_turn(
                turn_id,
                operation,
                path,
                after_content,
            )
            return
        controller.capture_workspace_write(operation, path, after_content)


@dataclass(slots=True)
class TurnScope:
    """One active user turn checkpoint scope."""

    controller: SessionCheckpointController
    turn_id: str

    def history_sink(self, transcript: TranscriptStore):
        """Return a history sink that also updates turn checkpoint state."""

        def append(item: ModelConversationItem) -> None:
            record = transcript.append_item(item)
            if record is not None:
                self.controller.turn_store.record_transcript_item(self.turn_id, record)

        return append

    def mark_result(self, result: "UserTurnResult") -> None:
        """Mark a completed runtime result."""
        note = result.continuation_note()
        if note:
            self.controller.turn_store.attach_continuation_note(self.turn_id, note)
        if result.reached_max_steps:
            self.controller.turn_store.max_steps_turn(self.turn_id)
            return
        errors = result.errors()
        if errors:
            self.controller.turn_store.fail_turn(self.turn_id, errors[0].code)
            return
        self.controller.turn_store.complete_turn(self.turn_id)

    def mark_failed(self, error_code: str) -> None:
        """Mark the scope failed."""
        self.controller.turn_store.fail_turn(self.turn_id, error_code)

    def mark_interrupted(self, error_code: str = "interrupted") -> None:
        """Mark the scope interrupted."""
        self.controller.turn_store.interrupt_turn(self.turn_id, error_code)

    def attach_summary(self, summary: SessionSummaryRecord | None) -> None:
        """Attach a latest summary id after maintenance, if present."""
        if summary is not None:
            self.controller.turn_store.attach_summary(self.turn_id, summary.summary_id)

    def close(self) -> None:
        """Clear active turn state."""
        self.controller.clear_current_turn(self.turn_id)


def checkpoint_resume_context_item(
    checkpoint: TurnCheckpointRecord | None,
) -> ModelConversationItem | None:
    """Return synthetic context describing an unfinished previous turn."""
    if checkpoint is None or checkpoint.resume_hint == "normal":
        return None
    if checkpoint.resume_hint not in {
        RESUME_HINT_NO_RESPONSE,
        RESUME_HINT_AFTER_TOOL,
        RESUME_HINT_MAX_STEPS,
        RESUME_HINT_FAILED,
        RESUME_HINT_INTERRUPTED,
    }:
        return None
    return ModelConversationItem.from_message(
        Message(
            role=MessageRole.ASSISTANT,
            content=_resume_context_text(checkpoint),
        )
    )


def _resume_context_text(checkpoint: TurnCheckpointRecord) -> str:
    base = (
        "The previous Deepmate user turn did not finish normally. "
        "This is recovery context, not a new user request."
    )
    if checkpoint.resume_hint == RESUME_HINT_NO_RESPONSE:
        detail = (
            "The last user message was recorded, but no final assistant response "
            "was recorded. Continue by answering the user's last visible request."
        )
    elif checkpoint.resume_hint == RESUME_HINT_AFTER_TOOL:
        detail = (
            "A tool exchange was recorded, but the final assistant response was not. "
            "Continue from the existing tool result instead of repeating the tool "
            "unless it is clearly necessary."
        )
    elif checkpoint.resume_hint == RESUME_HINT_MAX_STEPS:
        detail = (
            "The last turn reached max_steps before a final answer. First summarize "
            "what is already done, then continue with the next useful step."
        )
    elif checkpoint.resume_hint == RESUME_HINT_INTERRUPTED:
        detail = (
            "The last turn was interrupted. Re-check the visible transcript and "
            "workspace state before retrying side-effectful work."
        )
    else:
        detail = (
            "The last turn failed. Re-check the visible transcript and avoid assuming "
            "side effects completed unless there is evidence."
        )
    refs = (
        f"turn_id={checkpoint.turn_id}; "
        f"status={checkpoint.status}; "
        f"resume_hint={checkpoint.resume_hint}; "
        f"last_transcript_sequence={checkpoint.last_transcript_sequence}; "
        f"last_tool_exchange_sequence={checkpoint.last_tool_exchange_sequence}; "
        f"final_assistant_sequence={checkpoint.final_assistant_sequence}"
    )
    note = checkpoint.continuation_note.strip()
    if note:
        detail = f"{detail}\n\nContinuation note:\n{note}"
    return f"{base}\n\n{detail}\n\nCheckpoint refs: {refs}"
