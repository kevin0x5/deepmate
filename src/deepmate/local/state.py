"""Private state for Deepmate Local preparation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deepmate.storage import atomic_write_json, file_lock

LOCAL_STATE_VERSION = 1


@dataclass(frozen=True, slots=True)
class LocalModelPrepareState:
    """Last known local-model preparation state."""

    model_id: str
    stage: str
    message: str
    status: str = "running"
    failure_kind: str = ""
    updated_at: str = ""

    @classmethod
    def create(
        cls,
        *,
        model_id: str,
        stage: str,
        message: str,
        status: str = "running",
        failure_kind: str = "",
    ) -> "LocalModelPrepareState":
        return cls(
            model_id=model_id.strip(),
            stage=stage.strip(),
            message=message.strip(),
            status=status.strip() or "running",
            failure_kind=failure_kind.strip(),
            updated_at=_utc_now(),
        )

    @classmethod
    def from_json(cls, payload: object) -> "LocalModelPrepareState | None":
        if not isinstance(payload, dict):
            return None
        try:
            version = int(payload.get("version", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            return None
        if version != LOCAL_STATE_VERSION:
            return None
        raw = payload.get("local_model")
        if not isinstance(raw, dict):
            return None
        model_id = _string(raw.get("model_id"))
        stage = _string(raw.get("stage"))
        message = _string(raw.get("message"))
        if not model_id or not stage:
            return None
        return cls(
            model_id=model_id,
            stage=stage,
            message=message,
            status=_string(raw.get("status")) or "running",
            failure_kind=_string(raw.get("failure_kind")),
            updated_at=_string(raw.get("updated_at")),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "version": LOCAL_STATE_VERSION,
            "local_model": {
                "model_id": self.model_id,
                "stage": self.stage,
                "message": self.message,
                "status": self.status,
                "failure_kind": self.failure_kind,
                "updated_at": self.updated_at or _utc_now(),
            },
        }

    def is_ready(self) -> bool:
        """Return whether the last persisted prepare completed successfully."""
        return self.status == "ready" and self.stage == "ready"

    def is_incomplete(self) -> bool:
        """Return whether the previous prepare should be described as resumable."""
        return self.status in {"running", "failed"} or self.stage not in {"ready"}

    def user_message(self) -> str:
        """Return a friendly status line for `/local status`."""
        if self.is_ready():
            return self.message or "本地模型已就绪。"
        if self.status == "running":
            return f"上次进度 · {self.message or self.stage}。输入 /local 可以继续。"
        detail = " ".join(self.message.split())[:320]
        if self.stage == "cancelled" or self.failure_kind == "cancelled":
            return "已取消准备。输入 /local 可以继续已下载的分层。"
        if detail:
            return detail if "/local" in detail else f"{detail} 处理后输入 /local 重试。"
        return "上次本地模型没有准备完成。输入 /local 重试。"


class LocalModelStateStore:
    """Small JSON sidecar under data_dir for local-model preparation."""

    def __init__(self, data_dir: str | Path | None) -> None:
        self._path = _state_path(data_dir)

    @property
    def path(self) -> Path | None:
        return self._path

    def load(self) -> LocalModelPrepareState | None:
        if self._path is None:
            return None
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        return LocalModelPrepareState.from_json(payload)

    def save(self, state: LocalModelPrepareState) -> None:
        if self._path is None:
            return
        with file_lock(self._path):
            atomic_write_json(self._path, state.to_json())

    def record(
        self,
        *,
        model_id: str,
        stage: str,
        message: str,
        status: str = "running",
        failure_kind: str = "",
    ) -> LocalModelPrepareState:
        state = LocalModelPrepareState.create(
            model_id=model_id,
            stage=stage,
            message=message,
            status=status,
            failure_kind=failure_kind,
        )
        self.save(state)
        return state


def _state_path(data_dir: str | Path | None) -> Path | None:
    if data_dir is None:
        return None
    return Path(data_dir) / "local" / "model_state.json"


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
