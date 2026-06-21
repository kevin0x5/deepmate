"""Session-scoped subagent result records and on-demand retrieval."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deepmate.providers import ModelToolResult
from deepmate.storage.atomic import atomic_write_json, file_lock
from deepmate.subagents.types import SubagentRunRequest, SubagentRunResult
from deepmate.subagents.verification import SubagentResultReview

if TYPE_CHECKING:
    from deepmate.runtime.agent_loop import UserTurnResult

SUBAGENT_RESULT_REF_PREFIX = "subagent_"
_REF_RE = re.compile(r"^subagent_[A-Za-z0-9_.-]{1,96}$")
MAX_TEXT_CHARS = 8_000
MAX_ARGUMENT_TEXT_CHARS = 2_000
MAX_TOOL_RESULTS_PER_STEP = 12
SUBAGENT_RESULT_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class SubagentResultRecord:
    """Durable, session-scoped details for one child run."""

    ref: str
    parent_session_id: str
    created_at: str
    request: dict[str, object]
    result: dict[str, object]
    review: dict[str, object] | None = None
    steps: tuple[dict[str, object], ...] = field(default_factory=tuple)
    schema_version: int = SUBAGENT_RESULT_SCHEMA_VERSION


class SubagentResultStore:
    """Persist subagent run details under one parent session boundary."""

    def __init__(self, root: str | Path, parent_session_id: str) -> None:
        clean_session_id = _clean_segment(parent_session_id, fallback="unknown-session")
        self._root = Path(root).resolve()
        self._parent_session_id = clean_session_id
        self._session_dir = (
            self._root / "subagents" / clean_session_id
        ).resolve()
        if not _is_relative_to(self._session_dir, self._root):
            raise ValueError("subagent result store path escaped data root")

    @classmethod
    def in_data_dir(
        cls,
        data_dir: str | Path,
        parent_session_id: str,
    ) -> "SubagentResultStore":
        """Create a store rooted at Deepmate's runtime data directory."""
        return cls(data_dir, parent_session_id=parent_session_id)

    @property
    def parent_session_id(self) -> str:
        """Return the parent session boundary."""
        return self._parent_session_id

    def save(
        self,
        *,
        request: SubagentRunRequest,
        result: SubagentRunResult,
        turn: "UserTurnResult | None",
        review: SubagentResultReview | None = None,
    ) -> SubagentResultRecord:
        """Persist one subagent run record and return its read handle."""
        with file_lock(self._session_dir / ".result-store"):
            ref = self._unique_ref_for_run(result.run_id)
            record = SubagentResultRecord(
                ref=ref,
                parent_session_id=self._parent_session_id,
                created_at=datetime.now(UTC).isoformat(),
                request=_request_payload(request),
                result=_result_payload(result),
                review=review.to_payload() if review is not None else None,
                steps=_steps_payload(turn) if turn is not None else (),
            )
            path = self._path_for_ref(ref)
            atomic_write_json(path, _record_payload(record))
            try:
                path.chmod(0o600)
            except OSError:
                pass
            return record

    def load(self, ref: str) -> SubagentResultRecord | None:
        """Load a subagent result record by session-scoped ref."""
        clean_ref = ref.strip()
        if not _REF_RE.match(clean_ref):
            return None
        path = self._path_for_ref(clean_ref)
        with file_lock(self._session_dir / ".result-store"):
            if not path.exists():
                return None
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
        if not isinstance(payload, dict):
            return None
        record = _record_from_payload(payload)
        if record.parent_session_id != self._parent_session_id:
            return None
        return record

    def _path_for_ref(self, ref: str) -> Path:
        if not _REF_RE.match(ref):
            raise ValueError(f"invalid subagent result ref: {ref}")
        path = (self._session_dir / f"{ref}.json").resolve()
        if not _is_relative_to(path, self._session_dir):
            raise ValueError("subagent result ref path escaped session directory")
        return path

    def _unique_ref_for_run(self, run_id: str) -> str:
        ref = _ref_for_run(run_id)
        if not self._path_for_ref(ref).exists():
            return ref
        base = ref[:90]
        for index in range(2, 10_000):
            candidate = f"{base}-{index}"
            if not self._path_for_ref(candidate).exists():
                return candidate
        suffix = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")[-12:]
        return f"{SUBAGENT_RESULT_REF_PREFIX}{suffix}"


def read_subagent_record_payload(
    record: SubagentResultRecord,
    *,
    include_steps: bool = True,
    step_index: int | None = None,
) -> dict[str, object]:
    """Return a model-visible payload for one persisted subagent record."""
    payload: dict[str, object] = {
        "result_ref": record.ref,
        "schema_version": record.schema_version,
        "created_at": record.created_at,
        "request": record.request,
        "result": record.result,
    }
    if record.review is not None:
        payload["review"] = record.review
    if include_steps:
        steps = record.steps
        if step_index is not None:
            steps = tuple(
                step
                for step in record.steps
                if int(step.get("step_index", 0)) == step_index
            )
        payload["steps"] = list(steps)
    else:
        payload["step_count"] = len(record.steps)
    return payload


def _ref_for_run(run_id: str) -> str:
    clean = _clean_segment(run_id, fallback="run")
    return f"{SUBAGENT_RESULT_REF_PREFIX}{clean[:80]}"


def _request_payload(request: SubagentRunRequest) -> dict[str, object]:
    return {
        "run_id": request.run_id or "",
        "goal": request.goal,
        "input_context_chars": len(request.input_context),
        "output_contract": request.output_contract or "",
        "acceptance_criteria": list(request.acceptance_criteria),
        "allowed_tools": list(request.allowed_tools),
        "tool_access_mode": request.tool_access_mode.value,
        "model_purpose": request.model_purpose or "",
        "max_steps": request.max_steps,
        "parent_session_id": request.parent_session_id or "",
        "parent_activation_id": request.parent_activation_id or "",
    }


def _result_payload(result: SubagentRunResult) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_id": result.run_id,
        "status": result.status.value,
        "summary": result.summary,
        "artifact_refs": list(result.artifact_refs),
        "evidence_refs": list(result.evidence_refs),
    }
    if result.usage is not None:
        payload["usage"] = {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "cache_hit_input_tokens": result.usage.cache_hit_input_tokens,
            "cache_miss_input_tokens": result.usage.cache_miss_input_tokens,
            "reasoning_tokens": result.usage.reasoning_tokens,
        }
    if result.error is not None:
        payload["error"] = {
            "code": result.error.code,
            "message": result.error.message,
            "refs": list(result.error.refs),
        }
    return payload


def _steps_payload(turn: "UserTurnResult") -> tuple[dict[str, object], ...]:
    steps: list[dict[str, object]] = []
    for index, step in enumerate(turn.steps, start=1):
        usage = step.response.usage
        payload: dict[str, object] = {
            "step_index": index,
            "assistant_content": _bounded_text(step.response.content),
            "assistant_reasoning": _bounded_text(step.response.reasoning),
            "finish_reason": step.response.finish_reason,
            "tool_requests": [
                {
                    "name": request.name,
                    "id": request.id,
                    "arguments": _bounded_value(request.arguments),
                    "argument_error": request.argument_error,
                }
                for request in step.response.tool_requests
            ],
            "tool_results": [
                _tool_result_payload(result)
                for result in step.tool_results[:MAX_TOOL_RESULTS_PER_STEP]
            ],
            "errors": [
                {
                    "code": error.code,
                    "message": error.message,
                    "refs": list(error.refs),
                }
                for error in step.errors
            ],
            "events": [
                {
                    "kind": event.kind,
                    "summary": event.summary,
                    "refs": list(event.refs),
                }
                for event in step.events
            ],
        }
        if len(step.tool_results) > MAX_TOOL_RESULTS_PER_STEP:
            payload["tool_results_truncated"] = (
                len(step.tool_results) - MAX_TOOL_RESULTS_PER_STEP
            )
        if usage is not None:
            payload["usage"] = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_hit_input_tokens": usage.cache_hit_input_tokens,
                "cache_miss_input_tokens": usage.cache_miss_input_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
            }
        steps.append(payload)
    return tuple(steps)


def _tool_result_payload(result: ModelToolResult) -> dict[str, object]:
    return {
        "name": result.name,
        "request_id": result.request_id,
        "content": _bounded_text(result.content),
        "data": _bounded_value(result.data),
        "refs": list(result.refs),
        "is_error": result.is_error,
    }


def _record_payload(record: SubagentResultRecord) -> dict[str, object]:
    return {
        "schema_version": record.schema_version,
        "ref": record.ref,
        "parent_session_id": record.parent_session_id,
        "created_at": record.created_at,
        "request": record.request,
        "result": record.result,
        "review": record.review,
        "steps": list(record.steps),
    }


def _record_from_payload(payload: dict[str, Any]) -> SubagentResultRecord:
    steps = payload.get("steps")
    step_values = steps if isinstance(steps, list) else []
    schema_version = _int(payload.get("schema_version"), default=1)
    return SubagentResultRecord(
        ref=_text(payload.get("ref")),
        parent_session_id=_text(payload.get("parent_session_id")),
        created_at=_text(payload.get("created_at")),
        request=_dict(payload.get("request")),
        result=_dict(payload.get("result")),
        review=_optional_dict(payload.get("review")),
        steps=tuple(_dict(step) for step in step_values),
        schema_version=schema_version,
    )


def _bounded_text(value: str) -> str:
    if len(value) <= MAX_TEXT_CHARS:
        return value
    return value[:MAX_TEXT_CHARS] + f"\n...[truncated {len(value) - MAX_TEXT_CHARS} chars]"


def _bounded_value(value: object) -> object:
    if isinstance(value, str):
        if len(value) <= MAX_ARGUMENT_TEXT_CHARS:
            return value
        return (
            value[:MAX_ARGUMENT_TEXT_CHARS]
            + f"\n...[truncated {len(value) - MAX_ARGUMENT_TEXT_CHARS} chars]"
        )
    if isinstance(value, dict):
        return {
            str(key): _bounded_value(item)
            for key, item in value.items()
            if isinstance(key, str)
        }
    if isinstance(value, list):
        return [_bounded_value(item) for item in value[:100]]
    if isinstance(value, tuple):
        return [_bounded_value(item) for item in value[:100]]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _bounded_text(str(value))


def _clean_segment(value: str, *, fallback: str) -> str:
    clean = str(value).strip()
    if not clean:
        clean = fallback
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", clean).strip(".-") or fallback


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _dict(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


def _int(value: object, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _optional_dict(value: object) -> dict[str, object] | None:
    return dict(value) if isinstance(value, dict) else None


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
