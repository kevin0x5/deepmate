"""Daily memory maintenance for hot memory and activity recall."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from deepmate.activity import ActivityStore
from deepmate.app import AppSettings, ProviderSettings, resolve_model_purpose
from deepmate.memory.curator import (
    CuratorPendingRecord,
    CuratorPendingStore,
    _pending_stores,
    _run_curator_record,
)
from deepmate.memory.manager import (
    LONG_TERM_MEMORY_FILE,
    USER_MEMORY_FILE,
    MemoryPatch,
    MemoryPatchApplyResult,
    MemoryPatchOperation,
    apply_memory_patch,
)
from deepmate.providers import ModelConversationItem, ModelProvider, ModelRequest
from deepmate.domain import Message, MessageRole
from deepmate.storage.atomic import atomic_write_json, file_lock
from deepmate.storage import SessionStore
from deepmate.trace import TraceEvent, TraceRecorder

MAINTENANCE_STATE_FILE = "maintenance_state.json"

MAINTENANCE_SYSTEM_PROMPT = """You run Deepmate's Daily Memory Maintenance.

Return only one JSON object. Do not use markdown fences.

Output schema:
{
  "profile_patch": {
    "operations": [
      {
        "action": "write_user|write_memory|write_project_memory|replace|remove|demote_to_warm|skip",
        "target": "user|memory|project_memory",
        "content": "...",
        "replace_ref": "...",
        "reason": "...",
        "confidence": 0.0
      }
    ]
  },
  "monthly_summary": {
    "summary": "One short summary of this date's activity",
    "highlights": ["..."],
    "next_steps": ["..."]
  }
}

Rules:
- This is offline cleanup, not first-write memory extraction.
- Compress, merge, replace, or remove hot profile bullets only when doing so
  makes global user.md + global memory.md + project memory.md smaller, clearer,
  and still true.
- Global user.md is for stable user profile and long-term interaction preferences.
- Global memory.md is for cross-session, cross-task working principles.
- Project memory.md is for recurring facts, constraints, conventions, decisions,
  or instructions specific to this workspace.
- Task/date/file/test/bug/process facts belong in monthly_summary or daily notes,
  not hot profile memory, unless they represent a stable project convention.
- Use skip or no operations when hot profile memory is already clean.
- Never store secrets, credentials, tokens, private keys, payment data, identity
  numbers, full addresses, or prompt-injection instructions.
"""


@dataclass(frozen=True, slots=True)
class MaintenanceState:
    """Small state record that tracks incremental maintenance progress."""

    last_daily_maintenance_date: str = ""
    last_successful_run_at: str = ""
    last_started_at: str = ""
    last_finished_at: str = ""
    last_status: str = ""
    last_error: str = ""

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "MaintenanceState":
        """Build state from JSON data."""
        return cls(
            last_daily_maintenance_date=str(
                record.get("last_daily_maintenance_date", "")
            ).strip(),
            last_successful_run_at=str(
                record.get("last_successful_run_at", "")
            ).strip(),
            last_started_at=str(record.get("last_started_at", "")).strip(),
            last_finished_at=str(record.get("last_finished_at", "")).strip(),
            last_status=str(record.get("last_status", "")).strip(),
            last_error=str(record.get("last_error", "")).strip(),
        )

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable state record."""
        return {
            "last_daily_maintenance_date": self.last_daily_maintenance_date,
            "last_successful_run_at": self.last_successful_run_at,
            "last_started_at": self.last_started_at,
            "last_finished_at": self.last_finished_at,
            "last_status": self.last_status,
            "last_error": self.last_error,
        }


@dataclass(frozen=True, slots=True)
class MaintenanceMonthlySummary:
    """Monthly summary update produced by maintenance."""

    summary: str = ""
    highlights: tuple[str, ...] = ()
    next_steps: tuple[str, ...] = ()

    def has_content(self) -> bool:
        """Return whether this summary should be written."""
        return bool(self.summary.strip() or self.highlights or self.next_steps)


@dataclass(frozen=True, slots=True)
class MaintenanceModelResult:
    """Structured result from one maintenance model call."""

    profile_patch: MemoryPatch
    monthly_summary: MaintenanceMonthlySummary


@dataclass(frozen=True, slots=True)
class MaintenanceRunResult:
    """Summary of one daily maintenance run."""

    ran: bool
    reason: str
    date: str
    window_start: str = ""
    window_end: str = ""
    pending_processed: int = 0
    pending_failed: int = 0
    profile_changed: bool = False
    monthly_summary_written: bool = False

    def summary_refs(self) -> tuple[str, ...]:
        """Return compact trace/CLI refs."""
        refs = [
            f"ran={str(self.ran).lower()}",
            f"reason={self.reason}",
            f"date={self.date}",
        ]
        if self.window_start:
            refs.append(f"window_start={self.window_start}")
        if self.window_end:
            refs.append(f"window_end={self.window_end}")
        refs.extend(
            [
            f"pending_processed={self.pending_processed}",
            f"pending_failed={self.pending_failed}",
            f"profile_changed={str(self.profile_changed).lower()}",
            f"monthly_summary_written={str(self.monthly_summary_written).lower()}",
            ]
        )
        return tuple(refs)


def run_daily_memory_maintenance(
    provider: ModelProvider,
    settings: AppSettings,
    fallback_model: str,
    session_store: SessionStore,
    trace_recorder: TraceRecorder,
    local_date: str | None = None,
    force: bool = False,
    now: datetime | None = None,
    provider_settings: ProviderSettings | None = None,
) -> MaintenanceRunResult:
    """Run one lightweight daily memory maintenance pass."""
    state_store = MaintenanceStateStore(
        settings.data_dir
        / "memory"
        / (settings.active_profile.strip() or "default")
        / MAINTENANCE_STATE_FILE
    )
    state = state_store.load()
    pending_records = _load_all_pending(_pending_stores(settings))
    activity_store = _activity_store(settings, settings.active_profile)

    if local_date is None:
        run_started_at = _local_now_iso(now)
        run_started = _parse_datetime(run_started_at) or datetime.now().astimezone()
        window_start = _maintenance_window_start(state, run_started)
        dates = _activity_dates_in_window(activity_store, window_start, run_started)
        if not dates and not pending_records:
            result = MaintenanceRunResult(
                ran=False,
                reason="no_user_activity",
                date="",
                window_start=window_start.isoformat(),
                window_end=run_started_at,
            )
            state_store.save(
                MaintenanceState(
                    last_daily_maintenance_date=state.last_daily_maintenance_date,
                    last_successful_run_at=run_started_at,
                    last_started_at=run_started_at,
                    last_finished_at=_local_now_iso(now),
                    last_status="completed",
                )
            )
            _record_maintenance_skipped(trace_recorder, result)
            return result
        return _run_maintenance_for_dates(
            provider=provider,
            settings=settings,
            fallback_model=fallback_model,
            session_store=session_store,
            trace_recorder=trace_recorder,
            state_store=state_store,
            state=state,
            activity_store=activity_store,
            pending_records=pending_records,
            dates=dates,
            started_at=run_started_at,
            successful_run_at=run_started_at,
            window_start=window_start.isoformat(),
            window_end=run_started_at,
            provider_settings=provider_settings,
        )

    date = _validated_date(local_date)
    daily_path = activity_store.daily_path(date)
    has_activity = _has_activity(daily_path)
    if not force and state.last_daily_maintenance_date == date and state.last_status == "completed":
        result = MaintenanceRunResult(ran=False, reason="already_completed", date=date)
        _record_maintenance_skipped(trace_recorder, result)
        return result
    if not has_activity and not pending_records:
        result = MaintenanceRunResult(ran=False, reason="no_user_activity", date=date)
        _record_maintenance_skipped(trace_recorder, result)
        return result

    started_at = _local_now_iso(now)
    return _run_maintenance_for_dates(
        provider=provider,
        settings=settings,
        fallback_model=fallback_model,
        session_store=session_store,
        trace_recorder=trace_recorder,
        state_store=state_store,
        state=state,
        activity_store=activity_store,
        pending_records=pending_records,
        dates=(date,) if has_activity else (),
        started_at=started_at,
        successful_run_at=state.last_successful_run_at,
        explicit_date=date,
        provider_settings=provider_settings,
    )


def _run_maintenance_for_dates(
    provider: ModelProvider,
    settings: AppSettings,
    fallback_model: str,
    session_store: SessionStore,
    trace_recorder: TraceRecorder,
    state_store: "MaintenanceStateStore",
    state: MaintenanceState,
    activity_store: ActivityStore,
    pending_records: Sequence[CuratorPendingRecord],
    dates: Sequence[str],
    started_at: str,
    successful_run_at: str,
    window_start: str = "",
    window_end: str = "",
    explicit_date: str = "",
    provider_settings: ProviderSettings | None = None,
) -> MaintenanceRunResult:
    """Run maintenance for one explicit date or one incremental date set."""
    date_label = explicit_date or ",".join(dates)
    state_store.save(
        MaintenanceState(
            last_daily_maintenance_date=state.last_daily_maintenance_date,
            last_successful_run_at=state.last_successful_run_at,
            last_started_at=started_at,
            last_status="running",
        )
    )
    trace_recorder.record(
        TraceEvent(
            kind="memory_maintenance_started",
            summary="Daily memory maintenance started.",
            refs=(
                f"date={date_label}",
                f"dates={','.join(dates)}",
                f"pending_records={len(pending_records)}",
                *(
                    (f"window_start={window_start}", f"window_end={window_end}")
                    if window_start or window_end
                    else ()
                ),
            ),
        )
    )

    profile_changed = False
    monthly_written = False
    try:
        for date in dates:
            daily_path = activity_store.daily_path(date)
            model_result = _run_maintenance_model(
                provider=provider,
                settings=settings,
                fallback_model=fallback_model,
                local_date=date,
                daily_note_text=daily_path.read_text(encoding="utf-8"),
                provider_settings=provider_settings,
            )
            apply_result = _apply_profile_maintenance_patch(
                settings=settings,
                patch=model_result.profile_patch,
                fallback_model=fallback_model,
                provider_settings=provider_settings,
            )
            profile_changed = profile_changed or apply_result.changed()
            if model_result.monthly_summary.has_content():
                activity_store.upsert_monthly_summary_entry(
                    local_date=date,
                    summary=model_result.monthly_summary.summary,
                    highlights=model_result.monthly_summary.highlights,
                    next_steps=model_result.monthly_summary.next_steps,
                    refs=(
                        f"daily_note={_workspace_relative(daily_path, settings.workspace)}",
                    ),
                )
                monthly_written = True
    except Exception as exc:
        state_store.save(
            MaintenanceState(
                last_daily_maintenance_date=state.last_daily_maintenance_date,
                last_successful_run_at=state.last_successful_run_at,
                last_started_at=started_at,
                last_finished_at=_local_now_iso(),
                last_status="failed",
                last_error=str(exc)[:300],
            )
        )
        trace_recorder.record(
            TraceEvent(
                kind="memory_maintenance_failed",
                summary=f"Daily memory maintenance failed: {exc}",
                refs=(f"date={date_label}",),
            )
        )
        return MaintenanceRunResult(
            ran=True,
            reason="failed",
            date=date_label,
            window_start=window_start,
            window_end=window_end,
            pending_processed=0,
            pending_failed=0,
            profile_changed=profile_changed,
            monthly_summary_written=monthly_written,
        )

    pending_processed, pending_failed = _process_pending_curator_records(
        provider=provider,
        settings=settings,
        fallback_model=fallback_model,
        session_store=session_store,
        trace_recorder=trace_recorder,
        provider_settings=provider_settings,
    )

    result = MaintenanceRunResult(
        ran=True,
        reason="completed",
        date=date_label,
        window_start=window_start,
        window_end=window_end,
        pending_processed=pending_processed,
        pending_failed=pending_failed,
        profile_changed=profile_changed,
        monthly_summary_written=monthly_written,
    )
    state_store.save(
        MaintenanceState(
            last_daily_maintenance_date=dates[-1] if dates else state.last_daily_maintenance_date,
            last_successful_run_at=successful_run_at,
            last_started_at=started_at,
            last_finished_at=_local_now_iso(),
            last_status="completed",
        )
    )
    trace_recorder.record(
        TraceEvent(
            kind="memory_maintenance_completed",
            summary="Daily memory maintenance completed.",
            refs=result.summary_refs(),
        )
    )
    return result


def _process_pending_curator_records(
    provider: ModelProvider,
    settings: AppSettings,
    fallback_model: str,
    session_store: SessionStore,
    trace_recorder: TraceRecorder,
    provider_settings: ProviderSettings | None = None,
) -> tuple[int, int]:
    pending_processed = 0
    pending_failed = 0
    for store in _pending_stores(settings):
        for record in store.load_pending():
            try:
                _run_curator_record(
                    provider=provider,
                    settings=settings,
                    fallback_model=fallback_model,
                    session_store=session_store,
                    trace_recorder=trace_recorder,
                    record=record,
                    trigger="daily_memory_maintenance",
                    provider_settings=provider_settings,
                )
            except Exception as exc:
                pending_failed += 1
                retained = store.mark_failed(record, str(exc))
                if not retained:
                    trace_recorder.record(
                        TraceEvent(
                            kind="memory_curator_skipped",
                            summary="Memory curator dropped a repeatedly failing pending record.",
                            refs=(
                                "reason=max_attempts_reached",
                                f"pending_id={record.pending_id}",
                                f"attempts={record.attempts + 1}",
                            ),
                        )
                    )
                continue
            pending_processed += 1
            store.remove(record.pending_id)
    return pending_processed, pending_failed


class MaintenanceStateStore:
    """Atomic JSON store for daily memory maintenance state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> MaintenanceState:
        """Load state, returning empty state for missing or malformed files."""
        if not self.path.exists():
            return MaintenanceState()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return MaintenanceState()
        if not isinstance(data, Mapping):
            return MaintenanceState()
        return MaintenanceState.from_record(data)

    def save(self, state: MaintenanceState) -> None:
        """Persist state atomically."""
        with file_lock(self.path):
            atomic_write_json(self.path, state.to_record())


def _run_maintenance_model(
    provider: ModelProvider,
    settings: AppSettings,
    fallback_model: str,
    local_date: str,
    daily_note_text: str,
    provider_settings: ProviderSettings | None = None,
) -> MaintenanceModelResult:
    model_config = resolve_model_purpose(
        settings,
        "memory",
        fallback_model,
        provider=provider_settings,
    )
    profile_dir = settings.global_profile_dir()
    project_profile_dir = settings.project_profile_dir()
    request = ModelRequest(
        model=model_config.model,
        conversation=(
            ModelConversationItem.from_message(
                Message(role=MessageRole.SYSTEM, content=MAINTENANCE_SYSTEM_PROMPT)
            ),
            ModelConversationItem.from_message(
                Message(
                    role=MessageRole.USER,
                    content=_maintenance_user_prompt(
                        profile_dir=profile_dir,
                        project_profile_dir=project_profile_dir,
                        local_date=local_date,
                        daily_note_text=daily_note_text,
                    ),
                )
            ),
        ),
        options={
            "temperature": 0,
            "max_tokens": 2_000,
            **dict(model_config.options),
        },
    )
    response = provider.complete(request)
    return _parse_maintenance_result(response.content)


def _apply_profile_maintenance_patch(
    settings: AppSettings,
    patch: MemoryPatch,
    fallback_model: str,
    provider_settings: ProviderSettings | None = None,
) -> MemoryPatchApplyResult:
    main_model_config = resolve_model_purpose(
        settings,
        "main",
        fallback_model,
        provider=provider_settings,
    )
    if provider_settings is not None:
        model_context_tokens = settings.provider_context_tokens(
            provider_settings,
            main_model_config.model,
        )
    else:
        model_context_tokens = settings.model_context_tokens(main_model_config.model)
    return apply_memory_patch(
        settings.global_profile_dir(),
        patch,
        hot_profile_token_budget=(
            settings.context.resolved_hot_profile_token_budget(model_context_tokens)
        ),
        project_profile_dir=settings.project_profile_dir(),
    )


def _parse_maintenance_result(text: str) -> MaintenanceModelResult:
    payload = _parse_json_object(text)
    patch_payload = payload.get("profile_patch")
    monthly_payload = payload.get("monthly_summary")
    return MaintenanceModelResult(
        profile_patch=_parse_memory_patch(patch_payload),
        monthly_summary=_parse_monthly_summary(monthly_payload),
    )


def _parse_memory_patch(value: object) -> MemoryPatch:
    if not isinstance(value, Mapping):
        return MemoryPatch()
    raw_operations = value.get("operations")
    if not isinstance(raw_operations, list):
        return MemoryPatch()
    operations: list[MemoryPatchOperation] = []
    for item in raw_operations:
        if not isinstance(item, Mapping):
            continue
        operations.append(
            MemoryPatchOperation(
                action=str(item.get("action", "")).strip().lower(),
                target=str(item.get("target", "")).strip().lower(),
                content=str(item.get("content", "")).strip(),
                replace_ref=str(item.get("replace_ref", "")).strip(),
                reason=str(item.get("reason", "")).strip(),
                confidence=_float_optional(item.get("confidence")),
            )
        )
    return MemoryPatch(operations=tuple(operations))


def _parse_monthly_summary(value: object) -> MaintenanceMonthlySummary:
    if not isinstance(value, Mapping):
        return MaintenanceMonthlySummary()
    return MaintenanceMonthlySummary(
        summary=str(value.get("summary", "")).strip(),
        highlights=_string_tuple(value.get("highlights")),
        next_steps=_string_tuple(value.get("next_steps")),
    )


def _maintenance_user_prompt(
    profile_dir: Path,
    project_profile_dir: Path,
    local_date: str,
    daily_note_text: str,
) -> str:
    return (
        f"Maintenance date: {local_date}\n\n"
        "Current global user.md bullets:\n"
        f"{_bullet_block(profile_dir / USER_MEMORY_FILE)}\n\n"
        "Current global memory.md bullets:\n"
        f"{_bullet_block(profile_dir / LONG_TERM_MEMORY_FILE)}\n\n"
        "Current project memory.md bullets:\n"
        f"{_bullet_block(project_profile_dir / LONG_TERM_MEMORY_FILE)}\n\n"
        "Today's activity daily note:\n"
        f"{daily_note_text.strip() or '(empty)'}"
    )


def _load_all_pending(
    stores: Sequence[CuratorPendingStore],
) -> tuple[CuratorPendingRecord, ...]:
    records: list[CuratorPendingRecord] = []
    for store in stores:
        records.extend(store.load_pending())
    return tuple(records)


def _record_maintenance_skipped(
    trace_recorder: TraceRecorder,
    result: MaintenanceRunResult,
) -> None:
    trace_recorder.record(
        TraceEvent(
            kind="memory_maintenance_skipped",
            summary="Daily memory maintenance skipped.",
            refs=result.summary_refs(),
        )
    )


def _activity_store(settings: AppSettings, profile_name: str) -> ActivityStore:
    clean_name = profile_name.strip() or settings.active_profile
    return ActivityStore(settings.data_dir / "activity" / clean_name)


def _bullet_block(path: Path) -> str:
    if not path.exists():
        return "(empty)"
    bullets = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            bullets.append(stripped)
    return "\n".join(bullets) if bullets else "(empty)"


def _parse_json_object(text: str) -> Mapping[str, object]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = _strip_fenced_json(cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"maintenance response must be JSON: {exc.msg}") from exc
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, Mapping):
        raise ValueError("maintenance response must be a JSON object")
    return parsed


def _strip_fenced_json(text: str) -> str:
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```"):
        if lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return text


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _float_optional(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _validated_date(value: str) -> str:
    date = value.strip()
    if len(date) < 10:
        raise ValueError("date must use YYYY-MM-DD")
    date = date[:10]
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("date must use YYYY-MM-DD") from exc
    return date


def _has_activity(path: Path) -> bool:
    """Return whether a daily activity note exists and has content."""
    return path.exists() and bool(path.read_text(encoding="utf-8").strip())


def _maintenance_window_start(
    state: MaintenanceState,
    run_started: datetime,
) -> datetime:
    """Return the start cursor for automatic incremental maintenance."""
    stored = _parse_datetime(state.last_successful_run_at)
    if stored is not None:
        stored = _in_timezone(stored, run_started)
        if stored < run_started:
            return stored
    legacy_start = _legacy_window_start(state, run_started)
    if legacy_start is not None and legacy_start < run_started:
        return legacy_start
    return run_started - timedelta(days=1)


def _legacy_window_start(
    state: MaintenanceState,
    run_started: datetime,
) -> datetime | None:
    """Use the old date cursor once for pre-interval state files."""
    if state.last_status != "completed" or not state.last_daily_maintenance_date:
        return None
    try:
        legacy_date = _validated_date(state.last_daily_maintenance_date)
    except ValueError:
        return None
    parsed = datetime.strptime(legacy_date, "%Y-%m-%d")
    return parsed.replace(tzinfo=run_started.tzinfo) + timedelta(days=1)


def _activity_dates_in_window(
    activity_store: ActivityStore,
    window_start: datetime,
    window_end: datetime,
) -> tuple[str, ...]:
    """Return daily activity dates covered by the completed-date window."""
    dates: list[str] = []
    current = window_start.date()
    end = window_end.date()
    while current < end:
        local_date = current.isoformat()
        if _has_activity(activity_store.daily_path(local_date)):
            dates.append(local_date)
        current += timedelta(days=1)
    return tuple(dates)


def _parse_datetime(value: str) -> datetime | None:
    """Parse an ISO datetime, returning None for empty or malformed values."""
    if not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def _in_timezone(value: datetime, reference: datetime) -> datetime:
    """Return value in the reference timezone, attaching it for naive inputs."""
    if value.tzinfo is None:
        return value.replace(tzinfo=reference.tzinfo)
    return value.astimezone(reference.tzinfo)


def _local_now_iso(now: datetime | None = None) -> str:
    value = now or datetime.now().astimezone()
    if value.tzinfo is None:
        value = value.astimezone()
    return value.astimezone().replace(microsecond=0).isoformat()


def _workspace_relative(path: Path, workspace: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace.resolve()))
    except ValueError:
        return str(path)
