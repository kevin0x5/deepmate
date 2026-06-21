"""Backend service for the desktop pet frontend.

The service keeps Deepmate-owned pet behavior in Python and writes a small
frontend-ready state file consumed by the desktop UI process.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic

from deepmate.pet.copy import bounded_pet_text, fallback_pet_copy, generate_pet_copy
from deepmate.pet.events import PetEvent, event_for_care_reminder
from deepmate.pet.learning import (
    fetch_learning_candidates,
    generate_learning_suggestion,
    interest_tags_from_texts,
)
from deepmate.pet.policy import PetDisplayDecision, PetDisplayPolicy
from deepmate.pet.state import (
    PetProfile,
    PetStateStore,
    PetUserAction,
    default_pet_profile,
)
from deepmate.providers import ModelProvider

DEFAULT_POLL_SECONDS = 1.2
MAX_CARE_EMITTED_KEYS = 14
LEARNING_INITIAL_DELAY_SECONDS = 90.0
LEARNING_POLL_SECONDS = 30 * 60.0
CARE_INITIAL_DELAY_SECONDS = 45 * 60.0
CARE_POLL_SECONDS = 2 * 60 * 60.0
CARE_LONG_WORK_SECONDS = 2 * 60 * 60
CARE_ACTIVE_KINDS = {
    "task.started",
    "task.progress",
    "task.waiting",
}


@dataclass(frozen=True, slots=True)
class PetServiceSnapshot:
    """One frontend-ready state snapshot."""

    record: Mapping[str, object]
    event: PetEvent
    profile: PetProfile


class PetBackendService:
    """Maintain pet behavior and publish state for a frontend process."""

    def __init__(
        self,
        store: PetStateStore,
        *,
        provider: ModelProvider | None = None,
        model: str = "",
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        learning_initial_delay: float = LEARNING_INITIAL_DELAY_SECONDS,
        care_initial_delay: float = CARE_INITIAL_DELAY_SECONDS,
    ) -> None:
        self.store = store
        self.provider = provider
        self.model = model.strip()
        self.poll_seconds = max(0.2, float(poll_seconds))
        self.policy = PetDisplayPolicy()
        self.stop_event = Event()
        self.profile = self._load_profile()
        self.current_event = _event_from_record(self.store.offline_state())
        self.current_event_id = ""
        self.current_ui_signature = ""
        self._publish_lock = Lock()
        self._last_poll_error_log = 0.0
        self._learning_failure_count = 0
        self._last_learning_at = monotonic() - max(
            0.0,
            LEARNING_POLL_SECONDS - learning_initial_delay,
        )
        self._last_care_at = monotonic() - max(0.0, CARE_POLL_SECONDS - care_initial_delay)
        self._care_emitted_keys: set[str] = set()
        self._active_work_key = ""
        self._active_work_started_at: datetime | None = None

    def run_forever(self) -> None:
        """Run until ``stop`` is called or the process is interrupted."""
        while not self.stop_event.is_set():
            self.tick()
            self.stop_event.wait(self.poll_seconds)

    def stop(self) -> None:
        """Request the service loop to stop."""
        self.stop_event.set()

    def tick(self) -> PetServiceSnapshot | None:
        """Refresh state, maybe emit proactive cards, and publish UI state."""
        try:
            self._prune_care_emitted_keys()
            self.profile = self._load_profile()
            ui_state = self.store.load_pet_state()
            muted = _bool(
                ui_state.get("muted"),
                False,
            ) or _profile_muted_until_active(self.profile)
            collapsed = _bool(ui_state.get("collapsed"), False) or _bool(
                ui_state.get("mini"),
                False,
            )
            record = self.store.load_current_state() or self.store.offline_state()
            event = _event_from_record(record)
            current_event_changed = False
            with self._publish_lock:
                if event.event_id != self.current_event_id:
                    self.current_event = event
                    self.current_event_id = event.event_id
                    current_event_changed = True
            if current_event_changed:
                self._update_active_work_started_at(event)
            snapshot = self.publish_event(event, muted=muted)
            self._maybe_schedule_learning(muted=muted, collapsed=collapsed)
            care_snapshot = self._maybe_schedule_care(muted=muted, collapsed=collapsed)
            return care_snapshot or snapshot
        except Exception as exc:
            self._log_poll_error(exc)
            return None

    def publish_event(
        self,
        event: PetEvent,
        *,
        muted: bool = False,
        bubble_override: Mapping[str, object] | None = None,
    ) -> PetServiceSnapshot:
        """Publish one event as frontend UI state."""
        with self._publish_lock:
            decision = self.policy.decide(event)
            bubble = (
                dict(bubble_override)
                if bubble_override is not None
                else self._bubble_for_event(event, decision, muted=muted)
            )
            record = _ui_state_record(
                event,
                self.profile,
                decision,
                bubble=bubble,
                muted=muted,
            )
            signature = _ui_signature(record)
            if signature != self.current_ui_signature:
                self.store.save_ui_state(record)
                self.current_ui_signature = signature
            return PetServiceSnapshot(record=record, event=event, profile=self.profile)

    def append_action(self, action: str, payload: Mapping[str, object] | None = None) -> None:
        """Append a user action for Deepmate to process."""
        self.store.append_action(
            PetUserAction(
                action=action,
                created_at=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                payload=dict(payload or {}),
            )
        )

    def _load_profile(self) -> PetProfile:
        try:
            if not self.store.profile_path.exists():
                self.store.save_profile(default_pet_profile())
            return self.store.load_profile()
        except OSError:
            return default_pet_profile()

    def _bubble_for_event(
        self,
        event: PetEvent,
        decision: PetDisplayDecision,
        *,
        muted: bool,
    ) -> dict[str, object]:
        if muted or not decision.show_bubble:
            return _bubble_record("", decision, show=False)
        fallback = fallback_pet_copy(event, self.profile, max_chars=90)
        text = self._load_or_generate_copy(event, fallback)
        return _bubble_record(text, decision, show=bool(text.strip()))

    def _load_or_generate_copy(self, event: PetEvent, fallback: str) -> str:
        key = _copy_cache_key(event, self.profile)
        try:
            cache = self.store.load_copy_cache()
        except OSError:
            cache = {}
        cached = cache.get(key)
        if isinstance(cached, Mapping):
            text = _text(cached.get("text"))
            if text:
                return text
        if self.provider is None or not self.model or self.profile.bubble_generation == "frugal":
            return fallback
        result = generate_pet_copy(
            event,
            self.profile,
            provider=self.provider,
            model=self.model,
            max_chars=90,
        )
        text = result.text.strip() or fallback
        self._store_copy_cache(key, text, result.source)
        return text

    def _store_copy_cache(self, key: str, text: str, source: str) -> None:
        try:
            self.store.update_copy_cache(
                key,
                {
                    "text": text,
                    "source": source,
                    "cached_at": datetime.now(timezone.utc)
                    .astimezone()
                    .isoformat(timespec="seconds"),
                },
                limit=200,
            )
        except OSError:
            pass

    def _maybe_schedule_learning(self, *, muted: bool, collapsed: bool) -> None:
        now = monotonic()
        poll_seconds = _learning_poll_seconds(self.profile)
        if now - self._last_learning_at < poll_seconds:
            return
        self._last_learning_at = now
        if self.profile.learning_mode == "off" or muted or collapsed:
            return
        if _feedback_suppressed(
            self.store.load_learning_state(),
            "learning_suggestion",
        ):
            return
        Thread(target=self._learning_worker, daemon=True).start()

    def _learning_worker(self) -> None:
        try:
            state = self.store.load_learning_state()
            if _feedback_suppressed(state, "learning_suggestion"):
                self._learning_failure_count = 0
                return
            raw_shown = state.get("shown_urls", [])
            shown_urls = (
                {_text(item) for item in raw_shown if isinstance(item, str)}
                if isinstance(raw_shown, list)
                else set()
            )
            candidates = []
            for source in _learning_sources(state):
                candidates.extend(fetch_learning_candidates(source, timeout=8, limit=16))
            fresh = tuple(candidate for candidate in candidates if candidate.url not in shown_urls)
            if not fresh:
                self._learning_failure_count = 0
                return
            tags = interest_tags_from_texts(
                (
                    self.current_event.current_work_title,
                    self.current_event.summary,
                )
            )
            suggestion = generate_learning_suggestion(
                fresh,
                interest_tags=tags,
                current_work_summary=self.current_event.summary,
                profile=self.profile,
                provider=self.provider,
                model=self.model,
            )
            if suggestion is None:
                self._learning_failure_count = 0
                return
            if self._learning_display_suppressed():
                self._learning_failure_count = 0
                return
            shown_urls.add(suggestion.url)
            state["shown_urls"] = list(shown_urls)[-100:]
            state["last_suggestion"] = {
                "title": suggestion.title,
                "url": suggestion.url,
                "summary": suggestion.summary,
                "source": suggestion.source,
                "shown_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            }
            self.store.save_learning_state(state)
            text = bounded_pet_text(f"学习建议：{suggestion.summary} {suggestion.url}", 180)
            self._learning_failure_count = 0
            bubble = {
                "text": text,
                "show": True,
                "hold": False,
                "reason": "learning_suggestion",
                "priority": 40,
                "duration_ms": 16_000,
            }
            with self._publish_lock:
                event = self.current_event
            self.publish_event(event, bubble_override=bubble)
        except Exception as exc:
            self._learning_failure_count += 1
            if self._learning_failure_count in {3, 10} or self._learning_failure_count % 25 == 0:
                print(
                    "warning: desktop pet learning refresh failed "
                    f"{self._learning_failure_count} times: {exc}",
                    file=sys.stderr,
                )

    def _learning_display_suppressed(self) -> bool:
        profile = self._load_profile()
        ui_state = self.store.load_pet_state()
        muted = _bool(ui_state.get("muted"), False) or _profile_muted_until_active(
            profile
        )
        collapsed = _bool(ui_state.get("collapsed"), False) or _bool(
            ui_state.get("mini"),
            False,
        )
        return profile.learning_mode == "off" or muted or collapsed

    def _maybe_schedule_care(
        self,
        *,
        muted: bool,
        collapsed: bool,
    ) -> PetServiceSnapshot | None:
        now = monotonic()
        if now - self._last_care_at < CARE_POLL_SECONDS:
            return None
        self._last_care_at = now
        if not self.profile.proactive_care or muted or collapsed:
            return None
        care_reason = _care_reason(
            self.profile,
            self.current_event,
            active_started_at=self._active_work_started_at,
        )
        if not care_reason:
            return None
        if _feedback_suppressed(
            self.store.load_learning_state(),
            "proactive_care",
            "care.reminder",
            care_reason,
        ):
            return None
        care_key = f"{datetime.now(timezone.utc).astimezone().date().isoformat()}:{care_reason}"
        if care_key in self._care_emitted_keys:
            return None
        self._care_emitted_keys.add(care_key)
        self._prune_care_emitted_keys()
        event = event_for_care_reminder(
            workspace=self.current_event.workspace,
            session_id=self.current_event.session_id,
            title=self.current_event.current_work_title,
            summary=_care_summary(care_reason),
        )
        decision = self.policy.decide(event)
        if not decision.show_bubble:
            return None
        text = fallback_pet_copy(event, self.profile, max_chars=90)
        bubble = {
            "text": text,
            "show": True,
            "hold": decision.hold,
            "reason": decision.reason,
            "priority": decision.priority,
            "duration_ms": 12_000,
        }
        return self.publish_event(event, bubble_override=bubble)

    def _prune_care_emitted_keys(self) -> None:
        if len(self._care_emitted_keys) <= MAX_CARE_EMITTED_KEYS:
            return
        self._care_emitted_keys = set(
            sorted(self._care_emitted_keys)[-MAX_CARE_EMITTED_KEYS:]
        )

    def _update_active_work_started_at(self, event: PetEvent) -> None:
        key = _active_work_key(event)
        if not key:
            self._active_work_key = ""
            self._active_work_started_at = None
            return
        event_started_at = _parse_datetime(event.created_at) or datetime.now(
            timezone.utc
        ).astimezone()
        if key != self._active_work_key:
            self._active_work_key = key
            self._active_work_started_at = event_started_at
            return
        if (
            self._active_work_started_at is None
            or event_started_at < self._active_work_started_at
        ):
            self._active_work_started_at = event_started_at

    def _log_poll_error(self, exc: Exception) -> None:
        now = monotonic()
        if now - self._last_poll_error_log < 30:
            return
        self._last_poll_error_log = now
        print(f"warning: desktop pet refresh failed: {exc}", file=sys.stderr)


def run_pet_service(
    data_dir: str | Path,
    *,
    provider: ModelProvider | None = None,
    model: str = "",
    poll_seconds: float = DEFAULT_POLL_SECONDS,
) -> int:
    """Run only the backend pet service."""
    service = PetBackendService(
        PetStateStore.in_data_dir(data_dir),
        provider=provider,
        model=model,
        poll_seconds=poll_seconds,
    )
    try:
        service.run_forever()
    except KeyboardInterrupt:
        service.stop()
    return 0


def _ui_state_record(
    event: PetEvent,
    profile: PetProfile,
    decision: PetDisplayDecision,
    *,
    bubble: Mapping[str, object],
    muted: bool,
) -> dict[str, object]:
    actions = [action.to_record() for action in event.actions]
    return {
        "event_id": event.event_id,
        "kind": event.kind,
        "state": event.state.value,
        "severity": event.severity.value,
        "created_at": event.created_at,
        "workspace": event.workspace,
        "session_id": event.session_id,
        "task_id": event.task_id,
        "run_id": event.run_id,
        "title": event.current_work_title,
        "summary": event.summary,
        "refs": list(event.refs),
        "actions": actions,
        "bubble": dict(bubble),
        "display": {
            "show_bubble": decision.show_bubble,
            "priority": decision.priority,
            "reason": decision.reason,
            "hold": decision.hold,
        },
        "profile": {
            "pet_id": profile.pet_id,
            "species": profile.species,
            "name": profile.name,
            "style": profile.style,
            "bubble_generation": profile.bubble_generation,
            "learning_mode": profile.learning_mode,
            "proactive_care": profile.proactive_care,
            "startup": profile.startup,
            "report_cadence": profile.report_cadence,
            "quiet_hours": dict(profile.quiet_hours),
            "muted_until": profile.muted_until,
        },
        "muted": muted,
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }


def _bubble_record(
    text: str,
    decision: PetDisplayDecision,
    *,
    show: bool,
) -> dict[str, object]:
    return {
        "text": text,
        "show": show,
        "hold": decision.hold,
        "reason": decision.reason,
        "priority": decision.priority,
        "duration_ms": 0 if decision.hold else 7_500,
    }


def _event_from_record(record: Mapping[str, object]) -> PetEvent:
    try:
        return PetEvent.from_record(record)
    except (TypeError, ValueError):
        return PetEvent(
            kind="current_work.idle",
            summary="Deepmate is ready.",
        ).normalized()


def _copy_cache_key(event: PetEvent, profile: PetProfile) -> str:
    return f"{event.event_id}:{profile.pet_id}:{profile.bubble_generation}"


def _learning_sources(state: Mapping[str, object]) -> tuple[str, ...]:
    raw = state.get("sources")
    if isinstance(raw, list):
        return tuple(_text(item) for item in raw if _text(item))
    return ()


def _feedback_suppressed(state: Mapping[str, object], *keys: str) -> bool:
    raw = state.get("suppressed_until")
    if not isinstance(raw, Mapping):
        return False
    now = datetime.now(timezone.utc).astimezone()
    for key in keys:
        until = _parse_datetime(_text(raw.get(key)))
        if until is not None and until > now:
            return True
    return False


def _learning_poll_seconds(profile: PetProfile) -> float:
    if profile.learning_mode == "standard":
        return LEARNING_POLL_SECONDS
    if profile.learning_mode == "low":
        return 60 * 60.0
    return LEARNING_POLL_SECONDS


def _profile_muted_until_active(profile: PetProfile) -> bool:
    muted_until = _parse_datetime(profile.muted_until)
    if muted_until is None:
        return False
    return muted_until > datetime.now(timezone.utc).astimezone()


def _care_reason(
    profile: PetProfile,
    event: PetEvent,
    *,
    active_started_at: datetime | None = None,
) -> str:
    if event.kind not in CARE_ACTIVE_KINDS:
        return ""
    started_at = active_started_at or _parse_datetime(event.created_at)
    now = datetime.now(timezone.utc).astimezone()
    if (
        started_at is not None
        and (now - started_at).total_seconds() >= CARE_LONG_WORK_SECONDS
    ):
        return "long_work"
    if _in_quiet_hours(profile, now):
        return "quiet_hours"
    return ""


def _care_summary(reason: str) -> str:
    if reason == "quiet_hours":
        return "It is getting late. Save the current progress before going further."
    return "This has been running for a while. Consider taking a short break."


def _in_quiet_hours(profile: PetProfile, now: datetime) -> bool:
    quiet_hours = profile.quiet_hours
    if not isinstance(quiet_hours, Mapping):
        return False
    start = _parse_time(_text(quiet_hours.get("start")))
    end = _parse_time(_text(quiet_hours.get("end")))
    if start is None or end is None or start == end:
        return False
    current = now.timetz().replace(tzinfo=None)
    if start < end:
        return start <= current < end
    return current >= start or current < end


def _parse_datetime(value: str) -> datetime | None:
    if not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.astimezone()
    return parsed.astimezone()


def _parse_time(value: str) -> time | None:
    if not value.strip():
        return None
    try:
        return time.fromisoformat(value.strip())
    except ValueError:
        return None


def _active_work_key(event: PetEvent) -> str:
    if event.kind not in CARE_ACTIVE_KINDS:
        return ""
    parts = (
        event.workspace,
        event.session_id,
        event.task_id,
        event.run_id,
        event.current_work_title,
    )
    return "|".join(_text(part) for part in parts)


def _ui_signature(record: Mapping[str, object]) -> str:
    bubble = record.get("bubble")
    profile = record.get("profile")
    return "|".join(
        (
            _text(record.get("event_id")),
            _text(record.get("kind")),
            _text(record.get("state")),
            _text(record.get("summary")),
            _text(bubble.get("text") if isinstance(bubble, Mapping) else ""),
            _text(profile.get("pet_id") if isinstance(profile, Mapping) else ""),
            "muted" if _bool(record.get("muted"), False) else "live",
        )
    )


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _bool(value: object, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        clean = value.strip().lower()
        if clean in {"1", "true", "yes", "on"}:
            return True
        if clean in {"0", "false", "no", "off"}:
            return False
    return fallback
