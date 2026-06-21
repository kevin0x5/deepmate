"""Local state files for the desktop pet companion."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deepmate.pet.events import PetEvent, PetVisualState
from deepmate.storage import JsonlWriter, atomic_write_json, file_lock

PET_PROFILE_FILE = "pet_profile.json"
PET_STATE_FILE = "pet_state.json"
PET_UI_STATE_FILE = "ui_state.json"
PET_CURRENT_STATE_FILE = "current_state.json"
PET_EVENTS_FILE = "events.jsonl"
PET_ACTIONS_FILE = "actions.jsonl"
PET_ACTIONS_STATE_FILE = "actions_state.json"
PET_LEARNING_STATE_FILE = "pet_learning_state.json"
PET_COPY_CACHE_FILE = "pet_copy_cache.json"

PET_PRESETS: Mapping[str, tuple[str, str, str]] = {
    "dog": ("dog-happy", "dog", "happy"),
    "cat": ("cat-lazy", "cat", "lazy"),
    "squirrel": ("squirrel-lively", "squirrel", "lively"),
    "penguin": ("penguin-naive", "penguin", "naive"),
}


@dataclass(frozen=True, slots=True)
class PetProfile:
    """User-selected pet appearance and behavior settings."""

    pet_id: str = "dog-happy"
    species: str = "dog"
    name: str = ""
    style: str = "happy"
    bubble_generation: str = "smart"
    learning_mode: str = "off"
    proactive_care: bool = True
    startup: bool = False
    report_cadence: str = "normal"
    quiet_hours: Mapping[str, str] = field(
        default_factory=lambda: {"start": "22:30", "end": "08:30"}
    )
    muted_until: str = ""

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable profile record."""
        return {
            "pet_id": self.pet_id,
            "species": self.species,
            "name": self.name,
            "style": self.style,
            "bubble_generation": self.bubble_generation,
            "learning_mode": self.learning_mode,
            "proactive_care": self.proactive_care,
            "startup": self.startup,
            "report_cadence": self.report_cadence,
            "quiet_hours": dict(self.quiet_hours),
            "muted_until": self.muted_until,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "PetProfile":
        """Build a profile from a JSON-like record."""
        quiet_hours = record.get("quiet_hours", {})
        return cls(
            pet_id=_text(record.get("pet_id")) or "dog-happy",
            species=_text(record.get("species")) or "dog",
            name=_text(record.get("name")),
            style=_text(record.get("style")) or "happy",
            bubble_generation=_choice(
                record.get("bubble_generation"), {"smart", "frugal"}, "smart"
            ),
            learning_mode=_choice(
                record.get("learning_mode"), {"off", "low", "standard"}, "off"
            ),
            proactive_care=_bool(record.get("proactive_care"), True),
            startup=_bool(record.get("startup"), False),
            report_cadence=_choice(
                record.get("report_cadence"), {"quiet", "normal", "active"}, "normal"
            ),
            quiet_hours=quiet_hours if isinstance(quiet_hours, Mapping) else {},
            muted_until=_text(record.get("muted_until")),
        )


@dataclass(frozen=True, slots=True)
class PetUserAction:
    """One action written by the pet host for Deepmate to process."""

    action: str
    created_at: str
    payload: Mapping[str, object] = field(default_factory=dict)

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable action record."""
        return {
            "action": self.action.strip(),
            "created_at": self.created_at.strip(),
            "payload": dict(self.payload),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "PetUserAction":
        """Build an action from a JSON-like record."""
        payload = record.get("payload", {})
        return cls(
            action=_text(record.get("action")),
            created_at=_text(record.get("created_at")),
            payload=payload if isinstance(payload, Mapping) else {},
        )


class PetStateStore:
    """Small file-backed store shared by Deepmate and the desktop pet host."""

    def __init__(self, data_dir: str | Path) -> None:
        self.root = Path(data_dir) / "pet"

    @classmethod
    def in_data_dir(cls, data_dir: str | Path) -> "PetStateStore":
        """Return a store rooted in Deepmate's runtime data directory."""
        return cls(data_dir)

    @property
    def profile_path(self) -> Path:
        return self.root / PET_PROFILE_FILE

    @property
    def state_path(self) -> Path:
        return self.root / PET_STATE_FILE

    @property
    def current_state_path(self) -> Path:
        return self.root / PET_CURRENT_STATE_FILE

    @property
    def ui_state_path(self) -> Path:
        return self.root / PET_UI_STATE_FILE

    @property
    def events_path(self) -> Path:
        return self.root / PET_EVENTS_FILE

    @property
    def actions_path(self) -> Path:
        return self.root / PET_ACTIONS_FILE

    @property
    def actions_state_path(self) -> Path:
        return self.root / PET_ACTIONS_STATE_FILE

    @property
    def learning_state_path(self) -> Path:
        return self.root / PET_LEARNING_STATE_FILE

    @property
    def copy_cache_path(self) -> Path:
        return self.root / PET_COPY_CACHE_FILE

    def load_profile(self) -> PetProfile:
        """Load the selected pet profile or return the default dog."""
        record = _read_json_object(self.profile_path)
        return PetProfile.from_record(record) if record else default_pet_profile()

    def save_profile(self, profile: PetProfile) -> None:
        """Persist the selected pet profile."""
        with file_lock(self.profile_path):
            atomic_write_json(self.profile_path, profile.to_record())

    def select_pet(self, pet: str) -> PetProfile:
        """Select a built-in pet by species or preset id."""
        profile = default_pet_profile(pet)
        self.save_profile(profile)
        return profile

    def load_current_state(self) -> dict[str, object]:
        """Load the current display state."""
        return _read_json_object(self.current_state_path)

    def save_current_state(self, event: PetEvent) -> dict[str, object]:
        """Persist an event as the current pet state and append it to event log."""
        ready = event.normalized()
        record = ready.to_record()
        with file_lock(self.current_state_path):
            atomic_write_json(self.current_state_path, record)
        self.append_event(ready)
        return record

    def append_event(self, event: PetEvent) -> None:
        """Append a pet event to events.jsonl."""
        ready = event.normalized()
        if not ready.is_ready():
            return
        JsonlWriter(self.events_path).append(ready.to_record())

    def append_action(self, action: PetUserAction) -> None:
        """Append one user action from the pet host."""
        if not action.action.strip():
            return
        JsonlWriter(self.actions_path).append(action.to_record())

    def pending_actions(self, *, limit: int = 20) -> tuple[tuple[int, PetUserAction], ...]:
        """Return unprocessed user actions with their 1-based JSONL line indexes."""
        cursor = _int(_read_json_object(self.actions_state_path).get("processed_count"), 0)
        if not self.actions_path.exists():
            return ()
        pending: list[tuple[int, PetUserAction]] = []
        try:
            with self.actions_path.open(encoding="utf-8") as file:
                for index, line in enumerate(file, start=1):
                    if index <= cursor:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, Mapping):
                        continue
                    action = PetUserAction.from_record(payload)
                    if action.action.strip():
                        pending.append((index, action))
                    if len(pending) >= max(1, limit):
                        break
        except OSError:
            return ()
        return tuple(pending)

    def mark_actions_processed(self, processed_count: int) -> None:
        """Record how far Deepmate has consumed actions.jsonl."""
        if processed_count <= 0:
            return
        state = _read_json_object(self.actions_state_path)
        current = _int(state.get("processed_count"), 0)
        if processed_count <= current:
            return
        with file_lock(self.actions_state_path):
            atomic_write_json(
                self.actions_state_path,
                {"processed_count": processed_count},
            )

    def load_pet_state(self) -> dict[str, object]:
        """Load UI-only pet host state."""
        return _read_json_object(self.state_path)

    def save_pet_state(self, state: Mapping[str, object]) -> None:
        """Persist UI-only pet host state."""
        with file_lock(self.state_path):
            atomic_write_json(self.state_path, dict(state))

    def load_ui_state(self) -> dict[str, object]:
        """Load the frontend-ready pet display state."""
        return _read_json_object(self.ui_state_path)

    def save_ui_state(self, state: Mapping[str, object]) -> None:
        """Persist the frontend-ready pet display state."""
        with file_lock(self.ui_state_path):
            atomic_write_json(self.ui_state_path, dict(state))

    def load_learning_state(self) -> dict[str, object]:
        """Load learning-mode feedback and cache state."""
        return _read_json_object(self.learning_state_path)

    def save_learning_state(self, state: Mapping[str, object]) -> None:
        """Persist learning-mode feedback and cache state."""
        with file_lock(self.learning_state_path):
            atomic_write_json(self.learning_state_path, dict(state))

    def load_copy_cache(self) -> dict[str, object]:
        """Load cached visible pet copy."""
        return _read_json_object(self.copy_cache_path)

    def save_copy_cache(self, cache: Mapping[str, object]) -> None:
        """Persist cached visible pet copy."""
        with file_lock(self.copy_cache_path):
            atomic_write_json(self.copy_cache_path, dict(cache))

    def update_copy_cache(self, key: str, value: Mapping[str, object], *, limit: int = 200) -> None:
        """Update one cached pet copy entry under one file lock."""
        clean_key = key.strip()
        if not clean_key:
            return
        with file_lock(self.copy_cache_path):
            cache = _read_json_object(self.copy_cache_path)
            cache.pop(clean_key, None)
            cache[clean_key] = dict(value)
            if len(cache) > max(1, limit):
                keys = list(cache)[-max(1, limit) :]
                cache = {item: cache[item] for item in keys}
            atomic_write_json(self.copy_cache_path, cache)

    def offline_state(self) -> dict[str, object]:
        """Return a displayable offline state without writing it."""
        existing = self.load_current_state()
        if existing:
            return existing
        return {
            "event_id": "pet_current_work_idle",
            "kind": "current_work.idle",
            "summary": "Deepmate is ready.",
            "state": PetVisualState.IDLE.value,
            "current_work_title": "",
            "actions": [],
        }


def default_pet_profile(pet: str = "dog") -> PetProfile:
    """Return a default built-in pet profile."""
    clean = pet.strip().lower().replace("_", "-") if isinstance(pet, str) else "dog"
    preset = PET_PRESETS.get(clean)
    if preset is None:
        preset = next(
            (
                candidate
                for candidate in PET_PRESETS.values()
                if clean == candidate[0]
            ),
            PET_PRESETS["dog"],
        )
    pet_id, species, style = preset
    return PetProfile(pet_id=pet_id, species=species, style=style)


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _choice(value: object, allowed: set[str], fallback: str) -> str:
    clean = _text(value).lower()
    return clean if clean in allowed else fallback


def _bool(value: object, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        clean = value.strip().lower()
        if clean in {"true", "1", "yes", "on"}:
            return True
        if clean in {"false", "0", "no", "off"}:
            return False
    return fallback


def _int(value: object, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return fallback
    return fallback
