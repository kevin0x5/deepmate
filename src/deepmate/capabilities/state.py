"""Persistent capability runtime state.

This sidecar keeps Deepmate-specific governance data out of SKILL.md, MCP
schemas, and native tool schemas.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from deepmate.domain import CapabilityKind, ProfileRef
from deepmate.foundation import (
    non_negative_int,
    normalize_name,
    normal_datetime,
    utc_isoformat,
)
from deepmate.storage import atomic_write_json, file_lock

if TYPE_CHECKING:
    from deepmate.skills.catalog import SkillCard

CAPABILITY_INDEX_FILE = "capability_index.json"
DEFAULT_UNUSED_DAYS_TO_WARM = 7
DEFAULT_UNUSED_DAYS_TO_COLD = 14


class CapabilitySource(StrEnum):
    """Who owns the capability asset for lifecycle purposes."""

    LOCAL = "local"
    IMPORTED = "imported"
    GENERATED = "generated"


class CapabilityScope(StrEnum):
    """Where the capability is discovered or stored."""

    WORKSPACE = "workspace"
    PROFILE = "profile"


class CapabilityTemperature(StrEnum):
    """How active a capability is for default exposure."""

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


class CapabilityAssetState(StrEnum):
    """Whether the capability is active in Deepmate's local control plane."""

    ACTIVE = "active"
    ARCHIVED = "archived"


@dataclass(frozen=True, slots=True)
class SkillTemperaturePolicy:
    """Deterministic cooling thresholds for skills."""

    unused_days_to_warm: int = DEFAULT_UNUSED_DAYS_TO_WARM
    unused_days_to_cold: int = DEFAULT_UNUSED_DAYS_TO_COLD

    def is_ready(self) -> bool:
        """Return whether the policy has usable monotonic thresholds."""
        return (
            self.unused_days_to_warm > 0
            and self.unused_days_to_cold >= self.unused_days_to_warm
        )


@dataclass(frozen=True, slots=True)
class CapabilityState:
    """One persisted capability state entry."""

    capability_id: str
    kind: CapabilityKind
    name: str
    path_or_ref: str
    source: CapabilitySource = CapabilitySource.LOCAL
    scope: CapabilityScope = CapabilityScope.WORKSPACE
    temperature: CapabilityTemperature = CapabilityTemperature.HOT
    hidden: bool = False
    asset_state: CapabilityAssetState = CapabilityAssetState.ACTIVE
    created_at: str = ""
    updated_at: str = ""
    last_seen_at: str = ""
    last_used_at: str = ""
    invocation_count: int = 0
    last_selected_at: str = ""

    def is_ready(self) -> bool:
        """Return whether the state has enough identity to persist."""
        return bool(
            self.capability_id.strip()
            and self.kind.value.strip()
            and self.name.strip()
            and self.source.value.strip()
            and self.scope.value.strip()
            and self.temperature.value.strip()
            and self.asset_state.value.strip()
            and self.created_at.strip()
            and self.updated_at.strip()
        )

    def is_exposed_by_default(self) -> bool:
        """Return whether this state enters the default model capability surface."""
        return (
            self.asset_state == CapabilityAssetState.ACTIVE
            and not self.hidden
            and self.temperature != CapabilityTemperature.COLD
        )

    def exposure(self) -> str:
        """Return the derived exposure mode."""
        if not self.is_exposed_by_default():
            return "not-loaded"
        if self.temperature == CapabilityTemperature.WARM:
            return "name-only"
        return "name+description"

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable record."""
        return {
            "capability_id": self.capability_id,
            "kind": self.kind.value,
            "name": self.name,
            "path_or_ref": self.path_or_ref,
            "source": self.source.value,
            "scope": self.scope.value,
            "temperature": self.temperature.value,
            "hidden": self.hidden,
            "asset_state": self.asset_state.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_seen_at": self.last_seen_at,
            "last_used_at": self.last_used_at,
            "invocation_count": self.invocation_count,
            "last_selected_at": self.last_selected_at,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "CapabilityState":
        """Build one state entry from a stored JSON record."""
        return cls(
            capability_id=_text(record.get("capability_id")),
            kind=_capability_kind(record.get("kind")),
            name=_text(record.get("name")),
            path_or_ref=_text(record.get("path_or_ref")),
            source=_capability_source(record.get("source")),
            scope=_capability_scope(record.get("scope")),
            temperature=_capability_temperature(record.get("temperature")),
            hidden=bool(record.get("hidden", False)),
            asset_state=_capability_asset_state(record.get("asset_state")),
            created_at=_text(record.get("created_at")),
            updated_at=_text(record.get("updated_at")),
            last_seen_at=_text(record.get("last_seen_at")),
            last_used_at=_text(record.get("last_used_at")),
            invocation_count=_non_negative_int(record.get("invocation_count")),
            last_selected_at=_text(record.get("last_selected_at")),
        )


class CapabilityStateStore:
    """JSON sidecar store for profile-scoped capability state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def in_data_dir(
        cls,
        data_dir: str | Path,
        profile: ProfileRef | str,
    ) -> "CapabilityStateStore":
        """Return the profile-local capability state store."""
        profile_name = profile.name if isinstance(profile, ProfileRef) else str(profile)
        clean_profile = profile_name.strip() or "default"
        return cls(Path(data_dir) / "capabilities" / clean_profile / CAPABILITY_INDEX_FILE)

    def load(self) -> dict[str, CapabilityState]:
        """Load all persisted states keyed by capability id."""
        return self._load_unlocked()

    def _load_unlocked(self) -> dict[str, CapabilityState]:
        """Load all persisted states without acquiring a mutation lock."""
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("capability index must be a JSON object")
        raw_items = data.get("capabilities", [])
        if not isinstance(raw_items, list):
            raise ValueError("capability index requires a capabilities list")
        states: dict[str, CapabilityState] = {}
        for raw_item in raw_items:
            if not isinstance(raw_item, Mapping):
                continue
            state = CapabilityState.from_record(raw_item)
            if state.is_ready():
                states[state.capability_id] = state
        return states

    def save(self, states: Mapping[str, CapabilityState]) -> None:
        """Atomically persist all state entries."""
        with file_lock(self.path):
            self._save_unlocked(states)

    def _save_unlocked(self, states: Mapping[str, CapabilityState]) -> None:
        """Persist all state entries while the caller holds the mutation lock."""
        ready_states = sorted(
            (state for state in states.values() if state.is_ready()),
            key=lambda state: state.capability_id,
        )
        payload = {
            "version": 1,
            "capabilities": [state.to_record() for state in ready_states],
        }
        atomic_write_json(self.path, payload)

    def sync_workspace_skills(
        self,
        cards: Iterable[SkillCard],
        workspace: str | Path,
        now: datetime | None = None,
        policy: SkillTemperaturePolicy | None = None,
    ) -> dict[str, CapabilityState]:
        """Ensure discovered workspace skills have state and apply deterministic cooling."""
        current_time = _normal_datetime(now)
        policy = policy or SkillTemperaturePolicy()
        if not policy.is_ready():
            raise ValueError("skill temperature policy thresholds are invalid")
        with file_lock(self.path):
            states = self._load_unlocked()
            root = Path(workspace)
            for card in cards:
                capability_id = skill_capability_id(card.name, CapabilityScope.WORKSPACE)
                previous = states.get(capability_id)
                state = previous or _new_skill_state(card, root, current_time)
                state = replace(
                    state,
                    name=card.name.strip(),
                    path_or_ref=_workspace_relative(card.path, root),
                    last_seen_at=_isoformat(current_time),
                    updated_at=_isoformat(current_time),
                )
                states[capability_id] = _cooled_state(state, current_time, policy)
            self._save_unlocked(states)
        return states

    def cool_all_skills(
        self,
        now: datetime | None = None,
        policy: SkillTemperaturePolicy | None = None,
    ) -> dict[str, CapabilityState]:
        """Apply deterministic cooling to all persisted skill states."""
        current_time = _normal_datetime(now)
        policy = policy or SkillTemperaturePolicy()
        if not policy.is_ready():
            raise ValueError("skill temperature policy thresholds are invalid")
        with file_lock(self.path):
            states = {
                capability_id: _cooled_state(state, current_time, policy)
                for capability_id, state in self._load_unlocked().items()
            }
            self._save_unlocked(states)
        return states

    def skill_states_by_name(self) -> dict[str, CapabilityState]:
        """Return skill states keyed by normalized skill name."""
        return {
            _normalize_name(state.name): state
            for state in self.load().values()
            if state.kind == CapabilityKind.SKILL
        }

    def record_skill_selected(
        self,
        name: str,
        now: datetime | None = None,
    ) -> CapabilityState:
        """Record explicit skill selection and reset the skill to hot."""
        current_time = _normal_datetime(now)
        with file_lock(self.path):
            states = self._load_unlocked()
            capability_id = skill_capability_id(name, CapabilityScope.WORKSPACE)
            state = states.get(capability_id) or _placeholder_skill_state(
                name=name,
                current_time=current_time,
            )
            state = replace(
                state,
                temperature=CapabilityTemperature.HOT,
                asset_state=CapabilityAssetState.ACTIVE,
                hidden=False,
                updated_at=_isoformat(current_time),
                last_used_at=_isoformat(current_time),
                last_selected_at=_isoformat(current_time),
                invocation_count=state.invocation_count + 1,
            )
            states[capability_id] = state
            self._save_unlocked(states)
        return state

    def record_skill_installed(
        self,
        card: SkillCard,
        workspace: str | Path,
        now: datetime | None = None,
        source: CapabilitySource = CapabilitySource.IMPORTED,
    ) -> CapabilityState:
        """Record an installed skill as hot without polluting its SKILL.md."""
        current_time = _normal_datetime(now)
        root = Path(workspace)
        with file_lock(self.path):
            states = self._load_unlocked()
            capability_id = skill_capability_id(card.name, CapabilityScope.WORKSPACE)
            previous = states.get(capability_id)
            timestamp = _isoformat(current_time)
            state = previous or _new_skill_state(card, root, current_time)
            state = replace(
                state,
                name=card.name.strip(),
                path_or_ref=_workspace_relative(card.path, root),
                source=source,
                temperature=CapabilityTemperature.HOT,
                hidden=False,
                asset_state=CapabilityAssetState.ACTIVE,
                updated_at=timestamp,
                last_seen_at=timestamp,
            )
            states[capability_id] = state
            self._save_unlocked(states)
        return state

    def set_skill_state(
        self,
        name: str,
        action: str,
        now: datetime | None = None,
    ) -> CapabilityState:
        """Apply one manual skill state action."""
        clean_action = action.strip().lower()
        if clean_action not in {"heat", "cool", "hide", "restore"}:
            raise ValueError("skill state action must be heat, cool, hide, or restore")
        current_time = _normal_datetime(now)
        with file_lock(self.path):
            states = self._load_unlocked()
            capability_id = skill_capability_id(name, CapabilityScope.WORKSPACE)
            state = states.get(capability_id)
            if state is None:
                raise ValueError(f"skill state not found: {name}")
            if clean_action in {"heat", "restore"}:
                state = replace(
                    state,
                    temperature=CapabilityTemperature.HOT,
                    hidden=False,
                    asset_state=CapabilityAssetState.ACTIVE,
                )
            elif clean_action == "cool":
                state = replace(state, temperature=_next_cooler_temperature(state.temperature))
            elif clean_action == "hide":
                state = replace(state, hidden=True)
            state = replace(state, updated_at=_isoformat(current_time))
            states[capability_id] = state
            self._save_unlocked(states)
        return state


def skill_capability_id(
    name: str,
    scope: CapabilityScope = CapabilityScope.WORKSPACE,
) -> str:
    """Return the stable state id for a skill name."""
    clean_name = _normalize_name(name)
    if not clean_name:
        raise ValueError("skill name cannot be empty")
    return f"{CapabilityKind.SKILL.value}:{scope.value}:{clean_name}"


def _new_skill_state(
    card: SkillCard,
    workspace: Path,
    current_time: datetime,
) -> CapabilityState:
    timestamp = _isoformat(current_time)
    return CapabilityState(
        capability_id=skill_capability_id(card.name, CapabilityScope.WORKSPACE),
        kind=CapabilityKind.SKILL,
        name=card.name.strip(),
        path_or_ref=_workspace_relative(card.path, workspace),
        created_at=timestamp,
        updated_at=timestamp,
        last_seen_at=timestamp,
    )


def _placeholder_skill_state(name: str, current_time: datetime) -> CapabilityState:
    timestamp = _isoformat(current_time)
    return CapabilityState(
        capability_id=skill_capability_id(name, CapabilityScope.WORKSPACE),
        kind=CapabilityKind.SKILL,
        name=name.strip(),
        path_or_ref="",
        created_at=timestamp,
        updated_at=timestamp,
        last_seen_at=timestamp,
    )


def _cooled_state(
    state: CapabilityState,
    now: datetime,
    policy: SkillTemperaturePolicy,
) -> CapabilityState:
    if (
        state.kind != CapabilityKind.SKILL
        or state.asset_state != CapabilityAssetState.ACTIVE
    ):
        return state
    activity_time = (
        _parse_datetime(state.last_used_at)
        or _parse_datetime(state.created_at)
        or now
    )
    elapsed = now - _normal_datetime(activity_time)
    temperature = state.temperature
    if elapsed >= timedelta(days=policy.unused_days_to_cold):
        temperature = CapabilityTemperature.COLD
    elif elapsed >= timedelta(days=policy.unused_days_to_warm):
        temperature = CapabilityTemperature.WARM
    if temperature == state.temperature:
        return state
    return replace(state, temperature=temperature, updated_at=_isoformat(now))


def _next_cooler_temperature(
    temperature: CapabilityTemperature,
) -> CapabilityTemperature:
    if temperature == CapabilityTemperature.HOT:
        return CapabilityTemperature.WARM
    return CapabilityTemperature.COLD


def _workspace_relative(path: Path, workspace: Path) -> str:
    try:
        return str(path.relative_to(workspace))
    except ValueError:
        return str(path)


def _normalize_name(name: str) -> str:
    return normalize_name(name)


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _non_negative_int(value: object) -> int:
    return non_negative_int(value)


def _capability_kind(value: object) -> CapabilityKind:
    try:
        return CapabilityKind(_text(value))
    except ValueError:
        return CapabilityKind.SKILL


def _capability_source(value: object) -> CapabilitySource:
    try:
        return CapabilitySource(_text(value))
    except ValueError:
        return CapabilitySource.LOCAL


def _capability_scope(value: object) -> CapabilityScope:
    try:
        return CapabilityScope(_text(value))
    except ValueError:
        return CapabilityScope.WORKSPACE


def _capability_temperature(value: object) -> CapabilityTemperature:
    try:
        return CapabilityTemperature(_text(value))
    except ValueError:
        return CapabilityTemperature.HOT


def _capability_asset_state(value: object) -> CapabilityAssetState:
    try:
        return CapabilityAssetState(_text(value))
    except ValueError:
        return CapabilityAssetState.ACTIVE


def _normal_datetime(value: datetime | None) -> datetime:
    return normal_datetime(value)


def _parse_datetime(value: str) -> datetime | None:
    if not value.strip():
        return None
    try:
        return _normal_datetime(
            datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        )
    except ValueError:
        return None


def _isoformat(value: datetime) -> str:
    return utc_isoformat(value)
