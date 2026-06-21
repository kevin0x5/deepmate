"""Daily capability maintenance for skill temperature governance."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from deepmate.capabilities.state import (
    CapabilityAssetState,
    CapabilitySource,
    CapabilityState,
    CapabilityStateStore,
    CapabilityTemperature,
    SkillTemperaturePolicy,
)
from deepmate.domain import CapabilityKind
from deepmate.skills import SkillCard
from deepmate.storage.jsonl import JsonlWriter
from deepmate.trace import TraceEvent, TraceRecorder

CAPABILITY_PROPOSALS_FILE = "proposals.jsonl"
CAPABILITY_MAINTENANCE_STATE_FILE = "maintenance_state.json"
ARCHIVE_GENERATED_SKILL = "archive_generated_skill"
PENDING = "pending"


@dataclass(frozen=True, slots=True)
class CapabilityMaintenanceProposal:
    """One low-risk capability governance proposal."""

    proposal_id: str
    type: str
    capability_id: str
    capability_name: str
    status: str
    reason: str
    evidence_refs: tuple[str, ...] = field(default_factory=tuple)
    created_at: str = ""

    def is_ready(self) -> bool:
        """Return whether this proposal has enough identity to persist."""
        return bool(
            self.proposal_id.strip()
            and self.type.strip()
            and self.capability_id.strip()
            and self.capability_name.strip()
            and self.status.strip()
            and self.reason.strip()
            and self.created_at.strip()
        )

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable record."""
        return {
            "proposal_id": self.proposal_id,
            "type": self.type,
            "capability_id": self.capability_id,
            "capability_name": self.capability_name,
            "status": self.status,
            "reason": self.reason,
            "evidence_refs": list(self.evidence_refs),
            "created_at": self.created_at,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "CapabilityMaintenanceProposal":
        """Build a proposal from a JSONL record."""
        return cls(
            proposal_id=_text(record.get("proposal_id")),
            type=_text(record.get("type")),
            capability_id=_text(record.get("capability_id")),
            capability_name=_text(record.get("capability_name")),
            status=_text(record.get("status")) or PENDING,
            reason=_text(record.get("reason")),
            evidence_refs=_string_tuple(record.get("evidence_refs")),
            created_at=_text(record.get("created_at")),
        )


@dataclass(frozen=True, slots=True)
class CapabilityMaintenanceResult:
    """Summary of one capability maintenance run."""

    reason: str
    skills_seen: int
    states_seen: int
    cooled: int
    proposals_created: int
    proposals_path: Path
    state_path: Path

    def is_noop(self) -> bool:
        """Return whether the run had nothing to maintain."""
        return self.reason == "no_capabilities"


class CapabilityProposalStore:
    """Append-only proposal store for capability maintenance."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def in_state_store(cls, state_store: CapabilityStateStore) -> "CapabilityProposalStore":
        """Return the proposal store next to a capability index."""
        return cls(state_store.path.parent / CAPABILITY_PROPOSALS_FILE)

    def append(self, proposal: CapabilityMaintenanceProposal) -> None:
        """Append one ready proposal."""
        if not proposal.is_ready():
            raise ValueError("capability proposal is not ready")
        JsonlWriter(self.path).append(proposal.to_record())

    def load(self) -> tuple[CapabilityMaintenanceProposal, ...]:
        """Load proposal records from JSONL, skipping malformed lines."""
        if not self.path.exists():
            return ()
        proposals: list[CapabilityMaintenanceProposal] = []
        with self.path.open(encoding="utf-8") as file:
            for line in file:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, Mapping):
                    continue
                proposal = CapabilityMaintenanceProposal.from_record(record)
                if proposal.is_ready():
                    proposals.append(proposal)
        return tuple(proposals)

    def has_pending(self, proposal_type: str, capability_id: str) -> bool:
        """Return whether an equivalent pending proposal already exists."""
        clean_type = proposal_type.strip()
        clean_capability_id = capability_id.strip()
        return any(
            proposal.type == clean_type
            and proposal.capability_id == clean_capability_id
            and proposal.status == PENDING
            for proposal in self.load()
        )


def run_daily_capability_maintenance(
    *,
    cards: Iterable[SkillCard],
    workspace: str | Path,
    state_store: CapabilityStateStore,
    proposal_store: CapabilityProposalStore | None = None,
    trace_recorder: TraceRecorder | None = None,
    now: datetime | None = None,
    policy: SkillTemperaturePolicy | None = None,
) -> CapabilityMaintenanceResult:
    """Run low-cost daily capability maintenance."""
    current_time = _normal_datetime(now)
    card_tuple = tuple(cards)
    previous_states = state_store.load()
    state_store.sync_workspace_skills(
        card_tuple,
        workspace,
        now=current_time,
        policy=policy,
    )
    states = state_store.cool_all_skills(now=current_time, policy=policy)
    cooled = _cooled_count(previous_states, states)
    proposal_store = proposal_store or CapabilityProposalStore.in_state_store(state_store)
    proposals = _generated_archive_proposals(
        states=states.values(),
        proposal_store=proposal_store,
        created_at=_isoformat(current_time),
    )
    for proposal in proposals:
        proposal_store.append(proposal)
    reason = _maintenance_reason(
        skills_seen=len(card_tuple),
        states_seen=len(states),
        cooled=cooled,
        proposals_created=len(proposals),
    )
    _write_maintenance_state(
        state_store.path.parent / CAPABILITY_MAINTENANCE_STATE_FILE,
        {
            "last_run_at": _isoformat(current_time),
            "reason": reason,
            "skills_seen": len(card_tuple),
            "states_seen": len(states),
            "cooled": cooled,
            "proposals_created": len(proposals),
        },
    )
    _record_trace(
        trace_recorder,
        reason=reason,
        state_store=state_store,
        proposals_path=proposal_store.path,
        skills_seen=len(card_tuple),
        states_seen=len(states),
        cooled=cooled,
        proposals=proposals,
    )
    return CapabilityMaintenanceResult(
        reason=reason,
        skills_seen=len(card_tuple),
        states_seen=len(states),
        cooled=cooled,
        proposals_created=len(proposals),
        proposals_path=proposal_store.path,
        state_path=state_store.path,
    )


def _generated_archive_proposals(
    states: Iterable[CapabilityState],
    proposal_store: CapabilityProposalStore,
    created_at: str,
) -> tuple[CapabilityMaintenanceProposal, ...]:
    proposals: list[CapabilityMaintenanceProposal] = []
    for state in states:
        if not _needs_generated_archive_proposal(state):
            continue
        if proposal_store.has_pending(ARCHIVE_GENERATED_SKILL, state.capability_id):
            continue
        proposals.append(
            CapabilityMaintenanceProposal(
                proposal_id=uuid4().hex,
                type=ARCHIVE_GENERATED_SKILL,
                capability_id=state.capability_id,
                capability_name=state.name,
                status=PENDING,
                reason=(
                    "Generated skill is cold and active; review whether it should be "
                    "archived from the default skill set."
                ),
                evidence_refs=(
                    f"capability_id={state.capability_id}",
                    f"temperature={state.temperature.value}",
                    f"source={state.source.value}",
                    f"asset_state={state.asset_state.value}",
                    f"last_used_at={state.last_used_at}",
                    f"invocation_count={state.invocation_count}",
                ),
                created_at=created_at,
            )
        )
    return tuple(proposals)


def _needs_generated_archive_proposal(state: CapabilityState) -> bool:
    return (
        state.kind == CapabilityKind.SKILL
        and state.source == CapabilitySource.GENERATED
        and state.temperature == CapabilityTemperature.COLD
        and state.asset_state == CapabilityAssetState.ACTIVE
    )


def _cooled_count(
    before: Mapping[str, CapabilityState],
    after: Mapping[str, CapabilityState],
) -> int:
    return sum(
        1
        for capability_id, state in after.items()
        if capability_id in before
        and before[capability_id].temperature != state.temperature
    )


def _maintenance_reason(
    *,
    skills_seen: int,
    states_seen: int,
    cooled: int,
    proposals_created: int,
) -> str:
    if states_seen == 0 and skills_seen == 0:
        return "no_capabilities"
    if proposals_created:
        return "proposals_created"
    if cooled:
        return "cooled"
    return "completed"


def _record_trace(
    recorder: TraceRecorder | None,
    *,
    reason: str,
    state_store: CapabilityStateStore,
    proposals_path: Path,
    skills_seen: int,
    states_seen: int,
    cooled: int,
    proposals: tuple[CapabilityMaintenanceProposal, ...],
) -> None:
    if recorder is None:
        return
    recorder.record(
        TraceEvent(
            kind="capability_maintenance_completed",
            summary=f"Capability maintenance completed: {reason}.",
            refs=(
                f"reason={reason}",
                f"skills_seen={skills_seen}",
                f"states_seen={states_seen}",
                f"cooled={cooled}",
                f"proposals_created={len(proposals)}",
                f"state_path={state_store.path}",
                f"proposals_path={proposals_path}",
                *(f"proposal_id={proposal.proposal_id}" for proposal in proposals),
            ),
        )
    )


def _write_maintenance_state(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(tmp_name, path)
    finally:
        tmp_path = Path(tmp_name)
        if tmp_path.exists():
            tmp_path.unlink()


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item.strip())


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _normal_datetime(value: datetime | None) -> datetime:
    current = value or datetime.now(UTC)
    if current.tzinfo is None:
        return current.replace(tzinfo=UTC)
    return current.astimezone(UTC)


def _isoformat(value: datetime) -> str:
    return _normal_datetime(value).replace(microsecond=0).isoformat()
