"""Low-frequency self-evolution maintenance."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping

from deepmate.capabilities.state import (
    CapabilityAssetState,
    CapabilitySource,
    CapabilityState,
    CapabilityStateStore,
    CapabilityTemperature,
)
from deepmate.domain import ProfileRef
from deepmate.evolution.behavior import (
    extract_behavior_hints,
    normalize_behavior_hints,
    profile_behavior_path,
    replace_behavior_hints_section,
    workspace_behavior_path,
)
from deepmate.evolution.changes import (
    EvolutionChange,
    EvolutionChangeStore,
    applied_change,
)
from deepmate.evolution.evidence_mining import (
    EvolutionEvidenceBatch,
    collect_evidence_from_records,
    user_correction_candidates,
    workflow_candidates,
)
from deepmate.evolution.failure_patterns import (
    FailurePatternGuard,
    FailurePatternStore,
    update_failure_patterns_from_evidence,
)
from deepmate.evolution.generated_skills import (
    GeneratedSkillApplyResult,
    archive_generated_skill,
    apply_generated_skill_draft,
    generated_skill_drafts_from_workflows,
)
from deepmate.foundation import utc_isoformat
from deepmate.storage import JsonlWriter, atomic_write_json, atomic_write_text, file_lock
from deepmate.storage.session_store import SessionRecord, SessionStore
from deepmate.trace import TraceEvent, TraceRecorder

EVOLUTION_MAINTENANCE_STATE_FILE = "maintenance_state.json"
EVOLUTION_FITNESS_METRICS_FILE = "fitness_metrics.jsonl"
SENSITIVE_BEHAVIOR_HINT_PATTERNS = (
    re.compile(
        r"\b(?:api[ _-]?key|access[ _-]?token|password|passwd|secret|"
        r"user[ _-]?name|account(?:[ _-]?id)?|phone|mobile)\b\s*[:=]",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:bearer\s+|sk-|ghp_|github_pat_)[A-Za-z0-9._-]{8,}", re.IGNORECASE),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    re.compile(r"(?i)\b(?:[0-9a-f]{1,4}:){2,}[0-9a-f:]{1,4}\b"),
    re.compile(r"(?<!\d)(?:\+\d[\d ()-]{7,}\d|\d{10,15})(?!\d)"),
    re.compile(r"(?:/Users/|/home/)[^\s]+"),
    re.compile(r"\b[A-Za-z]:\\Users\\[^\s]+", re.IGNORECASE),
)


@dataclass(frozen=True, slots=True)
class EvolutionMaintenanceState:
    """Incremental cursor for self-evolution maintenance."""

    last_run_at: str = ""
    last_status: str = ""
    last_reason: str = ""

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable maintenance state record."""
        return {
            "last_run_at": self.last_run_at,
            "last_status": self.last_status,
            "last_reason": self.last_reason,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "EvolutionMaintenanceState":
        """Build state from a stored JSON object."""
        return cls(
            last_run_at=_text(record.get("last_run_at")),
            last_status=_text(record.get("last_status")),
            last_reason=_text(record.get("last_reason")),
        )


@dataclass(frozen=True, slots=True)
class EvolutionFitnessMetrics:
    """Small metrics snapshot for one evolution maintenance run."""

    recorded_at: str
    window_start: str
    window_end: str
    token_cost: int = 0
    loaded_skills_count: int = 0
    used_skills_count: int = 0
    tool_failure_count: int = 0
    user_correction_count: int = 0
    generated_skill_apply_count: int = 0
    rollback_count: int = 0

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable metrics record."""
        return {
            "recorded_at": self.recorded_at,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "token_cost": self.token_cost,
            "loaded_skills_count": self.loaded_skills_count,
            "used_skills_count": self.used_skills_count,
            "tool_failure_count": self.tool_failure_count,
            "user_correction_count": self.user_correction_count,
            "generated_skill_apply_count": self.generated_skill_apply_count,
            "rollback_count": self.rollback_count,
        }

    def is_empty(self) -> bool:
        """Return whether the metrics snapshot has no observed signal."""
        return not any(
            (
                self.token_cost,
                self.loaded_skills_count,
                self.used_skills_count,
                self.tool_failure_count,
                self.user_correction_count,
                self.generated_skill_apply_count,
                self.rollback_count,
            )
        )


@dataclass(frozen=True, slots=True)
class EvolutionMaintenanceResult:
    """Summary of one evolution maintenance run."""

    ran: bool
    reason: str
    behavior_changes: int
    failure_patterns_updated: int
    generated_skill_changes: int
    capability_state_changes: int
    trace_records_seen: int
    sessions_seen: int
    activity_notes_seen: int
    capability_states_seen: int
    metrics: EvolutionFitnessMetrics
    applied_log_path: Path
    maintenance_state_path: Path
    metrics_path: Path
    changed_paths: tuple[Path, ...] = ()


def run_evolution_maintenance(
    *,
    workspace: str | Path,
    data_dir: str | Path,
    profile: ProfileRef,
    now: datetime | None = None,
    trace_path: str | Path | None = None,
    sessions_dir: str | Path | None = None,
    activity_dir: str | Path | None = None,
    evidence_records: tuple[Mapping[str, object], ...] = (),
    trace_recorder: TraceRecorder | None = None,
    force: bool = False,
) -> EvolutionMaintenanceResult:
    """Run one profile's maintenance exclusively."""
    profile_name = profile.name.strip() or "default"
    lock_path = Path(data_dir) / "evolution" / profile_name / ".maintenance"
    with file_lock(lock_path):
        return _run_evolution_maintenance_unlocked(
            workspace=workspace,
            data_dir=data_dir,
            profile=profile,
            now=now,
            trace_path=trace_path,
            sessions_dir=sessions_dir,
            activity_dir=activity_dir,
            evidence_records=evidence_records,
            trace_recorder=trace_recorder,
            force=force,
        )


def _run_evolution_maintenance_unlocked(
    *,
    workspace: str | Path,
    data_dir: str | Path,
    profile: ProfileRef,
    now: datetime | None = None,
    trace_path: str | Path | None = None,
    sessions_dir: str | Path | None = None,
    activity_dir: str | Path | None = None,
    evidence_records: tuple[Mapping[str, object], ...] = (),
    trace_recorder: TraceRecorder | None = None,
    force: bool = False,
) -> EvolutionMaintenanceResult:
    """Run deterministic self-evolution maintenance for new signals."""
    workspace_path = Path(workspace)
    data_path = Path(data_dir)
    current_time = _normal_datetime(now)
    current_iso = utc_isoformat(current_time)
    profile_name = profile.name.strip() or "default"
    state_path = data_path / "evolution" / profile_name / EVOLUTION_MAINTENANCE_STATE_FILE
    metrics_path = data_path / "evolution" / profile_name / EVOLUTION_FITNESS_METRICS_FILE
    state = _load_maintenance_state(state_path)
    window_start = _parse_datetime(state.last_run_at)
    store = EvolutionChangeStore.in_data_dir(data_path, profile)
    pattern_store = FailurePatternStore.in_data_dir(data_path, profile)
    capability_state_store = CapabilityStateStore.in_data_dir(data_path, profile)
    trace_records = _read_jsonl_records_since(trace_path, window_start)
    session_records = _read_session_records_since(
        sessions_dir,
        workspace_path,
        profile,
        window_start,
    )
    activity_paths = _activity_paths_since(activity_dir, window_start)
    session_evidence_records = _session_evidence_records(session_records)
    activity_evidence_records = _activity_evidence_records(activity_paths)
    behavior_paths = _behavior_paths_since(workspace_path, profile, window_start)
    capability_states = _capability_states_since(capability_state_store, window_start)
    applied_changes = _changes_since(store, window_start)
    explicit_records = tuple(evidence_records)
    has_new_signal = bool(
        force
        or explicit_records
        or trace_records
        or session_records
        or activity_paths
        or behavior_paths
        or capability_states
        or applied_changes
    )
    if not has_new_signal:
        metrics = _build_metrics(
            records=(),
            batch=EvolutionEvidenceBatch(),
            generated_results=(),
            changes=store.load(),
            window_start=state.last_run_at,
            window_end=current_iso,
            recorded_at=current_iso,
        )
        _write_maintenance_state(
            state_path,
            EvolutionMaintenanceState(
                last_run_at=current_iso,
                last_status="completed",
                last_reason="no_new_signals",
            ),
            metrics,
        )
        _record_trace(
            trace_recorder,
            completed=False,
            result_reason="no_new_signals",
            metrics=metrics,
            state_path=state_path,
            metrics_path=metrics_path,
            trace_records_seen=0,
            sessions_seen=0,
            activity_notes_seen=0,
            capability_states_seen=0,
            changed_paths=(),
        )
        return EvolutionMaintenanceResult(
            ran=False,
            reason="no_new_signals",
            behavior_changes=0,
            failure_patterns_updated=0,
            generated_skill_changes=0,
            capability_state_changes=0,
            trace_records_seen=0,
            sessions_seen=0,
            activity_notes_seen=0,
            capability_states_seen=0,
            metrics=metrics,
            applied_log_path=store.path,
            maintenance_state_path=state_path,
            metrics_path=metrics_path,
    )

    changed_paths: list[Path] = []
    changed_path_keys: set[str] = set()
    behavior_change_count = 0
    for target_path in (
        workspace_behavior_path(workspace_path),
        profile_behavior_path(workspace_path, profile),
    ):
        change = _normalize_existing_behavior_file(
            target_path=target_path,
            workspace=workspace_path,
            store=store,
            now=current_time,
        )
        if change is not None:
            if _remember_changed_path(changed_paths, changed_path_keys, target_path):
                behavior_change_count += 1
    evidence_records_all = (
        *trace_records,
        *explicit_records,
        *session_evidence_records,
        *activity_evidence_records,
    )
    evidence_batch = collect_evidence_from_records(evidence_records_all)
    behavior_changes = _apply_behavior_hints_from_evidence(
        workspace=workspace_path,
        data_dir=data_path,
        profile=profile,
        evidence_batch=evidence_batch,
        now=current_time,
    )
    for behavior_change, target_path in behavior_changes:
        if _remember_changed_path(
            changed_paths,
            changed_path_keys,
            target_path,
        ):
            behavior_change_count += 1
    updated_patterns = update_failure_patterns_from_evidence(
        store=pattern_store,
        user_corrections=evidence_batch.user_corrections,
        tool_failures=evidence_batch.tool_failures,
        now=current_time,
        change_store=store,
        workspace=workspace_path,
    )
    guard = FailurePatternGuard.from_store(pattern_store)
    generated_results: list[GeneratedSkillApplyResult] = []
    for draft in generated_skill_drafts_from_workflows(
        workflow_candidates(evidence_batch.workflows)
    ):
        result = apply_generated_skill_draft(
            draft=draft,
            workspace=workspace_path,
            data_dir=data_path,
            profile=profile,
            guard=guard,
            state_store=capability_state_store,
            change_store=store,
            now=current_time,
        )
        generated_results.append(result)
        if result.skill_path is not None and result.is_applied():
            _remember_changed_path(changed_paths, changed_path_keys, result.skill_path)
    for state_to_archive in _generated_cold_active_states(capability_states):
        result = archive_generated_skill(
            skill_name=state_to_archive.name,
            workspace=workspace_path,
            data_dir=data_path,
            profile=profile,
            state_store=capability_state_store,
            change_store=store,
            now=current_time,
        )
        generated_results.append(result)
        if result.skill_path is not None and result.is_applied():
            _remember_changed_path(changed_paths, changed_path_keys, result.skill_path)
    generated_skill_changes = sum(1 for result in generated_results if result.is_applied())
    metrics = _build_metrics(
        records=(*trace_records, *explicit_records),
        batch=evidence_batch,
        generated_results=tuple(generated_results),
        changes=store.load(),
        window_start=state.last_run_at,
        window_end=current_iso,
        recorded_at=current_iso,
    )
    JsonlWriter(metrics_path).append(metrics.to_record())
    reason = _maintenance_reason(
        behavior_changes=behavior_change_count,
        failure_patterns_updated=len(updated_patterns),
        generated_skill_changes=generated_skill_changes,
        metrics=metrics,
    )
    _write_maintenance_state(
        state_path,
        EvolutionMaintenanceState(
            last_run_at=current_iso,
            last_status="completed",
            last_reason=reason,
        ),
        metrics,
    )
    _record_trace(
        trace_recorder,
        completed=True,
        result_reason=reason,
        metrics=metrics,
        state_path=state_path,
        metrics_path=metrics_path,
        trace_records_seen=len(trace_records),
        sessions_seen=len(session_records),
        activity_notes_seen=len(activity_paths),
        capability_states_seen=len(capability_states),
        changed_paths=tuple(changed_paths),
    )
    return EvolutionMaintenanceResult(
        ran=True,
        reason=reason,
        behavior_changes=behavior_change_count,
        failure_patterns_updated=len(updated_patterns),
        generated_skill_changes=generated_skill_changes,
        capability_state_changes=sum(
            1
            for result in generated_results
            if result.is_applied() and result.reason == "generated_skill_archived"
        ),
        trace_records_seen=len(trace_records),
        sessions_seen=len(session_records),
        activity_notes_seen=len(activity_paths),
        capability_states_seen=len(capability_states),
        metrics=metrics,
        applied_log_path=store.path,
        maintenance_state_path=state_path,
        metrics_path=metrics_path,
        changed_paths=tuple(changed_paths),
    )


def apply_behavior_hint_change(
    *,
    workspace: str | Path,
    data_dir: str | Path,
    profile: ProfileRef,
    hints: tuple[str, ...],
    target_scope: str = "workspace",
    summary: str = "Updated behavior hints.",
    evidence_refs: tuple[str, ...] = (),
    now: datetime | None = None,
) -> EvolutionChange | None:
    """Add explicit behavior hints to one behavior.md and record the patch."""
    clean_hints = normalize_behavior_hints(hints)
    if not clean_hints:
        return None
    workspace_path = Path(workspace)
    clean_scope = target_scope.strip().lower() or "workspace"
    if clean_scope not in {"workspace", "profile"}:
        raise ValueError("behavior hint target_scope must be workspace or profile")
    target_path = (
        profile_behavior_path(workspace_path, profile)
        if clean_scope == "profile"
        else workspace_behavior_path(workspace_path)
    )
    with file_lock(target_path):
        old_exists = target_path.exists()
        old_content = target_path.read_text(encoding="utf-8") if old_exists else ""
        merged_hints = normalize_behavior_hints(
            (*extract_behavior_hints(old_content), *clean_hints)
        )
        new_content = replace_behavior_hints_section(old_content, merged_hints)
        return _apply_behavior_file_update(
            target_path=target_path,
            workspace=workspace_path,
            store=EvolutionChangeStore.in_data_dir(data_dir, profile),
            old_content=old_content,
            new_content=new_content,
            old_exists=old_exists,
            summary=summary,
            evidence_refs=evidence_refs,
            now=now or datetime.now().astimezone(),
        )


def _normalize_existing_behavior_file(
    *,
    target_path: Path,
    workspace: Path,
    store: EvolutionChangeStore,
    now: datetime,
) -> EvolutionChange | None:
    with file_lock(target_path):
        if not target_path.exists():
            return None
        old_content = target_path.read_text(encoding="utf-8")
        hints = extract_behavior_hints(old_content)
        if not hints:
            return None
        normalized_hints = normalize_behavior_hints(hints)
        new_content = replace_behavior_hints_section(old_content, normalized_hints)
        return _apply_behavior_file_update(
            target_path=target_path,
            workspace=workspace,
            store=store,
            old_content=old_content,
            new_content=new_content,
            old_exists=True,
            summary="Normalized behavior hints.",
            evidence_refs=("maintenance=behavior_normalization",),
            now=now,
        )


def _apply_behavior_hints_from_evidence(
    *,
    workspace: Path,
    data_dir: Path,
    profile: ProfileRef,
    evidence_batch: EvolutionEvidenceBatch,
    now: datetime,
) -> tuple[tuple[EvolutionChange, Path], ...]:
    scoped: dict[str, list[tuple[str, tuple[str, ...]]]] = {
        "workspace": [],
        "profile": [],
    }
    for aggregate in user_correction_candidates(evidence_batch.user_corrections):
        correction = _best_user_correction_hint(aggregate.examples)
        if not correction:
            continue
        scope = _behavior_hint_scope(correction, evidence_batch)
        scoped[scope].append((correction, aggregate.source_refs))
    changes: list[tuple[EvolutionChange, Path]] = []
    for scope, candidates in scoped.items():
        hints = normalize_behavior_hints(tuple(item[0] for item in candidates))
        if not hints:
            continue
        change = apply_behavior_hint_change(
            workspace=workspace,
            data_dir=data_dir,
            profile=profile,
            hints=hints,
            target_scope=scope,
            summary="Added behavior hints from repeated user corrections.",
            evidence_refs=tuple(
                ref for _, refs in candidates for ref in refs
            ),
            now=now,
        )
        if change is not None:
            target = (
                profile_behavior_path(workspace, profile)
                if scope == "profile"
                else workspace_behavior_path(workspace)
            )
            changes.append((change, target))
    return tuple(changes)


def _behavior_hint_scope(correction: str, batch: EvolutionEvidenceBatch) -> str:
    matching = [
        evidence
        for evidence in batch.user_corrections
        if " ".join(evidence.correction.split()).lower() == correction.lower()
    ]
    explicit = {
        evidence.target_scope.strip().lower()
        for evidence in matching
        if evidence.target_scope.strip().lower() in {"workspace", "profile"}
    }
    if len(explicit) == 1:
        return explicit.pop()
    clean = correction.lower()
    profile_markers = (
        "i prefer",
        "my preference",
        "call me ",
        "respond in ",
        "reply in ",
        "always be concise",
        "我偏好",
        "请称呼我",
        "始终用中文",
        "始终使用中文",
    )
    return "profile" if any(marker in clean for marker in profile_markers) else "workspace"


def _best_user_correction_hint(examples: tuple[str, ...]) -> str:
    for example in examples:
        hint = _clean_behavior_hint_candidate(example)
        if hint:
            return hint
    return ""


def _clean_behavior_hint_candidate(value: str) -> str:
    text = " ".join(value.strip().split())
    if not text:
        return ""
    if any(pattern.search(text) for pattern in SENSITIVE_BEHAVIOR_HINT_PATTERNS):
        return ""
    if len(text) > 180:
        text = text[:177].rstrip() + "..."
    return text


def _apply_behavior_file_update(
    *,
    target_path: Path,
    workspace: Path,
    store: EvolutionChangeStore,
    old_content: str,
    new_content: str,
    old_exists: bool,
    summary: str,
    evidence_refs: tuple[str, ...],
    now: datetime,
) -> EvolutionChange | None:
    if old_exists and old_content == new_content:
        return None
    if not old_exists and not new_content.strip():
        return None
    atomic_write_text(target_path, new_content)
    try:
        timestamp = now.timestamp()
        os.utime(target_path, (timestamp, timestamp))
    except OSError:
        pass
    change = applied_change(
        change_type="behavior_patch",
        target_path=_workspace_relative(target_path, workspace),
        summary=summary,
        old_content=old_content,
        new_content=new_content,
        old_exists=old_exists,
        evidence_refs=evidence_refs,
        now_iso=utc_isoformat(now),
    )
    try:
        store.append(change)
    except Exception as append_error:
        try:
            if old_exists:
                atomic_write_text(target_path, old_content)
            else:
                target_path.unlink(missing_ok=True)
        except OSError:
            raise RuntimeError(
                f"behavior audit append failed and rollback was incomplete: {target_path}"
            ) from append_error
        raise
    return change


def _workspace_relative(path: Path, workspace: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace.resolve()))
    except ValueError:
        return str(path)


def _remember_changed_path(
    changed_paths: list[Path],
    changed_path_keys: set[str],
    path: Path,
) -> bool:
    key = str(path.resolve())
    if key in changed_path_keys:
        return False
    changed_path_keys.add(key)
    changed_paths.append(path)
    return True


def _load_maintenance_state(path: Path) -> EvolutionMaintenanceState:
    if not path.exists():
        return EvolutionMaintenanceState()
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return EvolutionMaintenanceState()
    if not isinstance(record, Mapping):
        return EvolutionMaintenanceState()
    return EvolutionMaintenanceState.from_record(record)


def _write_maintenance_state(
    path: Path,
    state: EvolutionMaintenanceState,
    metrics: EvolutionFitnessMetrics,
) -> None:
    payload = {
        **state.to_record(),
        "latest_metrics": metrics.to_record(),
    }
    atomic_write_json(path, payload)


def _read_jsonl_records_since(
    path: str | Path | None,
    since: datetime | None,
) -> tuple[Mapping[str, object], ...]:
    if path is None:
        return ()
    trace_path = Path(path)
    if not trace_path.exists():
        return ()
    records: list[Mapping[str, object]] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, Mapping):
            continue
        if "maintenance" in _text(record.get("kind")):
            continue
        if _is_record_after(record, "recorded_at", since):
            records.append(record)
    return tuple(records)


def _read_session_records_since(
    sessions_dir: str | Path | None,
    workspace: Path,
    profile: ProfileRef,
    since: datetime | None,
) -> tuple[SessionRecord, ...]:
    if sessions_dir is None:
        return ()
    store = SessionStore.in_directory(sessions_dir)
    sessions: list[SessionRecord] = []
    for session in store.list_recent(limit=10_000):
        if session.workspace.resolve() != workspace.resolve():
            continue
        if session.profile.name != profile.name:
            continue
        updated_at = _parse_datetime(session.updated_at)
        if since is None or (updated_at is not None and updated_at > since):
            sessions.append(session)
    return tuple(sessions)


def _activity_paths_since(
    activity_dir: str | Path | None,
    since: datetime | None,
) -> tuple[Path, ...]:
    if activity_dir is None:
        return ()
    root = Path(activity_dir)
    if not root.exists():
        return ()
    paths: list[Path] = []
    for path in sorted(root.rglob("*.md")):
        modified_at = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
        if since is None or _normal_datetime(modified_at) > since:
            paths.append(path)
    return tuple(paths)


def _session_evidence_records(
    sessions: tuple[SessionRecord, ...],
) -> tuple[Mapping[str, object], ...]:
    records: list[Mapping[str, object]] = []
    for session in sessions:
        transcript_records = _read_transcript_records(session.transcript_path)
        successful_tools: list[str] = []
        for record in transcript_records:
            payload = record.get("payload")
            if not isinstance(payload, Mapping):
                continue
            if record.get("kind") == "message" and payload.get("role") == "user":
                correction = _explicit_correction(_text(payload.get("content")))
                if correction:
                    records.append(
                        {
                            "kind": "user_correction",
                            "signature": correction,
                            "correction": correction,
                            "confidence": "0.7",
                            "source_group": f"session:{session.session_id}",
                            "source_ref": (
                                f"session:{session.session_id}:"
                                f"sequence:{record.get('sequence', 0)}"
                            ),
                        }
                    )
            if record.get("kind") != "tool_exchange":
                continue
            raw_results = payload.get("tool_results")
            if not isinstance(raw_results, list) or any(
                isinstance(result, Mapping) and result.get("is_error") is True
                for result in raw_results
            ):
                continue
            for result in raw_results:
                if not isinstance(result, Mapping):
                    continue
                name = _text(result.get("name"))
                if name and name not in successful_tools:
                    successful_tools.append(name)
        if len(successful_tools) >= 2:
            workflow_name = _workflow_name_from_tools(successful_tools)
            records.append(
                {
                    "kind": "workflow_success",
                    "signature": " -> ".join(successful_tools),
                    "name": workflow_name,
                    "description": "Repeat a validated sequence of successful tool calls.",
                    "steps": [f"Call {name}." for name in successful_tools],
                    "confidence": "0.65",
                    "source_group": f"session:{session.session_id}",
                    "source_ref": f"session:{session.session_id}:workflow",
                }
            )
    return tuple(records)


def _workflow_name_from_tools(tool_names: list[str]) -> str:
    """Build a non-personal workflow name without copying the session title."""
    labels = []
    for tool_name in tool_names[:3]:
        label = " ".join(re.findall(r"[A-Za-z0-9]+", tool_name)).title()
        if label:
            labels.append(label)
    return f"{' '.join(labels)} Workflow" if labels else "Generated Tool Workflow"


def _read_transcript_records(path: Path) -> tuple[Mapping[str, object], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return ()
    records: list[Mapping[str, object]] = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, Mapping):
            records.append(record)
    return tuple(records)


def _activity_evidence_records(
    paths: tuple[Path, ...],
) -> tuple[Mapping[str, object], ...]:
    records: list[Mapping[str, object]] = []
    for path in paths:
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for section in re.split(r"(?m)^## ", content)[1:]:
            fields: dict[str, str] = {}
            for line in section.splitlines()[1:]:
                match = re.match(r"^- ([a-z_]+):\s*(.+)$", line.strip())
                if match:
                    fields[match.group(1)] = match.group(2).strip()
            correction = _explicit_correction(fields.get("summary", ""))
            session_id = fields.get("session_id", "")
            if correction and session_id:
                records.append(
                    {
                        "kind": "user_correction",
                        "signature": correction,
                        "correction": correction,
                        "confidence": "0.6",
                        "source_group": f"session:{session_id}",
                        "source_ref": f"activity:{path.name}:session:{session_id}",
                    }
                )
    return tuple(records)


def _explicit_correction(content: str) -> str:
    clean = " ".join(content.strip().split())
    if not clean:
        return ""
    prefixes = (
        "correction:",
        "user correction:",
        "纠正：",
        "更正：",
    )
    lower = clean.lower()
    for prefix in prefixes:
        if lower.startswith(prefix.lower()):
            return clean[len(prefix) :].strip()
    if lower.startswith(("do not ", "don't ", "always ", "never ")):
        return clean
    if clean.startswith(("不要", "请始终", "以后不要", "以后请")):
        return clean
    return ""


def _behavior_paths_since(
    workspace: Path,
    profile: ProfileRef,
    since: datetime | None,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for path in (workspace_behavior_path(workspace), profile_behavior_path(workspace, profile)):
        if not path.exists():
            continue
        modified_at = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
        if since is None or _normal_datetime(modified_at) > since:
            paths.append(path)
    return tuple(paths)


def _capability_states_since(
    state_store: CapabilityStateStore,
    since: datetime | None,
) -> tuple[CapabilityState, ...]:
    states: list[CapabilityState] = []
    for state in state_store.load().values():
        updated_at = _parse_datetime(state.updated_at)
        if since is None or (updated_at is not None and updated_at > since):
            states.append(state)
    return tuple(states)


def _changes_since(
    store: EvolutionChangeStore,
    since: datetime | None,
) -> tuple[EvolutionChange, ...]:
    changes: list[EvolutionChange] = []
    for change in store.load():
        created_at = _parse_datetime(change.created_at)
        if since is None or (created_at is not None and created_at > since):
            changes.append(change)
    return tuple(changes)


def _generated_cold_active_states(
    states: tuple[CapabilityState, ...],
) -> tuple[CapabilityState, ...]:
    return tuple(
        state
        for state in states
        if state.source == CapabilitySource.GENERATED
        and state.temperature == CapabilityTemperature.COLD
        and state.asset_state == CapabilityAssetState.ACTIVE
    )


def _build_metrics(
    *,
    records: tuple[Mapping[str, object], ...],
    batch: EvolutionEvidenceBatch,
    generated_results: tuple[GeneratedSkillApplyResult, ...],
    changes: tuple[EvolutionChange, ...],
    window_start: str,
    window_end: str,
    recorded_at: str,
) -> EvolutionFitnessMetrics:
    return EvolutionFitnessMetrics(
        recorded_at=recorded_at,
        window_start=window_start,
        window_end=window_end,
        token_cost=_token_cost(records),
        loaded_skills_count=len(_loaded_skill_names(records)),
        used_skills_count=len(_used_skill_names(records)),
        tool_failure_count=len(batch.tool_failures),
        user_correction_count=len(batch.user_corrections),
        generated_skill_apply_count=sum(
            1 for result in generated_results if result.is_applied()
        ),
        rollback_count=sum(
            1
            for change in changes
            if change.change_type == "rollback"
            and _change_in_window(change, window_start, window_end)
        ),
    )


def _token_cost(records: tuple[Mapping[str, object], ...]) -> int:
    total = 0
    for record in records:
        refs = _refs_map(record)
        total += _int_ref(refs, "input_tokens")
        total += _int_ref(refs, "output_tokens")
        total += _int_ref(refs, "reasoning_tokens")
        total += _int_ref(refs, "summary_input_tokens")
        total += _int_ref(refs, "summary_output_tokens")
        total += _int_ref(refs, "summary_reasoning_tokens")
    return total


def _loaded_skill_names(records: tuple[Mapping[str, object], ...]) -> set[str]:
    names: set[str] = set()
    for record in records:
        kind = _text(record.get("kind"))
        refs = _refs_map(record)
        if kind == "native_tool_completed" and (
            "load_skill" in _record_refs(record) or refs.get("tool") == "load_skill"
        ):
            skill = refs.get("skill", "").strip()
            if skill:
                names.add(skill.lower())
    return names


def _used_skill_names(records: tuple[Mapping[str, object], ...]) -> set[str]:
    names: set[str] = set()
    for record in records:
        if _text(record.get("kind")) != "capability_selected":
            continue
        skill = _refs_map(record).get("skill", "").strip()
        if skill:
            names.add(skill.lower())
    return names


def _change_in_window(change: EvolutionChange, window_start: str, window_end: str) -> bool:
    created = _parse_datetime(change.created_at)
    end = _parse_datetime(window_end)
    start = _parse_datetime(window_start)
    if created is None or end is None:
        return False
    if created > end:
        return False
    return start is None or created > start


def _maintenance_reason(
    *,
    behavior_changes: int,
    failure_patterns_updated: int,
    generated_skill_changes: int,
    metrics: EvolutionFitnessMetrics,
) -> str:
    if generated_skill_changes:
        return "generated_skill_changes"
    if failure_patterns_updated:
        return "failure_patterns_updated"
    if behavior_changes:
        return "behavior_updated"
    if not metrics.is_empty():
        return "metrics_recorded"
    return "completed"


def _record_trace(
    recorder: TraceRecorder | None,
    *,
    completed: bool,
    result_reason: str,
    metrics: EvolutionFitnessMetrics,
    state_path: Path,
    metrics_path: Path,
    trace_records_seen: int,
    sessions_seen: int,
    activity_notes_seen: int,
    capability_states_seen: int,
    changed_paths: tuple[Path, ...],
) -> None:
    if recorder is None:
        return
    recorder.record(
        TraceEvent(
            kind=(
                "evolution_maintenance_completed"
                if completed
                else "evolution_maintenance_skipped"
            ),
            summary=f"Evolution maintenance {result_reason}.",
            refs=(
                f"reason={result_reason}",
                f"trace_records_seen={trace_records_seen}",
                f"sessions_seen={sessions_seen}",
                f"activity_notes_seen={activity_notes_seen}",
                f"capability_states_seen={capability_states_seen}",
                f"token_cost={metrics.token_cost}",
                f"loaded_skills_count={metrics.loaded_skills_count}",
                f"used_skills_count={metrics.used_skills_count}",
                f"tool_failure_count={metrics.tool_failure_count}",
                f"user_correction_count={metrics.user_correction_count}",
                f"generated_skill_apply_count={metrics.generated_skill_apply_count}",
                f"rollback_count={metrics.rollback_count}",
                f"state_path={state_path}",
                f"metrics_path={metrics_path}",
                *(f"changed_path={path}" for path in changed_paths),
            ),
        )
    )


def _is_record_after(
    record: Mapping[str, object],
    key: str,
    since: datetime | None,
) -> bool:
    if since is None:
        return True
    value = _parse_datetime(_text(record.get(key)))
    return value is not None and value > since


def _parse_datetime(value: str) -> datetime | None:
    if not value.strip():
        return None
    try:
        return _normal_datetime(datetime.fromisoformat(value.strip().replace("Z", "+00:00")))
    except ValueError:
        return None


def _normal_datetime(value: datetime | None) -> datetime:
    return (value or datetime.now().astimezone()).astimezone()


def _refs_map(record: Mapping[str, object]) -> dict[str, str]:
    refs: dict[str, str] = {}
    for ref in _record_refs(record):
        if "=" not in ref:
            continue
        key, value = ref.split("=", 1)
        refs[key.strip()] = value.strip()
    return refs


def _record_refs(record: Mapping[str, object]) -> tuple[str, ...]:
    refs = record.get("refs")
    if not isinstance(refs, list) and not isinstance(refs, tuple):
        return ()
    return tuple(ref.strip() for ref in refs if isinstance(ref, str) and ref.strip())


def _int_ref(refs: Mapping[str, str], key: str) -> int:
    try:
        return max(0, int(refs.get(key, "0")))
    except ValueError:
        return 0


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""
