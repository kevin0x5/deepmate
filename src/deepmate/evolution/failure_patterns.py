"""FailurePatternGuard for low-risk self-evolution gates."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from deepmate.domain import ProfileRef
from deepmate.evolution.changes import EvolutionChange, EvolutionChangeStore, applied_change
from deepmate.evolution.evidence_mining import (
    EvidenceAggregate,
    ToolFailureEvidence,
    UserCorrectionEvidence,
    tool_failure_candidates,
    user_correction_candidates,
)
from deepmate.foundation import normalize_name, normal_datetime, utc_isoformat
from deepmate.storage import JsonlWriter

FAILURE_PATTERNS_FILE = "failure_patterns.jsonl"
USER_CORRECTION_PATTERN = "user_correction"
TOOL_FAILURE_PATTERN = "tool_failure"
DEFAULT_USER_CORRECTION_STRENGTH = 5
DEFAULT_TOOL_FAILURE_STRENGTH = 4
DEFAULT_MAX_STRENGTH = 10
DEFAULT_BLOCK_STRENGTH = 5


@dataclass(frozen=True, slots=True)
class FailurePattern:
    """One known failure mode used to gate self-evolution changes."""

    signature: str
    kind: str
    strength: int
    source_refs: tuple[str, ...]
    last_seen_at: str

    def key(self) -> str:
        """Return the stable key for de-duplicating pattern updates."""
        return pattern_key(self.kind, self.signature)

    def is_ready(self) -> bool:
        """Return whether this pattern has enough data to persist."""
        return bool(
            self.signature.strip()
            and self.kind.strip()
            and self.strength > 0
            and self.last_seen_at.strip()
        )

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable JSONL record."""
        return {
            "signature": self.signature,
            "kind": self.kind,
            "strength": self.strength,
            "source_refs": list(self.source_refs),
            "last_seen_at": self.last_seen_at,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "FailurePattern":
        """Build a pattern from one JSONL record."""
        return cls(
            signature=_text(record.get("signature")),
            kind=_text(record.get("kind")),
            strength=_positive_int(record.get("strength")),
            source_refs=_string_tuple(record.get("source_refs")),
            last_seen_at=_text(record.get("last_seen_at")),
        )


@dataclass(frozen=True, slots=True)
class FailurePatternMatch:
    """Result of checking a proposed change against known failure patterns."""

    matched_patterns: tuple[FailurePattern, ...]
    blocked: bool
    reason: str = ""

    def has_match(self) -> bool:
        """Return whether any pattern matched."""
        return bool(self.matched_patterns)


class FailurePatternStore:
    """Append-only JSONL store for latest failure patterns."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def in_data_dir(
        cls,
        data_dir: str | Path,
        profile: ProfileRef | str,
    ) -> "FailurePatternStore":
        """Return the profile-local failure pattern store."""
        profile_name = profile.name if isinstance(profile, ProfileRef) else str(profile)
        clean_profile = profile_name.strip() or "default"
        return cls(Path(data_dir) / "evolution" / clean_profile / FAILURE_PATTERNS_FILE)

    def append(self, pattern: FailurePattern) -> None:
        """Append one ready failure pattern update."""
        if not pattern.is_ready():
            raise ValueError("failure pattern is incomplete")
        JsonlWriter(self.path).append(pattern.to_record())

    def load(self) -> tuple[FailurePattern, ...]:
        """Load latest pattern state keyed by kind and signature."""
        if not self.path.exists():
            return ()
        latest: dict[str, FailurePattern] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, Mapping):
                continue
            pattern = FailurePattern.from_record(record)
            if pattern.is_ready():
                latest[pattern.key()] = pattern
        return tuple(latest[key] for key in sorted(latest))

    def by_key(self) -> dict[str, FailurePattern]:
        """Return latest patterns keyed by kind/signature."""
        return {pattern.key(): pattern for pattern in self.load()}


class FailurePatternGuard:
    """Check proposed self-evolution changes against known failure modes."""

    def __init__(
        self,
        patterns: tuple[FailurePattern, ...],
        block_strength: int = DEFAULT_BLOCK_STRENGTH,
    ) -> None:
        self.patterns = tuple(pattern for pattern in patterns if pattern.is_ready())
        self.block_strength = max(1, block_strength)

    @classmethod
    def from_store(
        cls,
        store: FailurePatternStore,
        block_strength: int = DEFAULT_BLOCK_STRENGTH,
    ) -> "FailurePatternGuard":
        """Build a guard from the latest persisted pattern set."""
        return cls(store.load(), block_strength=block_strength)

    def check_text(
        self,
        text: str,
        refs: tuple[str, ...] = (),
    ) -> FailurePatternMatch:
        """Return whether the proposed change matches a blocking pattern."""
        haystack = _normalized_haystack(text, refs)
        matched = tuple(
            pattern
            for pattern in self.patterns
            if _pattern_matches(pattern, haystack)
        )
        blocked = any(pattern.strength >= self.block_strength for pattern in matched)
        reason = ""
        if blocked:
            strongest = max(matched, key=lambda pattern: pattern.strength)
            reason = (
                "blocked_by_failure_pattern:"
                f"{strongest.kind}:{strongest.signature}:strength={strongest.strength}"
            )
        return FailurePatternMatch(
            matched_patterns=matched,
            blocked=blocked,
            reason=reason,
        )


def update_failure_patterns_from_evidence(
    *,
    store: FailurePatternStore,
    user_corrections: tuple[UserCorrectionEvidence, ...] = (),
    tool_failures: tuple[ToolFailureEvidence, ...] = (),
    now: datetime | None = None,
    change_store: EvolutionChangeStore | None = None,
    workspace: str | Path | None = None,
) -> tuple[FailurePattern, ...]:
    """Create or strengthen patterns from deterministic evidence candidates."""
    aggregates = (
        *user_correction_candidates(user_corrections),
        *tool_failure_candidates(tool_failures),
    )
    return update_failure_patterns(
        store=store,
        aggregates=aggregates,
        now=now,
        change_store=change_store,
        workspace=workspace,
    )


def update_failure_patterns(
    *,
    store: FailurePatternStore,
    aggregates: tuple[EvidenceAggregate, ...],
    now: datetime | None = None,
    change_store: EvolutionChangeStore | None = None,
    workspace: str | Path | None = None,
) -> tuple[FailurePattern, ...]:
    """Apply deterministic pattern updates and optionally record applied changes."""
    current_time = normal_datetime(now)
    timestamp = utc_isoformat(current_time)
    latest = store.by_key()
    updated: list[FailurePattern] = []
    should_record_change = change_store is not None and workspace is not None
    old_exists = store.path.exists() if should_record_change else False
    old_content = (
        store.path.read_text(encoding="utf-8")
        if should_record_change and old_exists
        else ""
    )
    for aggregate in aggregates:
        if not aggregate.is_ready():
            continue
        kind = _pattern_kind(aggregate.kind)
        signature = normalize_name(aggregate.signature)
        key = pattern_key(kind, signature)
        previous = latest.get(key)
        pattern = _strengthened_pattern(
            previous=previous,
            kind=kind,
            signature=signature,
            source_refs=aggregate.source_refs,
            timestamp=timestamp,
        )
        store.append(pattern)
        latest[key] = pattern
        updated.append(pattern)
    if should_record_change and updated:
        new_content = store.path.read_text(encoding="utf-8")
        evidence_refs = _merge_strings(
            tuple(ref for pattern in updated for ref in pattern.source_refs)
        )
        summary = (
            f"Updated {len(updated)} failure pattern"
            f"{'' if len(updated) == 1 else 's'}."
        )
        assert change_store is not None
        assert workspace is not None
        change_store.append(
            applied_change(
                change_type="failure_pattern_update",
                target_path=_workspace_relative(store.path, Path(workspace)),
                summary=summary,
                old_content=old_content,
                new_content=new_content,
                old_exists=old_exists,
                evidence_refs=evidence_refs,
                validation_result="passed",
                decision="auto_apply_with_validation",
                now_iso=timestamp,
            )
        )
    return tuple(updated)


def pattern_key(kind: str, signature: str) -> str:
    """Return the stable key for one pattern."""
    return f"{normalize_name(kind)}:{normalize_name(signature)}"


def _strengthened_pattern(
    *,
    previous: FailurePattern | None,
    kind: str,
    signature: str,
    source_refs: tuple[str, ...],
    timestamp: str,
) -> FailurePattern:
    base_strength = (
        DEFAULT_USER_CORRECTION_STRENGTH
        if kind == USER_CORRECTION_PATTERN
        else DEFAULT_TOOL_FAILURE_STRENGTH
    )
    if previous is None:
        strength = base_strength
        refs = source_refs
    else:
        strength = min(DEFAULT_MAX_STRENGTH, max(previous.strength, base_strength) + 1)
        refs = _merge_strings((*previous.source_refs, *source_refs))
    return FailurePattern(
        signature=signature,
        kind=kind,
        strength=strength,
        source_refs=refs,
        last_seen_at=timestamp,
    )


def _pattern_kind(kind: str) -> str:
    clean = normalize_name(kind).replace(" ", "_")
    if clean == USER_CORRECTION_PATTERN:
        return USER_CORRECTION_PATTERN
    if clean == TOOL_FAILURE_PATTERN:
        return TOOL_FAILURE_PATTERN
    raise ValueError(f"unsupported failure pattern kind: {kind}")


def _pattern_matches(pattern: FailurePattern, haystack: str) -> bool:
    signature = normalize_name(pattern.signature)
    if not signature:
        return False
    return signature in haystack


def _normalized_haystack(text: str, refs: tuple[str, ...]) -> str:
    return normalize_name(" ".join((text, *refs)))


def _merge_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(value.strip().split())
        key = normalize_name(text)
        if not text or key in seen:
            continue
        seen.add(key)
        merged.append(text)
    return tuple(merged)


def _workspace_relative(path: Path, workspace: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace.resolve()))
    except ValueError:
        return str(path)


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _positive_int(value: object) -> int:
    return max(0, value) if isinstance(value, int) else 0


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
