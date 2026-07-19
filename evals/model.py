"""Stable data contracts shared by the Deepmate evaluation modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Mapping


JsonMap = Mapping[str, object]


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One versioned evaluation case loaded from a repository asset."""

    schema_version: int
    id: str
    version: int
    suite: str
    kind: str
    title: str
    okr: JsonMap
    environment: JsonMap
    agent: JsonMap
    permissions: JsonMap
    conversation: tuple[JsonMap, ...]
    budgets: JsonMap
    graders: tuple[JsonMap, ...]
    hard_failures: tuple[JsonMap, ...]
    metrics: tuple[str, ...]
    execution: JsonMap = field(
        default_factory=lambda: {
            "class": "deterministic-runtime",
            "modes": ("offline",),
        }
    )
    presets: tuple[str, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)
    path: Path | None = field(default=None, compare=False)
    digest: str = ""

    def is_safety(self) -> bool:
        return self.kind == "safety" or self.suite == "security"

    def supports_mode(self, mode: str) -> bool:
        modes = self.execution.get("modes", ("offline",))
        return mode in tuple(str(value) for value in modes)

    def to_record(self) -> dict[str, object]:
        record = asdict(self)
        record["path"] = str(self.path) if self.path else ""
        return record


@dataclass(frozen=True, slots=True)
class EvalTrial:
    """Identity and timing for one independent attempt of a case."""

    trial_id: str
    case_id: str
    index: int
    seed: int
    workspace: str
    data_dir: str
    started_at: str
    ended_at: str = ""
    wall_started_at: str = ""
    wall_ended_at: str = ""

    def to_record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvalRunManifest:
    """Provenance required before two evaluation runs may be compared."""

    run_id: str
    created_at: str
    deepmate_commit: str
    dirty: bool
    case_set_digest: str
    fixture_digest: str
    policy_digest: str
    provider: str
    model: str
    judge: str
    permissions_digest: str
    environment: JsonMap
    pricing: JsonMap = field(default_factory=dict)
    variants: tuple[str, ...] = field(default_factory=tuple)

    def comparison_key(self) -> tuple[object, ...]:
        return (
            self.case_set_digest,
            self.fixture_digest,
            self.policy_digest,
            self.permissions_digest,
            self.provider,
            self.model,
            self.judge,
            _stable_mapping(self.environment),
            _stable_mapping(self.pricing),
        )

    def to_record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvalGrade:
    """One deterministic, model-assisted, or safety verdict."""

    grader: str
    grader_type: str
    passed: bool
    score: float
    hard: bool
    message: str
    evidence: tuple[str, ...] = field(default_factory=tuple)
    valid: bool = True

    def to_record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvalMetrics:
    """Normalized metrics for a trial or an aggregate suite."""

    values: JsonMap = field(default_factory=dict)
    priced: bool = False

    def to_record(self) -> dict[str, object]:
        return {"values": dict(self.values), "priced": self.priced}


@dataclass(frozen=True, slots=True)
class EvalCaseResult:
    """Complete result for one case trial."""

    trial: EvalTrial
    grades: tuple[EvalGrade, ...]
    metrics: EvalMetrics
    final_output: str
    observations: JsonMap
    trace: tuple[JsonMap, ...]
    valid: bool = True
    invalid_reason: str = ""

    @property
    def passed(self) -> bool:
        return self.valid and all(grade.passed for grade in self.grades)

    @property
    def safety_passed(self) -> bool:
        return self.valid and all(
            grade.passed for grade in self.grades if grade.hard
        )

    def to_record(self) -> dict[str, object]:
        return {
            "trial": self.trial.to_record(),
            "grades": [grade.to_record() for grade in self.grades],
            "metrics": self.metrics.to_record(),
            "final_output": self.final_output,
            "observations": dict(self.observations),
            "trace": [dict(record) for record in self.trace],
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "passed": self.passed,
            "safety_passed": self.safety_passed,
        }


@dataclass(frozen=True, slots=True)
class EvalRunResult:
    """Suite result plus its comparison provenance."""

    manifest: EvalRunManifest
    cases: tuple[EvalCaseResult, ...]
    metrics: EvalMetrics
    output_dir: str
    valid: bool = True
    invalid_reason: str = ""
    framework_tests_passed: bool | None = None

    @property
    def safety_gate(self) -> bool:
        return self.valid and all(result.safety_passed for result in self.cases)

    @property
    def passed(self) -> bool:
        tests_passed = self.framework_tests_passed is not False
        return self.valid and tests_passed and self.safety_gate and all(
            result.passed for result in self.cases
        )

    def to_record(self) -> dict[str, object]:
        return {
            "manifest": self.manifest.to_record(),
            "cases": [case.to_record() for case in self.cases],
            "metrics": self.metrics.to_record(),
            "output_dir": self.output_dir,
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "framework_tests_passed": self.framework_tests_passed,
            "safety_gate": self.safety_gate,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class EvalComparison:
    """Validated delta between a baseline and a candidate run."""

    baseline_run_id: str
    candidate_run_id: str
    valid: bool
    deltas: JsonMap = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    invalid_reason: str = ""

    def to_record(self) -> dict[str, object]:
        return asdict(self)


def _stable_mapping(value: JsonMap) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
