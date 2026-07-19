"""Sanitized JSONL, JSON, and Markdown reporting for evaluation runs."""

from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from evals.loader import EvalConfigurationError, load_cases
from evals.model import (
    EvalCaseResult,
    EvalGrade,
    EvalMetrics,
    EvalRunManifest,
    EvalRunResult,
    EvalTrial,
)
from evals.privacy import scan_value


def write_run_artifacts(run: EvalRunResult, output_dir: str | Path) -> tuple[Path, ...]:
    """Persist a sanitized run and reject any report that fails privacy checks."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    record = run.to_record()
    findings = scan_value(record, source="run report")
    if findings:
        raise RuntimeError("evaluation report contains unsafe data and was not written")
    trials_path = root / "trials.jsonl"
    _atomic_write_text(
        trials_path,
        "".join(
            json.dumps(result.to_record(), ensure_ascii=False, sort_keys=True) + "\n"
            for result in run.cases
        ),
    )
    summary_path = root / "summary.json"
    _atomic_write_text(
        summary_path,
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    markdown_path = root / "report.md"
    _atomic_write_text(markdown_path, render_markdown(run))
    completed_path = root / "COMPLETED"
    _atomic_write_text(completed_path, f"{run.manifest.run_id}\n")
    (root / "RUNNING").unlink(missing_ok=True)
    return trials_path, summary_path, markdown_path


def render_markdown(run: EvalRunResult) -> str:
    """Render a compact outcome-first report with safety and OKR coverage."""
    failed = [result for result in run.cases if not result.passed]
    low_probes = [
        (result.trial.case_id, grade)
        for result in run.cases
        for grade in result.grades
        if grade.grader_type == "fact_probe" and not grade.passed
    ]
    lines = [
        f"# Deepmate Eval Run {run.manifest.run_id}",
        "",
        "## Outcome",
        "",
        f"- Valid: `{str(run.valid).lower()}`",
        f"- Passed: `{str(run.passed).lower()}`",
        f"- Safety Gate: `{str(run.safety_gate).lower()}`",
        (
            f"- Cases/Trials: `{run.metrics.values.get('cases', 0)}/"
            f"{run.metrics.values.get('trials', 0)}`"
        ),
        f"- Task Success: `{_format_metric(run.metrics.values.get('task_success'))}`",
        (
            "- Reliable Task Horizon: `"
            f"{run.metrics.values.get('reliable_task_horizon') or 'unavailable'}`"
        ),
        (
            "- Cost per Reliable Success: `"
            f"{_format_metric(run.metrics.values.get('cost_per_reliable_success'))}`"
        ),
        (
            "- Evolution Net Gain: `"
            f"{_format_metric(run.metrics.values.get('evolution_net_gain'))}`"
        ),
        "",
        "## Provenance",
        "",
        f"- Commit: `{run.manifest.deepmate_commit}`",
        f"- Dirty: `{str(run.manifest.dirty).lower()}`",
        f"- Provider/Model: `{run.manifest.provider}/{run.manifest.model}`",
        f"- Evaluator: `{run.manifest.judge}`",
        f"- Case Set Digest: `{run.manifest.case_set_digest}`",
        f"- Fixture Digest: `{run.manifest.fixture_digest}`",
        f"- Policy Digest: `{run.manifest.policy_digest}`",
        "",
        "## Failures",
        "",
    ]
    if failed:
        for result in failed:
            failed_graders = ", ".join(
                grade.grader for grade in result.grades if not grade.passed
            )
            lines.append(f"- `{result.trial.case_id}` trial {result.trial.index}: {failed_graders}")
    else:
        lines.append("- None.")
    lines.extend(("", "## Low Fact Probe Scores", ""))
    if low_probes:
        for case_id, grade in low_probes:
            lines.append(f"- `{case_id}`: `{grade.score:.3f}` ({grade.grader})")
    else:
        lines.append("- None.")
    coverage = okr_coverage(result.trial.case_id for result in run.cases)
    lines.extend(("", "## OKR Coverage", ""))
    for key_result, case_ids in sorted(coverage.items()):
        lines.append(f"- `{key_result}`: {len(case_ids)} case(s)")
    return "\n".join(lines) + "\n"


def okr_coverage(case_ids: Iterable[str] = ()) -> Mapping[str, tuple[str, ...]]:
    """Build coverage from authoritative case manifests, optionally filtered."""
    selected = set(case_ids)
    coverage: dict[str, list[str]] = defaultdict(list)
    for case in load_cases():
        if selected and case.id not in selected:
            continue
        for key_result in case.okr.get("key_results", ()):
            coverage[str(key_result)].append(case.id)
    return {key: tuple(sorted(values)) for key, values in coverage.items()}


def load_run(reference: str | Path, runs_root: str | Path) -> EvalRunResult:
    """Load a saved run by ID, directory, or summary path."""
    candidate = Path(reference)
    if not candidate.exists():
        candidate = Path(runs_root) / str(reference)
    if candidate.is_dir():
        candidate = candidate / "summary.json"
    if not candidate.is_file():
        raise EvalConfigurationError(f"run summary does not exist: {reference}")
    if candidate.is_symlink():
        raise EvalConfigurationError(f"run summary may not be a symlink: {candidate}")
    if not (candidate.parent / "COMPLETED").is_file():
        raise EvalConfigurationError(f"evaluation run is incomplete: {candidate.parent}")
    if candidate.stat().st_size > 10_000_000:
        raise EvalConfigurationError(f"run summary exceeds the size limit: {candidate}")
    raw = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise EvalConfigurationError(f"run summary is not an object: {candidate}")
    findings = scan_value(raw, source="loaded run summary")
    if findings:
        raise EvalConfigurationError("run summary failed privacy validation")
    return _run_from_record(raw)


def _atomic_write_text(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_from_record(raw: Mapping[str, object]) -> EvalRunResult:
    manifest_raw = _mapping(raw.get("manifest"))
    manifest = EvalRunManifest(
        run_id=str(manifest_raw.get("run_id", "")),
        created_at=str(manifest_raw.get("created_at", "")),
        deepmate_commit=str(manifest_raw.get("deepmate_commit", "")),
        dirty=bool(manifest_raw.get("dirty", False)),
        case_set_digest=str(manifest_raw.get("case_set_digest", "")),
        fixture_digest=str(manifest_raw.get("fixture_digest", "")),
        policy_digest=str(manifest_raw.get("policy_digest", "")),
        provider=str(manifest_raw.get("provider", "")),
        model=str(manifest_raw.get("model", "")),
        judge=str(manifest_raw.get("judge", "")),
        permissions_digest=str(manifest_raw.get("permissions_digest", "")),
        environment=_mapping(manifest_raw.get("environment")),
        pricing=_mapping(manifest_raw.get("pricing")),
        variants=tuple(str(value) for value in _sequence(manifest_raw.get("variants"))),
    )
    case_results: list[EvalCaseResult] = []
    for item in _sequence(raw.get("cases")):
        case_raw = _mapping(item)
        trial_raw = _mapping(case_raw.get("trial"))
        trial = EvalTrial(
            trial_id=str(trial_raw.get("trial_id", "")),
            case_id=str(trial_raw.get("case_id", "")),
            index=int(trial_raw.get("index", 0)),
            seed=int(trial_raw.get("seed", 0)),
            workspace=str(trial_raw.get("workspace", "")),
        data_dir=str(trial_raw.get("data_dir", "")),
        started_at=str(trial_raw.get("started_at", "")),
        ended_at=str(trial_raw.get("ended_at", "")),
        wall_started_at=str(trial_raw.get("wall_started_at", "")),
        wall_ended_at=str(trial_raw.get("wall_ended_at", "")),
    )
        grades = tuple(
            EvalGrade(
                grader=str(grade.get("grader", "")),
                grader_type=str(grade.get("grader_type", "")),
                passed=bool(grade.get("passed", False)),
                score=float(grade.get("score", 0)),
                hard=bool(grade.get("hard", False)),
                message=str(grade.get("message", "")),
                evidence=tuple(str(value) for value in _sequence(grade.get("evidence"))),
                valid=bool(grade.get("valid", True)),
            )
            for grade in (_mapping(value) for value in _sequence(case_raw.get("grades")))
        )
        metrics_raw = _mapping(case_raw.get("metrics"))
        metrics = EvalMetrics(
            values=_mapping(metrics_raw.get("values")),
            priced=bool(metrics_raw.get("priced", False)),
        )
        case_results.append(
            EvalCaseResult(
                trial=trial,
                grades=grades,
                metrics=metrics,
                final_output=str(case_raw.get("final_output", "")),
                observations=_mapping(case_raw.get("observations")),
                trace=tuple(_mapping(value) for value in _sequence(case_raw.get("trace"))),
                valid=bool(case_raw.get("valid", True)),
                invalid_reason=str(case_raw.get("invalid_reason", "")),
            )
        )
    metrics_raw = _mapping(raw.get("metrics"))
    return EvalRunResult(
        manifest=manifest,
        cases=tuple(case_results),
        metrics=EvalMetrics(
            values=_mapping(metrics_raw.get("values")),
            priced=bool(metrics_raw.get("priced", False)),
        ),
        output_dir=str(raw.get("output_dir", "")),
        valid=bool(raw.get("valid", True)),
        invalid_reason=str(raw.get("invalid_reason", "")),
        framework_tests_passed=(
            raw.get("framework_tests_passed")
            if isinstance(raw.get("framework_tests_passed"), bool)
            else None
        ),
    )


def _format_metric(value: object) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _mapping(value: object) -> Mapping[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: object) -> tuple[object, ...]:
    return tuple(value) if isinstance(value, (list, tuple)) else ()
