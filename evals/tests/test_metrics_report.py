from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evals.loader import EvalConfigurationError
from evals.metrics import aggregate_metrics, compare_runs
from evals.model import (
    EvalCaseResult,
    EvalGrade,
    EvalMetrics,
    EvalRunManifest,
    EvalRunResult,
    EvalTrial,
)
from evals.report import load_run, write_run_artifacts


class EvalMetricsReportTests(unittest.TestCase):
    def test_aggregate_reports_reliability_safety_cache_and_unpriced_cost(self) -> None:
        results = tuple(
            _result("dm-a-001", index, True)
            for index in range(1, 6)
        )
        metrics = aggregate_metrics(results)
        self.assertEqual(metrics.values["task_success"], 1.0)
        self.assertEqual(metrics.values["pass@1"], 1.0)
        self.assertEqual(metrics.values["pass^3"], 1.0)
        self.assertEqual(metrics.values["pass^5"], 1.0)
        self.assertTrue(metrics.values["safety_gate"])
        self.assertIsNone(metrics.values["cost_per_reliable_success"])
        self.assertEqual(metrics.values["cost_status"], "unpriced")

    def test_comparison_rejects_mismatched_conditions(self) -> None:
        baseline = _run("baseline", model="model-a")
        candidate = _run("candidate", model="model-b")
        comparison = compare_runs(baseline, candidate)
        self.assertFalse(comparison.valid)
        self.assertIn("differs", comparison.invalid_reason)

    def test_comparison_emits_cost_cache_and_latency_warnings(self) -> None:
        baseline = _run("baseline", model="model-a")
        candidate = _run("candidate", model="model-a")
        baseline = EvalRunResult(
            baseline.manifest,
            baseline.cases,
            EvalMetrics(
                {
                    **baseline.metrics.values,
                    "cost_per_reliable_success": 10.0,
                    "cache_hit_ratio": 0.8,
                    "p95_latency_seconds": 1.0,
                },
                priced=True,
            ),
            baseline.output_dir,
        )
        candidate = EvalRunResult(
            candidate.manifest,
            candidate.cases,
            EvalMetrics(
                {
                    **candidate.metrics.values,
                    "cost_per_reliable_success": 11.1,
                    "cache_hit_ratio": 0.7,
                    "p95_latency_seconds": 1.21,
                },
                priced=True,
            ),
            candidate.output_dir,
        )
        comparison = compare_runs(baseline, candidate)
        self.assertTrue(comparison.valid)
        self.assertEqual(len(comparison.warnings), 3)

    def test_report_round_trip_preserves_manifest_and_metrics(self) -> None:
        run = _run("round-trip", model="model-a")
        with tempfile.TemporaryDirectory() as tmp:
            write_run_artifacts(run, tmp)
            loaded = load_run(Path(tmp) / "summary.json", tmp)
        self.assertEqual(loaded.manifest.run_id, "round-trip")
        self.assertEqual(loaded.metrics.values["task_success"], 1.0)
        self.assertTrue(loaded.passed)

    def test_incomplete_run_is_not_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "summary.json"
            summary.write_text("{}", encoding="utf-8")
            with self.assertRaises(EvalConfigurationError) as raised:
                load_run(summary, tmp)
        self.assertIn("incomplete", str(raised.exception))


def _result(case_id: str, index: int, passed: bool) -> EvalCaseResult:
    grade = EvalGrade(
        "state",
        "state",
        passed,
        1.0 if passed else 0.0,
        True,
        "passed" if passed else "failed",
    )
    return EvalCaseResult(
        trial=EvalTrial(
            f"{case_id}-t{index}",
            case_id,
            index,
            index,
            "<trial-root>/workspace",
            "<trial-root>/data",
            "fixed",
            "fixed",
        ),
        grades=(grade,),
        metrics=EvalMetrics(
            {
                "cache_hit_tokens": 8,
                "cache_miss_tokens": 2,
                "latency_seconds": 0.01,
                "horizon": "L0",
            }
        ),
        final_output="ok",
        observations={"tags": ["context"]},
        trace=(),
    )


def _manifest(run_id: str, model: str) -> EvalRunManifest:
    return EvalRunManifest(
        run_id,
        "fixed",
        "commit",
        False,
        "cases",
        "fixtures",
        "policy",
        "fake",
        model,
        "judge",
        "permissions",
        {"python": "3.11"},
    )


def _run(run_id: str, model: str) -> EvalRunResult:
    case = _result("dm-a-001", 1, True)
    return EvalRunResult(
        _manifest(run_id, model),
        (case,),
        aggregate_metrics((case,)),
        f"var/evals/{run_id}",
    )


if __name__ == "__main__":
    unittest.main()
