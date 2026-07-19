from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from deepmate.evolution import collect_evidence_from_records
from deepmate.evolution.maintenance import _read_jsonl_records_since


class EvalEvolutionIsolationTests(unittest.TestCase):
    def test_eval_grader_and_hidden_sources_are_not_evolution_evidence(self) -> None:
        records = (
            _workflow("normal", "synthetic-source"),
            {**_workflow("eval", "var/evals/synthetic"), "eval_artifact": True},
            {**_workflow("grader", "synthetic-grader"), "source_kind": "grader"},
            {**_workflow("hidden", "synthetic-hidden"), "source_kind": "hidden_probe"},
        )
        batch = collect_evidence_from_records(records)
        self.assertEqual(len(batch.workflows), 1)
        self.assertEqual(batch.workflows[0].signature, "normal")

    def test_eval_artifact_trace_path_is_not_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "var" / "evals" / "run" / "trace.jsonl"
            trace.parent.mkdir(parents=True)
            trace.write_text(
                json.dumps(
                    {
                        "kind": "workflow_success",
                        "recorded_at": "2026-01-01T00:00:00+00:00",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(_read_jsonl_records_since(trace, None), ())


def _workflow(signature: str, source_ref: str) -> dict[str, object]:
    return {
        "kind": "workflow_success",
        "signature": signature,
        "name": "Synthetic Workflow",
        "description": "Synthetic evidence used by an isolation test.",
        "steps": ["Read fixture.", "Verify result."],
        "confidence": "0.9",
        "source_group": source_ref,
        "source_ref": source_ref,
    }


if __name__ == "__main__":
    unittest.main()
