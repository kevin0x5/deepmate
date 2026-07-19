from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evals.__main__ import EXIT_FAILURE, EXIT_INVALID, EXIT_PASS, main
from evals.loader import load_cases, select_cases
from evals.runner import run_suite


class EvalRunnerCliTests(unittest.TestCase):
    def test_runner_uses_real_agent_loop_and_writes_only_sanitized_paths(self) -> None:
        selected = select_cases(load_cases(), case_id="dm-tool-no-call-001")
        with tempfile.TemporaryDirectory() as tmp:
            run = run_suite(selected, output_root=tmp)
            summary = Path(tmp) / run.manifest.run_id / "summary.json"
            text = summary.read_text(encoding="utf-8")
        self.assertTrue(run.passed)
        self.assertTrue(any(record.get("name") == "deepmate turn" for record in run.cases[0].trace))
        self.assertNotIn(tmp, text)
        self.assertEqual(json.loads(text)["output_dir"], f"var/evals/{run.manifest.run_id}")

    def test_validate_and_objectives_commands_pass(self) -> None:
        self.assertEqual(main(["validate"]), EXIT_PASS)
        self.assertEqual(main(["objectives"]), EXIT_PASS)

    def test_missing_case_is_invalid(self) -> None:
        self.assertEqual(main(["run", "--case", "dm-missing-case-999"]), EXIT_INVALID)

    def test_matrix_rejects_scripted_provider(self) -> None:
        self.assertEqual(
            main(["matrix", "--suite", "core", "--models", "model-a"]),
            EXIT_INVALID,
        )

    def test_valid_ab_regression_uses_failure_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = main(
                [
                    "ab",
                    "--case",
                    "dm-cxt-freeze-refresh-001",
                    "--variant",
                    "control",
                    "--variant",
                    "candidate",
                    "--trials",
                    "1",
                    "--output-root",
                    tmp,
                ]
            )
        self.assertEqual(code, EXIT_FAILURE)


if __name__ == "__main__":
    unittest.main()
