from __future__ import annotations

import tempfile
import unittest

from evals.loader import load_cases, select_cases
from evals.runner import run_suite


class EvalPrPresetTests(unittest.TestCase):
    def test_pr_preset_passes_fifteen_core_and_fifteen_safety_cases(self) -> None:
        cases = select_cases(load_cases(), preset="pr")
        self.assertEqual(len(cases), 30)
        with tempfile.TemporaryDirectory() as tmp:
            run = run_suite(cases, output_root=tmp)
        self.assertTrue(run.valid)
        self.assertTrue(run.passed)
        self.assertTrue(run.safety_gate)
        self.assertEqual(run.metrics.values["pass@1"], 1.0)


if __name__ == "__main__":
    unittest.main()
