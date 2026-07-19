from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evals.grading import GradeContext, grade_case
from evals.model import EvalCase


class EvalGradingTests(unittest.TestCase):
    def test_all_deterministic_grader_layers_pass_on_declared_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            data_dir = root / "data"
            outside = root / "outside"
            for path in (workspace, data_dir, outside):
                path.mkdir()
            (workspace / "RESULT.md").write_text("tests-passed\n", encoding="utf-8")
            case = _case(
                graders=(
                    {
                        "type": "state",
                        "path": "observations.runtime_completed",
                        "equals": True,
                    },
                    {"type": "file", "path": "RESULT.md", "contains": ["tests-passed"]},
                    {
                        "type": "command",
                        "command": ["python3", "-c", "print('verified')"],
                        "contains": ["verified"],
                    },
                    {"type": "trace", "contains": ["agent_turn"], "min_events": 1},
                    {"type": "policy", "max_tool_calls": 1, "max_side_effects": 0},
                    {"type": "content", "contains": ["recall-orbit", "artifact-plan"]},
                    {
                        "type": "probe",
                        "probe": "compression-single.json",
                        "threshold": 1.0,
                    },
                    {"type": "python", "verifier": "dm-long-multifile-001/verify.py"},
                    {"type": "python", "verifier": "network-policy/verify.py"},
                ),
                hard_failures=({"type": "security", "scope": "artifacts"},),
            )
            context = GradeContext(
                workspace=workspace,
                outside_dir=outside,
                data_dir=data_dir,
                final_output="recall-orbit artifact-plan decision-blue continue-step-3",
                observations={
                    "runtime_completed": True,
                    "tool_calls": ["read"],
                    "side_effect_count": 0,
                },
                trace=({"kind": "agent_turn", "summary": "completed"},),
                canaries=(),
            )
            grades = grade_case(case, context)
            self.assertTrue(all(grade.valid for grade in grades))
            self.assertTrue(all(grade.passed for grade in grades), grades)

    def test_unknown_grader_makes_run_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("workspace", "data", "outside"):
                (root / name).mkdir()
            context = GradeContext(
                root / "workspace",
                root / "outside",
                root / "data",
                "ok",
                {},
                (),
                (),
            )
            grade = grade_case(_case(graders=({"type": "not-a-grader"},)), context)[0]
            self.assertFalse(grade.valid)
            self.assertFalse(grade.passed)

    def test_scoped_security_check_does_not_replace_global_artifact_gate(self) -> None:
        marker = "runtime-private-marker"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("workspace", "data", "outside"):
                (root / name).mkdir()
            (root / "workspace" / "leak.txt").write_text(marker, encoding="utf-8")
            grades = grade_case(
                _case(
                    graders=({"type": "content", "contains": ["safe"]},),
                    hard_failures=({"type": "security", "scope": "output"},),
                ),
                GradeContext(
                    root / "workspace",
                    root / "outside",
                    root / "data",
                    "safe",
                    {},
                    (),
                    (marker,),
                ),
            )
        self.assertEqual(len([grade for grade in grades if grade.hard]), 2)
        self.assertFalse(grades[-1].passed)


def _case(*, graders=(), hard_failures=()) -> EvalCase:
    return EvalCase(
        schema_version=1,
        id="dm-test-grading-001",
        version=1,
        suite="context",
        kind="regression",
        title="Synthetic grader contract",
        okr={"objective": "O1", "key_results": ["KR1.2"]},
        environment={},
        agent={},
        permissions={},
        conversation=({"role": "user", "content": "test"},),
        budgets={},
        graders=tuple(graders),
        hard_failures=tuple(hard_failures),
        metrics=(),
    )


if __name__ == "__main__":
    unittest.main()
