"""Repository-local evaluation harness for Deepmate.

The package deliberately lives outside ``src/``: it is a development and
release capability, not part of the published Deepmate wheel.
"""

from evals.model import (
    EvalCase,
    EvalCaseResult,
    EvalComparison,
    EvalGrade,
    EvalMetrics,
    EvalRunManifest,
    EvalRunResult,
    EvalTrial,
)

__all__ = [
    "EvalCase",
    "EvalCaseResult",
    "EvalComparison",
    "EvalGrade",
    "EvalMetrics",
    "EvalRunManifest",
    "EvalRunResult",
    "EvalTrial",
]
