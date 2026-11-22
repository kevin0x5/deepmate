"""Command-line entry point for the repository-local Deepmate eval harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from evals.loader import (
    EVALS_ROOT,
    EvalConfigurationError,
    load_cases,
    select_cases,
    validate_repository_assets,
)
from evals.metrics import compare_runs
from evals.privacy import committed_asset_paths, scan_paths
from evals.report import load_run, okr_coverage, render_markdown
from evals.runner import DEFAULT_OUTPUT_ROOT, run_suite


EXIT_PASS = 0
EXIT_FAILURE = 1
EXIT_INVALID = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 -m evals")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("objectives", help="show the OKR and case coverage")
    subparsers.add_parser("validate", help="validate all committed eval assets")

    run = subparsers.add_parser("run", help="run a preset, suite, or case")
    selection = run.add_mutually_exclusive_group(required=True)
    selection.add_argument("--preset", choices=("pr", "nightly", "release", "core"))
    selection.add_argument(
        "--suite",
        choices=(
            "context",
            "long-task",
            "tool-governance",
            "evolution",
            "local-model",
            "security",
            "core",
        ),
    )
    selection.add_argument("--case", dest="case_id")
    run.add_argument("--trials", type=int)
    _provider_arguments(run)
    run.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    run.add_argument("--skip-framework-tests", action="store_true")

    ab = subparsers.add_parser("ab", help="run one case under two or more variants")
    ab.add_argument("--case", dest="case_id", required=True)
    ab.add_argument("--variant", action="append", required=True)
    ab.add_argument("--trials", type=int, default=3)
    _provider_arguments(ab)
    ab.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))

    matrix = subparsers.add_parser("matrix", help="run one suite across models")
    matrix.add_argument("--suite", required=True)
    matrix.add_argument("--models", nargs="+", required=True)
    matrix.add_argument("--trials", type=int, default=1)
    matrix.add_argument(
        "--provider",
        default="fake",
        choices=("fake", "openai-compatible"),
    )
    matrix.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))

    compare = subparsers.add_parser("compare", help="compare equivalent saved runs")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--runs-root", default=str(DEFAULT_OUTPUT_ROOT))

    report = subparsers.add_parser("report", help="render a saved run report")
    report.add_argument("run_id")
    report.add_argument("--runs-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "objectives":
            return _objectives()
        if args.command == "validate":
            return _validate()
        if args.command == "run":
            return _run(args)
        if args.command == "ab":
            return _ab(args)
        if args.command == "matrix":
            return _matrix(args)
        if args.command == "compare":
            return _compare(args)
        if args.command == "report":
            return _report(args)
    except (
        EvalConfigurationError,
        ValueError,
        RuntimeError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return EXIT_INVALID
    return EXIT_INVALID


def _objectives() -> int:
    path = EVALS_ROOT / "OBJECTIVES.md"
    if not path.is_file():
        raise EvalConfigurationError("OBJECTIVES.md is missing")
    print(path.read_text(encoding="utf-8").rstrip())
    print("\nCase Coverage\n")
    for key_result, case_ids in sorted(okr_coverage().items()):
        print(f"- {key_result}: {len(case_ids)} case(s)")
    return EXIT_PASS


def _validate() -> int:
    cases = load_cases()
    findings = list(validate_repository_assets(cases))
    privacy_findings = scan_paths(committed_asset_paths(EVALS_ROOT))
    findings.extend(
        f"privacy:{finding.kind}:{Path(finding.source).name}:{finding.message}"
        for finding in privacy_findings
    )
    if findings:
        for finding in findings:
            print(f"- {finding}", file=sys.stderr)
        return EXIT_INVALID
    print(f"validated {len(cases)} cases; privacy and reference checks passed")
    return EXIT_PASS


def _run(args: argparse.Namespace) -> int:
    cases = load_cases()
    suite = "" if args.suite == "core" else (args.suite or "")
    preset = "core" if args.suite == "core" else (args.preset or "")
    selected = select_cases(
        cases,
        preset=preset,
        suite=suite,
        case_id=args.case_id or "",
        execution_mode="offline" if args.provider == "fake" else "online",
    )
    trials = args.trials or (
        3
        if args.preset == "nightly"
        else 5
        if args.preset == "release" and args.provider != "fake"
        else 1
    )
    framework_tests = None
    if args.preset == "pr" and not args.skip_framework_tests:
        framework_tests = _run_framework_tests()
    provider_factory = _provider_factory(args.provider)
    run = run_suite(
        selected,
        trials=trials,
        provider_name=args.provider,
        model=args.model,
        output_root=args.output_root,
        pricing=_pricing(),
        framework_tests_passed=framework_tests,
        provider_factory=provider_factory,
        environment_metadata=_provider_metadata(args.provider),
    )
    _print_run(run, Path(args.output_root))
    if not run.valid:
        return EXIT_INVALID
    return EXIT_PASS if run.passed else EXIT_FAILURE


def _ab(args: argparse.Namespace) -> int:
    if len(args.variant) < 2:
        raise EvalConfigurationError("ab requires at least two --variant values")
    selected = select_cases(
        load_cases(),
        case_id=args.case_id,
        execution_mode="offline" if args.provider == "fake" else "online",
    )
    provider_factory = _provider_factory(args.provider)
    runs = []
    for variant in args.variant:
        run = run_suite(
            selected,
            trials=args.trials,
            provider_name=args.provider,
            model=args.model,
            output_root=args.output_root,
            pricing=_pricing(),
            variants=(variant,),
            provider_factory=provider_factory,
            environment_metadata=_provider_metadata(args.provider),
        )
        runs.append(run)
    control = runs[0]
    control_values = control.metrics.values
    comparisons = []
    for variant, run in zip(args.variant, runs, strict=True):
        deltas = {}
        for key in ("task_success", "pass@1", "cache_hit_ratio", "p95_latency_seconds"):
            before = control_values.get(key)
            after = run.metrics.values.get(key)
            deltas[key] = (
                float(after) - float(before)
                if isinstance(before, (int, float)) and isinstance(after, (int, float))
                else None
            )
        comparisons.append(
            {
                "variant": variant,
                "run_id": run.manifest.run_id,
                "passed": run.passed,
                "deltas_from_control": deltas,
            }
        )
    print(json.dumps({"control": args.variant[0], "runs": comparisons}, indent=2, sort_keys=True))
    return EXIT_PASS if all(run.passed for run in runs) else EXIT_FAILURE


def _matrix(args: argparse.Namespace) -> int:
    if args.provider == "fake":
        raise EvalConfigurationError(
            "matrix requires an online provider; scripted responses cannot compare models"
        )
    cases = load_cases()
    selected = select_cases(
        cases,
        preset="core" if args.suite == "core" else "",
        suite="" if args.suite == "core" else args.suite,
        execution_mode="online",
    )
    models = tuple(
        model.strip()
        for value in args.models
        for model in value.split(",")
        if model.strip()
    )
    if not models:
        raise EvalConfigurationError("matrix requires at least one model")
    provider_factory = _provider_factory(args.provider)
    runs = [
        run_suite(
            selected,
            trials=args.trials,
            provider_name=args.provider,
            model=model,
            output_root=args.output_root,
            pricing=_pricing(),
            provider_factory=provider_factory,
            environment_metadata={
                **_provider_metadata(args.provider),
                **_matrix_metadata(),
            },
        )
        for model in models
    ]
    for model, run in zip(models, runs, strict=True):
        print(
            f"{model}: run={run.manifest.run_id} pass@1={run.metrics.values.get('pass@1')} "
            f"safety={run.safety_gate}"
        )
    return EXIT_PASS if all(run.passed for run in runs) else EXIT_FAILURE


def _compare(args: argparse.Namespace) -> int:
    baseline = load_run(args.baseline, args.runs_root)
    candidate = load_run(args.candidate, args.runs_root)
    comparison = compare_runs(baseline, candidate)
    print(json.dumps(comparison.to_record(), ensure_ascii=False, indent=2, sort_keys=True))
    if not comparison.valid:
        return EXIT_INVALID
    return EXIT_PASS if candidate.passed and candidate.safety_gate else EXIT_FAILURE


def _report(args: argparse.Namespace) -> int:
    run = load_run(args.run_id, args.runs_root)
    print(render_markdown(run).rstrip())
    if not run.valid:
        return EXIT_INVALID
    return EXIT_PASS if run.passed else EXIT_FAILURE


def _provider_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", default="fake", choices=("fake", "openai-compatible"))
    parser.add_argument("--model", default="deepmate-eval-scripted-v1")


def _provider_factory(name: str):
    if name == "fake":
        return None
    base_url = os.environ.get("DEEPMATE_EVAL_BASE_URL", "").strip()
    provider_credential = os.environ.get("DEEPMATE_EVAL_API_KEY", "").strip()
    if not base_url or not provider_credential:
        raise EvalConfigurationError(
            "openai-compatible provider requires DEEPMATE_EVAL_BASE_URL and "
            "DEEPMATE_EVAL_API_KEY; values are read at runtime and never persisted"
        )
    from deepmate.providers import ChatCompletionsProvider

    return lambda: ChatCompletionsProvider(base_url, provider_credential)


def _run_framework_tests() -> bool:
    completed = subprocess.run(
        [
            "python3", "-m", "pytest", "-q", "evals/tests",
            "--ignore=evals/tests/test_pr_preset.py",
        ],
        cwd=EVALS_ROOT.parent,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if completed.returncode:
        print(completed.stdout[-4000:], file=sys.stderr)
        print(completed.stderr[-4000:], file=sys.stderr)
    return completed.returncode == 0


def _matrix_metadata() -> dict[str, object]:
    """Read non-secret local-runtime provenance without recording endpoints."""
    names = {
        "model_digest": "DEEPMATE_EVAL_MODEL_DIGEST",
        "quantization": "DEEPMATE_EVAL_QUANTIZATION",
        "ollama_version": "DEEPMATE_EVAL_OLLAMA_VERSION",
        "hardware_class": "DEEPMATE_EVAL_HARDWARE_CLASS",
        "context_window": "DEEPMATE_EVAL_CONTEXT_WINDOW",
    }
    return {
        key: value
        for key, environment_name in names.items()
        if (value := os.environ.get(environment_name, "").strip())
    }


def _provider_metadata(name: str) -> dict[str, object]:
    if name == "fake":
        return {"measurement_source": "controlled-provider"}
    base_url = os.environ.get("DEEPMATE_EVAL_BASE_URL", "").strip()
    return {
        "measurement_source": "provider-reported",
        "provider_endpoint_digest": hashlib.sha256(base_url.encode("utf-8")).hexdigest(),
    }


def _pricing() -> dict[str, object]:
    input_price = os.environ.get(
        "DEEPMATE_EVAL_INPUT_PRICE_PER_MILLION",
        "",
    ).strip()
    output_price = os.environ.get(
        "DEEPMATE_EVAL_OUTPUT_PRICE_PER_MILLION",
        "",
    ).strip()
    if not input_price and not output_price:
        return {}
    if not input_price or not output_price:
        raise EvalConfigurationError(
            "pricing requires both input and output price provenance"
        )
    source = os.environ.get("DEEPMATE_EVAL_PRICE_SOURCE", "").strip()
    effective_at = os.environ.get("DEEPMATE_EVAL_PRICE_EFFECTIVE_AT", "").strip()
    if not source or not effective_at:
        raise EvalConfigurationError(
            "pricing requires DEEPMATE_EVAL_PRICE_SOURCE and "
            "DEEPMATE_EVAL_PRICE_EFFECTIVE_AT"
        )
    try:
        pricing = {
            "input_per_million": float(input_price),
            "output_per_million": float(output_price),
            "currency": (
                os.environ.get("DEEPMATE_EVAL_PRICE_CURRENCY", "USD").strip()
                or "USD"
            ),
            "source": source,
            "effective_at": effective_at,
        }
        cache_hit_price = os.environ.get(
            "DEEPMATE_EVAL_CACHE_HIT_INPUT_PRICE_PER_MILLION", ""
        ).strip()
        if cache_hit_price:
            pricing["cache_hit_input_per_million"] = float(cache_hit_price)
        return pricing
    except ValueError as exc:
        raise EvalConfigurationError("pricing values must be numeric") from exc


def _print_run(run, output_root: Path) -> None:
    print(f"run: {run.manifest.run_id}")
    print(f"output: {output_root / run.manifest.run_id}")
    print(f"valid: {str(run.valid).lower()}")
    print(f"passed: {str(run.passed).lower()}")
    print(f"safety_gate: {str(run.safety_gate).lower()}")
    print(f"pass@1: {run.metrics.values.get('pass@1')}")


if __name__ == "__main__":
    raise SystemExit(main())
