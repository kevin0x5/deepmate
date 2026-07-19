"""Case loading, schema validation, reference checks, and reproducible digests."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from evals.model import EvalCase


EVALS_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EVALS_ROOT.parent
CASES_ROOT = EVALS_ROOT / "cases"
FIXTURES_ROOT = EVALS_ROOT / "fixtures"
PROBES_ROOT = EVALS_ROOT / "probes"
GRADERS_ROOT = EVALS_ROOT / "graders"
BASELINES_ROOT = EVALS_ROOT / "baselines"
SCHEMA_PATH = EVALS_ROOT / "case.schema.json"
KNOWN_KEY_RESULTS = frozenset(
    {
        "KR1.1", "KR1.2", "KR1.3",
        "KR2.1", "KR2.2", "KR2.3",
        "KR3.1", "KR3.2",
        "KR4.1", "KR4.2", "KR4.3",
    }
)
EXPECTED_CASE_COUNTS = {
    "context": 8,
    "long-task": 7,
    "tool-governance": 7,
    "evolution": 5,
    "local-model": 3,
    "security": 15,
}


class EvalConfigurationError(ValueError):
    """The evaluation assets cannot produce a valid run."""


def load_cases(root: Path = CASES_ROOT) -> tuple[EvalCase, ...]:
    """Load and validate every case below ``root`` in stable order."""
    if not root.is_dir():
        raise EvalConfigurationError(f"case directory does not exist: {root}")
    cases = tuple(load_case(path) for path in sorted(root.rglob("*.yaml")))
    if not cases:
        raise EvalConfigurationError(f"no evaluation cases found under {root}")
    duplicate_ids = _duplicates(case.id for case in cases)
    if duplicate_ids:
        raise EvalConfigurationError(
            f"duplicate evaluation case ids: {', '.join(duplicate_ids)}"
        )
    return cases


def load_case(path: str | Path) -> EvalCase:
    """Load one JSON-compatible YAML case and resolve its references."""
    case_path = _safe_asset_path(Path(path), CASES_ROOT, "case")
    raw = _load_yaml_mapping(case_path)
    _validate_schema(raw, case_path)
    digest = digest_value(raw)
    case = EvalCase(
        schema_version=int(raw["schema_version"]),
        id=str(raw["id"]),
        version=int(raw["version"]),
        suite=str(raw["suite"]),
        kind=str(raw["kind"]),
        title=str(raw["title"]),
        okr=_mapping(raw["okr"], "okr", case_path),
        execution=_mapping(
            raw.get(
                "execution",
                {"class": "deterministic-runtime", "modes": ["offline"]},
            ),
            "execution",
            case_path,
        ),
        environment=_mapping(raw["environment"], "environment", case_path),
        agent=_mapping(raw["agent"], "agent", case_path),
        permissions=_mapping(raw["permissions"], "permissions", case_path),
        conversation=tuple(
            _mapping(item, "conversation item", case_path)
            for item in _sequence(raw["conversation"], "conversation", case_path)
        ),
        budgets=_mapping(raw["budgets"], "budgets", case_path),
        graders=tuple(
            _mapping(item, "grader", case_path)
            for item in _sequence(raw["graders"], "graders", case_path)
        ),
        hard_failures=tuple(
            _mapping(item, "hard failure", case_path)
            for item in _sequence(raw["hard_failures"], "hard_failures", case_path)
        ),
        metrics=tuple(str(value) for value in _sequence(raw["metrics"], "metrics", case_path)),
        presets=tuple(str(value) for value in raw.get("presets", ())),
        tags=tuple(str(value) for value in raw.get("tags", ())),
        path=case_path,
        digest=digest,
    )
    _validate_case_references(case)
    return case


def validate_repository_assets(cases: Iterable[EvalCase]) -> tuple[str, ...]:
    """Return human-readable validation findings; an empty tuple means valid."""
    loaded = tuple(cases)
    findings: list[str] = []
    suite_counts = {suite: 0 for suite in EXPECTED_CASE_COUNTS}
    for case in loaded:
        suite_counts[case.suite] = suite_counts.get(case.suite, 0) + 1
        key_results = tuple(str(value) for value in case.okr.get("key_results", ()))
        unknown = tuple(value for value in key_results if value not in KNOWN_KEY_RESULTS)
        if unknown:
            findings.append(f"{case.id}: unknown OKR references {', '.join(unknown)}")
        if case.is_safety() and not case.hard_failures:
            findings.append(f"{case.id}: safety case requires at least one hard failure")
    if len(loaded) != 45:
        findings.append(f"expected 45 cases, found {len(loaded)}")
    for suite, expected in EXPECTED_CASE_COUNTS.items():
        actual = suite_counts.get(suite, 0)
        if actual != expected:
            findings.append(f"suite {suite}: expected {expected}, found {actual}")
    pr_core = tuple(
        case for case in loaded if "pr" in case.presets and not case.is_safety()
    )
    pr_safety = tuple(
        case for case in loaded if "pr" in case.presets and case.is_safety()
    )
    if len(pr_core) != 15:
        findings.append(f"PR preset requires 15 core cases, found {len(pr_core)}")
    if len(pr_safety) != 15:
        findings.append(f"PR preset requires 15 safety cases, found {len(pr_safety)}")
    findings.extend(
        _validate_offline_pr_baseline(
            tuple(case for case in loaded if "pr" in case.presets)
        )
    )
    return tuple(findings)


def _validate_offline_pr_baseline(pr_cases: tuple[EvalCase, ...]) -> tuple[str, ...]:
    path = BASELINES_ROOT / "offline-pr-v1.json"
    if not path.is_file():
        return ("offline PR baseline is missing",)
    try:
        baseline = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ("offline PR baseline is invalid JSON",)
    digests = baseline.get("digests", {}) if isinstance(baseline, Mapping) else {}
    if not isinstance(digests, Mapping):
        return ("offline PR baseline digests are missing",)
    expected = {
        "case_set": case_set_digest(pr_cases),
        "fixtures": fixture_digest(pr_cases),
        "policy": policy_digest(pr_cases),
        "permissions": digest_value([dict(case.permissions) for case in pr_cases]),
    }
    return tuple(
        f"offline PR baseline {name} digest is stale"
        for name, value in expected.items()
        if digests.get(name) != value
    )


def select_cases(
    cases: Iterable[EvalCase],
    *,
    preset: str = "",
    suite: str = "",
    case_id: str = "",
    execution_mode: str = "",
) -> tuple[EvalCase, ...]:
    """Select cases without silently accepting an empty or ambiguous filter."""
    selected = tuple(cases)
    if preset:
        selected = tuple(case for case in selected if preset in case.presets)
    if suite:
        selected = tuple(
            case
            for case in selected
            if case.suite == suite or suite in case.tags
        )
    if case_id:
        selected = tuple(case for case in selected if case.id == case_id)
    if execution_mode:
        selected = tuple(case for case in selected if case.supports_mode(execution_mode))
    if not selected:
        filters = (
            f"preset={preset!r}, suite={suite!r}, case={case_id!r}, "
            f"execution_mode={execution_mode!r}"
        )
        raise EvalConfigurationError(f"no cases matched {filters}")
    return selected


def case_set_digest(cases: Iterable[EvalCase]) -> str:
    return digest_value([(case.id, case.version, case.digest) for case in cases])


def fixture_digest(cases: Iterable[EvalCase]) -> str:
    paths: set[Path] = set()
    for case in cases:
        fixture = case.environment.get("fixture")
        if isinstance(fixture, str) and fixture.strip():
            root = _fixture_path(fixture)
            paths.update(path for path in root.rglob("*") if _is_asset_file(path))
        for grader in (*case.graders, *case.hard_failures):
            probe = grader.get("probe")
            if isinstance(probe, str) and probe.strip():
                paths.add(_safe_asset_path(PROBES_ROOT / probe, PROBES_ROOT, "probe"))
            verifier = grader.get("verifier")
            if isinstance(verifier, str) and verifier.strip():
                paths.add(_safe_asset_path(GRADERS_ROOT / verifier, GRADERS_ROOT, "grader"))
    return digest_files(paths)


def policy_digest(cases: Iterable[EvalCase]) -> str:
    return digest_value(
        [
            {
                "id": case.id,
                "execution": dict(case.execution),
                "permissions": dict(case.permissions),
                "budgets": dict(case.budgets),
                "graders": [dict(item) for item in case.graders],
                "hard_failures": [dict(item) for item in case.hard_failures],
            }
            for case in cases
        ]
    )


def digest_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({item.resolve() for item in paths}):
        digest.update(str(path.relative_to(EVALS_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def digest_value(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_case_references(case: EvalCase) -> None:
    fixture = case.environment.get("fixture")
    if isinstance(fixture, str) and fixture.strip():
        _fixture_path(fixture)
    for grader in (*case.graders, *case.hard_failures):
        probe = grader.get("probe")
        if isinstance(probe, str) and probe.strip():
            _safe_asset_path(PROBES_ROOT / probe, PROBES_ROOT, "probe")
        verifier = grader.get("verifier")
        if isinstance(verifier, str) and verifier.strip():
            _safe_asset_path(GRADERS_ROOT / verifier, GRADERS_ROOT, "grader")
    _validate_case_contract(case)


def _validate_case_contract(case: EvalCase) -> None:
    allowed_modes = {"offline", "online", "local"}
    modes = case.execution.get("modes", ("offline",))
    if not isinstance(modes, (list, tuple)) or not modes:
        raise EvalConfigurationError(f"{case.id}: execution.modes must not be empty")
    unknown_modes = sorted(set(str(value) for value in modes).difference(allowed_modes))
    if unknown_modes:
        raise EvalConfigurationError(
            f"{case.id}: unsupported execution modes {', '.join(unknown_modes)}"
        )
    if case.execution.get("class", "deterministic-runtime") not in {
        "deterministic-runtime",
        "product-capability",
    }:
        raise EvalConfigurationError(f"{case.id}: unsupported execution class")
    if case.permissions.get("mode", "read_only") not in {
        "read_only",
        "workspace_write",
    }:
        raise EvalConfigurationError(f"{case.id}: unsupported permission mode")
    for key, lower, upper in (
        ("max_steps", 1, 64),
        ("provider_attempts", 1, 5),
    ):
        value = case.budgets.get(key)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or not lower <= value <= upper
        ):
            raise EvalConfigurationError(
                f"{case.id}: budgets.{key} must be between {lower} and {upper}"
            )
    required_by_grader = {
        "probe": "probe",
        "python": "verifier",
        "command": "command",
    }
    supported = {
        "state", "file", "command", "trace", "policy", "content",
        "probe", "security", "python",
    }
    for spec in (*case.graders, *case.hard_failures):
        grader_type = str(spec.get("type", ""))
        if grader_type not in supported:
            raise EvalConfigurationError(
                f"{case.id}: unsupported grader type {grader_type!r}"
            )
        required = required_by_grader.get(grader_type)
        if required and not spec.get(required):
            raise EvalConfigurationError(
                f"{case.id}: {grader_type} grader requires {required}"
            )


def _fixture_path(reference: str) -> Path:
    path = _safe_asset_path(FIXTURES_ROOT / reference, FIXTURES_ROOT, "fixture")
    if not path.is_dir():
        raise EvalConfigurationError(f"fixture must be a directory: {reference}")
    return path


def _safe_asset_path(path: Path, root: Path, label: str) -> Path:
    candidate = path.absolute()
    root_absolute = root.absolute()
    try:
        relative = candidate.relative_to(root_absolute)
    except ValueError as exc:
        raise EvalConfigurationError(f"{label} path escapes its asset root: {path}") from exc
    current = root_absolute
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise EvalConfigurationError(f"{label} may not use symlinks: {path}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise EvalConfigurationError(f"{label} path escapes its asset root: {path}") from exc
    if not resolved.exists():
        raise EvalConfigurationError(f"{label} does not exist: {path}")
    return resolved


def _load_yaml_mapping(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - JSON fixtures need no dependency.
            raise EvalConfigurationError(
                f"{path} is not JSON-compatible YAML and PyYAML is unavailable"
            ) from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise EvalConfigurationError(f"case must contain a mapping: {path}")
    return value


def _validate_schema(raw: Mapping[str, object], path: Path) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    try:
        import jsonschema  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - CI/dev normally has jsonschema.
        _minimal_schema_validation(raw, path)
        return
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(raw), key=lambda item: tuple(item.path))
    if errors:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
            for error in errors[:8]
        )
        raise EvalConfigurationError(f"invalid case {path}: {details}")


def _minimal_schema_validation(raw: Mapping[str, object], path: Path) -> None:
    required = {
        "schema_version", "id", "version", "suite", "kind", "title", "okr",
        "environment", "agent", "permissions", "conversation", "budgets",
        "graders", "hard_failures", "metrics",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise EvalConfigurationError(f"invalid case {path}: missing {', '.join(missing)}")
    if raw.get("schema_version") != 1:
        raise EvalConfigurationError(f"invalid case {path}: schema_version must be 1")
    if not re.fullmatch(r"dm-[a-z0-9]+(?:-[a-z0-9]+)+-[0-9]{3}", str(raw.get("id", ""))):
        raise EvalConfigurationError(f"invalid case id in {path}")
    allowed = required | {"execution", "presets", "tags"}
    unexpected = sorted(set(raw).difference(allowed))
    if unexpected:
        raise EvalConfigurationError(
            f"invalid case {path}: unexpected fields {', '.join(unexpected)}"
        )
    if not isinstance(raw.get("version"), int) or int(raw["version"]) < 1:
        raise EvalConfigurationError(f"invalid case {path}: version must be positive")
    if raw.get("suite") not in EXPECTED_CASE_COUNTS:
        raise EvalConfigurationError(f"invalid case {path}: unknown suite")
    if raw.get("kind") not in {"capability", "regression", "safety"}:
        raise EvalConfigurationError(f"invalid case {path}: unknown kind")
    if not isinstance(raw.get("title"), str) or len(str(raw["title"]).strip()) < 8:
        raise EvalConfigurationError(f"invalid case {path}: title is too short")
    for name in ("environment", "agent", "permissions", "budgets", "okr"):
        if not isinstance(raw.get(name), Mapping):
            raise EvalConfigurationError(f"invalid case {path}: {name} must be a mapping")
    okr = raw["okr"]
    assert isinstance(okr, Mapping)
    if okr.get("objective") not in {"O1", "O2", "O3", "O4"}:
        raise EvalConfigurationError(f"invalid case {path}: unknown objective")
    key_results = okr.get("key_results")
    if (
        not isinstance(key_results, list)
        or not key_results
        or any(str(value) not in KNOWN_KEY_RESULTS for value in key_results)
    ):
        raise EvalConfigurationError(f"invalid case {path}: invalid key_results")
    conversation = raw.get("conversation")
    if not isinstance(conversation, list) or not conversation:
        raise EvalConfigurationError(f"invalid case {path}: conversation is empty")
    for item in conversation:
        if (
            not isinstance(item, Mapping)
            or item.get("role") not in {"user", "assistant"}
            or not isinstance(item.get("content"), str)
            or not str(item.get("content", "")).strip()
        ):
            raise EvalConfigurationError(f"invalid case {path}: invalid conversation item")
    for name, allow_empty in (("graders", False), ("hard_failures", True)):
        values = raw.get(name)
        if not isinstance(values, list) or (not allow_empty and not values):
            raise EvalConfigurationError(f"invalid case {path}: invalid {name}")
        if any(
            not isinstance(item, Mapping)
            or not isinstance(item.get("type"), str)
            or not str(item.get("type", "")).strip()
            for item in values
        ):
            raise EvalConfigurationError(f"invalid case {path}: invalid {name} entry")
    for name in ("metrics", "presets", "tags"):
        values = raw.get(name, [])
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise EvalConfigurationError(f"invalid case {path}: {name} must be strings")


def _mapping(value: object, name: str, path: Path) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EvalConfigurationError(f"{path}: {name} must be a mapping")
    return dict(value)


def _sequence(value: object, name: str, path: Path) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise EvalConfigurationError(f"{path}: {name} must be a sequence")
    return tuple(value)


def _duplicates(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return tuple(sorted(duplicates))


def _is_asset_file(path: Path) -> bool:
    return (
        path.is_file()
        and not path.is_symlink()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )
