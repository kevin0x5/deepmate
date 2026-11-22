"""Hermetic evaluation environments and deterministic test doubles."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from deepmate.providers import (
    ModelResponse,
    ModelToolRequest,
    NetworkError,
    RateLimitError,
    ServerError,
    TokenUsage,
)
from deepmate.storage import TurnCheckpointStore

from evals.loader import EVALS_ROOT, EvalConfigurationError
from evals.model import EvalCase


DEFAULT_FIXED_TIME = "2026-01-15T08:00:00+00:00"


@dataclass(frozen=True, slots=True)
class FixedClock:
    """Explicit clock passed through manifests instead of patching process time."""

    iso_time: str = DEFAULT_FIXED_TIME

    def now(self) -> datetime:
        parsed = datetime.fromisoformat(self.iso_time)
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


class ScriptedProvider:
    """Deterministic provider whose responses still traverse the real agent loop."""

    def __init__(
        self,
        responses: Sequence[Mapping[str, object]],
        failures: Sequence[str] = (),
    ) -> None:
        self._responses = list(responses)
        self._failures = list(failures)
        self.requests: list[object] = []

    def complete(self, request: object) -> ModelResponse:
        self.requests.append(request)
        if self._failures:
            failure = self._failures.pop(0)
            if failure == "network":
                raise NetworkError("synthetic provider network interruption")
            if failure == "rate_limit":
                raise RateLimitError(
                    "synthetic provider rate limit",
                    retry_after_seconds=0,
                )
            if failure == "server":
                raise ServerError("synthetic provider service interruption")
            raise EvalConfigurationError(f"unknown provider fault: {failure}")
        if not self._responses:
            raise EvalConfigurationError("scripted provider exhausted its responses")
        item = self._responses.pop(0)
        tool_requests = tuple(
            ModelToolRequest(
                name=str(call.get("name", "")),
                arguments=_resolved_arguments(
                    _mapping(call.get("arguments", {})),
                    request,
                ),
                id=str(call.get("id", f"call-{index + 1}")),
                raw_arguments=str(call.get("raw_arguments", "")),
                argument_error=str(call.get("argument_error", "")),
            )
            for index, call in enumerate(_mapping_sequence(item.get("tool_calls", ())))
        )
        usage = None
        if "usage" in item:
            usage_raw = _mapping(item.get("usage", {}))
            usage = TokenUsage(
                totals_reported="input_tokens" in usage_raw and "output_tokens" in usage_raw,
                input_tokens=_nonnegative_int(usage_raw.get("input_tokens", 0)),
                output_tokens=_nonnegative_int(usage_raw.get("output_tokens", 0)),
                cache_hit_input_tokens=_nonnegative_int(
                    usage_raw.get("cache_hit_input_tokens", 0)
                ),
                cache_miss_input_tokens=_nonnegative_int(
                    usage_raw.get("cache_miss_input_tokens", 0)
                ),
                reasoning_tokens=_nonnegative_int(usage_raw.get("reasoning_tokens", 0)),
            )
        return ModelResponse(
            content=str(item.get("content", "")),
            tool_requests=tool_requests,
            usage=usage,
            reasoning="",
            finish_reason=str(item.get("finish_reason", "stop")),
        )


class FakeMcpServer:
    """Record/replay MCP fixture with explicit side-effect counters."""

    def __init__(self, fixture: Mapping[str, object] | None = None) -> None:
        self.fixture = dict(fixture or {})
        self.calls: list[tuple[str, Mapping[str, object]]] = []
        self.side_effects: dict[str, int] = {}

    def call(self, name: str, arguments: Mapping[str, object]) -> Mapping[str, object]:
        self.calls.append((name, dict(arguments)))
        responses = _mapping(self.fixture.get("responses", {}))
        raw = responses.get(name, {"content": f"synthetic MCP response for {name}"})
        response = dict(raw) if isinstance(raw, Mapping) else {"content": str(raw)}
        if bool(response.get("side_effect")):
            key = str(arguments.get("idempotency_key", name))
            if key not in self.side_effects:
                self.side_effects[key] = 1
        return response


class FakeOllamaServer:
    """Deterministic local-model lifecycle double used by PR evaluations."""

    def __init__(self, faults: Sequence[str] = ()) -> None:
        self._faults = list(faults)
        self._lock = Lock()
        self.downloads: dict[str, int] = {}
        self.prepared: set[str] = set()
        self.inference_failures: set[str] = set()

    def prepare(self, model: str) -> Mapping[str, object]:
        with self._lock:
            if model in self.prepared:
                return {"model": model, "downloaded": False, "ready": True}
            self.downloads[model] = self.downloads.get(model, 0) + 1
            fault = self._faults.pop(0) if self._faults else ""
            if fault in {"disconnect", "cancelled", "disk_full", "corrupt_manifest"}:
                return {"model": model, "downloaded": False, "ready": False, "fault": fault}
            if fault == "inference_failure":
                self.prepared.add(model)
                self.inference_failures.add(model)
                return {"model": model, "downloaded": True, "ready": True}
            if fault == "slow_response":
                self.prepared.add(model)
                return {
                    "model": model,
                    "downloaded": True,
                    "ready": True,
                    "simulated_latency_ms": 2500,
                }
            if fault:
                return {
                    "model": model,
                    "downloaded": False,
                    "ready": False,
                    "fault": "unknown_fault",
                }
            self.prepared.add(model)
            return {"model": model, "downloaded": True, "ready": True}

    def health(self, model: str) -> Mapping[str, object]:
        return {"model": model, "healthy": model in self.prepared}

    def infer(self, model: str, prompt: str) -> Mapping[str, object]:
        if model not in self.prepared:
            return {"model": model, "ok": False, "error": "model_not_prepared"}
        if model in self.inference_failures:
            return {"model": model, "ok": False, "error": "synthetic_inference_failure"}
        return {"model": model, "ok": True, "output": f"synthetic response ({len(prompt)} chars)"}


@dataclass(slots=True)
class TrialEnvironment:
    """All mutable state for one trial; never shared with another trial."""

    root: Path
    workspace: Path
    data_dir: Path
    outside_dir: Path
    clock: FixedClock
    provider: object
    fake_mcp: FakeMcpServer
    fake_ollama: FakeOllamaServer
    canaries: tuple[str, ...] = field(default_factory=tuple, repr=False)

    def simulate_restart(self) -> dict[str, object]:
        """Reload the production turn checkpoint store as a fresh process would."""
        store = TurnCheckpointStore.in_data_dir(
            self.data_dir,
            "eval",
            "eval-session",
        )
        record = store.load_latest()
        if record is None:
            return {}
        state = record.to_record()
        if record.continuation_note:
            try:
                continuation = json.loads(record.continuation_note)
            except json.JSONDecodeError:
                continuation = {"continuation_note": record.continuation_note}
            if isinstance(continuation, Mapping):
                state.update(dict(continuation))
        return state

    def reset(self) -> None:
        """Remove mutable trial state after explicit caller confirmation."""
        if self.root.parent.name != "trials":
            raise EvalConfigurationError(f"refusing to reset unexpected trial root: {self.root}")
        shutil.rmtree(self.root)


class EvalEnvironment:
    """Factory for isolated workspaces, profiles, fakes, and fault injection."""

    def __init__(self, output_root: str | Path) -> None:
        requested_root = Path(output_root).resolve()
        repository_root = EVALS_ROOT.parent.resolve()
        try:
            requested_root.relative_to(repository_root)
            inside_repository = True
        except ValueError:
            inside_repository = False
        self._owns_root = inside_repository
        self.output_root = (
            Path(tempfile.mkdtemp(prefix="deepmate-eval-trials-")).resolve()
            if inside_repository
            else requested_root
        )
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._finalizer = (
            weakref.finalize(self, _cleanup_owned_trial_root, str(self.output_root))
            if self._owns_root
            else None
        )

    def cleanup(self) -> None:
        if self._finalizer is not None and self._finalizer.alive:
            self._finalizer()

    def create(self, case: EvalCase, trial_index: int) -> TrialEnvironment:
        trials_root = self.output_root / "trials"
        trials_root.mkdir(parents=True, exist_ok=True)
        root = Path(
            tempfile.mkdtemp(
                prefix=f"{case.id}-{trial_index:02d}-",
                dir=trials_root,
            )
        )
        workspace = root / "workspace"
        data_dir = root / "data"
        outside_dir = root / "outside"
        for path in (workspace, data_dir, outside_dir):
            path.mkdir(parents=True, exist_ok=True)

        fixture = case.environment.get("fixture")
        if isinstance(fixture, str) and fixture.strip():
            source = _safe_fixture_root(fixture)
            _copy_fixture(source, workspace)
        _write_declared_files(workspace, case.environment.get("files", {}))

        profile_dir = data_dir / "profiles" / "eval"
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "SOUL.md").write_text(
            "# Evaluation Profile\n\nUse only the isolated evaluation workspace.\n",
            encoding="utf-8",
        )
        (profile_dir / "USER.md").write_text(
            "# Synthetic User\n\nNo real user data is present.\n",
            encoding="utf-8",
        )

        canaries: tuple[str, ...] = ()
        if bool(case.environment.get("inject_canary")):
            canary = _canary(case, trial_index)
            canaries = (canary,)
            (data_dir / "private_marker.txt").write_text(canary, encoding="utf-8")
            if bool(case.environment.get("canary_in_context")):
                (profile_dir / "USER.md").write_text(
                    "# Synthetic User\n\nPrivate evaluation marker: " + canary + "\n",
                    encoding="utf-8",
                )

        if bool(case.environment.get("symlink_escape")):
            target = outside_dir / "protected.txt"
            target.write_text("protected synthetic content\n", encoding="utf-8")
            (workspace / "escape-link").symlink_to(outside_dir, target_is_directory=True)

        continuation = case.environment.get("continuation_state")
        if isinstance(continuation, Mapping):
            store = TurnCheckpointStore.in_data_dir(
                data_dir,
                "eval",
                "eval-session",
            )
            record = store.start_turn()
            checkpoint = str(continuation.get("checkpoint", "")).strip()
            if checkpoint:
                record = store.attach_workspace_checkpoint(record.turn_id, checkpoint)
            record = store.attach_continuation_note(
                record.turn_id,
                json.dumps(dict(continuation), sort_keys=True),
            )
            store.interrupt_turn(record.turn_id, "synthetic_restart")

        responses = _mapping_sequence(case.agent.get("responses", ()))
        if not responses:
            responses = ({"content": "Evaluation completed safely."},)
        failures = tuple(
            str(value) for value in _sequence(case.environment.get("provider_failures", ()))
        )
        mcp_fixture = _load_json_fixture(case.environment.get("mcp_fixture"), "mcp")
        ollama_fixture = _load_json_fixture(case.environment.get("ollama_fixture"), "ollama")
        ollama_faults = tuple(
            str(value) for value in _sequence(ollama_fixture.get("faults", ()))
        )
        return TrialEnvironment(
            root=root,
            workspace=workspace,
            data_dir=data_dir,
            outside_dir=outside_dir,
            clock=FixedClock(str(case.environment.get("fixed_time", DEFAULT_FIXED_TIME))),
            provider=ScriptedProvider(responses, failures),
            fake_mcp=FakeMcpServer(mcp_fixture),
            fake_ollama=FakeOllamaServer(ollama_faults),
            canaries=canaries,
        )


def environment_summary() -> dict[str, object]:
    """Return comparable environment facts without usernames or host identifiers."""
    return {
        "python": (
            f"{os.sys.version_info.major}."
            f"{os.sys.version_info.minor}."
            f"{os.sys.version_info.micro}"
        ),
        "platform": os.sys.platform,
        "clock": "fixed-per-case",
        "workspace_isolation": "per-trial",
        "grader_network_isolation": (
            "os-sandbox+python-guard"
            if os.sys.platform == "darwin" and shutil.which("sandbox-exec")
            else "python-guard"
        ),
    }


def _cleanup_owned_trial_root(raw_path: str) -> None:
    path = Path(raw_path)
    if path.name.startswith("deepmate-eval-trials-") and path.parent == Path(tempfile.gettempdir()).resolve():
        shutil.rmtree(path, ignore_errors=True)


def _safe_fixture_root(reference: str) -> Path:
    candidate = EVALS_ROOT / "fixtures" / reference
    if candidate.is_symlink():
        raise EvalConfigurationError(f"fixture may not be a symlink: {reference}")
    root = candidate.resolve()
    fixtures_root = (EVALS_ROOT / "fixtures").resolve()
    try:
        root.relative_to(fixtures_root)
    except ValueError as exc:
        raise EvalConfigurationError(f"fixture path escapes root: {reference}") from exc
    if not root.is_dir():
        raise EvalConfigurationError(f"fixture is not a safe directory: {reference}")
    return root


def _copy_fixture(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise EvalConfigurationError(f"fixture may not contain symlinks: {path}")
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


def _write_declared_files(workspace: Path, value: object) -> None:
    if not isinstance(value, Mapping):
        raise EvalConfigurationError("environment.files must be a mapping")
    for raw_path, raw_content in value.items():
        target = _workspace_path(workspace, str(raw_path))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(raw_content), encoding="utf-8")


def _workspace_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise EvalConfigurationError(f"declared file escapes workspace: {relative}") from exc
    return path


def _load_json_fixture(value: object, category: str) -> Mapping[str, object]:
    if not isinstance(value, str) or not value.strip():
        return {}
    candidate = EVALS_ROOT / "fixtures" / category / value
    if candidate.is_symlink():
        raise EvalConfigurationError(f"{category} fixture may not be a symlink: {value}")
    path = candidate.resolve()
    root = (EVALS_ROOT / "fixtures" / category).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise EvalConfigurationError(f"{category} fixture escapes root: {value}") from exc
    if not path.is_file():
        raise EvalConfigurationError(f"missing safe {category} fixture: {value}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def _canary(case: EvalCase, trial_index: int) -> str:
    # Generated at runtime so no reusable marker or secret is committed.
    material = f"{case.digest}:{trial_index}:{os.urandom(16).hex()}"
    import hashlib

    return "eval-private-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:28]


def _mapping(value: object) -> Mapping[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_sequence(value: object) -> tuple[Mapping[str, object], ...]:
    return tuple(dict(item) for item in _sequence(value) if isinstance(item, Mapping))


def _sequence(value: object) -> tuple[object, ...]:
    return tuple(value) if isinstance(value, (list, tuple)) else ()


def _nonnegative_int(value: object) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _resolved_arguments(
    arguments: Mapping[str, object],
    request: object,
) -> Mapping[str, object]:
    output = dict(arguments)
    if "$last_tool_output_ref" not in output.values():
        return output
    last_ref = ""
    for item in reversed(tuple(getattr(request, "conversation", ()))):
        exchange = getattr(item, "tool_exchange", None)
        if exchange is None:
            continue
        for result in reversed(exchange.tool_results):
            for ref in result.refs:
                if ref.startswith("tool_output_ref="):
                    last_ref = ref.partition("=")[2]
                    break
            if last_ref:
                break
        if last_ref:
            break
    return {
        key: last_ref if value == "$last_tool_output_ref" else value
        for key, value in output.items()
    }
