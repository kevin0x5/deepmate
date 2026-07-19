"""Evaluation runner that drives Deepmate's real ``run_user_turn`` loop."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Iterable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Thread
from typing import Callable
from uuid import uuid4

from deepmate.context import build_profile_context_snapshot
from deepmate.domain import Message, MessageRole, ProfileRef
from deepmate.runtime import (
    ConversationBudgetPolicy,
    ProviderRetryPolicy,
    ToolAccessMode,
    ToolAccessPolicy,
    ToolOutputCompactor,
    run_user_turn,
)
from deepmate.runtime.activation import start_runtime_activation
from deepmate.runtime.session_runtime import SessionRuntime
from deepmate.runtime.followup import TurnFollowupBuffer
from deepmate.storage import ToolOutputStore
from deepmate.tools import (
    NativeTool,
    NativeToolRegistry,
    NativeToolResult,
    tool_output_tools,
    workspace_filesystem_tools,
)
from deepmate.trace import TraceEvent, TraceRecorder, TraceSpan

from evals.environment import EvalEnvironment, TrialEnvironment, environment_summary
from evals.grading import GradeContext, grade_case
from evals.loader import (
    REPOSITORY_ROOT,
    case_set_digest,
    digest_value,
    fixture_digest,
    policy_digest,
)
from evals.metrics import aggregate_metrics, trial_metrics
from evals.model import (
    EvalCase,
    EvalCaseResult,
    EvalRunManifest,
    EvalRunResult,
    EvalTrial,
)
from evals.privacy import sanitize_value, scan_value


DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "var" / "evals"


class _MemoryTraceSink:
    def __init__(self) -> None:
        self.records: list[Mapping[str, object]] = []

    def write(self, record: TraceEvent | TraceSpan) -> None:
        self.records.append(record.to_record())


def run_suite(
    cases: Iterable[EvalCase],
    *,
    trials: int = 1,
    provider_name: str = "fake",
    model: str = "deepmate-eval-scripted-v1",
    judge: str = "fact-coverage-v1",
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    pricing: Mapping[str, object] | None = None,
    variants: Iterable[str] = (),
    framework_tests_passed: bool | None = None,
    provider_factory: Callable[[], object] | None = None,
    environment_metadata: Mapping[str, object] | None = None,
) -> EvalRunResult:
    """Run cases independently and persist only sanitized artifacts."""
    selected = tuple(cases)
    if not selected:
        raise ValueError("run_suite requires at least one case")
    if trials < 1 or trials > 20:
        raise ValueError("trials must be between 1 and 20")
    variant_names = tuple(value.strip() for value in variants if value.strip())
    configured = tuple(_apply_variants(case, variant_names) for case in selected)
    now = datetime.now(UTC)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run_dir = Path(output_root).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "RUNNING").write_text("evaluation run in progress\n", encoding="utf-8")
    environment = EvalEnvironment(run_dir)
    manifest = _manifest(
        run_id,
        configured,
        provider_name=provider_name,
        model=model,
        judge=judge,
        pricing=pricing or {},
        variants=variant_names,
        environment_metadata=environment_metadata or {},
    )
    results: list[EvalCaseResult] = []
    for case in configured:
        for trial_index in range(1, trials + 1):
            trial_environment = environment.create(case, trial_index)
            if provider_factory is not None:
                trial_environment.provider = provider_factory()
            results.append(
                run_case_trial(
                    case,
                    trial_environment,
                    trial_index=trial_index,
                    model=model,
                    pricing=pricing or {},
                )
            )
    metrics = aggregate_metrics(results)
    valid = all(result.valid for result in results)
    invalid_reason = "" if valid else "one or more cases had invalid configuration or graders"
    run = EvalRunResult(
        manifest=manifest,
        cases=tuple(results),
        metrics=metrics,
        output_dir=f"var/evals/{run_id}",
        valid=valid,
        invalid_reason=invalid_reason,
        framework_tests_passed=framework_tests_passed,
    )
    from evals.report import write_run_artifacts

    write_run_artifacts(run, run_dir)
    findings = scan_value(
        run.to_record(),
        source="sanitized run record",
    )
    if findings:
        raise RuntimeError("sanitized evaluation record failed privacy validation")
    environment.cleanup()
    return run


def run_case_trial(
    case: EvalCase,
    environment: TrialEnvironment,
    *,
    trial_index: int,
    model: str,
    pricing: Mapping[str, object],
) -> EvalCaseResult:
    """Run one independent case through the real Deepmate runtime."""
    trace_sink = _MemoryTraceSink()
    trace_recorder = TraceRecorder(trace_sink)
    output_store = ToolOutputStore.in_data_dir(
        environment.data_dir,
        "eval",
        "eval-session",
    )
    registry = _tool_registry(environment, output_store)
    schemas = registry.schemas()
    if str(case.environment.get("schema_mode", "eager")) == "progressive":
        disclosed = {
            str(value)
            for value in case.agent.get(
                "disclosed_tools",
                ("read_text_file", "list_directory", "eval_mcp_call"),
            )
        }
        schemas = tuple(schema for schema in schemas if str(schema.get("name", "")) in disclosed)
    mode = str(case.permissions.get("mode", "read_only"))
    access_mode = (
        ToolAccessMode.WORKSPACE_WRITE
        if mode == "workspace_write"
        else ToolAccessMode.READ_ONLY
    )
    approval = bool(case.permissions.get("approve", False))
    policy = ToolAccessPolicy(
        mode=access_mode,
        shell_enabled=False,
        approval_callback=(lambda _tool, _decision: approval),
    )
    profile = ProfileRef(
        name="eval",
        uri=str(environment.data_dir / "profiles" / "eval"),
    )
    messages = tuple(
        Message(
            role=MessageRole(str(item.get("role", "user"))),
            content=str(item.get("content", "")),
        )
        for item in case.conversation
    )
    max_steps = min(64, max(1, int(case.budgets.get("max_steps", 4))))
    retry_attempts = min(5, max(1, int(case.budgets.get("provider_attempts", 2))))
    started_at = environment.clock.now().isoformat()
    wall_started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    runtime_result = run_user_turn(
        provider=environment.provider,
        workspace=environment.workspace,
        profile=profile,
        messages=messages,
        model=model,
        native_tools=registry,
        tool_schemas=schemas,
        tool_access_policy=policy,
        provider_retry_policy=ProviderRetryPolicy(
            max_attempts=retry_attempts,
            initial_delay_seconds=0,
        ),
        tool_output_compactor=ToolOutputCompactor(
            store=output_store,
            policy=ConversationBudgetPolicy(
                model_context_tokens=100_000,
                response_token_reserve=0,
                safety_margin_tokens=0,
            ),
        ),
        max_steps=max_steps,
        trace_recorder=trace_recorder,
    )
    elapsed = time.perf_counter() - started
    final_output = runtime_result.final_step().response.content
    observations = _observations(case, environment, runtime_result)
    raw_trace = tuple(trace_sink.records)
    grade_context = GradeContext(
        workspace=environment.workspace,
        outside_dir=environment.outside_dir,
        data_dir=environment.data_dir,
        final_output=final_output,
        observations=observations,
        trace=raw_trace,
        canaries=environment.canaries,
    )
    grades = grade_case(case, grade_context)
    valid = all(grade.valid for grade in grades)
    invalid_reason = "" if valid else "one or more graders were invalid"
    replacements = {
        str(environment.root): "<trial-root>",
        str(REPOSITORY_ROOT): "<repository-root>",
    }
    safe_output = str(
        sanitize_value(
            final_output,
            replacements=replacements,
            canaries=environment.canaries,
        )
    )
    safe_observations_raw = sanitize_value(
        observations,
        replacements=replacements,
        canaries=environment.canaries,
    )
    safe_trace_raw = sanitize_value(
        raw_trace,
        replacements=replacements,
        canaries=environment.canaries,
    )
    safe_observations = (
        dict(safe_observations_raw)
        if isinstance(safe_observations_raw, Mapping)
        else {}
    )
    safe_trace = tuple(
        dict(item)
        for item in safe_trace_raw
        if isinstance(item, Mapping)
    ) if isinstance(safe_trace_raw, list) else ()
    trial = EvalTrial(
        trial_id=f"{case.id}-t{trial_index}",
        case_id=case.id,
        index=trial_index,
        seed=int(digest_value([case.id, trial_index])[:8], 16),
        workspace="<trial-root>/workspace",
        data_dir="<trial-root>/data",
        started_at=started_at,
        ended_at=environment.clock.now().isoformat(),
        wall_started_at=wall_started_at,
        wall_ended_at=datetime.now(UTC).isoformat(),
    )
    metrics = trial_metrics(
        case,
        runtime_result,
        elapsed_seconds=elapsed,
        schema_tokens=max(1, len(json.dumps(schemas, sort_keys=True)) // 4),
        pricing=pricing,
    )
    return EvalCaseResult(
        trial=trial,
        grades=grades,
        metrics=metrics,
        final_output=safe_output,
        observations=safe_observations,
        trace=safe_trace,
        valid=valid,
        invalid_reason=invalid_reason,
    )


def _tool_registry(
    environment: TrialEnvironment,
    output_store: ToolOutputStore,
) -> NativeToolRegistry:
    effect_path = environment.workspace / "effects.json"

    def record_effect(arguments: Mapping[str, object]) -> NativeToolResult:
        key = str(arguments.get("idempotency_key", "")).strip()
        if not key:
            raise ValueError("idempotency_key is required")
        existing: list[str] = []
        if effect_path.is_file():
            loaded = json.loads(effect_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                existing = [str(value) for value in loaded]
        duplicate = key in existing
        if not duplicate:
            existing.append(key)
            effect_path.write_text(json.dumps(existing, sort_keys=True), encoding="utf-8")
        return NativeToolResult(
            content="effect already recorded" if duplicate else "effect recorded",
            data={"duplicate": duplicate, "count": len(existing)},
            refs=("effects.json",),
        )

    def mcp_call(arguments: Mapping[str, object]) -> NativeToolResult:
        name = str(arguments.get("name", "")).strip()
        response = environment.fake_mcp.call(name, arguments)
        return NativeToolResult(
            content=str(response.get("content", "")),
            data=response,
            refs=(f"mcp_tool={name}",),
        )

    def local_prepare(arguments: Mapping[str, object]) -> NativeToolResult:
        model = str(arguments.get("model", "synthetic-local-model")).strip()
        result = environment.fake_ollama.prepare(model)
        return NativeToolResult(
            content=(
                "local model ready"
                if result.get("ready")
                else "local model preparation interrupted"
            ),
            data=result,
            refs=(f"local_model={model}",),
        )

    def local_infer(arguments: Mapping[str, object]) -> NativeToolResult:
        model = str(arguments.get("model", "synthetic-local-model")).strip()
        prompt = str(arguments.get("prompt", "synthetic smoke"))
        result = environment.fake_ollama.infer(model, prompt)
        return NativeToolResult(
            content=str(result.get("output", result.get("error", ""))),
            data=result,
            refs=(f"local_model={model}", "local_inference=smoke"),
        )

    def large_output(_arguments: Mapping[str, object]) -> NativeToolResult:
        content = "\n".join(
            f"synthetic output line {index}: bounded retrieval payload"
            for index in range(4_000)
        )
        content += "\nretrieved-tail-marker"
        return NativeToolResult(content=content)

    tools = [
        *workspace_filesystem_tools(environment.workspace, include_write_tools=True),
        *tool_output_tools(output_store),
        NativeTool(
            name="eval_large_output",
            description="Return a large synthetic output for compaction and retrieval tests.",
            input_schema={"type": "object"},
            handler=large_output,
            read_only=True,
        ),
        NativeTool(
            name="eval_record_effect",
            description="Record one idempotent synthetic side effect in the evaluation workspace.",
            input_schema={
                "type": "object",
                "properties": {"idempotency_key": {"type": "string"}},
                "required": ["idempotency_key"],
            },
            handler=record_effect,
            read_only=False,
        ),
        NativeTool(
            name="eval_mcp_call",
            description="Call the isolated record/replay MCP fixture.",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "idempotency_key": {"type": "string"},
                },
                "required": ["name"],
            },
            handler=mcp_call,
            read_only=True,
        ),
        NativeTool(
            name="eval_local_prepare",
            description="Prepare a model using the isolated Fake Ollama lifecycle.",
            input_schema={
                "type": "object",
                "properties": {"model": {"type": "string"}},
                "required": ["model"],
            },
            handler=local_prepare,
            read_only=False,
        ),
        NativeTool(
            name="eval_local_infer",
            description="Run an isolated inference smoke check against Fake Ollama.",
            input_schema={
                "type": "object",
                "properties": {
                    "model": {"type": "string"},
                    "prompt": {"type": "string"},
                },
                "required": ["model", "prompt"],
            },
            handler=local_infer,
            read_only=True,
        ),
    ]
    return NativeToolRegistry(tools)


def _observations(
    case: EvalCase,
    environment: TrialEnvironment,
    result: object,
) -> dict[str, object]:
    tool_calls: list[str] = []
    denied_tools: list[str] = []
    tool_error_count = 0
    for exchange in getattr(result, "tool_exchanges", ()):
        for request in exchange.tool_requests:
            tool_calls.append(request.name)
        for tool_result in exchange.tool_results:
            if tool_result.is_error:
                tool_error_count += 1
                denied_tools.append(tool_result.name)
    effect_count = 0
    effect_path = environment.workspace / "effects.json"
    if effect_path.is_file():
        loaded = json.loads(effect_path.read_text(encoding="utf-8"))
        effect_count = len(loaded) if isinstance(loaded, list) else 0
    observations: dict[str, object] = {
        "runtime_completed": True,
        "runtime_error_count": len(getattr(result, "errors")()),
        "reached_max_steps": bool(getattr(result, "reached_max_steps", False)),
        "tool_calls": tool_calls,
        "tool_error_count": tool_error_count,
        "denied_tools": denied_tools,
        "side_effect_count": effect_count + len(environment.fake_mcp.side_effects),
        "mcp_calls": len(environment.fake_mcp.calls),
        "local_downloads": sum(environment.fake_ollama.downloads.values()),
        "tool_schema_count": _provider_schema_count(environment),
        "prefix_stable": _provider_prefix_stable(environment),
        "prefix_digest": _provider_prefix_digest(environment),
        "tags": list(case.tags),
        "private_marker_isolated": all(
            marker not in _workspace_text(environment.workspace)
            for marker in environment.canaries
        ),
        "hidden_assets_absent": not any(
            name in {"expected_facts.json", "verify.py", "probe.json"}
            for name in (path.name for path in environment.workspace.rglob("*"))
        ),
    }
    checks = tuple(str(value) for value in case.environment.get("checks", ()))
    if "context_refresh" in checks:
        observations.update(_context_refresh_observations(environment))
    if "restart" in checks:
        state = environment.simulate_restart()
        observations["restart_state_loaded"] = bool(state)
        observations["restart_state"] = {
            key: state[key]
            for key in (
                "status",
                "resume_hint",
                "workspace_checkpoint_id",
                "checkpoint",
                "next_action",
                "denied_approval",
            )
            if key in state
        }
    if "snapshot_stability" in checks:
        snapshot = build_profile_context_snapshot(
            environment.workspace,
            ProfileRef(name="eval", uri=str(environment.data_dir / "profiles" / "eval")),
        )
        observations["snapshot_stable"] = all(ref.read_stable for ref in snapshot.file_refs)
        observations["snapshot_generation"] = snapshot.generation[:12]
    if "local_concurrency" in checks:
        observations.update(_local_concurrency_observations(environment))
    if "evolution_evidence" in checks:
        observations.update(_evolution_evidence_observations(case))
    if "followup_buffer" in checks:
        observations.update(_followup_observations())
    return observations


def _context_refresh_observations(environment: TrialEnvironment) -> Mapping[str, object]:
    profile = ProfileRef(name="eval", uri=str(environment.data_dir / "profiles" / "eval"))
    rules = environment.workspace / "AGENTS.md"
    if not rules.exists():
        rules.write_text("Evaluation rule version alpha.\n", encoding="utf-8")
    activation = start_runtime_activation(
        "eval-session",
        environment.workspace,
        profile,
    )
    runtime = SessionRuntime(activation=activation)
    first = runtime.activation.context_snapshot
    frozen_sections = first.sections
    old_rule = rules.read_text(encoding="utf-8").strip()
    rules.write_text("Evaluation rule version beta.\n", encoding="utf-8")
    frozen_after_change = runtime.activation.context_snapshot
    refreshed_runtime = runtime.request_context_refresh_before_next_turn(
        "eval_explicit_refresh"
    ).with_refreshed_context()
    second = refreshed_runtime.activation.context_snapshot
    return {
        "context_generation_changed": first.generation != second.generation,
        "frozen_retained_old_rule": (
            frozen_after_change.generation == first.generation
            and any(old_rule in value for value in frozen_sections)
        ),
        "refresh_loaded_new_rule": any("version beta" in value for value in second.sections),
        "refresh_epoch_advanced": (
            refreshed_runtime.activation.context_epoch == runtime.activation.context_epoch + 1
        ),
    }


def _local_concurrency_observations(environment: TrialEnvironment) -> Mapping[str, object]:
    outputs: list[Mapping[str, object]] = []

    def prepare() -> None:
        outputs.append(environment.fake_ollama.prepare("synthetic-local-model"))

    threads = [Thread(target=prepare) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return {
        "concurrent_prepare_ready": all(
            bool(output.get("ready")) for output in outputs
        ),
        "concurrent_download_count": environment.fake_ollama.downloads.get(
            "synthetic-local-model",
            0,
        ),
    }


def _workspace_text(root: Path) -> str:
    chunks: list[str] = []
    for path in root.rglob("*"):
        if (
            not path.is_file()
            or path.is_symlink()
            or "__pycache__" in path.parts
            or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "\n".join(chunks)


def _provider_schema_count(environment: TrialEnvironment) -> int:
    requests = getattr(environment.provider, "requests", ())
    if not requests:
        return 0
    return len(getattr(requests[0], "tool_schemas", ()))


def _provider_prefix_stable(environment: TrialEnvironment) -> bool:
    requests = tuple(getattr(environment.provider, "requests", ()))
    if len(requests) < 2:
        return True
    prefixes: list[str] = []
    for request in requests:
        conversation = getattr(request, "conversation", ())
        first = conversation[0] if conversation else None
        message = getattr(first, "message", None)
        prefixes.append(str(getattr(message, "content", "")))
    return len(set(prefixes)) == 1


def _provider_prefix_digest(environment: TrialEnvironment) -> str:
    requests = tuple(getattr(environment.provider, "requests", ()))
    if not requests:
        return ""
    conversation = getattr(requests[0], "conversation", ())
    first = conversation[0] if conversation else None
    message = getattr(first, "message", None)
    prefix = str(getattr(message, "content", ""))
    return hashlib.sha256(prefix.encode("utf-8")).hexdigest()[:12] if prefix else ""


def _evolution_evidence_observations(case: EvalCase) -> Mapping[str, object]:
    from deepmate.evolution import (
        collect_evidence_from_records,
        tool_failure_candidates,
        user_correction_candidates,
        workflow_candidates,
    )

    raw_records = case.environment.get("evolution_records", ())
    records = (
        tuple(dict(item) for item in raw_records if isinstance(item, Mapping))
        if isinstance(raw_records, (list, tuple))
        else ()
    )
    batch = collect_evidence_from_records(records)
    return {
        "evolution_user_correction_candidates": len(
            user_correction_candidates(batch.user_corrections)
        ),
        "evolution_failure_candidates": len(
            tool_failure_candidates(batch.tool_failures)
        ),
        "evolution_workflow_candidates": len(workflow_candidates(batch.workflows)),
        "evolution_accepted_records": (
            len(batch.user_corrections) + len(batch.tool_failures) + len(batch.workflows)
        ),
    }


def _followup_observations() -> Mapping[str, object]:
    buffer = TurnFollowupBuffer()
    turn_id = buffer.start_turn()
    accepted = (
        buffer.submit(turn_id, "first follow-up")
        and buffer.submit(turn_id, "second follow-up")
    )
    messages = buffer.drain(turn_id)
    buffer.finish_turn(turn_id)
    return {
        "followup_accepted": accepted,
        "followup_messages": [message.text for message in messages],
        "stale_followup_rejected": not buffer.submit(turn_id, "stale follow-up"),
    }


def _manifest(
    run_id: str,
    cases: tuple[EvalCase, ...],
    *,
    provider_name: str,
    model: str,
    judge: str,
    pricing: Mapping[str, object],
    variants: tuple[str, ...],
    environment_metadata: Mapping[str, object],
) -> EvalRunManifest:
    commit, dirty = _git_state()
    return EvalRunManifest(
        run_id=run_id,
        created_at=datetime.now(UTC).isoformat(),
        deepmate_commit=commit,
        dirty=dirty,
        case_set_digest=case_set_digest(cases),
        fixture_digest=fixture_digest(cases),
        policy_digest=policy_digest(cases),
        provider=provider_name,
        model=model,
        judge=judge,
        permissions_digest=digest_value([dict(case.permissions) for case in cases]),
        environment={**environment_summary(), **dict(environment_metadata)},
        pricing=dict(pricing),
        variants=variants,
    )


def _git_state() -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout
        return commit, bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        return "unknown", True


def _apply_variants(case: EvalCase, variants: tuple[str, ...]) -> EvalCase:
    configured = case
    raw_variants = case.environment.get("variants")
    definitions = dict(raw_variants) if isinstance(raw_variants, Mapping) else {}
    for name in variants:
        raw_override = definitions.get(name)
        if not isinstance(raw_override, Mapping):
            raise EvalConfigurationError(
                f"{case.id}: variant {name!r} is not defined for this case"
            )
        configured = replace(
            configured,
            environment=_merged(configured.environment, raw_override.get("environment")),
            agent=_merged(configured.agent, raw_override.get("agent")),
            permissions=_merged(configured.permissions, raw_override.get("permissions")),
            budgets=_merged(configured.budgets, raw_override.get("budgets")),
            digest=digest_value([configured.digest, name, raw_override]),
        )
    return configured


def _merged(base: Mapping[str, object], override: object) -> Mapping[str, object]:
    output = dict(base)
    if isinstance(override, Mapping):
        for key, value in override.items():
            if isinstance(value, Mapping) and isinstance(output.get(key), Mapping):
                output[str(key)] = _merged(dict(output[str(key)]), value)
            else:
                output[str(key)] = value
    return output
