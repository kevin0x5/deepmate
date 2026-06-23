"""Command-line channel for the first runnable Deepmate path."""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import shlex
import sys
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import time_ns

from deepmate.activity import ActivityStore
from deepmate.app import (
    AppSettings,
    ModelCallConfig,
    ProviderSettings,
    load_settings,
    resolve_model_purpose,
)
from deepmate.app.settings import DEFAULT_MODEL_CONTEXT_TOKENS
from deepmate.app import save_provider_api_key, save_wecom_remote_settings
from deepmate.behavior import behavior_runtime_for_session
from deepmate.capabilities import (
    CapabilityMaintenanceResult,
    CapabilitySurface,
    CapabilityState,
    CapabilityStateStore,
    combine_surfaces,
    from_mcp_tool_catalog,
    from_mcp_tool_refs,
    from_native_tool_schemas,
    from_skill_cards,
    run_daily_capability_maintenance,
)
from deepmate.channels.interactive import (
    _close_remote_routes_for_local_turn,
    run_interactive_mode,
)
from deepmate.channels.tui import consume_workspace_switch_request, run_tui_mode
from deepmate.channels.wecom import WeComRunDependencies, run_wecom_remote_channel
from deepmate.channels.checkpointing import (
    SessionCheckpointController,
    SessionCheckpointWriteRouter,
)
from deepmate.channels.session_maintenance import (
    force_session_summary_checkpoint,
    run_session_maintenance,
    runtime_conversation_from_store,
    write_session_end_activity,
)
from deepmate.channels.skill_view import (
    discover_skill_cards,
    discover_workspace_skill_cards,
    format_capability_list,
    format_skill_document,
    format_skill_list,
    select_skill_documents,
    workspace_skill_catalog,
)
from deepmate.context import ContextWarning, build_profile_context_snapshot
from deepmate.cron import (
    handle_cron_command,
    maybe_create_cron_draft,
    run_due_jobs,
    run_job_now,
    watch_due_jobs,
)
from deepmate.qa import handle_qa_command, maybe_create_qa_audit, maybe_qa_agent_prompt
from deepmate.domain import Message, MessageRole
from deepmate.evolution import EvolutionChangeStore, run_evolution_maintenance
from deepmate.local.presets import (
    LOCAL_PROVIDER_API_KEY,
    LOCAL_PROVIDER_NAME,
    local_model_by_id,
    local_model_capabilities,
    local_model_by_runtime_name,
    recommended_local_model,
)
from deepmate.local import (
    LocalModelStateStore,
    OllamaLocalRuntime,
    ollama_api_url_from_provider_base_url,
)
from deepmate.mcp import (
    McpServerInventory,
    McpToolCatalog,
    McpToolExecutor,
    McpToolRef,
    McpUsageStateStore,
    discover_mcp_catalog,
    discover_mcp_tools,
    format_mcp_catalog_status,
    format_mcp_server_list,
)
from deepmate.memory import run_daily_memory_maintenance
from deepmate.pet.events import (
    event_for_task_achievement,
    event_for_turn_finished,
    event_for_turn_started,
)
from deepmate.pet.electron_host import electron_pet_command, run_pet_host
from deepmate.pet.pets import built_in_pet_ids
from deepmate.pet.state import PetStateStore, PetUserAction
from deepmate.providers import (
    AuthError,
    ChatCompletionsProvider,
    ModelCapabilities,
    ModelConversationItem,
    ModelRequest,
    ModelResponse,
    NetworkError,
    ProviderError,
    RateLimitError,
    ServerError,
    StreamDelta,
)
from deepmate.runtime import (
    ConversationBudgetPolicy,
    DeliveryReview,
    DeliveryReviewStatus,
    LoopGuardPolicy,
    ProviderRetryPolicy,
    SessionRuntime,
    ToolAccessMode,
    ToolAccessPolicy,
    ToolOutputCompactionPolicy,
    ToolOutputCompactor,
    ToolRepairPolicy,
    build_delivery_review_input,
    review_final_response,
    should_run_llm_delivery_review,
    start_session_runtime,
    start_runtime_activation,
)
from deepmate.providers.chat_completions import sanitize_model_request
from deepmate.runtime.hooks import (
    HookActor,
    HookDirective,
    HookEnvelope,
    HookEvent,
    HookLoadOptions,
    HookOutcome,
    HookRuntimeContext,
    HookSignalStore,
    HookTrustStore,
    format_hook_validation,
    format_hooks_status,
    load_hook_report,
)
from deepmate.runtime.conversation_budget import build_request_budget_report
from deepmate.runtime.model_request import build_model_request
from deepmate.runtime.sandbox import SandboxMode, SandboxPolicy, SandboxRunner
from deepmate.runtime.safety import SessionApprovalCache
from deepmate.skills import SkillCatalog
from deepmate.skills import (
    InstalledSkillManifestStore,
    format_installed_skill_list,
    format_skill_inspection,
    format_skill_install_result,
    format_skill_uninstall_result,
    format_skill_verify_result,
    inspect_skill_source,
    install_skill_source,
    uninstall_skill,
    update_skill_source,
    verify_skill_install,
)
from deepmate.storage import (
    SessionRecord,
    SessionStore,
    ToolOutputStore,
    TranscriptStore,
    TurnCheckpointStore,
    WorkspaceCheckpointStore,
    WorkspaceRewindPlan,
)
from deepmate.subagents import (
    SubagentOrchestrationPolicy,
    SubagentRuntime,
    SubagentToolExecutor,
)
from deepmate.subagents.store import SubagentResultStore
from deepmate.tasks import (
    TASK_CLEAR,
    TASK_STATUS,
    TaskContext,
    ExecuteDecision,
    ExecuteEvaluation,
    ExecuteLoopUpdate,
    TaskSessionController,
    TaskStage,
    TaskStore,
    apply_task_update_result,
    default_task_prompt,
    evaluate_execute_progress,
    evidence_from_result,
    execute_start_prompt,
    format_execute_outcome,
    generate_task_update,
    parse_task_prompt_command,
    persisted_task_stage,
    render_task_context_section,
    should_run_task_update,
    loop_update_from_evaluation,
)
from deepmate.tasks.store import DegenerateTaskPlanError, execution_contract_gaps
from deepmate.tools import (
    AgentBrowserBackend,
    NativeToolRegistry,
    browser_loader_tools,
    browser_tools,
    computer_tools,
    ComputerUseState,
    format_browser_install_result,
    format_browser_validation_result,
    install_browser_backend,
    mcp_loader_tools,
    skill_installer_tools,
    skill_loader_tools,
    shell_tools,
    tool_output_tools,
    validate_browser_backend,
    web_research_tools,
    workspace_artifact_tools,
    workspace_diagram_tools,
    workspace_document_tools,
    workspace_filesystem_tools,
    workspace_lsp_tools,
    workspace_report_tools,
    workspace_search_tools,
)
from deepmate.trace import (
    JsonlTraceSink,
    OtlpExportResult,
    TraceEvent,
    TraceRecorder,
    TraceSpan,
    export_otlp_traces,
    new_span_id,
    new_trace_id,
    summarize_trace_usage,
    trace_record_matches_kinds,
    trace_record_matches_session,
    trace_record_refs,
    trace_refs_to_map,
)

SESSION_VIEW_TRANSCRIPT_LIMIT = 20
SESSION_VIEW_TRACE_LIMIT = 20
SESSION_VIEW_PREVIEW_CHARS = 500


def _restart_with_workspace(
    workspace: Path,
    argv: Sequence[str] | None,
    *,
    session_id: str = "",
) -> int:
    """Replace the current process with the same TUI command for a new workspace."""
    updated = _argv_with_workspace(argv, workspace, session_id=session_id)
    print(f"info: opening workspace: {workspace}", file=sys.stderr)
    os.execv(sys.executable, [sys.executable, sys.argv[0], *updated])
    return 0


def _argv_with_workspace(
    argv: Sequence[str] | None,
    workspace: Path,
    *,
    session_id: str = "",
) -> list[str]:
    args = list(sys.argv[1:] if argv is None else argv)
    for index, value in enumerate(args):
        if value.startswith("--workspace="):
            args[index] = f"--workspace={workspace}"
            break
    else:
        index = -1
    if "--workspace" in args:
        flag_index = args.index("--workspace")
        if flag_index + 1 < len(args):
            args[flag_index + 1] = str(workspace)
        else:
            args.append(str(workspace))
    elif index < 0:
        args[:0] = ["--workspace", str(workspace)]
    args = _without_initial_prompt(args)
    if session_id.strip():
        args = _with_option_value(args, "--session-id", session_id.strip())
    if "--interactive" not in args and "--interactive-legacy" not in args:
        args.append("--interactive")
    return args


def _with_option_value(args: Sequence[str], option: str, value: str) -> list[str]:
    updated = list(args)
    if option in updated:
        index = updated.index(option)
        if index + 1 < len(updated) and not updated[index + 1].startswith("-"):
            updated[index + 1] = value
        else:
            updated.insert(index + 1, value)
        return updated
    updated.extend((option, value))
    return updated


def _without_initial_prompt(args: Sequence[str]) -> list[str]:
    options_with_values = {
        "--workspace",
        "--profile",
        "--provider",
        "--model",
        "--base-url",
        "--api-key-env",
        "--max-steps",
        "--task",
        "--remote",
        "--session-title",
        "--thinking",
        "--sandbox",
        "--skill",
        "--show-skill",
        "--inspect-skill",
        "--install-skill",
        "--update-skill",
        "--verify-skill",
        "--uninstall-skill",
        "--skill-name",
        "--skill-target",
        "--skill-state",
        "--rewind",
        "--rewind-to",
        "--rewind-mode",
        "--activity-date",
        "--activity-month",
        "--activity-lines",
        "--memory-maintenance-date",
        "--rollback-evolution-change",
        "--session-items",
        "--trace-events",
        "--trace-kind",
        "--otlp-endpoint",
        "--temperature",
        "--max-tokens",
        "--pet-select",
        "--pet-learning",
        "--pet-bubble",
        "--pet-name",
        "--pet-model",
        "--list-mcp-tools",
        "--reasoning-effort",
        "--behavior-learning",
        "--computer-learning",
        "-m",
    }
    dropped_options_with_values = {"--session-id"}
    cleaned: list[str] = []
    iterator = iter(range(len(args)))
    for index in iterator:
        value = args[index]
        if not value.startswith("-"):
            continue
        if value in dropped_options_with_values:
            if index + 1 < len(args) and not args[index + 1].startswith("-"):
                next(iterator, None)
            continue
        cleaned.append(value)
        if (
            value in options_with_values
            and index + 1 < len(args)
            and not args[index + 1].startswith("-")
        ):
            cleaned.append(args[index + 1])
            next(iterator, None)
    return cleaned


def main(argv: Sequence[str] | None = None) -> int:
    """Run one real model-backed Deepmate CLI session."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    prompt = " ".join(args.prompt).strip()
    task_stage_arg, task_prompt = _parse_task_cli(args.task, args.prompt)
    task_prompt_command = parse_task_prompt_command(task_prompt)
    task_control_command = (
        task_prompt_command.control
        if task_prompt_command is not None and task_prompt_command.is_control()
        else ""
    )
    task_requested = args.task is not None or task_prompt_command is not None
    if args.task is not None:
        prompt = task_prompt
    if task_prompt_command is not None and not task_prompt_command.is_control():
        task_stage_arg = task_prompt_command.stage
        prompt = task_prompt_command.prompt
    elif task_prompt_command is not None and task_prompt_command.is_control():
        prompt = task_prompt_command.prompt
    if (
        not prompt
        and not task_requested
        and not args.validate_runtime
        and not args.doctor
        and args.cron is None
        and not args.cron_runner
        and args.qa is None
        and not args.remote
        and not args.remote_validate
        and not args.setup_key
        and not args.setup_wecom
        and not args.interactive
        and not args.interactive_legacy
        and not args.activity_today
        and not args.activity_date
        and not args.activity_month
        and not args.run_memory_maintenance
        and not args.run_evolution_maintenance
        and not args.list_evolution_changes
        and not args.run_capability_maintenance
        and not args.rollback_evolution_change
        and not args.install_memory_maintenance_schedule
        and not args.list_sessions
        and not args.list_skills
        and not args.list_capabilities
        and not args.list_mcp
        and not args.mcp_status
        and not args.sandbox_status
        and not args.local_status
        and args.prepare_local_model is None
        and not args.validate_browser
        and not args.install_browser
        and not args.behavior_status
        and not args.pet
        and not args.pet_status
        and not args.pet_select
        and not args.pet_actions
        and not _pet_settings_requested(args)
        and not args.hooks_status
        and not args.validate_hooks
        and not args.trust_workspace
        and not args.list_mcp_tools
        and not args.show_skill
        and not args.inspect_skill
        and not args.install_skill
        and not args.update_skill
        and not args.verify_skill
        and not args.uninstall_skill
        and not args.list_skill_installs
        and not args.skill_state
        and not args.show_session
        and not args.validate_otlp
        and not args.rewind
    ):
        args.interactive = True
    if args.interactive_legacy:
        args.interactive = True
    if args.validate_runtime and args.interactive:
        print(
            "error: --validate-runtime cannot be combined with --interactive",
            file=sys.stderr,
        )
        return 2
    if args.cron is not None and args.interactive:
        print("error: --cron cannot be combined with --interactive", file=sys.stderr)
        return 2
    if args.qa is not None and args.interactive:
        print("error: --qa cannot be combined with --interactive", file=sys.stderr)
        return 2
    if args.cron_runner and args.interactive:
        print("error: --cron-runner cannot be combined with --interactive", file=sys.stderr)
        return 2
    if (args.cron_watch or args.cron_poll_seconds != 60) and not args.cron_runner:
        print(
            "error: --cron-watch and --cron-poll-seconds require --cron-runner",
            file=sys.stderr,
        )
        return 2
    if args.doctor and args.interactive:
        print("error: --doctor cannot be combined with --interactive", file=sys.stderr)
        return 2
    if args.remote and args.interactive:
        print("error: --remote cannot be combined with --interactive", file=sys.stderr)
        return 2
    if args.remote and prompt:
        print("error: --remote does not accept an initial prompt", file=sys.stderr)
        return 2
    if args.validate_runtime and task_requested:
        print(
            "error: --validate-runtime cannot be combined with --task",
            file=sys.stderr,
        )
        return 2
    evolution_commands = sum(
        bool(value)
        for value in (
            args.run_evolution_maintenance,
            args.list_evolution_changes,
            args.rollback_evolution_change,
        )
    )
    if evolution_commands > 1:
        print(
            "error: choose only one of --run-evolution-maintenance, "
            "--list-evolution-changes, or --rollback-evolution-change",
            file=sys.stderr,
        )
        return 2
    if args.validate_runtime:
        prompt = _runtime_validation_prompt(args, prompt)

    try:
        settings = load_settings(args.workspace)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if task_control_command:
        controller = TaskSessionController(settings.workspace)
        try:
            print(controller.handle_control(task_control_command))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    args.max_steps = _effective_max_steps(settings, args.max_steps)

    pet_store = PetStateStore.in_data_dir(settings.data_dir)
    session_store = SessionStore.in_directory(settings.data_dir / "sessions")

    if args.cron is not None:
        try:
            print(_handle_cron_cli(settings.workspace, args.cron))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.cron_runner:
        try:
            if args.cron_watch:
                print(
                    "Cron runner watching "
                    f"{settings.workspace} every {max(5, args.cron_poll_seconds)}s. "
                    "Press Ctrl+C to stop.",
                    flush=True,
                )
                watch_due_jobs(
                    workspace=settings.workspace,
                    poll_seconds=args.cron_poll_seconds,
                    report_sink=lambda line: print(line, flush=True),
                )
            else:
                print(run_due_jobs(workspace=settings.workspace))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("Cron runner stopped.", file=sys.stderr)
            return 130
        return 0
    if prompt and not task_requested and not args.cron_job_run:
        try:
            cron_message = maybe_create_cron_draft(prompt, workspace=settings.workspace)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if cron_message is not None:
            print(cron_message)
            return 0
        qa_prompt = maybe_qa_agent_prompt(prompt, workspace=settings.workspace)
        if qa_prompt is not None:
            prompt = qa_prompt

    if args.pet_select or _pet_settings_requested(args):
        try:
            profile = _apply_pet_profile_settings(pet_store, args)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(_format_pet_profile_update(profile, pet_store.load_learning_state().get("sources")))
        if not args.pet and not args.pet_status and not args.pet_actions:
            return 0
    if args.pet_actions:
        print(_format_pet_actions(pet_store, session_store))
        return 0
    if args.pet_status and not args.pet:
        try:
            print(_format_pet_status(pet_store))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.pet:
        provider, pet_model = _pet_copy_provider(settings, args)
        return run_pet_host(settings.data_dir, provider=provider, model=pet_model)

    task_store: TaskStore | None = None
    task_stage: TaskStage | None = None
    task_context: TaskContext | None = None
    task_cursor_stage: TaskStage | None = None
    task_controller: TaskSessionController | None = (
        TaskSessionController(settings.workspace) if args.interactive else None
    )
    if task_requested:
        if task_controller is None:
            task_controller = TaskSessionController(settings.workspace)
        task_store = task_controller.store
        task_store.ensure()
        task_cursor_stage = task_store.resolve_stage(None)
        task_stage = task_store.resolve_stage(task_stage_arg)
        task_controller.enable(task_stage)
        task_context = task_store.context_for_stage(task_stage)
        if not prompt:
            prompt = default_task_prompt(task_stage)
        if task_stage == TaskStage.EXECUTE:
            prompt = execute_start_prompt(prompt)
    if args.list_sessions:
        _print_sessions(session_store.list_recent())
        return 0
    if args.list_skills:
        _print_skills(settings.workspace)
        return 0
    if args.list_capabilities:
        try:
            _print_capabilities(settings, args.profile)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.trust_workspace:
        try:
            record = HookTrustStore.in_data_dir(settings.data_dir).trust_workspace(
                settings.workspace
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print("Workspace trusted for project hooks:")
        print(f"- workspace: {record.workspace}")
        print(f"- workspace_hash: {record.workspace_hash[:16]}")
        print(f"- trusted_at: {record.trusted_at}")
        return 0
    if args.hooks_status:
        try:
            report = load_hook_report(
                settings.workspace,
                settings.data_dir,
                _hook_load_options(settings),
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(format_hooks_status(report, settings.workspace))
        return 0
    if args.validate_hooks:
        try:
            report = load_hook_report(
                settings.workspace,
                settings.data_dir,
                _hook_load_options(settings),
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(format_hook_validation(report, settings.workspace))
        return 1 if report.has_errors() else 0
    if args.skill_state:
        try:
            state = _apply_skill_state_command(settings, args.skill_state, args.profile)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(
            f"skill {state.name}: temperature={state.temperature.value}, "
            f"exposure={state.exposure()}"
        )
        return 0
    if args.remote_validate:
        try:
            _validate_remote_settings(settings, args.remote)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.setup_key is not None:
        try:
            provider_settings = settings.provider(args.provider)
            path = _save_provider_key_command(
                settings,
                provider_settings.api_key_env,
                args.setup_key,
            )
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print("Model API key saved locally.")
        print(f"- provider: {provider_settings.name}")
        print(f"- key: {provider_settings.api_key_env}")
        print(f"- storage: {_workspace_relative(path, settings.workspace)}")
        return 0
    if args.setup_wecom is not None:
        try:
            path = _save_wecom_command(settings, args.setup_wecom)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print("Enterprise WeChat remote saved locally.")
        print(f"- storage: {_workspace_relative(path, settings.workspace)}")
        print("- next: deepmate --remote-validate --remote wecom")
        print("- start: deepmate --remote wecom")
        return 0
    if args.run_capability_maintenance:
        try:
            result = _run_capability_maintenance(settings, args.profile)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(_format_capability_maintenance_result(result, settings.workspace))
        return 0
    if args.run_evolution_maintenance:
        try:
            result = _run_evolution_maintenance(settings, args.profile)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(_format_evolution_maintenance_result(result, settings.workspace))
        return 0
    if args.list_evolution_changes:
        try:
            print(_format_evolution_changes(settings, args.profile))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.rollback_evolution_change:
        try:
            result = _rollback_evolution_change(
                settings,
                args.profile,
                args.rollback_evolution_change,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(
            "evolution rollback:"
            f"\n- change_id: {result.change_id}"
            f"\n- target: {_workspace_relative(result.target_path, settings.workspace)}"
            "\n- restored_hash: "
            f"{result.restored_hash[:12] if result.restored_hash else '(deleted)'}"
            f"\n- rollback_change_id: {result.rollback_change.change_id}"
        )
        return 0
    if args.list_mcp:
        print(format_mcp_server_list(settings.mcp_servers))
        return 0
    if args.mcp_status:
        try:
            _print_mcp_status(settings, args.profile)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.sandbox_status:
        _print_sandbox_status(settings, args)
        return 0
    if args.doctor:
        print(_doctor_text(settings, args))
        return 0
    if args.local_status:
        print(_local_status_text(settings, args))
        return 0
    if args.prepare_local_model is not None:
        result = _prepare_local_model_command(settings, args)
        print(result.message)
        return 0 if result.ok else 1
    if args.validate_browser:
        try:
            result = validate_browser_backend(settings.workspace)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(format_browser_validation_result(result))
        return 0 if result.ok() else 1
    if args.install_browser:
        try:
            result = install_browser_backend(settings.workspace)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(format_browser_install_result(result))
        return 0 if result.ok() else 1
    if args.behavior_status:
        behavior_runtime = behavior_runtime_for_session(
            data_dir=settings.deepmate_home,
            workspace=settings.workspace,
            profile=settings.profile_ref(args.profile),
            session_id="status",
            interaction_learning_enabled=args.behavior_learning,
            computer_learning_enabled=args.computer_learning,
            computer_use_enabled=args.computer_use,
        )
        print(behavior_runtime.status_text())
        return 0
    if args.list_mcp_tools:
        try:
            _print_mcp_tools(settings, args.list_mcp_tools)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.show_skill:
        try:
            _print_skill_detail(settings.workspace, args.show_skill)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.inspect_skill:
        try:
            result = inspect_skill_source(
                args.inspect_skill,
                settings.workspace,
                skill_name=args.skill_name or "",
            )
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(format_skill_inspection(result, settings.workspace))
        return 0 if not result.fatal_errors else 1
    if args.install_skill:
        try:
            result = _install_skill_command(settings, args, args.install_skill)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(format_skill_install_result(result, settings.workspace))
        print()
        print(
            format_skill_verify_result(
                _verify_skill_command(settings, args, result.skill.name),
                settings.workspace,
            )
        )
        return 0
    if args.update_skill:
        try:
            result = _update_skill_command(settings, args, args.update_skill)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(format_skill_install_result(result, settings.workspace))
        print()
        print(
            format_skill_verify_result(
                _verify_skill_command(settings, args, result.skill.name),
                settings.workspace,
            )
        )
        return 0
    if args.verify_skill:
        try:
            result = _verify_skill_command(settings, args, args.verify_skill)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(format_skill_verify_result(result, settings.workspace))
        return 0
    if args.uninstall_skill:
        try:
            result = _uninstall_skill_command(settings, args, args.uninstall_skill)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(format_skill_uninstall_result(result, settings.workspace))
        return 0
    if args.list_skill_installs:
        try:
            manifest_store = InstalledSkillManifestStore.in_data_dir(settings.data_dir)
            print(format_installed_skill_list(manifest_store, settings.workspace))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.activity_today or args.activity_date or args.activity_month:
        try:
            _print_activity(
                settings=settings,
                profile_name=args.profile,
                today=args.activity_today,
                date=args.activity_date,
                month=args.activity_month,
                lines=args.activity_lines,
            )
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.install_memory_maintenance_schedule:
        try:
            path = _install_memory_maintenance_schedule(settings)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(_format_memory_maintenance_schedule_install(path, settings.workspace))
        return 0
    if args.show_session:
        try:
            session = session_store.load(session_store.resolve_id(args.show_session))
            _print_session_detail(
                session_store=session_store,
                session=session,
                data_dir=settings.data_dir,
                trace_path=settings.trace_sink,
                activity_store=_activity_store(settings, session.profile.name),
                transcript_limit=args.session_items,
                trace_limit=args.trace_events,
                trace_kinds=args.trace_kind or (),
                export_otlp=args.export_otlp,
                otlp_endpoint=args.otlp_endpoint,
                settings=settings,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.validate_otlp:
        try:
            result = _validate_otlp_export(
                settings=settings,
                endpoint=args.otlp_endpoint,
            )
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print("OTLP validation completed.")
        print(f"- endpoint: {result.endpoint}")
        print(f"- spans: {result.spans_exported}/{result.spans_seen}")
        print(f"- status: {result.status_code}")
        if result.message:
            print(f"- message: {result.message}")
        return 0
    if args.rewind:
        try:
            return _handle_rewind_command(settings, session_store, args)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    if args.mcp is True and not settings.mcp_servers:
        _print_warning("--mcp ignored: no MCP servers configured")

    try:
        provider_settings = settings.provider(args.provider)
        _validate_provider_connection(provider_settings)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    api_key = provider_settings.api_key(settings.data_dir)
    remote_provider_settings = provider_settings
    remote_api_key = api_key
    auto_local_model = recommended_local_model()
    if not api_key and not args.interactive:
        print(
            _missing_api_key_message(settings, provider_settings),
            file=sys.stderr,
        )
        return 1

    options = _model_options(args)
    if args.validate_runtime and "max_tokens" not in options:
        options["max_tokens"] = 512
    selected_local_model = auto_local_model
    if provider_settings.name == LOCAL_PROVIDER_NAME:
        configured_local_model = local_model_by_runtime_name(
            args.model or provider_settings.primary_model()
        ) or local_model_by_id(args.model or provider_settings.primary_model())
        if configured_local_model is not None:
            selected_local_model = configured_local_model
        if "max_tokens" not in options:
            options["max_tokens"] = selected_local_model.max_tokens
        if not args.interactive:
            local_ready = _local_model_ready(settings, selected_local_model)
            if local_ready:
                api_key = api_key or LOCAL_PROVIDER_API_KEY
            else:
                print(
                    _local_model_not_ready_message(selected_local_model),
                    file=sys.stderr,
                )
                return 1
    provider = (
        ChatCompletionsProvider(
            base_url=provider_settings.base_url,
            api_key=api_key,
        )
        if api_key
        else _MissingApiKeyProvider(settings, provider_settings)
    )
    qa_model = _resolve_chat_model(args, provider_settings)
    if provider_settings.name == LOCAL_PROVIDER_NAME and not args.model:
        qa_model = selected_local_model.runtime_name
    if args.qa is not None:
        try:
            print(
                _handle_qa_cli(
                    settings.workspace,
                    args.qa,
                    provider=provider,
                    model=qa_model,
                    options=options,
                    allow_fallback=True,
                )
            )
        except (OSError, ValueError, json.JSONDecodeError, ProviderError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    if prompt and not task_requested and not args.cron_job_run:
        try:
            qa_message = maybe_create_qa_audit(
                prompt,
                workspace=settings.workspace,
                provider=provider,
                model=qa_model,
                options=options,
                allow_fallback=True,
            )
        except (OSError, ValueError, json.JSONDecodeError, ProviderError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if qa_message is not None:
            print(qa_message)
            return 0
    trace_recorder = TraceRecorder(JsonlTraceSink(settings.trace_sink))
    if args.run_memory_maintenance:
        hook_signal_store = HookSignalStore.in_data_dir(settings.data_dir)
        hook_report = load_hook_report(
            settings.workspace,
            settings.data_dir,
            _hook_load_options(settings),
        )
        hook_context = HookRuntimeContext.from_registry(
            hook_report.registry,
            signal_store=hook_signal_store,
        )
        before = _emit_cli_maintenance_hook(
            hook_context,
            HookEvent.MAINTENANCE_BEFORE_RUN,
            settings=settings,
            trace_recorder=trace_recorder,
            payload={
                "status": "before_run",
                "summary": "Daily memory maintenance requested.",
                "maintenance_kind": "memory",
                "profile": settings.profile_ref(args.profile).name,
                "local_date": args.memory_maintenance_date or "",
                "force": args.force_memory_maintenance,
            },
        )
        if before.directive != HookDirective.CONTINUE:
            print(before.reason or "maintenance blocked by hook", file=sys.stderr)
            return 1
        result = run_daily_memory_maintenance(
            provider=provider,
            settings=settings,
            fallback_model=provider_settings.primary_model(),
            session_store=session_store,
            trace_recorder=trace_recorder,
            local_date=args.memory_maintenance_date,
            force=args.force_memory_maintenance,
            provider_settings=provider_settings,
        )
        print(_format_memory_maintenance_result(result))
        _emit_cli_maintenance_hook(
            hook_context,
            HookEvent.MAINTENANCE_AFTER_RUN,
            settings=settings,
            trace_recorder=trace_recorder,
                payload={
                    "status": "completed" if result.reason != "failed" else "failed",
                    "summary": "Daily memory maintenance finished.",
                    "maintenance_kind": "memory",
                    "profile": settings.profile_ref(args.profile).name,
                    "date": result.date,
                    "reason": result.reason,
                    "pending_processed": result.pending_processed,
                    "pending_failed": result.pending_failed,
                    "profile_changed": result.profile_changed,
                    "monthly_summary_written": result.monthly_summary_written,
                },
                refs=result.summary_refs(),
            )
        if args.memory_maintenance_date is None:
            capability_result = _run_capability_maintenance(settings, args.profile)
            print()
            print(_format_capability_maintenance_result(capability_result, settings.workspace))
            _emit_cli_maintenance_hook(
                hook_context,
                HookEvent.MAINTENANCE_AFTER_RUN,
                settings=settings,
                trace_recorder=trace_recorder,
                payload={
                    "status": "completed",
                    "summary": "Daily capability maintenance finished.",
                    "maintenance_kind": "capability",
                    "profile": settings.profile_ref(args.profile).name,
                    "reason": capability_result.reason,
                    "skills_seen": capability_result.skills_seen,
                    "states_seen": capability_result.states_seen,
                    "cooled": capability_result.cooled,
                    "proposals_created": capability_result.proposals_created,
                },
                refs=(
                    f"reason={capability_result.reason}",
                    f"skills_seen={capability_result.skills_seen}",
                    f"states_seen={capability_result.states_seen}",
                    f"cooled={capability_result.cooled}",
                    f"proposals_created={capability_result.proposals_created}",
                ),
            )
            evolution_result = _run_evolution_maintenance(settings, args.profile)
            print()
            print(_format_evolution_maintenance_result(evolution_result, settings.workspace))
            _emit_cli_maintenance_hook(
                hook_context,
                HookEvent.MAINTENANCE_AFTER_RUN,
                settings=settings,
                trace_recorder=trace_recorder,
                payload={
                    "status": "completed" if evolution_result.ran else "skipped",
                    "summary": "Evolution maintenance finished.",
                    "maintenance_kind": "evolution",
                    "profile": settings.profile_ref(args.profile).name,
                    "reason": evolution_result.reason,
                    "behavior_changes": evolution_result.behavior_changes,
                    "failure_patterns_updated": evolution_result.failure_patterns_updated,
                    "generated_skill_changes": evolution_result.generated_skill_changes,
                    "capability_state_changes": evolution_result.capability_state_changes,
                    "trace_records_seen": evolution_result.trace_records_seen,
                    "sessions_seen": evolution_result.sessions_seen,
                    "activity_notes_seen": evolution_result.activity_notes_seen,
                },
                refs=(
                    f"ran={str(evolution_result.ran).lower()}",
                    f"reason={evolution_result.reason}",
                    f"behavior_changes={evolution_result.behavior_changes}",
                    f"failure_patterns_updated={evolution_result.failure_patterns_updated}",
                    f"generated_skill_changes={evolution_result.generated_skill_changes}",
                    f"capability_state_changes={evolution_result.capability_state_changes}",
                ),
            )
        return 0 if result.reason != "failed" else 1
    printed_context_warnings: set[tuple[str, str, tuple[str, ...]]] = set()
    mcp_executor: McpToolExecutor | None = None
    browser_backend: AgentBrowserBackend | None = None
    close_runtime_tools_after_post_turn = False
    try:
        if args.interactive and not args.session_id and not prompt:
            pet_session_id = _consume_pet_open_actions(pet_store, session_store)
            if pet_session_id:
                args.session_id = pet_session_id
        session = _load_or_create_session(
            session_store=session_store,
            settings=settings,
            profile_name=args.profile,
            session_id=args.session_id,
            title=args.session_title
            or ("Runtime validation" if args.validate_runtime else None),
            prompt=prompt,
            interactive=args.interactive and not args.validate_runtime,
        )
        pet_state_store = PetStateStore.in_data_dir(settings.data_dir)
        transcript = session_store.transcript_store(session)
        try:
            model = _resolve_chat_model(args, provider_settings)
            if provider_settings.name == LOCAL_PROVIDER_NAME and not args.model:
                model = selected_local_model.runtime_name
            if provider_settings.name == LOCAL_PROVIDER_NAME:
                preset = local_model_by_runtime_name(model) or local_model_by_id(model)
                if preset is not None:
                    model = preset.runtime_name
            provider_key_available = bool(api_key.strip())
            allow_missing_context_window = args.interactive and not provider_key_available
            model_context_tokens = _provider_model_context_tokens(
                settings,
                provider_settings,
                model,
                allow_missing_context_window=allow_missing_context_window,
            )
            model_capabilities = _provider_model_capabilities(
                settings,
                provider_settings,
                model,
            )
            maintenance_fallback_model = (
                model
                if provider_settings.name == LOCAL_PROVIDER_NAME
                else provider_settings.primary_model()
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        checkpoint_controller = _checkpoint_controller(settings, session)
        checkpoint_write_router = SessionCheckpointWriteRouter(checkpoint_controller)
        if task_store is not None:
            task_store.set_write_checkpoint(
                checkpoint_write_router.capture_workspace_write
            )
        activation = start_runtime_activation(
            session_id=session.session_id,
            workspace=session.workspace,
            profile=session.profile,
            context_snapshot=_profile_context_snapshot(
                settings,
                session.profile,
                model,
                provider_settings=provider_settings,
                task_context=task_context,
                allow_missing_context_window=allow_missing_context_window,
            ),
        )
        behavior_runtime = behavior_runtime_for_session(
            data_dir=settings.deepmate_home,
            workspace=settings.workspace,
            profile=session.profile,
            session_id=session.session_id,
            interaction_learning_enabled=args.behavior_learning,
            computer_learning_enabled=args.computer_learning,
            computer_use_enabled=args.computer_use,
        )
        if task_store is not None and task_stage is not None:
            task_store.save_state(
                persisted_task_stage(task_stage, task_cursor_stage),
                session_id=session.session_id,
                execute_status=(
                    "active" if task_stage == TaskStage.EXECUTE else ""
                ),
            )
        runtime = start_session_runtime(
            activation,
            conversation=runtime_conversation_from_store(
                session_store,
                session,
                transcript,
                warning_sink=_print_warning,
                turn_checkpoint_store=checkpoint_controller.turn_store,
            ),
            behavior_runtime=behavior_runtime,
        )
        hook_report = load_hook_report(
            settings.workspace,
            settings.data_dir,
            _hook_load_options(settings),
        )
        hook_signal_store = HookSignalStore.in_data_dir(settings.data_dir)
        hook_context = HookRuntimeContext.from_registry(
            hook_report.registry,
            signal_store=hook_signal_store,
        )
        _record_hook_load_trace(trace_recorder, hook_report, session.session_id)
        tool_output_store = ToolOutputStore.in_data_dir(
            settings.data_dir,
            session.profile.name,
            session.session_id,
        )
        interactive_defaults = args.interactive and not args.validate_runtime
        tool_access_policy = ToolAccessPolicy(
            ToolAccessMode.WORKSPACE_WRITE
            if args.workspace_write or interactive_defaults
            else ToolAccessMode.READ_ONLY,
            shell_enabled=args.shell,
        )
        approval_cache = SessionApprovalCache()
        native_tools = None
        tool_schemas = ()
        expose_read_tools = (
            args.read_only_tools
            or args.workspace_write
            or interactive_defaults
        )
        register_write_tools = args.workspace_write or interactive_defaults
        expose_network_tools = args.allow_network or interactive_defaults
        register_shell_tools = args.shell or interactive_defaults
        expose_computer_tools = args.computer_use
        def native_tool_factory(
            cache: SessionApprovalCache | None = approval_cache,
            session_id: str = session.session_id,
        ) -> NativeToolRegistry | None:
            built = _build_cli_native_tools(
                settings=settings,
                expose_read_tools=expose_read_tools,
                register_write_tools=register_write_tools,
                expose_network_tools=expose_network_tools,
                register_shell_tools=register_shell_tools,
                behavior_runtime=behavior_runtime,
                expose_computer_tools=expose_computer_tools,
                shell_enabled=args.shell,
                network_enabled=args.allow_network,
                env_change_enabled=args.allow_env_change,
                sandbox_mode=SandboxMode(args.sandbox),
                approval_cache=cache,
                checkpoint_write_router=checkpoint_write_router,
                hook_context=hook_context,
            )
            capability_store = CapabilityStateStore.in_data_dir(
                settings.data_dir,
                session.profile,
            )
            built = _attach_skill_installer_tools(
                native_tools=built,
                prompt=prompt,
                workspace=settings.workspace,
                data_dir=settings.data_dir,
                capability_state_store=capability_store,
                shell_enabled=args.shell,
                network_enabled=expose_network_tools,
                env_change_enabled=args.allow_env_change,
                sandbox_mode=SandboxMode(args.sandbox),
                approval_cache=cache,
            )
            retrieval_registry = _attach_tool_output_tools(
                native_tools=None,
                store=tool_output_store,
                enabled=True,
                exposed_by_default=True,
            )
            return _attach_browser_tools(
                native_tools=built,
                backend=AgentBrowserBackend(settings.workspace, session_name=session_id),
                preload=args.browser,
                approval_cache=cache,
                extra_schema_loader=lambda: tuple(
                    retrieval_registry.schemas()
                    if retrieval_registry is not None
                    else ()
                ),
            )

        native_tools = native_tool_factory(approval_cache)
        tool_schemas = native_tools.schemas() if native_tools is not None else ()
        capability_state_store = CapabilityStateStore.in_data_dir(
            settings.data_dir,
            session.profile,
        )
        mcp_catalog: McpToolCatalog | None = None
        mcp_tools = ()
        mcp_state_store = McpUsageStateStore.in_data_dir(
            settings.data_dir,
            session.profile,
        )
        if _mcp_enabled(args, settings):
            mcp_catalog = _discover_mcp_catalog(settings, mcp_state_store)
            mcp_tools = mcp_catalog.read_only_tools()
            if mcp_tools:
                mcp_executor = McpToolExecutor(
                    settings.mcp_servers,
                    (
                        mcp_catalog.all_tools()
                        if args.mcp_write
                        else mcp_tools
                    ),
                    settings.workspace,
                    usage_state_store=mcp_state_store,
                    allow_write_tools=args.mcp_write,
                    hook_context=hook_context,
                )
            elif settings.mcp_servers:
                _print_warning("MCP enabled, but no read-only MCP tools were discovered")

        skill_catalog, skill_warnings = _skill_catalog(
            settings.workspace,
            settings.data_dir,
            capability_state_store=capability_state_store,
        )
        selected_skills = select_skill_documents(
            skill_catalog,
            args.skill,
            settings.workspace,
            command_name="--skill",
        )
        selected_skill_states = tuple(
            capability_state_store.record_skill_selected(skill.name)
            for skill in selected_skills
            if _path_is_within(skill.path, settings.workspace)
        )
        for state in selected_skill_states:
            _record_capability_state_trace(
                trace_recorder,
                kind="capability_selected",
                summary=f"Skill selected: {state.name}.",
                state=state,
                session_id=session.session_id,
            )
        native_tools = _attach_skill_loader_tools(
            native_tools=native_tools,
            skill_catalog=skill_catalog,
            workspace=settings.workspace,
            capability_state_store=capability_state_store,
            data_dir=settings.data_dir,
            dynamic=True,
        )
        native_tools = _attach_mcp_loader_tools(
            native_tools=native_tools,
            mcp_catalog=mcp_catalog,
        )
        expose_tool_output_retrieval = tool_output_store.has_records()
        native_tools = _attach_tool_output_tools(
            native_tools=native_tools,
            store=tool_output_store,
            enabled=bool(native_tools is not None or mcp_tools or args.subagents),
            exposed_by_default=expose_tool_output_retrieval,
        )
        tool_schemas = _default_tool_schemas_for_model(native_tools, model)
        capability_surface, capability_warnings = _capability_surface(
            skill_catalog,
            tool_schemas,
            mcp_tools,
            capability_state_store=capability_state_store,
            mcp_catalog=mcp_catalog,
            model_context_tokens=model_context_tokens,
        )
        default_mcp_schema_tools = (
            mcp_catalog.default_schema_tools(model_context_tokens)
            if mcp_catalog is not None and local_model_by_runtime_name(model) is None
            else ()
        )
        all_tool_schemas = (
            *tool_schemas,
            *(tool.schema() for tool in default_mcp_schema_tools),
        )
        if local_model_by_runtime_name(model) is not None and prompt:
            all_tool_schemas = _schemas_with_local_prompt_extras(
                all_tool_schemas,
                native_tools,
                prompt,
            )
        conversation_budget_policy = _conversation_budget_policy(
            settings,
            model,
            provider_settings=provider_settings,
            allow_missing_context_window=allow_missing_context_window,
        )
        loop_guard_policy = _loop_guard_policy(settings)
        tool_repair_policy = _tool_repair_policy(settings)
        tool_output_compactor = ToolOutputCompactor(
            store=tool_output_store,
            policy=conversation_budget_policy,
            enabled=settings.tool_output.compaction_enabled,
            lossless_normalization=settings.tool_output.lossless_normalization,
            compaction_policy=_tool_output_compaction_policy(settings),
        )
        provider_retry_policy = ProviderRetryPolicy(
            max_attempts=settings.provider_retry.max_attempts,
            initial_delay_seconds=settings.provider_retry.initial_delay_seconds,
        )
        subagent_executor = None
        if args.subagents:
            subagent_result_store = SubagentResultStore.in_data_dir(
                settings.data_dir,
                session.session_id,
            )
            subagent_executor = _subagent_executor(
                provider=provider,
                settings=settings,
                profile=session.profile,
                model=model,
                capability_surface=capability_surface,
                native_tools=native_tools,
                mcp_executor=mcp_executor,
                tool_schemas=all_tool_schemas,
                selected_skills=selected_skills,
                activation=activation,
                provider_retry_policy=provider_retry_policy,
                options=options,
                trace_recorder=trace_recorder,
                tool_access_mode=tool_access_policy.mode,
                provider_settings=provider_settings,
                hook_context=hook_context,
                result_store=subagent_result_store,
            )
            all_tool_schemas = (*all_tool_schemas, *subagent_executor.schemas())
        for warning in skill_warnings:
            _print_warning(warning.message)
        for warning in capability_warnings:
            _print_warning(warning.message)

        def tui_native_tool_factory(
            cache: SessionApprovalCache | None = approval_cache,
            session_id: str = session.session_id,
        ) -> NativeToolRegistry | None:
            built = native_tool_factory(cache, session_id)
            current_catalog, _warnings = _skill_catalog(
                settings.workspace,
                settings.data_dir,
                capability_state_store=capability_state_store,
            )
            built = _attach_skill_loader_tools(
                native_tools=built,
                skill_catalog=current_catalog,
                workspace=settings.workspace,
                capability_state_store=capability_state_store,
                data_dir=settings.data_dir,
                dynamic=True,
            )
            built = _attach_mcp_loader_tools(
                native_tools=built,
                mcp_catalog=mcp_catalog,
            )
            return _attach_tool_output_tools(
                native_tools=built,
                store=tool_output_store,
                enabled=bool(built is not None or mcp_tools or args.subagents),
                exposed_by_default=tool_output_store.has_records(),
            )

        def post_turn_maintenance(
            user_prompt,
            current_session,
            current_transcript,
            current_runtime,
        ) -> SessionRuntime:
            return run_session_maintenance(
                provider=provider,
                settings=settings,
                fallback_model=maintenance_fallback_model,
                prompt=user_prompt,
                session_store=session_store,
                session=current_session,
                transcript=current_transcript,
                runtime=current_runtime,
                conversation_budget_policy=conversation_budget_policy,
                trace_recorder=trace_recorder,
                warning_sink=_print_warning,
                status_sink=_print_status,
                hook_context=hook_context,
                provider_settings=provider_settings,
            )

        def task_turn_maintenance(
            user_prompt,
            final_text,
            current_session,
            current_runtime,
        ) -> SessionRuntime | tuple[SessionRuntime, ExecuteLoopUpdate]:
            if task_controller is None or task_controller.active_stage is None:
                return current_runtime
            current_stage = task_controller.active_stage
            update = _run_task_post_turn(
                provider=provider,
                settings=settings,
                fallback_model=maintenance_fallback_model,
                task_store=task_controller.store,
                stage=current_stage,
                prompt=user_prompt,
                final_text=final_text,
                result=current_runtime.last_user_turn_result,
                trace_recorder=trace_recorder,
                session_id=current_session.session_id,
                provider_settings=provider_settings,
                turn_succeeded=True,
                achievement_required=current_stage == TaskStage.CHECKPOINT,
            )
            task_controller.finish_turn(current_stage)
            if update is not None:
                return current_runtime, update
            return current_runtime

        def context_snapshot_for(profile_ref, model_override: str | None = None):
            return _profile_context_snapshot(
                settings,
                profile_ref,
                model_override or model,
                provider_settings=provider_settings,
                task_context=(
                    task_controller.context()
                    if task_controller is not None
                    else task_context
                ),
            )

        def local_context_prepare(state, preset) -> bool:
            policy = _conversation_budget_policy(
                settings,
                preset.runtime_name,
                provider_settings=settings.provider(LOCAL_PROVIDER_NAME),
            )
            local_snapshot = context_snapshot_for(
                state.session.profile,
                model_override=preset.runtime_name,
            )
            request_for_budget = build_model_request(
                workspace=state.workspace,
                profile=state.session.profile,
                messages=(),
                model=preset.runtime_name,
                capability_surface=state.capability_surface,
                tool_schemas=state.tool_schemas,
                conversation=state.runtime.conversation,
                selected_skill_documents=state.selected_skill_documents,
                context_snapshot=local_snapshot,
                options=dict(state.options),
                capabilities=local_model_capabilities(preset.runtime_name),
            ).request
            request_for_budget = sanitize_model_request(request_for_budget)
            request_report = build_request_budget_report(
                request_for_budget,
                policy=policy,
            )
            if request_report.pressure_ratio < _local_context_prepare_ratio(preset):
                state.runtime = state.runtime.with_refreshed_context(local_snapshot)
                return False
            provider_for_summary = state.remote_provider or provider
            if isinstance(provider_for_summary, _MissingApiKeyProvider):
                state.runtime = state.runtime.with_refreshed_context(local_snapshot)
                return False
            try:
                state.runtime = force_session_summary_checkpoint(
                    provider=provider_for_summary,
                    settings=settings,
                    fallback_model=remote_provider_settings.primary_model(),
                    session_store=state.session_store,
                    session=state.session,
                    transcript=state.transcript,
                    runtime=state.runtime,
                    conversation_budget_policy=policy,
                    trace_recorder=state.trace_recorder,
                    warning_sink=_print_warning,
                    status_sink=_print_status,
                    hook_context=state.hook_context,
                    reason="before_local_model_switch",
                    provider_settings=remote_provider_settings,
                )
            except Exception:
                state.runtime = state.runtime.with_refreshed_context(local_snapshot)
                return False
            state.runtime = state.runtime.with_conversation(
                runtime_conversation_from_store(
                    state.session_store,
                    state.session,
                    state.transcript,
                    warning_sink=_print_warning,
                    turn_checkpoint_store=(
                        state.checkpoint_controller.turn_store
                        if state.checkpoint_controller is not None
                        else None
                    ),
                )
            )
            state.runtime = state.runtime.with_refreshed_context(local_snapshot)
            return True

        def session_end_maintenance(
            current_session,
            current_transcript,
            current_runtime,
            reason,
        ) -> None:
            write_session_end_activity(
                settings=settings,
                session_store=session_store,
                session=current_session,
                transcript=current_transcript,
                runtime=current_runtime,
                trace_recorder=trace_recorder,
                event="interactive_session_end",
                status="paused",
                summary=f"Interactive session ended: {reason}.",
                warning_sink=_print_warning,
            )

        if args.remote == "wecom":
            return run_wecom_remote_channel(
                WeComRunDependencies(
                    settings=settings,
                    provider=provider,
                    model=model,
                    provider_name=provider_settings.name,
                    session_store=session_store,
                    trace_recorder=trace_recorder,
                    capability_surface=capability_surface,
                    native_tools=native_tools,
                    mcp_tools=mcp_executor,
                    subagents=subagent_executor,
                    tool_access_policy=tool_access_policy,
                    tool_schemas=all_tool_schemas,
                    selected_skill_documents=selected_skills,
                    conversation_budget_policy=conversation_budget_policy,
                    provider_retry_policy=provider_retry_policy,
                    options=options,
                    model_capabilities=model_capabilities,
                    max_steps=args.max_steps,
                    loop_guard_policy=loop_guard_policy,
                    warning_sink=lambda warning: _print_context_warning(
                        warning,
                        printed_context_warnings,
                    ),
                    status_sink=_print_status,
                    tool_output_compactor=tool_output_compactor,
                    tool_repair_policy=tool_repair_policy,
                    hook_context=hook_context,
                    approval_cache=approval_cache,
                    behavior_runtime_factory=lambda remote_session: behavior_runtime_for_session(
                        data_dir=settings.deepmate_home,
                        workspace=remote_session.workspace,
                        profile=remote_session.profile,
                        session_id=remote_session.session_id,
                        interaction_learning_enabled=args.behavior_learning,
                        computer_learning_enabled=args.computer_learning,
                        computer_use_enabled=False,
                    ),
                )
            )

        if args.interactive and not args.interactive_legacy:
            tui_exit_code = run_tui_mode(
                provider=provider,
                provider_name=provider_settings.name,
                provider_api_key_env=provider_settings.api_key_env,
                provider_api_key_available=bool(api_key.strip()),
                model=model,
                default_model=provider_settings.primary_model(),
                upgrade_model=provider_settings.upgrade_model,
                workspace=settings.workspace,
                profile=session.profile,
                session_store=session_store,
                session=session,
                transcript=transcript,
                runtime=runtime,
                capability_surface=capability_surface,
                native_tools=native_tools,
                native_tool_factory=tui_native_tool_factory,
                mcp_tools=mcp_executor,
                subagents=subagent_executor,
                tool_access_policy=tool_access_policy,
                tool_schemas=all_tool_schemas,
                selected_skill_documents=selected_skills,
                mcp_servers=settings.mcp_servers,
                conversation_budget_policy=conversation_budget_policy,
                provider_retry_policy=provider_retry_policy,
                options=options,
                model_capabilities=model_capabilities,
                max_steps=args.max_steps,
                loop_guard_policy=loop_guard_policy,
                trace_recorder=trace_recorder,
                warning_sink=lambda warning: _print_context_warning(
                    warning,
                    printed_context_warnings,
                ),
                status_sink=_print_status,
                tool_output_compactor=tool_output_compactor,
                tool_repair_policy=tool_repair_policy,
                hook_context=hook_context,
                hook_load_options=_hook_load_options(settings),
                data_dir=settings.data_dir,
                capability_state_store=capability_state_store,
                approval_cache=approval_cache,
                maintenance_handler=post_turn_maintenance,
                task_controller=task_controller,
                task_maintenance_handler=task_turn_maintenance,
                session_end_handler=session_end_maintenance,
                context_snapshot_factory=context_snapshot_for,
                checkpoint_controller=checkpoint_controller,
                checkpoint_controller_factory=lambda session_ref: _checkpoint_controller(
                    settings,
                    session_ref,
                ),
                checkpoint_write_router=checkpoint_write_router,
                pet_state_store=pet_state_store,
                refresh_skill_surface_callback=lambda state: _refresh_tui_skill_surface(
                    state,
                    model_context_tokens=_state_model_context_tokens(settings, state),
                    mcp_catalog=mcp_catalog,
                    mcp_tools=mcp_tools,
                ),
                local_context_prepare_callback=local_context_prepare,
                behavior_runtime=behavior_runtime,
                initial_prompts=(prompt,) if prompt else (),
                show_reasoning=args.show_reasoning,
                remote_provider=(
                    ChatCompletionsProvider(
                        base_url=remote_provider_settings.base_url,
                        api_key=remote_api_key,
                    )
                    if remote_api_key
                    else _MissingApiKeyProvider(settings, remote_provider_settings)
                ),
                remote_provider_name=remote_provider_settings.name,
                remote_model=remote_provider_settings.primary_model(),
                remote_default_model=remote_provider_settings.primary_model(),
                remote_upgrade_model=remote_provider_settings.upgrade_model,
                local_provider_base_url=settings.provider(LOCAL_PROVIDER_NAME).base_url,
                local_provider_api_key=(
                    settings.provider(LOCAL_PROVIDER_NAME).api_key(settings.data_dir)
                    or LOCAL_PROVIDER_API_KEY
                ),
            )
            switch_target = consume_workspace_switch_request()
            if switch_target is not None:
                return _restart_with_workspace(
                    switch_target.workspace,
                    argv,
                    session_id=switch_target.session_id,
                )
            return tui_exit_code

        if args.interactive:
            return run_interactive_mode(
                provider=provider,
                model=model,
                workspace=settings.workspace,
                profile=session.profile,
                session_store=session_store,
                session=session,
                transcript=transcript,
                runtime=runtime,
                capability_surface=capability_surface,
                native_tools=native_tools,
                mcp_tools=mcp_executor,
                subagents=subagent_executor,
                tool_access_policy=tool_access_policy,
                tool_schemas=all_tool_schemas,
                selected_skill_documents=selected_skills,
                mcp_servers=settings.mcp_servers,
                conversation_budget_policy=conversation_budget_policy,
                provider_retry_policy=provider_retry_policy,
                options=options,
                max_steps=args.max_steps,
                loop_guard_policy=loop_guard_policy,
                trace_recorder=trace_recorder,
                warning_sink=lambda warning: _print_context_warning(
                    warning,
                    printed_context_warnings,
                ),
                model_capabilities=model_capabilities,
                status_sink=_print_status,
                tool_output_compactor=tool_output_compactor,
                tool_repair_policy=tool_repair_policy,
                hook_context=hook_context,
                hook_load_options=_hook_load_options(settings),
                data_dir=settings.data_dir,
                capability_state_store=capability_state_store,
                maintenance_handler=post_turn_maintenance,
                task_controller=task_controller,
                task_maintenance_handler=task_turn_maintenance,
                session_end_handler=session_end_maintenance,
                context_snapshot_factory=context_snapshot_for,
                checkpoint_controller=checkpoint_controller,
                checkpoint_controller_factory=lambda session_ref: _checkpoint_controller(
                    settings,
                    session_ref,
                ),
                checkpoint_write_router=checkpoint_write_router,
                initial_prompts=(prompt,) if prompt else (),
                show_reasoning=args.show_reasoning,
            )

        turn_scope = checkpoint_controller.start_turn(
            _load_latest_summary(session_store, session)
        )
        try:
            _close_remote_routes_for_local_turn(
                session_store=session_store,
                session=session,
                data_dir=settings.data_dir,
                trace_recorder=trace_recorder,
                runtime=runtime,
            )
            _safe_save_pet_event(
                pet_state_store,
                event_for_turn_started(
                    workspace=settings.workspace,
                    session_id=session.session_id,
                    prompt=prompt,
                    title=session.title,
                ),
            )
            turn = runtime.run_user_turn(
                provider=provider,
                messages=(Message(role=MessageRole.USER, content=prompt),),
                model=model,
                capability_surface=capability_surface,
                native_tools=native_tools,
                mcp_tools=mcp_executor,
                subagents=_bind_subagent_executor(
                    subagent_executor,
                    capability_surface=capability_surface,
                    native_tools=native_tools,
                    mcp_tools=mcp_executor,
                    tool_schemas=all_tool_schemas,
                    selected_skill_documents=selected_skills,
                    activation=runtime.activation,
                    tool_access_policy=tool_access_policy,
                    result_store=subagent_result_store if args.subagents else None,
                ),
                tool_access_policy=tool_access_policy,
                tool_schemas=all_tool_schemas,
                selected_skill_documents=selected_skills,
                conversation_budget_policy=conversation_budget_policy,
                provider_retry_policy=provider_retry_policy,
                options=options,
                max_steps=args.max_steps,
                loop_guard_policy=loop_guard_policy,
                trace_recorder=trace_recorder,
                warning_sink=lambda warning: _print_context_warning(
                    warning,
                    printed_context_warnings,
                ),
                history_sink=turn_scope.history_sink(transcript),
                status_sink=_print_status,
                tool_output_compactor=tool_output_compactor,
                tool_repair_policy=tool_repair_policy,
                hook_context=hook_context,
            )
        except KeyboardInterrupt:
            turn_scope.mark_interrupted()
            _safe_save_pet_event(
                pet_state_store,
                event_for_turn_finished(
                    workspace=settings.workspace,
                    session_id=session.session_id,
                    title=session.title,
                    summary="Turn interrupted.",
                    failed=True,
                ),
            )
            raise
        except Exception as exc:
            turn_scope.mark_failed(type(exc).__name__)
            _safe_save_pet_event(
                pet_state_store,
                event_for_turn_finished(
                    workspace=settings.workspace,
                    session_id=session.session_id,
                    title=session.title,
                    summary=f"{type(exc).__name__}: {exc}",
                    failed=True,
                ),
            )
            raise
        else:
            turn_scope.mark_result(turn.result)
        finally:
            turn_scope.close()
        runtime = turn.runtime
        result = turn.result
        if args.validate_runtime:
            final_text = _response_text(result.final_step().response)
            if not result.has_errors() and not result.reached_max_steps:
                runtime = run_session_maintenance(
                    provider=provider,
                    settings=settings,
                    fallback_model=maintenance_fallback_model,
                    prompt=prompt,
                    session_store=session_store,
                    session=session,
                    transcript=transcript,
                    runtime=runtime,
                    conversation_budget_policy=conversation_budget_policy,
                    trace_recorder=trace_recorder,
                    warning_sink=_print_warning,
                    status_sink=_print_status,
                    hook_context=hook_context,
                    provider_settings=provider_settings,
                )
                turn_scope.attach_summary(_load_latest_summary(session_store, session))
            write_session_end_activity(
                settings=settings,
                session_store=session_store,
                session=session,
                transcript=transcript,
                runtime=runtime,
                trace_recorder=trace_recorder,
                event="runtime_validation_end",
                status=_runtime_validation_session_status(result),
                summary=final_text or "Runtime validation finished.",
                warning_sink=_print_warning,
            )
            return _finish_runtime_validation(
                result=result,
                session_store=session_store,
                session=session,
                transcript=transcript,
                trace_path=settings.trace_sink,
                trace_recorder=trace_recorder,
                model=model,
                require_native_tool=args.read_only_tools or args.workspace_write,
                require_workspace_write=args.workspace_write,
                require_shell=args.shell,
                require_subagent=args.subagents,
                require_activity_note=True,
                mcp_tool_count=len(mcp_tools),
            )
        close_runtime_tools_after_post_turn = True
    except KeyboardInterrupt:
        print()
        return 130
    except AuthError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            f"hint: check {provider_settings.api_key_env} for provider "
            f"{provider_settings.name}",
            file=sys.stderr,
        )
        return 2
    except RateLimitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("hint: retry after the provider rate limit resets", file=sys.stderr)
        return 75
    except (NetworkError, ServerError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("hint: this may be retryable", file=sys.stderr)
        return 75
    except ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if not close_runtime_tools_after_post_turn and browser_backend is not None:
            browser_backend.close()
        if not close_runtime_tools_after_post_turn and mcp_executor is not None:
            mcp_executor.close()

    for error in result.errors():
        if _is_loop_guard_error(result, error):
            continue
        print(f"error: {error.message}", file=sys.stderr)
    if result.loop_guard_stop is not None:
        print(f"stopped: {result.loop_guard_stop.message}", file=sys.stderr)
        print("hint: run again with `continue` or `继续` to resume.", file=sys.stderr)
    elif result.reached_max_steps:
        print(
            f"error: reached --max-steps={args.max_steps} before final answer",
            file=sys.stderr,
        )

    final_step = result.final_step()
    final_text = _response_text(final_step.response)
    delivery_review = _review_delivery(
        user_request=prompt,
        final_response=final_text,
        result=result,
        trace_recorder=trace_recorder,
        session_id=session.session_id,
        activation_refs=runtime.activation.trace_refs(),
    )
    _safe_save_pet_event(
        pet_state_store,
        event_for_turn_finished(
            workspace=settings.workspace,
            session_id=session.session_id,
            title=session.title,
            summary=_pet_turn_summary(result, final_text, delivery_review.summary),
            failed=(
                result.has_errors()
                or result.reached_max_steps
                or delivery_review.status == DeliveryReviewStatus.BLOCKED
            ),
        ),
    )
    _print_delivery_review_warning(delivery_review)
    if args.show_reasoning and final_step.response.reasoning:
        print(final_step.response.reasoning)
        if final_step.response.content:
            print()
    if final_step.response.content:
        print(final_step.response.content)
    elif final_step.response.reasoning and not args.show_reasoning:
        print(final_step.response.reasoning)
    sys.stdout.flush()
    execute_update: ExecuteLoopUpdate | None = None
    if (
        task_store is not None
        and task_stage is not None
        and not result.has_errors()
    ):
        checkpoint_write_router.set_thread_turn_id(turn_scope.turn_id)
        try:
            execute_update = _run_task_post_turn(
                provider=provider,
                settings=settings,
                fallback_model=maintenance_fallback_model,
                task_store=task_store,
                stage=task_stage,
                prompt=prompt,
                final_text=final_text,
                result=result,
                trace_recorder=trace_recorder,
                session_id=session.session_id,
                provider_settings=provider_settings,
                turn_succeeded=not result.reached_max_steps,
                achievement_required=task_stage == TaskStage.CHECKPOINT,
            )
        finally:
            checkpoint_write_router.clear_thread_turn_id()
        if execute_update is not None and not execute_update.should_continue():
            print()
            print(format_execute_outcome(execute_update.evaluation))
    while (
        task_store is not None
        and task_stage == TaskStage.EXECUTE
        and execute_update is not None
        and execute_update.should_continue()
        and not result.has_errors()
        and not result.reached_max_steps
    ):
        prompt = execute_update.continuation
        runtime = runtime.request_context_refresh_before_next_turn(
            "task_execute_continuation"
        )
        print()
        print(format_execute_outcome(execute_update.evaluation))
        print()
        turn_scope = checkpoint_controller.start_turn(
            _load_latest_summary(session_store, session)
        )
        try:
            trace_recorder.record(
                TraceEvent(
                    kind="task_execute_continuation_started",
                    summary="Task execute continuation turn started.",
                    refs=(
                        f"session_id={session.session_id}",
                        f"turns={execute_update.turns}",
                    ),
                )
            )
            _safe_save_pet_event(
                pet_state_store,
                event_for_turn_started(
                    workspace=settings.workspace,
                    session_id=session.session_id,
                    prompt=prompt,
                    title=session.title,
                ),
            )
            turn = runtime.run_user_turn(
                provider=provider,
                messages=(Message(role=MessageRole.USER, content=prompt),),
                model=model,
                capability_surface=capability_surface,
                native_tools=native_tools,
                mcp_tools=mcp_executor,
                subagents=_bind_subagent_executor(
                    subagent_executor,
                    capability_surface=capability_surface,
                    native_tools=native_tools,
                    mcp_tools=mcp_executor,
                    tool_schemas=all_tool_schemas,
                    selected_skill_documents=selected_skills,
                    activation=runtime.activation,
                    tool_access_policy=tool_access_policy,
                    result_store=subagent_result_store if args.subagents else None,
                ),
                tool_access_policy=tool_access_policy,
                tool_schemas=all_tool_schemas,
                selected_skill_documents=selected_skills,
                conversation_budget_policy=conversation_budget_policy,
                provider_retry_policy=provider_retry_policy,
                options=options,
                max_steps=args.max_steps,
                loop_guard_policy=loop_guard_policy,
                trace_recorder=trace_recorder,
                warning_sink=lambda warning: _print_context_warning(
                    warning,
                    printed_context_warnings,
                ),
                history_sink=turn_scope.history_sink(transcript),
                status_sink=_print_status,
                tool_output_compactor=tool_output_compactor,
                tool_repair_policy=tool_repair_policy,
                hook_context=hook_context,
            )
        except KeyboardInterrupt:
            turn_scope.mark_interrupted()
            _safe_save_pet_event(
                pet_state_store,
                event_for_turn_finished(
                    workspace=settings.workspace,
                    session_id=session.session_id,
                    title=session.title,
                    summary="Turn interrupted.",
                    failed=True,
                ),
            )
            raise
        except Exception as exc:
            turn_scope.mark_failed(type(exc).__name__)
            _safe_save_pet_event(
                pet_state_store,
                event_for_turn_finished(
                    workspace=settings.workspace,
                    session_id=session.session_id,
                    title=session.title,
                    summary=f"{type(exc).__name__}: {exc}",
                    failed=True,
                ),
            )
            raise
        else:
            turn_scope.mark_result(turn.result)
        finally:
            turn_scope.close()
        runtime = turn.runtime
        result = turn.result
        for error in result.errors():
            if _is_loop_guard_error(result, error):
                continue
            print(f"error: {error.message}", file=sys.stderr)
        if result.loop_guard_stop is not None:
            print(f"stopped: {result.loop_guard_stop.message}", file=sys.stderr)
            print("hint: run again with `continue` or `继续` to resume.", file=sys.stderr)
        elif result.reached_max_steps:
            print(
                f"error: reached --max-steps={args.max_steps} before final answer",
                file=sys.stderr,
            )
        final_step = result.final_step()
        final_text = _response_text(final_step.response)
        delivery_review = _review_delivery(
            user_request=prompt,
            final_response=final_text,
            result=result,
            trace_recorder=trace_recorder,
            session_id=session.session_id,
            activation_refs=runtime.activation.trace_refs(),
        )
        _safe_save_pet_event(
            pet_state_store,
            event_for_turn_finished(
                workspace=settings.workspace,
                session_id=session.session_id,
                title=session.title,
                summary=_pet_turn_summary(result, final_text, delivery_review.summary),
                failed=(
                    result.has_errors()
                    or result.reached_max_steps
                    or delivery_review.status == DeliveryReviewStatus.BLOCKED
                ),
            ),
        )
        _print_delivery_review_warning(delivery_review)
        if args.show_reasoning and final_step.response.reasoning:
            print(final_step.response.reasoning)
            if final_step.response.content:
                print()
        if final_step.response.content:
            print(final_step.response.content)
        elif final_step.response.reasoning and not args.show_reasoning:
            print(final_step.response.reasoning)
        sys.stdout.flush()
        if result.has_errors():
            break
        checkpoint_write_router.set_thread_turn_id(turn_scope.turn_id)
        try:
            execute_update = _run_task_post_turn(
                provider=provider,
                settings=settings,
                fallback_model=maintenance_fallback_model,
                task_store=task_store,
                stage=TaskStage.EXECUTE,
                prompt=prompt,
                final_text=final_text,
                result=result,
                trace_recorder=trace_recorder,
                session_id=session.session_id,
                provider_settings=provider_settings,
                turn_succeeded=not result.reached_max_steps,
            )
        finally:
            checkpoint_write_router.clear_thread_turn_id()
        if execute_update is not None and not execute_update.should_continue():
            print()
            print(format_execute_outcome(execute_update.evaluation))
    if not result.has_errors() and not result.reached_max_steps:
        runtime = run_session_maintenance(
            provider=provider,
            settings=settings,
            fallback_model=maintenance_fallback_model,
            prompt=prompt,
            session_store=session_store,
            session=session,
            transcript=transcript,
            runtime=runtime,
            conversation_budget_policy=conversation_budget_policy,
            trace_recorder=trace_recorder,
            warning_sink=_print_warning,
            status_sink=_print_status,
            hook_context=hook_context,
            provider_settings=provider_settings,
        )
        turn_scope.attach_summary(_load_latest_summary(session_store, session))
    write_session_end_activity(
        settings=settings,
        session_store=session_store,
        session=session,
        transcript=transcript,
        runtime=runtime,
        trace_recorder=trace_recorder,
        event="session_end",
        status=_session_end_status(result, delivery_review),
        summary=final_text or "Session finished. See transcript for details.",
        warning_sink=_print_warning,
    )
    session_store.touch(session.session_id)
    if browser_backend is not None:
        browser_backend.close()
    if mcp_executor is not None:
        mcp_executor.close()
    if (
        result.has_errors()
        or result.reached_max_steps
        or delivery_review.status == DeliveryReviewStatus.BLOCKED
    ):
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepmate",
        description=(
            "Deepmate local agent. Run `deepmate` to open the TUI, or "
            "`deepmate --cli 'prompt'` for a one-shot CLI turn."
        ),
        epilog=(
            "Typical use: deepmate, deepmate --cli --workspace-write "
            "'edit README', or deepmate --show-session <id>."
        ),
    )
    parser.add_argument("prompt", nargs="*", help="User prompt for one Deepmate turn.")
    parser.add_argument(
        "--cli",
        action="store_true",
        help=(
            "Run a one-shot command-line turn instead of opening the TUI. "
            "For compatibility, providing a prompt without --cli also runs one turn."
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start the Textual TUI for a durable session. This is the default with no prompt.",
    )
    parser.add_argument(
        "--interactive-legacy",
        action="store_true",
        help="Start the legacy line-based interactive REPL.",
    )
    parser.add_argument(
        "--validate-runtime",
        action="store_true",
        help="Run a real provider-backed runtime smoke check and exit.",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help=(
            "Check the base install and optional heavy feature readiness without "
            "calling a model or installing dependencies."
        ),
    )
    parser.add_argument(
        "--cron",
        nargs="*",
        metavar="VALUE",
        help=(
            "Manage workspace cron jobs: add/list/status/show/approve/pause/"
            "resume/remove/run/tick."
        ),
    )
    parser.add_argument(
        "--cron-runner",
        action="store_true",
        help="Run due approved workspace cron jobs once and exit.",
    )
    parser.add_argument(
        "--cron-watch",
        action="store_true",
        help="With --cron-runner, keep polling due cron jobs until interrupted.",
    )
    parser.add_argument(
        "--cron-poll-seconds",
        type=int,
        default=60,
        help="Polling interval for --cron-runner --cron-watch; minimum 5 seconds.",
    )
    parser.add_argument(
        "--cron-job-run",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--qa",
        nargs="*",
        metavar="VALUE",
        help=(
            "Manage QA Audit workflows: audit/run/status/report/list. "
            "Artifacts are stored under qa/audits/."
        ),
    )
    parser.add_argument("--workspace", default=".", help="Deepmate workspace root.")
    parser.add_argument(
        "--task",
        nargs="?",
        const="",
        help=(
            "Run in project Task Mode. Use plan or execute for model turns; "
            "checkpoint creates an achievement; status and clear are local commands."
        ),
    )
    parser.add_argument("--provider", help="Provider name from config/providers.yaml.")
    parser.add_argument(
        "--remote",
        choices=("wecom",),
        help="Run one remote channel backend, for example wecom.",
    )
    parser.add_argument(
        "--remote-validate",
        action="store_true",
        help="Validate remote channel configuration and exit without connecting.",
    )
    parser.add_argument(
        "--setup-key",
        nargs="?",
        const="-",
        metavar="API_KEY",
        help=(
            "Save the provider API key into Deepmate's local private data dir. "
            "Omit the value to paste it through stdin."
        ),
    )
    parser.add_argument(
        "--setup-wecom",
        nargs="+",
        metavar="VALUE",
        help=(
            "Save Enterprise WeChat remote settings locally. Use "
            "--setup-wecom BOT_ID SECRET [ALLOWED_USERS] [GROUP_POLICY], or pass "
            "only BOT_ID and paste the secret through stdin."
        ),
    )
    parser.add_argument("--model", help="Override provider default model.")
    parser.add_argument(
        "--upgrade-model",
        action="store_true",
        help=(
            "Use the provider's configured stronger upgrade_model for this "
            "session unless --model is set."
        ),
    )
    parser.add_argument("--profile", help="Override active profile name.")
    parser.add_argument("--session-id", help="Resume an existing Deepmate session id.")
    parser.add_argument("--session-title", help="Title for a newly created session.")
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List recent Deepmate sessions and exit.",
    )
    parser.add_argument(
        "--list-skills",
        action="store_true",
        help="List workspace skills and exit.",
    )
    parser.add_argument(
        "--list-capabilities",
        action="store_true",
        help="List governed workspace capabilities and exit.",
    )
    parser.add_argument(
        "--list-mcp",
        action="store_true",
        help="List configured MCP servers and exit.",
    )
    parser.add_argument(
        "--mcp-status",
        action="store_true",
        help="Connect to configured MCP servers, show tool exposure and schema status, and exit.",
    )
    parser.add_argument(
        "--sandbox-status",
        action="store_true",
        help="Show local sandbox backend status and exit.",
    )
    parser.add_argument(
        "--local-status",
        action="store_true",
        help="Show local Ollama model status and exit.",
    )
    parser.add_argument(
        "--prepare-local-model",
        nargs="?",
        const="",
        metavar="PRESET",
        help=(
            "Prepare a local Ollama model and exit. Omit PRESET to use the "
            "recommended local model."
        ),
    )
    parser.add_argument(
        "--validate-browser",
        action="store_true",
        help=(
            "Run a local no-network smoke check for the optional built-in "
            "browser backend and exit."
        ),
    )
    parser.add_argument(
        "--install-browser",
        action="store_true",
        help=(
            "Install and initialize the optional agent-browser backend, then exit."
        ),
    )
    parser.add_argument(
        "--pet",
        action="store_true",
        help="Start the optional lightweight desktop pet companion.",
    )
    parser.add_argument(
        "--pet-status",
        action="store_true",
        help="Show desktop pet profile and current work state without calling a model.",
    )
    parser.add_argument(
        "--pet-actions",
        action="store_true",
        help="Show pending desktop pet actions without calling a model.",
    )
    parser.add_argument(
        "--pet-select",
        choices=("dog", "cat", "squirrel", "penguin", *built_in_pet_ids()),
        help="Select the built-in desktop pet appearance.",
    )
    parser.add_argument(
        "--pet-learning",
        choices=("off", "low", "standard"),
        help="Set desktop pet learning mode.",
    )
    parser.add_argument(
        "--pet-bubble",
        choices=("smart", "frugal"),
        help="Set desktop pet bubble generation mode.",
    )
    parser.add_argument("--pet-name", help="Set an optional desktop pet name.")
    parser.add_argument(
        "--pet-proactive-care",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable low-frequency proactive care bubbles.",
    )
    parser.add_argument(
        "--hooks-status",
        action="store_true",
        help="Show hook loading, trust, and diagnostic status and exit.",
    )
    parser.add_argument(
        "--validate-hooks",
        action="store_true",
        help="Validate hook packs without executing hooks and exit.",
    )
    parser.add_argument(
        "--trust-workspace",
        action="store_true",
        help="Trust this workspace for project hook packs stored under .deepmate/hooks.",
    )
    parser.add_argument(
        "--list-mcp-tools",
        metavar="SERVER",
        help="Connect to one configured MCP server, list its tools, and exit.",
    )
    parser.add_argument(
        "--show-skill",
        metavar="SKILL",
        help="Show one workspace skill's metadata and instructions.",
    )
    parser.add_argument(
        "--inspect-skill",
        metavar="SOURCE",
        help="Inspect a skill source path, archive, URL, GitHub repo, remote skill page, or install instruction.",
    )
    parser.add_argument(
        "--install-skill",
        metavar="SOURCE",
        help="Install a full skill bundle into the workspace and verify it.",
    )
    parser.add_argument(
        "--update-skill",
        metavar="SOURCE_OR_SKILL",
        help="Update an installed skill from its manifest source or an explicit source.",
    )
    parser.add_argument(
        "--verify-skill",
        metavar="SKILL",
        help="Verify that a skill is discoverable and loadable.",
    )
    parser.add_argument(
        "--uninstall-skill",
        metavar="SKILL",
        help="Uninstall an imported skill, or hide an untracked skill unless --force-skill is set.",
    )
    parser.add_argument(
        "--list-skill-installs",
        action="store_true",
        help="List skills installed through Deepmate's skill importer.",
    )
    parser.add_argument(
        "--skill-name",
        help="Select or override a skill name when inspecting/installing a source with multiple candidates.",
    )
    parser.add_argument(
        "--skill-target",
        default="workspace",
        help="Skill install target: workspace, deepmate, or a workspace-relative directory.",
    )
    parser.add_argument(
        "--force-skill",
        action="store_true",
        help="Allow skill install/update/uninstall to replace or delete an existing workspace skill target.",
    )
    parser.add_argument(
        "--skill-state",
        nargs=2,
        metavar=("ACTION", "SKILL"),
        help="Manage skill state without calling a model: heat, cool, hide, restore.",
    )
    parser.add_argument(
        "--run-capability-maintenance",
        action="store_true",
        help="Run low-cost capability state maintenance and exit.",
    )
    parser.add_argument(
        "--show-session",
        metavar="SESSION_ID",
        help="Show one session's metadata, recent transcript, and trace summary.",
    )
    parser.add_argument(
        "--rewind",
        metavar="SESSION_ID",
        help=(
            "Preview or apply a checkpoint rewind for one session. "
            "Use with --rewind-to."
        ),
    )
    parser.add_argument(
        "--rewind-to",
        metavar="TURN_ID",
        help="Target turn id for --rewind, for example turn_00003.",
    )
    parser.add_argument(
        "--rewind-mode",
        choices=("workspace", "conversation", "both"),
        default="workspace",
        help="What to rewind: workspace files, conversation transcript, or both.",
    )
    parser.add_argument(
        "--rewind-apply",
        action="store_true",
        help="Apply the rewind. Without this flag Deepmate only prints a preview.",
    )
    parser.add_argument(
        "--rewind-force",
        action="store_true",
        help="Apply workspace rewind even when current file contents differ.",
    )
    parser.add_argument(
        "--activity-today",
        action="store_true",
        help="Show today's activity daily note and exit.",
    )
    parser.add_argument(
        "--activity-date",
        metavar="YYYY-MM-DD",
        help="Show one activity daily note and exit.",
    )
    parser.add_argument(
        "--activity-month",
        metavar="YYYY-MM",
        help="Show one activity monthly summary, or list daily notes if absent.",
    )
    parser.add_argument(
        "--activity-lines",
        type=int,
        default=160,
        help="Maximum lines to print for activity views (default: 160).",
    )
    parser.add_argument(
        "--run-memory-maintenance",
        action="store_true",
        help=(
            "Run daily memory maintenance and exit; the automatic window also "
            "runs capability and evolution maintenance."
        ),
    )
    parser.add_argument(
        "--run-evolution-maintenance",
        action="store_true",
        help="Run lightweight self-evolution maintenance and exit.",
    )
    parser.add_argument(
        "--list-evolution-changes",
        action="store_true",
        help="List applied self-evolution changes and exit.",
    )
    parser.add_argument(
        "--rollback-evolution-change",
        metavar="CHANGE_ID",
        help="Rollback one applied self-evolution change and exit.",
    )
    parser.add_argument(
        "--memory-maintenance-date",
        metavar="YYYY-MM-DD",
        help=(
            "Explicit date for --run-memory-maintenance; by default it processes "
            "the window since the last successful maintenance run."
        ),
    )
    parser.add_argument(
        "--force-memory-maintenance",
        action="store_true",
        help="Run memory maintenance even if the date already completed.",
    )
    parser.add_argument(
        "--install-memory-maintenance-schedule",
        action="store_true",
        help=(
            "Install a macOS LaunchAgent that runs daily memory, capability, "
            "and evolution maintenance at 02:00."
        ),
    )
    parser.add_argument(
        "--session-items",
        type=int,
        default=SESSION_VIEW_TRANSCRIPT_LIMIT,
        help=(
            "Number of recent transcript items to show with --show-session "
            f"(default: {SESSION_VIEW_TRANSCRIPT_LIMIT})."
        ),
    )
    parser.add_argument(
        "--trace-events",
        type=int,
        default=SESSION_VIEW_TRACE_LIMIT,
        help=(
            "Number of recent trace events to show with --show-session "
            f"(default: {SESSION_VIEW_TRACE_LIMIT})."
        ),
    )
    parser.add_argument(
        "--trace-kind",
        action="append",
        help=(
            "Filter --show-session trace output by exact trace kind. "
            "Repeat to include multiple kinds."
        ),
    )
    parser.add_argument(
        "--export-otlp",
        action="store_true",
        help="With --show-session, export matching session spans to the configured OTLP HTTP endpoint.",
    )
    parser.add_argument(
        "--validate-otlp",
        action="store_true",
        help="Send a synthetic Deepmate trace to the configured OTLP HTTP endpoint and exit.",
    )
    parser.add_argument(
        "--otlp-endpoint",
        help="Override observability.otlp.endpoint for OTLP export or validation.",
    )
    parser.add_argument("--temperature", type=float, help="Model temperature.")
    parser.add_argument("--max-tokens", type=int, help="Maximum output tokens.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Hard safety cap for one user turn. Defaults to "
            "runtime.loop_guard.hard_step_cap."
        ),
    )
    parser.add_argument(
        "--read-only-tools",
        action="store_true",
        help="Expose read-only workspace filesystem tools to the model.",
    )
    parser.add_argument(
        "--workspace-write",
        action="store_true",
        help=(
            "Expose workspace filesystem write tools. Writes are limited to "
            "non-sensitive paths inside the current workspace."
        ),
    )
    parser.add_argument(
        "--shell",
        action="store_true",
        help=(
            "Expose a sandboxed workspace shell tool for tests, builds, and "
            "small verification commands."
        ),
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help=(
            "Preload the built-in browser tool family for this run. By default "
            "Deepmate exposes only a small loader and loads browser schemas on demand."
        ),
    )
    parser.add_argument(
        "--behavior-status",
        action="store_true",
        help="Show behavior learning and Computer Use status without calling a model.",
    )
    parser.add_argument(
        "--behavior-learning",
        dest="behavior_learning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable or disable learning from Deepmate-visible user interactions. "
            "Enabled by default."
        ),
    )
    parser.add_argument(
        "--computer-use",
        action="store_true",
        help=(
            "Enable current-task Computer Use context and tools for this session. "
            "This does not enable long-term computer behavior learning."
        ),
    )
    parser.add_argument(
        "--computer-learning",
        dest="computer_learning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable or disable learning from explicit real-computer observation. "
            "Disabled by default and independent from Computer Use."
        ),
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help=(
            "Expose lightweight web_search/web_fetch tools and allow shell commands "
            "to request network=on after policy checks."
        ),
    )
    parser.add_argument(
        "--allow-env-change",
        action="store_true",
        help=(
            "Allow shell commands that may modify package or environment state, "
            "such as pip/npm/brew install."
        ),
    )
    parser.add_argument(
        "--sandbox",
        choices=tuple(mode.value for mode in SandboxMode),
        default=SandboxMode.AUTO.value,
        help=(
            "Sandbox mode for external commands: auto, require, or off "
            "(default: auto)."
        ),
    )
    parser.add_argument(
        "--mcp",
        dest="mcp",
        action="store_true",
        default=None,
        help=(
            "Enable configured MCP servers for this agent run. MCP is enabled "
            "by default when servers are configured."
        ),
    )
    parser.add_argument(
        "--no-mcp",
        dest="mcp",
        action="store_false",
        help="Disable configured MCP servers for this agent run.",
    )
    parser.add_argument(
        "--mcp-write",
        action="store_true",
        help=(
            "Allow execution of MCP tools that are not marked read-only. "
            "Default MCP exposure remains read-only."
        ),
    )
    parser.add_argument(
        "--subagents",
        action="store_true",
        help="Expose an explicit bounded run_subagent tool to the model.",
    )
    parser.add_argument(
        "--skill",
        action="append",
        default=(),
        help="Load a named SKILL.md body into the system context for this run.",
    )
    parser.add_argument(
        "--thinking",
        choices=("enabled", "disabled"),
        help="DeepSeek V4 thinking mode switch.",
    )
    parser.add_argument(
        "--reasoning-effort",
        help="Provider reasoning effort, for example high or max.",
    )
    parser.add_argument(
        "--show-reasoning",
        action="store_true",
        help="Print provider reasoning content before the final answer.",
    )
    _group_parser_help(parser)
    return parser


def _group_parser_help(parser: argparse.ArgumentParser) -> None:
    """Move flat optional arguments into readable argparse help groups."""
    groups: dict[str, argparse._ArgumentGroup] = {}

    def group(title: str) -> argparse._ArgumentGroup:
        existing = groups.get(title)
        if existing is not None:
            return existing
        created = parser.add_argument_group(title)
        groups[title] = created
        return created

    option_groups = {
        "Core": {
            "--interactive",
            "--cli",
            "--interactive-legacy",
            "--validate-runtime",
            "--doctor",
            "--cron",
            "--cron-runner",
            "--cron-watch",
            "--cron-poll-seconds",
            "--qa",
            "--workspace",
            "--task",
            "--provider",
            "--model",
            "--upgrade-model",
            "--profile",
            "--temperature",
            "--max-tokens",
            "--max-steps",
            "--thinking",
            "--reasoning-effort",
            "--show-reasoning",
        },
        "Tool Access": {
            "--read-only-tools",
            "--workspace-write",
            "--shell",
            "--browser",
            "--behavior-status",
            "--local-status",
            "--prepare-local-model",
            "--behavior-learning",
            "--no-behavior-learning",
            "--computer-use",
            "--computer-learning",
            "--no-computer-learning",
            "--allow-network",
            "--allow-env-change",
            "--sandbox",
            "--mcp",
            "--no-mcp",
            "--mcp-write",
            "--subagents",
            "--skill",
        },
        "Sessions And Recovery": {
            "--session-id",
            "--session-title",
            "--list-sessions",
            "--show-session",
            "--rewind",
            "--rewind-to",
            "--rewind-mode",
            "--rewind-apply",
            "--rewind-force",
            "--session-items",
            "--trace-events",
            "--trace-kind",
            "--export-otlp",
            "--validate-otlp",
            "--otlp-endpoint",
        },
        "Remote": {
            "--remote",
            "--remote-validate",
            "--setup-key",
            "--setup-wecom",
        },
        "Skills And MCP": {
            "--list-skills",
            "--list-capabilities",
            "--list-mcp",
            "--mcp-status",
            "--list-mcp-tools",
            "--show-skill",
            "--inspect-skill",
            "--install-skill",
            "--update-skill",
            "--verify-skill",
            "--uninstall-skill",
            "--list-skill-installs",
            "--skill-name",
            "--skill-target",
            "--force-skill",
            "--skill-state",
            "--run-capability-maintenance",
        },
        "Hooks And Sandbox": {
            "--sandbox-status",
            "--hooks-status",
            "--validate-hooks",
            "--trust-workspace",
            "--validate-browser",
            "--install-browser",
        },
        "Pet": {
            "--pet",
            "--pet-status",
            "--pet-actions",
            "--pet-select",
            "--pet-learning",
            "--pet-bubble",
            "--pet-name",
            "--pet-proactive-care",
        },
        "Memory Activity Evolution": {
            "--activity-today",
            "--activity-date",
            "--activity-month",
            "--activity-lines",
            "--run-memory-maintenance",
            "--run-evolution-maintenance",
            "--list-evolution-changes",
            "--rollback-evolution-change",
            "--memory-maintenance-date",
            "--force-memory-maintenance",
            "--install-memory-maintenance-schedule",
        },
    }
    target_by_option = {
        option: title
        for title, options in option_groups.items()
        for option in options
    }
    option_group = next(
        (action_group for action_group in parser._action_groups if action_group.title == "options"),
        None,
    )
    if option_group is None:
        return
    remaining = []
    for action in option_group._group_actions:
        option = next((name for name in action.option_strings if name in target_by_option), "")
        if not option:
            remaining.append(action)
            continue
        group(target_by_option[option])._group_actions.append(action)
    option_group._group_actions = remaining


def _model_options(args: argparse.Namespace) -> dict[str, object]:
    options: dict[str, object] = {}
    if args.temperature is not None:
        options["temperature"] = args.temperature
    if args.max_tokens is not None:
        options["max_tokens"] = args.max_tokens
    if args.thinking is not None:
        options["thinking"] = {"type": args.thinking}
    if args.reasoning_effort is not None:
        options["reasoning_effort"] = args.reasoning_effort
    return options


def _resolve_chat_model(args: argparse.Namespace, provider_settings) -> str:
    """Return the main chat model selected by CLI flags and provider config."""
    explicit_model = (args.model or "").strip()
    if explicit_model:
        return explicit_model
    if getattr(args, "upgrade_model", False):
        upgrade_model = (getattr(provider_settings, "upgrade_model", "") or "").strip()
        if not upgrade_model:
            raise ValueError(
                "current provider has one configured model; use --model for a "
                "temporary model override"
            )
        return upgrade_model
    primary = provider_settings.primary_model()
    if not primary:
        raise ValueError(f"provider {provider_settings.name} missing model")
    return primary


def _validate_provider_connection(provider_settings: ProviderSettings) -> None:
    if not provider_settings.base_url.strip():
        raise ValueError(f"provider {provider_settings.name} missing base_url")


def _runtime_validation_prompt(args: argparse.Namespace, user_prompt: str = "") -> str:
    """Return the small validation prompt used by --validate-runtime."""
    lines = [
        "Runtime validation request.",
        "Return a concise final answer containing the phrase `deepmate runtime ok`.",
    ]
    if args.workspace_write:
        lines.extend(
            (
                "",
                "Native workspace write check:",
                "- First call the `write_text_file` tool once.",
                "- Use path `runtime_validation_write_test.txt`.",
                "- Use content `deepmate write ok`.",
                "- Use overwrite=true.",
                "- Then finish the answer.",
            )
        )
    elif args.read_only_tools:
        lines.extend(
            (
                "",
                "Native tool check:",
                "- First call the `list_directory` tool once with path `.`.",
                "- Then use the result to finish the answer.",
                "- Do not write files.",
            )
        )
    if args.subagents:
        lines.extend(
            (
                "",
                "Subagent check:",
                "- Also call `run_subagent` once for a tiny delegated check.",
                "- Use goal: `Reply with deepmate subagent ok in one sentence.`",
                "- Set max_steps=2.",
                "- Do not give the subagent any tools.",
            )
        )
    if getattr(args, "shell", False):
        lines.extend(
            (
                "",
                "Shell tool check:",
                "- Also call `run_shell_command` once.",
                "- Use command `pwd`.",
                "- Use cwd `.` and network `off`.",
                "- Then use the result to finish the answer.",
            )
        )
    if args.mcp:
        lines.extend(
            (
                "",
                "MCP check:",
                "- MCP discovery may expose read-only tools.",
                "- Do not call MCP tools unless their schema makes the validation task obvious.",
            )
        )
    if user_prompt.strip():
        lines.extend(("", "Additional user note:", user_prompt.strip()))
    return "\n".join(lines)


def _finish_runtime_validation(
    result,
    session_store: SessionStore,
    session: SessionRecord,
    transcript: TranscriptStore,
    trace_path: Path,
    trace_recorder: TraceRecorder,
    model: str,
    require_native_tool: bool,
    require_workspace_write: bool,
    require_shell: bool,
    require_subagent: bool,
    require_activity_note: bool,
    mcp_tool_count: int,
) -> int:
    """Print a concise runtime validation report and return a CLI exit code."""
    transcript_records = transcript.load_records()
    trace_records = _read_session_trace_records(trace_path, session.session_id)
    errors = _runtime_validation_errors(
        result=result,
        transcript_record_count=len(transcript_records),
        trace_records=trace_records,
        require_native_tool=require_native_tool,
        require_workspace_write=require_workspace_write,
        require_shell=require_shell,
        require_subagent=require_subagent,
        require_activity_note=require_activity_note,
    )
    status = "ok" if not errors else "failed"
    final_text = _response_text(result.final_step().response)
    trace_recorder.record(
        TraceEvent(
            kind="runtime_validation_finished",
            summary=f"Runtime validation {status}.",
            refs=(
                f"session_id={session.session_id}",
                f"status={status}",
                f"model={model}",
                f"transcript_records={len(transcript_records)}",
                f"trace_records={len(trace_records)}",
                "native_tool_completed="
                f"{_trace_count(trace_records, 'native_tool_completed')}",
                "activity_daily_note_written="
                f"{_trace_count(trace_records, 'activity_daily_note_written')}",
                "session_summary_completed="
                f"{_trace_count(trace_records, 'session_summary_completed')}",
                "subagent_tool_completed="
                f"{_trace_count(trace_records, 'subagent_tool_completed')}",
                "shell_tool_completed="
                f"{_trace_native_tool_ref_count(trace_records, 'run_shell_command')}",
                f"mcp_tools_discovered={mcp_tool_count}",
            ),
        )
    )
    session_store.touch(session.session_id)
    trace_records = _read_session_trace_records(trace_path, session.session_id)
    _print_runtime_validation_report(
        status=status,
        session=session,
        transcript=transcript,
        trace_path=trace_path,
        trace_records=trace_records,
        transcript_record_count=len(transcript_records),
        model=model,
        final_text=final_text,
        errors=errors,
        mcp_tool_count=mcp_tool_count,
    )
    return 0 if not errors else 1


def _runtime_validation_errors(
    result,
    transcript_record_count: int,
    trace_records: Sequence[Mapping[str, object]],
    require_native_tool: bool,
    require_workspace_write: bool,
    require_shell: bool,
    require_subagent: bool,
    require_activity_note: bool,
) -> tuple[str, ...]:
    errors = [error.message for error in result.errors()]
    if result.reached_max_steps:
        errors.append("runtime reached max_steps before final answer")
    if not _response_text(result.final_step().response):
        errors.append("model response was empty")
    if transcript_record_count < 1:
        errors.append("transcript was not written")
    if not trace_records:
        errors.append("trace was not written")
    if _trace_count(trace_records, "model_response_received") < 1:
        errors.append("model response trace was not written")
    if require_native_tool and _trace_count(trace_records, "native_tool_completed") < 1:
        errors.append("native read-only tool call was not completed")
    if require_workspace_write and not _trace_has_native_tool_ref(
        trace_records,
        "write_text_file",
    ):
        errors.append("workspace write tool call was not completed")
    if require_shell and not _trace_has_native_tool_ref(
        trace_records,
        "run_shell_command",
    ):
        errors.append("shell tool call was not completed")
    if require_subagent and _trace_count(trace_records, "subagent_tool_completed") < 1:
        errors.append("subagent tool call was not completed")
    if require_activity_note and _trace_count(trace_records, "activity_daily_note_written") < 1:
        errors.append("activity daily note was not written")
    return tuple(errors)


def _runtime_validation_session_status(result) -> str:
    if result.has_errors():
        return "failed"
    if result.reached_max_steps:
        return "max_steps_reached"
    return "completed"


def _print_runtime_validation_report(
    status: str,
    session: SessionRecord,
    transcript: TranscriptStore,
    trace_path: Path,
    trace_records: Sequence[Mapping[str, object]],
    transcript_record_count: int,
    model: str,
    final_text: str,
    errors: Sequence[str],
    mcp_tool_count: int,
) -> None:
    print("Runtime validation")
    print(f"  status: {status}")
    print(f"  session: {session.session_id}: {session.title}")
    print(f"  model: {model}")
    print(f"  transcript: {transcript.path} (items={transcript_record_count})")
    print(f"  trace: {trace_path} (events={len(trace_records)})")
    print(
        "  observed: "
        f"model_responses={_trace_count(trace_records, 'model_response_received')}, "
        f"native_tools={_trace_count(trace_records, 'native_tool_completed')}, "
        f"activity_notes={_trace_count(trace_records, 'activity_daily_note_written')}, "
        f"session_summaries={_trace_count(trace_records, 'session_summary_completed')}, "
        f"subagents={_trace_count(trace_records, 'subagent_tool_completed')}, "
        f"mcp_tools_discovered={mcp_tool_count}"
    )
    if final_text.strip():
        print(f"  final: {_preview_text(final_text, 240)}")
    if errors:
        print("  errors:")
        for error in errors:
            print(f"    - {error}")
    print(
        "  inspect: "
        f"PYTHONPATH=src python3 -m deepmate --show-session {session.session_id}"
    )


def _trace_count(records: Sequence[Mapping[str, object]], kind: str) -> int:
    clean_kind = kind.strip()
    return sum(
        1
        for record in records
        if str(record.get("kind", "")).strip() == clean_kind
    )


def _trace_has_native_tool_ref(
    records: Sequence[Mapping[str, object]],
    tool_name: str,
) -> bool:
    expected = tool_name.strip()
    return any(
        str(record.get("kind", "")).strip() == "native_tool_completed"
        and expected in {
            str(ref).strip()
            for ref in _iter_values(record.get("refs"))
        }
        for record in records
    )


def _trace_native_tool_ref_count(
    records: Sequence[Mapping[str, object]],
    tool_name: str,
) -> int:
    expected = tool_name.strip()
    return sum(
        1
        for record in records
        if str(record.get("kind", "")).strip() == "native_tool_completed"
        and expected in {str(ref).strip() for ref in _iter_values(record.get("refs"))}
    )


def _response_text(response) -> str:
    return response.content.strip() or response.reasoning.strip()


def _safe_save_pet_event(store: PetStateStore, event) -> None:
    try:
        store.save_current_state(event)
    except Exception:
        pass


def _pet_settings_requested(args: argparse.Namespace) -> bool:
    return bool(
        args.pet_learning is not None
        or args.pet_bubble is not None
        or args.pet_name is not None
        or args.pet_proactive_care is not None
    )


def _apply_pet_profile_settings(
    store: PetStateStore,
    args: argparse.Namespace,
):
    profile = store.select_pet(args.pet_select) if args.pet_select else store.load_profile()
    updates: dict[str, object] = {}
    if args.pet_learning is not None:
        updates["learning_mode"] = args.pet_learning
    if args.pet_bubble is not None:
        updates["bubble_generation"] = args.pet_bubble
    if args.pet_name is not None:
        updates["name"] = args.pet_name.strip()
    if args.pet_proactive_care is not None:
        updates["proactive_care"] = args.pet_proactive_care
    if updates:
        profile = replace(profile, **updates)
        store.save_profile(profile)
    return profile


def _format_pet_profile_update(profile, learning_sources: object = None) -> str:
    name = profile.name.strip() or "(none)"
    text = (
        "desktop pet updated:"
        f"\n- pet_id: {profile.pet_id}"
        f"\n- species: {profile.species}"
        f"\n- style: {profile.style}"
        f"\n- name: {name}"
        f"\n- bubble_generation: {profile.bubble_generation}"
        f"\n- learning_mode: {profile.learning_mode}"
        f"\n- proactive_care: {str(profile.proactive_care).lower()}"
    )
    if profile.learning_mode != "off":
        sources = (
            ", ".join(str(item) for item in learning_sources if str(item).strip())
            if isinstance(learning_sources, list)
            else ""
        )
        text += f"\n- learning_sources: {sources or '(none configured)'}"
    return text


def _pet_turn_summary(result, final_text: str, review_summary: str) -> str:
    if final_text.strip():
        return final_text
    if result.loop_guard_stop is not None:
        return result.loop_guard_stop.message
    errors = result.errors()
    if errors:
        return errors[0].message
    if result.reached_max_steps:
        return "Reached max steps before final answer."
    return review_summary or "Turn finished."


class _MissingApiKeyProvider:
    """Provider placeholder that lets interactive mode open before key setup."""

    def __init__(self, settings: AppSettings, provider_settings) -> None:
        self._settings = settings
        self._provider_settings = provider_settings
        self._message = _missing_api_key_message(settings, provider_settings)

    def complete(self, request: ModelRequest) -> ModelResponse:
        api_key = self._provider_settings.api_key(self._settings.data_dir)
        if not api_key:
            raise AuthError(self._message)
        provider = ChatCompletionsProvider(
            base_url=self._provider_settings.base_url,
            api_key=api_key,
        )
        return provider.complete(request)

    def complete_stream(self, request: ModelRequest, on_delta) -> ModelResponse:
        api_key = self._provider_settings.api_key(self._settings.data_dir)
        if not api_key:
            on_delta(StreamDelta(content=self._message))
            raise AuthError(self._message)
        provider = ChatCompletionsProvider(
            base_url=self._provider_settings.base_url,
            api_key=api_key,
        )
        return provider.complete_stream(request, on_delta)


def _missing_api_key_message(settings: AppSettings, provider_settings) -> str:
    secret_path = settings.data_dir / "secrets" / "providers.env"
    return "\n".join(
        (
            "Deepmate needs a model API key before it can answer.",
            "",
            "The simplest path is to save it locally for this workspace:",
            "  deepmate "
            f"--provider {shlex.quote(provider_settings.name)} "
            f"--workspace {shlex.quote(str(settings.workspace))} --setup-key",
            "",
            "Or, if you are already in the TUI:",
            "  /setup-key <your-api-key>",
            "",
            "Deepmate stores this under local runtime data, not in project source:",
            f"  {secret_path}",
            "",
            "Advanced shell option:",
            f"  export {provider_settings.api_key_env}=<your-api-key>",
        )
    )


def _local_status_text(settings: AppSettings, args: argparse.Namespace) -> str:
    provider_settings = settings.provider(LOCAL_PROVIDER_NAME)
    runtime = _local_runtime(provider_settings)
    status = runtime.status()
    prepared = runtime.prepared_model(start_server=False)
    lines = [
        "Local model status:",
        f"- ollama_installed: {str(status.installed).lower()}",
        f"- ollama_running: {str(status.running).lower()}",
        f"- available: {str(status.available).lower()}",
    ]
    if status.version:
        lines.append(f"- version: {status.version}")
    if prepared is not None:
        lines.append(f"- prepared_model: {prepared.id} ({prepared.runtime_name})")
    else:
        recommended = recommended_local_model()
        lines.append("- prepared_model: none")
        lines.append(f"- recommended: {recommended.id} ({recommended.ollama_ref})")
        lines.append(
            f"- next: deepmate --prepare-local-model {recommended.id}"
        )
    if status.message:
        lines.append(f"- message: {status.message}")
    return "\n".join(lines)


def _doctor_text(settings: AppSettings, args: argparse.Namespace) -> str:
    """Return a no-model, no-install health report for first-run setup."""
    lines = [
        "Deepmate doctor",
        "",
        "Base install:",
        f"- python: {sys.version.split()[0]}",
        f"- workspace: {settings.workspace}",
        f"- data_dir: {settings.data_dir}",
        f"- config: {_doctor_ok(_doctor_path_exists(settings.workspace / 'config' / 'deepmate.yaml'))}",
        f"- providers: {_doctor_ok(_doctor_path_exists(settings.workspace / 'config' / 'providers.yaml'))}",
        f"- textual_tui: {_doctor_ok(_doctor_module_available('textual'))}",
    ]
    cards, warnings = discover_skill_cards(settings.workspace, data_dir=settings.data_dir)
    builtin_count = sum(1 for card in cards if card.is_builtin())
    lines.append(f"- built_in_skills: {_doctor_ok(builtin_count > 0)} ({builtin_count})")
    if warnings:
        lines.extend(f"  warning: {warning.message}" for warning in warnings[:4])
    try:
        provider_settings = settings.provider(args.provider)
        provider_ok = bool(provider_settings.api_key(settings.data_dir))
        lines.append(
            "- model_key: "
            f"{_doctor_ok(provider_ok)} ({provider_settings.api_key_env})"
        )
        if not provider_ok:
            lines.append(
                "  next: deepmate "
                f"--provider {shlex.quote(provider_settings.name)} "
                f"--workspace {shlex.quote(str(settings.workspace))} --setup-key"
            )
    except ValueError as exc:
        lines.append(f"- model_key: missing ({exc})")

    sandbox_status = _sandbox_status(settings, args)
    lines.extend(
        (
            "",
            "Optional features:",
            f"- sandbox_backend: {_doctor_ok(sandbox_status.available)} ({sandbox_status.backend})",
            f"- desktop_pet: {_doctor_optional(electron_pet_command(settings.data_dir) is not None)}",
            "  setup: run /pet setup in the TUI, or set DEEPMATE_PET_ELECTRON to an existing Electron binary",
            f"- browser_backend: {_doctor_optional(AgentBrowserBackend(settings.workspace).is_available())}",
            "  install: deepmate --install-browser",
        )
    )
    try:
        local_status = _local_runtime(settings.provider(LOCAL_PROVIDER_NAME)).status()
        lines.append(
            "- local_model_runtime: "
            f"{_doctor_optional(local_status.available)} "
            f"(ollama_installed={_bool_text(local_status.installed)}, "
            f"running={_bool_text(local_status.running)})"
        )
        if not local_status.available:
            lines.append("  prepare: deepmate --prepare-local-model")
    except (OSError, ValueError) as exc:
        lines.append(f"- local_model_runtime: optional missing ({exc})")
        lines.append("  prepare: deepmate --prepare-local-model")
    lines.extend(
        (
            "",
            "Notes:",
            "- Optional features are loaded on demand; missing Electron, agent-browser, or Ollama does not block normal CLI/TUI use.",
            "- Desktop pet UI assets ship with Deepmate; Electron is the optional runtime.",
            "- For a real provider-backed smoke test, run: deepmate --validate-runtime --thinking disabled",
        )
    )
    return "\n".join(lines)


def _handle_cron_cli(workspace: Path, values: Sequence[str] | None) -> str:
    parts = tuple(value for value in (values or ()) if str(value).strip())
    if not parts:
        return handle_cron_command("status", workspace=workspace)
    action = parts[0].strip().lower()
    if action == "tick":
        return run_due_jobs(workspace=workspace)
    if action == "run":
        if len(parts) < 2:
            raise ValueError("--cron run requires a job id")
        return run_job_now(parts[1], workspace=workspace)
    return handle_cron_command(" ".join(parts), workspace=workspace)


def _handle_qa_cli(
    workspace: Path,
    values: Sequence[str] | None,
    *,
    provider,
    model: str,
    options: Mapping[str, object],
    allow_fallback: bool = False,
) -> str:
    parts = tuple(value for value in (values or ()) if str(value).strip())
    if not parts:
        return handle_qa_command("help", workspace=workspace)
    return handle_qa_command(
        " ".join(parts),
        workspace=workspace,
        provider=provider,
        model=model,
        options=options,
        allow_fallback=allow_fallback,
    )


def _doctor_path_exists(path: Path) -> bool:
    return path.exists()


def _doctor_module_available(module: str) -> bool:
    try:
        __import__(module)
    except Exception:
        return False
    return True


def _doctor_ok(ok: bool) -> str:
    return "ok" if ok else "missing"


def _doctor_optional(ok: bool) -> str:
    return "ready" if ok else "optional missing"


def _bool_text(value: bool) -> str:
    return str(value).lower()


def _prepare_local_model_command(settings: AppSettings, args: argparse.Namespace):
    provider_settings = settings.provider(LOCAL_PROVIDER_NAME)
    requested = (args.prepare_local_model or "").strip()
    if requested:
        preset = local_model_by_id(requested) or local_model_by_runtime_name(requested)
    else:
        preset = recommended_local_model()
    if preset is None:
        raise ValueError(f"unknown local model preset: {requested}")
    runtime = _local_runtime(provider_settings)

    def progress_sink(progress) -> None:
        print(progress.message, file=sys.stderr)

    return runtime.prepare_model(
        preset,
        progress=progress_sink,
        state_store=LocalModelStateStore(settings.data_dir),
        install_missing_runtime=False,
    )


def _local_model_ready(settings: AppSettings, preset) -> bool:
    runtime = _local_runtime(settings.provider(LOCAL_PROVIDER_NAME))
    try:
        prepared = runtime.prepared_model()
    except Exception:
        return False
    if prepared is None:
        return False
    return prepared.runtime_name == preset.runtime_name or prepared.id == preset.id


def _local_model_not_ready_message(preset) -> str:
    return "\n".join(
        (
            "Local model is not ready.",
            f"- required: {preset.id} ({preset.ollama_ref})",
            f"- prepare: deepmate --prepare-local-model {preset.id}",
            "- status: deepmate --local-status",
        )
    )


def _local_runtime(provider_settings: ProviderSettings) -> OllamaLocalRuntime:
    return OllamaLocalRuntime(
        api_url=ollama_api_url_from_provider_base_url(provider_settings.base_url)
    )


def _pet_copy_provider(settings: AppSettings, args: argparse.Namespace):
    try:
        provider_settings = settings.provider(args.provider)
    except ValueError:
        return None, ""
    api_key = provider_settings.api_key(settings.data_dir)
    if not api_key:
        return None, ""
    return (
        ChatCompletionsProvider(
            base_url=provider_settings.base_url,
            api_key=api_key,
        ),
        args.model or provider_settings.primary_model(),
    )


def _format_pet_status(store: PetStateStore) -> str:
    profile = store.load_profile()
    state = store.load_current_state() or store.offline_state()
    learning_state = store.load_learning_state()
    pending_actions = store.pending_actions(limit=100)
    lines = [
        "Desktop pet",
        f"- pet_id: {profile.pet_id}",
        f"- species: {profile.species}",
        f"- name: {profile.name.strip() or '(none)'}",
        f"- style: {profile.style}",
        f"- bubble_generation: {profile.bubble_generation}",
        f"- learning_mode: {profile.learning_mode}",
        f"- proactive_care: {str(profile.proactive_care).lower()}",
        f"- pending_actions: {len(pending_actions)}",
        f"- current_kind: {str(state.get('kind', '')).strip() or '(none)'}",
        f"- current_state: {str(state.get('state', '')).strip() or '(none)'}",
        f"- current_work: {str(state.get('current_work_title', '')).strip() or '(none)'}",
        f"- summary: {_preview_text(str(state.get('summary', '') or ''), 220) or '(none)'}",
    ]
    if profile.learning_mode != "off" and isinstance(learning_state.get("sources"), list):
        sources = ", ".join(
            str(item) for item in learning_state.get("sources", []) if str(item).strip()
        )
        lines.append(f"- learning_sources: {sources or '(none configured)'}")
    session_id = str(state.get("session_id", "")).strip()
    if session_id:
        lines.append(f"- session_id: {session_id}")
    return "\n".join(lines)


def _format_pet_actions(
    store: PetStateStore,
    session_store: SessionStore,
) -> str:
    pending = store.pending_actions(limit=50)
    lines = ["Desktop pet actions", f"- pending: {len(pending)}"]
    for index, action in pending:
        lines.append("")
        lines.append(f"[{index}] {action.action}")
        if action.created_at:
            lines.append(f"- created_at: {action.created_at}")
        lines.extend(_format_pet_action_payload(action, session_store))
    return "\n".join(lines)


def _format_pet_action_payload(
    action: PetUserAction,
    session_store: SessionStore,
) -> list[str]:
    payload = action.payload
    title = _text_value(payload.get("title"))
    workspace = _text_value(payload.get("workspace"))
    session_id = _text_value(payload.get("session_id"))
    lines: list[str] = []
    if title:
        lines.append(f"- title: {title}")
    if workspace:
        lines.append(f"- workspace: {workspace}")
    if session_id:
        try:
            session = session_store.load(session_store.resolve_id(session_id))
        except (OSError, ValueError, json.JSONDecodeError):
            lines.append(f"- session: {session_id} (not found)")
        else:
            lines.append(f"- session: {session.session_id}: {session.title}")
            lines.append(
                "- open: "
                f"PYTHONPATH=src python3 -m deepmate --interactive "
                f"--session-id {session.session_id}"
            )
    return lines or ["- payload: (empty)"]


def _consume_pet_open_actions(
    store: PetStateStore,
    session_store: SessionStore,
) -> str:
    pending = store.pending_actions(limit=50)
    processed = 0
    selected_session_id = ""
    for index, action in pending:
        if action.action != "open_current_work":
            continue
        processed = max(processed, index)
        session_id = _text_value(action.payload.get("session_id"))
        if not session_id:
            continue
        try:
            session = session_store.load(session_store.resolve_id(session_id))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        selected_session_id = session.session_id
    if processed:
        try:
            store.mark_actions_processed(processed)
        except OSError:
            pass
    if selected_session_id:
        _print_status(f"desktop pet requested current work: {selected_session_id}")
    return selected_session_id


def _review_delivery(
    user_request: str,
    final_response: str,
    result,
    trace_recorder: TraceRecorder,
    session_id: str,
    activation_refs: tuple[str, ...],
) -> DeliveryReview:
    review_input = build_delivery_review_input(
        user_request=user_request,
        final_response_draft=final_response,
        tool_exchanges=result.tool_exchanges,
        errors=result.errors(),
        reached_max_steps=result.reached_max_steps,
    )
    review = review_final_response(review_input)
    trace_recorder.record(
        TraceEvent(
            kind="delivery_review_finished",
            summary=review.summary,
            refs=(
                f"session_id={session_id}",
                f"status={review.status.value}",
                f"issues={len(review.issues)}",
                f"llm_review_suggested={should_run_llm_delivery_review(review_input)}",
                *review.issues,
                *activation_refs,
            ),
        )
    )
    return review


def _record_capability_state_trace(
    recorder: TraceRecorder,
    kind: str,
    summary: str,
    state: CapabilityState,
    session_id: str,
) -> None:
    recorder.record(
        TraceEvent(
            kind=kind,
            summary=summary,
            refs=(
                f"session_id={session_id}",
                f"capability_id={state.capability_id}",
                f"capability_kind={state.kind.value}",
                f"skill={state.name}",
                f"temperature={state.temperature.value}",
                f"exposure={state.exposure()}",
            ),
        )
    )


def _print_delivery_review_warning(review: DeliveryReview) -> None:
    if review.status == DeliveryReviewStatus.ACCEPTED:
        return
    if review.status == DeliveryReviewStatus.BLOCKED:
        print(f"error: {review.summary}", file=sys.stderr)
        return
    if review.issues:
        print(
            "warning: delivery review needs attention: "
            + ", ".join(review.issues),
            file=sys.stderr,
        )


def _title_from_prompt(prompt: str, limit: int = 48) -> str:
    title = " ".join(prompt.split())
    if not title:
        return "Untitled session"
    if len(title) <= limit:
        return title
    return title[: limit - 3].rstrip() + "..."


def _print_sessions(sessions: Sequence[SessionRecord]) -> None:
    if not sessions:
        print("No sessions found.")
        return
    print(f"{'SESSION ID: TITLE':<72}  UPDATED")
    for session in sessions:
        print(
            f"{session.session_id}: {session.title:<37}  "
            f"{session.updated_at}"
        )


def _print_skills(workspace: Path) -> None:
    cards, warnings = discover_skill_cards(workspace)
    for warning in warnings:
        _print_warning(warning.message)
    print(format_skill_list(cards, workspace))


def _print_capabilities(settings: AppSettings, profile_name: str | None = None) -> None:
    cards, warnings = discover_skill_cards(settings.workspace)
    for warning in warnings:
        _print_warning(warning.message)
    workspace_cards, _workspace_warnings = discover_workspace_skill_cards(
        settings.workspace
    )
    state_store = CapabilityStateStore.in_data_dir(
        settings.data_dir,
        settings.profile_ref(profile_name),
    )
    state_store.sync_workspace_skills(workspace_cards, settings.workspace)
    print(format_capability_list(cards, settings.workspace, state_store))


def _apply_skill_state_command(
    settings: AppSettings,
    action_and_name: Sequence[str],
    profile_name: str | None = None,
) -> CapabilityState:
    if len(action_and_name) != 2:
        raise ValueError("--skill-state requires ACTION and SKILL")
    action, name = action_and_name
    cards, warnings = discover_workspace_skill_cards(settings.workspace)
    for warning in warnings:
        _print_warning(warning.message)
    state_store = CapabilityStateStore.in_data_dir(
        settings.data_dir,
        settings.profile_ref(profile_name),
    )
    state_store.sync_workspace_skills(cards, settings.workspace)
    return state_store.set_skill_state(name, action)


def _run_capability_maintenance(
    settings: AppSettings,
    profile_name: str | None = None,
) -> CapabilityMaintenanceResult:
    cards, warnings = discover_workspace_skill_cards(settings.workspace)
    for warning in warnings:
        _print_warning(warning.message)
    state_store = CapabilityStateStore.in_data_dir(
        settings.data_dir,
        settings.profile_ref(profile_name),
    )
    return run_daily_capability_maintenance(
        cards=cards,
        workspace=settings.workspace,
        state_store=state_store,
        trace_recorder=TraceRecorder(JsonlTraceSink(settings.trace_sink)),
    )


def _format_capability_maintenance_result(
    result: CapabilityMaintenanceResult,
    workspace: Path,
) -> str:
    return "\n".join(
        (
            f"capability maintenance: {result.reason}",
            f"- skills_seen: {result.skills_seen}",
            f"- states_seen: {result.states_seen}",
            f"- cooled: {result.cooled}",
            f"- proposals_created: {result.proposals_created}",
            f"- state: {_workspace_relative(result.state_path, workspace)}",
            f"- proposals: {_workspace_relative(result.proposals_path, workspace)}",
        )
    )


def _run_evolution_maintenance(settings: AppSettings, profile_name: str | None = None):
    profile = settings.profile_ref(profile_name)
    return run_evolution_maintenance(
        workspace=settings.workspace,
        data_dir=settings.data_dir,
        profile=profile,
        trace_path=settings.trace_sink,
        sessions_dir=settings.data_dir / "sessions",
        activity_dir=settings.data_dir / "activity" / (profile.name.strip() or "default"),
        trace_recorder=TraceRecorder(JsonlTraceSink(settings.trace_sink)),
    )


def _format_evolution_maintenance_result(result, workspace: Path) -> str:
    lines = [
        f"evolution maintenance: {result.reason}",
        f"- ran: {str(result.ran).lower()}",
        f"- behavior_changes: {result.behavior_changes}",
        f"- failure_patterns_updated: {result.failure_patterns_updated}",
        f"- generated_skill_changes: {result.generated_skill_changes}",
        f"- capability_state_changes: {result.capability_state_changes}",
        f"- trace_records_seen: {result.trace_records_seen}",
        f"- sessions_seen: {result.sessions_seen}",
        f"- activity_notes_seen: {result.activity_notes_seen}",
        f"- capability_states_seen: {result.capability_states_seen}",
        f"- token_cost: {result.metrics.token_cost}",
        f"- loaded_skills_count: {result.metrics.loaded_skills_count}",
        f"- used_skills_count: {result.metrics.used_skills_count}",
        f"- tool_failure_count: {result.metrics.tool_failure_count}",
        f"- user_correction_count: {result.metrics.user_correction_count}",
        f"- generated_skill_apply_count: {result.metrics.generated_skill_apply_count}",
        f"- rollback_count: {result.metrics.rollback_count}",
        f"- applied_log: {_workspace_relative(result.applied_log_path, workspace)}",
        f"- maintenance_state: {_workspace_relative(result.maintenance_state_path, workspace)}",
        f"- metrics: {_workspace_relative(result.metrics_path, workspace)}",
    ]
    if result.changed_paths:
        lines.append("- changed_paths:")
        lines.extend(
            f"  - {_workspace_relative(path, workspace)}" for path in result.changed_paths
        )
    return "\n".join(lines)


def _format_evolution_changes(
    settings: AppSettings,
    profile_name: str | None,
    limit: int = 20,
) -> str:
    store = EvolutionChangeStore.in_data_dir(
        settings.data_dir,
        settings.profile_ref(profile_name),
    )
    changes = store.load()
    if not changes:
        return "No evolution changes found."
    recent = tuple(reversed(changes[-max(1, limit) :]))
    lines = [f"Evolution changes (showing {len(recent)} of {len(changes)}):"]
    for change in recent:
        raw_target = Path(change.target_path)
        target_path = raw_target if raw_target.is_absolute() else settings.workspace / raw_target
        target = _workspace_relative(target_path, settings.workspace)
        lines.append(
            "- "
            f"{change.change_id} "
            f"type={change.change_type} "
            f"status={change.status} "
            f"decision={change.decision} "
            f"target={target}"
        )
        summary = _text_value(change.summary)
        if summary:
            lines.append(f"  summary={summary}")
        if change.evidence_refs:
            lines.append("  refs=" + ", ".join(change.evidence_refs[:4]))
    return "\n".join(lines)


def _rollback_evolution_change(
    settings: AppSettings,
    profile_name: str | None,
    change_id: str,
):
    store = EvolutionChangeStore.in_data_dir(
        settings.data_dir,
        settings.profile_ref(profile_name),
    )
    return store.rollback(change_id, settings.workspace)


def _print_skill_detail(workspace: Path, name: str) -> None:
    catalog, warnings = _skill_catalog(workspace)
    for warning in warnings:
        _print_warning(warning.message)
    document = select_skill_documents(
        catalog,
        (name,),
        workspace,
        command_name="--show-skill",
    )[0]
    print(format_skill_document(document, workspace))


def _install_skill_command(settings: AppSettings, args: argparse.Namespace, source: str):
    state_store = CapabilityStateStore.in_data_dir(
        settings.data_dir,
        settings.profile_ref(args.profile),
    )
    return install_skill_source(
        source,
        settings.workspace,
        settings.data_dir,
        state_store,
        target=args.skill_target,
        skill_name=args.skill_name or "",
        force=args.force_skill,
    )


def _update_skill_command(settings: AppSettings, args: argparse.Namespace, source_or_name: str):
    state_store = CapabilityStateStore.in_data_dir(
        settings.data_dir,
        settings.profile_ref(args.profile),
    )
    return update_skill_source(
        source_or_name,
        settings.workspace,
        settings.data_dir,
        state_store,
        target=args.skill_target,
        skill_name=args.skill_name or "",
    )


def _verify_skill_command(settings: AppSettings, args: argparse.Namespace, name: str):
    state_store = CapabilityStateStore.in_data_dir(
        settings.data_dir,
        settings.profile_ref(args.profile),
    )
    return verify_skill_install(
        name,
        settings.workspace,
        settings.data_dir,
        state_store,
    )


def _uninstall_skill_command(settings: AppSettings, args: argparse.Namespace, name: str):
    state_store = CapabilityStateStore.in_data_dir(
        settings.data_dir,
        settings.profile_ref(args.profile),
    )
    return uninstall_skill(
        name,
        settings.workspace,
        settings.data_dir,
        state_store,
        force=args.force_skill,
    )


def _print_mcp_tools(settings: AppSettings, server_name: str) -> None:
    server = _mcp_server(settings, server_name)
    tools = discover_mcp_tools(server, settings.workspace)
    if not tools:
        print(f"No MCP tools discovered for server: {server.name}")
        return
    print(f"MCP tools for {server.name}:")
    for tool in tools:
        print(_format_mcp_tool_line(tool))


def _print_mcp_status(settings: AppSettings, profile_name: str | None = None) -> None:
    if not settings.mcp_servers:
        print("No MCP servers configured.")
        return
    state_store = McpUsageStateStore.in_data_dir(
        settings.data_dir,
        settings.profile_ref(profile_name),
    )
    catalog = discover_mcp_catalog(
        settings.mcp_servers,
        settings.workspace,
        state_store=state_store,
    )
    print(
        format_mcp_catalog_status(
            catalog,
            model_context_tokens=_default_model_context_tokens(settings),
        )
    )


def _print_sandbox_status(settings: AppSettings, args: argparse.Namespace) -> None:
    status = _sandbox_status(settings, args)
    print("Sandbox status:")
    print(f"- platform: {status.platform}")
    print(f"- mode: {status.mode.value}")
    print(f"- backend: {status.backend}")
    print(f"- available: {str(status.available).lower()}")
    print(f"- sandboxed: {str(status.sandboxed).lower()}")
    print(f"- workspace: {status.workspace}")
    print(f"- network_default: {status.network_default}")
    if status.warning:
        print(f"- warning: {status.warning}")


def _sandbox_status(settings: AppSettings, args: argparse.Namespace):
    policy = SandboxPolicy(
        workspace=settings.workspace,
        cwd=settings.workspace,
        network_enabled=args.allow_network,
        mode=SandboxMode(args.sandbox),
    )
    return SandboxRunner().status(policy)


def _hook_load_options(settings: AppSettings) -> HookLoadOptions:
    hooks = settings.hooks
    return HookLoadOptions(
        enabled=hooks.enabled,
        managed_hooks_only=hooks.managed_hooks_only,
        load_project_hooks=hooks.load_project_hooks,
        load_user_hooks=hooks.load_user_hooks,
        trace_matches=hooks.trace_matches,
        before_timeout_ms=hooks.before_timeout_ms,
        after_timeout_ms=hooks.after_timeout_ms,
        maintenance_timeout_ms=hooks.maintenance_timeout_ms,
    )


def _emit_cli_maintenance_hook(
    hook_context: HookRuntimeContext | None,
    event_name: HookEvent,
    *,
    settings: AppSettings,
    trace_recorder: TraceRecorder,
    payload: Mapping[str, object],
    refs: tuple[str, ...] = (),
) -> HookOutcome:
    if hook_context is None:
        return HookOutcome()
    outcome = hook_context.emit(
        HookEnvelope(
            event_name=event_name,
            actor=HookActor.MAINTENANCE,
            payload=payload,
            source_refs=(
                *refs,
                f"workspace={settings.workspace}",
                *hook_context.trace_refs(),
            ),
        )
    )
    if outcome.action_results or outcome.directive != HookDirective.CONTINUE:
        trace_recorder.record(
            TraceEvent(
                kind="hook_event_evaluated",
                summary=f"Hook event evaluated: {event_name.value}.",
                refs=(
                    f"hook_event={event_name.value}",
                    f"hook_directive={outcome.directive.value}",
                    *outcome.refs,
                ),
            )
        )
    return outcome


def _record_hook_load_trace(
    trace_recorder: TraceRecorder,
    report,
    session_id: str,
) -> None:
    loaded = report.loaded_counts()
    skipped = report.skipped_counts
    refs = (
        f"session_id={session_id}",
        f"hook_surface_tag={report.registry.surface_tag()}",
        *(f"loaded_{key}={value}" for key, value in loaded.items()),
        *(f"skipped_{key}={value}" for key, value in skipped.items()),
        f"workspace_trusted={str(report.workspace_trusted).lower()}",
    )
    trace_recorder.record(
        TraceEvent(
            kind="hooks_loaded",
            summary="Runtime hook registry loaded for activation.",
            refs=refs,
        )
    )
    for diagnostic in report.diagnostics[:20]:
        trace_recorder.record(
            TraceEvent(
                kind="hook_diagnostic",
                summary=diagnostic.message,
                refs=(
                    f"level={diagnostic.level.value}",
                    *(("hook_id=" + diagnostic.hook_id,) if diagnostic.hook_id else ()),
                    *diagnostic.refs[:3],
                ),
            )
        )


def _format_mcp_tool_line(tool: McpToolRef) -> str:
    marker = _mcp_read_only_marker(tool)
    description = tool.description.strip()
    description_suffix = (
        f" - {description}"
        if description
        else f" - {tool.display_description()} [description fallback]"
    )
    return f"- {tool.qualified_name()} [{marker}]{description_suffix}"


def _mcp_read_only_marker(tool: McpToolRef) -> str:
    if tool.is_read_only():
        return "read-only"
    if "readOnlyHint" not in tool.annotations:
        return "read-only-hint-missing"
    return "not-read-only"


def _mcp_server(settings: AppSettings, server_name: str):
    name = server_name.strip()
    if not name:
        raise ValueError("--list-mcp-tools requires a server name")
    for server in settings.mcp_servers:
        if server.name == name:
            return server
    raise ValueError(f"MCP server not found: {name}")


def _default_model_context_tokens(settings: AppSettings) -> int:
    main_model = settings.model_purpose("main")
    if main_model is not None and main_model.model.strip():
        return settings.model_context_tokens(main_model.model)
    try:
        provider_settings = settings.provider()
        return settings.provider_context_tokens(
            provider_settings,
            provider_settings.primary_model(),
        )
    except ValueError:
        return settings.model_context_tokens("")


def _print_activity(
    settings: AppSettings,
    profile_name: str | None,
    today: bool,
    date: str | None,
    month: str | None,
    lines: int,
) -> None:
    if lines < 1:
        raise ValueError("--activity-lines must be at least 1")
    selected = sum(bool(value) for value in (today, date, month))
    if selected != 1:
        raise ValueError(
            "choose exactly one of --activity-today, --activity-date, or --activity-month"
        )
    profile = settings.profile_ref(profile_name)
    store = _activity_store(settings, profile.name)
    if today:
        _print_activity_file(
            store.daily_path(_local_timestamp()[:10]),
            settings.workspace,
            lines,
            missing_message="No activity note for today.",
        )
        return
    if date:
        _print_activity_file(
            store.daily_path(date),
            settings.workspace,
            lines,
            missing_message=f"No activity note for {date}.",
        )
        return
    if month:
        summary_path = store.monthly_summary_path(month)
        if summary_path.exists():
            _print_activity_file(summary_path, settings.workspace, lines)
            return
        daily_dates = store.list_daily_dates(month)
        print(
            "Monthly activity summary not found: "
            f"{_workspace_relative(summary_path, settings.workspace)}"
        )
        if not daily_dates:
            print(f"No daily activity notes found for {month}.")
            return
        print(f"Daily activity notes for {month}:")
        for daily_date in daily_dates:
            path = store.daily_path(daily_date)
            print(f"- {daily_date}: {_workspace_relative(path, settings.workspace)}")


def _format_memory_maintenance_result(result) -> str:
    """Return a compact CLI summary for one maintenance run."""
    lines = [
        "memory maintenance:",
        f"- date: {result.date or '(none)'}",
        f"- ran: {str(result.ran).lower()}",
        f"- reason: {result.reason}",
        f"- pending_processed: {result.pending_processed}",
        f"- pending_failed: {result.pending_failed}",
        f"- profile_changed: {str(result.profile_changed).lower()}",
        f"- monthly_summary_written: {str(result.monthly_summary_written).lower()}",
    ]
    if result.window_start or result.window_end:
        lines.append(f"- window_start: {result.window_start}")
        lines.append(f"- window_end: {result.window_end}")
    return "\n".join(lines)


def _install_memory_maintenance_schedule(settings: AppSettings) -> Path:
    """Install a lightweight macOS LaunchAgent for daily maintenance."""
    if sys.platform != "darwin":
        raise ValueError("memory maintenance schedule install currently supports macOS")
    profile = settings.active_profile.strip() or "default"
    label = f"com.deepmate.memory-maintenance.{profile}"
    launch_agents = Path.home() / "Library" / "LaunchAgents"
    launch_agents.mkdir(parents=True, exist_ok=True)
    logs_dir = settings.data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    src_root = Path(__file__).resolve().parents[3]
    command = (
        f"cd {shlex.quote(str(settings.workspace))} "
        f"&& PYTHONPATH={shlex.quote(str(src_root))} python3 -m deepmate "
        f"--workspace {shlex.quote(str(settings.workspace))} "
        "--run-memory-maintenance"
    )
    plist = {
        "Label": label,
        "ProgramArguments": ["/bin/zsh", "-lc", command],
        "StartCalendarInterval": {"Hour": 2, "Minute": 0},
        "StandardOutPath": str(logs_dir / "memory-maintenance.out.log"),
        "StandardErrorPath": str(logs_dir / "memory-maintenance.err.log"),
        "WorkingDirectory": str(settings.workspace),
    }
    path = launch_agents / f"{label}.plist"
    with path.open("wb") as file:
        plistlib.dump(plist, file, sort_keys=True)
    return path


def _format_memory_maintenance_schedule_install(path: Path, workspace: Path) -> str:
    relative = _workspace_relative(path, workspace)
    quoted = shlex.quote(str(path))
    return "\n".join(
        (
            f"memory maintenance schedule installed: {relative}",
            "Load it with:",
            f"  launchctl bootstrap gui/$(id -u) {quoted}",
            "Run it once now with:",
            f"  launchctl kickstart -k gui/$(id -u)/{path.stem}",
        )
    )


def _handle_rewind_command(
    settings: AppSettings,
    session_store: SessionStore,
    args: argparse.Namespace,
) -> int:
    """Preview or apply a checkpoint rewind for one session."""
    session = session_store.load(session_store.resolve_id(args.rewind))
    turn_store = TurnCheckpointStore.in_data_dir(
        settings.data_dir,
        session.profile.name,
        session.session_id,
    )
    turns = turn_store.load_turns()
    if not turns:
        print(f"No checkpoints found for session: {session.session_id}")
        return 1
    if not args.rewind_to:
        print("usage: --rewind SESSION_ID --rewind-to TURN_ID [--rewind-apply]")
        print()
        _print_checkpoint_turns(turns)
        return 2

    target = turn_store.require_turn(args.rewind_to)
    workspace_store = WorkspaceCheckpointStore.in_data_dir(
        settings.data_dir,
        session.profile.name,
        session.session_id,
    )
    include_workspace = args.rewind_mode in {"workspace", "both"}
    include_conversation = args.rewind_mode in {"conversation", "both"}
    workspace_plan = (
        workspace_store.rewind_plan(target.turn_id, session.workspace)
        if include_workspace
        else None
    )
    conversation_plan = (
        _conversation_rewind_plan(session_store, session, target)
        if include_conversation
        else None
    )
    _print_rewind_preview(
        session=session,
        target_turn_id=target.turn_id,
        mode=args.rewind_mode,
        workspace_plan=workspace_plan,
        conversation_plan=conversation_plan,
        apply=args.rewind_apply,
        force=args.rewind_force,
    )
    if not args.rewind_apply:
        print()
        print("Preview only. Add --rewind-apply to apply this rewind.")
        return 0
    if (
        workspace_plan is not None
        and workspace_plan.has_conflicts()
        and not args.rewind_force
    ):
        print()
        print(
            "Workspace rewind has conflicts. Re-run with --rewind-force to apply anyway.",
            file=sys.stderr,
        )
        return 1
    if include_workspace:
        workspace_plan = workspace_store.apply_rewind(
            target.turn_id,
            session.workspace,
            force=args.rewind_force,
        )
    if include_conversation:
        _apply_conversation_rewind(session_store, session, turn_store, target)
    session_store.touch(session.session_id)
    print()
    print(f"Rewind applied to {target.turn_id}.")
    return 0


def _print_checkpoint_turns(turns) -> None:
    print("Available checkpoints:")
    for turn in turns:
        print(
            f"- {turn.turn_id}: status={turn.status}, "
            f"resume={turn.resume_hint}, last_sequence={turn.last_transcript_sequence}"
        )


def _conversation_rewind_plan(
    session_store: SessionStore,
    session: SessionRecord,
    target,
) -> dict[str, object]:
    transcript = session_store.transcript_store(session)
    records = transcript.load_records()
    keep_sequence = target.last_transcript_sequence
    remove_count = sum(1 for record in records if record.sequence > keep_sequence)
    summary = session_store.summary_store(session).load_latest()
    summary_action = "none"
    if summary is not None:
        summary_action = (
            "delete"
            if summary.covered_until_sequence > keep_sequence
            else "keep"
        )
    return {
        "keep_sequence": keep_sequence,
        "current_records": len(records),
        "remove_records": remove_count,
        "summary_action": summary_action,
        "summary_id": summary.summary_id if summary is not None else "",
    }


def _apply_conversation_rewind(
    session_store: SessionStore,
    session: SessionRecord,
    turn_store: TurnCheckpointStore,
    target,
) -> None:
    plan = _conversation_rewind_plan(session_store, session, target)
    transcript = session_store.transcript_store(session)
    transcript.truncate_after(int(plan["keep_sequence"]))
    if plan["summary_action"] == "delete":
        session_store.summary_store(session).delete_latest()
    turn_store.set_latest(target.turn_id)


def _print_rewind_preview(
    *,
    session: SessionRecord,
    target_turn_id: str,
    mode: str,
    workspace_plan: WorkspaceRewindPlan | None,
    conversation_plan: Mapping[str, object] | None,
    apply: bool,
    force: bool,
) -> None:
    print("Checkpoint rewind")
    print(f"  session: {session.session_id}: {session.title}")
    print(f"  target_turn: {target_turn_id}")
    print(f"  mode: {mode}")
    print(f"  action: {'apply' if apply else 'preview'}")
    if force:
        print("  force: true")
    if workspace_plan is not None:
        print()
        print("Workspace")
        if not workspace_plan.actions:
            print("  No workspace file changes after target turn.")
        for action in workspace_plan.actions:
            conflict = " conflict" if action.conflict else ""
            reason = f" ({action.reason})" if action.reason else ""
            print(f"  - {action.action}: {action.path}{conflict}{reason}")
    if conversation_plan is not None:
        print()
        print("Conversation")
        print(f"  keep_until_sequence: {conversation_plan['keep_sequence']}")
        print(f"  current_records: {conversation_plan['current_records']}")
        print(f"  remove_records: {conversation_plan['remove_records']}")
        summary_action = str(conversation_plan["summary_action"])
        summary_id = str(conversation_plan["summary_id"])
        summary_suffix = f" ({summary_id})" if summary_id else ""
        print(f"  summary: {summary_action}{summary_suffix}")


def _print_activity_file(
    path: Path,
    workspace: Path,
    lines: int,
    missing_message: str = "",
) -> None:
    print(f"Activity file: {_workspace_relative(path, workspace)}")
    if lines < 1:
        print("No activity lines requested.")
        return
    if not path.exists():
        print(missing_message or "Activity file not found.")
        return
    shown: deque[str] = deque(maxlen=lines)
    total_lines = 0
    with path.open(encoding="utf-8") as file:
        for raw_line in file:
            total_lines += 1
            shown.append(raw_line.rstrip("\n"))
    limit = min(lines, total_lines)
    print()
    for line in shown:
        print(line)
    if limit < total_lines:
        print(f"... ({total_lines - limit} more lines)")


def _print_session_detail(
    session_store: SessionStore,
    session: SessionRecord,
    data_dir: Path,
    trace_path: Path,
    activity_store: ActivityStore | None = None,
    transcript_limit: int = SESSION_VIEW_TRANSCRIPT_LIMIT,
    trace_limit: int = SESSION_VIEW_TRACE_LIMIT,
    trace_kinds: Sequence[str] = (),
    export_otlp: bool = False,
    otlp_endpoint: str | None = None,
    settings: AppSettings | None = None,
) -> None:
    if transcript_limit < 1:
        raise ValueError("--session-items must be at least 1")
    if trace_limit < 0:
        raise ValueError("--trace-events must be at least 0")

    transcript_items = session_store.transcript_store(session).load_items()
    summary_record = session_store.summary_store(session).load_latest()
    turn_store = TurnCheckpointStore.in_data_dir(
        data_dir,
        session.profile.name,
        session.session_id,
    )
    workspace_store = WorkspaceCheckpointStore.in_data_dir(
        data_dir,
        session.profile.name,
        session.session_id,
    )
    latest_checkpoint = turn_store.load_latest()
    workspace_checkpoints = workspace_store.load_checkpoints()
    trace_records = _read_session_trace_records(
        trace_path,
        session.session_id,
        trace_kinds=trace_kinds,
    )

    print("Session")
    print(f"  id: {session.session_id}")
    print(f"  title: {session.title}")
    print(f"  status: {session.status}")
    print(f"  workspace: {session.workspace}")
    print(f"  profile: {session.profile.name} ({session.profile.uri})")
    print(f"  created_at: {session.created_at}")
    print(f"  updated_at: {session.updated_at}")
    print(f"  transcript: {session.transcript_path}")
    if activity_store is not None:
        print(f"  activity_note: {_session_activity_note_path(session, activity_store, trace_records)}")
    if summary_record is not None:
        print(f"  summary: {session_store.summary_path(session.session_id)}")
        print(f"  summary_id: {summary_record.summary_id}")
        print(f"  summary_covered_until_sequence: {summary_record.covered_until_sequence}")
        print(f"  summary_source_model: {summary_record.source_model}")
    if latest_checkpoint is not None:
        print(
            "  checkpoint_latest: "
            f"{latest_checkpoint.turn_id} status={latest_checkpoint.status} "
            f"resume={latest_checkpoint.resume_hint}"
        )
        print(
            "  checkpoint_sequences: "
            f"user={latest_checkpoint.user_sequence}, "
            f"last={latest_checkpoint.last_transcript_sequence}, "
            f"tool={latest_checkpoint.last_tool_exchange_sequence}, "
            f"assistant={latest_checkpoint.final_assistant_sequence}"
        )
        if latest_checkpoint.workspace_checkpoint_id:
            print(
                "  checkpoint_workspace: "
                f"{latest_checkpoint.workspace_checkpoint_id}"
            )
    if workspace_checkpoints:
        snapshot_count = sum(len(record.files) for record in workspace_checkpoints)
        print(
            "  workspace_checkpoints: "
            f"{len(workspace_checkpoints)} turn(s), {snapshot_count} file snapshot(s)"
        )
    print()

    print(
        "Transcript "
        f"(showing last {min(len(transcript_items), transcript_limit)} "
        f"of {len(transcript_items)} items)"
    )
    if not transcript_items:
        print("  No transcript items.")
    else:
        start_index = max(0, len(transcript_items) - transcript_limit)
        for index, item in enumerate(transcript_items[start_index:], start_index + 1):
            print(f"  {_format_transcript_item(index, item)}")
    print()

    trace_label = "Trace"
    clean_trace_kinds = tuple(kind.strip() for kind in trace_kinds if kind.strip())
    if clean_trace_kinds:
        trace_label += f" kinds={','.join(clean_trace_kinds)}"
    print(
        f"{trace_label} "
        f"(showing last {min(len(trace_records), trace_limit)} "
        f"matching events)"
    )
    if trace_limit == 0:
        print("  Trace display disabled.")
    elif not trace_records:
        print("  No matching trace events.")
    else:
        usage_summary = summarize_trace_usage(trace_records)
        if not usage_summary.is_empty():
            print(f"  {_format_trace_usage_summary(usage_summary)}")
        for record in trace_records[-trace_limit:]:
            print(f"  {_format_trace_record(record)}")
    if export_otlp:
        print()
        _print_otlp_export(
            trace_records,
            endpoint=otlp_endpoint,
            settings=settings,
        )


def _format_transcript_item(index: int, item: ModelConversationItem) -> str:
    message = item.message
    if message is not None:
        return (
            f"{index:04d} {message.role.value}: "
            f"{_preview_text(message.content)}"
        )
    exchange = item.tool_exchange
    if exchange is None:
        return f"{index:04d} empty"
    tool_names = ", ".join(
        _text_value(getattr(request, "name", ""))
        for request in _iter_values(exchange.tool_requests)
        if _text_value(getattr(request, "name", ""))
    )
    result_refs = tuple(
        ref
        for result in _iter_values(exchange.tool_results)
        for ref in _iter_values(getattr(result, "refs", ()))
        if str(ref).strip()
    )
    details = [
        f"tools={tool_names or '-'}",
        f"results={len(_iter_values(exchange.tool_results))}",
    ]
    if result_refs:
        details.append(f"refs={', '.join(result_refs[:3])}")
    assistant_text = _text_value(exchange.assistant_content) or _text_value(
        exchange.assistant_reasoning
    )
    if assistant_text:
        details.append(f"assistant={_preview_text(assistant_text)}")
    return f"{index:04d} tool_exchange: " + "; ".join(details)


def _read_session_trace_records(
    trace_path: Path,
    session_id: str,
    trace_kinds: Sequence[str] = (),
) -> tuple[Mapping[str, object], ...]:
    if not trace_path.exists():
        return ()
    records: list[Mapping[str, object]] = []
    with trace_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            if trace_record_matches_session(
                record,
                session_id,
            ) and trace_record_matches_kinds(record, trace_kinds):
                records.append(record)
    return tuple(records)


def _session_activity_note_path(
    session: SessionRecord,
    activity_store: ActivityStore,
    trace_records: Sequence[Mapping[str, object]],
) -> str:
    for record in reversed(trace_records):
        if str(record.get("kind", "")).strip() != "activity_daily_note_written":
            continue
        refs = trace_refs_to_map(trace_record_refs(record))
        path = str(refs.get("activity_path", "")).strip()
        if path:
            return path
    return _workspace_relative(activity_store.daily_path(session.updated_at[:10]), session.workspace)


def _format_trace_record(record: Mapping[str, object]) -> str:
    if str(record.get("record_type", "")).strip() == "span":
        return _format_trace_span_record(record)
    recorded_at = str(record.get("recorded_at", "")).strip()
    kind = str(record.get("kind", "")).strip()
    summary = str(record.get("summary", "")).strip()
    refs = trace_refs_to_map(trace_record_refs(record))

    if kind == "model_response_received":
        return _format_model_response_trace(recorded_at, kind, refs)
    if kind == "conversation_budget_report":
        return _format_budget_trace(recorded_at, kind, refs)
    if kind == "request_budget_report":
        return _format_request_budget_trace(recorded_at, kind, refs)
    if kind == "model_request_prefix_fingerprint":
        return _format_prefix_fingerprint_trace(recorded_at, kind, refs)
    if kind == "session_summary_policy":
        return _format_session_summary_policy_trace(recorded_at, kind, refs)
    if kind.startswith("session_summary"):
        return _format_session_summary_trace(recorded_at, kind, refs, summary)
    if kind.startswith("memory_extraction"):
        return _format_memory_trace(recorded_at, kind, refs, summary)
    if kind.startswith("subagent_"):
        return _format_subagent_trace(recorded_at, kind, refs, summary)

    compact_refs = _compact_trace_refs(refs)
    suffix = f" ({compact_refs})" if compact_refs else ""
    return f"{recorded_at} {kind}: {summary}{suffix}"


def _print_otlp_export(
    trace_records: Sequence[Mapping[str, object]],
    *,
    endpoint: str | None,
    settings: AppSettings | None,
) -> None:
    configured = settings.otlp_traces if settings is not None else None
    clean_endpoint = (endpoint or (configured.endpoint if configured else "")).strip()
    if not clean_endpoint:
        raise ValueError(
            "OTLP endpoint is not configured. Set observability.otlp.endpoint "
            "or pass --otlp-endpoint."
        )
    spans = tuple(_span_from_trace_record(record) for record in trace_records)
    spans = tuple(span for span in spans if span is not None)
    headers = configured.headers if configured is not None else ()
    service_name = configured.service_name if configured is not None else "deepmate"
    service_version = configured.service_version if configured is not None else ""
    result = export_otlp_traces(
        spans,
        endpoint=clean_endpoint,
        headers=headers,
        service_name=service_name,
        service_version=service_version,
    )
    if result.is_success():
        print(
            "OTLP export completed: "
            f"spans={result.spans_exported}/{result.spans_seen}, "
            f"endpoint={result.endpoint}, status={result.status_code}"
        )
        return
    if result.spans_exported == 0:
        print(f"OTLP export skipped: {result.message}")
        return
    raise ValueError(
        "OTLP export failed: "
        f"endpoint={result.endpoint}, status={result.status_code}, message={result.message}"
    )


def _validate_otlp_export(
    *,
    settings: AppSettings,
    endpoint: str | None = None,
) -> OtlpExportResult:
    """Send a synthetic trace that exercises Deepmate's OTLP span shape."""
    configured = settings.otlp_traces
    clean_endpoint = (endpoint or configured.endpoint).strip()
    if not clean_endpoint:
        raise ValueError(
            "OTLP endpoint is not configured. Set observability.otlp.endpoint "
            "or pass --otlp-endpoint."
        )
    spans = _synthetic_otlp_validation_spans(settings)
    return export_otlp_traces(
        spans,
        endpoint=clean_endpoint,
        headers=configured.headers,
        service_name=configured.service_name,
        service_version=configured.service_version,
    )


def _synthetic_otlp_validation_spans(settings: AppSettings) -> tuple[TraceSpan, ...]:
    trace_id = new_trace_id()
    root_span_id = new_span_id()
    model_span_id = new_span_id()
    tool_span_id = new_span_id()
    started_at = time_ns()
    session_id = "deepmate-otlp-validation"
    model = settings.provider().primary_model()
    root = TraceSpan(
        name="deepmate otlp validation",
        kind="INTERNAL",
        trace_id=trace_id,
        span_id=root_span_id,
        started_at_unix_nano=started_at,
        ended_at_unix_nano=started_at + 30_000_000,
        status="OK",
        attributes={
            "session.id": session_id,
            "gen_ai.conversation.id": session_id,
            "gen_ai.operation.name": "invoke_agent",
            "langfuse.trace.name": "Deepmate OTLP validation",
            "deepmate.validation.kind": "otlp",
            "deepmate.workspace": str(settings.workspace),
        },
    )
    model_span = TraceSpan(
        name=f"chat {model}",
        kind="CLIENT",
        trace_id=trace_id,
        span_id=model_span_id,
        parent_span_id=root_span_id,
        started_at_unix_nano=started_at + 1_000_000,
        ended_at_unix_nano=started_at + 20_000_000,
        status="OK",
        attributes={
            "session.id": session_id,
            "gen_ai.conversation.id": session_id,
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": settings.default_provider,
            "gen_ai.request.model": model,
            "gen_ai.response.model": model,
            "gen_ai.usage.input_tokens": 12,
            "gen_ai.usage.output_tokens": 4,
            "deepmate.validation.kind": "otlp",
        },
    )
    tool_span = TraceSpan(
        name="execute_tool otlp_validation_tool",
        kind="INTERNAL",
        trace_id=trace_id,
        span_id=tool_span_id,
        parent_span_id=root_span_id,
        started_at_unix_nano=started_at + 21_000_000,
        ended_at_unix_nano=started_at + 29_000_000,
        status="OK",
        attributes={
            "session.id": session_id,
            "gen_ai.conversation.id": session_id,
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "otlp_validation_tool",
            "deepmate.tool.source": "synthetic",
            "deepmate.validation.kind": "otlp",
        },
    )
    return (root, model_span, tool_span)


def _span_from_trace_record(record: Mapping[str, object]) -> TraceSpan | None:
    if record.get("record_type") != "span":
        return None
    try:
        return TraceSpan(
            name=_text_value(record.get("name")),
            kind=_text_value(record.get("kind")),
            trace_id=_text_value(record.get("trace_id")),
            span_id=_text_value(record.get("span_id")),
            parent_span_id=_text_value(record.get("parent_span_id")),
            started_at_unix_nano=_int_object(record.get("started_at_unix_nano")),
            ended_at_unix_nano=_int_object(record.get("ended_at_unix_nano")),
            status=_text_value(record.get("status")) or "UNSET",
            attributes=_mapping_value(record.get("attributes")),
        )
    except (TypeError, ValueError):
        return None


def _format_trace_span_record(record: Mapping[str, object]) -> str:
    name = str(record.get("name", "")).strip() or "span"
    kind = str(record.get("kind", "")).strip() or "INTERNAL"
    status = str(record.get("status", "")).strip() or "UNSET"
    attributes = record.get("attributes")
    attrs = attributes if isinstance(attributes, Mapping) else {}
    duration = _span_duration(record)
    details = [
        f"kind={kind}",
        f"status={status}",
        f"duration={duration}",
    ]
    model = _text_value(attrs.get("gen_ai.response.model")) or _text_value(
        attrs.get("gen_ai.request.model")
    )
    if model:
        details.append(f"model={model}")
    input_tokens = _int_object(attrs.get("gen_ai.usage.input_tokens"))
    output_tokens = _int_object(attrs.get("gen_ai.usage.output_tokens"))
    if input_tokens > 0 or output_tokens > 0:
        details.append(f"input={input_tokens}")
        details.append(f"output={output_tokens}")
    cache_hit = _int_object(
        attrs.get("gen_ai.usage.cache_read.input_tokens")
    ) or _int_object(attrs.get("deepmate.usage.cache_hit_input_tokens"))
    cache_miss = _int_object(attrs.get("deepmate.usage.cache_miss_input_tokens"))
    if cache_hit > 0 or cache_miss > 0:
        total = cache_hit + cache_miss
        cache = f"cache_hit={cache_hit}, cache_miss={cache_miss}"
        if total > 0:
            cache += f", cache_hit_ratio={cache_hit / total:.0%}"
        details.append(cache)
    tool_name = _text_value(attrs.get("gen_ai.tool.name")) or _text_value(
        attrs.get("deepmate.tool.name")
    )
    if tool_name:
        details.append(f"tool={tool_name}")
    error_type = _text_value(attrs.get("error.type"))
    if error_type:
        details.append(f"error={error_type}")
    return f"span {name}: " + ", ".join(details)


def _span_duration(record: Mapping[str, object]) -> str:
    start = _int_object(record.get("started_at_unix_nano"))
    end = _int_object(record.get("ended_at_unix_nano"))
    if start <= 0 or end <= 0 or end < start:
        return "-"
    seconds = (end - start) / 1_000_000_000
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.1f}s"


def _format_trace_usage_summary(summary) -> str:
    cache_ratio = summary.cache_hit_ratio()
    cache_part = ""
    if cache_ratio is not None:
        cache_part = f", cache_hit_ratio={cache_ratio:.0%}"
    return (
        "usage: "
        f"model_responses={summary.model_response_events}, "
        f"input={summary.input_tokens}, "
        f"output={summary.output_tokens}, "
        f"reasoning={summary.reasoning_tokens}, "
        f"cache_hit={summary.cache_hit_input_tokens}, "
        f"cache_miss={summary.cache_miss_input_tokens}"
        f"{cache_part}"
    )


def _format_model_response_trace(
    recorded_at: str,
    kind: str,
    refs: Mapping[str, str],
) -> str:
    input_tokens = _int_ref(refs, "input_tokens")
    cache_hit = _int_ref(refs, "cache_hit_input_tokens")
    cache_miss = _int_ref(refs, "cache_miss_input_tokens")
    cache_ratio = ""
    if input_tokens > 0:
        cache_ratio = f", cache_hit_ratio={cache_hit / input_tokens:.0%}"
    return (
        f"{recorded_at} {kind}: "
        f"model={refs.get('model', '-')}, step={refs.get('step', refs.get('turn', '-'))}, "
        f"input={input_tokens}, output={refs.get('output_tokens', '0')}, "
        f"cache_hit={cache_hit}, cache_miss={cache_miss}{cache_ratio}, "
        f"reasoning={refs.get('reasoning_tokens', '0')}, "
        f"tools={refs.get('tool_requests', '0')}, "
        f"finish={refs.get('finish_reason', '-')}"
    )


def _format_budget_trace(
    recorded_at: str,
    kind: str,
    refs: Mapping[str, str],
) -> str:
    return (
        f"{recorded_at} {kind}: "
        f"items={refs.get('conversation_items', '0')}, "
        f"estimated={refs.get('estimated_history_tokens', '0')}, "
        f"budget={refs.get('history_token_budget', '0')}, "
        f"over_budget={refs.get('over_budget', 'false')}, "
        f"trimmed={refs.get('trimmed', 'false')}"
    )


def _format_request_budget_trace(
    recorded_at: str,
    kind: str,
    refs: Mapping[str, str],
) -> str:
    return (
        f"{recorded_at} {kind}: "
        f"estimated_input={refs.get('estimated_input_tokens', '0')}, "
        f"usable_input={refs.get('usable_input_tokens', '0')}, "
        f"pressure={refs.get('pressure_ratio', '0')}, "
        f"tool_output_ratio={refs.get('tool_output_ratio', '0')}"
    )


def _format_prefix_fingerprint_trace(
    recorded_at: str,
    kind: str,
    refs: Mapping[str, str],
) -> str:
    return (
        f"{recorded_at} {kind}: "
        f"step={refs.get('step', '-')}, "
        f"model={refs.get('model', '-')}, "
        f"prefix={refs.get('prefix_digest', '-')}, "
        f"system={refs.get('system_digest', '-')}, "
        f"tool_schemas={refs.get('tool_schema_count', '0')}, "
        f"options={refs.get('options_digest', '-')}"
    )


def _format_session_summary_policy_trace(
    recorded_at: str,
    kind: str,
    refs: Mapping[str, str],
) -> str:
    return (
        f"{recorded_at} {kind}: "
        f"action={refs.get('summary_action', '-')}, "
        f"reason={refs.get('summary_reason', '-')}, "
        f"checkpoint={refs.get('summary_should_checkpoint', 'false')}, "
        f"estimated_input={refs.get('estimated_input_tokens', '0')}"
    )


def _format_session_summary_trace(
    recorded_at: str,
    kind: str,
    refs: Mapping[str, str],
    summary: str,
) -> str:
    details = [
        f"summary_id={refs.get('summary_id', '-')}",
        f"covered_until={refs.get('covered_until_sequence', '-')}",
        f"source_items={refs.get('source_item_count', refs.get('source_items', '-'))}",
        f"reason={refs.get('summary_reason', '-')}",
    ]
    compact = ", ".join(item for item in details if not item.endswith("=-"))
    suffix = f" ({compact})" if compact else ""
    return f"{recorded_at} {kind}: {summary}{suffix}"


def _format_memory_trace(
    recorded_at: str,
    kind: str,
    refs: Mapping[str, str],
    summary: str,
) -> str:
    compact_refs = _compact_trace_refs(refs)
    suffix = f" ({compact_refs})" if compact_refs else ""
    return f"{recorded_at} {kind}: {summary}{suffix}"


def _format_subagent_trace(
    recorded_at: str,
    kind: str,
    refs: Mapping[str, str],
    summary: str,
) -> str:
    details = [
        f"run={refs.get('subagent_run_id', '-')}",
        f"assignment={refs.get('assignment_id', '-')}",
        f"attempt={refs.get('attempt', '-')}",
        f"stage={refs.get('stage', '-')}",
        f"status={refs.get('status', '-')}",
        f"review={refs.get('review_status', '-')}",
        f"max_steps={refs.get('max_steps', '-')}",
        f"child_runs={refs.get('child_runs', '-')}",
        f"accepted={refs.get('accepted_results', '-')}",
        f"blocking_gaps={refs.get('blocking_gaps', '-')}",
        f"revised={refs.get('revised', '-')}",
        f"artifacts={refs.get('artifact_refs', '0')}",
        f"evidence={refs.get('evidence_refs', '0')}",
    ]
    if "assignments" in refs:
        details.append(f"assignments={refs['assignments']}")
    if "max_child_runs" in refs:
        details.append(f"max_child_runs={refs['max_child_runs']}")
    if "tool_access_mode" in refs:
        details.append(f"tools={refs['tool_access_mode']}")
    if "model_purpose" in refs:
        details.append(f"model_purpose={refs['model_purpose']}")
    compact = ", ".join(item for item in details if not item.endswith("=-"))
    return f"{recorded_at} {kind}: {summary} ({compact})"


def _compact_trace_refs(refs: Mapping[str, str]) -> str:
    hidden_keys = {
        "activation_id",
        "parent_activation_id",
        "session_id",
        "parent_session_id",
        "current_session_id",
    }
    visible = [
        f"{key}={value}"
        for key, value in refs.items()
        if key not in hidden_keys
    ]
    return ", ".join(visible[:8])


def _int_ref(refs: Mapping[str, str], key: str) -> int:
    try:
        return int(refs.get(key, "0"))
    except (TypeError, ValueError):
        return 0


def _int_object(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return 0


def _preview_text(value: str, limit: int = SESSION_VIEW_PREVIEW_CHARS) -> str:
    text = " ".join(_text_value(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _text_value(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _mapping_value(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items() if str(key).strip()}


def _iter_values(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return ()


def _checkpoint_controller(
    settings: AppSettings,
    session: SessionRecord,
) -> SessionCheckpointController:
    return SessionCheckpointController.in_data_dir(
        settings.data_dir,
        workspace=session.workspace,
        profile=session.profile.name,
        session_id=session.session_id,
    )


def _load_latest_summary(
    session_store: SessionStore,
    session: SessionRecord,
):
    try:
        return session_store.summary_store(session).load_latest()
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _load_or_create_session(
    session_store: SessionStore,
    settings: AppSettings,
    profile_name: str | None,
    session_id: str | None,
    title: str | None,
    prompt: str,
    interactive: bool,
) -> SessionRecord:
    if session_id:
        return session_store.load(session_store.resolve_id(session_id))
    if interactive and not prompt.strip() and not (title or "").strip():
        latest = session_store.latest_for_workspace(settings.workspace)
        if latest is not None:
            _print_status(
                f"interactive session resumed: {latest.session_id} ({latest.title})"
            )
            return latest
    session = session_store.create(
        workspace=settings.workspace,
        profile=settings.profile_ref(profile_name),
        title=title or _title_from_prompt(prompt),
    )
    mode = "interactive session" if interactive else "session"
    _print_status(f"{mode} created: {session.session_id} ({session.title})")
    return session


def _activity_store(settings: AppSettings, profile_name: str) -> ActivityStore:
    clean_name = profile_name.strip() or settings.active_profile
    return ActivityStore(settings.data_dir / "activity" / clean_name)


def _local_timestamp() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def _workspace_relative(path: Path, workspace: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace.resolve()))
    except ValueError:
        return str(path)


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        resolved = path.resolve()
        base = root.resolve()
    except OSError:
        return False
    return resolved == base or base in resolved.parents


def _session_end_status(result, review: DeliveryReview) -> str:
    if result.loop_guard_stop is not None:
        return result.loop_guard_stop.reason.value
    if result.has_errors():
        return "failed"
    if result.reached_max_steps:
        return "max_steps_reached"
    if review.status == DeliveryReviewStatus.BLOCKED:
        return "blocked"
    return "completed"


def _skill_catalog(
    workspace: Path,
    data_dir: Path | None = None,
    capability_state_store: CapabilityStateStore | None = None,
) -> tuple[SkillCatalog | None, tuple[ContextWarning, ...]]:
    catalog, warnings = workspace_skill_catalog(workspace, data_dir=data_dir)
    if catalog is not None and capability_state_store is not None:
        workspace_cards, _workspace_warnings = discover_workspace_skill_cards(workspace)
        capability_state_store.sync_workspace_skills(workspace_cards, workspace)
    return catalog, warnings


def _attach_skill_loader_tools(
    *,
    native_tools: NativeToolRegistry | None,
    skill_catalog: SkillCatalog | None,
    workspace: Path,
    capability_state_store: CapabilityStateStore | None = None,
    data_dir: Path | None = None,
    dynamic: bool = False,
) -> NativeToolRegistry | None:
    """Attach the on-demand skill loader to this run's native tool registry."""
    catalog_provider: Callable[[], SkillCatalog | None] | None = None
    if dynamic:
        catalog_provider = lambda: _skill_catalog(
            workspace,
            data_dir,
            capability_state_store=capability_state_store,
        )[0]
    skill_tools = skill_loader_tools(
        skill_catalog,
        workspace,
        capability_state_store,
        catalog_provider=catalog_provider,
    )
    if not skill_tools:
        return native_tools
    registry = native_tools or NativeToolRegistry()
    for tool in skill_tools:
        registry.register(tool)
    return registry


def _build_cli_native_tools(
    *,
    settings: AppSettings,
    expose_read_tools: bool,
    register_write_tools: bool,
    expose_network_tools: bool,
    register_shell_tools: bool,
    behavior_runtime=None,
    expose_computer_tools: bool = False,
    shell_enabled: bool,
    network_enabled: bool,
    env_change_enabled: bool,
    sandbox_mode: SandboxMode,
    approval_cache: SessionApprovalCache | None,
    checkpoint_write_router: SessionCheckpointWriteRouter,
    hook_context: HookRuntimeContext,
) -> NativeToolRegistry | None:
    if not (
        expose_read_tools
        or register_write_tools
        or register_shell_tools
        or expose_network_tools
        or expose_computer_tools
    ):
        return None
    return NativeToolRegistry(
        (
            *(
                workspace_filesystem_tools(
                    settings.workspace,
                    include_write_tools=register_write_tools,
                    write_checkpoint=checkpoint_write_router.capture_workspace_write,
                    hook_context=hook_context,
                )
                if expose_read_tools or register_write_tools
                else ()
            ),
            *(workspace_search_tools(settings.workspace) if expose_read_tools else ()),
            *(workspace_lsp_tools(settings.workspace) if expose_read_tools else ()),
            *(workspace_document_tools(settings.workspace) if expose_read_tools else ()),
            *(workspace_artifact_tools(settings.workspace) if expose_read_tools else ()),
            *(
                workspace_report_tools(
                    settings.workspace,
                    write_checkpoint=checkpoint_write_router.capture_workspace_write,
                    hook_context=hook_context,
                )
                if register_write_tools
                else ()
            ),
            *(
                workspace_diagram_tools(
                    settings.workspace,
                    write_checkpoint=checkpoint_write_router.capture_workspace_write,
                    hook_context=hook_context,
                )
                if register_write_tools
                else ()
            ),
            *web_research_tools(network_enabled=expose_network_tools),
            *(
                computer_tools(
                    data_dir=settings.data_dir,
                    workspace=settings.workspace,
                    session_id=getattr(behavior_runtime, "session_id", ""),
                    state=ComputerUseState(
                        enabled=lambda: bool(
                            behavior_runtime is not None
                            and behavior_runtime.computer_use_enabled
                        ),
                        computer_learning_enabled=lambda: bool(
                            behavior_runtime is not None
                            and behavior_runtime.settings.computer_learning_enabled
                        ),
                    ),
                    exposed_by_default=expose_computer_tools,
                )
                if behavior_runtime is not None
                and (expose_read_tools or expose_computer_tools)
                else ()
            ),
            *(
                shell_tools(
                    settings.workspace,
                    shell_enabled=shell_enabled,
                    network_enabled=network_enabled,
                    env_change_enabled=env_change_enabled,
                    sandbox_mode=sandbox_mode,
                    approval_cache=approval_cache,
                    hook_context=hook_context,
                )
                if register_shell_tools
                else ()
            ),
        )
    )


def _replace_skill_loader_tools(
    registry: NativeToolRegistry | None,
    *,
    skill_catalog: SkillCatalog | None,
    workspace: Path,
    capability_state_store: CapabilityStateStore | None,
    data_dir: Path | None = None,
    dynamic: bool = False,
) -> NativeToolRegistry | None:
    existing = tuple(
        tool
        for tool in (registry.list_tools() if registry is not None else ())
        if tool.name != "load_skill"
    )
    refreshed = NativeToolRegistry(existing)
    return _attach_skill_loader_tools(
        native_tools=refreshed,
        skill_catalog=skill_catalog,
        workspace=workspace,
        capability_state_store=capability_state_store,
        data_dir=data_dir,
        dynamic=dynamic,
    )


def _refresh_tui_skill_surface(
    state,
    *,
    model_context_tokens: int,
    mcp_catalog: McpToolCatalog | None,
    mcp_tools: Sequence[McpToolRef],
) -> None:
    catalog, warnings = _skill_catalog(
        state.workspace,
        state.data_dir,
        capability_state_store=state.capability_state_store,
    )
    for warning in warnings:
        if state.warning_sink is not None:
            state.warning_sink(warning)
    state.native_tools = _replace_skill_loader_tools(
        state.native_tools,
        skill_catalog=catalog,
        workspace=state.workspace,
        capability_state_store=state.capability_state_store,
        data_dir=state.data_dir,
        dynamic=True,
    )
    state.tool_schemas = _default_tool_schemas_for_model(state.native_tools, state.model)
    state.capability_surface, _warnings = _capability_surface(
        catalog,
        state.tool_schemas,
        mcp_tools,
        capability_state_store=state.capability_state_store,
        mcp_catalog=mcp_catalog,
        model_context_tokens=model_context_tokens,
    )


def _state_model_context_tokens(settings: AppSettings, state) -> int:
    policy = getattr(state, "conversation_budget_policy", None)
    if policy is not None:
        return max(1, int(policy.model_context_tokens))
    return settings.model_context_tokens(getattr(state, "model", ""))


def _attach_skill_installer_tools(
    *,
    native_tools: NativeToolRegistry | None,
    prompt: str,
    workspace: Path,
    data_dir: Path,
    capability_state_store: CapabilityStateStore,
    shell_enabled: bool = False,
    network_enabled: bool = False,
    env_change_enabled: bool = False,
    sandbox_mode: SandboxMode = SandboxMode.AUTO,
    approval_cache: SessionApprovalCache | None = None,
) -> NativeToolRegistry | None:
    """Attach a stable skill-install loader while keeping concrete schemas on demand."""
    registry = native_tools or NativeToolRegistry()
    for tool in skill_installer_tools(
        workspace,
        data_dir,
        capability_state_store,
        shell_enabled=shell_enabled,
        network_enabled=network_enabled,
        env_change_enabled=env_change_enabled,
        sandbox_mode=sandbox_mode,
        approval_cache=approval_cache,
    ):
        registry.register(tool)
    return registry


def _hide_native_tool_schemas(
    registry: NativeToolRegistry,
    names: Sequence[str],
) -> NativeToolRegistry:
    """Keep tools executable while removing them from the default prompt surface."""
    hidden = {name.strip() for name in names if name.strip()}
    return NativeToolRegistry(
        replace(tool, exposed_by_default=False)
        if tool.name.strip() in hidden
        else tool
        for tool in registry.list_tools()
    )


def _default_tool_schemas_for_model(
    registry: NativeToolRegistry | None,
    model: str,
) -> tuple[Mapping[str, object], ...]:
    if registry is None:
        return ()
    if local_model_by_runtime_name(model) is None:
        return registry.schemas()
    return tuple(
        tool.schema()
        for tool in registry.list_tools()
        if tool.exposed_by_default and tool.name not in _local_hidden_tool_schema_names()
    )


def _local_hidden_tool_schema_names() -> set[str]:
    return {
        "lsp_definition",
        "lsp_references",
        "lsp_hover",
        "read_document",
        "inspect_table",
        "review_artifact",
        "render_html_report",
        "render_tech_diagram",
    }


def _schemas_with_local_prompt_extras(
    schemas: Sequence[Mapping[str, object]],
    registry: NativeToolRegistry | None,
    prompt: str,
) -> tuple[Mapping[str, object], ...]:
    if registry is None:
        return tuple(schemas)
    selected = list(schemas)
    existing = {
        str(schema.get("name", "")).strip()
        for schema in selected
        if hasattr(schema, "get")
    }
    for name in _local_extra_schema_names_for_prompt(prompt):
        if name in existing:
            continue
        tool = registry.get(name)
        if tool is None:
            continue
        selected.append(tool.schema())
        existing.add(name)
    return tuple(selected)


def _local_extra_schema_names_for_prompt(prompt: str) -> tuple[str, ...]:
    clean = prompt.lower()
    names: list[str] = []
    if _text_has_any(
        clean,
        (
            "定义",
            "引用",
            "调用链",
            "谁调用",
            "类型",
            "签名",
            "definition",
            "references",
            "hover",
        ),
    ):
        names.extend(("lsp_definition", "lsp_references", "lsp_hover"))
    if _text_has_any(
        clean,
        ("文档", "表格", ".docx", ".xlsx", ".pdf", ".csv", "document", "excel"),
    ):
        names.extend(("read_document", "inspect_table"))
    if _text_has_any(clean, ("检查交付", "验收", "review_artifact")):
        names.append("review_artifact")
    if _text_has_any(clean, ("报告", "html report", "render_html_report", "生成html")):
        names.append("render_html_report")
    if _text_has_any(clean, ("架构图", "流程图", "时序图", "diagram", "图表")):
        names.append("render_tech_diagram")
    return tuple(dict.fromkeys(names))


def _text_has_any(text: str, markers: Sequence[str]) -> bool:
    return any(marker in text for marker in markers)


def _attach_mcp_loader_tools(
    *,
    native_tools: NativeToolRegistry | None,
    mcp_catalog: McpToolCatalog | None,
) -> NativeToolRegistry | None:
    """Attach MCP search/load tools to this run's native tool registry."""
    loader_tools = mcp_loader_tools(mcp_catalog)
    if not loader_tools:
        return native_tools
    registry = native_tools or NativeToolRegistry()
    for tool in loader_tools:
        registry.register(tool)
    return registry


def _attach_browser_tools(
    *,
    native_tools: NativeToolRegistry | None,
    backend: AgentBrowserBackend,
    preload: bool = False,
    extra_schema_loader=None,
    approval_cache: SessionApprovalCache | None = None,
) -> NativeToolRegistry:
    """Attach built-in browser tools to this run's native tool registry."""
    registry = native_tools or NativeToolRegistry()
    concrete_tools = browser_tools(backend)
    if preload:
        tools = concrete_tools
    else:
        tools = (
            *browser_loader_tools(
                backend,
                load_tools=lambda: concrete_tools,
                extra_schema_loader=extra_schema_loader,
                approval_cache=approval_cache,
            ),
            *(
                replace(tool, exposed_by_default=False)
                for tool in concrete_tools
            ),
        )
    for tool in tools:
        registry.register(tool)
    return registry


def _only_browser_loader_is_exposed(native_tools: NativeToolRegistry) -> bool:
    schemas = native_tools.schemas()
    return bool(schemas) and tuple(schema["name"] for schema in schemas) == (
        "load_browser_tools",
    )


def _attach_tool_output_tools(
    *,
    native_tools: NativeToolRegistry | None,
    store: ToolOutputStore,
    enabled: bool,
    exposed_by_default: bool = True,
) -> NativeToolRegistry | None:
    """Attach session-scoped tool output retrieval when tool outputs may exist."""
    if not enabled:
        return native_tools
    registry = native_tools or NativeToolRegistry()
    for tool in tool_output_tools(store):
        registry.register(
            tool if exposed_by_default else replace(tool, exposed_by_default=False)
        )
    return registry


def _capability_surface(
    skill_catalog: SkillCatalog | None,
    native_tool_schemas: Sequence[Mapping[str, object]],
    mcp_tools: Sequence[McpToolRef] = (),
    capability_state_store: CapabilityStateStore | None = None,
    mcp_catalog: McpToolCatalog | None = None,
    model_context_tokens: int = 0,
) -> tuple[CapabilitySurface | None, tuple[ContextWarning, ...]]:
    surfaces: list[CapabilitySurface] = []
    if skill_catalog is not None:
        states = (
            capability_state_store.skill_states_by_name()
            if capability_state_store is not None
            else None
        )
        skill_surface = from_skill_cards(skill_catalog.list_cards(), states)
        if not skill_surface.is_empty():
            surfaces.append(skill_surface)

    if native_tool_schemas:
        surfaces.append(from_native_tool_schemas(native_tool_schemas))

    if mcp_catalog is not None:
        mcp_surface = from_mcp_tool_catalog(
            mcp_catalog,
            max(1, model_context_tokens),
        )
        if not mcp_surface.is_empty():
            surfaces.append(mcp_surface)
    elif mcp_tools:
        surfaces.append(from_mcp_tool_refs(mcp_tools))

    if not surfaces:
        return None, ()
    return combine_surfaces(surfaces), ()


def _subagent_executor(
    provider: ChatCompletionsProvider,
    settings: AppSettings,
    profile,
    model: str,
    capability_surface: CapabilitySurface | None,
    native_tools: NativeToolRegistry | None,
    mcp_executor: McpToolExecutor | None,
    tool_schemas: Sequence[Mapping[str, object]],
    selected_skills,
    activation,
    provider_retry_policy: ProviderRetryPolicy,
    options: Mapping[str, object],
    trace_recorder: TraceRecorder,
    tool_access_mode: ToolAccessMode,
    provider_settings: ProviderSettings | None = None,
    hook_context: HookRuntimeContext | None = None,
    result_store: SubagentResultStore | None = None,
) -> SubagentToolExecutor:
    subagent_model = _resolve_subagent_model(
        settings,
        model,
        options,
        provider_settings=provider_settings,
    )
    allowed_tools = tuple(
        str(schema.get("name", "")).strip()
        for schema in tool_schemas
        if str(schema.get("name", "")).strip()
    )

    def runtime_factory() -> SubagentRuntime:
        return SubagentRuntime(
            provider=provider,
            workspace=settings.workspace,
            profile=profile,
            model=subagent_model.model,
            capability_surface=capability_surface,
            native_tools=native_tools,
            mcp_tools=mcp_executor,
            tool_schemas=tuple(tool_schemas),
            selected_skill_documents=(),
            parent_tool_access_policy=ToolAccessPolicy(mode=tool_access_mode),
            activation=activation,
            conversation_budget_policy=_conversation_budget_policy(
                settings,
                subagent_model.model,
                provider_settings=provider_settings,
            ),
            provider_retry_policy=provider_retry_policy,
            tool_repair_policy=_tool_repair_policy(settings),
            options=subagent_model.options,
            model_capabilities=_provider_model_capabilities(
                settings,
                provider_settings,
                subagent_model.model,
            ),
            trace_recorder=trace_recorder,
        )

    return SubagentToolExecutor(
        runtime_factory=runtime_factory,
        default_allowed_tools=allowed_tools,
        parent_tool_access_mode=tool_access_mode,
        workflow_policy=SubagentOrchestrationPolicy(
            max_child_runs=settings.subagents.max_child_runs,
            max_workspace_write_child_runs=(
                settings.subagents.max_workspace_write_child_runs
            ),
            max_revise_attempts=settings.subagents.max_revise_attempts,
            max_child_steps=settings.subagents.max_child_steps,
            revise_step_extension=settings.subagents.revise_step_extension,
        ),
        hook_context=hook_context,
        result_store=result_store,
    )


def _bind_subagent_executor(
    subagents: SubagentToolExecutor | None,
    *,
    capability_surface: CapabilitySurface | None,
    native_tools: NativeToolRegistry | None,
    mcp_tools: McpToolExecutor | None,
    tool_schemas: Sequence[Mapping[str, object]],
    selected_skill_documents,
    activation,
    tool_access_policy: ToolAccessPolicy | None,
    result_store: SubagentResultStore | None = None,
) -> SubagentToolExecutor | None:
    if subagents is None:
        return None
    return subagents.bind_runtime(
        capability_surface=capability_surface,
        native_tools=native_tools,
        mcp_tools=mcp_tools,
        tool_schemas=tool_schemas,
        selected_skill_documents=tuple(selected_skill_documents),
        activation=activation,
        parent_tool_access_policy=tool_access_policy,
        result_store=result_store,
    )


def _resolve_subagent_model(
    settings: AppSettings,
    fallback_model: str,
    parent_options: Mapping[str, object],
    *,
    provider_settings: ProviderSettings | None = None,
) -> ModelCallConfig:
    if local_model_by_runtime_name(fallback_model) is not None:
        return ModelCallConfig(model=fallback_model, options=dict(parent_options))
    configured = (
        settings.model_purpose("subagent_worker")
        if _provider_has_dedicated_internal_models(provider_settings)
        else None
    )
    if configured is None:
        return resolve_model_purpose(
            settings,
            "subagent_worker",
            fallback_model,
            option_overrides=parent_options,
        )
    return resolve_model_purpose(
        settings,
        "subagent_worker",
        fallback_model,
    )


def _provider_has_dedicated_internal_models(
    provider_settings: ProviderSettings | None,
) -> bool:
    if provider_settings is None:
        return True
    return provider_settings.name in {"deepseek", LOCAL_PROVIDER_NAME}


def _provider_model_context_tokens(
    settings: AppSettings,
    provider_settings: ProviderSettings | None,
    model: str,
    *,
    allow_missing_context_window: bool = False,
) -> int:
    local_preset = local_model_by_runtime_name(model) or local_model_by_id(model)
    if local_preset is not None:
        return local_preset.effective_context_tokens
    if provider_settings is None:
        return settings.model_context_tokens(model)
    try:
        return settings.provider_context_tokens(provider_settings, model)
    except ValueError:
        if allow_missing_context_window:
            return min(32_768, DEFAULT_MODEL_CONTEXT_TOKENS)
        raise


def _provider_model_capabilities(
    settings: AppSettings,
    provider_settings: ProviderSettings | None,
    model: str,
):
    local_preset = local_model_by_runtime_name(model) or local_model_by_id(model)
    if local_preset is not None:
        return local_model_capabilities(model)
    if provider_settings is None:
        try:
            provider_settings = settings.provider()
        except ValueError:
            return ModelCapabilities()
    return settings.model_capabilities(provider_settings, model)


def _conversation_budget_policy(
    settings: AppSettings,
    model: str,
    *,
    provider_settings: ProviderSettings | None = None,
    allow_missing_context_window: bool = False,
) -> ConversationBudgetPolicy:
    local_preset = local_model_by_runtime_name(model)
    model_context_tokens = (
        local_preset.effective_context_tokens
        if local_preset is not None
        else _provider_model_context_tokens(
            settings,
            provider_settings,
            model,
            allow_missing_context_window=allow_missing_context_window,
        )
    )
    response_token_reserve = (
        local_preset.response_token_reserve
        if local_preset is not None
        else settings.context.resolved_response_token_reserve(model_context_tokens)
    )
    safety_margin_tokens = (
        local_preset.safety_margin_tokens
        if local_preset is not None
        else settings.context.resolved_safety_margin_tokens(model_context_tokens)
    )
    history_token_budget = (
        _local_history_token_budget(
            model_context_tokens,
            response_token_reserve,
            safety_margin_tokens,
        )
        if local_preset is not None
        else settings.context.resolved_history_token_budget(model_context_tokens)
    )
    return ConversationBudgetPolicy(
        history_token_budget=history_token_budget,
        history_window_mode=settings.context.history_window_mode,
        protect_recent_items=settings.context.protect_recent_items,
        model_context_tokens=model_context_tokens,
        response_token_reserve=response_token_reserve,
        safety_margin_tokens=safety_margin_tokens,
    )


def _local_history_token_budget(
    model_context_tokens: int,
    response_token_reserve: int,
    safety_margin_tokens: int,
) -> int:
    usable = max(1, model_context_tokens - response_token_reserve - safety_margin_tokens)
    return max(1, int(usable * 0.75))


def _local_context_prepare_ratio(preset) -> float:
    """Return when to summarize before switching to a local model."""
    if preset.effective_context_tokens <= 16_384:
        return 0.35
    if preset.effective_context_tokens <= 24_576:
        return 0.45
    if preset.effective_context_tokens <= 32_768:
        return 0.55
    return 0.65


def _effective_max_steps(settings: AppSettings, value: int | None) -> int:
    if value is not None:
        return max(1, value)
    return max(1, settings.loop_guard.hard_step_cap)


def _loop_guard_policy(settings: AppSettings) -> LoopGuardPolicy:
    return LoopGuardPolicy(
        enabled=settings.loop_guard.enabled,
        hard_step_cap=settings.loop_guard.hard_step_cap,
    )


def _is_loop_guard_error(result, error) -> bool:
    stop = getattr(result, "loop_guard_stop", None)
    if stop is None:
        return False
    return getattr(error, "code", "") == f"loop_guard_{stop.reason.value}"


def _tool_repair_policy(settings: AppSettings) -> ToolRepairPolicy:
    return ToolRepairPolicy(
        enabled=settings.tool_repair.enabled,
        reasoning_scavenge=settings.tool_repair.reasoning_scavenge,
        argument_repair=settings.tool_repair.argument_repair,
        max_identical_tool_calls=settings.tool_repair.max_identical_tool_calls,
        max_similar_tool_calls=settings.tool_repair.max_similar_tool_calls,
    )


def _tool_output_compaction_policy(settings: AppSettings) -> ToolOutputCompactionPolicy:
    return ToolOutputCompactionPolicy(
        small_output_ratio=settings.tool_output.small_output_ratio,
        medium_output_ratio=settings.tool_output.medium_output_ratio,
        huge_output_ratio=settings.tool_output.huge_output_ratio,
        compact_target_ratio=settings.tool_output.compact_target_ratio,
    )


def _parse_task_cli(
    task_value: str | None,
    prompt_parts: Sequence[str],
) -> tuple[TaskStage | None, str]:
    """Parse --task value and positional prompt into a stage and user text."""
    prompt = " ".join(prompt_parts).strip()
    if task_value is None:
        return None, prompt
    value = task_value.strip()
    stage = TaskStage.parse(value)
    if stage is not None:
        return stage, prompt
    if value.lower() in {TASK_STATUS, TASK_CLEAR}:
        return None, f"task/{value.lower()} {prompt}".strip()
    parts = [part for part in (value, prompt) if part]
    return None, " ".join(parts).strip()


def _run_task_update(
    *,
    provider: ChatCompletionsProvider,
    settings: AppSettings,
    fallback_model: str,
    task_store: TaskStore,
    stage: TaskStage,
    prompt: str,
    final_text: str,
    trace_recorder: TraceRecorder,
    session_id: str,
    provider_settings: ProviderSettings | None = None,
    turn_succeeded: bool = True,
    achievement_required: bool = False,
) -> None:
    if not turn_succeeded:
        trace_recorder.record(
            TraceEvent(
                kind="task_mode_update_skipped",
                summary="Task Mode update skipped because the main turn did not succeed.",
                refs=(
                    f"session_id={session_id}",
                    f"stage={stage.value}",
                    "reason=main_turn_failed",
                ),
            )
        )
        return
    if not should_run_task_update(
        stage,
        user_prompt=prompt,
        final_answer=final_text,
    ):
        trace_recorder.record(
            TraceEvent(
                kind="task_mode_update_skipped",
                summary=(
                    "Task Mode update skipped for low-signal execute turn."
                ),
                refs=(
                    f"session_id={session_id}",
                    f"stage={stage.value}",
                    "reason=low_signal_execute_turn",
                ),
            )
        )
        return
    try:
        model_config = resolve_model_purpose(
            settings,
            "memory",
            fallback_model,
            provider=provider_settings,
        )
        documents = task_store.read_documents()
        result = generate_task_update(
            provider,
            model=model_config.model,
            stage=stage,
            documents=documents,
            user_prompt=prompt,
            final_answer=final_text,
            achievement_required=achievement_required
            or stage == TaskStage.CHECKPOINT,
        )
        changed = apply_task_update_result(
            task_store,
            result,
            stage=stage,
            documents=documents,
            allow_achievement=turn_succeeded,
        )
    except DegenerateTaskPlanError as exc:
        _print_warning("Task Mode notes were not updated for this turn.")
        trace_recorder.record(
            TraceEvent(
                kind="task_mode_update_skipped",
                summary="Task Mode update skipped after a degenerate plan update.",
                refs=(
                    f"session_id={session_id}",
                    f"stage={stage.value}",
                    "reason=degenerate_plan",
                    f"detail={str(exc)[:200]}",
                    f"plan={_workspace_relative(task_store.plan_path, settings.workspace)}",
                ),
            )
        )
        return
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        ProviderError,
    ) as exc:
        _print_warning("Task Mode notes were not updated for this turn.")
        trace_recorder.record(
            TraceEvent(
                kind="task_mode_update_skipped",
                summary="Task Mode update skipped after the user turn.",
                refs=(
                    f"session_id={session_id}",
                    f"stage={stage.value}",
                    f"reason={type(exc).__name__}",
                    f"detail={str(exc)[:200]}",
                ),
            )
        )
        return
    trace_recorder.record(
        TraceEvent(
            kind="task_mode_updated",
            summary=f"Task Mode updated for stage {stage.value}.",
            refs=(
                f"session_id={session_id}",
                f"stage={stage.value}",
                f"changed_files={len(changed)}",
                *(
                    f"changed={_workspace_relative(path, settings.workspace)}"
                    for path in changed
                ),
                *(
                    (f"fallback_reason={result.fallback_reason}",)
                    if result.fallback_reason
                    else ()
                ),
            ),
        )
    )
    if result.achievement_md.strip():
        achievement_path = _task_achievement_path(changed, task_store)
        _safe_save_pet_event(
            PetStateStore.in_data_dir(settings.data_dir),
            event_for_task_achievement(
                workspace=settings.workspace,
                session_id=session_id,
                title=result.achievement_title,
                summary=_task_achievement_summary(result.achievement_md),
                path=(
                    _workspace_relative(achievement_path, settings.workspace)
                    if achievement_path is not None
                    else ""
                ),
            ),
        )


def _run_task_post_turn(
    *,
    provider: ChatCompletionsProvider,
    settings: AppSettings,
    fallback_model: str,
    task_store: TaskStore,
    stage: TaskStage,
    prompt: str,
    final_text: str,
    result,
    trace_recorder: TraceRecorder,
    session_id: str,
    provider_settings: ProviderSettings | None = None,
    turn_succeeded: bool = True,
    achievement_required: bool = False,
) -> ExecuteLoopUpdate | None:
    """Maintain Task Mode files and evaluate execute-loop progress."""
    if not turn_succeeded:
        _run_task_update(
            provider=provider,
            settings=settings,
            fallback_model=fallback_model,
            task_store=task_store,
            stage=stage,
            prompt=prompt,
            final_text=final_text,
            trace_recorder=trace_recorder,
            session_id=session_id,
            provider_settings=provider_settings,
            turn_succeeded=False,
        )
        return None
    _run_task_update(
        provider=provider,
        settings=settings,
        fallback_model=fallback_model,
        task_store=task_store,
        stage=stage,
        prompt=prompt,
        final_text=final_text,
        trace_recorder=trace_recorder,
        session_id=session_id,
        provider_settings=provider_settings,
        turn_succeeded=True,
        achievement_required=achievement_required or stage == TaskStage.CHECKPOINT,
    )
    if stage != TaskStage.EXECUTE:
        return None
    documents = task_store.read_documents()
    gaps = execution_contract_gaps(documents.plan)
    if gaps:
        evaluation = ExecuteEvaluation(
            decision=ExecuteDecision.BLOCKED,
            reason=(
                "task/plan.md no longer has a complete execution contract. "
                f"Missing: {', '.join(gaps)}."
            ),
            next_instruction="Return to task/plan and restore the missing contract sections.",
        )
        state = task_store.read_state()
        turns = (state.execute_turns if state is not None else 0) + 1
        task_store.save_state(
            TaskStage.EXECUTE,
            session_id=session_id,
            execute_status=evaluation.decision.value,
            execute_turns=turns,
            last_reason=evaluation.reason,
            next_instruction=evaluation.next_instruction,
        )
        trace_recorder.record(
            TraceEvent(
                kind="task_execute_evaluated",
                summary="Task execute evaluator blocked on missing contract sections.",
                refs=(
                    f"session_id={session_id}",
                    "decision=blocked",
                    f"turns={turns}",
                    f"missing={','.join(gaps)}",
                ),
            )
        )
        return loop_update_from_evaluation(evaluation, turns=turns)
    try:
        model_config = resolve_model_purpose(
            settings,
            "memory",
            fallback_model,
            provider=provider_settings,
        )
        state = task_store.read_state()
        evaluation = evaluate_execute_progress(
            provider,
            model=model_config.model,
            documents=documents,
            evidence=evidence_from_result(
                user_prompt=prompt,
                result=result,
                final_answer=final_text,
            ),
            state=state,
        )
        turns = (state.execute_turns if state is not None else 0) + 1
        task_store.save_state(
            TaskStage.EXECUTE,
            session_id=session_id,
            execute_status=evaluation.decision.value,
            execute_turns=turns,
            last_reason=evaluation.reason,
            next_instruction=evaluation.next_instruction,
        )
        update = loop_update_from_evaluation(evaluation, turns=turns)
        trace_recorder.record(
            TraceEvent(
                kind="task_execute_evaluated",
                summary=f"Task execute evaluator returned {evaluation.decision.value}.",
                refs=(
                    f"session_id={session_id}",
                    f"decision={evaluation.decision.value}",
                    f"turns={turns}",
                    f"reason={evaluation.reason[:200]}",
                ),
            )
        )
        if evaluation.decision == ExecuteDecision.ACHIEVED:
            _run_task_update(
                provider=provider,
                settings=settings,
                fallback_model=fallback_model,
                task_store=task_store,
                stage=TaskStage.EXECUTE,
                prompt=prompt,
                final_text=final_text + "\n\n" + format_execute_outcome(evaluation),
                trace_recorder=trace_recorder,
                session_id=session_id,
                provider_settings=provider_settings,
                turn_succeeded=True,
                achievement_required=True,
            )
        return update
    except (OSError, ValueError, json.JSONDecodeError, ProviderError) as exc:
        _print_warning("Task execute evaluation failed.")
        trace_recorder.record(
            TraceEvent(
                kind="task_execute_evaluation_failed",
                summary="Task execute evaluation failed.",
                refs=(
                    f"session_id={session_id}",
                    f"reason={type(exc).__name__}",
                    f"detail={str(exc)[:200]}",
                ),
            )
        )
        return None


def _task_achievement_path(
    changed: Sequence[Path],
    task_store: TaskStore,
) -> Path | None:
    for path in reversed(tuple(changed)):
        try:
            if path.parent == task_store.achievements_dir:
                return path
        except OSError:
            continue
    return None


def _task_achievement_summary(content: str) -> str:
    for line in content.splitlines():
        clean = line.strip().lstrip("-").strip()
        if clean and not clean.startswith("#"):
            return clean
    return "Task achievement saved."


def _profile_context_snapshot(
    settings: AppSettings,
    profile,
    model: str,
    *,
    provider_settings: ProviderSettings | None = None,
    task_context: TaskContext | None = None,
    allow_missing_context_window: bool = False,
):
    model_context_tokens = _provider_model_context_tokens(
        settings,
        provider_settings,
        model,
        allow_missing_context_window=allow_missing_context_window,
    )
    task_section = render_task_context_section(task_context)
    return build_profile_context_snapshot(
        workspace=settings.workspace,
        profile=profile,
        hot_profile_token_budget=(
            settings.context.resolved_hot_profile_token_budget(model_context_tokens)
        ),
        hot_profile_warn_tokens=(
            settings.context.hot_profile_warn_tokens(model_context_tokens)
        ),
        extra_sections=(task_section,) if task_section else (),
    )


def _mcp_enabled(args, settings: AppSettings) -> bool:
    """Return whether the agent run should connect configured MCP servers."""
    if args.mcp is False:
        return False
    return bool(settings.mcp_servers)


def _discover_mcp_catalog(
    settings: AppSettings,
    state_store: McpUsageStateStore,
) -> McpToolCatalog:
    """Discover MCP tools and preserve run-local server inventory."""
    inventories: list[McpServerInventory] = []
    for server in settings.mcp_servers:
        try:
            catalog = discover_mcp_catalog(
                (server,),
                settings.workspace,
                state_store=state_store,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            _print_warning(f"MCP server skipped during discovery: {server.name}: {exc}")
            continue
        inventories.extend(catalog.inventories)
    return McpToolCatalog(inventories=tuple(inventories), state_store=state_store)


def _discover_read_only_mcp_tools(settings: AppSettings) -> tuple[McpToolRef, ...]:
    tools: list[McpToolRef] = []
    for server in settings.mcp_servers:
        discovered = discover_mcp_tools(server, settings.workspace)
        tools.extend(tool for tool in discovered if tool.is_read_only())
    return tuple(tools)


def _print_warning(message: str) -> None:
    prefix = "warning"
    if sys.stderr.isatty():
        prefix = "\033[33mwarning\033[0m"
    print(f"{prefix}: {message}", file=sys.stderr)


def _print_status(message: str) -> None:
    print(f"info: {message}", file=sys.stderr)


def _validate_remote_settings(settings: AppSettings, remote: str | None) -> None:
    if remote != "wecom":
        raise ValueError("--remote-validate requires --remote wecom")
    try:
        settings.remote.wecom.validate_ready()
    except ValueError as exc:
        raise ValueError(_wecom_setup_message(settings, str(exc))) from exc
    print("remote wecom configuration ok")
    print(f"- bot_id: {settings.remote.wecom.bot_id}")
    print(f"- secret: {'set' if settings.remote.wecom.secret else 'missing'}")
    print(
        "- allowed_users: "
        + (",".join(settings.remote.wecom.allowed_users) or "(not set)")
    )
    print(f"- group_policy: {settings.remote.wecom.group_policy}")
    print(f"- workspace: {settings.workspace}")
    print(f"- data_dir: {settings.data_dir}")


def _wecom_setup_message(settings: AppSettings, reason: str) -> str:
    config_path = settings.workspace / "config" / "deepmate.yaml"
    return "\n".join(
        (
            f"{reason}",
            "",
            "Enterprise WeChat remote is not ready yet. Save the WeCom values "
            "locally, then validate again:",
            "",
            "  deepmate --setup-wecom <bot-id> <secret>",
            "",
            "  deepmate --remote-validate --remote wecom",
            "  deepmate --remote wecom",
            "",
            "Optional: append allowed users and group access to the same command:",
            "  deepmate --setup-wecom <bot-id> <secret> <user-id> readonly",
            "",
            "Advanced persistent config is here:",
            f"  {config_path}",
        )
    )


def _save_provider_key_command(
    settings: AppSettings,
    api_key_env: str,
    value: str,
) -> Path:
    clean = value.strip()
    if clean == "-":
        if sys.stdin.isatty():
            print("Paste the model API key, then press Enter:", file=sys.stderr)
        clean = sys.stdin.readline().strip()
    if not clean:
        raise ValueError("model API key is empty")
    return save_provider_api_key(settings.data_dir, api_key_env, clean)


def _save_wecom_command(settings: AppSettings, values: Sequence[str]) -> Path:
    parts = [value.strip() for value in values if value.strip()]
    if len(parts) < 1:
        raise ValueError("Enterprise WeChat bot id is required")
    if len(parts) > 4:
        raise ValueError("Use --setup-wecom BOT_ID SECRET [ALLOWED_USERS] [GROUP_POLICY]")
    bot_id = parts[0]
    secret = parts[1] if len(parts) >= 2 else "-"
    allowed_users = parts[2] if len(parts) >= 3 else ""
    group_policy = parts[3] if len(parts) >= 4 else ""
    if secret == "-":
        if sys.stdin.isatty():
            print("Paste the Enterprise WeChat secret, then press Enter:", file=sys.stderr)
        secret = sys.stdin.readline().strip()
    return save_wecom_remote_settings(
        settings.data_dir,
        bot_id=bot_id,
        secret=secret,
        allowed_users=allowed_users,
        group_policy=group_policy,
    )


def _print_context_warning(
    warning: ContextWarning,
    printed: set[tuple[str, str, tuple[str, ...]]],
) -> None:
    key = (warning.code, warning.message, warning.refs)
    if key in printed:
        return
    printed.add(key)
    _print_warning(warning.message)
