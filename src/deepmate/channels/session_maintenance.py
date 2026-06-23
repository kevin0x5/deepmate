"""Post-turn maintenance helpers for durable sessions."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from deepmate.activity import ActivityEntry, ActivityStore, preview_activity_text
from deepmate.app import AppSettings, ProviderSettings, resolve_model_purpose
from deepmate.channels.checkpointing import checkpoint_resume_context_item
from deepmate.domain import MessageRole
from deepmate.memory import (
    curator_pending_store,
    record_curator_pending_checkpoint,
    should_skip_memory_extraction,
)
from deepmate.memory.manager import (
    MemoryPatch,
    MemoryPatchOperation,
    apply_memory_patch,
)
from deepmate.providers import (
    ChatCompletionsProvider,
    ModelConversationItem,
)
from deepmate.runtime import (
    ConversationBudgetPolicy,
    HookActor,
    HookDirective,
    HookEnvelope,
    HookEvent,
    HookOutcome,
    HookRuntimeContext,
    SessionRuntime,
    SessionSummaryAction,
    SessionSummaryDecision,
    SessionSummary,
    SessionSummaryInput,
    SessionSummarySourceItem,
    decide_session_summary,
    generate_checkpoint_update,
    session_summary_to_conversation_item,
)
from deepmate.runtime.agent_loop import UserTurnResult
from deepmate.runtime.conversation_budget import estimate_conversation_item_tokens
from deepmate.storage import (
    SessionRecord,
    SessionStore,
    SessionSummaryRecord,
    ToolOutputStore,
    TranscriptRecord,
    TranscriptStore,
    TurnCheckpointStore,
)
from deepmate.storage.tool_output_store import tool_output_ref_value
from deepmate.trace import TraceEvent, TraceRecorder

StatusSink = Callable[[str], None]

SUMMARY_RECENT_RAW_TOKEN_BUDGET = 96_000
SUMMARY_MIN_RECENT_ITEMS = 2


@dataclass(frozen=True, slots=True)
class MemoryCheckpointHookResult:
    """Compact memory patch outcome for checkpoint hook payloads."""

    status: str
    operation_count: int = 0
    applied_count: int = 0
    skipped_count: int = 0
    budget_blocked_count: int = 0
    refs: tuple[str, ...] = ()


def runtime_conversation_from_store(
    session_store: SessionStore,
    session: SessionRecord,
    transcript: TranscriptStore,
    warning_sink: StatusSink | None = None,
    turn_checkpoint_store: TurnCheckpointStore | None = None,
) -> tuple[ModelConversationItem, ...]:
    """Return latest summary plus uncovered transcript, or raw transcript."""
    checkpoint_context = (
        checkpoint_resume_context_item(turn_checkpoint_store.load_latest())
        if turn_checkpoint_store is not None
        else None
    )
    try:
        summary = session_store.summary_store(session).load_latest()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _emit(warning_sink, f"session summary ignored: {exc}")
        return _with_checkpoint_context(transcript.load_items(), checkpoint_context)
    if summary is None:
        return _with_checkpoint_context(transcript.load_items(), checkpoint_context)
    return _with_checkpoint_context(
        (
            session_summary_to_conversation_item(
                summary.content,
                summary_id=summary.summary_id,
            ),
            *transcript.load_items_after(summary.covered_until_sequence),
        ),
        checkpoint_context,
    )


def _with_checkpoint_context(
    items: tuple[ModelConversationItem, ...],
    checkpoint_context: ModelConversationItem | None,
) -> tuple[ModelConversationItem, ...]:
    if checkpoint_context is None:
        return items
    return (*items, checkpoint_context)


def run_session_maintenance(
    provider: ChatCompletionsProvider,
    settings: AppSettings,
    fallback_model: str,
    prompt: str,
    session_store: SessionStore,
    session: SessionRecord,
    transcript: TranscriptStore,
    runtime: SessionRuntime,
    conversation_budget_policy: ConversationBudgetPolicy,
    trace_recorder: TraceRecorder,
    warning_sink: StatusSink | None = None,
    status_sink: StatusSink | None = None,
    hook_context: HookRuntimeContext | None = None,
    provider_settings: ProviderSettings | None = None,
) -> SessionRuntime:
    """Run the current post-turn maintenance chain."""
    _ = prompt  # Kept for the existing channel callback signature.
    record_successful_workflow_evidence(
        trace_recorder=trace_recorder,
        session=session,
        runtime=runtime,
    )
    _prune_tool_outputs_after_turn(
        settings=settings,
        session=session,
        transcript=transcript,
        trace_recorder=trace_recorder,
        runtime=runtime,
        warning_sink=warning_sink,
    )
    return _run_session_summary_after_turn(
        provider=provider,
        settings=settings,
        fallback_model=fallback_model,
        session_store=session_store,
        session=session,
        transcript=transcript,
        runtime=runtime,
        conversation_budget_policy=conversation_budget_policy,
        trace_recorder=trace_recorder,
        warning_sink=warning_sink,
        status_sink=status_sink,
        hook_context=hook_context,
        provider_settings=provider_settings,
    )


def force_session_summary_checkpoint(
    *,
    provider: ChatCompletionsProvider,
    settings: AppSettings,
    fallback_model: str,
    session_store: SessionStore,
    session: SessionRecord,
    transcript: TranscriptStore,
    runtime: SessionRuntime,
    conversation_budget_policy: ConversationBudgetPolicy,
    trace_recorder: TraceRecorder,
    warning_sink: StatusSink | None = None,
    status_sink: StatusSink | None = None,
    hook_context: HookRuntimeContext | None = None,
    reason: str = "forced",
    provider_settings: ProviderSettings | None = None,
) -> SessionRuntime:
    """Create a summary checkpoint now when older source material exists."""
    decision = SessionSummaryDecision(
        action=SessionSummaryAction.CHECKPOINT,
        reason=reason.strip() or "forced",
        refs=("forced=true",),
    )
    return _run_session_summary_checkpoint(
        provider=provider,
        settings=settings,
        fallback_model=fallback_model,
        session_store=session_store,
        session=session,
        transcript=transcript,
        runtime=runtime,
        conversation_budget_policy=conversation_budget_policy,
        trace_recorder=trace_recorder,
        warning_sink=warning_sink,
        status_sink=status_sink,
        hook_context=hook_context,
        decision=decision,
        provider_settings=provider_settings,
    )


def _prune_tool_outputs_after_turn(
    *,
    settings: AppSettings,
    session: SessionRecord,
    transcript: TranscriptStore,
    trace_recorder: TraceRecorder,
    runtime: SessionRuntime,
    warning_sink: StatusSink | None,
) -> None:
    try:
        records = transcript.load_records()
        refs = _tool_output_refs_from_records(records)
        store = ToolOutputStore.in_data_dir(
            settings.data_dir,
            session.profile.name,
            session.session_id,
        )
        deleted = store.prune_unreferenced(refs)
    except Exception as exc:
        _emit(warning_sink, f"tool output cleanup skipped: {exc}")
        return
    if deleted <= 0:
        return
    trace_recorder.record(
        TraceEvent(
            kind="tool_output_pruned",
            summary="Unreferenced session tool outputs were pruned.",
            refs=(
                f"deleted={deleted}",
                f"kept_refs={len(refs)}",
                *runtime.activation.trace_refs(),
            ),
        )
    )


def record_successful_workflow_evidence(
    *,
    trace_recorder: TraceRecorder,
    session: SessionRecord,
    runtime: SessionRuntime,
) -> None:
    """Record deterministic repeated-workflow evidence for evolution maintenance."""
    result = runtime.last_user_turn_result
    if result is None or result.has_errors() or result.reached_max_steps:
        return
    evidence = _workflow_evidence_from_result(result)
    if evidence is None:
        return
    signature, name, description, steps, reference_paths = evidence
    trace_recorder.record(
        TraceEvent(
            kind="workflow_success",
            summary=description,
            refs=(
                f"signature={signature}",
                f"name={name}",
                f"description={description}",
                f"session_id={session.session_id}",
                *(f"step={step}" for step in steps),
                *(f"path={path}" for path in reference_paths),
                *runtime.activation.trace_refs(),
            ),
        )
    )


def _workflow_evidence_from_result(
    result: UserTurnResult,
) -> tuple[str, str, str, tuple[str, ...], tuple[str, ...]] | None:
    tool_names = _successful_tool_names(result)
    if len(tool_names) < 2:
        return None
    signature = "tool workflow: " + " -> ".join(tool_names[:6])
    name = _workflow_name(tool_names)
    description = "Repeated successful workflow using " + ", ".join(tool_names[:4]) + "."
    steps = tuple(f"Run {name}." for name in tool_names[:8])
    paths = _workflow_reference_paths(result)
    return signature, name, description, steps, paths


def _successful_tool_names(result: UserTurnResult) -> tuple[str, ...]:
    names: list[str] = []
    seen_adjacent = ""
    for step in result.steps:
        for tool_result in step.tool_results:
            if tool_result.is_error:
                continue
            name = tool_result.name.strip()
            if not name or name == seen_adjacent:
                continue
            names.append(name)
            seen_adjacent = name
    return tuple(names)


def _workflow_name(tool_names: tuple[str, ...]) -> str:
    words = []
    for name in tool_names[:3]:
        words.extend(part for part in name.replace("-", "_").split("_") if part)
    title = " ".join(word.capitalize() for word in words[:6])
    return title or "Tool Workflow"


def _workflow_reference_paths(result: UserTurnResult) -> tuple[str, ...]:
    paths: list[str] = []
    seen: set[str] = set()
    for step in result.steps:
        for tool_result in step.tool_results:
            for ref in tool_result.refs:
                if not ref.startswith("path="):
                    continue
                path = ref.split("=", 1)[1].strip()
                if path and path not in seen:
                    seen.add(path)
                    paths.append(path)
    return tuple(paths[:8])


def write_session_end_activity(
    settings: AppSettings,
    session_store: SessionStore,
    session: SessionRecord,
    transcript: TranscriptStore,
    runtime: SessionRuntime,
    trace_recorder: TraceRecorder,
    event: str,
    status: str,
    summary: str,
    warning_sink: StatusSink | None = None,
) -> None:
    """Write the final activity entry for a session when transcript exists."""
    records = transcript.load_records()
    if not records:
        return
    try:
        latest_summary = session_store.summary_store(session).load_latest()
    except (OSError, ValueError, json.JSONDecodeError):
        latest_summary = None
    summary_text = _activity_summary_text(summary, latest_summary, runtime)
    _write_activity_entry(
        settings=settings,
        session_store=session_store,
        session=session,
        transcript=transcript,
        runtime=runtime,
        trace_recorder=trace_recorder,
        event=event,
        status=status,
        title=session.title,
        summary=summary_text,
        summary_record=latest_summary,
        warning_sink=warning_sink,
    )
    _record_memory_pending_after_session_end(
        settings=settings,
        session=session,
        runtime=runtime,
        transcript_records=records,
        latest_summary=latest_summary,
        trace_recorder=trace_recorder,
    )


def _run_session_summary_after_turn(
    provider: ChatCompletionsProvider,
    settings: AppSettings,
    fallback_model: str,
    session_store: SessionStore,
    session: SessionRecord,
    transcript: TranscriptStore,
    runtime: SessionRuntime,
    conversation_budget_policy: ConversationBudgetPolicy,
    trace_recorder: TraceRecorder,
    warning_sink: StatusSink | None,
    status_sink: StatusSink | None,
    hook_context: HookRuntimeContext | None,
    provider_settings: ProviderSettings | None,
) -> SessionRuntime:
    decision = _summary_decision(runtime)
    if decision is None or not decision.should_checkpoint():
        return runtime
    return _run_session_summary_checkpoint(
        provider=provider,
        settings=settings,
        fallback_model=fallback_model,
        session_store=session_store,
        session=session,
        transcript=transcript,
        runtime=runtime,
        conversation_budget_policy=conversation_budget_policy,
        trace_recorder=trace_recorder,
        warning_sink=warning_sink,
        status_sink=status_sink,
        hook_context=hook_context,
        decision=decision,
        provider_settings=provider_settings,
    )


def _run_session_summary_checkpoint(
    *,
    provider: ChatCompletionsProvider,
    settings: AppSettings,
    fallback_model: str,
    session_store: SessionStore,
    session: SessionRecord,
    transcript: TranscriptStore,
    runtime: SessionRuntime,
    conversation_budget_policy: ConversationBudgetPolicy,
    trace_recorder: TraceRecorder,
    warning_sink: StatusSink | None,
    status_sink: StatusSink | None,
    hook_context: HookRuntimeContext | None,
    decision: SessionSummaryDecision,
    provider_settings: ProviderSettings | None,
) -> SessionRuntime:
    try:
        summary_store = session_store.summary_store(session)
        previous = summary_store.load_latest()
        source_records = _summary_source_records(
            transcript.load_records(),
            previous,
            conversation_budget_policy,
        )
        if not source_records:
            trace_recorder.record(
                TraceEvent(
                    kind="session_summary_skipped",
                    summary="Session summary skipped because no older source segment is available.",
                    refs=(
                        "reason=no_source_segment",
                        *decision.trace_refs(),
                        *runtime.activation.trace_refs(),
                    ),
                )
            )
            return runtime

        summary_input = SessionSummaryInput(
            previous_summary=previous.content if previous is not None else "",
            previous_covered_item_count=(
                previous.covered_item_count if previous is not None else 0
            ),
            source_items=tuple(
                SessionSummarySourceItem(
                    sequence=record.sequence,
                    item=_record_item(record),
                )
                for record in source_records
            ),
        )
        trace_recorder.record(
            TraceEvent(
                kind="session_summary_checkpoint_started",
                summary="Session summary checkpoint started.",
                refs=(
                    f"source_items={len(source_records)}",
                    f"previous_summary={str(previous is not None).lower()}",
                    *decision.trace_refs(),
                    *runtime.activation.trace_refs(),
                ),
            )
        )
        model_config = resolve_model_purpose(
            settings,
            "summary",
            fallback_model,
            provider=provider_settings,
        )
        update = generate_checkpoint_update(
            provider=provider,
            model=model_config.model,
            summary_input=summary_input,
            profile_dir=settings.global_profile_dir(session.profile.name),
            project_profile_dir=settings.project_profile_dir(session.profile.name),
            options=model_config.options,
        )
        generated = update.session_summary
        record = _summary_record(session, generated)
        summary_store.save_latest(record)
        memory_patch_result = _apply_memory_patch_after_checkpoint(
            settings=settings,
            session=session,
            runtime=runtime,
            summary_input=summary_input,
            summary_record=record,
            memory_patch=update.memory_patch,
            trace_recorder=trace_recorder,
            hook_context=hook_context,
        )
    except Exception as exc:
        _emit(warning_sink, f"session summary skipped: {exc}")
        trace_recorder.record(
            TraceEvent(
                kind="session_summary_failed",
                summary=f"Session summary checkpoint failed: {exc}",
                refs=(
                    *decision.trace_refs(),
                    *runtime.activation.trace_refs(),
                ),
            )
        )
        return runtime

    updated = runtime.with_conversation(
        runtime_conversation_from_store(
            session_store,
            session,
            transcript,
            warning_sink=warning_sink,
        )
    )
    trace_recorder.record(
        TraceEvent(
            kind="session_summary_completed",
            summary="Session summary checkpoint completed.",
            refs=(
                f"summary_id={record.summary_id}",
                f"covered_until_sequence={record.covered_until_sequence}",
                f"covered_item_count={record.covered_item_count}",
                f"source_item_count={record.source_item_count}",
                f"estimated_source_tokens={record.estimated_source_tokens}",
                f"source_model={record.source_model}",
                *_summary_usage_refs(generated.usage),
                *decision.trace_refs(),
                *updated.activation.trace_refs(),
            ),
        )
    )
    _emit_maintenance_hook(
        hook_context,
        HookEvent.CHECKPOINT_CREATED,
        payload={
            "status": "completed",
            "summary": "Session summary checkpoint completed.",
            "summary_id": record.summary_id,
            "covered_until_sequence": record.covered_until_sequence,
            "covered_item_count": record.covered_item_count,
            "source_item_count": record.source_item_count,
            "activity_digest_available": True,
            "memory_patch_status": memory_patch_result.status,
            "memory_patch_operations": memory_patch_result.operation_count,
            "memory_patch_applied": memory_patch_result.applied_count,
            "memory_patch_skipped": memory_patch_result.skipped_count,
            "memory_patch_budget_blocked": memory_patch_result.budget_blocked_count,
            "session_id": session.session_id,
        },
        trace_recorder=trace_recorder,
        runtime=updated,
        session=session,
        refs=(
            f"summary_id={record.summary_id}",
            *memory_patch_result.refs,
        ),
    )
    _write_activity_entry(
        settings=settings,
        session_store=session_store,
        session=session,
        transcript=transcript,
        runtime=updated,
        trace_recorder=trace_recorder,
        event="session_summary_checkpoint",
        status="completed",
        title="Session summary checkpoint",
        summary=update.activity_digest.render(fallback=generated.content),
        summary_record=record,
        warning_sink=warning_sink,
    )
    _emit(
        status_sink,
        "session summary checkpoint saved: "
        f"{record.summary_id} (covered_until_sequence={record.covered_until_sequence})",
    )
    return updated


def _write_activity_entry(
    settings: AppSettings,
    session_store: SessionStore,
    session: SessionRecord,
    transcript: TranscriptStore,
    runtime: SessionRuntime,
    trace_recorder: TraceRecorder,
    event: str,
    status: str,
    title: str,
    summary: str,
    summary_record: SessionSummaryRecord | None = None,
    warning_sink: StatusSink | None = None,
) -> None:
    try:
        store = _activity_store(settings, _activity_profile_name(session))
        entry = ActivityEntry(
            timestamp=_local_timestamp(),
            event=event,
            status=status,
            title=preview_activity_text(title or session.title, 120),
            summary=summary or "Session activity recorded.",
            session_id=session.session_id,
            session_title=session.title,
            profile=session.profile.name,
            workspace=str(session.workspace),
            summary_id=summary_record.summary_id if summary_record is not None else "",
            covered_until_sequence=(
                summary_record.covered_until_sequence
                if summary_record is not None
                else 0
            ),
            transcript_path=_workspace_relative(transcript.path, settings.workspace),
            session_summary_path=(
                _workspace_relative(
                    session_store.summary_path(session.session_id),
                    settings.workspace,
                )
                if summary_record is not None
                else ""
            ),
            trace_path=_workspace_relative(settings.trace_sink, settings.workspace),
        )
        path = store.append_daily_entry(entry)
    except Exception as exc:
        _emit(warning_sink, f"activity note skipped: {exc}")
        trace_recorder.record(
            TraceEvent(
                kind="activity_daily_note_failed",
                summary=f"Activity daily note write failed: {exc}",
                refs=(
                    f"event={event}",
                    f"status={status}",
                    *runtime.activation.trace_refs(),
                ),
            )
        )
        return

    trace_recorder.record(
        TraceEvent(
            kind="activity_daily_note_written",
            summary="Activity daily note written.",
            refs=(
                f"event={event}",
                f"status={status}",
                f"activity_date={entry.local_date()}",
                f"activity_path={_workspace_relative(path, settings.workspace)}",
                f"session_id={session.session_id}",
                *runtime.activation.trace_refs(),
            ),
        )
    )


def _activity_store(settings: AppSettings, profile_name: str) -> ActivityStore:
    clean_name = profile_name.strip() or settings.active_profile
    return ActivityStore(settings.data_dir / "activity" / clean_name)


def _local_timestamp() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def _activity_profile_name(session: SessionRecord) -> str:
    return session.profile.name.strip() or "default"


def _workspace_relative(path: Path, workspace: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace.resolve()))
    except ValueError:
        return str(path)


def _activity_summary_text(
    fallback: str,
    latest_summary: SessionSummaryRecord | None,
    runtime: SessionRuntime,
) -> str:
    if _same_as_latest_summary(fallback, latest_summary):
        return "Session ended after the latest summary checkpoint."
    if _generic_session_end_summary(fallback):
        if latest_summary is not None and latest_summary.content.strip():
            return latest_summary.content
        last_response = _last_runtime_response_text(runtime)
        if last_response:
            return last_response
    return fallback.strip() or "Session activity recorded."


def _record_memory_pending_after_session_end(
    *,
    settings: AppSettings,
    session: SessionRecord,
    runtime: SessionRuntime,
    transcript_records: tuple[TranscriptRecord, ...],
    latest_summary: SessionSummaryRecord | None,
    trace_recorder: TraceRecorder,
) -> None:
    """Queue short-session user-authored text for later low-frequency curation.

    This keeps the hot path cheap: session end records only transcript sequence
    numbers after deterministic local filtering. The existing daily curator does
    any model-backed semantic decision later.
    """
    covered_until = latest_summary.covered_until_sequence if latest_summary else 0
    source_sequences = _memory_source_sequences_from_transcript_records(
        transcript_records,
        after_sequence=covered_until,
    )
    if not source_sequences:
        return
    if _curator_pending_exists(
        settings=settings,
        profile_name=session.profile.name,
        session_id=session.session_id,
        source_sequences=source_sequences,
    ):
        return
    pending = record_curator_pending_checkpoint(
        settings=settings,
        profile_name=session.profile.name,
        profile_uri=session.profile.uri,
        session_id=session.session_id,
        summary_id=(latest_summary.summary_id if latest_summary else "session_end"),
        source_sequences=source_sequences,
    )
    if pending is None:
        return
    trace_recorder.record(
        TraceEvent(
            kind="memory_curator_pending_recorded",
            summary="Session-end user activity queued for low-frequency memory curation.",
            refs=(
                "trigger=session_end",
                f"pending_id={pending.pending_id}",
                f"session_id={session.session_id}",
                f"source_item_count={len(source_sequences)}",
                *runtime.activation.trace_refs(),
            ),
        )
    )


def _memory_source_sequences_from_transcript_records(
    records: tuple[TranscriptRecord, ...],
    *,
    after_sequence: int = 0,
) -> tuple[int, ...]:
    sequences: list[int] = []
    for record in records:
        if record.sequence <= after_sequence:
            continue
        item = record.to_item()
        if item is None or item.message is None:
            continue
        if item.message.role != MessageRole.USER:
            continue
        content = item.message.content.strip()
        if not content:
            continue
        if should_skip_memory_extraction(content).should_skip:
            continue
        sequences.append(record.sequence)
    return tuple(sequences)


def _curator_pending_exists(
    *,
    settings: AppSettings,
    profile_name: str,
    session_id: str,
    source_sequences: tuple[int, ...],
) -> bool:
    try:
        pending_records = curator_pending_store(
            settings.data_dir,
            profile_name,
        ).load_pending()
    except Exception:
        return False
    return any(
        record.session_id == session_id and record.source_sequences == source_sequences
        for record in pending_records
    )


def _generic_session_end_summary(value: str) -> bool:
    text = " ".join(value.lower().split())
    return not text or text.startswith("interactive session ended:")


def _last_runtime_response_text(runtime: SessionRuntime) -> str:
    result = runtime.last_user_turn_result
    if result is None or not result.steps:
        return ""
    response = result.final_step().response
    return response.content.strip() or response.reasoning.strip()


def _same_as_latest_summary(
    value: str,
    latest_summary: SessionSummaryRecord | None,
) -> bool:
    if latest_summary is None:
        return False
    return value.strip() == latest_summary.content.strip()


def _summary_decision(runtime: SessionRuntime):
    result = runtime.last_user_turn_result
    if result is None or not result.steps:
        return None
    final_step = result.final_step()
    report = final_step.request_budget_report
    if report is None:
        return None
    return decide_session_summary(
        report,
        cache_hit_ratio=_cache_hit_ratio(final_step.response.usage),
        profile_context_changed=runtime.profile_context_changed,
        history_trimmed=any(
            step.conversation_budget_report is not None
            and step.conversation_budget_report.trimmed
            for step in result.steps
        ),
    )


def _summary_source_records(
    records: tuple[TranscriptRecord, ...],
    previous: SessionSummaryRecord | None,
    policy: ConversationBudgetPolicy,
) -> tuple[TranscriptRecord, ...]:
    if not records:
        return ()
    cutoff = _summary_cutoff_index(records, policy)
    if cutoff <= 0:
        return ()
    previous_sequence = previous.covered_until_sequence if previous is not None else 0
    return tuple(
        record
        for record in records[:cutoff]
        if record.sequence > previous_sequence
    )


def _summary_cutoff_index(
    records: tuple[TranscriptRecord, ...],
    policy: ConversationBudgetPolicy,
) -> int:
    normalized_policy = policy.normalized()
    protect_count = max(
        SUMMARY_MIN_RECENT_ITEMS,
        normalized_policy.protect_recent_items,
    )
    cutoff_by_items = max(0, len(records) - protect_count)
    cutoff_by_tokens = _recent_raw_token_cutoff(records)
    return max(cutoff_by_items, cutoff_by_tokens)


def _recent_raw_token_cutoff(records: tuple[TranscriptRecord, ...]) -> int:
    used_tokens = 0
    keep_start = len(records)
    for index in reversed(range(len(records))):
        item = _record_item(records[index])
        item_tokens = estimate_conversation_item_tokens(item)
        if keep_start < len(records) and (
            used_tokens + item_tokens > SUMMARY_RECENT_RAW_TOKEN_BUDGET
        ):
            break
        used_tokens += item_tokens
        keep_start = index
    return keep_start


def _record_item(record: TranscriptRecord) -> ModelConversationItem:
    item = record.to_item()
    if item is None:
        raise ValueError(f"transcript record cannot be restored: {record.sequence}")
    return item


def _summary_record(
    session: SessionRecord,
    summary: SessionSummary,
) -> SessionSummaryRecord:
    return SessionSummaryRecord.create(
        session_id=session.session_id,
        content=summary.content,
        covered_until_sequence=summary.covered_until_sequence,
        covered_item_count=summary.covered_item_count,
        source_item_count=summary.source_item_count,
        estimated_source_tokens=summary.estimated_source_tokens,
        source_model=summary.source_model,
        usage=_usage_record(summary.usage),
    )


def _usage_record(usage) -> dict[str, object]:
    if usage is None:
        return {}
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_hit_input_tokens": usage.cache_hit_input_tokens,
        "cache_miss_input_tokens": usage.cache_miss_input_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
    }


def _summary_usage_refs(usage) -> tuple[str, ...]:
    if usage is None:
        return ()
    return (
        f"summary_input_tokens={usage.input_tokens}",
        f"summary_output_tokens={usage.output_tokens}",
        f"summary_cache_hit_input_tokens={usage.cache_hit_input_tokens}",
        f"summary_cache_miss_input_tokens={usage.cache_miss_input_tokens}",
        f"summary_reasoning_tokens={usage.reasoning_tokens}",
    )


def _cache_hit_ratio(usage) -> float | None:
    if usage is None:
        return None
    total = usage.cache_hit_input_tokens + usage.cache_miss_input_tokens
    if total <= 0:
        return None
    return usage.cache_hit_input_tokens / total


def _apply_memory_patch_after_checkpoint(
    settings: AppSettings,
    session: SessionRecord,
    runtime: SessionRuntime,
    summary_input: SessionSummaryInput,
    summary_record: SessionSummaryRecord,
    memory_patch: MemoryPatch,
    trace_recorder: TraceRecorder,
    hook_context: HookRuntimeContext | None = None,
) -> MemoryCheckpointHookResult:
    source_text = _memory_source_text_from_summary_input(summary_input)
    if not source_text:
        trace_recorder.record(
            TraceEvent(
                kind="memory_extraction_skipped",
                summary="Memory extraction skipped because the summary source has no user-authored segment.",
                refs=("reason=no_user_source_segment", *runtime.activation.trace_refs()),
            )
        )
        return MemoryCheckpointHookResult(
            status="skipped",
            operation_count=len(memory_patch.operations),
            refs=("reason=no_user_source_segment",),
        )

    if not memory_patch.operations:
        trace_recorder.record(
            TraceEvent(
                kind="memory_patch_skipped",
                summary="Checkpoint memory patch skipped because no operations were proposed.",
                refs=(
                    "reason=no_memory_patch_operations",
                    f"summary_id={summary_record.summary_id}",
                    *runtime.activation.trace_refs(),
                ),
            )
        )
        return MemoryCheckpointHookResult(
            status="skipped",
            refs=("reason=no_memory_patch_operations",),
        )

    filtered_patch, unsupported_count = _filter_checkpoint_memory_patch(
        memory_patch,
        source_text,
    )
    if not filtered_patch.operations:
        trace_recorder.record(
            TraceEvent(
                kind="memory_patch_skipped",
                summary="Checkpoint memory patch skipped because proposed operations were not supported by user-authored source.",
                refs=(
                    "reason=unsupported_by_user_source",
                    f"unsupported={unsupported_count}",
                    f"summary_id={summary_record.summary_id}",
                    *runtime.activation.trace_refs(),
                ),
            )
        )
        return MemoryCheckpointHookResult(
            status="skipped",
            operation_count=len(memory_patch.operations),
            skipped_count=unsupported_count,
            refs=(
                "reason=unsupported_by_user_source",
                f"unsupported={unsupported_count}",
            ),
        )

    hot_profile_budget = runtime.activation.context_snapshot.hot_profile_token_budget
    try:
        apply_result = apply_memory_patch(
            settings.global_profile_dir(session.profile.name),
            filtered_patch,
            hot_profile_token_budget=hot_profile_budget,
            project_profile_dir=settings.project_profile_dir(session.profile.name),
        )
    except Exception as exc:
        _record_memory_pending_after_checkpoint(
            settings=settings,
            session=session,
            runtime=runtime,
            summary_input=summary_input,
            summary_record=summary_record,
            trace_recorder=trace_recorder,
            reason=f"apply_failed:{exc}",
        )
        return MemoryCheckpointHookResult(
            status="pending",
            operation_count=len(memory_patch.operations),
            refs=(f"reason=apply_failed:{type(exc).__name__}",),
        )

    if apply_result.budget_blocked:
        _record_memory_pending_after_checkpoint(
            settings=settings,
            session=session,
            runtime=runtime,
            summary_input=summary_input,
            summary_record=summary_record,
            trace_recorder=trace_recorder,
            reason="budget_blocked:" + ",".join(apply_result.budget_blocked),
        )
        return MemoryCheckpointHookResult(
            status="pending",
            operation_count=len(memory_patch.operations),
            applied_count=len(apply_result.applied_operations),
            skipped_count=len(apply_result.skipped) + unsupported_count,
            budget_blocked_count=len(apply_result.budget_blocked),
            refs=(
                "reason=budget_blocked",
                *apply_result.summary_refs(),
                f"unsupported={unsupported_count}",
            ),
        )

    result = MemoryCheckpointHookResult(
        status="applied" if apply_result.changed() else "skipped",
        operation_count=len(memory_patch.operations),
        applied_count=len(apply_result.applied_operations),
        skipped_count=len(apply_result.skipped) + unsupported_count,
        budget_blocked_count=len(apply_result.budget_blocked),
        refs=(
            *apply_result.summary_refs(),
            f"unsupported={unsupported_count}",
        ),
    )
    trace_recorder.record(
        TraceEvent(
            kind="memory_patch_applied",
            summary="Checkpoint memory patch applied to profile files.",
            refs=(
                "trigger=session_summary_checkpoint",
                f"summary_id={summary_record.summary_id}",
                *apply_result.summary_refs(),
                f"unsupported={unsupported_count}",
                *runtime.activation.trace_refs(),
            ),
        )
    )
    _emit_maintenance_hook(
        hook_context,
        HookEvent.MEMORY_PATCH_APPLIED,
        payload={
            "status": result.status,
            "summary": "Checkpoint memory patch applied to profile files.",
            "summary_id": summary_record.summary_id,
            "operation_count": result.operation_count,
            "applied": result.applied_count,
            "skipped": result.skipped_count,
            "budget_blocked": result.budget_blocked_count,
            "session_id": session.session_id,
        },
        trace_recorder=trace_recorder,
        runtime=runtime,
        session=session,
        refs=(f"summary_id={summary_record.summary_id}", *result.refs),
    )
    return result


def _filter_checkpoint_memory_patch(
    memory_patch: MemoryPatch,
    user_source_text: str,
) -> tuple[MemoryPatch, int]:
    """Keep checkpoint memory writes grounded in user-authored source."""
    supported_operations = []
    unsupported_count = 0
    for operation in memory_patch.normalized().operations:
        if _checkpoint_memory_operation_supported(operation, user_source_text):
            supported_operations.append(operation)
        else:
            unsupported_count += 1
    return MemoryPatch(operations=tuple(supported_operations)), unsupported_count


def _checkpoint_memory_operation_supported(
    operation: MemoryPatchOperation,
    user_source_text: str,
) -> bool:
    action = operation.action.strip().lower()
    if action in {"skip", "demote_to_warm"}:
        return True
    source_key = _memory_alignment_key(user_source_text)
    if action in {"write_user", "write_memory", "write_project_memory"}:
        return _memory_content_supported(operation.content, source_key)
    if action == "replace":
        return _memory_content_supported(operation.content, source_key)
    if action == "remove":
        return _memory_content_supported(operation.replace_ref, source_key)
    return True


def _memory_content_supported(candidate: str, source_key: str) -> bool:
    return any(
        _memory_text_supported(candidate_key, source_key)
        for candidate_key in _memory_alignment_candidate_keys(candidate)
    )


def _memory_text_supported(candidate_key: str, source_key: str) -> bool:
    if not candidate_key:
        return False
    if candidate_key in source_key:
        return True
    if len(candidate_key) <= 6:
        return False
    candidate_units = _memory_alignment_units(candidate_key)
    if not candidate_units:
        return False
    source_units = _memory_alignment_units(source_key)
    if not source_units:
        return False
    overlap = len(candidate_units & source_units)
    unit_ratio = overlap / len(candidate_units)
    candidate_chars = set(candidate_key)
    char_ratio = len(candidate_chars & set(source_key)) / len(candidate_chars)
    return char_ratio >= 0.95 and unit_ratio >= 0.85


def _memory_alignment_key(text: str) -> str:
    return "".join(char.casefold() for char in str(text) if char.isalnum())


def _memory_alignment_candidate_keys(text: str) -> tuple[str, ...]:
    key = _memory_alignment_key(text)
    variants = [key]
    for prefix in (
        "用户偏好",
        "用户希望",
        "用户要求",
        "用户需要",
        "用户通常",
        "用户倾向于",
        "用户喜欢",
        "用户是",
        "本项目",
        "项目",
        "theuserprefers",
        "userprefers",
        "theuserwants",
        "userwants",
        "theuserneeds",
        "userneeds",
        "thisproject",
        "project",
    ):
        if key.startswith(prefix) and len(key) > len(prefix):
            variants.append(key[len(prefix) :])
    return tuple(dict.fromkeys(variant for variant in variants if variant))


def _memory_alignment_units(key: str) -> set[str]:
    if not key:
        return set()
    units = {key[index : index + 2] for index in range(max(0, len(key) - 1))}
    if len(key) <= 6:
        units.update(key)
    return units


def _record_memory_pending_after_checkpoint(
    settings: AppSettings,
    session: SessionRecord,
    runtime: SessionRuntime,
    summary_input: SessionSummaryInput,
    summary_record: SessionSummaryRecord,
    trace_recorder: TraceRecorder,
    reason: str,
) -> None:
    pending = record_curator_pending_checkpoint(
        settings=settings,
        profile_name=session.profile.name,
        profile_uri=session.profile.uri,
        session_id=session.session_id,
        summary_id=summary_record.summary_id,
        source_sequences=_memory_source_sequences_from_summary_input(summary_input),
    )
    if pending is None:
        return
    trace_recorder.record(
        TraceEvent(
            kind="memory_patch_pending_recorded",
            summary="Memory patch pending record created after checkpoint failure.",
            refs=(
                "trigger=session_summary_checkpoint",
                f"reason={reason}",
                f"pending_id={pending.pending_id}",
                f"summary_id={pending.summary_id}",
                f"source_item_count={len(pending.source_sequences)}",
                *runtime.activation.trace_refs(),
            ),
        )
    )


def _memory_source_text_from_summary_input(summary_input: SessionSummaryInput) -> str:
    sections: list[str] = []
    for source in summary_input.source_items:
        item = source.item
        if item.message is None or item.message.role != MessageRole.USER:
            continue
        content = item.message.content.strip()
        if not content:
            continue
        sections.append(f"### User transcript item {source.sequence}\n{content}")
    return "\n\n".join(sections).strip()


def _memory_source_sequences_from_summary_input(
    summary_input: SessionSummaryInput,
) -> tuple[int, ...]:
    sequences: list[int] = []
    for source in summary_input.source_items:
        item = source.item
        if item.message is None or item.message.role != MessageRole.USER:
            continue
        if item.message.content.strip():
            sequences.append(source.sequence)
    return tuple(sequences)


def _tool_output_refs_from_records(records: Sequence[TranscriptRecord]) -> tuple[str, ...]:
    refs: list[str] = []
    seen: set[str] = set()
    for record in records:
        payload = record.payload
        raw_results = payload.get("tool_results")
        if not isinstance(raw_results, list):
            continue
        for raw_result in raw_results:
            if not isinstance(raw_result, dict):
                continue
            raw_refs = raw_result.get("refs")
            if not isinstance(raw_refs, list):
                continue
            for raw_ref in raw_refs:
                ref = tool_output_ref_value(str(raw_ref))
                if ref and ref not in seen:
                    seen.add(ref)
                    refs.append(ref)
    return tuple(refs)


def _emit_maintenance_hook(
    hook_context: HookRuntimeContext | None,
    event_name: HookEvent,
    *,
    payload: dict[str, object],
    trace_recorder: TraceRecorder,
    runtime: SessionRuntime,
    session: SessionRecord,
    refs: tuple[str, ...] = (),
) -> HookOutcome:
    if hook_context is None:
        return HookOutcome()
    outcome = hook_context.emit(
        HookEnvelope(
            event_name=event_name,
            actor=HookActor.MAINTENANCE,
            payload=payload,
            session_id=session.session_id,
            source_refs=(
                *refs,
                *runtime.activation.trace_refs(),
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
                    *runtime.activation.trace_refs(),
                ),
            )
        )
    return outcome


def _emit(sink: StatusSink | None, message: str) -> None:
    if sink is not None:
        sink(message)
