from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from deepmate.capabilities import (
    CapabilityAssetState,
    CapabilityStateStore,
    CapabilityTemperature,
)
from deepmate.activity import ActivityEntry, ActivityStore
from deepmate.domain import ProfileRef
from deepmate.evolution import (
    EvolutionChangeStore,
    GeneratedSkillDraft,
    apply_behavior_hint_change,
    apply_generated_skill_draft,
    run_evolution_maintenance,
)
from deepmate.trace import JsonlTraceSink, TraceEvent, TraceRecorder
from deepmate.storage.session_store import SessionStore


class EvolutionMaintenanceTests(unittest.TestCase):
    def test_no_new_signal_noops_and_records_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_dir = workspace / "var"
            profile = ProfileRef(name="default", uri="profiles/default")
            now = datetime(2026, 6, 6, 2, 0, tzinfo=UTC)

            result = run_evolution_maintenance(
                workspace=workspace,
                data_dir=data_dir,
                profile=profile,
                now=now,
                trace_path=data_dir / "trace.jsonl",
                sessions_dir=data_dir / "sessions",
                activity_dir=data_dir / "activity" / "default",
            )

            self.assertFalse(result.ran)
            self.assertEqual(result.reason, "no_new_signals")
            self.assertTrue(result.maintenance_state_path.exists())
            state = json.loads(result.maintenance_state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["last_reason"], "no_new_signals")
            self.assertEqual(state["last_run_at"], "2026-06-06T02:00:00+00:00")

    def test_last_run_cursor_prevents_reprocessing_trace_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_dir = workspace / "var"
            profile = ProfileRef(name="default", uri="profiles/default")
            trace_path = data_dir / "trace.jsonl"
            _write_trace(
                trace_path,
                (
                    _trace_record(
                        "user_correction",
                        "Do not rewrite imported skills.",
                        "2026-06-06T01:00:00+00:00",
                        ("signature=do not rewrite imported skills",),
                    ),
                    _trace_record(
                        "user_correction",
                        "Do not rewrite imported skills.",
                        "2026-06-06T01:05:00+00:00",
                        ("signature=do not rewrite imported skills",),
                    ),
                ),
            )

            first = run_evolution_maintenance(
                workspace=workspace,
                data_dir=data_dir,
                profile=profile,
                now=datetime(2026, 6, 6, 2, 0, tzinfo=UTC),
                trace_path=trace_path,
            )
            second = run_evolution_maintenance(
                workspace=workspace,
                data_dir=data_dir,
                profile=profile,
                now=datetime(2026, 6, 6, 3, 0, tzinfo=UTC),
                trace_path=trace_path,
            )

            self.assertTrue(first.ran)
            self.assertEqual(first.failure_patterns_updated, 1)
            self.assertFalse(second.ran)
            records = [
                json.loads(line)
                for line in first.applied_log_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [record["change_type"] for record in records],
                ["behavior_patch", "failure_pattern_update"],
            )

    def test_last_run_cursor_accepts_z_suffix_datetime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_dir = workspace / "var"
            profile = ProfileRef(name="default", uri="profiles/default")
            state_path = data_dir / "evolution" / "default" / "maintenance_state.json"
            state_path.parent.mkdir(parents=True)
            state_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "last_run_at": "2026-06-06T02:00:00Z",
                        "last_reason": "forced",
                    }
                ),
                encoding="utf-8",
            )
            trace_path = data_dir / "trace.jsonl"
            _write_trace(
                trace_path,
                (
                    _trace_record(
                        "user_correction",
                        "Do not rewrite imported skills.",
                        "2026-06-06T01:00:00Z",
                        ("signature=do not rewrite imported skills",),
                    ),
                ),
            )

            result = run_evolution_maintenance(
                workspace=workspace,
                data_dir=data_dir,
                profile=profile,
                now=datetime(2026, 6, 6, 3, 0, tzinfo=UTC),
                trace_path=trace_path,
            )

            self.assertFalse(result.ran)
            self.assertEqual(result.reason, "no_new_signals")

    def test_maintenance_updates_behavior_failure_pattern_generated_skill_and_metrics(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_dir = workspace / "var"
            profile = ProfileRef(name="default", uri="profiles/default")
            behavior_path = workspace / ".deepmate" / "behavior.md"
            behavior_path.parent.mkdir(parents=True)
            behavior_path.write_text(
                "# Behavior Hints\n\n-  Keep changes scoped.  \n- Keep changes scoped.\n",
                encoding="utf-8",
            )
            trace_path = data_dir / "trace.jsonl"
            _write_trace(
                trace_path,
                (
                    _trace_record(
                        "model_response_received",
                        "Model response.",
                        "2026-06-06T01:00:00+00:00",
                        ("input_tokens=10", "output_tokens=4", "reasoning_tokens=1"),
                    ),
                    _trace_record(
                        "native_tool_completed",
                        "Native tool completed: load_skill.",
                        "2026-06-06T01:01:00+00:00",
                        ("load_skill", "skill=Existing Skill"),
                    ),
                    _trace_record(
                        "capability_selected",
                        "Selected skill.",
                        "2026-06-06T01:02:00+00:00",
                        ("skill=Existing Skill",),
                    ),
                    _trace_record(
                        "user_correction",
                        "Do not rewrite imported skills.",
                        "2026-06-06T01:03:00+00:00",
                        ("signature=do not rewrite imported skills",),
                    ),
                    _trace_record(
                        "user_correction",
                        "Do not rewrite imported skills.",
                        "2026-06-06T01:04:00+00:00",
                        ("signature=do not rewrite imported skills",),
                    ),
                    _trace_record(
                        "workflow_success",
                        "Review generated skill patch safely.",
                        "2026-06-06T01:05:00+00:00",
                        (
                            "signature=generated skill patch review",
                            "name=Generated Skill Patch Review",
                            "step=Load generated SKILL.md",
                            "step=Validate frontmatter",
                        ),
                    ),
                    _trace_record(
                        "workflow_success",
                        "Review generated skill patch safely.",
                        "2026-06-06T01:06:00+00:00",
                        (
                            "signature=generated skill patch review",
                            "name=Generated Skill Patch Review",
                            "step=Load generated SKILL.md",
                            "step=Validate frontmatter",
                        ),
                    ),
                ),
            )

            result = run_evolution_maintenance(
                workspace=workspace,
                data_dir=data_dir,
                profile=profile,
                now=datetime(2026, 6, 6, 2, 0, tzinfo=UTC),
                trace_path=trace_path,
            )

            self.assertTrue(result.ran)
            self.assertEqual(result.behavior_changes, 1)
            self.assertEqual(result.failure_patterns_updated, 1)
            self.assertEqual(result.generated_skill_changes, 1)
            self.assertEqual(result.metrics.token_cost, 15)
            self.assertEqual(result.metrics.loaded_skills_count, 1)
            self.assertEqual(result.metrics.used_skills_count, 1)
            self.assertEqual(result.metrics.user_correction_count, 2)
            self.assertEqual(result.metrics.generated_skill_apply_count, 1)
            self.assertTrue(
                (
                    workspace
                    / "skills"
                    / "generated"
                    / "generated-skill-patch-review"
                    / "SKILL.md"
                ).exists()
            )
            self.assertTrue(result.metrics_path.exists())

    def test_maintenance_adds_behavior_hint_from_repeated_user_correction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_dir = workspace / "var"
            profile = ProfileRef(name="default", uri="profiles/default")
            trace_path = data_dir / "trace.jsonl"
            _write_trace(
                trace_path,
                (
                    _trace_record(
                        "user_correction",
                        "Do not rewrite imported skills.",
                        "2026-06-06T01:03:00+00:00",
                        ("signature=do not rewrite imported skills",),
                    ),
                    _trace_record(
                        "user_correction",
                        "Do not rewrite imported skills.",
                        "2026-06-06T01:04:00+00:00",
                        ("signature=do not rewrite imported skills",),
                    ),
                ),
            )

            result = run_evolution_maintenance(
                workspace=workspace,
                data_dir=data_dir,
                profile=profile,
                now=datetime(2026, 6, 6, 2, 0, tzinfo=UTC),
                trace_path=trace_path,
            )

            behavior = (workspace / ".deepmate" / "behavior.md").read_text(
                encoding="utf-8"
            )
            self.assertTrue(result.ran)
            self.assertEqual(result.behavior_changes, 1)
            self.assertIn("- Do not rewrite imported skills.", behavior)
            changes = EvolutionChangeStore.in_data_dir(data_dir, profile).load()
            self.assertTrue(
                any(
                    change.summary
                    == "Added behavior hints from repeated user corrections."
                    for change in changes
                )
            )

    def test_profile_scoped_corrections_write_profile_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_dir = workspace / "var"
            profile = ProfileRef(name="default", uri="profiles/default")
            records = (
                {
                    "kind": "user_correction",
                    "summary": "Always be concise.",
                    "refs": (
                        "signature=always be concise",
                        "target_scope=profile",
                        "source_group=session:1",
                    ),
                },
                {
                    "kind": "user_correction",
                    "summary": "Always be concise.",
                    "refs": (
                        "signature=always be concise",
                        "target_scope=profile",
                        "source_group=session:2",
                    ),
                },
            )

            result = run_evolution_maintenance(
                workspace=workspace,
                data_dir=data_dir,
                profile=profile,
                evidence_records=records,
                force=True,
            )

            profile_behavior = workspace / "profiles" / "default" / "behavior.md"
            self.assertEqual(result.behavior_changes, 1)
            self.assertIn("Always be concise.", profile_behavior.read_text(encoding="utf-8"))
            self.assertFalse((workspace / ".deepmate" / "behavior.md").exists())

    def test_session_transcripts_contribute_distinct_corrections_and_workflows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_dir = workspace / "var"
            sessions_dir = data_dir / "sessions"
            profile = ProfileRef(name="default", uri="profiles/default")
            store = SessionStore.in_directory(sessions_dir)
            for index in range(2):
                session = store.create(workspace, profile, title="Review Flow")
                records = (
                    {
                        "session_id": session.session_id,
                        "record_id": f"message-{index}",
                        "recorded_at": "2026-07-18T01:00:00+00:00",
                        "sequence": 1,
                        "kind": "message",
                        "payload": {
                            "role": "user",
                            "content": "Correction: Do not overwrite generated files.",
                        },
                    },
                    {
                        "session_id": session.session_id,
                        "record_id": f"tools-{index}",
                        "recorded_at": "2026-07-18T01:01:00+00:00",
                        "sequence": 2,
                        "kind": "tool_exchange",
                        "payload": {
                            "tool_results": [
                                {"name": "read_text_file", "is_error": False},
                                {"name": "write_text_file", "is_error": False},
                            ]
                        },
                    },
                )
                session.transcript_path.write_text(
                    "\n".join(json.dumps(record) for record in records) + "\n",
                    encoding="utf-8",
                )

            result = run_evolution_maintenance(
                workspace=workspace,
                data_dir=data_dir,
                profile=profile,
                sessions_dir=sessions_dir,
                now=datetime(2026, 7, 19, tzinfo=UTC),
            )

            self.assertEqual(result.sessions_seen, 2)
            self.assertEqual(result.behavior_changes, 1)
            self.assertEqual(result.generated_skill_changes, 1)
            generated_skills = tuple(
                (workspace / "skills" / "generated").glob("*/SKILL.md")
            )
            self.assertEqual(len(generated_skills), 1)
            generated_content = generated_skills[0].read_text(encoding="utf-8")
            self.assertNotIn("Review Flow", generated_content)

    def test_activity_notes_contribute_distinct_explicit_corrections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_dir = workspace / "var"
            activity_root = data_dir / "activity" / "default"
            activity_store = ActivityStore(activity_root)
            for index in range(2):
                activity_store.append_daily_entry(
                    ActivityEntry(
                        timestamp=f"2026-07-19T0{index + 1}:00:00+08:00",
                        event="session_end",
                        status="completed",
                        title="Preference correction",
                        summary="Correction: Respond in Chinese.",
                        session_id=f"session-{index}",
                        session_title="Preference correction",
                        profile="default",
                        workspace=str(workspace),
                    )
                )

            result = run_evolution_maintenance(
                workspace=workspace,
                data_dir=data_dir,
                profile=ProfileRef(name="default", uri="profiles/default"),
                activity_dir=activity_root,
            )

            profile_behavior = workspace / "profiles" / "default" / "behavior.md"
            self.assertEqual(result.activity_notes_seen, 1)
            self.assertEqual(result.behavior_changes, 1)
            self.assertIn("Respond in Chinese.", profile_behavior.read_text(encoding="utf-8"))

    def test_maintenance_does_not_persist_sensitive_behavior_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_dir = workspace / "var"
            profile = ProfileRef(name="default", uri="profiles/default")
            trace_path = data_dir / "trace.jsonl"
            correction = "Use api_key=sk-secretvalue123 for future requests."
            _write_trace(
                trace_path,
                (
                    _trace_record(
                        "user_correction",
                        correction,
                        "2026-06-06T01:03:00+00:00",
                        ("signature=sensitive",),
                    ),
                    _trace_record(
                        "user_correction",
                        correction,
                        "2026-06-06T01:04:00+00:00",
                        ("signature=sensitive",),
                    ),
                ),
            )

            result = run_evolution_maintenance(
                workspace=workspace,
                data_dir=data_dir,
                profile=profile,
                now=datetime(2026, 6, 6, 2, 0, tzinfo=UTC),
                trace_path=trace_path,
            )

            self.assertTrue(result.ran)
            self.assertEqual(result.behavior_changes, 0)
            self.assertFalse((workspace / ".deepmate" / "behavior.md").exists())

    def test_maintenance_filters_identity_and_network_behavior_hints(self) -> None:
        sensitive_corrections = (
            "Use account_id=customer-12345 for future requests.",
            "Always connect to 10.20.30.40 for this service.",
            "Call +86 138 0013 8000 before deployment.",
        )
        for index, correction in enumerate(sensitive_corrections):
            with self.subTest(correction=correction), tempfile.TemporaryDirectory() as tmp:
                workspace = Path(tmp)
                records = tuple(
                    {
                        "kind": "user_correction",
                        "summary": correction,
                        "refs": (
                            f"signature=sensitive-{index}",
                            f"source_group=session:{source_index}",
                        ),
                    }
                    for source_index in range(2)
                )

                result = run_evolution_maintenance(
                    workspace=workspace,
                    data_dir=workspace / "var",
                    profile=ProfileRef(name="default", uri="profiles/default"),
                    evidence_records=records,
                    force=True,
                )

                self.assertEqual(result.behavior_changes, 0)
                self.assertFalse((workspace / ".deepmate" / "behavior.md").exists())

    def test_maintenance_archives_cold_generated_skill_from_capability_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_dir = workspace / "var"
            profile = "default"
            state_store = CapabilityStateStore.in_data_dir(data_dir, profile)
            applied = apply_generated_skill_draft(
                draft=GeneratedSkillDraft(
                    name="Generated Cleanup",
                    description="Clean generated skill assets.",
                    steps=("Inspect generated SKILL.md.",),
                    source_refs=("test=setup",),
                ),
                workspace=workspace,
                data_dir=data_dir,
                profile=profile,
                state_store=state_store,
                now=datetime(2026, 6, 6, 1, 0, tzinfo=UTC),
            )
            self.assertTrue(applied.is_applied())
            state_store.set_skill_state(
                "Generated Cleanup",
                "cool",
                now=datetime(2026, 6, 6, 1, 10, tzinfo=UTC),
            )
            state_store.set_skill_state(
                "Generated Cleanup",
                "cool",
                now=datetime(2026, 6, 6, 1, 20, tzinfo=UTC),
            )

            result = run_evolution_maintenance(
                workspace=workspace,
                data_dir=data_dir,
                profile=ProfileRef(name="default", uri="profiles/default"),
                now=datetime(2026, 6, 6, 2, 0, tzinfo=UTC),
            )

            self.assertTrue(result.ran)
            self.assertEqual(result.capability_state_changes, 1)
            state = state_store.skill_states_by_name()["generated cleanup"]
            self.assertEqual(state.asset_state, CapabilityAssetState.ARCHIVED)
            self.assertEqual(state.temperature, CapabilityTemperature.COLD)

    def test_maintenance_trace_does_not_trigger_next_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_dir = workspace / "var"
            profile = ProfileRef(name="default", uri="profiles/default")
            trace_path = data_dir / "trace.jsonl"
            trace_recorder = TraceRecorder(JsonlTraceSink(trace_path))

            first = run_evolution_maintenance(
                workspace=workspace,
                data_dir=data_dir,
                profile=profile,
                now=datetime(2026, 6, 6, 2, 0, tzinfo=UTC),
                trace_path=trace_path,
                trace_recorder=trace_recorder,
                force=True,
            )
            second = run_evolution_maintenance(
                workspace=workspace,
                data_dir=data_dir,
                profile=profile,
                now=datetime(2026, 6, 6, 3, 0, tzinfo=UTC),
                trace_path=trace_path,
            )

            self.assertTrue(first.ran)
            self.assertFalse(second.ran)

    def test_applied_log_signal_records_rollback_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_dir = workspace / "var"
            profile = ProfileRef(name="default", uri="profiles/default")
            behavior_path = workspace / ".deepmate" / "behavior.md"
            behavior_path.parent.mkdir(parents=True)
            original = "# Behavior Hints\n\n- Original hint.\n"
            behavior_path.write_text(original, encoding="utf-8")
            change = apply_behavior_hint_change(
                workspace=workspace,
                data_dir=data_dir,
                profile=profile,
                hints=("Temporary hint.",),
                now=datetime(2026, 6, 6, 1, 0, tzinfo=UTC),
            )
            assert change is not None
            store = EvolutionChangeStore.in_data_dir(data_dir, profile)
            store.rollback(change.change_id, workspace)

            result = run_evolution_maintenance(
                workspace=workspace,
                data_dir=data_dir,
                profile=profile,
                trace_path=data_dir / "trace.jsonl",
            )

            self.assertTrue(result.ran)
            self.assertEqual(result.metrics.rollback_count, 1)
            self.assertTrue(result.metrics_path.exists())


def _trace_record(
    kind: str,
    summary: str,
    recorded_at: str,
    refs: tuple[str, ...],
) -> dict[str, object]:
    return {
        "kind": kind,
        "summary": summary,
        "recorded_at": recorded_at,
        "refs": list(refs),
    }


def _write_trace(path: Path, records: tuple[dict[str, object], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
