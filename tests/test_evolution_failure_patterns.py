from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from deepmate.evolution import (
    EvolutionChangeStore,
    FailurePatternGuard,
    FailurePatternStore,
    ToolFailureEvidence,
    UserCorrectionEvidence,
    collect_evidence_from_records,
    update_failure_patterns_from_evidence,
)


class EvolutionFailurePatternTests(unittest.TestCase):
    def test_collects_explicit_evidence_from_trace_like_records(self) -> None:
        batch = collect_evidence_from_records(
            (
                {
                    "kind": "tool_failed",
                    "summary": "write failed",
                    "refs": (
                        "tool_name=filesystem.write",
                        "error_signature=permission denied",
                    ),
                    "recorded_at": "2026-06-06T00:00:00+00:00",
                },
                {
                    "kind": "user_correction",
                    "summary": "Do not rewrite imported skills.",
                    "refs": ("signature=do not rewrite imported skills",),
                    "record_id": "correction-1",
                },
                {
                    "kind": "workflow_success",
                    "summary": "Review generated skill patch safely.",
                    "refs": (
                        "signature=generated skill patch review",
                        "name=Generated Skill Patch Review",
                        "step=Load generated SKILL.md",
                        "step=Validate frontmatter",
                    ),
                },
            )
        )

        self.assertEqual(len(batch.tool_failures), 1)
        self.assertEqual(batch.tool_failures[0].tool_name, "filesystem.write")
        self.assertEqual(len(batch.user_corrections), 1)
        self.assertEqual(
            batch.user_corrections[0].signature,
            "do not rewrite imported skills",
        )
        self.assertEqual(len(batch.workflows), 1)
        self.assertEqual(
            batch.workflows[0].steps,
            ("Load generated SKILL.md", "Validate frontmatter"),
        )

    def test_collects_native_tool_failure_from_bare_ref_and_summary(self) -> None:
        batch = collect_evidence_from_records(
            (
                {
                    "kind": "native_tool_failed",
                    "summary": "Permission denied",
                    "refs": ("write_text_file", "step=1"),
                    "recorded_at": "2026-06-06T00:00:00+00:00",
                },
            )
        )

        self.assertEqual(len(batch.tool_failures), 1)
        self.assertEqual(batch.tool_failures[0].tool_name, "write_text_file")
        self.assertEqual(batch.tool_failures[0].error_signature, "Permission denied")
        self.assertEqual(
            batch.tool_failures[0].source_ref,
            "trace:2026-06-06T00:00:00+00:00:native_tool_failed",
        )

    def test_same_tool_failure_below_threshold_does_not_create_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FailurePatternStore.in_data_dir(Path(tmp) / "var", "default")

            patterns = update_failure_patterns_from_evidence(
                store=store,
                tool_failures=(
                    ToolFailureEvidence("filesystem.write", "permission denied", "trace:1"),
                    ToolFailureEvidence("filesystem.write", "permission denied", "trace:2"),
                ),
                now=datetime(2026, 6, 6, tzinfo=UTC),
            )

            self.assertEqual(patterns, ())
            self.assertEqual(store.load(), ())

    def test_user_correction_threshold_creates_failure_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FailurePatternStore.in_data_dir(Path(tmp) / "var", "default")

            patterns = update_failure_patterns_from_evidence(
                store=store,
                user_corrections=(
                    UserCorrectionEvidence(
                        "do not rename imported skills",
                        "User corrected an imported skill rename.",
                        "session:1",
                    ),
                    UserCorrectionEvidence(
                        "do not rename imported skills",
                        "User repeated the same correction.",
                        "session:2",
                    ),
                ),
                now=datetime(2026, 6, 6, tzinfo=UTC),
            )

            self.assertEqual(len(patterns), 1)
            self.assertEqual(patterns[0].kind, "user_correction")
            self.assertEqual(patterns[0].strength, 5)
            self.assertEqual(patterns[0].source_refs, ("session:1", "session:2"))
            self.assertEqual(store.load()[0].signature, "do not rename imported skills")

    def test_same_tool_failure_threshold_creates_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FailurePatternStore.in_data_dir(Path(tmp) / "var", "default")

            patterns = update_failure_patterns_from_evidence(
                store=store,
                tool_failures=(
                    ToolFailureEvidence("filesystem.write", "permission denied", "trace:1"),
                    ToolFailureEvidence("filesystem.write", "permission denied", "trace:2"),
                    ToolFailureEvidence("filesystem.write", "permission denied", "trace:3"),
                ),
            )

            self.assertEqual(len(patterns), 1)
            self.assertEqual(patterns[0].kind, "tool_failure")
            self.assertEqual(patterns[0].strength, 4)
            self.assertEqual(patterns[0].source_refs, ("trace:1", "trace:2", "trace:3"))

    def test_pattern_match_strengthens_existing_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FailurePatternStore.in_data_dir(Path(tmp) / "var", "default")
            corrections = (
                UserCorrectionEvidence("preserve user assets", "Correction.", "session:1"),
                UserCorrectionEvidence("preserve user assets", "Correction again.", "session:2"),
            )

            update_failure_patterns_from_evidence(store=store, user_corrections=corrections)
            update_failure_patterns_from_evidence(
                store=store,
                user_corrections=(
                    UserCorrectionEvidence("preserve user assets", "Correction.", "session:3"),
                    UserCorrectionEvidence("preserve user assets", "Correction again.", "session:4"),
                ),
            )

            pattern = store.load()[0]
            self.assertEqual(pattern.strength, 6)
            self.assertEqual(
                pattern.source_refs,
                ("session:1", "session:2", "session:3", "session:4"),
            )

    def test_guard_blocks_text_that_matches_strong_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FailurePatternStore.in_data_dir(Path(tmp) / "var", "default")
            update_failure_patterns_from_evidence(
                store=store,
                user_corrections=(
                    UserCorrectionEvidence("delete imported skill", "Do not delete it.", "s1"),
                    UserCorrectionEvidence("delete imported skill", "Still wrong.", "s2"),
                ),
            )

            guard = FailurePatternGuard.from_store(store)
            match = guard.check_text("Generated skill step: delete imported skill files.")

            self.assertTrue(match.has_match())
            self.assertTrue(match.blocked)
            self.assertIn("blocked_by_failure_pattern", match.reason)

    def test_failure_pattern_change_records_one_data_dir_rollback_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            data_dir = Path(tmp) / "var"
            workspace.mkdir()
            store = FailurePatternStore.in_data_dir(data_dir, "default")
            change_store = EvolutionChangeStore.in_data_dir(data_dir, "default")

            patterns = update_failure_patterns_from_evidence(
                store=store,
                user_corrections=(
                    UserCorrectionEvidence("preserve generated skill", "Correction.", "s1"),
                    UserCorrectionEvidence("preserve generated skill", "Again.", "s2"),
                ),
                tool_failures=(
                    ToolFailureEvidence("shell", "permission denied", "t1"),
                    ToolFailureEvidence("shell", "permission denied", "t2"),
                    ToolFailureEvidence("shell", "permission denied", "t3"),
                ),
                now=datetime(2026, 6, 6, tzinfo=UTC),
                change_store=change_store,
                workspace=workspace,
            )

            self.assertEqual(len(patterns), 2)
            changes = change_store.load()
            self.assertEqual(len(changes), 1)
            self.assertEqual(changes[0].change_type, "failure_pattern_update")
            self.assertEqual(changes[0].target_path, str(store.path))

            result = change_store.rollback(changes[0].change_id, workspace)

            self.assertEqual(result.target_path, store.path)
            self.assertFalse(store.path.exists())
            self.assertEqual(len(change_store.load()), 2)


if __name__ == "__main__":
    unittest.main()
