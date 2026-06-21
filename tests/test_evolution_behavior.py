from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from deepmate.domain import ProfileRef
from deepmate.evolution import (
    EvolutionChangeStore,
    apply_behavior_hint_change,
    extract_behavior_hints,
    replace_behavior_hints_section,
    run_evolution_maintenance,
)


class EvolutionBehaviorTests(unittest.TestCase):
    def test_extracts_behavior_hints_only_from_behavior_section(self) -> None:
        markdown = "\n".join(
            (
                "# Notes",
                "- Not a behavior hint.",
                "",
                "# Behavior Hints",
                "",
                "- Prefer closed-loop execution.",
                "- Keep broad plans small.",
                "  Include validation when editing files.",
                "",
                "## Other",
                "- Ignored.",
            )
        )

        self.assertEqual(
            extract_behavior_hints(markdown),
            (
                "Prefer closed-loop execution.",
                "Keep broad plans small. Include validation when editing files.",
            ),
        )

    def test_replace_behavior_hints_section_preserves_other_sections(self) -> None:
        markdown = "\n".join(
            (
                "# Intro",
                "Keep this.",
                "",
                "# Behavior Hints",
                "",
                "- Old hint.",
                "",
                "# Tail",
                "Keep tail.",
            )
        )

        updated = replace_behavior_hints_section(
            markdown,
            ("New hint.", "New hint.", "Another hint."),
        )

        self.assertIn("# Intro\nKeep this.", updated)
        self.assertIn("# Tail\nKeep tail.", updated)
        self.assertIn("- New hint.", updated)
        self.assertIn("- Another hint.", updated)
        self.assertNotIn("Old hint", updated)
        self.assertEqual(updated.count("- New hint."), 1)

    def test_apply_behavior_hint_change_records_hashes_and_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            profile = ProfileRef(name="default", uri="profiles/default")
            data_dir = workspace / "var"
            behavior_path = workspace / ".deepmate" / "behavior.md"
            behavior_path.parent.mkdir(parents=True)
            original = "# Intro\nKeep.\n\n# Behavior Hints\n\n- Existing hint.\n"
            behavior_path.write_text(original, encoding="utf-8")

            change = apply_behavior_hint_change(
                workspace=workspace,
                data_dir=data_dir,
                profile=profile,
                hints=("Prefer concise closure.",),
                target_scope="workspace",
                summary="Added explicit collaboration preference.",
                evidence_refs=("test=evidence",),
            )

            self.assertIsNotNone(change)
            assert change is not None
            self.assertEqual(change.change_type, "behavior_patch")
            self.assertTrue(change.old_hash)
            self.assertTrue(change.new_hash)
            self.assertIn("Prefer concise closure.", behavior_path.read_text(encoding="utf-8"))

            store = EvolutionChangeStore.in_data_dir(data_dir, profile)
            loaded = store.load()
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].change_id, change.change_id)
            self.assertEqual(loaded[0].old_content, original)

            result = store.rollback(change.change_id, workspace)

            self.assertEqual(result.change_id, change.change_id)
            self.assertEqual(behavior_path.read_text(encoding="utf-8"), original)
            self.assertEqual(len(store.load()), 2)

    def test_change_store_load_skips_malformed_jsonl_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            profile = ProfileRef(name="default", uri="profiles/default")
            data_dir = workspace / "var"
            behavior_path = workspace / ".deepmate" / "behavior.md"
            behavior_path.parent.mkdir(parents=True)
            behavior_path.write_text("# Behavior Hints\n\n", encoding="utf-8")
            change = apply_behavior_hint_change(
                workspace=workspace,
                data_dir=data_dir,
                profile=profile,
                hints=("Prefer direct answers.",),
                target_scope="workspace",
                summary="Added behavior hint.",
            )
            store = EvolutionChangeStore.in_data_dir(data_dir, profile)
            path = store.path
            original_log = path.read_text(encoding="utf-8")
            path.write_text("{bad json\n" + original_log, encoding="utf-8")

            loaded = store.load()

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].change_id, change.change_id if change else "")

    def test_evolution_maintenance_normalizes_existing_behavior_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            profile = ProfileRef(name="default", uri="profiles/default")
            behavior_path = workspace / ".deepmate" / "behavior.md"
            behavior_path.parent.mkdir(parents=True)
            behavior_path.write_text(
                "# Behavior Hints\n\n-  Prefer closed-loop execution.  \n- Prefer closed-loop execution.\n",
                encoding="utf-8",
            )

            result = run_evolution_maintenance(
                workspace=workspace,
                data_dir=workspace / "var",
                profile=profile,
            )

            self.assertTrue(result.ran)
            self.assertEqual(result.behavior_changes, 1)
            self.assertEqual(
                behavior_path.read_text(encoding="utf-8"),
                "# Behavior Hints\n\n- Prefer closed-loop execution.\n",
            )
            records = [
                json.loads(line)
                for line in result.applied_log_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(records[0]["change_type"], "behavior_patch")
            self.assertTrue(records[0]["old_hash"])
            self.assertTrue(records[0]["new_hash"])


if __name__ == "__main__":
    unittest.main()
