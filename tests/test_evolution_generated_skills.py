from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from deepmate.capabilities import (
    CapabilityAssetState,
    CapabilitySource,
    CapabilityStateStore,
    CapabilityTemperature,
)
from deepmate.domain import ProfileRef
from deepmate.evolution import (
    EvolutionChangeStore,
    FailurePatternGuard,
    FailurePatternStore,
    GeneratedSkillDraft,
    UserCorrectionEvidence,
    WorkflowEvidence,
    apply_generated_skill_draft,
    apply_generated_skill_patch,
    archive_generated_skill,
    generated_skill_drafts_from_workflows,
    update_failure_patterns_from_evidence,
    workflow_candidates,
)
from deepmate.skills import SkillCatalog, SkillCard, load_skill_card


class EvolutionGeneratedSkillTests(unittest.TestCase):
    def test_repeated_workflow_threshold_generates_draft(self) -> None:
        workflows = (
            WorkflowEvidence(
                signature="review generated skill patch",
                name="Generated Skill Patch Review",
                description="Review and patch generated Deepmate skills.",
                steps=("Load generated SKILL.md.", "Validate frontmatter."),
                source_ref="session:1",
            ),
            WorkflowEvidence(
                signature="review generated skill patch",
                name="Generated Skill Patch Review",
                description="Review and patch generated Deepmate skills.",
                steps=("Load generated SKILL.md.", "Validate frontmatter."),
                source_ref="session:2",
            ),
        )

        drafts = generated_skill_drafts_from_workflows(workflow_candidates(workflows))

        self.assertEqual(len(drafts), 1)
        self.assertEqual(drafts[0].name, "Generated Skill Patch Review")
        self.assertEqual(
            drafts[0].steps,
            ("Load generated SKILL.md.", "Validate frontmatter."),
        )

    def test_generated_skill_draft_writes_catalog_and_hot_generated_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_dir = workspace / "var"
            profile = ProfileRef(name="default", uri="profiles/default")
            state_store = CapabilityStateStore.in_data_dir(data_dir, profile)
            draft = _draft()

            result = apply_generated_skill_draft(
                draft=draft,
                workspace=workspace,
                data_dir=data_dir,
                profile=profile,
                state_store=state_store,
                now=datetime(2026, 6, 6, tzinfo=UTC),
            )

            self.assertTrue(result.is_applied())
            assert result.skill_path is not None
            self.assertTrue(result.skill_path.exists())
            card = load_skill_card(result.skill_path)
            self.assertEqual(card.name, draft.name)
            catalog = SkillCatalog.from_paths((workspace / "skills",))
            self.assertIsNotNone(catalog.get(draft.name))

            states = state_store.skill_states_by_name()
            state = states["generated capability hygiene"]
            self.assertEqual(state.source, CapabilitySource.GENERATED)
            self.assertEqual(state.temperature, CapabilityTemperature.HOT)
            self.assertEqual(state.exposure(), "name+description")

    def test_strong_failure_pattern_blocks_generated_skill_apply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_dir = workspace / "var"
            profile = "default"
            pattern_store = FailurePatternStore.in_data_dir(data_dir, profile)
            update_failure_patterns_from_evidence(
                store=pattern_store,
                user_corrections=(
                    UserCorrectionEvidence("unsafe archive imported skill", "No.", "s1"),
                    UserCorrectionEvidence("unsafe archive imported skill", "No.", "s2"),
                ),
            )
            guard = FailurePatternGuard.from_store(pattern_store)

            result = apply_generated_skill_draft(
                draft=GeneratedSkillDraft(
                    name="Unsafe Skill Archiver",
                    description="Archive imported skills automatically.",
                    steps=("Run unsafe archive imported skill cleanup.",),
                    source_refs=("session:1",),
                ),
                workspace=workspace,
                data_dir=data_dir,
                profile=profile,
                guard=guard,
            )

            self.assertEqual(result.status, "blocked")
            self.assertFalse((workspace / "skills").exists())

    def test_generated_skill_draft_rejects_existing_skill_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            existing = workspace / "skills" / "existing" / "SKILL.md"
            existing.parent.mkdir(parents=True)
            existing.write_text(
                "---\n"
                "name: Generated Capability Hygiene\n"
                "description: Existing skill.\n"
                "---\n"
                "Body.",
                encoding="utf-8",
            )

            result = apply_generated_skill_draft(
                draft=_draft(),
                workspace=workspace,
                data_dir=workspace / "var",
                profile="default",
            )

            self.assertEqual(result.status, "rejected")
            self.assertIn("skill_name_already_exists", result.reason)
            self.assertFalse(
                (workspace / "skills" / "generated" / "generated-capability-hygiene").exists()
            )

    def test_generated_skill_patch_applies_and_invalid_patch_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_dir = workspace / "var"
            profile = "default"
            state_store = CapabilityStateStore.in_data_dir(data_dir, profile)
            applied = apply_generated_skill_draft(
                draft=_draft(),
                workspace=workspace,
                data_dir=data_dir,
                profile=profile,
                state_store=state_store,
            )
            assert applied.skill_path is not None
            original = applied.skill_path.read_text(encoding="utf-8")
            updated = original + "\n## Extra\n- Keep validation focused.\n"

            patched = apply_generated_skill_patch(
                skill_name="Generated Capability Hygiene",
                new_markdown=updated,
                workspace=workspace,
                data_dir=data_dir,
                profile=profile,
                state_store=state_store,
            )

            self.assertTrue(patched.is_applied())
            self.assertIn("Keep validation focused.", applied.skill_path.read_text(encoding="utf-8"))

            failed = apply_generated_skill_patch(
                skill_name="Generated Capability Hygiene",
                new_markdown="---\nname: Different\ndescription: Bad.\n---\nBody.",
                workspace=workspace,
                data_dir=data_dir,
                profile=profile,
                state_store=state_store,
            )

            self.assertEqual(failed.status, "rolled_back")
            self.assertEqual(applied.skill_path.read_text(encoding="utf-8"), updated)

    def test_imported_skill_patch_is_rejected_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_dir = workspace / "var"
            skill_path = workspace / "skills" / "imported" / "SKILL.md"
            skill_path.parent.mkdir(parents=True)
            original = "---\nname: Imported\ndescription: Imported skill.\n---\nBody."
            skill_path.write_text(original, encoding="utf-8")
            state_store = CapabilityStateStore.in_data_dir(data_dir, "default")
            state_store.record_skill_installed(
                SkillCard("Imported", "Imported skill.", skill_path),
                workspace,
                source=CapabilitySource.IMPORTED,
            )

            result = apply_generated_skill_patch(
                skill_name="Imported",
                new_markdown=original + "\nChanged.",
                workspace=workspace,
                data_dir=data_dir,
                profile="default",
                state_store=state_store,
            )

            self.assertEqual(result.status, "rejected")
            self.assertEqual(result.reason, "skill_is_not_generated")
            self.assertEqual(skill_path.read_text(encoding="utf-8"), original)

    def test_generated_skill_draft_and_archive_rollbacks_restore_capability_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_dir = workspace / "var"
            profile = "default"
            state_store = CapabilityStateStore.in_data_dir(data_dir, profile)
            applied = apply_generated_skill_draft(
                draft=_draft(),
                workspace=workspace,
                data_dir=data_dir,
                profile=profile,
                state_store=state_store,
            )
            assert applied.change is not None
            assert applied.skill_path is not None
            rollback = applied.change

            store = EvolutionChangeStore.in_data_dir(data_dir, profile)
            store.rollback(rollback.change_id, workspace)

            self.assertFalse(applied.skill_path.exists())
            self.assertEqual(state_store.load(), {})

            reapplied = apply_generated_skill_draft(
                draft=_draft(),
                workspace=workspace,
                data_dir=data_dir,
                profile=profile,
                state_store=state_store,
            )
            archived = archive_generated_skill(
                skill_name="Generated Capability Hygiene",
                workspace=workspace,
                data_dir=data_dir,
                profile=profile,
                state_store=state_store,
            )
            self.assertTrue(archived.is_applied())
            archived_state = state_store.skill_states_by_name()[
                "generated capability hygiene"
            ]
            self.assertEqual(archived_state.asset_state, CapabilityAssetState.ARCHIVED)
            self.assertEqual(archived_state.temperature, CapabilityTemperature.COLD)
            assert archived.change is not None

            store.rollback(archived.change.change_id, workspace)

            restored_state = state_store.skill_states_by_name()[
                "generated capability hygiene"
            ]
            self.assertEqual(restored_state.asset_state, CapabilityAssetState.ACTIVE)
            self.assertEqual(restored_state.temperature, CapabilityTemperature.HOT)
            self.assertTrue(reapplied.skill_path and reapplied.skill_path.exists())


def _draft() -> GeneratedSkillDraft:
    return GeneratedSkillDraft(
        name="Generated Capability Hygiene",
        description="Maintain generated Deepmate skill assets safely.",
        steps=(
            "Load the generated SKILL.md.",
            "Validate frontmatter and body before applying changes.",
            "Leave imported and local skills untouched.",
        ),
        source_refs=("session:generated-skill",),
    )


if __name__ == "__main__":
    unittest.main()
