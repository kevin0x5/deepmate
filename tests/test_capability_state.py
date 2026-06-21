from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from deepmate.capabilities import (
    CapabilityStateStore,
    CapabilityTemperature,
    SkillTemperaturePolicy,
)
from deepmate.domain import ProfileRef
from deepmate.skills import SkillCard


class CapabilityStateTests(unittest.TestCase):
    def test_workspace_skill_cools_by_thresholds_and_selection_restores_hot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            data_dir = root / "var"
            skill_path = workspace / "skills" / "writer" / "SKILL.md"
            skill_path.parent.mkdir(parents=True)
            skill_path.write_text(
                "---\nname: writer\ndescription: Draft writing skill.\n---\nBody.",
                encoding="utf-8",
            )
            card = SkillCard(
                name="writer",
                description="Draft writing skill.",
                path=skill_path,
            )
            store = CapabilityStateStore.in_data_dir(
                data_dir,
                ProfileRef(name="default", uri="profiles/default"),
            )
            start = datetime(2026, 6, 1, tzinfo=UTC)
            policy = SkillTemperaturePolicy(
                unused_days_to_warm=7,
                unused_days_to_cold=14,
            )

            states = store.sync_workspace_skills(
                (card,),
                workspace,
                now=start,
                policy=policy,
            )
            state = next(iter(states.values()))
            self.assertEqual(state.temperature, CapabilityTemperature.HOT)
            self.assertEqual(state.exposure(), "name+description")

            states = store.sync_workspace_skills(
                (card,),
                workspace,
                now=start + timedelta(days=8),
                policy=policy,
            )
            state = next(iter(states.values()))
            self.assertEqual(state.temperature, CapabilityTemperature.WARM)
            self.assertEqual(state.exposure(), "name-only")

            states = store.sync_workspace_skills(
                (card,),
                workspace,
                now=start + timedelta(days=13),
                policy=policy,
            )
            state = next(iter(states.values()))
            self.assertEqual(state.temperature, CapabilityTemperature.WARM)

            states = store.sync_workspace_skills(
                (card,),
                workspace,
                now=start + timedelta(days=15),
                policy=policy,
            )
            state = next(iter(states.values()))
            self.assertEqual(state.temperature, CapabilityTemperature.COLD)
            self.assertEqual(state.exposure(), "not-loaded")

            selected = store.record_skill_selected(
                "writer",
                now=start + timedelta(days=16),
            )
            self.assertEqual(selected.temperature, CapabilityTemperature.HOT)
            self.assertEqual(selected.exposure(), "name+description")
            self.assertEqual(selected.invocation_count, 1)

    def test_manual_skill_state_actions_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            data_dir = root / "var"
            skill_path = workspace / "skills" / "review" / "SKILL.md"
            skill_path.parent.mkdir(parents=True)
            skill_path.write_text(
                "---\nname: review\ndescription: Review code.\n---\nBody.",
                encoding="utf-8",
            )
            card = SkillCard(name="review", description="Review code.", path=skill_path)
            store = CapabilityStateStore.in_data_dir(data_dir, "default")
            store.sync_workspace_skills((card,), workspace)

            cooled = store.set_skill_state("review", "cool")
            self.assertEqual(cooled.temperature, CapabilityTemperature.WARM)
            hidden = store.set_skill_state("review", "hide")
            self.assertTrue(hidden.hidden)
            self.assertEqual(hidden.exposure(), "not-loaded")
            restored = store.set_skill_state("review", "restore")
            self.assertEqual(restored.temperature, CapabilityTemperature.HOT)
            self.assertFalse(restored.hidden)

            data = json.loads(store.path.read_text(encoding="utf-8"))
            self.assertEqual(data["version"], 1)
            self.assertEqual(data["capabilities"][0]["temperature"], "hot")

    def test_float_invocation_count_is_loaded_as_non_negative_int(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = CapabilityStateStore.in_data_dir(Path(tmp) / "var", "default")
            store.path.parent.mkdir(parents=True, exist_ok=True)
            store.path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "capabilities": [
                            {
                                "capability_id": "skill:workspace:writer",
                                "kind": "skill",
                                "name": "writer",
                                "path_or_ref": "skills/writer/SKILL.md",
                                "source": "local",
                                "scope": "workspace",
                                "temperature": "hot",
                                "asset_state": "active",
                                "created_at": "2026-06-01T00:00:00+00:00",
                                "updated_at": "2026-06-01T00:00:00+00:00",
                                "invocation_count": 2.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            state = store.load()["skill:workspace:writer"]

            self.assertEqual(state.invocation_count, 2)

    def test_z_suffix_datetimes_are_used_for_skill_cooling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = CapabilityStateStore.in_data_dir(Path(tmp) / "var", "default")
            store.path.parent.mkdir(parents=True, exist_ok=True)
            store.path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "capabilities": [
                            {
                                "capability_id": "skill:workspace:writer",
                                "kind": "skill",
                                "name": "writer",
                                "path_or_ref": "skills/writer/SKILL.md",
                                "source": "local",
                                "scope": "workspace",
                                "temperature": "hot",
                                "asset_state": "active",
                                "created_at": "2026-06-01T00:00:00Z",
                                "updated_at": "2026-06-01T00:00:00Z",
                                "last_used_at": "2026-06-01T00:00:00Z",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            states = store.cool_all_skills(
                now=datetime(2026, 6, 9, tzinfo=UTC),
                policy=SkillTemperaturePolicy(
                    unused_days_to_warm=7,
                    unused_days_to_cold=14,
                ),
            )

            self.assertEqual(
                states["skill:workspace:writer"].temperature,
                CapabilityTemperature.WARM,
            )


if __name__ == "__main__":
    unittest.main()
