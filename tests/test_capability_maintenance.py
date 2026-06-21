from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from deepmate.capabilities import (
    CapabilityAssetState,
    CapabilityProposalStore,
    CapabilitySource,
    CapabilityState,
    CapabilityStateStore,
    CapabilityTemperature,
    run_daily_capability_maintenance,
)
from deepmate.domain import CapabilityKind
from deepmate.skills import SkillCard
from deepmate.trace import TraceRecorder


class _TraceSink:
    def __init__(self) -> None:
        self.events = []

    def write(self, event) -> None:
        self.events.append(event)


class CapabilityMaintenanceTests(unittest.TestCase):
    def test_maintenance_cools_local_skill_without_archive_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            data_dir = root / "var"
            skill_path = workspace / "skills" / "local" / "SKILL.md"
            skill_path.parent.mkdir(parents=True)
            skill_path.write_text(
                "---\nname: local\ndescription: Local skill.\n---\nBody.",
                encoding="utf-8",
            )
            card = SkillCard(name="local", description="Local skill.", path=skill_path)
            state_store = CapabilityStateStore.in_data_dir(data_dir, "default")
            start = datetime(2026, 6, 1, tzinfo=UTC)
            state_store.sync_workspace_skills((card,), workspace, now=start)
            trace_sink = _TraceSink()

            result = run_daily_capability_maintenance(
                cards=(card,),
                workspace=workspace,
                state_store=state_store,
                trace_recorder=TraceRecorder(trace_sink),
                now=start + timedelta(days=15),
            )

            state = next(iter(state_store.load().values()))
            self.assertEqual(state.temperature, CapabilityTemperature.COLD)
            self.assertEqual(result.cooled, 1)
            self.assertEqual(result.proposals_created, 0)
            self.assertFalse(result.proposals_path.exists())
            self.assertEqual(
                trace_sink.events[0].kind,
                "capability_maintenance_completed",
            )
            self.assertIn("cooled=1", trace_sink.events[0].refs)

    def test_generated_cold_skill_creates_one_archive_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            data_dir = root / "var"
            state_store = CapabilityStateStore.in_data_dir(data_dir, "default")
            created_at = "2026-06-01T00:00:00+00:00"
            state_store.save(
                {
                    "skill:workspace:auto": CapabilityState(
                        capability_id="skill:workspace:auto",
                        kind=CapabilityKind.SKILL,
                        name="auto",
                        path_or_ref="skills/auto/SKILL.md",
                        source=CapabilitySource.GENERATED,
                        temperature=CapabilityTemperature.COLD,
                        asset_state=CapabilityAssetState.ACTIVE,
                        created_at=created_at,
                        updated_at=created_at,
                        last_used_at=created_at,
                    )
                }
            )
            proposal_store = CapabilityProposalStore.in_state_store(state_store)

            first = run_daily_capability_maintenance(
                cards=(),
                workspace=workspace,
                state_store=state_store,
                proposal_store=proposal_store,
                now=datetime(2026, 6, 20, tzinfo=UTC),
            )
            second = run_daily_capability_maintenance(
                cards=(),
                workspace=workspace,
                state_store=state_store,
                proposal_store=proposal_store,
                now=datetime(2026, 6, 21, tzinfo=UTC),
            )

            self.assertEqual(first.proposals_created, 1)
            self.assertEqual(second.proposals_created, 0)
            proposals = proposal_store.load()
            self.assertEqual(len(proposals), 1)
            self.assertEqual(proposals[0].type, "archive_generated_skill")
            self.assertEqual(proposals[0].status, "pending")
            record = json.loads(proposal_store.path.read_text(encoding="utf-8").strip())
            self.assertEqual(record["capability_id"], "skill:workspace:auto")


if __name__ == "__main__":
    unittest.main()
