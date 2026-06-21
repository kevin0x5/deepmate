from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deepmate.activity import ActivityEntry, ActivityStore, preview_activity_text


class ActivityStoreTests(unittest.TestCase):
    def test_append_daily_entry_creates_date_file_with_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ActivityStore(Path(tmp) / "activity" / "default")
            entry = ActivityEntry(
                timestamp="2026-05-30T15:42:10+08:00",
                event="session_end",
                status="completed",
                title="Activity recall design",
                summary="Session finished with an activity recall decision.",
                session_id="session-1",
                session_title="Design activity recall",
                profile="default",
                workspace="/workspace",
                summary_id="summary-1",
                covered_until_sequence=12,
                transcript_path="var/sessions/session-1.jsonl",
                session_summary_path="var/sessions/session-1.summary.json",
                trace_path="var/traces/trace.jsonl",
            )

            path = store.append_daily_entry(entry)

            self.assertEqual(path.name, "2026-05-30.md")
            content = path.read_text(encoding="utf-8")
            self.assertIn("# Activity Note - 2026-05-30", content)
            self.assertIn("## 15:42:10+08:00 - Activity recall design", content)
            self.assertIn("- event: session_end", content)
            self.assertIn("- summary_id: summary-1", content)
            self.assertIn("- profile: default", content)
            self.assertIn("- workspace: /workspace", content)
            self.assertIn("session_summary: var/sessions/session-1.summary.json", content)
            self.assertEqual(
                store.monthly_summary_path("2026-05").name,
                "2026-05.md",
            )

    def test_append_daily_entry_keeps_single_header_for_same_day(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ActivityStore(Path(tmp) / "activity" / "default")
            entry = ActivityEntry(
                timestamp="2026-05-30T10:00:00+08:00",
                event="session_end",
                status="completed",
                title="First",
                summary="First entry",
                session_id="session-1",
                session_title="First session",
                profile="default",
                workspace="/workspace",
            )

            store.append_daily_entry(entry)
            store.append_daily_entry(
                ActivityEntry(
                    timestamp="2026-05-30T11:00:00+08:00",
                    event="session_summary_checkpoint",
                    status="completed",
                    title="Second",
                    summary="Second entry",
                    session_id="session-1",
                    session_title="First session",
                    profile="default",
                    workspace="/workspace",
                )
            )

            content = store.daily_path("2026-05-30").read_text(encoding="utf-8")
            self.assertEqual(content.count("# Activity Note - 2026-05-30"), 1)
            self.assertIn("## 10:00:00+08:00 - First", content)
            self.assertIn("## 11:00:00+08:00 - Second", content)
            self.assertEqual(store.list_daily_dates("2026-05"), ("2026-05-30",))
            self.assertEqual(store.list_daily_dates("2026-06"), ())

    def test_upsert_monthly_summary_entry_replaces_same_date_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ActivityStore(Path(tmp) / "activity" / "default")

            path = store.upsert_monthly_summary_entry(
                local_date="2026-05-30",
                summary="Initial summary.",
                highlights=("First highlight.",),
                next_steps=("First next step.",),
                refs=("session_id=session-1",),
            )
            store.upsert_monthly_summary_entry(
                local_date="2026-05-30",
                summary="Updated summary.",
                highlights=("Updated highlight.",),
                next_steps=("Updated next step.",),
                refs=("session_id=session-2",),
            )
            store.upsert_monthly_summary_entry(
                local_date="2026-05-31",
                summary="Second day summary.",
            )

            content = path.read_text(encoding="utf-8")
            self.assertIn("# Activity Summary - 2026-05", content)
            self.assertEqual(content.count("## 2026-05-30"), 1)
            self.assertIn("- summary: Updated summary.", content)
            self.assertIn("Updated highlight.", content)
            self.assertIn("session_id=session-2", content)
            self.assertNotIn("Initial summary.", content)
            self.assertIn("## 2026-05-31", content)

    def test_preview_activity_text_compacts_whitespace_and_truncates(self) -> None:
        self.assertEqual(preview_activity_text(" hello\n world "), "hello world")
        self.assertEqual(preview_activity_text("abcdef", limit=5), "ab...")


if __name__ == "__main__":
    unittest.main()
