from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deepmate.storage import ToolOutputStore


class ToolOutputStoreTests(unittest.TestCase):
    def test_save_and_load_within_session_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolOutputStore.in_data_dir(
                Path(tmp),
                profile="default",
                session_id="session-a",
            )

            record = store.save(
                tool_name="pytest",
                tool_source="native",
                content_kind="log",
                content="raw output",
                estimated_tokens=12,
                request_id="call-1",
            )
            loaded = store.load(record.ref)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.content, "raw output")
        self.assertEqual(loaded.session_id, "session-a")
        self.assertEqual(loaded.profile, "default")

    def test_ref_is_session_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = ToolOutputStore.in_data_dir(root, "default", "session-a")
            second = ToolOutputStore.in_data_dir(root, "default", "session-b")

            record = first.save(
                tool_name="pytest",
                tool_source="native",
                content_kind="log",
                content="raw output",
                estimated_tokens=12,
                request_id="call-1",
            )

            self.assertIsNone(second.load(record.ref))

    def test_invalid_ref_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ToolOutputStore.in_data_dir(Path(tmp), "default", "session-a")

            self.assertIsNone(store.load("../escape"))
            self.assertIsNone(store.load("out_not_hex"))

    def test_prune_unreferenced_deletes_only_unkept_session_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ToolOutputStore.in_data_dir(root, "default", "session-a")
            kept = store.save(
                tool_name="pytest",
                tool_source="native",
                content_kind="log",
                content="keep",
                estimated_tokens=1,
                request_id="keep",
            )
            stale = store.save(
                tool_name="pytest",
                tool_source="native",
                content_kind="log",
                content="stale",
                estimated_tokens=1,
                request_id="stale",
            )

            deleted = store.prune_unreferenced((kept.ref,))

            self.assertEqual(deleted, 1)
            self.assertIsNotNone(store.load(kept.ref))
            self.assertIsNone(store.load(stale.ref))


if __name__ == "__main__":
    unittest.main()
