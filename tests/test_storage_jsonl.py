from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from deepmate.storage import JsonlWriter
import deepmate.storage.jsonl as jsonl_storage


class JsonlWriterTests(unittest.TestCase):
    def test_append_is_complete_under_threaded_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"

            def write_many(worker: int) -> None:
                writer = JsonlWriter(path)
                for index in range(50):
                    writer.append({"worker": worker, "index": index})

            threads = [
                threading.Thread(target=write_many, args=(worker,))
                for worker in range(8)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(records), 400)
        self.assertEqual(
            {(record["worker"], record["index"]) for record in records},
            {(worker, index) for worker in range(8) for index in range(50)},
        )

    def test_append_fsyncs_written_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"

            with patch("deepmate.storage.jsonl.os.fsync") as fsync:
                JsonlWriter(path).append({"event": "ready"})

            fsync.assert_called_once()

    def test_path_lock_cache_prunes_unlocked_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_locks = dict(jsonl_storage._PATH_LOCKS)
            try:
                jsonl_storage._PATH_LOCKS.clear()
                with patch("deepmate.storage.jsonl.MAX_PATH_LOCKS", 4):
                    for index in range(12):
                        jsonl_storage._path_lock(root / f"events-{index}.jsonl")
                self.assertLessEqual(len(jsonl_storage._PATH_LOCKS), 4)
            finally:
                jsonl_storage._PATH_LOCKS.clear()
                jsonl_storage._PATH_LOCKS.update(old_locks)

    def test_path_lock_cache_does_not_prune_live_writer_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_locks = dict(jsonl_storage._PATH_LOCKS)
            try:
                jsonl_storage._PATH_LOCKS.clear()
                writer = JsonlWriter(root / "active.jsonl")
                active_lock = writer._lock
                with patch("deepmate.storage.jsonl.MAX_PATH_LOCKS", 4):
                    for index in range(12):
                        jsonl_storage._path_lock(root / f"events-{index}.jsonl")
                self.assertIs(JsonlWriter(root / "active.jsonl")._lock, active_lock)
            finally:
                jsonl_storage._PATH_LOCKS.clear()
                jsonl_storage._PATH_LOCKS.update(old_locks)


if __name__ == "__main__":
    unittest.main()
