from __future__ import annotations

import gc
import json
import subprocess
import sys
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

    def test_registry_retains_only_locks_with_live_writers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            writer = JsonlWriter(root / "active.jsonl")
            for index in range(12):
                JsonlWriter(root / f"other-{index}.jsonl")
            gc.collect()
            self.assertIs(JsonlWriter(writer.path)._lock, writer._lock)
            self.assertEqual([path for path in jsonl_storage._PATH_LOCKS if path.parent == root], [writer.path])

    def test_collecting_a_writer_during_registry_lookup_does_not_deadlock(self) -> None:
        # Run in a child so a regression cannot hang the rest of the suite.
        code = """
import gc
from deepmate.storage.jsonl import JsonlWriter, _LOCKS_GUARD
writer = JsonlWriter("gc-check.jsonl")
writer.cycle = writer
del writer
with _LOCKS_GUARD:
    gc.collect()
"""
        subprocess.run([sys.executable, "-c", code], check=True, timeout=5)


if __name__ == "__main__":
    unittest.main()
