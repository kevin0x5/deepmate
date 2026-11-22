"""JSONL storage helpers."""

from __future__ import annotations

import json
import os
import threading
import weakref
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from deepmate.storage.atomic import file_lock

_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: weakref.WeakValueDictionary[Path, threading.Lock] = weakref.WeakValueDictionary()


def iter_jsonl_objects(path: Path) -> Iterator[dict[str, Any]]:
    """Read UTF-8 JSON objects one physical line at a time, skipping damaged rows."""
    try:
        file = path.open("rb")
    except FileNotFoundError:
        return
    with file:
        for line in file:
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(record, dict):
                yield record


def _path_lock(path: Path) -> threading.Lock:
    """Share a lock while any writer still owns it."""
    key = path.resolve(strict=False)
    with _LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _PATH_LOCKS[key] = lock
        return lock


class JsonlWriter:
    """Append dictionary records to a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = _path_lock(self.path)

    def append(self, record: dict[str, Any]) -> None:
        """Write one JSON record as a single line."""
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            with file_lock(self.path):
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as file:
                    file.write(line)
                    file.flush()
                    os.fsync(file.fileno())

    def append_bounded(
        self,
        record: dict[str, Any],
        *,
        max_bytes: int,
        backups: int = 2,
    ) -> None:
        """Append one record and rotate before exceeding a bounded log size."""
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        encoded_size = len(line.encode("utf-8"))
        size_limit = max(1024, int(max_bytes))
        backup_count = max(1, int(backups))
        with self._lock:
            with file_lock(self.path):
                self.path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    current_size = self.path.stat().st_size
                except OSError:
                    current_size = 0
                if current_size and current_size + encoded_size > size_limit:
                    oldest = self.path.with_name(f"{self.path.name}.{backup_count}")
                    oldest.unlink(missing_ok=True)
                    for index in range(backup_count - 1, 0, -1):
                        source = self.path.with_name(f"{self.path.name}.{index}")
                        target = self.path.with_name(f"{self.path.name}.{index + 1}")
                        if source.exists():
                            os.replace(source, target)
                    if self.path.exists():
                        os.replace(
                            self.path,
                            self.path.with_name(f"{self.path.name}.1"),
                        )
                with self.path.open("a", encoding="utf-8") as file:
                    file.write(line)
                    file.flush()
                    os.fsync(file.fileno())
