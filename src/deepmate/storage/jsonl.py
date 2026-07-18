"""JSONL storage helpers."""

from __future__ import annotations

import json
import os
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepmate.storage.atomic import file_lock

_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[Path, "_PathLockEntry"] = {}
MAX_PATH_LOCKS = 1024


@dataclass(slots=True)
class _PathLockEntry:
    lock: threading.Lock
    ref_count: int = 0


def _path_lock(path: Path) -> threading.Lock:
    """Return the per-path lock without pinning it past this lookup."""
    return _path_lock_entry(path)[1].lock


def _writer_path_lock(path: Path) -> tuple[Path, threading.Lock]:
    key = path.resolve(strict=False)
    with _LOCKS_GUARD:
        entry = _PATH_LOCKS.get(key)
        if entry is None:
            if len(_PATH_LOCKS) >= MAX_PATH_LOCKS:
                _prune_path_locks()
            entry = _PathLockEntry(threading.Lock())
            _PATH_LOCKS[key] = entry
        entry.ref_count += 1
        return key, entry.lock


def _path_lock_entry(path: Path) -> tuple[Path, _PathLockEntry]:
    key = path.resolve(strict=False)
    with _LOCKS_GUARD:
        entry = _PATH_LOCKS.get(key)
        if entry is None:
            if len(_PATH_LOCKS) >= MAX_PATH_LOCKS:
                _prune_path_locks()
            entry = _PathLockEntry(threading.Lock())
            _PATH_LOCKS[key] = entry
        return key, entry


def _release_path_lock(path: Path) -> None:
    with _LOCKS_GUARD:
        entry = _PATH_LOCKS.get(path)
        if entry is not None and entry.ref_count > 0:
            entry.ref_count -= 1


def _prune_path_locks() -> None:
    for path, entry in list(_PATH_LOCKS.items()):
        if entry.ref_count > 0 or entry.lock.locked():
            continue
        _PATH_LOCKS.pop(path, None)
        if len(_PATH_LOCKS) < MAX_PATH_LOCKS:
            return


class JsonlWriter:
    """Append dictionary records to a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock_key, self._lock = _writer_path_lock(self.path)
        self._lock_finalizer = weakref.finalize(self, _release_path_lock, self._lock_key)

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
