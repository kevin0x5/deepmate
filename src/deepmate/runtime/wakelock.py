"""Runtime wake lock helper for long-running Deepmate turns."""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class WakeHandle(Protocol):
    """A concrete platform wake handle."""

    def release(self) -> None:
        """Release the wake handle."""


@dataclass(frozen=True, slots=True)
class WakeConfig:
    """Runtime wake configuration."""

    enabled: bool = True
    post_turn_grace_minutes: int = 15


class RuntimeWakeSession:
    """Hold a wake lock during a turn and optionally keep a grace period."""

    def __init__(
        self,
        reason: str,
        config: WakeConfig | None = None,
        *,
        backend: "WakeBackend | None" = None,
    ) -> None:
        self.reason = reason.strip() if isinstance(reason, str) else ""
        self.reason = self.reason or "Deepmate turn"
        self.config = config or WakeConfig()
        self.backend = backend or WakeBackend()
        self._handle: WakeHandle | None = None
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._generation = 0

    def __enter__(self) -> "RuntimeWakeSession":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish_turn()

    def start(self) -> None:
        """Acquire active wake if enabled."""
        if not self.config.enabled:
            return
        with self._lock:
            self._generation += 1
            self._cancel_timer_locked()
            if self._handle is None:
                self._handle = self.backend.acquire(self.reason)

    def finish_turn(self) -> None:
        """Enter grace wake or release immediately."""
        if not self.config.enabled:
            return
        with self._lock:
            grace_seconds = max(0, self.config.post_turn_grace_minutes) * 60
            self._cancel_timer_locked()
            if grace_seconds <= 0:
                self._release_locked()
                return
            generation = self._generation
            self._timer = threading.Timer(
                grace_seconds,
                self._release_for_generation,
                args=(generation,),
            )
            self._timer.daemon = True
            self._timer.start()

    def release(self) -> None:
        """Release active/grace wake immediately."""
        with self._lock:
            self._generation += 1
            self._cancel_timer_locked()
            self._release_locked()

    def _release_for_generation(self, generation: int) -> None:
        with self._lock:
            if generation != self._generation:
                return
            self._cancel_timer_locked()
            self._release_locked()

    def _cancel_timer_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _release_locked(self) -> None:
        if self._handle is not None:
            self._handle.release()
            self._handle = None


class WakeBackend:
    """Platform wake backend."""

    def acquire(self, reason: str) -> WakeHandle:
        system = platform.system().lower()
        if system == "darwin":
            return _CaffeinateHandle.acquire(reason)
        if system == "linux":
            return _SystemdInhibitHandle.acquire(reason)
        if system == "windows":
            return _WindowsWakeHandle.acquire(reason)
        return _NoopWakeHandle()


@dataclass(slots=True)
class _NoopWakeHandle:
    def release(self) -> None:
        return


@dataclass(slots=True)
class _ProcessWakeHandle:
    process: subprocess.Popen

    def release(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()


class _CaffeinateHandle:
    @staticmethod
    def acquire(reason: str) -> WakeHandle:
        caffeinate = _which("caffeinate")
        if caffeinate is None:
            return _NoopWakeHandle()
        process = subprocess.Popen(
            [str(caffeinate), "-i", "-d", "-m", "-w", str(os.getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return _ProcessWakeHandle(process)


class _SystemdInhibitHandle:
    @staticmethod
    def acquire(reason: str) -> WakeHandle:
        systemd_inhibit = _which("systemd-inhibit")
        if systemd_inhibit is None:
            return _NoopWakeHandle()
        process = subprocess.Popen(
            [
                str(systemd_inhibit),
                "--what=idle:sleep",
                "--who=deepmate",
                f"--why={reason}",
                "--mode=block",
                "sleep",
                "infinity",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return _ProcessWakeHandle(process)


@dataclass(slots=True)
class _WindowsWakeHandle:
    previous: int

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002

    @staticmethod
    def acquire(reason: str) -> WakeHandle:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        flags = (
            _WindowsWakeHandle.ES_CONTINUOUS
            | _WindowsWakeHandle.ES_SYSTEM_REQUIRED
            | _WindowsWakeHandle.ES_DISPLAY_REQUIRED
        )
        previous = kernel32.SetThreadExecutionState(flags)
        return _WindowsWakeHandle(previous=previous)

    def release(self) -> None:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)


def _which(name: str) -> Path | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory:
            continue
        candidate = Path(directory) / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    return None
