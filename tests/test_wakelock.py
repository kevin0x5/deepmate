from __future__ import annotations

import unittest

from deepmate.runtime.wakelock import RuntimeWakeSession, WakeConfig


class WakeLockTests(unittest.TestCase):
    def test_stale_grace_timer_does_not_release_new_turn_handle(self) -> None:
        backend = _FakeWakeBackend()
        session = RuntimeWakeSession(
            None,
            WakeConfig(enabled=True, post_turn_grace_minutes=15),
            backend=backend,
        )

        session.start()
        first_handle = backend.handles[-1]
        session.finish_turn()
        session.start()
        session._release_for_generation(1)

        self.assertEqual(len(backend.handles), 1)
        self.assertFalse(first_handle.released)
        session.release()
        self.assertTrue(first_handle.released)


class _FakeWakeBackend:
    def __init__(self) -> None:
        self.handles: list[_FakeWakeHandle] = []

    def acquire(self, reason: str):
        handle = _FakeWakeHandle(reason)
        self.handles.append(handle)
        return handle


class _FakeWakeHandle:
    def __init__(self, reason: str) -> None:
        self.reason = reason
        self.released = False

    def release(self) -> None:
        self.released = True


if __name__ == "__main__":
    unittest.main()
