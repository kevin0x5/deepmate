from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evals.environment import EvalEnvironment, FakeMcpServer, FakeOllamaServer
from evals.loader import load_cases


class EvalEnvironmentTests(unittest.TestCase):
    def test_trials_have_distinct_workspace_and_data_roots(self) -> None:
        case = next(case for case in load_cases() if case.id == "dm-cxt-freeze-refresh-001")
        with tempfile.TemporaryDirectory() as tmp:
            factory = EvalEnvironment(tmp)
            first = factory.create(case, 1)
            second = factory.create(case, 2)
            self.assertNotEqual(first.workspace, second.workspace)
            self.assertNotEqual(first.data_dir, second.data_dir)
            (first.workspace / "only-first.txt").write_text("synthetic", encoding="utf-8")
            self.assertFalse((second.workspace / "only-first.txt").exists())

    def test_runtime_canaries_are_unique_and_outside_workspace(self) -> None:
        case = next(
            case
            for case in load_cases()
            if case.id == "dm-sec-context-cross-session-001"
        )
        with tempfile.TemporaryDirectory() as tmp:
            factory = EvalEnvironment(tmp)
            first = factory.create(case, 1)
            second = factory.create(case, 2)
            self.assertNotEqual(first.canaries, second.canaries)
            workspace_rules = (first.workspace / "AGENTS.md").read_text(
                encoding="utf-8"
            )
            self.assertFalse(any(value in workspace_rules for value in first.canaries))

    def test_fake_mcp_is_idempotent_by_key(self) -> None:
        server = FakeMcpServer({"responses": {"create": {"side_effect": True}}})
        server.call("create", {"idempotency_key": "stable-key"})
        server.call("create", {"idempotency_key": "stable-key"})
        self.assertEqual(server.side_effects, {"stable-key": 1})

    def test_fake_ollama_resumes_after_interruption(self) -> None:
        server = FakeOllamaServer(("disconnect",))
        self.assertFalse(server.prepare("synthetic-model")["ready"])
        self.assertTrue(server.prepare("synthetic-model")["ready"])
        self.assertEqual(server.downloads["synthetic-model"], 2)

    def test_fake_ollama_covers_deployment_and_inference_faults(self) -> None:
        for fault in ("disk_full", "cancelled", "corrupt_manifest"):
            with self.subTest(fault=fault):
                server = FakeOllamaServer((fault,))
                result = server.prepare("synthetic-model")
                self.assertFalse(result["ready"])
                self.assertEqual(result["fault"], fault)
        slow = FakeOllamaServer(("slow_response",))
        self.assertEqual(slow.prepare("synthetic-model")["simulated_latency_ms"], 2500)
        failed_inference = FakeOllamaServer(("inference_failure",))
        self.assertTrue(failed_inference.prepare("synthetic-model")["ready"])
        self.assertFalse(failed_inference.infer("synthetic-model", "prompt")["ok"])


if __name__ == "__main__":
    unittest.main()
