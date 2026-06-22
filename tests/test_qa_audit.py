from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from deepmate.channels.cli import main
from deepmate.channels.tui.commands import command_suggestions
from deepmate.qa import handle_qa_command, maybe_create_qa_audit
from deepmate.qa.commands import maybe_qa_agent_prompt
from deepmate.qa.discovery import discover_project
from deepmate.qa.model import AuditCase
from deepmate.providers import ModelRequest, ModelResponse
from deepmate.runtime.sandbox import SandboxRunResult
from deepmate.qa.store import QaAuditStore


class FakeQaProvider:
    def __init__(self, content: str | None = None) -> None:
        self.requests: list[ModelRequest] = []
        self.content = content or json.dumps(
            {
                "scope": [
                    "LLM-defined release readiness audit",
                    "Validate first-use and runtime evidence against the goal",
                ],
                "risk_model": [
                    "LLM identified that real user paths can fail despite passing unit tests",
                ],
                "permissions": [
                    "Read workspace files for project understanding.",
                    "Run existing local checks in a sandbox.",
                    "Use Computer Use for real visual and interaction validation when approved.",
                ],
                "cases": [
                    {
                        "case_id": "llm.release.tests.001",
                        "title": "LLM selected existing project tests",
                        "surface": "test_surface",
                        "risk_area": "regression",
                        "priority": "high",
                        "persona": "maintainer",
                        "scenario_brief": "Run the detected project test suite as a release gate.",
                        "steps": ["python3 -m unittest discover -s tests -v"],
                        "expected": ["Tests exit successfully with actionable output."],
                        "runner": "shell",
                        "tools": ["shell"],
                        "oracle": "exit_code",
                        "evidence_required": ["command_output"],
                    },
                    {
                        "case_id": "llm.ux.real.001",
                        "title": "LLM selected real user interaction review",
                        "surface": "experience_surface",
                        "risk_area": "real_user_experience",
                        "priority": "high",
                        "persona": "new user",
                        "scenario_brief": "Use Computer Use to inspect the first visible user journey.",
                        "steps": ["Observe the main entrypoint.", "Perform the primary action."],
                        "expected": ["The visible flow is understandable and recoverable."],
                        "runner": "computer",
                        "tools": ["computer_use"],
                        "oracle": "visual_interaction_review",
                        "evidence_required": ["screenshot", "interaction_log"],
                        "blocked_if": ["Computer Use permission is not granted."],
                    },
                    {
                        "case_id": "llm.release.decision.001",
                        "title": "LLM release decision synthesis",
                        "surface": "artifact_surface",
                        "risk_area": "decision_quality",
                        "priority": "high",
                        "persona": "project owner",
                        "scenario_brief": "Synthesize evidence into a release decision.",
                        "steps": ["Review all case evidence."],
                        "expected": ["Report states ready, not ready, or needs review."],
                        "runner": "artifact",
                        "tools": ["render_html_report"],
                        "oracle": "report_review",
                        "evidence_required": ["report.html"],
                    },
                ],
            },
            ensure_ascii=False,
        )

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(content=self.content)


def qa_command(text: str, *, workspace: Path, provider: FakeQaProvider | None = None) -> str:
    return handle_qa_command(
        text,
        workspace=workspace,
        provider=provider or FakeQaProvider(),
        model="qa-planner-model",
        options={"temperature": 0},
    )


class QaAuditTests(unittest.TestCase):
    def test_discovery_is_project_adaptive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "package.json").write_text(
                json.dumps(
                    {
                        "name": "web-product",
                        "scripts": {"test": "vitest", "dev": "vite --host 127.0.0.1"},
                        "devDependencies": {"electron": "^1.0.0"},
                    }
                ),
                encoding="utf-8",
            )
            (workspace / "README.md").write_text("# Web Product\n", encoding="utf-8")
            (workspace / "vite.config.ts").write_text("export default {}", encoding="utf-8")

            profile = discover_project(workspace)

            self.assertEqual(profile.project_name, "web-product")
            self.assertIn("node", profile.project_kinds)
            self.assertIn("ui_surface", profile.surfaces)
            self.assertIn("desktop_surface", profile.surfaces)
            self.assertIn("npm test", profile.test_commands)

    def test_qa_audit_creates_plan_cases_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
            (workspace / "tests").mkdir()
            (workspace / "tests" / "test_demo.py").write_text(
                "import unittest\n\nclass DemoTest(unittest.TestCase):\n"
                "    def test_ok(self):\n        self.assertTrue(True)\n",
                encoding="utf-8",
            )

            created = qa_command("/qa 发布前质量验收", workspace=workspace)
            self.assertIn("QA Audit 方案已生成", created)
            self.assertIn("测试方案概述", created)
            self.assertIn("测试大纲", created)
            self.assertIn("执行前权限清单", created)
            self.assertIn("确认方向无误后输入：/qa run", created)
            store = QaAuditStore(workspace)
            audit_id = store.latest_audit_id()
            paths = store.paths(audit_id)
            self.assertTrue(paths.plan.exists())
            self.assertTrue(paths.cases.exists())
            cases_text = paths.cases.read_text(encoding="utf-8")
            self.assertIn("scenario_brief", cases_text)
            self.assertIn("llm.release.tests.001", cases_text)
            status_before_run = handle_qa_command(f"/qa status {audit_id}", workspace=workspace)
            self.assertIn("permissions confirmed: no", status_before_run)
            self.assertIn("next: review the plan, then run /qa run", status_before_run)

            with patch("deepmate.qa.runner.SandboxRunner.run") as run_shell:
                run_shell.return_value = SandboxRunResult(
                    stdout="tests ok",
                    stderr="",
                    exit_code=0,
                    backend="sandbox-exec",
                    sandboxed=True,
                )
                run = handle_qa_command(f"/qa run {audit_id}", workspace=workspace)
            _command, policy = run_shell.call_args.args
            self.assertFalse(policy.network_enabled)
            self.assertEqual(run_shell.call_args.kwargs.get("timeout_seconds"), 120)
            self.assertIn("QA Audit 完成", run)
            state = store.read_state_mapping(audit_id)
            self.assertEqual(state.get("permissions_confirmed"), True)
            self.assertIn("requested_permissions", state)
            status_after_run = handle_qa_command(f"/qa status {audit_id}", workspace=workspace)
            self.assertIn("permissions confirmed: yes", status_after_run)
            self.assertTrue(paths.report_md.exists())
            self.assertTrue(paths.report_html.exists())
            self.assertIn("<!doctype html>", paths.report_html.read_text(encoding="utf-8"))
            self.assertTrue(any((paths.evidence / "commands").iterdir()))

            task = handle_qa_command(f"/qa task {audit_id}", workspace=workspace)
            self.assertIn("task/plan.md", task)
            self.assertIn(
                "QA Audit",
                (workspace / "task" / "plan.md").read_text(encoding="utf-8"),
            )

    def test_shell_command_failure_is_reported_as_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
            (workspace / "tests").mkdir()
            (workspace / "tests" / "test_demo.py").write_text(
                "import unittest\n\nclass DemoTest(unittest.TestCase):\n"
                "    def test_not_ok(self):\n        self.fail('boom')\n",
                encoding="utf-8",
            )

            qa_command("/qa 发布前质量验收", workspace=workspace)
            store = QaAuditStore(workspace)
            audit_id = store.latest_audit_id()
            with patch("deepmate.qa.runner.SandboxRunner.run") as run_shell:
                run_shell.return_value = SandboxRunResult(
                    stdout="",
                    stderr="boom",
                    exit_code=1,
                    backend="sandbox-exec",
                    sandboxed=True,
                )
                run = handle_qa_command(f"/qa run {audit_id}", workspace=workspace)

            self.assertIn("failed=1", run)
            report = store.paths(audit_id).report_md.read_text(encoding="utf-8")
            self.assertIn("Decision: Not ready", report)

    def test_shell_command_with_shell_syntax_runs_inside_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
            (workspace / "tests").mkdir()
            qa_command("/qa 发布前质量验收", workspace=workspace)
            store = QaAuditStore(workspace)
            audit_id = store.latest_audit_id()
            cases = list(store.read_cases(audit_id))
            rewritten = []
            for case in cases:
                if case.runner == "shell":
                    rewritten.append(AuditCase.from_mapping({**case.to_mapping(), "steps": ("pytest && echo done",)}))
                else:
                    rewritten.append(case)
            store.write_cases(audit_id, rewritten)

            with patch("deepmate.qa.runner.SandboxRunner.run") as run_shell:
                run_shell.return_value = SandboxRunResult(
                    stdout="pytest ok\ndone",
                    stderr="",
                    exit_code=0,
                    backend="sandbox-exec",
                    sandboxed=True,
                )
                run = handle_qa_command(f"/qa run {audit_id}", workspace=workspace)

            self.assertIn("failed=0", run)
            self.assertEqual(run_shell.call_args.args[0], "pytest && echo done")
            _command, policy = run_shell.call_args.args
            self.assertFalse(policy.network_enabled)
            self.assertTrue(
                any(
                    result.summary == "Command passed: pytest && echo done"
                    for result in store.read_results(audit_id)
                )
            )

    def test_shell_command_requires_sandbox_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
            (workspace / "tests").mkdir()
            qa_command("/qa 发布前质量验收", workspace=workspace)
            store = QaAuditStore(workspace)
            audit_id = store.latest_audit_id()

            with patch("deepmate.qa.runner.SandboxRunner.run") as run_shell:
                run_shell.side_effect = RuntimeError("sandbox backend is required")
                run = handle_qa_command(f"/qa run {audit_id}", workspace=workspace)

            self.assertIn("blocked=", run)
            self.assertTrue(
                any(
                    "QA sandbox policy" in result.summary
                    for result in store.read_results(audit_id)
                )
            )

    def test_shell_sandbox_backend_failure_is_blocked_not_project_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
            (workspace / "tests").mkdir()
            qa_command("/qa 发布前质量验收", workspace=workspace)
            store = QaAuditStore(workspace)
            audit_id = store.latest_audit_id()

            with patch("deepmate.qa.runner.SandboxRunner.run") as run_shell:
                run_shell.return_value = SandboxRunResult(
                    stdout="",
                    stderr="sandbox-exec: sandbox_apply: Operation not permitted",
                    exit_code=71,
                    backend="sandbox-exec",
                    sandboxed=True,
                )
                run = handle_qa_command(f"/qa run {audit_id}", workspace=workspace)

            self.assertIn("blocked=", run)
            self.assertTrue(
                any(
                    "sandbox backend failed" in result.summary
                    for result in store.read_results(audit_id)
                )
            )

    def test_shell_evidence_keeps_head_and_tail_of_long_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
            (workspace / "tests").mkdir()
            qa_command("/qa 发布前质量验收", workspace=workspace)
            store = QaAuditStore(workspace)
            audit_id = store.latest_audit_id()

            with patch("deepmate.qa.runner.SandboxRunner.run") as run_shell:
                run_shell.return_value = SandboxRunResult(
                    stdout="HEAD" + ("x" * 50_000) + "TAIL",
                    stderr="",
                    exit_code=0,
                    backend="sandbox-exec",
                    sandboxed=True,
                )
                handle_qa_command(f"/qa run {audit_id}", workspace=workspace)

            evidence = (store.paths(audit_id).evidence / "commands" / "llm.release.tests.001.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn("HEAD", evidence)
            self.assertIn("TAIL", evidence)
            self.assertIn("truncated", evidence)

    def test_state_json_corruption_reports_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "README.md").write_text("# Demo\n", encoding="utf-8")
            qa_command("/qa 发布前质量验收", workspace=workspace)
            store = QaAuditStore(workspace)
            audit_id = store.latest_audit_id()
            store.paths(audit_id).state.write_text("{bad json", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "invalid QA audit state JSON"):
                store.read_state_mapping(audit_id)

    def test_update_state_preserves_concurrent_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "README.md").write_text("# Demo\n", encoding="utf-8")
            qa_command("/qa 发布前质量验收", workspace=workspace)
            store = QaAuditStore(workspace)
            audit_id = store.latest_audit_id()

            threads = [
                threading.Thread(target=store.update_state, args=(audit_id,), kwargs={f"field_{index}": index})
                for index in range(8)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            state = store.read_state_mapping(audit_id)
            for index in range(8):
                self.assertEqual(state.get(f"field_{index}"), index)

    def test_natural_language_qa_request_is_intercepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "README.md").write_text("# Demo\n", encoding="utf-8")

            message = maybe_create_qa_audit(
                "帮我做一次 QA Audit，检查这个项目是否具备发布条件",
                workspace=workspace,
                provider=FakeQaProvider(),
                model="qa-planner-model",
            )

            self.assertIsNotNone(message)
            self.assertIn("QA Audit 方案已生成", message or "")

    def test_natural_language_qa_discussion_does_not_create_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "README.md").write_text("# Demo\n", encoding="utf-8")

            message = maybe_create_qa_audit(
                "发布前验收一般应该怎么做？",
                workspace=workspace,
            )

            self.assertIsNone(message)
            self.assertFalse((workspace / "qa").exists())

    def test_qa_prompt_exposes_agent_context_and_computer_use_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "package.json").write_text(
                json.dumps(
                    {
                        "name": "desktop-demo",
                        "devDependencies": {"electron": "^1.0.0"},
                    }
                ),
                encoding="utf-8",
            )
            qa_command("/qa 检查真实用户体验", workspace=workspace)

            store = QaAuditStore(workspace)
            cases_text = store.paths(store.latest_audit_id()).cases.read_text(encoding="utf-8")
            prompt = handle_qa_command("/qa prompt", workspace=workspace)

            self.assertIn("llm.ux.real.001", cases_text)
            self.assertIn("<qa_audit_context>", prompt)
            self.assertIn("Computer Use", prompt)

    def test_web_ui_audit_plans_real_interaction_without_running_computer_use(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "package.json").write_text(
                json.dumps({"name": "web-demo", "scripts": {"dev": "vite"}}),
                encoding="utf-8",
            )
            (workspace / "vite.config.ts").write_text("export default {}", encoding="utf-8")

            qa_command("/qa 检查真实用户体验", workspace=workspace)
            store = QaAuditStore(workspace)
            audit_id = store.latest_audit_id()
            paths = store.paths(audit_id)
            cases_text = paths.cases.read_text(encoding="utf-8")

            self.assertIn("llm.ux.real.001", cases_text)
            run = handle_qa_command(f"/qa run {audit_id}", workspace=workspace)
            self.assertIn("blocked=", run)
            report = paths.report_md.read_text(encoding="utf-8")
            self.assertIn("Web UI, TUI, CLI, and desktop", report)

    def test_natural_language_continue_returns_agent_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "README.md").write_text("# Demo\n", encoding="utf-8")
            qa_command("/qa 发布前验收", workspace=workspace)

            prompt = maybe_qa_agent_prompt("继续 QA Audit", workspace=workspace)

            self.assertIsNotNone(prompt)
            self.assertIn("<qa_audit_context>", prompt or "")
            self.assertIn("Use subagents", prompt or "")

    def test_tui_command_suggestions_include_qa(self) -> None:
        suggestions = "\n".join(command_suggestions())

        self.assertIn("/qa <goal>", suggestions)

    def test_cli_qa_command_requires_provider_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "README.md").write_text("# Demo\n", encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch.dict("os.environ", {}, clear=True), redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(("--workspace", str(workspace), "--qa", "发布前验收"))

            self.assertEqual(code, 1)
            self.assertNotIn("QA Audit 方案已生成", stdout.getvalue())
            self.assertIn("Deepmate needs a model API key", stderr.getvalue())

    def test_cli_qa_readme_release_readiness_example_creates_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "README.md").write_text("# Demo\n", encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            provider = FakeQaProvider()

            with (
                patch.dict(os.environ, {"STUB_API_KEY": "test-key"}),
                patch(
                    "deepmate.channels.cli.ChatCompletionsProvider",
                    lambda base_url, api_key: provider,
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                code = main(
                    (
                        "--workspace",
                        str(workspace),
                        "--qa",
                        "Run a release-readiness audit for the web app.",
                    )
                )

            self.assertEqual(code, 0, stderr.getvalue())
            self.assertIn("QA Audit 方案已生成", stdout.getvalue())
            self.assertIn("release-readiness audit", stdout.getvalue())
            self.assertTrue((workspace / "qa" / "audits").exists())

    def test_cli_qa_falls_back_when_planner_returns_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "README.md").write_text("# Demo\n", encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            provider = FakeQaProvider(content="not json")

            with (
                patch.dict(os.environ, {"STUB_API_KEY": "test-key"}),
                patch(
                    "deepmate.channels.cli.ChatCompletionsProvider",
                    lambda base_url, api_key: provider,
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                code = main(
                    (
                        "--workspace",
                        str(workspace),
                        "--qa",
                        "Run a release-readiness audit for the web app.",
                    )
                )

            self.assertEqual(code, 0, stderr.getvalue())
            self.assertIn("QA Audit 方案已生成", stdout.getvalue())
            self.assertTrue((workspace / "qa" / "audits").exists())


if __name__ == "__main__":
    unittest.main()
