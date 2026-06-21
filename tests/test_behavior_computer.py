from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from deepmate.behavior import (
    BehaviorRule,
    BehaviorRuleStore,
    behavior_runtime_for_session,
)
from deepmate.behavior.rules import (
    extract_explicit_behavior_rules,
    match_behavior_rules,
    render_behavior_turn_tail,
    workspace_hash,
)
from deepmate.channels.interactive import _handle_computer_command
from deepmate.channels.tui.commands import handle_tui_command
from deepmate.channels.tui.state import TuiRuntimeState
from deepmate.domain import Message, MessageRole, ProfileRef
from deepmate.providers import ModelResponse
from deepmate.runtime import (
    ToolAccessMode,
    ToolAccessPolicy,
    start_runtime_activation,
    start_session_runtime,
)
from deepmate.runtime.prefix_cache import model_request_prefix_fingerprint
from deepmate.storage import SessionStore
from deepmate.tools import (
    COMPUTER_TOOL_NAMES,
    ComputerUseState,
    NativeToolRegistry,
    computer_tools,
)
from deepmate.tools.computer import _bounds_from_text, _path_is_private
from deepmate.trace import TraceEvent


class _StubProvider:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("stub provider received too many requests")
        return self.responses.pop(0)


class _TraceRecorder:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def record(self, event: TraceEvent) -> None:
        self.events.append(event)


class BehaviorComputerTests(unittest.TestCase):
    def test_rule_store_matches_project_and_global_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            store = BehaviorRuleStore.in_data_dir(root / "var", "default")
            global_rule = store.upsert(
                BehaviorRule(rule_id="", text="回复先给结论", scope="global")
            )
            project_rule = store.upsert(
                BehaviorRule(
                    rule_id="",
                    text="这个项目先跑单元测试",
                    scope="workspace",
                    workspace_hash=workspace_hash(workspace),
                    tags=("test",),
                )
            )

            matched = match_behavior_rules(
                store.enabled_rules(),
                "请修复这个项目的测试",
                workspace_hash_value=workspace_hash(workspace),
            )

            self.assertIn(global_rule.rule_id, {rule.rule_id for rule in matched})
            self.assertIn(project_rule.rule_id, {rule.rule_id for rule in matched})
            rendered = render_behavior_turn_tail(matched)
            self.assertIn("<deepmate_behavior_context>", rendered)
            self.assertIn("回复先给结论", rendered)

    def test_explicit_rule_extraction_requires_user_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self.assertEqual(
                extract_explicit_behavior_rules(
                    "帮我修复这个测试",
                    workspace=workspace,
                ),
                (),
            )
            rules = extract_explicit_behavior_rules(
                "以后回复先给结论，不要太长。",
                workspace=workspace,
            )
            self.assertEqual(len(rules), 1)
            self.assertIn("回复先给结论", rules[0].text)

    def test_behavior_turn_tail_does_not_change_system_prefix_or_history(self) -> None:
        with _workspace() as workspace:
            data_dir = workspace / "var"
            profile = ProfileRef(name="default", uri="profiles/default")
            activation = start_runtime_activation(
                session_id="session_1",
                workspace=workspace,
                profile=profile,
            )
            behavior = behavior_runtime_for_session(
                data_dir=data_dir,
                workspace=workspace,
                profile=profile,
                session_id="session_1",
            )
            behavior.rule_store.upsert(
                BehaviorRule(rule_id="", text="回复先给结论", scope="global")
            )
            runtime = start_session_runtime(activation, behavior_runtime=behavior)
            provider = _StubProvider(
                [ModelResponse(content="one"), ModelResponse(content="two")]
            )

            first = runtime.run_user_turn(
                provider=provider,
                messages=(Message(role=MessageRole.USER, content="介绍一下"),),
                model="stub-model",
            )
            second = first.runtime.run_user_turn(
                provider=provider,
                messages=(Message(role=MessageRole.USER, content="继续介绍"),),
                model="stub-model",
            )

            first_prefix = model_request_prefix_fingerprint(provider.requests[0])
            second_prefix = model_request_prefix_fingerprint(provider.requests[1])
            self.assertEqual(first_prefix.system_digest, second_prefix.system_digest)
            first_roles = [
                item.message.role
                for item in provider.requests[0].conversation
                if item.message is not None
            ]
            self.assertEqual(first_roles[-1], MessageRole.USER)
            self.assertIn(
                "<deepmate_behavior_context>",
                provider.requests[0].conversation[-1].message.content,
            )
            self.assertNotIn(
                "<deepmate_behavior_context>",
                "\n".join(
                    item.message.content
                    for item in second.runtime.conversation
                    if item.message is not None
                ),
            )

    def test_forget_request_disables_matching_rule(self) -> None:
        with _workspace() as workspace:
            data_dir = workspace / "var"
            profile = ProfileRef(name="default", uri="profiles/default")
            behavior = behavior_runtime_for_session(
                data_dir=data_dir,
                workspace=workspace,
                profile=profile,
                session_id="session_1",
            )
            behavior.rule_store.upsert(
                BehaviorRule(rule_id="", text="回复先给结论", scope="global")
            )

            prepared = behavior.prepare_turn_tail(
                (Message(role=MessageRole.USER, content="忘记 回复先给结论"),)
            )

            self.assertEqual(len(prepared.disabled_rules), 1)
            self.assertEqual(behavior.rule_store.enabled_rules(), ())

    def test_computer_use_turn_tail_is_current_task_only(self) -> None:
        with _workspace() as workspace:
            data_dir = workspace / "var"
            profile = ProfileRef(name="default", uri="profiles/default")
            behavior = behavior_runtime_for_session(
                data_dir=data_dir,
                workspace=workspace,
                profile=profile,
                session_id="session_1",
                computer_use_enabled=True,
            )

            prepared = behavior.prepare_turn_tail(
                (Message(role=MessageRole.USER, content="打开网页并截图"),),
                tool_schema_names=("computer_screenshot", "computer_snapshot"),
            )

            self.assertEqual(behavior.rule_store.enabled_rules(), ())
            self.assertIn("<deepmate_computer_use>", prepared.messages[0].content)
            self.assertIn("Screenshot capability: available", prepared.messages[0].content)
            self.assertIn("UI snapshot capability: available", prepared.messages[0].content)

    def test_computer_tools_are_hidden_until_exposed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            enabled = False
            learning = False
            tools = computer_tools(
                data_dir=Path(tmp) / "var",
                workspace=Path(tmp),
                session_id="session_1",
                state=ComputerUseState(
                    enabled=lambda: enabled,
                    computer_learning_enabled=lambda: learning,
                ),
            )
            registry = NativeToolRegistry(tools)

            self.assertEqual(registry.schemas(), ())
            enabled = True
            status = registry.get("computer_status").call({})
            self.assertIn("enabled: true", status.content)

    def test_computer_tool_family_is_hidden_until_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            enabled = False
            tools = computer_tools(
                data_dir=Path(tmp) / "var",
                workspace=Path(tmp),
                session_id="session_1",
                state=ComputerUseState(
                    enabled=lambda: enabled,
                    computer_learning_enabled=lambda: False,
                ),
                exposed_by_default=False,
            )
            registry = NativeToolRegistry(tools)

            self.assertEqual(registry.schemas(), ())
            self.assertEqual(
                {tool.name for tool in registry.list_tools()},
                set(COMPUTER_TOOL_NAMES),
            )
            with self.assertRaises(ValueError):
                registry.get("computer_click").call({"x": 10, "y": 20})

            enabled = True
            self.assertIn("enabled: true", registry.get("computer_status").call({}).content)

    def test_computer_actions_are_not_read_only_and_require_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = computer_tools(
                data_dir=Path(tmp) / "var",
                workspace=Path(tmp),
                session_id="session_1",
                state=ComputerUseState(
                    enabled=lambda: True,
                    computer_learning_enabled=lambda: False,
                ),
            )
            registry = NativeToolRegistry(tools)

            read_only_by_name = {tool.name: tool.read_only for tool in registry.list_tools()}
            self.assertTrue(read_only_by_name["computer_status"])
            self.assertTrue(read_only_by_name["computer_snapshot"])
            self.assertTrue(read_only_by_name["computer_screenshot"])
            self.assertFalse(read_only_by_name["computer_click"])
            self.assertFalse(read_only_by_name["computer_type"])
            self.assertFalse(read_only_by_name["computer_key"])
            self.assertFalse(read_only_by_name["computer_open"])

            with (
                patch("deepmate.tools.computer.platform.system", return_value="Darwin"),
                self.assertRaisesRegex(ValueError, "readable screen observation"),
            ):
                registry.get("computer_click").call({"x": 10, "y": 20})

    def test_computer_click_requires_readable_snapshot_and_converts_screenshot_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calls = []

            def fake_run_process(command, **_kwargs):
                calls.append(tuple(command))
                if command[0] == "screencapture":
                    output = Path(command[-1])
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(
                        b"\x89PNG\r\n\x1a\n"
                        b"\x00\x00\x00\rIHDR"
                        b"\x00\x00\x07\x80"
                        b"\x00\x00\x04\x38"
                        b"\x08\x02\x00\x00\x00"
                        b"\x00\x00\x00\x00"
                    )
                return type("Completed", (), {"stdout": ""})()

            def fake_run_osascript(_script, *_args, **_kwargs):
                return type(
                    "Completed",
                    (),
                    {
                        "stdout": (
                            "Computer accessibility snapshot\n"
                            "- front_app: Safari\n"
                            "- bundle_id: com.apple.Safari\n"
                            "Windows:\n"
                            "- role=AXWindow name=\"Search\" bounds=0,0,960,540"
                        )
                    },
                )()

            tools = computer_tools(
                data_dir=Path(tmp) / "var",
                workspace=Path(tmp),
                session_id="session_1",
                state=ComputerUseState(
                    enabled=lambda: True,
                    computer_learning_enabled=lambda: False,
                ),
                exposed_by_default=True,
            )
            registry = NativeToolRegistry(tools)

            with (
                patch("deepmate.tools.computer.platform.system", return_value="Darwin"),
                patch(
                    "deepmate.tools.computer._screen_bounds",
                    return_value={
                        "left": 0,
                        "top": 0,
                        "right": 960,
                        "bottom": 540,
                        "width": 960,
                        "height": 540,
                    },
                ),
                patch("deepmate.tools.computer._run_process", side_effect=fake_run_process),
                patch("deepmate.tools.computer._run_osascript", side_effect=fake_run_osascript),
            ):
                screenshot = registry.get("computer_screenshot").call({})
                with self.assertRaisesRegex(ValueError, "readable screen observation"):
                    registry.get("computer_click").call(
                        {"x": 100, "y": 60, "coordinate_space": "screenshot_pixel"}
                    )
                registry.get("computer_snapshot").call({})
                clicked = registry.get("computer_click").call(
                    {"x": 100, "y": 60, "coordinate_space": "screenshot_pixel"}
                )

            self.assertEqual(screenshot.data["scale"], 2.0)
            self.assertEqual(screenshot.attachments[0]["type"], "image")
            self.assertEqual(screenshot.attachments[0]["source"], "computer_screenshot")
            self.assertEqual(clicked.data["x"], 50)
            self.assertEqual(clicked.data["y"], 30)
            self.assertEqual(clicked.data["coordinate_space"], "screenshot_pixel")

    def test_computer_optional_int_defaults_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scripts = []

            def fake_run_osascript(script, *args, **_kwargs):
                scripts.append((script, args))
                return type(
                    "Completed",
                    (),
                    {
                        "stdout": (
                            "Computer accessibility snapshot\n"
                            "- front_app: Safari\n"
                            "- bundle_id: com.apple.Safari\n"
                            "Windows:\n"
                            "- role=AXWindow name=\"Search\" bounds=0,0,100,100"
                        )
                    },
                )()

            tools = computer_tools(
                data_dir=Path(tmp) / "var",
                workspace=Path(tmp),
                session_id="session_1",
                state=ComputerUseState(
                    enabled=lambda: True,
                    computer_learning_enabled=lambda: False,
                ),
                exposed_by_default=True,
            )
            registry = NativeToolRegistry(tools)

            with (
                patch("deepmate.tools.computer.platform.system", return_value="Darwin"),
                patch(
                    "deepmate.tools.computer._screen_bounds",
                    return_value={
                        "left": 0,
                        "top": 0,
                        "right": 100,
                        "bottom": 100,
                        "width": 100,
                        "height": 100,
                    },
                ),
                patch("deepmate.tools.computer._run_osascript", side_effect=fake_run_osascript),
            ):
                snapshot = registry.get("computer_snapshot").call({})
                clicked = registry.get("computer_click").call({"x": 10, "y": 20})

            self.assertEqual(snapshot.data["max_depth"], 1)
            self.assertEqual(snapshot.data["max_items"], 10)
            self.assertEqual(clicked.data["count"], 1)

    def test_computer_key_supports_function_and_lock_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scripts = []

            def fake_run_osascript(script, *args, **_kwargs):
                scripts.append((script, args))
                return type(
                    "Completed",
                    (),
                    {
                        "stdout": (
                            "Computer accessibility snapshot\n"
                            "- front_app: Safari\n"
                            "- bundle_id: com.apple.Safari\n"
                            "Windows:\n"
                            "- role=AXWindow name=\"Search\" bounds=0,0,100,100"
                        )
                    },
                )()

            tools = computer_tools(
                data_dir=Path(tmp) / "var",
                workspace=Path(tmp),
                session_id="session_1",
                state=ComputerUseState(
                    enabled=lambda: True,
                    computer_learning_enabled=lambda: False,
                ),
                exposed_by_default=True,
            )
            registry = NativeToolRegistry(tools)

            with (
                patch("deepmate.tools.computer.platform.system", return_value="Darwin"),
                patch("deepmate.tools.computer._run_osascript", side_effect=fake_run_osascript),
            ):
                registry.get("computer_snapshot").call({})
                f1 = registry.get("computer_key").call({"key": "f1"})
                caps = registry.get("computer_key").call({"key": "capslock"})

            self.assertEqual(f1.data["key"], "f1")
            self.assertEqual(caps.data["key"], "capslock")
            self.assertTrue(any("key code 122" in script for script, _args in scripts))
            self.assertTrue(any("key code 57" in script for script, _args in scripts))

    def test_computer_private_path_detection_resolves_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            outside = root / "private.txt"
            workspace.mkdir()
            outside.write_text("secret", encoding="utf-8")

            self.assertFalse(_path_is_private("notes.txt", workspace))
            self.assertTrue(_path_is_private("../private.txt", workspace))
            self.assertTrue(_path_is_private(str(outside), workspace))

    def test_computer_bounds_parser_supports_negative_multiscreen_coordinates(self) -> None:
        self.assertEqual(
            _bounds_from_text("-1920,0,1728,1117"),
            {
                "left": -1920,
                "top": 0,
                "right": 1728,
                "bottom": 1117,
                "width": 3648,
                "height": 1117,
            },
        )

    def test_computer_click_accepts_negative_coordinates_inside_multiscreen_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calls = []

            def fake_run_osascript(script, *args, **_kwargs):
                calls.append((script, args))
                return type(
                    "Completed",
                    (),
                    {
                        "stdout": (
                            "Computer accessibility snapshot\n"
                            "- front_app: Safari\n"
                            "- bundle_id: com.apple.Safari\n"
                            "Windows:\n"
                            "- role=AXWindow name=\"Search\" bounds=-100,0,50,50"
                        )
                    },
                )()

            tools = computer_tools(
                data_dir=Path(tmp) / "var",
                workspace=Path(tmp),
                session_id="session_1",
                state=ComputerUseState(
                    enabled=lambda: True,
                    computer_learning_enabled=lambda: False,
                ),
                exposed_by_default=True,
            )
            registry = NativeToolRegistry(tools)

            with (
                patch("deepmate.tools.computer.platform.system", return_value="Darwin"),
                patch(
                    "deepmate.tools.computer._screen_bounds",
                    return_value={
                        "left": -1920,
                        "top": 0,
                        "right": 1728,
                        "bottom": 1117,
                        "width": 3648,
                        "height": 1117,
                    },
                ),
                patch("deepmate.tools.computer._run_osascript", side_effect=fake_run_osascript),
            ):
                registry.get("computer_snapshot").call({})
                clicked = registry.get("computer_click").call({"x": -100, "y": 20})

            self.assertEqual(clicked.data["x"], -100)
            self.assertEqual(clicked.data["y"], 20)
            self.assertTrue(calls)

    def test_computer_snapshot_protects_sensitive_apps_and_redacts_sensitive_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = computer_tools(
                data_dir=Path(tmp) / "var",
                workspace=Path(tmp),
                session_id="session_1",
                state=ComputerUseState(
                    enabled=lambda: True,
                    computer_learning_enabled=lambda: False,
                ),
                exposed_by_default=True,
            )
            registry = NativeToolRegistry(tools)

            def sensitive_app(_script, *_args, **_kwargs):
                return type(
                    "Completed",
                    (),
                    {
                        "stdout": (
                            "Computer accessibility snapshot\n"
                            "- front_app: Messages\n"
                            "- bundle_id: com.apple.MobileSMS"
                        )
                    },
                )()

            with (
                patch("deepmate.tools.computer.platform.system", return_value="Darwin"),
                patch("deepmate.tools.computer._run_osascript", side_effect=sensitive_app),
            ):
                protected = registry.get("computer_snapshot").call({})

            self.assertTrue(protected.data["protected"])
            self.assertIn("protected", protected.content)
            self.assertIn("risk=local_private_access", protected.refs)

            def sensitive_field(_script, *_args, **_kwargs):
                return type(
                    "Completed",
                    (),
                    {
                        "stdout": (
                            "Computer accessibility snapshot\n"
                            "- front_app: Safari\n"
                            "- bundle_id: com.apple.Safari\n"
                            "Windows:\n"
                            "- role=AXTextField name=\"API key\" value=\"sk-secret\" bounds=1,2,3,4"
                        )
                    },
                )()

            with (
                patch("deepmate.tools.computer.platform.system", return_value="Darwin"),
                patch("deepmate.tools.computer._run_osascript", side_effect=sensitive_field),
            ):
                redacted = registry.get("computer_snapshot").call({})

            self.assertIn('name="[redacted]"', redacted.content)
            self.assertIn('value="[redacted]"', redacted.content)
            self.assertNotIn("sk-secret", redacted.content)

    def test_computer_action_policy_requires_approval_separate_from_workspace_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = NativeToolRegistry(
                computer_tools(
                    data_dir=Path(tmp) / "var",
                    workspace=Path(tmp),
                    session_id="session_1",
                    state=ComputerUseState(
                        enabled=lambda: True,
                        computer_learning_enabled=lambda: False,
                    ),
                    exposed_by_default=True,
                )
            ).get("computer_click")
            self.assertIsNotNone(tool)
            decision = ToolAccessPolicy(ToolAccessMode.WORKSPACE_WRITE).check_native_tool(
                tool,
                {"x": 10, "y": 20},
            )
            approvals = []
            approved = ToolAccessPolicy(
                approval_callback=lambda approved_tool, approved_decision: approvals.append(
                    (approved_tool.name, approved_decision.refs)
                )
                or True
            ).check_native_tool(tool, {"x": 10, "y": 20})

            self.assertFalse(decision.allowed)
            self.assertTrue(decision.requires_approval)
            self.assertTrue(any(ref == "risk=computer_action" for ref in decision.refs))
            self.assertTrue(any(ref == "approval_key=computer:click" for ref in decision.refs))
            self.assertTrue(approved.allowed)
            self.assertEqual(approvals[0][0], "computer_click")

    def test_workspace_behavior_rule_ids_include_workspace_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            store = BehaviorRuleStore.in_data_dir(root / "var", "default")
            text = "这个项目先跑单元测试"

            first_rule = store.upsert(
                BehaviorRule(
                    rule_id="",
                    text=text,
                    scope="workspace",
                    workspace_hash=workspace_hash(first),
                )
            )
            second_rule = store.upsert(
                BehaviorRule(
                    rule_id="",
                    text=text,
                    scope="workspace",
                    workspace_hash=workspace_hash(second),
                )
            )

            self.assertNotEqual(first_rule.rule_id, second_rule.rule_id)
            self.assertEqual(len(store.enabled_rules()), 2)
            first_matches = match_behavior_rules(
                store.enabled_rules(),
                "这个项目测试怎么跑",
                workspace_hash_value=workspace_hash(first),
            )
            self.assertIn(first_rule.rule_id, {rule.rule_id for rule in first_matches})
            self.assertNotIn(second_rule.rule_id, {rule.rule_id for rule in first_matches})

    def test_behavior_off_does_not_write_learning_evidence_preview(self) -> None:
        with _workspace() as workspace:
            data_dir = workspace / "var"
            profile = ProfileRef(name="default", uri="profiles/default")
            behavior = behavior_runtime_for_session(
                data_dir=data_dir,
                workspace=workspace,
                profile=profile,
                session_id="session_1",
                interaction_learning_enabled=False,
            )
            runtime = start_session_runtime(
                start_runtime_activation(
                    session_id="session_1",
                    workspace=workspace,
                    profile=profile,
                ),
                behavior_runtime=behavior,
            )
            provider = _StubProvider([ModelResponse(content="final secret")])

            runtime.run_user_turn(
                provider=provider,
                messages=(Message(role=MessageRole.USER, content="prompt secret"),),
                model="stub-model",
            )
            trace_path = next((data_dir / "behavior" / "activity").glob("*.jsonl"))
            record = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[-1])

            self.assertFalse(record["evidence_enabled"])
            self.assertNotIn("prompt_preview", record)
            self.assertNotIn("final_preview", record)
            self.assertEqual(record["prompt_chars"], len("prompt secret"))

    def test_behavior_profile_switch_disables_computer_use_by_default(self) -> None:
        with _workspace() as workspace:
            data_dir = workspace / "var"
            profile = ProfileRef(name="default", uri="profiles/default")
            behavior = behavior_runtime_for_session(
                data_dir=data_dir,
                workspace=workspace,
                profile=profile,
                session_id="session_1",
                computer_use_enabled=True,
            )

            switched = behavior.with_profile(
                workspace=workspace,
                profile=profile,
                session_id="session_2",
            )

            self.assertTrue(behavior.computer_use_enabled)
            self.assertFalse(switched.computer_use_enabled)

    def test_tui_behavior_and_computer_commands_update_runtime(self) -> None:
        with _workspace() as workspace:
            state = _tui_state(workspace)
            behavior = behavior_runtime_for_session(
                data_dir=workspace / "var",
                workspace=workspace,
                profile=state.session.profile,
                session_id=state.session.session_id,
            )
            state.behavior_runtime = behavior
            state.runtime = state.runtime.with_behavior_runtime(behavior)
            state.native_tools = NativeToolRegistry(
                computer_tools(
                    data_dir=workspace / "var",
                    workspace=workspace,
                    session_id=state.session.session_id,
                    state=ComputerUseState(
                        enabled=lambda: behavior.computer_use_enabled,
                        computer_learning_enabled=(
                            lambda: behavior.settings.computer_learning_enabled
                        ),
                    ),
                )
            )

            off = handle_tui_command("/behavior off", state)
            on = handle_tui_command("/computer on", state)
            learning = handle_tui_command("/computer learning on", state)

            self.assertTrue(off.handled)
            self.assertFalse(behavior.settings.interaction_learning_enabled)
            self.assertTrue(on.handled)
            self.assertTrue(behavior.computer_use_enabled)
            self.assertTrue(learning.handled)
            self.assertFalse(behavior.settings.computer_learning_enabled)
            self.assertIn(
                "computer_screenshot",
                {str(schema.get("name")) for schema in state.tool_schemas},
            )
            self.assertIn(
                "computer_snapshot",
                {str(schema.get("name")) for schema in state.tool_schemas},
            )
            self.assertTrue(
                set(COMPUTER_TOOL_NAMES).issubset(
                    {str(schema.get("name")) for schema in state.tool_schemas}
                )
            )

    def test_legacy_interactive_computer_command_refreshes_schemas(self) -> None:
        with _workspace() as workspace:
            data_dir = workspace / "var"
            profile = ProfileRef(name="default", uri="profiles/default")
            behavior = behavior_runtime_for_session(
                data_dir=data_dir,
                workspace=workspace,
                profile=profile,
                session_id="session_1",
            )
            runtime = start_session_runtime(
                start_runtime_activation(
                    session_id="session_1",
                    workspace=workspace,
                    profile=profile,
                ),
                behavior_runtime=behavior,
            )
            registry = NativeToolRegistry(
                computer_tools(
                    data_dir=data_dir,
                    workspace=workspace,
                    session_id="session_1",
                    state=ComputerUseState(
                        enabled=lambda: behavior.computer_use_enabled,
                        computer_learning_enabled=(
                            lambda: behavior.settings.computer_learning_enabled
                        ),
                    ),
                )
            )

            with redirect_stdout(StringIO()):
                enabled = _handle_computer_command(
                    "/computer on",
                    runtime,
                    native_tools=registry,
                    tool_schemas=(),
                )
            self.assertTrue(behavior.computer_use_enabled)
            self.assertTrue(
                set(COMPUTER_TOOL_NAMES).issubset(
                    {str(schema.get("name")) for schema in enabled or ()}
                )
            )

            with redirect_stdout(StringIO()):
                disabled = _handle_computer_command(
                    "/computer off",
                    runtime,
                    native_tools=registry,
                    tool_schemas=enabled or (),
                )
            self.assertFalse(behavior.computer_use_enabled)
            self.assertFalse(
                set(COMPUTER_TOOL_NAMES)
                & {str(schema.get("name")) for schema in disabled or ()}
            )


def _workspace():
    temp = tempfile.TemporaryDirectory()
    root = Path(temp.name)
    (root / "profiles" / "default").mkdir(parents=True)
    (root / "profiles" / "default" / "identity.md").write_text("Identity.", encoding="utf-8")
    (root / "profiles" / "default" / "soul.md").write_text("Soul.", encoding="utf-8")
    (root / "profiles" / "default" / "user.md").write_text("", encoding="utf-8")
    (root / "profiles" / "default" / "memory.md").write_text("", encoding="utf-8")
    return _TempWorkspace(temp, root)


def _tui_state(workspace: Path) -> TuiRuntimeState:
    session_store = SessionStore.in_directory(workspace / "var" / "sessions")
    profile = ProfileRef(name="default", uri="profiles/default")
    session = session_store.create(workspace=workspace, profile=profile, title="tui test")
    runtime = start_session_runtime(
        start_runtime_activation(
            session_id=session.session_id,
            workspace=workspace,
            profile=profile,
        )
    )
    return TuiRuntimeState(
        provider=_StubProvider([ModelResponse(content="ok")]),
        provider_name="stub",
        provider_api_key_env="STUB_API_KEY",
        provider_api_key_available=True,
        model="stub-main",
        default_model="stub-main",
        upgrade_model="stub-pro",
        workspace=workspace,
        profile=profile,
        session_store=session_store,
        session=session,
        transcript=session_store.transcript_store(session),
        runtime=runtime,
        capability_surface=None,
        native_tools=None,
        native_tool_factory=None,
        mcp_tools=None,
        subagents=None,
        tool_access_policy=ToolAccessPolicy(ToolAccessMode.READ_ONLY),
        tool_schemas=(),
        selected_skill_documents=(),
        mcp_servers=(),
        conversation_budget_policy=None,
        provider_retry_policy=None,
        options={},
        max_steps=2,
        trace_recorder=_TraceRecorder(),
        warning_sink=None,
        data_dir=workspace / "var",
    )


class _TempWorkspace:
    def __init__(self, temp: tempfile.TemporaryDirectory, root: Path) -> None:
        self.temp = temp
        self.root = root

    def __enter__(self) -> Path:
        return self.root

    def __exit__(self, exc_type, exc, tb) -> None:
        self.temp.cleanup()


if __name__ == "__main__":
    unittest.main()
