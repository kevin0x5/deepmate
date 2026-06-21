from __future__ import annotations

import struct
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from deepmate.app import AppSettings, ProviderSettings, RemoteSettings, WeComRemoteSettings
from deepmate.channels.interactive import _close_remote_routes_for_local_turn, _handle_command
from deepmate.channels.wecom import WeComChannel, WeComInboundMessage, WeComRunDependencies
from deepmate.channels.wecom.channel import parse_wecom_message
from deepmate.channels.wecom.client import (
    MAX_WEBSOCKET_FRAME_BYTES,
    WeComClientConfig,
    WeComProtocolError,
    WeComWsClient,
)
from deepmate.channels.remote import RemoteBindingStore
from deepmate.domain import ProfileRef
from deepmate.providers import ModelResponse, ModelToolRequest
from deepmate.runtime import ToolAccessMode, ToolAccessPolicy
from deepmate.runtime.safety import SessionApprovalCache, ToolSafetyPolicy
from deepmate.runtime.wakelock import WakeBackend
from deepmate.storage import SessionStore
from deepmate.tools import NativeTool, NativeToolRegistry, NativeToolResult
from deepmate.trace import TraceRecorder


class StubProvider:
    def __init__(self, response: ModelResponse) -> None:
        self.response = response
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return self.response


class SequenceProvider:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("sequence provider received too many requests")
        return self.responses.pop(0)


class BlockingProvider:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        self.entered.set()
        self.release.wait(2)
        return ModelResponse(content="blocked done")


class FakeWakeSession:
    def __init__(self) -> None:
        self.started = 0
        self.finished = 0

    def start(self) -> None:
        self.started += 1

    def finish_turn(self) -> None:
        self.finished += 1


class RemoteWeComTests(unittest.TestCase):
    def test_wecom_ws_client_reassembles_fragmented_json_message(self) -> None:
        client = WeComWsClient(WeComClientConfig(bot_id="bot", secret="secret"))
        client._sock = _FakeSocket(
            _server_frame(b'{"cmd":"aibot_msg_callback","content":"he', fin=False)
            + _server_frame(b'llo","user_id":"alice","req_id":"req_1"}', opcode=0x0)
        )

        payload = client.recv_json()

        self.assertEqual(payload["content"], "hello")
        self.assertEqual(payload["req_id"], "req_1")

    def test_wecom_ws_client_recv_json_timeout_restores_socket_timeout(self) -> None:
        client = WeComWsClient(WeComClientConfig(bot_id="bot", secret="secret"))
        sock = _FakeSocket(_server_frame(b'{"cmd":"message","content":"ok"}'))
        sock.timeout = 12
        client._sock = sock

        payload = client.recv_json_timeout(0.1)

        self.assertEqual(payload["content"], "ok")
        self.assertEqual(sock.timeout, 12)

    def test_wecom_ws_client_rejects_oversized_frame_before_allocating_payload(self) -> None:
        client = WeComWsClient(WeComClientConfig(bot_id="bot", secret="secret"))
        length = MAX_WEBSOCKET_FRAME_BYTES + 1
        client._sock = _FakeSocket(bytes([0x81, 127]) + struct.pack("!Q", length))

        with self.assertRaisesRegex(WeComProtocolError, "too large"):
            client.recv_json()

    def test_parse_wecom_message_requires_real_reply_id(self) -> None:
        message = parse_wecom_message(
            {
                "cmd": "aibot_msg_callback",
                "content": "hello",
                "user_id": "alice",
            }
        )

        self.assertIsNone(message)

    def test_wecom_allowed_users_empty_denies_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = _settings(root)
            settings = replace(
                base,
                remote=RemoteSettings(
                    wecom=replace(base.remote.wecom, allowed_users=())
                ),
            )
            session_store = SessionStore.in_directory(settings.data_dir / "sessions")
            sent = []
            channel = WeComChannel(
                WeComRunDependencies(
                    settings=settings,
                    provider=StubProvider(ModelResponse(content="unused")),
                    model="stub-model",
                    session_store=session_store,
                    trace_recorder=TraceRecorder(_TraceSink()),
                ),
                sender=sent.append,
                wake_factory=lambda reason: FakeWakeSession(),
            )

            channel.handle_message(
                WeComInboundMessage("req_1", "mallory", "chat_1", "single", "hello")
            )

        self.assertIn("not enabled", "\n".join(_sent_text(item) for item in sent))

    def test_wecom_allowed_users_star_allows_any_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = _settings(root)
            settings = replace(
                base,
                remote=RemoteSettings(
                    wecom=replace(base.remote.wecom, allowed_users=("*",))
                ),
            )
            session_store = SessionStore.in_directory(settings.data_dir / "sessions")
            sent = []
            channel = WeComChannel(
                WeComRunDependencies(
                    settings=settings,
                    provider=StubProvider(ModelResponse(content="done")),
                    model="stub-model",
                    session_store=session_store,
                    trace_recorder=TraceRecorder(_TraceSink()),
                ),
                sender=sent.append,
                wake_factory=lambda reason: FakeWakeSession(),
            )

            channel.handle_message(
                WeComInboundMessage("req_1", "mallory", "chat_1", "single", "hello")
            )

        self.assertIn("done", "\n".join(_sent_content(item) for item in sent))

    def test_wecom_group_policy_deny_blocks_group_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = _settings(root)
            settings = replace(
                base,
                remote=RemoteSettings(
                    wecom=replace(base.remote.wecom, group_policy="deny")
                ),
            )
            session_store = SessionStore.in_directory(settings.data_dir / "sessions")
            sent = []
            channel = WeComChannel(
                WeComRunDependencies(
                    settings=settings,
                    provider=StubProvider(ModelResponse(content="unused")),
                    model="stub-model",
                    session_store=session_store,
                    trace_recorder=TraceRecorder(_TraceSink()),
                ),
                sender=sent.append,
                wake_factory=lambda reason: FakeWakeSession(),
            )

            channel.handle_message(
                WeComInboundMessage("req_1", "alice", "chat_1", "group", "hello")
            )

        self.assertIn("disabled in Enterprise WeChat group chats", _sent_text(sent[-1]))

    def test_wecom_group_policy_readonly_denies_write_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _settings(root)
            session_store = SessionStore.in_directory(settings.data_dir / "sessions")
            marker = {"called": False}

            def handle(arguments):
                marker["called"] = True
                return NativeToolResult(content="write ok")

            tools = NativeToolRegistry(
                (
                    NativeTool(
                        name="write_marker",
                        description="Write marker.",
                        input_schema={"type": "object"},
                        handler=handle,
                        read_only=False,
                    ),
                )
            )
            provider = SequenceProvider(
                [
                    ModelResponse(
                        content="need write",
                        tool_requests=(
                            ModelToolRequest(name="write_marker", id="call_1"),
                        ),
                    ),
                    ModelResponse(content="done"),
                ]
            )
            sent = []
            channel = WeComChannel(
                WeComRunDependencies(
                    settings=settings,
                    provider=provider,
                    model="stub-model",
                    session_store=session_store,
                    trace_recorder=TraceRecorder(_TraceSink()),
                    native_tools=tools,
                    tool_schemas=tools.schemas(),
                    tool_access_policy=ToolAccessPolicy(ToolAccessMode.READ_ONLY),
                ),
                sender=sent.append,
                wake_factory=lambda reason: FakeWakeSession(),
            )

            channel.handle_message(
                WeComInboundMessage("req_1", "alice", "chat_1", "group", "执行写入")
            )

        self.assertFalse(marker["called"])
        self.assertIn("readonly", "\n".join(_sent_text(item) for item in sent))

    def test_wecom_long_reply_is_split_by_utf8_bytes_without_truncating(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = _settings(root)
            settings = replace(
                base,
                remote=RemoteSettings(
                    wecom=replace(
                        base.remote.wecom,
                        max_messages_per_minute=100,
                    )
                ),
            )
            session_store = SessionStore.in_directory(settings.data_dir / "sessions")
            long_reply = "中文测试" * 2200
            sent = []
            channel = WeComChannel(
                WeComRunDependencies(
                    settings=settings,
                    provider=StubProvider(ModelResponse(content=long_reply)),
                    model="stub-model",
                    session_store=session_store,
                    trace_recorder=TraceRecorder(_TraceSink()),
                ),
                sender=sent.append,
                wake_factory=lambda reason: FakeWakeSession(),
            )

            channel.handle_message(
                WeComInboundMessage("req_1", "alice", "chat_1", "single", "hello")
            )

        reply_chunks = [
            _sent_content(item)
            for item in sent
            if "任务已完成" in _sent_content(item) or _sent_content(item).startswith("[")
        ]
        self.assertGreater(len(reply_chunks), 1)
        stream_payloads = [item for item in sent if item.get("msgtype") == "stream"]
        stream_ids = {_sent_stream_id(item) for item in stream_payloads}
        self.assertEqual(len(stream_ids), 1)
        self.assertEqual(
            [index for index, item in enumerate(stream_payloads) if _sent_stream_finish(item)],
            [len(stream_payloads) - 1],
        )
        for payload in sent:
            text = _sent_content(payload)
            if text:
                if payload.get("msgtype") == "stream":
                    self.assertLessEqual(len(text.encode("utf-8")), 20000)
                else:
                    self.assertLessEqual(len(text.encode("utf-8")), 1800)
        self.assertIn("中文测试", "".join(reply_chunks))

    def test_wecom_turn_reply_uses_stream_msgtype(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _settings(root)
            session_store = SessionStore.in_directory(settings.data_dir / "sessions")
            sent = []
            channel = WeComChannel(
                WeComRunDependencies(
                    settings=settings,
                    provider=StubProvider(ModelResponse(content="```python\nprint(1)\n```")),
                    model="stub-model",
                    session_store=session_store,
                    trace_recorder=TraceRecorder(_TraceSink()),
                ),
                sender=sent.append,
                wake_factory=lambda reason: FakeWakeSession(),
            )

            channel.handle_message(
                WeComInboundMessage("req_1", "alice", "chat_1", "single", "code")
            )

        streams = [item for item in sent if item.get("msgtype") == "stream"]
        self.assertGreaterEqual(len(streams), 2)
        self.assertFalse(_sent_stream_finish(streams[0]))
        self.assertTrue(_sent_stream_finish(streams[-1]))
        self.assertEqual(_sent_stream_id(streams[0]), _sent_stream_id(streams[-1]))
        self.assertIn("```python", _sent_stream(streams[-1]))

    def test_wecom_final_stream_records_passive_reply_window_risk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _settings(root)
            session_store = SessionStore.in_directory(settings.data_dir / "sessions")
            sent = []
            sink = _TraceSink()
            channel = WeComChannel(
                WeComRunDependencies(
                    settings=settings,
                    provider=StubProvider(ModelResponse(content="unused")),
                    model="stub-model",
                    session_store=session_store,
                    trace_recorder=TraceRecorder(sink),
                ),
                sender=sent.append,
                wake_factory=lambda reason: FakeWakeSession(),
            )
            message = WeComInboundMessage(
                "req_1",
                "alice",
                "chat_1",
                "single",
                "hello",
            )
            state = channel._state_for_message(
                message,
                first_prompt="hello",
                open_route=True,
            )
            with state.lock:
                state.progress_started_at = time.monotonic() - 10

            channel._send_stream_chunk(
                message,
                state,
                stream_id="dm-req_1",
                text="done",
                finish=True,
            )

        self.assertEqual(sent[-1]["req_id"], "req_1")
        self.assertIn(
            "wecom_passive_reply_window_exceeded",
            tuple(event.kind for event in sink.events),
        )

    def test_wecom_tail_keeps_progress_log_after_final_reply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _settings(root)
            session_store = SessionStore.in_directory(settings.data_dir / "sessions")
            sent = []
            channel = WeComChannel(
                WeComRunDependencies(
                    settings=settings,
                    provider=StubProvider(ModelResponse(content="final answer")),
                    model="stub-model",
                    session_store=session_store,
                    trace_recorder=TraceRecorder(_TraceSink()),
                ),
                sender=sent.append,
                wake_factory=lambda reason: FakeWakeSession(),
            )

            channel.handle_message(
                WeComInboundMessage("req_1", "alice", "chat_1", "single", "处理一下")
            )
            channel.handle_message(
                WeComInboundMessage("req_2", "alice", "chat_1", "single", "/tail")
            )

        self.assertIn("user: 处理一下", _sent_text(sent[-1]))
        self.assertIn("assistant: final answer", _sent_text(sent[-1]))

    def test_binding_store_round_trips_session_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RemoteBindingStore.in_data_dir(root / "var")
            session_store = SessionStore.in_directory(root / "var" / "sessions")
            session = session_store.create(
                root,
                ProfileRef(name="default", uri="profiles/default"),
                "Current task",
            )

            record = store.bind_session(
                channel="wecom",
                remote_user_id="*",
                session=session,
                bound_from="test",
            )
            loaded = store.get("wecom", "*")

            self.assertIsNotNone(loaded)
            self.assertEqual(loaded, record)
            self.assertEqual(loaded.session_id, session.session_id)

    def test_wecom_first_message_creates_remote_session_and_runs_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _settings(root)
            session_store = SessionStore.in_directory(settings.data_dir / "sessions")
            provider = StubProvider(ModelResponse(content="done"))
            sent = []
            wake = FakeWakeSession()
            channel = WeComChannel(
                WeComRunDependencies(
                    settings=settings,
                    provider=provider,
                    model="stub-model",
                    session_store=session_store,
                    trace_recorder=TraceRecorder(_TraceSink()),
                ),
                sender=sent.append,
                wake_factory=lambda reason: wake,
            )

            channel.handle_message(
                WeComInboundMessage(
                    req_id="req_1",
                    user_id="alice",
                    chat_id="chat_1",
                    chat_type="single",
                    content="@deepmate 帮我看 git status",
                )
            )

            bindings = RemoteBindingStore.in_data_dir(settings.data_dir).list_records()
            self.assertEqual(len(bindings), 1)
            self.assertEqual(bindings[0].remote_user_id, "alice")
            self.assertTrue(bindings[0].route_open)
            self.assertTrue(provider.requests)
            self.assertEqual(wake.started, 1)
            self.assertEqual(wake.finished, 1)
            self.assertTrue(any("企业微信接管已开启" in _sent_text(item) for item in sent))
            self.assertTrue(any("done" in _sent_content(item) for item in sent))

    def test_current_command_reports_existing_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _settings(root)
            session_store = SessionStore.in_directory(settings.data_dir / "sessions")
            session = session_store.create(
                root,
                settings.profile_ref(),
                "Bound task",
            )
            RemoteBindingStore.in_data_dir(settings.data_dir).bind_session(
                channel="wecom",
                remote_user_id="alice",
                session=session,
                bound_from="test",
            )
            sent = []
            channel = WeComChannel(
                WeComRunDependencies(
                    settings=settings,
                    provider=StubProvider(ModelResponse(content="unused")),
                    model="stub-model",
                    session_store=session_store,
                    trace_recorder=TraceRecorder(_TraceSink()),
                ),
                sender=sent.append,
                wake_factory=lambda reason: FakeWakeSession(),
            )

            channel.handle_message(
                WeComInboundMessage(
                    req_id="req_2",
                    user_id="alice",
                    chat_id="chat_1",
                    chat_type="single",
                    content="/current",
                )
            )

            self.assertIn("Bound task", "\n".join(_sent_text(item) for item in sent))
            record = RemoteBindingStore.in_data_dir(settings.data_dir).get(
                "wecom",
                "alice",
            )
            self.assertFalse(record.route_open)

    def test_status_and_tail_report_remote_ownership_and_recent_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _settings(root)
            session_store = SessionStore.in_directory(settings.data_dir / "sessions")
            sent = []
            channel = WeComChannel(
                WeComRunDependencies(
                    settings=settings,
                    provider=StubProvider(ModelResponse(content="done")),
                    model="stub-model",
                    session_store=session_store,
                    trace_recorder=TraceRecorder(_TraceSink()),
                ),
                sender=sent.append,
                wake_factory=lambda reason: FakeWakeSession(),
            )

            channel.handle_message(
                WeComInboundMessage("req_1", "alice", "chat_1", "single", "处理一下")
            )
            channel.handle_message(
                WeComInboundMessage("req_2", "alice", "chat_1", "single", "/status")
            )
            channel.handle_message(
                WeComInboundMessage("req_3", "alice", "chat_1", "single", "/tail")
            )

        output = "\n".join(_sent_content(item) for item in sent)
        self.assertIn("接管：已开启", output)
        self.assertIn("最近状态：已完成", output)
        self.assertIn("done", output)

    def test_remote_deploy_create_runs_from_wecom(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _settings(root)
            session_store = SessionStore.in_directory(settings.data_dir / "sessions")
            sent = []
            calls = []

            def fake_deploy(command, **kwargs):
                calls.append((command, kwargs))
                return "预览链接已创建：https://preview.example"

            channel = WeComChannel(
                WeComRunDependencies(
                    settings=settings,
                    provider=StubProvider(ModelResponse(content="unused")),
                    model="stub-model",
                    session_store=session_store,
                    trace_recorder=TraceRecorder(_TraceSink()),
                ),
                sender=sent.append,
                wake_factory=lambda reason: FakeWakeSession(),
            )

            with patch(
                "deepmate.channels.wecom.channel.handle_deploy_command",
                side_effect=fake_deploy,
            ):
                channel.handle_message(
                    WeComInboundMessage(
                        "req_1",
                        "alice",
                        "chat_1",
                        "single",
                        "/deploy report.html",
                    )
                )

        output = "\n".join(_sent_content(item) for item in sent)
        self.assertIn("预览链接已创建", output)
        self.assertEqual(calls[0][0], "/deploy report.html")
        self.assertEqual(calls[0][1]["workspace"], settings.workspace)
        self.assertEqual(calls[0][1]["data_dir"], settings.data_dir)
        self.assertTrue(calls[0][1]["owner_session_id"])

    def test_running_wecom_commands_do_not_become_followups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _settings(root)
            session_store = SessionStore.in_directory(settings.data_dir / "sessions")
            provider = BlockingProvider()
            sent = []
            channel = WeComChannel(
                WeComRunDependencies(
                    settings=settings,
                    provider=provider,
                    model="stub-model",
                    session_store=session_store,
                    trace_recorder=TraceRecorder(_TraceSink()),
                ),
                sender=sent.append,
                wake_factory=lambda reason: FakeWakeSession(),
            )

            channel.handle_message(
                WeComInboundMessage("req_1", "alice", "chat_1", "single", "开始长任务"),
                async_turn=True,
            )
            self.assertTrue(provider.entered.wait(1))
            channel.handle_message(
                WeComInboundMessage("req_2", "alice", "chat_1", "single", "/status")
            )
            channel.handle_message(
                WeComInboundMessage("req_3", "alice", "chat_1", "single", "补充要求")
            )
            provider.release.set()
            self.assertTrue(_wait_for_sent_text(sent, "blocked done", timeout=2.5))
            self.assertTrue(_wait_for_idle(channel, timeout=1.0))

        output = "\n".join(_sent_content(item) for item in sent)
        self.assertIn("任务状态：运行中", output)
        self.assertIn("补充要求：0 条", output)
        self.assertIn("已收到补充要求", output)
        self.assertIn("待处理补充要求：1 条", output)

    def test_long_running_turn_sends_heartbeat_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _settings(root)
            settings = _with_fast_heartbeat(settings)
            session_store = SessionStore.in_directory(settings.data_dir / "sessions")
            provider = BlockingProvider()
            sent = []
            channel = WeComChannel(
                WeComRunDependencies(
                    settings=settings,
                    provider=provider,
                    model="stub-model",
                    session_store=session_store,
                    trace_recorder=TraceRecorder(_TraceSink()),
                ),
                sender=sent.append,
                wake_factory=lambda reason: FakeWakeSession(),
            )

            channel.handle_message(
                WeComInboundMessage("req_1", "alice", "chat_1", "single", "开始长任务"),
                async_turn=True,
            )
            self.assertTrue(provider.entered.wait(1))
            self.assertTrue(_wait_for_sent_text(sent, "任务仍在执行。", timeout=2.5))
            provider.release.set()
            self.assertTrue(_wait_for_sent_text(sent, "blocked done", timeout=2.5))

        output = "\n".join(_sent_content(item) for item in sent)
        self.assertIn("任务仍在执行。", output)
        self.assertIn("最近状态：等待模型或工具返回", output)
        streams = [item for item in sent if item.get("msgtype") == "stream"]
        self.assertGreaterEqual(len(streams), 3)
        self.assertEqual(len({_sent_stream_id(item) for item in streams}), 1)
        self.assertFalse(_sent_stream_finish(streams[0]))
        self.assertFalse(_sent_stream_finish(streams[1]))
        self.assertTrue(_sent_stream_finish(streams[-1]))

    def test_heartbeat_is_suppressed_while_remote_approval_is_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _with_fast_heartbeat(_settings(root))
            settings = replace(
                settings,
                remote=replace(
                    settings.remote,
                    wecom=replace(settings.remote.wecom, approval_timeout_seconds=3),
                ),
            )
            session_store = SessionStore.in_directory(settings.data_dir / "sessions")

            def handle(arguments):
                return NativeToolResult(content="write ok")

            tools = NativeToolRegistry(
                (
                    NativeTool(
                        name="write_marker",
                        description="Write marker.",
                        input_schema={"type": "object"},
                        handler=handle,
                        read_only=False,
                    ),
                )
            )
            provider = SequenceProvider(
                [
                    ModelResponse(
                        content="need write",
                        tool_requests=(
                            ModelToolRequest(name="write_marker", id="call_1"),
                        ),
                    ),
                    ModelResponse(content="denied done"),
                ]
            )
            sent = []
            channel = WeComChannel(
                WeComRunDependencies(
                    settings=settings,
                    provider=provider,
                    model="stub-model",
                    session_store=session_store,
                    trace_recorder=TraceRecorder(_TraceSink()),
                    native_tools=tools,
                    tool_schemas=tools.schemas(),
                    tool_access_policy=ToolAccessPolicy(ToolAccessMode.READ_ONLY),
                ),
                sender=sent.append,
                wake_factory=lambda reason: FakeWakeSession(),
            )

            channel.handle_message(
                WeComInboundMessage("req_1", "alice", "chat_1", "single", "执行写入"),
                async_turn=True,
            )
            self.assertTrue(_wait_for_sent_text(sent, "需要远程审批", timeout=1.5))
            time.sleep(1.3)
            output_before_deny = "\n".join(_sent_content(item) for item in sent)
            self.assertNotIn("任务仍在执行。", output_before_deny)
            channel.handle_message(
                WeComInboundMessage("req_2", "alice", "chat_1", "single", "/deny")
            )
            self.assertTrue(_wait_for_sent_text(sent, "已拒绝这次操作", timeout=1.5))
            self.assertTrue(_wait_for_sent_text(sent, "denied done", timeout=2.5))

    def test_remote_close_stops_heartbeat_and_pending_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _with_fast_heartbeat(_settings(root))
            settings = replace(
                settings,
                remote=replace(
                    settings.remote,
                    wecom=replace(settings.remote.wecom, approval_timeout_seconds=3),
                ),
            )
            session_store = SessionStore.in_directory(settings.data_dir / "sessions")

            def handle(arguments):
                return NativeToolResult(content="write ok")

            tools = NativeToolRegistry(
                (
                    NativeTool(
                        name="write_marker",
                        description="Write marker.",
                        input_schema={"type": "object"},
                        handler=handle,
                        read_only=False,
                    ),
                )
            )
            provider = SequenceProvider(
                [
                    ModelResponse(
                        content="need write",
                        tool_requests=(
                            ModelToolRequest(name="write_marker", id="call_1"),
                        ),
                    ),
                    ModelResponse(content="closed done"),
                ]
            )
            sent = []
            channel = WeComChannel(
                WeComRunDependencies(
                    settings=settings,
                    provider=provider,
                    model="stub-model",
                    session_store=session_store,
                    trace_recorder=TraceRecorder(_TraceSink()),
                    native_tools=tools,
                    tool_schemas=tools.schemas(),
                    tool_access_policy=ToolAccessPolicy(ToolAccessMode.READ_ONLY),
                ),
                sender=sent.append,
                wake_factory=lambda reason: FakeWakeSession(),
            )

            channel.handle_message(
                WeComInboundMessage("req_1", "alice", "chat_1", "single", "执行写入"),
                async_turn=True,
            )
            self.assertTrue(_wait_for_sent_text(sent, "需要远程审批", timeout=1.5))
            channel.handle_message(
                WeComInboundMessage("req_2", "alice", "chat_1", "single", "/close")
            )
            self.assertTrue(
                _wait_for_sent_text(
                    sent,
                    "已关闭当前企业微信接管",
                    timeout=1.5,
                )
            )
            self.assertTrue(_wait_for_sent_text(sent, "closed done", timeout=2.5))

        output = "\n".join(_sent_content(item) for item in sent)
        self.assertNotIn("任务仍在执行。", output)

    def test_remote_approval_allows_one_workspace_write_tool_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _settings(root)
            settings = replace(
                settings,
                remote=replace(
                    settings.remote,
                    wecom=replace(settings.remote.wecom, approval_timeout_seconds=2),
                ),
            )
            session_store = SessionStore.in_directory(settings.data_dir / "sessions")
            marker = {"called": False}

            def handle(arguments):
                marker["called"] = True
                return NativeToolResult(content="write ok")

            tools = NativeToolRegistry(
                (
                    NativeTool(
                        name="write_marker",
                        description="Write marker.",
                        input_schema={"type": "object"},
                        handler=handle,
                        read_only=False,
                    ),
                )
            )
            provider = SequenceProvider(
                [
                    ModelResponse(
                        content="need write",
                        tool_requests=(
                            ModelToolRequest(name="write_marker", id="call_1"),
                        ),
                    ),
                    ModelResponse(content="approved done"),
                ]
            )
            sent = []
            channel = WeComChannel(
                WeComRunDependencies(
                    settings=settings,
                    provider=provider,
                    model="stub-model",
                    session_store=session_store,
                    trace_recorder=TraceRecorder(_TraceSink()),
                    native_tools=tools,
                    tool_schemas=tools.schemas(),
                    tool_access_policy=ToolAccessPolicy(ToolAccessMode.READ_ONLY),
                ),
                sender=sent.append,
                wake_factory=lambda reason: FakeWakeSession(),
            )

            channel.handle_message(
                WeComInboundMessage("req_1", "alice", "chat_1", "single", "执行写入"),
                async_turn=True,
            )
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline and not any(
                "需要远程审批" in _sent_text(item) for item in sent
            ):
                time.sleep(0.01)
            channel.handle_message(
                WeComInboundMessage("req_2", "alice", "chat_1", "single", "/approve")
            )
            self.assertTrue(_wait_for_condition(lambda: marker["called"], timeout=1.5))
            self.assertTrue(_wait_for_sent_text(sent, "approved done", timeout=2.5))
            self.assertTrue(_wait_for_idle(channel, timeout=1.0))

        output = "\n".join(_sent_content(item) for item in sent)
        self.assertTrue(marker["called"])
        self.assertIn("需要远程审批", output)
        self.assertIn("已允许这次操作", output)
        self.assertIn("approved done", output)

    def test_remote_status_refreshes_route_closed_by_local_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _settings(root)
            session_store = SessionStore.in_directory(settings.data_dir / "sessions")
            sent = []
            channel = WeComChannel(
                WeComRunDependencies(
                    settings=settings,
                    provider=StubProvider(ModelResponse(content="done")),
                    model="stub-model",
                    session_store=session_store,
                    trace_recorder=TraceRecorder(_TraceSink()),
                ),
                sender=sent.append,
                wake_factory=lambda reason: FakeWakeSession(),
            )

            channel.handle_message(
                WeComInboundMessage("req_1", "alice", "chat_1", "single", "处理一下")
            )
            store = RemoteBindingStore.in_data_dir(settings.data_dir)
            record = store.get("wecom", "alice")
            self.assertTrue(record.route_open)
            store.upsert(record.with_route(open=False))
            channel.handle_message(
                WeComInboundMessage("req_2", "alice", "chat_1", "single", "/status")
            )

        output = "\n".join(_sent_text(item) for item in sent)
        self.assertIn("接管：已关闭", output)

    def test_remote_approval_is_denied_after_route_is_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _settings(root)
            settings = replace(
                settings,
                remote=replace(
                    settings.remote,
                    wecom=replace(settings.remote.wecom, approval_timeout_seconds=2),
                ),
            )
            session_store = SessionStore.in_directory(settings.data_dir / "sessions")
            marker = {"called": False}

            def handle(arguments):
                marker["called"] = True
                return NativeToolResult(content="write ok")

            tools = NativeToolRegistry(
                (
                    NativeTool(
                        name="write_marker",
                        description="Write marker.",
                        input_schema={"type": "object"},
                        handler=handle,
                        read_only=False,
                    ),
                )
            )
            provider = SequenceProvider(
                [
                    ModelResponse(
                        content="need write",
                        tool_requests=(
                            ModelToolRequest(name="write_marker", id="call_1"),
                        ),
                    ),
                    ModelResponse(content="should not write"),
                ]
            )
            sent = []
            channel = WeComChannel(
                WeComRunDependencies(
                    settings=settings,
                    provider=provider,
                    model="stub-model",
                    session_store=session_store,
                    trace_recorder=TraceRecorder(_TraceSink()),
                    native_tools=tools,
                    tool_schemas=tools.schemas(),
                    tool_access_policy=ToolAccessPolicy(ToolAccessMode.READ_ONLY),
                ),
                sender=sent.append,
                wake_factory=lambda reason: FakeWakeSession(),
            )

            channel.handle_message(
                WeComInboundMessage("req_1", "alice", "chat_1", "single", "执行写入"),
                async_turn=True,
            )
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline and not any(
                "需要远程审批" in _sent_text(item) for item in sent
            ):
                time.sleep(0.01)
            store = RemoteBindingStore.in_data_dir(settings.data_dir)
            record = store.get("wecom", "alice")
            self.assertIsNotNone(record)
            store.upsert(record.with_route(open=False))
            channel.handle_message(
                WeComInboundMessage("req_2", "alice", "chat_1", "single", "/approve")
            )
            self.assertTrue(_wait_for_sent_text(sent, "企业微信接管已关闭", timeout=1.5))
            self.assertTrue(_wait_for_idle(channel, timeout=2.5))

        output = "\n".join(_sent_text(item) for item in sent)
        self.assertFalse(marker["called"])
        self.assertIn("企业微信接管已关闭", output)

    def test_remote_safety_approval_callback_allows_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _settings(root)
            settings = replace(
                settings,
                remote=replace(
                    settings.remote,
                    wecom=replace(settings.remote.wecom, approval_timeout_seconds=2),
                ),
            )
            session_store = SessionStore.in_directory(settings.data_dir / "sessions")
            sent = []
            approval_cache = SessionApprovalCache()
            channel = WeComChannel(
                WeComRunDependencies(
                    settings=settings,
                    provider=BlockingProvider(),
                    model="stub-model",
                    session_store=session_store,
                    trace_recorder=TraceRecorder(_TraceSink()),
                    approval_cache=approval_cache,
                ),
                sender=sent.append,
                wake_factory=lambda reason: FakeWakeSession(),
            )
            channel.handle_message(
                WeComInboundMessage("req_1", "alice", "chat_1", "single", "/current")
            )
            state = next(iter(channel._states.values()))
            policy = ToolSafetyPolicy(
                workspace=root,
                shell_enabled=True,
                network_enabled=False,
                approval_cache=approval_cache,
            )
            decision = policy.check_shell_command("python3 -m pip install example")
            self.assertFalse(decision.allowed)
            self.assertTrue(decision.requires_approval)

            result = {}
            worker = threading.Thread(
                target=lambda: result.setdefault(
                    "decision",
                    channel._approve_safety_decision(
                        WeComInboundMessage(
                            "req_2",
                            "alice",
                            "chat_1",
                            "single",
                            "approval",
                        ),
                        state,
                        decision,
                    ),
                )
            )
            worker.start()
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline and not any(
                "需要远程审批" in _sent_text(item) for item in sent
            ):
                time.sleep(0.01)
            channel.handle_message(
                WeComInboundMessage("req_3", "alice", "chat_1", "single", "/approve")
            )
            worker.join(1)

        output = "\n".join(_sent_text(item) for item in sent)
        self.assertEqual(str(result.get("decision")), "allow_once")
        self.assertIn("运行命令：python3 -m pip install example", output)

    def test_interactive_remote_command_binds_current_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session_store = SessionStore.in_directory(root / "var" / "sessions")
            session = session_store.create(
                root,
                ProfileRef(name="default", uri="profiles/default"),
                "Current task",
            )

            with redirect_stdout(StringIO()):
                command = _handle_command(
                    prompt="/remote --wecom",
                    session_store=session_store,
                    session=session,
                    transcript=session_store.transcript_store(session),
                    runtime=_RuntimeStub(),
                    workspace=root,
                    profile=session.profile,
                    mcp_servers=(),
                    trace_recorder=TraceRecorder(_TraceSink()),
                    context_snapshot_factory=None,
                    capability_state_store=None,
                    checkpoint_controller_factory=None,
                    hook_context=None,
                    hook_load_options=None,
                    data_dir=root / "var",
                )

            self.assertEqual(command.action, "handled")
            record = RemoteBindingStore.in_data_dir(root / "var").get("wecom", "*")
            self.assertIsNotNone(record)
            self.assertEqual(record.session_id, session.session_id)
            self.assertFalse(record.route_open)

    def test_interactive_remote_open_and_local_turn_close_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session_store = SessionStore.in_directory(root / "var" / "sessions")
            session = session_store.create(
                root,
                ProfileRef(name="default", uri="profiles/default"),
                "Current task",
            )
            runtime = _RuntimeStub()

            with redirect_stdout(StringIO()):
                command = _handle_command(
                    prompt="/remote --open wecom",
                    session_store=session_store,
                    session=session,
                    transcript=session_store.transcript_store(session),
                    runtime=runtime,
                    workspace=root,
                    profile=session.profile,
                    mcp_servers=(),
                    trace_recorder=TraceRecorder(_TraceSink()),
                    context_snapshot_factory=None,
                    capability_state_store=None,
                    checkpoint_controller_factory=None,
                    hook_context=None,
                    hook_load_options=None,
                    data_dir=root / "var",
                )

            self.assertEqual(command.action, "handled")
            store = RemoteBindingStore.in_data_dir(root / "var")
            self.assertTrue(store.get("wecom", "*").route_open)

            _close_remote_routes_for_local_turn(
                session_store=session_store,
                session=session,
                data_dir=root / "var",
                trace_recorder=TraceRecorder(_TraceSink()),
                runtime=runtime,
            )

            self.assertFalse(store.get("wecom", "*").route_open)

    def test_binding_default_session_replaces_channel_specific_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RemoteBindingStore.in_data_dir(root / "var")
            session_store = SessionStore.in_directory(root / "var" / "sessions")
            first = session_store.create(
                root,
                ProfileRef(name="default", uri="profiles/default"),
                "Remote task",
            )
            second = session_store.create(
                root,
                ProfileRef(name="default", uri="profiles/default"),
                "Local task",
            )
            store.bind_session(
                channel="wecom",
                remote_user_id="alice",
                session=first,
                bound_from="remote",
            )

            record = store.bind_default_session(
                channel="wecom",
                session=second,
                bound_from="interactive",
            )

            self.assertEqual(record.remote_user_id, "*")
            self.assertIsNone(store.get("wecom", "alice"))
            self.assertEqual(store.get("wecom", "*").session_id, second.session_id)

    def test_wecom_channel_picks_up_binding_switch_without_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _settings(root)
            session_store = SessionStore.in_directory(settings.data_dir / "sessions")
            first = session_store.create(root, settings.profile_ref(), "First")
            second = session_store.create(root, settings.profile_ref(), "Second")
            store = RemoteBindingStore.in_data_dir(settings.data_dir)
            store.bind_session(
                channel="wecom",
                remote_user_id="alice",
                session=first,
                bound_from="test",
            )
            provider = StubProvider(ModelResponse(content="done"))
            channel = WeComChannel(
                WeComRunDependencies(
                    settings=settings,
                    provider=provider,
                    model="stub-model",
                    session_store=session_store,
                    trace_recorder=TraceRecorder(_TraceSink()),
                ),
                binding_store=store,
                sender=lambda payload: None,
                wake_factory=lambda reason: FakeWakeSession(),
            )

            channel.handle_message(
                WeComInboundMessage(
                    req_id="req_1",
                    user_id="alice",
                    chat_id="chat_1",
                    chat_type="single",
                    content="/current",
                )
            )
            store.bind_default_session(
                channel="wecom",
                session=second,
                bound_from="interactive",
                route_open=False,
            )
            channel.handle_message(
                WeComInboundMessage(
                    req_id="req_2",
                    user_id="alice",
                    chat_id="chat_1",
                    chat_type="single",
                    content="继续处理",
                )
            )

            self.assertEqual(
                store.get("wecom", "*").session_id,
                second.session_id,
            )
            self.assertTrue(store.get("wecom", "*").route_open)

    def test_wecom_channel_does_not_reuse_cached_state_after_unbind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _settings(root)
            session_store = SessionStore.in_directory(settings.data_dir / "sessions")
            first = session_store.create(root, settings.profile_ref(), "First")
            store = RemoteBindingStore.in_data_dir(settings.data_dir)
            store.bind_session(
                channel="wecom",
                remote_user_id="alice",
                session=first,
                bound_from="test",
            )
            provider = StubProvider(ModelResponse(content="done"))
            channel = WeComChannel(
                WeComRunDependencies(
                    settings=settings,
                    provider=provider,
                    model="stub-model",
                    session_store=session_store,
                    trace_recorder=TraceRecorder(_TraceSink()),
                ),
                binding_store=store,
                sender=lambda payload: None,
                wake_factory=lambda reason: FakeWakeSession(),
            )

            channel.handle_message(
                WeComInboundMessage(
                    req_id="req_1",
                    user_id="alice",
                    chat_id="chat_1",
                    chat_type="single",
                    content="/current",
                )
            )
            store.remove_channel("wecom")
            channel.handle_message(
                WeComInboundMessage(
                    req_id="req_2",
                    user_id="alice",
                    chat_id="chat_1",
                    chat_type="single",
                    content="重新开始",
                )
            )

            bindings = store.list_records()
            self.assertEqual(len(bindings), 1)
            self.assertNotEqual(bindings[0].session_id, first.session_id)
            self.assertEqual(bindings[0].remote_user_id, "alice")

    def test_wake_backend_uses_caffeinate_wait_for_current_process(self) -> None:
        calls = []

        class Process:
            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, timeout=None):
                pass

        def fake_popen(args, **kwargs):
            calls.append(args)
            return Process()

        with (
            patch("deepmate.runtime.wakelock.platform.system", lambda: "Darwin"),
            patch("deepmate.runtime.wakelock._which", lambda name: Path("/usr/bin/caffeinate")),
            patch("deepmate.runtime.wakelock.subprocess.Popen", fake_popen),
        ):
            handle = WakeBackend().acquire("test")
            handle.release()

        self.assertEqual(calls[0][:4], ["/usr/bin/caffeinate", "-i", "-d", "-m"])
        self.assertIn("-w", calls[0])


class _RuntimeStub:
    class Activation:
        def trace_refs(self):
            return ()

    activation = Activation()


class _TraceSink:
    def __init__(self) -> None:
        self.events = []

    def write(self, event) -> None:
        self.events.append(event)


class _FakeSocket:
    def __init__(self, data: bytes = b"") -> None:
        self.data = bytearray(data)
        self.sent = bytearray()
        self.timeout = None

    def recv(self, size: int) -> bytes:
        if not self.data:
            return b""
        chunk = bytes(self.data[:size])
        del self.data[:size]
        return chunk

    def sendall(self, data: bytes) -> None:
        self.sent.extend(data)

    def gettimeout(self):
        return self.timeout

    def settimeout(self, value):
        self.timeout = value


def _server_frame(payload: bytes, *, opcode: int = 0x1, fin: bool = True) -> bytes:
    first = (0x80 if fin else 0) | opcode
    length = len(payload)
    if length < 126:
        return bytes([first, length]) + payload
    if length <= 0xFFFF:
        return bytes([first, 126]) + struct.pack("!H", length) + payload
    return bytes([first, 127]) + struct.pack("!Q", length) + payload


def _settings(root: Path) -> AppSettings:
    (root / "AGENTS.md").write_text("Remote test workspace rules.\n", encoding="utf-8")
    return AppSettings(
        workspace=root,
        data_dir=root / "var",
        active_profile="default",
        trace_sink=root / "var" / "traces" / "trace.jsonl",
        default_provider="test",
        providers={
            "test": ProviderSettings(
                name="test",
                base_url="https://example.test",
                default_model="stub-model",
                api_key_env="TEST_API_KEY",
            )
        },
        remote=RemoteSettings(
            wecom=WeComRemoteSettings(
                enabled=True,
                bot_id="ww_test",
                secret="secret",
                allowed_users=("alice",),
            )
        ),
    )


def _with_fast_heartbeat(settings: AppSettings) -> AppSettings:
    return replace(
        settings,
        remote=replace(
            settings.remote,
            wecom=replace(
                settings.remote.wecom,
                progress_heartbeat=True,
                progress_intervals_seconds=(1,),
            ),
        ),
    )


def _wait_for_sent_text(
    sent: list[dict[str, object]],
    needle: str,
    *,
    timeout: float,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(needle in _sent_content(item) for item in sent):
            return True
        time.sleep(0.01)
    return any(needle in _sent_content(item) for item in sent)


def _wait_for_condition(predicate, *, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _wait_for_idle(channel: WeComChannel, *, timeout: float) -> bool:
    return _wait_for_condition(
        lambda: channel._states and not any(
            state.is_processing for state in channel._states.values()
        ),
        timeout=timeout,
    )


def _sent_text(payload: dict[str, object]) -> str:
    text = payload.get("text")
    if isinstance(text, dict):
        content = text.get("content")
        if isinstance(content, str):
            return content
    return _sent_markdown(payload)


def _sent_markdown(payload: dict[str, object]) -> str:
    markdown = payload.get("markdown")
    if isinstance(markdown, dict):
        content = markdown.get("content")
        if isinstance(content, str):
            return content
    return ""


def _sent_stream(payload: dict[str, object]) -> str:
    stream = payload.get("stream")
    if isinstance(stream, dict):
        content = stream.get("content")
        if isinstance(content, str):
            return content
    return ""


def _sent_stream_id(payload: dict[str, object]) -> str:
    stream = payload.get("stream")
    if isinstance(stream, dict):
        stream_id = stream.get("id")
        if isinstance(stream_id, str):
            return stream_id
    return ""


def _sent_stream_finish(payload: dict[str, object]) -> bool:
    stream = payload.get("stream")
    if isinstance(stream, dict):
        return stream.get("finish") is True
    return False


def _sent_content(payload: dict[str, object]) -> str:
    return _sent_text(payload) or _sent_markdown(payload) or _sent_stream(payload)


if __name__ == "__main__":
    unittest.main()
