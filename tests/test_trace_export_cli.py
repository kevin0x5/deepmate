from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from deepmate.app import AppSettings, ProviderSettings
from deepmate.activity import ActivityStore
from deepmate.channels.cli import _print_session_detail, _validate_otlp_export
from deepmate.domain import ProfileRef
from deepmate.storage import SessionStore
from deepmate.trace import JsonlTraceSink, TraceEvent, TraceRecorder, TraceSpan
from deepmate.trace.exporter import OtlpExportResult


class TraceExportCliTests(unittest.TestCase):
    def test_show_session_export_otlp_exports_only_session_spans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore.in_directory(root / "var" / "sessions")
            session = store.create(
                workspace=root,
                profile=ProfileRef(name="default", uri="profiles/default"),
                title="Trace export",
            )
            trace_path = root / "var" / "traces" / "trace.jsonl"
            recorder = TraceRecorder(JsonlTraceSink(trace_path))
            recorder.record(
                TraceEvent(
                    kind="model_response_received",
                    summary="Model response.",
                    refs=(f"session_id={session.session_id}", "input_tokens=1"),
                )
            )
            recorder.record_span(
                TraceSpan(
                    name="chat deepseek-v4-flash",
                    kind="CLIENT",
                    trace_id="a" * 32,
                    span_id="b" * 16,
                    started_at_unix_nano=1_000,
                    ended_at_unix_nano=2_000,
                    status="OK",
                    attributes={"session.id": session.session_id},
                )
            )
            recorder.record_span(
                TraceSpan(
                    name="chat other",
                    kind="CLIENT",
                    trace_id="c" * 32,
                    span_id="d" * 16,
                    started_at_unix_nano=1_000,
                    ended_at_unix_nano=2_000,
                    status="OK",
                    attributes={"session.id": "other-session"},
                )
            )
            settings = AppSettings(
                workspace=root,
                data_dir=root / "var",
                active_profile="default",
                trace_sink=trace_path,
                default_provider="deepseek",
            )
            captured = {}

            def fake_export(spans, **kwargs):
                captured["spans"] = spans
                captured["kwargs"] = kwargs
                return OtlpExportResult(
                    endpoint="https://otel.example/v1/traces",
                    spans_seen=len(spans),
                    spans_exported=len(spans),
                    status_code=200,
                    message="OK",
                )

            stdout = io.StringIO()
            with (
                patch("deepmate.channels.cli.export_otlp_traces", fake_export),
                redirect_stdout(stdout),
            ):
                _print_session_detail(
                    session_store=store,
                    session=session,
                    data_dir=root / "var",
                    trace_path=trace_path,
                    trace_limit=0,
                    export_otlp=True,
                    otlp_endpoint="https://otel.example",
                    settings=settings,
                )

        self.assertEqual(len(captured["spans"]), 1)
        self.assertEqual(captured["spans"][0].attributes["session.id"], session.session_id)
        self.assertEqual(captured["kwargs"]["endpoint"], "https://otel.example")
        self.assertIn("OTLP export completed", stdout.getvalue())

    def test_show_session_uses_recorded_activity_note_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore.in_directory(root / "var" / "sessions")
            session = store.create(
                workspace=root,
                profile=ProfileRef(name="default", uri="profiles/default"),
                title="Activity path",
            )
            trace_path = root / "var" / "traces" / "trace.jsonl"
            recorder = TraceRecorder(JsonlTraceSink(trace_path))
            recorder.record(
                TraceEvent(
                    kind="activity_daily_note_written",
                    summary="Activity note written.",
                    refs=(
                        f"session_id={session.session_id}",
                        "activity_date=2026-06-22",
                        "activity_path=var/activity/default/daily/2026-06-22.md",
                    ),
                )
            )
            settings = AppSettings(
                workspace=root,
                data_dir=root / "var",
                active_profile="default",
                trace_sink=trace_path,
                default_provider="deepseek",
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                _print_session_detail(
                    session_store=store,
                    session=session,
                    data_dir=root / "var",
                    trace_path=trace_path,
                    activity_store=ActivityStore(root / "var" / "activity" / "default"),
                    trace_limit=0,
                    settings=settings,
                )

        self.assertIn(
            "activity_note: var/activity/default/daily/2026-06-22.md",
            stdout.getvalue(),
        )

    def test_validate_otlp_exports_synthetic_deepmate_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = AppSettings(
                workspace=root,
                data_dir=root / "var",
                active_profile="default",
                trace_sink=root / "var" / "traces" / "trace.jsonl",
                default_provider="deepseek",
                providers={
                    "deepseek": ProviderSettings(
                        name="deepseek",
                        base_url="https://api.deepseek.com",
                        default_model="deepseek-v4-flash",
                    )
                },
            )
            captured = {}

            def fake_export(spans, **kwargs):
                captured["spans"] = spans
                captured["kwargs"] = kwargs
                return OtlpExportResult(
                    endpoint="https://otel.example/v1/traces",
                    spans_seen=len(spans),
                    spans_exported=len(spans),
                    status_code=200,
                    message="OK",
                )

            with patch("deepmate.channels.cli.export_otlp_traces", fake_export):
                result = _validate_otlp_export(
                    settings=settings,
                    endpoint="https://otel.example",
                )

        spans = captured["spans"]
        self.assertTrue(result.is_success())
        self.assertEqual(len(spans), 3)
        self.assertEqual(spans[0].name, "deepmate otlp validation")
        self.assertEqual(spans[1].parent_span_id, spans[0].span_id)
        self.assertEqual(spans[2].parent_span_id, spans[0].span_id)
        self.assertEqual(spans[1].attributes["gen_ai.operation.name"], "chat")
        self.assertEqual(spans[2].attributes["gen_ai.tool.name"], "otlp_validation_tool")
        self.assertEqual(captured["kwargs"]["endpoint"], "https://otel.example")


if __name__ == "__main__":
    unittest.main()
