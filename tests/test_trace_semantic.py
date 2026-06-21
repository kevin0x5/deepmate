from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepmate.channels.cli import _format_trace_record, _read_session_trace_records
from deepmate.trace import (
    JsonlTraceSink,
    TraceRecorder,
    TraceSpan,
    build_otlp_traces_payload,
    export_otlp_traces,
    summarize_trace_usage,
    trace_record_matches_kinds,
    trace_record_matches_session,
    trace_refs_to_map,
    trace_semantic_attributes,
)


class TraceSemanticTests(unittest.TestCase):
    def test_trace_record_matches_session_refs(self) -> None:
        record = {
            "kind": "subagent_run_completed",
            "refs": ["parent_session_id=session-a", "subagent_run_id=run-1"],
        }

        self.assertTrue(trace_record_matches_session(record, "session-a"))
        self.assertFalse(trace_record_matches_session(record, "session-b"))

    def test_trace_record_matches_session_span_attributes(self) -> None:
        record = {
            "record_type": "span",
            "name": "chat deepseek-v4-flash",
            "attributes": {"session.id": "session-a"},
        }

        self.assertTrue(trace_record_matches_session(record, "session-a"))
        self.assertFalse(trace_record_matches_session(record, "session-b"))

    def test_trace_record_matches_exact_kind_filter(self) -> None:
        record = {"kind": "model_response_received", "refs": []}

        self.assertTrue(trace_record_matches_kinds(record, ()))
        self.assertTrue(trace_record_matches_kinds(record, ("model_response_received",)))
        self.assertFalse(trace_record_matches_kinds(record, ("mcp_tool_completed",)))

    def test_trace_record_matches_span_kind_filter(self) -> None:
        record = {
            "record_type": "span",
            "name": "chat deepseek-v4-flash",
            "attributes": {"session.id": "session-a"},
        }

        self.assertTrue(trace_record_matches_kinds(record, ("span",)))
        self.assertTrue(trace_record_matches_kinds(record, ("chat deepseek-v4-flash",)))
        self.assertFalse(trace_record_matches_kinds(record, ("model_response_received",)))

    def test_trace_refs_to_map_and_semantic_attributes(self) -> None:
        refs = (
            "model=deepseek-v4-flash",
            "provider=deepseek",
            "input_tokens=100",
            "output_tokens=20",
            "reasoning_tokens=7",
            "cache_hit_input_tokens=80",
            "cache_miss_input_tokens=20",
            "finish_reason=stop",
            "session_id=session-a",
        )

        self.assertEqual(trace_refs_to_map(refs)["model"], "deepseek-v4-flash")

        attributes = trace_semantic_attributes("model_response_received", refs)
        self.assertEqual(attributes["gen_ai.operation.name"], "chat")
        self.assertEqual(attributes["gen_ai.provider.name"], "deepseek")
        self.assertEqual(attributes["gen_ai.request.model"], "deepseek-v4-flash")
        self.assertEqual(attributes["gen_ai.response.model"], "deepseek-v4-flash")
        self.assertEqual(attributes["gen_ai.usage.input_tokens"], 100)
        self.assertEqual(attributes["gen_ai.usage.output_tokens"], 20)
        self.assertEqual(attributes["deepmate.usage.reasoning_tokens"], 7)
        self.assertEqual(attributes["deepmate.usage.cache_hit_input_tokens"], 80)
        self.assertEqual(attributes["deepmate.finish_reason"], "stop")

    def test_prefix_fingerprint_trace_has_model_semantics(self) -> None:
        attributes = trace_semantic_attributes(
            "model_request_prefix_fingerprint",
            ("model=deepseek-v4-flash", "prefix_digest=abc123"),
        )

        self.assertEqual(attributes["gen_ai.operation.name"], "chat")
        self.assertEqual(attributes["gen_ai.request.model"], "deepseek-v4-flash")

    def test_summarize_trace_usage(self) -> None:
        records = (
            {
                "kind": "model_response_received",
                "refs": [
                    "input_tokens=100",
                    "output_tokens=20",
                    "reasoning_tokens=7",
                    "cache_hit_input_tokens=80",
                    "cache_miss_input_tokens=20",
                ],
            },
            {
                "kind": "model_response_received",
                "refs": [
                    "input_tokens=50",
                    "output_tokens=10",
                    "cache_hit_input_tokens=25",
                    "cache_miss_input_tokens=25",
                ],
            },
            {"kind": "native_tool_completed", "refs": ["tool_source=native"]},
        )

        summary = summarize_trace_usage(records)

        self.assertEqual(summary.model_response_events, 2)
        self.assertEqual(summary.input_tokens, 150)
        self.assertEqual(summary.output_tokens, 30)
        self.assertEqual(summary.reasoning_tokens, 7)
        self.assertEqual(summary.cache_hit_input_tokens, 105)
        self.assertEqual(summary.cache_miss_input_tokens, 45)
        self.assertEqual(summary.cache_hit_ratio(), 0.7)

    def test_span_jsonl_record_is_readable_by_session_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            recorder = TraceRecorder(JsonlTraceSink(path))
            recorder.record_span(
                TraceSpan(
                    name="deepmate turn",
                    kind="INTERNAL",
                    trace_id="1" * 32,
                    span_id="1" * 16,
                    started_at_unix_nano=100_000_000,
                    ended_at_unix_nano=900_000_000,
                    status="OK",
                    attributes={
                        "session.id": "session-a",
                        "gen_ai.operation.name": "invoke_agent",
                        "gen_ai.usage.input_tokens": 100,
                        "gen_ai.usage.output_tokens": 20,
                    },
                )
            )
            span = TraceSpan(
                name="chat deepseek-v4-flash",
                kind="CLIENT",
                trace_id="1" * 32,
                span_id="2" * 16,
                started_at_unix_nano=1_000_000_000,
                ended_at_unix_nano=2_500_000_000,
                status="OK",
                attributes={
                    "session.id": "session-a",
                    "gen_ai.request.model": "deepseek-v4-flash",
                    "gen_ai.usage.input_tokens": 100,
                    "gen_ai.usage.output_tokens": 20,
                    "gen_ai.usage.cache_read.input_tokens": 80,
                    "deepmate.usage.cache_miss_input_tokens": 20,
                },
            )

            recorder.record_span(span)

            records = _read_session_trace_records(path, "session-a")
            self.assertEqual(len(records), 2)
            formatted = _format_trace_record(records[-1])
            self.assertIn("span chat deepseek-v4-flash", formatted)
            self.assertIn("duration=1.5s", formatted)
            self.assertIn("model=deepseek-v4-flash", formatted)
            self.assertIn("cache_hit_ratio=80%", formatted)
            turn_formatted = _format_trace_record(records[0])
            self.assertIn("span deepmate turn", turn_formatted)
            self.assertIn("input=100", turn_formatted)

    def test_otlp_payload_uses_http_json_shapes_for_langfuse(self) -> None:
        root = TraceSpan(
            name="deepmate turn",
            kind="INTERNAL",
            trace_id="a" * 32,
            span_id="b" * 16,
            started_at_unix_nano=1_000,
            ended_at_unix_nano=5_000,
            status="OK",
            attributes={
                "session.id": "session-a",
                "langfuse.trace.name": "Implement feature",
            },
        )
        model = TraceSpan(
            name="chat deepseek-v4-flash",
            kind="CLIENT",
            trace_id="a" * 32,
            span_id="c" * 16,
            parent_span_id="b" * 16,
            started_at_unix_nano=2_000,
            ended_at_unix_nano=4_000,
            status="OK",
            attributes={
                "session.id": "session-a",
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": "deepseek",
                "gen_ai.request.model": "deepseek-v4-flash",
                "gen_ai.response.model": "deepseek-v4-flash",
                "gen_ai.usage.input_tokens": 100,
                "gen_ai.usage.output_tokens": 20,
            },
        )
        failed_tool = TraceSpan(
            name="execute_tool search_files",
            kind="INTERNAL",
            trace_id="a" * 32,
            span_id="d" * 16,
            parent_span_id="b" * 16,
            started_at_unix_nano=4_000,
            ended_at_unix_nano=5_000,
            status="ERROR",
            attributes={
                "session.id": "session-a",
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "search_files",
                "error.type": "ValueError",
                "error.message": "bad query",
            },
        )

        payload = build_otlp_traces_payload(
            (root, model, failed_tool),
            service_version="0.1.0",
        )

        encoded = json.dumps(payload)
        self.assertIn("resourceSpans", payload)
        self.assertIn('"traceId": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"', encoded)
        self.assertIn('"spanId": "cccccccccccccccc"', encoded)
        self.assertIn('"parentSpanId": "bbbbbbbbbbbbbbbb"', encoded)
        self.assertIn('"kind": 3', encoded)
        self.assertIn('"key": "gen_ai.request.model"', encoded)
        self.assertIn('"key": "gen_ai.usage.input_tokens"', encoded)
        self.assertIn('"intValue": "100"', encoded)
        self.assertIn('"key": "session.id"', encoded)
        self.assertIn('"key": "gen_ai.conversation.id"', encoded)
        self.assertNotIn('"key": "langfuse.session.id"', encoded)
        self.assertIn('"key": "langfuse.trace.name"', encoded)
        self.assertIn('"code": 2', encoded)
        self.assertIn('"message": "bad query"', encoded)

    def test_otlp_payload_skips_incomplete_spans(self) -> None:
        incomplete = TraceSpan(
            name="chat deepseek-v4-flash",
            kind="CLIENT",
            trace_id="a" * 32,
            span_id="b" * 16,
            started_at_unix_nano=1_000,
            ended_at_unix_nano=0,
            status="UNSET",
            attributes={"session.id": "session-a"},
        )

        payload = build_otlp_traces_payload((incomplete,))

        self.assertEqual(payload, {"resourceSpans": []})

    def test_otlp_payload_derives_required_operation_name(self) -> None:
        span = TraceSpan(
            name="execute_tool search_files",
            kind="INTERNAL",
            trace_id="a" * 32,
            span_id="b" * 16,
            started_at_unix_nano=1_000,
            ended_at_unix_nano=2_000,
            status="OK",
            attributes={"session.id": "session-a"},
        )

        encoded = json.dumps(build_otlp_traces_payload((span,)))

        self.assertIn('"key": "gen_ai.operation.name"', encoded)
        self.assertIn('"stringValue": "execute_tool"', encoded)

    def test_otlp_export_posts_to_traces_endpoint_without_leaking_headers(self) -> None:
        span = TraceSpan(
            name="chat deepseek-v4-flash",
            kind="CLIENT",
            trace_id="a" * 32,
            span_id="b" * 16,
            started_at_unix_nano=1_000,
            ended_at_unix_nano=2_000,
            status="OK",
            attributes={"session.id": "session-a"},
        )
        captured = []

        class Response:
            status = 200
            reason = "OK"

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def getcode(self):
                return self.status

        def fake_urlopen(request, timeout):
            captured.append((request, timeout))
            return Response()

        with patch("deepmate.trace.exporter.urlopen", fake_urlopen):
            result = export_otlp_traces(
                (span,),
                endpoint="https://cloud.langfuse.com/api/public/otel",
                headers=(("Authorization", "Basic secret"),),
                service_name="deepmate-test",
            )

        self.assertTrue(result.is_success())
        self.assertEqual(result.endpoint, "https://cloud.langfuse.com/api/public/otel/v1/traces")
        request, timeout = captured[0]
        self.assertEqual(timeout, 10)
        self.assertEqual(request.get_header("Authorization"), "Basic secret")
        self.assertIn(b"resourceSpans", request.data)


if __name__ == "__main__":
    unittest.main()
