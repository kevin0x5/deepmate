from __future__ import annotations

import io
import json
import os
import re
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from deepmate.app import AppSettings
from deepmate.channels.checkpointing import SessionCheckpointWriteRouter
from deepmate.channels.cli import _build_cli_native_tools
from deepmate.providers import ModelToolRequest
from deepmate.runtime import (
    HookAction,
    HookActionType,
    HookDefinition,
    HookEvent,
    HookLayer,
    HookRegistry,
    HookRuntimeContext,
    HookSignalStore,
    SandboxMode,
    ToolAccessMode,
    ToolAccessPolicy,
    execute_native_tool_request,
)
from deepmate.tools import (
    NativeToolRegistry,
    web_research_tools,
    workspace_lsp_tools,
    workspace_artifact_tools,
    workspace_diagram_tools,
    workspace_document_tools,
    workspace_filesystem_tools,
    workspace_report_tools,
    workspace_search_tools,
)
from deepmate.tools.lsp import MAX_LSP_HEADER_BYTES, MAX_LSP_PAYLOAD_BYTES, _JsonRpcClient


class BuiltinWorkspaceToolTests(unittest.TestCase):
    def test_cli_read_only_tools_include_lsp_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            settings = AppSettings(
                workspace=workspace,
                data_dir=workspace / "var",
                active_profile="default",
                trace_sink=workspace / "var" / "trace.jsonl",
                default_provider="stub",
            )

            registry = _build_cli_native_tools(
                settings=settings,
                expose_read_tools=True,
                register_write_tools=False,
                expose_network_tools=False,
                register_shell_tools=False,
                shell_enabled=False,
                network_enabled=False,
                env_change_enabled=False,
                sandbox_mode=SandboxMode.AUTO,
                approval_cache=None,
                checkpoint_write_router=SessionCheckpointWriteRouter(),
                hook_context=HookRuntimeContext.from_registry(
                    HookRegistry.from_hooks(())
                ),
            )

            self.assertIsNotNone(registry)
            names = tuple(tool.name for tool in registry.list_tools())
            schemas = tuple(schema["name"] for schema in registry.schemas())
            self.assertIn("lsp_definition", names)
            self.assertIn("lsp_references", names)
            self.assertIn("lsp_hover", names)
            self.assertIn("lsp_definition", schemas)

    def test_search_content_and_files_are_bounded_to_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "README.md").write_text("Deepmate project", encoding="utf-8")
            (workspace / "src").mkdir()
            (workspace / "src" / "app.py").write_text(
                "def run():\n    return 'Deepmate search ok'\n",
                encoding="utf-8",
            )
            (workspace / ".env").write_text("SECRET=1", encoding="utf-8")
            registry = NativeToolRegistry(workspace_search_tools(workspace))

            content = execute_native_tool_request(
                ModelToolRequest(
                    name="search_content",
                    id="call_1",
                    arguments={"pattern": "deepmate", "path": ".", "max_matches": 5},
                ),
                registry,
            )
            files = execute_native_tool_request(
                ModelToolRequest(
                    name="search_files",
                    id="call_2",
                    arguments={"pattern": "**/*.py", "path": ".", "max_results": 5},
                ),
                registry,
            )

            self.assertIsNone(content.error)
            self.assertIsNotNone(content.model_result)
            self.assertIn("src/app.py", content.model_result.content)
            self.assertNotIn(".env", content.model_result.content)
            self.assertIsNone(files.error)
            self.assertIsNotNone(files.model_result)
            self.assertIn("src/app.py", files.model_result.content)

    def test_lsp_tools_are_read_only_and_degrade_when_server_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "app.py").write_text(
                "def run():\n    return 1\n",
                encoding="utf-8",
            )
            registry = NativeToolRegistry(
                workspace_lsp_tools(workspace, server_resolver=lambda _path: None)
            )

            self.assertTrue(all(tool.read_only for tool in registry.list_tools()))
            self.assertEqual(
                tuple(tool.name for tool in registry.list_tools()),
                ("lsp_definition", "lsp_references", "lsp_hover"),
            )
            result = execute_native_tool_request(
                ModelToolRequest(
                    name="lsp_definition",
                    id="call_1",
                    arguments={"file": "app.py", "line": 1, "column": 5},
                ),
                registry,
            )

            self.assertIsNone(result.error)
            self.assertIsNotNone(result.model_result)
            self.assertIn("LSP unavailable", result.model_result.content)
            self.assertFalse(result.native_result.data.get("available"))
            self.assertIn("lsp_available=false", result.model_result.refs)

    def test_lsp_tools_are_bounded_to_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            outside = Path(tmp).parent / "outside_lsp_test.py"
            outside.write_text("def outside():\n    pass\n", encoding="utf-8")
            registry = NativeToolRegistry(
                workspace_lsp_tools(workspace, server_resolver=lambda _path: None)
            )
            try:
                result = execute_native_tool_request(
                    ModelToolRequest(
                        name="lsp_hover",
                        id="call_1",
                        arguments={"file": str(outside), "line": 1, "column": 1},
                    ),
                    registry,
                )
            finally:
                if outside.exists():
                    outside.unlink()

            self.assertIsNotNone(result.error)
            self.assertIn("path must stay inside workspace root", result.error.message)

    def test_lsp_tools_degrade_for_large_source_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "large.py").write_text(
                "x = '" + ("a" * (2 * 1024 * 1024)) + "'\n",
                encoding="utf-8",
            )
            registry = NativeToolRegistry(
                workspace_lsp_tools(
                    workspace,
                    server_resolver=lambda _path: ("unused-language-server",),
                )
            )

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="lsp_hover",
                    id="call_1",
                    arguments={"file": "large.py", "line": 1, "column": 1},
                ),
                registry,
            )

            self.assertIsNone(result.error)
            self.assertIsNotNone(result.model_result)
            self.assertIn("Source file is too large", result.model_result.content)
            self.assertFalse(result.native_result.data["available"])

    def test_lsp_jsonrpc_rejects_oversized_request_payload(self) -> None:
        process = _FakeLspProcess()
        client = _JsonRpcClient(process, deadline=9999999999)

        with self.assertRaisesRegex(OSError, "request payload is too large"):
            client.notify("x" * MAX_LSP_PAYLOAD_BYTES, {})

    def test_lsp_jsonrpc_rejects_oversized_header_buffer(self) -> None:
        process = _FakeLspProcess()
        client = _JsonRpcClient(process, deadline=9999999999)

        with (
            patch("deepmate.tools.lsp.select.select", return_value=([process.stdout], [], [])),
            patch("deepmate.tools.lsp.os.read", return_value=b"x" * (MAX_LSP_HEADER_BYTES + 1)),
        ):
            with self.assertRaisesRegex(OSError, "header is too large"):
                client._read_message()

    def test_lsp_jsonrpc_matches_string_response_id_and_counts_skips(self) -> None:
        process = _FakeLspProcess()
        client = _JsonRpcClient(process, deadline=9999999999)
        messages = iter(
            (
                {"jsonrpc": "2.0", "method": "window/logMessage", "params": {}},
                {"jsonrpc": "2.0", "id": 999, "result": None},
                {"jsonrpc": "2.0", "id": "1", "result": {"ok": True}},
            )
        )

        with patch.object(client, "_read_message", side_effect=lambda: next(messages)):
            response = client.request("test/request", {})

        self.assertEqual(response["result"], {"ok": True})
        self.assertEqual(client.skipped_notifications, 1)
        self.assertEqual(client.skipped_unmatched_responses, 1)

    def test_search_files_default_includes_root_and_nested_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "README.md").write_text("Deepmate project", encoding="utf-8")
            (workspace / "src").mkdir()
            (workspace / "src" / "app.py").write_text("print('ok')", encoding="utf-8")
            (workspace / "config").mkdir()
            (workspace / "config" / "deepmate.yaml").write_text(
                "runtime: internal\n",
                encoding="utf-8",
            )
            (workspace / "profiles").mkdir()
            (workspace / "profiles" / "memory.md").write_text(
                "internal memory",
                encoding="utf-8",
            )
            (workspace / ".env").write_text("SECRET=1", encoding="utf-8")
            (workspace / ".env.staging").write_text("SECRET=2", encoding="utf-8")
            registry = NativeToolRegistry(workspace_search_tools(workspace))

            default_result = execute_native_tool_request(
                ModelToolRequest(name="search_files", id="call_1", arguments={}),
                registry,
            )
            all_result = execute_native_tool_request(
                ModelToolRequest(
                    name="search_files",
                    id="call_2",
                    arguments={"pattern": "**/*"},
                ),
                registry,
            )
            empty_path_result = execute_native_tool_request(
                ModelToolRequest(
                    name="search_files",
                    id="call_3",
                    arguments={"path": ""},
                ),
                registry,
            )

            for result in (default_result, all_result, empty_path_result):
                self.assertIsNone(result.error)
                self.assertIsNotNone(result.model_result)
                self.assertIn("README.md", result.model_result.content)
                self.assertIn("src/app.py", result.model_result.content)
                self.assertNotIn("config/deepmate.yaml", result.model_result.content)
                self.assertNotIn("profiles/memory.md", result.model_result.content)
                self.assertNotIn(".env", result.model_result.content)
                self.assertNotIn(".env.staging", result.model_result.content)

    def test_filesystem_directory_listing_hides_deepmate_internal_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "README.md").write_text("Project", encoding="utf-8")
            (workspace / "config").mkdir()
            (workspace / "config" / "deepmate.yaml").write_text(
                "runtime: internal\n",
                encoding="utf-8",
            )
            (workspace / "profiles").mkdir()
            (workspace / "profiles" / "memory.md").write_text(
                "internal memory",
                encoding="utf-8",
            )
            registry = NativeToolRegistry(workspace_filesystem_tools(workspace))

            listing = execute_native_tool_request(
                ModelToolRequest(name="list_directory", id="call_1", arguments={}),
                registry,
            )
            explicit_read = execute_native_tool_request(
                ModelToolRequest(
                    name="read_text_file",
                    id="call_2",
                    arguments={"path": "config/deepmate.yaml"},
                ),
                registry,
            )

            self.assertIsNone(listing.error)
            self.assertIsNotNone(listing.model_result)
            self.assertIn("README.md", listing.model_result.content)
            self.assertNotIn("config/", listing.model_result.content)
            self.assertNotIn("profiles/", listing.model_result.content)
            self.assertIsNone(explicit_read.error)
            self.assertIn("runtime: internal", explicit_read.model_result.content)

    def test_filesystem_write_denies_env_glob_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            registry = NativeToolRegistry(
                workspace_filesystem_tools(workspace, include_write_tools=True)
            )

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="write_text_file",
                    id="call_1",
                    arguments={
                        "path": ".env.staging",
                        "content": "SECRET=1\n",
                    },
                ),
                registry,
                ToolAccessPolicy(mode=ToolAccessMode.WORKSPACE_WRITE),
            )

        self.assertIsNotNone(result.error)
        self.assertIn("path is not allowed", result.error.message)

    def test_filesystem_write_uses_durable_atomic_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            registry = NativeToolRegistry(
                workspace_filesystem_tools(workspace, include_write_tools=True)
            )

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="write_text_file",
                    id="call_1",
                    arguments={"path": "note.txt", "content": "hello\n"},
                ),
                registry,
                ToolAccessPolicy(mode=ToolAccessMode.WORKSPACE_WRITE),
            )

            self.assertIsNone(result.error)
            self.assertEqual((workspace / "note.txt").read_text(encoding="utf-8"), "hello\n")
            self.assertFalse(tuple(workspace.glob(".note.txt.deepmate-*.tmp")))

    def test_search_files_ignores_symlink_loops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "real.txt").write_text("ok", encoding="utf-8")
            (workspace / "a").symlink_to("b")
            (workspace / "b").symlink_to("a")
            registry = NativeToolRegistry(workspace_search_tools(workspace))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="search_files",
                    id="call_1",
                    arguments={"pattern": "**/*", "kind": "any"},
                ),
                registry,
            )

            self.assertIsNone(result.error)
            self.assertIsNotNone(result.model_result)
            self.assertIn("real.txt", result.model_result.content)

    def test_read_text_file_reads_large_files_with_paging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "huge.log").write_bytes(b"x" * 2_000_001)
            registry = NativeToolRegistry(workspace_filesystem_tools(workspace))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="read_text_file",
                    id="call_1",
                    arguments={"path": "huge.log", "max_chars": 500},
                ),
                registry,
            )

            self.assertIsNone(result.error)
            self.assertIsNotNone(result.native_result)
            self.assertEqual(result.native_result.data.get("chars_read"), 500)
            self.assertTrue(result.native_result.data.get("truncated"))
            next_offset = result.native_result.data.get("next_offset")
            self.assertIsNotNone(next_offset)
            # Read the next page via offset
            result2 = execute_native_tool_request(
                ModelToolRequest(
                    name="read_text_file",
                    id="call_2",
                    arguments={"path": "huge.log", "max_chars": 500, "offset": next_offset},
                ),
                registry,
            )
            self.assertIsNone(result2.error)
            self.assertIsNotNone(result2.native_result)
            self.assertEqual(result2.native_result.data.get("chars_read"), 500)

    def test_read_text_file_tolerates_non_utf8_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "gbk.txt").write_bytes("中文测试内容".encode("gbk"))
            registry = NativeToolRegistry(workspace_filesystem_tools(workspace))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="read_text_file",
                    id="call_1",
                    arguments={"path": "gbk.txt"},
                ),
                registry,
            )

            # Non-UTF-8 bytes must not crash the tool; they are decoded with
            # replacement so the caller still gets readable output.
            self.assertIsNone(result.error)
            self.assertIsNotNone(result.native_result)
            self.assertGreater(result.native_result.data.get("chars_read"), 0)

    def test_read_text_file_reports_character_offset_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "zh.txt").write_text("中文测试", encoding="utf-8")
            registry = NativeToolRegistry(workspace_filesystem_tools(workspace))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="read_text_file",
                    id="call_1",
                    arguments={"path": "zh.txt", "max_chars": 2},
                ),
                registry,
            )

            self.assertIsNone(result.error)
            self.assertIsNotNone(result.native_result)
            data = result.native_result.data
            self.assertEqual(data["chars_read"], 2)
            self.assertEqual(data["next_offset"], 2)
            self.assertEqual(data["offset_unit"], "chars")
            self.assertEqual(data["bytes_total"], 12)
            self.assertFalse(data["chars_total_known"])

            result2 = execute_native_tool_request(
                ModelToolRequest(
                    name="read_text_file",
                    id="call_2",
                    arguments={"path": "zh.txt", "max_chars": 10, "offset": 2},
                ),
                registry,
            )

            self.assertIsNone(result2.error)
            self.assertEqual(result2.native_result.data["chars_total"], 4)

    def test_read_text_file_rejects_symlink_after_path_validation(self) -> None:
        if not getattr(os, "O_NOFOLLOW", 0):
            self.skipTest("O_NOFOLLOW is unavailable on this platform")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            outside = Path(tmp).parent / "outside-filesystem-symlink-test.txt"
            outside.write_text("secret", encoding="utf-8")
            (workspace / "validated.txt").symlink_to(outside)
            registry = NativeToolRegistry(workspace_filesystem_tools(workspace))

            try:
                with patch(
                    "deepmate.tools.filesystem._workspace_path",
                    return_value=workspace / "validated.txt",
                ):
                    result = execute_native_tool_request(
                        ModelToolRequest(
                            name="read_text_file",
                            id="call_1",
                            arguments={"path": "validated.txt"},
                        ),
                        registry,
                    )
            finally:
                outside.unlink(missing_ok=True)

            self.assertIsNotNone(result.error)
            self.assertIn("path is not a readable file", result.error.message)

    def test_edit_text_file_reports_non_utf8_as_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "gbk.txt").write_bytes("中文测试内容".encode("gbk"))
            registry = NativeToolRegistry(
                workspace_filesystem_tools(workspace, include_write_tools=True)
            )

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="edit_text_file",
                    id="call_1",
                    arguments={
                        "path": "gbk.txt",
                        "old_text": "中文",
                        "new_text": "文本",
                    },
                ),
                registry,
                ToolAccessPolicy(mode=ToolAccessMode.WORKSPACE_WRITE),
            )

            self.assertIsNotNone(result.error)
            self.assertIn("requires UTF-8 text", result.error.message)
            self.assertNotIn("UnicodeDecodeError", result.error.message)

    def test_read_document_extracts_docx_text_without_optional_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            docx = workspace / "brief.docx"
            _write_minimal_docx(docx, "Deepmate document text")
            registry = NativeToolRegistry(workspace_document_tools(workspace))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="read_document",
                    id="call_1",
                    arguments={"path": "brief.docx"},
                ),
                registry,
            )

            self.assertIsNone(result.error)
            self.assertIsNotNone(result.model_result)
            self.assertIn("Deepmate document text", result.model_result.content)
            self.assertEqual(result.model_result.data["format"], "docx")

    def test_read_document_detects_gbk_text_with_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "notes.md").write_bytes("中文测试内容".encode("gbk"))
            registry = NativeToolRegistry(workspace_document_tools(workspace))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="read_document",
                    id="call_1",
                    arguments={"path": "notes.md"},
                ),
                registry,
            )

            self.assertIsNone(result.error)
            self.assertIsNotNone(result.model_result)
            self.assertIn("中文测试内容", result.model_result.content)
            self.assertIn("Warning: Detected non-UTF-8", result.model_result.content)
            self.assertIn(result.model_result.data["encoding"], {"gb18030", "gbk"})
            self.assertTrue(result.model_result.data["warnings"])

    def test_read_document_reports_invalid_docx_as_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "fake.docx").write_bytes(b"PK\x03\x04 not real")
            registry = NativeToolRegistry(workspace_document_tools(workspace))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="read_document",
                    id="call_1",
                    arguments={"path": "fake.docx"},
                ),
                registry,
            )

            self.assertIsNotNone(result.error)
            self.assertIn("invalid DOCX file", result.error.message)
            self.assertNotIn("BadZipFile", result.error.message)

    def test_inspect_table_profiles_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "data.csv").write_text(
                "name,score\nA,10\nB,20\nB,\n",
                encoding="utf-8",
            )
            registry = NativeToolRegistry(workspace_document_tools(workspace))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="inspect_table",
                    id="call_1",
                    arguments={"path": "data.csv", "preview_rows": 2},
                ),
                registry,
            )

            self.assertIsNone(result.error)
            self.assertIsNotNone(result.model_result)
            self.assertIn("Rows: 3 total", result.model_result.content)
            profiles = result.model_result.data["column_profiles"]
            self.assertEqual(profiles[1]["name"], "score")
            self.assertEqual(profiles[1]["type"], "number")
            self.assertEqual(profiles[1]["missing"], 1)

    def test_inspect_table_jsonl_skips_comment_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "data.jsonl").write_text(
                "# header comment\n{\"a\": 1}\n// generated\n{\"a\": 2}\n",
                encoding="utf-8",
            )
            registry = NativeToolRegistry(workspace_document_tools(workspace))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="inspect_table",
                    id="call_1",
                    arguments={"path": "data.jsonl"},
                ),
                registry,
            )

            self.assertIsNone(result.error)
            self.assertIsNotNone(result.model_result)
            self.assertIn("Rows: 2 total", result.model_result.content)
            self.assertIn("Warning: Skipped 2 comment/blank JSONL line", result.model_result.content)
            self.assertEqual(result.model_result.data["total_rows"], 2)
            self.assertTrue(result.model_result.data["warnings"])

    def test_inspect_table_json_prefers_data_array_over_metadata_array(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "data.json").write_text(
                json.dumps(
                    {
                        "meta": [{"name": "source", "value": "generated"}],
                        "data": [{"name": "A", "score": 10}, {"name": "B", "score": 20}],
                    }
                ),
                encoding="utf-8",
            )
            registry = NativeToolRegistry(workspace_document_tools(workspace))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="inspect_table",
                    id="call_1",
                    arguments={"path": "data.json"},
                ),
                registry,
            )

            self.assertIsNone(result.error)
            self.assertEqual(result.model_result.data["columns"], ("name", "score"))
            self.assertEqual(result.model_result.data["total_rows"], 2)
            self.assertFalse(result.model_result.data["warnings"])

    def test_inspect_table_json_warns_when_selecting_ambiguous_array(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "data.json").write_text(
                json.dumps(
                    {
                        "meta": [{"kind": "source"}],
                        "payload": [
                            {"name": "A", "score": 10},
                            {"name": "B", "score": 20},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            registry = NativeToolRegistry(workspace_document_tools(workspace))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="inspect_table",
                    id="call_1",
                    arguments={"path": "data.json"},
                ),
                registry,
            )

            self.assertIsNone(result.error)
            self.assertEqual(result.model_result.data["columns"], ("name", "score"))
            self.assertTrue(result.model_result.data["warnings"])
            self.assertIn("Multiple JSON array fields", result.model_result.content)

    def test_inspect_table_reports_invalid_xlsx_as_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "fake.xlsx").write_bytes(b"PK\x03\x04 not real")
            registry = NativeToolRegistry(workspace_document_tools(workspace))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="inspect_table",
                    id="call_1",
                    arguments={"path": "fake.xlsx"},
                ),
                registry,
            )

            self.assertIsNotNone(result.error)
            self.assertIn("invalid XLSX file", result.error.message)
            self.assertNotIn("BadZipFile", result.error.message)

    def test_inspect_table_xlsx_without_cell_references_preserves_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            path = workspace / "no_refs.xlsx"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    "xl/workbook.xml",
                    """<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>""",
                )
                archive.writestr(
                    "xl/_rels/workbook.xml.rels",
                    """<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>""",
                )
                archive.writestr(
                    "xl/worksheets/sheet1.xml",
                    """<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row><c t="inlineStr"><is><t>name</t></is></c><c t="inlineStr"><is><t>score</t></is></c></row><row><c t="inlineStr"><is><t>A</t></is></c><c><v>10</v></c></row></sheetData></worksheet>""",
                )
            registry = NativeToolRegistry(workspace_document_tools(workspace))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="inspect_table",
                    id="call_1",
                    arguments={"path": "no_refs.xlsx"},
                ),
                registry,
            )

        self.assertIsNone(result.error)
        self.assertEqual(result.model_result.data["columns"], ("name", "score"))
        self.assertEqual(result.model_result.data["preview"][0]["name"], "A")
        self.assertEqual(result.model_result.data["preview"][0]["score"], 10)

    def test_review_artifact_flags_markdown_delivery_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "prd.md").write_text(
                "# PRD\n\n## Summary\n\n![Flow](artifacts/flow.svg)\n\nTODO\n",
                encoding="utf-8",
            )
            registry = NativeToolRegistry(workspace_artifact_tools(workspace))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="review_artifact",
                    id="call_1",
                    arguments={
                        "path": "prd.md",
                        "require_sources": True,
                        "require_summary": True,
                        "require_acceptance_criteria": True,
                        "require_diagram_captions": True,
                    },
                ),
                registry,
            )

            self.assertIsNone(result.error)
            self.assertIsNotNone(result.model_result)
            codes = {finding["code"] for finding in result.model_result.data["findings"]}
            self.assertFalse(result.model_result.data["passed"])
            self.assertIn("placeholder", codes)
            self.assertIn("missing_sources", codes)
            self.assertIn("missing_acceptance_criteria", codes)
            self.assertIn("missing_image_caption", codes)

    def test_review_artifact_passes_markdown_with_sources_and_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "prd.md").write_text(
                (
                    "# PRD\n\n"
                    "## Summary\n\nA concise product note.\n\n"
                    "## Acceptance Criteria\n\n- Given a user, when they export, then HTML is written.\n\n"
                    "## Sources\n\n- [README](README.md)\n"
                ),
                encoding="utf-8",
            )
            registry = NativeToolRegistry(workspace_artifact_tools(workspace))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="review_artifact",
                    id="call_1",
                    arguments={
                        "path": "prd.md",
                        "require_sources": True,
                        "require_summary": True,
                        "require_acceptance_criteria": True,
                    },
                ),
                registry,
            )

            self.assertIsNone(result.error)
            self.assertIsNotNone(result.model_result)
            self.assertTrue(result.model_result.data["passed"])

    def test_review_artifact_flags_markdown_starting_below_title(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "brief.md").write_text(
                "## Summary\n\nA concise note.\n",
                encoding="utf-8",
            )
            registry = NativeToolRegistry(workspace_artifact_tools(workspace))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="review_artifact",
                    id="call_1",
                    arguments={"path": "brief.md"},
                ),
                registry,
            )

            self.assertIsNone(result.error)
            self.assertIsNotNone(result.model_result)
            codes = {finding["code"] for finding in result.model_result.data["findings"]}
            self.assertFalse(result.model_result.data["passed"])
            self.assertIn("missing_title", codes)
            self.assertIn("heading_starts_too_deep", codes)

    def test_review_artifact_flags_html_asset_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "report.html").write_text(
                (
                    "<!doctype html><html><head><title>Report</title>"
                    '<meta name="viewport" content="width=device-width"></head>'
                    "<body><h1>Report</h1><h2>Sources</h2><a href=\"README.md\">README</a>"
                    '<p class="asset-note">SVG image not embedded: SVG contains unsafe markup.</p>'
                    "<style>@media print { body { color: #000; } }</style></body></html>"
                ),
                encoding="utf-8",
            )
            registry = NativeToolRegistry(workspace_artifact_tools(workspace))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="review_artifact",
                    id="call_1",
                    arguments={"path": "report.html", "require_sources": True},
                ),
                registry,
            )

            self.assertIsNone(result.error)
            self.assertIsNotNone(result.model_result)
            codes = {finding["code"] for finding in result.model_result.data["findings"]}
            self.assertIn("asset_not_embedded", codes)

    def test_review_artifact_flags_svg_marker_and_unsafe_markup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "diagram.svg").write_text(
                (
                    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
                    '<image href="https://example.com/pixel.png"/>'
                    '<path d="M0 0 L10 10" marker-end="url(#missing)"/></svg>'
                ),
                encoding="utf-8",
            )
            registry = NativeToolRegistry(workspace_artifact_tools(workspace))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="review_artifact",
                    id="call_1",
                    arguments={"path": "diagram.svg"},
                ),
                registry,
            )

            self.assertIsNone(result.error)
            self.assertIsNotNone(result.model_result)
            codes = {finding["code"] for finding in result.model_result.data["findings"]}
            self.assertIn("unsafe_svg_markup", codes)
            self.assertIn("missing_marker_defs", codes)

    def test_review_artifact_flags_svg_animation_and_top_target_markup(self) -> None:
        unsafe_snippets = (
            '<svg><animate attributeName="onload" to="alert(1)"/></svg>',
            '<svg><set attributeName="href" to="javascript:alert(1)"/></svg>',
            '<?xml-stylesheet href="http://example.test/x.css"?><svg/>',
            '<svg><a target="_top" href="#ok"><text>open</text></a></svg>',
            '<svg><object data="https://example.test/widget"></object></svg>',
            '<svg><use href="https://example.test/symbol.svg#x"/></svg>',
            '<svg><image href="https://example.test/pixel.png"/></svg>',
            '<svg><link href="https://example.test/x.css" rel="stylesheet"/></svg>',
            '<svg><meta http-equiv="refresh" content="0;url=https://example.test"/></svg>',
            '<svg requiredExtensions="https://example.test/ext"></svg>',
            '<svg externalResourcesRequired="true"></svg>',
            '<svg><style>@import "https://example.test/x.css";</style></svg>',
            '<svg><feImage href="https://example.test/pixel.png"/></svg>',
        )
        for index, markup in enumerate(unsafe_snippets):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as tmp:
                workspace = Path(tmp)
                (workspace / "diagram.svg").write_text(markup, encoding="utf-8")
                registry = NativeToolRegistry(workspace_artifact_tools(workspace))

                result = execute_native_tool_request(
                    ModelToolRequest(
                        name="review_artifact",
                        id="call_1",
                        arguments={"path": "diagram.svg"},
                    ),
                    registry,
                )

                self.assertIsNone(result.error)
                self.assertIsNotNone(result.model_result)
                codes = {
                    finding["code"]
                    for finding in result.model_result.data["findings"]
                }
                self.assertIn("unsafe_svg_markup", codes)

    def test_render_html_report_writes_with_checkpoint_and_hook(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            source = workspace / "report.md"
            output = workspace / "report.html"
            source.write_text(
                "# Deepmate Report\n\n## Summary\n\nA polished report.\n",
                encoding="utf-8",
            )
            observed: list[object] = []

            def checkpoint(operation: str, path: Path, after_content: str) -> None:
                observed.extend((operation, path.name, "<html" in after_content))

            registry = NativeToolRegistry(
                workspace_report_tools(
                    workspace,
                    write_checkpoint=checkpoint,
                    hook_context=_hook_context(HookEvent.WRITE_BEFORE, HookActionType.TRACE),
                )
            )

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="render_html_report",
                    id="call_1",
                    arguments={
                        "source_path": "report.md",
                        "output_path": "report.html",
                        "theme": "prussian",
                    },
                ),
                registry,
                ToolAccessPolicy(mode=ToolAccessMode.WORKSPACE_WRITE),
            )

            self.assertIsNone(result.error)
            self.assertTrue(output.is_file())
            self.assertIn("--primary: #003153", output.read_text(encoding="utf-8"))
            self.assertEqual(observed, ["render_html_report", "report.html", True])
            self.assertIsNotNone(result.model_result)
            self.assertEqual(result.model_result.data["warnings"], 0)

    def test_render_html_report_renders_relative_svg_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "artifacts").mkdir()
            (workspace / "artifacts" / "diagram.svg").write_text(
                '<svg xmlns="http://www.w3.org/2000/svg"></svg>',
                encoding="utf-8",
            )
            source = workspace / "report.md"
            output = workspace / "report.html"
            source.write_text(
                '# Deepmate Report\n\n![Runtime diagram](artifacts/diagram.svg "Runtime flow")\n',
                encoding="utf-8",
            )
            registry = NativeToolRegistry(workspace_report_tools(workspace))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="render_html_report",
                    id="call_1",
                    arguments={
                        "source_path": "report.md",
                        "output_path": "report.html",
                    },
                ),
                registry,
                ToolAccessPolicy(mode=ToolAccessMode.WORKSPACE_WRITE),
            )

            self.assertIsNone(result.error)
            html = output.read_text(encoding="utf-8")
            self.assertIn('<figure class="diagram" aria-label="Runtime diagram">', html)
            self.assertIn('<svg xmlns="http://www.w3.org/2000/svg"></svg>', html)
            self.assertIn("<figcaption>Runtime flow</figcaption>", html)
            self.assertIsNotNone(result.model_result)
            checks = {check["name"]: check for check in result.model_result.data["checks"]}
            self.assertTrue(checks["svg_images"]["passed"])

    def test_render_html_report_does_not_rewrite_inline_svg_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "artifacts").mkdir()
            (workspace / "artifacts" / "diagram.svg").write_text(
                (
                    '<svg xmlns="http://www.w3.org/2000/svg">'
                    "<text>[A](B) *not markdown*</text></svg>"
                ),
                encoding="utf-8",
            )
            (workspace / "report.md").write_text(
                "# Deepmate Report\n\n![Diagram](artifacts/diagram.svg)\n",
                encoding="utf-8",
            )
            registry = NativeToolRegistry(workspace_report_tools(workspace))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="render_html_report",
                    id="call_1",
                    arguments={
                        "source_path": "report.md",
                        "output_path": "report.html",
                    },
                ),
                registry,
                ToolAccessPolicy(mode=ToolAccessMode.WORKSPACE_WRITE),
            )

            self.assertIsNone(result.error)
            html = (workspace / "report.html").read_text(encoding="utf-8")
            self.assertIn("<text>[A](B) *not markdown*</text>", html)
            self.assertNotIn('<a href="B"', html)
            self.assertNotIn("<em>not markdown</em>", html)

    def test_render_html_report_keeps_image_syntax_inside_inline_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "artifacts").mkdir()
            (workspace / "artifacts" / "diagram.svg").write_text(
                '<svg xmlns="http://www.w3.org/2000/svg"></svg>',
                encoding="utf-8",
            )
            (workspace / "report.md").write_text(
                "# Deepmate Report\n\nUse `![Diagram](artifacts/diagram.svg)` literally.\n",
                encoding="utf-8",
            )
            registry = NativeToolRegistry(workspace_report_tools(workspace))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="render_html_report",
                    id="call_1",
                    arguments={
                        "source_path": "report.md",
                        "output_path": "report.html",
                    },
                ),
                registry,
                ToolAccessPolicy(mode=ToolAccessMode.WORKSPACE_WRITE),
            )

            self.assertIsNone(result.error)
            html = (workspace / "report.html").read_text(encoding="utf-8")
            self.assertIn("<code>![Diagram](artifacts/diagram.svg)</code>", html)
            self.assertNotIn('<figure class="diagram"', html)

    def test_render_html_report_does_not_inline_unsafe_svg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "artifacts").mkdir()
            (workspace / "artifacts" / "unsafe.svg").write_text(
                '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>',
                encoding="utf-8",
            )
            (workspace / "report.md").write_text(
                "# Deepmate Report\n\n![Unsafe](artifacts/unsafe.svg)\n",
                encoding="utf-8",
            )
            registry = NativeToolRegistry(workspace_report_tools(workspace))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="render_html_report",
                    id="call_1",
                    arguments={
                        "source_path": "report.md",
                        "output_path": "report.html",
                    },
                ),
                registry,
                ToolAccessPolicy(mode=ToolAccessMode.WORKSPACE_WRITE),
            )

            self.assertIsNone(result.error)
            html = (workspace / "report.html").read_text(encoding="utf-8")
            self.assertNotIn("<script>", html)
            self.assertNotIn('<img src="artifacts/unsafe.svg"', html)
            self.assertIn("SVG image not embedded: SVG contains unsafe markup.", html)
            self.assertIsNotNone(result.model_result)
            checks = {check["name"]: check for check in result.model_result.data["checks"]}
            self.assertFalse(checks["svg_images"]["passed"])

    def test_render_html_report_rejects_svg_event_namespace_and_css_bypasses(self) -> None:
        unsafe_payloads = (
            '<svg/onload=alert(1)></svg>',
            '<svg xmlns="http://www.w3.org/2000/svg"><svg:script>alert(1)</svg:script></svg>',
            '<svg xmlns="http://www.w3.org/2000/svg"><style>rect{width:expression(alert(1))}</style></svg>',
            '<svg xmlns="http://www.w3.org/2000/svg"><style>rect{fill:url(https://example.com/a.png)}</style></svg>',
        )
        for index, payload in enumerate(unsafe_payloads):
            with self.subTest(index=index):
                with tempfile.TemporaryDirectory() as tmp:
                    workspace = Path(tmp)
                    (workspace / "artifacts").mkdir()
                    (workspace / "artifacts" / "unsafe.svg").write_text(
                        payload,
                        encoding="utf-8",
                    )
                    (workspace / "report.md").write_text(
                        "# Deepmate Report\n\n![Unsafe](artifacts/unsafe.svg)\n",
                        encoding="utf-8",
                    )
                    registry = NativeToolRegistry(workspace_report_tools(workspace))

                    result = execute_native_tool_request(
                        ModelToolRequest(
                            name="render_html_report",
                            id="call_1",
                            arguments={
                                "source_path": "report.md",
                                "output_path": "report.html",
                            },
                        ),
                        registry,
                        ToolAccessPolicy(mode=ToolAccessMode.WORKSPACE_WRITE),
                    )

                    self.assertIsNone(result.error)
                    html = (workspace / "report.html").read_text(encoding="utf-8")
                    self.assertIn("SVG image not embedded", html)
                    self.assertNotIn(payload, html)

    def test_render_html_report_does_not_inline_svg_with_external_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "artifacts").mkdir()
            (workspace / "artifacts" / "external.svg").write_text(
                (
                    '<svg xmlns="http://www.w3.org/2000/svg">'
                    '<image href="https://example.com/pixel.png"/></svg>'
                ),
                encoding="utf-8",
            )
            (workspace / "report.md").write_text(
                "# Deepmate Report\n\n![External](artifacts/external.svg)\n",
                encoding="utf-8",
            )
            registry = NativeToolRegistry(workspace_report_tools(workspace))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="render_html_report",
                    id="call_1",
                    arguments={
                        "source_path": "report.md",
                        "output_path": "report.html",
                    },
                ),
                registry,
                ToolAccessPolicy(mode=ToolAccessMode.WORKSPACE_WRITE),
            )

            self.assertIsNone(result.error)
            html = (workspace / "report.html").read_text(encoding="utf-8")
            self.assertNotIn("https://example.com/pixel.png", html)
            self.assertNotIn('<img src="artifacts/external.svg"', html)
            self.assertIn("SVG image not embedded: SVG contains unsafe markup.", html)
            self.assertIsNotNone(result.model_result)
            checks = {check["name"]: check for check in result.model_result.data["checks"]}
            self.assertFalse(checks["svg_images"]["passed"])

    def test_render_html_report_requires_workspace_write_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "report.md").write_text("# Title", encoding="utf-8")
            registry = NativeToolRegistry(workspace_report_tools(workspace))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="render_html_report",
                    id="call_1",
                    arguments={
                        "source_path": "report.md",
                        "output_path": "report.html",
                    },
                ),
                registry,
                ToolAccessPolicy(mode=ToolAccessMode.READ_ONLY),
            )

            self.assertIsNotNone(result.error)
            self.assertEqual(result.error.code, "native_tool_denied")

    def test_render_tech_diagram_can_be_embedded_in_html_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "artifacts").mkdir()
            policy = ToolAccessPolicy(mode=ToolAccessMode.WORKSPACE_WRITE)
            diagram_registry = NativeToolRegistry(workspace_diagram_tools(workspace))

            diagram_result = execute_native_tool_request(
                ModelToolRequest(
                    name="render_tech_diagram",
                    id="call_1",
                    arguments={
                        "diagram_type": "timeline",
                        "output_path": "artifacts/roadmap.svg",
                        "title": "Delivery Roadmap",
                        "theme": "forest",
                        "milestones": [
                            {"label": "Design", "time": "Week 1"},
                            {"label": "Build", "time": "Week 2"},
                            {"label": "Review", "time": "Week 3"},
                        ],
                        "phases": [
                            {"label": "Foundation", "detail": "Design and build"},
                            {"label": "Closure", "detail": "Review and ship"},
                        ],
                    },
                ),
                diagram_registry,
                policy,
            )
            self.assertIsNone(diagram_result.error)

            (workspace / "report.md").write_text(
                "# Delivery Report\n\n"
                "## Roadmap\n\n"
                '![Delivery roadmap](artifacts/roadmap.svg "Delivery roadmap")\n',
                encoding="utf-8",
            )
            report_registry = NativeToolRegistry(workspace_report_tools(workspace))
            report_result = execute_native_tool_request(
                ModelToolRequest(
                    name="render_html_report",
                    id="call_2",
                    arguments={
                        "source_path": "report.md",
                        "output_path": "report.html",
                        "theme": "forest",
                    },
                ),
                report_registry,
                policy,
            )

            self.assertIsNone(report_result.error)
            html = (workspace / "report.html").read_text(encoding="utf-8")
            self.assertIn('<figure class="diagram" aria-label="Delivery roadmap">', html)
            self.assertIn("<svg", html)
            self.assertIn("Foundation", html)
            self.assertIn("<figcaption>Delivery roadmap</figcaption>", html)
            self.assertIsNotNone(report_result.model_result)
            checks = {
                check["name"]: check
                for check in report_result.model_result.data["checks"]
            }
            self.assertTrue(checks["svg_images"]["passed"])

    def test_render_tech_diagram_writes_svg_with_static_qa(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            output = workspace / "diagram.svg"
            observed: list[object] = []

            def checkpoint(operation: str, path: Path, after_content: str) -> None:
                observed.extend((operation, path.name, "<svg" in after_content))

            registry = NativeToolRegistry(
                workspace_diagram_tools(
                    workspace,
                    write_checkpoint=checkpoint,
                    hook_context=_hook_context(HookEvent.WRITE_BEFORE, HookActionType.TRACE),
                )
            )

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="render_tech_diagram",
                    id="call_1",
                    arguments={
                        "diagram_type": "agent_architecture",
                        "output_path": "diagram.svg",
                        "title": "Deepmate Runtime",
                        "theme": "prussian",
                        "groups": [
                            {"id": "runtime", "label": "Runtime"},
                            {"id": "capabilities", "label": "Capabilities"},
                        ],
                        "nodes": [
                            {
                                "id": "agent",
                                "label": "Agent Runtime",
                                "kind": "agent",
                                "group": "runtime",
                            },
                            {
                                "id": "tools",
                                "label": "Native Tools",
                                "kind": "tool",
                                "group": "capabilities",
                            },
                        ],
                        "edges": [
                            {
                                "source": "agent",
                                "target": "tools",
                                "label": "calls",
                                "flow": "control",
                            }
                        ],
                    },
                ),
                registry,
                ToolAccessPolicy(mode=ToolAccessMode.WORKSPACE_WRITE),
            )

            self.assertIsNone(result.error)
            self.assertTrue(output.is_file())
            svg = output.read_text(encoding="utf-8")
            self.assertIn("<svg", svg)
            self.assertIn("Deepmate Runtime", svg)
            self.assertEqual(observed, ["render_tech_diagram", "diagram.svg", True])
            self.assertIsNotNone(result.model_result)
            self.assertEqual(result.model_result.data["warnings"], 0)
            self.assertEqual(result.model_result.data["diagram_type"], "agent_architecture")

    def test_render_tech_diagram_covers_supported_types_and_themes(self) -> None:
        cases = (
            (
                "architecture",
                "prussian",
                {
                    "nodes": [
                        {"id": "client", "label": "Client", "kind": "user", "group": "Input"},
                        {"id": "runtime", "label": "Runtime", "kind": "agent", "group": "Core"},
                    ],
                    "edges": [
                        {"source": "client", "target": "runtime", "label": "request", "flow": "primary"}
                    ],
                },
            ),
            (
                "agent_architecture",
                "forest",
                {
                    "nodes": [
                        {"id": "agent", "label": "Agent Runtime", "kind": "agent", "group": "Core"},
                        {"id": "memory", "label": "Memory Store", "kind": "memory", "group": "State"},
                    ],
                    "edges": [
                        {"source": "agent", "target": "memory", "label": "read", "flow": "read"}
                    ],
                },
            ),
            (
                "flowchart",
                "graphite",
                {
                    "nodes": [
                        {"id": "plan", "label": "Plan", "kind": "process"},
                        {"id": "decide", "label": "Decide", "kind": "decision"},
                    ],
                    "edges": [
                        {"source": "plan", "target": "decide", "label": "next", "flow": "control"}
                    ],
                },
            ),
            (
                "sequence",
                "blueprint",
                {
                    "nodes": [
                        {"id": "user", "label": "User"},
                        {"id": "agent", "label": "Agent"},
                        {"id": "tool", "label": "Tool"},
                    ],
                    "edges": [
                        {"source": "user", "target": "agent", "label": "task", "flow": "primary"},
                        {
                            "source": "agent",
                            "target": "tool",
                            "label": "call",
                            "flow": "control",
                            "frame": "Tool use",
                        },
                    ],
                },
            ),
            (
                "comparison",
                "prussian",
                {
                    "columns": ["A", "B"],
                    "rows": [{"label": "Cost", "values": ["Low", "Medium"]}],
                },
            ),
            (
                "timeline",
                "forest",
                {
                    "milestones": [
                        {"label": "Design", "time": "Week 1"},
                        {"label": "Build", "time": "Week 2"},
                    ],
                    "phases": [{"label": "Foundation", "detail": "Design and build"}],
                },
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            registry = NativeToolRegistry(workspace_diagram_tools(workspace))
            policy = ToolAccessPolicy(mode=ToolAccessMode.WORKSPACE_WRITE)

            for index, (diagram_type, theme, payload) in enumerate(cases):
                output = f"diagram-{index}.svg"
                result = execute_native_tool_request(
                    ModelToolRequest(
                        name="render_tech_diagram",
                        id=f"call_{index}",
                        arguments={
                            "diagram_type": diagram_type,
                            "output_path": output,
                            "title": f"{diagram_type} diagram",
                            "theme": theme,
                            **payload,
                        },
                    ),
                    registry,
                    policy,
                )

                self.assertIsNone(result.error, f"{diagram_type} failed: {result.error}")
                svg = (workspace / output).read_text(encoding="utf-8")
                self.assertIn("<svg", svg)
                self.assertIn("viewBox=", svg)
                self.assertNotRegex(svg, r"TODO|TBD|FIXME|<what you need>")
                self.assertEqual(_missing_marker_refs(svg), set(), diagram_type)

    def test_render_tech_diagram_schema_covers_rendered_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            tool = NativeToolRegistry(workspace_diagram_tools(workspace)).get(
                "render_tech_diagram"
            )
            properties = tool.input_schema["properties"]

            for key in (
                "nodes",
                "edges",
                "groups",
                "columns",
                "rows",
                "milestones",
                "phases",
            ):
                self.assertIn(key, properties)

    def test_render_tech_diagram_returns_actionable_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            registry = NativeToolRegistry(workspace_diagram_tools(workspace))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="render_tech_diagram",
                    id="call_1",
                    arguments={
                        "diagram_type": "comparison",
                        "output_path": "bad.svg",
                        "columns": ["A", "B", "C"],
                        "rows": [{"label": "Fit", "values": ["Yes"]}],
                        "nodes": [{"id": "node_a", "label": "A"}],
                        "edges": [{"source": "node_a", "target": "missing"}],
                    },
                ),
                registry,
                ToolAccessPolicy(mode=ToolAccessMode.WORKSPACE_WRITE),
            )

            self.assertIsNone(result.error)
            self.assertIsNotNone(result.model_result)
            messages = " ".join(
                str(check["message"]) for check in result.model_result.data["checks"]
            )
            self.assertIn("known=node_a", messages)
            self.assertIn("3 columns", messages)

    def test_render_tech_diagram_requires_workspace_write_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            registry = NativeToolRegistry(workspace_diagram_tools(workspace))

            result = execute_native_tool_request(
                ModelToolRequest(
                    name="render_tech_diagram",
                    id="call_1",
                    arguments={
                        "diagram_type": "timeline",
                        "output_path": "roadmap.svg",
                        "milestones": [{"label": "Design", "time": "Week 1"}],
                    },
                ),
                registry,
                ToolAccessPolicy(mode=ToolAccessMode.READ_ONLY),
            )

            self.assertIsNotNone(result.error)
            self.assertEqual(result.error.code, "native_tool_denied")


class WebResearchToolTests(unittest.TestCase):
    def test_web_tools_are_only_returned_when_network_is_enabled(self) -> None:
        self.assertEqual(web_research_tools(network_enabled=False), ())
        self.assertEqual(
            tuple(tool.name for tool in web_research_tools(network_enabled=True)),
            ("web_search", "web_fetch"),
        )

    def test_web_fetch_extracts_readable_html_with_mocked_network(self) -> None:
        registry = NativeToolRegistry(web_research_tools(network_enabled=True))
        response = _FakeResponse(
            b"<html><head><title>Example</title></head><body><h1>Hello</h1><p>World</p></body></html>",
            "https://example.com/page",
            "text/html; charset=utf-8",
        )

        with patch(
            "deepmate.tools.url_safety.socket.getaddrinfo",
            return_value=[_addr("192.0.2.34")],
        ):
            with patch("deepmate.tools.web._open_public_request", return_value=response):
                result = execute_native_tool_request(
                    ModelToolRequest(
                        name="web_fetch",
                        id="call_1",
                        arguments={"url": "https://example.com/page"},
                    ),
                    registry,
                )

        self.assertIsNone(result.error)
        self.assertIsNotNone(result.model_result)
        self.assertIn("# Hello", result.model_result.content)
        self.assertEqual(result.model_result.data["title"], "Example")

    def test_web_fetch_uses_html_meta_charset(self) -> None:
        registry = NativeToolRegistry(web_research_tools(network_enabled=True))
        response = _FakeResponse(
            "<html><head><meta charset=\"gbk\"><title>中文</title></head><body><p>中文测试</p></body></html>".encode("gbk"),
            "https://example.com/page",
            "text/html",
        )

        with patch(
            "deepmate.tools.url_safety.socket.getaddrinfo",
            return_value=[_addr("192.0.2.34")],
        ):
            with patch("deepmate.tools.web._open_public_request", return_value=response):
                result = execute_native_tool_request(
                    ModelToolRequest(
                        name="web_fetch",
                        id="call_1",
                        arguments={"url": "https://example.com/page"},
                    ),
                    registry,
                )

        self.assertIsNone(result.error)
        self.assertIn("中文测试", result.model_result.content)
        self.assertEqual(result.model_result.data["title"], "中文")

    def test_web_fetch_detects_html_after_long_preamble(self) -> None:
        registry = NativeToolRegistry(web_research_tools(network_enabled=True))
        response = _FakeResponse(
            (
                "<!doctype html>\n"
                + ("<!-- long generated preamble -->\n" * 80)
                + "<html><head><title>Preamble</title></head><body><h1>Hello</h1></body></html>"
            ).encode(),
            "https://example.com/page",
            "text/plain; charset=utf-8",
        )

        with patch(
            "deepmate.tools.url_safety.socket.getaddrinfo",
            return_value=[_addr("192.0.2.34")],
        ):
            with patch("deepmate.tools.web._open_public_request", return_value=response):
                result = execute_native_tool_request(
                    ModelToolRequest(
                        name="web_fetch",
                        id="call_1",
                        arguments={"url": "https://example.com/page"},
                    ),
                    registry,
                )

        self.assertIsNone(result.error)
        self.assertIn("# Hello", result.model_result.content)
        self.assertEqual(result.model_result.data["title"], "Preamble")

    def test_web_fetch_marks_truncated_content_inline(self) -> None:
        registry = NativeToolRegistry(web_research_tools(network_enabled=True))
        response = _FakeResponse(
            b"abcdefghijklmnopqrstuvwxyz",
            "https://example.com/page",
            "text/plain; charset=utf-8",
        )

        with patch(
            "deepmate.tools.url_safety.socket.getaddrinfo",
            return_value=[_addr("192.0.2.34")],
        ):
            with patch("deepmate.tools.web._open_public_request", return_value=response):
                result = execute_native_tool_request(
                    ModelToolRequest(
                        name="web_fetch",
                        id="call_1",
                        arguments={
                            "url": "https://example.com/page",
                            "max_chars": 10,
                        },
                    ),
                    registry,
                )

        self.assertIsNone(result.error)
        self.assertIsNotNone(result.model_result)
        self.assertIn("[truncated - total=26 chars, returned=10 chars]", result.model_result.content)
        self.assertEqual(result.model_result.data["chars"], 10)
        self.assertEqual(result.model_result.data["total_chars"], 26)
        self.assertTrue(result.model_result.data["truncated"])

    def test_web_search_parses_duckduckgo_results_with_mocked_network(self) -> None:
        registry = NativeToolRegistry(web_research_tools(network_enabled=True))
        html = b"""
        <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fone">One</a>
        <div class="result__snippet">First result.</div>
        """
        response = _FakeResponse(html, "https://html.duckduckgo.com/html/?q=x", "text/html")

        with patch(
            "deepmate.tools.url_safety.socket.getaddrinfo",
            return_value=[_addr("198.51.100.232")],
        ):
            with patch("deepmate.tools.web._open_public_request", return_value=response):
                result = execute_native_tool_request(
                    ModelToolRequest(
                        name="web_search",
                        id="call_1",
                        arguments={"query": "deepmate", "max_results": 1},
                    ),
                    registry,
                )

        self.assertIsNone(result.error)
        self.assertIsNotNone(result.model_result)
        self.assertIn("https://example.com/one", result.model_result.content)
        self.assertEqual(result.model_result.data["backend"], "duckduckgo_html")

    def test_web_search_normalizes_url_like_query(self) -> None:
        registry = NativeToolRegistry(web_research_tools(network_enabled=True))
        html = b"""
        <a class="result__a" href="/l/?uddg=https%3A%2F%2Fopenai.com%2F">OpenAI</a>
        <div class="result__snippet">Homepage.</div>
        """
        response = _FakeResponse(html, "https://html.duckduckgo.com/html/?q=x", "text/html")
        seen_urls = []

        def fake_open(request):
            seen_urls.append(request.full_url)
            return response

        with patch(
            "deepmate.tools.url_safety.socket.getaddrinfo",
            return_value=[_addr("198.51.100.232")],
        ):
            with patch("deepmate.tools.web._open_public_request", side_effect=fake_open):
                result = execute_native_tool_request(
                    ModelToolRequest(
                        name="web_search",
                        id="call_1",
                        arguments={"query": "https://www.openai.com/", "max_results": 1},
                    ),
                    registry,
                )

        self.assertIsNone(result.error)
        self.assertIn("q=site%3Aopenai.com+openai.com", seen_urls[0])
        self.assertEqual(result.model_result.data["query"], "site:openai.com openai.com")
        self.assertEqual(result.model_result.data["original_query"], "https://www.openai.com/")

    def test_web_search_falls_back_when_duckduckgo_classes_change(self) -> None:
        registry = NativeToolRegistry(web_research_tools(network_enabled=True))
        html = ("""
        <html><body>
        <a class="new-result-title" href="/l/?uddg=https%3A%2F%2Fexample.com%2Ffallback">Fallback Result</a>
        <a href="https://duckduckgo.com/settings">Settings</a>
        </body></html>
        """).encode()
        response = _FakeResponse(html, "https://html.duckduckgo.com/html/?q=x", "text/html")

        with patch(
            "deepmate.tools.url_safety.socket.getaddrinfo",
            return_value=[_addr("198.51.100.232")],
        ):
            with patch("deepmate.tools.web._open_public_request", return_value=response):
                result = execute_native_tool_request(
                    ModelToolRequest(
                        name="web_search",
                        id="call_1",
                        arguments={"query": "deepmate", "max_results": 2},
                    ),
                    registry,
                )

        self.assertIsNone(result.error)
        self.assertIsNotNone(result.model_result)
        self.assertIn("Fallback Result", result.model_result.content)
        self.assertIn("https://example.com/fallback", result.model_result.content)
        self.assertIn("fallback link extraction", result.model_result.content)
        self.assertEqual(result.model_result.data["result_count"], 1)
        self.assertIn("fallback link extraction", result.model_result.data["parse_warning"])

    def test_web_search_uses_bing_when_duckduckgo_returns_no_results(self) -> None:
        registry = NativeToolRegistry(web_research_tools(network_enabled=True))
        duckduckgo_html = (
            "<html><body><main>" + ("No result markup. " * 80) + "</main></body></html>"
        ).encode()
        bing_html = b"""
        <html><body><ol id="b_results">
        <li class="b_algo">
          <h2><a href="https://example.com/bing">Bing Result</a></h2>
          <div class="b_caption"><p>Fallback snippet.</p></div>
        </li>
        </ol></body></html>
        """
        responses = [
            _FakeResponse(
                duckduckgo_html,
                "https://html.duckduckgo.com/html/?q=x",
                "text/html",
            ),
            _FakeResponse(
                bing_html,
                "https://www.bing.com/search?q=x",
                "text/html; charset=utf-8",
            ),
        ]

        with patch(
            "deepmate.tools.url_safety.socket.getaddrinfo",
            return_value=[_addr("192.0.2.34")],
        ):
            with patch("deepmate.tools.web._open_public_request", side_effect=responses):
                result = execute_native_tool_request(
                    ModelToolRequest(
                        name="web_search",
                        id="call_1",
                        arguments={"query": "deepmate", "max_results": 1},
                    ),
                    registry,
                )

        self.assertIsNone(result.error)
        self.assertIsNotNone(result.model_result)
        self.assertIn("Bing Result", result.model_result.content)
        self.assertIn("https://example.com/bing", result.model_result.content)
        self.assertEqual(result.model_result.data["backend"], "bing_html")
        self.assertEqual(result.model_result.data["result_count"], 1)
        self.assertIn("trying Bing HTML fallback", result.model_result.content)

    def test_web_search_filters_unsafe_result_urls(self) -> None:
        registry = NativeToolRegistry(web_research_tools(network_enabled=True))
        html = b"""
        <a class="result__a" href="/l/?uddg=http%3A%2F%2F127.0.0.1%2Fsecret">Local</a>
        <div class="result__snippet">Unsafe result.</div>
        <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fsafe">Safe</a>
        <div class="result__snippet">Safe result.</div>
        """
        response = _FakeResponse(html, "https://html.duckduckgo.com/html/?q=x", "text/html")

        with patch(
            "deepmate.tools.url_safety.socket.getaddrinfo",
            return_value=[_addr("192.0.2.34")],
        ):
            with patch("deepmate.tools.web._open_public_request", return_value=response):
                result = execute_native_tool_request(
                    ModelToolRequest(
                        name="web_search",
                        id="call_1",
                        arguments={"query": "deepmate", "max_results": 2},
                    ),
                    registry,
                )

        self.assertIsNone(result.error)
        self.assertIn("https://example.com/safe", result.model_result.content)
        self.assertNotIn("127.0.0.1", result.model_result.content)
        self.assertEqual(result.model_result.refs, ("https://example.com/safe",))
        self.assertEqual(result.model_result.data["result_count"], 1)
        self.assertIn("Filtered 1 unsafe", result.model_result.data["parse_warning"])

    def test_web_search_reports_unparseable_duckduckgo_html(self) -> None:
        registry = NativeToolRegistry(web_research_tools(network_enabled=True))
        html = ("<html><body><main>" + ("No result markup. " * 80) + "</main></body></html>").encode()
        response = _FakeResponse(html, "https://html.duckduckgo.com/html/?q=x", "text/html")

        with patch(
            "deepmate.tools.url_safety.socket.getaddrinfo",
            return_value=[_addr("198.51.100.232")],
        ):
            with patch("deepmate.tools.web._open_public_request", return_value=response):
                result = execute_native_tool_request(
                    ModelToolRequest(
                        name="web_search",
                        id="call_1",
                        arguments={"query": "deepmate", "max_results": 1},
                    ),
                    registry,
                )

        self.assertIsNone(result.error)
        self.assertIsNotNone(result.model_result)
        self.assertIn("HTML format may have changed", result.model_result.content)
        self.assertEqual(result.model_result.data["result_count"], 0)


def _write_minimal_docx(path: Path, text: str) -> None:
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "word/document.xml",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                "<w:body><w:p><w:r><w:t>"
                + escaped
                + "</w:t></w:r></w:p></w:body></w:document>"
            ),
        )


def _hook_context(event_name: HookEvent, action_type: HookActionType) -> HookRuntimeContext:
    signal_path = Path(tempfile.gettempdir()) / "deepmate-test-hook-signals.jsonl"
    registry = HookRegistry(
        (
            HookDefinition(
                hook_id="test-hook",
                event_name=event_name,
                layer=HookLayer.BUILTIN,
                actions=(HookAction(action_type=action_type),),
            ),
        )
    )
    return HookRuntimeContext.from_registry(
        registry,
        signal_store=HookSignalStore(signal_path),
    )


def _addr(address: str):
    return (2, 1, 6, "", (address, 443))


def _missing_marker_refs(svg: str) -> set[str]:
    refs = set(re.findall(r"url\(#([^)]+)\)", svg))
    defs = set(re.findall(r'<marker id="([^"]+)"', svg))
    return refs - defs


class _FakeResponse:
    def __init__(self, body: bytes, url: str, content_type: str) -> None:
        self._body = body
        self._url = url
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self, _size: int) -> bytes:
        return self._body

    def geturl(self) -> str:
        return self._url


class _FakeLspPipe:
    def __init__(self) -> None:
        self.data = b""

    def write(self, data: bytes) -> int:
        self.data += data
        return len(data)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None

    def fileno(self) -> int:
        return 123


class _FakeLspProcess:
    def __init__(self) -> None:
        self.stdin = _FakeLspPipe()
        self.stdout = _FakeLspPipe()
        self.stderr = None


if __name__ == "__main__":
    unittest.main()
