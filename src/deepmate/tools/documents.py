"""Read-only document and tabular inspection tools."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

from deepmate.tools.filesystem import _relative_path, _workspace_path
from deepmate.tools.registry import NativeTool, NativeToolResult

DEFAULT_MAX_CHARS = 30_000
MAX_CHARS = 100_000
DEFAULT_MAX_ROWS = 5_000
MAX_ROWS = 50_000
DEFAULT_PREVIEW_ROWS = 20
MAX_PREVIEW_ROWS = 100
MAX_DOCUMENT_BYTES = 20_000_000
MAX_ARCHIVE_MEMBER_BYTES = 30_000_000
TEXT_ENCODINGS = ("utf-8-sig", "utf-16", "gb18030", "gbk")
REPLACEMENT_CHAR = "\ufffd"


def workspace_document_tools(workspace_root: str | Path) -> tuple[NativeTool, ...]:
    """Return document reading and table inspection tools for one workspace."""
    root = Path(workspace_root).resolve()
    if not root.is_dir():
        raise ValueError(f"workspace root must be a directory: {root}")
    return (
        NativeTool(
            name="read_document",
            description="Extract readable text from a workspace document.",
            input_schema=_read_document_schema(),
            handler=lambda arguments: _read_document(root, arguments),
        ),
        NativeTool(
            name="inspect_table",
            description="Profile a CSV, TSV, JSON, JSONL, or XLSX table.",
            input_schema=_inspect_table_schema(),
            handler=lambda arguments: _inspect_table(root, arguments),
        ),
    )


def _read_document(root: Path, arguments: Mapping[str, object]) -> NativeToolResult:
    path = _workspace_path(root, _text(arguments, "path"))
    max_chars = _int(arguments, "max_chars", DEFAULT_MAX_CHARS, 1, MAX_CHARS)
    if not path.is_file():
        raise ValueError(f"path is not a file: {_relative_path(root, path)}")
    if path.stat().st_size > MAX_DOCUMENT_BYTES:
        raise ValueError(f"document exceeds {MAX_DOCUMENT_BYTES} bytes")

    suffix = path.suffix.lower()
    metadata: dict[str, object] = {"format": suffix.lstrip(".") or "text"}
    warnings: list[str] = []
    if suffix in {".txt", ".md", ".rst", ".log", ".json", ".yaml", ".yml"}:
        text, encoding, warning = _read_text_with_encoding_hint(path)
        metadata["encoding"] = encoding
        if warning:
            warnings.append(warning)
    elif suffix in {".html", ".htm"}:
        parser = _DocumentHtmlParser()
        html_text, encoding, warning = _read_text_with_encoding_hint(path)
        parser.feed(html_text)
        text = parser.render()
        metadata["title"] = parser.title
        metadata["encoding"] = encoding
        if warning:
            warnings.append(warning)
    elif suffix == ".docx":
        text, docx_metadata = _read_docx(path)
        metadata.update(docx_metadata)
    elif suffix == ".pdf":
        text, pdf_metadata = _read_pdf(path)
        metadata.update(pdf_metadata)
    elif suffix in {".csv", ".tsv", ".xlsx"}:
        raise ValueError("tabular document; use inspect_table instead")
    else:
        raise ValueError(
            "unsupported document format; supported: txt, md, rst, log, json, "
            "yaml, html, docx, pdf"
        )

    content = text[:max_chars]
    if len(text) > max_chars:
        warnings.append(
            f"Document truncated: returned {len(content)} of {len(text)} characters."
        )
        content = (
            content.rstrip()
            + f"\n\n[truncated - total={len(text)} chars, returned={len(content)} chars]"
        )
    if warnings:
        content = "\n".join(f"Warning: {warning}" for warning in warnings) + "\n\n" + content
    relative = _relative_path(root, path)
    return NativeToolResult(
        content=content or "(empty document)",
        data={
            "path": relative,
            "chars": min(len(text), max_chars),
            "bytes": path.stat().st_size,
            "truncated": len(text) > max_chars,
            "warnings": tuple(warnings),
            **metadata,
        },
        refs=(relative,),
    )


def _read_docx(path: Path) -> tuple[str, Mapping[str, object]]:
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ValueError("invalid DOCX file: not a readable zip archive") from exc
    with archive:
        try:
            document_xml = _read_zip_member(archive, "word/document.xml")
        except KeyError as exc:
            raise ValueError("DOCX is missing word/document.xml") from exc
        core_xml = (
            _read_zip_member(archive, "docProps/core.xml")
            if "docProps/core.xml" in archive.namelist()
            else b""
        )
    root = ElementTree.fromstring(document_xml)
    paragraphs: list[str] = []
    for paragraph in root.iter():
        if _local_name(paragraph.tag) != "p":
            continue
        parts = [
            node.text or ""
            for node in paragraph.iter()
            if _local_name(node.tag) in {"t", "tab", "br"}
        ]
        text = "".join("\t" if part == "" else part for part in parts).strip()
        if text:
            paragraphs.append(text)
    metadata: dict[str, object] = {"paragraphs": len(paragraphs)}
    if core_xml:
        core = ElementTree.fromstring(core_xml)
        for node in core.iter():
            name = _local_name(node.tag)
            if name in {"title", "creator", "subject"} and node.text:
                metadata[name] = node.text.strip()
    return "\n\n".join(paragraphs), metadata


def _read_text_with_encoding_hint(path: Path) -> tuple[str, str, str]:
    raw = path.read_bytes()
    best_text = raw.decode("utf-8", errors="replace")
    best_encoding = "utf-8"
    best_score = _decoded_text_score(best_text)
    warning = ""
    for encoding in TEXT_ENCODINGS:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        score = _decoded_text_score(text)
        if score < best_score:
            best_text = text
            best_encoding = encoding
            best_score = score
        if score == 0:
            break
    if best_encoding not in {"utf-8", "utf-8-sig"}:
        warning = f"Detected non-UTF-8 text encoding: {best_encoding}."
    elif best_text.count(REPLACEMENT_CHAR):
        warning = (
            "Text may be decoded with the wrong encoding; replacement characters were found."
        )
    return best_text, best_encoding, warning


def _decoded_text_score(text: str) -> int:
    replacements = text.count(REPLACEMENT_CHAR)
    control_chars = sum(
        1 for character in text if ord(character) < 32 and character not in "\n\r\t"
    )
    private_use = sum(1 for character in text if 0xE000 <= ord(character) <= 0xF8FF)
    surrogate_like = sum(1 for character in text if ord(character) > 0xFFFF)
    return replacements * 100 + control_chars * 20 + private_use * 10 + surrogate_like * 4


def _read_pdf(path: Path) -> tuple[str, Mapping[str, object]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ValueError(
            "PDF extraction requires optional dependency pypdf; "
            "install it with python3 -m pip install pypdf"
        ) from exc
    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(page for page in pages if page), {"pages": len(reader.pages)}


def _inspect_table(root: Path, arguments: Mapping[str, object]) -> NativeToolResult:
    path = _workspace_path(root, _text(arguments, "path"))
    max_rows = _int(arguments, "max_rows", DEFAULT_MAX_ROWS, 1, MAX_ROWS)
    preview_rows = _int(
        arguments,
        "preview_rows",
        DEFAULT_PREVIEW_ROWS,
        1,
        MAX_PREVIEW_ROWS,
    )
    sheet_name = _optional_text(arguments, "sheet", "")
    if not path.is_file():
        raise ValueError(f"path is not a file: {_relative_path(root, path)}")
    if path.stat().st_size > MAX_DOCUMENT_BYTES:
        raise ValueError(f"table exceeds {MAX_DOCUMENT_BYTES} bytes")

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        columns, rows, total_rows = _read_delimited(path, max_rows, suffix)
        sheet = ""
        warnings: tuple[str, ...] = ()
    elif suffix == ".jsonl":
        columns, rows, total_rows, skipped_rows = _read_jsonl(path, max_rows)
        sheet = ""
        warnings = (
            (f"Skipped {skipped_rows} comment/blank JSONL line(s).",)
            if skipped_rows
            else ()
        )
    elif suffix == ".json":
        columns, rows, total_rows, warnings = _read_json_table(path, max_rows)
        sheet = ""
    elif suffix == ".xlsx":
        columns, rows, total_rows, sheet = _read_xlsx(path, max_rows, sheet_name)
        warnings = ()
    else:
        raise ValueError("unsupported table format; supported: csv, tsv, json, jsonl, xlsx")

    profiles = [_profile_column(column, rows) for column in columns]
    preview = rows[:preview_rows]
    relative = _relative_path(root, path)
    content = _format_table_report(
        relative,
        columns,
        preview,
        profiles,
        rows_read=len(rows),
        total_rows=total_rows,
        sheet=sheet,
        warnings=warnings,
    )
    return NativeToolResult(
        content=content,
        data={
            "path": relative,
            "sheet": sheet,
            "columns": tuple(columns),
            "column_profiles": tuple(profiles),
            "rows_read": len(rows),
            "total_rows": total_rows,
            "truncated": total_rows > len(rows),
            "warnings": warnings,
            "preview": tuple(preview),
        },
        refs=(relative,),
    )


def _read_delimited(
    path: Path,
    max_rows: int,
    suffix: str,
) -> tuple[list[str], list[dict[str, object]], int]:
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as file:
        sample = file.read(8192)
        file.seek(0)
        delimiter = "\t" if suffix == ".tsv" else _detect_delimiter(sample)
        reader = csv.DictReader(file, delimiter=delimiter)
        columns = _unique_columns(reader.fieldnames or ())
        rows: list[dict[str, object]] = []
        total = 0
        for raw_row in reader:
            total += 1
            if len(rows) < max_rows:
                rows.append({column: raw_row.get(column) for column in columns})
    return columns, rows, total


def _read_jsonl(
    path: Path,
    max_rows: int,
) -> tuple[list[str], list[dict[str, object]], int, int]:
    rows: list[dict[str, object]] = []
    columns: list[str] = []
    total = 0
    skipped = 0
    with path.open(encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            clean = line.strip()
            if not clean or clean.startswith(("#", "//")):
                skipped += 1
                continue
            try:
                value = json.loads(clean)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc.msg}") from exc
            if not isinstance(value, Mapping):
                value = {"value": value}
            total += 1
            _extend_columns(columns, value.keys())
            if len(rows) < max_rows:
                rows.append(dict(value))
    return columns, rows, total, skipped


def _read_json_table(
    path: Path,
    max_rows: int,
) -> tuple[list[str], list[dict[str, object]], int, tuple[str, ...]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    warnings: list[str] = []
    if isinstance(value, Mapping):
        candidates = {
            str(key): item
            for key, item in value.items()
            if isinstance(item, list)
        }
        if candidates:
            selected_key, selected_value, ambiguous = _select_json_table_array(candidates)
            value = selected_value
            if ambiguous:
                warnings.append(
                    "Multiple JSON array fields found; selected "
                    f"'{selected_key}' as the most table-like value."
                )
        else:
            value = [value]
    if not isinstance(value, list):
        raise ValueError("JSON table must be an array or contain an array value")
    rows: list[dict[str, object]] = []
    columns: list[str] = []
    for item in value[:max_rows]:
        row = dict(item) if isinstance(item, Mapping) else {"value": item}
        _extend_columns(columns, row.keys())
        rows.append(row)
    return columns, rows, len(value), tuple(warnings)


def _select_json_table_array(
    candidates: Mapping[str, list[object]],
) -> tuple[str, list[object], bool]:
    preferred = ("data", "rows", "records", "items", "results")
    by_lower = {key.lower(): key for key in candidates}
    for preferred_key in preferred:
        key = by_lower.get(preferred_key)
        if key is not None:
            return key, candidates[key], False

    scored = sorted(
        candidates.items(),
        key=lambda item: _json_array_table_score(item[1]),
        reverse=True,
    )
    selected_key, selected_value = scored[0]
    ambiguous = len(candidates) > 1
    return selected_key, selected_value, ambiguous


def _json_array_table_score(value: Sequence[object]) -> tuple[int, int, int, int]:
    mapping_items = [item for item in value if isinstance(item, Mapping)]
    field_count = len({str(key) for item in mapping_items for key in item.keys()})
    return (
        1 if mapping_items else 0,
        len(mapping_items),
        field_count,
        len(value),
    )


def _read_xlsx(
    path: Path,
    max_rows: int,
    requested_sheet: str,
) -> tuple[list[str], list[dict[str, object]], int, str]:
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ValueError("invalid XLSX file: not a readable zip archive") from exc
    with archive:
        names = set(archive.namelist())
        workbook = ElementTree.fromstring(_read_zip_member(archive, "xl/workbook.xml"))
        relationships = ElementTree.fromstring(
            _read_zip_member(archive, "xl/_rels/workbook.xml.rels")
        )
        relationship_targets = {
            node.attrib.get("Id", ""): node.attrib.get("Target", "")
            for node in relationships
            if _local_name(node.tag) == "Relationship"
        }
        sheets = []
        for node in workbook.iter():
            if _local_name(node.tag) != "sheet":
                continue
            name = node.attrib.get("name", "")
            relation_id = next(
                (
                    value
                    for key, value in node.attrib.items()
                    if _local_name(key) == "id"
                ),
                "",
            )
            target = relationship_targets.get(relation_id, "")
            if target:
                target = target.lstrip("/")
                if not target.startswith("xl/"):
                    target = f"xl/{target}"
                sheets.append((name, target))
        if not sheets:
            raise ValueError("XLSX contains no worksheets")
        selected = next(
            (item for item in sheets if item[0] == requested_sheet),
            None,
        )
        if requested_sheet and selected is None:
            raise ValueError(
                f"sheet not found: {requested_sheet}; available: "
                + ", ".join(name for name, _ in sheets)
            )
        sheet_name, sheet_path = selected or sheets[0]
        if sheet_path not in names:
            raise ValueError(f"XLSX worksheet is missing: {sheet_path}")
        shared_strings = _xlsx_shared_strings(archive)
        sheet_root = ElementTree.fromstring(_read_zip_member(archive, sheet_path))

    raw_rows: list[list[object]] = []
    total_rows = 0
    for row_node in sheet_root.iter():
        if _local_name(row_node.tag) != "row":
            continue
        values: dict[int, object] = {}
        next_index = 0
        for cell in row_node:
            if _local_name(cell.tag) != "c":
                continue
            reference = cell.attrib.get("r", "")
            index = _column_index(reference) if reference else next_index
            values[index] = _xlsx_cell_value(cell, shared_strings)
            next_index = index + 1
        total_rows += 1
        if len(raw_rows) < max_rows + 1:
            width = max(values, default=-1) + 1
            raw_rows.append([values.get(index) for index in range(width)])
    if not raw_rows:
        return [], [], 0, sheet_name
    columns = _xlsx_headers(raw_rows[0])
    rows = [
        {column: row[index] if index < len(row) else None for index, column in enumerate(columns)}
        for row in raw_rows[1 : max_rows + 1]
    ]
    return columns, rows, max(0, total_rows - 1), sheet_name


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ElementTree.fromstring(_read_zip_member(archive, "xl/sharedStrings.xml"))
    strings: list[str] = []
    for item in root:
        if _local_name(item.tag) != "si":
            continue
        strings.append(
            "".join(
                node.text or ""
                for node in item.iter()
                if _local_name(node.tag) == "t"
            )
        )
    return strings


def _xlsx_cell_value(cell, shared_strings: Sequence[str]) -> object:
    cell_type = cell.attrib.get("t", "")
    value_node = next(
        (node for node in cell.iter() if _local_name(node.tag) == "v"),
        None,
    )
    inline = "".join(
        node.text or ""
        for node in cell.iter()
        if _local_name(node.tag) == "t"
    )
    raw = value_node.text if value_node is not None else inline
    if raw is None:
        return None
    if cell_type == "s":
        try:
            return shared_strings[int(raw)]
        except (ValueError, IndexError):
            return raw
    if cell_type in {"str", "inlineStr"}:
        return raw
    if cell_type == "b":
        return raw == "1"
    try:
        number = float(raw)
        return int(number) if number.is_integer() else number
    except ValueError:
        return raw


def _profile_column(column: str, rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    values = [row.get(column) for row in rows]
    present = [value for value in values if not _is_missing(value)]
    numeric = [_number(value) for value in present]
    numeric_values = [value for value in numeric if value is not None]
    if present and len(numeric_values) == len(present):
        inferred_type = "number"
    elif present and all(isinstance(value, bool) for value in present):
        inferred_type = "boolean"
    elif present and all(_looks_like_datetime(value) for value in present):
        inferred_type = "datetime"
    elif not present:
        inferred_type = "empty"
    else:
        inferred_type = "text"
    profile: dict[str, object] = {
        "name": column,
        "type": inferred_type,
        "non_null": len(present),
        "missing": len(values) - len(present),
        "unique": len({_stable_value(value) for value in present}),
    }
    if numeric_values:
        profile.update(
            {
                "min": min(numeric_values),
                "max": max(numeric_values),
                "mean": statistics.fmean(numeric_values),
                "median": statistics.median(numeric_values),
            }
        )
    else:
        profile["top_values"] = tuple(
            {"value": value, "count": count}
            for value, count in Counter(
                str(item)[:120] for item in present
            ).most_common(5)
        )
    return profile


def _format_table_report(
    path: str,
    columns: Sequence[str],
    preview: Sequence[Mapping[str, object]],
    profiles: Sequence[Mapping[str, object]],
    *,
    rows_read: int,
    total_rows: int,
    sheet: str,
    warnings: Sequence[str] = (),
) -> str:
    lines = [
        f"Table: {path}",
        f"Rows: {total_rows} total, {rows_read} profiled",
        f"Columns: {len(columns)}",
    ]
    if sheet:
        lines.append(f"Sheet: {sheet}")
    for warning in warnings:
        lines.append(f"Warning: {warning}")
    lines.extend(("", "Column profiles:"))
    for profile in profiles:
        detail = (
            f"- {profile['name']}: {profile['type']}; "
            f"missing={profile['missing']}; unique={profile['unique']}"
        )
        if "mean" in profile:
            detail += (
                f"; min={_compact_number(profile['min'])}; "
                f"max={_compact_number(profile['max'])}; "
                f"mean={_compact_number(profile['mean'])}"
            )
        lines.append(detail)
    if preview:
        lines.extend(("", "Preview:", json.dumps(preview, ensure_ascii=False, indent=2)))
    return "\n".join(lines)


class _DocumentHtmlParser(HTMLParser):
    SKIP = frozenset({"script", "style", "svg", "noscript", "template"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: list[str] = []
        self._title_parts: list[str] = []
        self._in_title = False

    @property
    def title(self) -> str:
        return " ".join(" ".join(self._title_parts).split())

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self.SKIP:
            self._skip_depth += 1
        elif not self._skip_depth and tag == "title":
            self._in_title = True
        elif not self._skip_depth and (
            tag in {"p", "div", "section", "article", "main", "li", "br"}
            or re.fullmatch(r"h[1-6]", tag)
        ):
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag == "title":
            self._in_title = False
        elif not self._skip_depth and (
            tag in {"p", "div", "section", "article", "main", "li"}
            or re.fullmatch(r"h[1-6]", tag)
        ):
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        clean = " ".join(data.split())
        if not clean:
            return
        if self._in_title:
            self._title_parts.append(clean)
        else:
            self._parts.append(clean + " ")

    def render(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self._parts).splitlines())
        return "\n".join(line for line in lines if line).strip()


def _detect_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _unique_columns(columns: Iterable[str | None]) -> list[str]:
    result: list[str] = []
    counts: Counter[str] = Counter()
    for index, raw in enumerate(columns, start=1):
        base = (raw or f"column_{index}").strip() or f"column_{index}"
        counts[base] += 1
        result.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return result


def _extend_columns(columns: list[str], values: Iterable[object]) -> None:
    for value in values:
        name = str(value)
        if name not in columns:
            columns.append(name)


def _xlsx_headers(values: Sequence[object]) -> list[str]:
    return _unique_columns(str(value) if value is not None else None for value in values)


def _column_index(reference: str) -> int:
    letters = "".join(character for character in reference if character.isalpha())
    result = 0
    for character in letters.upper():
        result = result * 26 + (ord(character) - ord("A") + 1)
    return max(0, result - 1)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _read_zip_member(archive: zipfile.ZipFile, name: str) -> bytes:
    info = archive.getinfo(name)
    if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
        raise ValueError(
            f"archive member exceeds {MAX_ARCHIVE_MEMBER_BYTES} bytes: {name}"
        )
    return archive.read(info)


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        clean = value.strip().replace(",", "")
        if clean.endswith("%"):
            clean = clean[:-1]
        try:
            number = float(clean)
            return number if math.isfinite(number) else None
        except ValueError:
            return None
    return None


def _looks_like_datetime(value: object) -> bool:
    if not isinstance(value, str):
        return False
    clean = value.strip().replace("Z", "+00:00")
    try:
        datetime.fromisoformat(clean)
        return True
    except ValueError:
        return False


def _stable_value(value: object) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def _compact_number(value: object) -> str:
    if not isinstance(value, (int, float)):
        return str(value)
    return f"{value:.4g}"


def _text(arguments: Mapping[str, object], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be non-empty text")
    return value.strip()


def _optional_text(arguments: Mapping[str, object], key: str, default: str) -> str:
    value = arguments.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be text")
    return value.strip()


def _int(
    arguments: Mapping[str, object],
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = arguments.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


def _read_document_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative document path."},
            "max_chars": {"type": "integer", "minimum": 1, "maximum": MAX_CHARS},
        },
        "required": ["path"],
        "additionalProperties": False,
    }


def _inspect_table_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative table path."},
            "sheet": {"type": "string", "description": "Optional XLSX sheet name."},
            "max_rows": {"type": "integer", "minimum": 1, "maximum": MAX_ROWS},
            "preview_rows": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_PREVIEW_ROWS,
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }
