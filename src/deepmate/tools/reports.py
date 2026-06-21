"""Deterministic self-contained HTML report rendering."""

from __future__ import annotations

import html
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree

from deepmate.runtime.hooks import HookDirective, HookEvent, HookRuntimeContext
from deepmate.tools.filesystem import (
    MAX_WRITE_CHARS,
    WorkspaceWriteCheckpoint,
    _atomic_write_text,
    _emit_write_hook,
    _hook_refs,
    _relative_path,
    _workspace_path,
)
from deepmate.tools.registry import NativeTool, NativeToolResult
from deepmate.tools.svg_security import UNSAFE_SVG_PATTERN

THEMES = ("prussian", "forest", "graphite")
LAYOUTS = ("brief", "report", "presentation")
PLACEHOLDER_PATTERN = re.compile(r"\b(?:TODO|TBD|FIXME)\b|<what you need>", re.IGNORECASE)
MAX_INLINE_SVG_BYTES = 1_000_000
IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\((\S+?)(?:\s+&quot;([^&]*)&quot;)?\)")


def workspace_report_tools(
    workspace_root: str | Path,
    *,
    write_checkpoint: WorkspaceWriteCheckpoint | None = None,
    hook_context: HookRuntimeContext | None = None,
) -> tuple[NativeTool, ...]:
    """Return the deterministic Markdown-to-HTML report tool."""
    root = Path(workspace_root).resolve()
    if not root.is_dir():
        raise ValueError(f"workspace root must be a directory: {root}")
    return (
        NativeTool(
            name="render_html_report",
            description="Render workspace Markdown into a polished self-contained HTML report.",
            input_schema=_render_html_report_schema(),
            handler=lambda arguments: _render_html_report(
                root,
                arguments,
                write_checkpoint=write_checkpoint,
                hook_context=hook_context,
            ),
            read_only=False,
        ),
    )


def _render_html_report(
    root: Path,
    arguments: Mapping[str, object],
    *,
    write_checkpoint: WorkspaceWriteCheckpoint | None,
    hook_context: HookRuntimeContext | None,
) -> NativeToolResult:
    source_path = _workspace_path(root, _text(arguments, "source_path"))
    output_path = _workspace_path(root, _text(arguments, "output_path"))
    theme = _choice(arguments, "theme", "prussian", THEMES)
    layout = _choice(arguments, "layout", "report", LAYOUTS)
    overwrite = _bool(arguments, "overwrite", False)
    title_override = _optional_text(arguments, "title", "")
    if not source_path.is_file():
        raise ValueError(f"source_path is not a file: {_relative_path(root, source_path)}")
    if source_path.suffix.lower() not in {".md", ".markdown"}:
        raise ValueError("source_path must be a Markdown file")
    if output_path.suffix.lower() not in {".html", ".htm"}:
        raise ValueError("output_path must end in .html or .htm")
    if output_path.exists() and not overwrite:
        raise ValueError("output_path already exists; pass overwrite=true to replace it")
    if output_path.exists() and not output_path.is_file():
        raise ValueError("output_path exists but is not a file")
    if not output_path.parent.is_dir():
        raise ValueError(f"parent directory does not exist: {output_path.parent}")

    markdown = source_path.read_text(encoding="utf-8")
    title = title_override or _document_title(markdown) or source_path.stem
    image_refs: list[Mapping[str, object]] = []
    body, headings, links = _render_markdown(
        markdown,
        workspace_root=root,
        asset_root=source_path.parent,
        image_refs=image_refs,
    )
    rendered = _html_document(title, body, theme=theme, layout=layout)
    if len(rendered) > MAX_WRITE_CHARS:
        raise ValueError(f"rendered report must be at most {MAX_WRITE_CHARS} characters")
    checks = _quality_checks(markdown, headings, links, image_refs, rendered)

    before_outcome = _emit_write_hook(
        hook_context,
        HookEvent.WRITE_BEFORE,
        root=root,
        path=output_path,
        operation="render_html_report",
        status="before",
        payload={
            "source_path": _relative_path(root, source_path),
            "content_size": len(rendered),
            "path_kind": "file",
            "theme": theme,
            "layout": layout,
        },
    )
    if before_outcome.directive != HookDirective.CONTINUE:
        raise ValueError(
            before_outcome.reason
            or f"Workspace write stopped by hook: {before_outcome.directive.value}"
        )
    if write_checkpoint is not None:
        write_checkpoint("render_html_report", output_path, rendered)
    _atomic_write_text(output_path, rendered)
    after_outcome = _emit_write_hook(
        hook_context,
        HookEvent.WRITE_AFTER,
        root=root,
        path=output_path,
        operation="render_html_report",
        status="completed",
        payload={
            "source_path": _relative_path(root, source_path),
            "content_size": len(rendered),
            "path_kind": "file",
            "theme": theme,
            "layout": layout,
        },
    )

    relative_source = _relative_path(root, source_path)
    relative_output = _relative_path(root, output_path)
    failed_checks = tuple(check for check in checks if not check["passed"])
    content = "\n".join(
        (
            f"Rendered {relative_output} from {relative_source}.",
            f"Theme: {theme}; layout: {layout}; title: {title}",
            (
                "Static QA passed."
                if not failed_checks
                else "Static QA warnings: "
                + "; ".join(str(check["message"]) for check in failed_checks)
            ),
        )
    )
    return NativeToolResult(
        content=content,
        data={
            "source_path": relative_source,
            "path": relative_output,
            "title": title,
            "theme": theme,
            "layout": layout,
            "chars_written": len(rendered),
            "checks": checks,
            "warnings": len(failed_checks),
        },
        refs=(relative_output, relative_source, *_hook_refs(after_outcome)),
    )


def _render_markdown(
    markdown: str,
    *,
    workspace_root: Path | None = None,
    asset_root: Path | None = None,
    image_refs: list[Mapping[str, object]] | None = None,
) -> tuple[str, tuple[tuple[int, str], ...], tuple[str, ...]]:
    lines = markdown.splitlines()
    output: list[str] = []
    headings: list[tuple[int, str]] = []
    links: list[str] = []
    paragraph: list[str] = []
    list_items: list[str] = []
    ordered_list = False
    table_rows: list[list[str]] = []
    in_code = False
    code_language = ""
    code_lines: list[str] = []

    def inline(text: str) -> str:
        return _inline_markdown(
            text,
            links,
            workspace_root=workspace_root,
            asset_root=asset_root,
            image_refs=image_refs,
        )

    def flush_paragraph() -> None:
        if paragraph:
            output.append(f"<p>{inline(' '.join(paragraph))}</p>")
            paragraph.clear()

    def flush_list() -> None:
        nonlocal ordered_list
        if list_items:
            tag = "ol" if ordered_list else "ul"
            output.append(f"<{tag}>" + "".join(list_items) + f"</{tag}>")
            list_items.clear()
        ordered_list = False

    def flush_table() -> None:
        if not table_rows:
            return
        header = table_rows[0]
        rows = table_rows[1:]
        output.append(
            "<div class=\"table-wrap\"><table><thead><tr>"
            + "".join(f"<th>{inline(cell)}</th>" for cell in header)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>"
                + "".join(f"<td>{inline(cell)}</td>" for cell in row)
                + "</tr>"
                for row in rows
            )
            + "</tbody></table></div>"
        )
        table_rows.clear()

    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_paragraph()
            flush_list()
            flush_table()
            if in_code:
                language_class = (
                    f' class="language-{html.escape(code_language)}"'
                    if code_language
                    else ""
                )
                output.append(
                    f"<pre><code{language_class}>"
                    + html.escape("\n".join(code_lines))
                    + "</code></pre>"
                )
                code_lines.clear()
                in_code = False
                code_language = ""
            else:
                in_code = True
                code_language = stripped[3:].strip()
            index += 1
            continue
        if in_code:
            code_lines.append(line)
            index += 1
            continue
        if not stripped:
            flush_paragraph()
            flush_list()
            flush_table()
            index += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            flush_list()
            flush_table()
            level = len(heading.group(1))
            text = heading.group(2).strip()
            headings.append((level, text))
            output.append(
                f'<h{level} id="{_slug(text)}">{inline(text)}</h{level}>'
            )
            index += 1
            continue
        if stripped.startswith(">"):
            flush_paragraph()
            flush_list()
            flush_table()
            output.append(
                f"<blockquote>{inline(stripped[1:].strip())}</blockquote>"
            )
            index += 1
            continue
        list_match = re.match(r"^([-*+]|\d+\.)\s+(.+)$", stripped)
        if list_match:
            flush_paragraph()
            flush_table()
            item_ordered = list_match.group(1)[0].isdigit()
            if list_items and ordered_list != item_ordered:
                flush_list()
            ordered_list = item_ordered
            list_items.append(f"<li>{inline(list_match.group(2))}</li>")
            index += 1
            continue
        if "|" in stripped and index + 1 < len(lines) and _is_table_divider(lines[index + 1]):
            flush_paragraph()
            flush_list()
            table_rows.append(_table_cells(stripped))
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                table_rows.append(_table_cells(lines[index]))
                index += 1
            flush_table()
            continue
        if stripped in {"---", "***", "___"}:
            flush_paragraph()
            flush_list()
            flush_table()
            output.append("<hr>")
            index += 1
            continue
        paragraph.append(stripped)
        index += 1

    flush_paragraph()
    flush_list()
    flush_table()
    if in_code:
        output.append("<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>")
    return "\n".join(output), tuple(headings), tuple(links)


def _inline_markdown(
    text: str,
    links: list[str],
    *,
    workspace_root: Path | None,
    asset_root: Path | None,
    image_refs: list[Mapping[str, object]] | None,
) -> str:
    escaped = html.escape(text)
    image_fragments: list[str] = []
    code_fragments: list[str] = []

    def replace_image(match: re.Match[str]) -> str:
        alt = match.group(1)
        src = html.unescape(match.group(2)).strip()
        caption = html.unescape(match.group(3) or "").strip()
        links.append(src)
        svg_result = _inline_svg(src, workspace_root=workspace_root, asset_root=asset_root)
        if image_refs is not None:
            image_refs.append(
                {
                    "src": src,
                    "alt": alt,
                    "caption": caption,
                    "inline_status": svg_result["status"],
                    "inline_reason": svg_result["reason"],
                }
            )
        if svg_result["content"]:
            caption_html = (
                f'<figcaption>{html.escape(caption)}</figcaption>' if caption else ""
            )
            fragment = (
                f'<figure class="diagram" aria-label="{alt}">'
                f"{svg_result['content']}"
                f"{caption_html}"
                "</figure>"
            )
            return _stash_image_fragment(image_fragments, fragment)
        if svg_result["status"] != "skipped":
            caption_html = (
                f'<figcaption>{html.escape(caption)}</figcaption>' if caption else ""
            )
            reason = html.escape(str(svg_result["reason"]))
            note = f"SVG image not embedded: {reason}."
            fragment = (
                f'<figure class="image-block">'
                f'<p class="asset-note">{note}</p>'
                f"{caption_html}"
                "</figure>"
            )
            return _stash_image_fragment(image_fragments, fragment)
        safe_src = html.escape(src, quote=True) if _safe_link(src) else ""
        if not safe_src:
            return html.escape(alt)
        caption_html = f'<figcaption>{html.escape(caption)}</figcaption>' if caption else ""
        fragment = (
            f'<figure class="image-block">'
            f'<img src="{safe_src}" alt="{alt}" loading="lazy">'
            f"{caption_html}"
            "</figure>"
        )
        return _stash_image_fragment(image_fragments, fragment)

    def replace_link(match: re.Match[str]) -> str:
        label = match.group(1)
        href = html.unescape(match.group(2)).strip()
        links.append(href)
        safe_href = html.escape(href, quote=True) if _safe_link(href) else "#"
        return f'<a href="{safe_href}">{label}</a>'

    def replace_code(match: re.Match[str]) -> str:
        return _stash_code_fragment(code_fragments, f"<code>{match.group(1)}</code>")

    escaped = re.sub(r"`([^`]+)`", replace_code, escaped)
    escaped = IMAGE_PATTERN.sub(replace_image, escaped)
    escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replace_link, escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", escaped)
    for index, fragment in enumerate(image_fragments):
        escaped = escaped.replace(_image_fragment_token(index), fragment)
    for index, fragment in enumerate(code_fragments):
        escaped = escaped.replace(_code_fragment_token(index), fragment)
    return escaped


def _stash_image_fragment(fragments: list[str], fragment: str) -> str:
    token = _image_fragment_token(len(fragments))
    fragments.append(fragment)
    return token


def _image_fragment_token(index: int) -> str:
    return f"@@DEEPMATE_IMAGE_{index}@@"


def _stash_code_fragment(fragments: list[str], fragment: str) -> str:
    token = _code_fragment_token(len(fragments))
    fragments.append(fragment)
    return token


def _code_fragment_token(index: int) -> str:
    return f"@@DEEPMATE_INLINE_CODE_{index}@@"


def _inline_svg(
    src: str,
    *,
    workspace_root: Path | None,
    asset_root: Path | None,
) -> Mapping[str, str]:
    if workspace_root is None or asset_root is None:
        return {"status": "skipped", "reason": "no workspace context", "content": ""}
    parsed = urlparse(src)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        return {"status": "skipped", "reason": "external or complex image ref", "content": ""}
    if Path(parsed.path).suffix.lower() != ".svg":
        return {"status": "skipped", "reason": "not an SVG image", "content": ""}
    try:
        path = (asset_root / parsed.path).resolve()
        path.relative_to(workspace_root)
    except ValueError:
        return {"status": "blocked", "reason": "image path escapes workspace", "content": ""}
    if not path.is_file() or path.stat().st_size > MAX_INLINE_SVG_BYTES:
        return {"status": "missing", "reason": "SVG missing or too large", "content": ""}
    text = path.read_text(encoding="utf-8", errors="replace")
    if UNSAFE_SVG_PATTERN.search(text):
        return {"status": "blocked", "reason": "SVG contains unsafe markup", "content": ""}
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError:
        return {"status": "invalid", "reason": "SVG XML parse failed", "content": ""}
    if root.tag.rsplit("}", 1)[-1] != "svg":
        return {"status": "invalid", "reason": "image root is not svg", "content": ""}
    return {"status": "inlined", "reason": "SVG inlined", "content": text}


def _html_document(title: str, body: str, *, theme: str, layout: str) -> str:
    palette = {
        "prussian": {
            "ink": "#17202a",
            "muted": "#5f6872",
            "paper": "#f7f6f2",
            "panel": "#ffffff",
            "primary": "#003153",
            "accent": "#b78032",
            "line": "#d9d8d2",
            "positive": "#2f5d50",
            "negative": "#a8483e",
        },
        "forest": {
            "ink": "#18231f",
            "muted": "#5d6964",
            "paper": "#f5f6f1",
            "panel": "#ffffff",
            "primary": "#1f4a3c",
            "accent": "#a87934",
            "line": "#d8ddd7",
            "positive": "#2e664d",
            "negative": "#a14c43",
        },
        "graphite": {
            "ink": "#202326",
            "muted": "#656b70",
            "paper": "#f4f5f5",
            "panel": "#ffffff",
            "primary": "#343a40",
            "accent": "#8b6b39",
            "line": "#d7dadd",
            "positive": "#3f665b",
            "negative": "#9d4d48",
        },
    }[theme]
    page_width = "1120px" if layout == "presentation" else "920px"
    section_gap = "48px" if layout == "presentation" else "32px"
    css_vars = "\n".join(f"      --{key}: {value};" for key, value in palette.items())
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
{css_vars}
      --page-width: {page_width};
      --section-gap: {section_gap};
    }}
    * {{ box-sizing: border-box; }}
    html {{ color-scheme: light; background: var(--paper); }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--paper);
      font: 16px/1.65 Inter, "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
      letter-spacing: 0;
    }}
    header {{
      padding: 56px max(24px, calc((100vw - var(--page-width)) / 2));
      color: #fff;
      background: var(--primary);
      border-bottom: 4px solid var(--accent);
    }}
    header p {{ max-width: 720px; margin: 10px 0 0; color: rgba(255,255,255,.78); }}
    main {{ width: min(var(--page-width), calc(100% - 40px)); margin: 0 auto; padding: 48px 0 72px; }}
    h1, h2, h3, h4, h5, h6 {{ margin: var(--section-gap) 0 12px; line-height: 1.25; }}
    h1 {{ margin: 0; color: #fff; font-size: 40px; font-weight: 650; }}
    main > h1:first-child {{ display: none; }}
    h2 {{ padding-bottom: 10px; color: var(--primary); font-size: 26px; border-bottom: 1px solid var(--line); }}
    h3 {{ color: var(--primary); font-size: 20px; }}
    h4, h5, h6 {{ font-size: 16px; }}
    p, li {{ max-width: 76ch; }}
    a {{ color: var(--primary); text-underline-offset: 3px; }}
    img {{ max-width: 100%; height: auto; display: block; border: 1px solid var(--line); border-radius: 6px; background: var(--panel); }}
    figure {{ margin: 28px 0; }}
    figcaption {{ margin-top: 8px; color: var(--muted); font-size: 13px; text-align: center; }}
    .diagram, .image-block {{ padding: 0; }}
    .diagram svg {{ width: 100%; height: auto; display: block; border: 1px solid var(--line); border-radius: 6px; background: var(--panel); }}
    .asset-note {{ max-width: none; margin: 0; padding: 14px 16px; color: var(--muted); background: var(--panel); border: 1px solid var(--line); border-radius: 6px; font-size: 13px; }}
    blockquote {{
      margin: 24px 0;
      padding: 16px 20px;
      color: var(--muted);
      background: var(--panel);
      border-left: 3px solid var(--accent);
    }}
    code {{ padding: 2px 5px; background: #e9ebeb; border-radius: 4px; font: 14px/1.5 "SFMono-Regular", Consolas, monospace; }}
    pre {{ overflow: auto; padding: 20px; color: #f4f4f2; background: #20272b; border-radius: 6px; }}
    pre code {{ padding: 0; color: inherit; background: transparent; }}
    .table-wrap {{ max-width: 100%; overflow-x: auto; margin: 24px 0; background: var(--panel); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 12px 14px; text-align: left; vertical-align: top; border-bottom: 1px solid var(--line); }}
    th {{ color: #fff; background: var(--primary); font-weight: 600; }}
    tbody tr:nth-child(even) {{ background: color-mix(in srgb, var(--paper) 72%, #fff); }}
    hr {{ margin: 40px 0; border: 0; border-top: 1px solid var(--line); }}
    footer {{ padding: 18px 24px; color: rgba(255,255,255,.72); text-align: center; background: var(--primary); font-size: 12px; }}
    @media (max-width: 640px) {{
      header {{ padding: 36px 20px; }}
      main {{ width: min(100% - 28px, var(--page-width)); padding-top: 28px; }}
      h1 {{ font-size: 30px; }}
      h2 {{ font-size: 22px; }}
    }}
    @media print {{
      body {{ background: #fff; font-size: 11pt; }}
      header {{ padding: 24mm 18mm 12mm; print-color-adjust: exact; -webkit-print-color-adjust: exact; }}
      main {{ width: auto; padding: 12mm 18mm; }}
      h2, h3, table, blockquote, pre, figure {{ break-inside: avoid; }}
      a {{ color: inherit; text-decoration: none; }}
      footer {{ display: none; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p>Prepared with Deepmate</p>
  </header>
  <main>
{body}
  </main>
  <footer>Generated by Deepmate</footer>
</body>
</html>
"""


def _quality_checks(
    markdown: str,
    headings: Sequence[tuple[int, str]],
    links: Sequence[str],
    image_refs: Sequence[Mapping[str, object]],
    rendered: str,
) -> tuple[Mapping[str, object], ...]:
    initial_heading_gap = bool(headings and headings[0][0] > 1)
    heading_jumps = [
        (previous[0], current[0])
        for previous, current in zip(headings, headings[1:])
        if current[0] > previous[0] + 1
    ]
    unsafe_links = [link for link in links if not _safe_link(link)]
    failed_images = [
        ref
        for ref in image_refs
        if str(ref.get("src", "")).lower().endswith(".svg")
        and ref.get("inline_status") not in {"inlined", "skipped"}
    ]
    checks = (
        {
            "name": "title",
            "passed": bool(_document_title(markdown)),
            "message": "Markdown should contain one top-level title.",
        },
        {
            "name": "heading_order",
            "passed": not heading_jumps and not initial_heading_gap,
            "message": "Heading levels should start at h1 and not skip levels.",
        },
        {
            "name": "links",
            "passed": not unsafe_links,
            "message": "Links must use http, https, mailto, or relative paths.",
        },
        {
            "name": "svg_images",
            "passed": not failed_images,
            "message": (
                "Workspace SVG images should exist and be safe to inline."
                if failed_images
                else "Workspace SVG images are inline-safe."
            ),
        },
        {
            "name": "placeholders",
            "passed": PLACEHOLDER_PATTERN.search(markdown) is None,
            "message": "Unresolved TODO/TBD/FIXME placeholder found.",
        },
        {
            "name": "responsive_print",
            "passed": (
                'name="viewport"' in rendered
                and "@media print" in rendered
                and "overflow-x: auto" in rendered
            ),
            "message": "Responsive, print, and table overflow rules are required.",
        },
    )
    return tuple(checks)


def _document_title(markdown: str) -> str:
    for line in markdown.splitlines():
        match = re.match(r"^#\s+(.+)$", line.strip())
        if match:
            return match.group(1).strip()
    return ""


def _safe_link(value: str) -> bool:
    if not value.strip():
        return False
    parsed = urlparse(value.strip())
    return not parsed.scheme or parsed.scheme in {"http", "https", "mailto"}


def _is_table_divider(line: str) -> bool:
    cells = _table_cells(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _slug(text: str) -> str:
    clean = re.sub(r"[^\w\u4e00-\u9fff]+", "-", text.lower(), flags=re.UNICODE)
    return clean.strip("-") or "section"


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


def _choice(
    arguments: Mapping[str, object],
    key: str,
    default: str,
    choices: Sequence[str],
) -> str:
    value = _optional_text(arguments, key, default).lower()
    if value not in choices:
        raise ValueError(f"{key} must be one of: {', '.join(choices)}")
    return value


def _bool(arguments: Mapping[str, object], key: str, default: bool) -> bool:
    value = arguments.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _render_html_report_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "source_path": {
                "type": "string",
                "description": "Workspace-relative Markdown source path.",
            },
            "output_path": {
                "type": "string",
                "description": "Workspace-relative .html output path.",
            },
            "title": {"type": "string"},
            "theme": {"type": "string", "enum": list(THEMES)},
            "layout": {"type": "string", "enum": list(LAYOUTS)},
            "overwrite": {"type": "boolean"},
        },
        "required": ["source_path", "output_path"],
        "additionalProperties": False,
    }
