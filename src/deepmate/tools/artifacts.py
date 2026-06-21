"""Read-only review tools for generated work artifacts."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from xml.etree import ElementTree

from deepmate.tools.filesystem import _relative_path, _workspace_path
from deepmate.tools.registry import NativeTool, NativeToolResult
from deepmate.tools.svg_security import UNSAFE_SVG_PATTERN

REVIEW_ARTIFACT_TOOL_NAME = "review_artifact"

ARTIFACT_TYPES = ("auto", "markdown", "html", "svg")
MAX_REVIEW_CHARS = 1_000_000
DEFAULT_MAX_PARAGRAPH_CHARS = 900
MAX_PARAGRAPH_CHARS = 4_000
PLACEHOLDER_PATTERN = re.compile(r"\b(?:TODO|TBD|FIXME)\b|<what you need>", re.IGNORECASE)
MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\((\S+?)(?:\s+\"([^\"]*)\")?\)")
def workspace_artifact_tools(workspace_root: str | Path) -> tuple[NativeTool, ...]:
    """Return read-only artifact review tools for one workspace."""
    root = Path(workspace_root).resolve()
    if not root.is_dir():
        raise ValueError(f"workspace root must be a directory: {root}")
    return (
        NativeTool(
            name=REVIEW_ARTIFACT_TOOL_NAME,
            description=(
                "Review a generated Markdown, HTML, or SVG artifact and return "
                "actionable findings for one auto-fix pass."
            ),
            input_schema=_review_artifact_schema(),
            handler=lambda arguments: _review_artifact(root, arguments),
        ),
    )


def _review_artifact(root: Path, arguments: Mapping[str, object]) -> NativeToolResult:
    path = _workspace_path(root, _text(arguments, "path"))
    if not path.is_file():
        raise ValueError(f"path is not a file: {_relative_path(root, path)}")
    artifact_type = _artifact_type(path, _choice(arguments, "artifact_type", "auto", ARTIFACT_TYPES))
    require_sources = _bool(arguments, "require_sources", False)
    require_summary = _bool(arguments, "require_summary", False)
    require_acceptance_criteria = _bool(arguments, "require_acceptance_criteria", False)
    require_diagram_captions = _bool(arguments, "require_diagram_captions", False)
    max_paragraph_chars = _int(
        arguments,
        "max_paragraph_chars",
        DEFAULT_MAX_PARAGRAPH_CHARS,
        200,
        MAX_PARAGRAPH_CHARS,
    )
    text, truncated = _read_review_text(path)
    findings: list[dict[str, object]] = []
    if truncated:
        findings.append(
            _finding(
                "warning",
                "review_truncated",
                f"Artifact is larger than {MAX_REVIEW_CHARS} characters; reviewed the first slice only.",
                "Split or shorten the artifact before final delivery review.",
            )
        )

    if artifact_type == "markdown":
        findings.extend(
            _review_markdown(
                text,
                require_sources=require_sources,
                require_summary=require_summary,
                require_acceptance_criteria=require_acceptance_criteria,
                require_diagram_captions=require_diagram_captions,
                max_paragraph_chars=max_paragraph_chars,
            )
        )
    elif artifact_type == "html":
        findings.extend(
            _review_html(
                text,
                require_sources=require_sources,
                require_summary=require_summary,
                require_diagram_captions=require_diagram_captions,
            )
        )
    elif artifact_type == "svg":
        findings.extend(_review_svg(text))
    else:
        raise ValueError(f"unsupported artifact_type: {artifact_type}")

    counts = Counter(str(finding["severity"]) for finding in findings)
    passed = not any(finding["severity"] in {"error", "warning"} for finding in findings)
    relative = _relative_path(root, path)
    if passed:
        content = f"Artifact review passed for {relative}."
    else:
        content = "\n".join(
            (
                f"Artifact review found {len(findings)} issue(s) for {relative}.",
                *(
                    f"- {finding['severity']}: {finding['code']}: {finding['message']}"
                    for finding in findings[:12]
                ),
            )
        )
        if len(findings) > 12:
            content += f"\n- info: truncated_findings: {len(findings) - 12} more finding(s)."

    return NativeToolResult(
        content=content,
        data={
            "path": relative,
            "artifact_type": artifact_type,
            "passed": passed,
            "findings": findings,
            "finding_counts": dict(counts),
            "truncated": truncated,
        },
        refs=(relative,),
    )


def _review_markdown(
    text: str,
    *,
    require_sources: bool,
    require_summary: bool,
    require_acceptance_criteria: bool,
    require_diagram_captions: bool,
    max_paragraph_chars: int,
) -> tuple[dict[str, object], ...]:
    findings: list[dict[str, object]] = []
    headings = _markdown_headings(text)
    top_titles = [heading for heading in headings if heading[0] == 1]
    if not top_titles:
        findings.append(
            _finding("warning", "missing_title", "Markdown has no top-level title.", "Add one # title.")
        )
    elif len(top_titles) > 1:
        findings.append(
            _finding(
                "warning",
                "multiple_titles",
                "Markdown has multiple top-level titles.",
                "Keep one # title and demote later top-level sections.",
            )
        )
    if headings and headings[0][0] > 1:
        findings.append(
            _finding(
                "warning",
                "heading_starts_too_deep",
                f"Markdown starts at h{headings[0][0]} before '{headings[0][1]}'.",
                "Start with one # title, then use ## for main sections.",
            )
        )
    findings.extend(_heading_jump_findings(headings))
    if PLACEHOLDER_PATTERN.search(text):
        findings.append(
            _finding(
                "error",
                "placeholder",
                "Artifact contains TODO/TBD/FIXME or unresolved placeholder text.",
                "Replace placeholders before final delivery.",
            )
        )
    findings.extend(_empty_markdown_section_findings(text))
    findings.extend(_long_paragraph_findings(text, max_paragraph_chars))
    if require_sources and not _markdown_has_sources(text):
        findings.append(
            _finding(
                "warning",
                "missing_sources",
                "Sources or references are required but no usable Sources section was found.",
                "Add a Sources section with links or workspace file references.",
            )
        )
    if require_summary and not _has_heading_named(headings, ("summary", "executive summary", "摘要", "总结")):
        findings.append(
            _finding(
                "warning",
                "missing_summary",
                "A summary section is required but was not found.",
                "Add a short Summary section near the top.",
            )
        )
    if require_acceptance_criteria and not _has_heading_named(
        headings,
        ("acceptance criteria", "acceptance", "验收标准", "验收"),
    ):
        findings.append(
            _finding(
                "warning",
                "missing_acceptance_criteria",
                "Acceptance criteria are required but no matching section was found.",
                "Add testable acceptance criteria.",
            )
        )
    if require_diagram_captions:
        for image_index, match in enumerate(MARKDOWN_IMAGE_PATTERN.finditer(text), start=1):
            caption = (match.group(3) or "").strip()
            if not caption:
                findings.append(
                    _finding(
                        "warning",
                        "missing_image_caption",
                        f"Image {image_index} has no Markdown image title for caption rendering.",
                        "Use syntax like ![Alt](path.svg \"Caption\").",
                    )
                )
    return tuple(findings)


def _review_html(
    text: str,
    *,
    require_sources: bool,
    require_summary: bool,
    require_diagram_captions: bool,
) -> tuple[dict[str, object], ...]:
    findings: list[dict[str, object]] = []
    lowered = text.lower()
    if "<title>" not in lowered or "</title>" not in lowered:
        findings.append(
            _finding("warning", "missing_html_title", "HTML has no <title> element.", "Set a document title.")
        )
    if "<h1" not in lowered:
        findings.append(
            _finding("warning", "missing_h1", "HTML has no visible or semantic h1.", "Keep one report title.")
        )
    if 'name="viewport"' not in lowered:
        findings.append(
            _finding(
                "warning",
                "missing_viewport",
                "HTML has no responsive viewport meta tag.",
                "Include a mobile-safe viewport tag.",
            )
        )
    if "@media print" not in lowered:
        findings.append(
            _finding(
                "warning",
                "missing_print_css",
                "HTML has no print stylesheet.",
                "Include print CSS for shareable reports.",
            )
        )
    if PLACEHOLDER_PATTERN.search(text):
        findings.append(
            _finding(
                "error",
                "placeholder",
                "HTML contains TODO/TBD/FIXME or unresolved placeholder text.",
                "Replace placeholders before final delivery.",
            )
        )
    if re.search(r"<\s*script\b", text, flags=re.IGNORECASE):
        findings.append(
            _finding(
                "error",
                "script_tag",
                "HTML contains a script tag.",
                "Remove scripts from static deliverables unless explicitly required.",
            )
        )
    if re.search(r"(?:href|src)\s*=\s*['\"]\s*javascript:", text, flags=re.IGNORECASE):
        findings.append(
            _finding(
                "error",
                "unsafe_link",
                "HTML contains a javascript: link or source.",
                "Remove unsafe links before delivery.",
            )
        )
    if "svg image not embedded:" in lowered or "class=\"asset-note\"" in lowered:
        findings.append(
            _finding(
                "warning",
                "asset_not_embedded",
                "HTML reports at least one image asset that was not embedded.",
                "Fix the referenced SVG or remove the image before final delivery.",
            )
        )
    if require_sources and not _html_has_sources(text):
        findings.append(
            _finding(
                "warning",
                "missing_sources",
                "Sources or references are required but no usable section was found.",
                "Add a Sources section with links or workspace file references.",
            )
        )
    if require_summary and not re.search(r">\s*(summary|executive summary|摘要|总结)\s*<", lowered):
        findings.append(
            _finding(
                "warning",
                "missing_summary",
                "A summary section is required but was not found.",
                "Add a short Summary section near the top.",
            )
        )
    if require_diagram_captions:
        diagram_figures = re.findall(
            r"<figure\b[^>]*class=\"[^\"]*\bdiagram\b[^\"]*\"[^>]*>(.*?)</figure>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        for index, figure in enumerate(diagram_figures, start=1):
            if not re.search(r"<figcaption\b", figure, flags=re.IGNORECASE):
                findings.append(
                    _finding(
                        "warning",
                        "missing_diagram_caption",
                        f"Diagram figure {index} has no figcaption.",
                        "Add a Markdown image title before rendering the report.",
                    )
                )
    return tuple(findings)


def _review_svg(text: str) -> tuple[dict[str, object], ...]:
    findings: list[dict[str, object]] = []
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as exc:
        return (
            _finding(
                "error",
                "invalid_svg_xml",
                f"SVG XML parse failed: {exc}.",
                "Regenerate or fix the SVG before delivery.",
            ),
        )
    if _local_name(root.tag) != "svg":
        findings.append(
            _finding("error", "not_svg", "Artifact root is not <svg>.", "Use an SVG file.")
        )
    if "viewBox=" not in text:
        findings.append(
            _finding(
                "warning",
                "missing_viewbox",
                "SVG has no viewBox.",
                "Add a viewBox so the diagram scales cleanly.",
            )
        )
    if PLACEHOLDER_PATTERN.search(text):
        findings.append(
            _finding(
                "error",
                "placeholder",
                "SVG contains TODO/TBD/FIXME or unresolved placeholder text.",
                "Replace placeholders before final delivery.",
            )
        )
    if UNSAFE_SVG_PATTERN.search(text):
        findings.append(
            _finding(
                "error",
                "unsafe_svg_markup",
                "SVG contains scripts, event handlers, foreignObject, or external hrefs.",
                "Remove unsafe or externally-loaded markup.",
            )
        )
    marker_refs = set(re.findall(r"url\(#([^)]+)\)", text))
    marker_defs = set(re.findall(r'<marker\b[^>]*\bid="([^"]+)"', text))
    missing = sorted(marker_refs - marker_defs)
    if missing:
        findings.append(
            _finding(
                "error",
                "missing_marker_defs",
                f"SVG references marker ids that are not defined: {', '.join(missing[:6])}.",
                "Regenerate the diagram or add the missing marker definitions.",
            )
        )
    if 'role="img"' not in text and "aria-label=" not in text:
        findings.append(
            _finding(
                "info",
                "missing_accessible_label",
                "SVG has no role=\"img\" or aria-label.",
                "Add an accessible label for shareable diagrams.",
            )
        )
    return tuple(findings)


def _artifact_type(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix == ".svg":
        return "svg"
    raise ValueError("artifact_type=auto supports Markdown, HTML, and SVG files")


def _read_review_text(path: Path) -> tuple[str, bool]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        text = handle.read(MAX_REVIEW_CHARS + 1)
    return text[:MAX_REVIEW_CHARS], len(text) > MAX_REVIEW_CHARS


def _markdown_headings(text: str) -> tuple[tuple[int, str], ...]:
    headings: list[tuple[int, str]] = []
    in_code = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if match:
            headings.append((len(match.group(1)), match.group(2).strip()))
    return tuple(headings)


def _heading_jump_findings(headings: Sequence[tuple[int, str]]) -> tuple[dict[str, object], ...]:
    findings: list[dict[str, object]] = []
    for previous, current in zip(headings, headings[1:]):
        if current[0] > previous[0] + 1:
            findings.append(
                _finding(
                    "warning",
                    "heading_jump",
                    f"Heading jumps from h{previous[0]} to h{current[0]} before '{current[1]}'.",
                    "Use sequential heading levels for easier scanning.",
                )
            )
    return tuple(findings)


def _empty_markdown_section_findings(text: str) -> tuple[dict[str, object], ...]:
    findings: list[dict[str, object]] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if not match:
            continue
        if len(match.group(1)) == 1:
            continue
        has_content = False
        for next_line in lines[index + 1 :]:
            stripped = next_line.strip()
            if re.match(r"^#{1,6}\s+", stripped):
                break
            if stripped and stripped not in {"---", "***", "___"}:
                has_content = True
                break
        if not has_content:
            findings.append(
                _finding(
                    "warning",
                    "empty_section",
                    f"Section '{match.group(2).strip()}' has no content.",
                    "Remove empty sections or add concise content.",
                )
            )
    return tuple(findings)


def _long_paragraph_findings(text: str, max_chars: int) -> tuple[dict[str, object], ...]:
    findings: list[dict[str, object]] = []
    paragraphs = re.split(r"\n\s*\n", text)
    for index, paragraph in enumerate(paragraphs, start=1):
        clean = " ".join(line.strip() for line in paragraph.splitlines())
        if not clean or clean.startswith("#") or clean.startswith("|") or clean.startswith("```"):
            continue
        if len(clean) > max_chars:
            findings.append(
                _finding(
                    "warning",
                    "long_paragraph",
                    f"Paragraph {index} is {len(clean)} characters.",
                    "Split long prose into shorter paragraphs or bullets.",
                )
            )
    return tuple(findings)


def _markdown_has_sources(text: str) -> bool:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^#{1,6}\s+(.+)$", line.strip())
        if not match or not _is_sources_heading(match.group(1)):
            continue
        section = "\n".join(_section_lines(lines, index + 1))
        if re.search(r"https?://|`[^`]+`|\[[^\]]+\]\([^)]+\)|[/\\][\w.-]+", section):
            return True
    return False


def _html_has_sources(text: str) -> bool:
    lowered = text.lower()
    if not re.search(r"id=\"(?:sources|references|参考来源|来源|参考)\"", lowered):
        if not re.search(r">\s*(sources|references|参考来源|来源|参考)\s*<", lowered):
            return False
    return bool(re.search(r"https?://|<a\b|<code>[^<]+</code>|[/\\][\w.-]+", text, re.IGNORECASE))


def _section_lines(lines: Sequence[str], start: int) -> tuple[str, ...]:
    section: list[str] = []
    for line in lines[start:]:
        if re.match(r"^#{1,6}\s+", line.strip()):
            break
        section.append(line)
    return tuple(section)


def _has_heading_named(headings: Sequence[tuple[int, str]], names: Sequence[str]) -> bool:
    normalized = {name.lower() for name in names}
    for _, heading in headings:
        clean = heading.strip().lower()
        if clean in normalized:
            return True
    return False


def _is_sources_heading(value: str) -> bool:
    return value.strip().lower() in {
        "sources",
        "source ledger",
        "references",
        "参考来源",
        "来源",
        "参考",
    }


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _finding(severity: str, code: str, message: str, suggestion: str) -> dict[str, object]:
    return {
        "severity": severity,
        "code": code,
        "message": message,
        "suggestion": suggestion,
    }


def _text(arguments: Mapping[str, object], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _choice(
    arguments: Mapping[str, object],
    key: str,
    default: str,
    choices: Sequence[str],
) -> str:
    value = arguments.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    value = value.strip().lower()
    if value not in choices:
        raise ValueError(f"{key} must be one of: {', '.join(choices)}")
    return value


def _bool(arguments: Mapping[str, object], key: str, default: bool) -> bool:
    value = arguments.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _int(arguments: Mapping[str, object], key: str, default: int, minimum: int, maximum: int) -> int:
    value = arguments.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


def _review_artifact_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative artifact path.",
            },
            "artifact_type": {
                "type": "string",
                "enum": list(ARTIFACT_TYPES),
                "description": "Use auto for .md/.html/.svg files.",
            },
            "require_sources": {
                "type": "boolean",
                "description": "Require a usable Sources or References section.",
            },
            "require_summary": {
                "type": "boolean",
                "description": "Require a Summary or equivalent section.",
            },
            "require_acceptance_criteria": {
                "type": "boolean",
                "description": "Require acceptance criteria for PRD/spec artifacts.",
            },
            "require_diagram_captions": {
                "type": "boolean",
                "description": "Require captions for Markdown images or HTML diagram figures.",
            },
            "max_paragraph_chars": {
                "type": "integer",
                "description": "Warn when Markdown paragraphs exceed this length.",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }
