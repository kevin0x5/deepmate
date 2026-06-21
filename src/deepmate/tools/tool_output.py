"""Native retrieval tool for compacted tool outputs."""

from __future__ import annotations

from collections.abc import Mapping

from deepmate.foundation import estimate_text_tokens
from deepmate.storage.tool_output_store import ToolOutputStore
from deepmate.tools.registry import NativeTool, NativeToolResult

RETRIEVE_TOOL_OUTPUT_NAME = "retrieve_tool_output"
DEFAULT_RETRIEVE_TOKENS = 2_000
MAX_RETRIEVE_TOKENS = 8_000


def tool_output_tools(store: ToolOutputStore | None) -> tuple[NativeTool, ...]:
    """Return native tools for retrieving session-scoped raw outputs."""
    if store is None:
        return ()
    return (
        NativeTool(
            name=RETRIEVE_TOOL_OUTPUT_NAME,
            description=(
                "Retrieve bounded excerpts from a Deepmate-compacted tool output "
                "by retrieval ref. Use this only when a compacted tool result says "
                "more exact detail is needed."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "Tool output ref returned by a compacted tool result.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional text to search for relevant raw-output excerpts.",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Optional maximum estimated tokens to return.",
                    },
                },
                "required": ["ref"],
            },
            handler=lambda arguments: _retrieve_tool_output(store, arguments),
        ),
    )


def _retrieve_tool_output(
    store: ToolOutputStore,
    arguments: Mapping[str, object],
) -> NativeToolResult:
    ref = _text(arguments.get("ref")).strip()
    if not ref:
        return NativeToolResult(
            content="tool_output_ref_required: ref is required.",
            refs=("tool_output_ref_required",),
        )
    record = store.load(ref)
    if record is None:
        return NativeToolResult(
            content=f"tool_output_ref_not_found: {ref}",
            refs=("tool_output_ref_not_found", f"tool_output_ref={ref}"),
        )
    query = _text(arguments.get("query")).strip()
    max_tokens = _max_tokens(arguments.get("max_tokens"))
    if query:
        content, matched_chunks = _query_excerpt(record.content, query, max_tokens)
    else:
        content, matched_chunks = _bounded_excerpt(record.content, max_tokens), 0
    header = (
        "[tool output retrieved]\n"
        f"Ref: {record.ref}\n"
        f"Tool: {record.tool_name}\n"
        f"Source: {record.tool_source}\n"
        f"Kind: {record.content_kind}\n"
        f"Original estimated tokens: {record.estimated_tokens}\n"
    )
    if query:
        header += f"Query: {query}\nMatched chunks: {matched_chunks}\n"
    return NativeToolResult(
        content=f"{header}\n{content}",
        refs=(
            f"tool_output_ref={record.ref}",
            f"tool_output_retrieved_tokens={estimate_text_tokens(content)}",
            f"matched_chunks={matched_chunks}",
        ),
    )


def _bounded_excerpt(text: str, max_tokens: int) -> str:
    if estimate_text_tokens(text) <= max_tokens:
        return text
    lines = text.splitlines()
    if not lines:
        return _truncate_chars(text, max_tokens)
    head_count = min(40, max(1, len(lines) // 4))
    tail_count = min(40, max(1, len(lines) // 4))
    excerpt = _with_omission(lines[:head_count], lines[-tail_count:], len(lines))
    while estimate_text_tokens(excerpt) > max_tokens and (head_count > 1 or tail_count > 1):
        if head_count >= tail_count and head_count > 1:
            head_count -= 1
        elif tail_count > 1:
            tail_count -= 1
        excerpt = _with_omission(lines[:head_count], lines[-tail_count:], len(lines))
    if estimate_text_tokens(excerpt) <= max_tokens:
        return excerpt
    return _truncate_chars(excerpt, max_tokens)


def _query_excerpt(text: str, query: str, max_tokens: int) -> tuple[str, int]:
    lines = text.splitlines()
    if not lines:
        return _bounded_excerpt(text, max_tokens), 0
    lowered = query.lower()
    matched_indexes = [
        index for index, line in enumerate(lines) if lowered in line.lower()
    ]
    if not matched_indexes:
        return (
            "No direct matches for query. Returning bounded raw-output excerpt.\n\n"
            + _bounded_excerpt(text, max_tokens),
            0,
        )
    chunks: list[str] = []
    seen: set[int] = set()
    for match in matched_indexes[:20]:
        start = max(0, match - 2)
        end = min(len(lines), match + 3)
        chunk_lines = []
        for index in range(start, end):
            if index in seen:
                continue
            seen.add(index)
            chunk_lines.append(f"{index + 1}: {lines[index]}")
        if chunk_lines:
            chunks.append("\n".join(chunk_lines))
    content = "\n\n---\n\n".join(chunks)
    while estimate_text_tokens(content) > max_tokens and len(chunks) > 1:
        chunks.pop()
        content = "\n\n---\n\n".join(chunks)
    if estimate_text_tokens(content) > max_tokens:
        content = _truncate_chars(content, max_tokens)
    return content, len(chunks)


def _with_omission(head: list[str], tail: list[str], total_lines: int) -> str:
    omitted = max(0, total_lines - len(head) - len(tail))
    marker = f"[... omitted {omitted} middle line(s) ...]"
    if omitted <= 0:
        return "\n".join(head)
    return "\n".join((*head, marker, *tail))


def _truncate_chars(text: str, max_tokens: int) -> str:
    if estimate_text_tokens(text) <= max_tokens:
        return text
    keep_chars = max(1, int(len(text) * max_tokens / max(1, estimate_text_tokens(text))))
    candidate = text[:keep_chars].rstrip()
    while estimate_text_tokens(candidate) > max_tokens and keep_chars > 1:
        keep_chars = max(1, keep_chars - max(1, keep_chars // 5))
        candidate = text[:keep_chars].rstrip()
    return candidate + "\n[... output bounded by retrieve budget ...]"


def _max_tokens(value: object) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_RETRIEVE_TOKENS
    return min(MAX_RETRIEVE_TOKENS, max(1, parsed))


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""
