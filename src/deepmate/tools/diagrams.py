"""Deterministic SVG technical diagram rendering tools."""

from __future__ import annotations

import html
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
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

RENDER_TECH_DIAGRAM_TOOL_NAME = "render_tech_diagram"

DIAGRAM_TYPES = (
    "architecture",
    "agent_architecture",
    "flowchart",
    "sequence",
    "comparison",
    "timeline",
)
THEMES = ("prussian", "forest", "graphite", "blueprint")
PLACEHOLDER_PATTERN = re.compile(r"\b(?:TODO|TBD|FIXME)\b|<what you need>", re.IGNORECASE)
MAX_NODES = 40
MAX_EDGES = 80
MAX_GROUPS = 12
MAX_ROWS = 30
MAX_COLUMNS = 8
MAX_MILESTONES = 30


@dataclass(frozen=True, slots=True)
class Theme:
    name: str
    canvas: str
    canvas_alt: str
    text: str
    muted: str
    border: str
    node: str
    node_alt: str
    primary: str
    secondary: str
    accent: str
    warning: str
    grid: str


THEME_MAP: Mapping[str, Theme] = {
    "prussian": Theme(
        name="prussian",
        canvas="#f8fafc",
        canvas_alt="#eef4f8",
        text="#102638",
        muted="#5f7182",
        border="#b8c9d8",
        node="#ffffff",
        node_alt="#e8f1f6",
        primary="#003153",
        secondary="#1f6f8b",
        accent="#b48a2c",
        warning="#b45309",
        grid="#d7e3ec",
    ),
    "forest": Theme(
        name="forest",
        canvas="#f7faf7",
        canvas_alt="#edf5ef",
        text="#183326",
        muted="#647466",
        border="#b8cdbc",
        node="#ffffff",
        node_alt="#e8f2e9",
        primary="#1f4d36",
        secondary="#4d7c59",
        accent="#a67c2d",
        warning="#a16207",
        grid="#d7e5d9",
    ),
    "graphite": Theme(
        name="graphite",
        canvas="#f7f7f5",
        canvas_alt="#eeeeeb",
        text="#242526",
        muted="#686b70",
        border="#c5c6c8",
        node="#ffffff",
        node_alt="#eceef0",
        primary="#2f3437",
        secondary="#5b6268",
        accent="#9c7a3a",
        warning="#9a3412",
        grid="#dddddc",
    ),
    "blueprint": Theme(
        name="blueprint",
        canvas="#082f49",
        canvas_alt="#0b3b5e",
        text="#e0f2fe",
        muted="#9bd8f4",
        border="#67e8f9",
        node="#0b3b5e",
        node_alt="#0d4c75",
        primary="#67e8f9",
        secondary="#38bdf8",
        accent="#fde047",
        warning="#fb7185",
        grid="#0ea5e9",
    ),
}

FLOW_STYLES: Mapping[str, tuple[str, str]] = {
    "primary": ("primary", ""),
    "data": ("primary", ""),
    "control": ("accent", ""),
    "trigger": ("accent", ""),
    "read": ("secondary", ""),
    "write": ("secondary", "5 4"),
    "async": ("muted", "5 4"),
    "event": ("muted", "5 4"),
    "transform": ("accent", "2 3"),
    "feedback": ("accent", ""),
}


@dataclass(frozen=True, slots=True)
class Box:
    node_id: str
    label: str
    kind: str
    x: float
    y: float
    width: float
    height: float

    @property
    def cx(self) -> float:
        return self.x + self.width / 2

    @property
    def cy(self) -> float:
        return self.y + self.height / 2


def workspace_diagram_tools(
    workspace_root: str | Path,
    *,
    write_checkpoint: WorkspaceWriteCheckpoint | None = None,
    hook_context: HookRuntimeContext | None = None,
) -> tuple[NativeTool, ...]:
    """Return deterministic technical diagram rendering tools."""
    root = Path(workspace_root).resolve()
    if not root.is_dir():
        raise ValueError(f"workspace root must be a directory: {root}")
    return (
        NativeTool(
            name=RENDER_TECH_DIAGRAM_TOOL_NAME,
            description=(
                "Render a structured technical diagram spec into a polished "
                "self-contained SVG file."
            ),
            input_schema=_render_tech_diagram_schema(),
            handler=lambda arguments: _render_tech_diagram(
                root,
                arguments,
                write_checkpoint=write_checkpoint,
                hook_context=hook_context,
            ),
            read_only=False,
        ),
    )


def _render_tech_diagram(
    root: Path,
    arguments: Mapping[str, object],
    *,
    write_checkpoint: WorkspaceWriteCheckpoint | None,
    hook_context: HookRuntimeContext | None,
) -> NativeToolResult:
    diagram_type = _choice(arguments, "diagram_type", "architecture", DIAGRAM_TYPES)
    output_path = _workspace_path(root, _text(arguments, "output_path"))
    theme_name = _choice(arguments, "theme", "prussian", THEMES)
    overwrite = _bool(arguments, "overwrite", False)
    title = _optional_text(arguments, "title", "Technical Diagram")
    subtitle = _optional_text(arguments, "subtitle", "")
    if output_path.suffix.lower() != ".svg":
        raise ValueError("output_path must end in .svg")
    if output_path.exists() and not overwrite:
        raise ValueError("output_path already exists; pass overwrite=true to replace it")
    if output_path.exists() and not output_path.is_file():
        raise ValueError("output_path exists but is not a file")
    if not output_path.parent.is_dir():
        raise ValueError(f"parent directory does not exist: {output_path.parent}")

    spec = dict(arguments)
    theme = THEME_MAP[theme_name]
    checks = _validate_spec(diagram_type, spec)
    rendered = _render_svg(diagram_type, spec, theme, title=title, subtitle=subtitle)
    checks.extend(_quality_checks(rendered, spec))
    if len(rendered) > MAX_WRITE_CHARS:
        raise ValueError(f"rendered diagram must be at most {MAX_WRITE_CHARS} characters")

    before_outcome = _emit_write_hook(
        hook_context,
        HookEvent.WRITE_BEFORE,
        root=root,
        path=output_path,
        operation=RENDER_TECH_DIAGRAM_TOOL_NAME,
        status="before",
        payload={
            "content_size": len(rendered),
            "diagram_type": diagram_type,
            "path_kind": "file",
            "theme": theme_name,
        },
    )
    if before_outcome.directive != HookDirective.CONTINUE:
        raise ValueError(
            before_outcome.reason
            or f"Workspace write stopped by hook: {before_outcome.directive.value}"
        )
    if write_checkpoint is not None:
        write_checkpoint(RENDER_TECH_DIAGRAM_TOOL_NAME, output_path, rendered)
    _atomic_write_text(output_path, rendered)
    after_outcome = _emit_write_hook(
        hook_context,
        HookEvent.WRITE_AFTER,
        root=root,
        path=output_path,
        operation=RENDER_TECH_DIAGRAM_TOOL_NAME,
        status="completed",
        payload={
            "content_size": len(rendered),
            "diagram_type": diagram_type,
            "path_kind": "file",
            "theme": theme_name,
        },
    )

    relative_output = _relative_path(root, output_path)
    failed_checks = tuple(check for check in checks if not check["passed"])
    node_count = len(_mapping_list(spec, "nodes", MAX_NODES))
    edge_count = len(_mapping_list(spec, "edges", MAX_EDGES))
    content = "\n".join(
        (
            f"Rendered {relative_output}.",
            (
                f"Diagram: {diagram_type}; theme: {theme_name}; "
                f"nodes={node_count}; edges={edge_count}."
            ),
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
            "path": relative_output,
            "diagram_type": diagram_type,
            "theme": theme_name,
            "nodes": node_count,
            "edges": edge_count,
            "checks": checks,
            "warnings": len(failed_checks),
            "chars_written": len(rendered),
        },
        refs=(relative_output, *_hook_refs(after_outcome)),
    )


def _render_svg(
    diagram_type: str,
    spec: Mapping[str, object],
    theme: Theme,
    *,
    title: str,
    subtitle: str,
) -> str:
    if diagram_type == "comparison":
        width, height, body = _render_comparison(spec, theme)
    elif diagram_type == "timeline":
        width, height, body = _render_timeline(spec, theme)
    elif diagram_type == "sequence":
        width, height, body = _render_sequence(spec, theme)
    elif diagram_type in {"architecture", "agent_architecture", "flowchart"}:
        width, height, body = _render_node_edge_diagram(diagram_type, spec, theme)
    else:
        raise ValueError(f"Unsupported diagram_type: {diagram_type}")
    return "\n".join(
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" role="img" aria-label="{_esc(title)}">',
            _defs(theme),
            f'  <rect width="{width}" height="{height}" fill="{theme.canvas}"/>',
            _grid(width, height, theme) if theme.name == "blueprint" else "",
            _title_block(title, subtitle, theme, width),
            body,
            "</svg>",
        )
    )


def _render_node_edge_diagram(
    diagram_type: str,
    spec: Mapping[str, object],
    theme: Theme,
) -> tuple[int, int, str]:
    nodes = _mapping_list(spec, "nodes", MAX_NODES)
    edges = _mapping_list(spec, "edges", MAX_EDGES)
    groups = _mapping_list(spec, "groups", MAX_GROUPS)
    if not nodes:
        nodes = ({"id": "topic", "label": "Topic", "kind": "process"},)
    group_order = _group_order(nodes, groups)
    row_count = max(1, len(group_order))
    row_height = 164
    width = 1040
    height = max(540, 190 + row_count * row_height)
    boxes = _layout_boxes(nodes, group_order, diagram_type, width)
    lines: list[str] = []

    for index, group_id in enumerate(group_order):
        label = _group_label(group_id, groups)
        y = 112 + index * row_height
        lines.append(
            f'  <rect x="34" y="{y}" width="{width - 68}" height="{row_height - 28}" '
            f'rx="14" fill="{theme.canvas_alt}" stroke="{theme.border}" '
            f'stroke-dasharray="6 5" opacity="0.86"/>'
        )
        lines.append(
            f'  <text x="52" y="{y + 26}" class="section">{_esc(label)}</text>'
        )

    node_map = {box.node_id: box for box in boxes}
    for edge in edges:
        source = _optional_text(edge, "source", "")
        target = _optional_text(edge, "target", "")
        if source not in node_map or target not in node_map:
            continue
        lines.append(_edge_svg(node_map[source], node_map[target], edge, theme))

    for box in boxes:
        lines.append(_node_svg(box, theme))

    legend = _legend_for_edges(edges, theme, height)
    if legend:
        lines.append(legend)
    return width, height, "\n".join(lines)


def _render_sequence(spec: Mapping[str, object], theme: Theme) -> tuple[int, int, str]:
    nodes = _mapping_list(spec, "nodes", MAX_NODES)
    edges = _mapping_list(spec, "edges", MAX_EDGES)
    participants = nodes or _participants_from_edges(edges)
    if not participants:
        participants = ({"id": "user", "label": "User"}, {"id": "system", "label": "System"})
    participants = participants[:8]
    width = max(760, 180 + len(participants) * 150)
    height = max(460, 206 + len(edges) * 62)
    x_positions = {
        _node_id(node, index): 90 + index * ((width - 180) / max(1, len(participants) - 1))
        for index, node in enumerate(participants)
    }
    lines: list[str] = []
    for index, node in enumerate(participants):
        node_id = _node_id(node, index)
        x = x_positions[node_id]
        label = _label(node, f"Participant {index + 1}")
        lines.append(
            f'  <rect x="{x - 58:.1f}" y="112" width="116" height="42" rx="10" '
            f'fill="{theme.node}" stroke="{theme.border}"/>'
        )
        lines.append(
            f'  <text x="{x:.1f}" y="138" text-anchor="middle" class="node-title-small">'
            f'{_esc(label)}</text>'
        )
        lines.append(
            f'  <line x1="{x:.1f}" y1="164" x2="{x:.1f}" y2="{height - 54}" '
            f'stroke="{theme.border}" stroke-dasharray="5 5"/>'
        )
    for index, edge in enumerate(edges):
        source = _optional_text(edge, "source", "")
        target = _optional_text(edge, "target", "")
        if source not in x_positions or target not in x_positions:
            continue
        y = 206 + index * 58
        x1 = x_positions[source]
        x2 = x_positions[target]
        color, dash = _flow_color(edge, theme)
        marker_id = _marker_for_edge(edge)
        marker = (
            f'marker-end="url(#{marker_id})"'
            if x1 <= x2
            else f'marker-start="url(#{marker_id}Reverse)"'
        )
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(
            f'  <line x1="{x1:.1f}" y1="{y}" x2="{x2:.1f}" y2="{y}" '
            f'stroke="{color}" stroke-width="2"{dash_attr} {marker}/>'
        )
        label = _optional_text(edge, "label", _optional_text(edge, "flow", "message"))
        frame_label = _optional_text(edge, "frame", "")
        if frame_label:
            lines.append(
                f'  <rect x="46" y="{y - 36}" width="{width - 92}" height="42" rx="10" '
                f'fill="{theme.canvas_alt}" stroke="{theme.border}" stroke-dasharray="5 4" '
                f'opacity="0.72"/>'
            )
            lines.append(
                f'  <text x="62" y="{y - 14}" class="muted">{_esc(_short(frame_label, 42))}</text>'
            )
        lines.append(
            f'  <rect x="{min(x1, x2) + abs(x2 - x1) / 2 - 48:.1f}" y="{y - 25}" '
            f'width="96" height="18" rx="9" fill="{theme.canvas}" opacity="0.94"/>'
        )
        lines.append(
            f'  <text x="{min(x1, x2) + abs(x2 - x1) / 2:.1f}" y="{y - 11}" '
            f'text-anchor="middle" class="edge-label">{_esc(_short(label, 24))}</text>'
        )
    return width, height, "\n".join(lines)


def _render_comparison(spec: Mapping[str, object], theme: Theme) -> tuple[int, int, str]:
    columns = _string_list(spec, "columns", MAX_COLUMNS)
    rows = _mapping_list(spec, "rows", MAX_ROWS)
    if not columns:
        columns = ("Option A", "Option B")
    if not rows:
        rows = ({"label": "Fit", "values": ["", ""]},)
    column_width = 164
    label_width = 210
    row_height = 58
    width = max(760, label_width + len(columns) * column_width + 80)
    height = max(380, 150 + (len(rows) + 1) * row_height)
    x0 = 40
    y0 = 116
    lines = [
        f'  <rect x="{x0}" y="{y0}" width="{width - 80}" height="{(len(rows) + 1) * row_height}" '
        f'rx="12" fill="{theme.node}" stroke="{theme.border}"/>'
    ]
    lines.append(
        f'  <rect x="{x0}" y="{y0}" width="{width - 80}" height="{row_height}" '
        f'rx="12" fill="{theme.canvas_alt}" stroke="{theme.border}"/>'
    )
    for index, column in enumerate(columns):
        x = x0 + label_width + index * column_width
        lines.append(
            f'  <text x="{x + column_width / 2}" y="{y0 + 29}" text-anchor="middle" '
            f'class="node-title-small">{_esc(_short(column, 24))}</text>'
        )
    for row_index, row in enumerate(rows):
        y = y0 + row_height * (row_index + 1)
        fill = theme.node if row_index % 2 == 0 else theme.canvas_alt
        lines.append(
            f'  <rect x="{x0}" y="{y}" width="{width - 80}" height="{row_height}" fill="{fill}" '
            f'opacity="0.7"/>'
        )
        label = _optional_text(row, "label", f"Row {row_index + 1}")
        lines.append(_multiline_text(label, x0 + 18, y + 23, 26, "label"))
        values = _values(row.get("values"), len(columns))
        for column_index, value in enumerate(values):
            x = x0 + label_width + column_index * column_width
            lines.append(
                _multiline_text(
                    value,
                    x + column_width / 2,
                    y + 20,
                    22,
                    "label",
                    anchor="middle",
                    max_lines=2,
                )
            )
    for index in range(len(columns) + 1):
        x = x0 + label_width + index * column_width
        lines.append(
            f'  <line x1="{x}" y1="{y0}" x2="{x}" y2="{y0 + (len(rows) + 1) * row_height}" '
            f'stroke="{theme.border}" opacity="0.75"/>'
        )
    return width, height, "\n".join(lines)


def _render_timeline(spec: Mapping[str, object], theme: Theme) -> tuple[int, int, str]:
    milestones = _mapping_list(spec, "milestones", MAX_MILESTONES)
    phases = _mapping_list(spec, "phases", MAX_MILESTONES)
    if not milestones:
        milestones = ({"label": "Start", "time": "T0"}, {"label": "Finish", "time": "T1"})
    width = max(760, 220 + len(milestones) * 140)
    height = 470 if phases else 420
    left = 88
    right = width - 88
    axis_y = 210
    step = (right - left) / max(1, len(milestones) - 1)
    lines = [
        f'  <line x1="{left}" y1="{axis_y}" x2="{right}" y2="{axis_y}" '
        f'stroke="{theme.primary}" stroke-width="3" marker-end="url(#arrowPrimary)"/>'
    ]
    if phases:
        phase_y = 304
        phase_width = (right - left) / max(1, len(phases))
        for index, phase in enumerate(phases):
            x = left + index * phase_width
            label = _optional_text(phase, "label", f"Phase {index + 1}")
            detail = _optional_text(phase, "detail", _optional_text(phase, "time", ""))
            lines.append(
                f'  <rect x="{x:.1f}" y="{phase_y}" width="{phase_width - 12:.1f}" height="58" '
                f'rx="12" fill="{theme.node}" stroke="{theme.border}"/>'
            )
            lines.append(
                f'  <text x="{x + 14:.1f}" y="{phase_y + 23}" class="node-title-small">'
                f'{_esc(_short(label, 22))}</text>'
            )
            if detail:
                lines.append(
                    f'  <text x="{x + 14:.1f}" y="{phase_y + 42}" class="muted">'
                    f'{_esc(_short(detail, 28))}</text>'
                )
    for index, item in enumerate(milestones):
        x = left + index * step
        label = _optional_text(item, "label", f"Milestone {index + 1}")
        time = _optional_text(item, "time", "")
        y = axis_y - 76 if index % 2 == 0 else axis_y + 54
        lines.append(f'  <circle cx="{x:.1f}" cy="{axis_y}" r="8" fill="{theme.accent}"/>')
        lines.append(
            f'  <line x1="{x:.1f}" y1="{axis_y}" x2="{x:.1f}" y2="{y + 12}" '
            f'stroke="{theme.border}" stroke-dasharray="4 4"/>'
        )
        lines.append(
            f'  <rect x="{x - 64:.1f}" y="{y}" width="128" height="54" rx="10" '
            f'fill="{theme.node}" stroke="{theme.border}"/>'
        )
        lines.append(
            f'  <text x="{x:.1f}" y="{y + 22}" text-anchor="middle" class="node-title-small">'
            f'{_esc(_short(label, 22))}</text>'
        )
        if time:
            lines.append(
                f'  <text x="{x:.1f}" y="{y + 40}" text-anchor="middle" class="muted">'
                f'{_esc(_short(time, 18))}</text>'
            )
    return width, height, "\n".join(lines)


def _layout_boxes(
    nodes: Sequence[Mapping[str, object]],
    group_order: Sequence[str],
    diagram_type: str,
    width: int,
) -> tuple[Box, ...]:
    boxes: list[Box] = []
    row_height = 164
    for row_index, group_id in enumerate(group_order):
        row_nodes = [
            node
            for node in nodes
            if _optional_text(node, "group", "default") == group_id
        ]
        count = max(1, len(row_nodes))
        box_width = 156 if diagram_type == "flowchart" else 168
        box_height = 76
        available = width - 180
        gap = available / max(1, count)
        for index, node in enumerate(row_nodes):
            x = 90 + gap * index + gap / 2 - box_width / 2
            y = 152 + row_index * row_height
            boxes.append(
                Box(
                    node_id=_node_id(node, len(boxes)),
                    label=_label(node, f"Node {len(boxes) + 1}"),
                    kind=_optional_text(node, "kind", "process"),
                    x=x,
                    y=y,
                    width=box_width,
                    height=box_height,
                )
            )
    return tuple(boxes)


def _node_svg(box: Box, theme: Theme) -> str:
    label_lines = _wrap_text(box.label, 22, max_lines=2)
    kind = _esc(_short(box.kind.replace("_", " "), 18))
    if box.kind == "decision":
        points = (
            f"{box.cx:.1f},{box.y:.1f} {box.x + box.width:.1f},{box.cy:.1f} "
            f"{box.cx:.1f},{box.y + box.height:.1f} {box.x:.1f},{box.cy:.1f}"
        )
        shape = (
            f'  <polygon points="{points}" fill="{theme.node}" stroke="{theme.border}" '
            f'stroke-width="1.4"/>'
        )
    elif box.kind in {"agent", "orchestrator"}:
        x1 = box.x + 14
        x2 = box.x + box.width - 14
        points = (
            f"{x1:.1f},{box.y:.1f} {x2:.1f},{box.y:.1f} "
            f"{box.x + box.width:.1f},{box.cy:.1f} {x2:.1f},{box.y + box.height:.1f} "
            f"{x1:.1f},{box.y + box.height:.1f} {box.x:.1f},{box.cy:.1f}"
        )
        shape = (
            f'  <polygon points="{points}" fill="{theme.node}" stroke="{theme.primary}" '
            f'stroke-width="1.8"/>'
        )
    elif box.kind in {"database", "memory", "vector_store", "store"}:
        shape = "\n".join(
            (
                f'  <path d="M {box.x:.1f} {box.y + 12:.1f} C {box.x:.1f} {box.y - 4:.1f}, '
                f'{box.x + box.width:.1f} {box.y - 4:.1f}, {box.x + box.width:.1f} {box.y + 12:.1f} '
                f'L {box.x + box.width:.1f} {box.y + box.height - 12:.1f} C '
                f'{box.x + box.width:.1f} {box.y + box.height + 4:.1f}, {box.x:.1f} '
                f'{box.y + box.height + 4:.1f}, {box.x:.1f} {box.y + box.height - 12:.1f} Z" '
                f'fill="{theme.node}" stroke="{theme.secondary}" stroke-width="1.5"/>',
                f'  <path d="M {box.x:.1f} {box.y + 12:.1f} C {box.x:.1f} {box.y + 28:.1f}, '
                f'{box.x + box.width:.1f} {box.y + 28:.1f}, {box.x + box.width:.1f} {box.y + 12:.1f}" '
                f'fill="none" stroke="{theme.secondary}" stroke-width="1.2"/>',
            )
        )
    else:
        shape = (
            f'  <rect x="{box.x:.1f}" y="{box.y:.1f}" width="{box.width}" height="{box.height}" '
            f'rx="12" fill="{theme.node}" stroke="{theme.border}" stroke-width="1.4"/>'
        )
    return "\n".join(
        (
            shape,
            _text_lines_svg(
                label_lines,
                box.cx,
                box.y + 28,
                "node-title-small",
                anchor="middle",
                line_height=15,
            ),
            f'  <text x="{box.cx:.1f}" y="{box.y + 60:.1f}" text-anchor="middle" '
            f'class="muted">{kind}</text>',
        )
    )


def _edge_svg(source: Box, target: Box, edge: Mapping[str, object], theme: Theme) -> str:
    color, dash = _flow_color(edge, theme)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    if abs(source.cy - target.cy) < 6:
        y = source.cy
        x1 = source.x + source.width if source.cx <= target.cx else source.x
        x2 = target.x if source.cx <= target.cx else target.x + target.width
        path = f"M {x1:.1f} {y:.1f} L {x2:.1f} {y:.1f}"
        lx = (x1 + x2) / 2
        ly = y - 10
    else:
        x1 = source.cx
        y1 = source.y + source.height if source.cy <= target.cy else source.y
        x2 = target.cx
        y2 = target.y if source.cy <= target.cy else target.y + target.height
        mid_y = (y1 + y2) / 2
        path = f"M {x1:.1f} {y1:.1f} L {x1:.1f} {mid_y:.1f} L {x2:.1f} {mid_y:.1f} L {x2:.1f} {y2:.1f}"
        lx = (x1 + x2) / 2
        ly = mid_y - 8
    label = _optional_text(edge, "label", _optional_text(edge, "flow", ""))
    lines = [
        f'  <path d="{path}" fill="none" stroke="{color}" stroke-width="2"{dash_attr} '
        f'marker-end="url(#{_marker_for_edge(edge)})"/>'
    ]
    if label:
        text = _short(label, 22)
        width = max(42, len(text) * 7 + 14)
        lines.append(
            f'  <rect x="{lx - width / 2:.1f}" y="{ly - 14:.1f}" width="{width}" height="18" '
            f'rx="9" fill="{theme.canvas}" opacity="0.94"/>'
        )
        lines.append(
            f'  <text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" class="edge-label">'
            f'{_esc(text)}</text>'
        )
    return "\n".join(lines)


def _defs(theme: Theme) -> str:
    markers = []
    for marker_id, color in (
        ("arrowPrimary", theme.primary),
        ("arrowSecondary", theme.secondary),
        ("arrowAccent", theme.accent),
        ("arrowMuted", theme.muted),
    ):
        markers.extend(
            (
                f'    <marker id="{marker_id}" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">',
                f'      <polygon points="0 0, 10 3.5, 0 7" fill="{color}"/>',
                "    </marker>",
                f'    <marker id="{marker_id}Reverse" markerWidth="10" markerHeight="7" refX="1" refY="3.5" orient="auto">',
                f'      <polygon points="10 0, 0 3.5, 10 7" fill="{color}"/>',
                "    </marker>",
            )
        )
    return "\n".join(
        (
            "  <defs>",
            *markers,
            "    <style>",
            "      text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; }",
            f"      .title {{ font-size: 26px; font-weight: 750; fill: {theme.text}; }}",
            f"      .subtitle {{ font-size: 13px; font-weight: 500; fill: {theme.muted}; }}",
            f"      .section {{ font-size: 12px; font-weight: 750; fill: {theme.primary}; letter-spacing: 1.2px; }}",
            f"      .node-title-small {{ font-size: 13px; font-weight: 700; fill: {theme.text}; }}",
            f"      .label {{ font-size: 13px; font-weight: 560; fill: {theme.text}; }}",
            f"      .edge-label {{ font-size: 11px; font-weight: 650; fill: {theme.muted}; }}",
            f"      .muted {{ font-size: 11px; font-weight: 520; fill: {theme.muted}; }}",
            f"      .caption {{ font-size: 12px; font-weight: 520; fill: {theme.muted}; }}",
            "    </style>",
            "  </defs>",
        )
    )


def _title_block(title: str, subtitle: str, theme: Theme, width: int) -> str:
    lines = [f'  <text x="40" y="52" class="title">{_esc(title)}</text>']
    if subtitle:
        lines.append(f'  <text x="40" y="76" class="subtitle">{_esc(subtitle)}</text>')
    lines.append(
        f'  <line x1="40" y1="92" x2="{max(120, width - 40)}" y2="92" '
        f'stroke="{theme.border}" opacity="0.75"/>'
    )
    return "\n".join(lines)


def _grid(width: int, height: int, theme: Theme) -> str:
    lines = []
    for x in range(0, width + 1, 32):
        lines.append(
            f'  <line x1="{x}" y1="0" x2="{x}" y2="{height}" stroke="{theme.grid}" opacity="0.12"/>'
        )
    for y in range(0, height + 1, 32):
        lines.append(
            f'  <line x1="0" y1="{y}" x2="{width}" y2="{y}" stroke="{theme.grid}" opacity="0.12"/>'
        )
    return "\n".join(lines)


def _legend_for_edges(edges: Sequence[Mapping[str, object]], theme: Theme, height: int) -> str:
    flows = []
    for edge in edges:
        flow = _optional_text(edge, "flow", "primary").lower()
        if flow not in flows:
            flows.append(flow)
    if len(flows) < 2:
        return ""
    x = 40
    y = height - 58
    lines = [
        f'  <rect x="{x}" y="{y}" width="{max(180, len(flows) * 120)}" height="38" '
        f'rx="12" fill="{theme.canvas_alt}" stroke="{theme.border}" opacity="0.92"/>'
    ]
    for index, flow in enumerate(flows[:6]):
        color, dash = _flow_color({"flow": flow}, theme)
        marker_id = _marker_for_edge({"flow": flow})
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        lx = x + 16 + index * 118
        lines.append(
            f'  <line x1="{lx}" y1="{y + 22}" x2="{lx + 28}" y2="{y + 22}" '
            f'stroke="{color}" stroke-width="2"{dash_attr} marker-end="url(#{marker_id})"/>'
        )
        lines.append(f'  <text x="{lx + 38}" y="{y + 26}" class="muted">{_esc(flow)}</text>')
    return "\n".join(lines)


def _flow_color(edge: Mapping[str, object], theme: Theme) -> tuple[str, str]:
    flow = _optional_text(edge, "flow", "primary").lower()
    color_key, dash = FLOW_STYLES.get(flow, ("primary", ""))
    return str(getattr(theme, color_key)), dash


def _marker_for_edge(edge: Mapping[str, object]) -> str:
    flow = _optional_text(edge, "flow", "primary").lower()
    color_key, _ = FLOW_STYLES.get(flow, ("primary", ""))
    return {
        "primary": "arrowPrimary",
        "secondary": "arrowSecondary",
        "accent": "arrowAccent",
        "muted": "arrowMuted",
    }.get(color_key, "arrowPrimary")


def _validate_spec(diagram_type: str, spec: Mapping[str, object]) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    nodes = _mapping_list(spec, "nodes", MAX_NODES)
    edges = _mapping_list(spec, "edges", MAX_EDGES)
    node_ids = {_node_id(node, index) for index, node in enumerate(nodes)}
    for node in nodes:
        node_id = _optional_text(node, "id", "(missing id)")
        label = _label(node, "")
        if not label:
            checks.append(_check(False, "node_label", f"node {node_id} label is empty"))
        if len(label) > 42:
            checks.append(
                _check(
                    False,
                    "text_length",
                    f"node {node_id} label may overflow: {_short(label, 24)}",
                )
            )
    for edge in edges:
        source = _optional_text(edge, "source", "")
        target = _optional_text(edge, "target", "")
        if diagram_type != "timeline" and nodes and (source not in node_ids or target not in node_ids):
            checks.append(
                _check(
                    False,
                    "edge_refs",
                    f"edge references unknown node: {source}->{target}; known={', '.join(sorted(node_ids))}",
                )
            )
    if diagram_type == "comparison":
        columns = _string_list(spec, "columns", MAX_COLUMNS)
        for index, row in enumerate(_mapping_list(spec, "rows", MAX_ROWS), start=1):
            values = row.get("values")
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
                if columns and len(values) != len(columns):
                    checks.append(
                        _check(
                            False,
                            "comparison_values",
                            f"comparison row {index} has {len(values)} values for {len(columns)} columns",
                        )
                    )
    if diagram_type == "timeline":
        for index, milestone in enumerate(_mapping_list(spec, "milestones", MAX_MILESTONES), start=1):
            if not _optional_text(milestone, "label", ""):
                checks.append(_check(False, "milestone_label", f"milestone {index} label is empty"))
    if diagram_type == "sequence" and len(nodes) > 8:
        checks.append(_check(False, "sequence_participants", "sequence diagrams support at most 8 participants; split the diagram or use architecture"))
    for key in ("title", "subtitle"):
        value = _optional_text(spec, key, "")
        if value and PLACEHOLDER_PATTERN.search(value):
            checks.append(_check(False, "placeholder", f"{key} contains an unresolved placeholder"))
    if not checks:
        checks.append(_check(True, "spec", "spec checks passed"))
    return checks


def _quality_checks(rendered: str, spec: Mapping[str, object]) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    try:
        ElementTree.fromstring(rendered)
        checks.append(_check(True, "xml", "SVG XML parsed"))
    except ElementTree.ParseError as exc:
        checks.append(_check(False, "xml", f"SVG XML parse failed: {exc}"))
    if PLACEHOLDER_PATTERN.search(rendered):
        checks.append(_check(False, "placeholder", "rendered SVG contains unresolved placeholder text"))
    else:
        checks.append(_check(True, "placeholder", "no unresolved placeholders"))
    if "viewBox=" in rendered:
        checks.append(_check(True, "viewbox", "viewBox present"))
    else:
        checks.append(_check(False, "viewbox", "viewBox missing"))
    return checks


def _check(passed: bool, code: str, message: str) -> dict[str, object]:
    return {"passed": passed, "code": code, "message": message}


def _group_order(
    nodes: Sequence[Mapping[str, object]],
    groups: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    order: list[str] = []
    for group in groups:
        group_id = _optional_text(group, "id", _optional_text(group, "label", "default"))
        if group_id and group_id not in order:
            order.append(group_id)
    for node in nodes:
        group_id = _optional_text(node, "group", "default")
        if group_id not in order:
            order.append(group_id)
    return tuple(order or ("default",))


def _group_label(group_id: str, groups: Sequence[Mapping[str, object]]) -> str:
    for group in groups:
        if _optional_text(group, "id", "") == group_id:
            return _optional_text(group, "label", group_id)
    return group_id if group_id != "default" else "System"


def _participants_from_edges(edges: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    seen: list[str] = []
    for edge in edges:
        for key in ("source", "target"):
            value = _optional_text(edge, key, "")
            if value and value not in seen:
                seen.append(value)
    return tuple({"id": value, "label": value} for value in seen)


def _node_id(node: Mapping[str, object], index: int) -> str:
    return _optional_text(node, "id", f"node_{index + 1}")


def _label(node: Mapping[str, object], default: str) -> str:
    return _optional_text(node, "label", _optional_text(node, "name", default))


def _values(value: object, count: int) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return tuple("" for _ in range(count))
    values = [str(item) for item in value[:count]]
    while len(values) < count:
        values.append("")
    return tuple(values)


def _wrap_text(text: str, max_chars: int, *, max_lines: int) -> tuple[str, ...]:
    words = " ".join(str(text).split()).split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word[:max_chars]
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if not lines:
        return ("",)
    if len(lines) == max_lines and len(" ".join(words)) > len(" ".join(lines)):
        lines[-1] = _short(lines[-1], max_chars)
    return tuple(lines[:max_lines])


def _text_lines_svg(
    lines: Sequence[str],
    x: float,
    y: float,
    css_class: str,
    *,
    anchor: str,
    line_height: int,
) -> str:
    parts = [
        f'  <text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" class="{css_class}">'
    ]
    for index, line in enumerate(lines):
        dy = "0" if index == 0 else str(line_height)
        parts.append(f'    <tspan x="{x:.1f}" dy="{dy}">{_esc(line)}</tspan>')
    parts.append("  </text>")
    return "\n".join(parts)


def _multiline_text(
    text: str,
    x: float,
    y: float,
    max_chars: int,
    css_class: str,
    *,
    anchor: str = "start",
    max_lines: int = 2,
) -> str:
    return _text_lines_svg(
        _wrap_text(text, max_chars, max_lines=max_lines),
        x,
        y,
        css_class,
        anchor=anchor,
        line_height=15,
    )


def _mapping_list(
    arguments: Mapping[str, object],
    key: str,
    max_items: int,
) -> tuple[Mapping[str, object], ...]:
    value = arguments.get(key, ())
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{key} must be an array")
    if len(value) > max_items:
        raise ValueError(f"{key} must contain at most {max_items} items")
    result = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"{key} items must be objects")
        result.append(item)
    return tuple(result)


def _string_list(
    arguments: Mapping[str, object],
    key: str,
    max_items: int,
) -> tuple[str, ...]:
    value = arguments.get(key, ())
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{key} must be an array")
    if len(value) > max_items:
        raise ValueError(f"{key} must contain at most {max_items} items")
    return tuple(str(item).strip() for item in value if str(item).strip())


def _choice(
    arguments: Mapping[str, object],
    key: str,
    default: str,
    choices: Sequence[str],
) -> str:
    value = _optional_text(arguments, key, default).lower().replace("-", "_")
    if value not in choices:
        raise ValueError(f"{key} must be one of: {', '.join(choices)}")
    return value


def _text(arguments: Mapping[str, object], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_text(arguments: Mapping[str, object], key: str, default: str) -> str:
    value = arguments.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value.strip()


def _bool(arguments: Mapping[str, object], key: str, default: bool) -> bool:
    value = arguments.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _short(text: str, limit: int) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def _render_tech_diagram_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "diagram_type": {"type": "string", "enum": list(DIAGRAM_TYPES)},
            "output_path": {
                "type": "string",
                "description": "Workspace-relative .svg output path.",
            },
            "title": {"type": "string"},
            "subtitle": {"type": "string"},
            "theme": {"type": "string", "enum": list(THEMES)},
            "nodes": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Diagram nodes or sequence participants.",
            },
            "edges": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Connections, messages, or flows between nodes.",
            },
            "groups": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Optional layer/group definitions for node diagrams.",
            },
            "columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Comparison matrix column labels.",
            },
            "rows": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Comparison matrix rows with label and values.",
            },
            "milestones": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Timeline milestones with label and optional time.",
            },
            "phases": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Timeline phase cards with label and optional detail.",
            },
            "overwrite": {"type": "boolean"},
        },
        "required": ["diagram_type", "output_path"],
        "additionalProperties": False,
    }
