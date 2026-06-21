---
name: tech-diagram
description: Built-in Work Kit for technical diagrams in reports, PRDs, architecture notes, and research briefs.
when_to_use: Use when the user asks for architecture diagrams, flows, sequences, timelines, comparison matrices, or a visual explanation of a technical system.
---
# Tech Diagram Work Kit

Use this Work Kit when a task needs a technical diagram as part of a concrete deliverable. Prefer a structured diagram spec plus `render_tech_diagram` over hand-written SVG.

## Workflow

1. Decide whether a diagram improves the deliverable. Use diagrams for architecture, runtime flow, tool calls, approval paths, user journeys, roadmap phases, and option comparisons.
2. Choose one diagram type:
   - `architecture` for systems, services, modules, data stores, and deployment shape.
   - `agent_architecture` for LLM, memory, tools, skills, MCP, browser, subagents, and runtime loops.
   - `flowchart` for process steps, user flows, approval paths, and decisions.
   - `sequence` for ordered interactions between user, agent, tools, remote channels, providers, and services.
   - `comparison` for option selection, competitor notes, capability matrices, and tradeoffs.
   - `timeline` for roadmaps, phases, milestones, and task evolution.
3. Extract structure from the current task context, local files, research notes, or user-provided material:
   - `nodes`: stable entities with `id`, `label`, `kind`, and optional `group`.
   - `edges`: connections with `source`, `target`, `label`, and `flow`.
   - `groups`: architecture layers or semantic regions.
   - `columns` and `rows`: comparison matrix content.
   - `milestones`: timeline labels and times.
   - `phases`: optional timeline phase cards with `label` and `detail`.
4. Pick a Deepmate theme:
   - `prussian` for technical strategy, architecture, and product reports.
   - `forest` for research, market, learning, and operational work.
   - `graphite` for risk, audit, review, and data-heavy material.
   - `blueprint` for engineering architecture and implementation plans.
5. Call `render_tech_diagram` with a workspace-relative `.svg` output path.
6. Read the static QA result. If it reports unknown edge refs, empty labels, long text, placeholders, or XML issues, fix the structured spec and call the tool again.
7. If the final deliverable is a report or PRD, reference the generated SVG from the Markdown source before calling `render_html_report`. Add an image title so it becomes a caption, for example `![Runtime flow](artifacts/runtime.svg "Runtime flow")`.

## Diagram Language

Use stable semantic `kind` values so diagrams stay consistent:

| Concept | Suggested `kind` |
| --- | --- |
| User or human actor | `user` |
| Agent or orchestrator | `agent` |
| LLM or model runtime | `model` |
| Tool or function | `tool` |
| Skill or Work Kit | `skill` |
| MCP server or external API | `api` |
| Browser or UI | `browser` |
| Working memory | `memory` |
| Persistent store, vector DB, graph DB | `database` / `vector_store` |
| Document or artifact | `document` |
| Queue, event bus, async channel | `queue` |
| Decision point | `decision` |
| Process step | `process` |

Use semantic `flow` values:

| Flow | Use for |
| --- | --- |
| `primary` / `data` | Main request, response, or data path |
| `control` / `trigger` | One component triggering another |
| `read` | Retrieval from memory, file, database, or web source |
| `write` | Store, update, file write, checkpoint, or archive |
| `async` / `event` | Non-blocking event, queue, notification, or remote callback |
| `transform` | Summarization, embedding, rendering, extraction, conversion |
| `feedback` | Reflection, retry, repair, review, or loopback |

## Spec Patterns

Architecture or agent architecture:

```json
{
  "diagram_type": "agent_architecture",
  "output_path": "artifacts/deepmate-agent-loop.svg",
  "title": "Deepmate Agent Loop",
  "theme": "prussian",
  "groups": [
    {"id": "input", "label": "Input"},
    {"id": "runtime", "label": "Runtime"},
    {"id": "capabilities", "label": "Capabilities"},
    {"id": "output", "label": "Output"}
  ],
  "nodes": [
    {"id": "user", "label": "User", "kind": "user", "group": "input"},
    {"id": "agent", "label": "Agent Runtime", "kind": "agent", "group": "runtime"},
    {"id": "tools", "label": "Native Tools", "kind": "tool", "group": "capabilities"},
    {"id": "report", "label": "Report", "kind": "document", "group": "output"}
  ],
  "edges": [
    {"source": "user", "target": "agent", "label": "task", "flow": "primary"},
    {"source": "agent", "target": "tools", "label": "calls", "flow": "control"},
    {"source": "tools", "target": "report", "label": "renders", "flow": "transform"}
  ]
}
```

Comparison:

```json
{
  "diagram_type": "comparison",
  "output_path": "artifacts/options.svg",
  "title": "Option Comparison",
  "theme": "graphite",
  "columns": ["Option A", "Option B", "Option C"],
  "rows": [
    {"label": "Cost", "values": ["Low", "Medium", "High"]},
    {"label": "Risk", "values": ["Low", "Low", "Medium"]}
  ]
}
```

Timeline:

```json
{
  "diagram_type": "timeline",
  "output_path": "artifacts/roadmap.svg",
  "title": "Roadmap",
  "theme": "blueprint",
  "milestones": [
    {"label": "Design", "time": "Week 1"},
    {"label": "Build", "time": "Week 2"},
    {"label": "Review", "time": "Week 3"}
  ],
  "phases": [
    {"label": "Foundation", "detail": "Design and build"},
    {"label": "Closure", "detail": "Review and ship"}
  ]
}
```

## Quality Bar

- Keep node labels short. Put explanation in surrounding report text, not inside the diagram.
- Use stable ids such as `agent_runtime`, `tool_registry`, or `remote_channel`; avoid spaces in ids.
- Use groups to express layers instead of placing every node in one row.
- Every edge must reference an existing node id.
- Use at least two different `flow` values only when the distinction adds meaning; otherwise keep the diagram calm.
- Do not ask the user to provide diagram JSON unless they explicitly want to edit the spec.
- Do not generate large hand-written SVG directly unless `render_tech_diagram` cannot express the required shape.
- Treat diagrams as part of the final work artifact, not as a separate decorative asset.
