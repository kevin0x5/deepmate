---
name: technical-architecture
description: Built-in Work Kit for project architecture explanations, module maps, runtime flows, and technical handoff reports.
when_to_use: Use when the user asks Deepmate to understand a codebase, explain project architecture, map modules, document runtime flow, or prepare a technical overview.
---
# Technical Architecture Work Kit

Use this Work Kit to turn a repository or technical folder into a concise architecture explanation that a teammate can read and act on. Keep it evidence-backed and delivery-oriented.

## Workflow

1. Start from the user's target scope. If no scope is given, inspect the workspace root with `search_files` before assuming architecture.
2. Read orientation files first when present: `README*`, `AGENTS.md`, `pyproject.toml`, package manifests, config files, and existing design docs the user points to.
3. Use `search_files` and `search_content` to identify entrypoints, runtime loops, tool registries, channels, storage, tests, and extension points.
4. Build a compact source ledger while reading. For every important claim, keep the supporting file path or URL and what it supports.
5. Produce Markdown with:
   - architecture summary
   - module responsibilities
   - runtime or request flow
   - data/state boundaries
   - extension points
   - risks or technical debt
   - sources
6. When a visual helps, load `tech-diagram` and call `render_tech_diagram`:
   - `architecture` for module boundaries.
   - `agent_architecture` for LLM/tool/memory/runtime loops.
   - `sequence` for request execution flow.
   - `timeline` only when explaining project evolution or migration steps.
7. If the user wants a shareable artifact, load `html-report`, reference generated SVG diagrams with captions, and call `render_html_report`.
8. Before final delivery, call `review_artifact` on the Markdown source with `require_sources=true`, `require_summary=true`, and `require_diagram_captions=true` if diagrams are present. If HTML was rendered, review it too with `require_sources=true`.
9. Fix actionable review findings once before final delivery. Do not ask the user to confirm routine source, placeholder, caption, or heading fixes.

## Output Shape

```markdown
# Project Architecture Overview

## Summary

## Module Map

| Area | Responsibility | Key Files |
| --- | --- | --- |

## Runtime Flow

## Data and State Boundaries

## Extension Points

## Risks and Technical Debt

## Sources
```

## Quality Bar

- Prefer direct file references over broad guesses.
- Separate confirmed implementation facts from inferred design intent.
- Keep diagrams structural, not decorative.
- Do not read the entire repository when a smaller scope answers the question.
- Do not invent technologies that are not visible in files or user context.
- Keep the Sources section compact: file path or URL, what it supports, and why it matters.
