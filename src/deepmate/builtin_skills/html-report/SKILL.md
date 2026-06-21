---
name: html-report
description: Built-in Work Kit for polished Markdown-to-HTML reports, briefs, and shareable work artifacts.
when_to_use: Use when the user asks for a polished report, brief, presentation-style HTML, data story, or shareable document.
---
# HTML Report Work Kit

Use this Work Kit when the deliverable should look polished without building a custom web app. The source of truth is Markdown; the final artifact is a self-contained HTML file rendered by `render_html_report`.

## Workflow

1. Gather or create the report content in Markdown. Keep headings semantic and tables clean.
2. Use `inspect_table` for CSV, TSV, JSON, JSONL, or XLSX inputs before writing analysis.
3. If the report needs an architecture view, process flow, option matrix, or roadmap, load `tech-diagram`, call `render_tech_diagram`, and reference the generated SVG from the Markdown with an image title for the caption.
4. Pick a theme:
   - `prussian` for technical, strategic, and product reports.
   - `forest` for market, learning, and operational reports.
   - `graphite` for risk, audit, and data-heavy reports.
5. Call `render_html_report` with the Markdown source and a workspace-relative HTML output path.
6. Read the tool's static QA warnings and fix the Markdown if needed.
7. Call `review_artifact` on the Markdown source and final HTML. Set `require_sources=true` when the report contains research, competitor claims, project analysis, or external facts; set `require_diagram_captions=true` when images or generated diagrams are present.
8. If `review_artifact` returns warnings or errors that can be fixed from context, update the Markdown or diagram spec once and render again before final delivery.

## Markdown Bar

- One `#` title.
- Use `##` sections for the main narrative.
- Keep tables narrow enough to scan.
- Avoid unresolved placeholders such as TODO, TBD, FIXME, or `<what you need>`.
- Put source links in the Markdown so the HTML remains auditable.
- Reference generated SVG diagrams with captions, for example `![Runtime flow](artifacts/runtime.svg "Runtime flow")`.
- For research or analysis reports, include a `## Sources` section. Keep it short: source title or file path, link/ref, and what it supports.

## Suggested Structure

```markdown
# Report Title

## Executive Summary

## Key Findings

## Evidence

| Item | Finding | Evidence | Implication |
| --- | --- | --- | --- |

## Recommendation

## Next Steps

## Sources
```
