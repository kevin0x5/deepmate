---
name: research-brief
description: Built-in Work Kit for concise research briefs, competitor notes, and technical option comparisons.
when_to_use: Use when the user asks Deepmate to research a topic, compare products, summarize public sources, or produce a decision-oriented brief.
---
# Research Brief Work Kit

Use this Work Kit to turn scattered sources into a concise brief with evidence, tradeoffs, and a recommendation. Keep the workflow lightweight and evidence-first.

## Workflow

1. Clarify the research question in one sentence. If the user already gave a clear question, do not ask for confirmation.
2. Search with `web_search` when public current information is needed and network is enabled. Fetch only the pages that are likely to matter with `web_fetch`.
3. Read local source documents with `read_document` when the user points to files. Use `search_content` for repo-local evidence.
4. Compare sources by relevance, date, licensing, pricing, maturity, operational fit, and risks. Do not treat search result snippets as final evidence when the linked page can be fetched.
5. Produce a brief that includes:
   - question
   - short answer
   - key findings
   - evidence table with source links or file refs
   - recommendation
   - risks and unknowns
6. When comparison, evolution, or capability shape is central to the brief, load `tech-diagram`: use `comparison` for option matrices, `timeline` for evolution, and `architecture` for technical landscape structure.
7. When the user asks for a shareable artifact, write Markdown first and call `render_html_report` to generate a polished HTML report.
8. Before final delivery, call `review_artifact` on the Markdown source with `require_sources=true` and `require_summary=true`. If HTML was rendered, also review the HTML with `require_sources=true`.
9. Fix actionable review findings once before final delivery. Do not ask the user to inspect routine missing-source, placeholder, caption, or heading warnings.

## Output Bar

- Prefer 3 to 7 findings.
- Separate confirmed facts from inferences.
- Cite source URLs or workspace file refs in the body.
- Include a compact `## Sources` section. Each source should state what it supports, not just list a URL.
- Avoid a long literature review unless the user explicitly asks for depth.
- For competitor tracking, include what changed, why it matters, and what action Deepmate or the project should take next.

## HTML Report Shape

Use a Markdown source with this structure:

```markdown
# Research Brief Title

## Short Answer

## Key Findings

## Evidence

| Source | What it says | Relevance | Confidence |
| --- | --- | --- | --- |

## Recommendation

## Risks and Unknowns

## Sources
```

Then call `render_html_report` with `theme="prussian"` for technical/product work or `theme="forest"` for market/industry briefs.
