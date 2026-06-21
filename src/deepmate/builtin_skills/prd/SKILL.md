---
name: prd
description: Built-in Work Kit for lightweight product requirement documents and acceptance-oriented specs.
when_to_use: Use when the user asks for a PRD, product spec, requirement analysis, feature proposal, user stories, scope, or acceptance criteria.
---
# PRD Work Kit

Use this Work Kit to create a concise product requirement document that can drive implementation. Keep the document useful for decisions and delivery, not ceremonial.

## Workflow

1. Restate the product goal and the target user. If missing, infer from project context and mark assumptions.
2. Inspect relevant project docs or code with `search_files`, `search_content`, and `read_document` before inventing behavior.
3. Write requirements as testable behavior. Prefer small sections and explicit non-goals over long prose.
4. Include decision changes when the user changed direction during the discussion.
5. When a visual would clarify the requirement, load `tech-diagram`: use `flowchart` for user paths, `sequence` for interactions, and `architecture` for module boundaries.
6. If asked for a shareable artifact, create Markdown and render it with `render_html_report`.
7. Before final delivery, call `review_artifact` on the Markdown source with `require_acceptance_criteria=true`. Also set `require_sources=true` when the PRD uses research, project docs, or code evidence.
8. Fix actionable review findings once before final delivery.

## PRD Structure

```markdown
# Feature Name PRD

## Goal

## Background

## Users and Jobs

## Scope

### In Scope

### Out of Scope

## Requirements

| ID | Requirement | Priority | Acceptance Criteria |
| --- | --- | --- | --- |

## User Stories

## Interaction and UX Notes

## Technical Notes

## Risks and Open Questions

## Rollout and Validation

## Sources
```

## Quality Bar

- Every important requirement should have an ID.
- Acceptance criteria must be observable and tied to the requirement they validate.
- Priorities should be practical: `P0`, `P1`, `P2`, or `later`.
- Do not over-spec internals when the user is still discussing product direction.
- Prefer Markdown as the source of truth and HTML as the presentation layer.
- Use `## Sources` when the PRD relies on user-provided docs, repository files, competitor facts, or research. Each source should say what requirement or decision it supports.
