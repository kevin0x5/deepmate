<p align="center">
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="#install"><img src="https://img.shields.io/badge/python-3.11%2B-green.svg" alt="Python 3.11+"></a>
  <a href="https://api-docs.deepseek.com"><img src="https://img.shields.io/badge/DeepSeek%20V4-optimized-5865F2.svg" alt="DeepSeek V4"></a>
</p>

# Deepmate

Deepmate is a cost-first, self-evolving local AI workbench with Task Mode, context management, local model setup, governed skills/MCP/tools, and DeepSeek V4-specific optimizations.

## Highlights

- **Prefix cache optimization** — Keep long-session background stable so Deepmate can reuse more context and avoid repeatedly paying for the same project instructions.
- **Context management** — Load only the relevant memory, skills, MCP tools, files, and command output for the current step; large outputs are compacted and can be retrieved when needed.
- **Task Mode** — Discuss and lock a plan first, then execute against that plan until the acceptance contract is met; `task/plan.md`, `task/evolution.md`, and `task/achievements/` preserve the work.
- **Computer Use** — When explicitly enabled, Deepmate can observe and operate the real UI to check visual layout, focus, keyboard flow, desktop apps, and other user-experience issues that unit tests miss.
- **One-command previews** — Preview generated reports, static pages, and local services with a local/LAN link; an external temporary link can be used when a preview relay is configured.
- **Local model support** — Prepare, check, and switch to Ollama-backed local models from Deepmate; missing Ollama does not block normal DeepSeek-backed use.
- **Desktop pet** — Optional Electron companion that follows work state, learns only when enabled, and gives lightweight progress, reminders, or feedback without becoming a second agent.

## Install

Requirements: Python 3.11+ and a DeepSeek API key.

```bash
git clone https://github.com/kevin0x5/deepmate.git
cd deepmate
python3 -m pip install -e .
export DEEPSEEK_API_KEY="<your-deepseek-api-key>"
```

## First Run

```bash
cd /path/to/your/project
deepmate --interactive
```

The TUI scopes file access, checkpoints, and tool permissions to the current workspace. For a one-shot check:

```bash
deepmate "Summarize this project in one paragraph."
```

## Features

| Capability | Description |
|------------|-------------|
| **LLM and local models** | DeepSeek V4 as the default model backend, lighter models for internal work, streaming responses, provider-aware retry handling, and optional Ollama local models for offline or lower-cost work. |
| **Prefix cache and cost control** | Stable reusable context across turns, cache-related runtime status, summary checkpoints only when needed, and controlled loading of skills, MCP schemas, and large tool outputs. |
| **Context management** | Project rules, memory, task state, skills, MCP tools, file references, and previous outputs are assembled for the current step without flooding the model with unrelated material. |
| **Task Mode** | `task/plan` defines the goal, acceptance contract, scope, plan, and verification strategy; `task/execute` works until the contract is achieved, blocked, or budget-limited. |
| **Project records** | Workspace task artifacts: `task/plan.md` for the current contract, `task/evolution.md` for key decisions and direction changes, and `task/achievements/` for completed stages. |
| **QA Audit** | `/qa <goal>` creates a project-aware test plan and editable cases; `/qa run` executes available checks, collects evidence, generates `report.html`, and can turn findings into a Task Mode repair plan. |
| **Tools** | Workspace file read/write/edit/search/diff, document and table inspection, HTML report rendering, SVG diagram rendering, sandboxed shell checks, browser automation, and LSP support. |
| **Computer Use** | On macOS, explicitly enable real-screen observation and actions such as snapshot, screenshot, click, type, key, open, and wait, mainly for UI validation and user-experience testing. |
| **Skills** | Standard `SKILL.md` bundles for repeatable work patterns such as PRDs, research briefs, HTML reports, diagrams, architecture notes, and advisor-style reviews; installation from local files, GitHub, archives, or URLs. |
| **MCP** | Configured MCP server discovery, on-demand schema loading, read-only default execution, and explicit opt-in for write-capable MCP tools. |
| **Subagents** | Bounded subtask delegation with tool allowlists, output expectations, and verification, without making multi-agent execution the default path. |
| **Checkpoint & Sessions** | Persistent sessions, transcripts, traces, file checkpoints, checkpoint rewind, session history inspection, and session tree / clone / fork for alternate directions. |
| **Memory and behavior learning** | Separate user-level and project-level memory, checkpoint-driven long-term memory updates, maintenance flows, and explicit collaboration preference learning without treating guesses as facts. |
| **Self-evolution** | Low-risk local improvements from repeated corrections, failures, and successful workflows, including behavior hints and generated skills with logged, rollbackable changes. |
| **Cron Jobs** | Recurring jobs from natural language or `/cron`, editable definitions in `cron/jobs.jsonl`, approval before execution, and workspace-local outputs. |
| **Preview Deploy** | Local/LAN previews for generated HTML, reports, static build directories, or existing local services, with managed status, TTL, stop, and replace lifecycle. |
| **Enterprise WeChat** | Remote channel for long-running work, approvals, status checks, progress heartbeats, and controlled remote commands. |
| **Desktop Pet** | Optional desktop companion for current work state, short progress or care messages, and learning suggestions when enabled. |
| **Hooks** | Governed lifecycle policies for approvals, blocking, tracing, checkpoints, memory/evolution signals, and project-specific rules; project hooks require workspace trust. |
| **Security** | Workspace-scoped writes, checkpoint-before-write behavior, shell/network/MCP write gates, environment sanitization, and approvals for sensitive actions. |
| **Observability** | Local trace evidence, session trace inspection, OTLP-compatible span export, and OpenTelemetry or Langfuse-compatible endpoint validation without sending prompts or file contents by default. |
| **TUI workbench** | Work in a Textual interface with sessions, file tree, previews, diffs, approvals, prompt queue, context usage, slash commands, and skill/MCP/task visibility. |

## Configuration

You normally do not need to edit YAML before first use. Deepmate creates
`config/deepmate.yaml`, `config/providers.yaml`, and local profile files with
safe defaults when it first opens a workspace.

The only required setup is a model key:

```bash
deepmate --setup-key
# or
export DEEPSEEK_API_KEY="<your-deepseek-api-key>"
```

Check the base install and optional feature readiness without a model call:

```bash
deepmate --doctor
```

Edit `config/deepmate.yaml` or `config/providers.yaml` only when you want to
switch providers, tune context budgets, or override the internal summary/memory
models.

Override per run:

```bash
deepmate --model deepseek-v4-pro --thinking enabled --reasoning-effort max "Review this design."
```

Local model:

```bash
deepmate --interactive          # then type /local to prepare and switch to Ollama
```

## Common Commands

```bash
# Read-only inspection
deepmate --read-only-tools "Find the main runtime entry points."

# Workspace writes (checkpointed)
deepmate --workspace-write "Update the README wording."

# Shell (sandboxed, off by default)
deepmate --shell "Run the focused tests for the TUI."

# Task Mode
deepmate --task plan "Plan the next auth module changes."
deepmate --task execute "Implement the approved plan."
deepmate --task checkpoint "Archive the current stage as an achievement."

# QA Audit
deepmate --qa "Run a release-readiness audit for the web app."
deepmate --qa run

# Cron Jobs
deepmate --cron add "Every weekday at 09:00, summarize project status into reports/daily"
deepmate --cron approve <job-id>
deepmate --cron-runner --cron-watch

# Skills
deepmate --list-skills
deepmate --show-skill research-brief

# MCP
deepmate --list-mcp
deepmate --mcp-status

# Hooks
deepmate --hooks-status
deepmate --validate-hooks
deepmate --trust-workspace

# Remote (Enterprise WeChat)
deepmate --remote wecom

# Pet
deepmate --pet

# Deploy preview
# In `deepmate --interactive`: /deploy ./dist

# Observability
deepmate --show-session <session-id> --trace-events 50
deepmate --show-session <session-id> --export-otlp --otlp-endpoint https://cloud.langfuse.com/api/public/otel
deepmate --validate-otlp --otlp-endpoint https://cloud.langfuse.com/api/public/otel

# Interactive session lineage
# In `deepmate --interactive`: /session tree, /tree, /session clone, /session fork

# Validate runtime without a model call
deepmate --validate-runtime --thinking disabled
```

## DeepSeek V4 Compatibility

- Model names: `deepseek-v4-flash` / `deepseek-v4-pro` (not deprecated V3 names)
- Context window: 1,000,000 tokens configured by default
- Reasoning: `thinking` and `reasoning_effort: max` supported
- No degraded-thinking workarounds recommended

Refs: [API docs](https://api-docs.deepseek.com/) · [Thinking mode](https://api-docs.deepseek.com/guides/thinking_mode) · [Pricing](https://api-docs.deepseek.com/quick_start/pricing)

## How It Works

```text
User surfaces
  CLI, TUI, Enterprise WeChat, optional desktop pet
Runtime
  Builds bounded context, calls the model, runs approved tools, checkpoints writes, and records evidence
Capabilities
  Native tools, skills, MCP, browser automation, Computer Use, and subagents are loaded when useful
Workspace records
  task/, qa/, cron/, reports, checkpoints, transcripts, and traces keep work inspectable and recoverable
Optional integrations
  Ollama local models, preview relay, Electron pet, OpenTelemetry / Langfuse-compatible export
```

## Safety and Privacy

- API keys and remote secrets are read from environment variables, never committed.
- Local runtime state is written under `var/` and ignored by Git.
- Workspace writes are path-guarded and checkpointed.
- Shell, network, and MCP write tools are off by default and require explicit flags.
- Tool outputs can be compacted and retrieved by handle rather than replayed in full.
- Traces are written locally by default and can be exported to OpenTelemetry OTLP endpoints such as Langfuse when configured.

## License

MIT

Third-party dependency notices are summarized in `THIRD_PARTY_NOTICES.md`.
