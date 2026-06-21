<p align="center">
  <a href="#许可证"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="#安装"><img src="https://img.shields.io/badge/python-3.11%2B-green.svg" alt="Python 3.11+"></a>
  <a href="https://api-docs.deepseek.com"><img src="https://img.shields.io/badge/DeepSeek%20V4-optimized-5865F2.svg" alt="DeepSeek V4"></a>
</p>

# Deepmate

Deepmate 是一个成本优先、可自进化的本地 AI 工作台，支持 Task Mode、context management、本地模型部署、skills/MCP/tools 治理等能力，并针对 DeepSeek V4 做了专项优化。

## 核心亮点

- **前缀缓存优化** — 保持长会话背景稳定，让 Deepmate 更容易复用上下文，减少重复发送同一批项目规则和说明的成本。
- **Context 管理** — 按当前步骤加载相关的 memory、skills、MCP tools、文件引用和命令输出；大型输出会压缩，并可按需取回。
- **Task Mode** — 先讨论并确认计划，再按验收契约持续执行；`task/plan.md`、`task/evolution.md` 和 `task/achievements/` 会保存任务过程。
- **Computer Use** — 显式开启后，Deepmate 可以观察和操作真实界面，用来检查视觉布局、焦点、键盘流、桌面应用和其他单元测试覆盖不到的体验问题。
- **一键预览** — 对生成的报告、静态页面或本地服务创建本地/局域网 preview link；配置 preview relay 后可生成临时外部链接。
- **本地模型支持** — 在 Deepmate 内准备、检查并切换 Ollama 本地模型；没有安装 Ollama 不影响正常使用 DeepSeek。
- **桌面宠物** — 可选 Electron companion，会跟随当前工作状态；只有开启后才学习，并用轻量进度、提醒或反馈辅助用户。

## 安装

要求：Python 3.11+ 和 DeepSeek API key。

```bash
git clone https://github.com/kevin0x5/deepmate.git
cd deepmate
python3 -m pip install -e .
export DEEPSEEK_API_KEY="<your-deepseek-api-key>"
```

## 首次运行

```bash
cd /path/to/your/project
deepmate --interactive
```

TUI 会将文件访问、checkpoint 和工具权限绑定到当前工作区。单轮检查：

```bash
deepmate "用一段话总结这个项目。"
```

## 功能概览

| 能力 | 说明 |
|------|------|
| **LLM 与本地模型** | 默认模型后端为 DeepSeek V4；轻量内部任务可使用更便宜的模型；支持 streaming、provider 重试提示，以及 Ollama 本地模型准备和切换。 |
| **Prefix Cache 与成本控制** | 稳定可复用的上下文、cache 相关运行状态、按需触发的 session summary，以及受控加载的 skills、MCP schema 和大型工具输出。 |
| **Context Management** | 根据当前步骤装配项目规则、memory、task 状态、skills、MCP tools、文件引用和历史输出，避免无关材料占用上下文。 |
| **Task Mode** | `task/plan` 定义目标、验收契约、范围、计划和验证策略；`task/execute` 按契约持续执行，直到达成、阻塞或预算受限。 |
| **项目记录** | 在工作区保存任务产物：`task/plan.md` 是当前契约，`task/evolution.md` 记录关键决策和方向变化，`task/achievements/` 记录阶段成果。 |
| **QA Audit** | 用 `/qa <目标>` 生成项目自适应测试方案和可编辑用例；用 `/qa run` 执行检查、保存 evidence、生成 `report.html`，并可把问题转成 Task Mode 修复计划。 |
| **Tools** | Workspace 文件读写、编辑、搜索、diff，文档和表格检查，HTML 报告与 SVG 图渲染，沙箱 shell，browser automation 和 LSP 支持。 |
| **Computer Use** | macOS 下显式开启后支持真实屏幕观察和操作，包括 snapshot、screenshot、click、type、key、open、wait，主要用于 UI 验证和用户体验测试。 |
| **Skills** | 标准 `SKILL.md` 技能包；内置 PRD、research brief、HTML report、tech diagram、technical architecture 和 advisor 类能力；支持从本地、GitHub、压缩包或 URL 安装。 |
| **MCP** | 已配置 MCP servers 的发现、按需 schema 加载、默认只读执行，以及写能力显式开启。 |
| **Subagents** | 有边界的子任务委派，包含工具白名单、输出约定和结果验证；多 Agent 不作为默认执行路径。 |
| **Checkpoint & Sessions** | Session、transcript、trace 和文件 checkpoint 持久化；支持历史查看、checkpoint rewind，以及 session tree、clone、fork。 |
| **Memory 与行为学习** | 用户级 memory 与项目级 memory 分离；长期记忆通过 checkpoint 和 maintenance 更新；明确协作偏好可学习，但不会把猜测当事实。 |
| **Self-evolution** | 基于重复纠正、失败模式和成功流程的低风险本地改进，例如 behavior hints 或 generated skills；变更有记录并可 rollback。 |
| **Cron Jobs** | 自然语言或 `/cron` 创建的循环定时任务、`cron/jobs.jsonl` 中可编辑的定义、运行前审批，以及写回工作区的输出。 |
| **Preview Deploy** | 生成 HTML、报告、静态构建目录或已有本地服务的本地/局域网 preview，并管理状态、TTL、停止和替换。 |
| **企业微信** | 提供远程通道，支持长任务状态、审批、进度心跳和受控远程命令。 |
| **Desktop Pet** | 可选桌面 companion，展示当前工作状态、简短进度和关怀提示；开启学习后可给出轻量建议。 |
| **Hooks** | 受治理的 lifecycle policies，用于审批、阻断、trace、checkpoint、memory/evolution signals 和项目规则；项目 hooks 需要 workspace trust。 |
| **Security** | Workspace 内路径保护、写入前 checkpoint、shell/network/MCP 写权限门控、环境变量脱敏和敏感操作审批。 |
| **Observability** | 本地 trace evidence、session trace 查看、OTLP-compatible span export，以及 OpenTelemetry 或 Langfuse-compatible endpoint 验证。 |
| **TUI Workbench** | Textual 界面包含 sessions、文件树、预览、diff、审批、prompt queue、context 用量、slash commands，以及 skill/MCP/task 状态。 |

## 配置

首次使用通常不需要手写 YAML。Deepmate 第一次进入工作区时会自动生成
`config/deepmate.yaml`、`config/providers.yaml` 和本地 profile 文件，并带有默认值。

唯一必须配置的是模型 key：

```bash
deepmate --setup-key
# 或
export DEEPSEEK_API_KEY="<your-deepseek-api-key>"
```

不调用模型检查基础安装和可选功能状态：

```bash
deepmate --doctor
```

只有在切换 provider、调整上下文预算，或覆盖 summary/memory 等内部模型时，才需要编辑
`config/deepmate.yaml` 或 `config/providers.yaml`。

运行时覆盖：

```bash
deepmate --model deepseek-v4-pro --thinking enabled --reasoning-effort max "审查这个设计。"
```

本地模型：

```bash
deepmate --interactive          # 启动后输入 /local 准备并切换到 Ollama
```

## 常用命令

```bash
# 只读项目检查
deepmate --read-only-tools "找出主要 runtime 入口并总结。"

# Workspace 写入（有 checkpoint）
deepmate --workspace-write "优化 README 的表达。"

# Shell（沙箱执行，默认关闭）
deepmate --shell "运行 TUI 相关的聚焦测试。"

# Task Mode
deepmate --task plan "规划认证模块的下一步修改。"
deepmate --task execute "执行已确认方案。"
deepmate --task checkpoint "把当前阶段归档为 achievement。"

# QA Audit
deepmate --qa "对这个 Web 应用做一次发布前质量验收。"
deepmate --qa run

# Cron Jobs
deepmate --cron add "每个工作日 09:00 总结项目状态并写入 reports/daily"
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

# 远程（企业微信）
deepmate --remote wecom

# Pet
deepmate --pet

# 部署预览
# 在 `deepmate --interactive` 中使用：/deploy ./dist

# Observability
deepmate --show-session <session-id> --trace-events 50
deepmate --show-session <session-id> --export-otlp --otlp-endpoint https://cloud.langfuse.com/api/public/otel
deepmate --validate-otlp --otlp-endpoint https://cloud.langfuse.com/api/public/otel

# Interactive session lineage
# 在 `deepmate --interactive` 中使用：/session tree、/tree、/session clone、/session fork

# 不调用模型验证 runtime
deepmate --validate-runtime --thinking disabled
```

## DeepSeek V4 兼容性

- 模型名称：`deepseek-v4-flash` / `deepseek-v4-pro`（不使用已废弃的 V3 名称）
- 上下文窗口：默认配置 1,000,000 tokens
- 推理能力：支持 `thinking` 和 `reasoning_effort: max`
- 不推荐降低 thinking 能力的 workaround

官方参考：[API 文档](https://api-docs.deepseek.com/) · [Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode) · [定价](https://api-docs.deepseek.com/quick_start/pricing)

## 工作方式

```text
用户入口
  CLI、TUI、企业微信、可选桌面宠物
Runtime
  装配有界 context，调用模型，执行已授权工具，写入 checkpoint，并记录 evidence
能力层
  Native tools、skills、MCP、browser automation、Computer Use 和 subagents 按需进入运行过程
工作区记录
  task/、qa/、cron/、reports、checkpoints、transcripts 和 traces 让工作可查看、可回溯
可选集成
  Ollama 本地模型、preview relay、Electron pet、OpenTelemetry / Langfuse-compatible export
```

## 安全与隐私

- API key 和远程 secret 从环境变量读取，不写入仓库。
- 本地运行状态默认写入 `var/`，被 Git 忽略。
- Workspace 写入有路径限制并创建 checkpoint。
- Shell、网络和 MCP 写工具默认关闭，需显式 flag 开启。
- 大型工具输出可压缩并通过 handle 按需取回，不完整回放进上下文。
- Traces 默认写入本地，可在配置后 export 到 OpenTelemetry OTLP endpoint，例如 Langfuse。

## 许可证

MIT

第三方依赖声明见 `THIRD_PARTY_NOTICES.md`。
