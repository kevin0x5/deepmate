<p align="center">
  <a href="#许可证"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="#安装"><img src="https://img.shields.io/badge/python-3.11%2B-green.svg" alt="Python 3.11+"></a>
  <a href="https://api-docs.deepseek.com"><img src="https://img.shields.io/badge/DeepSeek%20V4-optimized-5865F2.svg" alt="DeepSeek V4"></a>
</p>

# Deepmate

Deepmate 是一个成本优先的本地 AI 工作台，面向长任务项目工作，支持 Task Mode、上下文管理、本地模型部署，以及技能、MCP、工具治理，并针对 DeepSeek V4 做了专项优化。

## 核心亮点

- **前缀缓存** — 为会话保留稳定的上下文快照，让项目背景、规则和常用能力尽量复用，降低长任务里的重复上下文成本。
- **成本治理** — 根据使用情况管理技能和能力的冷热状态；MCP 描述和工具按需加载，大型输出先压缩，需要细节时再展开。
- **本地模型部署** — 直接在 Deepmate 中准备、检查并切换 Ollama 本地模型，适合本地处理的任务可以留在本机完成，减少远程模型调用。
- **自进化** — 从运行记录、会话、失败、纠正和成功流程中提取信号，沉淀为可回滚的行为提示、失败防护、技能草稿和能力状态。

## 安装

要求：Python 3.11+ 和 DeepSeek API key。

```bash
pip install deepmate
export DEEPSEEK_API_KEY="<your-deepseek-api-key>"
deepmate
```

默认情况下，`deepmate` 会打开 TUI，并将文件访问、检查点和工具权限限定在当前工作区。单轮 CLI 检查：

```bash
deepmate --cli "用一段话总结这个项目。"
```

源码开发安装：

```bash
git clone https://github.com/kevin0x5/deepmate.git
cd deepmate
python3 -m pip install -e .
```

## 功能概览

Deepmate 仍在快速开发中。核心 CLI/TUI 工作流本地优先；Ollama、Computer Use、企业微信、预览转发和桌面宠物等可选集成需要单独配置。

| 能力 | 说明 |
|------|------|
| **LLM Runtime** | 默认使用 DeepSeek V4，支持流式输出、按次切换模型，并显式配置 thinking mode、reasoning effort 等模型选项。 |
| **Local Model Deployment** | 在 Deepmate 内完成 Ollama 本地模型的准备、检查和切换，切换后仍可继续使用 TUI、会话和工具能力。 |
| **Context Management** | 把工作区规则、用户/项目记忆、任务状态、技能、MCP 工具、文件引用和历史输出，按当前回合需要装配给模型。 |
| **Context Snapshot、Prefix Cache 与 Cost Control** | 为会话固定工作区和用户上下文，并在需要时刷新；稳定前缀尽量复用，大型输出压缩后可按引用取回。 |
| **Task Mode** | 规划目标、锁定验收契约、按契约执行，并把 `task/plan.md`、`task/evolution.md`、`task/achievements/` 作为任务记录保存。 |
| **Workspace Tools** | 读写、编辑、搜索、diff 文件，检查文档和表格，渲染报告/图表，运行沙箱 shell，做浏览器自动化和 LSP 查询。 |
| **Skills** | 加载可复用的 `SKILL.md` 工作流，例如 PRD、调研简报、HTML 报告、技术图、架构说明和 advisor。 |
| **MCP** | 发现已配置的 MCP server，按需加载工具描述；默认只读执行，涉及写入时需要显式授权。 |
| **Memory** | 从用户输入中提取稳定的个人偏好、协作习惯、常用术语和长期工作原则，并过滤密钥、日志和一次性任务信息。 |
| **Subagents** | 在需要并行或隔离处理时，委派有边界的子任务，并限制工具、输出契约和验证方式。 |
| **Checkpoints** | 保存对话记录、运行证据和文件检查点，让工作区变更可检查、可恢复，并支持回退到历史检查点。 |
| **Session Tree** | 查看会话历史，并通过 tree、clone、fork 为不同方向创建分支。 |
| **Self-evolution** | 从运行记录、会话、活动摘要、失败和成功流程中提取信号，更新行为提示、失败防护、技能草稿和能力状态。 |
| **Computer Use** | macOS 下显式允许真实屏幕观察和操作，用于 UI 验证或桌面工作流。 |
| **Behavior Learning** | 学习本地工具和 Computer Use 的成功操作链路，把重复出现的操作路径沉淀为可复用 workflow，并进一步生成技能草稿。 |
| **QA 审计** | 生成项目自适应 QA 方案和可编辑用例，执行检查、生成报告，并可把问题转成 Task Mode 修复计划。 |
| **Preview Deploy** | 为生成的 HTML、报告、静态构建目录或已有本地服务创建本地/局域网预览，并可接入外部转发服务。 |
| **TUI Workbench** | 直接运行 `deepmate` 进入本地工作台，管理会话、文件浏览、预览、diff、审批、提示队列、上下文用量和斜杠命令。 |
| **Desktop Pet** | 用小型桌面伙伴展示当前工作状态、轻量进度反馈、提醒和学习建议。 |
| **Cron Jobs** | 通过自然语言或命令创建循环任务，运行前审批，并把输出保存在当前工作区。 |
| **Enterprise WeChat Remote** | 绑定远程会话，支持长任务、审批、状态检查、受控远程命令、进度心跳，并在工作运行期间保持设备不熄屏。 |
| **Hooks** | 在审批、阻断、运行证据、检查点、记忆/自进化信号和可信项目规则上应用生命周期策略。 |
| **Observability** | 查看本地运行证据、会话详情、token/cache 信号，并可导出到兼容 OTLP 的服务端点。 |
| **Security** | 限制工作区写入、写前创建检查点、门控 shell/network/MCP 写能力、脱敏环境变量，并对敏感操作请求审批。 |

## 配置

首次使用通常不需要手写 YAML。Deepmate 第一次进入工作区时会自动生成
`config/deepmate.yaml`、`config/providers.yaml` 和本地用户配置文件，并带有默认值。

唯一必须配置的是环境变量里的模型 key：

```bash
export DEEPSEEK_API_KEY="<your-deepseek-api-key>"
```

不调用模型检查基础安装和可选功能状态：

```bash
deepmate --doctor
```

只有在切换模型服务、调整上下文预算，或覆盖摘要/记忆等内部模型时，才需要编辑
`config/deepmate.yaml` 或 `config/providers.yaml`。

运行时覆盖：

```bash
deepmate --model deepseek-v4-pro --thinking enabled --reasoning-effort max "审查这个设计。"
```

本地模型：

```bash
deepmate          # 启动后输入 /local 准备并切换到 Ollama
```

## 常用命令

```bash
# 只读项目检查
deepmate --read-only-tools "找出主要 runtime 入口并总结。"

# 工作区写入（会先创建检查点）
deepmate --workspace-write "优化 README 的表达。"

# Shell（沙箱执行，默认关闭）
deepmate --shell "运行 TUI 相关的聚焦测试。"

# Task Mode
deepmate --task plan "规划认证模块的下一步修改。"
deepmate --task execute "执行已确认方案。"
deepmate --task checkpoint "把当前阶段归档为阶段成果。"

# QA 审计
deepmate --qa "对这个 Web 应用做一次发布前质量验收。"
deepmate --qa run

# 定时任务
deepmate --cron add "每个工作日 09:00 总结项目状态并写入 reports/daily"
deepmate --cron approve <job-id>
deepmate --cron-runner --cron-watch

# 技能
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

# 桌面宠物
deepmate --pet

# 部署预览
# 在 `deepmate` 中使用：/deploy ./dist

# 可观测性
deepmate --show-session <session-id> --trace-events 50
deepmate --show-session <session-id> --export-otlp --otlp-endpoint https://cloud.langfuse.com/api/public/otel
deepmate --validate-otlp --otlp-endpoint https://cloud.langfuse.com/api/public/otel

# 会话树
# 在 `deepmate` 中使用：/session tree、/tree、/session clone、/session fork

# 不调用模型验证运行时配置
deepmate --validate-runtime --thinking disabled
```

## DeepSeek V4 兼容性

- 模型名称：`deepseek-v4-flash` / `deepseek-v4-pro`（不使用已废弃的 V3 名称）
- 上下文窗口：默认配置 1,000,000 tokens
- 推理能力：支持 `thinking` 和 `reasoning_effort: max`
- 不推荐使用降低 thinking 能力的绕行方案

官方参考：[API 文档](https://api-docs.deepseek.com/) · [Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode) · [定价](https://api-docs.deepseek.com/quick_start/pricing)

## 工作方式

```text
用户入口
  CLI、TUI、企业微信、可选桌面宠物
运行时
  装配有界上下文，调用模型，执行已授权工具，写入检查点，并记录证据
能力层
  原生工具、skills、MCP、browser automation、Computer Use 和 subagents 按需进入运行过程
工作区记录
  task/、qa/、cron/、reports、checkpoints、transcripts 和 traces 让工作可查看、可回溯
可选集成
  Ollama 本地模型、预览转发、桌面宠物、OpenTelemetry / Langfuse 兼容导出
```

## 安全与隐私

- API key 和远程 secret 从环境变量读取，不写入仓库。
- 本地运行状态默认写入 `var/`，被 Git 忽略。
- 工作区写入有路径限制并创建检查点。
- Shell、网络和 MCP 写工具默认关闭，需要显式开启。
- 大型工具输出可压缩并通过 handle 按需取回，不完整回放进上下文。
- Traces 默认写入本地；配置后可导出到 OpenTelemetry OTLP endpoint，例如 Langfuse。

## 许可证

MIT
