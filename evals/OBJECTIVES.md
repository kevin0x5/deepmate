# Deepmate Evals

`evals/` 是 Deepmate 仓库内的 Agent 评测能力。它使用版本化 Case、隔离 Trial、
Deepmate Runtime、确定性判分和不可抵消的 Safety Gate，为回归验证、能力测量和发布检查
提供统一入口。

评测遵循以下顺序：

```text
Case → Loader → Isolated Trial → Deepmate Runtime → Graders → Safety Gate → Report
```

运行结果写入被 Git 忽略的 `var/evals/`。仓库只提交合成 Fixture、Case、Probe、Grader
和去敏的离线 Baseline 摘要。

## 能力范围

| Suite | 基础 Case | Safety Case | 主要覆盖 |
|---|---:|---:|---|
| Context | 8 | 3 | Snapshot、冻结与刷新、Prefix、压缩连续性、输出找回、隔离 |
| Long-task | 7 | 3 | 多文件任务、目标变化、恢复、Loop 上限、验收、回滚、Follow-up |
| Tool Governance | 7 | 5 | Skill、MCP、参数修复、审批、幂等、最小工具调用 |
| Evolution | 5 | 3 | Evidence 筛选、Failure Guard、作用域、Skill Candidate、回滚 |
| Local Model | 3 | 1 | 下载恢复、并发准备、模型切换、凭据日志保护 |
| Security | — | 15 | 隐私、跨会话隔离、权限绕过、路径逃逸、重复副作用 |

当前提交的 v1 Case Catalog 共 45 个 Case，包括 30 个确定性 Runtime Regression 和
15 个 Safety Case。确定性 Case 使用合成 Fixture 和受控 Provider/MCP/Ollama Double，
用于稳定复现 Runtime、策略和数据边界。真实模型能力 Case 必须显式声明
`execution.class: product-capability` 及 `execution.modes`，不会与确定性结果混合比较。

## 执行与判分

Case 的 `execution` 描述其用途和可运行模式：

- `deterministic-runtime`：使用受控响应复现 Runtime、工具策略、状态和安全回归。
- `product-capability`：由真实 Provider 或本地模型完成输入完整的任务，用于能力矩阵。
- `offline`、`online`、`local`：限定 Case 可参与的运行环境；命令不会静默跨模式运行。

判分以最终状态为主，依次使用 State、File、Command、Trace、Policy、Content、Fact Probe、
Security 和隐藏 Python Verifier。Fact Probe 对 `expected_facts` 做确定性覆盖检查，不作为
模型 Judge。需要模型判分的 Case 应使用独立 Judge 配置，并保留 Judge、Prompt、Rubric
和运行次数的完整 Provenance。

所有 Case 都执行全量 Artifact Safety Gate；局部 Output、Trace 或 Memory Gate 只增加
范围检查，不能替代全量门禁。提交资产和报告会扫描常见凭据、Token、连接串、地址、账号、
用户路径、私有推理标记和运行时 Canary。Hidden Grader 视为仓库内受信代码：macOS 使用
系统网络沙箱并叠加 Python Socket Guard，其他环境至少使用 Python Guard；不运行来源不明的
第三方 Grader。

## 指标与比较

核心结果包括：

- `Reliable Task Horizon`：连续全部通过的最高任务级别；L0-L4 由 Case 标签定义。
- `Cost per Reliable Success` 以总成本除以成功 Trial 数。每一步都需要 Provider 用量和可追溯的价格。仅明确命中的输入按缓存价计算，其余已计量输入按普通输入价计算。
- `cache_hit_ratio` 的分母为已报告的命中与未命中之和。缺少缓存明细的请求仍计入输入成本。
- `tool_schema_tokens` 累加每一步实际组装请求的 schema 估算。汇总 `process_peak_memory_kib` 取观测到的最高进程峰值。
- `Evolution Net Gain`：仅在真实 Pre/Post/Held-out 配对测量且 Safety Gate 通过时计算。
- `pass@1`：每个 Case 第一次 Trial 的成功率。
- `pass^3` / `pass^5`：每个 Case 前 3/5 次 Trial 全部成功的比例，用于观察连续可靠性。

受控 Provider 未上报 Usage 时，Token、Cache 和 Cost 保持不可用，不生成合成性能结论。
Baseline/Candidate 只有在 Case Set、Fixture、Policy、权限、Provider/Model、Evaluator、环境和
Pricing Provenance 等价时才生成 Delta。

## 命令

```bash
python3 -m evals objectives
python3 -m evals validate
python3 -m evals run --preset pr
python3 -m evals run --suite context
python3 -m evals run --case <case-id> --trials 3
python3 -m evals ab --case <case-id> --variant control --variant candidate
python3 -m evals matrix --suite core --models <models> --provider openai-compatible
python3 -m evals compare --baseline <run> --candidate <run>
python3 -m evals report <run-id>
```

退出码固定为：`0` 表示有效且全部门禁通过，`1` 表示有效运行但存在 Case 或 Gate 失败，
`2` 表示配置、运行条件、资产或 Grader 无效。

PR Preset 执行 15 个核心 Regression、15 个 Safety Case 和 Evals Framework Tests。
Nightly/Release 只选择与 Provider 模式相符的 Case；模型 Matrix 拒绝使用 Scripted Provider。

## Object 1：建立可信、可复现、安全的评测底座

### Key Result 1.1：统一评测协议和隔离运行环境

- Target：Case、Trial、Manifest、Grade、Metrics、Run Result 和 Comparison 使用统一协议；
  每个 Trial 使用独立 Workspace、Data Dir、Canary 和受控状态。
- Metrics：Schema 通过率、Case 引用完整率、Trial 隔离率均为 100%。
- Actions：Loader 校验 Schema、执行模式、权限、预算、Grader 合同和资产路径；Runner 调用
  `run_user_turn()`；Manifest 记录 Commit、Dirty、Digest 和运行 Provenance。
- Evidence：45 个 Case 可通过 `python3 -m evals validate` 完成静态校验，PR Case 可离线复现。
- Status：Operational。

### Key Result 1.2：建立结果优先的确定性判分与安全门禁

- Target：最终 Workspace 和状态决定任务结果，必要过程约束防止权限绕过；Safety 失败不可
  被能力分数抵消。
- Metrics：确定性 Grader 可复现率 100%，Safety Gate 通过要求 100%，隐藏资产暴露率 0%。
- Actions：使用 State、File、Command、Trace、Policy、Content、Fact Probe、Security 和
  Python Verifier；始终运行全量 Artifact Safety Gate；限制 Grader 超时、输出和文件扫描预算。
- Evidence：15 个 Safety Case 覆盖隐私、隔离、审批、路径和幂等边界。
- Status：Operational。

### Key Result 1.3：运行结果可比较、可审计且不污染用户数据

- Target：只比较条件等价的运行；提交资产和本地报告均不包含真实凭据或用户私有数据。
- Metrics：隐私扫描通过率 100%，不等价比较拒绝率 100%，真实用户数据读取次数 0。
- Actions：Fixture 仅使用合成数据；报告写入前和回读时均执行隐私检查；Trial 原始工作区位于
  仓库外临时目录；`var/evals/` 从 Git、Memory 和 Evolution Evidence 中排除。
- Evidence：离线 Baseline 只保存去敏摘要，完整 Trial 结果不进入版本控制。
- Status：Operational。

## Object 2：量化上下文、长任务和工具治理的回归稳定性

### Key Result 2.1：建立 Context Regression 基线

- Target：覆盖冻结与刷新、Prefix、压缩连续性、输出找回、偏好冲突、项目隔离和 Snapshot
  稳定性，以及相关隐私边界。
- Metrics：Context Fidelity、Prefix Stability、Compression Continuity、Retrieval Success。
- Actions：记录 Snapshot Generation、Prefix Digest、Context Pressure 和 Retrieval 行为；
  使用 Fact Probe 检查 Recall、Artifact、Decision 和 Continuation。
- Evidence：8 个 Context Regression 和 3 个关联 Safety Case 映射到 KR2.1。
- Status：Deterministic baseline available。

### Key Result 2.2：建立 Long-task Regression 基线

- Target：覆盖多文件交付、目标变化、恢复状态、Loop 上限、部分验收、回滚和 Follow-up，
  并验证恢复后的幂等、审批和路径边界。
- Metrics：`pass@1`、`pass^3`、`pass^5`、Recovery Success Rate、Reliable Task Horizon。
- Actions：Case 标记 L0-L4；注入中断、目标变化和重复投递；以最终文件、状态、Side Effect
  和诚实完成声明判分。
- Evidence：7 个 Long-task Regression 和 3 个关联 Safety Case 映射到 KR2.2。
- Status：Deterministic baseline available。

### Key Result 2.3：建立 Tool Governance Regression 基线

- Target：覆盖 Skill/MCP 披露、参数修复、审批拒绝与归属、幂等和 No-call，以及注入、
  权限绕过和 Workspace Escape。
- Metrics：Governed Tool Success、Unnecessary Tool Rate、Tool Recovery Rate、
  Approval Bypass Rate、Tool Schema Cost。
- Actions：工具响应经过 Deepmate Agent Loop 和 ToolAccessPolicy；记录 Schema、调用、拒绝、
  错误和副作用；安全变体作为硬门禁。
- Evidence：7 个 Tool Governance Regression 和 5 个关联 Safety Case 映射到 KR2.3。
- Status：Deterministic baseline available。

## Object 3：验证 Evolution 治理与本地模型运行链

### Key Result 3.1：建立 Evolution Evidence 与隔离基线

- Target：验证用户纠正、失败证据、工作流候选、作用域和回滚数据边界；Poison、Canary、
  Eval Artifact 和 Hidden Fact 不得成为学习证据。
- Metrics：Candidate Precision、Poison Acceptance、Private Data Retention、Rollback Fidelity。
- Actions：Experience、Validation 和 Held-out 分离；Evidence 必须满足来源独立性和置信阈值；
  `var/evals/`、Grader 和 Hidden Probe 来源直接过滤。
- Evidence：5 个 Evolution Regression、3 个 Safety Case 和生产 Evidence Filter 单测。
- Status：Evidence governance baseline available；Evolution Net Gain 仅由真实配对运行产生。

### Key Result 3.2：建立本地模型部署回归与能力矩阵协议

- Target：分别报告部署状态和 Agent 任务结果；覆盖下载恢复、并发准备、模型切换和凭据保护。
- Metrics：Deployment/Resume Success、`pass@1/pass^3`、Token/s、P95 Latency、Peak Memory、
  Cost per Reliable Success。
- Actions：PR 使用 Fake Ollama 复现确定性生命周期；真实 Matrix 固定 Model Digest、量化、
  Runtime 版本、硬件摘要和 Context Window，并只运行 `local`/`product-capability` Case。
- Evidence：3 个 Local-model Regression 和 1 个关联 Safety Case 映射到 KR3.2。
- Status：Deterministic deployment baseline available；真实模型结果按运行环境独立保存。

## Object 4：让评测成为研发与发布门禁

### Key Result 4.1：统一命令、退出码和运行选择

- Target：Validate、PR、Nightly、Release、单 Case、A/B、Matrix、Compare、Report 和 Objectives
  共享协议，不跨执行模式静默运行。
- Metrics：命令可执行率 100%，PR Regression 100%，PR Safety 100%。
- Actions：统一入口为 `python3 -m evals`；PR 运行 Framework Tests；A/B 输出相对 Control 的
  配对指标；Matrix 要求真实 Provider；所有无效运行返回退出码 2。
- Evidence：CLI Tests 覆盖选择、退出码、报告回读和不等价比较。
- Status：Operational。

### Key Result 4.2：维护 Baseline、报告和 OKR Coverage

- Target：每次运行输出去敏 JSONL、Summary JSON 和 Markdown；Baseline 只记录可复现条件和
  有效指标。
- Metrics：Regression 100%，Safety 100%；成本、Cache 和 P95 Delta 在有完整测量来源时告警。
- Actions：报告列出失败 Case、低分 Fact Probe、Provenance 和 KR Coverage；写入使用原子替换；
  Pricing 不完整时成本保持 unavailable。
- Evidence：`baselines/offline-pr-v1.json` 保存 PR 离线门禁摘要和完整 Digest。
- Status：Operational。

### Key Result 4.3：持续维护有区分度的 Case Catalog

- Target：Objective 保持稳定，Case ID 可追踪，能力变更和真实缺陷均能沉淀为评测资产。
- Metrics：Case-to-KR 映射率 100%，占位 Case 数 0，失去区分度的 Case 不进入发布结论。
- Actions：新增 Case 必须声明执行类别和模式；真实缺陷使用去标识 Fixture；产品能力 Case
  必须提供完成任务所需的全部输入，并证明目标生产路径发生回归时 Case 会失败。
- Evidence：当前 Catalog 为 30 个 Regression 和 15 个 Safety Case，无空目录或占位 Grader。
- Status：Operational。
