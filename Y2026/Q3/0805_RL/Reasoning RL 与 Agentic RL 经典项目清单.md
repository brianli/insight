---
title: Reasoning RL 与 Agentic RL 经典项目清单
created: 2026-08-06
tags:
  - rl
  - reasoning_rl
  - agentic_rl
  - llm
  - agents
  - github
related:
  - "[[Agentic RL：总体范式、主流模型与方法对比]]"
  - "[[Agentic RL 涌现 — 参考资料与学习路径]]"
  - "[[Agentic RL Survey 论文精读笔记]]"
  - "[[verl 的 R2、R3 到底重放什么？一文讲透 MoE 路由一致性]]"
---

# Reasoning RL 与 Agentic RL 经典项目清单

> [!note]
> 这里的“经典”不是简单按仓库年龄排序，而是指对理解算法、复现实验或搭建训练系统具有代表性和学习价值的项目。需要把“论文/算法复现”“训练基础设施”“环境与评测基准”分开看，不要把 LangChain、AutoGen 这类 Agent 编排框架混进来——它们不是 Agentic RL 项目。

## 一、Reasoning RL：优先看这些

### 1. `deepseek-ai/DeepSeek-Math`

[项目仓库](https://github.com/deepseek-ai/DeepSeek-Math)

Reasoning RL 的关键起点之一，核心贡献是：

- 提出 `GRPO`
- 在数学推理上使用 outcome-based reward
- 不依赖传统 PPO 的独立 critic/value model
- 展示小模型通过 RL 获得更强的数学推理能力

阅读重点：

```text
DeepSeekMath-RL
GRPO
reward normalization
group-relative advantage
math verifier
```

如果只想真正理解 `GRPO`，从这个项目和对应论文开始，不要先读一堆包装框架。

### 2. `deepseek-ai/DeepSeek-R1`

[项目仓库](https://github.com/deepseek-ai/DeepSeek-R1)

Reasoning RL 的标志性项目。

核心路径：

```text
R1-Zero:
Base Model
→ large-scale RL
→ spontaneous CoT / self-verification / reflection

R1:
cold-start SFT
→ reasoning RL
→ rejection sampling / SFT
→ further RL
```

它回答了一个关键问题：

> 不先做 SFT，是否可以仅靠可验证奖励让 base model 自己发展出复杂推理行为？

`R1-Zero` 的答案是可以，但会出现重复、格式差、语言混杂等问题；`R1` 则是工程上更完整的多阶段方案。

这是**必读论文 + 必看仓库**，但不是最适合第一次复现的项目。

### 3. `huggingface/open-r1`

[项目仓库](https://github.com/huggingface/open-r1)

目前最适合学习 DeepSeek-R1 训练路线的开源复现项目之一。

它分阶段复现：

1. R1 distillation
2. R1-Zero 风格纯 RL
3. 多阶段 reasoning post-training
4. 数学、代码、科学推理数据生成与评测

适合用来理解：

- 数据格式
- `GRPOTrainer`
- verifier / reward function
- distillation 与 RL 的边界
- 训练配置和评测流程

**学习价值高于直接读 DeepSeek-R1 仓库。**

### 4. `Jiayi-Pan/TinyZero`

[项目仓库](https://github.com/Jiayi-Pan/TinyZero)

这是最值得动手复现的 `R1-Zero` 极简版本。

特点：

- 基于 `veRL`
- 在 Countdown、乘法等简单任务上训练
- 小模型即可观察到 self-verification 和 search 行为
- 代码短，实验闭环清晰

注意：仓库已经明确提示它不再积极维护，推荐迁移到最新版 `veRL`。但作为**教学和理解 R1-Zero 机制的最小实验**，仍然非常有价值。

推荐顺序：

```text
TinyZero
→ veRL GRPO example
→ Open-R1
→ DeepSeek-R1
```

### 5. `Open-Reasoner-Zero/Open-Reasoner-Zero`

[项目仓库](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero)

比 TinyZero 更接近大规模 Reasoner-Zero 训练的完整复现。

特点：

- 开源训练数据
- 开源训练脚本
- 支持 `0.5B / 1.5B / 7B / 32B`
- 同时提供 PPO、Ray、vLLM 等训练基础设施
- 目标是用更少训练步数复现强 reasoning RL

适合研究：

- 大规模 rollout
- PPO 与 GRPO 的差异
- reasoning RL 的训练稳定性
- response length、reward、exploration 的关系

如果 TinyZero 是“机制演示”，Open-Reasoner-Zero 是“较完整的实验系统”。

### 6. `BytedTsinghua-SIA/DAPO`

[项目仓库](https://github.com/BytedTsinghua-SIA/DAPO)

`DAPO` 是 Reasoning RL 中非常重要的后续算法项目。

核心方向：

- Decoupled Clip
- Dynamic Sampling
- Token-level Policy Gradient Loss
- Overlong reward shaping
- 更稳定、更高效地训练长 CoT

项目同时开源：

- 算法
- 数据集 `DAPO-Math-17k`
- 训练脚本
- 训练记录
- 模型 checkpoint

它不是最早的 reasoning RL 项目，但对理解 **GRPO 之后如何处理长链推理、长度增长和训练坍塌** 很重要。

### 7. `OpenRLHF/OpenRLHF`

[项目仓库](https://github.com/OpenRLHF/OpenRLHF)

通用、高性能的 LLM RL 训练框架，支持：

- PPO
- GRPO
- DAPO
- REINFORCE++
- VLM RL
- multi-turn tool use
- vLLM
- Ray
- async RL

它不是某一个单独的 reasoning 方法，而是更像一个可扩展训练底座。

适合比较：

```text
PPO
GRPO
DAPO
REINFORCE++
VLM RL
Agentic RL
```

## 二、Agentic RL：核心项目

Agentic RL 与 Reasoning RL 的区别是：

```text
Reasoning RL:
一个问题 → 模型生成完整答案 → verifier 给 reward

Agentic RL:
观察环境
→ 思考
→ 调用工具 / 执行动作
→ 获得环境反馈
→ 再思考
→ 多轮交互
→ 完成任务
```

因此真正的 Agentic RL 难点不是单纯换一个 loss，而是：

- multi-turn trajectory
- environment state
- tool feedback
- sparse reward
- credit assignment
- rollout 并发
- context 管理
- 长程探索
- agent failure diagnosis

### 1. `mll-lab-nu/RAGEN`

[项目仓库](https://github.com/mll-lab-nu/RAGEN)

这是 Agentic RL 研究里非常值得优先读的项目。

核心概念：

- `RAGEN = Reasoning Agent`
- `StarPO`
- multi-turn trajectory-level RL
- Gym-compatible environment
- agent failure diagnosis
- reward variance filtering

内置环境包括：

- Sokoban
- FrozenLake
- WebShop
- DeepCoder
- SearchQA
- Lean
- Bandit
- Countdown
- Sudoku

它的价值不只是“训练一个 agent”，而是试图回答：

> 为什么 Agent RL 会 collapse？为什么模型会学成模板化行为？如何诊断训练失败？

适合研究：

```text
trajectory-level optimization
turn-wise vs trajectory-wise update
reasoning-action-reward interaction
agentic RL collapse
reward variance
```

我会把它列为 Agentic RL 的**研究型经典项目**。

### 2. `WooooDyy/AgentGym`

[项目仓库](https://github.com/WooooDyy/AgentGym)

Agentic RL 的环境与数据基础设施。

包含多类交互环境：

- WebShop
- WebArena
- ALFWorld
- SciWorld
- BabyAI
- TextCraft
- BIRD
- 工具调用任务
- 编程任务

同时提供：

- AgentTraj
- AgentEval
- 多环境统一接口
- agent trajectory 数据
- AgentEvol 训练方法

它更偏向：

```text
环境统一
数据构建
跨环境泛化
Agent trajectory
Agent evaluation
```

如果研究“通用 Agent 能不能通过经验跨任务成长”，AgentGym 是重要项目。

### 3. `WooooDyy/AgentGym-RL`

[项目仓库](https://github.com/WooooDyy/AgentGym-RL)

这是 AgentGym 的 RL 版本，专门针对：

- multi-turn decision making
- long-horizon tasks
- WebArena
- Search-R1 environment
- TextCraft
- BabyAI
- SciWorld

支持：

- PPO
- GRPO
- RLOO
- REINFORCE++

另一个关键点是 `ScalingInter-RL`：

> 逐步增加 agent 与环境的交互长度，而不是一开始就训练极长轨迹。

这是解决 Agentic RL exploration 和训练稳定性的一个实用思路。

### 4. `PeterGriffinJin/Search-R1`

[项目仓库](https://github.com/PeterGriffinJin/Search-R1)

这是目前最经典的**搜索型 Agentic RL** 项目之一。

目标：

```text
reasoning
↔ search tool call
↔ retrieved documents
↔ further reasoning
→ final answer
```

特点：

- 基于 `veRL`
- 支持 PPO、GRPO、REINFORCE
- 支持本地 sparse/dense retriever
- 支持在线搜索
- 让 base model 通过 RL 学会何时搜索、如何搜索
- 支持多轮搜索与推理交错

典型轨迹：

```text
<think>...
<begin_of_search>query<end_of_search>
<begin_of_documents>...</end_of_documents>
<think>...
<answer>...</answer>
```

它是连接两类 RL 的关键项目：

```text
Reasoning RL
+ Tool-use RL
+ Search Agent
```

### 5. `GAIR-NLP/DeepResearcher`

[项目仓库](https://github.com/GAIR-NLP/DeepResearcher)

Search-R1 之后更接近真实 Deep Research 的 Agentic RL 系统。

核心特点：

- 真实 Web search
- end-to-end RL
- 多轮搜索
- 规划
- 多来源交叉验证
- 自我反思
- 失败后调整研究方向
- 允许模型承认信息不足

它的研究意义在于：

> 奖励不是只判断最终答案，还要让模型在真实网页环境中学会研究过程。

适合研究：

- sparse outcome reward
- web environment rollout
- long-horizon research
- search trajectory
- research agent 的 credit assignment

### 6. `AgentR1/Agent-R1`

[项目仓库](https://github.com/AgentR1/Agent-R1)

更系统化的 Agentic RL 框架。

核心抽象：

```text
Step-level MDP
Environment
Observation
Action
Tool feedback
Context management
Reward
Policy update
```

它明确反对把整条多轮对话简单拼成一次长 prompt，而是把每一轮交互建模成 MDP transition。

支持的任务方向包括：

- GSM8K + tool
- HotpotQA
- ALFWorld
- WebShop
- academic paper search

适合研究：

- step-level credit assignment
- multi-turn context management
- tool/environment abstraction
- RLOO、PPO、GRPO 等算法比较

它更像是 Agentic RL 的“系统化架构版本”，不应与 TinyZero 这种最小复现混为一谈。

### 7. `PrimeIntellect-ai/prime-rl`

[项目仓库](https://github.com/PrimeIntellect-ai/prime-rl)

偏大规模、异步、生产级 Agentic RL。

核心特点：

- fully asynchronous RL
- 支持大规模 MoE
- FSDP2
- vLLM
- 多环境并行
- SWE 与 agentic environments
- tool calling
- Wiki Search
- Wordle
- coding / terminal agents

适合研究：

```text
async RL
off-policy effects
distributed rollout
large-scale agent training
sandboxed environments
```

不适合作为第一项目。它解决的是“怎么把 Agentic RL 跑到大规模”，不是“Agentic RL 为什么有效”。

### 8. `rllm-org/rllm`

[项目仓库](https://github.com/rllm-org/rllm)

很适合做 Agentic RL 实验的统一 harness。

抽象比较清楚：

```text
Agent
→ Traces
→ Rewards
→ RL Update
```

支持：

- 多种 agent harness
- Docker / Daytona / Modal sandbox
- `verl`
- Tinker
- Fireworks backend
- GRPO
- REINFORCE
- RLOO
- SWE-bench
- Terminal-Bench
- search agent
- coding agent

它的独特价值是：

> 同一份 Agent 代码可以用于 eval 和 RL training，不需要为训练重写一套 agent。

这对工程实验很实用，但从“经典性”看，它更像新一代基础设施，而不是奠基项目。

### 9. `microsoft/agent-lightning`

[项目仓库](https://github.com/microsoft/agent-lightning)

核心卖点：

- 几乎不改 Agent 代码
- 支持 LangChain、OpenAI Agent SDK、AutoGen、CrewAI 等
- 通过 tracing 收集 prompt、tool call、reward
- 支持多 Agent 系统中选择性优化部分 Agent

适合研究：

```text
agent tracing
trajectory collection
agent framework integration
multi-agent credit assignment
```

它解决的是“如何把已有 Agent 接入 RL”，不是“如何设计一个新的 Agentic RL 算法”。

### 10. `OpenPipe/ART`

[项目仓库](https://github.com/OpenPipe/ART)

偏易用和工程实践的 Agent RL 框架：

- 基于 GRPO
- multi-step agent
- Python 应用接入
- Trajectory 记录
- 自定义环境和 reward
- 支持 LangGraph
- 支持 MCP tool training

适合入门的例子：

- 2048
- Tic Tac Toe
- Codenames
- email search
- MCP server
- Temporal Clue

如果想快速写一个“自己的 Agent + environment + reward + GRPO”，ART 比 `verl` 更容易上手。

## 三、环境与基准

训练框架不是环境。需要分别理解两者。

### Web 交互

#### `ServiceNow/BrowserGym`

[项目仓库](https://github.com/ServiceNow/BrowserGym)

统一 Web Agent 环境与 benchmark 框架，支持：

- WebArena
- WorkArena
- WebLINX
- 自定义浏览器任务

适合作为 Web Agent RL 的环境层。

#### `web-arena-x/webarena`

[项目仓库](https://github.com/web-arena-x/webarena)

经典的可自托管真实 Web 环境，包含：

- 在线购物
- 论坛
- GitHub 类协作开发
- 内容管理

适合测量多步 Web navigation 和真实工具交互。

#### `princeton-nlp/WebShop`

[项目仓库](https://github.com/princeton-nlp/WebShop)

较早、较容易上手的 Web Agent 环境：

- 约 120 万商品
- 12,087 条自然语言购物指令
- compositional instruction
- query reformulation
- strategic exploration

适合第一个 Agentic RL 环境，因为比 WebArena 容易跑通。

### 工具调用

#### `OpenBMB/ToolBench`

[项目仓库](https://github.com/OpenBMB/ToolBench)

工具学习领域的经典项目：

- 数千个真实 API
- single-tool / multi-tool
- API retrieval
- ToolLLaMA
- tool execution traces
- DFSDT 数据生成

但要注意：它主要是**工具使用数据与 SFT/评测平台**，不是纯 Agentic RL 项目。可以作为 Agentic RL 的环境和数据来源。

### 真实用户—工具—Agent 交互

#### `sierra-research/tau-bench`

[项目仓库](https://github.com/sierra-research/tau-bench)

评测：

```text
用户模拟器
↔ Agent
↔ 领域工具/API
↔ policy constraints
```

适合研究：

- tool-agent-user interaction
- policy following
- multi-turn API execution
- transaction correctness
- stateful environment

如果研究 agentic tool use，只测单轮 function calling 是不够的；`τ-bench` 更接近真实任务。

### 电脑操作

#### `xlang-ai/OSWorld`

[项目仓库](https://github.com/xlang-ai/OSWorld)

真实桌面环境中的长程任务：

- 浏览器
- 文件管理
- Office
- 终端
- 多应用协同

适合 computer-use agent 的 RL / evaluation，但运行成本高，环境搭建也更复杂。

### 终端与编码

#### `harbor-framework/terminal-bench`

[项目仓库](https://github.com/harbor-framework/terminal-bench)

真实 terminal 环境中的复杂任务：

- 编译代码
- 配置服务
- 训练模型
- 系统调试
- 端到端工程任务

它非常适合 Agentic RL，因为 reward 可以来自：

```text
test script
unit test
build result
service health check
artifact validation
```

#### `SWE-bench/SWE-bench`

[项目仓库](https://github.com/SWE-bench/SWE-bench)

真实 GitHub issue 修复 benchmark。它本身不是 RL 训练框架，但非常适合作为：

- coding agent evaluation
- terminal sandbox
- patch-level reward
- long-horizon software engineering task

### 具身/文本环境

#### `alfworld/alfworld`

[项目仓库](https://github.com/alfworld/alfworld)

将文本环境与 embodied environment 对齐，适合：

- 高层规划
- 长程决策
- 文本动作
- 具身任务迁移

#### `StonyBrookNLP/appworld`

[项目仓库](https://github.com/StonyBrookNLP/appworld)

模拟多个日常应用与用户世界：

- 457 个 API
- 9 类日常应用
- 多人、多状态、可执行任务
- function calling
- interactive coding agent

比单纯的静态工具调用更适合研究 stateful agent。

## 四、分层推荐

### A. 想理解 Reasoning RL 原理

```text
1. DeepSeekMath
2. TinyZero
3. DeepSeek-R1
4. Open-R1
5. Open-Reasoner-Zero
6. DAPO
7. veRL
```

重点不是把所有仓库跑一遍，而是逐步回答：

```text
reward 从哪里来？
为什么 group-relative advantage 有效？
为什么不需要 critic？
模型如何产生更长 CoT？
什么时候会 reward hacking？
为什么 response length 会增长？
如何避免 reasoning collapse？
```

### B. 想理解 Agentic RL

```text
1. RAGEN
2. WebShop
3. Search-R1
4. AgentGym
5. AgentGym-RL
6. DeepResearcher
7. Agent-R1
8. rLLM / Agent Lightning
9. prime-rl
```

推荐第一批只跑：

```text
WebShop
Search-R1
RAGEN
```

这三个项目分别对应：

```text
简单交互环境
工具增强推理
多轮 Agent 训练与诊断
```

### C. 想搭建自己的 Agentic RL 实验

推荐栈：

```text
训练底座:
verl

环境:
WebShop / AppWorld / Terminal-Bench

Agent harness:
rLLM 或 Agent Lightning

算法:
GRPO → REINFORCE++ → DAPO / ARPO

观测:
trajectory logs
reward distribution
episode length
tool-call entropy
success rate
failure mode
```

不要一上来就用 OSWorld 或真实 Web。那是环境工程，不是算法研究。先用 WebShop 或一个自定义 Gym 环境跑通闭环：

```text
reset
→ observation
→ model action
→ environment feedback
→ trajectory
→ reward
→ policy update
→ evaluation
```

## 五、最值得优先 Clone 的 10 个仓库

| 优先级 | 项目 | 用途 |
|---:|---|---|
| 1 | `Jiayi-Pan/TinyZero` | 最小 R1-Zero 复现 |
| 2 | `deepseek-ai/DeepSeek-Math` | 理解 GRPO |
| 3 | `huggingface/open-r1` | 完整 reasoning RL pipeline |
| 4 | `verl-project/verl` | 主流 RL 训练底座 |
| 5 | `mll-lab-nu/RAGEN` | Agentic RL 与失败诊断 |
| 6 | `PeterGriffinJin/Search-R1` | 搜索工具 RL |
| 7 | `WooooDyy/AgentGym-RL` | 多环境、多轮 Agent RL |
| 8 | `GAIR-NLP/DeepResearcher` | 真实 Web deep research RL |
| 9 | `AgentR1/Agent-R1` | step-level Agentic RL 架构 |
| 10 | `rllm-org/rllm` | 通用 Agent harness 与训练接入 |

## 结论

- **Reasoning RL 的奠基链条：** `DeepSeekMath → DeepSeek-R1 → Open-R1 → DAPO`
- **Agentic RL 的主线：** `RAGEN → Search-R1 → AgentGym-RL → DeepResearcher → Agent-R1`
- **工程底座：** `veRL / OpenRLHF / SkyRL / rLLM / Agent Lightning`
- **环境与评测：** `WebShop / BrowserGym / WebArena / AppWorld / OSWorld / Terminal-Bench`

真正值得先做的不是收藏仓库，而是用 `TinyZero` 复现一次纯 reasoning RL，再用 `WebShop + veRL` 把单轮 reward 改成多轮环境 reward。后一步才算真正进入 Agentic RL。
