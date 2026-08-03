---
title: Agentic RL Survey 论文精读笔记
tags: [rl, agentic_rl, survey, emergence]
source: "https://arxiv.org/abs/2509.02547"
created: 2026-07-29
related:
  - "[[K3 规划与工具使用 — 从 Agentic RL 中涌现]]"
  - "[[Agentic RL 涌现 — 参考资料与学习路径]]"
---

## 论文信息

- **标题**: The Landscape of Agentic Reinforcement Learning for LLMs: A Survey
- **作者**: Guibin Zhang, Hejia Geng, Xiaohang Yu 等 (牛津、新加坡国立、AI Lab 等)
- **发表**: TMLR 2026
- **规模**: 500+ 篇文献综述
- **核心论点**: RL 是将 agent 能力从「静态启发式模块」转变为「自适应鲁棒行为」的关键机制

---

## 1. 形式化基础：从 LLM RL 到 Agentic RL

### 范式转变

| | 传统 PBRFT (DPO/RLHF) | Agentic RL |
|---|---|---|
| **MDP 类型** | 退化 MDP: T=1, γ=1 | POMDP: T>1, γ<1 |
| **状态空间** | 单一静态 prompt {s₀} | 动态状态 s_t, 部分可观测 o_t |
| **动作空间** | 纯文本序列 | 文本 + 环境交互动作 |
| **转移函数** | 确定性 → terminal | 动态 P(s_{t+1} \| s_t, a_t) |
| **奖励** | 单一标量 r(a) | 多步累积: 稀疏任务 reward + 密集子 reward |
| **优化目标** | 𝔼[r(a)] — 单轮 | 𝔼[∑γᵗR(s_t, a_t)] — 多轮折扣回报 |
| **数据** | 固定偏好数据集 | 可变动态环境 |

### RL 算法谱系

| 算法类别 | 代表算法 | 核心机制 | 在 K3 中的使用 |
|---|---|---|---|
| Policy Gradient | REINFORCE | ∇θJ = 𝔼[(ℛ-b)∇θ log πθ] | 基础 |
| PPO | PPO, VAPO, LitePPO | Clipped objective, trust region | K3 的前身 K2.5 使用 PPO-like |
| GRPO | GRPO, DAPO, Dr.GRPO | Group-relative advantage: Â = (ℛ-mean)/std | K3 使用类似机制 |
| DPO | DPO, SimPO, KTO | 从策略似然比导出隐式奖励 | — |

**Table 2** 详细对比了 PPO/DPO/GRPO 家族：

| 算法 | 是否需要 Value Network | 是否需要 Reference Model | Reward 形式 | 适合场景 |
|---|---|---|---|---|
| PPO | 是 | 否 | 显式 reward | 复杂环境 |
| DPO | 否 | 是 | 隐式 (来自偏好对) | 静态偏好数据 |
| GRPO | 否 | 否 | 组内相对 | LLM reasoning RL |

---

## 2. 六大能力分类

### 2.1 Planning (规划)

| 路径 | 描述 | 代表性工作 | K3 对应 |
|---|---|---|---|
| **RL as External Guide** | RL 训练辅助价值函数；LLM 在 MCTS 等搜索框架内生成 | RAP, LATS, MAPF-DT | ✗ |
| **RL as Internal Driver** | LLM 直接作为策略模型；通过环境交互的 trial-and-error 优化 | ETO, VOYAGER, Planner-R1, AdaPlan/PilotRL | **✓** |

未来方向: deliberation (深思) 与 intuition (直觉) 的合成 — 一个元策略决定何时探索、剪枝或定案。

### 2.2 Tool Using (工具使用)

三阶段演进:

| 阶段 | 方法 | 代表工作 | K3 所处位置 |
|---|---|---|---|
| **1. ReAct-style** | Prompt 工程或 SFT 教工具调用 | ReAct, Toolformer, AgentTuning | SFT 阶段 |
| **2. Tool-Integrated RL** | RL 从模仿转向结果驱动优化 | ToolRL, OTC-PO, ReTool, VTool-R1, ASPO | **RL 阶段** |
| **3. Long-horizon TIR (未来)** | 多轮工具使用 (瓶颈: 时间信用分配) | GiGPO, SpaRL | **K3 的目标** |

关键发现: ToolRL 证明即使从基础模型初始化且**没有任何模仿轨迹**，RL 训练也能涌现出:
- 错误代码的自修正
- 工具调用频率的自适应调整
- 多工具组合解决复杂子任务

### 2.3 Memory (记忆)

四个阶段:

| 阶段 | 描述 | 代表工作 |
|---|---|---|
| **RAG-style** | 外部数据存储，RL 可能调节查询时机 | MemoryBank, MemGPT, HippoRAG |
| **Token-level (Explicit)** | RL 控制的自然语言记忆池 | MemAgent, MEM1, Memory Token |
| **Token-level (Implicit/Latent)** | 潜在嵌入 token 作为记忆 | MemoryLLM, M+, Memory3 |
| **Prospective: Structured** | 时间图、原子笔记、层次化设计 | Zep, A-MEM (尚未 RL 控制) |

K3 的独特记忆方案: KDA 的固定大小循环状态 + External KV Cache Pool 的跨迭代持久化 — 结合了隐式 (KDA state) 和显式 (KV cache) 两种记忆。

### 2.4 Self-Improvement (自我提升)

三个层次:

| 层次 | 机制 | 代表工作 |
|---|---|---|
| **1. Verbal Self-correction** | 推理时的 prompt 启发式 (无梯度更新) | Reflexion, Self-refine, CRITIC |
| **2. Internalizing Self-correction** | RL 梯度更新将修正能力嵌入模型 | KnowSelf, SWEET-RL, ACC-Collab |
| **3. Iterative Self-training** | 自维持循环 (self-play + 搜索引导) | R-Zero, ISC, TTRL, SiriuS, MALT |

K3 处于 Level 2-3 之间: AET 的验证器闭环提供 Level 2 的 internalization，Partial Rollout 的跨迭代轨迹复用具备 Level 3 的 iterative 特征。

### 2.5 Reasoning (推理)

| 类型 | 类比 | 特征 | RL 的作用 |
|---|---|---|---|
| **Fast Reasoning** | System 1 | 快速、直觉、模式驱动 | 通过内部机制 (自洽性、ToT) 或置信度估计纠错 |
| **Slow Reasoning** | System 2 | 审慎、结构化、多步骤 | RL 优化 CoT、test-time scaling |

### 2.6 Perception & Others

- 多模态感知-动作循环
- 多 agent 协作和任务分工

---

## 3. 任务域分类

| 域 | 子任务 | K3 相关 benchmark |
|---|---|---|
| Search & Retrieval | HotPotQA, FEVER | BrowseComp, DeepSearchQA |
| GUI Navigation | WebArena, AndroidEnv | OSWorld, SaaS-Bench |
| Code Generation | SWE-bench, HumanEval | DeepSWE, SWE-Marathon, ProgramBench |
| Math Reasoning | MATH, AIME, AMC | GPQA Diamond, HLE-Full |
| Multi-Agent | Overcooked-AI, Melting Pot | Swarm Bench, MIRA Bench |
| Embodied Agents | Minecraft, ALFWorld | — |
| Scientific Discovery | 分子设计、蛋白质折叠 | 天体物理 case study |

---

## 4. 涌现的关键发现

### 4.1 从零涌现工具能力

> ToolRL 证明: 即使初始化自无模仿轨迹的基础模型，RL 训练也能自发产生 — 错误自修正、工具调用频率自适应、多工具组合。

### 4.2 涌现的推理链

> DeepSeek-R1 的 GRPO 训练在没有显式步骤监督的情况下产生了涌现的推理链。

### 4.3 涌现的自我进化

> R-Zero 等系统实现了从零数据开始的自我进化推理——通过 self-play 和搜索引导的 refinement 涌现。

### 4.4 证明工具驱动的涌现

> Lin & Xu (2025) 从理论上证明了: 确定性工具驱动的状态转移从根本上扩展了纯文本 RL 的边界。

---

## 5. 环境与框架大全

| 类别 | 代表环境 |
|---|---|
| Code | SWE-bench, HumanEval, MBPP, CodeContests |
| Math | MATH, GSM8K, AIME, AMC |
| Web | WebArena, MiniWoB++, VisualWebArena, WebShop |
| GUI | AndroidEnv, AppAgent, ScreenAgent |
| Game | NetHack, Minecraft, ALFWorld |
| Multi-Agent | Overcooked-AI, Melting Pot |
| Scientific | 分子设计, 蛋白质折叠, 实验室自动化 |

## 6. 开放挑战

| 挑战 | 描述 | K3 的处理 |
|---|---|---|
| **时间信用分配** | 长轨迹中难以归因 | Partial Rollout + 验证器结果反推 |
| **奖励设计** | 过程级 vs 仅结果级 | AET 的复合 reward (公开+隐藏验证器) |
| **可扩展性** | 多轮 RL 的计算成本 | Co-located RL + External KV Cache Pool |
| **泛化** | 跨域迁移 agent 能力 | Unified White-Box RL 的随机环境 |
| **安全与对齐** | RL 优化的 agent 保持人类价值观 | Reward hacking 检测 + Agent Behavior Bench |
| **RL 控制的结构化记忆** | 尚在初期阶段 | KDA state + External KV Cache 是独特方案 |

---

## 7. 与 K3 分析的对照

综述的框架与我们之前的分析高度一致:

| 我们的分析 | 综述的对应 |
|---|---|
| 「涌现不是被教出来」 | §4 涌现的关键发现 |
| 「四个涌现条件」 | §6 开放挑战 (稀疏奖励、环境多样性等) |
| 「规划 + 工具使用相互塑造」 | §3.1 + §3.2 的 integration |
| 「框架派 vs 训练派」 | §3.2 Tool Using 的 Stage 1 vs Stage 2 |
| 「Agent 模型的七个维度」 | §3 六大能力 + §4 任务域 |

**综述验证了我们的核心观点**: Agentic RL 的本质不是「RLHF 的升级版」，而是一个**从单步偏好优化到多步自主决策的范式转变**。K3 是这个范式在 3T 模型规模上的首次完整实践。
