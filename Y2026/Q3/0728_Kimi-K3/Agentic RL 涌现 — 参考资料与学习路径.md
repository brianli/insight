---
title: Agentic RL 涌现 — 参考资料与学习路径
tags: [rl, emergence, agent, reading_list, education]
created: 2026-07-29
related:
  - "[[K3 规划与工具使用 — 从 Agentic RL 中涌现]]"
  - "[[K3 Agent 能力全景]]"
  - "[[Agentic RL Survey 论文精读笔记]]"
---

## 阅读指南

按「基础 → 核心论文 → 前沿」三层递进组织。建议按顺序读，但可根据背景跳级。

| 你的背景 | 建议入口 |
|---|---|
| 对 RL 只有基础概念 | 从第 1 节开始 |
| 熟悉 PPO/RLHF | 从第 2 节开始 |
| 读过 DeepSeek-R1, Kimi K1.5 | 从第 3 节开始 |

---

## 1. 基础层：理解强化学习与涌现

这些是理解「为什么 agentic RL 能产生涌现」所需的理论基础。不需要全读完，但其中的概念会反复出现在后续论文中。

### 核心教材

- **Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018)**
  强化学习的圣经。重点章节:
  - Chapter 13: Policy Gradient Methods (PPO 的前身)
  - Chapter 16: Applications and Case Studies (RL 在真实世界中的挑战)
  - 在线免费: http://incompleteideas.net/book/the-book.html

### 关键论文 (RL 基础)

- **Schulman et al., "Proximal Policy Optimization Algorithms" (2017)**
  PPO 论文。K3 的 RL 算法 (Kimi K2.5 描述) 源自 PPO 的 trust-region 思想。理解 PPO 的 clipped objective 是理解「为什么 K3 对 stale data 有容忍度」的前提。
  arXiv: 1707.06347

- **Silver et al., "Reward is Enough" (2021)**
  一篇有争议但极具启发性的论文。核心论点: 最大化奖励的智能体自然会涌现出感知、规划、工具使用等能力 — 不需要为每个能力单独设计训练目标。这是理解 agentic RL 涌现的哲学基础。
  *Artificial Intelligence*, 299, 103535

### 关键概念

要理解 K3 的 RL 设计，需要先掌握这些概念在 LLM RL 语境中的含义:

| 概念 | 对应的 K3 设计 |
|---|---|
| **Sparse Reward** | 只检查最终输出和验证器结果 |
| **Exploration-Exploitation** | τ 递减课程, K=4 采样 |
| **Policy Gradient** | 基于 reward 更新模型参数 |
| **Trust Region / KL Penalty** | Per-token 正则化处理 off-policy staleness |
| **Curriculum Learning** | 8K→64K→256K→1M 上下文课程, τ 递减课程 |
| **On-Policy vs Off-Policy** | Partial rollout 引入 off-policy staleness |

---

## 2. 核心层：LLM 上的强化学习与推理涌现

这是「涌现」主题的核心文献。建议按发表时间顺序阅读，能看到思路的演进。

### 必读：推理能力的涌现

- **DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025)**
  **这是理解推理涌现的起点。** 核心贡献: 证明纯 RL (GRPO 算法) 可以激发 LLM 的链式推理能力，不需要手动标注推理链。R1-Zero 版本完全没有 SFT 推理数据，仅靠 RL 的 reward 信号就自发产生了 CoT 行为。
  - 关键词: GRPO, reasoning emergence, cold-start, multi-stage RL
  - Nature 发表: DOI 10.1038/s41586-025-09422-z

- **Kimi Team, "Kimi K1.5: Scaling Reinforcement Learning with LLMs" (2025)**
  K3 的 RL 方法论前身。详细描述了 long-CoT 和 short-CoT 的统一训练、partial rollout 机制、和多个 reasoning domain 上的 RL scaling。K3 的 §4.1.2 算法直接继承自此。
  - 关键词: long-CoT RL, partial rollout, on-policy distillation
  - arXiv: 2501.12599

### 核心算法

- **Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning" (2024) — 含 GRPO 算法定义**
  GRPO (Group Relative Policy Optimization) 被首次提出。GRPO 的核心思想是用组内相对分数替代 PPO 的 value network，使 RL 在 LLM 上的规模化成为可能。K3 中也使用了类似机制。
  arXiv: 2402.03300

### 工具使用与 Agent 环境

- **Yang et al., "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering" (2024)**
  LM 作为 coding agent 的基础工作。提出了 ACI (Agent-Computer Interface) 的概念 — 为 LM 优化的工具接口，影响深远。理解 SWE-agent 的 agent loop 才能理解 K3 的 Unified White-Box RL 中的 harness 设计。
  arXiv: 2405.15793

- **Jimenez et al., "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" (2024)**
  定义了 coding agent 评测标准。K3 的 DeepSWE, SWE-Marathon 等都建立在这个范式之上。
  arXiv: 2310.06770

### Scaling 理论基础

- **Sardana et al., "Beyond Exhaustive Rollouts: Asynchronous Multi-Agent RL Training with Heterogeneous Staleness" (2024) — 代表性 stale data 研究**
  虽然不是 K3 直接引用的论文，但其 stale data 容忍性分析与 K3 的 per-token 正则化设计思想一致。
  
- **Kaplan et al., "Scaling Laws for Neural Language Models" (2020)**
  Train-time scaling 的经典。K3 的 §3.2 scaling law 分析直接建立在此框架上。
  arXiv: 2001.08361

---

## 3. 前沿层：Agentic RL 与 K3 相关

### 第一优先级：Agentic RL 综述

- **Zhang et al., "The Landscape of Agentic Reinforcement Learning for LLMs: A Survey" (2025)**
  **这是关于 agentic RL 最全面的综述。** 500+ 篇文献综合，系统回答了:
  - Agentic RL 是什么 (区别于传统 RLHF)
  - 训练框架分类 (centralized RL, decentralized multi-agent, etc.)
  - 环境设计 (sandbox, interactive, open-ended)
  - Reward 设计 (verifiable, learned, hybrid)
  - 涌现行为分析
  - 未来方向
  arXiv: 2509.02547 | TMLR 版本: https://mlanthology.org/tmlr/2026/zhang2026tmlr-landscape/

### K3 系列技术报告

按时间顺序:

- **Kimi Team, "Kimi K2: Open Agentic Intelligence" (2025)**
  K3 上一代，引入了 co-located RL、外部 KV cache、partial rollout 等机制。K3 的 §5.3 RL 基础设施直接建立在此之上。
  arXiv: 2507.20534

- **Kimi Team, "Kimi K2.5: Visual Agentic Intelligence" (2026)**
  引入 Agent Swarm (多 agent 并发)、原生视觉 RL 训练。K3 的 Swarm Bench、vision-in-the-loop 训练继承自此。
  arXiv: 2602.02276

- **Kimi Team, "Kimi K3: Open Frontier Intelligence" (2026)**
  当前分析的论文本身。
  开源: https://github.com/MoonshotAI/Kimi-K3/tree/main

### K3 技术报告中引用的关键工作

K3 论文的 References 中最直接相关的 agentic RL 相关引用:

| 引用 | 论文 | 与 K3 的关系 |
|---|---|---|
| [59] | Kimi K2.5 Agent Swarm | RL 环境设计, swarm 训练 |
| [118] | Kimi K1.5 | RL 算法基础, partial rollout |
| [40] | DeepSeek-R1 | Reasoning 涌现, GRPO-like 算法 |
| [75] | Multi-teacher distillation | MOPD 蒸馏的理论基础 |
| [134] | On-policy distillation | MOPD 的 on-policy 属性来源 |
| [29] | DeepSeek-V4 | MOPD 在另一框架中的实践 |
| [84] | OpenAI o-series | Test-time scaling 概念 |
| [30] | DeepSeek-V3 | Auxiliary-loss-free load balancing |
| [23] | DeepSeekMoE | Shared + routed expert 架构 |

### 涌现相关的重要工作

- **Wei et al., "Emergent Abilities of Large Language Models" (2022)**
  定义和系统研究了 LLM 的涌现现象。虽然主要讨论 scale 带来的涌现 (不是 RL 带来的)，但提供了涌现的概念框架。
  arXiv: 2206.07682

- **Brown et al., "Language Models are Few-Shot Learners" (GPT-3, 2020)**
  首次大规模展示 LLM 的涌现行为。奠定了「scale 带来涌现」的认知基础。
  arXiv: 2005.14165

---

## 4. 补充材料：环境设计与 Tool Use

### Agent 环境

- **OpenAI, "Codex CLI" & Anthropic, "Claude Code"**
  不是论文，但作为产品级别的 agent harness，是理解 K3 的 Unified White-Box RL 中「为什么需要模拟多种 harness」的重要参考。
  - Codex: https://github.com/openai/codex
  - Claude Code: https://docs.anthropic.com/en/docs/claude-code

- **KVCache-AI, "AgentENV"**
  K3 使用的 microVM 沙箱，已开源。直接看代码可以理解 agent RL 环境的设计哲学。
  https://github.com/kvcache-ai/AgentENV

### 长上下文 Agent

- **Kimi Team, "Kimi Delta Attention" (2025) — referenced as [63] in K3**
  KDA 的原始论文。解释了线性注意力如何支持长上下文 agent。
  
- **Kimi Team, "Attention Residuals" (2026) — referenced as [57] in K3**
  AttnRes 的独立论文。解释了跨层信息流如何支持深层 agent model。

---

## 5. 学习路径建议

### 路径 A: 快速理解 (1-2 周)

```
1. Agentic RL Survey (§3, §4, §7)
   → 建立全貌认知

2. DeepSeek-R1 论文
   → 理解 reasoning 涌现的核心机制

3. Kimi K1.5 → K2 → K2.5 → K3 技术报告
   → 理解 Kimi 系列的 RL 方法论演进

4. AgentENV GitHub README + 架构文档
   → 理解 agent RL 环境的工程实现
```

### 路径 B: 深入理论 (4-8 周)

```
1. Sutton & Barto Ch.13 (Policy Gradient) + PPO 论文
   → 建立 RL 理论基础

2. GRPO 论文 + DAPO 论文
   → 理解 LLM 专属 RL 算法

3. Agentic RL Survey 全文
   → 系统掌握分类体系

4. DeepSeek-R1 + Kimi K1.5 全文 (带 appendix)
   → 深入理解算法细节

5. K3 技术报告 + 开源代码 (MoonEP, FlashKDA, AgentENV)
   → 理论与工程的完整闭环

6. Reward is Enough 论文
   → 思考涌现的哲学基础
```

### 路径 C: 实践路线

```
1. GRPO-Zero (GitHub: policy-gradient/GRPO-Zero)
   → 动手实现 GRPO

2. SWE-agent (GitHub)
   → 构建 coding agent 环境

3. AgentENV (GitHub)
   → 理解 microVM sandbox 设计

4. MiniTriton (K3 训练的 GPU 编译器, GitHub)
   → 看 K3 作为 agent 的实际产出

5. 用 GRPO 在小模型上复现简化的 agentic RL
   → 亲手体会涌现
```

---

## 6. 当前笔记体系

本系列笔记的完整索引:

```
Kimi K3 论文阅读报告              — 全局概览
K3 架构-训练范式-基础设施三层关系  — 模型从设计到运转的全栈
K3 Agent 能力全景                 — 七个维度的能力训练
K3 Agent 能力对比                 — 30+ benchmark 数据对比
K3 规划与工具使用 — 从 Agentic RL 中涌现  — 涌现机制的详细分析
Agentic RL 涌现 — 参考资料与学习路径    — 本文档
```
