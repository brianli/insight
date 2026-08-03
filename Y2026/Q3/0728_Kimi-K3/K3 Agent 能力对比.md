---
title: Kimi K3 Agent 能力对比 — Benchmark 七维度全景
tags: [llm, agent, benchmark, comparison]
created: 2026-07-28
related:
  - "[[K3 Agent 能力全景]]"
  - "[[Kimi K3 论文阅读报告]]"
---

## 方法论

将 K3 技术报告 Tables 2 & 3 中 30+ 个 agentic benchmark 映射到 K3 Agent 的七个能力维度。每个维度选出代表性 benchmark，对比 K3 与 Claude Fable 5 (max)、GPT-5.6 Sol (max)、Claude Opus 4.8 (max)、GPT-5.5 (xhigh) 和 GLM-5.2 (max)。

> 标注: **加粗 = 第一**，下划线 = 第二。数据来源: 报告 Tables 2/3, 第三方评测 Table 5。

---

## 1. Coding Agent

代表模型自主完成软件工程全流程的能力。

| Benchmark | K3 | Fable 5 | GPT-5.6 Sol | Opus 4.8 | GPT-5.5 | GLM-5.2 |
|---|---|---|---|---|---|---|
| DeepSWE | 67.5 | **70.0** | 73.0* | 59.0 | 67.0 | 46.2 |
| ProgramBench | **77.8** | 76.8 | 77.6 | 71.9 | 70.8 | 63.7 |
| Terminal-Bench 2.1 | 88.3 | 88.0 | **88.8** | 84.6 | 83.4 | 82.7 |
| FrontierSWE | 81.2 | **86.6** | 71.3 | 66.7 | 64.9 | 67.3 |
| SWE-Marathon | **42.0** | 35.0† | 39.0 | 40.0 | 14.0 | 13.0 |
| PostTrainBench | 36.6 | **41.4** | 34.6 | 34.1 | 28.4 | 34.3 |
| Kimi Code Bench 2.0‡ | 72.9 | _76.9_ | — | 71.7 | 66.0 | 64.2 |
| Kimi Webdev Bench | **+31.0%** win rate vs Opus 4.8 | — | — | — | — | — |

> \* GPT-5.6 Sol 在 DeepSWE 的 harness 可能不同于 K3 的评估设置 (技术报告仅报告得分, 不报告 harness 影响)。
> † Fable 5 在 SWE-Marathon 有 2 题拒答 + 3 题因访问限制无法测试，计为 0。
> ‡ Kimi Code Bench 2.0 各模型使用不同 harness。

**分析:**

- K3 领先: ProgramBench (+1.0 vs Fable 5), SWE-Marathon (+7.0 vs Opus 4.8), Kimi Webdev (+31% vs Opus 4.8)
- K3 竞争: Terminal-Bench 2.1 (仅比第一低 0.5)
- K3 落后: DeepSWE (-5.5 vs Sol), FrontierSWE (-5.4 vs Fable 5), PostTrainBench (-4.8 vs Fable 5)

**规律的识别**: K3 的优势随任务长度增加而放大。DeepSWE (中等长度) 落后 5.5 分, FrontierSWE (长) 落后 5.4 分, SWE-Marathon (超长) **领先** 7 分。这说明 K3 在长时间自主作业中的持续性和自纠错能力强于在短周期任务中的精度。

---

## 2. Web/Search Agent

代表模型在信息检索、深度研究和证据整合方面的能力。

| Benchmark | K3 | Fable 5 | GPT-5.6 Sol | Opus 4.8 | GPT-5.5 | GLM-5.2 |
|---|---|---|---|---|---|---|
| BrowseComp | **91.2** | 88.0 | 90.4 | 84.3 | 84.4 | — |
| DeepSearchQA (F1) | **95.0** | 94.2 | — | 93.1 | — | — |
| ResearchRubrics | **76.2** | — | 73.8 | 73.5 | 64.0 | 71.1 |
| Deep Research Bench‡ | **90.0** | — | 85.3 | 87.2 | 81.9 | 84.0 |

> ‡ 内部 benchmark。

**分析:**

K3 在所有搜索和研究类 benchmark 上均**排名第一**。这是 K3 最强的能力域。技术报告 §6.3 显示 WebDev Arena 上 K3 同样排名第一 (1678 Elo, 领先 Fable 5 的 1634)。

BrowseComp 尤其值得关注: 这是一个需要模型在 web 环境中自主搜索、筛选和验证信息的 benchmark。K3 的 91.2% 不仅略高于 Sol (90.4%), 且成本仅为 Sol 的一半 ($2.03 vs $4+)。

---

## 3. Autonomous Execution

代表模型在没有参考轨迹的情况下自主完成任务分解、工具选择、规划和验证的能力。

| Benchmark | K3 | Fable 5 | GPT-5.6 Sol | Opus 4.8 | GPT-5.5 | GLM-5.2 |
|---|---|---|---|---|---|---|
| AutomationBench | **30.8** | 29.1 | 29.7 | 27.2 | 22.7 | 12.9 |
| KAET (内部) | **83.5** | — | 85.4* | 78.7 | 79.7 | 74.7 |
| OSWorld-Verified | 84.8 | **85.0** | 83.0 | 83.4 | 79.0 | — |
| OSWorld 2.0 | 58.3 | **66.1** | 62.6 | 55.7 | 49.5 | — |
| SaaS-Bench | 60.1 | — | **61.4** | 56.1 | 43.8 | — |
| SpreadsheetBench 2 | 34.8 | 34.7 | 32.4 | 31.6 | 29.1 | 28.1 |

> \* KAET 上 GPT-5.6 Sol 85.4, K3 83.5。

**分析:**

- K3 领先: AutomationBench (+1.7 vs Fable 5)
- K3 落后: OSWorld 2.0 (-7.8 vs Fable 5), KAET (-1.9 vs Sol)
- K3 竞争: SpreadsheetBench 2 (差 0.1 vs Fable 5)

**规律的识别**: K3 在「全新的自主执行」类任务上表现最强 (AutomationBench), 但在「模拟已有产品交互」的任务上稍弱 (OSWorld, SaaS-Bench)。这可能是因为 Autonomous Execution Tasks (AET) 的训练重点在于从零构建解决方案，而 UI 操作类任务需要特定的交互范式知识。

---

## 4. Multi-Agent Orchestration

代表模型分解复杂目标、协调多个子 agent 并行执行、合成结果的能力。

| Benchmark | K3 | Fable 5 | GPT-5.6 Sol | Opus 4.8 | GPT-5.5 | GLM-5.2 |
|---|---|---|---|---|---|---|
| Swarm Bench‡ | **76.3** | — | 73.2 | 72.6 | 61.8 | 58.5 |
| MIRA Bench‡ | 64.1 | **72.9** | 62.2 | 59.8 | 54.6 | — |
| MCP-Atlas | 84.2 | **84.7** | 83.6 | 83.6 | 82.8 | 82.6 |
| MCPMark-Verified | **94.5** | 87.4 | 92.9 | 76.4 | 92.9 | — |

> ‡ 内部 benchmark。

**分析:**

- K3 领先: Swarm Bench (+3.1 vs Sol), MCPMark-Verified (+1.6 vs Sol)
- K3 落后: MIRA Bench (-8.8 vs Fable 5)

Swarm Bench 的特别之处: 这是 Kimi Agent harness 独有的并发 agent 协调范式 [59]。K3 在这个 benchmark 上的绝对优势表明 agent orchestration 是其专门训练的强项，而非通用能力的附带效果。

MIRA Bench 上 Fable 5 明显领先 (-8.8)，这个 bench 测试的是企业级多角色多系统协作，暗示 K3 在复杂企业工作流编排上仍有提升空间。

---

## 5. Long-Horizon Assistant

代表模型在持续多天的助手场景中处理并发事件、管理状态和完成任务的能力。

| Benchmark | K3 | Fable 5 | GPT-5.6 Sol | Opus 4.8 | GPT-5.5 | GLM-5.2 |
|---|---|---|---|---|---|---|
| 24/7 ClawBench 2.0‡ | 48.3 | 47.4 | **52.0** | 47.2 | 48.5 | 43.2 |
| Online Experience‡ | 77.9 | 74.2 | **84.0** | 69.4 | 73.7 | 64.0 |
| JobBench | 54.3 | **57.4** | 45.4 | 48.4 | 38.3 | 43.4 |
| Agent Behavior Bench‡ | 65.0 | 75.5 | **76.4** | 65.7 | 70.1 | — |
| Toolathlon-Verified | 76.5 | **77.9** | 74.9 | 76.2 | 73.5 | 59.9 |

> ‡ 内部 benchmark。

**分析:**

这是 K3 最弱的 agent 能力维度。所有 benchmark 均被 Fable 5 或 Sol 领先:

- Agent Behavior Bench: K3 65.0 vs Sol 76.4, Fable 5 75.5 (-10.4 to -11.4)
- Online Experience: 落后 Sol 6.1
- MIRA Bench (应归入此域): 落后 Fable 5 8.8
- JobBench: 落后 Fable 5 3.1

Agent Behavior Bench 评估的是 tool-use 行为、效率和纪律 — 这些是「过程质量」而非「结果正确性」。**K3 在过程质量上显著落后于两个顶级闭源模型，这是明确的改进方向。**

---

## 6. Professional Knowledge Work

代表模型在金融、法律、投行、税务等专业领域完成端到端工作流的能力。

| Benchmark | K3 | Fable 5 | GPT-5.6 Sol | Opus 4.8 | GPT-5.5 | GLM-5.2 |
|---|---|---|---|---|---|---|
| GDPval-AA v2 (Elo) | 1686 | **1747** | 1736 | 1593 | 1491 | 1510 |
| AA-Briefcase (Elo) | 1548 | **1583** | 1495 | 1354 | 1158 | 1260 |
| CorpFin v2 | 71.6 | **71.8** | 64.4 | 66.7 | 68.4 | 66.1 |
| Finance Agent v2 | 54.4 | **56.3** | 53.8 | 53.9 | 51.8 | 49.7 |
| Legal Research Bench | 44.2 | **49.5** | 48.1 | 43.8 | 40.4 | 31.3 |
| τ³-Banking | **33.4** | 26.8 | 33.0 | 27.6 | 31.3 | 26.8 |
| Harvey Lab-AA | **94.6** | 93.6 | 87.2 | 91.1 | 86.3 | 91.0 |
| Finance Bench‡ | 62.6 | — | **62.7** | 60.7 | 58.4 | 55.4 |
| DECK Bench‡ | **73.5** | 73.0 | 74.7* | 66.9 | 68.2 | 68.6 |
| Agents' Last Exam | 28.3 | 25.7 | **29.6** | 27.0 | 26.6 | 20.4 |

> ‡ 内部 benchmark。\* DECK Bench: GPT-5.6 Sol 74.7, K3 73.5。

**分析:**

- K3 领先: τ³-Banking (+6.6 vs Fable 5), Harvey Lab-AA (+1.0 vs Fable 5), DECK Bench (几乎并列)
- K3 竞争: CorpFin v2 (仅差 0.2 vs Fable 5), Finance Bench (差 0.1 vs Sol)
- K3 落后: GDPval-AA v2 (-61 Elo vs Fable 5), AA-Briefcase (-35 Elo), Legal Research (-5.3 vs Fable 5)

**规律的识别**: 专业领域 work 是 Fable 5 的优势域。K3 在数据密集型和流程型任务 (τ³-Banking, Harvey Lab, CorpFin) 上表现接近或领先，但在需要广泛知识覆盖和专业判断的 Elo 类 benchmark (GDPval-AA, AA-Briefcase) 上差距更大。这与 K3 在 HLE-Full 和 CritPt 上的短板一致 — **knowledge breadth 仍是瓶颈**。

---

## 7. Vision-in-the-Loop Agent

代表模型利用视觉反馈进行迭代推理和操作的能力。

| Benchmark | K3 | Fable 5 | GPT-5.6 Sol | Opus 4.8 | GPT-5.5 | GLM-5.2 |
|---|---|---|---|---|---|---|
| Agentic Vision Bench‡ | 78.3 | 81.1 | **82.9** | 82.8 | 76.9 | — |
| CharXiv (RQ) w/ tool | 91.3 | **93.5** | 89.1 | 89.9 | 89.0 | — |
| Math-Vision w/ tool | **97.8** | _98.6_ | _97.8_ | 97.1 | 96.8 | — |
| ZeroBench-main w/ tool (pass@5) | 41.0 | **46.0** | 35.0 | 34.0 | 41.0 | — |
| WorldVQA | 51.0 | **56.7** | 41.8 | 39.1 | 38.5 | — |

> ‡ 内部 benchmark。

**分析:**

- K3 领先: Math-Vision w/ tool = Sol (97.8)
- K3 竞争: 与 Fable 5 的差距在 2-5 个百分点
- K3 落后: Agentic Vision Bench (-4.6 vs Sol), WorldVQA (-5.7 vs Fable 5)

**规律的识别**: Vision-in-the-loop 的差距虽然存在但不大，且主要集中在「视觉理解 + 推理」的任务上 (如 Agentic Vision Bench, WorldVQA)。在「视觉 + 计算/代码」的任务上 (如 Math-Vision w/ tool)，K3 与 Fable 5 几乎持平。

---

## 汇总：K3 在各维度的竞争位置

### 位次热力图

| 能力维度 | K3 | Fable 5 | GPT-5.6 Sol | Opus 4.8 | GPT-5.5 | GLM-5.2 |
|---|---|---|---|---|---|---|
| Coding Agent | ★★★☆ | ★★★★ | ★★★☆ | ★★☆ | ★★ | ★ |
| Web/Search Agent | ★★★★★ | ★★★★ | ★★★★ | ★★★ | ★★★ | ★★ |
| Autonomous Execution | ★★★★ | ★★★★ | ★★★★ | ★★★ | ★★ | ★ |
| Multi-Agent Orchestration | ★★★★ | ★★★★ | ★★★★ | ★★★ | ★★ | ★ |
| Long-Horizon Assistant | ★★★ | ★★★★ | ★★★★★ | ★★★ | ★★★ | ★☆ |
| Professional Knowledge | ★★★★ | ★★★★★ | ★★★★ | ★★★ | ★★★ | ★★ |
| Vision-in-the-Loop Agent | ★★★☆ | ★★★★ | ★★★★ | ★★★ | ★★★ | — |

### 按领先/落后数量

| vs 对手 | K3 领先的 benchmark 数 | K3 落后的 benchmark 数 | 净领先 |
|---|---|---|---|
| vs Fable 5 | ~12/32 | ~16/32 | -4 |
| vs GPT-5.6 Sol | ~15/27 | ~8/27 | +7 |
| vs Opus 4.8 | ~25/30 | ~3/30 | +22 |
| vs GPT-5.5 | ~26/29 | ~3/29 | +23 |
| vs GLM-5.2 | ~27/27 | ~0/27 | +27 |

### 主要竞争对手的优劣势

| 对手 | 对 K3 的优势域 | K3 对对手的优势域 |
|---|---|---|
| **Fable 5** | Professional Knowledge (Elo 类), Long-Horizon Assistant (行为质量), FrontierSWE | Web/Search Agent (全部), Multi-Agent Orchestration (Swarm), SWE-Marathon, AutomationBench, Coding Cost Efficiency |
| **GPT-5.6 Sol** | Long-Horizon Assistant (Online Experience, Agent Behavior), DeepSWE | Web/Search Agent (全部), Multi-Agent Orchestration (Swarm, MCPMark), BrowseComp cost (1/2), Terminal-Bench 2.1 (接近) |

---

## 关键洞察

### 1. K3 的 agent 优势有明确的"体型"

**强项**: 搜索研究、自主执行、并行编排
**中项**: coding (长线任务显著更强)、专业工作 (数据密集型更强)
**弱项**: 长期助手的行为质量、UI 操作系统、广泛知识覆盖面

这不是「全能但什么都不突出」，而是**有明确结构的能力分布** — 反映了 RL 训练投入的重点方向 (搜索、coding、多 agent) 和相对投入较少的领域 (企业工作流、UI 交互)。

### 2. 与 Fable 5 的差距本质

Fable 5 对 K3 的最大优势不在结果正确性 (很多 benchmark 上 K3 分数接近甚至领先)，而在:

- **Elo 类指标**: GDPval-AA v2 (-61), AA-Briefcase (-35) — 反映的是知识广度、综合判断力
- **行为质量**: Agent Behavior Bench (-10.5) — 反映的是工具使用纪律、过程结构化程度
- **复杂 UI/OS**: OSWorld 2.0 (-7.8) — 反映的是对模拟环境的操作熟练度

这三类差距指向同样的根因: **K3 的知识面和交互细腻度还需要提升**。

### 3. SWE-Marathon 的第一名不是偶然

SWE-Marathon 是唯一需要 agent 在数百步中保持策略连贯 + 自纠错 + 一次性打通的 benchmark。K3 在这里领先 7 分，结合 DeepSWE (中等) 到 FrontierSWE (长) 到 SWE-Marathon (超长) 的排名上升趋势，清晰表明: **K3 为 long-horizon agentic 场景做了针对性优化，这个优势在常规 benchmark 上不明显，但在极限长度任务上充分释放。**

### 4. Web/Search 是 K3 最完整的优势域

BrowseComp, DeepSearchQA, ResearchRubrics, Deep Research Bench — K3 在这个维度上**全覆盖第一**。考虑到 search agent 是大多数实际应用的核心场景，这个优势的商业价值可能比单项 benchmark 分数更重要。

---

## 与七个能力维度的最终映射

| 能力维度 | K3 最强 benchmark | K3 最弱 benchmark | 与最强对手的差距 |
|---|---|---|---|
| Coding Agent | SWE-Marathon (+7.0 vs Opus) | PostTrainBench (-4.8 vs Fable 5) | 中等，任务越长越好 |
| Web/Search Agent | BrowseComp (+1.2 vs Sol) | — | 无短板 |
| Autonomous Execution | AutomationBench (+1.7 vs Fable 5) | OSWorld 2.0 (-7.8 vs Fable 5) | 中等，纯自主强、UI 弱 |
| Multi-Agent | Swarm Bench (+3.1 vs Sol) | MIRA (-8.8 vs Fable 5) | 中等，Swarm 范式强、企业范式弱 |
| Long-Horizon Assistant | 24/7 ClawBench (+0.9 vs Fable 5) | Agent Behavior (-11.4 vs Sol) | 较大，行为质量是明确短板 |
| Professional Knowledge | τ³-Banking (+6.6 vs Fable 5) | Legal Research (-5.3 vs Fable 5) | 较小但有方差 |
| Vision-in-the-Loop | Math-Vision w/ tool (持平 Sol) | WorldVQA (-5.7 vs Fable 5) | 中等偏小 |
