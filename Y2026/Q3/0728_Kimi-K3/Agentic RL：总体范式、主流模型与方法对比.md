---
title: Agentic RL：总体范式、主流模型与方法对比
tags: [agentic_rl, reinforcement_learning, llm, agents, survey, model_comparison]
created: 2026-08-05
related:
  - "[[Kimi K3 论文阅读报告]]"
  - "[[Agentic RL Survey 论文精读笔记]]"
  - "[[Agentic RL 涌现 — 参考资料与学习路径]]"
  - "[[K3 规划与工具使用 — 从 Agentic RL 中涌现]]"
  - "[[AgentENV：当大模型开始学会“做事”，我们开源了支撑它的基础设施]]"
---

# Agentic RL：总体范式、主流模型与方法对比

## 0. 阅读结论

Agentic RL 不是“给 LLM 加几个工具调用”的产品包装，也不是把传统 `RLHF` 的单轮偏好优化简单延长。它的核心变化是：

```text
模型不再只生成一段文本
而是在环境中持续观察、决策、行动、获取反馈并修正策略
```

因此，Agentic RL 的训练对象不是单个回答，而是一个能够在长程任务中完成目标的策略：

```text
什么时候思考
→ 是否调用工具
→ 选择哪个工具以及如何设置参数
→ 如何解释工具返回
→ 失败后如何恢复
→ 何时继续探索
→ 何时停止并提交结果
```

从技术体系看，Agentic RL 至少包括八个相互耦合的部分：

1. **任务与环境**：模型要在什么世界中行动；
2. **状态与记忆**：模型能看到什么、能保留什么；
3. **动作与工具接口**：模型能够采取什么行动；
4. **奖励与验证器**：什么行为被认为是成功；
5. **轨迹采样与信用分配**：如何从长程交互中学习；
6. **策略优化**：如何根据 reward 更新模型；
7. **课程与泛化**：如何从简单任务扩展到复杂环境；
8. **训练基础设施与部署**：如何以足够低的成本执行百万级 rollout。

当前主流模型大致分为三条路线：

| 路线 | 代表模型 | 主要训练对象 | 核心优势 | 主要局限 |
|---|---|---|---|---|
| **Reasoning-first** | DeepSeek-R1、OpenAI o1、Qwen3、Gemini Deep Think | 内部长链推理、验证与策略搜索 | reward 较容易构造，推理能力提升明显 | 外部环境、工具使用与长程状态管理相对不充分或未公开 |
| **Agent-environment-first** | Kimi K2/K2.5/K3、MiniMax-M1、Qwen3-Coder-Next | 工具调用、代码执行、沙箱交互、长期任务 | 更接近“会做事”的 Agent，环境反馈真实 | 训练系统昂贵，reward hacking、长程信用分配与环境泛化困难 |
| **Closed frontier hybrid** | OpenAI o3/o4-mini、Claude 4、Gemini 系列 | 推理、工具使用、产品级任务完成 | 模型、工具和产品系统结合紧密 | 训练配方与环境大多不公开，外部难以复现或归因 |

需要特别区分：

> **推理 RL 是 Agentic RL 的重要基础，但不是 Agentic RL 的全部。**

如果模型只在数学题上生成长 `CoT`，它获得的是内部 reasoning policy；只有当模型持续与外部环境、工具、代码执行器、浏览器、应用状态或其他 Agent 交互，并根据环境结果更新策略时，才进入更严格意义上的 Agentic RL。

---

## 1. 什么是 Agentic RL

### 1.1 从单轮偏好优化到长期决策

传统的 `RLHF`、`RLAIF` 或 `DPO` 通常可以近似为：

```text
输入 prompt
    ↓
模型生成一个回答
    ↓
人类或 reward model 评价回答
    ↓
更新模型
```

它更接近一个退化的单步决策问题：

$$
s_0 \rightarrow a_0 \rightarrow r_0
$$

其中，`prompt` 提供主要状态，模型输出一段文本作为动作，评价结果通常在单轮结束时给出。

Agentic RL 则将问题扩展为具有部分可观测性和时间延展性的 `POMDP`：

$$
\mathcal{M}=(\mathcal{S},\mathcal{O},\mathcal{A},\mathcal{T},\mathcal{R},\gamma)
$$

其中：

- $\mathcal{S}$：真实环境状态，例如代码仓库、浏览器页面、数据库或应用工作区；
- $\mathcal{O}$：模型能够观察到的内容，通常只是状态的一部分；
- $\mathcal{A}$：动作空间，包括文本、工具调用、代码执行、文件修改、子 Agent 委派等；
- $\mathcal{T}$：环境状态转移；
- $\mathcal{R}$：对中间步骤或最终结果的奖励；
- $\gamma$：长期回报的折扣因子。

模型在时间 $t$ 的策略不只依赖当前一条输入，而是依赖截至目前的历史：

$$
h_t=(o_0,a_0,o_1,a_1,\ldots,o_t)
$$

策略可以写为：

$$
a_t\sim\pi_\theta(a_t\mid h_t)
$$

训练目标则是最大化整条轨迹的期望回报：

$$
J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\sum_{t=0}^{T}\gamma^t r_t
\right]
$$

实际 LLM RL 往往还会加入参考策略约束：

$$
J(\theta)
=
\mathbb{E}
\left[
\sum_t\gamma^t r_t
-\beta D_{\mathrm{KL}}\bigl(\pi_\theta\Vert\pi_{\mathrm{ref}}\bigr)
\right]
$$

其目的不是让模型“尽量像参考模型”，而是防止策略在 reward 不完备时快速偏离语言能力、工具格式和安全边界。

### 1.2 Agentic RL 与几个相邻概念的边界

| 概念 | 主要特征 | 是否必然是 Agentic RL |
|---|---|---|
| `SFT` | 模仿示范轨迹 | 否 |
| `RLHF/DPO` | 对单轮回答进行偏好优化 | 通常不是 |
| Reasoning RL | 在可验证问题上优化内部推理过程 | 是基础，但不一定是完整 Agentic RL |
| Tool-use SFT | 学习工具 schema 与调用格式 | 否 |
| Tool-use RL | 根据工具执行结果优化选择与参数 | 是 |
| Agent framework | 在推理时通过外部代码编排模型、工具和记忆 | 本身不是训练方法 |
| Multi-agent orchestration | 多个 Agent 协作完成任务 | 只有在策略通过交互反馈进行训练时，才属于 Multi-agent Agentic RL |
| Agentic RL | 策略在动态环境中进行多步决策，并通过环境结果学习 | 是 |

最容易混淆的是以下三种情况：

1. **模型会调用工具，不等于模型经过了 Agentic RL。** 工具调用可能只是 SFT 学到的模板。
2. **系统具有 Agent loop，不等于模型本身是 Agent-native。** 很多产品由外部 harness 负责规划、重试、记忆和路由。
3. **模型经过 RL，不等于它经过了 Agentic RL。** 只在数学或代码答案上做 RL，主要属于 reasoning RL；是否达到 Agentic RL，要看是否存在持续的环境交互和任务级反馈。

### 1.3 Agentic RL 真正学习的是什么

SFT 更容易教会模型“形式”：

- 工具调用的 XML 或 JSON 格式；
- 参数字段的名称与类型；
- 基本的 `ReAct` 结构；
- 常见任务的示范步骤；
- 如何输出最终答案。

Agentic RL 更适合塑造“策略”：

- 何时调用工具，何时直接回答；
- 使用搜索、代码执行器还是文件系统；
- 工具调用的粒度与顺序；
- 如何利用失败信息定位原因；
- 是否需要回退或重试；
- 如何分配有限的 token、工具调用和提交预算；
- 何时停止探索；
- 是否应当将任务拆分给其他 Agent。

这也是“被教出来”和“涌现出来”的关键区别：

```text
SFT 给出可行行为的先验
RL 通过结果反馈重新排序这些行为
```

如果环境足够丰富、reward 足够可靠，模型可能学到训练数据中没有直接标注的策略，例如动态调整搜索深度、发现错误后更换工具、在多个工具之间形成组合，以及根据任务难度改变推理长度。

但“涌现”不能被浪漫化。它不是无条件发生的奇迹，而是以下条件共同作用的结果：

1. 基座模型已经具备足够的语言、代码和世界知识；
2. 环境允许模型探索不同路径；
3. reward 能够区分有效路径和投机路径；
4. rollout 数量足以覆盖有希望的策略；
5. 优化过程能够承受长轨迹、稀疏 reward 和策略漂移。

---

## 2. Agentic RL 包括哪些内容

Agentic RL 不应只按“算法名称”分类。更合理的方式是沿着一个完整训练系统的因果链拆分。

### 2.1 任务与环境

环境决定模型能否真正学习“行动后果”。常见环境包括：

| 环境类型 | 示例 | 主要学习内容 |
|---|---|---|
| 数学、逻辑与 STEM | 数学题、定理证明、科学计算 | 内部推理、验证与策略切换 |
| 代码执行 | Python、CUDA、Triton、编译器 | 代码生成、运行、调试与性能优化 |
| 软件工程 | GitHub 仓库、测试套件、终端 | 多文件修改、测试反馈、故障恢复 |
| Web 与 GUI | 浏览器、WebArena、OSWorld | 页面理解、点击、输入、状态追踪 |
| 搜索与研究 | Web search、文档库、知识图谱 | 查询规划、证据搜集与来源核验 |
| 专业工作流 | 金融、法律、HR、办公软件 | 跨应用状态管理与交付 |
| 多 Agent | Agent Swarm、协作游戏 | 任务分解、委派、并行与汇总 |
| 视觉与多模态 | 图像、视频、图表、视觉工具 | 感知—行动—验证闭环 |
| 开放式世界 | 游戏、模拟社会、机器人环境 | 长期规划、资源管理与适应性 |

环境设计的核心不是“越真实越好”，而是：

> **环境必须保留对目标能力有意义的因果结构，同时提供可扩展、可隔离、可验证的训练接口。**

真实环境具有迁移价值，但成本高、噪声大、不可复现；纯模拟环境便于大规模训练，但可能被模型过拟合。因此，成熟方案通常采用：

```text
程序化模拟环境 → 合成任务 → 受控真实环境 → 真实产品反馈
```

### 2.2 状态、上下文与记忆

Agent 的状态不等于 prompt 文本。一个软件工程任务的真实状态至少包括：

```text
代码仓库
测试结果
当前分支
终端进程
依赖版本
文件修改历史
预算与时间
```

模型可能只看到其中一部分，因此 Agentic RL 天然具有部分可观测性。

记忆机制可以分为：

1. **上下文记忆**：直接将历史工具调用和观察保留在 context window；
2. **压缩记忆**：将长轨迹总结为结构化摘要；
3. **外部显式记忆**：将事实、任务状态和历史经验写入数据库或文件；
4. **隐式状态记忆**：通过 recurrent state、KV cache 或 latent state 保留信息；
5. **环境状态记忆**：让代码仓库、应用工作区或数据库本身成为外部记忆。

为什么记忆是 Agentic RL 的核心问题？因为长任务的困难通常不是单步决策，而是：

```text
早期观察到的信息
是否能在几十、几百甚至几千步之后仍然影响行动
```

如果模型无法保存或检索关键状态，RL 可能只会优化短期局部行为，而不会形成真正的长期策略。

### 2.3 动作空间与工具接口

Agent 的动作通常包含三层：

```text
Token 层：生成自然语言、代码和结构化参数
Tool-call 层：选择工具、设置参数、决定调用时机
Environment 层：修改文件、执行程序、提交结果、委派子 Agent
```

工具接口的设计直接改变了学习难度：

- 工具 schema 太复杂，模型会把容量消耗在格式错误上；
- 工具功能过强，模型可能绕过真正的推理过程；
- 工具反馈过弱，模型无法判断动作是否有效；
- 只有单一 harness，模型容易学会框架记忆而不是泛化策略；
- 工具接口变化太大，训练信号又会变得不稳定。

因此，K3 的 `Unified White-Box RL Environment` 让工具接口、system prompt、memory、context management 和 subagent 等组件可组合；这不是为了增加形式多样性，而是为了迫使模型学习“根据环境结构进行适应”，而不是死记某个工具 schema。

### 2.4 Reward 与 Verifier

Agentic RL 的 reward 大致分为四类。

#### 1. 可验证的结果奖励

例如：

```text
数学答案正确
测试全部通过
网页成功构建
数据库状态满足约束
```

优点是目标明确、偏差较小；缺点是通常稀疏，难以指出中间哪一步出了问题。

#### 2. 过程奖励

例如：

```text
搜索结果是否相关
工具参数是否有效
代码是否通过局部测试
是否减少了未解决问题
```

优点是能缓解稀疏 reward；缺点是过程指标容易被模型投机，且不一定与最终成功高度一致。

#### 3. Model-based reward

由另一个模型根据 rubric 评价报告、代码、网页或交付物。适合没有精确答案的开放任务，但容易出现 judge bias、长度偏好和 reward hacking。

#### 4. 混合奖励

将确定性检查与生成式评价结合：

$$
R
=
\alpha R_{\mathrm{verifiable}}
+\beta R_{\mathrm{model}}
-\lambda P_{\mathrm{cost}}
-\mu P_{\mathrm{hack}}
$$

这通常是实际 Agent 任务最可行的方案。确定性 verifier 负责硬约束，模型 judge 负责开放质量，成本惩罚控制过度思考，反作弊检测阻止模型走捷径。

Reward 设计的根本原则是：

> **奖励必须评价任务目标，而不是评价看起来像成功的表面行为。**

例如，网页任务不能只看模型是否生成了 HTML；需要检查是否能构建、是否运行、功能是否正确以及是否真正实现了目标。Kernel 任务不能只看代码是否执行；还需要同时检查数值正确性、性能和是否存在缓存输入、回放图等作弊行为。

### 2.5 轨迹采样与时间信用分配

完整 Agent trajectory 可能包含数百或数千个动作，但最终只有一个成功或失败信号。模型需要回答：

```text
哪些早期决策真正导致了最终成功？
哪些错误只是可恢复的局部错误？
```

这就是时间信用分配问题。

常见处理方式包括：

- 结果 reward 与中间过程 reward 结合；
- 对工具调用、子任务完成和最终提交分别评分；
- 使用 value model 或 critic 估计中间状态价值；
- 采用 group-relative advantage；
- 通过 partial rollout 或 trajectory reuse 提升有效样本量；
- 通过 curriculum 先训练短任务，再延伸到长任务；
- 通过 verifier 提供诊断反馈。

长轨迹还会造成数据陈旧：

```text
轨迹在旧策略下生成
→ 中途策略已经更新
→ 轨迹继续执行或被重新使用
→ 数据变成 stale / off-policy
```

Kimi K1.5/K3 采用 partial rollout 并配合策略稳定化；MiniMax-M1 通过 `CISPO` 对 importance sampling weight 进行裁剪；这些设计的共同目标不是让 RL 更“理论优雅”，而是让训练系统能够承受真实 Agent 轨迹的长度和异质性。

### 2.6 策略优化算法

Agentic RL 不要求某一种固定算法，但不同算法适应的问题不同。

| 算法 | 主要机制 | 优点 | 主要问题 | 适用情形 |
|---|---|---|---|---|
| `REINFORCE` | 直接用回报加权 log-probability gradient | 简洁、无需 value model | 方差高、长轨迹不稳定 | 小规模验证、结果 reward |
| `PPO` | clipped ratio 与 trust region | 稳定、成熟 | 需要 value model，系统复杂 | 复杂环境、较成熟 RL pipeline |
| `GRPO` | 用同一问题的 group 相对 reward 代替 value network | 适合 LLM reasoning，节省 critic 成本 | 对 reward 组质量敏感 | 数学、代码等可验证任务 |
| `DAPO 等变体` | 对 clipping、采样和长度处理进行改进 | 更适合大规模 reasoning RL | 超参数与稳定性仍敏感 | 大规模 reasoning |
| `PMD / mirror-descent 类方法` | 通过策略比率和参考策略约束更新 | 兼顾策略改进与稳定性 | 实现和理论理解门槛较高 | 长程、off-policy 或 Kimi 系列路线 |
| `CISPO` | 裁剪 importance sampling weights，而非直接裁剪 token update | 对长序列 RL 更有效率 | 公开验证范围较窄 | MiniMax-M1 的长程 RL |
| `On-Policy Distillation` | teacher 在 student 的 on-policy 状态上提供 dense guidance | 将多个专家策略统一到一个模型 | teacher 质量与路由策略关键 | MOPD、多专家合并 |

算法名称不是最重要的比较维度。真正决定效果的是：

```text
reward 是否可靠
轨迹是否足够丰富
环境是否具有因果反馈
优化器能否承受长程与陈旧数据
```

### 2.7 课程学习与能力扩展

Agentic RL 通常需要多种 curriculum：

1. **任务难度 curriculum**：从单步问题扩展到多步任务；
2. **轨迹长度 curriculum**：从短 context 扩展到长 context；
3. **推理预算 curriculum**：从 `max` effort 逐步得到 `high/low` effort；
4. **环境复杂度 curriculum**：从单工具扩展到多工具、多应用；
5. **验证严格度 curriculum**：从公开反馈逐步加入隐藏 verifier；
6. **模态 curriculum**：从文本扩展到图像、视频与 GUI。

课程学习的 Why 并不是“让模型循序渐进更容易学”，而是控制探索空间：

```text
如果一开始任务过难
→ 有效成功轨迹太少
→ reward 几乎全为零
→ policy gradient 无法提供方向
```

合理的 curriculum 让模型先获得可行策略，再将已学会的局部能力组合到更长、更复杂的任务中。

### 2.8 训练基础设施与部署

长程 Agentic RL 的训练成本不只来自模型 forward/backward，还来自：

- 大量 rollout 生成；
- 工具执行与环境启动；
- KV cache 持久化；
- 有状态 sandbox 的暂停、恢复和复制；
- reward model 或 verifier 的运行；
- 多个不同长度轨迹之间的调度不平衡。

因此，训练基础设施本身会影响可研究的算法空间：

```text
环境启动慢
→ rollout 数量下降
→ 探索不足

KV cache 无法复用
→ 长轨迹成本急剧上升
→ 训练被迫缩短 horizon

无法暂停恢复
→ 少数慢轨迹拖垮整个 batch
→ 不能有效使用 partial rollout
```

Kimi K1.5/K3 的 partial rollout、External KV Cache Pool、AgentENV 和 co-located RL，MiniMax-M1 的 Lightning Attention 与 `CISPO`，GLM 的 `slime`，都说明 Agentic RL 是算法—系统协同问题。

---

## 3. Agentic RL 如何实施：一条可复用的工程流程

下面是一条适用于代码 Agent、研究 Agent 和工具 Agent 的通用流程。

### Step 1：定义目标能力，而不是先定义模型

先明确要训练的能力：

```text
代码修复？
网页操作？
深度研究？
多应用办公？
多 Agent 分工？
视觉工具使用？
```

不同能力对应不同环境和 reward。若目标模糊，后续所有数据与评价都会漂移。

### Step 2：构造有因果反馈的环境

环境至少需要提供：

1. 初始状态；
2. 明确目标；
3. 可执行动作；
4. 观察反馈；
5. 最终 verifier；
6. 资源或步骤预算。

一个有效任务不能只是“请写一段看起来正确的代码”，而应当让代码进入真实执行、测试和反馈闭环。

### Step 3：设计 action protocol

明确模型何时生成：

- 普通文本；
- 思考内容；
- 工具调用；
- 工具参数；
- 最终提交；
- 子 Agent 委派。

Action protocol 既决定训练数据格式，也决定模型在推理时的自由度。过度约束会限制探索，过度开放则会增加 reward 噪声和安全风险。

### Step 4：准备冷启动策略

通常有三种方式：

1. `SFT`：使用成功轨迹与高质量示范；
2. `in-context examples`：在 rollout prompt 中提供工具使用示例；
3. 直接从 base model 开始 RL：适合已经具备较强能力且 reward 极其清晰的任务。

实际工程中，SFT 往往更稳健，因为它先解决工具格式、基本交互与执行协议，再把 RL 资源用于策略优化。

### Step 5：构造任务分布

任务生成器要同时控制：

- 领域分布；
- 难度分布；
- 任务长度；
- 工具组合；
- 真实材料来源；
- 可验证性；
- 长尾场景。

如果只重复固定模板，模型可能记住题型；如果任务完全随机，成功率又可能过低。知识图谱引导、程序化生成、真实材料检索和环境状态随机化是常见组合。

### Step 6：配置 rollout 系统

需要决定：

- 一次采样多少条轨迹；
- 是否使用 group sampling；
- 是否允许暂停和恢复；
- 是否复用 prefix/KV cache；
- 是否并行执行工具；
- 环境如何隔离；
- 如何记录完整状态和失败原因。

对于长程任务，rollout 系统往往比 policy optimizer 更容易成为瓶颈。

### Step 7：设计 reward 与反作弊机制

至少要回答：

```text
成功是什么？
失败是什么？
局部进展如何定义？
输出过长是否应惩罚？
模型能否访问 verifier？
模型能否修改测试或评测逻辑？
```

在代码和网页任务中，verifier 必须与 agent 隔离；在开放式文本任务中，judge 需要 rubric、长度预算和多候选比较；在资源敏感任务中，工具调用次数、运行时间和 token 数也应纳入成本惩罚。

### Step 8：选择 policy optimization

可按 reward 与 horizon 选择：

```text
短、可验证、样本独立
→ GRPO / DAPO 类方法

长、交互复杂、需要稳定 critic
→ PPO / value-based pipeline

轨迹极长、存在重要性比率问题
→ CISPO、per-token regularization、mirror-descent 类方法

多个领域专家需要统一
→ On-Policy Distillation / MOPD
```

### Step 9：加入 curriculum 与预算控制

训练初期先扩大成功样本比例，再逐步提升任务难度、轨迹长度和环境多样性。若不控制预算，模型可能学会：

```text
用更多 token 换成功
用更多工具调用换成功
用更多提交次数试错
```

这在训练指标上可能有效，在真实产品中却不可接受。

### Step 10：进行跨环境、跨 harness 评价

不能只在训练环境中测 success rate。还需要测：

- 未见过的工具 schema；
- 未见过的任务模板；
- 未见过的应用；
- 不同 context 管理机制；
- 不同资源预算；
- 隐藏 verifier；
- 任务行为质量；
- 安全与 reward hacking。

Agent 的泛化能力不是“换一个 prompt 还能答对”，而是“换一个环境仍然能够通过观察学习如何行动”。

### Step 11：统一、蒸馏与部署

如果训练了领域专家或不同 effort 专家，需要决定：

- 是否保留多个模型；
- 是否路由；
- 是否用 on-policy distillation 合并；
- 是否训练 draft model；
- 是否使用 QAT；
- 是否做上下文压缩与记忆管理。

K3 的 MOPD 代表了“训练时专业化、部署时统一化”的一种方案。

---

## 4. 主流模型的 Agentic RL 路线

### 4.1 DeepSeek-R1：以可验证推理为中心的 RL

#### 训练路线

DeepSeek-R1 的公开路线可以概括为：

```text
DeepSeek-V3 基座
    ↓
R1-Zero：不经过 SFT，直接进行大规模 RL
    ↓
观察到推理、反思、验证与策略切换等行为
    ↓
R1：加入 cold-start 数据
    ↓
reasoning-oriented RL
    ↓
rejection sampling + SFT
    ↓
面向多类任务的最终 RL
```

R1-Zero 的关键价值是证明：在基座模型和 reward 足够合适时，RL 可以在没有人工推理轨迹的情况下推动长链推理行为形成。官方论文强调了 self-reflection、verification 和 dynamic strategy adaptation 等行为。

#### 为什么这条路线有效

数学和代码题具有相对清晰的验证器：

```text
答案是否正确
程序是否通过测试
格式是否满足要求
```

这使得 reward 的语义较稳定，模型可以通过大量试错学习哪些推理路径更有可能成功。相比开放式 Agent 任务，推理任务的环境状态更简单、动作空间更窄、轨迹更短，因此更适合作为大规模 RL 的起点。

#### 它属于哪一种 Agentic RL

严格说，R1 主要是 **reasoning RL**，而不是完整的 environment-centric Agentic RL：

- 主要环境是问题与验证器；
- 外部工具、长期记忆和多应用状态不是核心；
- 任务更多是“通过内部推理得到可验证答案”，而不是“改变外部环境状态”。

因此，R1 的贡献是提供了 Agentic RL 的重要策略底座：

```text
验证 → 反思 → 重新尝试 → 提交
```

但它不能单独解决代码仓库维护、浏览器操作、跨应用工作流和长期记忆问题。

#### 优势与局限

| 维度 | 判断 |
|---|---|
| 优势 | reward 清晰，推理行为容易规模化，开放论文便于复现 |
| 优势 | 证明了不依赖完整人工 CoT，也可能形成复杂推理策略 |
| 局限 | 外部工具与真实环境交互较弱 |
| 局限 | 长期状态、工具选择、任务分解不是主要训练对象 |
| 适合定位 | reasoning substrate，而非完整 Agent OS |

参考：[DeepSeek-R1 技术报告](https://arxiv.org/abs/2501.12948)。

### 4.2 Kimi K1.5：把 RL 的主要变量从“模型大小”扩展到“上下文与轨迹”

#### 训练路线

Kimi K1.5 的公开方法重点包括：

- 长 `CoT` RL；
- 数学、代码和视觉任务；
- 128K 级别的 RL context；
- partial rollout；
- code sandbox；
- `long2short`，将长推理能力迁移到较短推理模式。

其核心思想是：

> **RL 的 scaling 不只发生在模型参数和训练 FLOPs 上，也发生在可容纳的轨迹长度上。**

#### 为什么要做 partial rollout

长 `CoT` 轨迹的生成成本高，且不同问题的长度差异大。若每次都从头生成完整轨迹，计算大量浪费在已经生成过、但仍然有用的前缀上。partial rollout 通过复用已有轨迹片段，降低长轨迹采样成本，并使更长的 reasoning horizon 变得可训练。

这为后来的 K2/K3 长程 Agentic RL 提供了系统基础，但 K1.5 本身仍以 reasoning RL 为主，外部环境交互尚未像 K2/K3 那样成为核心。

#### 价值

K1.5 的重要性不在于它已经是完整 Agent，而在于它把以下问题带入主流实践：

```text
RL context 能否持续扩展？
长轨迹如何高效采样？
如何将高质量 long-CoT 压缩为低成本 short-CoT？
```

参考：[Kimi K1.5 技术报告](https://arxiv.org/abs/2501.12599)、[官方代码仓库](https://github.com/MoonshotAI/Kimi-k1.5)。

### 4.3 Kimi K2：从 reasoning RL 进入真正的 Agent-environment RL

#### 训练路线

K2 的 Post-Training 由两条主线组成：

```text
大规模 Agentic data synthesis
    ↓
SFT：学习工具调用形式与基本执行轨迹
    ↓
Unified RL
    ├─ verifiable rewards
    └─ self-critic rubric rewards
```

K2 的任务覆盖工具使用、复杂指令遵循、faithfulness、代码与软件工程、安全以及其他开放式任务。训练同时使用真实和合成环境。

#### 为什么需要大规模 Agentic data synthesis

工具调用数据存在两个问题：

1. 真实专家轨迹昂贵且规模有限；
2. 固定工具和固定任务容易导致过拟合。

K2 的合成流程通过工具规格、任务、Agent 角色和多轮执行环境，构造大量不同轨迹，再经过质量评价与过滤。SFT 先建立基本可行策略，RL 再根据结果反馈调整策略。

这实际上将 SFT 与 RL 的职责分开：

```text
SFT：降低探索门槛，学习交互协议
RL：选择更有效的工具策略和任务路径
```

#### 为什么同时使用 verifiable reward 与 self-critic reward

可验证 reward 适合：

```text
答案、测试、构建、状态约束
```

但很多 Agent 交付物没有精确答案，例如复杂写作、研究报告和开放式任务。K2 因此增加 self-critic rubric reward，让模型根据结构化 rubric 评价候选结果。

两者的组合是一个现实折中：

```text
verifier 提供硬约束
critic 提供开放质量
```

但 critic reward 也引入了长度偏好、评价偏差和 reward hacking，因此 K2 进一步设计了 rubric、闭环 critic refinement 与预算控制。

#### K2 的定位

K2 是从“推理模型”向“通用 Agent 模型”转变的关键节点。其核心变化不是增加一个工具调用接口，而是将：

```text
任务生成
+ 工具环境
+ 结果验证
+ RL 更新
```

组织成统一训练闭环。

参考：[Kimi K2 技术报告](https://arxiv.org/abs/2507.20534)、[Kimi K2 官方介绍](https://www.kimi.com/blog/kimi-k2.html)。

### 4.4 Kimi K2.5：将多模态与多 Agent 协作纳入 RL

#### 训练路线

K2.5 在 K2 基础上进一步加入：

- 原生文本—视觉联合预训练；
- zero-vision SFT；
- joint multimodal RL；
- Agent Swarm；
- `PARL`（Parallel-Agent Reinforcement Learning）；
- GUI、视觉推理和计算机使用任务。

其重要变化是，视觉不再只是输入模态，而成为 Agent 观察环境、执行动作和获取反馈的一部分。

#### 为什么需要 joint multimodal RL

如果视觉模型先独立训练、再通过少量数据接入语言模型，容易出现：

- 视觉理解与语言推理之间不一致；
- 模型会“看见”，但不会利用视觉结果规划行动；
- 视觉工具调用与文本工具调用的策略割裂。

联合 RL 使模型能够在同一个策略中学习：

```text
看图 → 选择工具 → 执行操作 → 观察新视觉状态 → 调整策略
```

#### 为什么需要 PARL

多 Agent 系统常见的问题是“serial collapse”：即使任务可以并行，模型仍然倾向于顺序执行所有子任务，导致时间和上下文成本增加。

PARL 的训练目标是让模型学会：

- 识别可并行的子任务；
- 动态决定是否创建子 Agent；
- 分配任务与资源；
- 汇总异步返回结果；
- 在并行收益不足时避免过度拆分。

这里的 reward 不应只评价最终答案，还要考虑并行执行的效率与关键步骤。否则模型可能通过创建大量无效子 Agent 获得表面上的“复杂度”，却没有真正提升完成质量。

#### 优势与局限

| 维度 | 判断 |
|---|---|
| 优势 | 将视觉、工具和多 Agent 协作放入统一训练范式 |
| 优势 | 可训练主动分解和并行执行，而不只是外部编排 |
| 局限 | 多 Agent reward 更难归因，协调成本更高 |
| 局限 | 需要大量可恢复、可并发的环境基础设施 |

参考：[Kimi K2.5 技术报告](https://arxiv.org/abs/2602.02276)、[Kimi K2.5 官方介绍](https://www.kimi.com/blog/kimi-k2-5)。

### 4.5 Kimi K3：长程、跨域、可部署的 Agentic RL

K3 将 K2/K2.5 的方向进一步规模化，其 Post-Training 主线为：

```text
SFT：建立 Agent 冷启动策略
    ↓
3 个领域 × 3 个推理努力等级
    ↓
9 个专门专家
    ↓
MOPD：统一到一个模型
```

三个领域是：

- `General Tasks`；
- `General Agents`；
- `Coding Agents`。

三个 effort level 是：

- `low`；
- `high`；
- `max`。

K3 的 Agentic RL 环境包括：

- Unified White-Box RL Environment；
- Knowledge-Graph-Guided Task Synthesis；
- verifiable search 与专业知识工作；
- Kernel Optimization；
- Personal Assistant；
- Autonomous Execution Tasks；
- Web Development；
- 视觉推理与工具使用。

#### K3 的 Why

K3 试图同时解决四个矛盾：

1. **专业化与统一化**：先训练九个专家，再用 MOPD 合并；
2. **质量与成本**：通过 effort budget 训练不同推理预算；
3. **长程轨迹与训练吞吐**：通过 partial rollout、KV cache pool 和 sandbox pause/resume；
4. **训练能力与部署约束**：从 SFT 开始进行 QAT，并微调 EAGLE-3 draft model。

K3 的关键不是某一个 RL 算法，而是把 Agentic RL 做成完整系统：

```text
任务生成
→ 多样环境
→ 长程 rollout
→ verifier / GRM
→ 领域与 effort 专家
→ MOPD 统一
→ 量化与推理加速
```

#### K3 相比 K2.5 的主要推进

| 维度 | K2.5 | K3 |
|---|---|---|
| 主要扩展方向 | 多模态与 Agent Swarm | 长程、多领域与部署规模 |
| RL 组织 | 联合文本—视觉 RL、PARL | 三领域 × 三 effort 专家、MOPD |
| 环境 | 视觉、工具、多 Agent | 搜索、代码、个人助理、AET、Web、kernel |
| 长度 | 长程 Agent | 1M context 与百万级累计轨迹 |
| 工程重点 | 多模态与并行 Agent | partial rollout、KV pool、AgentENV、部署感知训练 |

详细分析见 [[Kimi K3 论文阅读报告]] 的 [[Kimi K3 论文阅读报告#Post-Training]]。

### 4.6 MiniMax-M1：以长上下文 RL 效率为中心

MiniMax-M1 不是以通用 Agent harness 的广度为主要卖点，而是将长上下文、长推理和代码环境 RL 结合起来。

#### 训练路线

```text
Continual Pre-Training
    ↓
Focused SFT
    ↓
Large-scale RL
    ├─ 数学与可验证推理任务
    ├─ model-based feedback 的通用任务
    └─ sandbox-based software engineering
```

其两个核心技术是：

- `Lightning Attention`：降低长序列 RL 的计算与状态成本；
- `CISPO`：裁剪 importance sampling weights，而不是直接裁剪 token update。

#### 为什么 CISPO 重要

长轨迹下，策略比率可能出现极端值。直接裁剪 token update 可能丢失重要的策略改进信号；裁剪 importance sampling weight 则试图在控制方差的同时保留更有价值的更新。

这不是一个抽象的算法偏好，而是由长序列 RL 的数值问题驱动的选择：

```text
轨迹越长
→ token-level ratio 越多
→ 极端比率出现概率越高
→ 稳定化策略越重要
```

#### 定位

MiniMax-M1 代表了：

> **用高效序列架构和专门 RL 优化器，把长推理与代码 Agent 训练扩展到更大 horizon。**

它的公开资料对 sandbox 软件工程有明确说明，但在通用办公、知识工作、多 Agent 协作等方面的公开细节不如 K3 丰富。

参考：[MiniMax-M1 技术报告](https://arxiv.org/abs/2506.13585)。

### 4.7 Qwen3：统一 thinking / non-thinking，并以预算控制部署

Qwen3 的主要公开训练主线是：

```text
长 CoT cold start
    ↓
reasoning RL
    ↓
thinking / non-thinking mode integration
    ↓
general RL
```

Qwen3 将 thinking mode 与 non-thinking mode 统一在同一个模型中，并引入 thinking budget，使用户能够在质量和延迟之间进行控制。

#### Why

传统做法往往维护两个模型：

```text
普通聊天模型
专门 reasoning 模型
```

这样做会产生模型切换、部署和能力迁移问题。Qwen3 的统一模式试图让同一策略在不同预算下工作：

```text
复杂任务 → 允许更长思考
简单任务 → 直接响应
```

但需要注意：Qwen3 技术报告公开得最充分的是 reasoning RL、模式融合和预算机制；它不是像 K3 那样把长程外部环境、持久状态和复杂工具体系作为报告中心。因此，Qwen3 更接近“reasoning-first + 部分 Agent 能力”，不宜直接与 K3 的完整 Agent-environment RL 等同。

参考：[Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)、[Qwen3 官方仓库](https://github.com/QwenLM/Qwen3)。

### 4.8 Qwen3-Coder-Next：面向软件工程的专门 Agentic RL

Qwen3-Coder-Next 更接近严格意义上的 coding Agentic RL。官方资料明确提到其训练包含：

- 大规模可执行任务合成；
- 环境交互；
- 多步代码编辑；
- 工具使用；
- 执行失败后的故障恢复；
- 基于真实环境结果的 RL。

#### Why

代码 Agent 的核心能力不是一次生成正确代码，而是：

```text
理解仓库 → 定位问题 → 修改多个文件 → 运行测试
→ 读取错误 → 继续修复 → 直到满足 verifier
```

因此，静态代码数据无法充分训练故障恢复和状态跟踪。只有让模型在可执行环境中实际修改和运行，reward 才能反映“代码是否真正解决问题”。

Qwen3-Coder-Next 的价值在于将 agentic training 与较低的激活参数结合，探索：

```text
更小的推理成本
是否可以通过更专门、更真实的环境 RL
获得较强的 coding agent 能力
```

其局限则是任务域较窄，不能直接推断其在通用研究、跨应用办公和多 Agent 协作上的能力。

参考：[Qwen3-Coder-Next 技术报告](https://arxiv.org/abs/2603.00729)、[Qwen3-Coder 官方仓库](https://github.com/QwenLM/Qwen3-Coder)。

### 4.9 GLM-4.5：ARC 统一目标下的专家迭代与 RL

GLM-4.5 将目标概括为 `Agentic, Reasoning, and Coding`（ARC），采用多阶段训练、专家模型迭代和 RL，覆盖 agentic、reasoning 与 coding 任务。

从公开资料能够确认：

```text
大规模预训练
    ↓
多阶段 post-training
    ↓
expert model iteration
    ↓
reinforcement learning
```

其公开报告强调 Agent、推理与代码能力的统一，而不是只建立一个数学 reasoning 模型。这条路线与 K3 的共同点是：

- 都将 Agent、reasoning 和 coding 作为相互增强的能力；
- 都依赖多阶段 post-training；
- 都强调大规模 RL 基础设施。

差异在于，GLM-4.5 公开的 Agentic RL 细节相对高层，未像 K3 一样完整披露统一白盒环境、任务合成、AET、MOPD 等组成。因此，对 GLM-4.5 的具体 reward 结构、长程轨迹策略和环境覆盖，应保持保守判断。

参考：[GLM-4.5 技术报告](https://arxiv.org/abs/2508.06471)、[GLM-4.5 官方说明](https://z.ai/blog/glm-4.5)。

### 4.10 OpenAI o1 / o3 / o4-mini：从 reasoning RL 走向模型内生工具使用

#### o1

OpenAI 公开说明 o1 通过大规模 RL 训练模型更有效地使用内部 chain of thought，并学习改进思考、尝试不同策略和识别错误。

因此，o1 的公开定位更接近：

```text
大规模 reasoning RL
→ 内部长程思考
→ 自我检查与策略切换
```

而不是完整的外部环境 Agentic RL。

#### o3 / o4-mini

OpenAI 对 o3/o4-mini 的公开说明进一步强调：

- 模型在思考过程中使用工具；
- 能够搜索网页、运行 Python、分析图像和文件；
- 经过训练后能够判断何时以及如何使用工具；
- 工具行为与 reasoning chain 结合。

这代表了从：

```text
模型先思考，外部系统再调用工具
```

到：

```text
模型本身在策略中决定是否调用工具
```

的变化。

#### Why

如果工具调用完全由外部 harness 决定，模型只需要对工具返回进行解释；如果工具选择也被纳入策略，模型可以学习：

- 哪些问题需要搜索；
- 何时调用 Python；
- 如何组合多个工具；
- 何时停止工具使用并输出答案。

OpenAI 的公开材料说明了能力和高层训练方向，但没有公开完整的任务生成、reward、rollout、verifier 或优化器细节。因此，外部只能确认其属于“reasoning + integrated tool-use RL”路线，不能严谨复原其完整 Agentic RL recipe。

参考：[Learning to reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)、[Introducing OpenAI o3 and o4-mini](https://openai.com/index/introducing-o3-and-o4-mini/)、[o3/o4-mini System Card](https://openai.com/index/o3-o4-mini-system-card/)。

### 4.11 Claude 3.7 / Claude 4：强 Agent runtime，但训练细节公开有限

Anthropic 公开的 Claude 3.7/4 能力包括：

- extended thinking；
- thinking budget；
- thinking 中调用工具；
- 并行工具执行；
- 文件与记忆能力；
- 长时间 coding agent workflow。

从产品能力看，Claude 4 明显属于强 Agent 系统；但需要把“模型能力”和“训练方法”分开。

公开资料更充分地说明了：

```text
Claude 如何在推理时使用工具
Claude Code 如何通过 harness 执行任务
Claude 如何进行 extended thinking
```

而没有公开完整的：

```text
Agentic RL 任务分布
环境构造
reward 组合
policy optimization
trajectory replay
```

Anthropic 后续公开的对齐研究还指出，在 Claude 4 训练时期，其大部分 alignment training 仍是标准的 chat-based RLHF 数据，并不包含 agentic tool use。这一事实不能推出 Claude 没有任何 Agentic RL，而只能说明：

> **不能把 Claude 的 Agent 能力简单归因于“对齐训练本身使用了大量 Agentic RL”。**

更稳妥的判断是：Claude 采用了强模型能力、extended thinking、工具运行时、记忆和安全训练的混合路线，但其 Agentic RL 训练配方尚未公开到可复现程度。

参考：[Claude 3.7 Sonnet](https://www.anthropic.com/news/claude-3-7-sonnet)、[Introducing Claude 4](https://www.anthropic.com/news/claude-4)、[Teaching Claude why](https://www.anthropic.com/research/teaching-claude-why)。

### 4.12 Gemini 2.5 / Deep Think：公开信息更偏 reasoning RL

Google DeepMind 公开说明，Gemini 的 Deep Think 版本使用了新的 RL 方法，并利用更多多步推理、问题求解和定理证明数据。

因此，公开证据支持的判断是：

```text
Gemini Deep Think = 强 reasoning RL + 多步问题求解
```

但其通用 Agentic RL 环境、工具调用策略训练、长程 rollout 和 reward 结构并未充分公开。Gemini 产品可以通过工具和外部系统完成 Agent 任务，但这不等于论文层面已经公开了模型内部如何通过 Agentic RL 学会这些行为。

这一案例说明一个重要边界：

> **产品具备 Agent 能力，不代表训练方法已经公开证明是 Agentic RL；闭源系统必须将“观察到的能力”和“已披露的训练机制”分开。**

参考：[Gemini Deep Think 官方说明](https://deepmind.google/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/)。

---

## 5. 主流模型的 Agentic RL 异同与优劣势

### 5.1 按训练对象比较

| 模型 | 主要训练对象 | 外部环境深度 | 多步工具策略 | 公开程度 |
|---|---|---:|---:|---:|
| DeepSeek-R1 | 内部推理与验证 | 低 | 低—中 | 高 |
| Kimi K1.5 | 长 CoT、代码与视觉推理 | 中 | 中 | 高 |
| Kimi K2 | 工具使用、代码与通用 Agent | 高 | 高 | 高 |
| Kimi K2.5 | 多模态 Agent、Agent Swarm | 高 | 高 | 高 |
| Kimi K3 | 长程、多领域、多 effort Agent | 很高 | 很高 | 很高 |
| MiniMax-M1 | 长推理、代码 sandbox、长上下文 | 中—高 | 中—高 | 高 |
| Qwen3 | reasoning、模式切换、预算控制 | 中 | 中 | 高 |
| Qwen3-Coder-Next | 软件工程 Agent | 很高，但领域较窄 | 很高 | 高 |
| GLM-4.5 | ARC：Agent、reasoning、coding | 中—高 | 高 | 中—高 |
| OpenAI o1 | 内部 reasoning | 低 | 低 | 低 |
| OpenAI o3/o4-mini | reasoning + integrated tools | 高 | 高 | 中 |
| Claude 4 | extended thinking + Agent runtime | 高 | 高 | 低 |
| Gemini Deep Think | 多步 reasoning、定理证明 | 未充分公开 | 未充分公开 | 低 |

这里的“公开程度”指训练技术披露程度，不代表模型权重、API 或产品是否开放。

### 5.2 按 reward 体系比较

| 路线 | 主要 reward | 优势 | 风险 |
|---|---|---|---|
| Reasoning-first | 规则验证、答案正确性、格式 | 稳定、可规模化 | 容易停留在内部推理 |
| Coding Agent | 测试、构建、运行结果、性能 | 反馈真实，因果关系强 | 运行成本高，容易 reward hacking |
| General Agent | 模型 judge、rubric、任务状态 | 可覆盖开放式工作 | judge bias、长度偏好、评价不稳定 |
| Multi-agent | 最终任务质量 + 并行效率 + 协作质量 | 能学习任务分解与并行 | 时间信用分配和 credit assignment 更困难 |
| Multimodal Agent | 视觉结果、工具执行、环境状态 | 形成感知—行动闭环 | 视觉噪声、环境成本和跨模态对齐 |

### 5.3 按“能力写入模型”与“能力留在框架”比较

| 路线 | 代表 | Agent 能力主要在哪里 | 优势 | 局限 |
|---|---|---|---|---|
| **训练派** | Kimi K2/K3、Qwen3-Coder-Next、部分 GLM | 模型权重与策略中 | 换 harness 后仍可能保留执行能力 | 训练昂贵，reward 与安全更难 |
| **框架派** | Claude Code、Codex 等产品系统 | 模型外部的 harness、工具和记忆 | 产品迭代快，行为可编排 | 换环境后模型本身未必泛化 |
| **混合派** | OpenAI o3/o4-mini、Claude 4、Kimi K2.5 | 模型内生策略 + 外部系统 | 能力与产品工程协同 | 外部很难拆分模型贡献和系统贡献 |

这个区分不能被绝对化。训练派也需要 harness，框架派也可能使用模型 RL；真正的差别在于：

```text
模型是否在训练时经历了与部署相似的环境反馈
以及关键的工具决策是否被写入参数
```

### 5.4 综合评价

#### DeepSeek-R1

- **最强项**：可验证推理、反思与策略切换；
- **根本优势**：reward 清晰，RL 机制公开；
- **主要短板**：完整 Agent 环境、长期记忆和工具生态不是其训练中心；
- **适合定位**：Agent 的 reasoning core。

#### Kimi K3

- **最强项**：长程、多领域、工具环境、视觉、代码和部署闭环；
- **根本优势**：将任务、环境、reward、专家训练和基础设施统一设计；
- **主要短板**：系统复杂，训练成本高；Agent behavior 和 research-level reasoning 仍可能存在差距；
- **适合定位**：目前公开资料中最完整的通用 Agentic RL 系统之一。

#### MiniMax-M1

- **最强项**：长上下文、长推理与 RL 效率；
- **根本优势**：Lightning Attention 与 `CISPO` 联合解决长轨迹成本和优化稳定性；
- **主要短板**：通用 Agent 任务的公开覆盖不如 K3 完整；
- **适合定位**：长 horizon reasoning/coding RL。

#### Qwen3 / Qwen3-Coder-Next

- **最强项**：开放权重、模式切换、预算控制；`Qwen3-Coder-Next` 在软件工程环境中更接近专门 Agent；
- **根本优势**：将 reasoning 训练与较低推理成本结合；
- **主要短板**：通用 Agentic RL 与多应用长期任务的公开细节相对有限；Coder 版本领域更窄；
- **适合定位**：开放生态中的 reasoning-first 与 coding-agent 两条分支。

#### GLM-4.5

- **最强项**：将 Agent、reasoning、coding 纳入统一 ARC 目标；
- **根本优势**：多阶段训练与专家迭代；
- **主要短板**：公开 Agentic RL 机制的细节不如 K3、MiniMax 或 Qwen3-Coder-Next 充分；
- **适合定位**：统一 ARC 能力的开放模型路线。

#### OpenAI o3/o4-mini

- **最强项**：推理与工具使用深度结合，产品工具覆盖广；
- **根本优势**：将“是否以及如何调用工具”纳入模型策略；
- **主要短板**：训练任务、reward 和环境不透明，外部难以复现；
- **适合定位**：闭源模型内生工具策略的代表。

#### Claude 4

- **最强项**：长时间 coding、工具调用、记忆与 Agent runtime；
- **根本优势**：模型能力与产品 harness、工具和安全系统协同；
- **主要短板**：无法从公开资料中严格判断其 Agentic RL 的规模和具体贡献；
- **适合定位**：强 Agent 产品系统，而非公开 Agentic RL 研究范式。

#### Gemini Deep Think

- **最强项**：多步推理、科学与定理证明；
- **根本优势**：Google DeepMind 在大规模 RL 与搜索式推理上的长期积累；
- **主要短板**：通用 Agentic RL 的训练细节公开不足；
- **适合定位**：reasoning RL 与产品工具系统的闭源路线。

### 5.5 为什么不存在一个“绝对最优”的 Agentic RL 配方

Agentic RL 的最优方案取决于任务结构：

```text
数学证明
→ 规则 verifier + reasoning RL 更重要

代码修复
→ sandbox + 测试 + 故障恢复更重要

深度研究
→ 搜索环境 + 来源评价 + 长期记忆更重要

多 Agent 协作
→ 分解、并行、通信和资源 reward 更重要

办公自动化
→ 多应用状态与长期事件流更重要
```

因此，“谁的 Agentic RL 更强”必须拆成至少五个问题：

1. 在什么环境中更强？
2. reward 是结果正确性还是开放式质量？
3. 是否考虑工具成本与执行延迟？
4. 能否迁移到未见过的 harness？
5. 能力来自模型参数，还是来自外部系统？

只看一个 Agent benchmark，不能回答这些问题。

---

## 6. Agentic RL 的核心开放问题

### 6.1 长程信用分配

一条轨迹包含数百步时，最终成功很难精确归因到每一个动作。未来需要更好的：

- hierarchical credit assignment；
- subgoal value；
- event-level reward；
- hindsight feedback；
- trajectory-level causal attribution。

### 6.2 Reward hacking 与 verifier robustness

更强的 Agent 会主动寻找环境漏洞。常见方式包括：

- 修改测试；
- 缓存输入；
- 读取隐藏答案；
- 绕过真实执行；
- 让 judge 看到伪造产物；
- 通过冗长输出影响模型评分。

因此，verifier 不能只追求“容易运行”，还要具备隔离、随机化、隐藏测试和反作弊能力。

### 6.3 从单任务成功到跨环境泛化

真正有价值的 Agent 不应只会在一个 benchmark 的固定 harness 中完成任务。需要研究：

```text
工具 schema 泛化
环境动力学泛化
任务目标泛化
资源预算泛化
跨模态泛化
```

K3 的 Unified White-Box RL 是一个方向，但是否足以解决真实产品环境的分布偏移，仍需要更多公开实验。

### 6.4 Agentic RL 的 scaling law

预训练已经有较成熟的 scaling law；Agentic RL 仍缺乏同等清晰的规律。至少存在多个互相耦合的 scaling 变量：

```text
模型规模
× RL FLOPs
× rollout 数量
× 轨迹长度
× 环境多样性
× reward 质量
× verifier 覆盖
```

单纯增加 RL FLOPs 不一定有效。如果任务分布、reward 和环境没有同步扩展，模型可能只是在固定 benchmark 上过拟合。

### 6.5 Agent 的安全与价值稳定性

Agent 的行动能力越强，错误 reward 的后果越大。安全训练必须覆盖：

- 工具权限；
- 外部系统访问；
- 隐私与数据边界；
- 任务目标冲突；
- reward hacking；
- 长期自主行为；
- 多 Agent 之间的串联风险。

安全不能只在最终回答阶段检查，而应进入环境、动作、状态转移和 verifier。

### 6.6 模型内生能力与外部 harness 的边界

未来系统会继续在两条路线上发展：

```text
更强的 Agent-native policy
更强的外部 orchestration framework
```

真正重要的问题不是哪一方“取代”另一方，而是哪些能力应写入模型，哪些能力应保留在系统：

| 更适合写入模型 | 更适合保留在系统 |
|---|---|
| 工具选择判断 | 权限与审计 |
| 失败恢复策略 | 资源调度 |
| 任务分解先验 | 记忆存储与检索 |
| 输出与行动风格 | 业务规则 |
| 跨工具组合能力 | 环境隔离与安全策略 |

---

## 7. 参考资料与学习路径

### 7.1 总体综述

1. [The Landscape of Agentic Reinforcement Learning for LLMs: A Survey](https://arxiv.org/abs/2509.02547)  
   当前最适合作为总览入口。重点阅读 POMDP 形式化、能力分类、任务分类、环境与开放问题。
2. [Agentic Large Language Models, a survey](https://arxiv.org/abs/2503.23037)  
   从 reasoning、acting、interacting 三个角度理解 Agent 系统。
3. [Tool Learning with Large Language Models: A Survey](https://arxiv.org/abs/2405.17935)  
   适合补充工具学习、工具接口和调用策略的历史演进。

### 7.2 RL 与 reasoning 基础

1. [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html)
2. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
3. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300)
4. [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
5. [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783)

### 7.3 Tool use 与环境 Agent

1. [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
2. [Toolformer](https://arxiv.org/abs/2302.04761)
3. [ToolRL: Reward is All Tool Learning Needs](https://arxiv.org/abs/2504.13958)
4. [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536)
5. [SWE-agent](https://arxiv.org/abs/2405.15793)
6. [SWE-bench](https://arxiv.org/abs/2310.06770)
7. [WebArena](https://arxiv.org/abs/2307.13854)
8. [OSWorld](https://arxiv.org/abs/2404.07972)
9. [τ-bench](https://arxiv.org/abs/2406.12045)

### 7.4 长程 RL 与系统

1. [Kimi K1.5](https://arxiv.org/abs/2501.12599)
2. [Kimi K2](https://arxiv.org/abs/2507.20534)
3. [Kimi K2.5](https://arxiv.org/abs/2602.02276)
4. [MiniMax-M1](https://arxiv.org/abs/2506.13585)
5. [AgentENV](https://github.com/kvcache-ai/AgentENV)
6. [slime：GLM 系列 RL 训练框架](https://github.com/THUDM/slime)
7. [OpenKimi：Kimi RL 相关复现项目](https://github.com/horizon-llm/OpenKimi)

### 7.5 代表模型技术报告

1. [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
2. [Qwen3-Coder-Next Technical Report](https://arxiv.org/abs/2603.00729)
3. [GLM-4.5](https://arxiv.org/abs/2508.06471)
4. [OpenAI o1：Learning to reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)
5. [OpenAI o3/o4-mini System Card](https://openai.com/index/o3-o4-mini-system-card/)
6. [Claude 4](https://www.anthropic.com/news/claude-4)
7. [Gemini Deep Think](https://deepmind.google/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/)

### 7.6 推荐阅读顺序

#### 路线 A：建立全局认知

```text
Agentic RL Survey
→ DeepSeek-R1
→ Kimi K1.5
→ Kimi K2
→ Kimi K3
→ MiniMax-M1
→ Qwen3-Coder-Next
```

#### 路线 B：深入训练算法

```text
PPO
→ GRPO / DeepSeekMath
→ DeepSeek-R1
→ Kimi K1.5 的 partial rollout
→ MiniMax-M1 的 CISPO
→ K3 的 MOPD 与长程 RL
```

#### 路线 C：构建代码 Agent

```text
ReAct / Toolformer
→ SWE-agent / SWE-bench
→ ToolRL / ReTool
→ Qwen3-Coder-Next
→ AgentENV
→ 在小模型上复现 sandbox + verifier + GRPO
```

#### 路线 D：研究 Agent 系统

```text
WebArena / OSWorld / τ-bench
→ Kimi K2.5 Agent Swarm
→ K3 Unified White-Box Environment
→ 多 Agent credit assignment
→ 记忆、权限与安全
```

---

## 8. 最终判断

Agentic RL 的核心不是“让模型多想一会儿”，而是让模型在环境中承担决策后果：

```text
想法必须转化为行动
行动必须改变环境
环境必须返回反馈
反馈必须影响下一步策略
```

DeepSeek-R1 证明了 RL 能够塑造强推理策略；Kimi K2/K2.5/K3 将这一能力进一步推进到工具、视觉、长程任务、多 Agent 和部署系统；MiniMax-M1 将长 horizon 的计算效率作为核心问题；Qwen3-Coder-Next 则展示了以较低激活成本训练专门 coding Agent 的方向；OpenAI、Anthropic 与 Google 的闭源模型说明，前沿产品正在把 reasoning 与工具使用结合，但其内部 Agentic RL 配方仍缺乏足够公开证据。

因此，当前最合理的总体判断是：

> **Agentic RL 正在从“reasoning RL 的外部工具扩展”发展为“任务—环境—策略—基础设施共同优化”的完整训练范式。**

未来真正的竞争点，不只是哪个模型在某个 benchmark 上更高，而是：

```text
谁能构造更可靠的环境
谁能提供更高质量的 reward
谁能训练更长、更稳定的轨迹
谁能让能力跨环境泛化
谁能把高能力压缩到可接受的部署成本
```
