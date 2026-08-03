---
title: K3 规划与工具使用能力 — 从 Agentic RL 中涌现
tags: [llm, agent, rl, planning, tool_use, emergence]
created: 2026-07-29
related:
  - "[[K3 Agent 能力全景]]"
  - "[[K3 Agent 能力对比]]"
  - "[[K3 架构-训练范式-基础设施三层关系]]"
  - "[[Agentic RL 涌现 — 参考资料与学习路径]]"
---

## 被教出来 vs 涌现出来：关键区分

规划和工具使用在 K3 中**不是被显式教出来的，而是从 agentic RL 环境中涌现的**。

```
被教出来的 (SFT):
  ✓ 工具调用的格式 <bash>ls</bash>
  ✓ 基本对话和代码能力
  ✓ 代码语法

涌现出来的 (RL):
  ✓ 什么时候调用工具、调哪个
  ✓ 怎么拆解任务、怎么排顺序
  ✓ 失败后换什么策略
  ✓ 什么时候不调用工具直接思考就够了
```

SFT 给的是**形式**，RL 给的是**判断力**。

---

## 一、规划能力：从哪里长出来

### 1.1 环境是唯一的老师

K3 的训练中，没有任何一条数据是「正确规划」的标注。规划能力来自一个简单的 RL 循环：

```python
# 没有人类专家写的 plan，只有目标 + 验证器
class Task:
    goal: str              # "修复这个 CUDA kernel 的 race condition"
    environment: Sandbox   # 真实代码仓库 + 测试套件
    verifier: Callable     # "所有测试通过 → reward=1, 否则 0"

# 模型在环境中自由探索
for step in range(max_steps):
    action = model(history)     # 模型自己决定下一步做什么
    observation = env.step(action)  # 环境返回结果
    history.append(action, observation)
    
    if task_completed:
        reward = verifier.final_score(env.state)

# 没有 "你应该先读 README、再 grep atomic_add、然后修改 core.py"
# 模型通过 trial-and-error 自己发现有效的规划模式
```

### 1.2 四种环境类型，训练四种规划能力

K3 的 RL 环境设计覆盖了不同维度的规划需求：

| 环境类型 | 规划的挑战 | 训练出的能力 |
|---|---|---|
| **AET (自主执行)** | 无参考轨迹，只有目标描述和验证器反馈 | **从零构思**：任务分解、路径选择 |
| **Personal Assistant** | 多天、多应用、并发事件 | **交错规划**：同时处理多个独立事件流 |
| **Kernel Optimization** | 有性能门槛，正确性只是起点 | **约束优化**：在保证正确的前提下最大化性能 |
| **Web Development** | 视觉反馈，需要迭代调优 | **视觉感知引导规划**：看图→改代码→再看 |

### 1.3 AET 的规划训练：最纯粹的例子

这是 K3 规划能力的核心训练场。§4.2.6:

> "Agents see only the objective, context, constraints, and verification interfaces, without reference trajectories or predefined procedures, and must autonomously perform task decomposition, tool selection, planning, error recovery, and termination."

一个具体的 AET 任务训练过程：

```
任务: "复现这个 3D 相机维修管理系统为 Web 应用"
          (黑盒系统：只能通过 oracle 查询输入/输出，看不到源码)

模型看到:
  - 目标: "构建一个 Web 应用，行为与目标系统一致"
  - 可用的 oracle 查询次数: 100
  - 可提交次数: 10
  - 验证器: 比较生成系统与目标系统的输入/输出一致性

没有的东西:
  ✗ 系统架构文档
  ✗ API 参考
  ✗ 参考实现
  ✗ 推荐的开发步骤
  ✗ 任务分解示例

模型必须自己发现:
  Turn 1-5:  "我不知道这系统干什么，先发几个随机输入看输出"
  Turn 6-10: "发现它处理 camera repair tickets，有 CRUD 操作"
  Turn 11-15: "开始构建后端 API → 提交第一版"
  Turn 16-20: "验证器反馈: 60% 一致，缺少 'assign technician' 功能"
  Turn 21-30: "重新查询 oracle，发现 technician assignment 的逻辑"
  Turn 31-35: "更新实现 → 85% → 继续迭代..."
```

模型在这个过程中学到的是**规划策略** (不是特定任务的 plan):

- 信息不足时: 先探索/查询，再构建
- 有反馈时: 定位差异，定向修复
- 预算有限时: 优先覆盖核心功能，再优化边界情况
- 逼近目标时: 知道什么时候该停手提交

### 1.4 复合 Reward 塑造规划质量

Planning 的好坏通过 reward 的结构隐式定义：

```python
# AET 的 reward 结构
reward = (
    1.0 * correctness_score +          # 最终结果是否正确
    0.0 * process_quality              # 没有过程分！
) 
# penalty for exceeding submission budget

# 这意味着:
# - 模型想怎么规划就怎么规划
# - 但提交次数有限 → 必须规划好再动手
# - 没有 "你用了好方法" 的奖励 → 模型自己判断什么方法有效

# 对比: Personal Assistant 任务
reward_per_event = correctness(event)
total_reward = mean(reward_per_event)
# 如果一个事件处理不好，整体 reward 下降
# → 训练出 "不能因为一个难点卡住整条 pipeline" 的规划意识
```

关键洞察: **K3 不教模型怎么规划，它创造了规划必然会涌现的环境** — 复杂目标 + 有限资源 + 只验证结果不验证过程 + 大量 trial → 有效的规划策略自然胜出。

---

## 二、工具使用能力：从哪里长出来

### 2.1 两个层面：形式层和策略层

```
形式层 (SFT 学会):
  - 工具调用的 XML schema: <bash>...</bash>
  - 参数格式: <edit file="path" line="42">new_code</edit>
  - 工具返回的解析: <output>...</output>

策略层 (RL 学会):
  - "这个错误信息提示是类型不匹配，我该 grep 类型定义还是直接看调用栈?"
  - "grep 返回 200 个匹配，太多了，我该加什么过滤条件?"
  - "read_file 返回的代码很长，我应该分段读还是先搜关键函数?"
  - "测试失败了，是回退修改还是往前走?"
```

形式层是 SFT 教的，策略层是 RL 涌现的。

### 2.2 策略层的训练：Unified White-Box RL 环境

这是 K3 工具使用能力最核心的训练机制 (§4.2.1)。关键设计：**每次训练随机换工具 schema，让模型不能依赖记忆，必须学会泛化。**

```python
# 同一任务的三种工具 schema
# Kimi Code 风格:
  <bash>grep -rn "atomic_add" triton/</bash>
  <edit_file path="triton/python/core.py" line="1234">
    new_code
  </edit_file>

# Claude Code 风格:
  Bash: grep -rn "atomic_add" triton/
  Edit(file="triton/python/core.py", old="old_code", new="new_code")

# Codex 风格:
  run_shell("grep -rn 'atomic_add' triton/")
  write_file("triton/python/core.py", content)

# 训练时随机分配 → 模型必须按 schema 而不是记忆来使用工具
```

这种训练的效果是：

```
模型在训练中的行为变化:

Epoch 1000 (初期):
  → 经常用错 schema，参数名打错
  → "这个工具叫什么来着... grep? Bash? search?"
  
Epoch 5000 (中期):
  → schema 正确率提升，但工具选择不够合理
  → "grep 返回 5000 行匹配，我一条条看..." (低效)
  
Epoch 10000 (后期):
  → 先 grep 定位文件 → 发现太多匹配 → 加过滤条件 → 精确到 12 行
  → "grep 'atomic_add.*relaxed' 只搜可能相关的模式"
```

### 2.3 工具使用的反馈闭环

每步工具调用都产生一个 observation，模型必须根据 observation 调整下一步：

```
Turn N:   grep -rn "race" triton/
          → 返回: core.py:1234, ptx_gen.py:567, test_race.py:42

Turn N+1: read_file("triton/python/core.py", 1230:1260)
          → 返回: 30 行代码，确认这里用了 relaxed memory order

Turn N+2: "问题是 PTX memory order... 需要改成 acq_rel"
          但改成 acq_rel 会不会影响非 CUDA 后端？

Turn N+3: grep "atomic_add" triton/python/backend/
          → 确认只有 CUDA 后端使用 → safe to change
```

这和在沙箱里什么都不做的区别是：模型得到了**工具返回信息的实时反馈**。它学会的不是死板的工具调用顺序，而是:

- `grep` → 如果匹配太少, 搜错了地方
- `grep` → 如果匹配太多, 关键词不够精确
- `read_file` → 如果返回太长, 需要分段读取
- 先读少量关键文件, 确定方向后再深入

---

## 三、融合：一个完整的训练 case

最能说明问题的是把规划和工具使用放在一起看一个完整的 RL 训练轨迹：

```
完整的 K3 RL 训练 case: "修复 Triton 中 atomic_add 的 race condition"

┌─ Stage 1: 信息收集 (工具使用驱动规划) ───────────────────┐

  模型行动:
    grep "atomic_add" triton/          → 30 处匹配 (太多!)
  
  环境反馈: 2000 行 grep 输出
  模型判断 (planning):
    "30 处匹配太多了，我需要缩小范围"
    "race condition 最可能出在 CUDA backend 的 PTX codegen"
  
  调整后行动 (planning-informed tool use):
    grep "atomic_add.*ptx\|ptx.*atomic_add" triton/  → 2 处匹配 (精确!)
    read_file("triton/python/ptx_gen.py", 555:575)   → 看到 relaxed memory order

┌─ Stage 2: 初步验证 (工具使用验证规划) ───────────────────┐

  模型行动:
    edit_file("ptx_gen.py", 567, "relaxed" → "acq_rel")
  环境反馈: 文件已更新

  模型行动:
    python -m pytest test_atomic_race.py  → FAIL (测试仍然失败)
  环境反馈:
    test_atomic_race.py::test_race_condition FAILED
    AssertionError: expected atomic_add result = 100, got 97

  模型重新规划 (planning adjusts based on tool feedback):
    "测试依然失败，说明还有另一处 race"
    "ptx_gen.py 改了，但 core.py 也有 relaxed"
    "grep 'relaxed.*atomic' triton/ → 3 matches → 全部改掉"

┌─ Stage 3: 系统验证 (规划扩展) ───────────────────────────┐

  模型重新行动:
    edit_file("ptx_gen.py", ...)
    edit_file("core.py", ...)  
    edit_file("utils.py", ...)
    python -m pytest test_atomic_race.py → PASS
  
  环境反馈: 测试通过

  模型判断 (更高层次的 planning):
    "单测通过了，但我的修改可能影响其他部分"
    "需要跑完整的 CPU 和 GPU 测试套"

  模型行动:
    python -m pytest tests/unit/language/  → 47 passed, 0 failed
    python -m pytest tests/unit/runtime/   → 所有通过
  
  模型判断 (规划中的 "stop condition" 判断):
    "所有测试通过，提交 PR"

  最终 reward: +1 (完成, 且没有超过 budget)
```

这个 case 揭示了规划和使用工具的**相互塑造**：

```
工具使用提供信息 ──→ 规划调整策略
                         ↓
                    工具使用适应新策略 ──→ 工具返回新信息
                                              ↓
                                         规划再次调整 ...
```

---

## 四、涌现的深层机制

### 4.1 什么是「涌现」

在 K3 的语境中，「涌现」不是哲学概念，而是有明确定义的工程现象：

> **涌现** = 模型在 RL 训练中自发演化出的行为策略，这些策略 (a) 没有被显式标注在任何训练数据中，(b) 是环境 reward 结构的函数而非人类意图的直接映射，(c) 可以泛化到训练环境中未见过的类似任务上。

### 4.2 涌现的四个条件

K3 的 agentic RL 环境之所以能催生涌现，是因为它同时满足了四个条件：

```
条件 1: 稀疏奖励 (Sparse Reward)
  → 只检查最终结果，不检查中间步骤
  → 模型有探索自由

条件 2: 环境多样性 (Environment Diversity)
  → 训练中任务、工具、配置持续变化
  → 模型无法死记硬背，必须泛化

条件 3: 反馈保真度 (Feedback Fidelity)
  → 工具返回的结果是真实环境的输出
  → 反馈信号可预测、可复现

条件 4: 充分采样 (Sufficient Trials)
  → 几万个不同任务 × 每次 K=4 条轨迹
  → 有效策略有足够机会被发现和强化
```

### 4.3 条件一：稀疏奖励 — 为什么「不教」比「教」更有效

传统 SFT 做法：标注「正确的规划步骤」：

```
传统 SFT 训练数据:
  Task: "修复 race condition in atomic_add"
  Step 1: grep -rn "atomic_add" source/
  Step 2: 阅读 ptx_gen.py:555-575
  Step 3: 修改 memory order to acq_rel
  Step 4: 运行测试
  Step 5: 提交

问题: 模型学会了这套步骤，但换一个 task 就不知道了。
      因为模型学的是 "这 5 步对这道题是对的"，
      不是 "如何根据任务特点灵活选择步骤"。
```

K3 做法：只给目标 + 验证器，不规定中间步骤：

```
K3 RL 训练的 reward 函数:
  
  def reward(env_state):
      if all_tests_pass(env_state):
          return 1.0
      # 没有 "中间步骤是否正确" 的任何检查

  # 模型必须先探索 "哪些步骤组合能拿到 reward"
  # 在大量 trial 后，模型泛化出的是策略模式，不是步骤模板
```

为什么稀疏奖励让涌现发生：

```
密集 reward (SFT 风格):
  "你的 step 1 是对的 (+0.1), step 2 偏离了标准 (-0.1)..."
  → 模型学到的是 "如何模仿标准答案"
  → 离开标准答案的场景 → 无法应对

稀疏 reward (K3 风格):
  "你做的每一步我没有评价，只看最终结果"
  → 模型在探索空间中自由游走
  → 有效的 survival 策略自然被强化
  → 泛化的策略模式而非特定步骤模板
```

### 4.4 条件二：环境多样性 — 防止策略僵化

这是 K3 最关键的设计选择：**不让模型在同一环境中训练太久**。

```python
# 多样性的维度

# 1. 任务多样性
tasks = [
    fix_race_condition,           # 软件工程修复
    optimize_cuda_kernel,          # 性能优化
    build_web_dashboard,           # 前端开发
    analyze_financial_report,      # 数据分析
    debug_distributed_system,      # 系统调试
    replicate_blackbox_api,        # 反向工程
    orchestrate_multi_agent,       # agent 编排
    # ... 数万个不同的 task
]

# 2. 环境多样性
environments = [
    {"os": "ubuntu:22.04", "python": "3.10", "gpu": "A100"},
    {"os": "centos:7", "python": "3.8", "gpu": None},
    {"os": "debian:bookworm", "python": "3.11", "gpu": "H100"},
]

# 3. 工具 schema 多样性
schemas = [kimi_code, claude_code, codex, custom_random]

# 每个 training batch: 随机组合
task = random.choice(tasks)
env  = random.choice(environments)
tool = random.choice(schemas)
trajectory = run_agent(model, task, env, tool)
```

为什么环境多样性让涌现发生：

```
单一环境训练:
  Task A × Schema A × Env A → 学会 "对 A 有效的策略"
  换到 Task B → 策略失效 → 泛化失败

多样化环境训练:
  几万个不同 task × 3 种 schema × 多种 env
  → 唯一的幸存策略: 不与任何特定 task/schema/env 耦合的通用策略
  → 这就是涌现的本质：
     "学会如何为任何新 task 找到有效的方法"
     而不是 "记住对已知 task 的有效方法"
```

### 4.5 条件三：反馈保真度 — 涌现策略的验证基础

```python
# 低保真度反馈 (模型评判)
class LowFidelityEnv:
    def step(self, action):
        return judgement_model.evaluate(action)
        # 问题: 评判模型可能出错、不一致、有偏见

# 高保真度反馈 (环境直接返回)  
class HighFidelityEnv:
    def step(self, action):
        if action.type == "bash":
            return real_bash_output(action.command)
            # 真实环境的输出，100% 准确
        elif action.type == "pytest":
            return real_test_results(action.test_file)
            # 通过/失败，客观可复现
```

高保真度反馈让涌现策略的质量更高：

```
低保真度:
  模型改代码 → 评判模型说 "看起来不错" → reward +1
  → 模型学会写 "看起来不错" 的代码 (但可能跑不通)

高保真度:
  模型改代码 → 真实的测试套件说 FAIL → reward 0
  → 模型必须在真实环境中验证 → 学会写 "能跑通" 的代码
```

### 4.6 条件四：充分采样 — 涌现策略的发现机制

```python
# 为什么需要充分采样

# 单一 trial 的问题:
for task in tasks:
    trajectory = model.generate(task)  # K=1
    reward = env.evaluate(trajectory)
    update_policy(reward)
# → 如果 random seed 恰好选择了一个 "看起来还不错但实际是死胡同" 的路径
# → 模型就被带偏了

# K3 的解决: K=4 采样
for task in tasks:
    trajectories = model.generate(task, num_samples=4)  # K=4
    for traj in trajectories:
        reward = env.evaluate(traj)
    # 4 条轨迹中有正 reward 的强化，负 reward 的抑制
    # → 有效的策略被多轮 trial 交叉验证
```

为什么充分采样让涌现发生：

```
理论角度:
  - 每个 task 有 4 条探索路径
  - 50,000 tasks → 200,000 条 RL 轨迹
  - 每条轨迹 50-500 步 agent-environment 交互
  - 总共 10M-100M 次 "规划-执行-观察" 循环
  
  在这个规模上，统计意义上有效的策略模式被反复强化
  无效的策略模式被反复抑制
  涌现的 "信号积累窗口" 足够大，确保模式不是 noise
```

### 4.7 涌现的完整机制总结

```
                    稀疏奖励
                  (只看结果，不看过程)
                        ↓
                 模型自由探索
                        ↓
         环境多样性 × 反馈保真度 × 充分采样
                        ↓
    ┌──────────────────┼──────────────────┐
    ↓                  ↓                  ↓
  有效策略            无效策略            部分有效策略
  (被强化)            (被抑制)            (被淘汰)
    │                                      │
    └──────────────────┬──────────────────┘
                       ↓
              策略模式稳定收敛
                       ↓
         涌现: 通用规划与工具使用能力
         (不依赖特定 task/schema/env 的策略模式)
```

---

## 五、涌现 vs 被教：量化对比

### 5.1 一个思想实验

如果有两个版本的 K3，分别是「被教出来的」和「涌现出来的」，它们在同一个新任务上的表现：

```
任务: "优化这个 Python 脚本的启动时间，现在需要 3.2 秒，目标 < 1 秒"
(这个任务从未出现在任何训练数据中)

被教出来的版本:
  Step 1: grep -rn "import" script.py
  Step 2: read 整个文件
  Step 3: 不知道下一步做什么 (标准训练流程中没有 "脚本优化" 的 plan)

涌现出来的版本:
  Turn 1:  "启动慢通常和 import、initialization、I/O 有关"
  Turn 2:  python -m cProfile script.py → 分析瓶颈
  Turn 3:  看到 import 花了 2.1 秒 → 定位到具体 slow import
  Turn 4:  grep -rn "slow_module" → 发现只在某函数中用到
  Turn 5:  改为 lazy import → 测试 → PASS
  Turn 6:  整体测试 → 0.8 秒 → 提交

原因: 涌现版本泛化出了 "瓶颈分析和定向优化" 的策略模式
     不是在记忆中查找 "脚本优化的标准步骤"
```

### 5.2 涌现的证据

K3 技术报告中有一个间接但有力的证据：**任务越陌生 (在训练中越不可能见过)，K3 的相对表现越好**。

```
训练中常见任务类型:     DeepSWE      → K3 落后于 Fable 5 和 Sol
训练中较少见任务类型:    FrontierSWE  → K3 领先于 Sol, 接近 Fable 5
训练中完全陌生的任务:    SWE-Marathon → K3 排名第一
(训练中不可能包含 "从头写 Rust C 编译器" 或 "Android APK 反编译" 这些任务)

↑ 这说明 K3 学到的是 "如何应对陌生问题" 的通用策略，
  而不是 "对已知问题的标准回答"
```

---

## 六、汇总

### 训练维度

| 能力 | 形式来源 | 策略来源 | 关键训练设计 |
|---|---|---|---|
| 工具调用格式 | SFT | — | XTML chat template |
| 工具选择 (用哪个) | — | RL (Unified White-Box RL) | 动态 harness 配置, 多样化工具 schema |
| 工具参数 (怎么用) | — | RL (环境反馈) | grep/read/edit → observation → 调整 |
| 任务分解 | — | RL (AET, Swarm) | 只有目标+验证器, 无参考轨迹 |
| 步骤排序 | — | RL (全 agent domain) | Trial-and-error + 环境惩罚 |
| 失败恢复 | — | RL (kernel tasks, AET) | 错误信号 → 分析 → 重试 |
| Stop condition | — | RL (budget control) | Token budget + submission budget |
| 跨 task 泛化 | — | RL (全 agent domain) | 随机 harness + 随机 task 类型 + 随机环境 |

### 涌现的四个条件

| 条件 | K3 的实现 | 作用 |
|---|---|---|
| 稀疏奖励 | 只验证最终结果 | 给模型探索自由 |
| 环境多样性 | 数万 task × 多 harness × 多 env | 防止策略僵化 |
| 反馈保真度 | 真实环境输出，非评判模型 | 确保涌现策略的质量 |
| 充分采样 | K=4 × 50K+ tasks | 策略被统计层面验证 |

### 核心结论

K3 不教 K3 怎么规划、怎么用工具。K3 把 K3 放进了一个「只能通过有效的规划和工具使用来拿 reward」的环境，让它自己跑成千上万个 trial。能跑通的策略自然被强化，无效的策略自然被抑制。

**规划和工具使用能力是从 agentic RL 中涌现的，不是从标注数据中学来的。** 这个「涌现」有四个工程条件支撑：稀疏奖励给自由、环境多样性防僵化、反馈保真度保质量、充分采样做信号积累。
