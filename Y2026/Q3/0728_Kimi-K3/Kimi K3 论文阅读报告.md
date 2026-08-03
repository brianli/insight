---
title: Kimi K3 - Open Frontier Intelligence 论文阅读
tags: [paper_reading, llm, moe, rl, architecture]
---

## 论文信息

- **标题**: Kimi K3: Open Frontier Intelligence — Technical Report of Kimi K3
- **团队**: Moonshot AI (Kimi Team)
- **规模**: 2.8T 总参数, 104B 激活参数 (MoE), 1M token 上下文
- **核心改进**: 相比 Kimi K2 总体缩放效率提升 **2.5x**
- **定位**: 全球首个开源 3T 级 MoE 模型, 整体性能仅次于 Claude Fable 5 和 GPT-5.6 Sol, 远超其他开源和闭源模型

---

## 一句话总览

Kimi K3 沿着 **train-time scaling** 和 **test-time scaling** 两条轴同时推进到前沿: 将预训练推到 2.8T 参数, 并在 1M 上下文下进行强化的 long-horizon RL。这标志着开源模型第一次在 3T 级别与最强闭源模型正面竞争。

---

## 架构创新

### 1. Hybrid Attention (3:1 = KDA : Gated MLA)

每个 block 由 3 层 Kimi Delta Attention + 1 层 Gated MLA 组成, 末端额外一层 MLA 保证全局注意力。

**KDA (Kimi Delta Attention) 关键改进:**

- **Lower-bounded decay** ($g_{\text{min}} = -5$ Sigmoid): 将原本 Kimi Linear 中无界的 negative-Softplus 映射替换为有界 sigmoid, 使得 16-token tile 的累积 log-decay 限制在 (-80, 0), 从而**所有 causal tiles 都能用 Tensor Core 进行密集矩阵乘法**, 消除了原先对角线需显式 position-pair 计算的瓶颈。
- **Full-rank output gate**: 从 low-rank 改为 full-rank 的输入依赖输出门控, 增强表达能力。

**Gated MLA:**

- 继承 DeepSeek-V2 MLA 的 KV 压缩, 但使用 **NoPE** (无显式位置编码), 位置信息通过 KDA 层隐式编码。
- 新增 full-rank 输出门控, 与 KDA 统一。
- 训练中 attention 输出保持 FP32 以避免舍入误差。

### 2. Attention Residuals (AttnRes)

将 Transformer 在时间维度上的 attention 机制应用于**深度维度**: 每一层可以**选择性**地从 embedding 和所有前序层中检索表征。

- **Full AttnRes**: $O(L^2 d)$ 算法, 每层可通过 learnable pseudo-query 搜索所有前层
- **Block AttnRes**: 将 $L$ 层分成 $N$ 个 block, block 内输出求和压缩为一个表征, 跨 block 做 full attention。Kimi K3 使用 8 blocks, 12 层/block。将通信开销从 $O(Ld)$ 降至 $O(Nd)$。

### 3. Stable LatentMoE

- 896 个 routed experts, 每个 token 激活 16 个, sparsity = 56
- 2 个 shared experts 每层
- Latent MoE 维度 = 3584 (0.5x hidden dimension)

#### 它究竟把什么“变宽”了

Stable LatentMoE 归在 K3 的**宽度维（model width）**，但不是把主干 hidden dimension 直接从 $7168$ 扩大。它扩展的是**条件容量**：

- 专家池更大：896 个 routed experts；
- 每个 token 可组合的专业化路径更多：每 token 激活 16 个 routed experts；
- 总参数容量更大，但每个 token 只计算其中一小部分；
- 主干 hidden dimension 仍是 $d=7168$，routed expert 使用的是更窄的 latent dimension $\ell=3584=d/2$。

标准 MoE 让每个 routed expert 接收完整的 $d$ 维表示；LatentMoE 则把 routed path 压到低维空间：

$$
x\in\mathbb{R}^{d}
\xrightarrow{W_{\downarrow}}
z\in\mathbb{R}^{\ell}
\xrightarrow{\text{selected routed experts}}
u\in\mathbb{R}^{\ell}
\xrightarrow{\operatorname{RMSNorm},\,W_{\uparrow}}
y_{\text{routed}}\in\mathbb{R}^{d}
$$

K3 的 routed expert 路径可以概括为：

```text
x ∈ R^7168
  ├─ shared experts：直接在 7168 维处理公共变换
  └─ W↓：7168 → 3584
       └─ router 选择 16/896 个专家
            └─ 专家在 3584 维 latent space 中计算
                 └─ 加权聚合 + RMSNorm
                      └─ W↑：3584 → 7168
                           └─ 与 shared experts 相加
```

因此，LatentMoE 的杠杆是：**一次共享的降维/升维，换取大量 routed experts 的低维化**。它降低了 routed expert 的通信、权重读取和内部计算，使专家池可以继续扩大。

粗略地看，普通 MoE 的 routed 通信与每 token 激活数成正比于 $kd$；LatentMoE 降为 $k\ell$。当 $\ell=d/2$ 时，routed path 的表示通信宽度约减半。代价是：单个 routed expert 的工作空间变窄，所以这不是“每个专家都变宽”，而是**专家池变宽、单个专业分支变窄**。

#### 与 DeepSeekMoE 的差异

K3 的 Stable LatentMoE 沿用了 DeepSeekMoE 的基本组织方式：**shared experts + routed experts**。shared experts 负责所有 token 都需要的公共变换，routed experts 负责 token 条件化的专业变换。

但两者的主要差异在 routed expert 的工作空间：

| 维度 | DeepSeekMoE | K3 Stable LatentMoE |
| --- | --- | --- |
| Routed expert 输入 | 完整 $d$ 维表示 | 先压缩到 $\ell$ 维 |
| Routed expert 内部 | 在完整 hidden width 上计算 | 在低维 latent space 上计算 |
| 扩容思路 | 细粒度切分、增加专家组合 | 增加专家池，同时压低 routed path 宽度 |
| Shared experts | 隔离公共知识 | 隔离公共知识，并保留 full-width 公共路径 |
| 主要解决的问题 | 专家重复、专业化不足 | 专业化、通信/计算成本与极端稀疏稳定性 |

一句话：

> **DeepSeekMoE 主要把专家切得更细；LatentMoE 进一步把 routed expert 的通道压窄，让专家池可以更大。**

不要把两种 `latent` 混淆：

- **DeepSeek MLA 的 latent**：压缩 token 级 KV cache，解决序列长度与推理显存；
- **K3 LatentMoE 的 latent**：压缩 routed expert 的 channel representation，解决模型宽度、专家容量与 MoE 通信。

**三大稳定化改进:**

- **Normalized LatentMoE**：原始 LatentMoE 在专家聚合后直接做 $W_{\uparrow}u$。K3 改为：

  $$
  y_{\text{routed}}=W_{\uparrow}\operatorname{RMSNorm}(u)
  $$

  其中 $u=\sum_{i\in T_k(x)}p_iE_i^{routed}(W_{\downarrow}x)$。RMSNorm 让不同专家组合、不同 router 权重造成的尺度变化不会直接传入升维投影，也避免 routed branch 与 full-width shared branch 相加时幅值失衡。

- **SiTU-GLU**：SwiGLU 的 gate 和 up 分支都可能无界，极端值相乘会产生 activation outlier。K3 用 soft cap：

  $$
  \operatorname{softcap}(x,\beta)=\beta\tanh(x/\beta)
  $$

  $$
  \operatorname{SiTU\text{-}GLU}(x)
  =
  \beta_1\tanh\left(\frac{W_gx}{\beta_1}\right)
  \odot \sigma(W_gx)
  \odot
  \beta_2\tanh\left(\frac{W_ux}{\beta_2}\right)
  $$

  K3 取 $\beta_1=4,\beta_2=25$，在原点附近保留近似 SwiGLU 的行为，在大值区域逐渐饱和，并给出近似输出上界 $|f(x)|\leq\beta_1\beta_2=100$。这有利于控制多层矩阵乘法叠加、低精度训练和极端稀疏下的激活溢出。

- **Quantile Balancing (QB)**：K3 仍采用 auxiliary-loss-free routing，但不再依赖固定步长的 bias 反馈更新，而是根据 router score 的分位数直接设定 expert bias，使每个专家接近目标负载：

  $$
  q=\frac{mk}{n}
  $$

  其中 $m$ 是 batch token 数、$k$ 是每 token 激活专家数、$n$ 是专家总数。直观上，DeepSeek 的 bias 调整更像“根据冷热反馈逐步修正”，K3 QB 更像“根据全 batch 分布直接校准阈值”。大规模训练中使用 histogram 估计全局分位数，避免收集全量 router score。

#### 为什么叫 Stable

“Stable”不是 LatentMoE 天然稳定，而是 K3 针对极端稀疏的三类失稳分别加了约束：

| 机制 | 主要问题 | 稳定对象 |
| --- | --- | --- |
| 聚合后 RMSNorm | 不同专家组合导致 routed representation 尺度漂移 | 数值尺度 |
| SiTU-GLU | gate/up 乘法导致 activation outlier 和溢出 | 激活与梯度 |
| Quantile Balancing | 896 个专家的负载冷热不均 | 路由与专家训练机会 |
| MoonEP / static shape | Expert Parallel 执行形状波动、通信与 host sync | 系统执行 |

所以：

$$
\text{Stable LatentMoE}
=
\text{LatentMoE}
+
\text{数值稳定}
+
\text{路由稳定}
+
\text{系统稳定}
$$

K3 的关键不是单纯“把专家数量从 384/256 增加到 896”，而是同时处理了**容量扩展的成本**与**极端稀疏的稳定性**。

### 4. MoonViT-V2

- 从零开始用 next-token prediction 训练, **不使用 SigLIP 等对比预训练初始化**
- 主要动机: **训练稳定性**。SigLIP 初始化的视觉塔梯度范数更高且有频繁尖峰。
- 发现: 对比预训练对多模态语言模型的初始化**不是必需的**。
- 27 层 ViT, ~0.4B 参数, 支持图像和视频

### 5. Per-Head Muon

将 Muon 优化器的 Newton-Schulz 正交化从 full attention 投影矩阵粒度细化到**per-head**, 使各 head 的更新尺度更均衡。

---

## 预训练

### 数据

- 四大文本域: Web Text, Code, Mathematics, Knowledge
- 大规模视觉语料: captions, interleaved image-text, OCR, perception, video, visual coding
- 原生多模态训练 (language 和 vision 从训练开始即联合优化)

### Scaling Law

- 对比 Kimi K2: **cosine decay 优于 WSD** (在各自最优超参数下)
- 结论: 累积改进带来 2.5x 缩放效率提升

### 长上下文扩展

- 四阶段课程: 8K → 64K (预训练) → 256K → 1M (cooldown)
- NoPE 避免了 RoPE 缩放, 直接外推到 1M
- 合成长上下文数据: 排列拼接多模态文档和子任务, 迫使 attention 在 1M 尺度上运作

### 架构对比 (K2 vs K3)

| | K2 | K3 | 变化 |
|---|---|---|---|
| 总参数 | 1.04T | 2.78T | +167% |
| 激活参数 | 32.6B | 104.2B | +220% |
| 层数 | 61 | 93 | +52% |
| Routed Experts | 384 | 896 | +133% |
| 每 token 激活 experts | 8 | 16 | +100% |
| Shared Experts | 1 | 2 | +100% |
| 训练上下文 | 128K | 1M | 8x |
| Attention | MLA | Hybrid KDA-MLA | - |
| 激活函数 | SwiGLU | SiTU-GLU | - |

---

## Post-Training

### 三阶段范式

```
SFT → Domain-specific RL (3 domains × 3 effort levels = 9 experts) → MOPD 蒸馏统一
```

### RL 三域

- **General**: 搜索、知识、推理、视觉
- **General Agents**: 长时助手、深度研究、专业写作
- **Coding Agents**: SWE、kernel、web 开发

三个推理努力等级: low / high / max

### 关键算法

- **Partial Rollout**: 前 λ 比例轨迹完成即触发策略优化, 未完成的在下轮恢复。通过 token 级正则化容忍极端 off-policy staleness。
- **Reasoning Effort Budget Control**: 每道题目设定初始 budget $b_0(x)$, 超限轨迹 reward = -1。
- **Agentic GRM**: Agentic 方式执行评判: read outcome → generate rubric → score → record scorepad。加 output length budget 防止 reward hacking。

### MOPD (Multi-Teacher On-Policy Distillation)

多域多 effort 的 9 个专家模型, 通过 on-policy distillation 蒸馏为单一模型。reward 为:

$$r_{\text{opd}}(y_t) = \text{clip}\left(\log\frac{\pi_{\text{teacher}}(y_t)}{\pi_\theta(y_t)}, -R_{\max}, R_{\max}\right)$$

### 量化

- MoE expert 权重 → MXFP4, 激活 → MXFP8
- 从 SFT 阶段即开始 QAT, RL rollout 和 training 共享量化方案

### Decoding 加速

- 将 MTP 层 fine-tune 为 EAGLE-3 draft model
- 直接优化 $L_{\text{LK}}$ loss (acceptance rate), 而非 KL-divergence

---

## 基础设施

### KDA 系统协同设计

- **FlashKDA**: CUTLASS 实现的 chunkwise kernel, 重叠 intra-chunk 计算与 cross-chunk state propagation
- **KCP (KDA Context Parallelism)**: 利用 KDA 的线性循环特性, 每 rank 本地计算累积转移矩阵 $M_T$ 和局部状态 $\tilde{S}_T$, 一次 all-gather 后通过 prefix scan 恢复每个 rank 的入站状态。通信量不随序列长度增长。

### 3T 预训练基础设施

- **MoonEP**: 完美负载均衡的 EP 方案。通过有界冗余 expert 将每 rank 的 token 负载固定为 $S \times K$, 实现静态计算形状, 消除 per-layer host 同步。ILP 规划 + GPU 在线近似规划。
- **内存高效训练**: 统一 activation manager 抽象; FP8 量化 + offload; P2P Muon 正交化避免全参数 all-gather。
- **多模态编码器优化**: 动态 CP + encoder 计算填充 pipeline bubble。

### 1M Agentic RL

- **External KV Cache Pool**: write-back 策略, 将闲置 prefix 写入 CPU DRAM; KDA state 与 MLA KV cache 同步生命周期。
- **Auto-throttling Scheduler**: 根据 runtime 信号动态控制并发, 平衡早期利用率和后期 KV cache 压力。
- **AgentENV**: Firecracker microVM 沙箱。增量 checkpoint (133ms) / resume (49ms)。Pause/Resume/Fork/Snapshot 四个高级原语。训练期间共创建 51M+ 沙箱。

### 推理与在线服务

- **KDA-Aware Prefix Cache**: 统一 KDA state 和 MLA KV cache 的 paged pool 管理。前缀哈希在 512-token hash block 粒度 (而非物理 block 6144 tokens), 显著提升命中率。
- **高性能 Kernel**: KDA decoding 缓存投影输入而非 state, 支持 MTP 回滚。Block AttnRes 与 TP 融合。
- **Cache-Aware Affinity Scheduling**: 通过 consistent hashing 维护 cache locality, 双集群容错。
- **Budget-Based Admission Control**: 为不同请求类型分配独立资源预算。

---

## 实验结果

### 主要 Benchmark 定位

- **推理与知识**: GPQA Diamond 93.5 (并列第一); **HLE-Full 和 CritPt 落后**, 是明确弱点
- **Coding**: ProgramBench 第一 (77.8%); SWE-Marathon 第一 (42.0%, 领先 7 分); FrontierSWE 第二 (81.2%); DeepSWE 第三 (67.5%)
- **Agentic**: BrowseComp 第一 (91.2%); AutomationBench 第一 (30.8%); GDPval-AA v2 Elo 第三 (1686)
- **Vision**: OmniDocBench 第一 (91.1%); Math-Vision + Python 工具 97.8%

### 成本效率

在 Kimi Code Bench 2.0 上: Kimi K3 以 Claude Fable 5 **38% 的成本**达到相差 4 分的成绩, high effort 已等效于 Claude Opus 4.8 max effort, 成本为其 1/3。

### 网络安全评估

- Tier 1 (漏洞发现): 发现 16 个之前未知的漏洞, 包括 Linux 内核远程 DoS 和 Dirty-COW 级本地提权。
- Tier 2 (漏洞利用): 解决 14/36 tasks (38.9%), 高于 GLM-5.2 的 22.2%。但仍存在明显的能力差距 -- 人类专家可解决全部 36 题。

### Case Study 亮点

- **MiniTriton**: 完整 GPU 编译器 (DSL → MLIR → PTX), matmul 达 cuBLAS 90% 性能, 成功端到端训练 GPT。
- **芯片设计**: 48 小时内自主完成推理芯片原型 (100MHz, 8700+ tokens/s)。
- **科研编码**: 2 小时完成天体物理数值模拟 pipeline, 通常需 1-2 周。

---

## 关键洞察

1. **两条 scaling 轴不能割裂**。开源在 test-time scaling 进步快, 但 pretrain scale 停滞在 1T 级。K3 选择两者同时推进。
2. **Lower-bounded decay 是个巧妙的 "engineering trick"** -- 通过约束 decay 范围让 Tensor Core 全覆盖, 既是数值改进也是硬件利用改进。
3. **AttnRes 将 attention 思想泛化到深度维度**。block 分组是工程精简, 但保留了核心 idea: GPU 做矩阵乘法时最擅长的事情就是 attention。
4. **从零训练视觉塔 vs. 对比预训练**的结果颠覆常规认知。
5. **Quantile Balancing 优雅解决了 103 级 auxiliary-loss-free routing 的负载均衡问题**。
6. **成本效率是 K3 最被低估的优势**。在多个 benchmark 上以远低于竞品的成本达到接近顶级的分数。
7. **开源 3T 模型是基础设施挑战, 不只是算法**。MoonEP、KCP、AgentENV 等系统创新与架构创新同样重要。

---

## 局限性 & 待探索

- HLE-Full 和 CritPt 落后, **research-level reasoning 仍是明显短板**。
- Agent Behavior 质量 (而非结果正确性) 不及 Claude Fable 5 和 GPT-5.6 Sol。
- 网络安全 Tier 2 的表现与人类专家还有差距。
- 未讨论多语言能力和安全对齐细节。
- 训练 cost (FLOPs) 未明确披露。
- KDA decoding 的 speculative decoding 回滚方案依赖 projection caching (ReplaySSM 同类设计), 长批场景的 state 流量仍是问题。
