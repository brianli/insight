---
type: research
title: Long Context：从 1M 窗口到有效长程能力
created: 2026-08-05
updated: 2026-08-05
tags:
  - long_context
  - llm
  - attention
  - inference
  - training
  - kimi_k3
status: done
---

# Long Context：从 1M 窗口到有效长程能力

> [!warning]
> 本文讨论的是截至 **2026-08-05** 的公开资料。闭源模型通常只公开 context-window 规格和 benchmark，不公开 attention、position encoding、长上下文训练数据与基础设施细节。因此，文中明确区分“官方披露”“公开实现资料”和“推断”，不把未知部分补成事实。

# 1. 核心判断

“1M context window”不是一个单点技术指标，而是一套系统能力：

$$
\text{Long Context}
=
\text{Position Generalization}
+
\text{Token Mixing}
+
\text{Memory / KV Management}
+
\text{Long-Context Training}
+
\text{Distributed Systems}
+
\text{Effective-Context Evaluation}
$$

各家模型的共同路线不是“把 Transformer 的序列长度参数改成 $10^6$”，而是同时回答六个问题：

1. **位置**：模型如何理解远处的第 $500{,}000$ 个 token？
2. **计算**：如何避免每一层都对 1M 个 token 做完整二次 attention？
3. **记忆**：如何保存、压缩、传输和复用 1M token 的 KV / state？
4. **训练**：模型是否真的见过长依赖，还是只在推理时接受长输入？
5. **系统**：1M prefill、反向传播和多用户服务如何落到 GPU / TPU 集群？
6. **能力**：模型是否真的使用了远处信息，而不是只保留了“能接收”这个接口？

最重要的校正是：

> **Declared context length ≠ effective context length。**

“窗口”是模型能接收多少输入；“有效长程能力”是模型能否在输入变长、信息分散、干扰增加、需要多跳组合时仍然正确利用上下文。RULER、LongBench v2、MRCR 等工作反复说明，needle-in-a-haystack 只能测到长上下文能力的一小部分；即使模型支持 1M，长对话、多事实检索和深层推理仍可能显著退化。[RULER](https://github.com/NVIDIA/RULER)；[LongBench v2](https://longbench2.github.io/)；[LongBench v2 论文](https://arxiv.org/abs/2412.15204)

# 2. 为什么 1M 是一个系统级难题

### 2.1 标准 Transformer 的两个成本

对长度为 $n$、隐藏维度为 $d$ 的标准 causal self-attention：

$$
\text{Attention Compute}
\sim
O(n^2d)
$$

推理时还要保存每层的 KV cache：

$$
\text{KV Memory}
\sim
O(nLd_{\mathrm{kv}})
$$

其中 $L$ 是层数，$d_{\mathrm{kv}}$ 是每层 KV 的总宽度。

当 $n=10^6$ 时，问题不只是矩阵乘法变大，而是：

- prefill 需要处理约 $10^{12}$ 级别的 token-pair 关系；
- KV cache 随 token 数线性增长；
- batch 中不同请求长度差异巨大，造成 GPU 碎片和调度困难；
- 训练还要保存 activation、gradient 和 optimizer state；
- 多设备之间必须传递跨段的 attention / recurrent state。

### 2.2 长上下文至少有三种含义

| 含义 | 解决的问题 | 常见手段 |
|---|---|---|
| **窗口扩展** | 让模型接受更长输入 | RoPE scaling、YaRN、LongRoPE、NoPE |
| **长程建模** | 让模型真正利用远处信息 | 长上下文预训练、合成任务、全局 attention / memory |
| **长上下文服务** | 让请求能被经济地训练和部署 | KV 压缩、量化、context parallel、ring attention、cache reuse |

很多模型只在第一层意义上“支持 1M”。真正的 frontier 路线必须把三层打通。

# 3. 综述论文给出的技术地图

### 3.1 推荐先读的综述

| 资料 | 主要价值 | 阅读建议 |
|---|---|---|
| [A Comprehensive Survey on Long Context Language Modeling](https://arxiv.org/abs/2503.17407) | 将问题分成有效 / 高效 LCLM、训练部署和评估分析三部分 | 最优先，作为总目录 |
| [A Survey on Transformer Context Extension: Approaches and Evaluation](https://arxiv.org/abs/2503.13299) | 用 position、compression、RAG、attention pattern 四类组织 context extension | 适合建立方法分类 |
| [Thus Spake Long-Context Large Language Model](https://arxiv.org/abs/2502.17129) | 覆盖位置外推、KV cache、memory、架构、训练基础设施、后训练和评估 | 适合查缺补漏 |
| [Advancing Transformer Architecture in Long-Context Large Language Modeling](https://arxiv.org/abs/2311.12351) | 从 Transformer 架构全生命周期梳理长上下文设计 | 适合回看技术谱系 |
| [Beyond the Limits: A Survey of Techniques to Extend the Context Length](https://arxiv.org/abs/2402.02244) | 集中讨论上下文长度扩展技术 | 适合先看 position / efficient attention |

这些综述的共同结论可以压缩成：

```text
位置编码只是入口；
attention / memory 决定计算形态；
训练数据决定模型是否会使用长上下文；
系统基础设施决定方案能否落地；
评估决定“1M”是不是有效能力。
```

# 4. 六类技术路线

## 4.1 位置编码扩展：把“已学过的相对距离”搬到更长范围

主流 decoder-only LLM 大量使用 RoPE。RoPE 的问题是：模型在训练时只见过有限位置范围，直接把位置索引外推到几倍甚至几百倍后，旋转角度分布发生改变，注意力可能出现 aliasing、过度衰减或局部模式崩坏。

### Position Interpolation

Position Interpolation（PI）把长序列位置压缩回模型已经训练过的范围：

$$
p_{\mathrm{new}}
\mapsto
p_{\mathrm{old}}
=
p_{\mathrm{new}}\frac{L_{\mathrm{train}}}{L_{\mathrm{target}}}
$$

优点是简单、稳定、所需长文本微调成本低；代价是所有距离被统一压缩，近距离分辨率可能受损。

参考：[Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595)

### NTK-aware scaling

不是对所有 RoPE 频率等比例压缩，而是对不同频率维度使用不同缩放，尽量保留高频维度的局部精度，同时延长低频维度的可表示距离。

### YaRN

YaRN（Yet another RoPE extensioN）是目前开源模型中最常见的工程路线之一。它结合：

- 非均匀频率插值；
- attention temperature / attention scaling；
- 少量长上下文训练。

它的价值不只是“把位置拉长”，还要修正缩放后 attention 分布的温度。YaRN 报告称，相比早期方法，所需训练 token 和训练步数显著降低。[YaRN](https://arxiv.org/abs/2309.00071)

### LongRoPE

LongRoPE 对 RoPE 的缩放进一步做得不均匀：

- 不同 RoPE 维度使用不同缩放；
- 不同 token position 也可能采用不同策略；
- 配合少量长上下文微调；
- 尽量保持短上下文能力不退化。

论文报告将预训练模型扩展到超过 2M token。[LongRoPE](https://arxiv.org/abs/2402.13753)

### NoPE / iRoPE

另一条路线不是继续拉伸 RoPE，而是减少或取消部分层的位置编码：

- NoPE 层不直接给 Q/K 加 RoPE；
- 位置能力由局部 RoPE 层、attention pattern、递归 state 或其他机制承担；
- 这样可以减少超长位置上的旋转角度失真。

K3 的 Gated MLA 使用 NoPE，并由 KDA 的递归衰减机制提供位置和 recency 信息；Llama 4 的 iRoPE 则交错使用局部 RoPE attention 与全局 NoPE attention。

## 4.2 Attention pattern：不要让每一层都承担完整全局 attention

### Local + Global

把 attention 层分为：

```text
Local attention：只看局部 chunk / sliding window
Global attention：周期性看完整上下文
```

局部层负责高频、局部、精细模式；全局层负责跨段检索和长程信息汇聚。只要全局层的比例足够、训练数据确实要求跨段检索，就可以用远低于“每层全局 attention”的成本获得长程能力。

Llama 4 和 K3 都属于这一类，但实现不同：

- Llama 4：局部 RoPE + 全局 NoPE；
- K3：多数 KDA + 周期性 Gated MLA。

### Linear / Recurrent attention

将历史信息压缩成固定大小 state：

$$
S_t=F_t(S_{t-1},x_t)
$$

推理成本与历史长度不再线性增加，适合超长 decode；但固定 state 存在容量瓶颈和信息丢失风险。因此 frontier 模型通常不会完全抛弃全局 attention，而采用 hybrid：

$$
\text{Efficient State Mixing}
+
\text{Periodic Exact Retrieval}
$$

MiniMax 的 Lightning Attention 和 K3 的 KDA 都是这个思路。

### Sparse / block attention

让每个 query 只访问部分 token block：

$$
\mathcal{A}(q)
\subset
\{1,\ldots,n\}
$$

难点是如何选择 block。静态局部窗口便宜但可能漏掉远处证据；动态稀疏需要 index / routing 分支，增加实现复杂度。

## 4.3 KV cache 与 memory：窗口能接收，不等于能存得下

### MLA / latent KV

MLA 把每个 token 的 KV 压缩到 latent：

$$
c_t=W_cx_t,\qquad d_c\ll d_{\mathrm{kv}}
$$

推理时缓存 $c_t$，需要 attention 时再重建部分 K/V。它不改变 attention 的全局交互本质，但显著降低 KV cache。

DeepSeek-V2/V3 的 MLA 是这条路线的代表。它解决的是**存储与推理吞吐**，不是单独解决位置外推或训练长程能力。

### State cache

线性 / recurrent attention 缓存固定大小 state，而不是每个历史 token 的 KV。KDA 和 Lightning Attention 属于此类。

### KV quantization / offload / reuse

工程侧还会使用：

- FP8 / INT8 / INT4 KV cache；
- CPU / DRAM offload；
- prefix cache；
- shared prefix；
- 分层 cache；
- token dropping / merging / pruning。

这些方法不一定增强模型的长程理解能力，但能决定 1M 请求是否能服务。

## 4.4 长上下文训练：模型必须在训练中见过“远处有用”

### Continual pretraining

从已有短上下文模型继续训练，逐步扩大序列长度。优点是便宜；缺点是长程能力受原模型表示和位置机制限制。

### Long-context curriculum

典型路径：

$$
8K\rightarrow32K\rightarrow128K\rightarrow256K\rightarrow1M
$$

低长度阶段学习一般能力，高长度阶段学习跨段依赖。全程使用最大长度成本极高，所以长序列通常集中在训练后期或 cooldown。

### Long-context data

长文本本身不等于长程监督。高质量数据需要满足：

- 关键信息分散在较远位置；
- 问题不能靠局部 shortcut；
- 多文档之间存在交叉引用；
- 代码仓库需要跨文件依赖；
- 长对话需要跨轮次记忆；
- 多模态任务需要跨时间和模态引用。

因此各家都会做长文档清洗、长样本 upsample、拼接 / 重排、合成检索任务和长程 instruction tuning。

### Long-context post-training

后训练还要处理：

- 长上下文问答；
- 多跳检索；
- 代码仓库级修改；
- 长轨迹 agent；
- 长思维链；
- 长输入短输出（long-in-short-out）；
- 长输入长输出（long-in-long-out）。

模型能读 1M，不代表它能在 1M 输入上生成稳定的长答案；理解和生成是两个不同问题。

## 4.5 训练与推理基础设施

### Ring Attention / Context Parallelism

将序列切到多个设备，每个设备处理一个 block，并通过 ring / all-gather 交换 K/V 或 state。通信与计算重叠，避免单卡承担完整序列。

标准 attention 需要跨设备传递 KV block；linear attention 更适合传递固定 state transition。KCP、LASP+、varlen ring attention 都是对这个思想的结构化实现。

参考：[Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889)

### FlashAttention

FlashAttention 通过 tiling 和 IO-aware kernel 减少 HBM 与 SRAM 之间的读写，不改变 attention 的数学复杂度，但显著降低实际执行成本，是所有长上下文系统的基础组件之一。[FlashAttention](https://arxiv.org/abs/2205.14135)

### Prefill / decode 分离

1M context 的 prefill 和逐 token decode 是不同问题：

- prefill：矩阵运算密集、并行度高；
- decode：每次只生成少量 token，但要反复读取庞大 KV / state；
- 多用户 serving：请求长度高度不均衡，容易发生 head-of-line blocking。

因此生产系统需要 context parallel、paged KV、continuous batching、cache-aware scheduling、offload 和动态 admission control。

# 5. 代表模型逐一拆解

## 5.1 Gemini 1.5 / Gemini 2.5：MoE + 长上下文训练与 TPU 系统工程

### 公开规格

Gemini 1.5 技术报告把 Gemini 1.5 Pro 描述为高计算效率的多模态 MoE 模型，能够在文本、代码、音频和视频上处理百万级乃至更长上下文；报告还展示了跨模态 long-context retrieval、长文档 QA、长视频 QA 和长音频任务。[Gemini 1.5 技术报告](https://arxiv.org/abs/2403.05530)

Gemini 1.5 Pro 的产品路线曾从 1M 扩展到 2M context；Gemini 2.5 Pro 的官方资料继续把 1M long-context comprehension 作为核心能力。[Google Gemini 1.5 发布](https://blog.google/innovation-and-ai/products/google-gemini-next-generation-model-february-2024/)；[Gemini 1.5 2M context](https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio/)；[Gemini 2.5 Pro model card](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-2-5-Pro-Model-Card.pdf)

### 它公开讲清楚了什么

1. **MoE**：用稀疏激活扩展模型容量与计算效率；
2. **多模态统一长上下文**：文本、图像、音频、视频可以在同一 context 中处理；
3. **长上下文训练 / 评估**：不只做 NIAH，还测试长文档 QA、长视频 QA、长音频和 in-context learning；
4. **训练与 serving infrastructure**：报告明确说长上下文依赖新的训练和服务基础设施。

### 它没有公开讲清楚什么

公开报告没有完整披露：

- 精确 attention pattern；
- 是否使用 local / global sparse attention；
- 位置编码的具体实现；
- KV cache 是否使用类似 MLA 的压缩；
- 1M / 10M 实验中每层实际计算路径；
- 长上下文训练 token 的精确配比。

因此不能严谨地说 Gemini 1.5 是“靠某个公开的线性 attention 技巧”做到 1M。更稳妥的判断是：

> **Gemini 走的是“稀疏 MoE + 长上下文训练 + 大规模 TPU / serving 系统”的路线，但其 attention 和 position 细节是闭源的。**

### 关键启示

Gemini 的路线证明：如果基础设施足够强，模型不必公开采用显式 linear attention 才能实现百万级 context；但这种路线的技术不可复现性最高，外部研究者只能观察规格和行为，无法完整还原成本结构。

## 5.2 GPT-4.1：1M 产品能力，内部技术保持不透明

OpenAI 官方为 GPT-4.1、GPT-4.1 mini 和 GPT-4.1 nano 提供 1M context window，并报告了更好的 long-context comprehension。[GPT-4.1 官方发布](https://openai.com/index/gpt-4-1/)；[GPT-4.1 API model page](https://developers.openai.com/api/docs/models/gpt-4.1)

### 已知

- context window：1M；
- 强调代码库、长文档和多模态长上下文；
- 通过 Video-MME 等评估 long-context understanding；
- 产品层提供 cached input 等机制，降低重复长输入成本。

### 未知

OpenAI 没有公开 GPT-4.1 的：

- 模型层数、hidden size、attention 结构；
- RoPE / NoPE / ALiBi 等位置方案；
- 是否采用 MoE；
- 是否采用 sparse attention、KV compression 或 recurrent memory；
- 长上下文训练 curriculum 与数据比例；
- 1M prefill / decode 的基础设施方案。

因此 GPT-4.1 不能被归类到某个已证实的架构路线。准确说法是：

> **GPT-4.1 展示了“长上下文作为产品能力”的成熟化，但没有公开足够技术资料让外界判断它是通过位置扩展、稀疏 attention、KV 压缩还是系统工程实现的。**

这对研究判断很重要：**产品公开的 context limit 不是 architecture disclosure。**

## 5.3 Claude Sonnet 4 / 4.6 与 Opus 4.6：API 1M + 运行时上下文管理

Anthropic 在 2025 年将 Claude Sonnet 4 的 API context 扩展到 1M；Sonnet 4.6 和 Opus 4.6 的官方资料继续提供 1M context。[Sonnet 4 1M context](https://claude.com/blog/1m-context)；[Sonnet 4.6](https://www.anthropic.com/news/claude-sonnet-4-6)；[Opus 4.6](https://www.anthropic.com/news/claude-opus-4-6)

### 已知

- 1M context 主要面向大型代码库、研究文档和长 agent trajectory；
- Anthropic 还提供 context editing、memory 等运行时能力；
- API / 云平台 / 产品端的可用性和限制可能不同。

### 必须区分两层

```text
模型 context window：
    单次 forward 能看到多少 token

Context editing / memory：
    系统如何在多轮 agent 过程中压缩、删除、保存和恢复上下文
```

后者不能被当作模型本身拥有更长的 attention window。

### 技术公开程度

Anthropic 没有公开 Claude 4 的详细 model architecture 或长上下文训练 recipe。因此只能确认“1M 接口和产品能力”，无法确认内部采用哪一种 attention / position / KV route。

## 5.4 Llama 4 Scout / Maverick：iRoPE 的局部 RoPE + 全局 NoPE

### 公开规格

Meta 的模型卡给出：

- Llama 4 Scout：17B active、109B total、16 experts、10M context；
- Llama 4 Maverick：17B active、400B total、128 experts、1M context；
- 两者都是 native multimodal MoE。[Llama 4 model card](https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md)

### iRoPE 的核心

公开发布资料和 serving 实现将 Llama 4 的 iRoPE 描述为交错的两类 attention：

```text
Local RoPE attention：
    只看固定 chunk / 局部窗口
    保留 RoPE 的局部位置精度

Global NoPE attention：
    看完整 causal context
    不在超长位置上继续旋转 Q/K
```

公开 vLLM 资料将其描述为约 1:3 的 global / local 结构；Hugging Face 的 Llama 4 发布说明还提到，scaled softmax / temperature tuning 作用于 NoPE 层，而局部 RoPE 层因为只看短子序列不需要同样的温度修正。[vLLM Llama 4](https://vllm.ai/blog/2025-04-05-llama4)；[Hugging Face Llama 4 release](https://huggingface.co/blog/llama4-release)

### 为什么这条路线有效

标准 RoPE 的两个冲突是：

1. 需要位置区分；
2. 超长位置上旋转角度会失真。

iRoPE 把职责拆开：

- 局部 RoPE 层处理“附近 token 的精细相对位置”；
- 全局 NoPE 层处理“跨百万 token 的内容聚合”；
- 温度调节缓解全局 softmax 在序列特别长时概率质量过度稀释。

这是一种比纯 RoPE extrapolation 更结构化的方案。

### 需要谨慎的地方

Scout 的 10M 是官方 context specification，但完整 Meta 技术报告和训练细节并没有像 K3 那样公开。外部实现资料显示 Scout 在有限长度训练 / mid-training 后依靠 length generalization 推到更长窗口，因此“10M 可输入”与“10M 上所有任务都保持有效”必须分开评估。

## 5.5 Qwen3-Coder：256K native + YaRN 外推到 1M

Qwen3-Coder 官方资料明确区分：

- 原生支持 256K；
- 使用 YaRN 可扩展到 1M；
- 目标是 repository-scale code understanding。[Qwen3-Coder 官方仓库](https://github.com/QwenLM/Qwen3-Coder)；[Qwen3-Coder model card](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct)

### 路线

```text
标准 Transformer / RoPE backbone
    ↓
高质量长上下文语料与 long-context stage
    ↓
256K native context
    ↓
YaRN positional extrapolation
    ↓
1M context
```

Qwen3 技术报告描述了最后阶段的 long-context corpus；Qwen3-Coder 的产品资料则把 1M 明确标为 YaRN extension。[Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)

### 这条路线的性质

Qwen3-Coder 不是通过 KDA / Lightning Attention 改变 token mixing，而是：

- 继续使用成熟的 Transformer attention；
- 用 long-context training 让 256K 能力形成；
- 用 YaRN 扩展到 1M。

优点：

- 继承标准 Transformer 的精确 attention；
- 开源生态容易复现；
- 对代码仓库任务很直接。

代价：

- 1M 更多是“外推能力”，不是完整 1M native pretraining；
- full attention 的 prefill 和 KV cache 成本仍然存在；
- 1M 质量对 YaRN 配置、推理 kernel 和长上下文数据很敏感。

## 5.6 MiniMax-01 / MiniMax-M1：Lightning Attention + 周期性 Softmax

MiniMax-01 的官方仓库和技术报告把路线讲得很清楚：

- 456B total parameters；
- 45.9B activated；
- 80 layers；
- 每 7 层 Lightning Attention 后放 1 层 softmax attention；
- 训练 context 1M；
- 推理可外推到 4M；
- 使用 LASP+、varlen ring attention 和 Expert Tensor Parallel。[MiniMax-01 官方仓库](https://github.com/MiniMax-AI/MiniMax-01)；[MiniMax-01 技术报告](https://arxiv.org/abs/2501.08313)

### 核心架构

```text
Lightning → Lightning → ... → Lightning → Softmax → ...
       7 个高效层                         1 个精确全局层
```

Lightning Attention 将历史压缩成可递推的 state，长序列成本接近线性；周期性 softmax 层保留精确 token-to-token 检索能力。

### 系统协同

MiniMax 不只改 attention：

- `LASP+`：针对 linear attention 的 sequence parallel；
- `varlen ring attention`：处理长序列和可变长度；
- `ETP`：处理大规模 MoE；
- computation-communication overlap：尽量把通信藏在计算后面。

这说明 MiniMax 的核心观点是：

> **要做到百万 token，必须同时替换大多数 attention 的复杂度和重新设计 sequence parallel；单纯把 RoPE 拉长不够。**

### 训练长度与推理长度的差别

MiniMax-Text-01 训练到 1M，推理上报告 4M。后者是 inference extrapolation，不等于在 4M 上做过同等规模的完整训练。这个区分比“4M”这个数字本身更重要。

## 5.7 Kimi K3：KDA + Gated MLA + NoPE + KCP

K3 是这条路线里公开得最完整、也最适合深入复现的模型之一。技术报告称其 context window 为 1M，并把长上下文能力分解成架构、数据、训练 curriculum 和 context parallelism。[Kimi K3 技术报告](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)；[Kimi K3 arXiv](https://arxiv.org/abs/2607.24653)

### 架构

```text
KDA → KDA → KDA → Gated MLA → ...
```

- 69 层 KDA；
- 24 层 Gated MLA；
- backbone 末端额外 Gated MLA；
- MLA 使用 NoPE；
- KDA 使用固定大小 recurrent state；
- KDA 负责 position-sensitive / recency-aware mixing；
- MLA 负责 unrestricted global content interaction。

### 训练

$$
8K\rightarrow64K\rightarrow256K\rightarrow1M
$$

并且在长上下文数据中把任务所需信息分散到整个 1M 范围，避免模型只学到局部 shortcut。

### 系统

KDA Context Parallelism（KCP）把每个序列 segment 的影响表示成：

$$
S_{\mathrm{out}}^{(i)}
=
\widetilde S^{(i)}
+
M^{(i)}S_{\mathrm{in}}^{(i)}
$$

每个 rank 只交换固定大小的 state transition 与 local contribution，通过 prefix scan 恢复跨段 state。与标准 softmax attention 传递随序列增长的 KV block 不同，KCP 利用 KDA 的递归结构使通信规模主要由 state size 决定。

### K3 的路线定位

K3 不是“用一种线性 attention 取代 Transformer”，而是：

> **用 KDA 把大多数 token mixing 变成固定 state 递归，再用少量 MLA 保留精确全局检索，最后用 KCP 把递归状态并行化。**

这条路线在模型结构和系统结构之间的耦合最强。

## 5.8 DeepSeek-V3 / R1：MLA + YaRN，作为 128K 对照组

DeepSeek-V3 并不是 1M 模型，但它是理解长上下文工程边界的重要对照：

- 使用 MLA 压缩 KV cache；
- 使用 YaRN 做 context extension；
- 训练阶段从 4K → 32K → 128K，分两阶段扩展，每阶段约 1000 steps。[DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)

### 它解决了什么

- MLA：降低长序列 inference 的 KV memory；
- YaRN：把 RoPE / context window 从短长度扩展到 128K；
- progressive training：让模型逐步适应长上下文。

### 它没有解决什么

MLA 只压缩 KV，并不会自动把 $O(n^2)$ 的全局 attention 变成线性；YaRN 只改变位置映射，也不会自动产生长程任务能力。因此 DeepSeek-V3 的 128K 对照组清楚说明：

> **KV 压缩、位置扩展、长上下文训练是三个不同问题。**

# 6. 拉通对齐：各家相同在哪里

### 6.1 几乎所有路线都做 long-context training

无论是 Gemini、K3、MiniMax 还是 Qwen3-Coder，真正可靠的长上下文能力都不能只依赖 inference-time extrapolation。共同做法包括：

- 长文本 / 长代码 / 长视频数据；
- 长样本清洗和去重；
- 长上下文 curriculum；
- long-context QA / retrieval / coding 任务；
- 在后训练中强化长轨迹和长输入 instruction following。

唯一不同是公开程度：开放模型会公开一部分训练阶段，闭源模型通常只公开结果。

### 6.2 都在做某种“局部 + 全局”分工

| 模型 | 局部 / 高效路径 | 全局 / 精确路径 |
|---|---|---|
| Llama 4 | local RoPE chunked attention | global NoPE attention |
| MiniMax-01 | Lightning Attention | periodic Softmax |
| Kimi K3 | KDA | periodic Gated MLA |
| Gemini | 细节未公开；长上下文能力依赖 MoE 与系统工程 | 细节未公开 |
| Qwen3-Coder | 标准 attention + RoPE extension | full attention 本身 |
| DeepSeek-V3 | MLA 主要降低 KV 成本 | full attention with compressed KV |

这是一个很强的收敛趋势：

> **纯局部 attention 会漏远程信息；纯全局 attention 太贵；SOTA 更倾向于让便宜路径覆盖大多数计算，让少量全局路径负责纠错和精确检索。**

### 6.3 都在降低“每 token 的历史记忆成本”

不同模型只是压缩对象不同：

| 压缩对象 | 代表 |
|---|---|
| KV representation | DeepSeek MLA |
| Recurrent state | MiniMax Lightning、K3 KDA |
| Attention visibility | Llama 4 local chunks |
| Position representation | YaRN / LongRoPE / NoPE |
| Runtime context | Claude context editing / memory、各种 cache / offload |

### 6.4 都把长上下文看成系统协同问题

训练与 serving 不是“模型训练完之后再优化”的两个独立阶段：

- attention 结构决定 cache 形状；
- cache 形状决定 serving memory；
- sequence parallel 决定训练可扩展性；
- position strategy 决定模型能否 extrapolate；
- long data 决定训练是否值得付出这些系统成本。

# 7. 拉通对齐：各家不同在哪里

### 7.1 Native long context vs extrapolated long context

| 类型 | 代表 | 含义 |
|---|---|---|
| 更接近 native / long-length trained | Gemini 1.5、MiniMax 1M、K3 1M | 在较长训练阶段直接适配长上下文 |
| Native shorter + extrapolation | Qwen3-Coder 256K → 1M、MiniMax 1M → 4M、DeepSeek 128K route | 依靠位置缩放或结构外推超出训练长度 |
| 公开资料不足 | GPT-4.1、Claude 4.6 | 只知道产品窗口，不知道内部训练长度 |
| 极端 length generalization | Llama 4 Scout 10M | 公开规格极长，但完整 10M 训练与有效能力细节不透明 |

### 7.2 改 position 还是改 attention

```text
Qwen / DeepSeek：
    主要沿 RoPE / YaRN / long-context training 路线扩展

Llama 4：
    position 与 attention pattern 一起改，使用 iRoPE

MiniMax / Kimi：
    主要改 token mixing，使用 hybrid linear / recurrent attention

Gemini / GPT / Claude：
    内部细节未公开，只能从能力与基础设施行为推断
```

### 7.3 目标场景不同

| 模型 | 长上下文重点 |
|---|---|
| Gemini | 多模态长文档、音频、视频、跨模态检索 |
| GPT-4.1 | 代码库、长文档、API agent |
| Claude | 长代码库、长 agent workflow、文档综合 |
| Llama 4 Scout | 开放权重、极端超长输入、多模态 |
| Qwen3-Coder | repository-scale coding |
| MiniMax | 长 agent trajectory、百万级输入和更低 serving 成本 |
| Kimi K3 | 1M context 下的代码、知识工作、长程 agent 和 native vision |
| DeepSeek-V3 | 高效 128K open model，重点是 cost / quality 而非百万窗口 |

# 8. 一个更准确的模型分类

### Route A：大规模全局 attention + 训练 / 硬件堆栈

代表：Gemini、GPT、Claude（内部细节未公开）。

核心思想：

```text
不一定改掉标准 attention；
依靠 MoE、专用加速器、分布式 attention、长上下文训练和 serving 优化把它撑起来。
```

优势是表达能力直接，缺点是复现和成本不可见。

### Route B：Local-global sparse attention

代表：Llama 4。

核心思想：

```text
局部层保留位置精度；
全局层负责跨段检索；
通过 NoPE / temperature tuning 控制超长全局层。
```

优势是兼顾局部精度与全局成本，难点是全局层的频率和训练策略。

### Route C：RoPE extension

代表：Qwen3-Coder、DeepSeek-V3，以及大量开源模型。

核心思想：

```text
保留 Transformer attention；
把位置映射压缩 / 重标定到可泛化范围；
用少量长上下文训练修复分布偏移。
```

优势是实现简单、生态成熟；缺点是 attention 和 KV 成本仍在。

### Route D：Hybrid linear / recurrent attention

代表：MiniMax-01、Kimi K3。

核心思想：

```text
大多数层用固定状态处理长序列；
少量层做全局精确 attention；
用 context parallel 把 state / transition 并行化。
```

优势是长上下文 decode 和 cache 成本好；缺点是固定 state 的记忆容量与表达能力需要大量验证。

### Route E：外部 memory / context engineering

代表：所有模型的生产系统都会使用，但它不是模型本身的 context window。

包括：

- RAG；
- summary / compaction；
- memory tool；
- context editing；
- prefix cache；
- token pruning；
- retrieval planning。

它解决的是“如何管理无限工作历史”，不是“模型单次 forward 能否看 1M”。

# 9. 为什么“窗口越大越好”是一个不完整目标

长上下文的真实指标至少应包括：

1. **最大接收长度**；
2. **有效长度**：性能下降到阈值前的长度；
3. **多针检索**：多个相似证据是否都能找回；
4. **中间位置性能**：是否存在 lost-in-the-middle；
5. **长程组合**：信息是否能跨段、多跳组合；
6. **长输出稳定性**；
7. **成本 / latency**；
8. **多轮 agent 的 context retention**；
9. **长上下文下的 instruction following**；
10. **量化和 batch 增大后的退化**。

至少要组合使用：

- [RULER](https://github.com/NVIDIA/RULER)：合成、多任务、可调长度；
- [LongBench v2](https://arxiv.org/abs/2412.15204)：真实多任务、长文档、代码仓库、长对话；
- [MRCR / MRCR v2](https://github.com/openai/simple-evals)：多轮、多 needle、相似干扰；
- [Lost in the Middle](https://arxiv.org/abs/2307.03172)：位置敏感性；
- [LongMemEval](https://arxiv.org/abs/2410.10813)：长时记忆与多轮对话。

`Needle in a haystack` 只能回答“模型能否在某种合成环境下找一个 needle”，不能回答“模型是否能在 1M 代码库中完成真实修改”。

# 10. 对 K3 的最终定位

K3 的长上下文路线可以抽象成：

$$
\text{KDA}
\;(\text{固定 state、低成本长程混合})
+
\text{Gated MLA}
\;(\text{周期性全局精确检索})
+
\text{NoPE}
\;(\text{避免位置编码外推负担})
+
\text{Long Data}
\;(\text{迫使跨 1M 取证})
+
\text{Curriculum}
\;(\text{渐进扩展})
+
\text{KCP}
\;(\text{固定 state 的 context parallel})
$$

它与 MiniMax 最接近，都是 hybrid linear / global attention；与 Llama 4 的共同点是都把局部 / 高效路径和全局 / 精确路径分开；与 Qwen / DeepSeek 的不同点是它不是主要依赖 YaRN，而是改了 token mixing 本身。

K3 最值得研究的不是“它也有 1M”，而是：

> **它把长上下文从 position-extension 问题升级成 token mixing、state memory、训练数据和 distributed systems 的共同设计问题。**

# 11. 进一步学习路线

### 第一阶段：先建立 taxonomy

1. [[Kimi K3 论文阅读报告]] 的 §3.4；
2. [A Comprehensive Survey on Long Context Language Modeling](https://arxiv.org/abs/2503.17407)；
3. [A Survey on Transformer Context Extension](https://arxiv.org/abs/2503.13299)；
4. [Thus Spake Long-Context LLM](https://arxiv.org/abs/2502.17129)。

### 第二阶段：理解 position extension

1. [Position Interpolation](https://arxiv.org/abs/2306.15595)；
2. [YaRN](https://arxiv.org/abs/2309.00071)；
3. [LongRoPE](https://arxiv.org/abs/2402.13753)；
4. [Understanding RoPE Extensions](https://arxiv.org/abs/2406.13282)；
5. [Understanding RoPE Extensions of Long-Context LLMs](https://aclanthology.org/2025.coling-main.600/)。

### 第三阶段：理解 efficient attention

1. [Ring Attention](https://arxiv.org/abs/2310.01889)；
2. [FlashAttention](https://arxiv.org/abs/2205.14135)；
3. [DeepSeek-V3 / MLA](https://arxiv.org/abs/2412.19437)；
4. [MiniMax-01 / Lightning Attention](https://arxiv.org/abs/2501.08313)；
5. [Kimi Linear](https://arxiv.org/abs/2510.26692)；
6. [Kimi K3](https://arxiv.org/abs/2607.24653)。

### 第四阶段：理解有效能力与评估

1. [Lost in the Middle](https://arxiv.org/abs/2307.03172)；
2. [RULER](https://github.com/NVIDIA/RULER)；
3. [LongBench v2](https://arxiv.org/abs/2412.15204)；
4. [Beyond a Million Tokens](https://arxiv.org/abs/2510.27246)。

# 12. 最终结论

各家模型实现 Long Context 的共同答案是：

```text
长上下文位置适配
+ 长上下文训练数据
+ 渐进式长度 curriculum
+ 低成本 attention / memory
+ 分布式训练与 serving
+ 真实长程评估
```

差异在于它们把主要赌注放在哪里：

| 路线 | 主要赌注 |
|---|---|
| Gemini / GPT / Claude | 大规模基础设施与模型训练，细节闭源 |
| Llama 4 | local-global attention + iRoPE |
| Qwen3-Coder / DeepSeek | RoPE extension + long-context training |
| MiniMax | Lightning linear attention + periodic softmax |
| Kimi K3 | KDA + Gated MLA + NoPE + KCP |

最后一句：

> **1M 是输入容量；真正的 Long Context 是在百万 token 的干扰、距离、组合和成本下仍然能稳定地找对、想对、做对。**
