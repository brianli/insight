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

### 5. Per-Head Muon：为什么被放在 Model Architecture

#### 5.1 先给结论

K3 技术报告 §2.5 讨论的不是一个新的 forward layer，而是：

> **K3 如何更新 attention projection 的参数。**

Per-Head Muon 将 Q/K/V 投影矩阵沿 attention head 维度切分，对每个 head 的 momentum 独立执行 Newton–Schulz orthogonalization，再合并更新。目标是：

- 减少不同 attention head 的梯度 / momentum 尺度差异造成的相互压制；
- 让每个 head 的矩阵更新在自己的几何结构中归一化；
- 提升大规模训练时的稳定性；
- 略微降低 Newton–Schulz 正交化的优化器开销。

因此，§2.5 的关键不是“换了一个更好的 optimizer”，而是：

> **K3 既然把 attention 拆成了大量功能可能不同的 heads，就需要把参数更新也细化到 head 粒度，避免优化器把这些子系统重新粗暴耦合起来。**

需要严格区分两个图：

```text
forward graph：信息如何流动
    KDA / Gated MLA / AttnRes / Stable LatentMoE

backward-update graph：梯度如何变成参数更新
    Muon / Per-Head Muon / weight clipping / learning-rate schedule
```

Per-Head Muon 不改变推理时的 token 计算图，但改变了模型形成能力的训练动力学。

这里的 $g_t$ 是“优化器里的梯度”，不要和 KDA 公式中的 $g_t$（log-decay）混淆。两个符号相同，但语义完全不同。

#### 5.2 Muon 在做什么

普通 AdamW 的更新可以粗略写成：

$$
W_{t+1}
=
W_t-\eta \cdot \frac{m_t}{\sqrt{v_t}+\epsilon}
$$

它主要在 scalar 参数粒度上做自适应缩放。Muon 则把矩阵参数的 momentum 当成一个整体来处理：

$$
m_t=\beta m_{t-1}+(1-\beta)g_t
$$

然后对 $m_t$ 做 Newton–Schulz orthogonalization：

$$
\widetilde m_t\approx\operatorname{orth}(m_t)
$$

再用正交化后的矩阵更新参数：

$$
W_{t+1}=W_t-\eta\widetilde m_t
$$

这里的“正交化”不是把模型权重永久约束成正交矩阵，而是对**更新矩阵**做几何变换。若将矩阵更新写成 SVD：

$$
m=U\Sigma V^\top
$$

可以把 Muon 的作用粗略理解为：保留主要的左右奇异向量方向，同时弱化奇异值尺度差异对更新幅度的支配：

$$
\operatorname{orth}(m)\approx UV^\top
$$

因此，Muon 与 AdamW 的区别不是简单的“学习率更大或更小”，而是：

| 方法 | 主要处理对象 | 直觉 |
| --- | --- | --- |
| SGD | 原始梯度 | 梯度多大，更新就多大 |
| AdamW | 单个参数元素 | 每个 scalar 独立调节 |
| Muon | 矩阵更新的几何结构 | 不让少数奇异方向独占更新 |

上面对 Muon 的解释是机制层面的近似；技术报告 §2.5 本身没有展开 Newton–Schulz 的具体迭代式，而是直接说明其在 K3 中的结构化使用方式。

#### 5.3 Per-Head Muon 的数学结构

K3 有 96 个 attention heads，单 head dimension 为 128。以 Q 投影为例，完整矩阵可以按 head 拆成：

$$
W_q=
\begin{bmatrix}
W_q^{(1)}\\
W_q^{(2)}\\
\vdots\\
W_q^{(96)}
\end{bmatrix}
$$

其中：

$$
W_q^{(h)}\in\mathbb{R}^{128\times7168}
$$

完整 Q 投影矩阵的形状约为：

$$
W_q\in\mathbb{R}^{(96\times128)\times7168}
=\mathbb{R}^{12288\times7168}
$$

普通 full-matrix Muon 的处理方式是：

$$
\widetilde M_q=
\operatorname{orth}
\left(
\begin{bmatrix}
M_q^{(1)}\\
\vdots\\
M_q^{(96)}
\end{bmatrix}
\right)
$$

Per-Head Muon 则是：

$$
\widetilde M_q^{(h)}
=
\operatorname{orth}\left(M_q^{(h)}\right)
$$

最后再拼回完整更新矩阵：

$$
\widetilde M_q
=
\operatorname{concat}
\left(
\widetilde M_q^{(1)},
\ldots,
\widetilde M_q^{(96)}
\right)
$$

二者一般不等价：

$$
\operatorname{orth}
\left(
\operatorname{concat}_h M^{(h)}
\right)
\neq
\operatorname{concat}_h
\operatorname{orth}\left(M^{(h)}\right)
$$

Per-Head Muon 的作用，正是打断不必要的 head 间矩阵归一化耦合。它不会让 attention heads 在 forward 中彼此独立；Q/K/V 仍然会通过 attention 计算和 output projection 发生交互。被拆开的只是优化器对更新矩阵的处理粒度。

#### 5.4 为什么 full-matrix Muon 可能压制小尺度 head

K3 的 96 个 attention heads 虽然形状相同，但功能不一定相同。不同 head 可能偏向：

- 局部模式；
- 代码语法与括号结构；
- 长距离引用；
- 位置 / recency；
- KDA 状态写入与读取；
- MLA 的全局内容检索；
- 视觉 token 与文本 token 的交互。

因此，不同 head 的梯度或 momentum 尺度可能差异很大：

$$
\|M^{(1)}\|_F\gg\|M^{(2)}\|_F
$$

按照技术报告 §2.5 给出的直觉，full-matrix orthogonalization 把所有 heads 当成一个 coupled block。大梯度 / 大 momentum 的 head 更容易影响整体更新方向，而小尺度 head 得不到充分的独立归一化。

直观地说：

```text
full-matrix Muon
    所有 head 进入同一个矩阵级归一化器
    强信号 head 参与决定整体更新几何
    弱信号 head 的独立尺度可能被掩盖

Per-Head Muon
    每个 head 在自己的矩阵内部完成正交化
    head 的更新参考系彼此分开
    再把各 head 更新拼回 projection
```

这不是要把每个 head 的更新范数强行设成一样，而是：

> **让每个 head 的更新尺度由自己的矩阵结构决定，而不是由其他 head 的梯度规模间接决定。**

因此，Per-Head Muon 不能简单等同于“给每个 head 一个独立学习率”：

独立学习率只是：

$$
W_h\leftarrow W_h-\eta_hG_h
$$

Per-Head Muon 则是：

$$
W_h\leftarrow W_h-\eta\operatorname{orth}(M_h)
$$

前者只调节整体幅值，后者还改变矩阵更新的方向结构。

#### 5.5 为什么 K3 特别需要按 head 优化

##### 1. K3 的 attention 已经不是同质的标准 MHA

K3 采用约 3:1 的 Hybrid Attention：

- 69 层 KDA；
- 24 层 Gated MLA；
- backbone 末端额外增加一层 Gated MLA。

KDA 和 MLA 的职责不同：

| 模块 | 主要机制 | 更偏向的能力 |
| --- | --- | --- |
| KDA | 递归 state、delta update、channel-wise decay | 位置敏感、recency-aware、线性成本的历史混合 |
| Gated MLA | latent KV、全局 token-to-token attention | 不受限的全局内容检索 |

即便同一层内部的 head 形状相同，它们承担的统计角色也可能不同。优化器如果总把它们看成一个完整矩阵，就存在 forward 结构和 update 结构不匹配的问题。

##### 2. NoPE 将位置建模压力转移给 attention dynamics

K3 的 MLA 层使用 NoPE。报告的设计逻辑是：

- KDA 通过递归、decay 和 state update 提供 position-sensitive / recency-aware mixing；
- MLA 提供 unrestricted global content interaction。

这意味着不同 KDA head 可能学出不同的时间尺度：

- 有的 head 更保留近期信息；
- 有的 head 更保留远程状态；
- 有的 head 更偏向当前 token；
- 有的 head 更像稳定的历史 memory。

KDA 本身已经存在 per-head 的量：

$$
\alpha_t^{(h)},\quad
\beta_t^{(h)},\quad
A_h,\quad
b_\alpha^{(h)}
$$

这会进一步增强 head 的异质性。模型在 forward 中允许每个 head 拥有不同的时间动力学，optimizer 就不应再用一个过于粗粒度的全局矩阵归一化器处理所有 head。

##### 3. K3 将 head 数从 64 增加到 96

K3 与 K2 的相关变化是：

| 项目 | K2 | K3 |
| --- | ---: | ---: |
| Attention heads | 64 | 96 |
| Hidden size | 7168 | 7168 |
| Attention | MLA | Hybrid KDA–MLA |
| Training context | 128K | 1M |

hidden size 不变、head dimension 仍为 128，但 head 数增加意味着：

- 子空间分工更细；
- head 间梯度差异更容易放大；
- 更可能出现少数强 head 与大量弱 head；
- full-matrix update 的跨 head 耦合成本更高。

因此，Per-Head Muon 与 K3 的扩头、混合注意力和长上下文设计是配套的，而不是孤立的 optimizer trick。

#### 5.6 为什么它被放在 Model Architecture

从教科书分类看，Per-Head Muon 属于 optimization / training。但 K3 把它放在 §2 Model Architecture，原因不是排版偶然，而是作者采用了更宽的“架构”定义。

##### 第一层：K3 的架构是可训练系统，不只是 forward graph

K3 第 2 节把模型设计概括为沿三个信息流维度扩展：

| 维度 | K3 机制 | 解决的问题 |
| --- | --- | --- |
| Token mixing | KDA + Gated MLA | 长上下文中如何混合 token |
| Layer mixing | Attention Residuals | 深度方向如何选择历史表示 |
| Channel mixing | Stable LatentMoE | 如何扩大条件容量而不让每 token 成本爆炸 |
| Parameter update | Per-Head Muon | 如何让异质 attention heads 稳定学习 |

前三项改变 forward 的信息流，最后一项改变 backward 的参数流。K3 关注的不是单独某个模块，而是：

> **信息、表示、容量和梯度如何在一个超大模型中流动。**

##### 第二层：Per-Head Muon 是结构特化的，不是通用配置

如果报告只写“使用 AdamW、学习率为某值、weight decay 为某值”，它当然属于训练配方。

但 Per-Head Muon 依赖：

- Q/K/V projection 的矩阵形状；
- attention head 的切分方式；
- head dimension；
- 参数是否按 head block 组织；
- 分布式 optimizer 的参数分片；
- Newton–Schulz 对不同矩阵形状的计算方式。

如果 head 数、head dimension 或 projection layout 改变，Per-Head Muon 的实现也会随之改变。因此它更像：

- attention 的训练伴生结构；
- head-wise parameterization 的优化版本；
- architecture-aware optimizer。

##### 第三层：K3 的 scaling efficiency 依赖优化器把容量“兑现”

K3 报告声称相对 K2 有约 2.5× 的 overall scaling efficiency 提升。这个结果不能归因于某个模块的孤立收益，而是：

$$
\text{Scaling Efficiency}
=
f(
\text{attention},
\text{residual},
\text{MoE},
\text{vision},
\text{optimizer},
\text{data},
\text{systems}
)
$$

架构扩展如果没有匹配的优化动力学，新增的参数和子空间未必能转化为有效能力。Per-Head Muon 的定位就是把结构扩展转化为可训练性和有效 scaling。

#### 5.7 它与 K3 其他稳定化设计的关系

K3 的稳定性不是靠一个技巧，而是对不同动力学层次分别加约束：

| 机制 | 被稳定的对象 | 主要手段 |
| --- | --- | --- |
| KDA lower-bounded decay | token / recurrent state dynamics | 将 log-decay 限制在有限区间 |
| Attention Residuals | layer / depth dynamics | 深度方向的选择性聚合 |
| Stable LatentMoE | channel / expert dynamics | RMSNorm、SiTU-GLU、Quantile Balancing |
| Per-Head Muon | attention parameter dynamics | head 内独立的矩阵正交化 |

可以写成：

```text
KDA decay gate        → 稳定 token/state dynamics
AttnRes               → 稳定 layer/depth dynamics
Stable LatentMoE      → 稳定 channel/expert dynamics
Per-Head Muon         → 稳定 parameter/update dynamics
```

所以，Per-Head Muon 不是和 KDA、AttnRes、LatentMoE 平行的又一个 forward 模块，而是 K3 设计的“优化维度”：

$$
\text{K3 effective architecture}
=
\text{forward information flow}
+
\text{optimization dynamics}
$$

#### 5.8 与 §3.3 Training Recipe 的关系

报告在 §3.3 又写道：

> We optimize the model using the Per-Head Muon optimizer together with the weight-clipping mechanism introduced in Kimi K2.

这不是重复，而是两个层次：

```text
§2.5：为什么要设计 Per-Head Muon，它针对什么结构问题
§3.3：训练时如何实际使用它，并与哪些训练配方组合
```

§3.3 同时给出：

- Per-Head Muon；
- weight clipping；
- Quantile Balancing；
- cosine learning-rate schedule；
- 1% linear warmup；
- weight decay = 0.1。

这说明 Per-Head Muon 的设计理由在架构章节，具体部署方式在训练章节。

#### 5.9 不要过度解读：它不保证每个 head 都学得一样好

Per-Head Muon 只能解决一种优化耦合问题：

> 某个 head 的大梯度 / 大 momentum 不应过度决定其他 head 的矩阵更新尺度。

它不能保证：

- 每个 head 都有用；
- 所有 head 都被均匀使用；
- attention heads 不会冗余；
- 所有 head 的功能都能被解释；
- 训练一定不会发生 head collapse；
- KDA 与 MLA 的分工一定最优。

它也不是简单地把每个 head 的 update norm 强行设成完全一样。更准确的说法是：

> **每个 head 在自己的矩阵内部进行几何归一化，减少跨 head 尺度差异造成的间接压制。**

#### 5.10 证据强度与待验证问题

技术报告 §2.5 给出了清晰的机制动机，但公开证据仍然有限。报告没有详细提供：

- full-matrix Muon vs. Per-Head Muon 的独立 ablation；
- Per-Head Muon vs. AdamW 的对照；
- KDA 与 MLA 分别使用 Per-Head Muon 的结果；
- head-wise gradient norm / update norm 分布；
- 正交化前后的 singular-value spectrum；
- loss、scaling law 和下游 benchmark 的独立增益。

因此应做如下判断：

| 判断维度 | 评价 |
| --- | --- |
| 机制动机 | 中高 |
| 与 K3 结构的匹配性 | 高 |
| 独立性能贡献是否已证实 | 中低 |
| 工程价值 | 较高 |
| 能否单独解释 K3 能力提升 | 不能 |

不要把“设计合理”误写成“贡献已被证明”。目前我们知道它为什么可能有效，但不知道它对 K3 最终能力贡献了多少。

#### 5.11 对 2.5 的最终理解

K3 的四个主要架构维度可以这样记：

```text
序列维 token mixing
    └── KDA + Gated MLA
          解决：1M context 下如何混合 token

深度维 layer mixing
    └── Attention Residuals
          解决：93 层之间如何选择历史表示

宽度维 channel mixing
    └── Stable LatentMoE
          解决：如何扩大专家容量而不让每 token 成本爆炸

优化维 parameter update
    └── Per-Head Muon
          解决：如何让 96 个异质 attention heads 稳定学习
```

最重要的一句话：

> **KDA、AttnRes、Stable LatentMoE 在扩大 K3 的信息处理能力；Per-Head Muon 在保证这些新增自由度不会因为错误的更新几何而学不出来。**

下一步最值得做的验证实验是：对 full-matrix Muon、Per-Head Muon 和 AdamW 做三方对照，并分别在 KDA / MLA 上记录 gradient norm、update norm、singular-value spectrum、训练 loss scaling 以及下游任务结果。否则只能回答“为什么它可能有效”，不能回答“它到底贡献了多少”。

#### 5.12 从 $g_t$ 到 $m_t$：为什么它们是矩阵

这里最容易混淆的一点是：**梯度不是天然的标量或向量；梯度的形状跟着参数走。**

设某个 attention projection 的参数是矩阵：

$$
W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}
$$

那么损失函数对它的梯度定义为：

$$
G=\nabla_W\mathcal{L}
=
\left[
\frac{\partial\mathcal{L}}{\partial W_{ij}}
\right]
\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}
$$

也就是说，$G_{ij}$ 只是“损失对参数 $W_{ij}$ 的偏导数”。因为 $W$ 有多少个元素，梯度就有多少个对应元素，所以 $G$ 与 $W$ 同形状。

更严格地说，把矩阵空间看成带 Frobenius 内积的向量空间，梯度满足：

$$
d\mathcal{L}
=
\left\langle\nabla_W\mathcal{L},dW\right\rangle_F
=
\operatorname{tr}
\left(
(\nabla_W\mathcal{L})^\top dW
\right)
$$

因此：

$$
W\in\mathbb{R}^{m\times n}
\quad\Longrightarrow\quad
\nabla_W\mathcal{L}\in\mathbb{R}^{m\times n}
$$

##### 一个线性层的具体例子

令：

$$
y=Wx,
\qquad
W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}},
\quad
x\in\mathbb{R}^{d_{\mathrm{in}}},
\quad
\delta=\frac{\partial\mathcal{L}}{\partial y}
\in\mathbb{R}^{d_{\mathrm{out}}}
$$

因为：

$$
d\mathcal{L}
=
\delta^\top dy
=
\delta^\top(dW)x
$$

所以：

$$
g
=
\nabla_W\mathcal{L}
=
\delta x^\top
\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}
$$

它是一个外积矩阵。对一个 batch，若输入矩阵为 $X$、上游梯度为 $\Delta$，则：

$$
G
=
\frac{1}{B}\Delta^\top X
$$

仍然与 $W$ 同形状。训练框架中的 `g_t` 通常就是这个 batch 梯度，可能还经过 data-parallel 的聚合。

对 K3 的 Q 投影，可以粗略写成：

$$
W_q\in\mathbb{R}^{(96\times128)\times7168}
=
\mathbb{R}^{12288\times7168}
$$

因此：

$$
g_t,\;m_t\in\mathbb{R}^{12288\times7168}
$$

在 Per-Head Muon 中，第 $h$ 个 head 的切片则是：

$$
W_q^{(h)},\;g_t^{(h)},\;m_t^{(h)}
\in\mathbb{R}^{128\times7168}
$$

##### 那么 $m_t$ 为什么也是矩阵

因为 momentum 不是一个额外的标量，而是**为每个矩阵参数位置保存一个历史梯度状态**：

$$
m_t
=
\beta m_{t-1}
+
(1-\beta)g_t
$$

对每个元素展开就是：

$$
(m_t)_{ij}
=
\beta(m_{t-1})_{ij}
+
(1-\beta)(g_t)_{ij}
$$

所以：

- $\beta$ 通常是一个标量超参数；
- $g_t$ 是当前 step 的梯度矩阵；
- $m_{t-1}$ 是上一 step 的 momentum 矩阵；
- $m_t$ 是当前 step 的 momentum 矩阵。

它本质上是对**每一个权重位置的梯度做指数滑动平均**。矩阵结构不是由 momentum 公式创造的，而是继承自它所服务的参数 $W$。

这也解释了一个边界：如果参数是 bias、LayerNorm 的 scale 或其他一维向量，那么对应的梯度和 momentum 就是向量；如果参数是标量，则是标量。K3 报告说 Muon 用于 matrix parameters，而不是把所有参数都强行送进 Muon。

#### 5.13 Newton–Schulz orthogonalization 到底做了什么

先把问题说准确：

> Newton–Schulz 不是把 K3 的权重 $W$ 变成正交矩阵，而是把当前的 momentum / update matrix $m_t$ 变成一个**近似 polar factor**，再用这个方向更新 $W$。

##### 目标：去掉更新矩阵中不均衡的奇异值

对 momentum 矩阵做 SVD：

$$
m=U\Sigma V^\top
$$

其中：

- $U$：输出空间中的方向；
- $V$：输入空间中的方向；
- $\Sigma=\operatorname{diag}(\sigma_1,\sigma_2,\ldots)$：各方向的更新强度。

Muon 希望使用：

$$
\operatorname{polar}(m)
=
UV^\top
$$

而不是直接使用：

$$
m=U\Sigma V^\top
$$

所以它保留 $U,V$ 描述的“朝哪个输入方向、推向哪个输出方向”，但把不同方向的强度 $\sigma_i$ 拉平为近似相同的尺度。

更正式地，$m$ 的 polar decomposition 可以写成：

$$
m=QH
$$

其中：

$$
Q=UV^\top,
\qquad
H=V\Sigma V^\top
$$

$Q$ 是 polar factor。对于方阵，$Q^\top Q=I$；对于长方形矩阵，它通常是 partial isometry：

- 若矩阵高于宽，通常满足 $Q^\top Q=I$；
- 若矩阵宽于高，通常满足 $QQ^\top=I$。

因此，“orthogonalization”是工程上的简称；严格说，它对长方形矩阵产生的是具有正交行或正交列的 partial orthogonal matrix，而不一定是方阵意义上的 orthogonal matrix。

##### 一个最小例子

假设当前 momentum 为：

$$
m=
\begin{bmatrix}
10&0\\
0&1
\end{bmatrix}
$$

直接使用 momentum，意味着第一个方向的更新幅度是第二个方向的 10 倍。

它的 SVD 中：

$$
U=V=I,
\qquad
\Sigma=
\begin{bmatrix}
10&0\\
0&1
\end{bmatrix}
$$

polar factor 是：

$$
Q=UV^\top=I
$$

于是 Muon 大致把更新改成：

$$
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
$$

它没有改变两个方向本身，只是取消了“第一个方向因为奇异值更大，就独占 10 倍更新幅度”的现象。

所以可以用一句话理解：

> **Muon 保留更新矩阵的方向结构，压平其奇异值谱。**

这和只除以 Frobenius norm 不一样。若只做：

$$
\widehat m=\frac{m}{\|m\|_F}
$$

奇异值比例仍然是 $10:1$；它只是把整张矩阵统一缩放，并没有消除方向之间的相对不均衡。

#### 5.14 Newton–Schulz 如何近似这个 polar factor

直接用 SVD 求 $UV^\top$ 很昂贵，尤其是 K3 这种大规模分布式训练。Newton–Schulz 提供了一个只使用矩阵乘法和加法的迭代近似。

一个常见的 polar iteration 写成：

$$
X_0=\frac{m}{s}
$$

其中 $s$ 是足以控制初始谱范围的缩放因子，然后反复迭代：

$$
X_{k+1}
=
\frac{1}{2}
X_k
\left(3I-X_k^\top X_k\right)
$$

为了看清它在做什么，设：

$$
X_k=U\Sigma_kV^\top
$$

由于：

$$
X_k^\top X_k
=
V\Sigma_k^2V^\top
$$

代入迭代式后，$U,V$ 基本保持不变，只有奇异值按下面的标量函数变化：

$$
\sigma_{k+1}
=
\frac{1}{2}\sigma_k(3-\sigma_k^2)
$$

这个函数的目标是把非零 $\sigma_k$ 推向 $1$：

$$
\sigma_k\rightarrow1
$$

于是：

$$
X_k
\rightarrow
UV^\top
$$

也就是 polar factor。

更底层地说，polar factor 可以写成：

$$
\operatorname{polar}(m)
=
m(m^\top m)^{-1/2}
$$

Newton–Schulz 实际上是在用迭代方式近似 $(m^\top m)^{-1/2}$，但不显式做矩阵开方或 SVD。

Muon 的工程实现通常只做固定次数的低阶多项式迭代，而不是追求精确收敛。这样可以把正交化转化为 GPU 擅长的 GEMM，避免显式 SVD 的高成本。官方实现使用的是有限步数的 quintic Newton–Schulz 迭代，而不是上面用于解释的无限收敛 cubic 形式；代码明确说明，有限步输出不一定精确等于 $UV^\top$，而是保留相近奇异方向、把奇异值推入有限范围的近似结果。[Muon 官方实现](https://github.com/KellerJordan/Muon/blob/master/muon.py)

需要注意：上面的三次迭代式是帮助理解的标准形式。Muon 实际实现会使用归一化和更适合有限步数的多项式系数，具体细节取决于实现版本。

#### 5.15 为什么要做：Muon 的收益在哪里

##### 收益一：更新不再被少数奇异方向支配

原始 momentum 的奇异值可能是：

$$
\Sigma=\operatorname{diag}(100,10,1,0.1)
$$

直接更新会让第一个方向成为绝对主导；polar update 则将有效方向的尺度拉到相近水平。

这不是“让所有参数元素一样大”，而是让**矩阵的主方向**获得更均衡的更新机会。对于线性层来说，这比逐元素归一化更贴近它本身的矩阵结构。

##### 收益二：保留矩阵结构，而不是把矩阵粗暴 flatten 成向量

Adam 类方法把参数看成大量 scalar，逐元素处理：

$$
\Delta W_{ij}
\propto
\frac{m_{ij}}{\sqrt{v_{ij}}+\epsilon}
$$

但矩阵的功能依赖输入子空间和输出子空间之间的整体关系。Muon 直接在矩阵的奇异方向上处理更新，因此对线性层、attention projection 这样的二维参数更“结构感知”。

这不是说 AdamW 错了，而是二者的归一化坐标系不同：

| 方法 | 归一化参考系 | 可能保留 / 改变的东西 |
| --- | --- | --- |
| AdamW | 参数坐标轴 $(i,j)$ | 逐元素尺度 |
| Muon | 矩阵奇异方向 | 整体子空间方向与谱形状 |

##### 收益三：对 K3 的 Per-Head Muon，能够进一步平衡 head 之间的更新

这是 K3 采用 per-head 变体的关键。

完整 Q 投影可以看成把所有 head 堆叠起来：

$$
M=
\begin{bmatrix}
M^{(1)}\\
\vdots\\
M^{(96)}
\end{bmatrix}
$$

如果对整个高矩阵做 polar update：

$$
Q^\top Q=I
$$

这只能约束整个矩阵的列空间；不同 head 对应的行 block 的 Frobenius norm 不必相同。

而如果每个 head 单独做 polar update，且每个 head 的矩阵为 $128\times7168$：

$$
Q_hQ_h^\top=I_{128}
$$

于是：

$$
\|Q_h\|_F^2
=
\operatorname{tr}(Q_hQ_h^\top)
=128
$$

只要各 head 形状相同，每个 head 的更新矩阵就天然拥有相同的 Frobenius 规模（忽略实现中的额外缩放、近似误差和学习率因素）。

这给出一个比“让 head 更均衡”更准确的解释：

> **full-matrix Muon 约束整个 projection 的全局几何；Per-Head Muon 约束每个 head 自己的局部几何，因此不会让某些 head 的大尺度更新轻易吞掉其他 head 的更新预算。**

这是从矩阵形状和 polar 条件推导出的机制解释，不是 K3 报告中已经单独验证过的 ablation 结论。

##### 收益四：比精确 SVD 更适合大规模 GPU 训练

精确 SVD 能直接给出 $UV^\top$，但在大矩阵、分布式训练和每一步都要执行的 optimizer path 中代价很高。Newton–Schulz 使用固定次数矩阵乘法，可以更容易地：

- 映射到 GPU Tensor Core / GEMM；
- 与通信和其他 optimizer work overlap；
- 对固定形状做 kernel 优化；
- 避免显式构造完整 SVD 中间结果。

这也是 K3 报告称 per-head block 的 Newton–Schulz iteration 略微降低 optimizer overhead 的原因。

#### 5.16 这个收益不是免费的

不能把 orthogonalization 解释成“永远更稳定、永远更快”。它有明确代价和风险：

1. **丢掉了奇异值幅度信息。** 如果某个方向本来就应该小，polar update 仍可能给它相近的更新尺度。
2. **噪声方向可能被放大。** 低奇异值方向如果主要是 minibatch noise，强行拉平谱可能不是好事。
3. **需要合适的学习率和缩放。** 正交化只决定方向结构，不自动解决 update magnitude 的全部问题。
4. **不适合所有参数。** embedding、bias、Norm scale 等一维或特殊结构参数通常需要其他 optimizer 处理。
5. **K3 的独立收益仍未被充分证明。** 报告给出了合理的机制动机，但没有公开完整的 AdamW / full-Muon / Per-Head-Muon 三方 ablation。

因此，最准确的结论是：

> **Newton–Schulz orthogonalization 把“矩阵更新的方向”和“各方向的强度”分离：保留前者，压平后者。它的潜在收益是让矩阵参数更充分、更均衡地利用可学习子空间；K3 的 Per-Head 版本又把这种均衡从整个 Q/K/V projection 细化到每个 attention head。**

#### 5.17 用一句伪代码把整个过程串起来

```python
# W: 一个 attention projection matrix
# g: 当前 minibatch 对 W 的梯度，shape 与 W 相同
# m: W 专属的 momentum buffer，shape 与 W 相同

m = beta * m + (1 - beta) * g

# full Muon:
u = newton_schulz_polar(m)
W = W - lr * u

# Per-Head Muon:
for h in heads:
    m_h = split_by_head(m, h)
    u_h = newton_schulz_polar(m_h)
    u = concat_by_head(u_h)
W = W - lr * u
```

最终要记住三层关系：

```text
W：模型正在学习的矩阵参数
g：当前 batch 对 W 的矩阵梯度
m：g 的时间平滑矩阵
polar(m)：保留矩阵更新方向、压平奇异值尺度后的更新方向
```

#### 5.18 对前述解释的一个必要校正

把 Newton–Schulz 写成：

$$
X_{k+1}
=
\frac12X_k(3I-X_k^\top X_k)
$$

是为了说明“为什么它会把奇异值推向 1”的标准教学版本；K3 实际采用的 Muon 具体多项式、迭代步数、转置方向、数值精度和 update scale 需要以实现为准。官方 Muon 实现使用 quintic 迭代，并明确说明有限步结果并非精确的 $UV^\top$。[Muon 官方代码](https://github.com/KellerJordan/Muon/blob/master/muon.py)

因此，本文中：

$$
\operatorname{orth}(m)\approx UV^\top
$$

表示 **exact polar orthogonalization 的理想化解释**，不是说 K3 每一步真的计算了 SVD，也不是说实际 update 的每个奇异值都严格等于 1。

关于收益，也要区分“机制解释”和“已被 K3 单独证明的收益”：

- Muon 的矩阵正交化、weight decay 和 update scale 设计已有针对大模型训练的独立研究；
- 相关 scaling-law 实验报告了相对 AdamW 的计算效率优势，但那是特定模型、数据和超参数下的实证，不应直接当作 K3 的独立 ablation 结论；
- 最新理论工作分析了有限步 Newton–Schulz 如何逼近 exact-polar 理想化，以及其相对 momentum SGD 的 rank dependence，但这仍然不能替代 K3 自己的 full-Muon / Per-Head-Muon 对照实验。[Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)；[Convergence of Muon with Newton–Schulz](https://arxiv.org/abs/2601.19156)

#### 5.19 本轮追问：$g_t$ / $m_t$ 的矩阵形状与 Newton–Schulz 的真实作用

##### 1. 先抓住核心

$$
\boxed{
W\text{ 是什么形状，}\;
\nabla_W\mathcal{L}\text{、momentum buffer、Muon update 就首先是什么形状}
}
$$

讨论 K3 的 Q/K/V projection 时，参数本身是矩阵，因此：

$$
W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}
$$

其梯度：

$$
g_t
=
\nabla_W\mathcal{L}_t
\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}
$$

其 momentum：

$$
m_t
=
\beta m_{t-1}+(1-\beta)g_t
\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}
$$

所以 $m_t$ 不是一个“动量标量”，而是**每一个矩阵权重位置各自保存的历史梯度平滑值**。

另外，K3 里有两个同名的 $g_t$：

- 优化器里的 $g_t$：gradient；
- KDA 里的 $g_t$：log-decay。

它们只是符号碰巧相同，不能混为一谈。

##### 2. 为什么 $g_t$ 是矩阵

考虑一个线性层：

$$
y=Wx
$$

其中：

$$
W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}},
\quad
x\in\mathbb{R}^{d_{\mathrm{in}}},
\quad
\delta=\frac{\partial\mathcal{L}}{\partial y}
\in\mathbb{R}^{d_{\mathrm{out}}}
$$

反向传播给出的矩阵梯度是：

$$
g
=
\nabla_W\mathcal{L}
=
\delta x^\top
\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}
$$

第 $i,j$ 个元素为：

$$
g_{ij}
=
\frac{\partial\mathcal{L}}{\partial W_{ij}}
$$

也就是：

> **每一个权重 $W_{ij}$ 都有一个对应的偏导数 $g_{ij}$，所以把所有偏导数按原来的权重布局排起来，得到的就是梯度矩阵。**

对 batch，若将输入样本按列堆成 $X$、上游梯度按列堆成 $\Delta$，则：

$$
G
=
\frac{1}{B}\Delta X^\top
$$

仍然和 $W$ 同形状。不同框架若将 token 放在行上，公式会写成 $\Delta^\top X$；形状结论不变。

以 K3 的 Q projection 为例：

$$
W_q\in\mathbb{R}^{(96\times128)\times7168}
=\mathbb{R}^{12288\times7168}
$$

因此：

$$
g_t,\;m_t
\in\mathbb{R}^{12288\times7168}
$$

按 head 切分后，第 $h$ 个 head 的局部矩阵是：

$$
W_q^{(h)},\;g_t^{(h)},\;m_t^{(h)}
\in\mathbb{R}^{128\times7168}
$$

##### 3. 为什么 $m_t$ 也是矩阵

把 momentum 公式逐元素展开：

$$
(m_t)_{ij}
=
\beta(m_{t-1})_{ij}
+
(1-\beta)(g_t)_{ij}
$$

所以每一个位置 $(i,j)$ 都有自己的历史平均：

```text
m[0, 0]：W[0, 0] 的历史梯度平滑值
m[0, 1]：W[0, 1] 的历史梯度平滑值
...
m[i, j]：W[i, j] 的历史梯度平滑值
```

矩阵形式只是把这些标量状态保留在和参数相同的二维布局中。$\beta$ 通常是一个共享的标量超参数；它不是把整个矩阵压成一个数。

因此，优化器状态的层级是：

```text
W：模型参数矩阵
g：当前 batch 对 W 的梯度矩阵
m：g 的指数滑动平均矩阵
U：对 m 做矩阵级变换后的 update 矩阵
```

##### 4. Newton–Schulz 到底在做什么

设当前要处理的矩阵更新为 $M$。做 SVD：

$$
M=U\Sigma V^\top
$$

其中：

- $U,V$ 描述更新作用的输入 / 输出方向；
- $\Sigma$ 描述各个方向的更新强度。

普通 momentum 直接使用：

$$
\Delta W\propto M=U\Sigma V^\top
$$

如果 $\Sigma$ 很不均衡，例如：

$$
\Sigma=\operatorname{diag}(10,1,0.1)
$$

那么第一个奇异方向会获得约 10 倍于第二个方向的更新强度。

理想化的 polar orthogonalization 想得到：

$$
Q=UV^\top
$$

也就是：

- 保留“沿哪些输入 / 输出方向更新”；
- 去掉奇异值 $\Sigma$ 对方向之间的巨大幅度差异。

因此可以把它理解为：

> **Newton–Schulz 不是在寻找新的梯度方向，而是在重塑同一组矩阵方向的相对更新强度。**

##### 5. 它为什么叫 orthogonalization

对方阵，$Q=UV^\top$ 满足：

$$
Q^\top Q=I
$$

对 K3 的 per-head Q/K/V block，矩阵是 $128\times7168$，属于宽矩阵。理想化情况下更准确的关系是：

$$
QQ^\top=I_{128}
$$

这表示每个 head 的 128 个输出方向在更新矩阵里被整理成近似正交、近似等尺度的方向。

严格说，长方形矩阵对应的是 partial isometry；“orthogonalization”是工程上更简洁的说法，不代表它是方阵意义上的正交矩阵。

##### 6. Newton–Schulz 如何实现这个目标

教学上常用的 cubic 版本是：

$$
X_0=\frac{M}{s}
$$

$$
X_{k+1}
=
\frac12X_k(3I-X_k^\top X_k)
$$

若：

$$
X_k=U\Sigma_kV^\top
$$

则 $U,V$ 基本不变，奇异值逐个经过：

$$
\sigma_{k+1}
=
\frac12\sigma_k(3-\sigma_k^2)
$$

该标量迭代会把合适区间内的 $\sigma_k$ 推向 1，于是：

$$
X_k\rightarrow UV^\top
$$

更底层的目标是近似 polar factor：

$$
\operatorname{polar}(M)
=
M(M^\top M)^{-1/2}
$$

对于宽矩阵，可用等价的左侧形式。Newton–Schulz 的价值在于：不显式做 SVD 或矩阵平方根，而是用若干次矩阵乘法逼近它。

##### 7. Muon 实际并不精确计算 $UV^\top$

这里必须修正一个容易被教学公式误导的点：

> **标准 cubic 迭代可以收敛到理想化的 polar factor，但官方 Muon 实现使用有限步的 quintic Newton–Schulz，不追求每个奇异值精确收敛到 1。**

官方实现先对矩阵做缩放，然后使用 quintic 多项式迭代：

$$
X\leftarrow aX+b(XX^\top)X+c(XX^\top)^2X
$$

代码中的一组系数为：

$$
a=3.4445,\qquad b=-4.7750,\qquad c=2.0315
$$

官方代码明确说明：有限步输出更接近 $US'V^\top$，其中 $S'$ 的奇异值大致落在有限区间，而不是严格等于 1；它是一个**近似的、工程化的 zeroth-power / orthogonalized update**。[Muon 官方实现](https://github.com/KellerJordan/Muon/blob/master/muon.py)

所以前面写：

$$
\operatorname{orth}(M)\approx UV^\top
$$

应理解为**理想化解释**，不是说实际训练每一步都运行了一次精确 SVD。

##### 8. 一个最小例子

假设：

$$
M=
\begin{bmatrix}
10&0\\
0&1
\end{bmatrix}
$$

直接使用 $M$：

```text
方向 1：更新强度 10
方向 2：更新强度 1
```

第一个方向主导更新。

理想化 polar factor：

$$
M=I
\begin{bmatrix}
10&0\\
0&1
\end{bmatrix}
I
$$

所以：

$$
\operatorname{polar}(M)=I
$$

这保留了两个坐标方向，但不让第一个方向因为奇异值是 10 就获得 10 倍更新。

注意，单纯做全局 Frobenius 归一化：

$$
\widehat M=\frac{M}{\|M\|_F}
$$

只能把整体缩小，奇异值比例仍是 $10:1$。它解决的是“整张矩阵太大”，不是“不同矩阵方向之间过度不均衡”。

##### 9. 为什么要这样做

收益不是“梯度变得更正确”，而是引入一种矩阵级更新偏置：

1. **防止少数奇异方向吞掉更新预算。** 大方向不会仅凭更大的 singular value 就独占绝大部分 update。
2. **保留矩阵子空间结构。** Muon 不是把矩阵 flatten 成一条向量，而是在矩阵的奇异方向上处理更新。
3. **与线性层 / attention projection 的几何更匹配。** Q/K/V 本来就是把输入子空间映射到输出子空间，矩阵级处理比逐 scalar 归一化更直接。
4. **为 Per-Head Muon 提供局部均衡。** 每个 head 单独正交化后，各 head 不再共享一个跨 block 的全局谱竞争环境。
5. **工程上可用 GEMM 近似，不需要每一步跑 SVD。** 官方实现将 Newton–Schulz 描述为 GPU 上可稳定运行的矩阵正交化路径，并且只建议对 hidden weight matrices 使用 Muon，embedding、输出头、bias 和 norm gain 等参数仍使用其他优化器。[Muon 官方实现](https://github.com/KellerJordan/Muon/blob/master/muon.py)

##### 10. Per-Head 版本为什么比 full-matrix 更符合 K3

完整投影矩阵：

$$
M=
\begin{bmatrix}
M^{(1)}\\
\vdots\\
M^{(96)}
\end{bmatrix}
$$

如果整体正交化，它主要约束整张矩阵的全局列 / 行空间；某些 head 仍可能在整体矩阵中占据更大的更新尺度。

如果每个 head 独立处理：

$$
Q_h=\operatorname{NS}(M^{(h)})
$$

则每个 head 在自己的局部矩阵空间里完成谱整理。对于理想化的 $128\times7168$ block：

$$
Q_hQ_h^\top=I_{128}
$$

从而：

$$
\|Q_h\|_F^2
=
\operatorname{tr}(Q_hQ_h^\top)
=128
$$

这解释了“per-head equalizes update scale”的数学直觉：不是把每个 scalar 设成一样，而是让每个同形状 head 的局部更新矩阵具有相近的整体尺度。

##### 11. 代价与边界

这个方法不是无条件优越：

- 它会削弱原始梯度中的奇异值幅度信息；
- 很小的奇异方向可能是噪声，也可能是尚未学会的有用子空间，正交化无法自动区分；
- 需要配合正确的学习率 / update scale 和 weight decay；
- 并非所有参数都适合 Muon；
- K3 报告没有公开 full-Matrix Muon、Per-Head Muon、AdamW 的完整三方 ablation。

因此必须区分：

```text
机制收益：让矩阵更新的方向得到更均衡的机会
实证收益：是否更快收敛、是否更低 loss、是否更强 benchmark
```

后者不能只由公式推出。独立的 Moonlight 研究报告，在特定 scaling-law 实验中声称 Muon 相对 AdamW 约有 2× computational efficiency；但那不是 K3 对 Per-Head Muon 独立贡献的证明，且该研究同时强调 weight decay 和 per-parameter update scale 对扩展到大模型很关键。[Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)

##### 12. 最短记忆版

```text
W：要学习的矩阵
g_t：当前 batch 对 W 的矩阵梯度
m_t：g_t 的时间平滑矩阵

普通 momentum：
    直接用 m_t 更新，奇异值大的方向更新更强

Newton–Schulz：
    不改变主要奇异向量方向
    通过矩阵多项式把奇异值谱压到更均衡的范围

Per-Head Muon：
    不对整个 Q/K/V projection 一锅端
    而是每个 attention head 单独做这件事
```

---

## 预训练

### 3.3 Training Recipe：让架构真正学出来

这一节的本质不是介绍新模块，而是说明：**K3 如何把原生多模态架构训练成一个统一模型。**

核心做法有四点：

1. **从一开始联合训练视觉和语言**：图像、视频与文本 token 交错进入同一个 backbone，共同优化 next-token prediction，而不是先训练语言模型、再外挂视觉塔做事后对齐。核心目标是让视觉表示从一开始就服从语言与任务目标。
2. **用与架构匹配的优化稳定机制**：Per-Head Muon 处理 attention 矩阵更新，weight clipping 控制权重异常，Quantile Balancing 保证 MoE 专家负载均衡。
3. **采用保守的学习率配方**：cosine decay、1% linear warmup、weight decay = 0.1，控制 2.8T 参数模型的训练动态。
4. **逐步增加上下文长度**：先以 8K token 训练，后续扩展到 64K；更长的 256K → 1M curriculum 属于 §3.4 的 long-context extension。

一句话概括：

> **§3.3 解决的不是“模型能不能表达”，而是“视觉、语言、稀疏专家和注意力子空间能不能在同一套训练动力学下稳定形成”。**

其中最重要的是第一点：K3 把多模态融合从“后处理对齐问题”改成了“从训练第一天就共同学习的问题”；其余配置则是在保证这个超大、稀疏、长上下文系统不失稳。

### 数据

- 四大文本域: Web Text, Code, Mathematics, Knowledge
- 大规模视觉语料: captions, interleaved image-text, OCR, perception, video, visual coding
- 原生多模态训练 (language 和 vision 从训练开始即联合优化)

### Scaling Law

- 对比 Kimi K2: **cosine decay 优于 WSD** (在各自最优超参数下)
- 结论: 累积改进带来 2.5x 缩放效率提升

### 3.4 Long-Context Extension：如何做到 1M 窗口

> 长上下文各模型路线的系统横向对比见：[[Long Context：从1M窗口到有效长程能力]]

#### 先给结论

K3 的 1M context 不是靠某一个“超长位置编码技巧”实现的，而是同时解决了三个问题：

```text
架构：能不能处理 1M token
数据：模型有没有学会使用远处信息
系统：训练 1M token 是否负担得起
```

#### 1. 架构：NoPE + KDA 让长上下文在机制上可扩展

K3 的 MLA 层不使用显式 positional embedding（NoPE），位置和 recency 信息主要由 KDA 的递归 state、channel-wise decay 和 input-dependent gate 隐式编码：

$$
S_t=M_tS_{t-1}+\beta_tk_tv_t^\top
$$

其中 $S_t$ 是固定大小的 recurrent state，不随历史长度 $t$ 线性增长。这样做的直接收益是：

- 不需要为 1M context 重新调 RoPE frequency base；
- 不需要做 RoPE interpolation / YaRN；
- KDA 的状态大小主要由 head dimension 决定，而不是由 token 数决定；
- KDA 可以承担大多数序列混合，避免所有层都执行标准全局 attention。

K3 使用约 3:1 的 KDA : Gated MLA：

```text
KDA → KDA → KDA → Gated MLA → ...
```

- KDA：以固定大小的递归状态高效处理长序列、recency 和状态记忆；
- Gated MLA：周期性提供全局内容检索，避免线性 attention 只依赖压缩状态而丢失精确远程信息。

但要严谨：**KDA 不是让整个模型完全变成线性复杂度**。Gated MLA 仍然承担全局 attention 成本；K3 的策略是让大多数层走便宜的 KDA，少数层保留全局检索能力。

NoPE 解决的是“位置编码如何外推”；KDA 解决的是“历史信息如何以固定状态持续传递”。二者结合，才使 1M extension 在架构上可行。

#### 2. 数据：让模型真的必须跨越 1M 取信息

“输入长度支持 1M”不等于“模型会使用 1M”。如果训练样本中的答案总在局部上下文里，模型会学会局部 shortcut，即使上下文窗口标成 1M，也不会真正检索远处内容。

因此 K3 做了三件事：

1. 清理自然长文档和视频：去重、质量过滤、结构校验、视频 frame perceptual hashing，去除截断文件、二进制垃圾和低质量日志；
2. 对稀缺的长且连贯的数据进行 upsample，避免它们被海量短文本淹没；
3. 合成超长多模态样本：排列、拼接多个文档和子任务，使任务所需信息分散在整个 1M context 中，只有跨远距离检索才能完成。

核心不是“把序列塞到 1M”，而是：

> **把监督信号放到 1M 的距离上，迫使模型学习跨长距离取证。**

#### 3. 训练：渐进式扩大窗口，而不是一开始就训练 1M

K3 使用四阶段 context curriculum：

$$
8K\rightarrow64K\rightarrow256K\rightarrow1M
$$

其中：

- 8K → 64K：在常规预训练阶段逐步扩展；
- 256K → 1M：在 cooldown 阶段进行长上下文适应。

原因很现实：1M 序列的计算、显存和通信成本极高，不能让全部训练 token 都使用 1M。K3 将昂贵的超长序列训练集中在训练后段的一小部分预算里：

```text
先学通用语言能力
    ↓
再适应更长的依赖
    ↓
最后用少量 1M 训练校准长程行为
```

这同时解决两个问题：

- 训练成本可控；
- 模型从短依赖逐步过渡到长依赖，不会一开始就面对极端优化难度。

#### 4. 系统：KCP 把 1M 序列切到多个设备上

即使 KDA 使用固定大小 state，1M token 的激活、梯度和计算仍然很重。因此 K3 在 §5.1.2 使用 KDA Context Parallelism（KCP）。

将序列切成 $P$ 个 segment，每个 rank 处理一个 segment。每个 rank 不传输完整历史 KV，而是本地计算两个固定规模对象：

1. **cumulative transition**：当前 segment 对输入 state 的变换；
2. **zero-state contribution**：从零 state 开始、由当前 segment 自己产生的 state。

对第 $i$ 个 segment，可抽象为：

$$
S_{\text{out}}^{(i)}
=
\widetilde S^{(i)}
+
M^{(i)}S_{\text{in}}^{(i)}
$$

其中：

- $M^{(i)}$：该 segment 的累计状态转移；
- $\widetilde S^{(i)}$：该 segment 从零状态产生的局部内容；
- $S_{\text{in}}^{(i)}$：该 segment 接收到的历史 state。

各 rank 通过一次 all-gather 交换这些固定大小的 $M^{(i)}$ 和 $\widetilde S^{(i)}$，再用 prefix scan 恢复每个 segment 的输入 state。

关键点是：KDA 的 delta update 不是简单加法，不能像普通 linear attention 那样直接把各段 state 相加；KCP 显式保留“状态转移 + 局部写入”，所以可以精确重建跨 segment 的递归结果。

因此通信传递的是固定大小的 state fragments，而不是随 1M token 增长的完整 KV blocks。这让 KDA 的长序列训练可以扩展到多个设备。

#### 5. 1M window 与 1M capability 不是一回事

K3 做到的是三层意义上的“1M”：

| 层次 | 含义 |
| --- | --- |
| 架构支持 | 模型可以接收并处理最多 1M token |
| 训练适配 | 模型在 256K → 1M 阶段见过长程依赖 |
| 能力形成 | 合成数据迫使模型从整个 1M 范围检索信息 |

所以：

> **NoPE 让位置机制不因长度扩展而失效；KDA 让大多数序列处理成本可控；长上下文数据让模型学会使用远处信息；渐进式 curriculum 和 KCP 让训练这件事负担得起。**

这四个部分缺一不可。只改 RoPE，得到的只是“能输入更长”；只改 KDA，得到的只是“更便宜地处理长序列”；只有架构、数据、训练和系统共同配合，才可能得到真正可用的 1M context。

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

Agentic RL 的独立综述、主流模型路线与方法对比见 [[Agentic RL：总体范式、主流模型与方法对比]]。

### 本节所要解决的问题

Post-Training 的目标并非一般意义上的“对齐”或“指令跟随增强”，而是将一个已经具备大规模知识与基础推理能力的预训练模型，进一步塑造成能够在百万级上下文中长期执行任务、根据预算调整推理强度、跨领域完成工作的单一 Agent，并使其行为适应实际部署中的精度、延迟与成本约束。

因此，本节的核心问题可以表述为：

> **如何把预训练得到的能力转化为可执行、可验证、可调成本，并最终可部署的 Agent 策略。**

### 总体训练链

K3 的 Post-Training 采用如下三阶段范式：

```text
预训练基座
    ↓
SFT：建立可探索的 Agent 冷启动策略
    ↓
Domain-specific RL：按任务领域和推理预算训练专门策略
    ↓
MOPD：将专门策略统一到单一模型
    ↓
QAT + EAGLE-3：使训练结果适应部署约束并降低推理成本
```

其中，`4.1 Method` 负责说明策略如何被初始化、优化和合并；`4.2 RL Task Synthesis and Agentic Environments` 负责说明任务、环境与奖励信号如何被构造。二者共同构成 Agentic RL 的闭环。

实际训练过程并不是严格的一次性流水线，而是：

```text
构造任务与环境
    ↓
模型 rollout
    ↓
执行结果、验证器反馈与 reward
    ↓
策略更新
    ↓
暴露新的失败模式
    └────────→ 继续改进任务、环境与 verifier
```

报告采用“先讲方法、后讲任务与环境”的顺序，主要是为了先建立策略优化的抽象主线，再解释训练信号的来源。该顺序是论文叙事顺序，不等同于实际研发中的时间顺序。

### 4.1.1 SFT：建立 Agent 的冷启动策略

SFT 在 K3 中的主要作用不是直接获得最终能力，而是为后续 RL 提供一个高质量的初始策略。预训练模型虽然掌握了大量知识和语言生成能力，但并不天然具备以下行为：

- 正确理解并调用工具；
- 将复杂目标拆分为多个可执行步骤；
- 读取工具返回并据此调整后续行动；
- 在长程任务中保持状态并完成交付；
- 在失败后进行恢复，而不是直接终止。

因此，K3 的 SFT 数据重点覆盖复杂的 Agent trajectory。数据由此前 Kimi 系列中的领域专门模型生成，经过多阶段验证和人工参与标注，再通过 XTML-based chat template 统一序列化。XTML 的意义在于，将用户输入、模型输出、思考过程、工具调用、工具返回和多模态观察等不同交互类型编码为一致的训练结构，使复杂 Agent 交互能够转化为统一的 token-level learning problem。

SFT 的功能可以概括为：

```text
预训练模型的知识与推理能力
    +
基本的工具使用与执行轨迹
    ↓
具备可探索性的 Agent 初始策略
```

如果跳过这一步，直接对复杂长程任务进行 RL，模型往往会产生大量无效探索：工具格式错误、过早终止、行动与目标脱节，以及极其稀疏的有效 reward。SFT 因此承担的是“建立可探索行为分布”的作用，而不是简单模仿最终答案。

### 4.1.2 RL：按领域与推理预算进行专业化训练

SFT 提供冷启动基础，RL 则用于进一步获得高阶推理与执行能力。K3 没有将所有任务混合为一个单一 RL 专家，而是沿两个维度进行拆分。

#### 三个任务领域

| 领域 | 任务范围 | 主要能力 |
|---|---|---|
| `General Tasks` | 通用经验、视觉、推理、faithfulness、搜索、知识工作 | 广泛的认知与判断能力 |
| `General Agents` | 长时助手、深度研究、段落级写作 | 长程规划、工具使用与交付 |
| `Coding Agents` | SWE、coding experience、kernel、web development | 软件工程、GPU kernel 与应用开发 |

#### 三个推理努力等级

```text
low / high / max
```

两个维度交叉后得到：

```text
3 个领域 × 3 个推理努力等级 = 9 个专家模型
```

这种拆分同时处理两个冲突。

第一，**专业化能力与多任务负迁移之间的冲突**。知识、长程 Agent 和代码执行具有不同的任务结构与 reward 定义。如果完全使用一个 RL objective，某一领域的行为偏好可能干扰其他领域。例如，代码任务强调可运行性与性能，知识任务强调事实性，长程 Agent 任务则强调状态跟踪与最终交付。先训练领域专家，可以让不同能力在各自的 reward landscape 中充分优化。

第二，**结果质量与推理成本之间的冲突**。`max`、`high` 和 `low` 并不只是质量等级，而是不同的质量—成本工作点：

- `max`：允许较高推理与执行预算，追求更高成功率；
- `high`：在质量与成本之间取得平衡；
- `low`：减少思考 token、工具调用与延迟，适合成本敏感场景。

对于长程 Agent，额外的推理并不只意味着更多输出 token，还会增加工具调用次数、环境执行成本、上下文长度以及 KV cache 占用。因此，推理努力等级是一个需要显式训练的策略变量，而不是仅在推理阶段通过提示词控制的表面参数。

#### Partial Rollout：处理长程轨迹的尾部延迟

传统同步 RL 往往需要等待一个 batch 中的全部 rollout 完成后再更新策略。然而，长程 Agent 轨迹的长度差异很大，少数极慢轨迹会拖延整个训练 iteration。

K3 的 Partial Rollout 机制为每轮维护 $N\times K$ 条活跃轨迹。当完成比例达到 $\lambda$ 后，即先对已完成的轨迹进行策略优化；尚未完成的轨迹被暂停，并在下一轮开始时恢复：

```text
启动 N × K 条轨迹
    ↓
完成 λN K 条轨迹
    ↓
立即进行策略优化
    ↓
暂停的轨迹进入队列
    ↓
下一轮恢复执行
```

该机制降低了长程任务的尾部延迟，但也产生了新的稳定性问题：单条轨迹可能跨越多个 iteration，导致训练数据出现明显的 staleness，并进入更强的 off-policy regime。K3 通过 per-token regularization 限制策略更新幅度，使优化能够容忍这类陈旧数据。

这表明 K3 的 RL 设计首先受到长程 Agent 场景的系统约束，然后再对优化算法进行适配；它不是脱离执行系统而独立设计的纯算法方案。

#### Reasoning Effort RL：控制推理预算

K3 为每个问题 $x$ 估计一个初始 token budget $b_0(x)$。对于一条轨迹 $y$，如果其总 token 数超过预算阈值：

$$
T(y) > \tau b_0(x)
$$

则将该轨迹的任务 reward 覆盖为 $-1$。其中，$\tau$ 是预算倍率；通用任务中的 $T(y)$ 主要衡量 thinking tokens，Agent 任务中的 $T(y)$ 则包括思考过程与工具调用参数在内的累计输出 token。

训练采用阶段式 curriculum：

```text
较大的 τ → 训练 max-effort 专家
逐步降低 τ → 训练 high-effort 专家
进一步降低 τ → 训练 low-effort 专家
```

该机制的作用有二：

1. 防止模型通过无限延长思考或增加工具调用来获取不受约束的 reward；
2. 将推理成本显式纳入策略学习，使模型形成多个可部署的质量—延迟工作点。

#### Agentic GRM：为不可验证任务提供结构化奖励

对于开放式知识工作、专业写作和复杂 Agent 交付物，通常不存在简单的 exact-match 答案。K3 因此采用 `Agentic Generative Reward Model`，但要求评判模型遵循固定协议：

```text
1. 阅读结果、产品或文本输出
2. 生成评分 rubric
3. 根据 rubric 评价每个候选结果
4. 将 rubric 对应的分数写入 scorepad
```

该协议将“凭直觉打分”转化为较为结构化的评价过程。与此同时，K3 增加了类似 reasoning-effort control 的 verbosity budget：如果候选输出长度超过基准长度的指定倍数，则在二元比较中自动失败，以抑制模型通过无意义的冗长输出进行 reward hacking。

### 4.1.3 MOPD：将专业化策略统一到单一模型

如果只保留九个领域—努力等级专家，部署时就需要在多个模型之间进行路由；如果从一开始就训练一个统一模型，又容易产生领域之间的 reward interference。因此，K3 采用“训练时专业化、部署时统一化”的路径：

```text
3 domains × 3 effort levels
    ↓
9 个专门策略
    ↓
Multi-Teacher On-Policy Distillation
    ↓
一个统一模型
```

`MOPD` 的核心不是权重平均，也不是将九个模型简单压缩为一个模型，而是在 student 自己的 on-policy trajectory 上，由与当前领域 $d$ 和努力等级 $e$ 对应的 teacher 提供 token-level guidance。

对于输入 $x$、已有前缀 $y_{<t}$ 和当前 token $y_t$，报告给出的 per-token OPD reward 为：

$$
r_{\mathrm{opd}}^d(y_t\mid e,x,y_{<t})
=
\operatorname{clip}
\left(
\operatorname{sg}
\log
\frac{
\pi_{\mathrm{teacher}}^{(d,e)}(y_t\mid x,y_{<t})
}{
\pi_\theta(y_t\mid e,x,y_{<t})
},
-R_{\max},R_{\max}
\right)
$$

其中：

- teacher 对某个 token 的偏好高于 student 时，该 token 获得正向指导；
- `clip` 用于限制极端 advantage signal，稳定 RL 训练；
- `sg` 表示 teacher 不参与反向传播；
- on-policy 的含义是，teacher 指导的是 student 自己实际访问到的状态，而不是仅拟合 teacher 离线生成的轨迹。

这一点对长程 Agent 尤其重要。student 在实际执行过程中可能偏离 teacher 的轨迹，离线蒸馏只能覆盖 teacher 的状态分布，而 MOPD 能够在 student 自己的行为分布上提供修正。报告还指出，该 dense reward 可以直接嵌入既有 RL 框架，并与 Partial Rollout 等长轨迹优化机制结合。

### 4.1.4 Deployment-Aware Post-Training

K3 将部署约束纳入 Post-Training，而不是在训练结束后再单独处理。这一安排有两个层面。

#### MXFP4 Quantization-Aware Training

MoE expert 权重占据了模型参数内存的主要部分。K3 将其量化为 `MXFP4`，输入激活使用 `MXFP8`；attention projections、latent MoE projections、shared experts 和 MoE routers 等非 expert 组件则保留更高精度。

QAT 从 SFT 阶段开始，并贯穿整个 Post-Training，包括 SFT 和 RL。训练与 rollout 使用相同量化方案，从而避免：

```text
高精度训练与 rollout
    ↓
部署时突然切换为低精度
    ↓
模型概率分布、工具行为与策略稳定性发生变化
```

因此，`4.1.4` 虽然在章节顺序上位于 MOPD 之后，但它并不是严格意义上的最后一个训练阶段；QAT 实际上从 SFT 开始就参与训练。

#### Draft Model Fine-Tuning

K3 将预训练阶段的 MTP layer 微调为 `EAGLE-3` 风格的 draft model。target model 保持冻结，只更新 draft layer 与 feature-fusion projection。draft model 使用 target model 不同深度的特征，并通过 speculative decoding 提高生成速度。

K3 没有以传统 KL divergence 作为主要目标，而是直接优化无损 speculative sampling 的 acceptance rate：

$$
L_{\mathrm{LK}}
=
-\log \sum_{x\in V}\min\bigl(p(x),q(x)\bigr)
$$

其中 $p$ 是 target model 的分布，$q$ 是 draft model 的分布。这样做的原因是：对于容量受限的 draft model，最小化 KL 并不等价于最大化实际 token acceptance rate；而后者才直接决定 speculative decoding 的加速效果。

### 4.2 RL Task Synthesis and Agentic Environments

Agentic RL 的瓶颈不仅是策略优化算法，还包括：

```text
高质量任务
+ 可执行环境
+ 可靠 verifier
+ 可扩展 reward
```

因此，4.2 不是对任务案例的简单罗列，而是在回答“RL 的训练信号如何被持续、可靠地供给”这一问题。

#### Unified White-Box RL Environment

如果始终使用单一 Agent harness，模型可能过拟合于固定的工具 schema、system prompt、上下文管理机制、memory、skills 或 subagent 结构，最终学到的是“适应该框架的方法”，而不是更一般的任务解决能力。

K3 将 Agent harness 抽象为一组可配置、可组合的模块，并动态构造不同环境，使模型接触多种工具接口、提示词、上下文策略、技能和记忆机制。该设计的目标是提升跨 harness generalization。

#### Knowledge-Graph-Guided Task Synthesis

任务来源决定了 Post-Training 的质量与覆盖范围。纯随机生成或热门主题采样容易产生重复任务，也难以覆盖长尾专业知识。K3 因此构建自演化的层级知识图谱：

```text
粗粒度领域
    ↓
子领域
    ↓
细粒度概念
    ↓
原子知识点
```

系统从不同粒度的节点或相关节点组合中采样关键词，结合祖先节点的上下文信息检索真实材料，再由 synthesis agent 生成不同类型的训练任务。该过程同时控制：

- 领域覆盖；
- 知识粒度；
- 长尾概念比例；
- 真实材料 grounding；
- 任务类型分布。

#### Verifiable Problems in Agentic Environments

K3 的任务设计重点从“生成看似合理的回答”转向“通过一系列动作使环境达到目标状态”。代表性任务包括：

- 多步信息搜索与证据汇总；
- 投资银行、数据分析、法律实践等专业工作流；
- 使用 Python 工具进行裁剪、放大、变换和计算的视觉推理；
- 需要多轮工具调用与中间结果验证的复杂任务。

这些任务共同训练以下闭环：

```text
理解目标 → 制定计划 → 执行动作 → 观察结果 → 验证 → 调整
```

#### Kernel Optimization Tasks

Kernel 任务覆盖 CUDA、Triton、CuTe DSL、Gluon、ThunderKittens 和 TileLang 等编程方式，以及 BF16、FP8 和 FP4 等数值格式。奖励同时评价：

- 数值正确性；
- 相对于 expert implementation 的性能；
- 接近硬件 roofline 的程度；
- 是否存在 CUDA graph replay、input caching、降低精度等 reward-hacking 行为。

它训练的不是普通代码补全，而是：

```text
理解算子 → 编写 kernel → 运行验证 → 测量性能 → 迭代优化
```

#### Personal Assistant Tasks

K3 构造 Gmail、Notion、Slack 和 Canvas 等应用的可复现模拟环境，并在多个模拟日期中持续推进事件流。单次 rollout 可能包含数千次工具调用和数百万 token 的上下文。

该类任务主要训练：

- 持久状态跟踪；
- 跨应用协调；
- 长期计划；
- 事件驱动的行动；
- 在持续变化的环境中维护一致性。

这与单轮问答或短链工具调用有本质区别。

#### Autonomous Execution Tasks

`Autonomous Execution Tasks (AET)` 采用 verify-in-the-loop 的环境范式。每个任务只提供初始状态、目标、约束、工具动作空间、执行预算和独立 verifier，不提供参考轨迹或预定义程序。

模型必须自主完成：

```text
任务分解 → 工具选择 → 规划 → 执行 → 错误恢复 → 终止
```

奖励依据 verifier 对最终环境状态的评价，而不是模型自我报告的完成状态。为降低 reward hacking 风险，K3 将 agent 与 verifier 隔离，并结合公开 verifier 的诊断反馈、隐藏 verifier 的 held-out evaluation 以及有限提交预算下的惩罚机制。

#### Web Development Tasks

Web development 任务覆盖网站、交互式游戏、3D/WebGL 场景、数据可视化、SVG 和全栈应用。每个任务在容器化 sandbox 中运行，并使用多种 Agent scaffold，以提升跨 scaffold 泛化能力。

奖励由确定性检查与模型评价共同组成，包括：

- 项目是否成功构建；
- 应用是否正常运行；
- 功能行为是否正确；
- 结构与像素级相似度；
- 源代码检查；
- 对最终交互产物的观察与评价。

这类任务形成了典型的工程闭环：

```text
需求理解 → 代码实现 → 构建运行 → 观察结果 → 修复 → 交付
```

### 第4部分的写作逻辑

第4部分采用当前结构，主要有以下原因。

#### 1. 先说明策略，再说明训练信号

报告先给出：

```text
SFT → RL → MOPD
```

读者先获得 Post-Training 的抽象主线，再理解任务合成、环境配置和 verifier 如何为 RL 提供数据与 reward。若一开始直接介绍 knowledge graph、sandbox 和具体任务，容易陷入环境细节而无法把握训练目标。

#### 2. 突出 K3 与传统对齐流水线的区别

传统 Post-Training 常被概括为：

```text
SFT → preference optimization → safety alignment
```

K3 则将重点放在：

```text
SFT → long-horizon domain RL → verifiable environments → MOPD → deployment
```

这表明 K3 将 Post-Training 理解为 Agent policy engineering，而不仅是对话风格或偏好对齐。

#### 3. 处理“专业化训练与统一部署”的矛盾

多领域、多推理预算需要专业化策略；实际部署则希望使用一个统一模型。K3 因此先允许九个专家分别优化，再通过 MOPD 合并能力。MOPD 是整条逻辑链的闭合环节，缺少它，领域专家与最终产品之间就会存在明显断层。

#### 4. 将长程 Agent RL 作为算法—系统协同问题

当任务轨迹延伸到数百或数千次工具调用、甚至百万级上下文时，RL 已经不能仅靠策略梯度算法解决。Partial Rollout、KV cache 管理、可恢复 sandbox、动态并发控制和 verifier 隔离都成为训练闭环的一部分。第4节与第5节因此是相互衔接的：

```text
第4节：策略、任务、环境与 reward
第5节：承载这些策略与环境的训练、rollout 和服务系统
```

#### 5. 将部署目标前置到训练阶段

QAT 从 SFT 开始，RL rollout 与训练采用相同量化配置；EAGLE-3 则直接优化 token acceptance rate。报告由此强调，K3 的目标不是只在离线 benchmark 上得到高分，而是在真实推理成本下提供可用能力。

### 第4部分的“问题—机制”对应关系

| 要解决的问题 | 对应机制 |
|---|---|
| 预训练模型如何变成可执行 Agent | SFT cold-start policy |
| 如何获得高阶推理与执行能力 | Domain-specific RL |
| 如何控制推理成本 | Reasoning Effort Budget Control |
| 如何处理不等长长程轨迹 | Partial Rollout |
| 如何评价开放式交付物 | Agentic GRM |
| 如何抑制冗长输出与环境投机 | Verbosity budget、anti-hacking verifier |
| 如何将专家能力统一到一个模型 | MOPD |
| 如何避免过拟合单一 Agent harness | Unified White-Box RL Environment |
| 如何扩展任务覆盖与长尾知识 | Knowledge-Graph-Guided Task Synthesis |
| 如何训练真实的环境交互能力 | Verifiable Problems、AET、Personal Assistant Tasks |
| 如何使训练行为接近部署行为 | QAT |
| 如何降低长程推理延迟 | EAGLE-3 draft model |

### 需要避免的三种误读

1. **不要将 `4.1.4 Deployment-Aware Post-Training` 理解为训练结束后的最后一步。** QAT 从 SFT 阶段开始，并持续覆盖 RL；该小节只是集中说明部署约束。
2. **不要将 `4.2` 理解为方法完成后才开始构建环境。** 真实流程是任务、环境、rollout、verifier 和策略更新不断迭代；报告只是为了叙述清晰，将训练信号的来源单独放在后面说明。
3. **不要将 MOPD 理解为九个模型的简单压缩或权重合并。** 它是在 student 自己的 on-policy 状态分布上，由对应 teacher 提供 token-level dense guidance 的策略统一过程。

### 小结

K3 Post-Training 的主张可以概括为：

> **Agent 能力需要通过真实、长程、可验证的环境交互由 RL 写入策略；训练阶段可以按领域和推理预算进行专业化，部署阶段则通过 MOPD 统一为单一模型，并从训练早期开始纳入量化与推理成本约束。**

因此，第4部分的结构不是单纯的算法分类，而是一条完整的因果链：

```text
能力目标
    → 行为初始化
    → 专门能力放大
    → 推理成本控制
    → 多专家统一
    → 任务与 reward 供给
    → reward hacking 防护
    → 训练—部署一致性
```

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
