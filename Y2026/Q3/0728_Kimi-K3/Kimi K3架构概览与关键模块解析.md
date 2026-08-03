---
title: "Kimi K3架构概览与关键模块解析"
source: "https://mp.weixin.qq.com/s/tZuBzGUUdQanVobi35OMVw"
author:
  - "[[kaiyuan]]"
published:
created: 2026-07-28
description: "Kimi K3是Moonshot AI发布的旗舰开源模型：总参数2.8T、激活104B，原生多模态（MoonViT-V2），上下文1M，面向长程编程、知识工作与推理等场景。相较Kimi K2，官方称在架构与数据配方共同作用下，整体scaling效率约提升2.5倍。"
tags:
  - "clippings"
---
kaiyuan InfraTech *2026年7月28日 17:30* 点击 **蓝字** ，关注我们

Kimi K3 <sup>[1]</sup> 是Moonshot AI发布的旗舰开源模型：总参数2.8T、激活104B，原生多模态（MoonViT-V2），上下文1M，面向长程编程、知识工作与推理等场景。相较Kimi K2，官方称在架构与数据配方共同作用下，整体scaling效率约 **提升2.5倍** 。

**1 整体架构**

![Image](https://mmbiz.qpic.cn/mmbiz_jpg/uIP3tuXZx8CWfs1JndSAib4P9KqUmOwR9NIb8cPA6ic8QEQOnRqOhEibAmLUaDtwUv9XLLJL2vWf8w8oAnnHubzq4eQt68uQJJYohZfyRKbS4U/640?wx_fmt=jpeg&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=1)

高清图地址：https://github.com/CalvinXKY/InfraTech/tree/main/models/kimi\_k\_3 <sup>[2]</sup>

从架构上看，K3可以粗略理解为在三个维度上同时改结构：

- **序列维（长度）** ：用Kimi Delta Attention（KDA）配合Gated MLA做混合注意力（层构成：69 KDA + 24 Gated MLA，约3:1），支撑更长上下文下的效率与表达。
- **深度维（层间）** ：用Attention Residuals（AttnRes）替代均匀累加的标准残差，让深层能按需回看更早层的表示。
- **宽度维（容量）** ：用Stable LatentMoE把稀疏度推高到896个专家中激活16个（共享专家2），并配合Quantile Balancing等机制稳住路由。

关键规格：

| 项目 | 公开信息 |
| --- | --- |
| 总参数量 / 激活参数量 | 2.8T / 104B |
| 层数 / 稠密层 | 93 / 1 |
| 注意力层构成 | 69 KDA + 24 Gated MLA |
| 注意力隐藏维度 / 头数 | 7168 / 96 |
| Latent MoE维度 | 3584 |
| 专家FFN隐层 / 专家数 / 每token激活 / 共享专家 | 3072 / 896 / 16 / 2 |
| 词表 / 上下文 | 160K / 1M |
| 注意力 / 跨层连接 / MoE框架 | KDA & Gated MLA / AttnRes / Stable LatentMoE |
| 激活函数 | SiTU-GLU |
| 视觉编码器 | MoonViT-V2（401M） |
| 量化 | MXFP4 weights + MXFP8 activations（自SFT起QAT） |

更完整的模型卡片与参数表见仓库说明：Kimi K3 README\[https://github.com/CalvinXKY/InfraTech/tree/main/models/kimi\_k\_3\]。

**2 关键模块**

**2.1**

**Attention Residuals（AttnRes）**

标准Transformer残差是把各层输出按固定权重（通常为1）一路加下去。深度变大后，有两个常见麻烦：一是早期层的贡献被后面层冲淡（PreNorm dilution），二是隐状态幅度容易随深度失控。

AttnRes的核心想法很直接：把固定相加换成对历史层表示做一次随输入变化的注意力聚合，让当前层按需挑选更有用的早期表示，而不是均匀叠加所有历史信息。

 $\mathbf{h}_l = \sum_{i=0}^{l-1} \alpha_{i \to l} \cdot \mathbf{v}_i$ 

其中权重𝛼 <sub>𝑖→𝑙</sub> 由每层一个可学习的 𝑤 <sub>𝑙</sub> 与历史表示交互后经softmax得到。也就是说，残差连接本身变成了沿深度方向的attention。

工程上Full AttnRes要对前面所有层做聚合，显存大约是𝑂(𝐿𝑑)量级，大模型上偏贵。因此更常用的是Block AttnRes：

- **step1** ：把网络按深度切成若干block，block内部仍用普通残差累加。
- **step2** ：只在block边界保存一份block级表示，跨block时对这些block表示（再加当前block内的partial sum）做注意力。
- **step3** ：block数量大约到8量级时，kimi称可收回Full AttnRes的大部分收益，同时把额外开销压到可接受范围。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uIP3tuXZx8DNTdJyt0TGRdFBqE2TQUEhSvY1umFMpQZkv3C55uTv2QfaYvnDdrdibUYicnhFobxGfppQ3ASCMMUzMRG0UYz5g3OjuBdrricCkI/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=2)

K3共93层，架构图也画了Block n−1 / n−2 / n−3一类跨层回看：信息不仅沿序列流动，也沿深度被选择性检索。AttnRes在Kimi Linear等实验里对多步推理、代码类任务帮助更明显，是K3深度方向上的骨架之一。

通俗解释可参考：

[AttnRes快速看：Kimi优化残差的新方案](https://mp.weixin.qq.com/s?__biz=MzYyMjA5NzMwOQ==&mid=2247489766&idx=1&sn=0bab7ddaec402bdf4a8a4ba97574a900&scene=21#wechat_redirect)

**2.2**

**KDA（Kimi Delta Attention）**

KDA来自Kimi Linear这条线，是对Gated DeltaNet（GDN）的细化。线性注意力把历史压缩进有限大小的状态𝑆，靠递推更新，避免标准attention随序列长度二次膨胀的KV压力；代价是状态容量有限，gate/衰减怎么设计就变得很关键。

GDN里常见的衰减系数𝛼往往是标量：对整个状态𝑆 <sub>𝑡−1</sub> 做同一种衰减。Kimi这边的直觉是：状态里有𝑑 <sub>𝑘</sub> ×𝑑 <sub>𝑣</sub> 这么多分量，凭什么共用一个衰减？于是把衰减从“一个数”改成“𝑑 <sub>𝑘</sub> 个不同的𝛼”，再对角化后乘进原递推，这就是KDA相对GDN的关键改动。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uIP3tuXZx8B4gSVYE15JBjp6gXjx57lNjZJ5TibrhFkWloLEDpHleialNeg9MloFJbTF5yTUHTZxDQZME30RJkQYKhfRmtoZSdwOK388hd74w/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=3)

对GDN的变化可以概括成两步：

- **step1** ：上采样/投影，让输出变成多个𝛼值，而不是单个标量。
- **step2** ：用对角矩阵承接这组𝛼，使状态不同维度可以有不同的衰减强度。

K3注意力层为69 KDA + 24 Gated MLA（约3:1）：多数层走线性注意力省KV与算力，少数层保留全局Gated MLA兜底长程依赖。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uIP3tuXZx8CEbM249LcicQ31lZEUekWGUYcEp0MCdnvy513KYVH3Lh8FE8micFzqYQNdDLSibwTzjJa890ibSuePchg2DRFa8iag0vjL3pjyTrlc/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=4)

推理侧KDA递推状态与传统prefix cache不完全同构，目前已向vLLM等推理框架提供对应支持。工程上可关注FlashKDA\[https://github.com/MoonshotAI/FlashKDA\]与FLA中的KDA算子。

通俗推导可参考：

[Kimi最新的KimiLinear技术亮点是什么？一起来看下](https://mp.weixin.qq.com/s?__biz=MzYyMjA5NzMwOQ==&mid=2247485745&idx=1&sn=3763b72b5bfb05f8dcec02f2c38ae8b0&scene=21#wechat_redirect)

**2.3**

**Gated MLA**

混合注意力里的“全局”层走的是Gated MLA（KimiMLAAttention，且mla\_use\_output\_gate=true），末层再多挂一层MLA。

相对上一代（K2/K2.5一类MLA），压缩秩与head切分基本不动，主要变化有两点：

**头数** ：64改为96，单头dim不变，Q/K/V/O宽度整体×1.5。

**Output gate** ：attention输出在o\_proj前乘sigmoid(g\_proj(h))，即所谓Gated。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uIP3tuXZx8AFzLCkicJx7rcRuOGxDic7FXuPMYGbweYcBkrJXmT7eIgDDNRLabqEGcg4wUdic2baLsUpJ1GbaLOLFfrpnvk1N0rqicyc81znsUQ/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=5)

关键尺寸对照：

| 项 | 上一代MLA | K3 Gated MLA |
| --- | --- | --- |
| q\_lora\_rank | 1536 | 1536 |
| kv\_lora\_rank | 512 | 512 |
| qk\_nope / qk\_rope / v\_head | 128 / 64 / 128 | 128 / 64 / 128 |
| qk\_head\_dim（nope+rope） | 192 | 192 |
| num\_attention\_heads | 64 | 96 |
| output gate | 无 | 有（g\_proj） |

第一层（layer\_idx=0）是KDA + Dense FFN，并不是MLA + Dense；Gated MLA只出现在上述全局层，且这些层的FFN已是MoE。

**2.4**

**Stable LatentMoE**

K2/K2.5已是较大稀疏MoE；K3进一步把专家池扩到896、每 token 激活16个、共享专家2个。这里的“宽度扩展”不是把主干 hidden size 直接变大，而是扩大**条件专家容量**：专家池更大、每个 token 的专业化组合更多，同时保持每个 token 只激活少量专家。

K3 主 hidden dimension 为 $d=7168$，LatentMoE dimension 为 $\ell=3584=d/2$。routed expert 不直接接收完整 $d$ 维表示，而是先经过共享的降维投影，再在低维 latent space 中完成专家计算，最后映射回 $d$ 维：

$$
x\in\mathbb{R}^{d}
\xrightarrow{W_{\downarrow}}
z\in\mathbb{R}^{\ell}
\xrightarrow{\text{Top-16 routed experts}}
u\in\mathbb{R}^{\ell}
\xrightarrow{\operatorname{RMSNorm},\,W_{\uparrow}}
y_{\mathrm{routed}}\in\mathbb{R}^{d}
$$

完整结构是：

```text
x ∈ R^7168
  ├─ 2 个 shared experts：直接在 7168 维处理公共变换
  └─ W↓：7168 → 3584
       └─ router 选择 16/896 个 routed experts
            └─ 每个专家在 3584 维 latent space 中计算
                 └─ 加权聚合 + RMSNorm
                      └─ W↑：3584 → 7168
                           └─ 与 shared experts 输出相加
```

这使得 routed path 的表示通信宽度从约 $kd$ 降到 $k\ell$，专家内部的主计算也从完整 hidden width 转移到 latent width。由于降维/升维投影在专家之间共享，专家数量增加时不需要复制这部分成本。

因此更准确的表述是：

> **不是每个专家变宽，而是专家池变宽、专家组合空间变大；单个 routed expert 在更窄的 latent space 中工作。**

### Stable LatentMoE 与 DeepSeekMoE

K3 沿用了 DeepSeekMoE 的 **shared experts + routed experts** 组织方式：shared experts 承担所有 token 都需要的公共变换，routed experts 提供 token 条件化的专业变换。

关键差异在 routed expert 的工作空间：

| 维度 | DeepSeekMoE | K3 Stable LatentMoE |
| --- | --- | --- |
| Routed expert 输入 | 完整 $d$ 维表示 | 先压缩到 $\ell$ 维 |
| 专家内部计算 | 完整 hidden width | 低维 latent width |
| 扩容方式 | 细粒度专家分割与更多组合 | 大专家池 + 低维 routed path |
| Shared experts | 隔离公共知识、减少 routed 专家重复 | 同样隔离公共知识，并保留 full-width 公共路径 |
| 主要目标 | 专家专业化与组合性 | 专业化、通信/计算效率与极端稀疏稳定性 |

一句话：

> **DeepSeekMoE 把专家切得更细；Stable LatentMoE 进一步把 routed expert 压窄，让专家池可以继续扩大。**

配套手段主要包括：

- **Normalized LatentMoE** ：专家加权聚合后先做 RMSNorm，再进行 $W_{\uparrow}$ 升维，减少不同专家组合和 router 权重造成的尺度漂移，避免 routed branch 与 shared branch 相加时幅值失衡。
- **SiTU-GLU（Sigmoid Tanh Unit + GLU）** ：相对 SwiGLU，gate 和 up 分支都用 $\beta\tanh(x/\beta)$ 做 soft cap。K3 取 $\beta_1=4,\beta_2=25$，保留原点附近近似 SwiGLU 的行为，同时让大值区域有界，控制 activation outlier、低精度溢出和梯度异常。
- **Quantile Balancing** ：仍采用 auxiliary-loss-free routing，但根据 router score 的目标分位数直接设置 expert bias，而不是依赖固定步长的冷热反馈更新。目标负载为 $q=mk/n$，其中 $m$ 是 batch token 数、$k$ 是每 token 激活专家数、$n$ 是专家总数；大规模训练中用 histogram 估计全局分位数。
- **Per-Head Muon** ：把Muon优化扩展到按attention head独立适配，利于大规模下更细粒度的学习动态。
- **Fully balanced expert-parallel** ：训练侧强调静态shape、关键路径上无host sync，减轻大EP规模下因专家不均衡带来的吞吐抖动。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uIP3tuXZx8DOGqcQnn3TpiceJl7QSgf8ibibkyFJBbdzPiauTHgxR7IrxfaZe9gERo0cPvPx9PBbt9VictjCfXLhxrPYxDdQvJiar5OMm59wYNV0I/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=6)

Routed 专家实际路径是：先把 7168 投影到 3584，再在专家内做 3584→3072→3584 的 SiTU-GLU，最后映回 7168，并与 shared experts 相加。K3 从 SFT 阶段起做量化感知训练（权重 MXFP4、激活 MXFP8）。

这里的 `latent` 不要与 DeepSeek MLA 的 latent 混淆：

- MLA latent 压缩的是 token 级 KV 表示，目标是降低 KV cache；
- LatentMoE latent 压缩的是 routed expert 的 channel 表示，目标是扩大专家容量并降低 MoE 通信/计算成本。

Stable 的含义是三层稳定化：

1. **数值稳定**：聚合后 RMSNorm 控制 routed representation 尺度；
2. **激活稳定**：SiTU-GLU 为 gate/up 分支提供平滑上限；
3. **路由与系统稳定**：Quantile Balancing 均衡专家负载，MoonEP 用静态 shape 和 bounded redundant experts 兑现硬件利用率。

**2.5**

**视觉模块**

K3视觉编码器为MoonViT-V2（约401M），相对K2.5/K2.6的MoonViT（约400M）为同骨架升级：仍是NaViT式3D patch + 2D RoPE + sd2\_tpool merge。参数量几乎不变，但宽度、注意力、Norm、Projector与预处理上限有实质变化。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uIP3tuXZx8CIO02VrXevCIw6w0lz09mTsvps4f3892u8KDT5soGEksaAnjjP53tmuahO0GQPTKgia5BPNLeOMCNicAKeH9TiabekbicUXAgalbo/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=7)

视觉与语言融合的常见步骤如下：

- **step1 预处理** ：经ImageProcessor完成抽帧（视频）、尺寸调整、padding，再切分为Patch。
- **step2 Patch嵌入** ：对patch做卷积式嵌入，并叠加位置编码（含时间与空间）。
- **step3 编码** ：经多层Encoder堆叠，提取高层视觉特征。
- **step4 Patch池化与merge** ：通过merge\_kernel等在空间（及配置下的时间）维度聚合，压缩视觉token数量。
- **step5 MM Projector** ：将视觉hidden维度映射到与文本一致的text\_hidden\_size。
- **step6 序列拼接** ：按占位符把视觉特征写入文本embedding序列，得到inputs\_embeds与attention\_mask。

与上一代规格对照：

| 项目 | MoonViT（K2.6） | MoonViT-V2（K3） |
| --- | --- | --- |
| 参数量 / 层数 / patch | 400M / 27 / 14 | 401M / 27 / 14 |
| vt\_hidden\_size | 1152 | 1024 |
| vt\_intermediate\_size | 4304 | 4096 |
| 注意力头 / 通道 | 16头，QKV=1152（每头72） | 12头，qkv\_hidden\_size=1536（每头128） |
| Norm / 激活 | LayerNorm / gelu | RMSNorm / gelu\_pytorch\_tanh |
| MM Projector | patchmerger（pre-LN） | patchmergerv2（无pre-norm，post RMSNorm） |
| 图像patch上限 | in\_patch\_limit=16384 | 65536 |

计算过程可参考：

[VLM视觉-语言融合流程解析（Kimi K2.5/VL）](https://mp.weixin.qq.com/s?__biz=MzYyMjA5NzMwOQ==&mid=2247489786&idx=1&sn=1be775571fa50dc89b12c5a87b766c99&scene=21#wechat_redirect)

**3 相关资料**

- 技术报告（PDF）\[https://github.com/MoonshotAI/Kimi-K3/blob/main/k3\_tech\_report.pdf\]
- 官方仓库\[https://github.com/MoonshotAI/Kimi-K3\]
- 模型权重（Hugging Face）\[https://huggingface.co/moonshotai/Kimi-K3\]
- 模型权重（ModelScope）\[https://www.modelscope.cn/models/moonshotai/Kimi-K3\]
- 官方博客\[https://www.kimi.com/zh-cn/blog/kimi-k3\]

---

参考:

- \[1\]https://github.com/MoonshotAI/Kimi-K3/blob/main/k3\_tech\_report.pdf
- \[2\]https://github.com/CalvinXKY/InfraTech/tree/main/models/kimi\_k\_3

想深耕AI Infra领域？欢迎访问InfraTech库！内容涵盖大模型基础、PyTorch/vLLM/SGLang框架入门、性能加速等核心方向，配套50+知识干货及适合初学者的notebook练习： **https://github.com/CalvinXKY/InfraTech**

扫码关注我们，了解更多AI Infra基础知识。

![Image](https://mmbiz.qpic.cn/mmbiz_jpg/uIP3tuXZx8CExibG7alrmsJyZGQicZI71qribynwtt8vrAGDP8DeTxRh6lo2hia5qfSKrIaMicQiaaAdDoicT4iajD0YElIwm392bW6vYP3BEYicia7oQ/640?wx_fmt=jpeg&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=8)

大模型基础知识 · 目录
