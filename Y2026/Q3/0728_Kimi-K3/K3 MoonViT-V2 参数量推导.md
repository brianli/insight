---
title: "K3 MoonViT-V2 参数量推导"
created: 2026-08-04
tags:
  - "model/kimi/k3"
  - "multimodal"
  - "vision"
  - "parameter-count"
  - "architecture"
source:
  - "[[K3 MoonViT-V2 数学与 PyTorch 实现]]"
  - "[[Kimi K3 论文阅读报告]]"
  - "[[k3_tech_report.pdf]]"
paper_path:
  kimi_k3: "/Users/brianli/workspace/insight/Y2026/Q3/0728_Kimi-K3/k3_tech_report.pdf"
  kimi_vl: "/Users/brianli/paper/aimodel/kimi/Team et al.-2025-Kimi-VL Technical Report.pdf"
---

# K3 MoonViT-V2 参数量推导

## 结论先行

K3 技术报告给出的：

> `Total Parameters of ViT = 401M`

不是只用“27 层 × 普通 ViT block”粗略相乘就能得到。要还原这个数字，至少需要知道：

- $27$ 层视觉 Transformer；
- 视觉残差宽度 $d=1024$；
- QKV 投影宽度 $d_q=1536$，不是 $1024$；
- 每个视觉 FFN 中间维度 $d_f=4096$；
- patch size $P=14$；
- 视觉输入通道 $C=3$；
- 可学习二维位置表初始化为 $64\times64$；
- 线性层无 bias；
- `pixel shuffle` 和时间平均池化本身不引入参数；
- multimodal projector 不计入 `Total Parameters of ViT`。

按这组配置进行近似重建：

$$
N_{\text{vision}}
\approx
401.214\text{M}
$$

四舍五入后就是报告里的：

$$
\boxed{401\text{M}}
$$

关键判断：

> **401M 主要来自 27 个视觉 Transformer block；额外的 4.2M 左右主要来自 $64\times64\times1024$ 的二维可学习位置表。**

## 1. 先分清三个参数量口径

MoonViT-V2 这条视觉路径至少有三种口径：

| 口径 | 是否包含 | 近似参数量 |
| --- | --- | ---: |
| Vision Transformer / `vision_tower` | patch embedding、位置表、27 个视觉 block、末端 norm | **401.2M** |
| Multimodal projector | `pixel shuffle` 后将 $4d$ 映射到 LLM hidden size 的两层 MLP | **约 46.1M** |
| 视觉前端总计 | `vision_tower + mm_projector` | **约 447.3M** |

K3 报告 Table 1 中的 `Total Parameters of ViT = 401M` 应理解为第一种口径，而不是把 projector 也算进去。

这点很重要：如果把 projector 也算进去，会得到约 $447$M，而不是报告里的 $401$M。

## 2. 用到的视觉配置

结合 K3 报告、现有架构笔记和本地已有的公开配置摘录，可采用如下参数：

| 符号 | 含义 | 值 |
| --- | --- | ---: |
| $L$ | Vision Transformer 层数 | $27$ |
| $d$ | residual stream / `vt_hidden_size` | $1024$ |
| $H$ | Attention heads | $12$ |
| $d_h$ | 每个 head 的 QKV 宽度 | $128$ |
| $d_q=H d_h$ | Q/K/V 总投影宽度 | $1536$ |
| $d_f$ | FFN intermediate size | $4096$ |
| $P$ | patch size | $14$ |
| $C$ | 输入通道数 | $3$ |
| $G_h,G_w$ | 可学习位置表初始网格 | $64,64$ |
| $d_{\text{LLM}}$ | K3 语言 Backbone 宽度 | $7168$ |

其中最容易错的是：

$$
d_q=12\times128=1536\neq d=1024
$$

这意味着视觉 Attention 的 QKV 投影是**非方阵**：

$$
W_{qkv}\in\mathbb{R}^{3d_q\times d}
$$

而不是标准 ViT 中常见的：

$$
W_{qkv}\in\mathbb{R}^{3d\times d}
$$

公开的 Kimi-VL 报告明确描述了 MoonViT 的原生分辨率、位置编码和 projector 路线；K3 报告则给出 MoonViT-V2 的 27 层和 401M 总量。`vt_hidden_size=1024`、`qkv_hidden_size=1536`、`intermediate_size=4096` 等细粒度视觉配置并未全部出现在 K3 论文表格中，本推导使用当前知识库已有的公开配置摘录进行数值还原。因此，下文的 $401.214$M 是**高度吻合的重建**，不是仅凭 K3 PDF 表格即可严格唯一推出的官方逐项核算。

## 3. 一个视觉 Transformer Block 有多少参数

以下按 K3 的 bias-free、标准两层 FFN、两个 RMSNorm 的 block 统计。具体实现若把空间/时间路径拆成更多独立 norm，差异只有几万参数，不影响 $401$M 的百万级结论。

### 3.1 QKV 投影

输入 residual width 为 $d=1024$，Q/K/V 各自输出 $d_q=1536$：

$$
W_{qkv}\in\mathbb{R}^{3d_q\times d}
$$

参数量：

$$
N_{qkv}
=
3d_qd
=
3\times1536\times1024
=
4,718,592
$$

### 3.2 Attention 输出投影

Q/K/V 的多头结果拼回 $d_q=1536$，再投影回 residual width $d=1024$：

$$
W_o\in\mathbb{R}^{d\times d_q}
$$

参数量：

$$
N_o
=
dd_q
=
1024\times1536
=
1,572,864
$$

因此 Attention 部分：

$$
N_{\text{attn}}
=
N_{qkv}+N_o
=
4d d_q
=
6,291,456
$$

这里的 $4dd_q$ 来自：

$$
3dd_q+dd_q
$$

即 QKV 三个投影加一个输出投影。

### 3.3 两层视觉 FFN

K3 的视觉塔使用 `GELU` 路线，而不是 gated FFN。因此按两层线性层统计：

$$
W_{\text{up}}\in\mathbb{R}^{d_f\times d}
$$

$$
W_{\text{down}}\in\mathbb{R}^{d\times d_f}
$$

参数量：

$$
\begin{aligned}
N_{\text{ffn}}
&=
d_fd+dd_f\\
&=
2dd_f\\
&=
2\times1024\times4096\\
&=
8,388,608
\end{aligned}
$$

不要误用 SwiGLU 的三投影公式：

$$
3dd_f
$$

如果误把视觉 FFN 当成 SwiGLU，每层会多算：

$$
dd_f
=
4,194,304
$$

27 层会多算约 $113.25$M，结果会明显偏离 $401$M。

### 3.4 RMSNorm

每个 RMSNorm 只有一个长度为 $d$ 的可学习 scale：

$$
N_{\text{rmsnorm}}=d
$$

按每个 block 两个 RMSNorm：

$$
N_{\text{norm/block}}=2d=2,048
$$

### 3.5 单层 block 合计

$$
\begin{aligned}
N_{\text{block}}
&=
N_{\text{attn}}
+
N_{\text{ffn}}
+
N_{\text{norm/block}}\\
&=
6,291,456
+
8,388,608
+
2,048\\
&=
14,682,112
\end{aligned}
$$

27 层：

$$
\begin{aligned}
N_{\text{blocks}}
&=
27\times14,682,112\\
&=
396,417,024
\end{aligned}
$$

这已经占到了约 $396.4$M，说明参数主体确实是 27 个视觉 block。

## 4. 视觉塔的其他参数

### 4.1 Patch Embedding

采用每帧二维 patch projection，且时间维度由后续 temporal pooling 处理：

$$
W_{\text{patch}}
\in
\mathbb{R}^{d\times(CP^2)}
$$

参数量：

$$
\begin{aligned}
N_{\text{patch}}
&=
CP^2d\\
&=
3\times14^2\times1024\\
&=
602,112
\end{aligned}
$$

K3 的视觉路径支持视频，但“视频能力”不等于 patch embedding 一定使用可学习的 temporal tubelet。若 temporal tubelet size 为 $t_p>1$，这一项应改成：

$$
N_{\text{patch}}(t_p)
=
C t_p P^2d
$$

当前 $401$M 的重建采用 $t_p=1$，与公开参数规模更吻合。

### 4.2 可学习二维位置表

Kimi-VL 报告说明 MoonViT 使用可插值的 learned fixed-size absolute positional embedding；K3 的公开 vision config 采用 $64\times64$ 初始化网格的路线。

位置表为：

$$
E_{\text{pos}}
\in
\mathbb{R}^{G_h\times G_w\times d}
$$

参数量：

$$
\begin{aligned}
N_{\text{pos}}
&=
G_hG_wd\\
&=
64\times64\times1024\\
&=
4,194,304
\end{aligned}
$$

这正是从 $397$M 补到 $401$M 的关键一项。

推理时，输入图像的 patch 网格可能不是 $64\times64$，因此位置表需要插值：

$$
E_{\text{pos}}^{H_p\times W_p}
=
\operatorname{Interpolate}
\left(
E_{\text{pos}}^{64\times64}
\right)
$$

插值改变的是计算结果，不改变参数数量。

### 4.3 末端 Norm

如果视觉塔末端还有一个 RMSNorm：

$$
N_{\text{final norm}}=d=1024
$$

### 4.4 `pixel shuffle` 与时间池化

这两个操作本身不引入可学习参数：

#### `pixel shuffle`

$$
[B,N,D_v]
\longrightarrow
[B,N/4,4D_v]
$$

只是 reshape/permute：

$$
N_{\text{pixel shuffle}}=0
$$

#### 时间平均池化

`sd2_tpool` 的 mean pooling 也是无参数操作：

$$
X_{t}
\longrightarrow
\frac{1}{F}
\sum_{t=1}^{F}X_t
$$

因此：

$$
N_{\text{temporal pool}}=0
$$

不能因为它改变了 Tensor Shape，就把它当成一个带权重的层。

## 5. 合计：为什么是 401M

把上面的主要部分相加：

$$
\begin{aligned}
N_{\text{vision}}
&=
N_{\text{blocks}}
+
N_{\text{patch}}
+
N_{\text{pos}}
+
N_{\text{final norm}}\\
&=
396,417,024
+
602,112
+
4,194,304
+
1,024\\
&=
401,214,464
\end{aligned}
$$

换算为百万参数：

$$
\frac{401,214,464}{10^6}
=
401.214464\text{M}
$$

报告写成整数：

$$
\boxed{401\text{M}}
$$

这不是巧合：$401.214$M 与公开权重拆分中约 $401.2$M 的 vision tower 参数量一致。

## 6. Multimodal Projector 为什么另算

`pixel shuffle` 后，视觉 token 的 channel width 为：

$$
4d=4096
$$

K3 语言 Backbone hidden size 为：

$$
d_{\text{LLM}}=7168
$$

两层 MLP projector 可以抽象为：

$$
4096
\xrightarrow{W_1}
d_p
\xrightarrow{\sigma}
7168
$$

如果 projector 中间维度取 $d_p=4096$，且忽略 bias：

$$
\begin{aligned}
N_{\text{proj}}
&=
(4d)d_p+d_pd_{\text{LLM}}\\
&=
4096\times4096
+
4096\times7168\\
&=
16,777,216+29,360,128\\
&=
46,137,344
\end{aligned}
$$

即：

$$
N_{\text{proj}}\approx46.1\text{M}
$$

因此视觉前端合计约为：

$$
\begin{aligned}
N_{\text{vision front-end}}
&\approx
401.214464\text{M}
+
46.137344\text{M}\\
&\approx
447.351808\text{M}
\end{aligned}
$$

所以：

```text
401M  = MoonViT-V2 vision tower
46M   = multimodal projector
447M  ≈ 视觉前端总参数
```

这就是为什么不能拿 `401M` 直接理解为“图像进入 LLM 前的全部参数”。

## 7. 一个可复用的 Python 参数计算器

```python
from dataclasses import dataclass


@dataclass
class MoonViTConfig:
    layers: int = 27
    hidden: int = 1024
    qkv_hidden: int = 1536
    intermediate: int = 4096
    heads: int = 12
    patch_size: int = 14
    in_channels: int = 3
    pos_grid_h: int = 64
    pos_grid_w: int = 64
    block_norms: int = 2
    has_final_norm: bool = True
    patch_temporal_size: int = 1
    use_bias: bool = False


def linear_params(
    in_dim: int,
    out_dim: int,
    bias: bool = False,
) -> int:
    return in_dim * out_dim + (out_dim if bias else 0)


def count_moonvit_v2(
    cfg: MoonViTConfig,
    include_projector: bool = False,
    projector_hidden: int = 4096,
    llm_hidden: int = 7168,
    projector_bias: bool = False,
):
    d = cfg.hidden
    q = cfg.qkv_hidden
    f = cfg.intermediate

    # Non-square QKV: d -> 3q; output: q -> d.
    qkv = linear_params(d, 3 * q, cfg.use_bias)
    out = linear_params(q, d, cfg.use_bias)

    # Standard GELU FFN: d -> f -> d.
    ffn_up = linear_params(d, f, cfg.use_bias)
    ffn_down = linear_params(f, d, cfg.use_bias)

    norms = cfg.block_norms * d
    per_block = qkv + out + ffn_up + ffn_down + norms
    blocks = cfg.layers * per_block

    patch = (
        cfg.in_channels
        * cfg.patch_temporal_size
        * cfg.patch_size
        * cfg.patch_size
        * d
    )
    pos = cfg.pos_grid_h * cfg.pos_grid_w * d
    final_norm = d if cfg.has_final_norm else 0

    vision_tower = blocks + patch + pos + final_norm

    projector = 0
    if include_projector:
        merged_dim = 4 * d
        projector = (
            linear_params(merged_dim, projector_hidden, projector_bias)
            + linear_params(projector_hidden, llm_hidden, projector_bias)
        )

    return {
        "qkv_per_block": qkv,
        "out_per_block": out,
        "ffn_up_per_block": ffn_up,
        "ffn_down_per_block": ffn_down,
        "norms_per_block": norms,
        "per_block": per_block,
        "all_blocks": blocks,
        "patch_embed": patch,
        "position_table": pos,
        "final_norm": final_norm,
        "vision_tower": vision_tower,
        "projector": projector,
        "vision_front_end": vision_tower + projector,
    }


cfg = MoonViTConfig()
report = count_moonvit_v2(
    cfg,
    include_projector=True,
)

for name, value in report.items():
    print(f"{name:24s}: {value:>12,d} "
          f"({value / 1e6:8.3f}M)")
```

按上述默认配置，核心输出约为：

```text
per_block              :   14,682,112 (  14.682M)
all_blocks             :  396,417,024 ( 396.417M)
patch_embed            :      602,112 (   0.602M)
position_table         :    4,194,304 (   4.194M)
final_norm             :        1,024 (   0.001M)
vision_tower           :  401,214,464 ( 401.214M)
projector              :   46,137,344 (  46.137M)
vision_front_end       :  447,351,808 ( 447.352M)
```

## 8. 三个最容易算错的地方

### 错误一：把 QKV 当成方阵

错误假设：

$$
d_q=d=1024
$$

正确配置：

$$
d_q=1536
$$

每个 block 因此多出：

$$
4d(d_q-d)
=
4\times1024\times512
=
2,097,152
$$

27 层多出：

$$
56,623,104
\approx56.6\text{M}
$$

这会导致严重低估。

### 错误二：把视觉 FFN 当成 SwiGLU

K3 视觉塔使用 `GELU` 路线，按两层 FFN 计算：

$$
2dd_f
$$

如果误用 SwiGLU：

$$
3dd_f
$$

会多算约 $113.2$M。

### 错误三：把 `pixel shuffle` 当成有参数的卷积

`pixel shuffle` 只是把：

$$
[B,N,D]
\rightarrow
[B,N/4,4D]
$$

它本身没有权重：

$$
N_{\text{pixel shuffle}}=0
$$

真正有参数的是后面的 MLP projector，而且它属于 multimodal projector，不属于报告中的 `Total Parameters of ViT`。

## 9. 对当前 PyTorch 参考实现的影响

原来的教学版 `SelfAttention` 如果写成：

```python
nn.Linear(dim, 3 * dim)
nn.Linear(dim, dim)
```

它对应的是：

$$
d_q=d=1024
$$

因此不能复现 K3 MoonViT-V2 的参数量。

K3-compatible 的 Attention 至少应写成：

```python
class K3VisionAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,
        qkv_hidden_size: int = 1536,
        bias: bool = False,
    ):
        super().__init__()
        self.qkv = nn.Linear(
            hidden_size,
            3 * qkv_hidden_size,
            bias=bias,
        )
        self.out = nn.Linear(
            qkv_hidden_size,
            hidden_size,
            bias=bias,
        )
```

对应的矩阵形状是：

$$
\begin{aligned}
W_{qkv}&\in\mathbb{R}^{4608\times1024}\\
W_o&\in\mathbb{R}^{1024\times1536}
\end{aligned}
$$

而不是标准方形 Attention 的：

$$
W_{qkv}\in\mathbb{R}^{3072\times1024},
\qquad
W_o\in\mathbb{R}^{1024\times1024}
$$

## 最终总结

K3 的 MoonViT-V2 参数量可以按以下主公式理解：

$$
\boxed{
N_{\text{ViT}}
\approx
L
\left(
4dd_q
+
2dd_f
+
2d
\right)
+
CP^2d
+
G_hG_wd
+
d
}
$$

代入：

$$
\boxed{
27
\left(
4\times1024\times1536
+
2\times1024\times4096
+
2\times1024
\right)
+
3\times14^2\times1024
+
64^2\times1024
+
1024
}
$$

得到：

$$
\boxed{
401,214,464
\approx
401\text{M}
}
$$

真正决定这个数字的不是图像分辨率，也不是视频帧数，而是：

1. 27 个视觉 block；
2. 非方形的 $1024\to1536$ QKV；
3. $4096$ 维标准 GELU FFN；
4. $64\times64\times1024$ 可学习二维位置表；
5. projector 是否计入统计口径。

关联笔记：

- [[K3 MoonViT-V2 数学与 PyTorch 实现]]
- [[K3 Native Vision 深入解析]]
- [[Kimi K3 论文阅读报告]]
