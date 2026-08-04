---
title: "K3 Fig.6 Gradient Norms 解读"
created: 2026-08-04
tags:
  - "model/kimi/k3"
  - "multimodal"
  - "training"
  - "optimization"
source:
  - "[[K3 Native Vision 深入解析]]"
  - "[[K3 MoonViT-V2 数学与 PyTorch 实现]]"
  - "[[k3_tech_report.pdf]]"
---

# K3 Fig.6 Gradient Norms 解读

## 1. Fig.6 画的是什么

Fig.6 比较两种视觉塔在联合多模态预训练过程中的 **vision-tower gradient norm**：

- 蓝线：`MoonViT-3D (SigLIP init.)`；
- 红线：`MoonViT-V2 (from scratch)`；
- 横轴：训练 step，约从 $7\times10^3$ 到 $3\times10^4$；
- 纵轴：视觉塔梯度范数；
- 左图：完整训练轨迹；
- 右图：训练 step $14\text{k}\sim16\text{k}$ 的局部放大。

报告的结论是：

> 相比 SigLIP 初始化的 MoonViT-3D，从零训练的 MoonViT-V2 具有更低的梯度范数和更少的尖峰，因此优化更稳定。

必须准确理解：图 6 主要证明的是**优化动态差异**，不是直接证明某个视觉塔的最终视觉能力更强。报告还声称 MoonViT-V2 在视觉评测上匹配 SigLIP 初始化基线，但那是另一个结论。

## 2. Gradient norm 的数学定义

设视觉塔参数为：

$$
\theta_v=
\left\{
\theta_1,\theta_2,\ldots,\theta_L
\right\}
$$

多模态训练目标为：

$$
\mathcal{L}(\theta_v,\theta_{\text{LLM}},\theta_{\text{proj}})
$$

视觉塔梯度是：

$$
g_v
=
\nabla_{\theta_v}\mathcal{L}
$$

如果把所有视觉塔参数的梯度展平并拼接：

$$
\operatorname{vec}(g_v)
=
\left[
\operatorname{vec}(\nabla_{\theta_1}\mathcal{L});
\ldots;
\operatorname{vec}(\nabla_{\theta_L}\mathcal{L})
\right]
$$

那么图中的 gradient norm 通常可以理解为其 L2 范数：

$$
\left\|g_v\right\|_2
=
\sqrt{
\sum_{\ell=1}^{L}
\left\|
\nabla_{\theta_\ell}\mathcal{L}
\right\|_F^2
}
$$

其中：

- $\nabla_{\theta_\ell}\mathcal{L}$：第 $\ell$ 个视觉模块参数的梯度；
- $\|\cdot\|_F$：矩阵的 Frobenius 范数；
- $\|g_v\|_2$：整个 vision tower 在当前训练 step 接收到的总更新信号强度。

如果参数使用更新规则：

$$
\theta_{t+1}
=
\theta_t-\eta_t g_t
$$

那么在没有 gradient clipping、预条件器和混合精度缩放的简单情况下，参数更新幅度满足：

$$
\left\|\Delta\theta_t\right\|_2
\approx
\eta_t\left\|g_t\right\|_2
$$

所以 gradient norm 不是“模型能力分数”，而是：

> **当前损失函数要求这组参数改变多大，以及当前反向传播信号有多强的指标。**

## 3. 它的物理含义是什么

可以把视觉塔想象成一个由很多可调旋钮组成的系统。每个训练 step，loss 都会通过反向传播告诉它：

```text
哪些旋钮要往上拧
哪些旋钮要往下拧
整体要改多大
```

gradient norm 测量的是所有旋钮建议变化量的总体强度：

- norm 小：当前参数已经能较好地服务当前目标，或当前误差信号较弱；
- norm 大：当前参数与当前训练目标之间存在较强不匹配，或该 batch 产生了强烈误差信号；
- norm 突然尖峰：某个 batch、某类输入或某条反向路径突然要求视觉塔大幅修正。

更准确地说，它是**损失函数对视觉塔参数的敏感度的总体量级**：

$$
\|g_v\|_2
\quad\text{大致反映}\quad
\left\|
\frac{\partial\mathcal{L}}{\partial\theta_v}
\right\|_2
$$

它不是物理世界中的力，但可以类比为“训练系统施加在视觉塔上的总推力”。

## 4. 为什么 higher gradient norms 不等于更强

这是最容易误读的地方。

梯度大只说明：

$$
\left\|
\nabla_{\theta_v}\mathcal{L}
\right\|
\text{大}
$$

不说明：

- 视觉表示更好；
- 学习速度一定更快；
- 最终 loss 一定更低；
- 模型最终能力一定更强。

大梯度可能来自：

1. 模型离当前目标较远；
2. 不同模块之间存在尺度不匹配；
3. 输入 batch 是异常样本；
4. 激活值或 loss 对某些方向过度敏感；
5. 预训练视觉表示正在被联合目标强行重塑；
6. 某条梯度路径发生放大。

在训练早期，较大梯度有时是正常的；真正值得警惕的是：

> **在相同训练阶段和相近训练目标下，梯度长期偏高且频繁出现异常尖峰。**

## 5. 为什么 frequent spikes 通常意味着训练不稳定

### 5.1 梯度尖峰意味着单步更新突然变大

考虑最简单的 SGD：

$$
\theta_{t+1}
=
\theta_t-\eta g_t
$$

如果某一步梯度从正常值 $g_t$ 突然变为 $g_t+\Delta g_t$，那么更新变化为：

$$
\Delta\theta_t
=
-\eta\Delta g_t
$$

梯度尖峰越高，单步参数跳跃越大。模型可能从当前稳定区域被推到另一个区域，造成：

- loss 突然上升；
- 激活统计发生跳变；
- 后续梯度进一步放大；
- 训练出现振荡甚至发散。

### 5.2 梯度尖峰会破坏学习率的统一假设

学习率 $\eta$ 通常是按整体训练过程设定的。它隐含假设是：

> 大多数 step 的梯度尺度处于相对可控的范围。

如果梯度偶尔尖峰，学习率会面临两难：

- 学习率足够大：正常 step 学得快，但尖峰 step 可能跳飞；
- 学习率足够小：尖峰安全，但正常 step 学得慢。

可以用一个粗略安全条件表示：

$$
\eta_t\|g_t\|
\ll
\text{局部稳定区域的尺度}
$$

频繁尖峰会让固定或缓慢变化的学习率难以同时满足这个条件。

### 5.3 梯度尖峰会造成跨模态接口震荡

K3 中视觉塔并不是单独训练，而是通过 projector 接到语言模型：

$$
\text{image/video}
\rightarrow
\text{MoonViT-V2}
\rightarrow
\text{Projector}
\rightarrow
\text{K3 Backbone}
\rightarrow
\mathcal{L}_{\text{NTP}}
$$

视觉塔梯度可写成链式法则：

$$
\nabla_{\theta_v}\mathcal{L}
=
\frac{\partial \mathcal{L}}{\partial Z_{\text{vis}}}
\frac{\partial Z_{\text{vis}}}{\partial V}
\frac{\partial V}{\partial\theta_v}
$$

其中：

- $V$：视觉塔输出；
- $Z_{\text{vis}}$：projector 映射后的视觉 embedding。

如果视觉塔输出尺度、方向或特征几何突然变化，projector 和语言 Backbone 看到的输入分布也会变化。于是：

```text
视觉塔突然大更新
    ↓
视觉 embedding 分布跳变
    ↓
LLM 需要重新适应视觉 token
    ↓
反向梯度再次变大
```

这可能形成反馈回路：

$$
\text{representation shift}
\rightarrow
\text{interface mismatch}
\rightarrow
\text{large gradient}
\rightarrow
\text{larger representation shift}
$$

因此，在联合多模态训练中，视觉塔梯度稳定尤其重要。

### 5.4 尖峰可能触发数值风险

大梯度往往与以下现象相关：

- 激活 outlier；
- loss scale 过大；
- mixed precision 下 overflow；
- gradient scaler 频繁回退；
- optimizer state 被异常更新；
- gradient clipping 频繁触发。

如果使用 FP16、BF16、FP8 或更激进的量化训练，偶发大值更容易影响数值范围。

不过要严谨区分：

> **梯度尖峰是训练不稳定的风险信号，不是“不稳定”的单独充分证明。**

还需要结合：

- loss 是否同步振荡；
- 梯度 clipping 是否频繁触发；
- activation 是否出现 outlier；
- optimizer scale 是否反复回退；
- 训练是否最终发散；
- 下游性能是否受损。

K3 的 Fig.6 是一个消融实验中的强信号，但不是完整稳定性证明。

## 6. 如何读 Fig.6 左图

左图展示约 $7\text{k}$ 到 $30\text{k}$ step 的完整训练轨迹。

应关注三件事：

### 6.1 蓝线的基线水平更高

`MoonViT-3D (SigLIP init.)` 的梯度范数整体高于红色的 `MoonViT-V2 (from scratch)`。

这意味着在相同训练阶段，SigLIP 初始化的视觉塔受到的联合训练修正总体更强：

$$
\mathbb{E}_t[\|g_v^{\text{SigLIP}}(t)\|]
>
\mathbb{E}_t[\|g_v^{\text{scratch}}(t)\|]
$$

在 K3 报告的解释中，这说明预训练视觉塔接入 LLM 后需要承受更大的优化调整。

### 6.2 蓝线尖峰更高、更频繁

蓝线存在若干明显高峰，红线也有波动，但尖峰总体较少、幅度较低。

这意味着 SigLIP 初始化版本的训练更新更不均匀：

$$
\operatorname{Var}_t
\left[
\|g_v(t)\|
\right]
\text{更大}
$$

或者说，训练信号的尾部更重：

$$
\Pr\left(
\|g_v\|>\tau
\right)
\text{更高}
$$

这比单纯看平均梯度更重要。平均值相近时，尾部尖峰仍可能决定训练是否稳定。

### 6.3 不能把纵轴直接当作“学习速度”

蓝线更高不意味着蓝色模型“学得更快”，红线更低也不意味着“没有学习”。

更合理的读法是：

- 蓝线：视觉塔在联合训练中被强烈、间歇性地重塑；
- 红线：视觉塔从零开始沿当前语言建模目标逐步形成表示，更新更平滑。

## 7. 如何读 Fig.6 右图

右图放大 $14\text{k}\sim16\text{k}$ 区间，用来显示完整图中不容易看清的局部动态。

右图的价值是揭示：

- 蓝线的高峰不是个别视觉噪点；
- 在短时间窗口内，蓝线仍然反复出现尖峰；
- 红线主要集中在较窄的低幅值区间；
- 两种初始化的优化动态差异是持续性的，而不是某个单独 step 造成的。

可以把右图理解为对左图结论的局部放大：

```text
左图：整体趋势——蓝线更高、更不平滑
右图：局部证据——尖峰在短窗口内反复出现
```

## 8. 为什么 SigLIP 初始化可能出现这种现象

报告给出的直接解释是：

> 将预训练视觉编码器接入 LLM 后，联合优化变得不稳定。

更深入地看，可能有三类原因。

### 8.1 表示目标错位

SigLIP 优化的是图像—文本全局匹配；K3 预训练优化的是多模态 next-token prediction。

$$
\mathcal{L}_{\text{SigLIP}}
\neq
\mathcal{L}_{\text{K3 NTP}}
$$

接入后，LLM 梯度要求视觉塔学习：

- 细粒度文字；
- 空间布局；
- 坐标；
- 代码与渲染关系；
- 视频过程；
- Agent 环境状态。

原有视觉表示需要被重新塑形，形成较大的梯度。

### 8.2 表示尺度与接口几何错位

视觉塔输出经过 projector 后进入语言 Backbone。若预训练视觉输出的尺度、均值、协方差或方向与 LLM 期望不同，则 projector 需要承担较大的校准任务。

简化表示为：

$$
Z_{\text{vis}}=P(V)
$$

若 $V$ 的统计分布与当前 LLM hidden state 分布差异较大，则：

$$
\left\|
\frac{\partial\mathcal{L}}{\partial V}
\right\|
$$

可能在某些 batch 被放大。

### 8.3 预训练参数并不等于联合训练的好初始化

初始化“更有知识”与初始化“更适合当前联合目标”是两件事。

SigLIP 权重可能已经形成较强的视觉特征结构，但这些结构也可能限制了视觉塔向 K3 目标调整的方向。对于从零训练的 MoonViT-V2：

```text
没有旧视觉目标留下的强约束
    ↓
直接接受 NTP 梯度
    ↓
逐步形成与语言 Backbone 兼容的视觉表示
```

这不是说预训练一定有害，而是说在 K3 规模下，**旧目标带来的表示惯性可能比初始化收益更昂贵**。

## 9. 为什么从零训练的梯度更低、更平滑

可能的机制是：

1. 视觉塔没有必须保留的 SigLIP 表示结构；
2. 视觉塔从第一步就围绕 K3 的 NTP 目标塑形；
3. 视觉输出统计与 projector、LLM 共同演化；
4. 模态接口不是后期突然接入，而是共同形成；
5. 梯度不需要频繁纠正“旧视觉目标”和“新语言目标”的冲突。

因此：

$$
\text{from-scratch}
\rightarrow
\text{target-aligned representation}
\rightarrow
\text{smoother gradient flow}
$$

但这是对 Fig.6 的机制解释，不是从单张图中直接识别出的因果定律。

## 10. 从优化器角度看 gradient norm

K3 使用了 Muon 及其 per-head 变体等优化设计，真实参数更新不是简单 SGD：

$$
\Delta\theta_t
\neq
-\eta g_t
$$

而是某种带预条件、动量、正交化或裁剪的变换：

$$
\Delta\theta_t
=
\mathcal{U}(g_t,m_t,\theta_t,\eta_t)
$$

因此 gradient norm 与最终实际更新量不完全等价。

但它仍然是重要诊断信号，因为：

- gradient norm 大意味着原始反向信号大；
- optimizer 需要更强地预处理或裁剪；
- 同样的 optimizer 超参数下，参数更新风险更高；
- 尖峰说明训练信号的分布尾部更重。

如果使用 gradient clipping：

$$
g_t^{\text{clip}}
=
g_t\cdot
\min\left(
1,\frac{\tau}{\|g_t\|}
\right)
$$

当 $\|g_t\|>\tau$ 时，实际更新被截断，但这带来两个问题：

1. 梯度方向还在，但幅值信息被丢弃；
2. 频繁 clipping 说明模型不断产生超出优化器设计范围的更新。

所以 clipping 可以防止训练立即炸掉，但不能把频繁尖峰变成健康训练。

## 11. 一个最小 PyTorch 监控示例

### 11.1 计算 vision tower 的全局梯度范数

```python
from math import sqrt
import torch
from torch import nn


def grad_norm(module: nn.Module, norm_type: float = 2.0) -> float:
    """
    计算 module 当前已经累积的全局梯度范数。
    必须在 loss.backward() 之后、optimizer.zero_grad() 之前调用。
    """
    grads = [
        p.grad.detach()
        for p in module.parameters()
        if p.grad is not None
    ]

    if not grads:
        return 0.0

    if norm_type == float("inf"):
        return max(g.abs().max().item() for g in grads)

    total = torch.zeros((), device=grads[0].device)
    for g in grads:
        total = total + g.float().pow(norm_type).sum()

    return total.pow(1.0 / norm_type).item()
```

它近似计算：

$$
\left\|g_v\right\|_2
=
\left(
\sum_i\|g_i\|_2^2
\right)^{1/2}
$$

### 11.2 训练循环中记录

```python
history = []

for step, batch in enumerate(loader):
    optimizer.zero_grad(set_to_none=True)

    out = model(
        input_ids=batch["input_ids"],
        labels=batch["labels"],
        visual_items=batch["visual_items"],
    )
    loss = out["loss"]
    loss.backward()

    vision_grad_norm = grad_norm(model.vision_tower)
    total_grad_norm = grad_norm(model)

    # 先记录原始梯度，再决定是否 clipping。
    history.append(
        {
            "step": step,
            "loss": loss.detach().item(),
            "vision_grad_norm": vision_grad_norm,
            "total_grad_norm": total_grad_norm,
        }
    )

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )
    optimizer.step()
```

诊断时最好同时记录：

- vision tower gradient norm；
- projector gradient norm；
- LLM backbone gradient norm；
- loss；
- activation RMS；
- clipping rate；
- mixed-precision scaler 状态。

### 11.3 计算尖峰率

不能只看某个最大值，可以定义一个稳健阈值：

```python
import numpy as np


def spike_rate(values, multiplier: float = 5.0) -> float:
    """
    用 median + multiplier * MAD 定义尖峰。
    比均值+标准差更不容易被尖峰本身污染。
    """
    x = np.asarray(values, dtype=np.float64)
    median = np.median(x)
    mad = np.median(np.abs(x - median)) + 1e-12
    threshold = median + multiplier * 1.4826 * mad
    return float(np.mean(x > threshold)), float(threshold)
```

可以比较：

$$
\operatorname{SpikeRate}
=
\frac{1}{T}
\sum_{t=1}^{T}
\mathbf{1}
\left[
\|g_v(t)\|>\tau
\right]
$$

但注意：阈值 $\tau$ 应在相同模型、相同 batch、相同训练阶段和相同混合精度设置下比较。

## 12. 应该怎样严谨解读 Fig.6

正确表述：

> Fig.6 显示，在 K3 的预训练消融中，SigLIP 初始化的 MoonViT-3D 具有更高且更尖锐的 vision-tower gradient norms，而从零训练的 MoonViT-V2 梯度更低、更平滑。这说明前者在与 LLM 联合优化时承受更强、更不均匀的参数修正，存在更高的优化不稳定风险；它支持 K3 选择从零联合训练，但不能单凭该图证明从零训练在所有任务、规模和数据条件下都优于视觉预训练初始化。

错误表述：

- “梯度越小，模型越聪明”；
- “梯度越大，模型一定要爆炸”；
- “蓝线更高，所以 SigLIP 一定没学到东西”；
- “红线平滑，所以模型一定收敛到更优解”；
- “Fig.6 单独证明了 SigLIP 初始化有害”。

## 最终总结

$$
\text{gradient norm}
\approx
\text{当前训练目标对视觉塔施加的总更新信号强度}
$$

$$
\text{higher baseline}
\Rightarrow
\text{更强的持续重塑压力}
$$

$$
\text{frequent spikes}
\Rightarrow
\text{单步更新风险、尺度错配和数值问题的概率更高}
$$

K3 Fig.6 的真正信息是：

> **SigLIP 初始化的视觉塔虽然带着预训练视觉能力进入多模态训练，但它与 K3 的语言建模目标和共享 Backbone 接口之间存在更大的动态错配；MoonViT-V2 从零开始沿统一 NTP 目标塑形，因此梯度信号更低、更平滑。**

关联笔记：

- [[K3 Native Vision 深入解析]]
- [[K3 MoonViT-V2 数学与 PyTorch 实现]]
- [[Kimi K3 论文阅读报告]]
