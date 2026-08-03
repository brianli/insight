https://mp.weixin.qq.com/s/sT0Zcgir0hNQMZzVWB8pFA

#model/kimi/k3 

最近开源的 Kimi K3 的出色表现让不少人惊叹，大家想必也很好奇它到底做了什么才能有这样的效果。最近我翻阅了 Kimi K3 的技术报告，其中架构上的创新点主要是：Attention Residuals、KDA（Kimi Delta Attention）、LatentMoE，以及 MXFP4/MXFP8 量化训练这四个。

这篇文章，我们就来好好聊聊其中的 **Attention Residuals**。

## 概述

其实 `Attention Residuals` 在今年已经被关注过一次了，就是年初那条引发热议的新闻——《Kimi新架构让马斯克叹服！17岁高中生作者一战成名_腾讯新闻》——所提出的方法正是 Attention Residuals。

`Attention Residuals`（AttnRes）是对**标准残差连接（Residual Connections）** 的一种替代方案。它与 HC（Hyper-Connections）的思路不同：HC 想去调和 Pre-Norm 与 Post-Norm 之间的矛盾，而 AttnRes 干脆另起炉灶——直接把残差连接里"固定单位权重的均匀累加"换成可学习的、输入依赖的**深度注意力（Depth-wise Attention）**，在层间直接做 Attention 来替代原有的 Residual。该方法在几乎不增加计算量和参数量的前提下，能够带来显著的性能提升，且具有极高的普适性——无论是稠密模型还是混合专家模型（MoE），无论是视觉任务还是文本模态，均能取得收益。特别是在大语言模型（LLMs）的预训练中，以相同计算量可取得约 **1.25** 倍的效果提升[1]。

## Attention Residuals

`Attention Residuals` 的核心思想在于：把残差连接中"固定单位权重的均匀累加"替换为"沿深度方向的 softmax 注意力"，使每一层都能以可学习、输入依赖的权重，从先前各层表示中做选择性检索。实验表明，该方法不仅训练过程比标准 Pre-Norm 残差更稳定，还能有效抑制隐藏状态幅值随深度的无界增长，并让梯度在层间分布得更均匀，从而呈现出更健康的训练动态。

怎么理解"固定单位权重的均匀累加"呢？我们来看看标准的残差连接：

这里我们换一种写法，它能让我们看清更本质的东西。

先记 ，那么就有 ，约定 ，易得

于是它可以等价地写成：

也就是说，从  的视角看，残差连接就是把  等权求和，作为  的输入来得到 。一个很自然的推广，就是**把等权求和换成加权求和**：

意识到"换成加权求和"这条路可行之后，接下来的问题自然是： 该取什么形式？作者经过反复实验，最终确定：

这便是 Attention Residuals，其结构如图所示（其中 (b) 为 Full AttnRes）。

![[Y2026/Q3/0728_Kimi-K3/assets/Image 18.webp|Image]]

  

不过在 Kimi 真实的训练环境下，还需要一套进一步降低通信与显存的方案，于是就有了上图 (c) 的 Block 版。

### Block Attention Residuals

`Block AttnRes` 将  层划分为  个块：块内仍用标准残差求和把各层输出压缩成单个块表示，块间只对  个块级表示做注意力聚合。这一做法把显存与通信开销从  降到了 。

具体而言，定义 （嵌入向量始终作为一个独立的源——因为在 Full 版的注意力矩阵里，模型明显给嵌入层分配了可观的权重）。对于第  个块中的第  层，值矩阵为：

其中  是当前块内的累积部分和。键与注意力权重的计算方式与 Full AttnRes 一致。最终输出层聚合所有  个块表示。伪代码如下：

```
def block_attn_res(blocks: list[Tensor], partial_block: Tensor,                   proj: Linear, norm: RMSNorm) -> Tensor:    # blocks: N 个已完成块的表示，每个为 [B, T, D]    # partial_block: [B, T, D]（当前块内的部分和 b_n^i）    V = torch.stack(blocks + [partial_block])      # [N+1, B, T, D]    K = norm(V)    logits = torch.einsum('d, n b t d -> n b t', proj.weight.squeeze(), K)    h = torch.einsum('n b t, n b t d -> b t d', logits.softmax(0), V)    return hdef forward(self, blocks, hidden_states):    partial_block = hidden_states    # 在 Attention 之前先做一次 block attnres    h = block_attn_res(blocks, partial_block, self.attn_res_proj, self.attn_res_norm)    # 到达块边界时，把当前部分和作为一个新块存入历史    if self.layer_number % (self.block_size // 2) == 0:        blocks.append(partial_block)        partial_block = None    # Self-Attention 层    attn_out = self.attn(self.attn_norm(h))    partial_block = partial_block + attn_out if partial_block is not None else attn_out    # 在 MLP 之前再做一次 block attnres    h = block_attn_res(blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm)    # MLP 层    mlp_out = self.mlp(self.mlp_norm(h))    partial_block = partial_block + mlp_out    return blocks, partial_block
```

**效率**：显存从  降到 ，计算从  降到 。块数  在两个极端之间插值： 时退化为 Full AttnRes， 时退化为标准残差连接。实验显示，只需划分约 8 个块（即固定 ），就能拿回 AttnRes 绝大部分的收益。

此外，伪查询  作为可学习参数，与层的前向计算完全解耦。这一设计让同一个块内所有层可以批量计算注意力查询，把原本逐层的  次内存读取摊销成 1 次批量读取。再配合跨阶段缓存与两阶段计算策略，Block AttnRes 的端到端训练开销被压到 4% 以下，推理开销压到 2% 以下。

### Why Attention Residuals

#### 标准残差与先前方法的关系

标准残差连接，可以看作 `Attention Residuals` 在**注意力权重均匀分布时的特例**。

当伪查询  初始化为零时，所有 ，注意力权重退化为  的均匀分布，AttnRes 也就退化成了标准残差的等权求和形式：

其中  表示第  层输出对第  层输入的贡献权重。标准残差的  是全 1 下三角矩阵（对应等权求和），而 Full AttnRes 的  是秩为  的稠密矩阵。

类似地，HC/mHC 也可以看成深度方向上的线性注意力。记  为  个并行流的状态，其递推展开后的有效权重为：

其中  为累积转移矩阵。这对应于 -半可分结构，等价于带矩阵值状态的深度线性注意力。而 AttnRes 把它推广成了深度方向上的 **softmax 注意力**。

因此，标准残差连接以及已有的推广模型（Highway、HC/mHC 等），都可以视为**深度方向的线性注意力**；AttnRes 则是把这一族方法推广到了**深度方向的 softmax 注意力**。

#### 序列与深度的对偶性

`Attention Residuals` 的设计灵感，来自序列维度与深度方向之间形式上的对偶关系。残差连接通过固定递推  在深度方向传播信息，正如 RNN 通过固定递推在时间方向传播信息。

这一对偶可以更具体地展开：

|序列维度|深度维度|
|---|---|
|标准线性注意力|标准残差|
|Highway / 门控线性注意力|Highway Network|
|DeltaNet / DDL|DDL|
|softmax 注意力（Transformer）|**Attention Residuals**|

正如 Transformer 用 softmax 注意力替代了 RNN 的时序递推，`Attention Residuals` 也用层间 softmax 注意力替代了残差连接的深度递推，完成了深度方向上与序列方向如出一辙的"线性 → softmax"跃迁。

## 实验

一方面，Attention Residuals 明显提升了训练过程的收敛效率，在 Scaling Law 实验中优于 Baseline：

![[Image 17.png|Image]]

Block AttnRes 在最大规模下达到 1.692 的验证损失，等效于相对 Baseline 约 **1.25 倍**的计算优势。Full AttnRes 与 Block AttnRes 之间的差距随规模增大而缩小，在最大规模下仅差 0.001，说明 Block 版以极小的开销就找回了 Full 版绝大部分收益。

另一方面，AttnRes 在训练动态上也表现出明显的改善：

![[Image 18.png|Image]]

- • **验证损失**：AttnRes 全程保持更低的验证损失，且差距在衰减阶段进一步拉大，最终损失显著更低。
    
- • **输出幅值**：Baseline 受 Pre-Norm 稀释问题困扰，隐藏状态幅值随深度单调增长，深层被迫学习越来越大的输出来维持影响力；Block AttnRes 把增长限制在块内，并在块边界通过选择性聚合"重置"累积，形成有界的周期性模式。
    
- • **梯度幅值**：Baseline 的残差权重固定为 1，无法调节梯度跨深度的流动，导致最早几层的梯度过大；AttnRes 可学习的 softmax 权重在多个源之间引入了竞争，使梯度分布显著更均匀。
    

在下游任务上，AttnRes 在所有评估基准上都匹配或超越 Baseline，其中多步推理类任务的提升尤为明显：

![[Image 19.png|Image]]

研究还对比了不同残差变体的特性：

- • **Attention Residuals** 的注意力模式呈**对角主导**特征——每层最关注自己的直接前驱，同时对嵌入层（source 0）保持持久的非平凡权重，并偶发地出现对角外的集中区域，说明模型学到了超出标准残差路径的跳跃连接。
    
- • **标准残差** 的混合矩阵是全 1 下三角，每层等权累加所有先前层，无法做选择性检索。
    
- • **DenseFormer** 使用固定的、输入无关的标量系数，效果与 Baseline 基本持平（1.767 vs 1.766），说明显式的输入相关权重至关重要。
    
- • **mHC** 通过  个并行流引入输入相关性，把损失改善到 1.747；但 AttnRes 以更低的每层内存 I/O 开销（ vs ）取得了更优的结果（Full AttnRes 1.737，Block AttnRes 1.746）。
    

## 总结

1. 1. Attention Residuals 是对标准残差连接的一种替代方案。在实际架构选型中，除了朴素的残差连接，也可以考虑它的改进变体（如 HC/mHC）或本文介绍的 Attention Residuals。
    
2. 2. 该方法用层间 softmax 注意力替代朴素的等权累加，在大语言模型预训练中展现出显著的性能提升；与此同时，它只额外引入"每层一个可学习向量"的极小参数与开销，因此具备广泛的应用潜力（典型如 Kimi K3）。
    

**不过**，Full AttnRes 在大规模分布式训练中会带来显存与通信开销的增加（），必须借助 Block 分块、跨阶段缓存和两阶段计算等策略加以优化，才能把端到端开销控制在可接受的范围（训练 < 4%，推理 < 2%）。

## 参考文章

- • Attention Residuals - arXiv:2603.15031
    
- • Attention Residuals 回忆录 - 科学空间 | Scientific Spaces