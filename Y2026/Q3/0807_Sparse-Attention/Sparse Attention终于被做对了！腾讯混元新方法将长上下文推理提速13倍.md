https://mp.weixin.qq.com/s/LRs5yKYiv0EBoJoXrMW0_w

智猩猩AI整理

编辑：林夕

做长上下文的人，有三个问题永远躲不掉。

算力随序列长度平方级增长，外推能力说不准什么时候就崩，KV Cache跟着长度线性膨胀。

Sparse attention看起来是唯一出路。把上下文切成chunk，每次query只看最相关的几个，剩下的KV丢CPU上。开销恒定，内存也省了，听着非常理想。

但这些年试下来，行业里有个公开的秘密，还没有一种分块稀疏注意力，真能追平全注意力。

问题出在哪？Chunk选不准。

用简单的均值池化或者取最大logits来估计每个chunk的重要性，放到真实场景里，基本是在系统性错估。真正关键的信息块，经常就这么被漏掉了。

为此，腾讯混元团队联合多家机构提出了HiLS-Attention，该方法让原生稀疏注意力在效果上媲美甚至超越全注意力（短文本能力无损、长文本外推512倍、复杂检索反超50%），同时在512K超长上下文中实现13倍以上的推理加速，打破了长久以来困扰业界的效率与效果不可兼得困境。

![[Y2026/Q3/0807_Sparse-Attention/assets/Image.webp|Image]]

- 论文标题：HierarchicalSparseAttentionDoneRight: TowardInfiniteContextModeling
    
- 论文链接：https://arxiv.org/pdf/2607.02980
    
- github链接：  
    
    https://github.com/Tencent-Hunyuan/HiLS-Attention
    

_**01**_

**用分层Softmax**

**给Chunk选择装上“眼睛”  
**

稀疏注意力要先回答一个问题：对于当前query，哪些chunk值得看？

最朴素的做法叫Block Sparse Attention。它把每个chunk里所有token和query的注意力logit加起来，得到一个chunk mass，然后选Top-K。

问题是，这个chunk mass需要完整算一遍QK，和全注意力一样贵。那稀疏的意义就没了。

为了省算力，大家只能退而求其次，用压缩后的chunk表示来估计chunk mass。NSA、MoBA这些做法用的是mean-pooling，把一个chunk里的key平均成一个summary key。

这个做法在一种情况下靠谱，就是chunk内部logit分布比较均匀。

但真实文本里经常出现另一种情况，某个token一枝独秀，其他token都是打酱油的。这时候mean会严重低估这个chunk的重要性，max反而更准确。

真实分布介于两者之间，mean和max都会丢信息，chunk就选错了。

还有一件更隐蔽的事。

这些summary key只在选chunk的时候用一次，选完就被扔了。语言模型的loss梯度流不到summary上，它根本学不会抬高有用chunk、压低没用chunk。

研究团队把这两个问题拆开看。

（1）怎么构造一个数学表达能力够强的chunk重要性估计函数。

（2）怎么让这个chunk summary能被语言模型loss端到端地训练。

真实chunk mass可以写成LogSumExp形式。HiLS对它做一阶泰勒展开，得到一个既简洁又优雅的近似，包含两项。

**（1）相关性项**，summary key衡量query与chunk的整体相关性。

**（2）熵偏置项**，捕捉对应注意力分布的熵。

分布均匀时，熵接近log S；分布集中时，熵接近0。它把mean和max各自缺失的那一半信息补上了，让代理分数在任意分布下都能贴近真实chunk重要性。

为了让summary可学习，HiLS在每个chunk末尾塞了一个特殊的landmark token。

但真正关键的，是HiLS把softmax拆成了两级。

第一级是**块间softmax**，根据landmark算出的chunk-mass surrogate决定每个chunk整体分到多少注意力。

第二级是**块内softmax**，在被选中的chunk内部决定token之间的相对权重。

因为surrogate score直接出现在前向注意力权重里，语言模型loss的梯度可以一路反传到landmark表示。chunk选择第一次变成了端到端可学习的过程。

![[Y2026/Q3/0807_Sparse-Attention/assets/Image 1.webp|Image]]

HiLS-Attention架构总览

**光有理论不够，工程上还有四个细节得搞定。**

**（1）HoPE位置编码**

标准RoPE在8K训练长度下会让HiLS性能低于全注意力基线。HiLS改用HoPE，保留短周期维度，把长周期维度换成NoPE，避免训练长度外的分布外旋转。

**（2）低秩查询校准（Q-Cal）**

token级query不一定适合估计chunk级重要性。HiLS加一个低秩适配器，只引入0.6%额外参数。

**（3）GQA兼容**

同一组查询头共享KV头，HiLS用组内各头分数的最大值聚合，确保任意头觉得重要的块都不会被漏掉。

**（4）高效Kernel**  

NSA一次处理一个query token，HiLS 把相邻M个query token打包，计算它们选中chunk的并集上的注意力。Tensor Core利用率更高，重叠的K/V chunk 还能复用。

![[Y2026/Q3/0807_Sparse-Attention/assets/Image 2.webp|Image]]

NSA（左）和 HiLS（右）的 kernel 设计对比

![[Y2026/Q3/0807_Sparse-Attention/assets/Image 3.webp|Image]]

 _相邻 query token 的 chunk 重叠率分析_

_**02**_

**效率与效果首次**

**同时打破二选一困境**  

论文在345M到7B三个参数规模上做了系统验证，实验结果高度一致。

![[Y2026/Q3/0807_Sparse-Attention/assets/Image 4.webp|Image]]

超长上下文外推能力

  

![[Y2026/Q3/0807_Sparse-Attention/assets/Image 5.webp|Image]]

推理速度优于Full Attention

  

![[Y2026/Q3/0807_Sparse-Attention/assets/Image 6.webp|Image]]

短/中上下文任务性能保持

  

![[Y2026/Q3/0807_Sparse-Attention/assets/Image 7.webp|Image]]

 YaRN 外推范围内性能保持

先看345M小模型从头训练的结果，上下文长度8K。

HiLS在训练长度8K 上困惑度与全注意力几乎持平（4.94 vs 4.96），短文本能力没有打折。

真正的差距出现在长上下文。8K训练长度下，全注意力一到32K直接炸到100以上，而HiLS在512K时PPL仍然只有5.95。HiLS用8K训练就能稳定跑到512K，没有外推崩溃。

![[Y2026/Q3/0807_Sparse-Attention/assets/Image 8.webp|Image]]

345M 模型在不同上下文长度下的困惑度对比

  

![[Y2026/Q3/0807_Sparse-Attention/assets/Image 9.webp|Image]]

训练步数 vs 困惑度（左）与 RULER 准确率（右）

  

更反直觉的是VT（变量追踪）任务，HiLS在8K就拿到72分，而全注意力只有34分，高出一倍。

研究团队分析，压缩过程本身有去噪效果，无关token的微小噪声在压缩中互相抵消，共享语义信号反而被保留，检索表征变得更干净。

S-N和MK-MQ在8K到4M范围内准确率保持在90%上下，几乎没有衰减。

![[Y2026/Q3/0807_Sparse-Attention/assets/Image 10.webp|Image]]

 345M 模型 RULER 检索结果（S-N / MK-MQ / VT）

困惑度稳住之后，更关键的问题是检索能力会不会丢。RULER长上下文检索的结果打消了这个顾虑。

![[Y2026/Q3/0807_Sparse-Attention/assets/Image.svg|Image]]

 上下文检索结果

  

345M小模型的表现已经很说明问题了，但真正落地的场景面对的是7B甚至更大规模的模型。

HiLS对此也给出了轻量迁移方案，以OLMo3-1025-7B为基础，只续训50B token即可转为HiLS-Attention，不需要推倒重来。

RULER平均分97.42，超过YaRN扩展基线；128K和256K验证PPL分别只有2.55和3.10，均优于全注意力基线。

![[Y2026/Q3/0807_Sparse-Attention/assets/Image.svg|Image]]

在Olmo3预训练语料上，不同上下文长度下的验证困惑度

![[Y2026/Q3/0807_Sparse-Attention/assets/Image.svg|Image]]

OLMo3-7B继续预训练的一般下游任务结果

LongBench-v1总分33.2，超过OLMo3+YaRN 32K的31.7。OpenCompass 11项短上下文基准性能基本保持。

![[Y2026/Q3/0807_Sparse-Attention/assets/Image.svg|Image]]

**LongBench-v1评测基准说明**

效果过关了，推理效率怎么样？这是稀疏注意力最原始的出发点。

HiLS的prefill延迟随上下文近线性增长，全注意力是二次增长；decode延迟则几乎是常数。

两条曲线16K左右交叉，过了这个长度差距越拉越大。到512K时，prefill快13.5倍，decode快15.7倍。效果好、跑得快，两件事第一次同时做到了。

![[Y2026/Q3/0807_Sparse-Attention/assets/Image.svg|Image]]

Prefill与Decode延迟随上下文长度变化曲线

  

_**03**_

**长上下文建模的新基线**  

HiLS-Attention不只是又一个稀疏注意力实现。它在数学层面同时处理了两个根本问题。

一是chunk重要性估计的表达力。LogSumExp线性化加熵偏置，把mean和max两种极端统一起来。

二是chunk选择的端到端可学习性。分层softmax把surrogate score直接嵌进前向注意力，让语言模型loss的梯度自然流到landmark token。

稀疏注意力不仅没有损失精度，在部分长上下文任务上反而比全注意力更强。加上原生稀疏训练、轻量CPT迁移、SGLang生产推理支持，HiLS-Attention给长上下文LLM提供了一条效果更好、成本更低的路。

**END**

**关注+星标，获取AI前沿进展与优质开源项目**