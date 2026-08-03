https://mp.weixin.qq.com/s/aA72wF8LJrnXrW6uCLHikA

#model/kimi/k3 

![[Image.gif|Image]]

## 

放弃残差之后，底层架构该怎么写？苏剑林用这篇文章，还原了他们把 Attention 加进层间连接的全过程。

**©PaperWeekly 原创 · 作者 |** 苏剑林

**单位 |** 科学空间

**研究方向 |** NLP、神经网络

这篇文章介绍我们的一个最新作品 Attention Residuals（AttnRes）[1]，顾名思义，这是用 Attention 的思路去改进 Residuals。

![[Y2026/Q3/0728_Kimi-K3/assets/Image 19.webp|Image]]

不少读者应该都听说过 Pre Norm/Post Norm 之争，但这说到底只是 Residuals 本身的“内斗”，包括后来很多 Normalization 的变化都是如此。

比较有意思的变化是 HC [2]，它开始走扩大残差流的路线，但也许是效果上的不稳定，并没有引起太多反响。

后来的故事大家可能都知道了，去年底 DeepSeek 的 mHC [3] 改进了 HC，并在更大规模实验上验证了它的有效性。

相比于进一步扩大残差流，我们选择了另一条激进的路线：直接在层间做 Attention 来替代 Residuals。

当然，全流程走通还是有很多细节和工作的，这里就简单回忆一下相关的心路历程。

![[Y2026/Q3/0728_Kimi-K3/assets/Image 20.webp|Image]]

〓 AttnRes 示意图

![[Y2026/Q3/0728_Kimi-K3/assets/Image 21.webp|Image]]

层间注意

按照惯例，我们还是从 Residuals [4] 说起，大家应该耳熟能详了，它的形式为

![[Y2026/Q3/0728_Kimi-K3/assets/Image 22.webp|Image]]

这里我们换另外一种写法，它能让我们看出更深刻的东西。

先记 ，那么有 ，约定 ，那么易得 ，于是它可以等价地写成：

![[Y2026/Q3/0728_Kimi-K3/assets/Image 23.webp|Image]]

即从  的视角看，Residuals 是将  等权求和作为  的输入来得到 ，那么一个自然的推广就是换成加权求和：

![[Y2026/Q3/0728_Kimi-K3/assets/Image 24.webp|Image]]

这便是 AttnRes 的萌芽。上式还给  多加了两个约束，我们先来讨论一下它们的必要性：

1. 约束  保证了同一个  对不同层的贡献始终是同向的，避免出现一层想要增大  而另一层却想要缩小  的不一致性，直觉上对模型的学习更加友好；

2. 我们用的  是带 In Norm 的，会对输入先做 RMSNorm，由于 RMSNorm(x)=RMSNorm(cx) 对  都恒成立，所以加权平均和加权求和完全等价，约束  不会降低表达力。

![[Image 25.webp|Image]]

超级连接

在开展 AttnRes 之前，我们先简单回顾一下 HC（Hyper-Connections），并证明它也可以理解为层间 Attention，从而表明层间 Attention 确实是一条更为本质的路线。

HC 将 Residuals 改为：

![[Image 26.webp|Image]]

其中：

![[Image 27.webp|Image]]

经典选择是 k=4。

简单来说，状态变量扩大到 k 倍，输入到  前，用一个  矩阵将它变回 1 倍，输出后再用  将它变回 k 倍，最后跟  调节过的  相加。

如果不限定  的形式，那么像 Post Norm、Highway [5] 都是 HC 的特例。

类似地记 ，那么 ，约定 ，那么它也可以展开成：

![[Image 28.webp|Image]]

其中  定义为 。

进一步约定 ，我们就可以写出：

![[Image 29.webp|Image]]

注意每一个  都是  矩阵，相当于一个标量，所以它也是式 (3) 的层间 Attention 形式。

熟悉[线性注意力](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247705148&idx=1&sn=7876b114499952dca8fad437c5e9f708&scene=21#wechat_redirect)的读者应该很快能理解这个结果，HC 其实相当于“旋转 90 度”的 DeltaNet。

实践中，三个  矩阵由  激活的简单线性层计算而来，这导致连乘起来的  有爆炸或坍缩的风险，也无法保证  的非负性。

后来 mHC 做了改进，它先将三个  都改为 Sigmoid 激活，保证了  非负，然后交替归一化  使其满足双随机性，由双随机矩阵对乘法的封闭性保证  的稳定，最后实验也验证了这些改动的有效性。

不过，也有一些新实验如[你的DeepSeek mHC可能根本不需要“m”约束](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247718198&idx=1&sn=877a6e15d019329017b730a5ee344c52&scene=21#wechat_redirect)显示  直接设为单位阵就足够好了。

![[Image 30.webp|Image]]

众人拾柴

让我们回到 AttnRes 上。在意识到 AttnRes 的可行性后，接下来的问题便是， 该取什么形式呢？

一个很自然的想法是照着标准的[Scaled Dot-Product Attention](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247486960&idx=1&sn=1b4b9d7ec7a9f40fa8a9df6b6f53bbfb&scene=21#wechat_redirect)来，但当时笔者想着先快速尝试一下，所以选了个更简单的形式：

![[Image 31.webp|Image]]

其中  是一个可训练的向量参数，即直接以一个数据无关的静态向量为 Q、而 K、V 都是  去做 Softmax Attention，这便是第一版 AttnRes。

令人惊喜的是，就这么个简单的设计，它相比 Residuals 的提升已经非常显著！

当笔者在组内分享了 AttnRes 的初步实验结果后，@张宇 [6] 和@广宇 [7] 表现出极大的兴趣，然后一起参与进来，开始在更大规模的模型上做一些验证，发现结果都很让人欣喜。

期间，我们还尝试了一些更复杂的设计，发现它们大多不如这个简单的版本，只有给 K 多加个 RMSNorm 的操作，能取得比较稳定的收益，这构成了终版的 AttnRes 形式：

![[Image 32.webp|Image]]

然而，AttnRes 毕竟是一个密集型的层间连接方案，在 K2 甚至更大规模上的训练和推理是否可行呢？

令人振奋的是，@V哥 [8] 经过精妙的分析，首先肯定了推理上的可行性，其中的“点睛之笔”正是开始图方便的静态 Q 设计！

这使得我们计算完  后就可以提前计算 t > s 的注意力 ，给 Infra 带来了足够的挪腾空间。

但很不幸的是，训练的同学如@王哥 [9] 经过仔细分析，判断出在我们当前的训练环境下，AttnRes 还是不够可行（说白了还是穷），需要一个进一步降低通信和显存的方案，于是就有了下面的 Block 版。

相应地，之前的版本我们称为 Full 版。

![[Image 33.webp|Image]]

分块版本

从 Full AttnRes 到 Block AttnRes，相当于以往的将平方 Attention 线性化的过程，各种已有的 Efficient Attention 的思路都可以套上去试试。

比如我们第一个尝试的就是 SWA（Sliding Window Attention），然而发现实际效果很糟糕，甚至还不如 Residuals。

经过反思，笔者认为可以这样理解：Residuals 本身已经是一个非常强的 Baseline，它对应于所有状态向量的等权求和，任何新设计想要超越它，那么至少在形式上要能够覆盖它。

Full AttnRes 显然能满足这个条件，但是加上 SWA 后并不满足，它扔掉了一部分状态，无法覆盖“所有状态向量等权求和”这一特例。

由此我们意识到，对于 AttnRes 来说，“压缩”可能要比“稀疏”要更有效，而且压缩也不用太精细，简单的加权求和可能足矣。

经过一番构思和打磨后，@张宇和@广宇提出了论文中的 Block AttnRes 设计，它结合了分 Block 处理和求和压缩的思想，取得了接近 Full 版的效果。

Block AttnRes 的想法大致是这样的：

首先 Embedding 层单独作为一个 Block，这是因为通过观察 Full 版的 Attention 矩阵（这也是 Attention 概念的好处，可以随时可视化注意力模式）发现，模型偏向于给 Embedding 层可观的 Attention，所以有必要将 Embedding 独立出来。

剩下的每 m 层作为一个 Block，Block 内通过求和做压缩，以求和结果为单位算 Block 间 Attention。

实验显示，只需固定分 8 个 Block 左右，就可以获得 AttnRes 大部分收益。

经过评估，训练和推理的同学一致认为，Block AttnRes 的额外开销很小，相比于其效果提升是完全值得的。

详细分析看@王哥和@V哥的帖子，如果要数字的话，大致是 5% 以内的开销，换取 25% 的收益。

于是所有同学全力推动它进主线，这又是一段充实且愉快的经历，就不展开谈了。

![[Image 34.webp|Image]]

矩阵视角

值得一提的是，我们还可以通过 Attention 矩阵，将 Residuals、HC/mHC、Full AttnRes、Block AttnRes 统一起来，这也是一个比较有趣的理解视角，示例如下。

其中 ，Block AttnRes 版对应 m=3，以及 ，后面这个记号我们在《让炼丹更科学一些：新恒等式，新学习率》[10] 也用过。

Residuals  

![[Image 35.webp|Image]]

HC/mHC  

![[Image 36.webp|Image]]

Full AttnRes  

![[Image 37.webp|Image]]

Block AttnRes  

![[Image 38.webp|Image]]

![[Image 39.webp|图片]]

**相关工作**

从计划做 AttnRes 后，笔者和众多小伙伴们就一直沉浸在打磨、验证、加速的过程中。

部分读者可能知道，笔者的研究风格是，先尽自己最大努力去推导和求解，直到遇到困难或者完全解决后才寻找相关文献。

恰好笔者遇到了一群相似的小伙伴，恰好这一次 AttnRes 的探索整体都比较顺畅，所以直到所有测试都基本通过，开始准备技术报告时，我们才开始去调研相关文献。

但也正因如此，“不查不知道，一查吓一跳”，原来 Dense Connection、Depth Attention 相关工作已经非常多。

除了经典的 DenseNet [11] 外，我们调研到的还有 DenseFormer [12]、ANCRe [13]、MUDDFormer [14]、MRLA [15]、Dreamer [16] 等，甚至 BERT 之前的 ELMo [17] 都部分地应用了类似设计，这些我们都收录到了参考文献中。

技术报告发出去后，陆续收到了一些读者的评论，指出了一些还没收录到的相关工作，如 SKNets [18]、LIMe [19]、DCA [20] 等，对此我们表示抱歉和感谢，并承诺在后续修订后会把尽量它们补充上去。

但不管是读者还是作者本人，还请对此保持理性，文献综述是件不容易的事情，有些遗漏在所难免，我们对所有相关的工作都致以崇高的敬意。

同时，我们也呼吁大家多关注 AttnRes 在“Depth Attention”概念以外的工作量。

我们非常同意，在 2026 年的今天，“Depth Attention”或者说“Layer Attention”是一个毫无新意的想法。

但如何将它用于足够大的模型，作为 Residuals 足够强的替代品，同时还满足训练和推理的效率需求，并不是一件容易的事情。

据我们所知，AttnRes 是首个实现这一点的工作。

![[Image 40.webp|图片]]

**文章小结**

本文介绍了我们在模型架构上的最新结果 Attention Residuals（AttnRes），它用层间 Attention 来替代朴素的 Residuals，并通过精细的设计使其能满足训练和推理的效率要求，最终成功地将它拓展到足够大的模型上。

![[Image.svg|图片]]

**参考文献**

![[Image 1.svg|图片]]

[1] https://papers.cool/arxiv/2603.15031

[2] https://papers.cool/arxiv/2409.19606

[3] https://papers.cool/arxiv/2512.24880

[4] https://papers.cool/arxiv/1512.03385

[5] https://papers.cool/arxiv/1512.03385

[6] https://x.com/yzhang_cs

[7] https://x.com/nathancgy4

[8] https://zhuanlan.zhihu.com/p/2017528295286133070

[9] https://www.zhihu.com/question/2016993095078684011/answer/2017381145474508331

[10] https://kexue.fm/archives/11494

[11] https://papers.cool/arxiv/1608.06993

[12] https://papers.cool/arxiv/2402.02622

[13] https://papers.cool/arxiv/2602.09009

[14] https://papers.cool/arxiv/2502.12170

[15] https://papers.cool/arxiv/2302.03985

[16] https://papers.cool/arxiv/2601.21582

[17] https://papers.cool/arxiv/1802.05365

[18] https://papers.cool/arxiv/1903.06586

[19] https://papers.cool/arxiv/2502.09245

[20] https://papers.cool/arxiv/2502.06785

**更多阅读**

[![[Image 20.png|Image]]](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247701843&idx=1&sn=817c0b9a4589052831da2170c0157ec0&scene=21#wechat_redirect)

[![[Image 41.webp|Image]]](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247718294&idx=1&sn=98574ef7e60acb19a3894a5663f78549&scene=21#wechat_redirect)

[![[Image 21.png|Image]]](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247713461&idx=1&sn=9cd02b0ef3871afe7817c03553b16930&scene=21#wechat_redirect)

![[Image 1.gif|Image]]

**#投 稿 通 道#**

 **让你的文字被更多人看到** 

如何才能让更多的优质内容以更短路径到达读者群体，缩短读者寻找优质内容的成本呢？**答案就是：你不认识的人。**

总有一些你不认识的人，知道你想知道的东西。PaperWeekly 或许可以成为一座桥梁，促使不同背景、不同方向的学者和学术灵感相互碰撞，迸发出更多的可能性。 

PaperWeekly 鼓励高校实验室或个人，在我们的平台上分享各类优质内容，可以是**最新论文解读**，也可以是**学术热点剖析**、**科研心得**或**竞赛经验讲解**等。我们的目的只有一个，让知识真正流动起来。

📝 **稿件基本要求：**

• 文章确系个人**原创作品**，未曾在公开渠道发表，如为其他平台已发表或待发表的文章，请明确标注 

• 稿件建议以 **markdown** 格式撰写，文中配图以附件形式发送，要求图片清晰，无版权问题

• PaperWeekly 尊重原作者署名权，并将为每篇被采纳的原创首发稿件，提供**业内具有竞争力稿酬**，具体依据文章阅读量和文章质量阶梯制结算

📬 **投稿通道：**

• 投稿邮箱：hr@paperweekly.site 

• 来稿请备注即时联系方式（微信），以便我们在稿件选用的第一时间联系作者

• 您也可以直接添加小编微信（**pwbot02**）快速投稿，备注：姓名-投稿

![[Image 42.webp|Image]]

**△长按添加PaperWeekly小编**

🔍

现在，在**「知乎」**也能找到我们了

进入知乎首页搜索**「PaperWeekly」**

点击**「关注」**订阅我们的专栏吧

·

![[Image 43.webp|Image]]