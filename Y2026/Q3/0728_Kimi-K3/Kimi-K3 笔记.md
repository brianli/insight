## 核心贡献

- **Pre-training at the open frontier**. We train a 2.8T-parameter native multimodal MoE model with 104B activated parameters and a 1M-token context window. KDA, AttnRes, Stable LatentMoE, refined data and training recipes collectively improve overall scaling efficiency by approximately 2.5× over Kimi K2. 
- **Reinforcement learning for multi-effort test-time scaling**. We conduct RL across general, agentic, and coding domains and multiple reasoning-effort levels, then consolidate the resulting capabilities into a unified model. 
- **Infrastructure for multi-trillion-parameter, million-token intelligence**. We introduce KDA systems co-designs; MoonEP and memory-efficient infrastructure for 2.8T-parameter MoE pre-training; a co-located RL system with resumable sandboxes for million-token agentic trajectories; and more infrastructure innovations. 
- **An open frontier model**. We release the full Kimi K3 model weights, making frontier intelligence available for research, deployment, and further innovation

## 模型架构

在模型架构层面上，主要是围绕着序列长度（KDA/GatedMLA）/网络深度（AttnRes）/模型宽度（Stable LatentMoE）三个信息流维度组织（sequence length, network depth, and model width），最终结果以 2.5x scaling efficiency 增益呈现。
![[k3_tech_report.pdf#page=3&rect=71,300,545,738|k3_tech_report, p.3]]

### Stable LatentMoE：宽的是条件容量，不是主干 hidden size

K3 的 Stable LatentMoE 将模型扩展到 896 个 routed experts，每 token 激活 16 个，并配置 2 个 shared experts。它归在“模型宽度”维度，但不是把主干 hidden dimension 直接变大；K3 的主 hidden dimension 仍为 $7168$，扩展的是**专家池规模、专家组合空间与总参数容量**。

和普通 MoE 让每个 routed expert 接收完整 hidden representation 不同，LatentMoE 先将 $d=7168$ 压到 $\ell=3584$，让 routed experts 在低维 latent space 中计算，再映射回 $7168$ 维。shared experts 保留 full-width 路径，承担公共变换：

$$
7168 \rightarrow 3584
\rightarrow
\text{Top-16/896 routed experts}
\rightarrow 3584
\rightarrow 7168
$$

因此它的核心杠杆是：

> **专家池变宽、单个 routed expert 变窄；用一次共享的降维/升维换取更大的专家池。**

相对 DeepSeekMoE，K3 继承了 `shared experts + routed experts` 的组织方式，但 DeepSeek 主要通过细粒度专家分割增加专业化；Stable LatentMoE 进一步把 routed expert 的工作空间压到 latent dimension，从而降低专家通信、权重读取和内部计算成本。K3 再用聚合后 RMSNorm、SiTU-GLU 和 Quantile Balancing，分别控制聚合尺度、激活爆炸和 896 专家的负载不均。

注意两个 `latent` 不是一回事：

- DeepSeek MLA 的 latent：压缩 KV cache，服务于长序列；
- K3 LatentMoE 的 latent：压缩 routed expert 的 channel representation，服务于模型容量与 MoE 效率。

### 序列长度挑战

K3 模型究竟是如何解决序列长度挑战的呢？即如何从之前的 128K 的序列长度拉长到 1M 的序列长度呢？一句话回答：KDA + GatedMLA 混合注意力，并辅助 NoPE 和 chunkwise 并行训练。KDA → KDA → KDA → Gated MLA → KDA → KDA → KDA → Gated MLA → ...这种 3:1 混合的方式进行注意力操作，即 3 层 KDA来高效地处理 99% 的序列上下文，而 1 层 Gated MLA 来周期性地提供全局精确注意力来确保精确检索任何位置的内容。

标准 self-attention 两个开销：
- 计算：每个 query 和所有 key 做内积，计算公式为 $\mathcal{O}(n^2 \cdot d)$ 
- 显存：KV Cache 存储每个 token 的 key 和 value，计算公式为$\mathcal{O}(n \cdot d \cdot \ell)$ 
在 K3 模型中，上面三个参数对应值分别为$n = 10^6, d = 7168, \ell = 93$
按照这个参数估计计算量：$$\text{Attention FLOPs} \approx 2 \cdot n^2 \cdot d \cdot \ell \approx 1.3 \times 10^{15}$$估计存储量：$$\text{KV Cache} \approx 2 \cdot n \cdot d \cdot \ell \cdot 2\ \text{bytes (BF16)} \approx 2.6\ \text{TB}$$
一块 H100 的卡提供 80GB 显存，若上面 2.6TB 的话显存存储都需要 32 张卡以上，因此直接使用 标准的 self-attention 是行不通的。长上下文问题的本质不是"能不能做"，而是 **"能不能用线性（而非二次）成本做"**。

KDA （Kimi Delta Attention）核心思路：用固定大小的递推状态替代逐token 的 KV 矩阵。
递推状态$S_t \in \mathbb{R}^{d_k \times d_v}$ 是固定大小的矩阵。更新规则：  $$\boxed{S_t = \big(I - \beta_t k_t k_t^\top\big) \cdot \text{Diag}(\alpha_t) \cdot S_{t-1} + \beta_t k_t v_t^\top}$$
$$
  \begin{aligned}
  S_t &= \text{Diag}(\alpha_t) S_{t-1} - \beta_t k_t k_t^\top \text{Diag}(\alpha_t) S_{t-1} + \beta_t k_t v_t^\top \\
      &= \text{Diag}(\alpha_t) S_{t-1} + \beta_t k_t \left(v_t^\top - k_t^\top \text{Diag}(\alpha_t) S_{t-1}\right) \\
      &= \text{Diag}(\alpha_t) S_{t-1} + \beta_t k_t \left(v_t - (\text{Diag}(\alpha_t) S_{t-1})^\top k_t\right)^\top
  \end{aligned}
  $$
  $$\boxed{S_t = \underbrace{\text{Diag}(\alpha_t) S_{t-1}}_{\text{衰减后的旧 state}} + \beta_t \cdot k_t \cdot 
  \underbrace{\bigg(v_t - \underbrace{\big(\text{Diag}(\alpha_t) S{t-1}\big)^\top k_t}_{\text{对 } k_t \text{ 
  的预测值}}\bigg)^\top}_{\text{预测误差（= delta！）}}}$$
  $$\begin{aligned}
  \text{状态矩阵 }S &: \text{从 keys 预测 values 的线性模型——"我对世界的理解"} \\
  \text{当前输入 }(k_t, v_t) &: \text{新的训练样本——"这个 token 告诉我什么"} \\
  \text{DELTA }&= v_t - \underbrace{S_{t-1}^\top k_t}_{\text{模型的预测}} \\[4pt]
  \text{DELTA }&= \text{实际值 }-\text{ 预测值}
  \end{aligned}$$
  只有当预测有误（delta ≠ 0）时，$S_t$ 才会被更新。如果旧 state 已经完美预测了当前 $v_t$，delta = 0，$S_t$ 不变。
$$S_t = \sum_{i=1}^{t} w_i \cdot k_i v_i^\top$$

其中 $w_i$ 是第 $i$ 个 token 经过衰减和选择性擦除后残留的权重。
$\tilde{o}_t$ 的物理含义非常直接：这是从 state $S_t$ 中，根据当前 query $q_t$ 检索出来的"记忆回答"。$\tilde{o}_t$ 就是 KDA 版本的 attention 输出——从固定大小的压缩记忆中，根据当前 query 检索出的、对全部历史 value 的加权聚合结果。
  $$\tilde{o}_t = S_t^\top q_t$$
$$\boxed{\tilde{o}_t = \sum_{i=1}^{t} \underbrace{w_i \cdot (k_i^\top q_t)}_{\text{历史 token i 与当前 query 的匹配度}} \cdot
   \underbrace{v_i}_{\text{历史 token i 的内容}}}$$
将上面公式进行详细拆解：
  - 第一部分是遗忘，$\alpha_t \in (0,1)^{d_k}$，每个 channel 有自己的 decay。$\alpha$ 越接近 0（衰减越大），旧信息被忘得越多。KimiLinear 用 $g = -e^A \cdot \text{Softplus}(z) \in (-\infty, 0)$ 产生 decay。这意味着 $\alpha = e^g$ 可以接近 0——某些
  channel 的留存率可以极小。在 K3 中将这块调整为$$g = g_{\text{min}} \cdot \text{Sigmoid}(e^A z), \quad g_{\text{min}} = -5 \implies \alpha = e^g > e^{-5} \approx 6.7 \times 10^{-3}$$
  $$\text{Diag}(\alpha) \cdot S = \begin{bmatrix} \alpha_1 \cdot S_{1,1} & \alpha_1 \cdot S_{1,2} \\ \alpha_2 \cdot S_{2,1} &
   \alpha_2 \cdot S_{2,2} \end{bmatrix}$$
   - 第二部分选择性擦除，当新 token 到来，如果它的 key $k_t$ 和 state 里已有的信息冲突，delta rule 会把旧信息"擦掉"一部分，程度由 $\beta_t$ 控制。
   - 第三部分写入，把新 token 的 key-value 关联加入 state。
   - 第四部分读取，用当前 query $q_t$ 去查询 state，$S_t^\top q_t$ 把 $d_k \times d_v$ 的 state 压缩成 $d_v$ 维的输出向量。全程 $\mathcal{O}(d_k \cdot d_v)$ 计算量，和 $t$ 无关。

MLA 本身用了 latent KV 压缩（$c_t = W_c x_t \in \mathbb{R}^{d_c}$，$d_c \ll d$），KV cache 从 $\mathcal{O}(n \cdot d)$ 降到 $\mathcal{O}(n \cdot d_c)$。在 K3 中，MLA 只有 24 层（93 层中的 24 层），进一步压低了 KV cache 总量。
