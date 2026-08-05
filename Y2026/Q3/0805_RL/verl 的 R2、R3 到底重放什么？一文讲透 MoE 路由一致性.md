https://mp.weixin.qq.com/s/pibO8ATJjcXDL1qmyYUOmQ
#AIInfra #mp/硅海行舟 

**版本边界**｜verl **v0.8.0**（2026-06-01，commit 7aed6b2）

稳定接口以 v0.8.0 为准；兼容性提醒核对至 2026-07-25 main（commit 983cb0f2）

同一个 MoE（Mixture of Experts，混合专家）模型、同一批 token，为什么在 rollout（轨迹生成）和训练时会走向不同专家？R3 论文在 Qwen3-30B-A3B、SGLang rollout 与 Megatron training 的特定实验中，观察到约 **10%** 的逐层路由决策不一致；层层累积后，**94% 的 token 至少有一层选择了不同专家**。

这不是一个只影响几个小数位的问题。MoE Router 的 top-k 是离散选择：分数略微抖动，token 就可能从 Expert A 跳到 Expert B，后续输出和 log probability 都会换一条计算路径。近端策略优化（PPO）、群组相对策略优化（GRPO）一类算法又恰好依赖“旧策略与新策略”的 importance ratio（重要性比率）；比较对象悄悄换了路径，ratio、KL 散度和裁剪统计就可能被额外噪声污染。

**先给结论**：verl 的 R2、R3 不是奖励模型，也不是新的策略优化算法。它们是 **MoE Router Replay** 模式——记录一次 forward 选中的 top-k 专家，并在后续 forward 中重放这组专家索引。R2 对齐训练引擎内部，R3 把 rollout 也纳入对齐。

## 一、先把三个 forward 放到一张图里

理解 R2/R3，关键不是背缩写，而是分清一批轨迹通常经历的三次计算：

**1. Rollout**：推理引擎生成 response，并记录采样轨迹。

**2. Recompute / compute_log_prob**：训练引擎重新前向，计算旧策略 log probability，供 importance ratio 使用。

**3. Actor update**：训练引擎用当前参数再次前向、反向，更新策略。

![[Y2026/Q3/0805_RL/assets/Image.webp|Image]]

图 1：R2 从训练侧的 recompute 开始记录并在 update 重放；R3 从 rollout 开始记录，并贯穿后续两个训练侧 forward。示意图，作者绘制。

在 dense 模型里，数值差异通常仍落在同一条算子路径上；在 MoE 中，Router 还要执行 top-k。推理与训练可能使用不同内核、精度、并行切分或执行顺序，边界附近的专家分数排序因此可能翻转。更进一步，即使都在训练引擎里，参数更新、随机性或批处理形态也会让前后两次 forward 选中不同专家。

## 二、R2：对齐训练侧的 recompute 与 update

verl 文档把 R2 模式称为 **Router Replay**。从操作上看，它做的是：在 actor 的 `compute_log_prob` 中记录每个 token、每个 MoE 层的 top-k expert indices；进入 actor update 后，不再让 top-k 自由改选，而是重放刚才的索引。

它要解决的是一个很具体的问题：计算 old log prob 与计算新策略 log prob 时，尽量让两者沿同一组专家路径比较。这样 importance ratio 的变化更接近“参数更新造成的概率变化”，而不是掺入“专家集合突然换了”的离散跳变。

**R2 的边界**：它没有记录 rollout 引擎实际走过的路由。因此，如果主要误差来自 vLLM/SGLang 与训练引擎之间，R2 只能保证“训练侧两次一致”，不能保证“生成时与训练时一致”。

还有一个容易混淆的点：R2 不是“激活重计算”，也不是把整个 forward 的中间结果缓存下来。它缓存的是专家索引，数据量比完整激活小得多，但仍会随 token 数、MoE 层数和 top-k 增长。

## 三、R3：把 rollout 的真实路径带进训练

R3 在论文中指 **Rollout Routing Replay**。它把记录点前移到 rollout：推理后端在生成 token 时导出 top-k expert indices；随后训练侧的 `compute_log_prob` 和 actor update 都重放 rollout 选择的专家。

于是，三次 forward 的离散路径被钉在同一参考路由上：response 是沿哪些专家生成的，训练就沿哪些专家解释它、更新它。R3 论文在前述实验设置中估算，train–infer KL 从约 **1.5e-3** 降至 **7.5e-4**；论文中的 dense Qwen3-8B 对照约为 **6.4e-4**。这些数字说明跨引擎路由差异可能是 MoE 特有的一块系统误差，但不是对所有模型、后端和集群的收益保证。

R3 也并非只有在一批 rollout 被做多次 update 时才有意义。即使每批数据只更新一次，rollout 与训练引擎之间仍可能存在数值和内核差异；这正是它比 R2 多覆盖的一段。

## 四、原理：固定选择，不冻结 Router

Router 先根据 token 表示 _x_ 和路由权重 _Wr_ 计算训练侧 logits：

普通 forward 会从 _s_train 重新做 top-k。重放时则使用参考掩码 _I_ref：R2 的参考来自 recompute，R3 的参考来自 rollout。被选专家的门控权重仍用当前训练侧 logits 重新归一化：

|   |   |
|---|---|
|_g_replay,i = _I_ref,i exp(_s_train,i)∑Mj=1_I_ref,j exp(_s_train,j)|（2）|

这里 _I_ref,i 是“专家 _i_ 是否在参考 top-k 中”的 0/1 掩码，_M_ 是专家总数，_g_replay,i 是第 _i_ 个专家对应的重放门控权重。离散的专家集合被固定，但 _s_train 仍由当前参数计算，归一化权重仍会变化，梯度也仍可回传到 _W_r。因此，**Router 没被冻结；被冻结的只是这一次样本比较所走的离散路径**。

![[Y2026/Q3/0805_RL/assets/Image 1.webp|Image]]

图 2：路由重放只复用 top-k expert indices；当前训练参数仍负责计算 logits、门控权重与梯度。示意图，作者绘制。

这套设计的精髓是把“连续学习”和“离散可比性”拆开：门控权重继续学习，专家路径在一次策略比较中保持一致。

## 五、在 verl v0.8.0 中怎么开

R2/R3 不是一个全局通用开关，而是落在具体 actor engine 的配置下面。稳定版示例给出了 Megatron 与 VeOmni 两条路径：

# Megatron：R2 或 R3

actor_rollout_ref.actor.megatron.router_replay.mode=R2

# VeOmni：R2 或 R3

actor_rollout_ref.actor.veomni.router_replay.mode=R3

# R3 还必须让 rollout 后端导出路由

actor_rollout_ref.rollout.enable_rollout_routing_replay=True

把模式值换成 `R2`、`R3` 或 `disabled`。R2 不需要、也不应同时开启 rollout route capture；R3 则必须开启。VeOmni 路径还要求 `use_remove_padding=True`。

但“配置项存在”不等于任意组合都能工作。v0.8.0 示例覆盖了 Megatron + Qwen3-30B-A3B，以及 VeOmni + Qwen3.5-35B-A3B 等组合；VeOmni 合入时的模型 hook 明确涉及 Qwen3 MoE 与 Qwen3.5 MoE。其他 MoE 家族、普通 FSDP actor 或不同 rollout backend，应该先查对应实现和测试，不能从示例外推为通用支持。

**快速演进提醒**：截至 2026-07-25 的 main，Qwen3.5 这类 hybrid-attention MoE 在 vLLM R3 路径上要求 vLLM ≥ 0.22.0；THD Megatron batch 中的 R2 路由保留仍有开放修复 PR。另一个缓解 R3 路由张量内存压力的提案已关闭且未合并，也侧面说明这部分开销不能忽略。升级前要按模型、训练引擎、rollout 后端和版本做一张兼容矩阵。

## 六、什么时候需要：先证明“路由”真是问题

R2/R3 不应该因为“MoE 训练看起来不稳”就默认全开。更稳妥的顺序是先做对照诊断：固定同一批 token 和同一份权重，分别记录 rollout、compute_log_prob、update 的专家索引，再联合观察 train–infer KL、importance ratio 长尾、clip fraction 和路由不一致率。

![[Y2026/Q3/0805_RL/assets/Image 2.webp|Image]]

图 3：选择 R2/R3 的核心不是“哪个更强”，而是异常跨过了哪一段边界。示意图，作者绘制。

**适合先试 R2：**MoE actor 的异常主要出现在训练侧 old/new log prob 比较；rollout 与 recompute 的路由已经接近，或当前后端无法稳定导出 rollout routes。

**适合评估 R3：**同权重下 rollout 与训练引擎的路由差异明显，train–infer KL 偏大，且版本、模型 hook 和 route export 链路都已验证。

**通常不需要：**dense 模型；路由一致性本来就好；或者根因明显是权重同步、量化差异、奖励异常、学习率、数据质量与 policy lag。路由重放不会替代这些诊断。

## 七、局限：对齐路径也会付出代价

**1. 它只消除一类误差。** R2/R3 针对的是离散专家选择不一致。推理量化、权重同步延迟、采样分布偏差、奖励噪声、优势估计和优化器设置仍需分别处理。

**2. 路由会陈旧。** 如果同一批 rollout 被复用多次，或 policy lag 很大，缓存路由来自较旧 Router。当前 Router 已经更偏好新的专家，却仍被迫沿旧专家前向，新增偏好可能得不到实际激活与梯度。这也是后续 PR2 工作提出 predictive replay 的动机；verl v0.8.0 并未实现 PR2。

**3. 有存储与通信成本。** 每个 response token、每个 MoE 层都可能携带 top-k 索引，长序列和深层模型会把这部分元数据放大。R3 还要跨 rollout/训练边界传递它。R3 论文报告其实现中的整体 rollout latency 开销低于 3%，但这是特定设置的结果，不等于总训练零成本。

**4. 一致性与探索存在张力。** GSPO 论文指出，固定路由可能限制 MoE 的有效容量。短窗口内固定路径能让概率比更可比；固定得过久，则可能阻碍 Router 及时切换到更合适的专家。两者并不矛盾，关键在 replay 的时间跨度。

**5. 工程耦合比算法描述更强。** R3 需要推理后端导出与训练模型逐层、逐 token 对齐的 routes，还要处理 padding、packed sequence、并行切分与 response mask。模型结构或后端版本变化，都可能让“索引还能传”不等于“语义仍对应”。

## 八、最后的判断

R2/R3 的价值，不是让 MoE Router 永远不变，而是让一次策略更新中的“旧策略—新策略”比较尽量发生在同一条离散计算路径上。

只需要训练侧可比性，用 R2；真正的问题跨越 rollout 与训练引擎，再考虑 R3。两者都应建立在观测证据和兼容性验证上，并选择**最小但足够的对齐范围**。如果没有测到路由不一致，保持 disabled 往往比多开一个复杂机制更诚实。

参考资料

[1] verl v0.8.0 Router Replay README：github.com/verl-project/verl/blob/v0.8.0/examples/router_replay/README.md

[2] Ma et al., “Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers”：arxiv.org/abs/2510.11370

[3] Zheng et al., “Group Sequence Policy Optimization”：arxiv.org/abs/2507.18071

[4] Dong et al., “PR2: Predictive Routing Replay for MoE-Based LLM Reinforcement Learning”：arxiv.org/abs/2606.00395