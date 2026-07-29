#agent/evomap 

我是 Lena。上周我有一整个下午对着 Dify 的 "Create App" 界面发呆，卡在一个看起来简单、其实不简单的问题上：**我要做的这个东西，应该选 Agent 还是 Workflow？**

这种决定的特点是——一旦你想明白了就觉得显而易见，但第一次碰的时候是真的会让人困惑。两边都能调用 LLM，两边都能用工具，两边都能产出结果。那它们到底差在哪——以及什么时候选错了会让你白白返工好几个小时？

下面是我在两种模式之间来回切换、把 Dify 文档刷得比我愿意承认的次数还多、把同一个项目重写了两遍之后，搞明白的东西。

## 速答：要灵活就选 Agent，要可控就选 Workflow

如果你只有 30 秒：**Dify 的 Agent 是实时推理、自己决定的——它自己挑用什么工具、什么时候用。** Dify 的 Workflow 走的是你提前定义好的路径——每一步都看得见，每一处分支都是显式的。

任务开放性高的时候用 Agent。任务步骤稳定的时候用 Workflow。这是核心的分界线。

但实际情况比这复杂得多，所以请继续往下看。

## Dify Agent vs Workflow 对比表

|维度|Dify Agent|Dify Workflow / Chatflow|
|---|---|---|
|控制流程|LLM 在运行时决定路径|你在可视化画布上把路径定义出来|
|记忆与上下文|内置对话记忆（最多 500 条消息）|Workflow：无记忆。Chatflow：每个 LLM 节点可配置记忆|
|工具使用|模型自主决定调用哪些工具|工具作为固定节点摆放——你决定它们什么时候运行|
|调试|比较难追踪——模型的推理是一个黑盒|节点级单步调试，单节点测试运行|
|可靠性|可预测性低——取决于模型的推理质量|高度可预测——同样的输入，同样的路径|

我搭东西的时候反复回来翻这张表。最让我意外的是两种模式下调试体验的差异。这点我下面会讲。

### 控制流程

在 Dify Agent 里，你写一个 prompt，挂上工具，由模型决定执行顺序。你不画流程图，你描述一个目标，然后 LLM 用 [Function Calling 或 ReAct](https://docs.dify.ai/en/use-dify/build/agent) 这类策略推理出步骤——决定调用哪个工具、处理返回结果，然后再决定下一步。

在 Workflow 里，你站在一块画布上。你拖节点——LLM、Knowledge Retrieval、If/Else、Code、HTTP Request——再用线把它们连起来。根据 [Dify 的 workflow 文档](https://docs.dify.ai/en/use-dify/build/workflow-chatflow)，Workflow 从头到尾跑一次就结束，Chatflow 是在同一套节点系统之上多加了一层对话层。

这个区别在我亲自试两边之前感觉还很抽象。Agent 像是把工具箱递给某个人然后说"你看着办"。Workflow 像是写一份组装说明书。

### 记忆与上下文

这一点我没料到。**Agent 会自动保留对话历史**——最多 500 条消息或 2,000 token，哪个先到达就以哪个为准。超过上限的旧消息会被丢掉。

Workflow 没有记忆。每次跑都是完全从零开始。Chatflow 确实支持记忆，但你要在每个 LLM 节点里手动配置，而且和会话 ID、session 变量绑定。我直到自己的 workflow "忘了"两次运行之间的所有东西、花了 20 分钟以为是自己写坏了之后，才意识到这一点。

### 工具使用

Agent 动态挑工具。你把工具挂在 agent 上，由 LLM 在运行时根据用户的 query 决定用哪些。

Workflow 把工具当固定节点用。你把 "Google Search" 节点或 "HTTP Request" 节点精确摆在你想要的位置。执行流走到那个节点时，工具就会运行——不涉及模型的决策。

### 调试

……这一点我停在这里想了一会儿。

Dify Workflow 有 [step-run 功能](https://docs.dify.ai/en/guides/workflow/debug-and-preview/step-run)，能让你单独测试某个节点。你可以给单个节点喂测试输入，看它具体输出什么。这对生产工作来说是个大事——你能精确隔离出哪个节点行为不对。

Agent 给不了你这种能力。你能在对话日志里看到模型的工具调用，但要追踪它**为什么**选这条路径要难得多。推理发生在模型内部，日志记的是发生了什么，未必能解释为什么。

### 可靠性

这个权衡我现在也不敢说完全理解了，但我注意到的是：**同样的输入，workflow 走出来的输出路径是同一条。** 如果你的 If/Else 分支把账单问题路由到左、技术问题路由到右，每一次都会这么走。

Agent 可能会给你惊喜。模型可能用不同顺序调用工具、或者干脆跳过其中一个，看它怎么解读这次的 query。这是灵活性的代价。

## 什么时候用 Dify Agent

### 任务需要推理或者开放性的工具使用

如果用户什么都可能问——"分析这份数据"、"先搜 X 再总结"、"对比这两个东西"——而你没法提前预测步骤，那 Agent 是合理选择。

数据分析助手就是 Dify 自己的例子：agent 自主获取数据、生成图表、总结结论。你只描述 persona，挂上工具。

### 用户的交互会改变路径

那种"下一步取决于用户刚才说了什么"的多轮对话——这是 Agent 的地盘。内置记忆意味着 agent 还记得你三条消息前讨论过的东西。想在 Workflow 里复刻这件事，意味着手动管理 session 变量，复杂度会很快上去。

## 什么时候用 Dify Workflow

### 过程的步骤是稳定的

报告生成、工单分类、文档处理、邮件路由。如果在你动手之前，就能在白板上用方框和箭头把流程画出来——它就是一个 workflow。

我把工单分发器搭成了 Chatflow：用户消息进来 → LLM 分类意图 → If/Else 路由到不同的回复模板 → Answer 节点输出结果。每一步都看得见。每一处分支都可测。如果分类错了，我知道具体要修哪个节点。

### 可预测性比灵活性更重要

需要**一致、可审计行为**的生产系统，应该放在 workflow 里。IBM 把 [agentic workflows](https://www.ibm.com/think/topics/agentic-workflows) 定义为"agent 在最少人工干预下做决策的 AI 驱动流程"——但实际上很多生产团队想要的恰好相反：**对决策路径保持最大程度的人为控制**，AI 只负责每一步内部的重活。

Workflow 给你的就是这个。你控制路径，LLM 负责每个节点里的生成。

## 混合模式：workflow 里嵌 agent、agent 外面套 workflow

……这里事情开始变得有意思。

自从 Dify 引入了 [Agent Node 作为 workflow 组件](https://dify.ai/blog/dify-agent-node-introduction-when-workflows-learn-autonomous-reasoning) 之后，你可以把 agent 的推理能力**嵌进 workflow 内部**。Workflow 负责整体结构——分类、路由、条件分支——而 Agent Node 在某些特定步骤接管、负责需要自主推理的部分。

可以这样想：你的 workflow 是高速公路，Agent Node 是一个匝道，LLM 可以下去自由探索，然后回到主路。

Agent Node 支持可插拔策略——Function Calling 和 ReAct 是内置的，你也可以从 Dify Marketplace 装其他的。这意味着你可以拥有一个严格受控的 pipeline，其中某一步说"这部分你自己想办法"。

我试过一个简单版本：Start → Question Classifier（LLM 节点）→ 如果是技术问题 → 带有文档工具的 Agent Node → Answer。Classifier 做确定性的路由，agent 负责开放性的部分。对我的用例来说，比单独用任何一种模式都好用。

反过来也成立——你可以把一个 Workflow 发布成 [MCP server](https://docs.dify.ai/en/use-dify/publish/publish-mcp)，让外部 agent 把它当工具调用。Workflow 就成了更大的 agentic 系统里的一个可靠子例程。

## Dify 项目里常见的错误

### 用 agent 处理确定性任务

如果你的流程是"接收输入 → 分类 → 路由 → 响应"，而且步骤从不变化，就别用 agent。你会得到不一致的路由、更难调试的过程、更高的 token 成本。一个 workflow 用四个节点就能搞定，零模糊。

我早期就犯过这个错。我把一个 FAQ 机器人搭成了 agent，因为感觉这样设置更简单。一开始确实是——直到同样的问题开始给出不一样的答案，而我除了翻聊天日志之外没办法搞清楚为什么。

### 用 workflow 处理模糊任务

反过来的错误。如果用户的问题可能需要两个工具、也可能需要五个工具，你又预测不出来，那为每种可能性都搭一条 workflow 分支会变成维护噩梦。这种时候就该让 agent（或者 workflow 里的 Agent Node）登场了。

我开始看到的一种模式：**workflow 当骨架，agent 做关节。** 结构在需要刚性的地方保持刚性。灵活性在重要的地方才出现。

## FAQ

### Dify 里 Agent 和 Workflow 有什么区别？

Agent 用 LLM 的推理能力来决定调用哪些工具、按什么顺序调用。Workflow 在可视化画布上定义一套固定的步骤顺序。Agent 灵活但可预测性差。Workflow 可预测，但每一条路径都要你自己设计。

### 什么时候用 Dify Agent，什么时候用 Dify Workflow？

开放性任务用 Agent——研究助手、数据分析、对话机器人。稳定、可重复的过程用 Workflow——工单路由、报告生成、批量处理。

### Agent 和 Workflow 在记忆和上下文上有何不同？

Agent 自动保留对话历史（最多 500 条消息）。Workflow 在两次运行之间没有记忆。Chatflow 通过 session 变量支持可配置的记忆。

### 我能在同一个项目里把 Dify Agent 和 Workflow 结合起来用吗？

可以。Agent Node 把自主推理嵌进 Workflow 内部。你也可以把 Workflow 发布成工具，供 agent 调用。

把两种模式都玩了一遍之后，我一直在想这件事：真正难的不是选 Agent 还是 Workflow，而是搞清楚**你的系统里哪些部分需要推理、哪些部分需要轨道。** 这个 agent-vs-workflow 的问题，会在 AI 系统设计的每一层反复出现——而且越来越多的答案都是：两者都要，仔细组合。如果你在搭那种需要**跨任务保留学到的东西**——而不仅是单次对话内部——的 agent，那是 [另一类问题](https://evomap.ai/blog)，值得花点心思去关注。

……我会继续观察这块怎么演化。

## 往期文章

- 如果你还在纠结 agent 应该放在系统架构的哪一层，[What Are Agentic Workflows? Types, Examples, and Production Reality](https://evomap.ai/blog/what-are-agentic-workflows-types-examples-production-reality?utm_source=chatgpt.com) 把讨论扩展到 Dify 之外，拆解了在真实部署里用到的主要 workflow 模式。
- "workflow 当结构、agent 做灵活性"这种混合模式在看过 [MCP CLI, GEP, and the Three Layers of the Agent Stack](https://evomap.ai/blog/mcp-cli-gep-three-layers-agent-stack?utm_source=chatgpt.com) 之后会更好理解，这篇文章梳理了编排、工具和可复用能力是如何拼在一起的。
- 想了解工具变成标准化接口之后 agent 的行为怎么变化，[What Is MCP? The AI Tool Connection Standard Explained](https://evomap.ai/blog/what-is-mcp-ai-tool-connection-standard?utm_source=chatgpt.com) 解释了 workflow 和 agent 系统底下越来越常用的那个协议。
- 确定性流水线 vs 灵活 agent 之间的调试权衡，也连到 [Agent Hooks: How AI Execution Chains Actually Work](https://evomap.ai/blog/agent-hooks-ai-execution-chains?utm_source=chatgpt.com)，尤其是你在设计多步执行流的时候。
- 如果最后那一段让你开始想"agent 跑成功一次之后会怎样"，[Agent Skills vs GEP Assets: The Real Difference](https://evomap.ai/blog/agent-skills-vs-gep-assets-real-difference?utm_source=chatgpt.com) 探讨的是——记住一个成功的 workflow 和把它变得可复用，未必是同一回事。