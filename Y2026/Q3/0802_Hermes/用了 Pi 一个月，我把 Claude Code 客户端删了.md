https://mp.weixin.qq.com/s/FyFYiOPwqsRrK4RHjFgQUA

![[Y2026/Q3/0802_Hermes/assets/Image 16.webp|用了 Pi 一个月，我把 Claude Code 客户端删了 封面图]]

删掉 Claude Code 客户端的那一刻，我自己也愣了一下。

## 01Pi 的秘密：把 Agent 做“薄”

Pi 的哲学一句话：**让工具适应你的工作流，而不是让你的工作流适应工具。**

它的 README 直接写了——默认不内置子 Agent、不内置 Plan Mode、不内置大量专用工具。主动不做。

默认状态下，Pi 只给模型四个工具：`read`、`write`、`edit`、`bash`。

Claude Code 启动时要加载多少东西？oh-my-pi 有 32 个内置工具。Pi 就 4 个。这不是功能缺失，是刻意设计。

为什么重要？因为**每次模型调用，所有工具的定义、Schema、描述、参数，都变成上下文塞给模型**。塞得越少，启动越快，Token 消耗越低。我说的“发你好不到 1 秒就回”，不是什么黑魔法——它就是少塞了很多东西。

![[Y2026/Q3/0802_Hermes/assets/Image 17.webp|模型前为什么会变重]]

模型前为什么会变重

### Pi 不只是小——这四个设计改变了我的体验

**模型自由切换。** 这是我最在乎的。Pi 支持 Anthropic、OpenAI、Gemini、xAI、Groq、Cerebras、Mistral、OpenRouter，以及本地 llama.cpp 和 Ollama。同一个界面随意切。简单咨询用快模型，重度重构用强推理，敏感代码跑本地。

Claude Code 的核心优势是 Claude 模型与官方 Harness 的深度调优——但这意味着你基本接受了 Anthropic 的产品节奏和价格。对我来说，**模型选择权本身就是一个功能**。

**会话树。** Pi 的会话保留完整分支关系。你可以回到 20 轮之前的节点，从那里分叉一条新对话，原来的分支还在。做技术探索时，同时走三条路，保留所有上下文，不用复制粘贴。

**实时可见性。** TUI 底部实时显示模型、思考等级、Token 消耗、缓存命中、费用和上下文占用。Agent 在做什么、花了多少钱、上下文还剩多少，全透明。

**扩展边界清晰。** TypeScript Extensions、Skills、Packages、SDK、RPC。能力不够自己加，不想加一样用。不强塞选择给你。

### 但这些短板，Pi 的 README 自己写了

**开箱能力有限。** 四个工具能做的事，本质取决于模型通过 `bash` 自己搞定的能力。跨文件重命名？没有原生 LSP，`rg` + 手动改。类型错误？编译报错再说。对复杂项目来说，这有时候很疼。

**没有完整沙箱。** Pi 以当前用户权限运行，没有内置文件/进程/网络隔离。工具少不代表安全，只是攻击面小。README 自己说得很清楚。

**需要组装能力。** 适合有工程经验、愿意自己配置的人。如果你要安装即用的完整功能，它不是你的菜。

说白了：**Pi 是乐高积木，不是成品楼。** 有些人的需求，不需要一整栋楼。

## 02用 Pi 三周后，我遇到了一个它搞不定的问题

Rust 项目，跨文件重命名，涉及 14 个文件、3 个 crate、一堆 `pub use` 重导出。用 Pi 的路径：

1. `rg`
    
     搜旧函数名 → 50+ 个命中
    
2. 逐个文件读，区分真引用和同名变量
    
3. 手动改 14 个文件
    
4. `cargo check`
    
     → 漏了 3 个
    
5. 再改 → 再 check
    
6. 来回三轮才过
    

这时候我意识到了：**Pi 不是万能的——它压根没打算万能。**

装了 oh-my-pi。然后发现，它跟 Pi 已经是两个物种了。

### oh-my-pi 是一个“终端版可编程 IDE”

oh-my-pi 的 README 定位：**"A coding agent with the IDE wired in."** 没吹牛。

工具列表让人眼花：32 个内置工具、14 类 LSP 操作、28 类 DAP 调试操作，约 5.5 万行 Rust 核心。从代码量看，它已经不是 Pi 的增强版，而是独立的大平台。

但数字不重要。重要的是它解决了什么真问题。

### LSP：让 Agent 真正“看懂”代码

同一个 Rust 重命名的例子。在 oh-my-pi 里：

1. 调 `lsp rename`
    
2. Language Server 返回 WorkspaceEdit，覆盖所有引用
    
3. 一次性应用
    

**不是快了——是少了三轮返工。**

![[Y2026/Q3/0802_Hermes/assets/Image 18.webp|跨文件重命名 Pi vs oh-my-pi]]

跨文件重命名 Pi vs oh-my-pi

oh-my-pi 的 LSP 不是“跑一下编译器”的包装，而是完整 LSP Client 层：自动检测 Server、管理进程生命周期、JSON-RPC 通信、diagnostics 缓存、结果去重截断、超时取消、多 Server 路由。

对比主流 Agent 的 LSP 集成：

- **Claude Code**
    
    ：2.0.74 加了 LSP 工具（Go to Definition/Find References/Hover），但 Issue 区大量用户反馈“看不到启动状态”“配置不生效”“No LSP server available”。在快速成熟中，还不完整。
    
- **Codex CLI**
    
    ：原生 LSP Issue 开放状态，高热度功能请求。官方态度谨慎——觉得 LSP 是给 IDE 设计的，不一定适合 Agent。
    
- **oh-my-pi**
    
    ：14 类可调用 LSP 操作，原生集成，跟编辑流程深度绑定。目前三者里最完整的。
    

关键洞察：**需要精确符号级操作时，Harness 的工具能力比模型能力更重要。** 模型可能知道函数的所有调用位置，但用 grep + 肉眼找，错误率远超 LSP 一次性语义查询。

### DAP：原生调试，别再用 print 大法了

多数 AI Agent 修 Bug 的方式：

1. 读日志
    
2. 加 print
    
3. 再跑一遍
    
4. 看输出
    

oh-my-pi 多了一条路：**原生 DAP 调试**。launch、attach、设断点、单步、查线程、看调用栈、读变量、读写内存——模型可以直接调。

对 C/C++ 崩溃、Rust Native 问题、Go 死锁、多线程竞态这种“日志看不出问题”的场景，是降维打击。

### Hashline：弱模型也能当好程序员的秘密

这是 oh-my-pi 最被低估的东西。

常规 Agent 编辑文件：模型输出 patch → 找匹配位置 → 应用 → 失败 → 重试。常见失败：旧文本不匹配、缩进差异、多个相同片段、文件已被改过。

![[Y2026/Q3/0802_Hermes/assets/Image 19.webp|Hashline 为什么更稳]]

Hashline 为什么更稳

oh-my-pi 的做法：给代码行附加短内容哈希，模型引用哈希锚点，不重复输出旧代码。作者基准测试声称：某些模型上输出 Token 降低 61%。

这个数字是作者自测的，不是第三方审计。但逻辑没问题：**弱模型经常不是不会写代码，而是 patch 格式失败了。** 降低编辑的机械难度，小模型也能干活。

### 还有三件事，让 oh-my-pi 不只是“功能多”

**多模型角色路由。** 按 `default/smol/slow/plan/commit` 自动分配。快模型搜代码，强模型改架构，便宜模型写 changelog。比全程用最贵的模型合理。

**子 Agent + Advisor。** 并行子 Agent 加速多模块研究，Advisor 模型逐轮审查。Token 消耗增加，但高风险变更值这个价。

**统一工具表面。**`pr://`、`issue://`、`agent://`、`skill://` 都当路径，用同样的 `read`/`search`/`write`。模型不用为每个服务学不同参数 Schema。

### oh-my-pi 的代价

功能多不是免费的。

**复杂度。** 代码量和依赖远超 Pi，是大型 Agent 平台而非小型 Fork。功能耦合、版本回归、配置漂移——复杂系统的通病全有。

**资源消耗。** 可能同时启动 Language Server、Debug Adapter、Browser、Python/JS Kernel、子 Agent、Advisor、Memory Backend。内存和 CPU 明显高于原版 Pi。

**发布太猛。** 版本更新极快。好处是 Bug 修得快，坏处是配置可能不兼容、文档跟不上。

**安全面大。** SSH、Browser、GitHub、Debugger、进程内存、远程协作——能力越强，越需要沙箱和审批。

## 03一张图说清：四种 Agent 各自的位置

![[Y2026/Q3/0802_Hermes/assets/Image 20.webp|四种 Agent 各自适合什么任务]]

四种 Agent 各自适合什么任务

**Pi** 的护城河是“轻”。单文件修改、快速问答、脚本编写、自己配置扩展——最省 Token、最快、最自由。

**oh-my-pi** 的护城河是“全”。跨文件重构、类型修复、复杂调试、大型项目导航——更完整的工具链换更少的返工轮次。

**Claude Code** 的护城河是“集成”和“模型调优”。Claude 在自家 Harness 上表现最好，插件生态成熟。

**Codex CLI** 的护城河是“模型能力”。GPT 系编码和推理很强，但 Harness 偏薄，LSP 还在排队。

**核心判断不是谁更好——是你的任务属于哪种。**

## 04我现在两个都要

![[Y2026/Q3/0802_Hermes/assets/Image 21.webp|先看任务再选 Agent]]

先看任务再选 Agent

Pi 和 oh-my-pi 配置目录不同，完全可以并存。

我的日常：

- **快速问答、小改文件、脚本**
    
     → Pi（Gemini Flash 或 Groq，秒回，几乎不花钱）
    
- **跨文件重构、类型修复、代码审查**
    
     → oh-my-pi（Claude Sonnet，LSP + Hashline 少返工）
    
- **复杂调试、Native 崩溃**
    
     → oh-my-pi（DAP，告别 print 大法）
    

不是选一个最好的工具。是**按任务匹配工具能力**。

如果你现在只用 Claude Code，别急着删——先装一个 Pi，切到 Gemini Flash，发一句“你好”。亲眼看到“1 秒回”是什么体验，再决定。

删不删不重要。重要的是，**别让工具替你决定你能做什么。**

### 相关项目开源地址

- **Pi**
    
    https://github.com/earendil-works/pi
    
- **oh-my-pi**
    
    https://github.com/can1357/oh-my-pi
    

如果这篇内容对你有帮助，欢迎关注这个公众号，后续会更新更多精彩内容。