https://mp.weixin.qq.com/s/qxhfFwafX9A1W6sRC_Q9Ig
该公众号持续在更新 Hermes/OpenClaw 方面的内容

v0.20.0 的代号叫 Herald（传令官）。在希腊神话里，Hermes 是众神的信使，负责传话、宣告和引路。这个名字不是随便取的：这一版 Hermes 真正学会了说话（实时语音对话），学会了跟其他 Agent 通信（A2A v1.0），学会了向外部系统主动报告事件（签名 webhook），还学会了给自己的研究结论标注可信来源（grounded citations）。

3,650 个 commit，1,400 个合并 PR，1,200 个关闭的 issue，647 位贡献者。这是自 v0.19.0 Quicksilver 以来积累的一个大窗口，release notes 是完整的策划稿，不是补丁汇总。

## 一、语音：从语音信箱到对话

这是 v0.20.0 最革命性的变化，也是用户感知最直接的。

之前的语音模式是"语音信箱"：你说一句，等整个回复生成完毕，然后听一段完整的音频文件。Hermes 不会边想边说，你也不能打断它。v0.20.0 把这个模型彻底改了。

**流式 TTS 加打断。** Hermes 现在逐句合成语音，边生成边说。你可以在它说话的任意时刻直接开口打断，它会停下来听你说，模型也被通知"用户打断了你"。底层实现是逐句流水线：下一句在当前这句播放时就开始渲染，等待感被压到最低。这个能力覆盖 CLI 语音模式、桌面端，以及所有支持音频的 gateway 平台。打断是双向的：生成过程中和播放过程中都能打断，模型都知道。

**设备端唤醒词。** 你可以自定义一个唤醒短语（比如"hey Hermes"），检测在本地运行，待机时音频不会离开你的机器。不同唤醒词可以路由到不同 profile：说一个词唤醒工作助手，说另一个唤醒家里的智能家居控制。说"stop"就能在任何平台结束语音对话，不用碰键盘。

**全平台语音。** 在 WhatsApp、飞书、钉钉、LINE、QQ、Photon、微信上发语音消息，Hermes 自动转写并回复；自动 TTS 回复按平台适配（支持 opus 的平台发 opus，字幕正确附加）。STT（语音转文字）变成了独立的一等工具类目，有 GUI 开关、dashboard 下拉菜单和统一语言解析。以前转写结果经常回来的是错误语言，这个 bug 类被系统性修掉了。OpenAI 的 gpt-transcribe 也在支持列表里。

一条容易被忽略的细节：所有 TTS provider 共享一套统一的"念白文本预处理"，把 markdown 标记、代码块、URL 从语音中清理掉。你不会听到 Hermes 念出"井号井号标题"或"反引号 print 括号"。

## 二、Agent 通信三件套

如果说语音解决的是"人和 Hermes 的通信"，那这一节的三件事解决的是"Hermes 和外部世界的通信"。它们合起来把 Hermes 从单体助手变成了可以被编排、可以向外触发、可以被信任的 agent 节点。

**A2A v1.0。** 新的 bundled 插件实现了 Agent-to-Agent 协议。Hermes 可以发现、对话、被其他 A2A 兼容的 agent 驱动。这关闭了 issue #514，仓库里最古老的 feature request 之一。如果你在搭多 agent 系统，Hermes 有了一个标准的接入协议。A2A 的意义要看生态发展，但协议层面的落地本身是一个定位信号。

**Outbound webhooks。** 之前集成 Hermes 意味着轮询或在平台上监听。现在 Hermes 会主动向你注册的 HTTP 端点推送签名生命周期事件：session 活动、turn 完成、tool 事件，带 HMAC 签名验证，你的接收端可以验真。接 CI、智能家居、dashboard 不用轮询了。Hermes 从"你问它答"变成了可以触发后续流程的事件源。

**Grounded citations。** 新的 grounded-citations skill 让 Hermes 做研究时每个论断都带可验证来源。引文不是模型编的，它会和实际页面文本做匹配，链接到确切的证据位置。还有一个 fact-checking 模式：给它任何文档或论断，它会告诉你哪些查证通过、哪些不成立、哪些无法验证。如果你用 Hermes 做调研，这是"听起来对"和"有据可查"的区别。

## 三、桌面端：从聊天客户端到平台

v0.19.1 已经能看到桌面端 IDE 化的趋势（命令面板、tab 管理、消息回应），v0.20.0 把这个方向推到了"平台"的门槛上。

**Artifacts。** 版本化的卡片，带沙箱实时预览的右侧查看器。Hermes 生成的 HTML 或应用可以在聊天旁边的安全沙箱里直接运行，不用复制到浏览器试。Artifacts 是版本化的，你可以回到之前的版本。

**Plugin SDK。** 一个真正的插件 SDK 落地了，Kanban（看板任务系统）是创始插件。插件可以用 `ctx.download` 给用户传文件，放浮动面板，开多个 GUI 窗口。Widget 应用模型也来了：把应用抽象成 state + reducer + render，附带三个参考实现和网格布局引擎。第三方可以在 Hermes 桌面端里构建真正的应用，而不只是聊天插件。

**Quick-entry 和远程后端。** 全局热键，从操作系统任意位置捕获一个想法，直接投递到任意 session。⌘O 可以把文件夹打开为项目。桌面端还支持 SSH 远程后端连接模式，从笔记本连到服务器上的 Hermes 实例，事件驱动的实时同步取代了持续轮询。

**桌面 UX 细节波。** 编辑器拿到了附件拾取器（文件/文件夹/链接）、composer undo 栈、双 ESC 丢弃草稿、双 Enter 发送队列消息。两键模型切换（⌘⇧M），⌘K 里的 YOLO 开关有实时状态。iMessage 式的 emoji 回应来了（opt-in，双向）。会话侧边栏加了日期分割线、置顶区和可选的过期会话自动归档。原生桌面登录改用系统浏览器加 PKCE（RFC 8252），不再依赖 webview cookie。

桌面端的 60fps 性能优化也到了第二波：流式渲染成本不再随对话长度增长，五个流式 tab 同时拖拽保持 60fps，后台空闲 CPU 接近零。状态诊断工具（渲染 + store 变更计数器）和一条禁止 atom-mirrored ref 的 lint 规则，让 stale-read 这类 bug 不能再回来。Playwright E2E 测试套件也落地了，带视觉回归 diff。

## 四、CLI 和工具链

**CLI 新命令。** `!command` 直接跑 shell 命令，不花模型 turn，等于在 Hermes 里开了个快捷终端。`/init` 扫描项目生成或更新 AGENTS.md（让 agent 理解项目上下文的配置文件）。`/diff` 看暂存区/全部/session 的代码变更。`/context` 分析上下文窗口里到底是什么占满了空间。`/focus` 切换低输出视图，被隐藏的内容可以恢复。Ctrl+S 可以暂存写到一半的提示词。`hermes import-agent` 一键把 Claude Code 或 Codex CLI 的配置迁移到 Hermes。

**中途纠偏（redirects）。** 如果 Hermes 跑偏了，你不用 `/stop` 再重新解释。直接在它工作时打一句修正，当前 turn 会被重定向：已经在做的工作保留，原始 prompt 保留，agent 根据新指导调整方向。配合双 ESC 丢弃草稿和编辑器 undo 栈，纠偏感觉像编辑，不是重启。

**工具自我修复。** 一组系统性的自恢复升级，让 agent 在工具摩擦上少浪费 turn。terminal 输出截断会自动溢出到可回读的文件，并告诉你截断前的总大小。patch 工具能检测"已经应用过的编辑"并直接返回成功，不匹配时用空白字符可视化帮你诊断。write_file 写完后验证磁盘上的实际内容。搜索零匹配时自动探测近似项并恢复。默认工具调用迭代上限从 90 提到 500，长时间自主任务不再撞到人造天花板。

**委派和子 agent。** delegate_task 加了结构化超时/停滞元数据和实时的 per-child 状态查看（`/agents` 命令）。子 agent 现在可以用 execute_code。子 agent 的工具历史在 `subagent_stop` 里以脱敏形式暴露，还有一套公开的子 agent 生命周期 API 给插件用。

## 五、审批和安全：让自主更可控

agent 能力越强，控制就越重要。v0.20.0 在审批和安全上做了系统性加固。

**智能审批成熟。** `hermes approvals suggest` 会挖掘你的审批历史，自动生成白名单提案：你点过的"允许"会变成下次的"自动允许"。运营者可以自定义 smart-approval 策略。连续拒绝会触发熔断器，一个行为异常的循环会被冻住。docker/podman 的 daemon-redirect 命令现在需要审批。session 级别的失控循环也有了上限（web_search 和 delegate_task 各有帽，思路来自 Claude Code）。

**出站安全。** Iron-proxy 凭据注入出站防火墙重新落地。DNS-pinned 的 SSRF 安全 fetch 和 Slack CDN 白名单防止了服务端请求伪造。配置键脱敏的正则表达式消除了 ReDoS 风险。压缩的每个文本边界都应用了严格脱敏。

**凭据管理。** 一键 token 轮换加可操作的启动错误提示。Bitwarden 可选的加密 break-glass 缓存。vault 注入的密钥按 profile home 限定作用域。凭据池（credential pool）做了 reset-aware 的主源恢复：碰到限流后留在 fallback 上直到限流窗口重置，而不是立刻切回去再失败。`${env:VAR}` SecretRef 在 config.yaml 和 MCP config 之间做了对齐。

**Windows 硬化。** text-mode subprocess 的解码 bug 类被全仓库关闭（这个 bug 在 Linux 上永远不会暴露，只在 Windows 生产环境触发）。控制台闪烁在所有 daemon、环境探测、LSP、安装器路径上被隐藏。残余的编码缺口（MCP stdio、gateway 更新 I/O、STT/TTS、桌面 spawn）也做了修补。

**状态和数据库完整性。** 四个 session-state 修复（安全关闭追踪、flush-cursor 类修复、行重试、usage-PK 治愈）。FTS（全文搜索）布局升级到 v23，加了 `hermes sessions optimize` 和 CJK 双字分词：中文搜索终于不会再漏匹配了。读路径做了 per-thread 只读连接分离。

## 六、压缩和性能：长对话不卡，冷启动不慢

**压缩智能化。** 上下文压缩做了一次深度重构。大窗口模型主动修剪 tool-result（而不是被动等到撑满才压缩）。逐 turn 微压缩把成本分摊，不再是一次性大停顿。最近 N 条用户消息保证存活（`compression.min_tail_user_messages` 可配），不会出现"刚说的话被压缩没了"。进度感知超时不再惩罚慢速摘要模型。ghost-skill 防御确保被修剪的 skill 不会在 session 里留幽灵标记。阈值可以按模型和绝对 token 数配置。严格脱敏应用在每次压缩的文本边界。

**冷启动和热路径。** `hermes -w` 冷启动从约 14 秒降到约 1.8 秒。`hermes update` 空操作快了 2-6 秒。重的 SDK 改成懒加载，不阻塞启动路径。config 读取停止深拷贝（telemetry gate 快了 54 倍），29 个调用点用只读 config loader（读取成本低 28 倍）。流式热循环里去掉了 per-chunk 的 repr() 调用，token 统计便宜了约 3 倍。turn flush 批进一个 SQLite 事务。Anthropic 原生 API 的 prompt 缓存现在覆盖 tool schema 且不丢历史。DeepSeek 在 OpenCode gateway 上也加了 prompt 缓存。

**MCP 懒启动。** 配置的 MCP 服务器不再在 session 启动时全部加载。一个新的指纹键控磁盘缓存记录每个服务器的 tool schema，只在需要时才启动对应的服务器进程。这对配置了很多 MCP 服务器的人来说是体感质变：启动快了，内存省了。

**Session 架构。** 19 个 session 级的独立 dict 被合并成一个 turn/conversation/persistent-scoped 对象（SessionState）。TurnContext 和 TurnRunner 做了接缝提取。CommandDef 上声明式的 busy_policy。这些重构对用户不可见，但它们是后续功能迭代的基座。

## 七、生态扩展

**新平台。** Buzz（Block 的 Nostr messenger）作为 bundled gateway 落地，原生 WebSocket 传输加 NIP-42 认证。Vercel AI Gateway provider 和 Vercel Sandbox terminal backend 做了现代化改造后重新上线。Relay 完成了四阶段 parity：媒体、交互提示、线程生命周期、打字指示器。HSP（Hermes Sync Protocol）的个人和组织级 skill 同步落地，token 门控的命名空间发现。

**已连平台的增强。** Slack 原生 Block Kit clarify 按钮和可选的 emoji 触发。Discord 自动线程 session 按 prospective_thread_id 做键，reply 引用直接从 ID 构建。WhatsApp 的入站已读回执可配。Photon 拿到原生投票、效果和富链接。Kanban 唤醒现在会恢复创建者的 DM/线程 session，per-task 可以指定模型和思考深度。

**新模型。** Gemini 3.1 Pro 和 3.6 Flash 进目录（3.6-flash 成为 aux 默认），claude-opus-5 在 OpenRouter 和 Nous Portal 上线，deepseek-v4-flash-0731 可用。Bedrock Converse API 加了 prompt 缓存。模型选择器做了策展默认值和可折叠 provider 列表。

**运行时。** Node 26 成为硬性要求，brew 和 pip/PyPI wheel 渠道退役。官方安装渠道是 shell 安装器、Docker 和 Nix。托管 Node/uv 会在 bare PATH 之前解析，过期的托管版本会自动治愈到目标主版本。

## 个人结论

v0.20.0 是 Hermes 从"打字的助手"变成"会说话、能通信、可编排的 agent 平台"的转折点。语音对话是用户感知最强的变化：Hermes 终于可以像人一样被打断、被唤醒、隔着房间对话。A2A、webhook 和 grounded citations 一起，把 Hermes 推进了多 agent 系统和事件驱动工作流的基础设施层。桌面端跨过了"平台"的门槛，CLI 拿到了 power user 等待已久的命令套件，审批和安全系统跟上了 agent 的能力增长，压缩和冷启动终于不再让人等。3,650 个 commit 的窗口，Herald 这个代号，名副其实。