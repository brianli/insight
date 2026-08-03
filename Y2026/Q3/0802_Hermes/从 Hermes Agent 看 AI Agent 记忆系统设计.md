https://mp.weixin.qq.com/s/0veGSPLVGzB8AroK4q6mGg

![[Y2026/Q3/0802_Hermes/assets/Image.webp|Image]]

## 1. Agent 到底应该怎么“记住事情”

讨论 Agent 记忆时，最容易陷入一个误区：以为记得越多越好。但 Hermes Agent 展示的是另一种思路：记忆首先是一个工程取舍问题。

一个长期运行的 AI Agent，应该如何在“记住更多”和“保持上下文可控”之间取平衡？

好的 Agent 记忆系统不是“什么都塞进 prompt”，而是把信息按热度、稳定性、结构化程度和读取成本分层。

Hermes 可以看成 4 套记忆系统、5 个讲解层次。4 套系统是内置记忆、会话记忆、程序记忆和外部记忆；展开讲解时，session 压缩单独作为第三层来看：

- 内置热记忆：MEMORY.md / USER.md，小而精，常驻 prompt。
- 会话记忆：state.db / session_search，完整历史会话按需召回。
- session 压缩：context compression / session lineage，长会话过长后保留连续性。
- 程序记忆：Skills，保存“遇到这类任务怎么做”。
- 外部记忆：Honcho 等 external provider，提供深层用户建模和语义记忆。

这套分层背后的主线是：保持系统提示词稳定，充分利用 prompt cache；大体量、低频、历史性的信息通过工具按需取回。

下面重点看 Hermes 如何把“记忆”拆成可缓存、可检索、可审计、可扩展的几类工程组件。

## 2. 总体架构与 Prompt Runtime

可以把 Hermes 的 memory/runtime 看成“五层记忆 + prompt runtime”：

![[Y2026/Q3/0802_Hermes/assets/Image 1.webp|Image]]

这张图里最关键的分界是：

- MEMORY.md / USER.md 是 always-on，但容量小。
- state.db 是完整历史，但不自动进 prompt。
- compression / lineage 是长会话连续性层：把当前会话压短，同时保留逻辑会话关系。
- skills 是过程知识，默认只给索引，具体内容需要 skill_view 加载。
- external provider 是增强层，不应该替代基础 memory 和 session archive。

从 prompt 组装顺序看，Hermes 的关键不是“把所有记忆都拼进去”，而是先构造稳定前缀，再把动态内容放到后面或工具返回里：

![[Y2026/Q3/0802_Hermes/assets/Image 2.webp|Image]]

Prompt 构建：为什么 Hermes 强调稳定前缀

Hermes 的系统提示词不是每轮临时拼出来的散装字符串，而是分层组装，并尽量保持前缀稳定。

![[Y2026/Q3/0802_Hermes/assets/Image 3.webp|Image]]

源码里的注释明确说明：系统提示词会在 session 内缓存，只有 context compression 等事件才会重建。一个细节是时间也被处理成 date-only：

  

```
Conversation started: Tuesday, May 19, 2026Session ID: ...Model: ...Provider: ...
```

不放分钟级时间，是为了避免每次重建 prompt 时因为时间字符串变化导致缓存失效。

Prompt cache 的基本原理

Prompt cache 可以粗略理解为：模型服务商把一段已经处理过的 prompt 前缀缓存下来。下一次请求如果前缀 token 序列足够一致，就可以复用前缀对应的计算结果，而不是从头重新处理。

![[Y2026/Q3/0802_Hermes/assets/Image 4.webp|Image]]

从 Agent 视角看，缓存命中通常带来两个收益：

- 更低延迟：长系统提示词、工具说明、skills index 不必每轮完整重算。
- 更低成本：部分服务商会对 cache read tokens 采用更低计费，或至少减少实际计算压力。

它不是“语义相似就能命中”，而更接近“前缀内容一致才有机会命中”。

提高 prompt cache 命中的技巧

Hermes 的实现可以抽象成几条通用技巧：

|   |   |
|---|---|
|做法|目的|
|稳定内容放前面|身份、工具规则、skills index 尽量保持字节级稳定|
|动态内容工具化|搜索结果、日志、历史回忆不要改写 system prompt|
|冻结 memory 快照|会话中途写盘，不立即刷新当前 prompt|
|控制时间粒度|放日期，不放分钟秒级动态时间|
|确定性排序|skills、tools、context files 顺序稳定|
|大历史留在检索层|历史会话进|

这里的关键不是某个供应商的 cache API，而是架构原则：越稳定、越高频、越规则化的内容越应该靠前；越动态、越低频、越大体量的内容越应该工具化。

## 3. 第一层：MEMORY.md和USER.md热记忆

## 这一层最适合用“会话启动快照”来理解：文件在磁盘上是 live 的，但进入 prompt 的是 session 开始时的 frozen snapshot。

## ![[Y2026/Q3/0802_Hermes/assets/Image 5.webp|Image]]

存储位置和容量

Hermes 的内置记忆存在：

  

```
~/.hermes/memories/MEMORY.md~/.hermes/memories/USER.md
```

默认容量：

|   |   |   |
|---|---|---|
|文件|作用|默认限制|
|MEMORY.md|Agent 的环境事实、项目约定、工具经验、长期可复用事实|2200 字符|
|USER.md|用户画像、沟通偏好、工作方式、明确纠正|1375 字符|

注意这里限制的是 字符数，不是 token 数。这样实现更简单，也和模型无关。

文件格式

每条记忆用 § 分隔：

  

```
用户偏好：回答要直接，少写寒暄。§当前机器是 macOS，默认 shell 是 zsh。§某项目运行测试要用 make test，不要直接 pytest。
```

启动时 Hermes 会读取文件并渲染成系统提示词块，类似：

  

```
══════════════════════════════════════════════MEMORY (your personal notes) [67% — 1,474/2,200 chars]══════════════════════════════════════════════...
```

两类文件分别存什么

MEMORY.md 和 USER.md 都会进入系统提示词，但语义不同：

|   |   |   |
|---|---|---|
|文件|存储内容|作用|
|MEMORY.md|Agent 对环境、项目、工具和流程的稳定认知|让 Agent 下次回到同一工作环境时少走弯路|
|USER.md|用户画像、沟通偏好、协作方式和明确纠正|让 Agent 记住应该如何和这个用户协作|

简单例子：

  

```
MEMORY.md当前工作区 ~/project 主要用于 项目规范文档和技术方案沉淀。§项目规范文档应保持表达清晰，避免混入一次性上下文和临时备注。USER.md用户偏好中文输出，回答要直接、结构紧凑，少写背景铺垫。§用户在流程治理类任务中偏好“先判断，再给最小修改方案”。
```

这里不需要人工给每条信息做复杂分类。Hermes 的 memory 工具 schema 会提示模型：用户偏好、身份和协作方式通常写入 user；环境事实、项目约定、工具经验通常写入 memory。

Frozen snapshot：写入立即落盘，但不立刻进入当前 prompt

这是设计重点。

Hermes 在 session 开始时读取 memory 文件，生成一个 _system_prompt_snapshot。之后本轮会话中即便调用 memory(add/replace/remove) 修改了文件：

- 文件会立即写盘。
- tool response 会看到 live state。
- 但当前 session 的系统 prompt 不会被改。
- 新记忆通常要到下一次 session，或系统 prompt 因 compression 被重建后才进入 prompt。

这个机制牺牲了一点“马上生效”的直觉，换来 prompt cache 的稳定性。

memory 工具能力

memory 工具有三个动作：

|   |   |
|---|---|
|action|作用|
|add|新增一条记忆|
|replace|用|
|remove|用|

它没有 read action。原因是 memory 本来就会在 session 开始时注入系统 prompt；如果要修改，工具响应会返回当前条目和容量。

几个工程细节值得借鉴：

- replace/remove 用短子串定位，不需要暴露内部 ID。
- 如果子串命中多条不同记录，会要求更具体，避免误删。
- 精确重复条目不会重复添加。
- 写文件使用 lock + temp file + atomic replace，避免并发读到半截文件。
- 新条目会做安全扫描，拦截 prompt injection、凭证外泄、不可见 Unicode 等风险。

## 4. 第二层：会话记忆与search-session

## `当前源码里的核心文件是` hermes_state.py，数据库使用SQLite，默认路径：

  

```
~/.hermes/state.db
```

它负责持久化 session metadata、完整 message history，以及 FTS5 检索索引。

先看数据关系：

![[Y2026/Q3/0802_Hermes/assets/Image 6.webp|Image]]

这张图里有两个重点：

- messages 是事实来源，FTS 表只是搜索索引。
- 索引维护由 SQLite trigger 自动完成，应用层不需要手动双写。

SQLite state.db：核心表结构

sessions

sessions 表表示一次会话或会话分支。主要字段：

|   |   |
|---|---|
|字段|含义|
|id|session 主键|
|source|来源平台，如|
|user_id|平台用户 ID|
|model|使用的模型|
|model_config|模型配置 JSON|
|system_prompt|本 session 的系统提示词快照|
|parent_session_id|父 session，用于 compression、branch、delegation 形成 lineage|
|started_at|会话开始/结束时间|
|end_reason|结束原因，如 compression|
|message_count|消息数|
|tool_call_count|工具调用数|
|input_tokens|token 计数|
|cache_read_tokens|prompt cache 相关计数|
|reasoning_tokens|reasoning token 计数|
|estimated_cost_usd|成本估计/实际成本|
|title|人类可读标题，带唯一索引|
|handoff_*|跨平台 handoff 状态|

这个表不只是“聊天列表”，而是运行时账本：它同时服务 resume、search、cost、compression lineage 和跨平台会话。

messages

messages 表保存每条消息。主要字段：

|   |   |
|---|---|
|字段|含义|
|id|自增主键|
|session_id|所属 session|
|role|user|
|content|文本或 JSON 编码后的结构化内容|
|tool_call_id|工具调用 ID|
|tool_calls|assistant 发起的工具调用 JSON|
|tool_name|tool message 对应的工具名|
|timestamp|写入时间|
|token_count|单条消息 token 估计|
|finish_reason|模型完成原因|
|reasoning|reasoning 相关字段|
|codex_reasoning_items|Codex runtime 兼容字段|

多模态或结构化 content 不能直接绑定进 SQLite，所以 Hermes 会用一个 sentinel 前缀把 list/dict JSON 化：

  

```
\x00json:{...}
```

读取时再 decode 回原结构。

其他表和索引

还有：

- schema_version：记录 schema 版本。
- state_meta：key-value 元数据。
- idx_sessions_source、idx_sessions_parent、idx_sessions_started、idx_messages_session 等普通索引。
- idx_sessions_title_unique：session title 唯一索引。

FTS5 表：普通全文检索 + CJK trigram 检索

Hermes 建了两个 FTS5 virtual table：

  

```
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(content);CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_trigram USING fts5(    content,    tokenize='trigram');
```

区别：

- messages_fts：默认 FTS5 tokenizer，适合英文、代码标识符、常规关键词。
- messages_fts_trigram：trigram tokenizer，主要解决中文、日文、韩文这类 CJK 搜索的问题。

FTS 表里索引的不只是 content，而是拼接了：

  

```
message.content + tool_name + tool_calls
```

这意味着可以搜到：

- 用户/助手说过的话。
- 调过哪个工具。
- 工具调用参数里出现过的关键词。

FTS 索引通过 SQLite trigger 维护：

- AFTER INSERT ON messages：插入 FTS row。
- AFTER DELETE ON messages：删除 FTS row。
- AFTER UPDATE ON messages：先删旧 row，再插新 row。

这样应用层只需要写 messages 表，索引维护由数据库触发器完成。

并发和可靠性

state.db 默认启用 WAL：

- 多读一写更适合 gateway 多平台并发场景。
- 写操作用 BEGIN IMMEDIATE，并带随机 jitter retry，降低多个 Hermes 进程争抢写锁时的 convoy effect。
- 如果文件系统不支持 WAL，比如 NFS/SMB/FUSE，降级到 journal_mode=DELETE，牺牲并发但保证功能可用。
- 每隔一定写入次数做 best-effort WAL checkpoint，避免 WAL 文件无限增长。

这说明 state.db 不是玩具实现，而是被当作多入口、多进程共享状态库来设计。

session_search 到底怎么 search

当前 session_search 是一个单工具三形态接口，模式不是显式传 mode，而是通过参数推断：

|   |   |   |
|---|---|---|
|调用方式|模式|作用|
|不传参数|browse|浏览最近 session|
|传|discovery|全文检索历史 session|
|传|scroll|围绕某条消息继续滚动读取|

什么时候会触发 SQLite FTS 查询

这里要把 写索引 和 查索引 分开看。

![[Y2026/Q3/0802_Hermes/assets/Image 7.webp|Image]]

因此 Hermes 的 SQLite FTS 不是每轮自动运行的 RAG 组件。正常同一会话内，模型直接读 active context；如果早期上下文被 compression 压掉，主路径也是依赖 compression summary 延续任务，而不是每轮从 FTS 回捞当前 lineage。

FTS 查询主要发生在模型主动调用 session_search 时，典型触发是跨会话回忆：

  

```
我们之前讨论过 X 吗？上次那个 Y 最后怎么定的？帮我找一下之前处理 Z 的会话。我最近在做什么？上次做到哪了？
```

所以这里的边界应该这样理解：

|   |   |
|---|---|
|问题|主要机制|
|当前会话如何连续|active context + compression summary，不靠|
|什么时候查 FTS|需要跨会话回忆，且模型主动调用|
|长期稳定偏好放哪|MEMORY.md|

三种模式：Browse / Discovery / Scroll

![[Y2026/Q3/0802_Hermes/assets/Image 8.webp|Image]]

Browse：无查询时列最近会话

调用：

  

```
session_search()
```

流程：

1. 1. 调 list_sessions_rich。
    
2. 2. 默认排除 source = tool 的工具型 session。
    
3. 3. 跳过当前 session lineage，避免模型搜索自己已经在上下文里的内容。
    
4. 4. 默认跳过 child/delegation session。
    
5. 5. 返回 session id、title、source、started_at、last_active、message_count、preview。
    

适合问题：

- “我最近在做什么？”
- “上次那个任务是哪一个 session？”
- “帮我找最近几次会话。”

Discovery：传 query 做全文检索

调用：

  

```
session_search(query="auth refactor", limit=3)
```

流程：

1. 1. 默认只搜 user,assistant 角色，避免 tool output 噪声。
    
2. 2. 调 db.search_messages(query, limit=50)，先扩大命中数，方便后续按 lineage 去重。
    
3. 3. 排除当前 session lineage。
    
4. 4. 按 parent_session_id 解析到 lineage root，同一条会话链只保留一个命中。
    
5. 5. 对每个命中调用 get_anchored_view：
    

- 返回命中消息附近 ±5 条窗口。
- 返回 session 开头 bookend_start 前 3 条 user/assistant 消息。
- 返回 session 结尾 bookend_end 后 3 条 user/assistant 消息。

7. 6. 返回 snippet、match_message_id、messages_before、messages_after 等定位信息。
    

这不是“把整段历史扔给模型”，而是给模型一个足够判断的三段式视图：

![[Y2026/Q3/0802_Hermes/assets/Image 9.webp|Image]]

这个设计比单纯 snippet 更好：搜索命中本身常常只是中间过程，bookends 能让模型快速判断“当时目标是什么”和“最后结论是什么”。

Scroll：沿着命中继续读

调用：

  

```
session_search(  session_id="20260519_...",  around_message_id=12345,  window=10)
```

流程：

1. 1. 校验 session_id 和 around_message_id。
    
2. 2. window 限制在 [1, 20]。
    
3. 3. 拒绝滚动当前 session lineage，因为当前上下文里已经有。
    
4. 4. 调 get_messages_around：
    

- 取 anchor 前 N 条。
- 取 anchor 自身。
- 取 anchor 后 N 条。
- 按 message id 升序返回。

6. 5. 如果 anchor 实际在 child session 中，会尝试按 lineage 自动 rebind。
    

这让 session_search 支持渐进式检索：

1. 1. 先 discovery 找到相关 session。
    
2. 2. 再 scroll 往前/往后看更多上下文。
    
3. 3. 直到 messages_before 或 messages_after 小于 window，说明到达边界。
    

FTS 查询细节

search_messages 支持 FTS5 查询语法：

- 普通关键词：docker deployment
- 精确短语："docker networking"
- 布尔：docker OR kubernetes、python NOT java
- 前缀：deploy*
- 时间排序：默认按 FTS rank；也可 newest 或 oldest

查询进入 SQLite 前会做 sanitize：

- 保留成对引号里的精确短语。
- 清理容易导致 FTS5 syntax error 的特殊字符。
- 去掉开头/结尾孤立的 AND/OR/NOT。
- 将带点、横线、下划线的词加引号，比如 my-app.config.ts，避免被 FTS 拆成多个词。

中文检索单独处理：

- 如果包含 CJK 且每个 CJK token 至少 3 个字，走 messages_fts_trigram。
- 如果是 1-2 个中文字符，trigram 不适合，降级为 LIKE。
- 短中文 OR 查询会拆成多个 token，各自做 LIKE。

这说明 Hermes 没有把搜索问题简单丢给 vector DB，而是先把 lexical search 做到了工程可用。

## 5. 第三层：context compression 和 session lineage

长对话不能无限增长。Hermes 有 context compression 机制，把中间大量消息压缩成摘要，保留头尾和当前任务必要状态。

这一层可以单独看作压缩记忆：它不负责跨会话搜索，也不是长期偏好存储，而是解决“同一个长会话怎么继续跑下去”的问题。一边让当前上下文变短，一边用 parent-child lineage 保留原始会话的可追溯关系。

在 state.db 里，compression 会影响 session 结构：

- 原 session 会以 end_reason = compression 结束。
- 新 session 作为 child 继续，parent_session_id 指向原 session。
- 搜索和浏览时会通过 lineage 还原“同一个逻辑会话”。

这就是为什么 session_search 要做 lineage dedupe：

  

```
root session  └── compression child        └── further child
```

从用户视角这是一段连续会话；从存储视角是多条 session row。

源码里还可以看到一个实现演进：旧版本中曾有 flush_memories 这类压缩前记忆冲刷机制，但当前 release note 已经标记为移除。当前更重要的是：

- compression 前 external memory provider 可以通过 on_pre_compress(messages) 接收即将被压缩的上下文。
- compression 后 system prompt 会失效并重建，内置 memory 会重新从磁盘加载。
- background review 在每轮之后异步判断是否需要写 memory 或更新 skill。

这比“压缩前强行让模型写 memory”更稳一些，因为 memory 写入不应该只绑定在压缩时机上。

## 6. 第四层：Skills 作为程序记忆

hermes把 skills 当做 procedural memory。

![[Y2026/Q3/0802_Hermes/assets/Image 10.webp|Image]]

MEMORY.md 适合保存事实：

  

```
用户喜欢回答简洁。项目 A 测试命令是 make test。
```

Skills 适合保存流程：

  

```
遇到某类需求拆解任务时，先读 request，再校验 I/O 边界，最后写 result_file。
```

Hermes 的 skills 机制有几个特点：

1. 1. 系统 prompt 里默认不塞全部 skill 内容，只塞 compact skill index。
    
2. 2. 如果任务匹配某个 skill，模型必须用 skill_view(name) 加载完整 SKILL.md。
    
3. 3. Skill 可以带 references/、templates/、scripts/ 等支持文件。
    
4. 4. 后台 review 会在任务结束后判断是否需要更新 memory 或 skill。
    
5. 5. Curator 会维护 agent-created skills，鼓励合并成 class-level umbrella skill，而不是一堆一次性小 skill。
    

所以 skills 解决的是另一个问题：

不是“我知道什么”，而是“遇到这类任务我应该怎么做”。

这对团队内部 Agent 很关键。比如：

- 代码 review 的固定口径。
- 线上问题排查流程。
- 需求拆解输出格式。
- 数据库变更评审步骤。
- 某些平台工具的正确调用方式。

这些都不应该塞进普通 memory，因为它们是流程知识，结构比一条事实复杂。

## 7. 第五层：External memory provider

Hermes 的 MemoryManager 允许内置 memory 加一个 external provider。当前设计限制：

- 内置 provider 始终存在。
- 最多注册一个非内置 external provider，避免工具 schema 膨胀和多个记忆后端冲突。

这一层解决的是前四层不擅长的问题：语义召回、用户建模、跨会话事实抽取、知识图谱和更长周期的个性化。

![[Y2026/Q3/0802_Hermes/assets/Image 11.webp|Image]]

Provider 接入点

MemoryProvider 抽象类把 external memory 拆成几类接入点：

|   |   |   |
|---|---|---|
|接入点|时机|作用|
|system_prompt_block()|prompt 组装时|注入静态说明、provider 状态和工具使用规则|
|prefetch(query)|每次模型调用前|返回和当前问题相关的外部记忆上下文|
|queue_prefetch(query)|每轮结束后|后台预取下一轮可能要用的记忆，减少同步等待|
|sync_turn(user, assistant)|每轮结束后|把本轮用户/助手内容异步写入外部后端|
|get_tool_schemas()|模型显式调用工具时|暴露 provider 自己的搜索、推理、写入工具|
|on_session_end(messages)|会话结束时|做会话级总结、事实抽取或 flush|
|on_session_switch(...)|resume / branch / reset / compression 时|让 provider 更新内部 session 绑定|
|on_pre_compress(messages)|compression 前|提前抽取即将被压缩掉的信息|
|on_memory_write(...)|内置 memory 写入时|把|

这个设计有两个工程含义：

- external provider 既可以“自动注入上下文”，也可以“只暴露工具，等模型需要时再查”。
- provider 写入通常应该异步化，否则每轮对话都会被外部网络延迟拖慢。

Honcho 例子：三种 recall 模式

以 Honcho provider 为例，它不是简单的 vector search，而是偏“AI-native 用户建模”。它会维护 session summary、user representation、peer card、AI self-representation 等结构化上下文。

Honcho 的召回模式可以理解成三类：

|   |   |   |
|---|---|---|
|模式|行为|适合场景|
|context|自动把相关用户上下文注入到模型调用前|想要无感个性化，但不希望模型显式操作记忆|
|tools|不自动注入，只暴露|想控制成本和上下文污染，让模型按需查|
|hybrid|自动注入上下文，同时开放工具|想要默认个性化，也允许复杂问题显式深查|

它的 prefetch 也分层：基础 context 可以缓存并按 cadence 刷新；更贵的 dialectic reasoning 可以作为补充层异步预热。这样做是为了避免“每一轮都同步调用外部 LLM/服务”。

External provider 适合补什么

它最适合补内置层的这些短板：

|   |   |
|---|---|
|内置层短板|external provider 可以补的能力|
|FTS 主要靠关键词|语义召回、同义改写、多跳关联|
|MEMORY.md|更大规模的用户画像和事实库|
|session archive 是原始日志|自动抽取事实、偏好、关系、主题演化|
|SQLite 本地为主|跨设备、跨平台、跨 workspace 的长期记忆|
|skills 保存流程，不保存用户模型|建模“这个用户如何决策、偏好什么、反复关注什么”|

官方文档和插件里能看到 Honcho、Mem0、Hindsight、Holographic、RetainDB、ByteRover、Supermemory 等 provider 或相关集成。它们的定位不完全一样：有的偏语义检索，有的偏知识图谱，有的偏用户画像，有的偏长期上下文管理。

但它仍然只是增强层

external provider 不应该替代基础层：

- MEMORY.md / USER.md 仍然是最高频、最稳定、最 prompt-cache 友好的热记忆。
- state.db 仍然是完整、可审计、可定位的本地会话事实来源。
- session_search 仍然负责找回真实消息窗口，而不是让模型只相信二次摘要。
- skills 仍然承载“怎么做事”的流程沉淀。

所以更稳的架构不是“接一个高级 memory provider 就完事”，而是：

  

```
本地热记忆负责稳定偏好+ 本地 session archive 负责证据回溯+ skills 负责流程复用+ external provider 负责语义和用户建模增强
```

## 8. 工程取舍与设计启发

为什么不用一个 vector DB 搞定所有记忆

很多 Agent memory 方案会直接说“把历史都 embedding，检索 top-k”。Hermes 的实现更保守：

- 常驻 memory 是小文件。
- 会话 archive 是 SQLite。
- 搜索先做 FTS5。
- 中文走 trigram 或 LIKE。
- skills 用文件系统组织。
- external provider 作为增强，而不是基础依赖。

优点：

- 可解释，能看到到底搜到了哪条 message。
- 本地优先，部署简单。
- 容易删除和审计。
- 不依赖 embedding 模型质量。
- 对代码、命令、路径、错误字符串这类精确文本更友好。

缺点：

- 语义召回弱。
- 查询词质量影响很大。
- 没有自动总结时，模型需要自己阅读窗口。
- 用户画像和长期偏好需要额外 provider 才能做深。

为什么 memory 要小

小 memory 不是能力不足，而是刻意限制。

如果常驻 memory 太大：

- 每轮请求 token 成本上升。
- prompt cache 命中价值下降。
- 过期事实更难治理。
- 模型更容易被低价值历史干扰。
- 安全风险扩大，因为 memory 会进入系统 prompt。

因此 memory 应该像缓存里的 hot set，而不是数据库。

为什么 session_search 返回真实消息

当前实现不再用 LLM 摘要，有一个明显好处：

检索结果可审计。

如果返回摘要，模型和用户都要相信摘要模型没有漏掉关键信息；如果返回窗口和 bookends，主模型可以自己判断，必要时继续 scroll。

这更适合开发者工具，因为开发者通常需要原始证据：

- 当时用户到底怎么说。
- 工具到底调了什么。
- 错误文本是什么。
- 最后结论是不是已经验证过。

为什么 skills 是 memory 的一部分

只记住事实还不够。Agent 经常失败在“流程不稳定”：

- 明明上次学过某个项目怎么测，下次又乱跑命令。
- 明明用户纠正过输出格式，下次又按通用格式。
- 明明某类问题有固定排查顺序，下次又凭感觉。

这类经验不适合写成一句 memory，而应该成为可执行、可维护、可附带脚本的 skill。

对做 Agent / Copilot / 工具的启发

记忆分层模型

可以借鉴 Hermes 的“五层记忆 + 当前任务状态”模型：

![[Y2026/Q3/0802_Hermes/assets/Image 12.webp|Image]]

|   |   |   |
|---|---|---|
|信息类型|应放位置|读取方式|
|用户偏好、稳定约定|hot memory|每次 prompt 自动带上|
|历史对话、任务过程|session archive|按需搜索|
|长会话连续性|compression memory|上下文超限时压缩、接续、重建 prompt|
|操作流程、排查路径|skills / playbooks|任务匹配时加载|
|长期用户模型、语义关系|external provider|prefetch / recall|
|当前任务状态|当前上下文或 artifact|当前任务内使用，不作为长期记忆层|

不要把“任务进度”误存成长期记忆

很多 Agent memory 会越用越脏，根因是把临时状态升格成长期事实。

更好的规则：

- memory 保存长期稳定事实。
- session_search 回忆过去任务进度。
- artifact/result file 保存当前任务交付物。
- skill 保存可复用流程。

先做可审计检索，再谈语义智能

企业内部场景里，很多查询其实是精确文本：

- 错误码。
- 接口名。
- 服务名。
- 配置 key。
- 表名。
- PRD 标题。
- 命令行参数。

这类内容 FTS5 往往比纯 embedding 更可靠。可以先做：

  

```
SQLite/Postgres + FTS-> message/window/bookends-> 用户或模型确认-> 必要时再接 semantic layer
```

prompt cache 是架构约束，不是优化细节

一旦 Agent 运行频繁，prompt cache 会影响：

- 延迟。
- 成本。
- 多轮体验。
- 是否敢放长系统规则。

因此系统 prompt 应该有稳定分层：

- 稳定 identity / policy / tool rules。
- session 级上下文。
- 小型 memory snapshot。
- 动态检索结果不要改 system prompt，放在当前 turn 的工具返回或用户消息附加上下文里。

技能库需要治理

Skills 会自然膨胀。Hermes 的 curator 思路很有参考价值：

- 不鼓励“一次任务一个 skill”。
- 更倾向 class-level umbrella skill。
- 支持 references/templates/scripts。
- 保护 bundled/hub/pinned skills。
- 可以归档，不直接硬删除。

## 9. 风险和不足

Hermes 的设计很务实，但不是没有问题。

FTS 不是语义搜索

如果用户问：

  

```
上次那个我们讨论过的“权限隔离问题”是什么？
```

但历史里实际写的是：

  

```
tenant-level access control
```

纯关键词可能搜不到。需要：

- 更好的 query rewriting。
- 同义词扩展。
- 语义 provider。
- 或者让模型先生成多个候选 query。

memory 容量小，治理要求高

小 memory 的前提是有好的写入判断。如果模型乱写：

- 重要事实会被低价值事实挤掉。
- 用户偏好可能过期。
- 修正可能被写得过于绝对。
- 一次性环境问题可能变成长期禁令。

所以 memory 写入应该有强约束和可视化维护能力。

session archive 有隐私和合规问题

state.db 保存完整会话，包括 tool calls、reasoning 相关字段、系统 prompt 快照等。

这意味着必须考虑：

- 用户是否知道完整历史会被保存。
- 如何删除某个 session。
- 如何删除包含敏感信息的 message。
- 是否需要加密。
- 团队共享环境下 session 是否隔离。
- 备份和迁移时是否会泄露。

skills 自更新可能固化错误经验

background review 会主动更新 skills，这很强，但也有风险：

- 把一次 workaround 固化成通用规则。
- 把当前环境故障写成长期限制。
- 生成过窄 skill，污染 skill index。
- 多个 skill 之间重复或冲突。

所以需要 curator、保护规则、人工 review，以及明确的“什么不能保存”。

Compression 摘要有损

压缩能降低上下文，但摘要一定有损。Hermes 用 session lineage 保存历史，并可以通过 session_search 找回原始会话，这降低了风险。

但如果当前任务依赖中间细节，而压缩摘要漏了，模型仍可能偏航。因此：

- 关键决策应写 artifact。
- 重要长期规则应写 memory/skill。
- 历史细节应能通过 session_search 找回。