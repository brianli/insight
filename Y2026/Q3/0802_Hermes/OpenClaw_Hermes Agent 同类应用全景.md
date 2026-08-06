---
title: OpenClaw_Hermes Agent 同类应用全景（中国大陆）
type: research
created: 2026-08-06
updated: 2026-08-06
research_method: gh CLI + 官方产品页/文档
status: active
scope: 中国大陆 AI Agent 产品与应用
as_of: 2026-08-06
tags:
  - AI-Agent
  - OpenClaw
  - Hermes-Agent
  - 中国AI产品
  - 产品研究
---

# OpenClaw / Hermes Agent 同类应用全景

> [!warning]
> 本文不是“国内 AI 应用大全”。只纳入能让模型**调用工具、操作浏览器/桌面/手机、执行多步任务，或让用户持续运行/编排 Agent** 的产品。纯聊天、纯知识库、纯写作、只面向代码仓库的 Coding Agent，不算 OpenClaw / Hermes Agent 的直接同类。
>
> **资料截止：2026 年 8 月 6 日。** 产品迭代很快；“独有特性”指在本次纳入样本中由官方资料明确支持、且不是所有同类都已普遍具备的差异，不是永久的行业唯一。

## 先给结论

> [!warning]
> 上一版范围过宽，把 Agent 平台、手机 GUI Agent、Coding Agent、App Builder 和垂直 Skills 都混进来了。本文现在只讨论四类：**OpenClaw/Hermes/CoPaw/LobsterAI 这种个人 Agent、桌面 Agent、可自托管 Agent runtime，以及它们的直接开源替代/衍生项目**。
>
> 产品资料与 GitHub 状态截至 **2026 年 8 月 6 日**。GitHub stars 会变化，只作为生态热度信号，不作为产品质量证明。

真正值得放在同一张桌子上的，不是“所有带 Agent 字样的产品”，而是以下这些：

- **基准产品**：OpenClaw、Hermes Agent、QwenPaw（原 CoPaw）、LobsterAI。
- **国内/中文生态直接同类**：CowAgent、AstrBot、ZLAgent。
- **腾讯产品线**：QClaw、WorkBuddy；前者偏个人微信远控，后者偏职场/团队多模态工作台。
- **字节跳动产品/项目**：DeerFlow 2.0、火山引擎 ArkClaw、Agent TARS / UI-TARS Desktop；分别偏长任务 harness、云端 7×24 Agent、GUI/浏览器/电脑操作。
- **其他直接同类云端产品**：Kimi Claw、MiniMax MaxClaw。它们都属于 OpenClaw 的产品化部署，而不是普通聊天或 Agent 开发平台。
- **轻量开源替代/衍生项目**：nanobot、nano-claw、mini-claw、FlashClaw。
- 其他 Agent 平台、手机 GUI Agent、Coding Agent、App Builder、垂直 Skills 和纯控制面项目全部删除，不再占据本文篇幅。

### 1.1 比较基准

本文将目标产品定义为 **Personal Agent Runtime / Desktop Agent**，至少满足以下五项中的四项：

1. **能持续运行**：有 Gateway、后台服务、会话服务、定时任务或消息渠道，而非一次性脚本。
2. **能动手执行**：至少能操作文件、终端、浏览器、网络/API或本地应用中的两类工具。
3. **有持久状态**：具备跨会话记忆、工作区、技能、配置或任务历史。
4. **用户拥有部署/控制权**：能在个人电脑、服务器、NAS、容器或云主机运行，支持 BYOK 或本地模型。
5. **可扩展**：支持 Skills、Plugins、MCP、Agent 多实例、模型切换或自定义工具。

单纯“能创建 Agent”的 SaaS 平台、只做手机 GUI 操作的模型、只做代码仓库的 Coding Agent，不列为直接同类。

### 1.2 GitHub 定向扫描

本轮使用 `gh` CLI 做了三类扫描：

- 直接搜索 `OpenClaw`、`Hermes Agent`、`CoPaw`、`QwenPaw`、`LobsterAI`、`nanobot`、`personal AI assistant`、`desktop AI agent`、`self-hosted personal agent`、`QClaw`、`WorkBuddy`、`ArkClaw`、`Kimi Claw`、`MaxClaw` 等关键词；
- 检查腾讯、字节跳动、YuanbaoTeam 等官方/关联 GitHub 组织的仓库、README、fork/source 字段、更新时间和产品链接；
- 按“独立 runtime / 桌面成品 / 云端产品化 / GUI 执行栈 / 渠道或能力插件 / 噪声”分层，而不是按 stars 粗暴排名。

**重要校正**：`agentscope-ai/CoPaw` 已经演进/改名为 **QwenPaw**，当前 GitHub 主仓库是 `agentscope-ai/QwenPaw`。文中统一写作 **QwenPaw（原 CoPaw）**，避免把同一项目算成两个产品。

**腾讯归属校正**：

- `Tencent/openclaw` 是 `openclaw/openclaw` 的 fork，不是腾讯自研 runtime；
- `Tencent/openclaw-weixin` 是腾讯维护的 OpenClaw 微信 Channel Plugin；
- `Tencent/yuanbao-openclaw-plugin` 是将腾讯元宝机器人接入 OpenClaw 的官方渠道插件；
- `qiuzhi2046/Qclaw` 是个人账号下的开源桌面管家，README 明确写明“基于 OpenClaw”，且已暂停更新；它不是腾讯官方 QClaw 源码仓库；
- **腾讯 QClaw 官方产品确实存在**，官方入口是 `qclaw.qq.com`。之前只看 GitHub，错把“没有官方源码仓库”误读成“没有官方产品”。

## 2. 直接同类总表

| 产品/项目 | 团队/来源 | 层级 | 当前状态（2026-08-06） | 核心特性 | 与 OpenClaw/Hermes 的关系 |
|---|---|---|---|---|---|
| [[OpenClaw]] | OpenClaw 社区 | 基准 runtime | 活跃开源项目 | Gateway、频道、工具、Skills、Plugins、节点、记忆、后台运行 | 目标参照：个人设备上的通用 Agent |
| [[Hermes Agent]] | Nous Research | 基准 runtime | 活跃开源项目 | 自我进化 Skills、分层记忆、会话检索、cron、子 Agent、多终端后端 | 目标参照：强调学习闭环和长期成长 |
| [[QwenPaw（原 CoPaw）]] | AgentScope / Qwen 生态 | 国内基准 runtime | 活跃开源项目；QwenPaw 2.x | 本地/云部署、Agent OS、ReMe 三层记忆、安全门、MCP/A2A/ACP、Skills、插件、桌面与多渠道 | 国内大厂开源、最完整的直接同类之一 |
| [[LobsterAI]] | 网易有道 | 桌面成品 + runtime 组合 | 活跃开源项目/桌面产品 | Cowork、真实桌面文件/终端/浏览器、Office/视频/研究、Skills、MCP、定时任务、IM 远程控制 | 基于 OpenClaw；把 runtime 包进桌面成品 |
| [[QClaw]] | 腾讯电脑管家 | 桌面成品 + OpenClaw 产品化 | 商业化产品，macOS/Windows | 本地文件、浏览器、邮件、Skills、记忆、微信远程控制、内置模型、一键关联 OpenClaw | 腾讯面向个人用户的微信入口型“小龙虾” |
| [[WorkBuddy]] | 腾讯云 CodeBuddy | 桌面 Agent 工作台 | 商业化产品，面向个人/团队/企业 | 多模态办公、任务规划、结果交付、多 Agents、Skills、专家团、腾讯生态连接 | 腾讯面向职场/团队的产品化 Agent，公开资料强调办公工作台，不把其底层是否完全等同 OpenClaw 写死 |
| [[CowAgent]] | zhayujie / 中文开源社区 | 国内直接同类 | 活跃开源项目 | 规划、工具、Skills、三层记忆、Deep Dream、自进化、MCP、多渠道、桌面客户端 | 目标上直接对齐 OpenClaw/Hermes |
| [[AstrBot]] | AstrBotDevs | 国内直接同类 | 活跃开源项目 | IM 优先、Agent、MCP、Skills、知识库、插件市场、沙箱、WebUI、桌面部署 | 更偏 IM/机器人入口，但具备 OpenClaw alternative 级 runtime 能力 |
| [[ZLAgent]] | 中文开源社区 | IM-first Agent runtime | 活跃但早期 | 微信/企微/Webhook、长期记忆、Skills、GraphRAG、MCP、任务检查点、审批与恢复 | 小而明确的 Hermes 记忆 + OpenClaw 工具路线 |
| [[nanobot]] | HKUDS | 轻量 runtime | 活跃开源项目 | WebUI、Gateway、工具、记忆、MCP、多 Agent、自动化、后台运行、多渠道 | OpenClaw 的轻量化/易读替代 |
| [[nano-claw]] | hustcc | 轻量 runtime | 开源项目 | 约 4,500 行 TypeScript、持久记忆、Skills、子 Agent、cron、heartbeat、多渠道 | nanobot 的 TS 实现，受 OpenClaw 启发 |
| [[mini-claw]] | 中文开源社区 | 轻量 runtime | 早期/低活跃 | 多渠道、文件与 Shell 工具、定时任务、会话持久化、Skills、多模型 | 直接自称“迷你的 OpenClaw、CoPaw” |
| [[FlashClaw]] | 中文开源社区 | 轻量 runtime | 早期/低活跃 | 插件化、热加载、飞书/Telegram、长期记忆、定时任务、后台服务 | 中国本土轻量“龙虾”实现 |
| [[DeerFlow 2.0]] | 字节跳动 / Volcengine 开源生态 | 长任务 SuperAgent harness | 活跃开源项目 | 沙箱、记忆、Skills、子 Agent、Gateway/IM、长时任务、Web UI、本地/云部署 | 与 Hermes/OpenClaw 同层的长任务 Agent harness；更偏研究、代码、创作与多 Agent 编排 |
| [[ArkClaw]] | 火山引擎 / 字节跳动生态 | 云端 Agent 产品 | 商业化云服务 | 7×24 在线、云端持久存储、调度 + 执行双引擎、多个智能伙伴、飞书/腾讯生态协同 | 更像云端产品化的 OpenClaw + Hermes 组合，而不是本地 runtime |
| [[Agent TARS / UI-TARS Desktop]] | 字节跳动 Seed | GUI/浏览器/电脑操作 Agent 栈 | 活跃开源项目 | 本地/远程电脑操作、浏览器操作、视觉 grounding、CLI/Web UI、GUI Agent 模型 | 与 LobsterAI 的桌面执行面相邻，但不是完整个人 Agent runtime；更偏 Computer Use |
| [[Kimi Claw]] | 月之暗面 | 云端 OpenClaw 产品化 | 商业化/会员产品 | 一键云端部署、Kimi K3、5000+ ClawHub Skills、40GB 云存储、长期记忆、定时任务、多端/多渠道 | 基础模型厂商直接把 OpenClaw 云端化，强调模型、存储与技能生态绑定 |
| [[MaxClaw]] | MiniMax | 云端 OpenClaw 产品化 | 商业化云服务 | 10 秒一键部署、长期记忆、多端协作、内置工具、24/7 AI 专家团队、云端工作区 | MiniMax 版本的 OpenClaw 产品化，强调专家 Agent 和免运维 |

## 3. 核心产品拆解

### 3.1 OpenClaw：个人设备上的 Gateway 型 Agent

**它解决什么问题**：把模型、工具、消息渠道、设备节点和 Skills 接到一个本地 Gateway 里，让同一个个人 Agent 出现在电脑、手机和聊天工具中。

**关键能力**

- Gateway 作为 sessions、tools、events 和 channels 的本地控制面；
- 支持 CLI、TUI、Control UI，以及 WhatsApp、Telegram、Slack、Discord、Signal、iMessage 等频道；
- 工具、Skills、Plugins、节点、语音、Canvas、摄像头和屏幕操作可扩展；
- 支持本地模型和云模型；
- 主会话工具默认运行在宿主机，也可配置 sandbox。

> [!tip] **独有重点：生态广度与 Gateway 抽象**
> OpenClaw 的强项不是某一个记忆算法，而是把“一个人拥有的设备、渠道、工具和 Agent 状态”统一到 Gateway。它更像个人 Agent 的操作层。

**官方资料**

- [OpenClaw GitHub](https://github.com/openclaw/openclaw)
- [OpenClaw 官网](https://openclaw.ai)
- [OpenClaw 文档](https://docs.openclaw.ai)

### 3.2 Hermes Agent：有学习闭环的 Agent runtime

**它解决什么问题**：让 Agent 不只是记住聊天，而是在长期使用中沉淀 Skills、用户模型和任务经验。

**关键能力**

- 内置学习闭环：复杂任务后创建 Skills，使用中修补 Skills；
- 分层记忆：当前上下文、完整历史、长期个人知识；
- FTS5 会话搜索与 LLM 摘要；
- cron 调度、后台任务和跨渠道投递；
- 子 Agent、并行工作流和 RPC 工具调用；
- 支持本地、Docker、SSH、Singularity、Modal、Daytona、Vercel Sandbox 等执行后端；
- Telegram、Discord、Slack、WhatsApp、Signal、CLI 等入口。

> [!tip] **独有重点：自我进化闭环**
> Hermes 的相对独特性不是“有记忆”，而是把**任务经验 → Skill 创建/修补 → 下次复用**做成 runtime 的核心机制，再用会话搜索和用户建模补上长期连续性。

**官方资料**

- [Hermes Agent GitHub](https://github.com/NousResearch/hermes-agent)
- [Hermes Agent 官网/文档](https://hermes-agent.nousresearch.com/)

### 3.3 QwenPaw（原 CoPaw）：国内大厂的完整 Agent OS 路线

**它解决什么问题**：提供一个本地或云端可部署的个人 Agent，兼顾开源模型、本地数据、工具扩展、安全和多渠道运行。

**关键能力**

- 本地或云端部署，提供 QwenPaw-Flash 2B/4B/9B Agent 模型和本地 runtime；
- 三层记忆：live working context、完整原文历史、ReMe 自进化个人知识库；
- Agent OS：Resources、Governance、Sandbox 三个工作区支柱；
- Kernel-level Sandbox、Tool Guard、File Guard、Skill Scanner、Access Policy；
- Skills、Plugins、MCP，并支持 ACP/A2A 连接层；
- 多 Agent 并行、TUI、Console、桌面 App；
- 钉钉、飞书、微信、Discord、Telegram、iMessage、QQ 等渠道；
- 支持 cron、浏览器、文档、新闻和自定义工作流。

> [!tip] **独有重点：模型、记忆与治理一体化**
> QwenPaw 的相对独特性在于：它不是只开源一个 Agent loop，而是把**本地模型、Agent OS、记忆、沙箱、协议连接层和多渠道**一起做成产品。相比 OpenClaw，它更强调默认治理；相比 Hermes，它更强调本地模型和 Agent OS 结构。

**当前名称校正**

- GitHub 当前主仓库：`agentscope-ai/QwenPaw`；
- 本文所说的“CoPaw”指该项目的前身/旧称，不另算一个产品；
- QwenPaw README 的历史版本和社区搜索仍可能出现 CoPaw 名称，更新产品资料时要按仓库当前名称核对。

**官方资料**

- [QwenPaw GitHub](https://github.com/agentscope-ai/QwenPaw)
- [QwenPaw 文档](https://qwenpaw.agentscope.io/)
- [QwenPaw Release Notes](https://qwenpaw.agentscope.io/release-notes)
- [ReMe GitHub](https://github.com/agentscope-ai/ReMe)

### 3.4 LobsterAI：把 OpenClaw 包进桌面 Cowork 成品

**它解决什么问题**：让普通用户不用先理解 Gateway、CLI 和配置文件，就能把一个能操作真实桌面环境的 Agent 装到 macOS/Windows 上。

**架构定位**

LobsterAI README 明确写出：

- `Cowork` 是产品/session 层；
- `OpenClaw` 是底层 runtime 和 gateway；
- LobsterAI 把本地持久化、权限、UI 状态、Artifacts、Agents、Memory 和 IM 绑定放在桌面应用中；
- 通过 OpenClaw 完成 Agent 执行。

**关键能力**

- 桌面文件、终端、浏览器、文档、表格、幻灯片、视频和网页研究；
- 28 个内置 Skills，覆盖 Office、PDF、浏览器、研究、邮件、内容创作等；
- MCP Server；
- 定时任务；
- 微信、企微、钉钉、飞书、QQ、Telegram、Discord、网易 IM、POPO、邮件远程控制；
- 多 Agent、Expert Kits、Artifacts；
- SQLite 本地会话与应用数据；
- 敏感文件、终端和网络动作需要权限确认。

> [!tip] **独有重点：成品化桌面体验 + OpenClaw runtime**
> LobsterAI 不是另起炉灶重写 Agent runtime，而是把 OpenClaw 嵌入桌面 Cowork 产品。它的竞争点是**真正的桌面入口、权限 UI、Artifacts、Office 能力和手机远程控制**，这是 OpenClaw CLI 本身不直接提供的产品层。

**官方资料**

- [LobsterAI GitHub](https://github.com/netease-youdao/LobsterAI)
- [LobsterAI 官网/下载](https://lobsterai.youdao.com/)

### 3.5 CowAgent：中文生态里最直接的 OpenClaw/Hermes 对标

**它解决什么问题**：让个人 Agent 具备规划、执行、记忆、Skills、知识库和自进化，而不是只做 IM 聊天机器人。

**关键能力**

- 多步规划并循环调用工具；
- 三层记忆：上下文、daily memory、`MEMORY.md`；
- nightly Deep Dream，把碎片记忆沉淀成长记忆；
- Self-Evolution，自动改进 Skills、跟进未完成任务、合并知识；
- 内置文件、终端、浏览器、调度、记忆、搜索和 MCP 工具；
- Skill Hub、GitHub、ClawHub、URL 安装与自然语言创建 Skills；
- Web、微信、飞书、钉钉、企微、QQ、Telegram、Slack 等渠道；
- macOS/Windows Desktop client、Web Console、Docker/Server 部署。

> [!tip] **独有重点：中文渠道覆盖 + Deep Dream 记忆沉淀 + Skill Hub**
> CowAgent 的差异是把 Hermes 式的记忆/自进化理念和国内 IM 渠道、桌面客户端、Skill Hub 结合起来。它比单纯的 IM Bot 更接近完整个人 Agent runtime。

**官方资料**

- [CowAgent GitHub](https://github.com/zhayujie/CowAgent)
- [CowAgent 官网](https://cowagent.ai)
- [CowAgent 文档](https://docs.cowagent.ai/)

### 3.6 AstrBot：IM-first，但已具备直接同类的 runtime 结构

**它解决什么问题**：以微信/QQ/飞书等 IM 作为入口，提供插件、Agent、MCP、知识库和沙箱能力。

**关键能力**

- Agent、MCP、Skills、知识库、Persona 和上下文压缩；
- 1000+ 社区插件；
- Agent Sandbox，隔离执行代码、Shell 和会话资源；
- Web ChatUI、桌面应用和 Launcher；
- 多 IM 平台接入；
- 可进行 proactive agent 与自动化。

> [!tip] **独有重点：插件生态和 IM 入口**
> AstrBot 的产品重心明显比 OpenClaw 更偏 IM Bot/机器人平台，但它已经拥有本地部署、工具调用、Agent Sandbox、插件市场和 Web/Desktop 入口，因此仍属于直接同类，而不是普通聊天框架。

**边界**

- 如果你要的是“桌面上的 Agent 工作台”，AstrBot 不如 LobsterAI；
- 如果你要的是“消息渠道、插件和机器人生态”，AstrBot 更有吸引力。

**官方资料**

- [AstrBot GitHub](https://github.com/AstrBotDevs/AstrBot)
- [AstrBot 官网](https://astrbot.app)

### 3.7 ZLAgent：IM-first 的小型自主 Agent runtime

**定位**：一个面向个人的微信/企微/Webhook Agent，重点放在长期记忆、GraphRAG、Skills、MCP 和长任务控制，而非桌面 GUI。

**关键能力**

- 微信、企微和 Webhook 入口；
- 长期记忆；
- Skills 系统；
- GraphRAG；
- MCP 工具；
- FastAPI + PostgreSQL/pgvector + Neo4j + Redis；
- Docker 一键部署；
- 任务、检查点、审批、恢复策略等 runtime 结构已在 README 的实现清单中出现。

> [!tip] **独有重点：把 Hermes 记忆思路和企业微信/微信入口做成小型可部署 runtime**
> ZLAgent 的价值不是功能数量，而是架构取舍很明确：IM-first、持久记忆、GraphRAG、MCP 和可恢复长任务。它仍处于早期，不能和 QwenPaw、CowAgent 的成熟度等量齐观。

**官方资料**

- [ZLAgent GitHub](https://github.com/Kkkirito-123/ZLAgent)

### 3.8 nanobot：极简、可读、可自托管的 OpenClaw 替代

**定位**：用较小的 Python 代码规模提供个人 Agent 所需的 Gateway、WebUI、工具、记忆、MCP、自动化和聊天渠道。

**关键能力**

- WebUI 和 Gateway 双入口；
- 文件、Shell、浏览器/网络等工具；
- 持久记忆；
- MCP；
- 多 Agent 工作流；
- 自动化和后台运行；
- 多渠道接入；
- 可在个人设备或服务器自托管。

> [!tip] **独有重点：用小代码规模换取可读性和低部署成本**
> nanobot 的差异是“足够完整但不臃肿”。它不是在产品层击败 OpenClaw，而是让开发者更容易读懂、改造和部署一套个人 Agent runtime。

**官方资料**

- [nanobot GitHub](https://github.com/HKUDS/nanobot)
- [nanobot 文档](https://nanobot.wiki)

### 3.9 nano-claw：nanobot 的 TypeScript 轻量实现

**定位**：受 OpenClaw 启发、基于 nanobot 思路的 TypeScript/Node.js 轻量个人 Agent。

**关键能力**

- 约 4,500 行核心 TypeScript；
- Agent loop、持久记忆、Skills、子 Agent；
- Shell、文件等工具；
- Telegram、Discord、钉钉等渠道；
- cron、heartbeat、session 管理；
- 多模型 provider。

> [!tip] **独有重点：TypeScript 生态里的最小 OpenClaw-like 实现**
> nano-claw 不是大而全的产品，而是给熟悉 Node.js/TypeScript 的开发者一个极小、可改的 runtime。它适合研究和二次开发，不应按成品体验评价。

**官方资料**

- [nano-claw GitHub](https://github.com/hustcc/nano-claw)

### 3.10 mini-claw：Java 生态的轻量实现

**定位**：一个直接自称“迷你的 OpenClaw、CoPaw”的 Java Agent 项目。

**关键能力**

- 多渠道：CLI、飞书、HTTP/SSE；
- 文件读写、搜索、Shell、日期等工具；
- 自定义 Skills；
- Quartz cron 调度；
- JsonSession 会话持久化；
- OpenAI/Anthropic 多模型切换。

> [!tip] **独有重点：Java/Spring 工程栈的龙虾 runtime**
> mini-claw 的意义在于把 OpenClaw 类 runtime 带进 Java/Spring WebFlux/Quartz 生态；目前项目规模和活跃度较小，适合做架构参考，不适合直接当生产基座。

**官方资料**

- [mini-claw GitHub](https://github.com/memglongdeqiangse/mini-claw)

### 3.11 FlashClaw：插件化、热加载和飞书优先

**定位**：中文社区的轻量 TypeScript Agent，强调插件化和国内渠道。

**关键能力**

- 插件化渠道和工具；
- 运行时热加载插件；
- 飞书、Telegram；
- 长期记忆；
- cron、间隔和一次性定时任务；
- Windows 后台服务安装、诊断和安全审计。

> [!tip] **独有重点：小 runtime 的插件热加载与中国渠道优先**
> FlashClaw 的差异不在复杂的 Agent OS，而在开发体验：功能像 Minecraft Mod 一样可插拔，运行中加载，且对飞书接入和 Windows 后台服务做了明确包装。

**官方资料**

- [FlashClaw GitHub](https://github.com/GuLu9527/flashclaw)

### 3.12 腾讯：QClaw、WorkBuddy 与 OpenClaw 渠道生态

本轮通过 `gh` 检查了 `Tencent`、`TencentCloud`、`Tencent-Hunyuan`、`YuanbaoTeam` 组织，以及 `QClaw`、`WorkBuddy`、`Yuanbao OpenClaw` 等关键词；再用腾讯官方产品页和文档核对商业产品。结论不是“腾讯没有类似产品”，而是：**腾讯有两个直接产品，但没有把完整 runtime 以独立官方 GitHub 项目公开出来**。

#### QClaw：腾讯电脑管家的个人微信远控型 Agent

**定位**：腾讯电脑管家基于 OpenClaw 开源生态打造的本地化个人 AI Agent。

**关键能力**

- macOS / Windows 桌面客户端；
- 本地文件、浏览器、邮件等系统资源操作；
- 微信扫码绑定，通过微信远程下达任务；
- 内置国产模型、Skills 生态与持续记忆；
- 一键关联已经安装的 OpenClaw；
- 面向办公、创作、开发等个人场景。

> [!tip] **相对独有：微信远程控制本地电脑**
> QClaw 的产品壁垒不是重新发明 Agent loop，而是把“本地执行 + 微信入口 + 小白安装”组合起来。它比 OpenClaw 更容易上手，比 WorkBuddy 更偏个人用户；真正的关键验收点是权限边界、数据是否留在本地和远程操作的安全确认。

**官方资料**
- [QClaw 官网](https://qclaw.qq.com/)
- [QClaw 产品文档：是什么及与 OpenClaw 的区别](https://qclaw.qq.com/docs/205521621464268800)
- [QClaw 新手指引](https://qclaw.qq.com/docs/205441750814556160)

#### WorkBuddy：腾讯云的职场/团队 Agent 工作台

**定位**：腾讯云 CodeBuddy 推出的桌面 AI Agent 办公工具。官方描述是：用户用一句话描述需求，WorkBuddy 自主规划和执行多模态复杂任务，并交付可验收结果。

**关键能力**

- 本地授权文件夹读取与批量处理；
- 文档、表格、PPT、数据分析、深度研究、邮件等多模态任务；
- 多 Agents 并行工作；
- Skills、专家、专家团协作；
- 桌面端与主流 IM / 小程序入口；
- 腾讯文档、会议、邮箱、微信、企微、ima 等腾讯生态连接；
- 面向个人、团队和企业，提供权限、用量和企业服务。

> [!tip] **相对独有：专家团 + 腾讯办公生态的交付工作台**
> WorkBuddy 更像“职场版 LobsterAI”：重点不是让用户自己搭 runtime，而是让多个专家 Agent 协作完成文档、数据、研究和办公交付。公开产品页强调工作台与专家团；至于其底层是否完全复用 OpenClaw，不应仅凭社区文章下结论。

**官方资料**

- [WorkBuddy 官方产品页](https://copilot.tencent.com/work/)
- [WorkBuddy 官方文档](https://www.workbuddy.ai/docs/zh/workbuddy/Overview)
- [WorkBuddy Agents](https://copilot.tencent.com/agents)

#### 腾讯公开 GitHub 生态：渠道与能力层，而不是独立 runtime

- `Tencent/openclaw`：GitHub API 标记为 fork，父仓库是 `openclaw/openclaw`；不能算腾讯自研 runtime。
- `Tencent/clawhub`：`openclaw/clawhub` 的 fork，属于 Skill Directory 相关代码。
- `Tencent/openclaw-weixin`：腾讯维护的 OpenClaw 微信 Channel Plugin，支持二维码登录、多账号和 Gateway 接入。
- `Tencent/yuanbao-openclaw-plugin`：把腾讯元宝机器人通过 WebSocket 接入 OpenClaw，支持私聊、群聊、权限策略和 slash commands。
- `Tencent/openclaw-tencent-provider`：腾讯云模型 Provider 插件。
- `Tencent/BrowserSkill`：把已登录浏览器桥接给 OpenClaw、Hermes、WorkBuddy 等 Agent，属于浏览器能力层。
- `Tencent/SkillHone`：跨 OpenClaw/Hermes/Codex 等 runtime 的 Skills 持续进化工具，属于生态能力层。

> [!note] **QClaw 的 GitHub 误判纠正**
> `qiuzhi2046/Qclaw` 是个人账号下的开源 OpenClaw 桌面管家，README 明确写明项目已暂停更新。它不是腾讯官方 QClaw 源码仓库；但腾讯官方 QClaw 产品确实存在，官方入口是 `qclaw.qq.com`。

**官方/一手资料**

- [Tencent/yuanbao-openclaw-plugin](https://github.com/Tencent/yuanbao-openclaw-plugin)
- [Tencent/openclaw-weixin](https://github.com/Tencent/openclaw-weixin)
- [Tencent/openclaw-tencent-provider](https://github.com/Tencent/openclaw-tencent-provider)
- [Tencent/BrowserSkill](https://github.com/Tencent/BrowserSkill)
- [Tencent/SkillHone](https://github.com/Tencent/SkillHone)
- [Tencent/openclaw](https://github.com/Tencent/openclaw)（fork，不是腾讯自研 runtime）
- [qiuzhi2046/Qclaw](https://github.com/qiuzhi2046/Qclaw)（个人开源桌面管家，不是腾讯官方源码）

本轮通过 `gh` 检查了 `Tencent`、`TencentCloud`、`Tencent-Hunyuan`、`YuanbaoTeam` 组织，以及 `QClaw`、`WorkBuddy`、`Yuanbao OpenClaw` 等关键词。结论要拆开看：

### 3.13 DeerFlow 2.0：字节跳动的长任务 SuperAgent harness

**定位**：DeerFlow 2.0 已不只是早期 Deep Research 项目，而是字节跳动开源的长时任务 SuperAgent harness。README 明确把它定义为能编排 **sub-agents、memory、sandboxes、tools、skills 和 message gateway** 的系统，任务可运行数分钟到数小时。

**关键能力**

- 长时任务与多 Agent 编排；
- Sandbox 执行环境；
- Memory、Skills、Tools；
- Sub-agents；
- Gateway 与 IM Channels，包括微信、企微、飞书、钉钉、Telegram 等；
- Web UI、Gateway、CLI/本地开发和 Docker 部署；
- 本地或服务器运行，支持多模型 Provider；
- 支持 MCP 与 Skills 扩展。

> [!tip] **相对独有：长任务 SuperAgent + 沙箱 + 多 Agent**
> DeerFlow 与 OpenClaw/Hermes 的共同点是 runtime/harness 层，而不是单一业务 Agent。它的重心比 OpenClaw 更偏“研究、代码、创作和多 Agent 长任务”，比 LobsterAI 更偏后端/服务器和可编排执行。

**边界**

- DeerFlow 2.0 的推荐部署更偏 Linux/Docker/服务器；它不是 LobsterAI 那种面向普通用户的桌面成品。
- `bytedance/trae-agent` 主要是软件工程 Coding Agent，本轮仍排除；不要因为同属字节就把 Trae Agent 算进来。

**官方资料**

- [DeerFlow GitHub](https://github.com/bytedance/deer-flow)
- [DeerFlow 官网](https://deerflow.tech)

### 3.14 Agent TARS / UI-TARS Desktop：字节跳动的 GUI Agent 栈

**定位**：字节跳动 Seed 团队的 GUI/Computer Use 方向。`bytedance/UI-TARS-desktop` README 将其拆成两部分：**Agent TARS** 是把 GUI Agent 和 Vision 带进终端、电脑、浏览器和产品的通用多模态 Agent stack；**UI-TARS Desktop** 是由 UI-TARS 模型驱动的本地 GUI Agent 桌面应用。

**关键能力**

- 本地电脑操作；
- 远程 Computer Operator；
- 远程 Browser Operator；
- 浏览器 GUI、DOM 或混合策略；
- CLI、Web UI、桌面 GUI；
- Windows/macOS/Browser 跨平台；
- UI-TARS 模型和 Seed 系列模型；
- 本地处理路线，支持本地或远程 Operator。

> [!tip] **相对独有：视觉 GUI 操作与模型栈一体化**
> UI-TARS Desktop 的优势不是长期记忆、Skills 或 IM Gateway，而是“看屏幕、定位控件、操作电脑”的模型和执行栈。它应放在 LobsterAI 旁边作为桌面执行能力对照，而不是与 OpenClaw/Hermes 的长期个人 Agent runtime 完全等价。

**官方资料**

- [UI-TARS Desktop GitHub](https://github.com/bytedance/UI-TARS-desktop)
- [UI-TARS 模型 GitHub](https://github.com/bytedance/UI-TARS)
- [Agent TARS 官网](https://agent-tars.com)

### 3.15 ArkClaw：火山引擎的云端 7×24 Agent

**定位**：火山引擎提供的云端 SaaS AI 助手。官方产品页把 ArkClaw 定义为零门槛、7×24 在线的专属智能服务，并明确展示了与 Hermes Agent 的协同：ArkClaw 负责连接与决策，Hermes 负责执行与学习。

**关键能力**

- 一键云端部署，无需自己维护 VPS/Docker；
- 7×24 在线，支持调度与后台执行；
- 创建多个智能伙伴，共享或协同处理任务；
- 云端持久存储与工作区；
- 连接飞书及火山引擎/字节生态；
- 与 Hermes Agent 形成“决策/连接 + 执行/学习”的组合。

> [!tip] **相对独有：把 Hermes 的学习闭环商品化为云端服务**
> ArkClaw 的独特性不是“又一个云端聊天框”，而是把 OpenClaw/Hermes 的运行模式包装成免运维 SaaS，并把模型、云资源、渠道和存储一体化。它与 LobsterAI 的差异是：LobsterAI 强在本地桌面，ArkClaw 强在云端常驻。

**官方资料**

- [火山引擎 ArkClaw](https://www.volcengine.com/product/arkclaw)

### 3.16 Kimi Claw：月之暗面的云端 OpenClaw

**定位**：月之暗面把 OpenClaw 直接产品化为云端 7×24 AI Agent。官方页面明确写出一键部署、Kimi K3、5,000+ ClawHub Skills、40GB 云端存储、长期记忆和定时任务。

**关键能力**

- 一键云端部署，免服务器、免命令行；
- Kimi K3 原生模型；
- 5,000+ ClawHub Skills 即时可用；
- 40GB 云端存储；
- 长期记忆与跨会话个性化；
- 主动定时任务；
- 支持网页、消息渠道和 Android 设备部署/连接。

> [!tip] **相对独有：基础模型厂商直接绑定 OpenClaw 云产品**
> Kimi Claw 的核心是“模型 + 云部署 + Skill 生态”的一体化。它不像 QClaw 那样强调本地微信远控，也不像 LobsterAI 那样强调本地桌面，而是把 OpenClaw 变成 Kimi 会员/云基础设施中的常驻 Agent。

**官方资料**

- [Kimi Claw](https://www.kimi.com/bot)
- [Kimi Claw 功能介绍](https://www.kimi.com/zh-cn/resources/kimi-claw-introduction)

### 3.17 MaxClaw：MiniMax 的云端 OpenClaw 产品

**定位**：MiniMax 官方云端 AI Agent，明确基于 OpenClaw 框架构建，主打零代码、10 秒一键部署、长期记忆、多端协作和内置工具。

**关键能力**

- 10 秒一键部署；
- 无需服务器、Docker 或 API Key；
- 长期记忆；
- 多端协作；
- 内置工具与 24×7 AI 专家团队；
- 云端工作区与文件管理。

> [!tip] **相对独有：专家 Agent + OpenClaw 云端化**
> MaxClaw 的差异是把 MiniMax 自己的专家 Agent 能力叠加到 OpenClaw 产品形态上。它与 Kimi Claw 都是云端化，但 MaxClaw 更强调专家团队与免配置入口。

**官方资料**

- [MaxClaw 中文官方页](https://agent.minimaxi.com/activity/max-claw)
- [MaxClaw English official page](https://agent.minimax.io/activity/max-claw)

## 4. 能力矩阵

评分只根据公开 README 和仓库元数据做定性判断，不是统一 benchmark。

| 产品/项目 | Runtime | 桌面入口 | 本地/自托管 | 持久记忆 | Skills/Plugins | MCP | 定时/后台 | 多渠道 | 安全/审批 | 主要差异 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| OpenClaw | 高 | 中 | 高 | 中 | 高 | 中-高 | 高 | 高 | 中 | Gateway 与生态广度 |
| Hermes Agent | 高 | 中 | 高 | 高 | 高 | 中 | 高 | 高 | 中-高 | 自我进化闭环 |
| QwenPaw（原 CoPaw） | 高 | 中-高 | 高 | 高 | 高 | 高 | 高 | 高 | 高 | Agent OS 与治理 |
| LobsterAI | 高（嵌入 OpenClaw） | 高 | 高 | 高 | 高 | 高 | 高 | 高 | 高 | 桌面 Cowork 成品化 |
| CowAgent | 高 | 中-高 | 高 | 高 | 高 | 高 | 高 | 高 | 中 | Deep Dream + Skill Hub |
| AstrBot | 高 | 中 | 高 | 中 | 高 | 高 | 中-高 | 高 | 高 | IM 与插件生态 |
| ZLAgent | 高 | 低 | 高 | 高 | 高 | 高 | 中-高 | 高 | 高 | GraphRAG + 可恢复任务 |
| nanobot | 高 | 中 | 高 | 中 | 中-高 | 高 | 高 | 中-高 | 中 | 轻量、可读、易自托管 |
| nano-claw | 高 | 低-中 | 高 | 中 | 高 | 中 | 高 | 中 | 中 | TypeScript 极简实现 |
| mini-claw | 高 | 低 | 高 | 中 | 中 | 低-中 | 高 | 中 | 低-中 | Java/Spring 实现 |
| FlashClaw | 高 | 低-中 | 高 | 中 | 高 | 中 | 高 | 中 | 中 | 热加载与飞书优先 |
| QClaw | 高（基于/关联 OpenClaw） | 高 | 高 | 中-高 | 高 | 中 | 高 | 高（微信） | 中-高 | 微信远程控制本地电脑 |
| WorkBuddy | 高 | 高 | 中-高 | 中-高 | 高 | 中-高 | 高 | 高 | 高 | 专家团 + 职场交付工作台 |
| DeerFlow 2.0 | 高 | 低-中 | 高 | 高 | 高 | 高 | 高 | 高 | 高 | 长任务 SuperAgent harness |
| ArkClaw | 高（产品化） | 低 | 云端 | 高 | 高 | 中-高 | 高 | 中-高 | 高 | 云端 7×24 + Hermes 协同 |
| Agent TARS / UI-TARS Desktop | 中-高 | 高 | 中-高 | 低-中 | 中 | 高 | 中 | 低 | 中 | GUI/浏览器/电脑操作栈 |
| Kimi Claw | 高（云端 OpenClaw） | 低 | 云端 | 高 | 高 | 中 | 高 | 中-高 | 中-高 | Kimi 模型 + 5,000+ Skills + 云存储 |
| MaxClaw | 高（云端 OpenClaw） | 低 | 云端 | 高 | 中-高 | 中 | 高 | 中-高 | 中-高 | MiniMax 专家 Agent + 免运维部署 |

## 5. 独有特性总表

> [!note]
> “独有”采用相对口径：指在这组直接同类中最突出的差异，不宣称全球永久唯一。

| 产品/项目 | 最值得高亮的独有/差异特性 | 主要来源 |
|---|---|---|
| OpenClaw | Gateway 把设备、频道、工具、Skills、Plugins、节点统一起来 | Gateway + 生态 |
| Hermes Agent | 复杂任务后自动创建/修补 Skills，形成自我进化闭环 | 学习循环 + 记忆 |
| QwenPaw（原 CoPaw） | 本地模型、Agent OS、ReMe 记忆、安全治理、MCP/A2A/ACP 一体化 | 模型 + runtime + governance |
| LobsterAI | OpenClaw runtime + 桌面 Cowork + 权限 UI + Artifacts + Office + IM 远控 | 成品桌面层 |
| CowAgent | Deep Dream 记忆沉淀、Self-Evolution、中文渠道与 Skill Hub | 记忆 + 生态 |
| AstrBot | 1000+ 插件、IM-first、多模态、Agent Sandbox 和桌面部署 | 插件 + IM + 沙箱 |
| ZLAgent | GraphRAG + 长期记忆 + MCP + 微信/企微 + 可恢复任务结构 | 记忆 + IM + 长任务 |
| nanobot | 小代码规模提供 WebUI、Gateway、MCP、自动化和多渠道 | 轻量可读 |
| nano-claw | TypeScript/Node.js 的约 4,500 行 OpenClaw-like 实现 | 语言生态 + 极简 |
| mini-claw | Java/Spring WebFlux + Quartz 的龙虾 runtime | Java 生态 |
| FlashClaw | 插件热加载、飞书优先、Windows 后台服务与安全诊断 | 插件 + 本地运维 |
| QClaw | 微信远程控制本地电脑、低门槛安装、OpenClaw 生态兼容 | 微信入口 + 本地执行 |
| WorkBuddy | 多 Agents、150+ 专家团、腾讯办公生态和可验收交付 | 专家协作 + 办公生态 |
| DeerFlow 2.0 | 沙箱、记忆、Skills、子 Agent、Gateway 与数分钟到数小时长任务 | 长任务 + 编排 |
| ArkClaw | 云端 7×24、调度/执行双引擎、Hermes 协同 | 云端常驻 + 学习闭环 |
| Agent TARS / UI-TARS Desktop | GUI Agent 模型、视觉 grounding、本地/远程 Computer Operator 与 Browser Operator | GUI 执行 + 模型栈 |
| Kimi Claw | Kimi K3、5,000+ ClawHub Skills、40GB 云存储、长期记忆 | 模型 + 云基础设施 |
| MaxClaw | 10 秒部署、MiniMax 专家 Agent、长期记忆与多端协作 | 专家 Agent + 云端化 |

## 6. 按目的选型

### 6.1 我想要一个真正的个人 Agent

- **最完整、生态最大**：OpenClaw。
- **最强调长期成长**：Hermes Agent。
- **国内大厂开源路线**：QwenPaw（原 CoPaw）。
- **国内桌面成品路线**：LobsterAI。
- **中文生态与本地 IM**：CowAgent。
- **IM/插件优先**：AstrBot。
- **腾讯个人微信远控**：QClaw。
- **腾讯职场/团队交付**：WorkBuddy。
- **字节长任务与多 Agent**：DeerFlow 2.0。
- **火山云端 7×24**：ArkClaw。
- **Kimi 云端 OpenClaw**：Kimi Claw。
- **MiniMax 云端 OpenClaw**：MaxClaw。

### 6.2 我想要一个国内直接替代品

优先顺序：**LobsterAI → QwenPaw → CowAgent → AstrBot → ZLAgent**。

- LobsterAI：最像“装完就用”的桌面 Cowork；
- QwenPaw：最像完整 Agent OS/runtime；
- CowAgent：最像中文生态的 Hermes/OpenClaw 综合体；
- AstrBot：最像 IM/插件生态型 Agent；
- ZLAgent：最像小型、可自己掌控的 IM-first runtime；
- QClaw：最像微信远控型个人龙虾；
- WorkBuddy：最像职场/团队版桌面 Agent；
- DeerFlow：最像服务器侧的长任务 SuperAgent harness；
- ArkClaw：最像云端常驻的 OpenClaw + Hermes 产品化版本；
- Kimi Claw：最像模型厂商绑定 Skills、存储和云端 Agent 的版本；
- MaxClaw：最像“专家团队 + OpenClaw 云端化”的版本。

### 6.3 我是开发者，想研究 Agent runtime

- Python：QwenPaw、Hermes、CowAgent、nanobot、ZLAgent；
- TypeScript：OpenClaw、LobsterAI、nano-claw、FlashClaw；
- Java：mini-claw；
- Rust：QwenPaw 的部分运行/沙箱生态，以及其他 Claw 系列项目；
- Python/全栈长任务：DeerFlow；
- TypeScript/GUI Agent：Agent TARS。

### 6.4 我只想降低安装和运维门槛

- LobsterAI：直接使用桌面 Cowork，适合不想先理解 CLI/Gateway 的用户；
- QwenPaw：提供脚本安装、Docker、桌面 Beta 和本地/云端部署路径；
- OpenClaw：适合愿意自己掌握 Gateway、频道和运行环境的用户；
- nanobot：适合希望以较小复杂度自托管个人 Agent 的开发者。

## 7. 仍然要重点观察的分水岭

### 7.1 Runtime 与产品壳必须分开

- OpenClaw、Hermes、QwenPaw、CowAgent、nanobot：核心是 Agent runtime；
- LobsterAI：产品壳 + OpenClaw runtime；
- QClaw、WorkBuddy、ArkClaw、Kimi Claw、MaxClaw：面向不同入口的产品化/云端层；
- nanobot、nano-claw、mini-claw、FlashClaw：轻量化或语言生态衍生实现。

把这些全部叫“竞品”会掩盖真正差异：有的在造完整 runtime，有的在做成品桌面入口，有的在用更小代码规模复刻核心闭环。

### 7.2 这条赛道真正的能力门槛

1. **长期记忆是否真的可用**：能否检索、纠错、压缩、解释和删除；
2. **Skills 是否会进化**：能否创建、修补、版本化、审计和回滚；
3. **后台任务是否可靠**：断网、重启、模型失败后能否恢复；
4. **工具权限是否可控**：文件、终端、网络和账号权限是否细分；
5. **任务完成是否可验证**：是否有 artifacts、测试、状态检查和证据链；
6. **模型是否可替换**：是否支持本地模型、BYOK、多 Provider 和 fallback；
7. **成本是否可预测**：长期运行、浏览器、记忆检索和重试的成本如何；
8. **部署是否真正属于用户**：数据、工作区、密钥和 session 能否导出。

## 8. 后续跟踪清单

| 字段 | 观察问题 |
|---|---|
| Runtime 边界 | 是独立引擎、桌面产品壳，还是控制面？ |
| 本地控制 | 是否真的能操作本地文件、终端、浏览器和应用？ |
| 记忆 | 是聊天历史、文件记忆，还是结构化长期用户模型？ |
| Skills | 能否由 Agent 自动生成、修补和回滚？ |
| 后台 | 是否支持 cron、heartbeat、daemon、重启恢复？ |
| 渠道 | 电脑、Web、微信、飞书、钉钉、Telegram 是否共享同一 session？ |
| 安全 | 是否有审批、沙箱、凭据隔离、网络出口和审计？ |
| 生态 | MCP、Plugins、Skill Hub、A2A/ACP 是否真实可用？ |
| 证据 | 任务结果是否能通过文件、测试、页面状态或 artifact 验证？ |
| 活跃度 | 最近 commit、release、issue 响应和破坏性变更如何？ |

## 9. 来源与核验说明

本版优先使用 `gh api` / `gh search repos` 读取公开 GitHub 元数据、README 和官方仓库链接；产品能力只写仓库明确声明的内容。对公开资料不足或已停止维护的项目，使用“早期”“观察项”“待实测”限定，不把社区二次开发或同名仓库当成官方产品。

### 官方/一手入口

- [OpenClaw GitHub](https://github.com/openclaw/openclaw)
- [OpenClaw 官网](https://openclaw.ai)
- [OpenClaw 文档](https://docs.openclaw.ai)
- [Hermes Agent GitHub](https://github.com/NousResearch/hermes-agent)
- [Hermes Agent 官网/文档](https://hermes-agent.nousresearch.com/)
- [QwenPaw GitHub](https://github.com/agentscope-ai/QwenPaw)
- [QwenPaw 文档](https://qwenpaw.agentscope.io/)
- [LobsterAI GitHub](https://github.com/netease-youdao/LobsterAI)
- [LobsterAI 官网/下载](https://lobsterai.youdao.com/)
- [CowAgent GitHub](https://github.com/zhayujie/CowAgent)
- [CowAgent 文档](https://docs.cowagent.ai/)
- [AstrBot GitHub](https://github.com/AstrBotDevs/AstrBot)
- [AstrBot 官网](https://astrbot.app)
- [ZLAgent GitHub](https://github.com/Kkkirito-123/ZLAgent)
- [nanobot GitHub](https://github.com/HKUDS/nanobot)
- [nanobot 文档](https://nanobot.wiki)
- [nano-claw GitHub](https://github.com/hustcc/nano-claw)
- [mini-claw GitHub](https://github.com/memglongdeqiangse/mini-claw)
- [FlashClaw GitHub](https://github.com/GuLu9527/flashclaw)
- [QClaw 官网](https://qclaw.qq.com/)
- [QClaw 产品文档](https://qclaw.qq.com/docs/205521621464268800)
- [WorkBuddy 官方产品页](https://copilot.tencent.com/work/)
- [WorkBuddy 官方文档](https://www.workbuddy.ai/docs/zh/workbuddy/Overview)
- [Tencent/yuanbao-openclaw-plugin](https://github.com/Tencent/yuanbao-openclaw-plugin)
- [Tencent/openclaw-weixin](https://github.com/Tencent/openclaw-weixin)
- [Tencent/openclaw-tencent-provider](https://github.com/Tencent/openclaw-tencent-provider)
- [Tencent/BrowserSkill](https://github.com/Tencent/BrowserSkill)
- [Tencent/SkillHone](https://github.com/Tencent/SkillHone)
- [DeerFlow GitHub](https://github.com/bytedance/deer-flow)
- [DeerFlow 官网](https://deerflow.tech)
- [火山引擎 ArkClaw](https://www.volcengine.com/product/arkclaw)
- [Agent TARS / UI-TARS Desktop](https://github.com/bytedance/UI-TARS-desktop)
- [UI-TARS 模型](https://github.com/bytedance/UI-TARS)
- [Agent TARS 官网](https://agent-tars.com)
- [Kimi Claw](https://www.kimi.com/bot)
- [Kimi Claw 功能介绍](https://www.kimi.com/zh-cn/resources/kimi-claw-introduction)
- [MaxClaw 中文官方页](https://agent.minimaxi.com/activity/max-claw)

### 仓库内关联笔记

- [[Hermes Agent 源码解读三大核心机制]]
- [[Hermes Agent 双层记忆架构拆解]]
- [[Hermes Agent v0.20.0：Hermes 会说话了]]
- [[从 Hermes Agent 看 AI Agent 记忆系统设计]]
- [[用了 Pi 一个月，我把 Claude Code 客户端删了]]

^sources
