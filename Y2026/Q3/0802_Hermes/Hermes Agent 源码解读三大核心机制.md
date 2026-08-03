https://mp.weixin.qq.com/s/bzFGlNeHbiUmKG4AxIQ6Bg

从源代码角度解读，Hermes Agent 的核心是一个精心设计的“自我进化”系统。其源码结构清晰展示了它如何通过**闭环学习、持久记忆和自主技**能生成三大核心机制，实现与传统 Agent 完全不同的能力增长路径。

## 一、核心源码模块解析

   Hermes Agent 的源码围绕“智能体进化”这一核心理念组织，主要包含以下几个关键模块：

#### 1.1 Agent 核心循环（`run_agent.py` ）

  `AIAgent` 类是系统的核心抽象，其 `run_conversation()` 方法实现了一个**完全同步的对话循环**：

![[Y2026/Q3/0802_Hermes/assets/Image 14.webp|Image]]

这个同步设计看似简单，实则是深思熟虑的工程决策——AI Agent 的核心瓶颈在于 LLM API 延迟而非 I/O 并发，同步循环让代码更易调试和维护。

**关键代码细节**：

- **迭代预算机制**：子 Agent 拥有独立预算，不会消耗父 Agent 的配额，防止单一任务耗尽全局资源。
    
- **消息格式标准**：严格遵循 OpenAI 格式 `{"role": "system/user/assistant/tool"}`，推理内容存储在 `assistant_msg["reasoning"]` 中，实现多模型无摩擦切换。
    
- **参数类型安全**：`coerce_tool_args()` 函数将 LLM 返回的字符串参数与 JSON Schema 进行比对自动强转，避免工具调用失败。
    

#### 1.2 记忆系统（`tools/memory_tool.py`）

记忆系统采用**有界策展式设计**，代码实现非常克制：

```
class MemoryStore:    def __init__(self, memory_char_limit=2200, user_char_limit=1375):        self.memory_entries: List[str] = []        self.user_entries: List[str] = []        self.memory_char_limit = memory_char_limit        self.user_char_limit = user_char_limit        self._system_prompt_snapshot: Dict[str, str] = {"memory": "", "user": ""}
```

**核心设计巧思**：

- **两个分离文件**：`MEMORY.md`（Agent 的环境知识）和 `USER.md`（用户偏好）。
    
- **有界容量**：硬性限制 2200/1375 字符，这不是技术限制而是**设计哲学**——倒逼 Agent 做信息压缩，过时的自然被挤掉。
    
- **“超限即失败”策略**：当添加条目超限时，`add` 操作直接返回错误而非静默丢弃，并返回 `current_entries` 让模型看到所有现有条目，引导其执行 `replace` 或 `remove` 操作。
    

**冻结快照机制**：

每次会话启动时，Memory 加载后立即捕获快照注入系统提示词，整个会话期间这个快照保持不变：

```
def load_from_disk(self):    # 从磁盘加载    self.memory_entries = self._read_file(mem_dir / "MEMORY.md")    self.user_entries = self._read_file(mem_dir / "USER.md")    # 去重条目（保留顺序，保留首次出现）    self.memory_entries = list(dict.fromkeys(self.memory_entries))    self.user_entries = list(dict.fromkeys(self.user_entries))    # 会话开始时冻结快照，之后不再变动     self._system_prompt_snapshot = {        "memory": self._render_block("memory", self.memory_entries),        "user": self._render_block("user", self.user_entries),    }
```

这样做的好处是：**前缀缓存（Prefix Caching）** 在整个会话期间有效，大幅降低 API 成本。

#### 1.3 技能系统（`tools/skill_manager_tool.py`）

技能系统是 Hermes “自我进化”能力的核心载体，其实现包含几个关键函数：

**技能自动创建**（`_create_skill`）：

```
def _create_skill(name: str, content: str, category: str = None):    # 1. 校验 name 格式（小写字母、数字、连字符）    # 2. 校验 frontmatter 结构（必须包含 name 和 description）    # 3. 检查同名冲突（跨本地和外部目录全量搜索）    # 4. 创建目录并原子写入 SKILL.md（临时文件 + os.replace()）    # 5. 安全扫描（与社区 Hub 安装技能同等待遇）    # 6. 清除系统 Prompt 缓存，使新技能立即生效
```

**技能精确修补**（`_patch_skill`）：

```
def _patch_skill(name, old_string, new_string):    # 使用模糊匹配引擎，容忍空白符/缩进差异    new_content, match_count = fuzzy_find_and_replace(content, old_string, new_string)    # 校验 patch 后 frontmatter 结构完整    # 原子写入 + 安全扫描 + 回滚机制patch动作采用精准的 find-and-replace 方式，而非全量替换，这大幅降低了 Token 消耗。模糊匹配机制专门为 LLM 生成内容的不精确性设计。
```

**触发机制**：系统 Prompt 中内置了驱动指令：

```
SKILLS_GUIDANCE = (    "After completing a complex task (5+ tool calls), fixing a tricky error, "    "or discovering a non-trivial workflow, save the approach as a "    "skill with skill_manage so you can reuse it next time.\n"    "When using a skill and finding it outdated, incomplete, or wrong, "    "patch it immediately with skill_manage(action='patch') — don't wait to be asked. "    "Skills that aren't maintained become liabilities.")
```

#### 1.4 工具注册层

`ToolRegistry` 单例是整个工具系统的脊柱。每个工具文件在模块导入时调用 `registry.register()` 声明：

- Schema 定义
    
- 处理器函数
    
- 工具集归属
    
- 可用性检查函数（如有 API Key 依赖）
    

**优雅降级机制**：需要 API Key 的工具在 Key 未配置时自动从工具列表中隐藏，而非报错。`get_tool_definitions()` 还会**动态重建某些工具的 Schema**——例如 `execute_code` 工具的描述中列出的可用工具，会随 API Key 配置状态动态变化，避免模型产生“幻觉工具调用”。

## 二、Self-Improving 闭环的实现流程

结合源码分析，Hermes Agent 的自我进化闭环是这样运行的：

![[Y2026/Q3/0802_Hermes/assets/Image 15.webp|Image]]

这个流程的关键在于**每完成一个复杂任务，Agent 都会自动将经验固化为可复用的技能文档**，并且在使用中持续优化。与 OpenClaw 需要手写技能不同，Hermes 的技能是“长”出来的。

##  **三、与传统 Agent 的本质区别**

从源码设计理念来看，Hermes 与 OpenClaw 代表了两种完全不同的路径：

|**维度**|**Hermes** **Agent**|**OpenClaw**|
|---|---|---|
|**核心定位**|自我成长型 Agent|多渠道消息路由网关|
|核心架构|Python 进程 + SQLite|Node.js Gateway + WebSocket|
|**技能来源**|Agent 自动生成并迭代|开发者手写或从社区安装|
|**记忆方式**|有界策展式（自动压缩淘汰）|模块化插件（可更换后端）|
|**学习能力**|内置闭环（经验→技能→复用）|无内置学习机制|
|**工具系统**|Python 自注册 + 动态 Schema|TypeScript 插件生态 + 沙箱|
|**技术栈**|Python 93.6%|TypeScript 90.3%|

核心差异在于：**OpenClaw 的 Skill 是你写多少它会多少；****Hermes** **的 Skill 是用得越久越厚**——每一次踩坑都在加固能力的“护城河”。

## 四、关键工程亮点

从源码中还能发现几个值得注意的工程细节：

1. **子 Agent 委托机制**：`delegate_task` 工具创建的子 Agent 拥有隔离的上下文（不继承父会话）、独立的 task_id、受限的工具集，最大委托深度为 2 防止递归失控。批量模式下最多 3 个子 Agent 并行运行，使用 `ThreadPoolExecutor`。
    
2. **原子文件操作**：技能写入使用临时文件 + `os.replace()`，防止写入中断导致文件损坏。
    
3. **前缀缓存优化**：冻结快照机制让整个会话的系统提示不变，支持 Anthropic 等提供商的 prompt caching 功能，大幅降低成本。
    
4. **安全扫描机制**：新创建的技能会经过与社区 Hub 安装技能同等的安全审查，扫描失败则自动回滚删除。
    

这种“让 Agent 自己长能力”的设计，使得 Hermes 特别适合需要长期运行、持续积累的**开发运维、自动化办公、知识管理等场景**。

_源码地址：__https://github.com/NousResearch/hermes-agent_