# 12 — Design Decision Log

本文件记录会影响多个模块的架构决策。普通实现任务不得自行更改这些决定。

## D001 — 固定四 Agent 链路

**Decision**：MVP 使用 Planner -> Retriever -> Executor -> Summarizer 固定顺序。

**Reason**：降低调度复杂度，把开发资源集中在协议/状态/记忆/评测。

## D002 — 不采用现成 Agent orchestration framework 作为核心

**Decision**：MVP Runtime 自行实现轻量 in-process orchestrator。

**Reason**：赛题重点是系统机制；避免框架接管与版本兼容拖慢提交。

## D003 — Shared Memory 为主要非文本传输路径

**Decision**：NumPy float32 state 存入 `multiprocessing.shared_memory`，消息传 StateRef；提供 inproc fallback。

**Reason**：机制简单、真实非文本、与操作系统主题相关、容易测量。

## D004 — Artifact 单独处理大文本/JSON

**Decision**：State 只承载数值/紧凑状态；长文本、证据、工具输出走 ArtifactStore。

**Reason**：防止“把任何东西都叫 state”，并保持指标可解释。

## D005 — SQLite + NumPy memory

**Decision**：MVP 不引入外部向量数据库。

**Reason**：足以满足关键词/标签/语义检索，部署稳定，openEuler 风险低。

## D006 — Mock-first

**Decision**：无 API Key/网络时核心测试和实验 smoke 必须运行。

**Reason**：避免模型和接口成为交付单点故障。

## D007 — A/B/C/D 分层消融

**Decision**：A Text；B Structured；C Structured+State；D Full。

**Reason**：分别解释协议、非文本状态和记忆带来的影响。

## D008 — text baseline 从相同 Envelope 语义生成

**Decision**：先形成统一内部事件，再根据模式走 TextAdapter 或 Structured Codec。

**Reason**：减少“两个完全不同实现”导致的不公平对比。

## D009 — Dashboard 不使用 Node 构建链

**Decision**：MVP 使用原生 HTML/CSS/JS + 可选 FastAPI。

**Reason**：降低依赖、部署和 openEuler 风险。

## D010 — Sandbox 明确是 best-effort restricted execution

**Decision**：使用临时目录、subprocess、timeout、输出限制、Linux rlimit 等，但不宣称为强安全隔离。

**Reason**：时间与安全工程成本不允许实现真正强沙箱；如实说明比过度承诺更可靠。
