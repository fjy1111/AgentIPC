# AgentIPC

> 面向多智能体协作的结构化通信、共享状态与持久记忆运行时

AgentIPC 是面向 openEuler / 通用 Linux 环境的多智能体协作原型系统。项目目标不是构建一个大而全的 Agent 框架，而是围绕比赛要求中的三个系统层问题形成最小但完整、可运行、可验证、可复现的实现：

1. **低开销结构化通信**：用统一协议代替 Agent 间冗长自然语言透传，并提供握手、注册、能力发现和协议映射。
2. **非文本中间状态传递**：使用 NumPy 数值向量与 Linux/Python Shared Memory 保存状态本体，消息只携带 `StateRef`。
3. **共享记忆复用**：使用 SQLite + 轻量向量检索保存、查询和复用跨任务记忆。

项目同时保留纯文本协作模式，在完全相同的任务、Agent 和执行逻辑下进行 A/B/C/D 实验：

- A：Text Baseline
- B：Structured Protocol
- C：Structured + State
- D：Structured + State + Memory

## 当前开发原则

本项目采用“架构设计与实现分离”的协作方式：

- 总规划设计：负责架构、接口、任务拆分、依赖关系和验收标准。
- 实现 GPT：每个对话框一次只负责一个小任务，不跨任务扩展功能。
- Codex：只负责运行、测试、静态检查和回传任务报告；默认不修改代码。

任何实现者开始工作前都必须阅读：

1. `AGENTS.md`
2. `docs/00_PROJECT_CHARTER.md`
3. `docs/01_ARCHITECTURE.md`
4. `docs/03_DEVELOPMENT_PLAN.md`
5. `docs/04_MODULE_CONTRACTS.md`
6. 当前被分配任务对应的验收条件

## MVP 边界

MVP 必须具备：

- 4 个 Agent：Planner / Retriever / Executor / Summarizer
- text / structured 双模式
- 统一结构化通信协议
- capability register / discover / handshake
- Shared Memory 非文本状态交换
- Artifact 内容寻址与引用
- SQLite 共享记忆
- keyword / tag / semantic / hybrid 检索
- Mock LLM + OpenAI-compatible LLM
- Hash Embedding + 可选真实 embedding
- CodeAct 受限 Python 执行
- 两组各 10 轮连续任务
- A/B/C/D benchmark
- 指标记录与报告
- 最小 Dashboard
- openEuler 24.03-LTS-SP3 安装、测试、验证脚本

## MVP 明确不做

在 MVP 验收完成前，禁止主动加入：LangGraph、CrewAI、AutoGen 接管、Redis、Milvus、Qdrant、HyperGraph Memory、LLMLingua、AutoForm、LLM hidden state、多机分布式、Kubernetes、eBPF、WASM、GPU 强依赖、多数据集大规模 benchmark。

## 目录目标

```text
agentipc/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── configs/
├── src/agentipc/
│   ├── agents/
│   ├── artifacts/
│   ├── evaluation/
│   ├── memory/
│   ├── protocol/
│   ├── providers/
│   ├── runtime/
│   ├── sandbox/
│   └── state/
├── scenarios/
├── dashboard/
├── scripts/
├── tests/
├── results/
└── docs/
```

详细设计、任务拆分和验收规则见 `docs/`。
