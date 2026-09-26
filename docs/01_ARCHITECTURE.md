# 01 — System Architecture

## 1. 总体架构

```text
                      ┌─────────────────────────┐
                      │       Agent Runtime      │
                      │                         │
                      │ Orchestrator / Router   │
                      │ Capability Registry     │
                      │ RunContext              │
                      │ Metrics / Trace         │
                      └────────────┬────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
        Protocol Layer         State Layer        Memory Layer
        AgentEnvelope          StateHub           MemoryService
        TextAdapter            SharedMemory       SQLiteStore
        Codec/Validation       StateRef           VectorIndex
              │                    │                    │
              └──────────────┬─────┴───────┬───────────┘
                             │             │
                       ArtifactStore     Sandbox
                             │             │
                             └──────┬──────┘
                                    │
        ┌──────────────┬────────────┼──────────────┬──────────────┐
        ▼              ▼            ▼              ▼
     Planner        Retriever     Executor       Summarizer
```

## 2. Agent 角色

### Planner

输入：用户任务。

输出：结构化计划，例如步骤、所需能力、检索主题、是否需要工具执行。

在开启 State Exchange 时，Planner 还负责生成 `PlanState` 并写入 `StateHub`。

### Retriever

输入：任务计划、知识库、可选 StateRef、可选历史 Memory。

输出：证据集合或 ArtifactRef。

Retriever 必须真实消费 Planner 提供的非文本状态，例如使用计划向量影响候选排序。

### Executor

输入：计划、证据或 ArtifactRef。

输出：结构化工具结果。

在 CodeAct 场景下，通过 Sandbox 执行受限 Python。

### Summarizer

输入：上游结果与引用。

输出：最终答案、简短证据摘要，以及需要写入 Memory 的候选记录。

## 3. Runtime

Runtime 不负责“聪明”，只负责：

- 启动 Agent
- 路由消息
- 能力发现
- 模式切换
- 引用解析
- 调用 State / Artifact / Memory
- 记录指标
- 错误传播
- 生成 trace

Runtime 的核心编排顺序固定：

```text
Planner -> Retriever -> Executor -> Summarizer
```

MVP 不做动态图规划。

## 4. 通信模式

### Text mode

- 相同的语义事件先形成内部 `AgentEnvelope`。
- `TextAdapter` 将其展开为自然语言消息。
- 需要的中间内容以文本形式 materialize。
- 指标记录文本字符/token。

### Structured mode

- 直接传递经过 schema 校验的 `AgentEnvelope`。
- 小结构可直接内联。
- 大内容用 ArtifactRef。
- 非文本状态用 StateRef。
- 历史知识用 MemoryRef。

因此 text / structured 的业务语义相同，仅载体不同。

## 5. 非文本状态路径

```text
Planner
  │
  ├── encode plan -> numpy.float32 vector
  │
  ├── StateHub.put(...)
  │        │
  │        └── SharedMemory stores raw bytes
  │
  └── AgentEnvelope(state_refs=[StateRef])
                          │
                          ▼
                      Retriever
                          │
                          ├── StateHub.resolve(ref)
                          └── vector influences retrieval ranking
```

状态本体不得先转换为 JSON 再声称为“非文本状态传递”。

## 6. Artifact 路径

较大文本/JSON/工具输出：

```text
producer -> ArtifactStore.put(payload) -> artifact://sha256/... -> consumer
```

ArtifactStore 是内容寻址存储，重复内容应返回相同或可复用的内容 hash。

## 7. Memory 路径

```text
Task N result
   │
   └── MemoryService.write(...)
             │
             ├── SQLite metadata/payload
             └── VectorIndex embedding

Task N+1
   │
   └── MemoryService.retrieve(query)
             │
             └── memory refs -> Agent consumer
```

MVP 记忆类型仅：

- evidence
- experience
- result

## 8. Provider 抽象

### LLMProvider

至少实现：

- MockLLMProvider
- OpenAICompatibleProvider

### EmbeddingProvider

至少实现：

- HashEmbeddingProvider
- 可选 SentenceTransformerEmbeddingProvider

所有核心测试必须用 Mock + Hash 完成。

## 9. Evaluation

正式实验矩阵：

| 配置 | Text | Structured | State | Memory |
|---|---:|---:|---:|---:|
| A | ✓ | × | × | × |
| B | × | ✓ | × | × |
| C | × | ✓ | ✓ | × |
| D | × | ✓ | ✓ | ✓ |

四组使用相同 Agent、任务、provider、随机种子和成功判定。

## 10. Dashboard

MVP Dashboard 不负责执行复杂逻辑，只负责：

- 读取运行结果
- 展示 Agent 时间线
- 展示消息与 refs
- 展示状态传输
- 展示 Memory 命中
- 展示 A/B/C/D 对比指标

优先实现静态 HTML/JS + 简单 API，避免额外前端工程复杂度。
