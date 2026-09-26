# 06 — Multi-GPT Development Workflow

## 1. 基本工作流

```text
总规划设计师
    │
    ├── 指定 Task ID
    │
    ▼
实现 GPT
    │
    ├── 浏览仓库
    ├── 只完成单 Task
    ├── 输出修改说明
    └── 提交代码
    │
    ▼
Codex
    │
    ├── 安装/运行
    ├── 测试
    ├── 收集日志
    └── 输出任务报告
    │
    ▼
总规划设计师
    │
    ├── 判断 PASS / REWORK / DESIGN CHANGE
    └── 分配下一 Task
```

## 2. 一个实现 GPT 一次只收到一个 Task

推荐提示词：

```text
你现在负责 AgentIPC 的单一开发任务 <TASK-ID>。

开始前必须阅读：
- AGENTS.md
- docs/00_PROJECT_CHARTER.md
- docs/01_ARCHITECTURE.md
- docs/03_DEVELOPMENT_PLAN.md 中 <TASK-ID>
- docs/04_MODULE_CONTRACTS.md

规则：
1. 只完成 <TASK-ID>，不要继续下一个任务。
2. 只修改该任务允许修改的文件。
3. 不改变公共接口；发现设计问题就报告，不自行扩大修改。
4. 不下载模型/数据集，不引入未经批准的新框架。
5. 为新增行为添加对应测试。
6. 完成后输出：修改文件、实现内容、测试建议、已知限制。

现在先浏览仓库相关内容，然后完成 <TASK-ID>。
```

## 3. Codex 提示词模板

```text
你现在只负责运行与验证 AgentIPC 的 <TASK-ID>，默认不要修改代码。

请先阅读：
- AGENTS.md
- docs/03_DEVELOPMENT_PLAN.md 中 <TASK-ID>
- docs/05_TEST_ACCEPTANCE.md
- docs/07_CODEX_RUNBOOK.md

然后：
1. 检查 git diff，确认修改范围。
2. 执行任务指定测试。
3. 执行必要的相关回归测试。
4. 不下载大型模型或数据集。
5. 不自动修复失败。
6. 按 docs/07_CODEX_RUNBOOK.md 模板返回 TASK REPORT。
```

## 4. 状态流转

每个 Task 只能是：

```text
TODO
IMPLEMENTING
READY_FOR_VERIFY
PASS
REWORK
BLOCKED
SUPERSEDED
```

只有 Codex 验证成功后才标 `PASS`。

## 5. 失败分类

### Implementation failure

实现与契约不符、测试失败、明显 bug。

处理：原实现 GPT 或新修复 GPT 处理同 Task。

### Environment failure

例如依赖安装、系统能力、端口冲突。

处理：先修复环境/脚本，不要修改业务逻辑掩盖问题。

### Design failure

任务无法在现有公共契约下合理实现。

处理：停止编码，由总规划设计师更新契约和受影响任务。

## 6. 禁止并行修改同一公共文件

以下文件属于高冲突区：

- `protocol/envelope.py`
- `runtime/context.py`
- `runtime/orchestrator.py`
- `evaluation/metrics.py`
- `config.py`

同一时间只允许一个 Task 修改这些文件。

## 7. 合并顺序

严格按依赖顺序合并。实现 GPT 可以提前生成代码，但未满足依赖的任务不应直接进入主分支。
