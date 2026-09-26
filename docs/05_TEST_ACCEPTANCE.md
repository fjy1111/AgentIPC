# 05 — Test and Acceptance Strategy

## 1. 测试层级

### Unit

单模块，不启动完整 Runtime。

覆盖：
- schema
- codec
- registry
- shared memory
- artifact store
- sqlite memory
- vector search
- providers
- sandbox
- metrics

### Integration

至少两模块真实连接。

重点：
- Envelope encode/decode
- Registry + Router
- StateHub producer -> consumer
- Artifact producer -> consumer
- Memory write -> next task retrieve

### End-to-End

完整：

```text
Planner -> Retriever -> Executor -> Summarizer
```

并验证 text / structured 模式。

### Stability

连续至少 10 轮：
- 无资源泄漏导致崩溃
- shared memory 正确释放
- SQLite 可持续写入
- trace 不互相覆盖

## 2. 每个任务的最小验收结构

一个实现任务只有满足下面三项才算完成：

1. 任务指定测试通过。
2. 原有测试没有新增失败。
3. Codex 任务报告中无“未解释失败”。

## 3. State 验收关键点

禁止只写这种测试：

```python
assert ref.uri.startswith("shm://")
```

必须验证：

```text
producer vector
  -> put
  -> ref
  -> resolve
  -> downstream calculation
  -> expected result
```

并校验 shape/dtype/checksum。

## 4. Memory 验收关键点

必须验证跨任务持久化：

```text
Task A write
close/reopen store（至少一个测试）
Task B retrieve
mark_used
reuse_count changes
```

## 5. Text / Structured 公平性验收

同一测试用例中：

- 业务最终结果应一致或满足同一判定标准。
- Agent 顺序一致。
- provider 一致。
- structured 不允许偷偷跳过业务步骤。

## 6. Benchmark 结果验收

结果目录至少包含：

```text
results/<timestamp>-<suite>/
├── config.json
├── env.json
├── raw.jsonl
├── summary.json
└── report.md
```

结果目录不得默认覆盖已有目录。

## 7. openEuler 验收

最终至少执行：

```bash
python3 --version
cat /etc/os-release
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
pytest -q
agentipc doctor
agentipc demo --provider mock
agentipc benchmark --suite smoke --provider mock
```

若系统 Python 不满足版本要求，安装脚本必须给出明确提示，而不是静默失败。

## 8. 不允许伪造的证据

以下内容不能用作最终性能结论：

- 手工填写的 summary
- 只有 README 数字没有 raw result
- 不同模型之间直接比较通信优化效果
- 不同任务集之间直接计算节省率
- 失败运行被人为删除后只保留成功样本
