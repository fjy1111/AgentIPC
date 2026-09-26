# 07 — Codex Verification Runbook

Codex 在本项目中的默认职责是执行与验证，而不是自动开发。

## 1. 验证前检查

必须记录：

```bash
git status --short
git diff --stat
python --version
```

如任务限制修改文件，应检查实际 diff 是否越界。

## 2. 默认验证顺序

```text
1. targeted test
2. related module tests
3. existing core smoke tests
4. optional lint/type check
5. task-specific command
```

不要一开始就运行超大 benchmark。

## 3. 失败时规则

- 不自动改代码。
- 保留第一现场错误。
- 给出最小复现命令。
- 区分代码错误、测试错误、环境错误、依赖错误。
- 如果测试卡住，记录最后输出与可疑资源。

## 4. Task Report 模板

```markdown
# TASK REPORT — <TASK-ID>

## Verdict
PASS | REWORK | BLOCKED

## Environment
- OS:
- Python:
- Commit:
- Working tree:

## Changed Files
- expected:
- actual:
- scope violation: yes/no

## Commands Run
1. `...`
2. `...`

## Results
- targeted tests: x passed / y failed
- related tests: ...
- smoke tests: ...

## Failures
### Failure 1
- command:
- error:
- minimal reproduction:
- likely category: implementation/environment/design/test

## Contract Check
- public interface unchanged: yes/no/unknown
- new dependency introduced: yes/no
- model/dataset downloaded: yes/no

## Notes
...
```

## 5. PASS 条件

Codex 只有在以下条件都满足时才能给 PASS：

- 指定验收命令成功
- 修改范围没有明显越界
- 无新增未解释回归失败
- 没有违反公共契约
- 没有偷偷引入禁止依赖

否则必须是 REWORK 或 BLOCKED。
