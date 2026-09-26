# 09 — Reference and Originality Policy

## 1. 可参考内容

允许学习公开项目中的：

- 模块分层思想
- 常见设计模式
- 实验组织方式
- 通用工程实践
- 测试策略
- 文档结构
- 标准库和第三方库的常规使用方式

参考项目包括但不限于：

- https://github.com/stander1/agent
- https://github.com/tkj-lp/low-cost-mult-agent-memory
- https://github.com/xuke5776-debug/os-dasai-v2

## 2. 本项目从参考项目吸收的“思想级”经验

### stander1/agent

- Runtime 与业务 Agent 分离
- StateRef / MemoryRef
- 可观测性与可复现实验
- 交付工程意识

### low-cost-mult-agent-memory

- 多 Agent 执行链可视化
- 连续任务与 CodeAct 场景
- 前端展示实验过程

### os-dasai-v2

- mock-first
- 需求追踪矩阵
- 结构化协议 / State / Memory 分层
- A/B/C/D 消融实验方法

## 3. 禁止行为

不得：

- 整文件复制后重命名
- 大段代码复制后仅改变量名
- 复制对方协议字段和类结构作为本项目正式协议
- 复制对方任务数据作为本项目原创任务
- 复制对方报告与 README 文案
- 删除许可证或来源信息后使用代码
- 以“让评委看不出来”为目标处理来源

## 4. 本项目独立性来源

AgentIPC 通过以下设计建立独立实现：

- 独立协议 `agentipc/0.1`
- 独立 AgentEnvelope 契约
- 独立 SharedMemory StateHub
- 独立 ArtifactStore
- 独立三类型 Memory 模型
- 独立 Knowledge Chain / CodeAct Chain 任务
- 独立开发任务分解与验收体系

## 5. 引用开源代码时

如果未来确实需要直接使用小段第三方代码：

1. 先确认许可证兼容。
2. 保留原许可证要求。
3. 在对应文件或 NOTICE 中注明来源。
4. 不把第三方代码描述为自主创新点。
