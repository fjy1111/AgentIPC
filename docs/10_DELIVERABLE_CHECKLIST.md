# 10 — Final Deliverable Checklist

## A. 源码

- [ ] 完整 Git 仓库
- [ ] `pyproject.toml`
- [ ] `.env.example`
- [ ] 默认配置
- [ ] 单元测试
- [ ] 集成测试
- [ ] 场景数据
- [ ] openEuler 脚本

## B. 系统功能

- [ ] 4 Agent 协同
- [ ] text mode
- [ ] structured mode
- [ ] handshake/register/discover
- [ ] StateRef + Shared Memory
- [ ] ArtifactRef
- [ ] Memory store/retrieve/reuse
- [ ] CodeAct
- [ ] 10+10 连续任务
- [ ] Dashboard

## C. 实验

- [ ] A Text Baseline
- [ ] B Structured
- [ ] C Structured + State
- [ ] D Full System
- [ ] 固定 seed
- [ ] 相同任务条件
- [ ] raw results
- [ ] summary
- [ ] 环境信息
- [ ] 通信开销
- [ ] state 次数/字节
- [ ] memory hit
- [ ] latency
- [ ] repeated work
- [ ] success rate

## D. 文档

- [ ] README
- [ ] 系统设计文档
- [ ] 协议设计
- [ ] 状态交换设计
- [ ] 共享记忆设计
- [ ] 部署文档
- [ ] 实验报告
- [ ] 测试报告
- [ ] 已知限制
- [ ] 参考与许可证说明

## E. openEuler

- [ ] openEuler 24.03-LTS-SP3 环境信息截图/日志
- [ ] 安装成功
- [ ] pytest 成功
- [ ] demo 成功
- [ ] benchmark 成功
- [ ] Dashboard 可启动

## F. 演示视频

建议顺序：

1. 30 秒：问题背景与核心架构
2. 60 秒：四 Agent + structured protocol
3. 45 秒：StateRef / Shared Memory
4. 45 秒：Memory 跨任务复用
5. 60 秒：A/B/C/D Dashboard
6. 30 秒：openEuler 环境与一键运行
7. 30 秒：总结与限制
