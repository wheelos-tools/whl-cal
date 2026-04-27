---
audience: user
stability: stable
P26-04-27
---


# Docs vs Context — 使用者文档 与 知识库的职责划分

目标：明确两套文档的角色，方便维护、搜索和面向不同读者的需求。

1. 定义与定位

- docs/（用户文档、运行手册、设计文档）
  - 受众：集成工程师、QA、最终用户、开发者的快速上手需求。
  - 内容特点：可执行的步骤（Quick Start）、示例命令、关键输出位置、验收规则、快速排障指南、受控的设计文档（设计摘要 & 参数说明）。
  - 要求：保持简洁、可复制、包含最小示例，且每次变更应伴随 smoke-test 验证并更新 `last_tested` 字段（若有）。

- context/（知识库，长期演化）
  - 受众：架构师、算法工程师、维护者。
  - 内容特点：理论推导、设计选择理由、验证结论、实验日志、验证点、长期待办（todos）、演化记录。
  - 要求：记录为什么做出某个设计、何时验证、哪些数据支持结论；当 docs 中做“简化说明”时，把详细论据放在 context 并链接回来。

2. 建议的协作与同步策略

- 每次对 pipeline 或关键参数作出变动：
  - 更新 docs/ 中的用户可见使用步骤（如果影响 CLI 或 output），并运行 smoke test；
  - 在 context/ 中写入变更原因、实验对比和验证数据的引用；
  - 在 docs 的变更说明中加入一句："See context/… for rationale and validation" 并在 context 中反向链接到 docs 的版本/提交。

- 文档元数据（可选）
  - 在 docs 文件头部添加简短元数据：audience: user|dev，stability: stable|experimental，last_tested: YYYY-MM-DD。
  - 这能帮助快速判断文档是否已通过 smoke-test 或需要人工复核。

3. 目录结构建议（已部分实现）

- docs/
  - quickstart_index.md  # 入口
  - lidar2camera_quickstart.md  # 用户快速上手 + 验收
  - lidar2camera_test_quickstart.md  # smoke 测试说明
  - lidar2camera_design.md  # 设计 + 参数（开发者阅读）
  - docs_vs_context.md  # 本文件

- context/
  - lidar2camera_context.md  # 深度背景、验证结论与后续任务
  - knowledge_base/  # 更广的工程 / 验证记录

4. 最终建议（短期可执行）

- 把 docs 作为“受控、可运行”的使用手册；把 context 作为“可增长的知识库”。
- 在 PR 模板或 CI 流程中加入：如果修改了接口或默认参数，必须更新 docs 并运行 smoke test。CI 可选通过轻量 workflow 自动验证。
- 把关键验收阈值写到 config.yaml 的 metrics 部分，并在 docs 中摘录以便用户快速查看。

5. 测试与验证

- 本仓库提供 `tools/run_lidar2camera_smoke.py`，用于合成数据的快速回归。文档里应包含一条 `PYTHONPATH=.` 运行说明（已在 quickstart 中加入）。


