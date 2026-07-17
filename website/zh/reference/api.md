---
title: API 参考
description: 公开 API 的模块入口与稳定性边界。
---

# API 参考

首发不复制整份 API 文档。请以仓库的 [`docs/API.md`](https://github.com/Linductor-alkaid/executor/blob/master/docs/API.md) 为完整签名、配置和兼容语义的唯一事实源。

| 模块 | 优先入口 |
| --- | --- |
| 普通任务与生命周期 | `Executor` Facade |
| 配置、状态与失败 | `ExecutorConfig`、状态查询、failure callback |
| 通信组件 | `executor::comm` |
| 实时执行器 | Facade 注册、启动、推送与状态查询 |
| GPU | GPU 注册、提交与调度 |
| 高级接口 | 独立实例、Manager 与执行器指针 |

教程会在需要时链接到具体 API；不确定从哪里开始时，先看[选择提交接口](/zh/guides/choosing-submit-api)。
