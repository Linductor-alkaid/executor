---
title: 高级与原理
description: 在 Facade 的默认道路不够用时，再进入资源隔离和底层执行路径。
---

# 高级与原理

大多数应用从 `Executor` Facade 开始。需要独立资源隔离时使用独立 `Executor` 实例；需要自定义周期源、直接控制实时/GPU 执行器或分析队列实现时，再进入公开高级接口。

1. [何时使用高级逃生口](/zh/advanced/escape-hatches)：实例隔离、`ExecutorManager` 与直接执行器指针的责任边界。
2. [接入自定义周期源](/zh/advanced/custom-cycle-manager)：实现 `ICycleManager` 并管理其生命周期。
3. [任务如何穿过执行器](/zh/advanced/execution-paths)：普通与实时任务的当前实现路径。
4. [无锁与性能实验](/zh/advanced/lockfree-and-performance)：`LockFreeTaskExecutor`、内部队列和 benchmark 的适用边界。

以下“原理”页面描述当前实现，帮助调试与性能分析；除 `include/executor/` 下公开接口外，不承诺 `src/` 内类型、数据结构或调度细节的兼容性。设计事实源见 [`docs/design/executor.md`](https://github.com/Linductor-alkaid/executor/blob/master/docs/design/executor.md)、[`docs/design/lockfree_user_api.md`](https://github.com/Linductor-alkaid/executor/blob/master/docs/design/lockfree_user_api.md) 和 [`docs/API.md`](https://github.com/Linductor-alkaid/executor/blob/master/docs/API.md)。
