---
title: 高级与原理
description: 在 Facade 的默认道路不够用时，再进入资源隔离和底层执行路径。
---

# 高级与原理

大多数应用从 `Executor` Facade 开始。需要独立资源隔离时使用独立 `Executor` 实例；需要自定义周期源、直接控制实时/GPU 执行器或分析队列实现时，再进入公开高级接口。

一次普通任务的大致路径是：

```text
Executor → ExecutorManager → ThreadPoolExecutor → 调度与 worker
```

内部实现细节不构成稳定 API。设计事实源见 [`docs/design/executor.md`](https://github.com/Linductor-alkaid/executor/blob/master/docs/design/executor.md)、[`docs/design/lockfree_user_api.md`](https://github.com/Linductor-alkaid/executor/blob/master/docs/design/lockfree_user_api.md) 和 [`docs/API.md`](https://github.com/Linductor-alkaid/executor/blob/master/docs/API.md)。
