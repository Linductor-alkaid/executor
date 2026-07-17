---
title: 可靠性
description: 让异常、超时、拒绝与关闭行为保持可观察。
---

# 可靠性

可靠性不是只多写一个 `try/catch`。调用方需要区分立即结果、提交拒绝、任务失败、等待超时和长期趋势，并为每类事件保留正确的观察出口。

1. [失败可观察性](/zh/reliability/failure-observability)：组合 `future`、failure callback、累计计数和最近事件。
2. [监控与采样](/zh/reliability/monitoring)：按开销和所需精度收集任务统计。

通信组件的 drop、覆盖、stale 与 missed phase 属于组件本地事件，不会自动进入 `ExecutorFailureStatus`；请在[通信组件选型](/zh/guides/choosing-communication)中为它们设置对应观察路径。
