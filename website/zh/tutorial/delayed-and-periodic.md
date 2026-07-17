---
title: 延迟重试与健康检查
description: 用 Facade 的延迟和软周期任务处理设备重试与后台健康检查。
---

# 延迟重试与健康检查

## 学习目标

用 `submit_delayed()` 安排一次设备重试，用 `submit_periodic()` 运行健康检查，并通过状态查询与 `cancel_task()` 结束周期工作。

## 场景问题

设备暂不可用时，立即重试只会制造额外负载；另一方面，后台健康检查需要隔一段时间重复运行，但必须能被停止并观察到是否真正运行过。

## 推荐方案

这两种工作都使用 `Executor` Facade，而不是直接操作底层线程池：

<<< @/../examples/tutorial/03_delayed_periodic.cpp{1-31}

完整源码：[`examples/tutorial/03_delayed_periodic.cpp`](https://github.com/Linductor-alkaid/executor/blob/master/examples/tutorial/03_delayed_periodic.cpp)。

```bash
./build/examples/tutorial/tutorial_03_delayed_periodic
```

## 预期输出

```text
retry complete
health checks=6
periodic status=running, cancelled=yes
```

健康检查次数受调度影响，实际值可能不同；关键事实是取消前状态存在且 `execution_count > 0`。

## 状态与取消

- `submit_periodic()` 返回任务 ID；保存它，才能调用 `cancel_task()`。
- `get_periodic_task_status(id)` 返回单个仍在注册的任务状态；取消后会得到空结果。
- `get_all_periodic_task_status()` 适合监控页或关闭前检查所有仍注册的周期任务。
- 延迟任务返回 `future`，因此其返回值、执行异常和拒绝提交仍由 `get()` 观察。

## 为什么这样做

延迟与周期调度属于 Facade 的能力：调用方不需要取得线程池对象或管理计时器线程。周期任务没有逐次 `future` 可保存，至少应保留任务 ID、查看执行/失败状态，并在不再需要时取消它。

## 失败如何观察

周期回调的异常不会由调用 `submit_periodic()` 的位置返回；运行中的任务应检查周期状态，并在长期服务中配置失败状态或回调。`cancel_task()` 返回 `false` 表示 ID 已不存在、已取消或不是当前实例的周期任务，不能把它当作已经安全停止的证明。

## 何时不选

`submit_periodic()` 是软周期后台调度：它不承诺控制循环的准点性，也不表达单周期预算。需要处理 CAN 或电机控制等严格周期工作时，应使用专用实时执行器。

## 下一步阅读

[批量处理一组传感器帧](/zh/tutorial/batch)处理同一时刻提交很多相似工作的情况。
