---
title: 有界等待与状态快照
description: 以有界等待、WaitResult 和状态快照安全完成一次流水线收尾。
---

# 有界等待与状态快照

## 学习目标

在阶段切换或关闭前，以超时为边界等待异步任务，并根据 `WaitResult`、完成状态和执行器状态作出下一步处理。

## 场景问题

规划流水线准备切换地图或退出。无限期等待会让调用线程失去响应；只看一个布尔值又不足以诊断是队列仍有工作、任务正在运行，还是执行器已经停止。

## 推荐方案

新代码优先使用 `wait_for_completion_ex(timeout)`：

```cpp
auto result = executor.wait_for_completion_ex(std::chrono::milliseconds{200});
if (!result.completed) {
    // result.timed_out 为 true；result.status 是超时瞬间的状态快照。
    std::cerr << "pending=" << result.status.pending_tasks << '\n';
}
```

`WaitResult` 同时给出 `completed`、`timed_out`、传入的 timeout 和 `CompletionStatus`。超时时，Facade 会记录 `WaitTimeout`，可在可靠性专题中通过失败状态和回调统一处理。

## 选择等待接口

| 目的 | 接口 | 结果 |
| --- | --- | --- |
| 兼容旧调用方 | `wait_for_completion()` | 最多等待默认时长；超时不抛出。 |
| 只要完成/超时判断 | `try_wait_for_completion(timeout)` | `bool`。 |
| 以任意 chrono 时长表达边界 | `wait_for_completion_for(timeout)` | `bool`。 |
| 需要诊断超时 | `wait_for_completion_ex(timeout)` | `WaitResult` 与状态快照。 |

`is_idle()` 用于快速判断默认异步执行器当前是否空闲；`get_completion_status()` 提供初始化、排队、活跃和待完成数量的快照。需要确认执行器生命周期时，再查询 `get_async_executor_status()`。

## 为什么这样做

有界等待把“尚未完成”变成调用方可以处理的业务状态，而不是无期限卡住。状态快照还能区分执行器未初始化、队列积压与运行中任务，为重试、降级或故障报告保留事实。

## 失败如何观察

超时不是任务异常，也不代表任务已经取消；检查 `result.timed_out` 和 `result.status`，并查询失败状态中的 `wait_timeout_count`。单个任务的返回值和异常仍应由各自的 `future.get()` 处理。

## 正确收尾顺序

1. 停止产生新任务，例如取消不再需要的周期任务。
2. 以业务可接受的 timeout 调用 `wait_for_completion_ex()`。
3. 完成时调用 `shutdown(true)`；超时时记录快照并按业务策略重试、降级或调用 `shutdown(false)`。

不要在超时后假设任务已经停止：超时只说明它们尚未全部完成。

## 下一步阅读

至此，普通业务流水线只使用了 Facade。继续了解统一失败回调与长期运行诊断，请阅读[可靠性专题](/zh/reliability/)；需要严格控制周期时，进入实时与通信专题。
