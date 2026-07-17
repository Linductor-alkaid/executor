---
title: 让控制命令优先
description: 使用 submit_priority 让少量控制工作在普通分析之前进入队列。
---

# 让控制命令优先

## 学习目标

在同一条机器人流水线中，让紧急 `ControlCommand` 比普通帧分析更早排队，同时保留 `future` 的结果和异常观察路径。

## 场景问题

传感器帧解析可以稍后开始，但一条停止或限速命令不应被普通分析任务长期挡在队列后面。

## 推荐方案

使用 `submit_priority(priority, task)`。优先级从 `0`（低）到 `3`（关键）；普通任务使用默认优先级。下面的示例以单工作线程演示两类工作均可正确完成。

<<< @/../examples/tutorial/02_priority.cpp{4,7-19,21-24}

完整源码：[`examples/tutorial/02_priority.cpp`](https://github.com/Linductor-alkaid/executor/blob/master/examples/tutorial/02_priority.cpp)。

```bash
./build/examples/tutorial/tutorial_02_priority
```

## 预期输出

```text
priority tasks=analysis,control
```

## 为什么这样做

- `submit_priority()` 仍返回 `future`；调用 `get()` 可获得结果，也可重新抛出任务异常。
- 优先级只影响等待队列的取出顺序；已经开始运行的低优先级任务不会被抢占。
- 多工作线程、已有执行中的任务和同优先级 FIFO 都会影响实际观察到的完成顺序，因此不要把这个示例的输出顺序当作调度时序证明。

## 失败如何观察

- 空任务或关闭后提交会形成可观察的拒绝/异常结果；不要忽略返回的 `future`。
- 控制任务本身失败时，在其 `future.get()` 处处理；长期运行的统一失败统计将在[可靠性专题](/zh/reliability/)展开。

## 何时不选

需要确定的周期、抖动预算、CPU 亲和性或运行中抢占时，不要提高普通线程池优先级来模拟实时性；请进入后续的专用实时线程教程。

## 下一步阅读

[延迟重试与健康检查](/zh/tutorial/delayed-and-periodic)处理“现在不做”和“定期做”两种新问题。
