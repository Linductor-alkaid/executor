---
title: 任务如何穿过执行器
description: 以当前实现说明普通与实时任务的调度、执行、状态和关闭路径。
---

# 任务如何穿过执行器

## 学习目标

建立调试和性能分析所需的执行心智模型，同时把当前内部路径与稳定公开 API 明确区分。

## 普通任务路径

```text
Executor Facade
  → ExecutorManager
    → ThreadPoolExecutor (IAsyncExecutor)
      → ThreadPool
        → PriorityScheduler → TaskDispatcher
          → WorkerLocalQueue / worker 执行
```

Facade 负责 future、failure event、任务图和生命周期入口；manager 持有默认异步执行器及注册表；`ThreadPoolExecutor` 将公开异步接口映射到线程池。当前线程池从优先级调度器取任务，分发到 worker 本地队列，并结合负载信息尝试工作窃取。线程池可依据配置进行动态扩缩容，关闭时停止接收/分发并按 `shutdown(wait_for_tasks)` 决定是否等待。

这些内部模块帮助解释“为什么队列堆积”或“为什么任务在另一 worker 执行”，但不是替代 `submit()`、`get_completion_status()` 和监控 API 的用户入口。

## 实时任务路径

```text
Executor Facade
  → ExecutorManager → RealtimeThreadExecutor
    → 周期触发（内置 sleep_until 或 ICycleManager）
      → cycle_callback → 有界实时队列 drain → 状态更新
```

每个周期先执行 `cycle_callback`，再处理已入队工作。`max_tasks_per_cycle` 限制单周期 drain；回调耗时超过周期会增加 `cycle_timeout_count`，并跳过错过的节拍重新调相，避免追赶式抖动。实时队列使用预分配任务包装和有界容量，因此入队可能因未运行、空任务、队列满或对象池耗尽被拒绝；这些计数应通过 `RealtimeExecutorStatus` 观察。

## 关闭与状态

普通 `wait_for_completion_ex()` 只等待默认异步执行器，不能证明实时 callback 或实时队列已经完成。实时流水线需要自己的确认、阶段门或停止顺序。状态 API 是当前运行情况的快照；调试内部路径时应以它们和 failure/comm events 为证据，而不是依赖线程调度偶然顺序。

## 实现说明，不是承诺

路径中的 `ThreadPool`、`PriorityScheduler`、`TaskDispatcher`、worker 队列与实际窃取策略可能在不改变公开行为的版本中重构。把它们当作“当前发生什么”的说明，而不是可以直接调用的 API。

## 下一步阅读

需要高频单消费者任务聚合时阅读[无锁与性能实验](/zh/advanced/lockfree-and-performance)。
