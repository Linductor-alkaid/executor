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

### 从 `submit()` 到 worker

1. `Executor::submit()` 先向默认 `IAsyncExecutor` 提交一个包装任务。包装任务兑现 `promise`，在用户函数抛异常时把异常写回 future，并把异常继续抛给底层执行器，使 future 与服务级 failure 观察保持一致。
2. 默认 manager 首次取得异步执行器时可用 `std::call_once` 进行懒初始化；显式初始化已经发生或 shutdown 后，Facade 将相应拒绝变成可诊断结果。
3. `ThreadPoolExecutor` 在自身 mutex 下取得一个 `shared_ptr<ThreadPool>` 快照，再把任务交给线程池。这一快照使 stop 与提交并发时，提交路径不会解引用已释放的线程池对象。
4. `ThreadPool` 把任务按 `CRITICAL → HIGH → NORMAL → LOW` 放入 `PriorityScheduler`。当前实现为四个优先级队列分别设置 mutex；优先级保证的是取队列顺序，不是运行中抢占。
5. `TaskDispatcher` 从调度器批量取任务，以 `LoadBalancer` 选择目标 worker，并推入该 worker 的本地队列。若 resize 使 worker ID 失效，或本地队列已满，任务会重新入调度器，而不是在“已出队、未入本地队列”的窗口静默丢失。
6. worker 先从自己的队列取任务；空闲时根据负载优先从负载更高的 worker 窃取，无法形成差异时才回退到随机起点。任务函数完成后，线程池更新完成/失败与耗时统计，并唤醒等待者。

Facade 负责 future、failure event、任务图和生命周期入口；manager 持有默认异步执行器及注册表；`ThreadPoolExecutor` 将公开异步接口映射到线程池。当前线程池从优先级调度器取任务，分发到 worker 本地队列，并结合负载信息尝试工作窃取。

### 同步点与不变量

当前普通线程池不是“全无锁”实现：优先级队列使用分级 mutex，默认 `WorkerLocalQueue` 的 push/pop/steal 使用 mutex，local-queue 向量与 resize 通过 `shared_mutex` 协调。这样做换取了 resize、回收与无任务丢失的清晰边界；启用的无锁 worker queue 也是可选实现路径，不能从公开 API 推断其必然存在。

等待完整性的核心不变量是：已接受任务最终要么执行并计入完成/失败，要么在拒绝/超时路径让其 future 成为就绪异常；dispatcher 的回入队逻辑正是为维护这一不变量。`wait_for_completion_ex()` 观察的是默认异步执行器的快照，不是全进程所有后台活动。

这些内部模块帮助解释“为什么队列堆积”或“为什么任务在另一 worker 执行”，但不是替代 `submit()`、`get_completion_status()` 和监控 API 的用户入口。

## 实时任务路径

```text
Executor Facade
  → ExecutorManager → RealtimeThreadExecutor
    → 周期触发（内置 sleep_until 或 ICycleManager）
      → cycle_callback → 有界实时队列 drain → 状态更新
```

### 一个周期内发生什么

1. 内置周期源以 `sleep_until(next_cycle_time)` 驱动；若注入 `ICycleManager`，外部周期源负责触发同一轮 `cycle_loop()`。
2. 每轮先运行 `cycle_callback`。回调异常被异常处理器捕获，避免杀死周期线程；这不代表业务成功，应用仍需自己的失败观察协议。
3. 然后 `process_tasks()` 从 MPSC 队列最多取出 `max_tasks_per_cycle` 项，执行任务并归还包装对象。预算为 `0` 才表示不限；默认预算让突发生产不会在一个周期内无限 drain。
4. 统计更新使用原子计数、整数 EMA 和 CAS 更新最大周期时间；超过 `cycle_period_ns` 时增加 `cycle_timeout_count`。若下一计划时刻已经落后，循环重设为“现在加一个周期”，跳过错过节拍而不是零等待追赶。

### 接受、拒绝与停止竞态

`push_task_ex()` 先登记一个 in-flight producer，再检查 `running_`，从预分配池获得 wrapper，并尝试入 MPSC 队列。空任务、未运行、对象池耗尽和队列满分别累加可见拒绝计数。停止路径先禁止新生产、等待已登记 producer 退出，再让单消费者 drain，目标是不让“已接受任务”在最终 drain 之后凭空出现。

队列本身是有界 MPSC 无锁队列，但这不意味着整条实时路径没有锁或分配：当前 `ObjectPool` 为避免 ABA、外来指针和重复释放，用 mutex 保护 free list；用户 callback、异常处理器和外部 `ICycleManager` 也可能引入锁、系统调用或分配。因此文档中的“避免锁、无限等待与运行期分配”是实时设计目标和调用方约束，不是对当前每条内部指令的绝对承诺。真实周期预算必须以 trace、状态和目标平台测量验证。

## 关闭与状态

普通 `wait_for_completion_ex()` 只等待默认异步执行器，不能证明实时 callback 或实时队列已经完成。实时流水线需要自己的确认、阶段门或停止顺序。状态 API 是当前运行情况的快照；调试内部路径时应以它们和 failure/comm events 为证据，而不是依赖线程调度偶然顺序。

## 实现说明，不是承诺

路径中的 `ThreadPool`、`PriorityScheduler`、`TaskDispatcher`、worker 队列与实际窃取策略可能在不改变公开行为的版本中重构。把它们当作“当前发生什么”的说明，而不是可以直接调用的 API。

对应源码入口：[`src/executor/executor.cpp`](https://github.com/Linductor-alkaid/executor/blob/master/src/executor/executor.cpp)、[`src/executor/executor_manager.cpp`](https://github.com/Linductor-alkaid/executor/blob/master/src/executor/executor_manager.cpp)、[`src/executor/thread_pool/thread_pool.cpp`](https://github.com/Linductor-alkaid/executor/blob/master/src/executor/thread_pool/thread_pool.cpp)、[`src/executor/thread_pool/task_dispatcher.hpp`](https://github.com/Linductor-alkaid/executor/blob/master/src/executor/thread_pool/task_dispatcher.hpp) 与 [`src/executor/realtime_thread_executor.cpp`](https://github.com/Linductor-alkaid/executor/blob/master/src/executor/realtime_thread_executor.cpp)。阅读源码时请同时参考测试：它们描述外部可依赖的行为，而不是由本文固定内部结构。

## 下一步阅读

需要高频单消费者任务聚合时阅读[无锁与性能实验](/zh/advanced/lockfree-and-performance)。
