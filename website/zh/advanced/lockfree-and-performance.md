---
title: 无锁与性能实验
description: 在明确的单消费者高频场景评估 LockFreeTaskExecutor，并正确理解内部队列、对象池和 benchmark。
---

# 无锁与性能实验

## 学习目标

判断 `LockFreeTaskExecutor` 是否比 Facade 更适合一个高频聚合场景，并用正确的统计、压力测试和 benchmark 验证结论。

## 何时使用 LockFreeTaskExecutor

它面向多生产者、单消费者的任务聚合，例如高频日志、异步事件或性能敏感的单消费者分发：

```cpp
executor::LockFreeTaskExecutor queue(1024);
queue.start();

if (!queue.push_task([] { process_event(); })) {
    // 空任务、已停止、队列满或对象池耗尽：执行背压策略。
}
queue.stop();
```

完整示例：[`examples/lockfree_task_executor_example.cpp`](https://github.com/Linductor-alkaid/executor/blob/master/examples/lockfree_task_executor_example.cpp)。它不是普通 `Executor` 的“更快替代品”：没有 future 返回值、任务依赖、统一 failure callback 或多 worker 执行能力。多数业务任务仍应先使用 Facade。

## 生命周期与可观察性

调用 `start()` 启动唯一消费者，`stop()` 停止并等待线程。`push_task()` 失败必须由调用方处理；检查 `pending_count()`、`processed_count()`、`exception_count()`、`rejected_empty_count()` 与 `get_queue_stats()`。任务异常由 worker 捕获并累计；如需异常对象，注册 `set_exception_handler()`，且 handler 本身要短小、线程安全、不抛出。

## 当前内部结构

### 生产、消费与停止

每次 `push_task()` 先拒绝空函数，再通过 `active_pushes_` 登记生产者，随后从固定容量 `ObjectPool` 获取 wrapper，写入 `std::function` 后尝试压入有界 `LockFreeQueue`。任一环节失败都返回 `false`；调用方要把它视作明确背压，而不是重试到无限期。

唯一消费者一次最多 `pop_batch(32)` 项，逐个执行并归还 wrapper。空队列时采用三级退避：短暂 CPU pause、自旋后 `yield`，再退化为 1 微秒 sleep；停止时先阻止新 producer、等待已登记 producer 离开，再停止 worker 并 drain 剩余队列。这个顺序避免 producer 在最终 drain 后继续放入任务。

### “无锁”的实际范围

当前实现中的 `LockFreeQueue` 提供有界入队/出队；`ObjectPool` 预分配任务包装以避免热路径分配。对象池的正确性优先于“所有东西都无锁”：其 acquire/release 使用 mutex 保护 free list，以避免 ABA、外来指针和重复释放问题。异常 handler 的读取/更新同样受 mutex 保护。换言之，队列是无锁的，执行器整体不是 lock-free 进度保证；这是用可验证的内存回收正确性换取的工程选择。

线程池内的 `WorkerLocalQueue`、工作窃取和对象池同样是实现细节，不能从 `src/` 直接集成。

## 如何验证性能

| 问题 | 验证方法 |
| --- | --- |
| 是否存在数据竞争或生命周期错误？ | TSAN、并发压力测试和边界测试。 |
| 满队列时业务是否正确降级？ | 断言失败返回、drop/拒绝计数与恢复行为。 |
| 特定硬件上是否更快？ | 固定构建类型、CPU、负载、线程数和数据规模后运行 benchmark。 |
| 是否改善真实端到端延迟？ | 测量提交、排队、执行和尾延迟，而非只看单次 microbenchmark。 |

性能数字只在测量环境内成立。记录版本、编译器、构建类型、硬件、频率/功耗策略、任务体、并发度与统计方法；不要把一次基准结果写成接口承诺。

对应源码入口：[`src/executor/lockfree_task_executor.cpp`](https://github.com/Linductor-alkaid/executor/blob/master/src/executor/lockfree_task_executor.cpp)、[`src/executor/util/lockfree_queue.hpp`](https://github.com/Linductor-alkaid/executor/blob/master/src/executor/util/lockfree_queue.hpp) 和 [`src/util/object_pool.hpp`](https://github.com/Linductor-alkaid/executor/blob/master/src/util/object_pool.hpp)。

## 下一步阅读

回到[API 参考](/zh/reference/api)确认公开接口；性能报告和内部优化历史可作为事实来源，但不是用户程序的依赖契约。
