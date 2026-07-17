---
title: 批量处理传感器帧
description: 在需要或不需要逐项结果时选择 submit_batch、submit_batch_no_future 与 submit_batch_priority。
---

# 批量处理传感器帧

## 学习目标

一次提交一批同类处理工作，并根据是否需要逐项结果选择带 `future` 或 fire-and-forget 路径。

## 场景问题

一个采集周期收到了多帧独立数据。逐个 `submit()` 当然可行，但提交端明确知道这些工作是一批，应该先表达这一语义，再决定是否需要结果。

## 推荐方案

需要逐项完成确认时，使用 `submit_batch()`：

<<< @/../examples/tutorial/04_batch.cpp{1-29}

完整源码：[`examples/tutorial/04_batch.cpp`](https://github.com/Linductor-alkaid/executor/blob/master/examples/tutorial/04_batch.cpp)。

```bash
./build/examples/tutorial/tutorial_04_batch
```

## 预期输出

```text
batch processed=6, completed=yes
```

## 三种批量路径

| 需求 | 接口 | 如何观察失败 |
| --- | --- | --- |
| 每项需要完成或异常结果 | `submit_batch()` | 对每个 `future` 调用 `get()`。 |
| 不需要逐项结果 | `submit_batch_no_future()` | 结合失败回调/状态，并使用有界等待或关闭语义。 |
| 一整批控制工作更紧急 | `submit_batch_priority(priority, tasks)` | 与 `submit_batch()` 一样逐项检查 future。 |

示例先检查 `submit_batch()` 的全部 future，再提交 no-future 批次，并以 `wait_for_completion_for()` 做有界收尾。

## 为什么这样做

批量接口表达“这些工作属于同一提交批次”，可以减少重复提交路径的开销；但收益取决于任务体、数量、线程数、硬件和构建配置。不要承诺固定倍率，性能判断请运行仓库的 [`benchmark_batch_submit_real.cpp`](https://github.com/Linductor-alkaid/executor/blob/master/tests/benchmark_batch_submit_real.cpp)。

## 常见错误

- **把 no-future 当成无失败**：它只省去 future 管理，不会让任务异常消失；需要为长期运行程序配置失败可观察路径。
- **只因任务数量多就改用 batch**：少量或彼此有依赖的任务，普通提交或任务依赖通常更清晰。
- **用优先级替代背压策略**：队列持续堆积时，优先级不能消除容量和截止时间问题。

## 下一步阅读

[让加载、感知和规划按依赖执行](/zh/tutorial/dependencies)说明如何表达先后关系，而不是在任务体内阻塞等待。
