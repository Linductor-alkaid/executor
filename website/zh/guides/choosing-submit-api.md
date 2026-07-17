---
title: 如何选择提交接口
description: 从时机、顺序、结果和失败语义出发，为一次性、周期、批量与依赖任务选择 Executor Facade。
---

# 如何选择提交接口

选择接口时先描述业务约束，不要先比较函数名。最有用的四个问题是：工作何时可以开始、是否有先后关系、调用方是否需要逐项结果，以及错过时间预算后应该发生什么。

## 30 秒选择表

| 你遇到的问题 | 默认接口 | 必须保留的控制量 | 失败观察 |
| --- | --- | --- | --- |
| 立即执行一次后台工作 | `submit()` | `future` | `future.get()` |
| 少数控制工作应先排队 | `submit_priority()` | `future`、priority | `future.get()` |
| 到某个相对时间后重试 | `submit_delayed()` | `future`、delay | `future.get()` |
| 普通后台健康检查 | `submit_periodic()` | task ID | 周期状态、callback |
| 一批独立工作需要逐项结果 | `submit_batch()` | 全部 futures | 对每个 future 调用 `get()` |
| 一批独立工作不需要逐项结果 | `submit_batch_no_future()` | 业务批次 ID | failure status/callback、等待状态 |
| 整批控制工作应先排队 | `submit_batch_priority()` | futures、priority | 逐项 `get()` |
| 后续工作必须等前置成功 | `submit_after()` / `when_all()` | `TaskHandle` 与 future | 依赖失败传播到后续 future |

默认接口只表达一项主要语义。需要“先等两个任务成功，再高优先级执行”的复杂策略时，先用依赖表达正确性，再判断是否真的需要优先级；不要用排队时序模拟依赖。

## 决策流程

```text
这段工作需要固定周期和 jitter 预算吗？
├─ 是 ──> 专用实时任务，不使用普通 submit_* 代替
└─ 否
   ├─ 会重复运行吗？
   │  ├─ 是 ──> 软周期维护：submit_periodic
   │  └─ 否
   ├─ 必须等待其他任务成功吗？
   │  ├─ 是 ──> submit_with_handle + submit_after / when_all
   │  └─ 否
   ├─ 同时产生大量独立同类工作吗？
   │  ├─ 是 ──> submit_batch（默认保留 futures）
   │  └─ 否
   └─ 立即、优先或延迟 ──> submit / submit_priority / submit_delayed
```

## 默认选择：`submit()`

当任务可以立即进入共享线程池，且没有依赖、周期或批次语义时，使用 `submit()`。它是最容易理解和验证的路径：

```cpp
auto future = executor.submit([frame] { return decode(frame); });
auto decoded = future.get();
```

按值捕获 `frame` 能让任务拥有稳定输入。若捕获引用、裸指针或 `this`，必须证明对象生命周期长于任务；这是异步接入中最常见、也最隐蔽的错误之一。

不应把阻塞循环、永久监听或没有退出条件的服务线程放进共享线程池。它们会长期占用 worker，使短任务排队；这类工作更适合由组件拥有的 `std::jthread`，或在确有周期语义时使用实时任务路径。

## 优先级：只改变等待队列

`submit_priority()` 适合少量必须在普通积压前被选取的工作，例如控制命令相对于日志解析。优先级取值为 `LOW`、`NORMAL`、`HIGH`、`CRITICAL` 对应的整数范围；使用项目内统一映射，避免不同模块自行发明数值。

它不提供：

- 抢占已经运行的低优先级任务；
- deadline 或最大响应时间保证；
- 多 worker 环境中的完成顺序；
- 对阻塞任务造成的线程池饥饿免疫。

如果所有调用方都把任务标成最高级，优先级就失去区分能力。上线前应检查各等级占比，并为控制平面保留最高等级。

## 延迟：表达“最早何时可运行”

`submit_delayed(delay_ms, ...)` 表示经过相对延迟后把工作交给普通执行器。它适合退避重试、短期延后清理和去抖，不是精确定时器：到期时若线程池繁忙，任务仍可能继续等待。

需要绝对业务时间时，应用应先计算剩余时长，并处理系统时间变化；需要严格周期时使用专用实时任务。延迟 future 仍然是返回值和异常边界，不能因为任务稍后执行就忽略它。

## 周期：适合维护，不适合控制

`submit_periodic()` 适合健康检查、缓存刷新和统计上报。它返回 task ID，而不是每次执行对应的一组 futures，因此使用者应保存 ID 并同时设计：

- 何时调用 `cancel_task()`；
- 如何查看 `execution_count`、`failed_count`、`consecutive_failure_count` 和最近错误；
- 连续失败达到阈值后是告警、降级还是停止；
- 任务执行时间接近或超过 period 时业务如何处理。

周期任务的 callback 应短小且可重复。如果它需要永久阻塞、严格相位或固定 jitter，普通周期接口已经不是正确抽象。

## 批量：先证明语义相同，再谈性能

使用 batch API 的前提是一批任务相互独立、由同一生产者同时产生，并且具有相同的调度语义。批量接口可能减少重复提交路径开销，但实际收益取决于任务数、任务体、线程数、硬件和构建类型，不承诺固定倍率。

默认使用 `submit_batch()` 并消费所有 futures。只有以下条件同时成立时才考虑 `submit_batch_no_future()`：

1. 调用方明确不需要逐项返回值或完成确认；
2. 已建立 failure callback/status 或业务事件；
3. 关闭前有有界等待或允许丢弃的明确策略；
4. 能将服务级失败关联回自己的批次或输入。

fire-and-forget 只是省略逐项 future，不等于 failure-and-forget。

## 依赖：表达成功关系，不要阻塞 worker

“加载模型 → 并行预处理 → 规划”属于任务图，而不是优先级问题。先用 `submit_with_handle()` 获得 `TaskHandle`，再用 `submit_after()` 或 `when_all()` 建立依赖。前置任务失败时，后续任务不会照常执行，其 future 会体现依赖失败。

不要在任意业务 lambda 中用 `future.get()` 隐藏任务关系；显式 handle 能校验依赖并统一传播失败。但当前 dependent wrapper 仍会进入线程池等待前置状态，并非完全非阻塞调度。大量依赖链在低线程数下仍可能占满 worker，应先提交前置任务、限制在途图规模，并在最小线程配置下验证。需要大规模动态 DAG 时选择专门图调度器。

`TaskHandle` 只在创建它的 Executor 实例内有效。无效 handle、未知 handle 或跨实例混用属于配置错误，不应作为正常重试路径。

## 时间预算与取消要分开设计

`wait_for_completion_ex(timeout)` 的超时表示预算内未全部完成，并返回 active、queued、pending 等状态快照；它不安全地杀死正在运行的 C++ 函数。业务层若需要取消，应让任务显式检查停止信号、为 I/O 设置超时，或将长工作切成可中断步骤。

同理，`shutdown(true)` 是有序退出策略，不是任意任务都能及时结束的保证。一个永久阻塞的业务函数仍然会破坏退出预算。

## 上线前核对

- 每个 future 都有明确 owner：调用 `get()`、转交给谁，或为什么可以丢弃。
- 每个周期 task ID 都有取消者，并定义连续失败策略。
- 优先级只用于排队，不承担正确性或实时保证。
- batch 中的任务相互独立；需要顺序时改用依赖。
- 所有任务都有有界运行或协作停止机制。
- 队列积压、提交拒绝和等待超时都有可观察路径。
- 通信数据没有被硬塞进任务调度语义；跨线程流数据按[通信组件选型](/zh/guides/choosing-communication)处理。

下一步可沿[循序教程](/zh/tutorial/)运行对应示例；准备接入长期服务时，继续完成[生产接入检查清单](/zh/guides/production-readiness)。完整签名、状态字段和兼容接口以[API 参考](/zh/reference/api)为准。
