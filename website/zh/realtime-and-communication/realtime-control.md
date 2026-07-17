---
title: 启动专用实时控制循环
description: 使用 Executor Facade 注册、诊断启动、推送和停止一个专用周期线程。
---

# 启动专用实时控制循环

## 学习目标

从 CAN 或控制循环的固定周期需求出发，使用 `register_realtime_task_ex()`、`start_realtime_task_ex()`、`try_push_realtime_task()` 和状态查询完成最小的可诊断路径。

## 何时需要专用线程

`submit_periodic()` 适合健康检查、刷新和允许抖动的后台工作。控制循环若需要固定周期、周期预算、优先级或 CPU 亲和性，应使用一个专用实时线程；它仍受操作系统、权限和硬件约束，不是绝对时限保证。

## 推荐方案

教程示例在非特权环境中主动关闭内存锁和 timer slack 请求，以便验证基本路径：

<<< @/../examples/tutorial/07_realtime.cpp{1-39}

完整源码：[`examples/tutorial/07_realtime.cpp`](https://github.com/Linductor-alkaid/executor/blob/master/examples/tutorial/07_realtime.cpp)。

```bash
./build/examples/tutorial/tutorial_07_realtime
```

## 预期输出

```text
realtime started=yes, command=queued, cycles=observed, command ran=yes
```

## 生命周期与队列

1. 填写最小 `RealtimeThreadConfig`：名称、周期和 `cycle_callback`。
2. 用 `_ex` 变体注册并启动；失败时读取 `ExecutorResult::error_code` 和 `message`。
3. 通过 `push_realtime_task()` 或 `try_push_realtime_task()` 投递常规控制工作；返回 `false` 表示未入队。
4. 用 `get_realtime_executor_status()` 和 `get_realtime_task_list()` 观察运行状态，完成后调用 `stop_realtime_task()`。

实时队列是有界入口：入队成功只表示将在后续周期处理，并不表示任务已完成。`max_tasks_per_cycle` 默认是 `64`；剩余工作会留给下一周期，以保护周期预算。周期回调超时后，运行时会跳过已错过的节拍并重新以“当前时间加一个周期”调相，避免追赶造成抖动风暴；通过 `cycle_timeout_count` 观察超时。紧急停止必须走应用自己的硬件或安全控制旁路，不能等待实时队列消费。

## 配置与降级

默认配置会尽力申请实时优先级、CPU 亲和性、内存锁和低 timer slack。Linux 的 `SCHED_FIFO`、`mlockall`、容器 cpuset 和 Windows 调度能力可能受权限或平台限制；库会安全继续运行，但这不代表请求已生效。

部署时检查 `RealtimeExecutorStatus` 的 `priority_applied`、`cpu_affinity_applied`、`memory_locked` 和 `timer_slack_applied`，并结合 `cycle_timeout_count`、`dropped_task_count`、`queue_full_count` 与 `pool_exhausted_count` 设定告警。空 `cpu_affinity` 是自适应选择，显式配置则应由部署环境验证其有效性。

## 下一步阅读

[传递每一条消息](/zh/realtime-and-communication/channels)选择普通数据流或实时周期内有限消费；需要可观察的配置和状态传递，请继续阅读下一章。
