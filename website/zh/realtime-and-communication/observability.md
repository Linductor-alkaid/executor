---
title: 通信可观察性
description: 用 CommStats 和 CommEventCallback 观察通信背压、覆盖、陈旧读取、延迟与时序错误。
---

# 通信可观察性

## 学习目标

把通信错误视为组件本地的协议状态，并通过 `stats()` 与 `set_event_callback()` 观察它们，而不是误以为都会出现在 `ExecutorFailureStatus`。

## 两个入口

```cpp
channel.set_event_callback([](const executor::comm::CommEvent& event) {
    report_comm_event(event.component_name, event.message);
});

const auto stats = channel.stats();
if (stats.dropped_count != 0 || stats.timeout_count != 0) {
    raise_backpressure_alert(stats);
}
```

`CommStats` 是本地累计快照，可包含发送、接收、drop、覆盖、stale read、关闭后发送、超时、handler 异常、missed phase、当前/峰值深度、producer/consumer lag 和延迟。`CommEventCallback` 适合少量诊断事件；callback 自身抛出的异常会被隔离，不会改变通信操作的返回值或组件状态。

## 根据语义设置告警

| 组件 | 重点字段/事件 | 含义 |
| --- | --- | --- |
| `MpscChannel` | `dropped_count`、`current_depth`、`timeout_count` | 背压、积压或等待失败。 |
| `RealtimeChannel` | `dropped_count`、`handler_exception_count`、lag | 单周期预算不足、满队列或 handler 失败。 |
| `LatestMailbox` | `overwritten_count`、`stale_read_count` | 更新过快或消费者没有拿到更高版本。 |
| `DoubleBuffer` | sequence、`stale_read_count`、latency | 读者未见新快照或状态发布延迟。 |
| `PhaseGate` / `Sequencer` | `timeout_count`、`missed_phase_count` | 阶段未到达、已关闭或顺序被跳过。 |

阈值必须来自业务周期和数据重要性。例如 mailbox 的覆盖在“只要最新目标值”场景中可以正常，而在必须保存每条审计消息的场景中则说明选择了错误组件。

## 与 Executor 失败状态的边界

`CommStats` 和 `CommEventCallback` 默认不汇总到 `ExecutorFailureStatus`，也不会调用 `Executor::set_failure_callback()`。若服务需要统一告警，在组件 callback 中把低频事件桥接到自己的监控系统；不要在高频数据路径中默认写日志，以免诊断本身破坏实时预算。

## 下一步阅读

回到[完整机器人数据流水线](/zh/tutorial/complete-robot-pipeline)按故障注入步骤验证背压与退出，或回顾[如何选择通信组件](/zh/guides/choosing-communication)。
