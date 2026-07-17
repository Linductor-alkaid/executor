---
title: 传递最新值、快照和阶段
description: 为最新配置、完整状态与启动顺序分别选择 LatestMailbox、DoubleBuffer、PhaseGate 和 Sequencer。
---

# 传递最新值、快照和阶段

## 学习目标

按数据语义选择 `LatestMailbox<T>`、`DoubleBuffer<T>`、`PhaseGate` 或 `Sequencer`，而不是以共享对象和标志位拼凑同步。

## 最小流水线

教程示例把帧、配置、状态和启动阶段组合在一起：

<<< @/../examples/tutorial/08_communication.cpp{1-29}

完整源码：[`examples/tutorial/08_communication.cpp`](https://github.com/Linductor-alkaid/executor/blob/master/examples/tutorial/08_communication.cpp)。

```bash
./build/examples/tutorial/tutorial_08_communication
```

## 预期输出

```text
frame=7, gain=3, state=21, phase=ready
```

## 只保留最新配置

`LatestMailbox<T>` 只保留最近一次 `publish()` 的值。实时消费者以 sequence 读取新版本，避免重复使用旧配置：

```cpp
uint64_t seen = 0;
ControlConfig config;
if (mailbox.try_load_newer_than(seen, config, seen)) {
    apply_config(config);
}
```

覆盖旧配置会增加 `overwritten_count`；没有更高 sequence 时是 stale read，而不是新消息丢失。

## 发布完整状态

`DoubleBuffer<T>` 适合单写多读。writer 用 `publish()` 或 `update()` 完整构造非活动缓冲后一次发布；reader 的 `load()` 或 `load_newer_than()` 得到按值复制的 `Snapshot<T>`，不会看到半更新对象。多写者先用 `MpscChannel` 汇聚到一个状态 owner；大型对象还要评估复制成本。

## 阶段与精确顺序

`PhaseGate` 适合初始化、标定、运行等单调推进的阶段：`advance_to()` 不能倒退或重复；`wait_for()` 区分成功、`Timeout` 和 `Closed`，`wait_for_exact()` 还能观察被跳过的 phase（`MissedPhase`）。

需要精确 ticket 顺序时使用 `Sequencer`：`next_ticket()` 分配序号，`publish(ticket)` 推进发布进度，`wait_until_published(ticket, timeout)` 在目标已被越过时返回 `MissedPhase`。它不是数据队列，不能替代 `MpscChannel`。

## 下一步阅读

[通信可观察性](/zh/realtime-and-communication/observability)说明如何用本地统计与事件处理 drop、覆盖、陈旧读取和 missed phase。
