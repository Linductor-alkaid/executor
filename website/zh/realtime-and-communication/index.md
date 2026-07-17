---
title: 实时与通信
description: 按数据语义选择专用实时线程和跨线程通信组件。
---

# 实时与通信

普通 `submit_periodic()` 是线程池上的软周期工作。固定周期控制、周期预算和实时队列需要专用实时线程；它们不是同一个抽象层。

跨线程传递数据时，先按语义选择：只保留最新配置使用 `LatestMailbox`；每条消息都要消费使用 `MpscChannel`；周期内有限消费使用 `RealtimeChannel`；完整状态快照使用 `DoubleBuffer`；阶段顺序使用 `PhaseGate` 或 `Sequencer`。

完整 API 与场景示例见 [`docs/API.md`](https://github.com/Linductor-alkaid/executor/blob/master/docs/API.md) 和 [`examples/comm_robot_pipeline.cpp`](https://github.com/Linductor-alkaid/executor/blob/master/examples/comm_robot_pipeline.cpp)。
