---
title: 实时与通信
description: 按数据语义选择专用实时线程和跨线程通信组件。
---

# 实时与通信

普通 `submit_periodic()` 是线程池上的软周期工作。固定周期控制、周期预算和实时队列需要专用实时线程；它们不是同一个抽象层。

如果还没有看过组件如何组合，先阅读[完整机器人数据流水线](/zh/tutorial/complete-robot-pipeline)：它先建立角色、数据所有权和退出协议，再把每条边映射到具体组件。

1. [启动专用实时控制循环](/zh/realtime-and-communication/realtime-control)：把综合示例中的普通周期模拟替换为可诊断实时 Facade。
2. [传递每一条消息](/zh/realtime-and-communication/channels)：普通帧流与实时周期内有限 drain。
3. [传递最新值、快照和阶段](/zh/realtime-and-communication/state-and-phases)：配置、完整状态与启动顺序。
4. [通信可观察性](/zh/realtime-and-communication/observability)：统一理解 `CommStats` 与本地事件 callback。

完整 API 见 [`docs/API.md`](https://github.com/Linductor-alkaid/executor/blob/master/docs/API.md)；综合页面的可运行事实源是 [`examples/comm_robot_pipeline.cpp`](https://github.com/Linductor-alkaid/executor/blob/master/examples/comm_robot_pipeline.cpp)。
