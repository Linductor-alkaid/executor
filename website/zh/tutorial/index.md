---
title: 循序教程
description: 以机器人数据流水线逐步学习 Executor Facade。
---

# 循序教程

教程使用同一条机器人数据流水线：采集 `SensorFrame`、解析 `ParsedFrame`、生成 `Plan`，并把 `ControlCommand` 送入控制循环。每章只增加一个业务问题和一组必要 API。

当前从[第一个任务](/zh/quick-start/first-task)开始。后续章节将依次覆盖优先级、延迟与周期、批量、依赖、失败可观察性、实时线程、通信和 GPU；每章都会先加入 `examples/tutorial/` 并通过构建验证，再发布页面。

需要直接按问题选接口时，进入[场景指南](/zh/guides/choosing-submit-api)。
