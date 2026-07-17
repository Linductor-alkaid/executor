---
title: 循序教程
description: 以机器人数据流水线逐步学习 Executor Facade。
---

# 循序教程

教程使用同一条机器人数据流水线：采集 `SensorFrame`、解析 `ParsedFrame`、生成 `Plan`，并把 `ControlCommand` 送入控制循环。每章只增加一个业务问题和一组必要 API。

从[第一个任务](/zh/quick-start/first-task)开始，然后沿着下面的业务问题推进：

1. [让控制命令优先](/zh/tutorial/priority)：普通分析与紧急控制命令共用线程池。
2. [延迟重试与健康检查](/zh/tutorial/delayed-and-periodic)：稍后重试设备，并取消可观察的软周期任务。
3. [批量处理传感器帧](/zh/tutorial/batch)：根据是否需要 future 选择批量提交路径。
4. [加载、感知与规划依赖](/zh/tutorial/dependencies)：让规划在前置工作完成后执行。
5. [有界等待与状态快照](/zh/tutorial/waiting-and-status)：在切换阶段或关闭前可靠收尾。

后续章节将覆盖失败可观察性、实时线程、通信和 GPU；每章都会先加入 `examples/tutorial/` 并通过构建验证，再发布页面。

需要直接按问题选接口时，进入[场景指南](/zh/guides/choosing-submit-api)。
