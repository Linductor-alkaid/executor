---
title: 可靠性
description: 让异常、超时、拒绝与关闭行为保持可观察。
---

# 可靠性

可靠性页面关注“失败如何被看见”：`future.get()` 传递任务结果和异常，`ExecutorResult` 诊断初始化与注册失败，状态与 callback 反映长期运行中的拒绝、超时和丢任务。

开始阅读[返回值与异常](/zh/quick-start/return-values-and-errors)和[初始化与关闭](/zh/quick-start/lifecycle)。完整字段与兼容语义以 [`docs/API.md`](https://github.com/Linductor-alkaid/executor/blob/master/docs/API.md) 为准。

后续专题将补充 failure callback、超时快照、监控采样和背压案例。
