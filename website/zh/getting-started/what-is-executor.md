---
title: Executor 是什么
description: Executor 为普通异步任务和专用实时线程提供统一的 C++ Facade。
---

# Executor 是什么

Executor 是一个 C++20 任务执行库：普通工作适合在线程池中运行；严格周期和抖动控制则使用专用实时线程。它把常见入口集中在 `executor::Executor` Facade 中，让大多数程序无需直接管理线程池或调度器。

## 它解决什么问题

当程序需要在后台解析数据、并行处理一批工作或等待某个结果时，手写线程通常意味着要自己处理生命周期、异常传递和关闭顺序。`Executor::submit()` 返回 `std::future`，把结果和任务异常带回调用方。

## 默认从 Facade 开始

普通业务任务优先使用 Facade：

```text
你的业务代码 → Executor Facade → 线程池与调度器
```

只有在需要独立资源隔离、自定义周期源或直接控制 GPU/实时执行器时，才进入高级接口。`submit_periodic()` 是线程池上的软周期任务；它不替代专用实时线程。

## 下一步

继续阅读[构建与安装](/zh/quick-start/build)，然后运行[第一个任务](/zh/quick-start/first-task)。
