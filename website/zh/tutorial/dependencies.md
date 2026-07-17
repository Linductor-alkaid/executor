---
title: 让加载、感知和规划按依赖执行
description: 用 TaskHandle、submit_after 与 when_all 构建可观察的任务依赖。
---

# 让加载、感知和规划按依赖执行

## 学习目标

使用 `submit_with_handle()` 保留任务结果和依赖句柄；以 `submit_after()` 和 `when_all()` 表达“加载与感知完成后再规划”。

## 场景问题

规划不能在地图加载和感知预处理完成前启动。若在一个工作线程任务内直接 `future.get()` 等待，很容易占住可执行其他前置工作的线程，甚至在资源紧张时形成阻塞链。

## 推荐方案

先提交带句柄的前置任务，汇合句柄后再提交规划：

<<< @/../examples/tutorial/05_dependencies.cpp{1-25}

完整源码：[`examples/tutorial/05_dependencies.cpp`](https://github.com/Linductor-alkaid/executor/blob/master/examples/tutorial/05_dependencies.cpp)。

```bash
./build/examples/tutorial/tutorial_05_dependencies
```

## 预期输出

```text
plan score=42
```

## 构建关系

- 单个前置任务：`submit_after(load.handle, plan)`。
- 多个前置任务：`submit_after({load.handle, sense.handle}, plan)`；示例使用等价的 `when_all()` 先创建汇合句柄。
- 还要继续依赖规划时，使用 `submit_after_with_handle()`；它与 `submit_with_handle()` 一样同时给出 `future` 和 `TaskHandle`。
- `when_all(handles)` 只表达“全部完成”的汇合点，可嵌套使用。

## 失败如何观察

前置任务失败时，其 `future.get()` 会重新抛出异常；依赖任务默认不会运行，并在自己的 `future.get()` 上给出失败。无效句柄和跨 `Executor` 实例的句柄也是可观察的拒绝，不要把句柄跨实例保存或混用。

## 何时不选

任务图适合明确的完成依赖，不适合连续数据流、最新值覆盖或严格周期阶段；这些问题将在通信与实时章节使用相应组件处理。

## 下一步阅读

[有界等待与状态快照](/zh/tutorial/waiting-and-status)说明如何在关闭或阶段切换时等待工作完成，而不无限阻塞。
