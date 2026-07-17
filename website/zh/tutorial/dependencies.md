---
title: 让加载、感知和规划按依赖执行
description: 用 TaskHandle、submit_after 与 when_all 构建可观察的任务依赖。
---

# 让加载、感知和规划按依赖执行

## 学习目标

使用 `submit_with_handle()` 保留任务结果和依赖句柄；以 `submit_after()` 和 `when_all()` 表达“加载与感知完成后再规划”。

## 场景问题

规划不能在地图加载和感知预处理完成前启动。在业务 lambda 中直接 `future.get()` 会隐藏依赖关系，也无法让 Facade 校验句柄并统一传播前置失败。

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

## 当前容量边界

任务句柄让完成关系和失败传播显式化，但当前实现不是完全非阻塞的 DAG 调度器：dependent wrapper 会进入普通线程池，并在前置状态确定前等待。低线程数、大量长依赖链，或在前置任务之前集中提交 dependent，都可能占住 worker。

实际使用时：

- 先提交所有前置任务，再提交依赖任务；
- 不在 dependent 中继续进行无界阻塞；
- 控制同时在途的依赖链数量；
- 用生产配置中的最小线程数进行压力测试；
- 大规模动态 DAG 需要专门图调度器时，不要只靠增大线程池和队列规避问题。

## 失败如何观察

前置任务失败时，其 `future.get()` 会重新抛出异常；依赖任务默认不会运行，并在自己的 `future.get()` 上给出失败。无效句柄和跨 `Executor` 实例的句柄也是可观察的拒绝，不要把句柄跨实例保存或混用。

## 何时不选

任务图适合数量可控、完成关系明确的依赖，不适合大规模非阻塞 DAG、连续数据流、最新值覆盖或严格周期阶段；这些问题应使用专门图调度器、通信组件或实时路径。

## 下一步阅读

[有界等待与状态快照](/zh/tutorial/waiting-and-status)说明如何在关闭或阶段切换时等待工作完成，而不无限阻塞。
