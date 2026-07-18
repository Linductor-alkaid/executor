---
title: 第一个任务
description: 使用 submit 和 future.get 执行任务、获取返回值并观察异常。
---

# 第一个任务

## 学习目标

使用 `Executor::instance()` 获取 Facade，提交一个返回 `42` 的任务，并通过 `future.get()` 获取结果和异常。

## 场景问题

你需要把一段计算移到后台执行，但调用方仍要可靠地获得结果或得知它失败了。

## 推荐方案

使用 `submit()`。它返回 `std::future`：任务成功时 `get()` 返回值，任务抛出异常时 `get()` 在调用方线程重新抛出异常。首次 `submit()` 会使用默认配置完成懒初始化。

<<< @/../examples/tutorial/01_first_task.cpp{4,7-16,18-24,26}

完整源码：[`examples/tutorial/01_first_task.cpp`](https://github.com/Linductor-alkaid/executor/blob/master/examples/tutorial/01_first_task.cpp)。

```bash
./build/examples/tutorial/tutorial_01_first_task
```

## 预期输出

```text
answer=42
task failed: expected tutorial failure
```

## 为什么这样做

- `future.get()` 不只是取值，也是异常传播边界；忽略它会让失败不再由这里观察。
- 默认配置允许懒初始化，适合这个最小示例。
- 需要自定义线程数、队列容量或监控时，必须在第一次提交前调用 `initialize_ex()`。
- 示例最后调用 `shutdown()`；这里没有待处理任务，`shutdown()` 与 `shutdown(true)` 的完成结果相同。业务程序应按下一页的规则选择关闭语义。

## 常见错误

- **只提交，不保存 future**：适合明确的 fire-and-forget 任务，但你失去返回值与异常传播路径。
- **把 `submit_periodic()` 当实时任务**：它是普通线程池的软周期调度；严格控制循环请使用后续的实时教程。
- **首次提交后再初始化**：配置可能已经无法按预期生效；请先完成初始化。

## 下一步阅读

先阅读[提交自己的函数与数据](/zh/quick-start/task-inputs-and-ownership)，学习如何传入自由函数、成员函数、参数和业务对象；然后由[返回值与异常](/zh/quick-start/return-values-and-errors)解释如何选择和补充失败观察路径。[初始化与关闭](/zh/quick-start/lifecycle)说明何时显式管理生命周期。
