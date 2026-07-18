---
title: 提交自己的函数与数据
description: 把自由函数、成员函数、lambda、参数和资源安全地提交给 Executor。
---

# 提交自己的函数与数据

## 学习目标

把项目中已经实现的函数交给 `submit()`，理解参数何时被复制、移动或引用，并让任务执行期间所需对象保持有效。

## `submit()` 接受什么

`submit()` 的调用形式是：

```cpp
auto future = executor.submit(可调用对象, 参数...);
```

“可调用对象”可以是自由函数、lambda、函数对象或成员函数指针。Executor 根据函数返回类型生成对应的 `std::future<T>`；参数不需要预先包装成无参数函数。

最直接的情况是把已有函数和实参分别传入：

<<< @/../examples/tutorial/11_task_inputs.cpp{12-19,39-43}

`frame` 和 `2` 会被保存到异步任务中，worker 稍后执行等价于 `score_frame(frame, 2)` 的调用。修改提交方后续持有的 `frame`，不会改变任务已经保存的副本。

## 提交成员函数

成员函数还需要一个调用对象。推荐让任务持有 `std::shared_ptr`，使对象至少活到任务执行结束：

<<< @/../examples/tutorial/11_task_inputs.cpp{21-31,45-46}

不要未经证明地传裸 `this` 或局部对象地址。`executor.submit(&Planner::make_plan, this, frame)` 语法可能成立，但如果 owner 在任务开始前销毁，worker 会访问悬空对象。若对象本来就由稳定的服务 owner 管理，也必须让关闭顺序先停止提交、等待任务，再销毁 owner。

## 用 lambda 组织任务输入

lambda 适合在提交点组合多个输入、执行少量预处理，或调用重载函数。默认优先按值捕获：

<<< @/../examples/tutorial/11_task_inputs.cpp{48-51}

`[frame, offset]` 把两个值复制进闭包，任务不依赖提交函数的栈帧。不要随手写 `[&]`：异步任务通常在当前作用域结束后才运行，引用捕获很容易悬空，而且调用方后续修改数据会形成竞态。

如果输入很大，先判断复制是否真是瓶颈。常见选择是移动独占资源，或共享不可变对象：

```cpp
auto model = std::make_shared<const Model>(load_model());
auto result = executor.submit([model, frame] {
    return infer(*model, frame);
});
```

## 把独占资源移入任务

需要把 `std::unique_ptr`、buffer handle 等资源的所有权交给任务时，用移动捕获：

<<< @/../examples/tutorial/11_task_inputs.cpp{53-56}

提交后原 `payload` 为空，资源只由任务闭包拥有。这比传裸指针更容易证明生命周期。移动后的对象不能再作为调用方输入使用。

当前实现使用 `std::bind` 保存直接传给 `submit(fn, args...)` 的参数，执行时绑定参数通常作为已保存的左值参与调用。因此，函数若要求按值取得 `std::unique_ptr` 或要求 `T&&`，不要写 `submit(fn, std::move(value))` 并期待执行时再次得到右值；使用上面的移动捕获 lambda，在闭包内部决定何时移动。

## 什么时候可以传引用

普通实参默认按衰减后的值保存。确实需要让任务操作原对象时，必须显式使用 `std::ref()` 或 `std::cref()`：

<<< @/../examples/tutorial/11_task_inputs.cpp{33-35,58-59,61-65}

引用没有延长对象生命周期，也没有提供线程安全。这个例子成立是因为：

1. `processed` 是线程安全的 `std::atomic<int>`；
2. 调用方在 `processed` 离开作用域前执行 `counted.get()`；
3. 没有其他代码在无同步的情况下访问同一可变状态。

如果无法同时证明这三点，就传值、移动所有权，或使用具有明确同步协议的 `shared_ptr`。`future.get()` 能等待任务结束，但它不能修复任务执行期间已经发生的数据竞争。

## 输入所有权选择表

| 需求 | 推荐写法 | 任务实际依赖 | 主要风险 |
| --- | --- | --- | --- |
| 小型输入，任务读取即可 | `submit(fn, value)` 或 `[value]` | 自己的副本 | 复制成本 |
| 独占资源交给任务 | `[value = std::move(value)]` | 独占所有权 | 提交方不能继续使用已移动对象 |
| 多任务共享只读大对象 | 捕获 `shared_ptr<const T>` | 共享生命周期 | 共享计数与对象常驻成本 |
| 调用成员函数 | 成员函数指针 + `shared_ptr` | 对象至少活到完成 | 不要用可能悬空的裸对象指针 |
| 修改调用方原对象 | `std::ref(value)` | 外部对象与同步协议 | 悬空引用、数据竞争、关闭顺序 |

## 这些规则也适用于其他提交接口

`submit_priority(priority, fn, args...)`、`submit_delayed(delay, fn, args...)`、`submit_with_handle(fn, args...)` 和 `submit_after(handle, fn, args...)` 使用相同的 callable 与参数模型。延迟越长、依赖链越长，引用输入越危险，因为任务真正开始的时间更难预测。

周期任务和 batch 的形状不同：`submit_periodic()` 接受可重复调用的 `void()` 任务，通常用 lambda 显式捕获长期状态；batch 接受一组独立的 `void()` callable，每个任务都应各自拥有稳定输入。

## 常见编译与运行问题

- **重载函数无法推导**：用 lambda 调用目标重载，或显式转换函数指针类型。
- **参数类型不匹配**：先在同步代码中确认 `std::invoke(fn, args...)` 可以成立，再提交。
- **只读成员函数报错**：检查成员函数的 `const` 限定和传入对象类型是否一致。
- **偶发崩溃或错误数据**：优先检查引用捕获、裸指针、`this` 和无锁共享可变对象。
- **任务还没开始对象就销毁**：改为值捕获、移动捕获或 `shared_ptr`，并修正关闭顺序。

## 构建和运行

```bash
cmake --build build --target tutorial_11_task_inputs
./build/examples/tutorial/tutorial_11_task_inputs
```

预期输出：

```text
score=42, plan=local-frame-7, adjusted=26, owned=9
processed=1
```

下一步阅读[返回值与异常](/zh/quick-start/return-values-and-errors)，理解这些输入在任务成功或失败后如何通过 future 交还控制权；需要按时机选择接口时阅读[如何选择提交接口](/zh/guides/choosing-submit-api)。
