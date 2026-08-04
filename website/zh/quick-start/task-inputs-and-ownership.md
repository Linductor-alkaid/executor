---
title: 提交自己的函数与数据
description: 通过 submit_auto(lambda) 安全传递输入、共享对象和独占资源。
---

# 提交自己的函数与数据

## 学习目标

把有限工作需要的输入捕获进 lambda 后交给 `submit_auto(lambda)`；理解值捕获、移动捕获和共享所有权，并让任务执行期间所需对象保持有效。

## 默认入口：`submit_auto(lambda)`

普通路径使用按值捕获的 lambda。可运行教程从这个模式开始，再展示多值和移动捕获：

<<< @/../examples/tutorial/11_task_inputs.cpp{34-50}

`frame`、`offset` 和移动后的 `payload` 分别由闭包拥有；future 表示完成或异常。修改提交方后续持有的 `frame` 不会改变任务保存的副本。`submit_auto(lambda)` 安全地选择默认异步后端，不会从 callable 推断应走 GPU、无锁或实时路径。`get_last_routing_decision()` 可解释选路，但不预约后端，也不取代 future。

## 提交成员函数

成员函数也使用同一条规则：将稳定 owner 和输入按值捕获到 lambda。推荐使用 `std::shared_ptr`，使对象至少活到任务执行结束：

<<< @/../examples/tutorial/11_task_inputs.cpp{20-29,52-55}

不要捕获裸 `this` 或局部对象地址：若 owner 在任务开始前销毁，worker 会访问悬空对象。即使对象由稳定的服务 owner 管理，也必须让关闭顺序先停止提交、等待任务，再销毁 owner。

## 用 lambda 组织任务输入

lambda 适合在提交点组合多个输入、执行少量预处理，或调用重载函数。默认优先按值捕获；上面的 `score` 与 `adjusted` 就分别演示一个和多个值的捕获。`[frame, offset]` 把两个值复制进闭包，任务不依赖提交函数的栈帧。不要随手写 `[&]`：异步任务通常在当前作用域结束后才运行，引用捕获很容易悬空，而且调用方后续修改数据会形成竞态。

如果输入很大，先判断复制是否真是瓶颈。常见选择是移动独占资源，或共享不可变对象：

```cpp
auto model = std::make_shared<const Model>(load_model());
auto result = executor.submit_auto([model, frame] {
    return infer(*model, frame);
});
```

## 把独占资源移入任务

需要把 `std::unique_ptr`、buffer handle 等资源的所有权交给任务时，用移动捕获：

<<< @/../examples/tutorial/11_task_inputs.cpp{47-50}

提交后原 `payload` 为空，资源只由任务闭包拥有。这比传裸指针更容易证明生命周期。移动后的对象不能再作为调用方输入使用。

移动捕获让闭包成为资源的明确 owner；若业务函数需要按值取得 `std::unique_ptr` 或 `T&&`，在 lambda 内决定何时 `std::move`，不要把资源借给异步任务。

## 共享和修改状态

若任务需要修改跨线程共享的状态，让 lambda 捕获带有明确同步语义的 owner，而不是借用调用方栈上的引用：

<<< @/../examples/tutorial/11_task_inputs.cpp{57-60}

这个例子用 `shared_ptr<atomic<int>>` 同时表达共享生命周期和原子访问。`shared_ptr` 只延长对象生命周期，并不会让任意对象自动线程安全；若状态不是原子类型，仍需要自己的 mutex、消息传递或其他同步协议。

不要使用 `[&]` 或裸引用来传递异步输入：它们既不延长生命周期，也不提供线程安全。`future.get()` 能等待任务结束，但不能修复任务运行期间已经发生的数据竞争。

## 输入所有权选择表

| 需求 | 推荐写法 | 任务实际依赖 | 主要风险 |
| --- | --- | --- | --- |
| 小型输入，任务读取即可 | `submit_auto([value] { ... })` | 自己的副本 | 复制成本 |
| 独占资源交给任务 | `[value = std::move(value)]` | 独占所有权 | 提交方不能继续使用已移动对象 |
| 多任务共享只读大对象 | 捕获 `shared_ptr<const T>` | 共享生命周期 | 共享计数与对象常驻成本 |
| 调用成员函数 | `[owner, value] { return owner->method(value); }` | 对象至少活到完成 | 不要用可能悬空的裸对象指针 |
| 共享可变状态 | 捕获带同步协议的 `shared_ptr` | 共享 owner 与同步规则 | 数据竞争、关闭顺序 |

## 这些规则也适用于其他提交接口

当业务明确需要 priority、delay、periodic、batch 或 dependency 时，进入对应显式接口；无论哪条 future 路径，都优先把稳定输入绑定到 lambda。延迟越长、依赖链越长，借用输入越危险，因为任务真正开始的时间更难预测。

周期任务和 batch 的形状不同：`submit_periodic()` 接受可重复调用的 `void()` 任务，通常用 lambda 显式捕获长期状态；batch 接受一组独立的 `void()` callable，每个任务都应各自拥有稳定输入。

实时、GPU 与通信也不是普通参数包的简单变体：实时 callback 和动态实时任务接收预绑定的 `void()` callable，GPU callable 可接收后端 stream，通信组件传递的是值对象。请在对应的[实时控制](/zh/realtime-and-communication/realtime-control)、[GPU 提交](/zh/gpu/register-and-submit)和[通信通道](/zh/realtime-and-communication/channels)页面按各自输入契约使用。

## 常见编译与运行问题

- **重载函数无法推导**：在 lambda 内调用目标重载，让编译器从调用参数推导。
- **参数类型不匹配**：先在同步代码中确认 lambda 内调用可以成立，再提交。
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
