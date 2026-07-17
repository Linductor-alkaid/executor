---
layout: home
hero:
  name: Executor 使用手册
  text: 从第一个任务开始构建可靠的 C++ 并发程序
  tagline: 十分钟完成构建、提交任务、获取结果并观察异常。
  actions:
    - theme: brand
      text: 十分钟开始使用
      link: /zh/quick-start/first-task
    - theme: alt
      text: 了解 Executor
      link: /zh/getting-started/what-is-executor
features:
  - title: 先完成一个任务
    details: 从 submit() 和 future.get() 开始，不要求先理解线程池、GPU 或实时调度器。
  - title: 按场景选择接口
    details: 需要优先级、延迟、周期、批量或依赖时，再逐步引入对应的 Facade API。
  - title: 示例可验证
    details: 页面核心代码对应仓库中的教程示例，并由根 CMake 工程持续编译和 smoke test。
---

## 一眼看懂

```cpp
auto& executor = executor::Executor::instance();
auto answer = executor.submit([] { return 42; });
std::cout << answer.get() << '\n';
executor.shutdown();
```

`get()` 同时获取结果和重新抛出任务中的异常。完整代码与预期输出见[第一个任务](/zh/quick-start/first-task)。

## 从这里继续

- 第一次使用：从[构建与安装](/zh/quick-start/build)到[第一个任务](/zh/quick-start/first-task)。
- 按场景选型：[选择提交接口](/zh/guides/choosing-submit-api)。
- 深入原理：后续将覆盖实时线程、通信、GPU 与底层执行路径。

<div class="version-note">适用版本：v0.2.3-dev（当前开发快照）</div>
