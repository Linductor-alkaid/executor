---
layout: home
hero:
  name: Executor 使用手册
  text: 为 C++20 程序提供可靠的任务执行与线程管理
  tagline: 十分钟完成构建、提交任务、获取结果并观察异常。
  actions:
    - theme: brand
      text: 十分钟开始使用
      link: /zh/quick-start/build
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
- 准备接入服务：[生产接入检查清单](/zh/guides/production-readiness)。
- 深入原理：[高级与原理](/zh/advanced/)；按需了解实时线程、通信、GPU 与底层执行路径。

## 发布信息

| 项目 | 当前支持 |
| --- | --- |
| 平台 | Linux、Windows |
| 语言 | C++20 |
| 构建系统 | CMake 3.16+ |
| 版本 | `v0.2.3` 源码基线的开发快照（含待发布能力） |
| 持续集成 | [GitHub Actions](https://github.com/Linductor-alkaid/executor/actions/workflows/c-cpp.yml) |
| 许可证 | [MIT](https://github.com/Linductor-alkaid/executor/blob/master/LICENSE) |

<div class="version-note">本手册以当前开发快照为准，不能将待发布能力视为既有稳定版承诺；稳定发布后会明确标注适用 tag。</div>
