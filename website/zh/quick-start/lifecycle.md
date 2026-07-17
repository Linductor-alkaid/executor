---
title: 初始化与关闭
description: 在需要配置时先初始化，并在程序边界有序关闭 Executor。
---

# 初始化与关闭

## 场景问题

最小程序可以依赖懒初始化；但当线程数、队列容量或监控需要自定义时，必须在首次提交前固定配置。

## 推荐方案

```cpp
executor::ExecutorConfig config;
config.min_threads = 2;
config.max_threads = 4;

auto& executor = executor::Executor::instance();
auto initialized = executor.initialize_ex(config);
if (!initialized) {
    throw std::runtime_error(initialized.message);
}

// submit() ... future.get() ...
executor.shutdown(true);
```

`initialize_ex()` 返回带错误码和消息的 `ExecutorResult`，比兼容的 `bool` 初始化接口更适合诊断。`shutdown(true)` 会等待已接受的任务完成；RAII 与进程退出路径仍提供兜底，但应用的正常退出边界应显式关闭。

## 常见错误

- 首次 `submit()` 后再修改初始化配置。
- 进程退出前既不等待 future，也不调用关闭，使业务完成顺序不清楚。
- 把任务异常和初始化失败混为一谈：前者通常经 future 观察，后者由 `ExecutorResult` 表达。

## 下一步

查看[如何选择提交接口](/zh/guides/choosing-submit-api)，按业务问题进入优先级、延迟、周期和批量教程。
