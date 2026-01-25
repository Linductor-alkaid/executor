# Changelog

本文档记录 executor 项目的版本变更。版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

---

## [0.1.0] - 2025-01-24

### 新增

- **Executor Facade**：统一 API `Executor::instance()` / 实例化模式，`initialize`、`shutdown`、`wait_for_completion`
- **任务提交**：`submit`、`submit_priority`、`submit_delayed`、`submit_periodic`、`cancel_task`
- **实时任务**：`register_realtime_task`、`start_realtime_task`、`stop_realtime_task`、`get_realtime_executor`、`get_realtime_task_list`
- **监控**：`enable_monitoring`、`get_async_executor_status`、`get_realtime_executor_status`、`get_task_statistics`、`get_all_task_statistics`
- **执行器管理**：`ExecutorManager` 单例/实例化，默认异步执行器 + 实时执行器注册表，RAII 生命周期
- **线程池**：动态扩缩容、优先级调度、工作窃取、负载均衡、任务分发
- **专用实时线程**：`RealtimeThreadExecutor`，周期回调、线程优先级、CPU 亲和性，可选 `ICycleManager` 集成
- **配置**：`ExecutorConfig`、`ThreadPoolConfig`、`RealtimeThreadConfig`
- **构建与安装**：CMake 3.16+，静态/动态库选项，`find_package(executor)` 支持（`executorConfig.cmake`、`executorConfigVersion.cmake`）
- **测试与示例**：单元/集成/性能/压力测试，`basic_submit`、`realtime_can`、`multi_project`、`monitor_example`

### 依赖与平台

- C++20，仅标准库 + `pthread`（Linux），无第三方必需依赖
- Linux/windows 下已验证；

---

## 迁移指南

当前为首次发布，无历史版本可迁移。若未来有破坏性变更，将在此补充迁移说明。

参见 [docs/API.md](docs/API.md) 与 [docs/design/executor.md](docs/design/executor.md)。
