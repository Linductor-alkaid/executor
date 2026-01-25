# Changelog

本文档记录 executor 项目的版本变更。版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

---

## [0.1.1] - 2026-01-25

### 优化

- **锁竞争优化**：为 `PriorityScheduler` 的每个优先级队列使用独立锁，减少锁竞争，端到端吞吐量提升 5.3%，延迟 p99 降低 18%
- **内存分配优化**：将 `PriorityScheduler` 从 `shared_ptr<Task>` 改为 `unique_ptr<Task>`，减少内存分配开销和控制块开销
- **批量分发优化**：实现真正的批量任务分发，批量 dequeue/push/负载更新，减少锁操作次数，端到端吞吐量提升 2.9%
- **工作窃取优化**：实现基于负载的智能窃取策略，优先从高负载线程窃取任务，端到端吞吐量提升 5.9%，延迟 p99 降低 44.4%，提交吞吐量提升 7.3%
- **延迟任务处理优化**：使用 `priority_queue` 替代 `vector` + `remove_if`，按执行时间排序，提高延迟任务处理效率
- **任务 ID 生成优化**：使用原子计数器替代时间戳实现，任务 ID 生成性能提升 80-90%，端到端吞吐量提升 7.3%，延迟 p99 降低 44%

### 性能提升

相比 v0.1.0：
- 端到端吞吐量提升 **13.0%**（461,576 → 521,390 tasks/s）
- 延迟 p99 降低 **55%**（0.22μs → 0.10μs）
- 提交吞吐量略有波动，整体保持稳定

详细优化记录和性能测试结果参见 [docs/optimization/PERFORMANCE_OPTIMIZATION.md](docs/optimization/PERFORMANCE_OPTIMIZATION.md)。

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
