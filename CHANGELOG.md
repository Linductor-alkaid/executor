# Changelog

本文档记录 executor 项目的版本变更。版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

---

## [0.2.2] - 2026-03-13

### 新增

- **批量任务提交 API**：新增 `submit_batch()` 和 `submit_batch_no_future()` 方法
  - `submit_batch()`：批量提交任务并返回 `std::future<void>` 列表
  - `submit_batch_no_future()`：批量提交任务，不返回 future，性能更高（fire-and-forget）
  - 单线程场景性能提升显著：**5-16x 加速**（500-2000 个任务）
  - 适用场景：单线程提交大量任务（500+ 个）
  - 不推荐场景：多线程并发提交（建议使用循环 `submit()`）
  - 底层优化：一次获取锁批量提交，减少锁竞争和内存分配开销
- **无锁任务执行器**：新增 `LockFreeTaskExecutor` 类，提供高性能无锁任务执行能力
  - 支持 MPSC（多生产者单消费者）模式，使用 CAS 操作保证线程安全
  - 单生产者性能：762万 ops/s，P50 延迟 171ns
  - 2生产者性能：528万 ops/s，效率 35%
  - 完全向后兼容单生产者场景（仅 3% 性能损失）
  - 适用场景：高频日志收集、实时事件处理、多线程任务聚合
- **API 接口**：`start()`、`stop()`、`push_task()`、`is_running()`、`pending_count()`、`processed_count()`
- **使用示例**：`examples/lockfree_task_executor_example.cpp` 演示基本用法、日志收集、事件处理

### 测试

- **批量提交性能测试**：
  - `benchmark_batch_submit`：单线程批量提交基准测试
  - `benchmark_batch_scales`：多规模性能测试（500/1000/2000 任务）
  - `benchmark_batch_submit_concurrent`：多线程并发提交测试
  - `test_batch_no_future`：功能正确性测试
- **并发安全性测试**：`test_lockfree_mpsc` 包含 6 项测试
  - 多生产者并发提交、高竞争正确性、队列满处理
  - 数据竞争检测、动态生产者、压力测试（32生产者）
- **性能基准测试**：
  - `benchmark_lockfree_task_executor`：基础延迟和吞吐量测试
  - `benchmark_lockfree_mpsc`：简化吞吐量测试
  - `benchmark_lockfree_mpsc_full`：完整性能评估（延迟/吞吐量/背压/可扩展性）

### 文档

- **API 手册更新**：[docs/API.md](docs/API.md) 新增批量提交 API 说明（第3.3节）
  - 性能特性表格：不同场景下的性能提升数据
  - 适用场景说明：推荐和不推荐的使用场景
  - 最佳实践：代码示例和性能测试数据
- **API 手册更新**：[docs/API.md](docs/API.md) 新增第5节"无锁任务执行器 API"
- **设计文档**：[docs/design/lockfree_user_api.md](docs/design/lockfree_user_api.md) 详细设计方案和使用指南
- **性能基线**：[docs/performance/lockfree_task_executor_baseline.md](docs/performance/lockfree_task_executor_baseline.md) 性能测试结果和分析

---

## [0.2.1] - 2026-03-09

### 新增

- **OpenCL 执行器**：实现 `OpenCLExecutor`，支持跨平台异构计算（Intel/AMD/NVIDIA GPU）
- **OpenCL 动态加载**：运行时加载 OpenCL 库，无静态链接，OpenCL 不可用时安全降级
- **GPU 设备查询 API**：新增 `enumerate_cuda_devices()`、`enumerate_opencl_devices()`、`enumerate_all_devices()`、`get_recommended_backend()` 函数，用户可查询系统可用 GPU 设备及推荐后端
- **设备信息增强**：`GpuDeviceInfo` 新增 `vendor` 字段，标识 GPU 厂商（NVIDIA/AMD/Intel）
- **统一内存支持**：CUDA 执行器支持统一内存（Unified Memory），新增 `allocate_unified_memory()`、`free_unified_memory()`、`prefetch_memory()` 方法；配置选项 `enable_unified_memory`；CPU 与 GPU 可共享内存无需显式传输
- **构建与示例**：`EXECUTOR_ENABLE_OPENCL` 选项；示例 `gpu_opencl`、`gpu_device_query`、`gpu_unified_memory`

### 文档

- **OpenCL 环境搭建指南**：[docs/setup/opencl_setup.md](docs/setup/opencl_setup.md)，包含 Linux/Windows 环境配置、常见问题排查

详细设计见 [docs/design/gpu_executor.md](docs/design/gpu_executor.md)。

---

## [0.2.0] - 2026-01-29

### 新增

- **GPU 执行器（CUDA）**：`IGpuExecutor` 接口，CUDA 执行器实现，与 ExecutorManager/Executor Facade 集成
- **GPU 任务与配置**：`register_gpu_executor`、`submit_gpu`、`get_gpu_executor`、`get_gpu_executor_status`、`get_gpu_executor_names`；`GpuExecutorConfig`、`GpuTaskConfig`、`GpuDeviceInfo`、`GpuExecutorStatus`
- **CUDA 动态加载**：运行时加载 CUDA 库，无静态链接，CUDA 不可用时安全降级
- **GPU 内存与流**：设备内存分配/释放、主机↔设备/设备↔设备拷贝（含异步）、流创建/销毁/同步、流回调
- **多 GPU 设备**：按设备 ID 注册多个执行器；设备间 P2P 拷贝为**实验性**，未在多 GPU 实机充分测试
- **GPU 内存池与监控**：可选内存池（`GpuMemoryManager`）、kernel 与内存统计、异常处理与错误码转换
- **GPU 任务队列**：优先级、批量提交、任务依赖（`submit_kernel_after`）
- **构建与示例**：`EXECUTOR_ENABLE_GPU`、`EXECUTOR_ENABLE_CUDA` 选项；示例 `gpu_basic`、`gpu_multi_device`

### 其他

- **CI**：C/C++ 工作流重构，依赖升级至 v4
- **文档与测试**：实时线程周期精度记录与外部接入说明；定时器优化；消除测试中数据竞态

详细设计见 [docs/design/gpu_executor.md](docs/design/gpu_executor.md)。

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
