# Changelog

本文档记录 executor 项目的版本变更。版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

---

## [0.2.2] - 2026-06-18

### 修复

- **v0.2.1 紧跟的 CI 修复**：见下方 `### 测试 / CI` 段。
- **无锁基础设施稳定性**（无 PR）：修复 benchmark 超时、`shutdown(true)` 任务挂起、`dispatch_batch` 任务丢失、周期任务取消竞态、无锁队列容量检测与内存可见性问题。
- **ObjectPool ABA 关键修复**（无 PR）：将 `ObjectPool` free list 从无锁 CAS 改为 mutex 保护，彻底消除 ABA 导致的 SEGFAULT；影响 `LockFreeTaskExecutor`、`RealtimeThreadExecutor` 等使用 `ObjectPool<Task>` 的执行器，`acquire()` / `release()` 接口保持兼容。
- **ObjectPool 入参与释放防护** [#3]：拒绝 `capacity=0` 构造，避免无效对象池配置。
- **ObjectPool release 防护**（无 PR）：新增 double-free / foreign pointer guard，防止重复释放或外部指针污染对象池。
- **ObjectPool::release O(1)** [#41]：优化释放路径，保留正确性防护并降低释放开销。
- **ThreadPool WorkerLocalQueue** [#1] [#6]：修复 `empty()` 判断逻辑与 `steal()` 竞态。
- **ThreadPool 状态统计** [#2]：修复 `get_status().idle_threads` 的 `size_t` 下溢。
- **ThreadPool 并发关闭** [#11]：修复 `stop()` / `shutdown()` 并发调用导致 double-join UB。
- **ThreadPool resize / dispatch** [#14]：worker 无效时任务重新入队，避免 resize 期间丢任务。
- **ThreadPool try_steal_task** [#15]：对 `local_queues_` 使用 `shared_lock`，修复 resize 并发访问竞态。
- **ThreadPool resize UAF**（无 PR）：使用 `shared_lock` 防护 resize 期间的队列生命周期。
- **LoadBalancer 数据竞争** [#12]：将 `strategy_` 改为 atomic。
- **LockFreeQueue 数据竞争** [#4] [#13]：`size()` 使用 acquire ordering，`stats_enabled_` 改为 atomic。
- **LockFree batch 异常安全** [#31]：`push_tasks_batch` 在对象池耗尽与部分入队场景下保持资源回收正确。
- **LockFreeTaskExecutor 构造泄漏** [#33]：使用 `unique_ptr` 替换裸指针，修复构造失败路径泄漏。
- **LockFreeTaskExecutor 异常可见性** [#44]：任务异常可被统计与观察，避免后台吞掉故障。
- **GPU submit_kernel_after** [#7] [#28]：`submit_kernel_after` 不阻塞 GPU worker，并修复依赖任务 UAF。
- **CudaExecutor wait_for_completion UAF** [#39]：修复等待完成期间的生命周期问题。
- **CudaExecutor submit 不死锁** [#43]：修复提交路径中可能出现的死锁。
- **Realtime 周期任务预算** [#40]：周期任务超预算时正确记录与处理。
- **simple_cycle_loop skip-late** [#42]：周期循环对过晚周期执行跳过策略，降低积压。
- **set_thread_priority nice** [#45]：Linux 下真正应用 nice 值，修复优先级配置未生效问题。
- **Windows 编译调整**（无 PR）：修复 Windows 平台编译兼容性。

### 新增

- **LockFreeTaskExecutor**（无 PR）：新增 MPSC 无锁任务执行器，提供 `start()`、`stop()`、`push_task()`、`is_running()`、`pending_count()`、`processed_count()` 与队列统计接口，适用于高频日志、实时事件与多线程任务聚合。
- **批量任务提交 API**（无 PR）：新增 `submit_batch()` 与 `submit_batch_no_future()`，单线程 500-2000 任务场景可获得 **5-16x** 加速。
- **LockFreeTaskExecutor 批量提交**（无 PR）：新增 `push_tasks_batch()`，支持尽力批量入队与实际入队数量回传。
- **智能调度接口**（无 PR）：新增智能调度与自适应调度能力，为后续 facade 默认优化提供基础。
- **实时 push_task 背压可见性** [#32]：新增 `push_task_ex()` 与 `dropped_task_count` / `failed_pushes` 等状态字段；`push_task()` 保持 void 兼容，背压丢任务可被观测。
- **软任务超时** [#24]：新增 `task_timeout_ms` 软超时语义，执行前 `elapsed >= timeout` 时跳过并计入 `timeout_count`；C++ 无安全线程终止机制，执行中的任务不被强制中断。

### 优化

- **无锁 MPSC 基础设施**（无 PR）：从 MPSC 队列、无锁任务执行器、批量提交一路演进到序列号 MPSC 队列、False Sharing 消除、CAS 重试策略优化、性能监控优化与 worker local queue 无锁化。
- **无锁工作线程队列**（无 PR）：`WorkerLocalQueue` 改造为无锁实现，提交吞吐量 **441,500 → 488,698 tasks/s（+10.7%）**，端到端吞吐量 **433,083 → 442,009 tasks/s（+2.1%）**。
- **Linux 实时性加固** [#16]：`RealtimeThreadExecutor` 增加 `mlockall`、`timer_slack`、线程命名等加固，1ms 周期 jitter p99 从 61 µs 压至约 15-20 µs。
- **Default-Optimal Facade (P019)** [#19] [#20] [#21] [#22]：
  - `enable_memory_lock` / `timer_slack_ns` 从 opt-in 改为 opt-out，默认开启实时性优化。
  - `min_threads` / `max_threads = 0` 时自动探测 `hardware_concurrency()`，`work_stealing` 默认开启。
  - 线程池 `cpu_affinity` 空时自动分配 [0..hw-1]，实时线程按周期自适应优先级。
  - 实时线程 `cpu_affinity` 空时自动绑核；多实时线程使用 round-robin 自动亲和性。
  - 1ms 周期 jitter p99 从 54.64 µs 降至 **1.77-6.64 µs**，降低 **89-97%**。
- **ThreadPool soft timeout** [#24]：执行前跳过超时任务并计数，避免误导用户认为执行中任务会被强杀。
- **LockFree spin+yield** [#26]：用 spin+yield 替换 100µs busy-sleep，降低无锁执行器等待延迟。
- **Realtime Windows timer 数据竞争** [#27]：`timer_period_ms_` 改为 atomic。
- **Realtime 多线程亲和性** [#29]：多个实时线程自动 round-robin 分配 CPU affinity。
- **GPU 性能优化**（无 PR）：补充 GPU 性能优化与性能测试报告。

### 文档

- **ObjectPool ABA 设计说明**（无 PR）：新增 ABA 修复设计文档，说明从 CAS free list 切换到 mutex 的正确性取舍。
- **API / README / CHANGELOG / MIGRATION 同步** [#17]：同步公开 API、默认值、迁移说明与发布记录。
- **README 拆分** [#18]：拆分英文 `README.md` 与中文 `README_zh.md`。
- **P019 facade 文档同步** [#23]：同步默认即最优 facade 哲学、实时性默认值与性能描述。
- **批量提交与软超时语义** [#25]：补充 `push_tasks_batch` 与 `task_timeout_ms` soft timeout 说明。
- **LockFreeQueue empty / size 语义** [#30]：说明 `empty()` 与 `size()` 在并发场景下的近似语义。
- **API 背压字段** [#46]：补充 `push_task_ex()`、`dropped_task_count`、`failed_pushes`、`queue_capacity` 等背压字段说明。

### 测试 / CI

- **v0.2.1 紧跟的 CI 修复**（无 PR）：连续修复 5 个 CI 问题，稳定 0.2.1 后续发布分支。
- **benchmark_batch_* 超时修复**（无 PR）：修复批量提交 benchmark 测试超时。
- **benchmark_lockfree_task_executor timeout**（无 PR）：CTest timeout 从 30s 调整为 120s，避免高吞吐压测误判超时。
- **无锁队列与批量提交测试清理**（无 PR）：移除无用测试文件并补强批量、并发、工作窃取相关测试。
- **benchmark latency 阈值调整**（无 PR）：放宽 `latency_single_task` P99 限制到 100µs，降低 CI 环境噪声误报。
- **Code Coverage mlockall 跳过**（无 PR）：覆盖率任务中跳过 `mlockall`，避免 OOM。
- **CUDA 测试头文件包含**（无 PR）：为 `test_unified_memory` 与 `test_gpu_dep_async` 补充 CUDA headers。

### 构建

- **Windows 编译调整**（无 PR）：修复 Windows 平台构建问题。
- **CMake 4.x CUDA Toolkit**（无 PR）：从 `CUDAToolkit` 推导 `CUDA_INCLUDE_DIRS`，兼容 CMake 4.x。
- **CUDA / no-CUDA 安装策略**（无 PR）：CUDA executor 运行时通过 `dlopen libcuda` 动态加载；deb 发布使用带 CUDA 的完整构建，用户机器无 CUDA 时运行时自动降级。

### 性能基准

- **任务提交吞吐量**：`benchmark_baseline` 从 v0.2.0 的 456,703 tasks/s 保持同档并在 commit path 达到约 488K+，约 **+7%+**。
- **MPSC 工作窃取场景**：提交吞吐 **441,500 → 488,698 tasks/s（+10.7%）**；端到端 **433,083 → 442,009 tasks/s（+2.1%）**。
- **实时线程 1ms 周期 jitter**：p99 **61.30 µs → 1.77-6.64 µs（-89% ~ -97%）**；avg **54.47 µs → 1-2 µs（约 -95%）**。
- **LockFreeTaskExecutor SPSC**：10K 任务提交平均 **97.29 ns**，p50 **29 ns**，p99 **1013 ns**，吞吐 **8,242,895 ops/s**。
- **LockFreeTaskExecutor 端到端**：100K 任务端到端吞吐 **5,942,007 ops/s**。
- **批量提交**：单线程 500-2000 任务场景 **5-16x** 加速。

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
