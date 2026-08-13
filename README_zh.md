# Executor

<p align="center">
  <img src="docs/executor.svg" width="160" alt="Executor 标志">
</p>

[![CI](https://github.com/Linductor-alkaid/executor/actions/workflows/c-cpp.yml/badge.svg)](https://github.com/Linductor-alkaid/executor/actions/workflows/c-cpp.yml) [![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus)](https://isocpp.org/) [![CMake](https://img.shields.io/badge/CMake-3.16%2B-064F8C?logo=cmake)](https://cmake.org/) [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-1793D1)](https://github.com)

> English version: [README.md](README.md)

> 轻量级 C++ 任务执行与线程管理库，提供统一的线程池与专用实时线程管理，支持任务提交、优先级调度、实时周期任务及基础监控；可选 GPU（CUDA/OpenCL）执行器，与 CPU 执行器统一管理。

---

## 特性

- **混合执行模式**
  线程池（普通并发任务）+ 专用实时线程（高实时性任务，如 CAN 通信、传感器采集）

- **默认即最优 Facade**
  零配置用户自动获得当前平台下的最优行为：
  - **自适应线程数**（`min/max_threads` = 0 sentinel，`ExecutorManager` 初始化时探测 `hardware_concurrency()`，探测失败退到 (2, 4)）
  - **工作窃取默认开启**（无锁实现，`max_threads == 1` 时自动关闭）
  - **线程池自动 CPU 亲和性**（空 affinity → 自动分配 [0..hw-1]，保留用户覆盖）
  - **实时线程自动 CPU 亲和性**（空 → 通过 `g_next_rt_cpu_hint` 在当前允许 CPU 集合内 round-robin 自动选择；可用 CPU 数量 <= 1 时不设置亲和性；保留用户覆盖）
  - **自适应实时线程优先级**（`thread_priority` = 0 → 自动建议：cycle ≤ 1 ms → 80，≤ 10 ms → 50，> 10 ms → 0）
  自动决策在平台探测或调优不可用时会退到安全默认，用户显式设值**始终保留**。任务失败、提交拒绝、丢任务和超时不属于调优失败，必须通过 future、返回值、状态计数或监控统计保持可观察；用户可以选择不响应，但库不能吞掉。

- **软任务超时**
  `task_timeout_ms` 是执行前软超时：任务在队列中等待超过配置阈值后，worker 开始执行前会跳过该任务并递增 `timeout_count`。
  已经开始执行的任务不会被强制中断，因为 C++ 没有安全的线程强杀机制。
  长耗时任务应在任务内部自行检查取消条件或 deadline。

- **Linux 实时性加固**
  `RealtimeThreadConfig` 对进程内存锁采用保守语义：
  - `enable_process_memory_lock`（默认 `false` — Linux `mlockall` 影响整个进程和后续映射；仅在完成内存预算评估后显式启用；拒绝原因见 `process_memory_lock_errno`）
  - `timer_slack_ns`（默认 `1` — 尽力设置 1 ns slack 以规避内核 50 µs 默认值；平台不支持或权限不足时回退；`0` 为显式 opt-out）
  - `thread_name`（仍为 `""` 默认 — 库不猜测用户业务命名）
  参考示例：`tests/test_realtime_hardening.cpp`

- **统一 API**
  `Executor` Facade 提供 `submit`、`submit_priority`、`submit_delayed`、`submit_periodic`、`submit_batch`、`submit_batch_no_future` 及实时任务注册

- **基于意图的自动路由**
  `submit_auto()` 为普通 CPU 和显式 CPU/GPU 双路径任务保留 `future` 完成语义；`dispatch_auto()` 为指定无锁或实时后端报告有界队列是否接收；`start_worker()` 管理长期、可中断的 Blocking I/O worker。路由决策、failure event 与 `get_executor_capabilities()` 使后端选择和状态可观察，而不会把不同执行模型伪装成同一种完成契约。

- **批量任务提交**
  `submit_batch()` 和 `submit_batch_no_future()` 可一次性提交大量任务，减少重复提交路径开销。当前版本不承诺固定加速比；收益会随任务数量、任务体、线程数、硬件和构建配置变化，请以本地 benchmark 结果为准。当前记录见 [docs/performance/batch_submit_baseline_2026-07-09.json](docs/performance/batch_submit_baseline_2026-07-09.json)。

- **实时 facade 推送与背压计数器**
  新代码优先使用 `Executor::push_realtime_task()` / `try_push_realtime_task()` 推送实时任务，无需接触 `IRealtimeExecutor*`。失败会返回 `false`，并进入 failure event 及 `dropped_task_count`、`queue_full_count`、`pool_exhausted_count` 等状态计数；既有 `push_task()` API 继续兼容。

- **集成边界清晰**
  `submit_periodic()` 是普通异步线程池上的周期提交，不等同于专用实时线程。实时线程按周期消费有界队列；`wait_for_completion()` 只等待异步执行器；紧急停止由独立硬零旁路处理；PID 等有状态算法的线程安全性由应用保持。完整边界与部署状态见 [API 集成契约](docs/API.md#43-集成契约周期队列与安全路径)。

- **可诊断的 facade 配置 API**
  `initialize_ex()`、`register_realtime_task_ex()`、`start_realtime_task_ex()`、`register_gpu_executor_ex()` 返回 `ExecutorResult`，可通过 `InvalidConfig`、`DuplicateName`、`NotFound`、`BackendUnavailable`、`StartFailed` 等稳定错误码判断原因；旧 `bool` API 委托到这些路径。

- **失败与生命周期可观察**
  `Executor::set_failure_callback()` 让 facade 用户订阅任务异常、提交拒绝、实时丢任务、任务超时、GPU 失败和等待超时；未设置回调时，失败仍保留在 `get_failure_status()` / `get_recent_failures()` 中。`wait_for_completion_ex()` 返回带 `CompletionStatus` 快照的 `WaitResult`，超时调用方仍能看到 active、queued、pending 任务数。

- **通信 Facade 可观察性**
  `executor::comm` 组件提供本地 `CommStats`，可观察 drop、overwrite、stale read、missed phase、timeout、深度、lag 和 latency。可选 `set_event_callback()` 在内部同步之外执行并隔离 callback 异常，但 callback 的配置与调用属于诊断/控制面工作，不是实时安全操作。通信事件默认不计入 `ExecutorFailureStatus` 的任务失败。

- **通信同步核心无锁化**
  `MpscChannel<T>` 与 `RealtimeChannel<T>` 使用构造期预分配的有界 MPSC 存储，并限定一个逻辑消费者。`LatestMailbox<T>` 与未绑定的 SWMR `DoubleBuffer<T>` 使用四个固定的 reader-pin 快照槽，使非平凡 `T` 的复制不依赖存在 data race 的 seqlock。快照 `try_publish()` 是非等待、系统级 lock-free 操作，但竞争中的 CAS 可重试，不承诺单次调用有界或 wait-free；`try_load()` 最多尝试四个槽。`PhaseGate` 与 `Sequencer` 使用非阻塞原子核心；若平台所需同步原子并非 lock-free，组件会在构造时拒绝。`is_synchronization_lock_free()` 只描述这些内部同步原子，不覆盖 `T` 的操作、时钟、字符串/结果构造、callback、预分配存储之外的分配或 OS 调度。超时等待与保证成功的兼容 API 通过 spin/yield 重试，只适合普通控制线程。`Topic<T>` 连同 publish fan-out 仍使用 mutex 与动态分配，不是实时原语。
  快照 sequence 是有限的 56 位域：耗尽时 `try_publish()` 返回 `false`，重试型兼容操作则以 `std::overflow_error` 报告永久耗尽，不会无限 spin。phase 与 ticket 状态仅支持小于 `2^63` 的值；phase/ticket wait 对更大的输入立即返回 `InvalidArgument`，相位绑定 LET 发布还要求 phase 小于 `2^63 - 1`。

- **相位绑定值的可选 LET 契约**
  `PhaseGate` 可通过 `bind_to_phase_gate()` 显式绑定 `DoubleBuffer<T>` 或 `LatestMailbox<T>`。写侧用 `publish_for_current_phase()` 提交相位 N 的值后，读侧只能在相位门进入 N+1 时通过 `load_for_current_phase()` 读取。绑定模式使用固定双槽 SWSR 存储：每相位最多一个值、没有 FIFO 语义、重复提交会被拒绝。它不是独立的 `LetChannel<T>` 类型。未绑定 `DoubleBuffer` 仍是最新完整快照，未绑定 `LatestMailbox` 仍是 latest-wins，`RealtimeChannel` 不自动具有 LET 语义。

- **可选 GPU（CUDA/OpenCL）**
  GPU 执行器接口与 CUDA/OpenCL 实现：kernel 提交、设备内存与流管理、多设备、内存池、监控；运行时动态加载，无 GPU 时安全降级；设备查询 API 自动推荐最佳后端

- **可配置**
  线程数、队列容量、优先级、CPU 亲和性、工作窃取、监控开关等

- **单例 / 实例化**
  支持进程内共享或按项目隔离的独立实例（RAII 生命周期）

- **可选监控**
  任务统计、单后端状态查询和统一生命周期 `Executor::get_snapshot()`；快照覆盖各后端、失败摘要和聚合计数，采用低频 best-effort 语义；可选 `ICycleManager` 集成以精确控制实时周期

- **最小依赖**
  仅依赖 C++ 标准库与平台特定 API（Linux: `pthread`、`rt`；Windows: Win32 API），无第三方必需依赖；GPU 为可选模块（CUDA/OpenCL 头文件 + 运行时动态加载）

- **跨平台支持**
  支持 Linux 和 Windows，自动适配平台特性（如 Windows 高精度定时器）

## 依赖与要求

| 项目 | 要求 |
|------|------|
| **C++ 标准** | C++20 |
| **构建系统** | CMake 3.16+ |
| **平台** | **Linux**：`pthread`、`rt`（实时扩展）<br>**Windows**：Visual Studio 2019+ / MSVC 14.0+，Win32 API |
| **GPU（可选）** | 启用 `EXECUTOR_ENABLE_GPU` 时：<br>- CUDA：需 CUDA Toolkit（头文件），运行时动态加载<br>- OpenCL：需 OpenCL 头文件，运行时动态加载<br>无静态链接，GPU 不可用时安全降级 |

### 平台特定说明

#### Linux
- 需要 `pthread` 和 `librt`（实时扩展库）
- 支持高精度定时器和实时调度策略

#### Windows
- 支持 Visual Studio 2019 及更高版本（MSVC 14.0+）
- 对于短周期实时任务（周期 < 20ms），自动启用高精度定时器（`timeBeginPeriod`）
- 定时器精度：默认 15.6ms，启用高精度后可达 1ms
- 注意：高精度定时器会增加系统功耗，仅在需要时自动启用

## 快速开始

### 构建

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

启用 GPU 支持时：

```bash
# 启用 CUDA（NVIDIA GPU）
cmake -B build -DCMAKE_BUILD_TYPE=Release -DEXECUTOR_ENABLE_GPU=ON -DEXECUTOR_ENABLE_CUDA=ON

# 启用 OpenCL（Intel/AMD/NVIDIA GPU）
cmake -B build -DCMAKE_BUILD_TYPE=Release -DEXECUTOR_ENABLE_GPU=ON -DEXECUTOR_ENABLE_OPENCL=ON

# 同时启用 CUDA 和 OpenCL
cmake -B build -DCMAKE_BUILD_TYPE=Release -DEXECUTOR_ENABLE_GPU=ON -DEXECUTOR_ENABLE_CUDA=ON -DEXECUTOR_ENABLE_OPENCL=ON

cmake --build build
```

查询系统 GPU 设备：

```bash
./build/examples/gpu_device_query
```

### 运行测试

```bash
ctest --test-dir build
```

### 基本用法

可不显式调用 `initialize`/`shutdown`，库会兜底；仍推荐在需要自定义配置或退出时显式调用。

```cpp
#include <executor/executor.hpp>
#include <iostream>

int main() {
    executor::ExecutorConfig config;
    config.min_threads = 4;
    config.max_threads = 16;

    auto& ex = executor::Executor::instance();
    auto init = ex.initialize_ex(config);
    if (!init) {
        std::cerr << "init failed: " << init.message << "\n";
        return 1;
    }

    ex.set_failure_callback([](const executor::ExecutorFailureEvent& event) {
        std::cerr << "executor failure: " << event.message << "\n";
    });

    auto future = ex.submit([]() { return 42; });
    int result = future.get();  // 如任务抛异常，也会在这里重新抛出

    auto wait = ex.wait_for_completion_ex(std::chrono::seconds(1));
    if (!wait.completed) {
        std::cerr << "pending tasks: " << wait.status.pending_tasks << "\n";
    }

    auto failures = ex.get_failure_status();
    std::cout << "result=" << result
              << ", task failures=" << failures.task_exception_count << "\n";
    ex.shutdown();
    return 0;
}
```

> 用户场景式通信示例见 [examples/comm_robot_pipeline.cpp](examples/comm_robot_pipeline.cpp)：在一条传感器采集、规划、记录、实时控制、状态监控流水线中展示 Topic 向 planner/recorder 独立扇出及逐订阅者背压、realtime commands、latest config、startup gate、state snapshot、task dependencies 和 comm observability。[examples/lifecycle_snapshot.cpp](examples/lifecycle_snapshot.cpp) 展示队列积压、任务失败和 shutdown 后的 `Executor` 生命周期 snapshot。更多示例见 [examples/](examples/)（需 `-DEXECUTOR_BUILD_EXAMPLES=ON` 构建）；可通过 `./build/examples/lifecycle_snapshot` 运行该 CPU 示例；GPU 示例 `gpu_basic`、`gpu_multi_device` 需同时启用 GPU）。

### 提交 API 选择

| 工作类型 | 推荐 API | 返回结果表示 |
| --- | --- | --- |
| 普通有限 CPU 工作 | `submit_auto(lambda)` | `future` 的完成或异常 |
| 独立 CPU 与 GPU 实现 | `submit_auto(cpu_gpu_task(cpu, gpu))` | 已选路径的完成或异常 |
| 指定 MPSC 无锁低延迟后端 | `dispatch_auto(LowLatency, task)` | 有界队列已接收任务 |
| 指定周期实时后端 | `dispatch_auto(RealtimeQueue, task)` | 实时队列已接收任务 |
| 长期可中断 I/O 循环 | `start_worker(BlockingWorkerSpec)` | worker 启动与生命周期 handle |

`Auto` 不会静默选择无锁或实时后端。自动路由不能验证 callable 的实时安全、线程安全、GPU 内存所有权或 I/O 可中断性；应用仍须自行负责这些边界。

## 文档

| 文档 | 说明 |
|------|------|
| [使用手册网站](website/README.md) | 中文优先的 VitePress 使用手册；包含从构建到第一个任务的学习路径 |
| [BUILD.md](docs/BUILD.md) | 构建、安装、`find_package`、选项与发布包 |
| [API.md](docs/API.md) | API 使用说明与主要类型 |
| [MIGRATION.md](docs/MIGRATION.md) | 迁移指南（版本升级说明） |
| [阻塞 I/O worker 教程](website/zh/realtime-and-communication/blocking-io-workers.md) | 管理长期、可中断 worker 的所有权与停止，不引入协议依赖 |
| [executor.md](docs/design/executor.md) | 架构与设计 |
| [gpu_executor.md](docs/design/gpu_executor.md) | GPU 执行器扩展设计（CUDA 等） |
| [cpp-project-design.md](docs/design/cpp-project-design.md) | 项目结构与实现 |
| [COVERAGE.md](docs/COVERAGE.md) | 代码覆盖率（gcov/lcov） |
| [Executor 集成 skill](docs/skill/executor-integration/SKILL.md) | 面向 AI 的渐进式集成指南；agent 将 Executor 接入应用前应优先阅读 |
| [Executor 维护者 skill](docs/skill/executor-maintainer/SKILL.md) | 面向贡献者的 AI 优先仓库维护导航图 |

### 使用 AI 集成

当 AI 运行在其他项目中时，先让它读取 Executor 检出目录中的 [`docs/skill/executor-integration/SKILL.md`](docs/skill/executor-integration/SKILL.md)，再开始接入。如果该 AI 无法访问该检出目录，将完整的 `docs/skill/executor-integration/` 目录复制到应用仓库，并在该项目的 agent 指令中引用其中的 `SKILL.md`。具体做法见 [skill 接入方式](docs/skill/executor-integration/references/adoption.md)。

## 安装与集成

### 安装

```bash
cmake --install build --prefix /usr/local
```

### 在项目中使用

通过 `find_package(executor)` 集成：

```cmake
find_package(executor REQUIRED)
add_executable(myapp main.cpp)
target_link_libraries(myapp PRIVATE executor::executor)
```

或使用 `add_subdirectory`：

```cmake
add_subdirectory(path/to/executor)
target_link_libraries(myapp PRIVATE executor::executor)
```

> 📖 详细说明见 [docs/BUILD.md](docs/BUILD.md)

---

## 平台兼容性

### 测试状态

- ✅ **Linux**：完全支持，所有测试通过
- ✅ **Windows**：支持，已通过编译和测试验证
  - 编译：Visual Studio 2019+ / MSVC 14.0+
  - 测试：所有单元测试和集成测试通过
  - 实时精度：短周期任务自动启用高精度定时器

### 已知限制

- **Windows 定时器精度**：虽然启用了高精度定时器，但由于系统调度器的限制，短周期（< 10ms）的精度可能不如 Linux
- **实时调度**：Windows 不支持 Linux 的实时调度策略（SCHED_FIFO/SCHED_RR），使用线程优先级代替

### 实时线程周期精度（误差）

以下为 **实时线程**（`register_realtime_task` + `RealtimeThreadExecutor` 周期回调）在不同周期下的 jitter（实际触发时刻 − 期望时刻，单位 μs）统计。运行 `./build/tests/benchmark_realtime_precision --json`（Windows 下为 `.\build\tests\Debug\benchmark_realtime_precision.exe --json`）可复现。

**更高实时精度需求**：若需进一步压低 jitter（如硬实时、高频率周期），建议接入 **周期管理器**（`RealtimeThreadConfig::cycle_manager`，实现 `ICycleManager`），由外部统一驱动周期并配合实时调度（如 Linux `SCHED_FIFO`）、CPU 隔离等使用。详见 [API.md 第 8 节](docs/API.md) 与 [examples/realtime_can.cpp](examples/realtime_can.cpp)。

#### Linux

完整 JSON 见 [docs/optimization/realtime_precision_linux.json](docs/optimization/realtime_precision_linux.json)。

| 周期 | jitter_us (min) | jitter_us (avg) | jitter_us (p50) | jitter_us (p95) | jitter_us (p99) |
|------|-----------------|-----------------|-----------------|-----------------|-----------------|
| 1 ms | 0.00 | 59.98 | 54.64 | 64.34 | 64.34 |
| 5 ms | 0.00 | 90.47 | 91.39 | 129.46 | 129.46 |
| 10 ms | 0.00 | 81.40 | 85.71 | 104.31 | 104.31 |
| 50 ms | 0.00 | 89.74 | 85.11 | 108.31 | 108.31 |
| 100 ms | 0.00 | 108.96 | 109.16 | 141.39 | 141.39 |

#### Windows

完整 JSON 见 [docs/optimization/realtime_precision_windows.json](docs/optimization/realtime_precision_windows.json)。Windows 非实时系统，调度与定时器分辨率会导致周期回调普遍偏晚；长周期下误差更大，仅适合软实时或对数毫秒级抖动不敏感的场景。

| 周期 | jitter_us (min) | jitter_us (avg) | jitter_us (p50) | jitter_us (p95) | jitter_us (p99) |
|------|-----------------|-----------------|-----------------|-----------------|-----------------|
| 1 ms | 109.00 | 109.00 | 109.00 | 109.00 | 109.00 |
| 5 ms | 0.00 | 1146.65 | 1077.90 | 1947.10 | 1947.10 |
| 10 ms | 0.00 | 1041.09 | 1159.20 | 1530.40 | 1530.40 |
| 50 ms | 0.00 | 7967.12 | 7344.40 | 14731.90 | 14731.90 |
| 100 ms | 0.00 | 10888.87 | 8839.40 | 16736.00 | 16736.00 |

## 版本

当前版本：**v0.3.1**

变更记录见 [CHANGELOG.md](CHANGELOG.md)

---

## 📄 许可

见[LICENSE](LICENSE)
