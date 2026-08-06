# Executor

<p align="center">
  <img src="docs/executor.svg" width="160" alt="Executor logo">
</p>

[![CI](https://github.com/Linductor-alkaid/executor/actions/workflows/c-cpp.yml/badge.svg)](https://github.com/Linductor-alkaid/executor/actions/workflows/c-cpp.yml) [![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus)](https://isocpp.org/) [![CMake](https://img.shields.io/badge/CMake-3.16%2B-064F8C?logo=cmake)](https://cmake.org/) [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-1793D1)](https://github.com)

> 中文文档: [README_zh.md](README_zh.md)

> A lightweight C++ task execution and thread management library providing a unified thread pool and dedicated real-time thread management. Supports task submission, priority scheduling, real-time periodic tasks, and basic monitoring. Optional GPU (CUDA/OpenCL) executor managed alongside the CPU executor through a unified API.

---

## Features

- **Hybrid Execution Modes**
  Thread pool (general concurrent tasks) + dedicated real-time thread (high real-time tasks such as CAN communication, sensor acquisition)

- **Default-Optimal Facade**
  Zero-config users get the best behavior on their platform automatically:
  - **Adaptive thread count** (`min/max_threads` = 0 sentinel, `ExecutorManager` probes `hardware_concurrency()` at init, falls back to (2, 4) on failure)
  - **Work-stealing by default** (lock-free implementation, auto-disabled when `max_threads == 1`)
  - **Auto CPU affinity for thread pool** (empty affinity → auto-allocate [0..hw-1], preserves user override)
  - **Auto CPU affinity for real-time threads** (empty → round-robin auto-select via `g_next_rt_cpu_hint` across the currently allowed CPU set; no affinity is set when one or fewer CPUs are available; preserves override)
  - **Adaptive real-time thread priority** (`thread_priority` = 0 → auto-recommend 80 if cycle ≤ 1 ms, 50 if ≤ 10 ms, 0 if > 10 ms)
  Auto-decisions fall back to safe defaults when platform probing or tuning is unavailable, and user-supplied values are **always preserved**. Task failures, rejected submissions, drops, and timeouts are not considered tuning failures: they must remain observable through futures, return values, status counters, or monitoring statistics.

- **Soft Task Timeout**
  `task_timeout_ms` is a pre-execution soft timeout: if a queued task has waited longer than the configured timeout before a worker starts it, the task is skipped and `timeout_count` is incremented.
  In-progress tasks are never forcefully interrupted because C++ has no safe thread-kill mechanism.
  Long-running tasks should check their own cancellation or deadline condition internally.

- **Linux Real-Time Hardening**
  `RealtimeThreadConfig` uses conservative process-memory-lock semantics:
  - `enable_process_memory_lock` (default `false` — Linux `mlockall` affects the whole process and future mappings; explicitly opt in only after sizing the memory budget; `process_memory_lock_errno` reports denied requests)
  - `timer_slack_ns` (default `1` — best-effort 1 ns slack to avoid kernel's 50 µs default; unsupported or denied calls fall back; `0` is now explicit opt-out)
  - `thread_name` (still `""` by default — library doesn't guess user business names)
  Reference example: `tests/test_realtime_hardening.cpp`

- **Unified API**
  `Executor` facade provides `submit`, `submit_priority`, `submit_delayed`, `submit_periodic`, `submit_batch`, `submit_batch_no_future`, and real-time task registration

- **Intent-Based Routing**
  `submit_auto()` keeps future semantics for ordinary CPU and explicit CPU/GPU dual-path work; `dispatch_auto()` reports bounded queue admission for named lock-free or realtime backends; `start_worker()` owns a long-lived interruptible Blocking I/O worker. Routing decisions, failure events, and `get_executor_capabilities()` make backend selection and state observable without pretending these models share one completion contract.

- **Batch Task Submission**
  `submit_batch()` and `submit_batch_no_future()` submit many tasks through a single batch API and can reduce repeated submission overhead. The current version does not promise a fixed speedup; gains depend on task count, task body, thread count, hardware, and build configuration. Treat local benchmark results as authoritative. Current benchmark record: [docs/performance/batch_submit_baseline_2026-07-09.json](docs/performance/batch_submit_baseline_2026-07-09.json), commands `cmake --build build --target benchmark_batch_scales benchmark_batch_submit_real benchmark_batch_submit_concurrent -j2`, `./build/tests/benchmark_batch_scales`, `./build/tests/benchmark_batch_submit_real`, and `./build/tests/benchmark_batch_submit_concurrent`, date 2026-07-09.

- **Lockfree MPSC Benchmark Baseline**
  The current-commit record is [tests/benchmarks/baselines/db589fb.json](tests/benchmarks/baselines/db589fb.json). Configure a Release build with tests enabled, then run `cmake --build build --target benchmark_lockfree_mpsc_full benchmark_lockfree_task_executor benchmark_lockfree_mpsc` and `taskset -c 24 ./build/tests/benchmark_lockfree_mpsc --json > tests/benchmarks/baselines/<short-sha>.json`. Pin the entire benchmark process to one CPU allowed by `taskset -pc $$` and record `/sys/devices/system/cpu/cpu<N>/cpufreq/scaling_governor`; comparison runs should use the same governor and pinning policy. The command emits raw benchmark JSON; the committed sidecar schema wraps it with `schema_version`, `benchmark` (target, metric definition, queue capacity, `enable_stats`, `reservation_wait_yields`), `capture` (commit, platform, CPU pinning/governor, compiler), and `raw_results` (producer count, P50/P99 latency, throughput). `benchmark-baseline.yml` runs on manual dispatch and nightly schedule, creates the same JSON envelope as an artifact, and reports drift without failing the workflow.

- **Real-Time Facade Push and Backpressure Counters**
  Use `Executor::push_realtime_task()` / `try_push_realtime_task()` to push real-time work without touching `IRealtimeExecutor*`. Failures return `false` and are observable via failure events plus counters such as `dropped_task_count`, `queue_full_count`, and `pool_exhausted_count`; the existing `push_task()` API remains compatible.

- **Clear Integration Boundaries**
  `submit_periodic()` schedules periodic work on the general asynchronous pool and is not a dedicated real-time thread. Real-time threads consume bounded queues per cycle; `wait_for_completion()` waits only for the asynchronous executor; a separate hard-zero bypass handles emergency stops; and application code retains the thread-safety responsibility for stateful algorithms such as PIDs. See the [API integration contract](docs/API.md#43-集成契约周期队列与安全路径) for the complete boundaries and deployment status.

- **Diagnosable Facade Setup APIs**
  `initialize_ex()`, `register_realtime_task_ex()`, `start_realtime_task_ex()`, and `register_gpu_executor_ex()` return `ExecutorResult` with stable error codes such as `InvalidConfig`, `DuplicateName`, `NotFound`, `BackendUnavailable`, and `StartFailed`; legacy `bool` APIs delegate to these paths.

- **Failure and Lifecycle Observability**
  `Executor::set_failure_callback()` lets facade users subscribe to task exceptions, rejected submissions, real-time drops, task timeouts, GPU failures, and wait timeouts. Failures are also retained in `get_failure_status()` / `get_recent_failures()` when no callback is installed. `wait_for_completion_ex()` returns `WaitResult` with a `CompletionStatus` snapshot so timeout callers can still see active, queued, and pending task counts.

- **Communication Facade Observability**
  `executor::comm` components expose local `CommStats` counters for drops, overwrites, stale reads, missed phases, timeouts, depth, lag, and latency. Optional `set_event_callback()` hooks are isolated from the data path: callback exceptions are swallowed and communication events are not counted as `ExecutorFailureStatus` task failures by default.

- **Optional GPU (CUDA/OpenCL)**
  GPU executor interface with CUDA/OpenCL implementations: kernel submission, device memory and stream management, multi-device, memory pool, monitoring. `add_stream_callback` is currently CUDA-only; check `supports_stream_callback()` before use, and inspect `get_status().last_error_message` when it returns `false`. OpenCL callback support via `cl_event` polling is a follow-up. Runtime dynamic loading with safe graceful degradation when no GPU is available. Device query API automatically recommends the best backend.

- **Configurable**
  Thread count, queue capacity, priority, CPU affinity, work stealing, monitoring toggle, and more

- **Singleton / Instance-Based**
  Supports a shared in-process singleton or isolated independent instances per project (RAII lifecycle)

- **Optional Monitoring**
  Task statistics, per-backend status queries, and the unified lifecycle `Executor::get_snapshot()` API. The snapshot includes backend states, failure summaries, and aggregate counters with low-frequency best-effort semantics. Optional `ICycleManager` integration for precise real-time cycle control

- **Minimal Dependencies**
  Depends only on the C++ standard library and platform-specific APIs (Linux: `pthread`, `rt`; Windows: Win32 API). No required third-party dependencies. GPU is an optional module (CUDA/OpenCL headers + runtime dynamic loading).

- **Cross-Platform Support**
  Supports Linux and Windows with automatic adaptation of platform features (e.g., Windows high-resolution timers)

## Dependencies & Requirements

| Item | Requirement |
|------|-------------|
| **C++ Standard** | C++20 |
| **Build System** | CMake 3.16+ |
| **Platform** | **Linux**: `pthread`, `rt` (real-time extensions)<br>**Windows**: Visual Studio 2019+ / MSVC 14.0+, Win32 API |
| **GPU (optional)** | When `EXECUTOR_ENABLE_GPU` is enabled:<br>- CUDA: CUDA Toolkit (headers required), runtime loaded dynamically<br>- OpenCL: OpenCL headers required, runtime loaded dynamically<br>No static linking; safe graceful degradation when GPU is unavailable |

### Platform-Specific Notes

#### Linux
- Requires `pthread` and `librt` (real-time extension library)
- Supports high-resolution timers and real-time scheduling policies

#### Windows
- Supports Visual Studio 2019 and later (MSVC 14.0+)
- For short-cycle real-time tasks (cycle < 20 ms), high-resolution timers (`timeBeginPeriod`) are automatically enabled
- Timer precision: 15.6 ms by default; up to 1 ms with high-resolution mode enabled
- Note: high-resolution timers increase system power consumption and are only enabled automatically when needed

## Quick Start

### Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

To enable GPU support:

```bash
# Enable CUDA (NVIDIA GPU)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DEXECUTOR_ENABLE_GPU=ON -DEXECUTOR_ENABLE_CUDA=ON

# Enable OpenCL (Intel/AMD/NVIDIA GPU)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DEXECUTOR_ENABLE_GPU=ON -DEXECUTOR_ENABLE_OPENCL=ON

# Enable both CUDA and OpenCL
cmake -B build -DCMAKE_BUILD_TYPE=Release -DEXECUTOR_ENABLE_GPU=ON -DEXECUTOR_ENABLE_CUDA=ON -DEXECUTOR_ENABLE_OPENCL=ON

cmake --build build
```

Enumerate GPU devices on the system:

```bash
./build/examples/gpu_device_query
```

### Run Tests

```bash
ctest --test-dir build
```

### Basic Usage

Explicit calls to `initialize`/`shutdown` are optional — the library provides a fallback; however, explicit calls are still recommended when custom configuration or controlled shutdown is needed.

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
    int result = future.get();  // also rethrows the task exception, if any

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

> For a user-scenario communication example, see [examples/comm_robot_pipeline.cpp](examples/comm_robot_pipeline.cpp): sensor frames, realtime commands, latest config, startup gating, state snapshots, task dependencies, and comm observability in one pipeline. Build examples with `-DEXECUTOR_BUILD_EXAMPLES=ON`. GPU examples `gpu_basic` and `gpu_multi_device` also require GPU support to be enabled.

### Choosing a Submission API

| Work type | Recommended API | Result means |
| --- | --- | --- |
| Ordinary finite CPU work | `submit_auto(lambda)` | `future` completion or exception |
| Separate CPU and GPU implementations | `submit_auto(cpu_gpu_task(cpu, gpu))` | selected path completion or exception |
| Named MPSC low-latency backend | `dispatch_auto(LowLatency, task)` | bounded queue accepted the task |
| Named periodic realtime backend | `dispatch_auto(RealtimeQueue, task)` | realtime queue accepted the task |
| Long-lived interruptible I/O loop | `start_worker(BlockingWorkerSpec)` | worker startup and lifecycle handle |

`Auto` does not silently choose lock-free or realtime backends. Automatic routing cannot verify callable realtime safety, thread safety, GPU memory ownership, or I/O interruptibility; applications retain those responsibilities.

## Documentation

| Document | Description |
|----------|-------------|
| [User guide website](website/README.md) | Chinese-first VitePress user guide, including the build-to-first-task learning path |
| [BUILD.md](docs/BUILD.md) | Build, install, `find_package`, options, and release packages |
| [API.md](docs/API.md) | API usage and primary types |
| [MIGRATION.md](docs/MIGRATION.md) | Migration guide (version upgrade notes) |
| [Blocking I/O worker tutorial](website/en/realtime-and-communication/blocking-io-workers.md) | Own and stop a long-lived, interruptible worker without adding a protocol dependency |
| [executor.md](docs/design/executor.md) | Architecture and design |
| [gpu_executor.md](docs/design/gpu_executor.md) | GPU executor extension design (CUDA, etc.) |
| [cpp-project-design.md](docs/design/cpp-project-design.md) | Project structure and implementation |
| [COVERAGE.md](docs/COVERAGE.md) | Code coverage (gcov/lcov) |

## Installation & Integration

### Install

```bash
cmake --install build --prefix /usr/local
```

### Use in Your Project

Integrate via `find_package(executor)`:

```cmake
find_package(executor REQUIRED)
add_executable(myapp main.cpp)
target_link_libraries(myapp PRIVATE executor::executor)
```

Or use `add_subdirectory`:

```cmake
add_subdirectory(path/to/executor)
target_link_libraries(myapp PRIVATE executor::executor)
```

> 📖 For detailed instructions see [docs/BUILD.md](docs/BUILD.md)

---

## Platform Compatibility

### Test Status

- ✅ **Linux**: Fully supported, all tests passing
- ✅ **Windows**: Supported, verified by compilation and testing
  - Build: Visual Studio 2019+ / MSVC 14.0+
  - Tests: All unit tests and integration tests passing
  - Real-time precision: high-resolution timers automatically enabled for short-cycle tasks

### Known Limitations

- **Windows Timer Precision**: Despite high-resolution timers being enabled, precision for short cycles (< 10 ms) may be lower than Linux due to system scheduler constraints
- **Real-Time Scheduling**: Windows does not support Linux real-time scheduling policies (SCHED_FIFO/SCHED_RR); thread priorities are used instead

### Real-Time Thread Cycle Precision (Jitter)

The table below shows jitter statistics (actual trigger time − expected time, in µs) for **real-time threads** (`register_realtime_task` + `RealtimeThreadExecutor` cycle callback) at various cycle periods. Run `./build/tests/benchmark_realtime_precision --json` (Windows: `.\build\tests\Debug\benchmark_realtime_precision.exe --json`) to reproduce.

**For higher real-time precision**: if you need to reduce jitter further (e.g., hard real-time, high-frequency cycles), consider integrating a **cycle manager** (`RealtimeThreadConfig::cycle_manager`, implementing `ICycleManager`) to drive cycles externally in conjunction with real-time scheduling (e.g., Linux `SCHED_FIFO`), CPU isolation, etc. See [API.md section 8](docs/API.md) and [examples/realtime_can.cpp](examples/realtime_can.cpp).

#### Linux

Full JSON: [docs/optimization/realtime_precision_linux.json](docs/optimization/realtime_precision_linux.json).

| Period | jitter_us (min) | jitter_us (avg) | jitter_us (p50) | jitter_us (p95) | jitter_us (p99) |
|--------|-----------------|-----------------|-----------------|-----------------|-----------------|
| 1 ms   | 0.00 | 59.98 | 54.64 | 64.34 | 64.34 |
| 5 ms   | 0.00 | 90.47 | 91.39 | 129.46 | 129.46 |
| 10 ms  | 0.00 | 81.40 | 85.71 | 104.31 | 104.31 |
| 50 ms  | 0.00 | 89.74 | 85.11 | 108.31 | 108.31 |
| 100 ms | 0.00 | 108.96 | 109.16 | 141.39 | 141.39 |

#### Windows

Full JSON: [docs/optimization/realtime_precision_windows.json](docs/optimization/realtime_precision_windows.json). Windows is not a real-time OS; scheduler and timer resolution cause cycle callbacks to fire consistently late. Errors are larger at longer periods. Suitable only for soft real-time or scenarios tolerant of millisecond-level jitter.

| Period | jitter_us (min) | jitter_us (avg) | jitter_us (p50) | jitter_us (p95) | jitter_us (p99) |
|--------|-----------------|-----------------|-----------------|-----------------|-----------------|
| 1 ms   | 109.00 | 109.00 | 109.00 | 109.00 | 109.00 |
| 5 ms   | 0.00 | 1146.65 | 1077.90 | 1947.10 | 1947.10 |
| 10 ms  | 0.00 | 1041.09 | 1159.20 | 1530.40 | 1530.40 |
| 50 ms  | 0.00 | 7967.12 | 7344.40 | 14731.90 | 14731.90 |
| 100 ms | 0.00 | 10888.87 | 8839.40 | 16736.00 | 16736.00 |

## Version

Current version: **v0.3.1**

See [CHANGELOG.md](CHANGELOG.md) for the change log.

---

## 📄 License

See [LICENSE](LICENSE)
