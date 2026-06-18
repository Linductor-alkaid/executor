# Executor

[![CI](https://github.com/Linductor-alkaid/executor/actions/workflows/c-cpp.yml/badge.svg)](https://github.com/Linductor-alkaid/executor/actions/workflows/c-cpp.yml) [![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus)](https://isocpp.org/) [![CMake](https://img.shields.io/badge/CMake-3.16%2B-064F8C?logo=cmake)](https://cmake.org/) [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-1793D1)](https://github.com)

> 中文文档: [README_zh.md](README_zh.md)

> A lightweight C++ task execution and thread management library providing a unified thread pool and dedicated real-time thread management. Supports task submission, priority scheduling, real-time periodic tasks, and basic monitoring. Optional GPU (CUDA/OpenCL) executor managed alongside the CPU executor through a unified API.

---

## Features

- **Hybrid Execution Modes**
  Thread pool (general concurrent tasks) + dedicated real-time thread (high real-time tasks such as CAN communication, sensor acquisition)

- **Default-Optimal Facade (P019)**
  Zero-config users get the best behavior on their platform automatically:
  - **Adaptive thread count** (`min/max_threads` = 0 sentinel, `ExecutorManager` probes `hardware_concurrency()` at init, falls back to (2, 4) on failure)
  - **Work-stealing by default** (lock-free implementation, auto-disabled when `max_threads == 1`)
  - **Auto CPU affinity for thread pool** (empty affinity → auto-allocate [0..hw-1], preserves user override)
  - **Auto CPU affinity for real-time threads** (empty → bind core 0 if hw >= 2, else OS-free; preserves override)
  - **Adaptive real-time thread priority** (`thread_priority` = 0 → auto-recommend 80 if cycle ≤ 1 ms, 50 if ≤ 10 ms, 0 if > 10 ms)
  All auto-decisions **fail silent** and user-supplied values are **always preserved**.

- **Linux Real-Time Hardening (P016 + P019-A)**
  `RealtimeThreadConfig` defaults are now opt-out:
  - `enable_memory_lock` (default `true` — `mlockall` to avoid page-fault jitter; failure silent)
  - `timer_slack_ns` (default `1` — 1 ns slack to avoid kernel's 50 µs default; failure silent; `0` is now explicit opt-out)
  - `thread_name` (still `""` by default — library doesn't guess user business names)
  Reference example: `tests/test_realtime_hardening.cpp`

- **Unified API**
  `Executor` facade provides `submit`, `submit_priority`, `submit_delayed`, `submit_periodic`, `submit_batch`, `submit_batch_no_future`, and real-time task registration

- **Batch Task Submission**
  `submit_batch()` and `submit_batch_no_future()` efficiently submit large numbers of tasks, with **5–16x** throughput improvement in single-threaded scenarios (500–2000 tasks)

- **Optional GPU (CUDA/OpenCL)**
  GPU executor interface with CUDA/OpenCL implementations: kernel submission, device memory and stream management, multi-device, memory pool, monitoring. Runtime dynamic loading with safe graceful degradation when no GPU is available. Device query API automatically recommends the best backend.

- **Configurable**
  Thread count, queue capacity, priority, CPU affinity, work stealing, monitoring toggle, and more

- **Singleton / Instance-Based**
  Supports a shared in-process singleton or isolated independent instances per project (RAII lifecycle)

- **Optional Monitoring**
  Task statistics and executor state queries. Optional `ICycleManager` integration for precise real-time cycle control

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

int main() {
    // Configure the executor
    executor::ExecutorConfig config;
    config.min_threads = 4;
    config.max_threads = 16;

    // Initialize and submit tasks
    auto& ex = executor::Executor::instance();
    ex.initialize(config);

    auto future = ex.submit([]() {
        return 42;
    });

    int result = future.get();
    ex.shutdown();
    return 0;
}
```

> For more examples see [examples/](examples/) (build with `-DEXECUTOR_BUILD_EXAMPLES=ON`). GPU examples `gpu_basic` and `gpu_multi_device` also require GPU support to be enabled.

## Documentation

| Document | Description |
|----------|-------------|
| [BUILD.md](docs/BUILD.md) | Build, install, `find_package`, options, and release packages |
| [API.md](docs/API.md) | API usage and primary types |
| [MIGRATION.md](docs/MIGRATION.md) | Migration guide (version upgrade notes) |
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

Current version: **v0.2.2**

See [CHANGELOG.md](CHANGELOG.md) for the change log.

---

## 📄 License

See [LICENSE](LICENSE)
