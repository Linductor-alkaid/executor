# Executor

[![CI](https://github.com/Linductor-alkaid/executor/actions/workflows/c-cpp.yml/badge.svg)](https://github.com/Linductor-alkaid/executor/actions/workflows/c-cpp.yml) [![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus)](https://isocpp.org/) [![CMake](https://img.shields.io/badge/CMake-3.16%2B-064F8C?logo=cmake)](https://cmake.org/) [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-1793D1)](https://github.com)

> 轻量级 C++ 任务执行与线程管理库，提供统一的线程池与专用实时线程管理，支持任务提交、优先级调度、实时周期任务及基础监控。

---

## 特性

- **混合执行模式**  
  线程池（普通并发任务）+ 专用实时线程（高实时性任务，如 CAN 通信、传感器采集）

- **统一 API**  
  `Executor` Facade 提供 `submit`、`submit_priority`、`submit_delayed`、`submit_periodic` 及实时任务注册

- **可配置**  
  线程数、队列容量、优先级、CPU 亲和性、工作窃取、监控开关等

- **单例 / 实例化**  
  支持进程内共享或按项目隔离的独立实例（RAII 生命周期）

- **可选监控**  
  任务统计、执行器状态查询；可选 `ICycleManager` 集成以精确控制实时周期

- **最小依赖**  
  仅依赖 C++ 标准库与平台特定 API（Linux: `pthread`、`rt`；Windows: Win32 API），无第三方必需依赖

- **跨平台支持**  
  支持 Linux 和 Windows，自动适配平台特性（如 Windows 高精度定时器）

## 依赖与要求

| 项目 | 要求 |
|------|------|
| **C++ 标准** | C++20 |
| **构建系统** | CMake 3.16+ |
| **平台** | **Linux**：`pthread`、`rt`（实时扩展）<br>**Windows**：Visual Studio 2019+ / MSVC 14.0+，Win32 API |

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

### 运行测试

```bash
ctest --test-dir build
```

### 基本用法

```cpp
#include <executor/executor.hpp>

int main() {
    // 配置执行器
    executor::ExecutorConfig config;
    config.min_threads = 4;
    config.max_threads = 16;

    // 初始化并提交任务
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

> 更多示例见 [examples/](examples/)（需 `-DEXECUTOR_BUILD_EXAMPLES=ON` 构建）。

## 文档

| 文档 | 说明 |
|------|------|
| [BUILD.md](docs/BUILD.md) | 构建、安装、`find_package`、选项与发布包 |
| [API.md](docs/API.md) | API 使用说明与主要类型 |
| [MIGRATION.md](docs/MIGRATION.md) | 迁移指南（版本升级说明） |
| [executor.md](docs/design/executor.md) | 架构与设计 |
| [cpp-project-design.md](docs/design/cpp-project-design.md) | 项目结构与实现 |
| [COVERAGE.md](docs/COVERAGE.md) | 代码覆盖率（gcov/lcov） |

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

**更高实时精度需求**：若需进一步压低 jitter（如硬实时、高频率周期），建议接入 **周期管理器**（`RealtimeThreadConfig::cycle_manager`，实现 `ICycleManager`），由外部统一驱动周期并配合实时调度（如 Linux `SCHED_FIFO`）、CPU 隔离等使用。详见 [API.md 第 7 节](docs/API.md) 与 [examples/realtime_can.cpp](examples/realtime_can.cpp)。

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

当前版本：**v0.1.1**

变更记录见 [CHANGELOG.md](CHANGELOG.md)

---

## 📄 许可

见[LICENSE](LICENSE)
