---
title: 注册并提交 GPU 工作
description: 在启用 CUDA 或 OpenCL 的设备上注册 GPU 执行器、提交工作并查询运行状态。
---

# 注册并提交 GPU 工作

## 学习目标

在真实可用的 CUDA 或 OpenCL 环境中，通过 Facade 注册一个 GPU executor、提交 kernel 工作并查询状态；资源管理需求再进入高级接口。

## 构建与设备检查

选择一个后端并构建：

```bash
# CUDA（NVIDIA）
cmake -B build -DEXECUTOR_BUILD_EXAMPLES=ON -DEXECUTOR_ENABLE_GPU=ON -DEXECUTOR_ENABLE_CUDA=ON

# OpenCL（Intel / AMD / NVIDIA）
cmake -B build -DEXECUTOR_BUILD_EXAMPLES=ON -DEXECUTOR_ENABLE_GPU=ON -DEXECUTOR_ENABLE_OPENCL=ON
cmake --build build
./build/examples/gpu_device_query
```

设备查询为空时停止在诊断页，不要假设 `device_id = 0` 一定存在。

## 注册、提交与观察

```cpp
executor::gpu::GpuExecutorConfig config;
config.name = "cuda0";
config.backend = executor::gpu::GpuBackend::CUDA;
config.device_id = 0;

const auto registered = executor.register_gpu_executor_ex("cuda0", config);
if (!registered) {
    return run_on_cpu();
}

executor::gpu::GpuTaskConfig task;
auto completed = executor.submit_gpu("cuda0", [](void* stream) {
    launch_kernel(stream);
}, task);
completed.get();

const auto status = executor.get_gpu_executor_status("cuda0");
```

`submit_gpu()` 返回 `future<void>`；调用 `get()` 仍是单次 kernel 提交的异常边界。查询 `get_gpu_executor_names()`、单个 `get_gpu_executor_status()` 或 `get_all_gpu_executor_status()` 可观察已注册 executor、队列、活跃/完成/失败 kernel、内存使用与 `last_error_message`。

## 配置边界

`GpuExecutorConfig` 至少需要非空名称、有效 backend、非负 `device_id`、正的队列容量与 stream 数。`GpuTaskConfig` 包含 grid/block、共享内存、stream、异步和优先级；非默认 stream 必须来自该 executor 的 `create_stream()`，销毁后不能继续使用。

本页保持在 Facade：注册、提交和状态查询。确实需要设备内存、stream 生命周期、统一内存或 P2P 时，才通过 `get_gpu_executor()` 进入高级资源控制，并承担它新增的生命周期责任。

## 性能与复现

不要把“GPU 已注册”推导为“工作负载更快”。任何结论至少记录 GPU 型号、驱动、后端、数据规模、kernel、构建类型、测量方式和 CPU 对照。多设备、内存和 stream 专题示例见 [`examples/gpu_basic.cpp`](https://github.com/Linductor-alkaid/executor/blob/master/examples/gpu_basic.cpp)、[`examples/gpu_multi_device.cpp`](https://github.com/Linductor-alkaid/executor/blob/master/examples/gpu_multi_device.cpp) 与 [`examples/gpu_opencl.cpp`](https://github.com/Linductor-alkaid/executor/blob/master/examples/gpu_opencl.cpp)。

## 下一步阅读

[CPU/GPU 自动选择](/zh/gpu/automatic-scheduling)说明何时让调度器按任务特征选择执行路径。
