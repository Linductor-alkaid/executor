---
title: CPU/GPU Automatic Selection
description: Use submit_auto and GpuScheduler configuration to choose CPU or GPU from task characteristics.
---

# CPU/GPU Automatic Selection

## Goal

Understand how `submit_auto()` chooses a path from task characteristics, when to adjust `GpuScheduler::Config`, and why choosing GPU does not automatically handle an unavailable backend.

## Default selection rules

The scheduler decides in order:

1. Choose GPU when `TaskCharacteristics::prefer_gpu` is true.
2. With adaptive scheduling enabled and at least two CPU and two GPU history samples for similar work, choose the side with lower predicted time.
3. Otherwise choose GPU when data size meets `data_size_threshold` (default 1 MiB) and compute intensity meets `compute_intensity_threshold` (default 2.0).
4. Choose CPU otherwise.

```cpp
executor::gpu::TaskCharacteristics work;
work.data_size_bytes = bytes;
work.compute_intensity = 3.0F;

auto future = executor.submit_auto(work, "cuda0",
    [](void* stream) { run_work(stream); }, gpu_task_config);
future.get();
```

The CPU branch invokes the same callable with a null stream; the GPU branch calls `submit_gpu()`. The callable must correctly handle CPU `nullptr`, and the named executor must already be registered before the GPU branch runs.

## One callable spans two execution environments

Unlike `submit_gpu()`, which may accept a no-argument callable, `submit_auto()` requires a callable that accepts `void*`, since CPU explicitly calls `kernel(nullptr)`:

```cpp
auto data = std::make_shared<WorkData>(prepare_work());
auto future = executor.submit_auto(work, "cuda0",
    [data](void* stream) {
        if (stream == nullptr) {
            run_cpu(*data);
        } else {
            run_gpu(stream, *data);
        }
    }, gpu_task_config);
```

Captured input must meet both CPU and GPU lifetime/thread-safety requirements. Do not let the null branch use a device pointer or the GPU branch use a host view valid only in the submitter stack. Both branches must produce the same business semantics. When inputs are fundamentally different, explicitly selecting `submit()` or `submit_gpu()` is often clearer.

## No implicit fallback

`submit_auto()` decides only from characteristics and scheduler history. If it chooses GPU but `cuda0` is unregistered/unavailable, submission fails explicitly—it does not secretly use CPU. First call `register_gpu_executor_ex()` and inspect status before admitting GPU characteristics or `prefer_gpu`; if registration fails, use `submit()` directly for CPU fallback.

## Tune configuration from measurement

```cpp
auto config = executor.get_scheduler_config();
config.data_size_threshold = 4 * 1024 * 1024;
config.compute_intensity_threshold = 2.5F;
config.enable_adaptive = true;
config.history_size = 200;
executor.update_scheduler_config(config);
```

Thresholds come from real benchmarks, not intuition. Adaptive history is useful only after representative CPU/GPU performance is recorded; recollect after changing hardware, driver, data shape, or kernel. Scheduling policy does not replace backend availability checks or task-level exception handling.

For stream, resource, or multi-device control, use the advanced interfaces deliberately; complete fields and semantics remain in the [API reference](https://github.com/Linductor-alkaid/executor/blob/master/docs/API.md).
