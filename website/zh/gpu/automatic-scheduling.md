---
title: CPU/GPU 自动选择
description: 使用 submit_auto 和 GpuScheduler 配置基于任务特征选择 CPU 或 GPU。
---

# CPU/GPU 自动选择

## 学习目标

理解 `submit_auto()` 如何依据任务特征选择路径，何时调整 `GpuScheduler::Config`，以及为什么“选择 GPU”不等于自动处理后端不可用。

## 默认选择规则

调度器按以下顺序决定：

1. `TaskCharacteristics::prefer_gpu` 为真时选择 GPU。
2. 启用自适应调度且相似任务的 CPU/GPU 历史各至少有两条时，选择预测耗时更短的一侧。
3. 否则，当数据大小达到 `data_size_threshold`（默认 1 MiB）且计算强度达到 `compute_intensity_threshold`（默认 2.0）时选择 GPU。
4. 其余情况选择 CPU。

```cpp
executor::gpu::TaskCharacteristics work;
work.data_size_bytes = bytes;
work.compute_intensity = 3.0F;

auto future = executor.submit_auto(work, "cuda0",
    [](void* stream) { run_work(stream); }, gpu_task_config);
future.get();
```

CPU 路径以空 stream 调用同一个 callable；GPU 路径调用 `submit_gpu()`。因此 callable 必须能正确处理 CPU 的 `nullptr` stream，也必须在 GPU 路径调用前已成功注册指定 executor。

## 一个 callable 必须覆盖两种输入环境

与 `submit_gpu()` 可接受无参数 callable 不同，`submit_auto()` 的 callable 必须能够以 `void*` 参数调用，因为 CPU 分支会明确执行 `kernel(nullptr)`：

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

捕获的数据必须同时满足 CPU 与 GPU 路径的生命周期和线程安全要求。不要让 `nullptr` 分支误用设备指针，也不要让 GPU 分支访问只在提交栈帧有效的 host view。两条路径应产生相同业务语义；若它们需要完全不同的输入结构，应用显式选择 `submit()` 或 `submit_gpu()` 往往更清晰。

## 不会隐式回退的情况

`submit_auto()` 的决策只依据特征和调度器历史；若它选择 GPU 而 `cuda0` 未注册或不可用，提交会明确失败，不会偷偷改走 CPU。推荐流程是先完成 `register_gpu_executor_ex()`、检查状态，再允许 GPU 特征或 `prefer_gpu` 进入调度器；注册失败时由应用直接使用 `submit()` 处理 CPU 回退。

## 调整配置

```cpp
auto config = executor.get_scheduler_config();
config.data_size_threshold = 4 * 1024 * 1024;
config.compute_intensity_threshold = 2.5F;
config.enable_adaptive = true;
config.history_size = 200;
executor.update_scheduler_config(config);
```

阈值应来自真实基准而非直觉。自适应历史只有在应用记录了具有代表性的 CPU/GPU 性能后才有意义；变更硬件、驱动、数据形状或 kernel 后应重新收集数据。调度配置是策略，不替代后端可用性检查和任务级异常处理。

## 下一步阅读

需要资源、stream 或多设备控制时进入后续高级专题；完整字段和实现语义以[API 参考](/zh/reference/api)为准。
