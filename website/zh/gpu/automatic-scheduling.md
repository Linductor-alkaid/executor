---
title: CPU/GPU 自动选择
description: 用 cpu_gpu_task 表达独立路径、配置回退，并理解 legacy submit_auto 的兼容边界。
---

# CPU/GPU 自动选择

## 学习目标

理解 `cpu_gpu_task()` + `submit_auto()` 如何依据任务特征选择 CPU 或 GPU，何时调整 `GpuScheduler::Config`，以及为什么“选择 GPU”不等于自动处理后端不可用。

如果你只有普通 CPU lambda，请返回[执行模型与路由边界](/zh/guides/execution-models-and-routing)：默认 `Auto` 不会隐式把它改投 GPU。

## 推荐路径：独立 CPU/GPU callable

新代码为 CPU 与 GPU 路径提供独立 callable，不再用 `nullptr` stream 在一个 callable 中猜测执行环境：

```cpp
auto future = executor.submit_auto(
    executor::cpu_gpu_task(
        [data] { run_cpu(*data); },
        [data](void* stream) { run_gpu(stream, *data); })
        .name("segment")
        .data_size(bytes)
        .compute_intensity(3.0F)
        .preferred_executor("cuda0")
        .fallback(executor::FallbackPolicy::AllowCpu));
future.get();
```

`AllowCpu` 在 GPU 未注册、未运行、错误或达到已知硬容量时允许 CPU 路径，并由 `RoutingDecision::fell_back` 解释；默认 `NoFallback` 则让 future 就绪为异常。`RequireRequestedBackend` 必须显式指定可提交的 GPU executor。实际提交仍可能与 stop 或容量竞争，调用方仍要处理 future 异常和 failure event。

## 默认选择规则

调度器按以下顺序决定：

1. `TaskCharacteristics::prefer_gpu` 为真时选择 GPU。
2. 启用自适应调度且相似任务的 CPU/GPU 历史各至少有两条时，选择预测耗时更短的一侧。
3. 否则，当数据大小达到 `data_size_threshold`（默认 1 MiB）且计算强度达到 `compute_intensity_threshold`（默认 2.0）时选择 GPU。
4. 其余情况选择 CPU。

`CpuGpuTask` 以 `data_size`、`compute_intensity` 和 `prefer_gpu()` 传递任务特征；调度器仍按显式 GPU 偏好、自适应历史和阈值选择候选。GPU 必须已注册、运行、无后端错误且未达到已知硬容量，才会进入候选。

## 兼容路径：legacy 四参数 overload

`0.3.x` 保留以下 overload 以兼容既有代码；它的 CPU 路径仍以空 stream 调用同一个 callable，GPU 未就绪时也不会隐式 CPU 回退：

```cpp
auto future = executor.submit_auto(characteristics, "cuda0",
    [data](void* stream) {
        if (stream == nullptr) {
            run_cpu(*data);
        } else {
            run_gpu(stream, *data);
        }
    }, gpu_task_config);
```

该模式仅用于渐进迁移。若两条路径需要不同输入或生命周期，新代码使用前一节的 `cpu_gpu_task()`；在下一个允许破坏性变更的主版本前，此 overload 不添加编译期弃用标记。

## 不会隐式回退的情况

新双路径 `submit_auto()` 会按 `FallbackPolicy` 处理不可用 GPU：仅 `AllowCpu` 会回退 CPU；`NoFallback` 和 legacy overload 都会明确失败，不会偷偷改走 CPU。推荐流程是先完成 `register_gpu_executor_ex()`、检查状态，再允许 GPU 特征或 `prefer_gpu` 进入调度器。

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
