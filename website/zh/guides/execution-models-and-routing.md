---
title: 执行模型与路由边界
description: 理解默认自动路由、future 完成、有界接收和 worker 生命周期何时应当分开处理。
---

# 执行模型与路由边界

`Executor` 的统一 Facade 让普通开发者可以从 `submit_auto(lambda)` 开始，但它不会把所有后端伪装成同一种线程池。进入专家路径之前，先分清调用方实际得到的结果模型。

## 三种结果，不是一种成功

| 模型 | 入口 | 调用方确认的事实 | 没有确认的事实 |
| --- | --- | --- | --- |
| completion | `submit_auto()`、`submit()`、CPU/GPU 双路径任务 | 工作完成，或 future 中有异常 | 实时周期、worker 或其他后端已结束 |
| admission | `dispatch_auto()` | 指定有界队列接受了本次任务 | 任务已执行、没有 drop 或业务已生效 |
| lifecycle | `start_worker()` | worker 已注册/启动，或得到启动失败 | 协议已握手、设备可用或首条数据已到达 |

这一区分保护调用方不做错误等待：`wait_for_completion()` 只等待默认异步的 future 型工作；它不等待实时周期、无锁队列或长期 worker。

## 默认自动路由做什么

默认普通任务：

```cpp
auto future = executor.submit_auto([] { return transform(); });
auto value = future.get();
```

`Auto` 只选择默认异步后端。它的价值是新手不必先注册或查找独立执行器，同时每次决定仍可查询：

```cpp
const auto decision = executor.get_last_routing_decision();
```

`RoutingDecision` 回答“根据声明的意图和能力快照选择了什么、为什么”；它不是提交 reservation，也不取代 `future`、`DispatchResult` 或状态计数。

## 何时路由 GPU

GPU 不是普通 lambda 的隐式加速器。只有业务具有不同的 CPU 和 GPU 实现时，才构造双路径任务：

```cpp
auto future = executor.submit_auto(
    cpu_gpu_task([input] { run_cpu(input); },
                 [input](void* stream) { run_gpu(input, stream); })
        .preferred_executor("cuda0")
        .fallback(FallbackPolicy::AllowCpu));
```

GPU 运行条件和队列容量仍会在实际投递时变化。`AllowCpu` 会以 `RoutingDecision::fell_back` 解释 CPU 回退；`NoFallback` 或 `RequireRequestedBackend` 的拒绝则通过 future 异常和 failure event 观察。后端注册、设备和 stream 细节进入[GPU 专题](/zh/gpu/)。

## 何时路由有界队列

无锁和实时路径是 opt-in。只有调用方已经知道单消费者、有界背压或周期消费正是业务语义时，才能使用 `LowLatency` 或 `RealtimeQueue`，并且必须填写运行中后端的 `preferred_executor`。

```cpp
TaskOptions options;
options.intent = ExecutionIntent::LowLatency;
options.preferred_executor = "telemetry";
auto admission = executor.dispatch_auto(options, [] { publish(); });
```

`accepted == true` 只表示队列接收；即使接收后，任务也可能在执行时抛异常，或在实时周期中晚于你期望的时间被处理。未启动、队列满、对象池耗尽和关闭竞争都应通过 `DispatchResult`、failure event 和后端状态计数处理，绝不静默投递到默认线程池。

## 何时创建长期 worker

Blocking I/O 的工作单位不是一次 callable，而是可被唤醒和停止的长期循环：

```cpp
BlockingWorkerSpec spec{"serial-rx", config, std::move(worker)};
auto handle = executor.start_worker(std::move(spec));
if (!handle.started()) {
    report(handle.start_result().message);
}
```

`WorkerHandle::request_stop()` 只请求 stop 并唤醒，`WorkerHandle::stop()` 再 join。worker 自己必须实现 `run(stop_token)` 和不抛异常的 `wakeup()`；详细约束见[Blocking I/O worker](/zh/realtime-and-communication/blocking-io-workers)。

## 能力快照与实际投递

`get_executor_capabilities()` 可用于控制面、监控 UI 或投递前诊断：它列出后端类型、名称、运行状态、协议支持、pending work 和 capacity hint。它不是锁定后端的预约；快照与实际投递之间仍可能发生 stop 或容量变化。

把它用于解释和预检，而不是把“快照显示可用”当作“任务必然完成”的承诺。

## 下一步阅读

现在可以回到[如何选择提交接口](/zh/guides/choosing-submit-api)做场景选型；需要专家控制时，再进入[实时控制](/zh/realtime-and-communication/realtime-control)、[GPU 与降级](/zh/gpu/)、[Blocking I/O worker](/zh/realtime-and-communication/blocking-io-workers)或[高级与原理](/zh/advanced/)。
