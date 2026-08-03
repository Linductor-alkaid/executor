---
title: Execution Models and Routing Boundaries
description: Learn when default automatic routing, future completion, bounded admission, and worker lifecycle must be treated separately.
---

# Execution Models and Routing Boundaries

Executor's unified Facade lets ordinary developers begin with `submit_auto(lambda)`, but it does not pretend that every backend is the same kind of thread pool. Before taking an expert path, distinguish the result model your caller actually receives.

## Three results, not one success

| Model | Entry | What the caller confirms | What it does not confirm |
| --- | --- | --- | --- |
| Completion | `submit_auto()`, `submit()`, CPU/GPU dual-path task | Work completed, or its future contains an exception | A real-time cycle, worker, or other backend has ended |
| Admission | `dispatch_auto()` | The named bounded queue accepted this task | The task ran, no item dropped, or business effect occurred |
| Lifecycle | `start_worker()` | A worker registered/started, or startup failed | Protocol handshake, device availability, or first input |

This distinction prevents invalid waiting: `wait_for_completion()` waits only for default asynchronous future work, not real-time cycles, lock-free queues, long-lived workers, or GPU activity.

## What default automatic routing does

For ordinary work:

```cpp
auto future = executor.submit_auto([] { return transform(); });
auto value = future.get();
```

`Auto` chooses only the default asynchronous backend. It removes the need to find or register a separate executor before a first task, while each choice remains inspectable:

```cpp
const auto decision = executor.get_last_routing_decision();
```

`RoutingDecision` explains what was selected and why from declared intent and a capability snapshot. It is neither a reservation nor a replacement for a future, `DispatchResult`, or status counter.

## When to route GPU work

GPU is not implicit acceleration for an ordinary lambda. Use it only when the business operation has independent CPU and GPU implementations:

```cpp
auto future = executor.submit_auto(
    cpu_gpu_task([input] { run_cpu(input); },
                 [input](void* stream) { run_gpu(input, stream); })
        .preferred_executor("cuda0")
        .fallback(FallbackPolicy::AllowCpu));
```

`AllowCpu` records a CPU fallback through `RoutingDecision::fell_back`; `NoFallback` and `RequireRequestedBackend` reject through the future and failure events. Registration, devices, and stream details belong in the [GPU topic](/en/gpu/).

## When to route bounded queues

Lock-free and real-time paths are explicit opt-ins. Use `LowLatency` or `RealtimeQueue` only when a single consumer, bounded backpressure, or periodic consumption is already the business contract, and name a running backend:

```cpp
TaskOptions options;
options.intent = ExecutionIntent::LowLatency;
options.preferred_executor = "telemetry";
auto admission = executor.dispatch_auto(options, [] { publish(); });
```

`accepted == true` means queue admission only. A stopped backend, full queue, exhausted object pool, or shutdown race can reject; none silently falls back to the default thread pool.

## When to create a long-lived worker

Blocking I/O is a wakeable, stoppable loop rather than one callable:

```cpp
BlockingWorkerSpec spec{"serial-rx", config, std::move(worker)};
auto handle = executor.start_worker(std::move(spec));
if (!handle.started()) report(handle.start_result().message);
```

`WorkerHandle::request_stop()` requests stop and wakeup; `WorkerHandle::stop()` also joins. The worker must implement `run(stop_token)` and non-throwing `wakeup()`; see [Blocking I/O workers](/en/realtime-and-communication/blocking-io-workers).

## Capability snapshots are not reservations

`get_executor_capabilities()` is useful for control planes, monitoring, and submission diagnostics. It is advisory: a backend can stop or fill between the snapshot and submission. Use it to explain and preflight, never as a promise that a task will complete.

## Next steps

Return to [Choose a Submission API](/en/guides/choosing-submit-api) for scenario selection. Enter [real-time control](/en/realtime-and-communication/realtime-control), [GPU](/en/gpu/), [Blocking I/O workers](/en/realtime-and-communication/blocking-io-workers), or [Advanced](/en/advanced/) only when the corresponding constraint is explicit.
