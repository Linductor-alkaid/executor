---
title: Monitoring and Sampling
description: Use task statistics and sampling rate to observe long-term Executor trends at acceptable overhead.
---

# Monitoring and Sampling

## Goal

Query success, failure, timeout, and execution-time statistics by task type, choosing monitoring enablement and sampling for high-frequency paths.

## Recommended path

Use `ExecutorConfig::enable_monitoring` for the initial setting; switch it at runtime with `enable_monitoring()` and query statistics:

```cpp
executor.enable_monitoring(true);
executor.set_monitoring_sampling_rate(0.1);  // Sample about 10% of tasks.

const auto default_stats = executor.get_task_statistics("default");
const auto all_stats = executor.get_all_task_statistics();
```

`TaskStatistics` contains total, success, failure, timeout, total execution time, and maximum/minimum execution time. An unknown task type returns zero values. Work completed while monitoring is disabled does not increment new counters.

Full example: [`examples/monitoring_sampling_example.cpp`](https://github.com/Linductor-alkaid/executor/blob/master/examples/monitoring_sampling_example.cpp).

| Goal | Setting | Trade-off |
| --- | --- | --- |
| Debugging, low-rate control path, exact counts | `1.0` | Records every task; most complete observation |
| Trend/regression monitoring for high-throughput service | `0.01`–`0.1` | Lower monitoring cost; statistics are a sample, not per-task accounting |
| Extremely sensitive path or no statistics needed yet | `enable_monitoring(false)` | No new statistics; loses task-trend visibility |

`set_monitoring_sampling_rate(rate)` accepts `0.0` through `1.0`. Sampling supports trend comparisons, anomaly detection, and capacity planning. Retain a future or business state for the definite result of one task.

## Do not confuse three data classes

- `TaskStatistics`: execution aggregate by task type, for trends.
- `get_async_executor_status()`: current queue, active, completed, and failed snapshot, for congestion and lifecycle diagnosis.
- `ExecutorFailureStatus`: cumulative failure kinds recorded by the Facade, for alerts and root-cause routing.

Start from a question: use status for “is work backing up?”, failure status for “are failures increasing?”, and task statistics for “is long-term execution time getting worse?”.

Continue with [choose a submission API](/en/guides/choosing-submit-api) for task semantics, or [choose a communication component](/en/guides/choosing-communication) for cross-thread data semantics.
