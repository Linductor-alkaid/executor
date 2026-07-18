---
title: Failure Observability
description: Combine futures, failure callbacks, status counts, and recent events into a failure-observation path for a long-running service.
---

# Failure Observability

## Goal

Distinguish a result, failure trend, and recent diagnostic event. A task exception must not disappear silently even when a caller does not immediately consume its future.

## Four observation paths

| Question | Default entry | Scope |
| --- | --- | --- |
| Did this call succeed, and what is its result? | `future.get()` | One result-bearing task; exception rethrows here |
| What failure just happened in the service? | `set_failure_callback()` | Immediately bridge to logs, alerts, or telemetry |
| How many failures of this type accumulated? | `get_failure_status()` | Health checks, dashboards, threshold alerts |
| What is the context of recent failures? | `get_recent_failures()` | Diagnosis, support bundle, bounded history |

## Recommended path

Set a callback after initialization, then retain `get()` where a result is needed:

<<< @/../examples/tutorial/06_observability.cpp{1-29}

```bash
./build/examples/tutorial/tutorial_06_observability
```

```text
failures=1, callback=1, recent=1
```

`future.get()` remains the result and exception boundary for one task. Callback, counts, and recent events are additional service-level observation; they do not replace it.

## Recent-event retention

- `get_recent_failures(0)` returns the entire current buffer; a positive argument returns that many newest events.
- `set_recent_failure_capacity(n)` configures ring-buffer capacity. At `0`, events are not retained, but cumulative counts and callback still work.
- `clear_recent_failures()` clears only diagnostic history; it does **not** reset cumulative `get_failure_status()` counts.

Choose capacity from memory budget and incident investigation window. Do not keep unbounded process-local history.

## Callback boundary

The failure callback runs on Executor's failure-recording path. Keep it short and nonblocking, and own any external I/O policy. An exception thrown by the callback is isolated and does not terminate a worker/background thread. For complex handling, enqueue a small event into application logging or alert infrastructure.

## Failures are not interchangeable

`TaskException`, `SubmitRejected`, `WaitTimeout`, real-time drops, GPU failure, and safe tuning fallback can all enter `ExecutorFailureStatus`, but have different meanings. Task exception needs a business-result decision; wait timeout means unfinished work; tuning fallback may still run safely. Communication events remain in local `executor::comm` callbacks/statistics by default.

Next: [monitoring and sampling](/en/reliability/monitoring) for throughput, success/failure, and execution-time trends; [bounded waiting and status](/en/tutorial/waiting-and-status) for wait timeout decisions.
