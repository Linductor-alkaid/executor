---
title: Versions and Migration
description: Entry points for the development snapshot, releases, and API migration.
---

# Versions and Migration

## Current scope

The latest release record is `v0.3.1`. This site uses that stable version as its baseline while following later `master` development; capabilities without a stable tag are not version promises. This first English edition does not maintain historical versioned sites.

| What to check | Source of truth |
| --- | --- |
| Released versions and breaking changes | [CHANGELOG.md](https://github.com/Linductor-alkaid/executor/blob/master/CHANGELOG.md) |
| Recommended migrations from older APIs | [MIGRATION.md](https://github.com/Linductor-alkaid/executor/blob/master/docs/MIGRATION.md) |
| Build options, compilers, and backends | [BUILD.md](https://github.com/Linductor-alkaid/executor/blob/master/docs/BUILD.md) |
| Complete current signatures | [API.md](https://github.com/Linductor-alkaid/executor/blob/master/docs/API.md) |

## Moving from `bool` to `_ex`

The legacy entry points remain compatible when a caller only needs success or failure. New code that must log, alert, or fall back safely should prefer the diagnostic `_ex` variants and inspect `ExecutorResult::error_code` and `message`.

| Migration | Use it when |
| --- | --- |
| `initialize(config)` → `initialize_ex(config)` | Configuration, repeated initialization, or post-shutdown failures need distinct causes. |
| `register_realtime_task(...)` → `register_realtime_task_ex(...)` | You need to distinguish invalid configuration, duplicate names, or platform startup failures. |
| `register_gpu_executor(...)` → `register_gpu_executor_ex(...)` | You need to distinguish invalid configuration from `BackendUnavailable`. |
| `wait_for_completion()` → `wait_for_completion_for()` / `_ex()` | Waiting must be bounded or timeout status must be recorded. |
| `IRealtimeExecutor::push_task()` → `Executor::try_push_realtime_task()` | Rejection, backpressure, and failure events must be observable. |

`_ex` is not a second business API that is always superior. Its value is connecting a failure reason to logs, alerts, or a fallback path.

## 0.3.1: from backend-first to intent-first

New code begins with `submit_auto(lambda)`, then enters a specialist path only when the business explicitly requires independent CPU/GPU implementations, bounded admission, or a long-lived worker lifecycle:

| Existing style or requirement | 0.3.1 recommended entry | Boundary that remains unchanged |
| --- | --- | --- |
| Ordinary `submit(lambda)` | Gradually adopt `submit_auto(lambda)` | Both return futures; `submit()` remains the explicit default-pool entry. |
| One callable branches on a null CPU/GPU stream | `cpu_gpu_task(cpu, gpu)` plus `submit_auto()` | The legacy four-argument overload remains available in `0.3.x` without implicit fallback. |
| Direct lock-free `push_task()` | Register, start, then use `dispatch_auto(LowLatency)` | `accepted` means admission only; single-consumer and backpressure semantics remain. |
| Direct real-time `push_task()` | Use `dispatch_auto(RealtimeQueue)` after start | `accepted` does not mean a later cycle completed and never falls back to the pool. |
| Register and start an I/O worker separately | `start_worker(BlockingWorkerSpec)` | `WorkerHandle` retains wakeup, stop token, startup timeout, and exit reason. |

Automatic routing does not infer callable real-time safety, thread safety, GPU-memory ownership, or I/O interruptibility. `get_executor_capabilities()` is only an advisory snapshot; each actual submission must still handle stop races and backpressure.

## Upgrade checklist

1. Read the target version's CHANGELOG and verify that each used capability exists in that tag.
2. Reconfigure and build with the target compiler, operating system, and any GPU or real-time permissions.
3. Keep observation paths for futures, return values, and status counters; use `_ex` at setup boundaries that need diagnosis.
4. Recheck real-time affinity, memory locking, timer slack, GPU backend, driver, and device status.
5. Run tests and tutorial smoke tests, then retest timeout, backpressure, and performance behavior under target load.

Chinese and English guides share the published information architecture. Check [translation status](/translation-status) whenever a new public page or language counterpart is added.
