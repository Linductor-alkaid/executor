---
title: Versions and Migration
description: Entry points for the development snapshot, releases, and API migration.
---

# Versions and Migration

## Current scope

The CMake project and latest release baseline are `v0.2.3`. This site describes the `master` development snapshot, including planned `0.3.0` communication and task-graph capabilities. Unless a stable tag has been published, do not treat those capabilities as available in the `v0.2.3` release. This first English edition does not maintain historical versioned sites.

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

## Upgrade checklist

1. Read the target version's CHANGELOG and verify that each used capability exists in that tag.
2. Reconfigure and build with the target compiler, operating system, and any GPU or real-time permissions.
3. Keep observation paths for futures, return values, and status counters; use `_ex` at setup boundaries that need diagnosis.
4. Recheck real-time affinity, memory locking, timer slack, GPU backend, driver, and device status.
5. Run tests and tutorial smoke tests, then retest timeout, backpressure, and performance behavior under target load.

The Chinese guide contains the complete currently published topic set. Each new Chinese capability is listed in the [translation status](/translation-status) until its English counterpart is available.
