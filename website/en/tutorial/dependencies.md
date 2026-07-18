---
title: Load, Sense, Then Plan
description: Build observable task dependencies with TaskHandle, submit_after, and when_all.
---

# Load, Sense, Then Plan

## Goal

Use `submit_with_handle()` to retain both a task result and a dependency handle, then express “load and sense before planning” with `submit_after()` and `when_all()`.

<<< @/../examples/tutorial/05_dependencies.cpp{1-25}

```bash
./build/examples/tutorial/tutorial_05_dependencies
```

```text
plan score=42
```

## Build the relationship

- One prerequisite: `submit_after(load.handle, plan)`.
- Several prerequisites: `submit_after({load.handle, sense.handle}, plan)`, or use `when_all()` to make an explicit join handle.
- If later work depends on the plan, use `submit_after_with_handle()` to retain a new handle as well as a future.
- `when_all(handles)` represents only a join where all work completes; joins may be nested.

## Inputs are saved at dependent submission

`submit_with_handle(fn, args...)` and `submit_after(handle, fn, args...)` use the same callable and argument model as `submit()`. A dependent's inputs are saved when that dependent is submitted, not when its prerequisites later succeed.

```cpp
PlanConfig config = load_plan_config();
auto plan = executor.submit_after(prerequisites, run_planner, config);
```

Later changes to the caller's `config` do not change the stored copy. A `TaskHandle` carries completion state, not a return value. Keep the prerequisite future or write the result to an explicitly owned shared object, then read it only after the dependency is satisfied.

References must live from dependent submission until eventual completion, including the time spent waiting for prerequisites. Longer chains should save immutable configuration by value or use `shared_ptr`, not `[&]` captures of a caller stack frame.

## Capacity boundary

Handles make completion and failure propagation explicit, but the current implementation is not a fully non-blocking DAG scheduler. Dependent wrappers enter the ordinary pool and can wait there for prerequisite state. Low worker counts, long chains, or submitting many dependents before their prerequisites can occupy workers.

Submit prerequisites first, avoid unbounded blocking inside a dependent, limit in-flight chains, and pressure-test with the production minimum worker count. Use a dedicated graph scheduler for large dynamic DAGs instead of increasing threads and queue capacity.

## Failure and exit

If a prerequisite fails, its future rethrows and the dependent does not run; its future reports failure too. Default-constructed handles and handles from another `Executor` instance are observable rejection. Do not persist or mix handles across instance or request lifetimes.

Test prerequisite failure, invalid/cross-instance handles, multiple long chains at low worker count, and shutdown before a graph completes. Stop creating graphs, retain their futures, then wait within a budget or record unfinished business IDs. In-memory task graphs do not survive process restart.

Use `submit_after_with_handle()` for further dependent work. For a continuous frame stream use a channel, not a task graph; for monotonic lifecycle stages use `PhaseGate`.

Next: [bounded waiting and status](/en/tutorial/waiting-and-status).
