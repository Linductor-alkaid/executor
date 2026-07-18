---
title: Batch Sensor Frames
description: Choose submit_batch, submit_batch_no_future, or submit_batch_priority for independent work.
---

# Batch Sensor Frames

## Goal

Submit a group of similar independent tasks at once, choosing a future-owning or fire-and-forget path based on whether each completion must be observed.

<<< @/../examples/tutorial/04_batch.cpp{1-29}

```bash
./build/examples/tutorial/tutorial_04_batch
```

```text
batch processed=6, completed=yes
```

| Need | API | Failure observation |
| --- | --- | --- |
| Each item needs completion or exception | `submit_batch()` | Call `get()` on every future. |
| Per-item results are unnecessary | `submit_batch_no_future()` | Use failure callback/status plus bounded waiting or shutdown semantics. |
| The entire group is urgent | `submit_batch_priority(priority, tasks)` | Inspect every future as with `submit_batch()`. |

## Bind every task's inputs

Batch APIs take independently callable, already-bound `void()` tasks, usually a `std::vector<std::function<void()>>`; they do not provide `submit_batch(fn, args...)`.

```cpp
std::vector<std::function<void()>> tasks;
for (SensorFrame frame : frames) {
    tasks.push_back([frame, processor] { processor->process(frame); });
}
auto futures = executor.submit_batch(tasks);
```

Each closure owns a `frame` copy and shares `processor` lifetime. Never capture a loop variable as `[&frame]`: a later iteration or departed scope can leave tasks with the same object or a dangling reference. For large data, use a buffer handle with a defined return protocol or `shared_ptr<const FrameData>`, not an unproven view.

The list and its callables must currently be copyable. Use a shared owner for move-only resources, or submit items individually with move-capture lambdas. Batch futures report completion of each `void()` callable; they do not collect business return values.

## Boundaries to preserve

Batch tasks must be independent, produced together, and have the same scheduling meaning. Future positions match input positions, but completion order is unspecified. A failed task does not roll back completed siblings; a batch is not a business transaction.

Test an exception in the middle of a batch, an empty callable, submission after shutdown, failure in a no-future task, and work exceeding an exit budget. Consume all futures and keep captured objects alive until all work completes.

For different per-item results, submit individually or provide an indexed result container. For dependencies, use handles and `when_all()`; for continuously arriving work, use upstream rate limiting, chunking, and capacity budgets rather than unbounded queue growth.

Next: [load, sense, then plan](/en/tutorial/dependencies).
