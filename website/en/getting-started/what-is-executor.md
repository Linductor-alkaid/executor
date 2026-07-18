---
title: What is Executor?
description: Decide what Executor solves, what it does not guarantee, and where to begin.
---

# What is Executor?

Executor is a C++20 task execution and thread-management library. Its public Facade brings together where work runs, how callers receive results, and how a runtime shuts down. Dedicated paths remain available for periodic control loops, cross-thread communication, and optional GPU work.

Start with the workload, not the thread-pool implementation:

1. Is this one background calculation, soft periodic maintenance, or periodic control with a jitter budget?
2. Does the caller need a result, per-item completion, or only service-level failure reporting?
3. Does data travel as task input, or continuously between long-running threads?

## When it fits

- You have short background work in several components and want shared execution resources.
- Callers need results and task exceptions through `std::future`.
- You need priority, delay, soft periodic scheduling, batches, or dependencies without maintaining a scheduler.
- A service needs observable rejected submissions, exceptions, wait timeouts, or real-time queue drops.

## What it is not

- It is not a coroutine runtime or a dataflow framework.
- `submit_priority()` changes ordinary queue order; it does not provide deadlines or preempt work already running.
- `submit_periodic()` is soft periodic work on the ordinary pool, not a hard-real-time control loop.
- A timeout does not safely terminate arbitrary C++ code. Tasks must remain bounded and own their own cancellation or deadline logic.

If your program only has one or two long-lived threads with clear ownership, `std::jthread` may be simpler. Add Executor when it removes operational responsibility rather than merely hiding `std::thread` creation.

## The usual first path

```cpp
auto& executor = executor::Executor::instance();
auto result = executor.submit([] { return parse_frame(); });

try {
    consume(result.get());
} catch (const std::exception& error) {
    report(error);
}
```

`submit()` queues work on the default asynchronous executor. A worker stores either the return value or exception in the future, and the caller observes it at `get()`. Submitting work successfully and completing work successfully are different events.

Continue with [build and install](/en/quick-start/build) when you are ready to run the first example.
