---
title: Blocking I/O Workers
description: Own, wake, and join a long-lived blocking worker through the Executor Facade.
---

# Blocking I/O Workers

## Use this path for a long-lived wait

Use `BlockingIoExecutor` when a component owns one long-lived loop that can block and must still stop cleanly. It is not a task queue, a real-time control loop, a protocol adapter, or a device-integration framework.

Use the thread pool for finite queueable work. Use a dedicated real-time thread for fixed-period control. The worker's protocol, inputs, outputs, retry policy, and safety behavior remain the library consumer's responsibility.

## The worker contract

Implement `IBlockingIoWorker::run(stop_token)` and `wakeup()`.

- `run()` may wait, but must return after a stop request becomes observable.
- `wakeup()` must release the current wait, may be called repeatedly, and must not throw.
- A stop token alone does not interrupt an arbitrary external wait. If the wait primitive cannot be awakened directly, use a bounded timeout and check the token after every return.

The Facade owns the worker after registration. It starts the dedicated thread, calls `wakeup()` during stopping, and joins before releasing the worker.

## Runnable mock worker

The tutorial uses a condition variable only to demonstrate the lifecycle without a protocol or hardware dependency:

<<< @/../examples/tutorial/12_blocking_io_worker.cpp{1-78}

```bash
./build/examples/tutorial/tutorial_12_blocking_io_worker
```

```text
blocking worker started=yes, stopped=yes, wakeups=1
```

## Lifecycle and status

1. Configure a nonempty `BlockingIoConfig::thread_name` and register a `std::unique_ptr<IBlockingIoWorker>`.
2. Prefer `register_blocking_io_worker_ex()` and `start_blocking_io_worker_ex()` when callers need `ExecutorResult` diagnostics.
3. Observe `get_blocking_io_worker_status(name)`. `ready` describes executor-thread setup only; it does not mean a protocol, device, or first input is ready.
4. Call `stop_blocking_io_worker(name)` to request stop, wake the worker, and join it. Repeated calls are safe.

`Executor::shutdown()` applies the same stop/wake/join rule to every registered I/O worker, including `shutdown(false)`. Do not detach a worker or retain references to it after shutdown.

## What remains outside Executor

This library deliberately does not decide message ownership, queue policy, data freshness, reconnect behavior, device safety actions, or deployment tuning. Define and test those concerns in the application that implements the worker.

For complete signatures and status fields, see the [API reference](https://github.com/Linductor-alkaid/executor/blob/master/docs/API.md#45-blocking-io-worker-api). Next: return to [real-time control](/en/realtime-and-communication/realtime-control) when the work instead has a fixed-period budget.
