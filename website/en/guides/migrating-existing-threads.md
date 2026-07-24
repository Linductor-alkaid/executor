---
title: Migrate Existing Thread Code
description: Gradually migrate std::thread, std::async, and a hand-written queue to Executor while preserving ownership and shutdown semantics.
---

# Migrate Existing Thread Code

Migration does not mean eliminating every `std::thread`. Move short, queueable work to shared execution resources and make results, failure, overload, and shutdown observable. Permanent loops, blocking device reads, and work with a strict period can still require a dedicated thread or real-time task.

For a permanent blocking loop that needs Facade ownership and join semantics, implement `IBlockingIoWorker` and follow [blocking I/O workers](/en/realtime-and-communication/blocking-io-workers). That library path owns lifecycle only; protocol and device behavior remain with the consumer.

This guide follows a frame-parsing service: it accepts `Frame`, parses in parallel, returns results to the caller, stops taking frames during exit, drains accepted work, and reports anything unfinished.

## Assign responsibility before changing code

Legacy thread functions often combine input acceptance, execution-resource creation, business work, and shutdown. Answer these independently first:

| Responsibility | Required question |
| --- | --- |
| Input ownership | Are arguments and captured objects still valid when the task executes? |
| Execution resource | Is it short work, a permanent loop, soft periodic maintenance, or strict periodic control? |
| Completion | Who owns the future, when retrieves it, and where handles exceptions? |
| Overload | When input exceeds consumption, should it queue, reject, overwrite, or degrade? |
| Lifecycle | Who stops producers, drains work, and finally shuts down Executor? |

Replacing APIs before these answers merely transfers the old races to a new abstraction.

## Start from detached threads

```cpp
void ParserService::accept(Frame frame) {
    std::thread([this, frame = std::move(frame)]() mutable {
        auto parsed = parse(frame);
        publish(parsed);
    }).detach();
}
```

This creates input-driven numbers of system threads, can dereference `this` after service destruction, loses a result channel for exceptions, and leaves shutdown unable to identify in-flight frames. First reject new input and define ownership of already accepted work.

## Let the service borrow Executor

The application owns the runtime; a service borrows it:

```cpp
class ParserService {
public:
    explicit ParserService(executor::Executor& executor) : executor_(executor) {}
private:
    executor::Executor& executor_;
};
```

Use `Executor::instance()` for ordinary process-wide sharing, or an independent instance for tests, plugins, or a subsystem that needs isolated drain/shutdown. In either case, exactly one owner calls `initialize_ex()` and `shutdown()`.

## Move result-bearing work first

```cpp
std::future<ParsedFrame> ParserService::accept(Frame frame) {
    return executor_.submit([frame = std::move(frame)]() mutable {
        return parse(frame);
    });
}

auto parsed = parser.accept(std::move(frame));
try {
    publish(parsed.get());
} catch (const std::exception& error) {
    report_parse_failure(error);
}
```

The task owns its input and does not capture `this`. Keep futures visible during migration: they expose formerly silent exceptions and completion assumptions. Omit a per-item future only for explicitly fire-and-forget work with a failure callback/status and a business correlation ID.

## Move from `std::async` and hand-written queues

For applications that used `std::async`, the surface change is usually direct:

```cpp
// Before
return std::async(std::launch::async, parse, std::move(frame));
// After
return executor_.submit(parse, std::move(frame));
```

The important change is lifecycle: application-owned Executor resources outlive a future destructor, so requests still consume futures and shutdown still stops producers before draining. Do not blindly replace code that depended on an unspecified/deferred `std::async` policy; decide whether it needed asynchronous work or lazy caller-thread evaluation.

For a `queue + mutex + condition_variable + workers` implementation, freeze the old entry point and record its capacity/rejection/exit behavior. Migrate one independent short-work class to `submit()`, compare active/queued state, inject exceptions/backlog/shutdown races, then remove old workers only after every producer moves. Do not use one giant Executor queue to mimic every data path: FIFO messages need a bounded channel; latest-only state needs a mailbox; computations needing results need tasks.

## Make dependencies explicit

Replace worker-side `future.get()` waits with handles:

```cpp
auto load = executor_.submit_with_handle(load_model);
auto plan = executor_.submit_after(load.handle, build_plan);

load.future.get();
auto result = plan.get();
```

The Facade now validates the relationship and propagates prerequisite failure. A handle belongs only to its originating Executor. Current dependent wrappers may still wait in the pool; submit prerequisites first, bound graph size, and pressure-test at the target minimum worker count. Use a specialized scheduler for large nonblocking DAGs.

## Keep permanent loops in the right place

Do not submit a never-ending blocking read loop to the shared pool. Use an owned, stoppable `std::jthread` for device I/O and submit only short post-read computation; use `submit_periodic()` for soft maintenance; use a dedicated real-time task for a jitter-budgeted loop; use `executor::comm` for sustained cross-thread data transfer.

## Establish a shutdown protocol

Expose draining at the business boundary:

```cpp
std::future<ParsedFrame> ParserService::accept(Frame frame) {
    if (!accepting_.load(std::memory_order_acquire)) {
        throw std::runtime_error("parser is draining");
    }
    return executor_.submit(parse, std::move(frame));
}

parser.stop_accepting();
auto drained = executor.wait_for_completion_ex(std::chrono::seconds{2});
if (!drained.completed) log_pending(drained.status.pending_tasks);
executor.shutdown(drained.completed);
```

A wait timeout does not cancel work, and `shutdown(false)` cannot create safe interruption points. I/O and long tasks still need their own timeout or cooperative stop mechanism. A shut-down Executor cannot be reinitialized; rebuild the isolated component/runtime to support restart.

## Migration acceptance matrix

| Scenario | Expected observation |
| --- | --- |
| Normal parse | Future returns a result; completion count increases |
| Parse throws | `future.get()` throws; failure status/callback observes it |
| Queue rejects | Future reports submission rejection; rejection count rises |
| Wait budget expires | `WaitResult.timed_out` with a pending snapshot |
| Submission after draining | Business entry rejects before Executor |
| Service object destroyed | No in-flight task captures its invalid reference |
| Process exit | Producers stop, work drains, then task-owned objects are destroyed |

Use low worker counts, small queues, and deliberately blocking work. A normal-path test cannot prove that migration removed races.

| Need | Default choice |
| --- | --- |
| Long-lived stoppable blocking-I/O owner | `std::jthread` |
| Few local parallel calculations | `std::async` or a synchronous algorithm |
| Short work across modules with unified capacity/diagnostics | Executor `submit()` |
| Bounded multistage work with clear completion relations | Executor task dependencies |
| Strict periodic loop | Executor real-time task |
| Sustained cross-thread data | `executor::comm` |

Continue with the [production readiness checklist](/en/guides/production-readiness), or review suspicious designs against [concurrency architecture antipatterns](/en/guides/concurrency-antipatterns).
