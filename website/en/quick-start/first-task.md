---
title: Your First Task
description: Run work with submit and future.get, retrieve its result, and observe its exception.
---

# Your First Task

## Goal

Use `Executor::instance()` to submit work returning `42`, then retrieve the result and an exception through `future.get()`.

## Recommended approach

Use `submit()`. It returns a `std::future`: `get()` returns the value after success and rethrows a task exception on the calling thread. The first submission lazily initializes Executor with its default configuration.

<<< @/../examples/tutorial/01_first_task.cpp{4,7-16,18-24,26}

Full source: [`examples/tutorial/01_first_task.cpp`](https://github.com/Linductor-alkaid/executor/blob/master/examples/tutorial/01_first_task.cpp).

```bash
./build/examples/tutorial/tutorial_01_first_task
```

## Expected output

```text
answer=42
task failed: expected tutorial failure
```

## Why this matters

- `future.get()` is both a result operation and an exception-observation boundary.
- Default configuration is appropriate for this minimal example.
- Configure thread counts, queue capacity, or monitoring with `initialize_ex()` before the first submission.
- The example ends with `shutdown()`. Applications must choose their shutdown semantics at a real component boundary.

## Common mistakes

- Submitting without retaining the future when you still need a return value or task exception.
- Treating `submit_periodic()` as a real-time task; it is soft periodic scheduling on the ordinary pool.
- Configuring the singleton after the first submission has already initialized it.

Read [submit functions and data](/en/quick-start/task-inputs-and-ownership) next. It explains how to pass free functions, member functions, parameters, and business objects safely; [return values and errors](/en/quick-start/return-values-and-errors) then covers failure observation. Read [initialization and shutdown](/en/quick-start/lifecycle) when you need explicit lifecycle control.
