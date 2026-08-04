---
title: Submit Functions and Data
description: Safely pass inputs, shared objects, and exclusive resources through submit_auto(lambda).
---

# Submit Functions and Data

## Goal

Capture the inputs required by finite work in a lambda, then pass it to `submit_auto(lambda)`. Understand value capture, move capture, and shared ownership, and keep every required object alive until the task finishes.

## Default: `submit_auto(lambda)`

Use a value-capturing lambda for the normal path. The runnable tutorial starts here, then shows multi-value and move captures:

<<< @/../examples/tutorial/11_task_inputs.cpp{34-50}

The closures respectively own `frame`, `offset`, and moved `payload`; their futures represent completion or exception. Mutating the submitter's later `frame` does not change the copy held by the task. `submit_auto(lambda)` safely selects the default asynchronous backend; it does not infer a GPU, lock-free, or real-time route from the callable. `get_last_routing_decision()` can explain the selected path but does not reserve it or replace the future.

## Submit a member function

A member function follows the same rule: capture a stable owner and inputs by value in a lambda. Prefer `std::shared_ptr` so the object survives until execution finishes:

<<< @/../examples/tutorial/11_task_inputs.cpp{20-29,52-55}

Do not capture a raw `this` pointer or address of a local object: a worker dereferences a dangling object if its owner is destroyed before execution begins. Even a service-owned object needs shutdown order that stops new submissions, waits for its tasks, then destroys the owner.

## Organize inputs with a lambda

Use a lambda to combine inputs at the submission point, do a small amount of preprocessing, or select an overload. Capture by value by default; the `score` and `adjusted` submissions above respectively show one and multiple captured values. `[frame, offset]` copies both values into the closure, so the task does not depend on the submitting function's stack frame. Avoid casually using `[&]`: an asynchronous task commonly runs after the current scope ends, and reference captures can dangle or race with later mutations.

For large inputs, first establish that copying is actually a bottleneck. Typical alternatives are moving an exclusively owned resource or sharing an immutable object:

```cpp
auto model = std::make_shared<const Model>(load_model());
auto result = executor.submit_auto([model, frame] {
    return infer(*model, frame);
});
```

## Move exclusive ownership into a task

To transfer a `std::unique_ptr`, buffer handle, or other exclusive resource, use a move capture:

<<< @/../examples/tutorial/11_task_inputs.cpp{47-50}

After submission, the original `payload` is empty and the closure exclusively owns the resource. This is easier to reason about than a raw pointer. Do not use the moved-from object again as a caller input.

Move capture makes the closure the explicit resource owner. If a business function needs a `std::unique_ptr` or `T&&` by value, decide where to `std::move` inside the lambda instead of lending the resource to asynchronous work.

## Share and modify state

When a task must modify cross-thread state, capture an owner with an explicit synchronization contract instead of borrowing a reference from the submitter's stack:

<<< @/../examples/tutorial/11_task_inputs.cpp{57-60}

This example uses `shared_ptr<atomic<int>>` to express both shared lifetime and atomic access. `shared_ptr` extends a lifetime only; it does not make an arbitrary object thread-safe. Non-atomic state still needs its own mutex, message-passing, or synchronization protocol.

Do not use `[&]` or a raw reference for asynchronous input: neither extends lifetime nor supplies thread safety. `future.get()` waits for completion but cannot repair a race that occurred while the task ran.

## Choose ownership deliberately

| Need | Recommended form | What the task depends on | Main risk |
| --- | --- | --- | --- |
| Small read-only input | `submit_auto([value] { ... })` | Its own copy | Copying cost |
| Transfer an exclusive resource | `[value = std::move(value)]` | Exclusive ownership | Submitter cannot reuse the moved value |
| Share a large immutable object | Capture `shared_ptr<const T>` | Shared lifetime | Reference-count and residency cost |
| Invoke a member function | `[owner, value] { return owner->method(value); }` | Object survives completion | A raw object pointer can dangle |
| Share mutable state | Capture a `shared_ptr` with a synchronization contract | Shared owner and synchronization rules | Data race, shutdown order |

## API-specific input shapes

When the business explicitly requires priority, delay, periodic scheduling, batching, or dependencies, enter the matching explicit API. On every future path, bind stable inputs to a lambda by default. Longer delays and dependency chains make borrowed inputs more dangerous because execution begins later.

Periodic tasks and batches have different shapes: `submit_periodic()` takes repeatable `void()` work, and batches take independently bound `void()` callables. Real-time callbacks and queue entries also use pre-bound `void()` work. CPU/GPU routing requires the separate CPU and GPU callables of `cpu_gpu_task()`; it is not an ordinary argument-pack variant. Read the corresponding [real-time](/en/realtime-and-communication/realtime-control), [GPU](/en/gpu/register-and-submit), and [communication](/en/realtime-and-communication/channels) contracts before using those paths.

## Build and run

```bash
cmake --build build --target tutorial_11_task_inputs
./build/examples/tutorial/tutorial_11_task_inputs
```

Expected output:

```text
score=42, plan=local-frame-7, adjusted=26, owned=9
processed=1
```

Next, read [return values and errors](/en/quick-start/return-values-and-errors) to see how these inputs return control through a future after success or failure; use [Choose a Submission API](/en/guides/choosing-submit-api) when timing or result-model requirements change.
