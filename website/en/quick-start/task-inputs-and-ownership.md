---
title: Submit Functions and Data
description: Safely submit free functions, member functions, lambdas, arguments, and resources to Executor.
---

# Submit Functions and Data

## Goal

Submit functions already implemented by your project with `submit()`. Understand when task inputs are copied, moved, or referenced, and keep every required object alive until the task has finished.

## What `submit()` accepts

`submit()` has this shape:

```cpp
auto future = executor.submit(callable, arguments...);
```

The callable may be a free function, lambda, function object, or member-function pointer. Executor derives the matching `std::future<T>` from its return type. Arguments do not need to be wrapped in a zero-argument function first.

Pass an existing function and its arguments directly:

<<< @/../examples/tutorial/11_task_inputs.cpp{12-19,39-43}

`frame` and `2` are stored in the asynchronous task. A worker later performs the equivalent of `score_frame(frame, 2)`. Mutating the submitter's later copy of `frame` does not change the value saved for the task.

## Submit a member function

A member function also needs an object. Prefer a `std::shared_ptr` owned by the task so the object survives until execution finishes:

<<< @/../examples/tutorial/11_task_inputs.cpp{21-31,45-46}

Do not pass a raw `this` pointer or address of a local object without proving the lifetime. `executor.submit(&Planner::make_plan, this, frame)` may compile, but a worker dereferences a dangling object if its owner is destroyed before execution begins. Even a service-owned object needs shutdown order that stops new submissions, waits for its tasks, then destroys the owner.

## Organize inputs with a lambda

Use a lambda to combine inputs at the submission point, do a small amount of preprocessing, or select an overload. Capture by value by default:

<<< @/../examples/tutorial/11_task_inputs.cpp{48-51}

`[frame, offset]` copies both values into the closure, so the task does not depend on the submitting function's stack frame. Avoid casually using `[&]`: an asynchronous task commonly runs after the current scope ends, and reference captures can dangle or race with later mutations.

For large inputs, first establish that copying is actually a bottleneck. Typical alternatives are moving an exclusively owned resource or sharing an immutable object:

```cpp
auto model = std::make_shared<const Model>(load_model());
auto result = executor.submit([model, frame] {
    return infer(*model, frame);
});
```

## Move exclusive ownership into a task

To transfer a `std::unique_ptr`, buffer handle, or other exclusive resource, use a move capture:

<<< @/../examples/tutorial/11_task_inputs.cpp{53-56}

After submission, the original `payload` is empty and the closure exclusively owns the resource. This is easier to reason about than a raw pointer. Do not use the moved-from object again as a caller input.

The current implementation stores direct `submit(fn, args...)` arguments through `std::bind`; bound arguments normally participate in invocation as stored lvalues. If a function needs to take a `std::unique_ptr` by value or requires `T&&`, do not expect `submit(fn, std::move(value))` to provide another rvalue later. Use a move-capture lambda and decide where to move from inside the closure.

## When a reference is appropriate

Ordinary arguments are stored as decayed values. If a task must operate on the original object, opt in explicitly with `std::ref()` or `std::cref()`:

<<< @/../examples/tutorial/11_task_inputs.cpp{33-35,58-59,61-65}

A reference neither extends lifetime nor provides thread safety. The example works because:

1. `processed` is a thread-safe `std::atomic<int>`.
2. The caller invokes `counted.get()` before `processed` leaves scope.
3. No other code touches the same mutable state without synchronization.

If you cannot prove all three, pass a value, move ownership, or use a `shared_ptr` with an explicit synchronization protocol. `future.get()` waits for completion, but cannot repair a data race that already occurred.

## Choose ownership deliberately

| Need | Recommended form | What the task depends on | Main risk |
| --- | --- | --- | --- |
| Small read-only input | `submit(fn, value)` or `[value]` | Its own copy | Copying cost |
| Transfer an exclusive resource | `[value = std::move(value)]` | Exclusive ownership | Submitter cannot reuse the moved value |
| Share a large immutable object | Capture `shared_ptr<const T>` | Shared lifetime | Reference-count and residency cost |
| Invoke a member function | Member pointer plus `shared_ptr` | Object survives completion | A raw object pointer can dangle |
| Modify the caller's object | `std::ref(value)` | External object and synchronization protocol | Dangling reference, race, shutdown order |

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

Next, read [return values and errors](/en/quick-start/return-values-and-errors) to see how these inputs return control through a future after success or failure.
