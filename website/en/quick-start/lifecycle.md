---
title: Initialization and Shutdown
description: Configure Executor before first use and stop it deliberately at an application boundary.
---

# Initialization and Shutdown

The minimal program can rely on lazy initialization. When you need custom thread counts, queue capacity, or monitoring, fix configuration before the first submission.

```cpp
executor::ExecutorConfig config;
config.min_threads = 2;
config.max_threads = 4;

auto& executor = executor::Executor::instance();
auto initialized = executor.initialize_ex(config);
if (!initialized) {
    throw std::runtime_error(initialized.message);
}

// submit() ... future.get() ...
executor.shutdown(true);
```

`initialize_ex()` returns an `ExecutorResult` with an error code and message, which is more useful for diagnosis than the compatible `bool` initializer. `shutdown(true)` waits for accepted asynchronous work; when the library's default wait limit is exceeded, it records a timeout diagnostic and continues with non-waiting shutdown. It is not an infinite-wait guarantee.

The singleton has a `shutdown(false)` process-exit fallback, and independent `Executor` instances clean up through destruction. Neither removes the application's responsibility to decide whether accepted work must finish at its business boundary.

## Common mistakes

- Changing initialization configuration after the first `submit()` completed lazy initialization.
- Exiting without waiting for owned futures or choosing a shutdown policy.
- Treating initialization errors and task errors as the same signal: inspect `ExecutorResult` for setup and futures for task execution.

For version-specific compatibility and upgrade guidance, read [versions and migration](/en/reference/version-and-migration).
