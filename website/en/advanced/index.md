---
title: Advanced and Internals
description: Move beyond the Facade only when resource isolation or lower-level execution paths are genuinely needed.
---

# Advanced and Internals

Most applications begin with the `Executor` Facade. Use an independent `Executor` only for isolated resources; enter public advanced interfaces only for custom cycle sources, direct real-time/GPU control, or queue-implementation analysis.

Start with the [source architecture map](/en/advanced/source-architecture) to establish modules, ownership, and synchronization domains, then follow one path:

1. [Advanced escape hatches](/en/advanced/escape-hatches): responsibility boundaries for instance isolation, `ExecutorManager`, and direct executor pointers.
2. [Custom cycle source](/en/advanced/custom-cycle-manager): implement `ICycleManager` and own its lifecycle.
3. [How tasks travel through Executor](/en/advanced/execution-paths): ordinary and real-time state transitions, promise fulfillment, and completion invariants.
4. [Lock-free and performance experiments](/en/advanced/lockfree-and-performance): `LockFreeTaskExecutor`, MPSC slot protocol, object pool, and backoff.
5. [Performance measurement and regression gates](/en/advanced/performance-measurement): a unified protocol for throughput, tail latency, jitter, correctness, and layered gates.

These internal pages describe the current implementation for debugging and performance analysis. Only public declarations under `include/executor/` carry integration and compatibility expectations; types, structures, and scheduling details under `src/` may change without public-behavior change. Fact sources include [`docs/design/executor.md`](https://github.com/Linductor-alkaid/executor/blob/master/docs/design/executor.md), [`docs/design/lockfree_user_api.md`](https://github.com/Linductor-alkaid/executor/blob/master/docs/design/lockfree_user_api.md), and [`docs/API.md`](https://github.com/Linductor-alkaid/executor/blob/master/docs/API.md).
