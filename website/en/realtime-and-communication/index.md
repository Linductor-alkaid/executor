---
title: Real-Time and Communication
description: Choose dedicated real-time threads and cross-thread communication components by data semantics.
---

# Real-Time and Communication

Ordinary `submit_periodic()` is soft periodic work on a thread pool. Fixed-period control, cycle budgets, and real-time queues require a dedicated real-time thread; these are different abstraction layers.

Start with the [complete robot pipeline](/en/tutorial/complete-robot-pipeline), which establishes roles, data ownership, and shutdown protocol before mapping each edge to a component.

1. [Dedicated real-time control loop](/en/realtime-and-communication/realtime-control): replace the portable periodic simulation with a diagnosable real-time Facade.
2. [Deliver every message](/en/realtime-and-communication/channels): ordinary frame flow and bounded draining inside a real-time cycle.
3. [Latest values, snapshots, and phases](/en/realtime-and-communication/state-and-phases): configuration, complete state, and startup order.
4. [Communication observability](/en/realtime-and-communication/observability): understand `CommStats` and local event callbacks.
5. [Capacity and alerts](/en/realtime-and-communication/capacity-and-alerting): turn cumulative statistics into window rates, margins, alert levels, and actions.

The complete API is in [`docs/API.md`](https://github.com/Linductor-alkaid/executor/blob/master/docs/API.md); the integrated runnable fact source is [`examples/comm_robot_pipeline.cpp`](https://github.com/Linductor-alkaid/executor/blob/master/examples/comm_robot_pipeline.cpp).
