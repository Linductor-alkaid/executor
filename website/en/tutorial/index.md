---
title: Step-by-Step Tutorials
description: Learn the Executor Facade through a robot data pipeline and a server-side import.
---

# Step-by-Step Tutorials

The tutorials follow one robot data pipeline: acquire `SensorFrame`, produce a `ParsedFrame`, create a `Plan`, then send a `ControlCommand` to a control loop. Each chapter introduces one business problem and only the APIs needed to solve it.

Start with [your first task](/en/quick-start/first-task) and [submitting functions and data](/en/quick-start/task-inputs-and-ownership), then proceed through:

1. [Prioritize control commands](/en/tutorial/priority): urgent control and ordinary analysis share a pool.
2. [Delayed retry and health checks](/en/tutorial/delayed-and-periodic): retry later and stop observable soft-periodic work.
3. [Batch sensor frames](/en/tutorial/batch): choose a batch path based on whether each task needs a future.
4. [Load, sense, then plan](/en/tutorial/dependencies): run planning only after prerequisite work completes.
5. [Bounded waiting and status](/en/tutorial/waiting-and-status): finish safely before a phase change or shutdown.
6. [Complete robot pipeline](/en/tutorial/complete-robot-pipeline): connect startup dependencies, frame streams, configuration, commands, snapshots, diagnostics, and shutdown.
7. [Service data import](/en/tutorial/service-data-import): apply the same dependency, batch, partial-failure, and bounded-drain model to a server request.

Every page identifies scale and concurrency assumptions, ownership of asynchronous objects, failure injection, in-flight work during exit, and when changing requirements require a different abstraction. Treat those sections as an architecture-review checklist.

For API selection by problem, see the Chinese [scenario guide](/zh/guides/choosing-submit-api); detailed real-time and communication material currently remains in Chinese.
