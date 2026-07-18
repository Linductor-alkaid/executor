---
title: Reliability
description: Keep exceptions, timeouts, rejections, and shutdown behavior observable.
---

# Reliability

Reliability is more than adding a `try/catch`. A caller must distinguish an immediate result, submission rejection, task failure, wait timeout, and long-term trend, with a suitable observation path for each.

1. [Failure observability](/en/reliability/failure-observability): combine futures, failure callbacks, cumulative counts, and recent events.
2. [Monitoring and sampling](/en/reliability/monitoring): collect task statistics with the required precision and acceptable overhead.
3. [Troubleshoot by symptom](/en/reliability/troubleshooting): use a fixed check order for no execution, backlog, wait timeout, stalled shutdown, real-time drops, and unavailable GPU.
4. [Linux and Windows deployment](/en/reliability/platform-deployment): verify builds, CPU availability, real-time permissions, memory locking, and platform-specific fallback actually take effect.

Communication drops, overwrites, stale reads, and missed phases remain component-local events; they do not automatically enter `ExecutorFailureStatus`. Define their observation path in [choose a communication component](/en/guides/choosing-communication).
