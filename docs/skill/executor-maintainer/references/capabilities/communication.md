# Communication Primitives

## Use It When

Search terms: `MpscChannel`, `SpscChannel`, `Topic`, `TopicSubscription`, fan-out, pub/sub, `RealtimeChannel`, latest value, mailbox, snapshot, `DoubleBuffer`, phase, `PhaseGate`, sequencer, drop, overwrite.

Choose by data semantics, not by familiarity with a container. Single-consumer FIFO, independent-subscriber FIFO, latest-only state, immutable snapshots, and phase ordering have different loss and visibility guarantees.

## Public Boundary

- `include/executor/comm.hpp`: aggregate include.
- `include/executor/comm/`: channels, in-process Topic/subscriptions, mailbox, double buffer, phase gate, types, and realtime-memory helpers.

## Implementation Trail

Read the header for the chosen primitive and `src/executor/comm/realtime_memory.cpp` where relevant. Comm components maintain their own result/status/event surfaces rather than automatically entering Facade failure status.

## Observable Contract

- Bounded channels report full/drop outcomes; callers cannot assume delivery from a publish attempt alone.
- `Topic<T>` fans post-subscription events out to independent bounded subscription queues. `TopicPublishResult` reports matched, delivered, and rejected subscriber counts; no subscribers is a successful empty delivery.
- A subscription's slow/full/closed state affects only that subscription. Publish snapshot and registry removal are the publish/unsubscribe linearization points; an in-flight snapshot may deliver or reject after concurrent unsubscribe without accessing destroyed state.
- Topic/subscription close wakes waiters and permits queued messages to drain. Topic does not provide replay, acknowledgements, cross-subscriber atomicity, networking, or hard-realtime guarantees.
- `LatestMailbox` intentionally overwrites old values; it is not FIFO.
- `DoubleBuffer` is a snapshot mechanism, and phase-bound variants express logical-time visibility rather than general queueing.
- Current mutex-backed primitives do not make a hard-realtime guarantee.

## Change Safeguards

Preserve producer/consumer semantics, stale/overwrite/drop statistics, and event callback behavior. For Topic changes, preserve move-only RAII subscription ownership, stable in-flight subscription lifetime, and per-subscription backpressure isolation. Run the directly matching `tests/test_comm_*.cpp` family and the comm facade harness for cross-component changes; run `test_comm_topic` under TSAN for registry/lifetime changes.

## Related Material

`website/en/guides/choosing-communication.md`, `website/en/tutorial/complete-robot-pipeline.md`, and `website/en/realtime-and-communication/state-and-phases.md`.
