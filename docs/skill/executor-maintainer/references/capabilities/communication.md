# Communication Primitives

## Use It When

Search terms: `MpscChannel`, `SpscChannel`, `RealtimeChannel`, latest value, mailbox, snapshot, `DoubleBuffer`, phase, `PhaseGate`, sequencer, drop, overwrite.

Choose by data semantics, not by familiarity with a container. FIFO delivery, latest-only state, immutable snapshots, and phase ordering have different loss and visibility guarantees.

## Public Boundary

- `include/executor/comm.hpp`: aggregate include.
- `include/executor/comm/`: channels, mailbox, double buffer, phase gate, types, and realtime-memory helpers.

## Implementation Trail

Read the header for the chosen primitive and `src/executor/comm/realtime_memory.cpp` where relevant. Comm components maintain their own result/status/event surfaces rather than automatically entering Facade failure status.

## Observable Contract

- Bounded channels report full/drop outcomes; callers cannot assume delivery from a publish attempt alone.
- `LatestMailbox` intentionally overwrites old values; it is not FIFO.
- `DoubleBuffer` is a snapshot mechanism, and phase-bound variants express logical-time visibility rather than general queueing.
- Current mutex-backed primitives do not make a hard-realtime guarantee.

## Change Safeguards

Preserve producer/consumer semantics, stale/overwrite/drop statistics, and event callback behavior. Run the directly matching `tests/test_comm_*.cpp` family and the comm facade harness for cross-component changes.

## Related Material

`website/en/guides/choosing-communication.md` and `website/en/realtime-and-communication/state-and-phases.md`.
