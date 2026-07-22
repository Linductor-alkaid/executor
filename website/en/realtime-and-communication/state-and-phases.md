---
title: Latest Values, Snapshots, and Phases
description: Choose LatestMailbox, DoubleBuffer, PhaseGate, or Sequencer for current configuration, complete state, and startup order.
---

# Latest Values, Snapshots, and Phases

## Goal

Choose `LatestMailbox<T>`, `DoubleBuffer<T>`, `PhaseGate`, or `Sequencer` by data semantics instead of assembling synchronization from a shared object and flags.

## Minimal pipeline

<<< @/../examples/tutorial/08_communication.cpp{1-29}

```bash
./build/examples/tutorial/tutorial_08_communication
```

```text
frame=7, gain=3, state=21, phase=ready
```

## Retain only current configuration

`LatestMailbox<T>` retains the most recent `publish()` value. A real-time consumer uses a sequence to avoid reusing an old configuration:

```cpp
uint64_t seen = 0;
ControlConfig config;
if (mailbox.try_load_newer_than(seen, config, seen)) {
    apply_config(config);
}
```

Overwriting a prior configuration increments `overwritten_count`; no higher sequence is a stale read, not a lost new message. `publish(value)` copies an lvalue and `publish(std::move(value))` moves it into the mailbox; readers copy the current value to their output object. The mailbox never keeps a reference to a publisher local, but pointers/views inside `T` still need application-managed backing lifetime. For a large immutable configuration, use `shared_ptr<const Config>` after complete validation.

## Publish complete state

`DoubleBuffer<T>` is for a single writer and multiple readers. `publish()` or `update()` builds the inactive buffer completely then releases it at once; `load()` or `load_newer_than()` returns a value-copy `Snapshot<T>`, never a partially updated object. Funnel multiple writers through an `MpscChannel` to one state owner, and assess copying cost for large state.

The current implementation uses a mutex to preserve complete snapshots; it does not promise lock-free reads. Choose it for its value-snapshot and ownership semantics, not as a hard-real-time primitive.

`update()` modifies the inactive buffer synchronously on the writer path; it is not an asynchronous Executor submission. Its references need only cover that immediate call, but it still obeys the one-writer constraint. A reader's snapshot remains its own copy after later publication.

## Phases and strict order

Use `PhaseGate` for monotonic setup/calibration/running stages. `advance_to()` cannot repeat or regress; `wait_for()` distinguishes success, `Timeout`, and `Closed`; `wait_for_exact()` also exposes a skipped stage as `MissedPhase`.

Use `Sequencer` for strict ticket order: `next_ticket()` allocates, `publish(ticket)` advances, and `wait_until_published(ticket, timeout)` returns `MissedPhase` when the target has already been passed. It is not a data queue and cannot replace `MpscChannel`.

Next: [communication observability](/en/realtime-and-communication/observability).
