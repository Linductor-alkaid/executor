---
title: Deliver Every Message
description: Use MpscChannel for ordinary data flow and a budgeted RealtimeChannel for periodic consumption.
---

# Deliver Every Message

## Goal

Distinguish “every frame must be handled” from “a real-time cycle may handle only a bounded number of commands,” then choose `MpscChannel<T>` or `RealtimeChannel<T>` accordingly.

## Ordinary flow: `MpscChannel`

A frame stream from acquisition to planning commonly requires FIFO consumption. `MpscChannel<T>` is a bounded multi-producer/single-consumer channel; an ordinary consumer can use `receive_for()` to bound its wait.

```cpp
executor::comm::ChannelOptions options;
options.capacity = 256;
options.drop_policy = executor::comm::DropPolicy::RejectNewest;

executor::comm::MpscChannel<SensorFrame> frames(options);
frames.try_send(SensorFrame{});

SensorFrame frame;
if (frames.receive_for(frame, std::chrono::milliseconds(10))) {
    plan(frame);
}
frames.close();
```

`try_send()` and `try_receive()` do not wait. `send_for()` and `receive_for()` use `CommResult` to distinguish `Timeout`, `Closed`, and other outcomes. `close()` stops new production and wakes waiters, but already queued data can still drain.

## A channel transfers values, not task functions

Communication components have no callable/argument binding model. Producers submit a `T`: lvalues copy, rvalues move. After successful sending, the consumer owns the channel's message rather than a reference to the producer stack.

```cpp
SensorFrame frame = capture_frame();
if (!frames.try_send(std::move(frame))) {
    handle_backpressure();
}
```

Do not rely on the original contents after moving. A raw pointer, span, or view transfers only that lightweight object, not its backing storage; the application keeps the buffer valid until consumption. Prefer a buffer handle with defined return rules or a smart pointer, and define pool-exhaustion backpressure.

## Real-time cycle: `RealtimeChannel`

A real-time thread must not wait on a condition variable or clear unlimited backlog. Bound per-cycle work with `drain_for_cycle()`:

```cpp
executor::comm::RealtimeChannelOptions options;
options.capacity = 128;
options.max_items_per_cycle = 8;

executor::comm::RealtimeChannel<ControlCommand> commands(options);
commands.try_send(ControlCommand{});
commands.drain_for_cycle([](const ControlCommand& command) {
    apply_command(command);
});
```

`max_items == 0` uses configured `max_items_per_cycle`; only a configuration value of `0` means unlimited. Preserve a production bound so backlog cannot consume the entire control period. A handler exception stops this drain, increments `handler_exception_count`, emits `HandlerException`, and continues propagating the exception.

The handler receives `const T&` valid only for that call. Copy or transfer it to an owned object before retaining it asynchronously. Keep the handler bounded and nonblocking because it runs inside the real-time cycle.

| Policy | Meaning | Typical use |
| --- | --- | --- |
| `RejectNewest` | Reject new message; producer observes failure | Every message matters; caller applies backpressure |
| `DropOldest` | Discard oldest, accept newest | Newer data matters more but remains queue semantics |
| `KeepLatest` | Retain only newest message | Near state semantics; `LatestMailbox` may fit directly |

Never ignore `try_send()`. Capacity, drop policy, peak depth, and drop count form a backpressure contract, not a performance value to increase arbitrarily. For latest-only configuration, continue with [latest values, snapshots, and phases](/en/realtime-and-communication/state-and-phases).
