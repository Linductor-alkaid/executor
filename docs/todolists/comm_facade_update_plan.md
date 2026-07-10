# 通信与并发辅助 Facade 更新计划

本文档承接阶段 7：[通信与并发辅助 Facade 设计](../design/comm_facade.md)。

目标是提供 executor 级别的高层通信/并发抽象，让控制线程、采集线程、规划线程、通信线程之间的数据传递和时序控制默认安全、可观察，并且适合实时线程集成。

---

## 当前基线

- [x] `util::LockFreeQueue<T>` 已提供 MPSC 无锁队列、容量、批量 push/pop、失败 push 和峰值统计。
- [x] `RealtimeThreadExecutor` 已有实时 push、队列满/未运行/对象池耗尽等背压计数。
- [x] `Executor` facade 已有 failure event、recent failures、wait result、周期任务状态等诊断入口。
- [x] `TaskDependencyManager` 已有依赖图、ready 检查、完成标记和 cycle 检测。
- [x] `tests/harness/test_comm_facade_usage.cpp` 已有 disabled 用例，占位覆盖 channel、mailbox、phase gate、double buffer、realtime channel。
- [x] 已新增公开的 `executor::comm` 命名空间和聚合头。
- [x] Typed Channel 场景下用户无需自行组合锁、原子变量、条件变量或底层队列来表达跨线程通信。

---

## 任务映射

| id | category | area | severity | effort | title |
|----|----------|------|----------|--------|-------|
| P1 | feature | inter-thread communication | high | M | Typed Channel：类型安全 MPSC/SPSC 数据通道，解决共享变量 + 锁的易错用法 |
| P2 | feature | timing/order control | high | M/L | PhaseGate / Sequencer：显式控制线程间步骤顺序，解决“先后时序不确定” |
| P3 | feature | real-time integration | high | M | Realtime Mailbox：实时线程周期内消费最新数据 / 队列数据，带背压和丢弃策略 |
| P4 | feature | data race avoidance | medium | M | Snapshot / DoubleBuffer：读写线程无锁交换状态快照，替代共享 mutable state |
| P5 | feature | task graph | medium | L | submit_after / when_all：把已有 TaskDependencyManager 暴露为任务时序 API |
| P6 | diagnostics | observability | medium | M | 通信时序监控：drop、latency、stale、missed phase、producer/consumer lag |

---

## 设计原则

1. **默认拒绝静默丢失**：满队列默认 `RejectNewest`，返回 `false` 并累计统计。
2. **实时线程不阻塞**：实时消费 API 使用 `try_*` / `drain_for_cycle()`，不做无限等待。
3. **类型安全优先**：用户传递 `T`、`Snapshot<T>`、`TaskHandle`，不传裸 `void*` 或共享 mutable 引用。
4. **生命周期显式**：所有通道支持 `close()`，关闭后发送和等待行为必须可预测。
5. **观察入口一致**：每个组件有 `stats()`；可选 `CommEventCallback`，后续再接入 `Executor` 聚合状态。
6. **先保守可用，再优化内部实现**：第一版可以用锁保护非实时/非平凡类型路径，但 API 不泄露锁语义。

---

## 阶段 7.0：API 骨架与文档入口

### 任务

- [x] 新增 `include/executor/comm.hpp` 聚合头。
- [x] 新增 `include/executor/comm/types.hpp`：
  - `CommErrorCode`
  - `CommResult`
  - `DropPolicy`
  - `CommStats`
  - `CommEventKind`
  - `CommEvent`
  - `CommEventCallback`
- [x] 新增空实现/最小声明文件，保证后续组件有稳定命名空间。
- [x] 在 `docs/API.md` 增加阶段 7 API 索引和最小使用提示。

### 验收

- [x] `#include <executor/comm.hpp>` 可编译。
- [x] `executor::comm` 通用类型有单元测试覆盖默认值、bool 转换和错误码字符串化。

---

## 阶段 7.1：P1 Typed Channel

### 任务

- [x] 实现 `MpscChannel<T>`：
  - `try_send(const T&)`
  - `try_send(T&&)`
  - `try_receive(T&)`
  - `send_for(...)`
  - `receive_for(...)`
  - `close()`
  - `stats()`
- [x] 实现 `SpscChannel<T>` 类型别名或轻量封装，后续可替换为 SPSC 优化实现。
- [x] 支持 `ChannelOptions`：
  - `capacity`
  - `drop_policy`
  - `enable_stats`
  - `name`
- [x] 满队列默认拒绝新消息；`DropOldest` 必须增加 `dropped_count`。
- [x] 关闭后发送返回失败，消费者可 drain 剩余数据。

### 测试

- [x] 启用/替换 `FacadeCommUsage.SensorProducerPlannerConsumer`。
- [x] 单生产者单消费者 FIFO 顺序测试。
- [x] 多生产者单消费者并发压力测试。
- [x] 队列满返回 false 且 `dropped_count` 或 `closed_send_count` 增加。
- [x] `close()` 唤醒阻塞 `receive_for()`。
- [x] 非平凡类型传递测试，如 `std::string` 或 move-only 类型视实现范围决定。

### 验收

- [x] 用户无需手写 mutex 即可完成 producer/consumer 数据传递。
- [x] 队列满、关闭后提交、超时等待均可通过返回值和统计观察。

---

## 阶段 7.2：P3 LatestMailbox / RealtimeChannel

### 任务

- [x] 实现 `LatestMailbox<T>`：
  - `publish(...)`
  - `try_load(T&)`
  - `try_load_newer_than(last_seen, out, new_sequence)`
  - `sequence()`
  - `stats()`
- [x] 实现 `RealtimeChannel<T>`：
  - `try_send(...)`
  - `drain_for_cycle(handler, max_items)`
  - `close()`
  - `stats()`
- [x] 支持 `RealtimeChannelOptions`：
  - `capacity`
  - `max_items_per_cycle`
  - `drop_policy`
  - `enable_stats`
  - `name`
- [x] 定义 handler 抛异常时的最小语义：停止本轮 drain、增加 `handler_exception_count`、记录 `HandlerException` 事件并继续外抛。
- [x] 与 `RealtimeThreadConfig::max_tasks_per_cycle` 的文档语义对齐：`0` 表示不限，非 0 表示单周期预算。

### 测试

- [x] 启用/替换 `FacadeCommUsage.ConfigThreadRealtimeControlThread`。
- [x] 启用/替换 `FacadeCommUsage.RealtimeCycleDrainsMessages`。
- [x] mailbox 多次 publish 后实时线程只消费最新值。
- [x] `try_load_newer_than()` 不重复消费旧 sequence。
- [x] `drain_for_cycle()` 不超过每周期预算。
- [x] 满队列策略计数正确。

### 验收

- [x] 实时周期内可用非阻塞 API 消费最新配置或有限条消息。
- [x] producer 过快时，drop/overwrite/lag 均可观察。

---

## 阶段 7.3：P2 PhaseGate / Sequencer

### 任务

- [x] 实现 `PhaseGate`：
  - `current_phase()`
  - `advance()`
  - `advance_to(phase)`
  - `has_reached(phase)`
  - `wait_for(phase, timeout)`
  - `close()`
  - `stats()`
- [x] 实现 `Sequencer`：
  - `next_ticket()`
  - `publish(ticket)`
  - `is_published(ticket)`
  - `wait_until_published(ticket, timeout)`
- [x] 定义 phase 倒退、重复 advance、close 后 wait 的返回语义。
- [x] missed phase 计数进入 `CommStats::missed_phase_count`。

### 测试

- [x] 启用/替换 `FacadeCommUsage.InitThreadWorkerThread`。
- [x] wait-before-advance 正常唤醒。
- [x] wait-after-advance 立即成功。
- [x] close 唤醒所有 waiter。
- [x] missed phase 场景返回 `MissedPhase` 并计数。
- [x] 并发 waiter 压力测试。

### 验收

- [x] 用户无需手写 condition variable 即可表达启动、采集、规划、通信等阶段顺序。
- [x] 时序错过或等待超时可观察。

---

## 阶段 7.4：P4 Snapshot / DoubleBuffer

### 任务

- [ ] 实现 `Snapshot<T>`：
  - `value`
  - `sequence`
  - `timestamp`
- [ ] 实现 `DoubleBuffer<T>`：
  - `publish(T)`
  - `update(fn)`
  - `load()`
  - `load_newer_than(last_seen, out)`
  - `sequence()`
  - `stats()`
- [ ] 明确第一版写入模型：单写多读；多写场景建议通过 channel 汇聚。
- [ ] 文档说明大型对象的复制成本和后续 `SnapshotPtr<T>` 扩展方向。

### 测试

- [ ] 启用/替换 `FacadeCommUsage.StateWriterMonitorReader`。
- [ ] 读者只能看到完整发布后的状态。
- [ ] 多读者并发读取无 data race。
- [ ] `load_newer_than()` 避免重复消费旧状态。
- [ ] writer 高频更新时 reader 不读到半更新状态。

### 验收

- [ ] 共享 mutable state 的典型读写场景可迁移到不可变快照。
- [ ] 读者有 sequence 判断新旧数据。

---

## 阶段 7.5：P5 submit_after / when_all

### 任务

- [ ] 设计并实现 `TaskHandle`。
- [ ] 在 `Executor` facade 增加：
  - `submit_after(dependency, task)`
  - `submit_after(dependencies, task)`
  - `when_all(dependencies)`
- [ ] 内部复用或扩展 `TaskDependencyManager`。
- [ ] 定义依赖失败传播策略：
  - 默认不执行 dependent task。
  - dependent future 返回可诊断异常。
  - 依赖图 cycle 返回 `ExecutorResult` 或 ready future 异常。
- [ ] 增加依赖状态裁剪，避免长生命周期服务中任务 ID 无限增长。

### 测试

- [ ] `submit_after(A, B)` 保证 B 在 A 完成后执行。
- [ ] `when_all(A, B)` 等待全部完成后触发 C。
- [ ] A 失败时 B 不执行且 future 可观察。
- [ ] cycle 或无效 handle 返回可诊断失败。
- [ ] shutdown 期间 pending dependency 不静默丢失。

### 验收

- [ ] 用户可通过 facade 表达任务时序，而不是手写 promise/future 链或轮询 `TaskDependencyManager`。

---

## 阶段 7.6：P6 通信时序监控

### 任务

- [ ] 每个组件补齐 `stats()` 字段：
  - drop
  - latency
  - stale
  - missed phase
  - producer lag
  - consumer lag
  - peak depth
- [ ] 增加 `set_event_callback()`，用于低频诊断事件。
- [ ] 评估 `Executor` 级聚合：
  - `set_comm_event_callback(...)`
  - `get_comm_status()`
- [ ] 文档明确：通信事件默认不计入任务失败计数。
- [ ] 增加 README / README_zh / docs/API.md 最小示例。

### 测试

- [ ] drop/overwrite/stale/missed phase/timeout 各有明确计数测试。
- [ ] callback 抛异常不会破坏通信组件状态。
- [ ] 高频路径未注册 callback 时不产生额外日志。

### 验收

- [ ] 用户能回答“丢了多少、延迟多大、是否读旧、是否错过 phase、生产/消费谁落后”。

---

## 阶段 7.7：示例与迁移材料

### 任务

- [ ] 新增 `examples/comm_channel.cpp`：采集线程到规划线程。
- [ ] 新增 `examples/realtime_mailbox.cpp`：配置线程到实时控制线程。
- [ ] 新增 `examples/phase_gate_startup.cpp`：初始化顺序控制。
- [ ] 新增 `examples/double_buffer_state.cpp`：监控线程读取状态快照。
- [ ] 更新 `docs/MIGRATION.md`：从共享变量、手写锁、底层队列迁移到 comm facade。

### 验收

- [ ] 示例优先展示 `executor::comm`，不要求用户先理解底层 `LockFreeQueue`。
- [ ] 文档说明何时选择 Channel、Mailbox、DoubleBuffer、PhaseGate、TaskHandle。

---

## 推荐实施顺序

1. 阶段 7.0：通用类型、聚合头、文档入口。
2. 阶段 7.1：Typed Channel，先解决最常见 producer/consumer。
3. 阶段 7.2：LatestMailbox / RealtimeChannel，补齐实时配置和周期 drain。
4. 阶段 7.3：PhaseGate / Sequencer，解决严格步骤顺序。
5. 阶段 7.4：DoubleBuffer，替代共享 mutable state。
6. 阶段 7.6：通信时序监控贯穿补齐，也可随每个组件同步落地。
7. 阶段 7.5：submit_after / when_all，作为较大任务图扩展单独推进。
8. 阶段 7.7：示例、README/API/MIGRATION 同步。

这个顺序先覆盖最高频、最高风险的跨线程数据传递和实时消费，再扩展任务图 API。

---

## 测试矩阵

- [ ] `test_comm_channel.cpp`
  - [ ] FIFO
  - [ ] MPSC 并发
  - [ ] full/drop/close/timeout
- [ ] `test_comm_mailbox.cpp`
  - [ ] latest wins
  - [ ] sequence freshness
  - [ ] overwrite/stale stats
- [ ] `test_comm_realtime_channel.cpp`
  - [ ] cycle drain budget
  - [ ] no blocking wait on realtime drain path
  - [ ] backpressure counters
- [ ] `test_comm_phase_gate.cpp`
  - [ ] wait/advance
  - [ ] close wakeup
  - [ ] missed phase
- [ ] `test_comm_double_buffer.cpp`
  - [ ] complete snapshot
  - [ ] concurrent readers
  - [ ] no repeated stale consume
- [ ] `test_executor_task_graph_facade.cpp`
  - [ ] submit_after
  - [ ] when_all
  - [ ] failure propagation
  - [ ] cycle/invalid handle diagnostics
- [ ] `tests/harness/test_comm_facade_usage.cpp`
  - [ ] replace disabled placeholders with compiling usage tests.

---

## Open Questions

- 第一版 `MpscChannel<T>` 是否支持 move-only 非平凡类型，还是先支持 copy/move constructible 普通对象。
- `DoubleBuffer<T>::load()` 对大对象复制成本较高，是否同步提供 `SnapshotPtr<T>`。
- `send_for()` / `receive_for()` 是否需要可中断 stop token，还是先通过 `close()` 唤醒。
- `submit_after()` 返回 `TaskHandle + future` 的组合形式如何设计，才能同时保留返回值和依赖能力。
- 通信诊断是否需要进入现有 `FailureKind`，还是保持独立 `CommEvent` 聚合。
