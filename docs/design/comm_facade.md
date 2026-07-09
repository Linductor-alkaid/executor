# 通信与并发辅助 Facade 设计

本文档是阶段 7 的设计稿，目标是把常见跨线程通信模式提升到 `executor::comm`
层，让用户默认走类型安全、生命周期清晰、可观察的抽象，而不是直接组合
`mutex`、`atomic`、`condition_variable`、共享可变对象和底层无锁队列。

关联计划：[通信与并发辅助 Facade 更新计划](../todolists/comm_facade_update_plan.md)。

---

## 背景与问题

多线程共享状态时，常见错误并不来自“没有工具”，而是来自工具组合太低层：

- 写线程和读线程谁先发生不确定，用户容易消费旧状态或半更新状态。
- 控制线程、采集线程、规划线程、通信线程之间有严格步骤顺序，但代码里只剩零散的锁和条件变量。
- `notify_one()` 漏掉、wait predicate 写错、ABA、relaxed/acquire/release 用错都会变成偶发问题。
- 队列满、覆盖旧值、实时周期内来不及消费时，如果没有统一状态，问题会表现为“系统偶尔不响应”。
- 实时线程不应被普通锁、堆分配、无限等待拖住。

项目已有的基础能力：

- `executor::util::LockFreeQueue<T>`：MPSC 无锁队列，要求 `T` trivially copyable，已有容量、失败 push、峰值等统计。
- `RealtimeThreadExecutor`：已有周期执行、每周期任务预算、实时 push 背压计数。
- `Executor` facade：已有 failure event、recent failure、wait result、周期任务状态等可观察性入口。
- `TaskDependencyManager`：已有依赖图、ready 检查、完成标记和 cycle 检测，但还没有高层任务时序 API。

阶段 7 不再把这些基础类型直接暴露给普通用户，而是提供更明确的通信语义。

---

## 设计目标

1. **默认写不出 data race**：API 不暴露可并发写读的裸引用；跨线程数据通过值、不可变快照或受控写入窗口传递。
2. **时序显式化**：线程间“先初始化后工作”“先采集后规划”“第 N 相位后再执行”通过 `PhaseGate` / `Sequencer` 表达。
3. **背压不静默**：队列满、覆盖、过期、关闭后提交都必须通过返回值和统计可见。
4. **实时路径非阻塞**：实时线程消费 API 默认无锁、无无限等待；可能阻塞的 API 在命名和文档中显式标注。
5. **可诊断但低开销**：每个组件提供本地 `stats()`；可选接入 `Executor` 的诊断回调，但不把通信抖动混为任务失败。
6. **渐进实现**：先支持单消费者、有限容量和明确策略；多消费者广播、复杂 reactive graph 留到后续版本。

## 非目标

- 不替代完整 actor framework、Rx 框架或分布式消息系统。
- 不承诺所有组件 lock-free；承诺的是用户层无 data race 和实时消费路径不阻塞。
- 不强制所有类型都走同一内部实现。平凡可复制小对象可复用无锁队列，复杂对象可用受控锁或 `shared_ptr<const T>` 快照。

---

## 命名空间与头文件

建议新增聚合头：

```cpp
#include <executor/comm.hpp>
```

核心命名空间：

```cpp
namespace executor::comm {
// Channel, mailbox, phase gate, double buffer, task sequencing helpers.
}
```

建议文件布局：

- `include/executor/comm.hpp`：聚合头。
- `include/executor/comm/types.hpp`：通用结果、策略、统计、事件类型。
- `include/executor/comm/channel.hpp`：`MpscChannel<T>` / `SpscChannel<T>`。
- `include/executor/comm/mailbox.hpp`：`LatestMailbox<T>` / `RealtimeChannel<T>`。
- `include/executor/comm/phase_gate.hpp`：`PhaseGate` / `Sequencer`。
- `include/executor/comm/double_buffer.hpp`：`Snapshot<T>` / `DoubleBuffer<T>`。
- `src/executor/comm/*.cpp`：非模板实现和诊断格式化。

---

## 通用类型

```cpp
namespace executor::comm {

enum class CommErrorCode {
    Ok,
    Closed,
    Full,
    Empty,
    Timeout,
    Stale,
    MissedPhase,
    InvalidArgument,
    NotReady
};

struct CommResult {
    bool ok = true;
    CommErrorCode error_code = CommErrorCode::Ok;
    std::string message;

    explicit operator bool() const noexcept { return ok; }
};

enum class DropPolicy {
    RejectNewest,  // 默认：满时拒绝新消息，不静默丢。
    DropOldest,    // 满时丢最旧消息，记录 drop。
    KeepLatest     // 保留最新值，适合 mailbox，不适合普通 FIFO 语义。
};

struct CommStats {
    uint64_t sent_count = 0;
    uint64_t received_count = 0;
    uint64_t dropped_count = 0;
    uint64_t overwritten_count = 0;
    uint64_t stale_read_count = 0;
    uint64_t closed_send_count = 0;
    uint64_t timeout_count = 0;
    uint64_t missed_phase_count = 0;
    uint64_t current_depth = 0;
    uint64_t peak_depth = 0;
    uint64_t capacity = 0;
    uint64_t producer_lag = 0;
    uint64_t consumer_lag = 0;
    std::chrono::nanoseconds max_latency{0};
    std::chrono::nanoseconds avg_latency{0};
};

enum class CommEventKind {
    Dropped,
    Overwritten,
    ClosedSend,
    Timeout,
    StaleRead,
    MissedPhase,
    ProducerLag,
    ConsumerLag,
    LatencyHigh
};

struct CommEvent {
    CommEventKind kind;
    std::string component_name;
    std::string message;
    uint64_t sequence = 0;
    std::chrono::steady_clock::time_point timestamp =
        std::chrono::steady_clock::now();
};

using CommEventCallback = std::function<void(const CommEvent&)>;

} // namespace executor::comm
```

说明：

- `CommResult` 用于失败可解释的控制操作，如 `close()`、`advance_to()`、`wait_for()`。
- 高频数据路径优先提供 `bool try_*`，避免每条消息构造字符串。
- `CommStats` 是本地累计统计；`CommEventCallback` 是可选诊断，不建议默认对每次 drop 分配字符串。
- 通信事件默认不是 `FailureKind::TaskException`。它们可以汇总到 Executor 诊断面板，但不污染任务失败计数。

---

## P1：Typed Channel

Typed Channel 解决“共享变量 + 锁”易错的问题。它提供明确的所有权转移和容量语义。

### API 草案

```cpp
namespace executor::comm {

struct ChannelOptions {
    size_t capacity = 1024;
    DropPolicy drop_policy = DropPolicy::RejectNewest;
    bool enable_stats = true;
    std::string name;
};

template<class T>
class MpscChannel {
public:
    explicit MpscChannel(ChannelOptions options = {});

    bool try_send(const T& value);
    bool try_send(T&& value);

    template<class Rep, class Period>
    CommResult send_for(T value, std::chrono::duration<Rep, Period> timeout);

    bool try_receive(T& out);

    template<class Rep, class Period>
    CommResult receive_for(T& out, std::chrono::duration<Rep, Period> timeout);

    void close();
    bool is_closed() const;
    bool empty() const;
    size_t size_approx() const;
    size_t capacity() const;
    CommStats stats() const;
    void set_event_callback(CommEventCallback callback);
};

template<class T>
using SpscChannel = MpscChannel<T>; // 初期可复用实现，后续替换为 SPSC 优化。

} // namespace executor::comm
```

### 语义

- 默认有界容量，满时 `try_send()` 返回 `false` 并增加 `dropped_count` 或 `closed_send_count`，不静默丢。
- `DropPolicy::RejectNewest` 是默认策略，适合命令、任务和事件流。
- `DropPolicy::DropOldest` 适合日志、遥测；必须增加 `dropped_count`。
- FIFO channel 不使用 `KeepLatest` 作为默认。只需要最新状态时应使用 `LatestMailbox<T>`。
- `close()` 后生产者不能再提交；消费者可继续 drain 已有数据，直到空。
- `receive_for()` 可阻塞普通线程；实时线程应使用 `try_receive()` 或 `RealtimeChannel::drain_for_cycle()`。

### 实现建议

- 对 `std::is_trivially_copyable_v<T>` 且容量固定的场景，可复用 `util::LockFreeQueue<T>`。
- 对非平凡类型，第一版可使用内部互斥锁保护环形缓冲区，但不把锁暴露给用户。
- 后续可引入 node pool 或 move-only ring buffer，减少非平凡类型的分配和锁竞争。

---

## P2：PhaseGate / Sequencer

PhaseGate 解决“线程间步骤顺序不确定”的问题。用户不再手写 condition variable predicate。

### PhaseGate API 草案

```cpp
namespace executor::comm {

class PhaseGate {
public:
    explicit PhaseGate(std::string name = {});

    uint64_t current_phase() const;

    CommResult advance_to(uint64_t phase);
    CommResult advance();

    bool has_reached(uint64_t phase) const;

    template<class Rep, class Period>
    CommResult wait_for(uint64_t phase,
                        std::chrono::duration<Rep, Period> timeout);

    CommResult close();
    bool is_closed() const;
    CommStats stats() const;
    void set_event_callback(CommEventCallback callback);
};

} // namespace executor::comm
```

### Sequencer API 草案

```cpp
namespace executor::comm {

class Sequencer {
public:
    uint64_t next_ticket();
    CommResult publish(uint64_t ticket);
    bool is_published(uint64_t ticket) const;

    template<class Rep, class Period>
    CommResult wait_until_published(uint64_t ticket,
                                    std::chrono::duration<Rep, Period> timeout);
};

} // namespace executor::comm
```

### 语义

- phase 单调递增，不允许倒退。
- `wait_for(p)` 在当前 phase 已超过 `p` 时成功返回；调用方要求精确消费某个 phase 时可使用 Sequencer。
- 如果调用方声明必须看到 phase N，但当前已经是 N+K，返回 `MissedPhase` 并增加统计。
- `close()` 唤醒所有 waiter，返回 `Closed`。
- 内部可使用 `mutex + condition_variable`，但 predicate 和 notify 由组件封装。

---

## P3：Realtime Mailbox / RealtimeChannel

实时线程有两类常见消费模式：

- 每周期只关心最新配置或目标值：使用 `LatestMailbox<T>`。
- 每周期 drain 一批消息但不能无限处理：使用 `RealtimeChannel<T>`。

### LatestMailbox API 草案

```cpp
namespace executor::comm {

template<class T>
class LatestMailbox {
public:
    explicit LatestMailbox(std::string name = {});

    void publish(const T& value);
    void publish(T&& value);

    // 返回 false 表示从未发布过值。
    bool try_load(T& out) const;

    // 若 sequence 未变化，返回 false，避免重复消费旧数据。
    bool try_load_newer_than(uint64_t last_seen_sequence,
                             T& out,
                             uint64_t& new_sequence) const;

    uint64_t sequence() const;
    CommStats stats() const;
    void set_event_callback(CommEventCallback callback);
};

} // namespace executor::comm
```

### RealtimeChannel API 草案

```cpp
namespace executor::comm {

struct RealtimeChannelOptions {
    size_t capacity = 1024;
    size_t max_items_per_cycle = 64;
    DropPolicy drop_policy = DropPolicy::RejectNewest;
    bool enable_stats = true;
    std::string name;
};

template<class T>
class RealtimeChannel {
public:
    explicit RealtimeChannel(RealtimeChannelOptions options = {});

    bool try_send(const T& value);
    bool try_send(T&& value);

    // 实时线程调用：不阻塞，不分配，不超过 max_items。
    template<class Fn>
    size_t drain_for_cycle(Fn&& handler, size_t max_items = 0);

    void close();
    CommStats stats() const;
    void set_event_callback(CommEventCallback callback);
};

} // namespace executor::comm
```

### 实时约束

- `drain_for_cycle()` 不等待，不调用 condition variable。
- `max_items == 0` 表示使用配置中的 `max_items_per_cycle`。
- handler 抛异常时不在实时路径做复杂恢复；建议返回已处理数量，并记录轻量错误计数。是否把异常桥接到 `Executor` failure event 由集成层决定。
- 队列满策略必须显式，默认拒绝新消息并返回 `false`。

---

## P4：Snapshot / DoubleBuffer

Snapshot / DoubleBuffer 解决“读到半更新状态”和“共享 mutable state”的问题。

### API 草案

```cpp
namespace executor::comm {

template<class T>
struct Snapshot {
    T value;
    uint64_t sequence = 0;
    std::chrono::steady_clock::time_point timestamp;
};

template<class T>
class DoubleBuffer {
public:
    explicit DoubleBuffer(T initial = {});

    // 单写线程或外部保证写入串行。多写版本后续可提供 MultiWriterDoubleBuffer。
    template<class Fn>
    uint64_t update(Fn&& writer);

    uint64_t publish(T value);

    Snapshot<T> load() const;
    bool load_newer_than(uint64_t last_seen_sequence, Snapshot<T>& out) const;

    uint64_t sequence() const;
    CommStats stats() const;
};

} // namespace executor::comm
```

### 语义

- 读者只看到完整发布后的快照。
- `update(fn)` 在非当前读缓冲区上修改，完成后一次性发布。
- `load_newer_than()` 帮助读者避免重复消费旧状态。
- 初期目标是单写多读；多写场景建议先用 `MpscChannel` 汇聚到一个状态 owner。
- 对大型不可复制对象，后续可增加 `SnapshotPtr<T>`，内部使用 `std::shared_ptr<const T>`。

---

## P5：submit_after / when_all

项目已有 `TaskDependencyManager`，但用户仍缺一个不用手写任务 ID 和轮询的时序 API。

### API 草案

```cpp
namespace executor {

class TaskHandle {
public:
    std::string id() const;
    bool valid() const;
};

template<class F, class... Args>
auto Executor::submit_after(const TaskHandle& dependency,
                            F&& f,
                            Args&&... args)
    -> std::future<std::invoke_result_t<F, Args...>>;

template<class F, class... Args>
auto Executor::submit_after(const std::vector<TaskHandle>& dependencies,
                            F&& f,
                            Args&&... args)
    -> std::future<std::invoke_result_t<F, Args...>>;

TaskHandle Executor::when_all(std::vector<TaskHandle> dependencies);

} // namespace executor
```

### 语义

- `submit_after(A, f)` 表示 `f` 只在 A 完成后进入执行器。
- dependency 失败时的默认策略建议为“不执行 dependent task，并让 dependent future 得到异常”，避免用户消费无效中间状态。
- `when_all()` 返回一个逻辑 handle，可作为后续任务依赖。
- 内部可先复用 `TaskDependencyManager`；第一版可限制依赖对象来自同一个 `Executor` 实例。
- 后续可扩展 `when_any()`、取消传播和失败策略。

---

## P6：通信时序监控

每个通信组件至少提供本地 `stats()`。可选地，`Executor` 增加通信诊断聚合：

```cpp
namespace executor {

struct CommStatusSnapshot {
    std::vector<executor::comm::CommStats> components;
    uint64_t total_dropped = 0;
    uint64_t total_stale_reads = 0;
    uint64_t total_missed_phases = 0;
    uint64_t total_latency_high = 0;
};

class Executor {
public:
    void set_comm_event_callback(executor::comm::CommEventCallback callback);
    CommStatusSnapshot get_comm_status() const;
};

} // namespace executor
```

建议监控指标：

- `drop`：满队列拒绝、DropOldest、KeepLatest 覆盖。
- `latency`：消息从 publish/send 到 receive/drain 的时间。
- `stale`：读者重复读旧 sequence 或显式要求新值但没有新值。
- `missed phase`：等待者发现目标步骤已经错过。
- `producer lag`：生产 sequence 与消费 sequence 的差。
- `consumer lag`：队列深度、未消费消息数或 phase 差。

默认只累计数字；高频事件不默认写日志。超过阈值时才触发 `CommEventCallback`，避免诊断本身干扰实时行为。

---

## 示例

### 采集线程到规划线程

```cpp
executor::comm::MpscChannel<SensorFrame> frames({.capacity = 256});

// producer threads
if (!frames.try_send(read_sensor())) {
    auto stats = frames.stats();
    // stats.dropped_count 可用于告警或降采样。
}

// planner thread
SensorFrame frame;
while (frames.try_receive(frame)) {
    plan(frame);
}
```

### 配置线程到实时控制线程

```cpp
executor::comm::LatestMailbox<ControlConfig> config_box;

// config thread
config_box.publish(load_config());

// realtime cycle
uint64_t seen = 0;
ControlConfig cfg;
uint64_t seq = 0;
if (config_box.try_load_newer_than(seen, cfg, seq)) {
    seen = seq;
    apply_config(cfg);
}
```

### 初始化顺序控制

```cpp
executor::comm::PhaseGate gate("startup");

std::thread worker([&] {
    auto ready = gate.wait_for(2, std::chrono::seconds(3));
    if (ready) {
        run_worker();
    }
});

initialize_io();
gate.advance_to(1);
initialize_planner();
gate.advance_to(2);
```

### 状态快照

```cpp
executor::comm::DoubleBuffer<SystemState> state;

// writer
state.update([](SystemState& next) {
    next.position = read_position();
    next.velocity = read_velocity();
});

// readers
auto snapshot = state.load();
render(snapshot.value);
```

---

## 风险与待决问题

- `MpscChannel<T>` 对非平凡类型的第一版实现是否接受内部锁，需要用 benchmark 和 TSAN 压力测试确认。
- `DoubleBuffer<T>` 的 `load()` 是否要求 `T` 可复制；大型状态可能需要第二套 `SnapshotPtr<T>` API。
- `submit_after()` 的失败传播策略会影响用户预期，需在 API 文档中明确默认值并提供可选策略。
- 通信事件是否进入 `ExecutorFailureEvent` 需要谨慎。建议先保留独立 `CommEvent`，避免把正常背压误报为任务失败。
