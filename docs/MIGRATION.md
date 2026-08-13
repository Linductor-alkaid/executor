# 迁移指南

本文档说明不同 executor 版本之间的迁移方式。若你从旧版本升级，请按对应版本节的说明操作。

---

## Unreleased：通信同步核心无锁化

本节适用于当前 `master` 开发快照。
现有通信类型与主要调用方式保持兼容，但内部同步和实时使用边界发生了以下变化：

- `MpscChannel<T>` / `RealtimeChannel<T>` 改为构造期预分配的有界 MPSC 节点池，数据路径不再使用
  mutex；仍要求一个逻辑消费者。
- `LatestMailbox<T>` / 未绑定的 `DoubleBuffer<T>` 改为四个固定 reader-pin 快照槽，复制非平凡
  `T` 时不依赖存在 data race 的 seqlock。`try_load()` 最多检查四个槽；`try_publish()` 是系统级
  lock-free，但竞争 CAS 可重试，不能声明为单次调用有界或 wait-free。
- `PhaseGate` / `Sequencer` 使用原子状态核心。带 timeout 的 wait API 与保证成功的兼容 API 仍会
  spin/yield，只适合普通控制线程。
- 新增 `is_synchronization_lock_free()`；兼容的 `is_lock_free()` 返回相同结果。所需原子不是
  lock-free 时，组件在构造时拒绝运行，而不是静默退化到库内部锁。
- channel 的 `close()` 只关闭新的 producer 准入；关闭前已经准入的 producer 仍可完成发布。
  需要判断所有已接受消息均已排空时使用 `is_drained()`，不要只把某次 `empty()` 当作终态。
- `Topic<T>` 是明确例外：subscription registry 和 `publish()` fan-out 快照仍使用 mutex 与动态
  分配，整体不是实时或 lock-free 路径。

上述“同步无锁”和“内部固定存储”不覆盖 `T` 的复制/移动/析构、时钟、诊断 callback、缺页、
调用方分配或 OS 调度。迁移实时路径时应使用非等待 API、关闭高频 callback，并在目标硬件上验证
完整调用链的最坏耗时与页面驻留情况。

---

## 从 0.3.0 升级到 0.3.1：统一 Facade 与自动路由

0.3.1 除实时进程内存锁配置项外是向后兼容扩展。`submit()`、`submit_gpu()`、四参数 legacy `submit_auto(TaskCharacteristics, name, kernel, config)`、实时和 Blocking I/O 的既有入口及返回类型均保持不变。

### 破坏性变更：实时进程内存锁配置

`RealtimeThreadConfig::enable_memory_lock` 已更名为 `enable_process_memory_lock`，并改为默认关闭，以纠正 Linux `mlockall` 的进程级语义。若需要进程级内存锁，改用 `enable_process_memory_lock = true`，并检查 `RealtimeExecutorStatus::process_memory_lock_applied` 与 `process_memory_lock_errno`；仅在完成整个进程的 memlock 内存预算评估后显式启用。

### 推荐迁移路径

- 普通短任务可从 `submit()` 逐步迁移到 `submit_auto(lambda)`；它默认只选择异步线程池，并通过 `get_last_routing_decision()` 提供可解释的默认决策。
- 需要两条独立实现时使用 `cpu_gpu_task(cpu, gpu)`。默认 `FallbackPolicy::NoFallback`：GPU 不可提交会使 future 进入异常；只有显式 `AllowCpu` 才会回退 CPU。带返回值的 CPU/GPU 自动任务尚未提供。
- 已验证的 MPSC 单消费者路径使用 `dispatch_auto()` + `LowLatency`，周期实时工作使用 `dispatch_auto()` + `RealtimeQueue`。两者必须指定已启动后端，返回的 `DispatchResult` 仅表示接收，不表示完成。
- 长期可中断 I/O 推荐改用 `start_worker(BlockingWorkerSpec{...})`。`WorkerHandle` 统一启动结果、状态查询和停止，但不改变 `wakeup()`、stop token、启动超时或退出原因契约。
- 通过 `get_executor_capabilities()` 枚举所有后端状态；它是预检快照，不能替代处理实际投递竞争和背压。
- 低频健康检查、等待/关闭超时现场和故障支持包可新增 `get_snapshot()` 或 `get_snapshot_text()`；它们是只读 best-effort 诊断，不触发默认异步执行器懒初始化，也不替代提交 reservation、任务结果或后端专属状态 API。可运行的最小示例见 `examples/lifecycle_snapshot.cpp`。

### 兼容与后续版本

- legacy CPU/GPU `submit_auto` 在整个 `0.3.x`（包括 0.3.1）保持现有“GPU 未就绪即失败、无隐式 CPU 回退”的行为，暂不添加编译期弃用标记。
- 后续允许破坏性变更的主版本才会进入 legacy overload 的弃用/移除窗口；`CpuGpuTask<T>` 的返回值支持和 `ExecutionReport<T>` 也仅在该窗口评估。
- 自动路由不能证明 callable 的实时安全、线程安全、GPU 内存所有权或 I/O 可中断性；这些仍由应用设计、部署和测试。

---

## 0.3.0：Blocking I/O worker

`BlockingIoExecutor` 是向后兼容的库级扩展，用于替代由调用方手写、长期阻塞且需要有序停止的 `std::thread` / `std::jthread`。它不提供协议、设备或业务流程迁移：调用方保留自己的 worker 实现、消息数据面和安全策略。

### 推荐迁移路径

1. 将长期循环封装为 `IBlockingIoWorker`：把主体放入 `run(std::stop_token)`，实现不抛异常且可重复调用的 `wakeup()`。
2. 保证停止可达：`wakeup()` 要直接解除等待；不能直接唤醒时使用有限 timeout，并在每次返回后检查 `stop_token`。不要依赖 stop token 自动中断外部库调用。
3. 用 `register_blocking_io_worker_ex()` 注册，再用 `start_blocking_io_worker_ex()` 启动；将 `ExecutorResult` 的拒绝原因写入调用方日志或诊断。
4. 用 `stop_blocking_io_worker()` 或 `Executor::shutdown()` 收敛生命周期。不要 detach worker，也不要在 `shutdown(false)` 时假定 I/O worker 会继续运行。

### 不适用的迁移

- 有限、可排队的工作仍应使用线程池；不要为短任务创建 I/O worker。
- 固定周期控制回调仍应使用 `RealtimeThreadExecutor`；不要在 `cycle_callback` 内等待长期 I/O。
- 协议解析、设备重连、数据新鲜度、命令语义和安全动作不属于 `executor`，由调用方独立设计和验证。

---

## 从 0.2.3 升级到 0.3.0

0.3.0 重点新增通信与并发辅助 facade，把常见跨线程通信、实时周期消费、快照读取和任务时序控制提升到 `Executor` / `executor::comm` 公开层。已有手写同步代码可以继续工作；新代码建议优先迁移到下列组件，以获得统一生命周期、背压和诊断统计。

### 推荐迁移到通信与并发辅助 facade

阶段 7 新增 `executor::comm`，用于替代常见的手写共享变量、mutex、condition_variable、底层无锁队列和 promise/future 链。综合示例见 [examples/comm_robot_pipeline.cpp](../examples/comm_robot_pipeline.cpp)，它模拟传感器采集、规划、实时控制和状态监控流水线。

迁移建议：

- 采集线程到规划线程的有界数据流：从“共享 vector + mutex”或直接使用底层队列，迁移到 `MpscChannel<T>` / `SpscChannel<T>`。满队列、关闭、超时通过返回值和 `CommStats` 可见。
- 配置线程到实时控制线程的“只要最新值”：从共享配置对象和原子 flag，迁移到 `LatestMailbox<T>`。实时线程用 sequence 避免重复消费旧配置。
- 实时周期内处理有限条命令：从实时线程里阻塞等待队列，迁移到 `RealtimeChannel<T>::drain_for_cycle()`，并设置每周期预算。该 facade 当前提供有界、非等待的调用语义，但其内部仍使用 mutex；硬实时或无锁要求需使用经验证的专用实现。
- 监控线程读取系统状态：从共享 mutable state，迁移到 `DoubleBuffer<T>` / `Snapshot<T>`，读者只看到完整发布后的快照。
- 启动、初始化、阶段顺序：从手写 condition variable predicate，迁移到 `PhaseGate` / `Sequencer`。
- 任务级依赖：从手写 promise/future 链或轮询 `TaskDependencyManager`，迁移到 `TaskHandle`、`submit_with_handle()`、`submit_after()` 和 `when_all()`。
- 诊断：每个通信组件都有 `stats()`；低频事件可通过 `set_event_callback()` 接入日志或监控。通信事件默认不计入 `ExecutorFailureStatus`，需要统一上报时由业务在 callback 中桥接。

### 选择指南

| 旧写法/需求 | 推荐 facade |
|-------------|-------------|
| producer/consumer 传递每条数据 | `MpscChannel<T>` / `SpscChannel<T>` |
| 控制配置只关心最新值 | `LatestMailbox<T>` |
| 实时周期内 drain 有限命令 | `RealtimeChannel<T>` |
| 多读者读取完整系统状态 | `DoubleBuffer<T>` / `Snapshot<T>` |
| 启动顺序、阶段推进 | `PhaseGate` |
| 精确 ticket 顺序 | `Sequencer` |
| 任务完成后再执行后续任务 | `TaskHandle` + `submit_after()` / `when_all()` |

### 破坏性变更

**无。** 0.3.0 保持 0.2.3 公开 API 兼容；通信 facade、任务图 facade、统计和场景示例均为向后兼容扩展。旧的共享变量、手写锁、底层队列和 promise/future 链仍可继续使用，但新代码推荐逐步迁移到 `executor::comm` 和 `Executor` facade。

---

## 从 0.2.2 升级到 0.2.3

0.2.3 是向后兼容版本，重点补齐 `Executor` facade 的失败可观察性、可诊断结果和等待生命周期状态。已有代码可以继续使用旧 `bool` API；新代码建议迁移到下列可诊断入口。

### 推荐迁移到可观察 facade

- 初始化、实时注册/启动、GPU 注册建议从旧 `bool` API 迁移到 `initialize_ex()`、`register_realtime_task_ex()`、`start_realtime_task_ex()`、`register_gpu_executor_ex()`，失败时读取 `ExecutorResult::error_code` 和 `message`。
- 普通任务仍通过 `future.get()` 获取返回值和重新抛出的任务异常；同时可通过 `Executor::set_failure_callback()`、`get_failure_status()`、`get_recent_failures()` 监控未被调用方立即消费的失败趋势。
- 实时任务推送建议从 `auto* rt = get_realtime_executor(...); rt->push_task(...)` 迁移到 `Executor::push_realtime_task()` / `try_push_realtime_task()`，以便不存在、未启动、队列满、对象池耗尽等失败同时通过返回值、failure event 和 `RealtimeExecutorStatus` 计数可见。
- 等待任务完成时，新代码优先使用 `wait_for_completion_for(timeout)` 或 `wait_for_completion_ex(timeout)`；后者在超时时返回 `WaitResult::status.pending_tasks`、`active_tasks`、`queued_tasks`，并累计 `wait_timeout_count`。
- 旧 API 均保持兼容；迁移的目的不是改变执行模型，而是让已有失败路径带上可诊断结果和统一监控入口。

### 破坏性变更

**无。** 0.2.3 保持 0.2.2 公开 API 兼容；新增 result、failure callback、facade push 和 wait result API 均为向后兼容扩展。

---

## 从 0.2.1 升级到 0.2.2

0.2.2 是向后兼容版本，**没有破坏性变更**。已有 0.2.1 代码可以直接重新编译使用；需要注意的是，部分 facade 默认值改为"默认即最优"，零配置用户会自动获得更积极的线程池与实时线程配置。

### 默认值变化：默认即最优 Facade

- `RealtimeThreadConfig.enable_memory_lock` 默认 `true`：Linux 下尽力尝试 `mlockall`，降低分页导致的实时抖动；平台不支持或权限不足时安全回退，不改变任务状态。
- `RealtimeThreadConfig.timer_slack_ns` 默认 `1`：Linux 下将 timer slack 调到 1 ns；设置为 `0` 表示显式 opt-out。
- `ThreadPoolConfig.min_threads` / `max_threads` 默认 `0`：作为 sentinel，初始化时自动探测 `hardware_concurrency()`；探测失败退到安全默认。
- `ThreadPoolConfig.enable_work_stealing` 默认 `true`：`max_threads == 1` 时自动关闭。
- `cpu_affinity` 为空时自动分配：线程池使用 [0..hw-1]；实时线程空 affinity 时通过 `g_next_rt_cpu_hint` 在当前允许 CPU 集合内 round-robin 自动选择，可用 CPU 数量 <= 1 时不设置亲和性；显式配置始终保留。

### 新增 API

- `IRealtimeExecutor::push_task_ex(std::function<void()>) -> bool`：背压可见版本的实时任务推送 API。返回 `true` 表示成功入队，返回 `false` 表示任务因空任务、队列满或对象池耗尽被丢弃；`push_task()` 的 `void` 签名保留以保证兼容。
- `RealtimeExecutorStatus` 新增背压字段：`dropped_task_count`、`failed_pushes`、`peak_queue_size`、`queue_capacity`，用于观察实时任务队列是否出现丢任务。
- `task_timeout_ms` 软超时：当任务开始执行前发现排队时间 `elapsed >= timeout` 时跳过任务并增加 `timeout_count`。执行中的任务不会被强制中断。

### 失败可观察性约定

Facade 的默认调优可以安全回退，但运行时任务状态不能静默丢失。任务异常、提交拒绝、实时队列丢任务和超时应通过 `future`、返回值、状态计数或监控统计暴露；调用方可以选择不响应这些信号，但库不应让失败无迹可寻。

### 破坏性变更

**无。** 0.2.2 保持 0.2.1 公开 API 兼容；新增字段、默认值和 API 均为向后兼容扩展。

### 升级检查清单

- [ ] 如果业务不希望库自动锁内存或调整 timer slack，显式设置 `enable_memory_lock = false` 或 `timer_slack_ns = 0`。
- [ ] 如果线程池线程数或 CPU 亲和性必须固定，显式设置 `min_threads`、`max_threads` 与 `cpu_affinity`，不要依赖默认 sentinel。
- [ ] 实时任务推送路径建议从 `push_task()` 迁移到 `push_task_ex()`，并监控 `dropped_task_count`。
- [ ] 使用 `task_timeout_ms` 时确认它是软超时：长任务需要在任务内部自行检查取消条件。
- [ ] 打包或安装 GPU 版本时确认 CUDA/OpenCL 为可选运行时依赖；无 GPU 或无 CUDA 驱动时会运行时降级。

---

## 从无到有（首次使用）

**0.1.0** 为首个发布版本，无需迁移。直接参考 [README.md](../README.md)、[docs/API.md](API.md) 与 [docs/BUILD.md](BUILD.md) 集成即可。

---

变更摘要见 [CHANGELOG.md](../CHANGELOG.md)。
