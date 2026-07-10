# 迁移指南

本文档说明不同 executor 版本之间的迁移方式。若你从旧版本升级，请按对应版本节的说明操作。

---

## 从 0.2.3 升级到 0.3.0

0.3.0 重点新增通信与并发辅助 facade，把常见跨线程通信、实时周期消费、快照读取和任务时序控制提升到 `Executor` / `executor::comm` 公开层。已有手写同步代码可以继续工作；新代码建议优先迁移到下列组件，以获得统一生命周期、背压和诊断统计。

### 推荐迁移到通信与并发辅助 facade

阶段 7 新增 `executor::comm`，用于替代常见的手写共享变量、mutex、condition_variable、底层无锁队列和 promise/future 链。综合示例见 [examples/comm_robot_pipeline.cpp](../examples/comm_robot_pipeline.cpp)，它模拟传感器采集、规划、实时控制和状态监控流水线。

迁移建议：

- 采集线程到规划线程的有界数据流：从“共享 vector + mutex”或直接使用底层队列，迁移到 `MpscChannel<T>` / `SpscChannel<T>`。满队列、关闭、超时通过返回值和 `CommStats` 可见。
- 配置线程到实时控制线程的“只要最新值”：从共享配置对象和原子 flag，迁移到 `LatestMailbox<T>`。实时线程用 sequence 避免重复消费旧配置。
- 实时周期内处理有限条命令：从实时线程里阻塞等待队列，迁移到 `RealtimeChannel<T>::drain_for_cycle()`，并设置每周期预算。
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
