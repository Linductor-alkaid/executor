# Facade 完整度与失败可观察性更新计划

本文档承接 README / API 文档中新的对外宣称：

> 自动调优可以安全回退；任务异常、提交拒绝、实时队列丢任务、超时等运行时状态必须通过 `future`、返回值、状态计数或监控统计保持可观察。用户可以选择不响应，但库不能让失败无迹可寻。

目标不是把所有失败都变成抛异常，也不是把 facade 做成静默万能层；目标是让普通用户优先面对 `Executor` 这一层完成常见工作流，并且在任务状态异常时有统一、低摩擦的观察入口。

---

## 当前基线

- [x] README / README_zh 已从“失败静默”改为“自动调优安全回退，任务失败可观察”。
- [x] docs/API.md 已更新 facade 哲学：默认即最优，失败可观察。
- [x] docs/MIGRATION.md 已补充失败可观察性约定。
- [x] 实时执行器 `push_task_ex()` 已返回入队结果，并通过 `dropped_task_count` 暴露丢任务。
- [x] 旧 `push_task()` 虽不返回结果，但失败仍累计到 `dropped_task_count`。
- [x] 普通异步任务异常尚未统一进入 facade 可见的失败通道。
- [ ] `submit_periodic()` 丢弃 future，周期任务异常可能无可见出口。
- [ ] `Executor` facade 尚无统一失败回调 / 最近错误 / 失败事件查询入口。
- [ ] 多个注册/初始化 API 仍只返回 `bool`，失败原因不可诊断。
- [ ] 实时任务推送仍要求用户先拿底层 `IRealtimeExecutor*`，facade 完整度不足。
- [ ] `wait_for_completion()` 超时返回不可区分，调用方无法知道是否真的完成。

---

## 设计原则

1. **Facade 优先**：新手路径应尽量只需要 `#include <executor/executor.hpp>` 和 `Executor` 方法；底层接口仍保留给高级用户。
2. **兼容优先**：不破坏现有 `submit()`、`push_task()`、`initialize()` 等签名；新增 `try_*`、`*_ex`、诊断结构或观察接口。
3. **失败不静默**：任务异常、提交拒绝、背压丢弃、软超时、等待超时必须至少落到一种可见渠道。
4. **自动调优不是任务失败**：CPU 探测、affinity、priority、mlock、timer slack 等平台调优失败可以安全回退，但应在诊断/调试层可解释，不应污染任务失败计数。
5. **状态可监控**：所有长期运行或后台行为都要能从状态结构、统计结构或事件回调中被观察。

---

## 阶段 1：统一失败事件模型

### 任务

- [x] 新增 `FailureEvent` / `ExecutorFailureEvent` 类型，建议字段：
  - `FailureKind kind`：`TaskException`、`SubmitRejected`、`TaskTimeout`、`RealtimeDrop`、`GpuFailure`、`WaitTimeout`、`TuningFallback`。
  - `std::string executor_name`
  - `std::string task_id`
  - `std::string message`
  - `std::exception_ptr exception`
  - `std::chrono::steady_clock::time_point timestamp`
- [x] 新增 `ExecutorFailureStatus` 或扩展现有状态查询，至少包含累计计数：
  - `task_exception_count`
  - `submit_rejected_count`
  - `timeout_count`
  - `realtime_drop_count`
  - `wait_timeout_count`
  - `tuning_fallback_count`
- [x] 在 `Executor` facade 暴露：
  - `set_failure_callback(...)`
  - `get_failure_status()`
  - `get_recent_failures(size_t max_count)`
  - `clear_recent_failures()`
- [x] 为 failure ring buffer 设置固定容量，避免长期运行进程无限增长；容量可配置或给出合理默认。

### 验收

- [ ] 用户只通过 `Executor` 就能订阅任务失败与提交拒绝。
- [x] 未设置 callback 时，失败仍会累计到状态/最近事件，而不是消失。
- [x] callback 自身抛异常不会杀死 worker 或后台线程。

---

## 阶段 2：修正普通异步任务失败可见性

### 问题

当前 `IAsyncExecutor::submit()` 在 wrapper 内捕获用户异常并写入 `promise`。底层 `ThreadPool::execute_task()` 看到 wrapper 正常返回，可能将任务统计为成功；如果用户不 `future.get()`，失败不可见。

### 任务

- [x] 调整 `submit()` / `submit_priority()` / `submit_batch()` 的包装方式，使用户任务异常同时：
  - 写入对应 `future`
  - 触发统一 failure event
  - 更新失败统计或专用 facade 失败计数
- [x] 明确 `AsyncExecutorStatus::failed_tasks` 的语义：
  - 方案 A：计入用户任务异常。
  - 方案 B：保持底层执行失败语义，新增 facade 级 `task_exception_count`。
  - 推荐方案 A，但需评估现有测试与 `wait_for_completion()` 不变量。
- [x] 对 `submit_batch_no_future()` 增加强制可见性：
  - 用户任务异常没有 future 可承载，必须进入 failure event / status counter。
- [x] 对提交被拒绝的场景计数：
  - shutdown 后提交
  - 空 batch
  - executor 未可用

### 验收

- [x] 单个 `submit([]{ throw ...; })` 即使不调用 `future.get()`，状态中也能观察到失败。
- [x] `submit_batch_no_future()` 内部任务抛异常时，failure callback 被调用且计数递增。
- [x] shutdown 后提交被拒绝时，调用方通过 future/返回值/事件至少一种方式可见。

---

## 阶段 3：修正延迟与周期任务状态

### 问题

`submit_periodic()` 当前周期回调只调用 `executor->submit(task)`，但丢弃返回的 `future`；周期任务异常可能只进入被丢弃的 future。延迟任务也需要确认提交失败和执行失败均可观察。

### 任务

- [ ] 为 `submit_periodic()` 增加内部任务包装：
  - 用户任务异常触发 failure event。
  - 每次周期执行失败累计到周期任务状态。
  - 可选：连续失败阈值、最后一次失败时间、最后错误消息。
- [ ] 新增周期任务状态查询：
  - `get_periodic_task_status(task_id)`
  - `get_all_periodic_task_status()`
- [ ] 定义周期任务失败后的默认行为：
  - 默认继续调度，但记录失败。
  - 可选配置：失败后停止、失败后退避、连续 N 次失败后停止。
- [ ] `submit_delayed()` 到期提交失败时，应设置 promise 异常并触发 failure event。
- [ ] 定时器线程中 facade/manager 已 shutdown 时，延迟/周期任务不可悄悄丢失，应写入 failure event 或设置 future 异常。

### 验收

- [ ] 周期任务抛异常但用户没有 future 时，仍能从 failure status 或 periodic status 观察到。
- [ ] `cancel_task()` 对不存在 ID 的失败保持返回值可见，并可选记录诊断事件。
- [ ] shutdown 期间未执行的 delayed/periodic 任务处理策略有文档说明和测试覆盖。

---

## 阶段 4：补全实时任务 facade

### 问题

实时背压可见性在底层已有基础，但普通用户仍需 `get_realtime_executor()` 后调用底层 `push_task_ex()`，这削弱 facade 完整度。

### 任务

- [ ] 在 `Executor` facade 增加：
  - `bool push_realtime_task(const std::string& name, std::function<void()> task)`
  - `bool try_push_realtime_task(...)`
  - 可选 `submit_realtime_task(...)` 命名别名，需避免与普通 `submit` 混淆。
- [ ] facade push 失败时统一记录 `RealtimeDrop` 或 `SubmitRejected` 事件：
  - 实时 executor 不存在
  - 未启动
  - 空任务
  - 队列满
  - 对象池耗尽
- [ ] 保留 `get_realtime_executor()` 作为高级逃生口，但 README/API 示例优先使用 facade push。
- [ ] `RealtimeExecutorStatus` 可选新增：
  - `rejected_not_running_count`
  - `rejected_empty_task_count`
  - `pool_exhausted_count`
  - `queue_full_count`
  以便把 `dropped_task_count` 拆出原因。

### 验收

- [ ] 不接触 `IRealtimeExecutor*`，用户即可注册、启动、推送、观察实时任务。
- [ ] facade push 失败同时通过返回值和状态计数可见。
- [ ] 旧 `push_task()` 仍兼容，但文档标注为兼容入口，推荐新代码使用 facade/`push_task_ex()`。

---

## 阶段 5：可诊断的初始化与注册 API

### 问题

`initialize()`、`register_realtime_task()`、`register_gpu_executor()` 等 API 返回 `bool`，失败原因丢失。

### 任务

- [ ] 新增轻量 Result 类型，例如：
  - `ExecutorResult`
  - `ExecutorErrorCode`
  - `std::string message`
- [ ] 新增非破坏性 API：
  - `initialize_ex(config) -> ExecutorResult`
  - `register_realtime_task_ex(name, config) -> ExecutorResult`
  - `start_realtime_task_ex(name) -> ExecutorResult`
  - `register_gpu_executor_ex(name, config) -> ExecutorResult`
- [ ] 旧 `bool` API 保留，并委托到 `_ex` 后只返回 `ok`。
- [ ] 常见错误码：
  - `AlreadyInitialized`
  - `AlreadyShutdown`
  - `InvalidConfig`
  - `DuplicateName`
  - `NotFound`
  - `BackendUnavailable`
  - `StartFailed`
  - `PermissionDenied`
- [ ] `_ex` 失败同时写入 failure/diagnostic event，但不把配置错误混为任务失败。

### 验收

- [ ] 用户看到 `false` 时，可以通过 `_ex` 或最近诊断知道原因。
- [ ] GPU 未编译/运行时不可用能清晰区分。
- [ ] RT 配置无效能返回具体字段原因。

---

## 阶段 6：等待与生命周期可观察

### 问题

`wait_for_completion()` 内部最多等待 300 秒，但返回 `void`，超时不可区分。

### 任务

- [ ] 新增：
  - `bool wait_for_completion_for(std::chrono::duration<...>)`
  - `WaitResult wait_for_completion_ex(timeout)`
  - `bool is_idle()` 或 `get_completion_status()`
- [ ] 旧 `wait_for_completion()` 保持阻塞语义，文档说明其行为。
- [ ] 等待超时记录 `WaitTimeout` 事件并累计计数。
- [ ] shutdown 路径中如果等待超时，应有可观察诊断，而不是继续假装全部完成。

### 验收

- [ ] 测试能明确区分完成返回和超时返回。
- [ ] wait timeout 后状态能说明仍有 active / queued / pending 任务。

---

## 阶段 7：通信/并发辅助 facade

### 背景

`tests/harness/test_comm_facade_usage.cpp` 已有 disabled 用例，说明项目已经识别到常见通信模式仍缺 facade。

### 任务

- [ ] 设计 `executor::comm` facade API：
  - `MpscChannel<T>`
  - `LatestMailbox<T>`
  - `PhaseGate`
  - `DoubleBuffer<T>`
  - `RealtimeChannel<T>`
- [ ] 每个通信组件必须定义失败可见性：
  - push 失败返回值
  - drop/overwrite 计数
  - close/shutdown 状态
  - 高水位或容量状态
- [ ] 启用并补全 `test_comm_facade_usage.cpp` 中 disabled 用例。
- [ ] 在 README/API 加入最小示例，避免用户直接拼底层队列、锁和生命周期。

### 验收

- [ ] 典型 producer/consumer、latest-state、phase handoff、realtime drain 场景无需用户直接操作底层无锁队列。
- [ ] 所有丢弃、覆盖、关闭后的提交均可观察。

---

## 阶段 8：测试矩阵

- [ ] `test_executor_failure_observability.cpp`
  - [ ] `submit` 用户异常可观察。
  - [ ] `submit_priority` 用户异常可观察。
  - [ ] `submit_batch` 部分任务异常可观察。
  - [ ] `submit_batch_no_future` 用户异常可观察。
  - [ ] shutdown 后提交拒绝可观察。
- [ ] `test_periodic_failure_observability.cpp`
  - [ ] 周期任务抛异常后无 future 也可观察。
  - [ ] 连续失败计数和最后错误可查询。
  - [ ] cancel 不存在任务返回 false 且可诊断。
- [ ] `test_realtime_facade_push.cpp`
  - [ ] facade 推送成功。
  - [ ] 不存在 RT executor 推送失败可见。
  - [ ] 未启动 / 已停止推送失败可见。
  - [ ] 队列满 / 池耗尽推送失败可见。
- [ ] `test_executor_result_diagnostics.cpp`
  - [ ] 初始化重复。
  - [ ] shutdown 后初始化/提交。
  - [ ] RT 无效配置。
  - [ ] GPU 后端不可用。
- [ ] `test_wait_completion_result.cpp`
  - [ ] wait 完成返回成功。
  - [ ] wait 超时返回 timeout，并保留 pending 状态。

---

## 阶段 9：文档与示例同步

- [ ] README / README_zh
  - [ ] 示例优先使用 `Executor` facade 完成普通任务、实时任务、状态观察。
  - [ ] 明确 `future.get()` 是获取返回值/异常的方式，但状态计数也能监控失败趋势。
- [ ] docs/API.md
  - [ ] 新增 failure event / result / wait result API 文档。
  - [ ] 标注旧 bool API 与新 `_ex` API 的关系。
  - [ ] 修正 `AsyncExecutorStatus::failed_tasks` 语义。
- [ ] docs/MIGRATION.md
  - [ ] 增加从旧接口迁移到 facade push、`*_ex` result、failure callback 的建议。
- [ ] examples
  - [ ] 新增 `examples/failure_observability.cpp`。
  - [ ] 新增或更新 `examples/realtime_can.cpp`，优先展示 facade push 和 drop 监控。
  - [ ] 新增 `examples/periodic_monitoring.cpp`。

---

## 推荐实施顺序

1. 阶段 1：统一失败事件模型。
2. 阶段 2：普通异步任务失败可见。
3. 阶段 3：延迟/周期任务失败可见。
4. 阶段 4：实时任务 facade push。
5. 阶段 5：`*_ex` 诊断 API。
6. 阶段 6：等待与生命周期可观察。
7. 阶段 8：补齐回归测试矩阵。
8. 阶段 9：同步文档与示例。
9. 阶段 7：通信/并发辅助 facade，可作为后续较大版本推进。

这个顺序先保证新宣称不落空，再补用户体验层面的完整 facade。
