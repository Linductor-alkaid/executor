# Changelog

本文档记录 executor 项目的版本变更。版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

---

## [Unreleased]

任务协作取消与定时句柄（客户端反馈台账 P1-2/P1-3 收敛，设计见
`docs/design/task_cancellation_and_timers.md`）：facade 新增任务级协作取消与可取消、
可重排的定时句柄；取消是协作请求而非抢占，取消计数进入独立生命周期字段而非
failure 体系。

### 新增

- **任务级协作取消（C1）**：新增 `include/executor/task_cancellation.hpp`
  （`TaskCancelled` / `TaskCancellationReason` / `TaskCancellationResponse` /
  `CancellationStatus`）。`submit_cancellable` / `submit_cancellable_priority` /
  `submit_cancellable_after` 把 `StopToken` 注入为 callable 首参数；
  `request_task_cancel(const TaskHandle&)` 提供排队取消与运行中协作请求，
  幂等且不写 failure 事件；`submit_with_handle` / `submit_after_with_handle`
  天然获得排队取消能力；取消 registry 有界（默认 65536，
  `set_cancellation_registry_capacity()` 可调），容量耗尽明确拒绝。
- **定时句柄（T1）**：新增 `include/executor/timer.hpp`（`TimerHandle` /
  `ScopedTimerHandle` / `TimerStatus` / `TimerOperationResult`）。
  `submit_delayed_with_handle` / `submit_delayed_cancellable_with_handle` /
  `submit_periodic_with_handle` / `submit_periodic_cancellable_with_handle`
  提供取消、重排与状态查询；内部定时器改为 registry + generation heap，
  变更 1ms 内可见的 steady 时钟分片等待（5ms 延迟任务的平均到期误差从约 5.5ms
  降至约 0.9ms；不用 condition_variable 定时等待，规避 gcc-11 libtsan 对
  pthread_cond_clockwait 未拦截导致的 double-lock 误报），
  stale entry 有界压缩；终态 record 只保留有界元数据。
- **定时器互操作指南（S1）**：新增 `docs/external_event_loop_interop.md` 与可编译
  示例 `examples/event_loop_interop.cpp`（托管事件循环、strand 延续盲区纪律、
  PhaseGate 批次收尾）；中英文网站同步上线指南页。
- **监控扩展**：`ExecutorSnapshot` schema 2 → 3，新增 `cancellation`
  （`CancellationStatus`）与 `timers`（`TimerStatusSummary`）独立字段与快照文本行；
  `Executor::get_cancellation_status()` / `get_timer_status_summary()` 查询入口。
- **教程与测试**：新增教程 `examples/tutorial/13_cancellation_and_timers.cpp`、
  `tests/test_task_cancellation.cpp`、`tests/test_timer_handle.cpp`、文档一致性测试
  `tests/test_api_doc_cancellation_fields.cpp`，以及整库
  `EXECUTOR_STOP_TOKEN_FORCE_FALLBACK` 强制实例化的
  `tests/test_task_cancellation_fallback.cpp`。
- **comm 指引（G1 轻量项）**：中英文"如何选择通信组件"指南新增
  "什么时候允许裸回调"一节，明确裸 `std::function` 回调的适用边界。

### 变更

- shutdown 清理未到期 delayed 任务：future 异常由
  `std::runtime_error("Timer stopped...")` + `SubmitRejected` 事件改为
  `TaskCancelled(Shutdown)`，不再记录 failure 事件；可观察性转移到定时计数。
- `ExecutorSnapshot::schema_version` 2 → 3（纯新增字段）；解析快照文本的下游
  工具需按新 schema 更新（迁移说明见 `docs/MIGRATION.md`）。

---

## Android 适配一期

Android 适配一期：核心库可在 NDK 工具链下以 CPU-only 配置交叉编译为静态库/共享库，
并纳入官方模拟器与真实 ARM64 runner 的验证流程。Android 上的线程优先级、CPU 亲和性、
`mlockall` 与 timer slack 均保持 best-effort，不承诺硬实时；CUDA/OpenCL 不进入一期。

### 新增

- **Android CPU-only 构建支持**：新增 `if(ANDROID)` CMake 平台分支，bionic 下不再错误
  链接 `librt`，也不导出 `libatomic`；Android 构建默认关闭 GPU/CUDA，用户仍可显式覆盖。
- **便携 StopToken/JThread 兼容层**：新增 `include/executor/stop_token.hpp`。桌面平台
  `executor::StopToken` 是 `std::stop_token` 别名，保持既有 override 源码与 ABI 兼容；
  Android libc++ 未启用 jthread 时使用自有 `StopSource` / `StopToken` / `detail::JThread`。
- **Android 线程与 affinity 默认值**：bionic 下使用 `sched_setaffinity` /
  `sched_getaffinity`；默认线程池上限为 4，自动 affinity 来自 cgroup 允许 cpuset；
  短周期实时线程不再自动申请 `SCHED_FIFO`。
- **Android 构建与设备脚本**：新增 `scripts/build_android.sh`、
  `scripts/run_android_tests.sh`、`scripts/capture_android_device_info.sh`，以及
  `tests/android_smoke.cpp` 等无 GTest standalone 测试。
- **Android CI**：新增 NDK r26c / r28b 交叉编译 workflow；新增手动触发的
  `arm64-concurrency` workflow，覆盖 4 核、单核 pinned、ASan/UBSan 和可配置 MPSC soak。
- **Android 打包与平台文档**：新增 `docs/PACKAGE_ANDROID.md`，覆盖 NDK CMake、AGP、
  `c++_shared` 打包、JNI shutdown 生命周期与 Prefab/AAR 模板；中英文网站首页、构建页
  与平台部署核对页同步补充 Android CPU-only 能力边界。

### 修复与改进

- 修复 `test_multithread_mpsc` 在慢速 ARM64 模拟环境下消费者过早退出导致误报的测试逻辑。
- Android 下实时调优路径统一为 best-effort：priority / affinity / mlock / timer slack
  失败只写入状态字段，不改变任务接受结果。
- 为 Android 平台裁剪 NDK clang 不支持的 warning 选项，并守卫仅适用于 desktop Linux
  的 `/proc` 测试。
- **修复 P-260816-001**：`ExecutorManager::shutdown()` 不再在持有
  `default_async_mutex_` 时执行默认执行器的阻塞排空（`stop(wait_for_tasks)` /
  `wait_for_completion()` 含 worker join）。此前池内任务在排空期间再入
  `submit()` / 状态查询等持锁读路径会与 shutdown 互相等待形成自死锁；现在改为
  锁内置闩并快照执行器、锁外排空，置闩后读路径立即走拒绝分支，与 ThreadPool
  自身"先停止接收新任务，再等待已接受任务完成"的关停顺序对齐。并发 shutdown
  调用者通过新增的条件变量等待排空完成，保持"第二个调用者等第一个排空结束后
  才返回"的旧语义。新增回归测试 `tests/test_shutdown_drain_reentrancy.cpp`
  （旧实现在该测试下确定性死锁）。
- **修复 P-260816-002（H1）**：`Executor` 定时器线程启停竞态。旧实现先原子置位
  `timer_running_` 再创建线程并给 `timer_thread_` 赋值，并发 `shutdown` 会在成员尚未
  赋值时读取它（数据竞争 UB）并跳过 join；随后赋值出的 joinable 线程成员在析构时触发
  `std::terminate`，竞态窗口内提交的延迟任务 future 永久悬挂。现在：`timer_thread_` /
  `timer_state_` / 测试工厂由 `timer_thread_mutex_` 保护，赋值完成后才置位运行标志；
  每代线程持有独立的停止标志，停止只对本代置位（join 期间并发重启不会复活旧线程，
  join 必定返回）；join 在锁外执行。`submit_delayed` 在入队临界区内检查停止位，
  消除"入队后无人处理"的悬挂 future；`set_timer_thread_factory_for_test` 同步化。
  新增回归测试 `tests/test_timer_thread_lifecycle_race.cpp`（旧实现下延迟任务
  future 永久悬挂 / `std::terminate`，测试确定性命中）。

### 验证

- 官方 Android 模拟器（API 30 x86_64，KVM）：6/6 standalone 测试通过。
- qemu-user + NDK bionic 静态 ARM64：6/6 测试通过。
- GitHub ARM64 runner（Neoverse-N2，4 核）：6/6、单核 pinned、ASan/UBSan、600 秒
  MPSC soak 均通过；结果见 `docs/performance/android_a3_validation.md`。
- big.LITTLE Android 真机验证已登记为发布前 gate，正式版本不得在未完成该项时宣称
  已在 big.LITTLE 设备验证。
- P-260816-001 修复验证：树内 Debug 构建 105/105 ctest 通过（除 benchmark 标签外）；
  关停/生命周期相关测试（`test_shutdown_drain_reentrancy`、`test_concurrent_stop_submit`、
  `test_thread_pool_self_shutdown` 等）在 ThreadSanitizer 下无警告；该回归测试已加入
  CI TSan 任务清单。
- P-260816-002 修复验证：树内 Debug 构建 106/106 ctest 通过（除 benchmark 标签外）；
  定时器相关测试（`test_timer_thread_lifecycle_race`、`test_periodic_failure_observability`、
  `test_realtime_timer_period_race`、`test_timer_period_guard`、`test_executor_facade`）在
  ThreadSanitizer 下无警告；该回归测试同样加入 CI TSan 任务清单。

---

## [0.4.0] - 2026-08-13

0.4.0 聚焦通信与并发执行路径的确定性边界：核心通信组件采用构造期固定存储和原子同步，新增进程内 Topic 扇出、LET 阶段通信、实时分配诊断及延迟分位数观测；任务图句柄保留和线程池真实扩缩容也获得明确的容量与并发语义。既有主要公开调用方式保持兼容，但“同步无锁”仅描述组件内部原子与固定存储，完整实时性仍须由调用方在目标环境验证。

### 新增

- **进程内 Topic / Subscription**：新增 `comm::Topic<T>` 与 move-only RAII `TopicSubscription<T>`，将订阅后的事件扇出到每个订阅者的独立有界 FIFO；发布结果报告匹配、成功和拒绝订阅数，逐订阅者保留独立 drop policy、统计、回调和关闭唤醒语义。该原语明确不提供网络传输、重放、可靠确认或硬实时保证。
- **LET 阶段通信契约**：`PhaseGate`、`DoubleBuffer` 与 `LatestMailbox` 新增可选的 phase-bound LET 模式。绑定后，发布仅发生在当前相位，读取仅暴露上一完成相位的数据；相位切换会拒绝活跃读写，避免跨周期读取或写入。
- **实时内存分配诊断**：新增 `comm::RealtimeAllocationGuard`、`RealtimeAllocationViolationPolicy` 和线程局部统计，可记录受保护实时路径中的分配次数、字节数、组件与阶段；Linux 构建可通过 `EXECUTOR_ENABLE_REALTIME_ALLOCATION_GUARD` 启用，`RealtimeThreadConfig::enable_allocation_guard` 控制周期回调的 opt-in 诊断。
- **通信延迟分位数**：`CommStats` 增加固定大小延迟直方图及近似 `p50_latency`、`p99_latency`，同时保留累计、平均和最大延迟统计。
- **有界任务图句柄保留**：`ExecutorConfig::task_graph_retention_capacity` 和对应运行时设置 API 控制终态 `TaskHandle` 的保留上限。被淘汰的句柄会明确拒绝为过期；仍被活跃依赖链引用的节点不会提前淘汰。
- **线程池真实扩缩容**：`ThreadPool::resize()` 与 `ThreadPoolResizer` 现创建或移除真实 worker，且仅接受初始化配置的线程数范围；缩容前迁移本地队列任务并 join 被移除 worker，返回时状态稳定。

### 修复与改进

- **通信同步核心无锁化**：`MpscChannel` / `RealtimeChannel` 改为构造期预分配的有界 MPSC 节点池，
  `LatestMailbox` / 未绑定 `DoubleBuffer` 改为固定 reader-pin 快照槽，`PhaseGate` / `Sequencer`
  改为原子状态核心；新增同步原子 lock-free 查询与构造期平台校验。该保证不覆盖 payload、callback、
  时钟、缺页或 OS 调度，`Topic` fan-out 仍属于 mutex 与动态分配支持的非实时控制面。
- **线程池扩缩容并发安全**：本地 worker 队列改以原子发布的 `shared_ptr` 快照访问，调度、窃取与 resize 通过读写锁协调，避免队列替换期间的悬空访问和 UAF；shutdown 与 resize 的 join 路径也已串行化。
- **任务调度边界**：移除 `TaskDispatcher` 的旧引用构造路径；空本地队列快照不会从调度器取走任务或发生越界访问。
- **实时契约实现边界**：LET 绑定要求固定双缓冲容量和不抛异常的复制语义；每个相位只允许一次发布，缺失上一相位数据或相位转换中读取会返回明确的通信错误。
- **兼容性**：调整 C++20 实现以兼容 GCC 10（项目仍建议使用 GCC 11 或更高版本）。

### 文档与测试

- 更新中英文 README、API、通信设计文档、教程站点与 sitemap，补充 LET 状态/相位、通信选择、延迟观测和失败可观测性示例说明。
- 新增 Topic fan-out/独立背压/并发生命周期、通信实时内存、LET `PhaseGate`、邮箱与双缓冲、任务图保留/过期语义的测试；扩展线程池扩缩容、调度 fallback 与并发 UAF 回归测试，并约束扩缩容压力用例的工作负载。
- 新增面向使用者的 `executor-integration` 渐进式接入指南，以及面向维护者的能力索引与维护参考。

---

## [0.3.1] - 2026-08-06

0.3.1 是统一 `Executor` Facade、完整生命周期监控与按意图自动路由的功能版本。除实时进程内存锁配置项外，它保留各执行模型真实的完成、接收和生命周期语义，而不将它们统一伪装为 `future`。

### 新增

- **任务意图与 CPU/GPU 双路径**：新增 `TaskOptions`、`TaskBuilder`、`ExecutionIntent`、`FallbackPolicy`、`cpu_gpu_task()` 和 `CpuGpuTask`。普通 `submit_auto(lambda)` 默认选择异步线程池；CPU/GPU 双路径仅在 GPU 可提交时使用 GPU，`AllowCpu` 才允许显式回退。
- **可解释路由与能力发现**：新增 `RoutingDecision`、routing callback、最近路由决策缓冲及 `get_executor_capabilities()`。路由说明与 failure event 分离：前者解释选择，后者报告实际拒绝或执行失败。
- **有界 dispatch**：新增 `DispatchResult` 和 `dispatch_auto()`。`LowLatency` 只投递到用户指定、运行中的无锁执行器；`RealtimeQueue` 只投递到指定、运行中的实时队列。返回值只表示队列接收，不表示任务完成。
- **无锁统一管理**：`ExecutorManager` 现注册、启动、停止并枚举 `LockFreeTaskExecutor`，跨异步、GPU、实时、Blocking I/O 和无锁后端保证名称唯一；关闭时先从无锁注册表摘除并停止。
- **Blocking I/O 统一控制面**：新增 `BlockingWorkerSpec`、`WorkerHandle` 和 `start_worker()`，封装注册、启动、状态查询及 stop/wakeup/join，同时保留 `IBlockingIoWorker` 的 stop token、启动超时和退出原因契约。
- **完整生命周期 Monitor**：新增 `ExecutorSnapshot`、`ExecutorLifecycleState`、`Executor::get_snapshot()` 和稳定文本导出，统一汇总生命周期、默认异步、Realtime、Blocking I/O、GPU、失败状态、最近失败事件、任务统计及聚合计数；snapshot 明确 `schema_version`、序号、采集时间、`partial` 和一致性说明。
- **故障现场诊断**：等待完成或 shutdown 超时、初始化/注册/启动失败路径可通过 snapshot callback 获取完整现场；诊断异常与业务执行、future、worker 和 shutdown 隔离。
- **有界在途任务诊断**：线程池和任务图支持 `Pending`、`Queued`、`Running`、`DependencyBlocked` 等生命周期状态，提供容量、采样率、状态计数、最老任务年龄和有限任务条目；容量溢出会计数并标记诊断不完整，不保存 callable、payload 或异常对象。
- **一致性校验与性能基线**：Manager 增加轻量 `state_epoch`，snapshot 采集前后最多重试两次，持续变化时标记 `epoch_changed`；idle initialized async 场景完成采集与文本格式化基线，epoch 校验不使用全局大锁。

### 测试与文档

- 新增自动路由阶段测试，覆盖默认路由、CPU 回退/拒绝、路由 callback 隔离、路由缓冲语义、无锁队列满、实时未启动/有界接收、Blocking worker 生命周期和能力枚举。
- 新增生命周期快照测试，覆盖未初始化不触发懒初始化、全部后端汇总、等待/关闭故障现场、in-flight 容量溢出、并发 shutdown，以及持续注册变化下的 epoch 有界重试和 partial 标记。
- `API.md`、`MIGRATION.md`、中英文 README 和 Blocking I/O 教程补充 API 选择表、结果语义、迁移路径及自动路由边界。
- `API.md`、生命周期 Monitor 设计文档和实施计划同步 snapshot 字段、best-effort/partial 语义、有限在途诊断、state epoch 和稳定文本导出说明。
- 新增 `examples/lifecycle_snapshot.cpp`，可在 CPU-only 构建中演示任务积压、任务失败、稳定文本导出和 shutdown 后的生命周期快照；该示例作为 CTest smoke test 运行。
- 新增生命周期快照性能基线，记录 idle initialized async 场景的采集/格式化耗时、格式化分配次数和输出字节数。

### 破坏性变更

- **实时进程内存锁配置**：`RealtimeThreadConfig::enable_memory_lock` 更名为 `enable_process_memory_lock`，并改为默认关闭，以明确其 Linux `mlockall` 的进程级语义。需要该能力的调用方必须改用新字段并显式设置为 `true`，同时检查 `RealtimeExecutorStatus::process_memory_lock_applied` 与 `process_memory_lock_errno`。

### 兼容性

- 除上述配置项外，`submit()`、`submit_gpu()`、实时和 Blocking I/O 的既有入口保持可用；生命周期 snapshot 为新增只读 API，既有单项状态和统计 API 无需迁移。
- legacy 四参数 CPU/GPU `submit_auto(TaskCharacteristics, name, kernel, config)` 在 `0.3.x` 保持既有“GPU 未就绪即失败、无隐式 CPU 回退”的行为，暂不添加编译期弃用标记。
- 带返回值的 CPU/GPU 自动任务、`ExecutionReport<T>` 和 legacy overload 的弃用/移除仅在后续允许破坏性变更的主版本评估。

---

## [0.3.0] - 2026-07-27

0.3.0 是面向跨线程通信、任务依赖编排和长期阻塞 I/O 生命周期管理的向后兼容功能版本。0.2.3 的公开 API 保持可用；新代码可逐步采用 `executor::comm`、任务图 facade 和 `BlockingIoExecutor`。

### 新增

- **通信与并发 facade**：新增安装头文件 `executor/comm.hpp` 及 `executor::comm` 命名空间，提供统一结果、错误码、统计与事件回调；公开 `MpscChannel<T>`、`SpscChannel<T>`、`LatestMailbox<T>`、`RealtimeChannel<T>`、`DoubleBuffer<T>` / `Snapshot<T>`、`PhaseGate` 和 `Sequencer`，覆盖有界消息流、最新值、实时周期 drain、一致快照和启动/顺序协调。
- **通信背压与可观测性**：channel 支持容量、超时、关闭和丢弃策略；通信组件可查询 `CommStats`，并可通过 `set_event_callback()` 获取低频事件。实时 channel 明确为有界、非等待 facade，周期内可设置 drain 预算。
- **任务图 facade**：`Executor` 新增 `TaskHandle`、`TaskSubmission<T>`、`submit_with_handle()`、`submit_after()`、`submit_after_with_handle()` 和 `when_all()`，用于在同一 `Executor` 实例内表达任务依赖与汇合；依赖失败、无效/跨实例 handle 与环路会以异常结果和 `SubmitRejected` 暴露。
- **阻塞 I/O worker**：新增 `IBlockingIoWorker`、`IBlockingIoExecutor`、`BlockingIoConfig` 与 `BlockingIoExecutorStatus`，并通过 `Executor` / `ExecutorManager` 提供注册、启动、停止和状态查询。worker 以 `run(std::stop_token)` + `wakeup()` 协作停止，适用于调用方持有的长期可中断 I/O 循环。
- **教程与用户网站**：新增中英文 VitePress 使用手册、完整教程示例和 GitHub Pages 部署/校验流程，覆盖任务提交、依赖、通信、实时控制、GPU、可观测性、部署和故障排查；新增阻塞 I/O worker 指南与教程示例。

### 修复

- **线程池和负载均衡并发安全**：修复 worker 队列丢失唤醒、完成排空、监控停止/异常生命周期、初始化失败回滚，以及动态扩缩容时 `LoadBalancer` 访问 worker 容器的数据竞争。
- **安全停止与生命周期**：worker 内调用 `stop()` 时安全转交 join；CUDA 并发停止串行化并修复重启后的 stopping state / waiter 注册；OpenCL 启动 CAS、建线程失败回滚及 stop/cleanup 与公开操作的生命周期互斥得到加固。
- **无锁执行器正确性**：修复 MPSC 槽位预留与发布导致的 head-of-line 阻塞、精确批量预留取消空洞、取消预留前的让步，以及 `LockFreeTaskExecutor::start()` 建线程失败后的回滚。
- **输入与诊断边界**：线程池拒绝空任务；`LockFreeQueue::backoff_multiplier` 增加边界校验；OpenCL kernel 异常写入 `last_error_message`；`submit_gpu()` 找不到执行器时记录 facade failure；明确 `dropped_task_count` 语义。
- **构建告警隔离**：CUDA/OpenCL 供应商头改为 `SYSTEM` include，严格告警仅应用于 executor 库目标，并修复 C++20 / 编译器告警问题。

### 测试 / CI

- **并发回归覆盖**：新增通信 facade、任务图、阻塞 I/O、线程池扩缩容、worker 自停止、CUDA 并发停止、OpenCL 生命周期和 MPSC 并发测试。
- **持续集成加固**：TSAN 与无锁 CI 覆盖每次变更；coverage / 无锁工作流聚焦功能测试；修复线程池、实时和 CUDA 环境相关的 flaky 测试，并让全部教程示例参与构建检查。

### 文档

- **迁移与边界说明**：`MIGRATION.md` 新增 0.2.3 → 0.3.0 的通信、任务图与阻塞 I/O 迁移建议，明确这些 API 均为兼容扩展；API 文档补充通信 facade、任务图、I/O worker、实时丢弃计数与集成边界。
- **性能记录**：更新批量提交 benchmark 基线数据；性能收益仍依赖任务规模、硬件、线程数与构建配置，应以本地测试结果为准。

### 兼容性

- **无破坏性变更**：0.3.0 保持 0.2.3 公开 API 兼容。通信 facade 不替代调用方的协议、设备重连、数据语义或安全策略；实时通信 facade 的内部实现不构成硬实时或无锁保证，存在此类要求时应使用经验证的专用实现。

---

## [0.2.3] - 2026-07-08

0.2.3 是面向 `Executor` facade 完整度与运行时失败可观察性的向后兼容版本。已有 0.2.2 代码可以继续编译使用；新代码建议优先使用 `_ex`、failure callback、facade 实时推送和可诊断等待 API。

### 新增

- **统一失败事件模型**：新增 `FailureKind`、`ExecutorFailureEvent`、`ExecutorFailureStatus` 与 `ExecutorFailureCallback`，覆盖任务异常、提交拒绝、任务超时、实时丢任务、GPU 失败、等待超时和调优回退等事件类型。
- **Facade 失败观察入口**：`Executor` 新增 `set_failure_callback()`、`get_failure_status()`、`get_recent_failures()`、`clear_recent_failures()` 和 `set_recent_failure_capacity()`；未设置 callback 时，失败仍会累计到状态计数和最近事件缓冲。
- **可诊断 Result API**：新增 `ExecutorResult`、`ExecutorErrorCode` 与 `executor_error_code_to_string()`；`initialize_ex()`、`register_realtime_task_ex()`、`start_realtime_task_ex()`、`register_gpu_executor_ex()` 可返回稳定错误码与说明消息，旧 `bool` API 保持兼容并委托到 `_ex`。
- **等待与生命周期状态**：新增 `CompletionStatus`、`WaitResult`、`wait_for_completion_for()`、`wait_for_completion_ex()`、`try_wait_for_completion()`、`is_idle()` 和 `get_completion_status()`，等待超时时可查看 active / queued / pending 任务快照。
- **周期任务状态查询**：新增 `PeriodicTaskStatus`、`get_periodic_task_status()` 与 `get_all_periodic_task_status()`，记录周期任务执行次数、失败次数、连续失败次数、最后错误与下一次执行时间。
- **实时 facade 推送**：新增 `Executor::push_realtime_task()` 与 `try_push_realtime_task()`，用户无需获取底层 `IRealtimeExecutor*` 即可推送实时任务；失败通过返回值、failure event 和状态计数同时可见。
- **实时拒绝原因计数**：`RealtimeExecutorStatus` 新增 `rejected_not_running_count`、`rejected_empty_task_count`、`pool_exhausted_count`、`queue_full_count`，将 `dropped_task_count` 的主要原因拆开观测。
- **失败可观察示例**：新增/更新 `examples/failure_observability.cpp`、`examples/periodic_monitoring.cpp` 与 `examples/realtime_can.cpp`，展示 `_ex` 初始化、failure callback、wait result、周期状态和实时 facade 推送。

### 修复

- **普通异步任务异常可见**：通过 `Executor::submit()` / `submit_priority()` / `submit_batch()` 提交的用户任务即使调用方没有立即 `future.get()`，也会记录 `TaskException` failure event，并让底层失败统计保持可见。
- **fire-and-forget 批量任务异常可见**：`submit_batch_no_future()` 的用户任务异常进入 failure event / status counter，避免无 future 场景下静默丢失异常。
- **提交拒绝可见**：未初始化、shutdown 后提交、空 batch、执行器不可用等提交失败路径记录 `SubmitRejected`，有 future 的路径会设置异常。
- **延迟与周期任务失败可见**：`submit_delayed()` 到期提交失败会设置 promise 异常并记录 failure event；`submit_periodic()` 的周期回调异常会更新周期任务状态并触发 failure event，默认继续调度。
- **等待超时可诊断**：`wait_for_completion()` 保持兼容签名，但超时不再无声返回；`wait_for_completion_ex()` 返回超时状态快照并累计 `wait_timeout_count`，shutdown 等待超时也会留下诊断事件。
- **Callback 异常隔离**：failure callback 自身抛出的异常不会杀死 worker、定时器线程或后台执行路径。
- **实时推送失败归因**：facade 推送在实时 executor 不存在、未运行、空任务、队列满或对象池耗尽时统一返回 `false` 并记录对应 failure event。

### 文档

- **README / README_zh**：同步 `Executor` facade 的失败可观察性、可诊断 `_ex` API、实时推送入口与 `wait_for_completion_ex()` 示例。
- **API.md**：补充 `ExecutorResult`、failure status、periodic status、wait result、实时背压字段和软超时 future 异常语义。
- **MIGRATION.md**：新增“从 0.2.2 升级到 0.2.3”，明确无破坏性变更，并给出推荐迁移入口。
- **Facade 可观察性计划**：`docs/todolists/facade_observability_update_plan.md` 标记阶段 1-6 完成，保留通信 facade 阶段作为后续工作。
- **Deb 打包指南**：发布命令、版本检查清单和 CUDA 完整包说明更新到 `0.2.3`。

### 测试 / CI

- **失败可观察性测试**：新增/补强 `test_executor_failure_observability`、`test_periodic_failure_observability`、`test_thread_pool_timeout_future`，覆盖任务异常、批量无 future、周期任务失败和软超时 future 异常。
- **可诊断 API 测试**：新增 `test_executor_result_diagnostics`，覆盖初始化、实时注册/启动、GPU 注册等 `_ex` 错误码。
- **等待超时测试**：新增/补强 `test_wait_completion_result`、`test_wait_for_completion_timeout_observable`，验证完成、超时、未初始化状态和 failure counter。
- **实时 facade 推送测试**：新增 `test_realtime_facade_push`，覆盖成功推送、不存在 executor、未运行、空任务、停止后推送和队列满失败路径。
- **通信 facade 规划测试**：保留 `tests/harness/test_comm_facade_usage.cpp` 的 disabled 用例，作为后续 `executor::comm` facade API 的需求锚点；0.2.3 不发布该 API。

### 兼容性

- **无破坏性变更**：0.2.3 保持 0.2.2 公开 API 兼容；新增 API 均为扩展。旧 `initialize()`、`register_realtime_task()`、`start_realtime_task()`、`register_gpu_executor()`、`wait_for_completion()` 和底层 `push_task()` 继续可用。

---

## [0.2.2] - 2026-06-18

### 修复

- **v0.2.1 紧跟的 CI 修复**：见下方 `### 测试 / CI` 段。
- **无锁基础设施稳定性**（无 PR）：修复 benchmark 超时、`shutdown(true)` 任务挂起、`dispatch_batch` 任务丢失、周期任务取消竞态、无锁队列容量检测与内存可见性问题。
- **ObjectPool ABA 关键修复**（无 PR）：将 `ObjectPool` free list 从无锁 CAS 改为 mutex 保护，彻底消除 ABA 导致的 SEGFAULT；影响 `LockFreeTaskExecutor`、`RealtimeThreadExecutor` 等使用 `ObjectPool<Task>` 的执行器，`acquire()` / `release()` 接口保持兼容。
- **ObjectPool 入参与释放防护** [#3]：拒绝 `capacity=0` 构造，避免无效对象池配置。
- **ObjectPool release 防护**（无 PR）：新增 double-free / foreign pointer guard，防止重复释放或外部指针污染对象池。
- **ObjectPool::release O(1)** [#41]：优化释放路径，保留正确性防护并降低释放开销。
- **ThreadPool WorkerLocalQueue** [#1] [#6]：修复 `empty()` 判断逻辑与 `steal()` 竞态。
- **ThreadPool 状态统计** [#2]：修复 `get_status().idle_threads` 的 `size_t` 下溢。
- **ThreadPool 并发关闭** [#11]：修复 `stop()` / `shutdown()` 并发调用导致 double-join UB。
- **ThreadPool resize / dispatch** [#14]：worker 无效时任务重新入队，避免 resize 期间丢任务。
- **ThreadPool try_steal_task** [#15]：对 `local_queues_` 使用 `shared_lock`，修复 resize 并发访问竞态。
- **ThreadPool resize UAF**（无 PR）：使用 `shared_lock` 防护 resize 期间的队列生命周期。
- **LoadBalancer 数据竞争** [#12]：将 `strategy_` 改为 atomic。
- **LockFreeQueue 数据竞争** [#4] [#13]：`size()` 使用 acquire ordering，`stats_enabled_` 改为 atomic。
- **LockFree batch 异常安全** [#31]：`push_tasks_batch` 在对象池耗尽与部分入队场景下保持资源回收正确。
- **LockFreeTaskExecutor 构造泄漏** [#33]：使用 `unique_ptr` 替换裸指针，修复构造失败路径泄漏。
- **LockFreeTaskExecutor 异常可见性** [#44]：任务异常可被统计与观察，避免后台吞掉故障。
- **GPU submit_kernel_after** [#7] [#28]：`submit_kernel_after` 不阻塞 GPU worker，并修复依赖任务 UAF。
- **CudaExecutor wait_for_completion UAF** [#39]：修复等待完成期间的生命周期问题。
- **CudaExecutor submit 不死锁** [#43]：修复提交路径中可能出现的死锁。
- **Realtime 周期任务预算** [#40]：周期任务超预算时正确记录与处理。
- **simple_cycle_loop skip-late** [#42]：周期循环对过晚周期执行跳过策略，降低积压。
- **set_thread_priority nice** [#45]：Linux 下真正应用 nice 值，修复优先级配置未生效问题。
- **Windows 编译调整**（无 PR）：修复 Windows 平台编译兼容性。

### 新增

- **LockFreeTaskExecutor**（无 PR）：新增 MPSC 无锁任务执行器，提供 `start()`、`stop()`、`push_task()`、`is_running()`、`pending_count()`、`processed_count()` 与队列统计接口，适用于高频日志、实时事件与多线程任务聚合。
- **批量任务提交 API**（无 PR）：新增 `submit_batch()` 与 `submit_batch_no_future()`，单线程 500-2000 任务场景可获得 **5-16x** 加速。
- **LockFreeTaskExecutor 批量提交**（无 PR）：新增 `push_tasks_batch()`，支持尽力批量入队与实际入队数量回传。
- **智能调度接口**（无 PR）：新增智能调度与自适应调度能力，为后续 facade 默认优化提供基础。
- **实时 push_task 背压可见性** [#32]：新增 `push_task_ex()` 与 `dropped_task_count` / `failed_pushes` 等状态字段；`push_task()` 保持 void 兼容，背压丢任务可被观测。
- **软任务超时** [#24]：新增 `task_timeout_ms` 软超时语义，执行前 `elapsed >= timeout` 时跳过并计入 `timeout_count`；C++ 无安全线程终止机制，执行中的任务不被强制中断。

### 优化

- **无锁 MPSC 基础设施**（无 PR）：从 MPSC 队列、无锁任务执行器、批量提交一路演进到序列号 MPSC 队列、False Sharing 消除、CAS 重试策略优化、性能监控优化与 worker local queue 无锁化。
- **无锁工作线程队列**（无 PR）：`WorkerLocalQueue` 改造为无锁实现，提交吞吐量 **441,500 → 488,698 tasks/s（+10.7%）**，端到端吞吐量 **433,083 → 442,009 tasks/s（+2.1%）**。
- **Linux 实时性加固** [#16]：`RealtimeThreadExecutor` 增加 `mlockall`、`timer_slack`、线程命名等加固，1ms 周期 jitter p99 从 61 µs 压至约 15-20 µs。
- **Default-Optimal Facade (P019)** [#19] [#20] [#21] [#22]：
  - `enable_memory_lock` / `timer_slack_ns` 从 opt-in 改为 opt-out，默认开启实时性优化。
  - `min_threads` / `max_threads = 0` 时自动探测 `hardware_concurrency()`，`work_stealing` 默认开启。
  - 线程池 `cpu_affinity` 空时自动分配 [0..hw-1]，实时线程按周期自适应优先级。
  - 实时线程 `cpu_affinity` 空时自动绑核；多实时线程使用 round-robin 自动亲和性。
  - 1ms 周期 jitter p99 从 54.64 µs 降至 **1.77-6.64 µs**，降低 **89-97%**。
- **ThreadPool soft timeout** [#24]：执行前跳过超时任务并计数，避免误导用户认为执行中任务会被强杀。
- **LockFree spin+yield** [#26]：用 spin+yield 替换 100µs busy-sleep，降低无锁执行器等待延迟。
- **Realtime Windows timer 数据竞争** [#27]：`timer_period_ms_` 改为 atomic。
- **Realtime 多线程亲和性** [#29]：多个实时线程自动 round-robin 分配 CPU affinity。
- **GPU 性能优化**（无 PR）：补充 GPU 性能优化与性能测试报告。

### 文档

- **ObjectPool ABA 设计说明**（无 PR）：新增 ABA 修复设计文档，说明从 CAS free list 切换到 mutex 的正确性取舍。
- **API / README / CHANGELOG / MIGRATION 同步** [#17]：同步公开 API、默认值、迁移说明与发布记录。
- **README 拆分** [#18]：拆分英文 `README.md` 与中文 `README_zh.md`。
- **P019 facade 文档同步** [#23]：同步默认即最优 facade 哲学、实时性默认值与性能描述。
- **批量提交与软超时语义** [#25]：补充 `push_tasks_batch` 与 `task_timeout_ms` soft timeout 说明。
- **LockFreeQueue empty / size 语义** [#30]：说明 `empty()` 与 `size()` 在并发场景下的近似语义。
- **API 背压字段** [#46]：补充 `push_task_ex()`、`dropped_task_count`、`failed_pushes`、`queue_capacity` 等背压字段说明。

### 测试 / CI

- **v0.2.1 紧跟的 CI 修复**（无 PR）：连续修复 5 个 CI 问题，稳定 0.2.1 后续发布分支。
- **benchmark_batch_* 超时修复**（无 PR）：修复批量提交 benchmark 测试超时。
- **benchmark_lockfree_task_executor timeout**（无 PR）：CTest timeout 从 30s 调整为 120s，避免高吞吐压测误判超时。
- **无锁队列与批量提交测试清理**（无 PR）：移除无用测试文件并补强批量、并发、工作窃取相关测试。
- **benchmark latency 阈值调整**（无 PR）：放宽 `latency_single_task` P99 限制到 100µs，降低 CI 环境噪声误报。
- **Code Coverage mlockall 跳过**（无 PR）：覆盖率任务中跳过 `mlockall`，避免 OOM。
- **CUDA 测试头文件包含**（无 PR）：为 `test_unified_memory` 与 `test_gpu_dep_async` 补充 CUDA headers。

### 构建

- **Windows 编译调整**（无 PR）：修复 Windows 平台构建问题。
- **CMake 4.x CUDA Toolkit**（无 PR）：从 `CUDAToolkit` 推导 `CUDA_INCLUDE_DIRS`，兼容 CMake 4.x。
- **CUDA / no-CUDA 安装策略**（无 PR）：CUDA executor 运行时通过 `dlopen libcuda` 动态加载；deb 发布使用带 CUDA 的完整构建，用户机器无 CUDA 时运行时自动降级。

### 性能基准

- **任务提交吞吐量**：`benchmark_baseline` 从 v0.2.0 的 456,703 tasks/s 保持同档并在 commit path 达到约 488K+，约 **+7%+**。
- **MPSC 工作窃取场景**：提交吞吐 **441,500 → 488,698 tasks/s（+10.7%）**；端到端 **433,083 → 442,009 tasks/s（+2.1%）**。
- **实时线程 1ms 周期 jitter**：p99 **61.30 µs → 1.77-6.64 µs（-89% ~ -97%）**；avg **54.47 µs → 1-2 µs（约 -95%）**。
- **LockFreeTaskExecutor SPSC**：10K 任务提交平均 **97.29 ns**，p50 **29 ns**，p99 **1013 ns**，吞吐 **8,242,895 ops/s**。
- **LockFreeTaskExecutor 端到端**：100K 任务端到端吞吐 **5,942,007 ops/s**。
- **批量提交**：单线程 500-2000 任务场景 **5-16x** 加速。

---

## [0.2.1] - 2026-03-09

### 新增

- **OpenCL 执行器**：实现 `OpenCLExecutor`，支持跨平台异构计算（Intel/AMD/NVIDIA GPU）
- **OpenCL 动态加载**：运行时加载 OpenCL 库，无静态链接，OpenCL 不可用时安全降级
- **GPU 设备查询 API**：新增 `enumerate_cuda_devices()`、`enumerate_opencl_devices()`、`enumerate_all_devices()`、`get_recommended_backend()` 函数，用户可查询系统可用 GPU 设备及推荐后端
- **设备信息增强**：`GpuDeviceInfo` 新增 `vendor` 字段，标识 GPU 厂商（NVIDIA/AMD/Intel）
- **统一内存支持**：CUDA 执行器支持统一内存（Unified Memory），新增 `allocate_unified_memory()`、`free_unified_memory()`、`prefetch_memory()` 方法；配置选项 `enable_unified_memory`；CPU 与 GPU 可共享内存无需显式传输
- **构建与示例**：`EXECUTOR_ENABLE_OPENCL` 选项；示例 `gpu_opencl`、`gpu_device_query`、`gpu_unified_memory`

### 文档

- **OpenCL 环境搭建指南**：[docs/setup/opencl_setup.md](docs/setup/opencl_setup.md)，包含 Linux/Windows 环境配置、常见问题排查

详细设计见 [docs/design/gpu_executor.md](docs/design/gpu_executor.md)。

---

## [0.2.0] - 2026-01-29

### 新增

- **GPU 执行器（CUDA）**：`IGpuExecutor` 接口，CUDA 执行器实现，与 ExecutorManager/Executor Facade 集成
- **GPU 任务与配置**：`register_gpu_executor`、`submit_gpu`、`get_gpu_executor`、`get_gpu_executor_status`、`get_gpu_executor_names`；`GpuExecutorConfig`、`GpuTaskConfig`、`GpuDeviceInfo`、`GpuExecutorStatus`
- **CUDA 动态加载**：运行时加载 CUDA 库，无静态链接，CUDA 不可用时安全降级
- **GPU 内存与流**：设备内存分配/释放、主机↔设备/设备↔设备拷贝（含异步）、流创建/销毁/同步、流回调
- **多 GPU 设备**：按设备 ID 注册多个执行器；设备间 P2P 拷贝为**实验性**，未在多 GPU 实机充分测试
- **GPU 内存池与监控**：可选内存池（`GpuMemoryManager`）、kernel 与内存统计、异常处理与错误码转换
- **GPU 任务队列**：优先级、批量提交、任务依赖（`submit_kernel_after`）
- **构建与示例**：`EXECUTOR_ENABLE_GPU`、`EXECUTOR_ENABLE_CUDA` 选项；示例 `gpu_basic`、`gpu_multi_device`

### 其他

- **CI**：C/C++ 工作流重构，依赖升级至 v4
- **文档与测试**：实时线程周期精度记录与外部接入说明；定时器优化；消除测试中数据竞态

详细设计见 [docs/design/gpu_executor.md](docs/design/gpu_executor.md)。

---

## [0.1.1] - 2026-01-25

### 优化

- **锁竞争优化**：为 `PriorityScheduler` 的每个优先级队列使用独立锁，减少锁竞争，端到端吞吐量提升 5.3%，延迟 p99 降低 18%
- **内存分配优化**：将 `PriorityScheduler` 从 `shared_ptr<Task>` 改为 `unique_ptr<Task>`，减少内存分配开销和控制块开销
- **批量分发优化**：实现真正的批量任务分发，批量 dequeue/push/负载更新，减少锁操作次数，端到端吞吐量提升 2.9%
- **工作窃取优化**：实现基于负载的智能窃取策略，优先从高负载线程窃取任务，端到端吞吐量提升 5.9%，延迟 p99 降低 44.4%，提交吞吐量提升 7.3%
- **延迟任务处理优化**：使用 `priority_queue` 替代 `vector` + `remove_if`，按执行时间排序，提高延迟任务处理效率
- **任务 ID 生成优化**：使用原子计数器替代时间戳实现，任务 ID 生成性能提升 80-90%，端到端吞吐量提升 7.3%，延迟 p99 降低 44%

### 性能提升

相比 v0.1.0：
- 端到端吞吐量提升 **13.0%**（461,576 → 521,390 tasks/s）
- 延迟 p99 降低 **55%**（0.22μs → 0.10μs）
- 提交吞吐量略有波动，整体保持稳定

详细优化记录和性能测试结果参见 [docs/optimization/PERFORMANCE_OPTIMIZATION.md](docs/optimization/PERFORMANCE_OPTIMIZATION.md)。

---

## [0.1.0] - 2025-01-24

### 新增

- **Executor Facade**：统一 API `Executor::instance()` / 实例化模式，`initialize`、`shutdown`、`wait_for_completion`
- **任务提交**：`submit`、`submit_priority`、`submit_delayed`、`submit_periodic`、`cancel_task`
- **实时任务**：`register_realtime_task`、`start_realtime_task`、`stop_realtime_task`、`get_realtime_executor`、`get_realtime_task_list`
- **监控**：`enable_monitoring`、`get_async_executor_status`、`get_realtime_executor_status`、`get_task_statistics`、`get_all_task_statistics`
- **执行器管理**：`ExecutorManager` 单例/实例化，默认异步执行器 + 实时执行器注册表，RAII 生命周期
- **线程池**：动态扩缩容、优先级调度、工作窃取、负载均衡、任务分发
- **专用实时线程**：`RealtimeThreadExecutor`，周期回调、线程优先级、CPU 亲和性，可选 `ICycleManager` 集成
- **配置**：`ExecutorConfig`、`ThreadPoolConfig`、`RealtimeThreadConfig`
- **构建与安装**：CMake 3.16+，静态/动态库选项，`find_package(executor)` 支持（`executorConfig.cmake`、`executorConfigVersion.cmake`）
- **测试与示例**：单元/集成/性能/压力测试，`basic_submit`、`realtime_can`、`multi_project`、`monitor_example`

### 依赖与平台

- C++20，仅标准库 + `pthread`（Linux），无第三方必需依赖
- Linux/windows 下已验证；

---

## 迁移指南

当前为首次发布，无历史版本可迁移。若未来有破坏性变更，将在此补充迁移说明。

参见 [docs/API.md](docs/API.md) 与 [docs/design/executor.md](docs/design/executor.md)。
