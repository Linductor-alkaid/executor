# Blocking I/O Executor 实施与使用手册更新计划

本文档把 [阻塞 I/O 执行器扩展设计](../design/blocking_io_executor.md) 拆分为可提交、
可验证的实现计划。目标是在不改变现有 async、实时、GPU 和 `executor::comm` 语义的前提下，
新增用于 LCM、socket、串口和 CAN 等长期接收循环的 `BlockingIoExecutor`。

本计划中的网站工作必须跟随公开 API、编译示例和测试完成后落地；设计稿不是公共 API，不能先在
使用手册中承诺尚未实现的类型或实时性质。

---

## 当前基线

- [x] `ThreadPoolExecutor` 管理有限的异步任务；不适合永久占用 worker 的 transport loop。
- [x] `RealtimeThreadExecutor` 管理固定周期、短回调和有界任务消费；阻塞 I/O 会计入周期预算。
- [x] `ExecutorManager` 已统一管理默认 async、RT、GPU 的注册与 shutdown。
- [x] `executor::comm` 已提供 `MpscChannel`、`LatestMailbox`、`RealtimeChannel`、`DoubleBuffer`、
  `PhaseGate` 和本地 `CommStats`/`CommEventCallback`。
- [x] 现有通信事件默认独立于 `ExecutorFailureStatus`。
- [x] 网站已有中英文“实时与通信”专题、导航、编译并 smoke-test 的教程机制，以及内容维护规则。
- [ ] 还没有针对专属、可阻塞、可唤醒 transport worker 的 executor 类型、状态或 facade API。

## 实施原则与不可变约束

1. **接口分离**：I/O executor 不继承 `IAsyncExecutor` 或 `IRealtimeExecutor`，不提供通用 `push_task()`。
2. **停止必须可中断**：`stop_token` 只表达请求；每个 worker 必须提供 `wakeup()` 来解除阻塞等待。
3. **普通调度默认值**：I/O worker 默认不申请 `SCHED_FIFO`、不自动绑核、不自动 memory lock。
4. **控制/I/O 解耦**：接收、校验、解码和发布在 I/O worker；控制在 RT loop；重计算在 async/GPU。
5. **通信语义不混淆**：状态、命令、诊断和配置各自选择 mailbox/channel/snapshot；不以“队列最快”为选择依据。
6. **错误分域**：启动和注册失败进入 `ExecutorResult`；运行期 transport/协议/背压进入 I/O 或通信诊断面。
7. **不 detach**：未 join 的 worker 不得脱离 Executor 生命周期，防止 transport 与数据面 use-after-free。
8. **文档以事实为源**：公开示例必须编译；中英文页面使用相同源码和命令；完整签名仍以 `docs/API.md` 为准。

---

## 阶段 I0：接口、配置和状态骨架

### 任务

- [x] 新增公开头 `include/executor/blocking_io.hpp`。
- [x] 定义 `BlockingIoConfig`：线程名、可选 affinity、可选 memory lock、启动 ready 超时。
- [x] 定义 `IBlockingIoWorker`：
  - [x] `run(std::stop_token)`。
  - [x] `wakeup() noexcept`。
  - [x] 由执行器以一次性 ready signal 确认线程属性完成后进入 worker。
- [x] 定义 `IBlockingIoExecutor`：`start()`、`request_stop()`、`stop()`、`get_name()`、`get_status()`。
- [x] 定义 `BlockingIoExecutorStatus` 和 stop reason：运行/ready/stop 请求、线程属性结果、wakeup 计数和最后错误。
- [x] 在 `include/executor/types.hpp` 中增加稳定状态字段；transport 特有统计保留在 adapter/通信层。
- [x] 编写 config 校验函数：空名、负 timeout、空 worker 均以 `InvalidConfig` 拒绝。

### 验收

- [x] 公共头可独立包含，且不引入 LCM、POSIX fd 或硬件 SDK 依赖。
- [x] 所有状态字段有初始化值和线程安全读取规则。
- [x] API 明确声明：没有有限等待或可唤醒路径的 worker 不满足契约。

### 测试

- [x] 新增 `test_blocking_io_types.cpp`：默认值、状态快照和公开头可独立编译。
- [x] 编译测试：消费者只依赖公开头，不需要链接实际 transport。

---

## 阶段 I1：单 worker 执行器与严格停止语义

### 任务

- [x] 新增实现文件 `src/executor/blocking_io_executor.hpp/.cpp`。
- [x] 内部使用 `std::jthread`，由 executor 唯一拥有 worker 和线程。
- [x] 启动路径：设置运行状态 -> 创建线程 -> 在线程内配置名称/affinity -> ready -> 执行 `run(stop_token)`。
- [x] 启动 ready 超时会 request-stop、wakeup、join、保存 `StartFailed` 原因并回滚运行状态。
- [x] `request_stop()` 按固定顺序执行：标记 stop -> `request_stop()` -> `worker.wakeup()`。
- [x] `stop()` 对已 join worker 幂等，串行化 stop，并在不持有 registry 锁时 join。
- [x] 捕获 `run()` 未处理异常；区分“请求后正常返回”“未请求的提前返回”“worker 异常”。
- [x] 不实现 join timeout 后 detach。
- [x] 仅在用户显式配置时执行 affinity 或 memory lock；记录请求是否真正生效。

### 验收

- [x] 没有阻塞 I/O 时，start/stop/析构均完成资源收敛；ASAN 执行路径通过。
- [x] 受控阻塞 worker 收到 wakeup 后可退出且 `stop()` 返回。
- [x] 多个线程同时调用 stop、重复启动、从 worker 自身回调请求停止都有定义行为，不产生双 join 或 `std::terminate`。
- [x] `std::jthread` 只存在于实现层，对调用方不泄漏线程所有权。

### 测试

- [x] 新增 `test_blocking_io_executor.cpp`。
- [x] 覆盖线程创建失败回滚、ready 超时、worker 启动异常、非预期返回、运行期异常。
- [x] 用 condition variable mock 实现可控阻塞 worker，验证 `wakeup()` 与 join。
- [x] 验证 stop 幂等、并发 stop、析构清理、自停止和状态计数。
- [ ] TSAN 测试生命周期状态读取与并发 stop；ASAN/LSAN 覆盖异常启动和停止路径。
  当前环境已完成 TSAN 编译与 ASAN 执行；TSAN runtime memory mapping 和 LSAN ptrace 扫描受宿主环境限制，留给 CI/目标机执行。

---

## 阶段 I2：ExecutorManager 与 Facade 集成

### 任务

- [x] 在 `ExecutorManager` 增加 I/O executor registry、访问锁、注册/查询/列举 API。
- [x] 在 `Executor` facade 增加：
  - [x] `register_blocking_io_worker[_ex]()`。
  - [x] `start_blocking_io_worker[_ex]()`。
  - [x] `stop_blocking_io_worker()`。
  - [x] `get_blocking_io_worker_status()` 与名称列表查询。
- [x] 注册时检查 RT、GPU、I/O 名称冲突；第一版在 facade 做跨 registry 检查。
- [x] 复用 `ExecutorResult` / `ExecutorErrorCode`：`InvalidConfig`、`DuplicateName`、`NotFound`、`StartFailed`。
- [x] 只对注册/启动等 executor 生命周期失败记录 facade failure；worker 运行期异常不记为 task exception。
- [x] 更新 `ExecutorManager::shutdown()`：先移出 I/O executor 所有权并释放 registry 锁，再 request-stop/wakeup/join。
- [x] 保持 `Executor::shutdown()` 先停止 timer thread 的当前语义；I/O join 对 `shutdown(false)` 仍是强制的生命周期收敛。
- [x] 审查单例 `atexit` 与实例模式析构路径；二者复用 `ExecutorManager::shutdown()`，实例析构已有 I/O worker 测试，单例 atexit 由既有 Facade 回归覆盖。

### 验收

- [x] I/O worker 可以通过 facade 得到 RT/GPU 一致的命名、注册、启动、状态和停止体验。
- [x] shutdown 不在 I/O registry 锁内调用外部 worker 代码或 join，避免锁级反转。
- [x] 既有 async/RT/GPU API 与 Facade 回归测试保持行为兼容。

### 测试

- [x] 新增 facade/manager 注册、重名、未找到、重复启动、状态查询测试。
- [x] 新增 I/O 注册失败和运行期 worker 异常的 facade 诊断隔离断言。
- [x] 添加 timer、RT、I/O、async 同时存在的 mixed shutdown 测试；GPU 仍由可选后端测试覆盖。
- [ ] 在 registry 操作和 stop 并发时运行 TSAN。

---

## 阶段 I3：通信数据面与状态新鲜度

### 任务

- [ ] 为每类 transport 消息声明明确业务语义和组件选择：
  - [ ] 最新状态/目标：`LatestMailbox<T>` 或 `DoubleBuffer<T>`。
  - [ ] 每条必须处理的事件、诊断和日志：有界 `MpscChannel<T>`。
  - [ ] RT 周期内有限命令：`RealtimeChannel<T>`，设定单周期预算。
  - [ ] 启动、重连和配置切换：`PhaseGate` 或 mailbox。
- [ ] 在 adapter 中发布消息前完成长度、版本、校验和和反序列化验证；不要让 RT 回调触碰原始 packet。
- [ ] 给状态消息定义 `sequence`、时间戳、最大可接受 age 和陈旧状态下的应用安全动作。
- [ ] 将 transport 层计数（poll、recv、decode error、断链）与 `CommStats`（drop、overwrite、lag、latency）并列呈现。
- [ ] 将高代价解码、记录和规划转交 async/GPU，但保留有界队列与拒绝/丢弃策略。
- [ ] 为动态配置、关闭和重连定义 channel close / phase transition 语义，禁止 worker 在 RT 消费者未退出时销毁共享数据面。

### 验收

- [ ] I/O worker 只承担 transport 和轻量数据准备；RT callback 不调用阻塞接收。
- [ ] 所有跨线程边都有容量、满队列策略、所有权、状态年龄和告警来源。
- [ ] 状态流的 overwrite 不被误报为命令丢失；命令流不误用 `KeepLatest`。

### 测试

- [ ] I/O producer 到 mailbox/channel 的 sequence、覆盖、满队列、close、陈旧值测试。
- [ ] RT 回调在无初始状态、状态过期、解码错误、channel 关闭时进入定义的业务降级路径。
- [ ] 高生产速率下验证不会无限积压，也不会使 RT `cycle_timeout_count` 异常增长。
- [ ] 结合 comm event callback 检查高频事件无 callback 时不产生额外日志/分配。

---

## 阶段 I4：Transport adapter 与 LCM 首个接入

### 任务

- [ ] 抽出应用当前独立 `std::jthread` 的 LCM 接收循环为 `LcmBlockingIoWorker`；不在 executor 核心库引入 LCM 必选依赖。
- [ ] 优先实现 LCM fd 与 wakeup fd 的 `poll` 路径：收到数据后 `handleTimeout(0)`，收到 wakeup 后退出。
- [ ] 若目标 LCM 版本不能获取 fd，使用有限 `handleTimeout(timeout)` 退化路径；记录并测试最坏 stop 延迟。
- [ ] `wakeup()` 使用平台封装：Linux 优先 eventfd，其他平台使用等价机制；清晰处理 EAGAIN/EINTR。
- [ ] LCM 回调只执行轻量解码和发布；复杂计算提交给 async/GPU executor。
- [ ] 为 socket、串口、CAN 给出同一 adapter 契约和 mock，不在第一版实现所有协议。
- [ ] 为 transport 断开/重连定义退避、状态更新与事件阈值；第一版只允许 adapter 内重连，不自动重启整个 executor。

### 验收

- [ ] LCM I/O 的阻塞不出现在 RT 线程或普通任务池。
- [ ] `stop()` 在目标 transport 的已声明上界内返回；无限 `handle()` 路径不得进入生产 adapter。
- [ ] LCM 可选依赖不会影响核心库的无 LCM 构建和测试。

### 测试

- [ ] transport mock 验证 fd-ready、wakeup-ready、timeout、EINTR、decode failure、断链和重连。
- [ ] 可选 LCM 集成测试只在依赖存在时启用；无 LCM 环境以 mock 覆盖完整停止契约。
- [ ] 在真实目标设备记录 stop 延迟、收包到发布延迟、状态年龄、RT 周期超时与 CPU 使用。

---

## 阶段 I5：安全关闭、可观测性与部署检查

### 任务

- [ ] 定义应用级 `prepare_shutdown()` 模式：先让执行器外部输出进入安全状态，再由 Manager 停 I/O、RT、GPU、async。
- [ ] 不让通用 Executor 猜测设备安全顺序；为不同设备说明“安全命令先发”与“先断接收”两种差异。
- [ ] 定义 I/O worker health 判断：最近成功接收、连续 decode/transport error、状态 age、channel lag。
- [ ] 选择低频阈值事件，而不是每个包/每次覆盖都写日志。
- [ ] 为 I/O worker 状态接入项目既有监控查询或提供独立查询；不把 `CommStats` 重复汇总到 task failure。
- [ ] 在部署清单中增加 Linux 权限、cpuset、IRQ affinity、socket buffer、fd 限制、memory lock、日志路径和设备断连验证。
- [ ] 以实际硬件数据决定是否需要 affinity 或普通优先级调整；第一版不因测量前猜测而开启 FIFO。

### 验收

- [ ] 明确安全关键应用必须显式 shutdown，`atexit` 只是兜底。
- [ ] 运行状态、通信背压和状态新鲜度可分别诊断，且告警动作有业务归属。
- [ ] 高负载与设备断连不会让 shutdown、RT 控制或监控面无界阻塞。

### 测试

- [ ] 端到端启动/运行/断链/重连/安全关闭测试。
- [ ] 反复创建销毁 Executor 实例和单例 atexit 路径测试。
- [ ] 目标硬件上的延迟、CPU、RT deadline 和错误恢复基线报告。

---

## 阶段 D1：源码、API 与迁移文档

本阶段在 I0-I3 的公开 API 和测试稳定后执行；LCM 细节随 I4 更新。

### 任务

- [ ] 更新 `docs/API.md`：接口签名、`BlockingIoConfig`、状态字段、错误码、启动/停止契约、普通调度默认值。
- [ ] 更新 `docs/MIGRATION.md`：从裸 `std::thread`/`std::jthread`、线程池常驻 lambda、RT callback 内阻塞 poll 迁移到 I/O worker。
- [ ] 更新 `README.md` 与 `README_zh.md` 的能力边界和入口链接；不把它宣传为硬实时 transport。
- [ ] 保持 [阻塞 I/O 执行器扩展设计](../design/blocking_io_executor.md) 为决策依据，并在实现完成后标注已落地/未落地部分。
- [ ] 为每个公开状态与异常路径补充 API 注释和可搜索的术语：worker、wakeup、bounded wait、state age、transport error。

### 验收

- [ ] 签名、默认值、错误码与测试一致；不存在仍要求读者猜测的 shutdown 语义。
- [ ] 迁移文档明确哪些旧模式可保留，哪些模式会占用线程池或破坏 RT 周期。

---

## 阶段 D2：使用手册和网站更新计划

网站采用当前 VitePress 双语结构。内容更新遵循现有设计约束：保持文档型、任务导向、无营销式
页面重构；使用现有主题、导航与版式，不新增装饰性页面或视觉组件。每个公开示例都必须来自编译
并 smoke-test 的源码，不能在中英文页面维护两份手工复制的 C++ 代码。

### 页面与导航

- [ ] 新增中文页面 `website/zh/realtime-and-communication/blocking-io-workers.md`。
  - [ ] 说明何时使用 I/O worker，何时使用 RT loop 或普通 async。
  - [ ] 说明 `wakeup()`、有限等待和“不能仅靠 stop token”的原因。
  - [ ] 给出状态/命令/诊断三类数据流选型表，并说明当前通信组件的锁与实时边界。
  - [ ] 给出启动、状态 age、断链和应用级安全关闭决策，而不是仅罗列 API。
- [ ] 新增英语对应页面 `website/en/realtime-and-communication/blocking-io-workers.md`，使用同一示例、命令与 API 名称。
- [ ] 更新中英文 `realtime-and-communication/index.md`：在 RT control 之后增加 I/O worker 页面入口，明确它是 RT 之外的专属 transport 模型。
- [ ] 更新 `website/.vitepress/config.mjs` 两个 locale 的“实时与通信” sidebar；中文和英文路由同步加入该页。
- [ ] 更新 `website/translation-status.md`：新增页面后保持该专题的 published 对应关系；未完成英文翻译时应显式标记 `Needs translation`，不发布空页面。
- [ ] 更新 `website/en/maintenance.md` 与中文对应维护页的“Real-time and communication”事实源：增加 blocking I/O public headers、测试和新教程编号。

### 教程与可运行事实源

- [ ] 新增编译/烟测示例，建议为 `examples/tutorial/12_blocking_io_worker.cpp`，使用不依赖 LCM 的 fake transport：
  - [ ] fake worker 在受控条件变量或 pipe/eventfd 上阻塞。
  - [ ] 演示 facade 注册、启动、向 `LatestMailbox`/有界 channel 发布、读取状态、request stop/wakeup/join。
  - [ ] 验证状态计数和 clean shutdown，不以 sleep 作为正确性的唯一依据。
- [ ] 在 `examples/tutorial/CMakeLists.txt` 注册示例与 CTest smoke test；命名和现有 tutorial 编号保持一致。
- [ ] 网站页面用 VitePress `<<< @` 嵌入已编译示例的最小连续行段，并链接完整源码；不要将 LCM SDK 设为文档构建的前置条件。
- [ ] 如 I4 的可选 LCM 集成测试已经存在，在页面中单独标注为“部署适配参考”，与无依赖教程区分，避免用户误以为核心库捆绑 LCM。
- [ ] 更新中英文 tutorial index，仅在该示例构建、运行和定位已稳定时加入；否则保留在专题页，不打乱基础教程序列。

### 既有页面的交叉更新

- [ ] 更新中英文 `realtime-control.md`：明确阻塞 `poll`/`handleTimeout` 不属于 `cycle_callback`，链接 I/O worker 页面。
- [ ] 更新中英文 `guides/choosing-communication.md`：增加“外部 transport 接收”决策分支，强调先按消息语义选 mailbox/channel，再选择运行线程模型。
- [ ] 更新中英文 `realtime-and-communication/observability.md` 与 `capacity-and-alerting.md`：区分 transport health、`CommStats`、状态 age 和 RT deadline，给出组合告警原则。
- [ ] 更新中英文 `guides/migrating-existing-threads.md`、`guides/concurrency-antipatterns.md` 与 `guides/production-readiness.md`：覆盖裸 jthread、线程池常驻 loop、RT 内阻塞 I/O、无 wakeup shutdown 的迁移与检查。
- [ ] 更新中英文 `tutorial/complete-robot-pipeline.md`：将 I/O worker 置于传感器/网络 ingress 边界，保留现有 pipeline 示例的教学范围并说明生产替换点。
- [ ] 更新中英文 API reference 入口页：链接 `docs/API.md` 的新 I/O executor 章节，而不在网页重复完整签名表。

### 网站验证

- [ ] 运行网站构建和链接检查，确认 base `/executor/` 下中英文新路由、sidebar、语言切换和交叉链接有效。
- [ ] 检查窄屏与宽屏下的长 API 名、表格和 code block 不溢出；不改变既有主题或页面信息架构。
- [ ] 确认每个 `<<< @` 指向存在的源码和稳定行段；示例修改时同步复查页面片段。
- [ ] 执行新教程的 CTest smoke test；示例变更时运行相关 runtime/communication 测试。
- [ ] 在 release 文档检查中确认版本范围、API、迁移材料、教程、中文页、英文页和 translation status 同步。

### 手册完成标准

- [ ] 用户能判断“这是固定周期控制、有限异步工作，还是专属可阻塞 I/O”。
- [ ] 用户知道 `stop_token` 不能独自中断底层 I/O，并能选择 wakeup fd 或有限 timeout。
- [ ] 用户能把状态、命令和诊断分别接到正确的通信组件，并能观测 age、drop、error 和 shutdown。
- [ ] 用户不会从手册推导出未承诺的硬实时、无锁或自动重连保证。

---

## 建议合并顺序

1. I0：公开类型和配置，先评审 API 边界。
2. I1：独立执行器与严格停止单测，先证明可 join。
3. I2：Manager/Facade 集成及 shutdown 并发安全。
4. I3：通信数据面和状态新鲜度测试。
5. I4：LCM/fd 适配器与实际部署测量，保持可选依赖。
6. I5：安全关闭、观测和硬件基线。
7. D1：API、迁移、README 与设计状态同步。
8. D2：教程、双语网站、导航和链接检查。

每阶段应以独立可回滚提交合入。D2 不得先于 I2 的公开 API 与 I1 的停止契约测试发布；LCM-specific
内容不得先于 I4 的可选集成测试和目标环境验证发布。

## 风险与待决项

- [ ] 确认目标 LCM 版本是否公开可 poll 的 fd；若没有，明确有限 `handleTimeout()` 上界并在部署中测量。
- [ ] 确认支持平台及 wakeup 原语：Linux eventfd、POSIX pipe、Windows `CancelIoEx` 等；接口保持平台无关。
- [ ] 明确 `LatestMailbox`/`DoubleBuffer` 在目标 RT 预算内的锁与复制成本；若不可接受，另立经过测量的 RT snapshot 扩展，不在 I/O executor 内绕过通信组件。
- [ ] 明确 I/O worker 的 ready 定义：transport 已打开、已订阅，还是已收到第一帧；它决定 startup timeout 的业务含义。
- [ ] 明确每类消息的丢弃策略、最大允许状态 age、断链降级动作和安全命令发送责任。
- [ ] 评估全局 name registry 是否作为本计划的一部分落地；若延后，必须有跨 registry 冲突测试。
- [ ] 在真实硬件上测量 affinity/IRQ/优先级调优收益前，不暴露或默认开启 FIFO。
