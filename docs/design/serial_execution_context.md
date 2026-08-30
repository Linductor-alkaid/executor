# SerialExecutionContext 设计

## 范围

S2 提供一个不依赖第三方事件循环的 FIFO 串行上下文，以及
`Executor::submit_on`/`submit_on_with_handle` 纳管入口。提交首先经过默认
executor 的 admission、任务统计、失败记录与取消 registry，再由上下文单线程
按提交顺序执行 callable。

## 生命周期

`SerialExecutionContext` 析构或显式 `shutdown()` 后拒绝新任务，并排空已经接收
的排队回调。已开始的回调运行至返回；其异常通过返回的 future 传播。上下文不
拥有 `Executor`，两者可独立关闭。调用方应在关闭 executor 前停止上下文提交，
并在需要时先 `context.shutdown()` 收敛外部状态。

## 取消与监控

返回的 `TaskHandle` 是 facade 纳管任务句柄。排队阶段取消保证 callable 不执行，
future 以既有 `TaskCancelled` 异常完成；运行阶段取消仍是协作请求。上下文本身
不提供抢占，也不改变外部事件循环的线程或销毁语义。快照中的 admission、执行、
异常和取消计数来自 facade；上下文队列深度不作为独立 backend 状态承诺。

该 API 不绑定 asio strand。需要对象始终在同一外部 strand 上访问和销毁的 timer
仍属于 T2，继续使用应用侧 timer。

## 派发与结算结构（W0 冻结，2026-08-30）

输入：Mira 台账 EXE-20260830-002（多 worker 饥饿）与 EXE-20260830-003（栈同步对象
竞争）。两者同源于旧 facade wrapper 的结构：wrapper 在 worker 栈上创建
`mutex`/`condition_variable` 并等待串行 callback 完成后再返回。冻结形状为
**派发/结算分离**（计划 W0 候选 1）：

- 池 worker 只执行一个有界、非阻塞的发布任务：`try_begin_execution()` 仲裁后调用
  `context.post_reserved(ticket, callback)` 并立即返回。发布任务不等待 callback、
  不读取业务 future、不持有任何栈同步对象。
- 业务 promise、ready CAS 标志与 published 标志全部由 `shared_ptr` 拥有（稳定
  生命周期），callback 仅捕获 `shared_ptr`；wrapper 返回与 callback 尾部之间不再
  存在共享栈对象，竞争按构造消除。
- 业务 future 的结算方恰好一个，由共享 `promise_ready` CAS 仲裁：
  - 串行线程 callback：调用用户 callable → 任务图/取消计数终态化 → registry
    finalize → `promise` 置值/置异常。终态化先于 future 就绪，延续既有观测不变式。
  - 排队取消：`TaskCancellationState` completion sink 结算 `TaskCancelled` 并
    `abandon(ticket)`；queued soft timeout、提交拒绝、registry 耗尽路径同构。
  - 发布被拒（context shutdown）：发布任务以 `ExecutorStopping` 结算。
  - 派发任务被池丢弃（`shutdown(false)` 清队列）：捕获于发布任务的 TicketGuard
    析构时 `abandon(ticket)` 并以 `ExecutorStopping` 结算（仅当未发布且相位未
    终态，不与取消/超时结算竞争）。
- FIFO 语义不变：ticket 由 `SerialExecutionContext` 既有 `reserve`/`post_reserved`/
  `abandon` 协议维护；未发布（取消/超时/拒绝）路径释放 ticket，不阻塞后续
  ticket 的顺序释放。
- 公开签名不变：`submit_on`/`submit_on_with_handle` 返回类型、句柄取消能力与
  `TaskSubmission` 结构保持一致；句柄仍经取消 registry 纳管。

被否决的候选 2（tracked 机制外部结算模式）需要修改 `submit_tracked_with_hook`
正常路径的无 CAS `set_value`，侵入所有 tracked 提交的热路径，收益不优于候选 1。
