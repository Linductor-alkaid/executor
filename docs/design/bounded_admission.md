# 总量有界 Admission 设计（A0 冻结，2026-08-30）

输入：Mira 台账 EXE-20260830-001。默认异步提交缺少跨 scheduler 与 worker 本地
队列的总量有界 admission：`ExecutorConfig::queue_capacity` 只构造每 worker 本地
有界队列，本地队列满时任务回推到无容量上限的 `PriorityScheduler`，`try_submit`
仅在池停止或空任务时拒绝，过载表现为无界内存增长且调用方无法得到结构化拒绝。

## 冻结决策

### D-1 配置面

`ExecutorConfig::max_in_flight_tasks`（`size_t`，默认 `0`）：

- `0` 表示不启用（完全保持既有行为，提交热路径零额外开销）；
- 正值 N 表示该 `Executor` facade 实例的默认异步提交总量上限，计数范围是
  **已接纳但尚未结算**的提交（scheduler 全局队列 + worker 本地队列 + 正在执行）。

边界归属：上限属于 facade 实例，不属于底层 `ThreadPool`。直接使用
`ThreadPool`/`PriorityScheduler` 的调用方不受保护（API.md 标注）。与
`queue_capacity` 的关系：后者仍是每 worker 本地队列的构造参数与扩缩容阈值，
两者语义独立，任何一处不得宣称 `queue_capacity` 是总量背压边界。

运行期可调：`set_max_in_flight_tasks(size_t)` 调小不驱逐已接纳任务，只约束
后续提交；`get_max_in_flight_tasks()`/`get_in_flight_submissions()` 提供观测。

### D-2 计数与释放

- 接纳：提交路径上的单次 `fetch_add`，超过上限即回滚并拒绝（允许计数瞬时
  过冲，但接纳决策不超过 N）。
- 释放：每个提交恰好一次，终态集合 = 正常完成、任务异常、排队取消、执行前
  超时、提交拒绝（池停止/registry 耗尽等）、context shutdown 拒绝、池丢弃
  （`shutdown(false)` 清队列）。
- 释放载体：`AdmissionReleaser`（共享指针拥有的恰好一次释放器，内部
  `atomic_bool` CAS）。所有可能结算 future 的闭包（任务包装、completion sink、
  on_timeout、on_rejected、TicketGuard）都持有该释放器：
  - 显式路径在 **future 结算之前**调用 `release()`（延续 PR #177
    "计数先于 future"的观测不变式：future 就绪后查询 status 已看到最终计数）；
  - 析构兜底覆盖池丢弃等无人结算的路径（闭包销毁即释放）。
- 与取消 registry 的顺序：**先 admission，后 registry**。admission 拒绝的提交
  不占用 registry 槽位；admission 接纳后 registry 耗尽的路径在结算前同时释放
  admission。任何失败路径按获取的逆序释放，不产生泄漏窗口。

### D-3 拒绝分类

- `FailureKind` 追加 `CapacityExhausted`（枚举尾部追加，源码兼容）。
- `ExecutorFailureStatus` 追加 `capacity_exhausted_count`；`record_failure`
  的计数 switch 同步扩展。
- 新公开异常 `CapacityExhaustedException : std::runtime_error`
  （`include/executor/types.hpp`，与 `TimedOutException` 同区）。
- 拒绝行为：**不抛出**。对应 future 立即以 `CapacityExhaustedException` 就绪，
  failure 事件同步记录（kind=`CapacityExhausted`），与 stopping
  （`ExecutorStopping`）、invalid input（`std::invalid_argument`）经异常类型与
  failure kind 双通道可区分。

### D-4 覆盖路径（一期）

经如下入口提交的默认异步任务全部计数：`submit`、`submit_with_handle`、
`submit_priority`、`submit_after(_with_handle)`、`submit_cancellable{,_priority,
_after}`、`submit_batch`、`submit_batch_priority`、`submit_batch_no_future`、
`submit_on(_with_handle)`。

- batch 语义：**逐任务独立接纳（部分接纳合法）**。被拒任务在返回的 futures
  向量中对应位置以 `CapacityExhaustedException` 就绪并各记一次 failure 事件；
  `submit_batch_no_future` 被拒任务仅记录事件。
- `submit_on_with_handle` 拒绝时释放 context ticket（既有 abandon 路径），
  不阻塞后续 FIFO。

明确不在一期（文档声明，不做静默）：`submit_delayed*`/`submit_periodic*`
（timer registry 派发路径）、GPU、realtime、Blocking I/O、lockfree。

### D-5 其他边界

- 不为 CRITICAL 优先级提供旁路；需要余量时由配置容量表达。
- shutdown 与并发 submit：池停止后的拒绝走既有 on_rejected 语义并释放容量；
  `shutdown(true)` drain 使计数归零（每个任务终态各自释放）。
- 扩缩容：admission 压力一期不作为 scale-up 信号，仅可经
  `get_in_flight_submissions()` 观测。
- 无任何无界 fallback 绕过：拒绝是唯一过载出口。

## 验收（对齐 Mira 台账上游验收标准）

1. 单 worker、总容量 N 时，第 N+1 个尚未结算的 submit 明确拒绝且 future 就绪。
2. 完成、任务异常、排队取消、执行前超时各释放一次容量（并发终态下恰好一次）。
3. shutdown 与并发 submit 不越过容量，也不留下未就绪 future。
4. failure/status 能区分 capacity rejection 与 stopping、invalid input。
5. `submit_on_with_handle` 的 context ticket 在 rejection 时被释放，后续 FIFO
   不阻塞。
6. 容量耗尽期间突发提交不产生 scheduler 无界堆积（内存有界），解除后恢复。
7. `max_in_flight_tasks == 0`（默认）时全部既有测试行为不变。
