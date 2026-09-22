# Dependency-Driven Scheduling 设计（调度侧唤醒）

状态：评审中（v0.5.2 主线，主清单阶段 21）
关联：[项目任务清单](../todolists/todolist.md) 阶段 21、
[性能审查台账](../todolists/performance_audit_2026-09_plan.md) PA-6/P4、
[总量有界 admission](bounded_admission.md)、
[任务协作取消与定时句柄](task_cancellation_and_timers.md)

---

## 1. 背景与问题

`submit_after` / `submit_after_with_handle` 当前采用"先入队、worker 上干等"模型：

1. 提交时任务 wrapper **立即入队**执行器
   （`include/executor/executor.hpp` `submit_tracked_with_hook` 尾部
   `try_submit_task` / `try_submit_priority_task`），图节点标记 `DependencyBlocked`；
2. worker 取到 wrapper 后在 `task_graph_cv_.wait(lock, pred)` 上**无限期阻塞**
   （executor.hpp 任务包装器内，谓词检查依赖失败/依赖全成/自身取消请求），
   期间**占用一个 worker 线程**；
3. 每个依赖任务终态时 `mark_task_graph_succeeded` / `mark_task_graph_failed`
   执行 `task_graph_cv_.notify_all()`（src/executor/executor.cpp 共 4 处），
   **唤醒全部等待者**，各自在全局 `task_graph_mutex_` 下重新求值谓词。

由此产生三个问题：

- **worker 占用**：N 个依赖阻塞任务可耗尽线程池。默认 worker 数小于依赖图宽度时，
  被阻塞的 wrapper 占住 worker，而其依赖还排在队列后面，形成饿死/挂死窗口。
  现有代码仅对"提交被拒"这一种情况做了封口（`on_rejected`，见 executor.hpp
  `submit_with_rejection_observer` 注释自述的风险），worker 占用本身无解。
- **惊群唤醒**：每 terminal 一次 `notify_all`，等待者规模 O(依赖等待任务总数)，
  全部在单一 `task_graph_mutex_` 上串行重查——正是 performance_audit PA-6
  登记的热点。
- **吞吐损失**：阻塞等待的 worker 本可用于执行依赖任务；等待时间完全计入
  worker 占用时长，扭曲负载统计与扩缩容决策。

## 2. 目标

- 依赖未满足的 dependent 任务**从不入队、从不接触 worker**，停在调度侧
  parked 结构中；
- 依赖终态由调度侧级联解析（**定向唤醒**）：最后一个依赖成功时入队执行；
  任一依赖失败/被取消时**立即结算**依赖异常，future 直接就绪；
- `task_graph_cv_`（facade 中唯一的条件变量）整体退役，删除全部 4 处
  `notify_all`；
- 公开 API 零变化：`submit_after` / `submit_after_with_handle` / `when_all` /
  `request_task_cancel` / `submit_cancellable_after` 签名与语义保持兼容。

## 3. 非目标

- **不做图全局锁分片**（PA-6 备选方案一）：改造后 `task_graph_mutex_` 的持有
  时间缩为纯簿记（终态标记 + 级联解析 + 出队收集），不再有长时间持锁等待，
  分片收益需重新实测后另行立项。
- **不改 when_all 虚节点机制**：`WhenAll` 状态节点已在
  `resolve_task_graph_dependents_locked` 中级联解析，本设计复用同一骨架。
- **不引入新的公开 API**。

## 4. 复用的既有骨架

| 骨架 | 位置 | 作用 |
|---|---|---|
| 反向边 `task_graph_dependents_` | executor.cpp | 依赖终态时找到受影响 dependents |
| 级联解析 `resolve_task_graph_dependents_locked` | executor.cpp | WhenAll 传递结算，扩展为同时驱动 parked 出队 |
| `TaskDependencyManager`（add/mark_completed/remove/prune，含环检测） | task/task_dependency_manager.* | 依赖计数与生命周期，环检测继续把守提交路径 |
| `TaskCancellationState` 单一 phase CAS + completion sink | task_cancellation.hpp | 取消/超时/开始执行仲裁，parked 任务沿用同一仲裁 |
| `DependencyBlocked` 生命周期 | types.hpp | 监控语义保留，入队时补记 Queued |
| 有界 admission（"计数先于 future"不变式） | bounded_admission.md | parked 期间继续占额，释放时序逐路径保持 |

## 5. 设计

### 5.1 数据结构

```cpp
struct TaskGraphNode {
    TaskGraphState state = TaskGraphState::Pending;
    std::exception_ptr exception;
    std::string error_message;
    std::vector<std::string> dependencies;
    // 新增：未满足依赖计数（在 task_graph_mutex_ 下维护）
    size_t unmet_count = 0;
    // 新增：parked 提交载荷；仅 Pending 且 unmet_count>0 时非空
    std::shared_ptr<ParkedSubmission> parked;
};

struct ParkedSubmission {
    std::function<void()> wrapper;          // 既有 task_wrapper，原样
    std::function<void(std::exception_ptr)> on_timeout;
    std::optional<int> priority;
    std::shared_ptr<IAsyncExecutor> executor_snapshot;  // 提交时快照
};
```

parked 载荷放节点内而非独立 map：生命周期与图节点严格同步，终态即清空，
不需要额外的表和清理遍历；retention 修剪已保证"有 active dependent 的依赖
节点不被驱逐"，反向边机制原样覆盖 parked 场景。

### 5.2 提交路径（`submit_tracked_with_hook` 依赖图变体）

前置步骤不变：`allocate_task_handle` → `register_task_graph_dependencies`
（环检测/句柄有效性拒绝路径不变）→ admission → cancellation registry →
executor snapshot。变化仅在**入队决策**：

1. **依赖已全部 Succeeded**（提交时即满足）：走既有立即入队路径，零回归；
2. **存在依赖已 Failed**：`dependency_failure_locked` 取异常 → 结算 future、
   `mark_task_graph_failed`、release admission、finalize registry ——
   与现行 worker 侧 fast path 等价，但**不占 worker**；
3. **否则 parked**：节点保持 `Pending`，`unmet_count = 未终态依赖数`，
   存 `parked` 载荷，lifecycle 记 `DependencyBlocked`。**不入队**。
   queued soft timeout 计时器照旧武装（见 6.1 决策 D1）。

### 5.3 依赖终态级联（调度侧唤醒核心）

`mark_task_graph_succeeded` / `mark_task_graph_failed` 在锁内
`resolve_task_graph_dependents_locked(task_id)` 中扩展一段 parked 处理：

```
对 current 的每个 dependent：
  if (WhenAll 虚节点)  → 既有级联逻辑，原样
  if (dependent.parked 非空):
    if (current 失败):
      → dependent 结算失败：取 dependency_failure_locked 异常（reclassify）、
        mark_task_graph_failed(dependent)、settle future、release admission、
        finalize、complete_terminal；parked 清空；并入 terminal_ids 级联
    else:
      --unmet_count == 0 且无依赖失败
      → 收集进 ready 列表
锁释放后：
  对 ready 列表逐个：先在锁内复查 node.state == Pending 且 parked 非空
  （与取消/超时竞态的最后一道闸），取出 parked 载荷并清空，
  然后调用 executor_snapshot->try_submit_task / try_submit_priority_task
  → 拒绝则走既有 on_rejected（节点 Failed）
```

要点：

- **出队（executor submit）必须在 `task_graph_mutex_` 外执行**：wrapper 开始
  执行时会经 `mark_task_graph_running` 重新取该锁，持锁调用 submit 会形成
  锁序倒置风险；锁内只做"复查 + 取载荷 + 清 parked"，锁外做 submit。
- **exactly-once**：parked 载荷取出即清空，是入队动作的唯一凭据；并发侧的
  取消/超时路径同样在锁内清空 parked 并走各自结算，双方以"谁清空了 parked"
  线性化，future 结算继续由 `promise_ready` CAS 兜底恰好一次。
- **失败级联深度**：沿用现有迭代式 `ready_ids` 栈，无递归。
- **级联期间仍持锁**：级联只做图簿记与 future/promise 结算（内存操作），
  无阻塞调用；executor submit 已移出锁外。

### 5.4 取消 parked 任务

`request_task_cancel` 对 parked 任务的既有 queued-cancel 路径继续成立
（completion sink 立即满足 future、`mark_task_graph_failed`、finalize），
新增一件事：该路径已持 `task_graph_mutex_`（notify 处），补一句
`node.parked.reset()`。此后依赖终态级联到该节点时发现 parked 为空且状态
已终态，自然跳过。依赖任务被取消时对下游的唤醒，由
`mark_task_graph_failed` 内的级联天然覆盖（替代现行 `notify_all`）。

### 5.5 shutdown 与 drain

- `shutdown(true)`：`wait_for_completion_ex` 的在途计数**已包含**
  `DependencyBlocked` 任务（`record_in_flight_task_pending` 在
  allocate_task_handle 即登记），语义自动正确——依赖正常推进后 parked 逐个
  ready、入队、执行，等待自然收敛；worker 不再被阻塞任务占用后，收敛速度
  只取决于真实工作量。
- `shutdown(false)`：执行器队列可能被弃，parked 任务若依赖尚在飞行也永远
  不会 ready。shutdown 路径对**全部仍 parked 的节点**统一结算：以
  "executor shutting down" 异常 settle future、`mark_task_graph_failed`、
  release admission、finalize——与"不等待未完成任务、尽快结束"的既定语义
  （safety_net_design）一致。
- ready 出队时 executor snapshot 已失效：`try_submit_*` 返回 false → 既有
  `on_rejected`，节点 Failed，不新增路径。

### 5.6 `task_graph_cv_` 退役

唯一等待点是任务包装器内的 `task_graph_cv_.wait`；改造后包装器只在依赖
满足后执行，等待逻辑整体删除。4 处 `notify_all`
（mark_task_graph_succeeded / mark_task_graph_failed / when_all /
request_task_cancel）中前两处的唤醒职能由级联接管；when_all 与取消路径的
notify 在删除等待者后失去对象，一并移除。成员变量删除。

## 6. 语义决策记录

### D1 queued soft timeout 的计时起点 —— **维持现状：提交即起算**

超时定时器在提交时武装，parked 期间同样可触发
（`try_timeout_before_start` 的 phase CAS 在 parked 下同样成立）。
触发时除既有结算外，需在锁内清空 `node.parked` 并按失败级联下游。

*备选*：改为"ready 入队时起算"被否决——改变现行可观测语义
（"任务在预算内未开始执行"包含依赖等待期），且依赖挂起将完全无界。
维持现状零迁移成本；若下游反馈需要区分"图等待"与"队列积压"，可作为
后续可选项再立项。

### D2 shutdown(false) 对 parked 的处理 —— **失败结算**

见 5.5。所有 parked 节点以异常结算，不静默丢弃 future。

### D3 admission 占额 —— **parked 期间继续占额**

"已受理未终态即占额"语义不变，`max_in_flight_tasks` 天然覆盖
依赖图整体规模（这正是下游要的有界性）；"计数先于 future"不变式逐路径
保持：依赖失败结算、超时、取消、shutdown drain 四条 parked 结算路径均
先 `release_admission()` 再 settle future，与现行提交拒绝路径同序。

### D4 priority 与 executor snapshot —— **提交时定格，ready 时沿用**

parked 载荷保存提交时的 `priority` 与 executor snapshot，出队原样使用。
不重新解析执行器：提交与执行使用同一快照是现行语义（拒绝路径也依赖它），
ready 时"换一个当前执行器"会破坏心智模型。

## 7. 行为差异（相对现行实现）

| 场景 | 现行 | 改造后 |
|---|---|---|
| 依赖等待期 worker 占用 | 占用 1 个 worker | 不占用 |
| 依赖等待期任务生命周期 | DependencyBlocked（worker 上） | DependencyBlocked（parked）→ ready 时 Queued |
| 依赖失败时下游结算时机 | 依赖终态 notify 唤醒后，worker 求值 | 依赖终态级联即时结算 |
| 提交时依赖已全部成功 | 入队后 wrapper 复查直接通过 | 直接入队（等价） |
| 超时语义 | 提交起算，可在等待期触发 | 不变（D1） |
| `max_in_flight_tasks` | 含等待中任务 | 不变（D3） |
| 惊群 `notify_all` | 每 terminal 一次 | 删除 |

API.md / MIGRATION.md / 网站同步点：生命周期观测一节补"DependencyBlocked
不再占用 worker；ready 后补记 Queued"。

## 8. 测试与验收

功能面（由 Independent-Verification-Agent 编写与执行）：

1. 依赖顺序：宽依赖（N dependency → 1 dependent）、深链（链长 > worker 数）、
   菱形；结果与执行顺序断言；
2. **无 worker 占用回归**：worker 数 = 2，提交 2 个长任务 + 依赖它们的
   8 个 dependent + 一批普通任务——普通任务不得饿死（现行实现可复现饿死，
   作为对照基线）；
3. 失败级联：中游失败，下游全部以依赖异常结算且不占 worker；
   下游的下游级联正确；
4. parked 取消：依赖未满足时 `request_task_cancel`，future 即时就绪，
   依赖终态后不重复结算；
5. 超时：parked 期间触发 queued soft timeout，结算 TimedOut、下游级联失败；
6. when_all 混合：parked 任务作为 when_all 输入、when_all 作为 parked
   依赖，级联收敛；
7. admission：parked 占额、依赖失败/超时/取消/shutdown 四路径释放恰好一次、
   计数先于 future；
8. shutdown(true) 收敛含 parked 链；shutdown(false) 全部 parked 异常结算；
9. retention：parked dependent 存续期间其依赖节点不被 trim 驱逐。

并发与性能面：

- TSAN 全量 + 专项：级联 resolve 与取消/超时三方竞态的压力测试；
- PA-6 验收口径：tracked 提交/完成路径多 worker 争用基准改善；每 terminal
  唤醒数从 O(等待者) 降为 O(1)（定向入队）；
- 既有 `test_executor_task_graph` / `test_task_graph_rejected_dependency` /
  `test_task_cancellation` / `test_concurrent_stop_submit` 全量回归。

## 9. 实施切分

1. **PR-1**：parked 提交路径 + 依赖终态级联入队（含 §8 功能面 1/2/3）；
2. **PR-2**：取消/超时/shutdown 三条 parked 结算路径 + retention 交互
   （§8 功能面 4–9）；
3. **PR-3**：`task_graph_cv_` 退役、监控 Queued 补记、基准与 PA-6 验收数据、
   API/MIGRATION/网站同步。

每个 PR 独立可回滚；任何路径发现与"恰好一次结算"或"计数先于 future"
冲突，先停下评审，不以性能名义放宽正确性约束。
