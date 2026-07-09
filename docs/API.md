# Executor API 使用说明

本文档说明 `executor` 库的主要 API、配置与类型，便于集成与扩展。完整接口定义见头文件 `include/executor/`。

---

## 1. 包含头文件

对外仅需包含 Facade 头文件：

```cpp
#include <executor/executor.hpp>
```

该头文件已包含 `config.hpp`、`types.hpp`、`interfaces.hpp`、`executor_manager.hpp`。使用 GPU API 时需启用 `EXECUTOR_ENABLE_GPU` 并包含 GPU 相关头文件（见 [BUILD.md](BUILD.md) 构建选项）。

---

## 2. 核心类：Executor

### 2.1 获取实例

| 方式 | 说明 |
|------|------|
| `Executor::instance()` | 单例，使用全局 `ExecutorManager`，进程内共享 |
| `Executor()` | 实例化模式，创建独立 `ExecutorManager`，资源隔离 |

### 2.2 初始化与关闭

```cpp
bool initialize(const ExecutorConfig& config);  // 初始化默认异步执行器（线程池）
ExecutorResult initialize_ex(const ExecutorConfig& config);
void shutdown(bool wait_for_tasks = true);      // 关闭所有执行器
void wait_for_completion();                     // 最多等待 300s，超时记录 WaitTimeout
bool try_wait_for_completion(std::chrono::milliseconds timeout);
template<class Rep, class Period>
bool wait_for_completion_for(std::chrono::duration<Rep, Period> timeout);
WaitResult wait_for_completion_ex(std::chrono::milliseconds timeout);
bool is_idle() const;
CompletionStatus get_completion_status() const;
```

- **懒初始化**：若不调用 `initialize(config)`，首次提交任务时会使用默认配置自动初始化（不抛异常）。需要自定义线程数、队列容量等时，请在首次提交前显式调用 `initialize(config)`。
- **退出时自动关闭（单例）**：使用单例时，若未显式调用 `shutdown()`，进程退出时会自动关闭所有执行器。若需在退出前等待未完成任务完成，请在业务逻辑中显式调用 `shutdown(true)`。
- `shutdown(true)` 会先通过 facade 等待队列中任务完成后再退出；如果等待超过 `kDefaultWaitForCompletionTimeout`，会记录 `WaitTimeout` 诊断并走非等待关闭路径，避免假装全部完成。
- `wait_for_completion()` 使用公开常量 `executor::kDefaultWaitForCompletionTimeout`，当前为 300 秒；保留 `void` 签名以兼容旧调用方，但超时会记录 `FailureKind::WaitTimeout`。
- `try_wait_for_completion(timeout)` 返回 `true` 表示所有已提交异步任务在 `timeout` 内完成；返回 `false` 表示等待超时且仍有任务未完成。超时不是 panic，也不抛异常；调用方可继续通过 `get_failure_status().wait_timeout_count` 或 `get_recent_failures()` 观察。
- `wait_for_completion_for(timeout)` 是支持任意 `std::chrono::duration` 的 bool 入口；`wait_for_completion_ex(timeout)` 返回 `WaitResult`，其中包含 `completed`、`timed_out`、`timeout`、`message` 和 `CompletionStatus` 快照。
- `get_completion_status()` 返回默认异步执行器的完成状态快照，包括 `is_initialized`、`is_running`、`is_idle`、`active_tasks`、`queued_tasks`、`pending_tasks`、`completed_tasks` 和 `failed_tasks`；`is_idle()` 是其中 `is_idle` 的便捷入口。状态查询不会触发默认执行器懒初始化。
- `initialize_ex(config)` 返回 `ExecutorResult`，可区分 `AlreadyInitialized`、`AlreadyShutdown`、`InvalidConfig`、`StartFailed` 等原因；旧 `initialize()` 保持 `bool` 签名，并委托到 `_ex` 后只返回 `ok`。

**注意事项**：懒初始化后不可再通过 `initialize()` 更换配置（已初始化则返回 false）。atexit 使用 `shutdown(false)`，不等待未完成任务。避免在静态析构中使用 Executor。

---

## 3. 任务提交 API（线程池）

### 3.1 基本提交

```cpp
template<typename F, typename... Args>
auto submit(F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type>;
```

- 提交到默认线程池，返回 `std::future`。
- 支持任意可调用对象及参数，`future.get()` 获取返回值或异常。

### 3.2 优先级提交

```cpp
template<typename F, typename... Args>
auto submit_priority(int priority, F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type>;
```

- `priority`：`0`=LOW，`1`=NORMAL，`2`=HIGH，`3`=CRITICAL（对应 `TaskPriority`）。
- 高优先级任务优先被调度。

### 3.3 批量提交

```cpp
template<typename F>
std::vector<std::future<void>> submit_batch(const std::vector<F>& tasks);

template<typename F>
void submit_batch_no_future(const std::vector<F>& tasks);
```

- **`submit_batch`**：批量提交任务，返回 `std::future<void>` 列表，适合需要等待任务完成的场景。
- **`submit_batch_no_future`**：批量提交任务，不返回 future（fire-and-forget），省去逐个 future 的管理开销，实际性能以 benchmark 为准。

#### 性能特性

`submit_batch()` / `submit_batch_no_future()` 的目标是减少重复提交路径开销，但当前版本不承诺固定加速比。实际结果会随任务数量、任务体、线程数、硬件、系统负载和构建配置变化；某些轻量任务或小批量场景可能低于循环 `submit()`。需要性能结论时，请优先运行本地 benchmark。

| 场景 | 经验建议 | 推荐使用 |
|------|---------|---------|
| 单线程提交大量任务 | 先 benchmark；若无需逐个 future，可优先尝试批量提交 | `submit_batch_no_future` |
| 单线程提交少量任务 | 收益不稳定，通常无需专门批量化 | 循环 `submit()` 或按实测选择 |
| 多线程并发提交 | 批量准备成本和锁竞争收益会互相抵消 | 按本地实测选择，默认可用循环 `submit()` |

#### 适用场景

**✅ 推荐使用批量提交**：
- 单线程需要提交大量任务，并且本地 benchmark 显示批量路径更快
- 任务准备开销小（如简单的 lambda）
- 不需要立即获取每个任务的 future（使用 `submit_batch_no_future`）

**⚠️ 不推荐使用批量提交**：
- 多线程并发提交（每个线程准备任务列表的开销会抵消收益）
- 任务数量较少（< 100 个）
- 需要在提交过程中动态决定是否继续提交

#### 最佳实践

```cpp
// ✅ 推荐：单线程批量提交大量任务
Executor executor;

std::vector<std::function<void()>> tasks;
tasks.reserve(1000);  // 预分配内存

for (int i = 0; i < 1000; ++i) {
    tasks.push_back([i]() {
        process_data(i);
    });
}

// 使用无 future 版本；实际收益请以本地 benchmark 为准
executor.submit_batch_no_future(tasks);

// 或使用有 future 版本（如需等待完成）
auto futures = executor.submit_batch(tasks);
for (auto& f : futures) {
    f.wait();  // 等待所有任务完成
}
```

```cpp
// ⚠️ 不推荐：多线程并发批量提交
std::vector<std::thread> threads;
for (int t = 0; t < 4; ++t) {
    threads.emplace_back([&executor]() {
        std::vector<std::function<void()>> tasks;
        for (int i = 0; i < 1000; ++i) {
            tasks.push_back([i]() { process(i); });
        }
        // 每个线程准备任务列表的开销较大
        executor.submit_batch_no_future(tasks);
    });
}

// ✅ 推荐：多线程直接提交
std::vector<std::thread> threads;
for (int t = 0; t < 4; ++t) {
    threads.emplace_back([&executor]() {
        for (int i = 0; i < 1000; ++i) {
            executor.submit([i]() { process(i); });
        }
    });
}
```

#### 性能测试数据

> 复现元数据：数据日期 2026-07-09，结果来源 [docs/performance/batch_submit_baseline_2026-07-09.json](performance/batch_submit_baseline_2026-07-09.json)。benchmark commands: `cmake --build build --target benchmark_batch_scales benchmark_batch_submit_real benchmark_batch_submit_concurrent -j2`，`./build/tests/benchmark_batch_scales`，`./build/tests/benchmark_batch_submit_real`，`./build/tests/benchmark_batch_submit_concurrent`。commit: `2ea0c37`。CPU: 13th Gen Intel(R) Core(TM) i9-13900KF，32 logical CPUs。OS: Linux 6.8.0-124-generic x86_64。compiler: GCC 11.4.0。build type: Release。
>
> 说明：以下数据来自当前版本的一次本地 benchmark 运行，不构成固定性能承诺。不同 benchmark 的计时范围不同，不能混合成同一个加速比口径。

单线程提交路径耗时（`benchmark_batch_scales`，使用 `submit_batch_no_future`；只计提交调用耗时，完成等待在计时后）：

| 任务数 | 循环 submit | submit_batch_no_future | 实测加速比 |
|--------|-------------|------------------------|------------|
| 500    | 5528 μs     | 1152 μs                | 4.80x      |
| 1000   | 7901 μs     | 770 μs                 | 10.26x     |
| 2000   | 17487 μs    | 1291 μs                | 13.55x     |
| 5000   | 45182 μs    | 3272 μs                | 13.81x     |

单线程真实负载端到端耗时（`benchmark_batch_submit_real`，使用 `submit_batch`；计提交并等待所有 future）：

| 任务数 | 循环 submit | submit_batch | 实测加速比 |
|--------|-------------|--------------|------------|
| 1000   | 9 ms        | 3 ms         | 3.00x      |
| 5000   | 41 ms       | 16 ms        | 2.56x      |
| 10000  | 38 ms       | 33 ms        | 1.15x      |
| 50000  | 381 ms      | 290 ms       | 1.31x      |

多线程并发端到端耗时（`benchmark_batch_submit_concurrent`，使用 `submit_batch`；计并发提交并等待所有 future）：

| 线程数 | 每线程任务数 | 总任务数 | 循环 submit | submit_batch | 实测加速比 | 建议 |
|--------|-------------|----------|-------------|--------------|------------|------|
| 2      | 5000        | 10000    | 55 ms       | 39 ms        | 1.41x      | 按本地实测选择 |
| 4      | 2500        | 10000    | 43 ms       | 38 ms        | 1.13x      | 按本地实测选择 |
| 8      | 1250        | 10000    | 36 ms       | 39 ms        | 0.92x      | 默认可用循环 `submit()` |
| 16     | 625         | 10000    | 33 ms       | 34 ms        | 0.97x      | 默认可用循环 `submit()` |
| 16     | 5000        | 80000    | 423 ms      | 493 ms       | 0.86x      | 默认可用循环 `submit()` |
| 32     | 312         | 9984     | 31 ms       | 36 ms        | 0.86x      | 默认可用循环 `submit()` |

**结论**：批量提交是可选的提交路径优化，不是固定倍率性能承诺。单线程大批量、无需 future 的场景可优先 benchmark `submit_batch_no_future()`；多线程并发提交和轻量任务场景应以实测选择循环 `submit()` 或批量提交。

#### 空任务拒绝

底层 `ThreadPool::try_submit(std::function<void()>)`、`ThreadPool::try_submit_priority(...)` 和 `ThreadPool::try_submit_batch(...)` 会拒绝空的 `std::function<void()>`。单任务路径返回 `false`，带回调的 overload 会向回调传入 `std::invalid_argument("empty task")`；批量路径只要发现任意空任务就返回 `false`，并且不会部分提交同一批次中的其他任务。

`ThreadPoolExecutor` / `IAsyncExecutor` / `Executor` facade 的 future API 不会同步抛出该拒绝；`submit(empty_function)` 和包含空任务的 `submit_batch(...)` 会返回已经 ready 的 future，`future.get()` 抛 `std::invalid_argument("empty task")`。`Executor` facade 同时将该情况记录为 `SubmitRejected`。

### 3.4 软超时

`task_timeout_ms` 是线程池任务的**执行前软超时**。worker 准备执行任务时会检查 `now - submit_time`；若 elapsed >= timeout，则跳过该任务并将线程池内部 timeout 计数与 `TaskStatistics::timeout_count` 加 1。通过 `ThreadPool::submit()`、`Executor::submit()`、priority submit 或 batch submit 暴露的 `std::future` 会被显式置为异常状态，`future.get()` 抛 `executor::TimedOutException`（例如 `Task timed out after 100ms`），不会变成 `std::future_error(broken_promise)`。

```cpp
executor::ExecutorConfig config;
config.task_timeout_ms = 100;  // 100 ms soft timeout

auto& ex = executor::Executor::instance();
ex.initialize(config);
```

| 行为 | 结果 |
|------|------|
| 任务排队超时，执行前检测到 | 跳过执行，`timeout_count++`；若有 future，`future.get()` 抛 `TimedOutException` |
| 任务已经开始执行后超时 | 不强制中断，继续运行到任务自行返回 |
| `task_timeout_ms = 0` | 不检查超时（默认行为） |

C++ 没有安全的通用线程强杀机制，因此 soft timeout 不会终止执行中的任务。排队超时是独立观测事件：它增加 timeout 计数，但不增加 `fail_count` / `failed_tasks`。长耗时任务应在任务内部自行检查取消条件或 deadline。

### 3.5 任务背压

实时执行器的 `push_task()` 为兼容旧接口仍返回 `void`。新代码优先使用 `Executor` facade 的 `push_realtime_task()` / `try_push_realtime_task()`；需要底层逃生口时再直接使用 `IRealtimeExecutor::push_task_ex()`。

```cpp
if (!ex.try_push_realtime_task("can_rx", []() {
    read_can_frame();
})) {
    // 实时执行器不存在、未运行、空任务、队列满或对象池耗尽导致失败
}

auto status = ex.get_realtime_executor_status("can_rx");
if (status.dropped_task_count > 0) {
    // 可用于告警、扩容或降级
}
```

| API / 字段 | 说明 |
|------------|------|
| `Executor::push_realtime_task(name, task)` / `try_push_realtime_task(name, task)` | 推荐 facade 入口；失败同时通过返回值、failure event 和状态计数可见 |
| `push_task(std::function<void()>)` | 兼容旧接口，不返回入队结果；失败会累计到状态计数 |
| `push_task_ex(std::function<void()>) -> bool` | 底层逃生口，`true` 表示成功入队，`false` 表示任务被丢弃 |
| `dropped_task_count` | 累计丢任务数，覆盖未运行、空任务、对象池耗尽、队列满；不受 `enable_stats` 影响 |
| `rejected_not_running_count` / `rejected_empty_task_count` | 按未运行和空任务拆分的拒绝计数 |
| `pool_exhausted_count` / `queue_full_count` | 按对象池耗尽和队列满拆分的拒绝计数 |
| `failed_pushes` | 底层队列失败入队数，仅 `enable_stats=true` 时统计 |
| `peak_queue_size` / `queue_capacity` | 用于分析实时任务队列水位与背压比例 |

### 3.6 延迟与周期任务

> ⚠️ **API 范围提示**：`submit_delayed`、`submit_periodic`、`cancel_task` **仅在 `Executor` Facade 类（`include/executor/executor.hpp`）中提供**，**不属于** `IAsyncExecutor`、`IExecutor` 或 `ThreadPool` 的接口。用户直接对底层 `ThreadPool` 实例调用这些方法会编译失败。延迟与周期任务统一由 Facade 内部的 `ExecutorManager` 调度，底层 `ThreadPool` 不感知任务时间维度。

```cpp
template<typename F, typename... Args>
auto submit_delayed(int64_t delay_ms, F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type>;

std::string submit_periodic(int64_t period_ms, std::function<void()> task);
bool cancel_task(const std::string& task_id);
```

- `submit_delayed`：延迟 `delay_ms` 毫秒后执行，返回 `future`。
- `submit_periodic`：按 `period_ms` 周期重复执行，返回任务 ID。
- `cancel_task`：取消对应周期性任务。

---

## 4. 实时任务 API（专用线程）

### 4.1 注册、启动、停止

```cpp
bool register_realtime_task(const std::string& name,
                            const RealtimeThreadConfig& config);
ExecutorResult register_realtime_task_ex(const std::string& name,
                                         const RealtimeThreadConfig& config);
bool start_realtime_task(const std::string& name);
ExecutorResult start_realtime_task_ex(const std::string& name);
void stop_realtime_task(const std::string& name);
bool push_realtime_task(const std::string& name, std::function<void()> task);
bool try_push_realtime_task(const std::string& name, std::function<void()> task);
```

- 每个 `name` 对应一个专用实时线程，按 `RealtimeThreadConfig` 周期执行 `cycle_callback`。
- 先 `register_realtime_task`，再 `start_realtime_task`；`stop_realtime_task` 停止该线程。
- `register_realtime_task_ex` / `start_realtime_task_ex` 返回可诊断结果：空名或非法配置为 `InvalidConfig`，重复注册为 `DuplicateName`，启动不存在的实时执行器为 `NotFound`，重复启动为 `AlreadyInitialized`。

### 4.2 获取执行器与列表

```cpp
IRealtimeExecutor* get_realtime_executor(const std::string& name);
std::vector<std::string> get_realtime_task_list() const;
```

- `push_realtime_task` / `try_push_realtime_task`：推荐任务推送入口；失败返回 `false`，并写入 failure event / 状态计数。
- `get_realtime_executor`：高级逃生口，用于直接访问 `push_task_ex` 等底层操作；若不存在返回 `nullptr`。
- `get_realtime_task_list`：当前已注册的实时任务名称列表。

---

## 5. 无锁任务执行器 API

### 5.1 概述

`LockFreeTaskExecutor` 是高性能的无锁任务执行器，支持 **MPSC（多生产者单消费者）** 模式。通过无锁队列和 CAS 操作避免互斥锁开销，提供极低延迟和高吞吐。

**适用场景**：
- 高频日志收集（多线程写入日志）
- 实时事件处理（多个事件源）
- 传感器数据采集（多传感器并发）
- 多线程环境下的任务聚合
- 性能敏感的异步任务分发

**技术特性**：
- 支持多个线程并发调用 `push_task()`
- 单个消费者线程处理任务
- 使用 CAS (Compare-And-Swap) 保证线程安全
- 完全向后兼容单生产者场景

**限制**：
- 固定队列容量，满时提交失败
- 仅支持 `std::function<void()>` 任务
- 单消费者（不支持多消费者）

### 5.2 包含头文件

```cpp
#include <executor/lockfree_task_executor.hpp>
```

### 5.3 基本用法

#### 单生产者场景（SPSC）

```cpp
// 创建执行器（队列容量1024）
executor::LockFreeTaskExecutor exec(1024);

// 启动消费者线程
exec.start();

// 提交任务
bool success = exec.push_task([]() {
    // 任务逻辑
});

if (!success) {
    // 队列满，处理背压
}

// 停止执行器（会处理剩余任务）
exec.stop();
```

#### 多生产者场景（MPSC）

```cpp
executor::LockFreeTaskExecutor exec(4096);
exec.start();

// 多个线程可以安全地并发提交任务
std::vector<std::thread> producers;
for (int i = 0; i < 4; ++i) {
    producers.emplace_back([&exec, i]() {
        for (int j = 0; j < 1000; ++j) {
            exec.push_task([i, j]() {
                // 处理任务
                std::cout << "Thread " << i << " task " << j << "\n";
            });
        }
    });
}

for (auto& t : producers) {
    t.join();
}

exec.stop();
```

### 5.4 API 接口

```cpp
class LockFreeTaskExecutor {
public:
    explicit LockFreeTaskExecutor(size_t queue_capacity = 1024,
                                  size_t backoff_multiplier = 2,
                                  bool enable_stats = false);
    ~LockFreeTaskExecutor();

    bool start();                                    // 启动消费者线程；stop() 后不可再次启动
    void stop();                                     // 停止接收新任务、处理已接受任务并等待
    bool is_running() const;                         // 检查运行状态

    // 单任务提交（线程安全，支持多生产者并发）
    // stop() 开始后返回 false；未 start 前仍可用于预填充队列
    bool push_task(std::function<void()> task);

    // 批量提交（原子语义：全成或全败）
    // tasks: 任务数组指针；count: 数组长度；pushed: 实际入队任务数（输出）
    // 返回值：true = 全部入队（pushed == count）；
    //         false = 空输入、stop() 已开始、内部对象池耗尽或队列空间不足，
    //                 没有任务入队，pushed 保持 0
    bool push_tasks_batch(const std::function<void()>* tasks,
                          size_t count,
                          size_t& pushed);

    size_t pending_count() const;                    // 队列中待处理任务数（近似值）
    uint64_t processed_count() const;                // 已处理任务总数

    QueueStats get_queue_stats() const;              // 队列性能统计（需 enable_stats=true）

    // 异常观测与自定义处理（require enable_stats=true）
    // exception_count() 返回 get_queue_stats() 期间累积的 task 异常次数
    // set_exception_handler() 允许替换默认「记录到 stats + 忽略」的行为,
    // 改由用户回调处理(例如记录到全局 logger、计数、转发到 ThreadPool 兜底)
    // QueueStats 字段参考下方 5.5 节。
    uint64_t exception_count() const;
    uint64_t rejected_empty_count() const;            // 空任务提交拒绝次数
    void set_exception_handler(std::function<void(std::exception_ptr)> handler);
};
```

#### `push_tasks_batch` 详解

| 项目 | 说明 |
|------|------|
| 时间复杂度 | O(count)，一次性申请所有 TaskWrapper，组装后单次调用 exact batch 入队 |
| 线程安全 | 与 `push_task` 相同，线程安全，可多生产者并发调用 |
| 原子语义 | 返回 true 时 `pushed == count`；返回 false 时没有任务入队且 `pushed == 0`。队列空间不足时不会部分入队 |
| 返回 false 时机 | (a) `tasks == nullptr` 或任一 `tasks[i]` 为空；(b) `stop()` 已开始，执行器拒绝新任务；(c) 对象池（ObjectPool）容量不足以一次性分配 count 个 wrapper；或 (d) 队列剩余空间不足以容纳整个 batch。以上情况下都不会有任务入队，`pushed` 为 0 |
| 批量统计 | 每次成功的 `push_tasks_batch` 调用会令 `get_queue_stats().batch_pushes` 递增 1，`total_pushes` 递增 `count`（P-260623-004：与队列 batch 统计语义一致） |
| 空任务统计 | 空任务属于提交拒绝，不进入队列，不增加 `processed_count()` 或 `exception_count()`；可通过 `rejected_empty_count()` 或 `get_queue_stats().rejected_empty_count` 观察 |

#### 停止后的提交语义

`LockFreeTaskExecutor` 区分“从未启动”和“已停止”状态：从未调用 `start()` 前仍允许 `push_task()` / `push_tasks_batch()` 预填充队列；一旦 `stop()` 开始，新的提交会被拒绝并返回 `false`。`stop()` 会等待已经进入提交路径的生产者完成，再让消费者线程处理所有已接受任务并退出，因此 `stop()` 返回后不会有静默接受但无人消费的任务残留在队列中。

**典型用法：**

```cpp
executor::LockFreeTaskExecutor exec(4096);
exec.start();

// 准备批量任务
std::vector<std::function<void()>> tasks;
tasks.reserve(100);
for (int i = 0; i < 100; ++i) {
    tasks.push_back([i]() { process(i); });
}

// 批量提交，检查实际入队数
size_t pushed = 0;
bool ok = exec.push_tasks_batch(tasks.data(), tasks.size(), pushed);
if (!ok) {
    // 空输入、对象池耗尽、队列空间不足或 stop() 后拒绝；
    // 没有任何任务入队，需要等待、修正输入或降级处理
} else {
    // pushed == tasks.size()
}
```

### 5.5 性能特性（LockFreeTaskExecutor）

> 数据来源：[docs/performance/lockfree_task_executor_baseline.md](performance/lockfree_task_executor_baseline.md)（2026-03-13 性能基线，10,000 任务单生产者场景）。后续基线更新请同步刷新本表。

#### 单生产者性能（SPSC）

| 指标 | 值 |
|------|-----|
| 平均延迟 | 97.29 ns |
| P50 延迟 | 29.00 ns |
| P99 延迟 | 1,013 ns |
| 吞吐量 | 8,242,895 ops/s |

#### 多生产者性能（MPSC）

| 生产者数 | 总吞吐量 | 单生产者吞吐量 | P99延迟 | 效率 |
|---------|---------|---------------|---------|------|
| 1       | 535万/s | 535万/s       | 4322ns  | 100% |
| 2       | 389万/s | 194万/s       | 3971ns  | 36%  |
| 4       | 340万/s | 85万/s        | 7576ns  | 16%  |
| 8       | 232万/s | 29万/s        | 9316ns  | 5%   |
| 16      | 186万/s | 12万/s        | 32090ns | 2%   |

**性能说明**：
- 单生产者性能接近理论最优（P99延迟 4.3μs）
- 2个生产者性能下降可控（36% 效率，P99延迟 4.0μs）
- 4+ 生产者因 CAS 竞争导致效率显著下降
- 16生产者场景下延迟升至 32μs，效率降至 2%

> 数据来源：docs/performance/lockfree_mpsc_baseline.json，基准时间 2026-06-25 12:34 CST。测试环境：队列容量16384，持续1秒吞吐量测试。更新请运行 `build/tests/benchmark_lockfree_mpsc_full` 并解析输出重新生成 JSON。

### 5.6 最佳实践

#### ✅ 推荐做法

**1. 选择合适的生产者数量**
```cpp
// 推荐：1-2 个生产者
executor::LockFreeTaskExecutor exec(4096);

// ✅ 单生产者：最佳性能
std::thread producer([&]() {
    exec.push_task([]() { /* ... */ });
});

// ✅ 2个生产者：性能良好
std::thread p1([&]() { exec.push_task([]() { /* ... */ }); });
std::thread p2([&]() { exec.push_task([]() { /* ... */ }); });
```

**2. 合理设置队列容量**
```cpp
// 根据任务频率和处理速度设置
// 低频场景：1024-2048
executor::LockFreeTaskExecutor low_freq(1024);

// 高频场景：4096-16384
executor::LockFreeTaskExecutor high_freq(8192);

// 容量必须是 2 的幂（会自动调整）
```

**3. 正确处理队列满的情况**
```cpp
bool success = exec.push_task([]() { /* ... */ });
if (!success) {
    // 策略1：重试（适合关键任务）
    while (!exec.push_task([]() { /* ... */ })) {
        std::this_thread::yield();
    }

    // 策略2：丢弃（适合日志等非关键任务）
    // 直接忽略

    // 策略3：降级（适合有备选方案的场景）
    // 使用其他执行器或同步执行
}
```

**4. 避免在 lambda 中捕获悬空引用**
```cpp
std::atomic<int> counter{0};

// ❌ 错误：捕获局部变量引用
{
    int local_var = 42;
    exec.push_task([&local_var]() {
        // local_var 可能已被销毁！
        std::cout << local_var << "\n";
    });
}

// ✅ 正确：捕获值或使用全局/静态变量
exec.push_task([value = 42]() {
    std::cout << value << "\n";
});

// ✅ 正确：捕获 shared_ptr
auto data = std::make_shared<int>(42);
exec.push_task([data]() {
    std::cout << *data << "\n";
});

// ✅ 正确：使用原子变量
exec.push_task([&counter]() {
    counter.fetch_add(1);
});
```

**5. 确保正确的生命周期管理**
```cpp
// ✅ 正确：执行器在生产者之前创建，之后销毁
{
    executor::LockFreeTaskExecutor exec(1024);
    exec.start();

    std::thread producer([&]() {
        exec.push_task([]() { /* ... */ });
    });

    producer.join();
    exec.stop();  // 会处理剩余任务
}  // 执行器在这里销毁
```

#### ⚠️ 注意事项

**1. 避免过多生产者**
```cpp
// ❌ 不推荐：4+ 生产者效率低
std::vector<std::thread> producers;
for (int i = 0; i < 16; ++i) {  // 效率仅 1-2%
    producers.emplace_back([&]() {
        exec.push_task([]() { /* ... */ });
    });
}

// ✅ 推荐：使用线程池或批量提交
ThreadPoolConfig config;
config.min_threads = 16;
config.max_threads = 16;

ThreadPool pool;
pool.initialize(config);
pool.submit([&]() {
    // 单个线程批量提交
    for (int i = 0; i < 1000; ++i) {
        exec.push_task([]() { /* ... */ });
    }
});
```

**2. 避免在任务中执行耗时操作**
```cpp
// ❌ 错误：阻塞消费者线程
exec.push_task([]() {
    std::this_thread::sleep_for(std::chrono::seconds(1));  // 阻塞！
    // 或者执行 I/O、网络请求等耗时操作
});

// ✅ 正确：任务应该是轻量级的
exec.push_task([]() {
    // 快速处理，微秒级
    process_data();
});

// ✅ 正确：耗时操作使用其他执行器
exec.push_task([&async_exec]() {
    async_exec.submit([]() {
        // 在其他线程执行耗时操作
        std::this_thread::sleep_for(std::chrono::seconds(1));
    });
});
```

**3. 注意 `pending_count()` 的近似性**
```cpp
// ⚠️ 注意：pending_count() 返回近似值
size_t count = exec.pending_count();
// 在多生产者场景下，实际值可能略有不同
// 不要依赖精确值做关键决策
```

**4. 避免在析构函数中提交任务**
```cpp
class MyClass {
    executor::LockFreeTaskExecutor& exec_;
public:
    ~MyClass() {
        // ❌ 危险：析构时提交任务可能导致问题
        exec_.push_task([this]() {
            // this 可能已被销毁！
        });
    }
};
```

### 5.7 异常行为和故障排查

#### 常见问题

**1. 段错误（Segmentation Fault）**

**原因**：Lambda 捕获了悬空引用
```cpp
// ❌ 问题代码
void bad_example() {
    executor::LockFreeTaskExecutor exec(1024);
    exec.start();

    int local = 42;
    exec.push_task([&local]() {
        std::cout << local << "\n";  // local 可能已销毁
    });

    // 函数返回，local 被销毁
}
```

**解决方案**：
```cpp
// ✅ 方案1：按值捕获
exec.push_task([local]() {
    std::cout << local << "\n";
});

// ✅ 方案2：使用 shared_ptr
auto data = std::make_shared<int>(42);
exec.push_task([data]() {
    std::cout << *data << "\n";
});
```

**2. 任务丢失**

**原因**：队列满时未处理失败情况
```cpp
// ❌ 问题代码
exec.push_task([]() { /* ... */ });  // 忽略返回值
```

**解决方案**：
```cpp
// ✅ 检查返回值
if (!exec.push_task([]() { /* ... */ })) {
    // 处理失败：重试、丢弃或降级
}
```

**3. 性能下降**

**原因**：生产者过多导致 CAS 竞争
```cpp
// ❌ 问题代码：16个生产者，效率仅 1%
for (int i = 0; i < 16; ++i) {
    threads.emplace_back([&]() {
        exec.push_task([]() { /* ... */ });
    });
}
```

**解决方案**：
```cpp
// ✅ 减少生产者数量到 1-2 个
for (int i = 0; i < 2; ++i) {
    threads.emplace_back([&]() {
        exec.push_task([]() { /* ... */ });
    });
}
```

**4. 死锁或挂起**

**原因**：在任务中等待执行器停止
```cpp
// ❌ 问题代码
exec.push_task([&exec]() {
    exec.stop();  // 死锁！消费者线程等待自己
});
```

**解决方案**：
```cpp
// ✅ 在外部停止
exec.stop();
```

### 5.8 性能调优建议

1. **队列容量**：根据峰值任务频率设置，避免频繁队列满
2. **生产者数量**：优先使用 1-2 个生产者
3. **任务粒度**：保持任务轻量级（< 10 微秒）
4. **内存对齐**：队列容量设为 2 的幂以优化性能
5. **CPU 亲和性**：考虑将消费者线程绑定到特定 CPU 核心

详细示例见 [examples/lockfree_task_executor_example.cpp](../examples/lockfree_task_executor_example.cpp)。

---

## 6. 监控 API

```cpp
void enable_monitoring(bool enable);
void set_monitoring_sampling_rate(double rate);

AsyncExecutorStatus get_async_executor_status() const;
RealtimeExecutorStatus get_realtime_executor_status(const std::string& name) const;

TaskStatistics get_task_statistics(const std::string& task_type) const;
std::map<std::string, TaskStatistics> get_all_task_statistics() const;
```

- `enable_monitoring`：开启/关闭任务监控（默认可在 `ExecutorConfig::enable_monitoring` 配置）。
- `set_monitoring_sampling_rate`：设置监控采样率（0.0–1.0），1.0 表示每次任务都采样，较低值可减少监控开销。
- `get_async_executor_status`：线程池名称、运行状态、活跃/完成/失败任务数、队列大小、平均任务时间等。
- `get_realtime_executor_status`：实时线程名称、运行状态、周期、周期计数、超时计数、平均/最大周期时间等。
- `get_task_statistics` / `get_all_task_statistics`：按 `task_type` 或全部的成功/失败/超时次数及执行时间统计。

---

## 7. 配置与类型

### 7.0 Facade 哲学：默认即最优，失败可观察

executor 库遵循以下原则 (P019 三阶段 + P019C companion):

1. **默认即最优** — 零配置用户拿到平台/负载下最好的行为
2. **自动决策** — 库在内部探测环境（`hw_concurrency`、timer slack）选最优路径
3. **自动降级可诊断** — 平台探测或系统级调优不可用时退到安全默认，不把调优失败伪装成任务失败
4. **任务失败可观察** — 任务异常、提交拒绝、实时队列丢任务、超时等运行时失败必须通过 `future`、返回值、状态计数或监控统计暴露；调用方可以选择不处理，但库不应让失败无迹可寻
5. **用户覆盖** — 显式设的非默认/非空值永远保留

实现：

- `ThreadPoolConfig.min_threads` / `max_threads` = 0（sentinel，自适应）
- `ThreadPoolConfig.enable_work_stealing` = `true`（默认开）
- `ThreadPoolConfig.cpu_affinity` 空 → auto-allocate [0..hw-1]
- `RealtimeThreadConfig.enable_memory_lock` = `true`（尽力调用 mlockall；不可用或权限不足时安全回退）
- `RealtimeThreadConfig.timer_slack_ns` = 1（尽力设置 1 ns；不可用或权限不足时安全回退）
- `RealtimeThreadConfig.cpu_affinity` 空 → bind core 0（hw >= 2）
- `RealtimeThreadConfig.thread_priority` = 0 → 自适应按 `cycle_period_ns` 建议
- `task_timeout_ms > 0`: 软超时 (执行前 skip + 记录 timeout_count; future 抛 `TimedOutException`; 不计入 fail_count; C++ 无安全 kill 机制, 执行中不强制中断)

### 7.1 ExecutorConfig / ThreadPoolConfig（线程池配置）

用于 `Executor::initialize()` / `ExecutorConfig` / `ThreadPoolConfig`：

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `min_threads` | `size_t` | `0` | 0 = 自适应 sentinel；`ExecutorManager::initialize` 时按 `hw_concurrency` 计算（min 2） |
| `max_threads` | `size_t` | `0` | 0 = 自适应 sentinel；默认 hw；探测失败退到 (2, 4) |
| `queue_capacity` | `size_t` | `1000` | 任务队列容量 |
| `thread_priority` | `int` | `0` | 线程优先级（Linux SCHED_FIFO 1–99，Windows `SetThreadPriority`） |
| `cpu_affinity` | `std::vector<int>` | 空 | 空 = 自适应 sentinel；`ExecutorManager` 自动填 [0..hw-1]；显式设值保留 |
| `task_timeout_ms` | `int64_t` | `0` | > 0: 软超时 (执行前 check elapsed >= timeout 则 skip + 记录 timeout_count; 暴露的 future 抛 `TimedOutException`; 不计入 fail_count; 0 = 不超时; 注意: 执行中不强制中断, C++ 无安全 kill 机制) |
| `enable_work_stealing` | `bool` | `true` | 无锁工作窃取；`max_threads == 1` 时自动关；-10.7% 性能退化关闭 |
| `enable_monitoring` | `bool` | `true` | 是否启用监控 |

内部动态 resize 扩容时，新增 worker 的负载元数据会重置为零负载，并将 `last_update` 初始化为当前 `std::chrono::steady_clock::now()`。

### 7.2 RealtimeThreadConfig（实时线程）

用于 `register_realtime_task()`：

| 字段 | 类型 | 说明 |
|------|------|------|
| `thread_name` | `std::string` | 线程名称（Linux 下通过 `pthread_setname_np` 设置，便于 top/perf 识别） |
| `cycle_period_ns` | `int64_t` | 周期（纳秒），如 2 000 000 表示 2 ms |
| `thread_priority` | `int` | 线程优先级（如 SCHED_FIFO 1–99）；== 0 时按 `cycle_period_ns` 自适应建议（≤1 ms → 80，≤10 ms → 50，>10 ms → 0）；显式设值保留 |
| `cpu_affinity` | `std::vector<int>` | CPU 亲和性；空 = 自适应 sentinel，实时线程 start 时绑核 0（hw >= 2）；显式设值保留 |
| `cycle_callback` | `std::function<void()>` | 每周期执行的回调 |
| `cycle_manager` | `ICycleManager*` | 可选，外部周期管理器；默认 nullptr 使用内置周期 |
| `max_tasks_per_cycle` | `uint64_t` | 单周期内最多处理的任务数；`0` 表示不限（保留旧行为，但生产环境建议 > 0 以保周期确定性）；默认 64 |
| `enable_memory_lock` | `bool` | 是否尽力调用 `mlockall` 锁定内存（避免分页抖动）；默认 `true`（opt-out，不可用或权限不足时安全回退） |
| `timer_slack_ns` | `uint64_t` | Linux timer slack（纳秒）；默认 1（1 ns，尽力设置，不可用或权限不足时安全回退）；`0` = 显式 opt-out 保留内核默认 |

### 7.3 状态与统计类型

- **AsyncExecutorStatus**：`name`、`is_running`、`active_tasks`、`completed_tasks`、`failed_tasks`、`queue_size`、`avg_task_time_ms`。`failed_tasks` 表示底层异步执行器已执行并以失败结束的任务数；通过 `Executor` facade 提交的用户任务异常也会让 wrapper 重新抛出，因此会计入该字段，同时计入 facade 的 `ExecutorFailureStatus::task_exception_count`。执行前软超时使用独立 timeout 计数，不计入 `failed_tasks`。
- **ThreadPoolStatus**：`include/executor/config.hpp:67` 仍定义此结构（与 `AsyncExecutorStatus` 字段几乎重合），并且仍是底层 `ThreadPool::get_status()` 的返回类型；`ThreadPoolExecutor::get_status()` 会读取它并映射为 `AsyncExecutorStatus`。通过 `Executor` facade 或异步执行器编写的新代码优先使用 `AsyncExecutorStatus`；直接使用底层 `ThreadPool` 时仍应按 `ThreadPoolStatus` 处理。若未来要弃用或移除该类型，需要先提供替代的底层状态 API，并在声明处添加 deprecation 标记。
- **RealtimeExecutorStatus**：
  - `name` (std::string)：执行器名称。
  - `is_running` (bool)：是否运行中。
  - `cycle_period_ns` (int64_t)：配置周期（纳秒）。
  - `cycle_count` (int64_t)：累计周期计数。
  - `cycle_timeout_count` (int64_t)：超时周期计数。
  - `avg_cycle_time_ns` (double)：平均周期执行时间（纳秒）。
  - `max_cycle_time_ns` (double)：最大周期执行时间（纳秒）。
  - `dropped_task_count` (uint64_t)：累计丢任务数（队列满 + 对象池耗尽，**始终累计**，不受 `enable_stats` 影响；P-001 260615 引入的背压可见性核心指标，应作为告警依据）。
  - `failed_pushes` (uint64_t)：LockFreeQueue 失败入队数（仅 `enable_stats=true` 时由底层队列统计；与 `dropped_task_count` 的子集：仅含"队列满"那一部分）。
  - `peak_queue_size` (uint64_t)：队列峰值长度（仅 `enable_stats=true`）。
  - `queue_capacity` (uint64_t)：RT 无锁队列固定容量（用于 `dropped/queue_capacity` 比率分析）。
  - `rejected_not_running_count` (uint64_t)：未运行/已停止时拒绝的累计数。
  - `rejected_empty_task_count` (uint64_t)：空任务拒绝累计数。
  - `pool_exhausted_count` (uint64_t)：对象池耗尽拒绝累计数。
  - `queue_full_count` (uint64_t)：队列满拒绝累计数。
- **TaskStatistics**：`total_count`、`success_count`、`fail_count`、`timeout_count`、`total_execution_time_ns`、`max_`/`min_execution_time_ns`。执行前软超时增加 `timeout_count`，不增加 `fail_count`。
- **ExecutorFailureStatus**：`task_exception_count`、`submit_rejected_count`、`timeout_count`、`realtime_drop_count`、`gpu_failure_count`、`wait_timeout_count`、`tuning_fallback_count`、`total_count`。`wait_for_completion()` 或 `try_wait_for_completion(timeout)` 等待超时时记录 `FailureKind::WaitTimeout` 并增加 `wait_timeout_count`；这只表示等待动作超时，不表示任务被取消、panic 或抛异常。
- **ExecutorResult**：`ok`、`error_code`、`message`，用于 `initialize_ex`、`register_realtime_task_ex`、`start_realtime_task_ex`、`register_gpu_executor_ex`。常见 `ExecutorErrorCode`：`AlreadyInitialized`、`AlreadyShutdown`、`InvalidConfig`、`DuplicateName`、`NotFound`、`BackendUnavailable`、`StartFailed`、`PermissionDenied`。`_ex` 失败会写入 failure/diagnostic event，但配置错误不会计入 `task_exception_count`。
- **CompletionStatus**：`executor_name`、`is_initialized`、`is_running`、`is_idle`、`active_tasks`、`queued_tasks`、`pending_tasks`、`completed_tasks`、`failed_tasks`。由 `get_completion_status()` 和 `WaitResult::status` 返回；状态查询不会触发默认异步执行器懒初始化。
- **WaitResult**：`completed`、`timed_out`、`timeout`、`status`、`message`。由 `wait_for_completion_ex(timeout)` 返回；超时会记录 `FailureKind::WaitTimeout` 并保留当时的 pending 状态快照。
- **CycleStatistics**：`name`、`period_ns`、`cycle_count`、`timeout_count`、`avg_cycle_time_ns`、`max_cycle_time_ns`、`is_running`。由 `ICycleManager::get_statistics()` 返回。

### 7.4 TaskPriority

```cpp
enum class TaskPriority { LOW = 0, NORMAL = 1, HIGH = 2, CRITICAL = 3 };
```

与 `submit_priority(int priority, ...)` 的整型对应。

---

## 8. GPU 执行器 API（可选，需 EXECUTOR_ENABLE_GPU）

GPU 执行器与 CPU 执行器接口分离，通过 `Executor` 注册与提交 GPU kernel，详见 [GPU 执行器设计](design/gpu_executor.md)。

### 8.1 注册与任务提交

```cpp
bool register_gpu_executor(const std::string& name,
                            const gpu::GpuExecutorConfig& config);
ExecutorResult register_gpu_executor_ex(const std::string& name,
                                         const gpu::GpuExecutorConfig& config);

template<typename KernelFunc>
auto submit_gpu(const std::string& executor_name,
               KernelFunc&& kernel,
               const gpu::GpuTaskConfig& config)
    -> std::future<void>;
```

- `register_gpu_executor`：按 `config.backend` 创建并注册 GPU 执行器；当前支持 `GpuBackend::CUDA` 和 `GpuBackend::OPENCL`。对应后端还需在编译时启用 `EXECUTOR_ENABLE_CUDA` / `EXECUTOR_ENABLE_OPENCL`，并且运行时设备、驱动和平台可用；否则创建或启动会失败并返回 `false`。
- `register_gpu_executor_ex`：推荐在需要诊断时使用，可区分 `InvalidConfig`、`DuplicateName`、`BackendUnavailable` 和 `StartFailed`。例如未编译对应后端、SYCL/HIP 尚未实现、运行时创建失败都会返回 `BackendUnavailable` 或更具体的启动失败信息。
- `submit_gpu`：向指定 GPU 执行器提交 kernel；kernel 可为 `void()` 或 `void(void*)`（流句柄，CUDA 下为 `cudaStream_t`，OpenCL 下为 `cl_command_queue`）。

### 8.2 查询与状态

```cpp
IGpuExecutor* get_gpu_executor(const std::string& name);
std::vector<std::string> get_gpu_executor_names() const;
gpu::GpuExecutorStatus get_gpu_executor_status(const std::string& name) const;
```

### 8.3 GPU 执行器接口（IGpuExecutor）

通过 `get_gpu_executor(name)` 获取指针后，可调用：

- **内存**：`allocate_device_memory`、`free_device_memory`；`copy_to_device`、`copy_to_host`、`copy_device_to_device`（均支持异步与流 ID）
- **统一内存**：`allocate_unified_memory`、`free_unified_memory`、`prefetch_memory`（host / device 方向均可）
- **P2P 传输**：`copy_from_peer`（跨 GPU 设备对等拷贝）
- **批量执行**：`submit_kernels_batch`（一次性提交一组 kernel+config，返回等长 `std::vector<std::future<void>>`；关停时每个输入均保证返回一个 future，详见 P-001 commit）
- **流**：`create_stream`、`destroy_stream`、`synchronize_stream`、`add_stream_callback`
- **执行**：`submit_kernel(kernel, config)`（返回 `std::future<void>`）、`synchronize`、`wait_for_completion`
- **状态**：`get_name`、`get_device_info`、`get_status`、`start`、`stop`

### 8.4 配置与类型

- **GpuExecutorConfig**：`name`、`backend`（支持 CUDA/OpenCL；分别要求 `EXECUTOR_ENABLE_CUDA` / `EXECUTOR_ENABLE_OPENCL` 且运行时可用）、`device_id`、`max_queue_size`、`memory_pool_size`、`default_stream_count`、`enable_monitoring`、`enable_unified_memory`（启用 `allocate_unified_memory` 等统一内存 API，CUDA 后端需要 `EXECUTOR_ENABLE_CUDA` 且硬件支持 managed memory）。`backend` 默认是 `GpuBackend::CUDA`；需要自动选择时可先调用 `gpu::get_recommended_backend()`，推荐逻辑会优先可用 CUDA 设备，其次 OpenCL，最后回到 CUDA 默认值。`device_id` 必须非负；`ExecutorManager::create_gpu_executor` 会拒绝负值，直接构造 `OpenCLExecutor` 时也会记录无效配置并在 `start()` 阶段拒绝负 `device_id`，不会用负下标访问设备数组。
- **GpuTaskConfig**：`grid_size`、`block_size`、`shared_memory_bytes`、`stream_id`、`async`；可选 `priority`。`stream_id == 0` 表示默认流/队列；非 0 值必须来自 `create_stream()` 且尚未 `destroy_stream()`，负数、越界或已销毁的 `stream_id` 不会回退到默认流/队列，相关 copy/submit 操作会失败。
- **CUDA stream 生命周期**：CUDA 后端内部用引用计数 wrapper 管理 `cudaStream_t`。`destroy_stream(stream_id)` 会先从 stream 表中摘除该 slot，并标记旧 wrapper 已销毁；已经拿到旧 wrapper 的并发操作会在 wrapper 锁下完成或观察到销毁状态，不会在已销毁的裸 `cudaStream_t` 上继续调用 CUDA API。销毁后的 copy/prefetch/callback/P2P 操作返回 `false`；销毁后的 `submit_kernel`/`submit_kernels_batch` future 抛出 `gpu::InvalidStreamException`。后续 `create_stream()` 可复用已摘除的 slot，但销毁前已提交的任务仍绑定旧 wrapper，不会误用新 stream。
- **GpuDeviceInfo**：设备名称、后端、设备 ID、厂商、总/空闲内存、计算能力等
- **GpuExecutorStatus**：名称、运行状态、活跃/完成/失败 kernel 数、队列大小、平均 kernel 时间、内存使用、`last_error_message`（最近一次启动/运行失败原因；空表示无错误；CUDA/OpenCL kernel 异常和无效 stream_id 均会记录）等
- **GpuScheduler**：GPU 任务调度器，支持优先级队列与批量提交策略；可通过 `GpuScheduler::Config` 配置
- **KernelLaunchOptimizer**：自动调优 kernel 启动参数（grid/block 尺寸），减少 kernel 配置开销；`KernelLaunchOptimizer::Config` 可定制
- **TaskSchedulerOptimizer**：优化 GPU 任务调度顺序，提高流水线利用率；`TaskSchedulerOptimizer::Config` 可定制
- **TransferOptimizer**：优化主机↔设备数据传输（合并小传输、异步流水线）；`TransferOptimizer::Config` 可定制

多 GPU 设备间 P2P 拷贝（`copy_from_peer`）为实验性功能。示例见 [examples/gpu_basic.cpp](../examples/gpu_basic.cpp)、[examples/gpu_multi_device.cpp](../examples/gpu_multi_device.cpp)、[examples/gpu_opencl.cpp](../examples/gpu_opencl.cpp)。

### 8.5 GPU 设备查询 API

在创建 GPU 执行器前，可查询系统可用设备及推荐后端：

```cpp
#include <executor/gpu/device_query.hpp>

// 枚举所有 CUDA 设备
std::vector<gpu::GpuDeviceInfo> enumerate_cuda_devices();

// 枚举所有 OpenCL 设备
std::vector<gpu::GpuDeviceInfo> enumerate_opencl_devices();

// 枚举所有 GPU 设备（CUDA + OpenCL）
std::vector<gpu::GpuDeviceInfo> enumerate_all_devices();

// 获取推荐后端（NVIDIA GPU 优先 CUDA，AMD/Intel GPU 使用 OpenCL）
gpu::GpuBackend get_recommended_backend(int device_id = 0);
```

**使用示例**：

```cpp
// 查询所有设备
auto devices = executor::gpu::enumerate_all_devices();
for (const auto& dev : devices) {
    std::cout << "Device " << dev.device_id << ": "
              << dev.name << " (" << dev.vendor << ")\n"
              << "  Backend: " << (dev.backend == executor::gpu::GpuBackend::CUDA ? "CUDA" : "OpenCL") << "\n"
              << "  Memory: " << (dev.total_memory_bytes / 1024 / 1024) << " MB\n";
}

// 自动选择推荐后端
auto backend = executor::gpu::get_recommended_backend(0);
executor::gpu::GpuExecutorConfig config;
config.backend = backend;
config.device_id = 0;
exec.register_gpu_executor("gpu0", config);
```

命令行工具：`gpu_device_query` 示例程序可直接查询系统 GPU 设备。

---

## 9. ICycleManager 接口（可选周期管理器）

`ICycleManager` 是可选接口，用于为实时线程提供更精确的周期控制和监控。若不提供，executor 使用内置的简单周期实现（基于 `std::this_thread::sleep_until`）。

### 9.1 接口定义

```cpp
class ICycleManager {
public:
    virtual ~ICycleManager() = default;

    // 注册周期任务
    virtual bool register_cycle(const std::string& name,
                                int64_t period_ns,
                                std::function<void()> callback) = 0;

    // 启动周期任务
    virtual bool start_cycle(const std::string& name) = 0;

    // 停止周期任务
    virtual void stop_cycle(const std::string& name) = 0;

    // 获取周期统计信息（可选）
    virtual CycleStatistics get_statistics(const std::string& name) const = 0;
};
```

### 9.2 使用场景

**内置周期（默认）**：
- 使用 `std::this_thread::sleep_until` 实现简单周期控制
- 适合大多数场景，无需额外实现
- 配置 `RealtimeThreadConfig::cycle_manager = nullptr`（默认）

**ICycleManager（可选）**：
- 需要更精确的周期控制（如硬件定时器、RTOS 周期管理）
- 需要统一的周期监控和统计（多个实时线程共享同一周期管理器）
- 需要自定义周期超时检测和恢复策略
- 需要与外部周期管理系统集成

### 9.3 实现示例

以下示例实现一个基于 `sleep_until` 的简单周期管理器：

```cpp
#include <executor/executor.hpp>
#include <thread>
#include <mutex>
#include <unordered_map>

class SimpleCycleManager : public executor::ICycleManager {
public:
    struct CycleInfo {
        std::string name;
        int64_t period_ns = 0;
        std::function<void()> callback;
    };

    bool register_cycle(const std::string& name, int64_t period_ns,
                       std::function<void()> callback) override {
        std::lock_guard<std::mutex> lock(mutex_);
        cycles_[name] = {name, period_ns, std::move(callback)};
        stop_requested_[name] = false;
        return true;
    }

    bool start_cycle(const std::string& name) override {
        CycleInfo info;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = cycles_.find(name);
            if (it == cycles_.end()) {
                return false;
            }
            info = it->second;
            stop_requested_[name] = false;
        }

        // 在独立线程中运行周期循环
        std::thread cycle_thread([this, name, info]() {
            auto next_cycle_time = std::chrono::steady_clock::now();
            const auto period_ns = std::chrono::nanoseconds(info.period_ns);

            while (true) {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (stop_requested_[name]) {
                        break;
                    }
                }

                // 执行周期回调
                if (info.callback) {
                    info.callback();
                }

                // 等待下一个周期
                next_cycle_time += period_ns;
                std::this_thread::sleep_until(next_cycle_time);
            }
        });
        cycle_thread.detach();

        return true;
    }

    void stop_cycle(const std::string& name) override {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_requested_[name] = true;
    }

    executor::CycleStatistics get_statistics(const std::string& name) const override {
        executor::CycleStatistics stats;
        stats.name = name;
        // 可在此添加统计信息收集逻辑
        return stats;
    }

private:
    std::unordered_map<std::string, CycleInfo> cycles_;
    std::unordered_map<std::string, bool> stop_requested_;
    mutable std::mutex mutex_;
};
```

### 9.4 注入到实时线程配置

在 `RealtimeThreadConfig` 中设置 `cycle_manager` 指针：

```cpp
// 创建周期管理器实例（用户管理生命周期）
SimpleCycleManager cycle_manager;

// 配置实时任务
executor::RealtimeThreadConfig rt_config;
rt_config.thread_name = "can_channel_0";
rt_config.cycle_period_ns = 2000000;  // 2ms
rt_config.thread_priority = 99;
rt_config.cpu_affinity = {0};
rt_config.cycle_callback = []() {
    // 周期回调逻辑
    // 注意：当使用 ICycleManager 时，此回调由周期管理器调用
};
rt_config.cycle_manager = &cycle_manager;  // 注入周期管理器

// 注册并启动实时任务
auto& exec = executor::Executor::instance();
exec.register_realtime_task("can_channel_0", rt_config);
exec.start_realtime_task("can_channel_0");

// ... 使用实时任务 ...

// 停止实时任务（周期管理器会自动停止周期）
exec.stop_realtime_task("can_channel_0");

// 注意：cycle_manager 的生命周期需由用户管理，确保在使用期间有效
```

### 9.5 工作流程

1. **注册阶段**：`RealtimeThreadExecutor::start()` 调用 `cycle_manager->register_cycle(name, period_ns, callback)`，注册周期任务。
2. **启动阶段**：`RealtimeThreadExecutor::start()` 调用 `cycle_manager->start_cycle(name)`，周期管理器开始按周期调用回调。
3. **执行阶段**：周期管理器在每个周期调用 `callback`（即 `RealtimeThreadConfig::cycle_callback`），实时线程在此回调中执行周期逻辑。
4. **停止阶段**：`RealtimeThreadExecutor::stop()` 调用 `cycle_manager->stop_cycle(name)`，周期管理器停止周期循环。

### 9.6 注意事项

- **生命周期管理**：`cycle_manager` 指针的生命周期需由用户管理，确保在实时线程运行期间有效。
- **线程安全**：`ICycleManager` 的实现需保证线程安全（如使用互斥锁保护内部状态）。
- **统计信息**：`get_statistics()` 可用于监控周期执行时间、超时次数等，便于性能分析。
- **多实例共享**：一个 `ICycleManager` 实例可管理多个实时线程的周期，便于统一监控和管理。

### 9.7 完整示例

参见 [examples/realtime_can.cpp](../examples/realtime_can.cpp)，其中展示了 `SimpleCycleManager` 的完整实现和使用。

---

## 10. 底层接口（可选）

- **IAsyncExecutor**：异步执行器抽象（线程池实现），提供 `submit`、`submit_priority`、`get_status`、`start`、`stop`、`wait_for_completion`。
- **IRealtimeExecutor**：实时执行器抽象，提供 `start`、`stop`、`push_task`、`get_status`。
- **ExecutorManager**：管理默认异步执行器与实时执行器注册表；通常通过 `Executor` 间接使用，也可直接调用（见设计文档）。

---

## 11. 使用模式简述

- **单例**：`Executor::instance()` + `initialize`，同一进程内多模块共享线程池。
- **实例化**：`Executor ex; ex.initialize(config);`，独立实例，RAII 析构时释放执行器，适合多项目/多模块隔离。
- **实时场景**：`register_realtime_task` + `start_realtime_task`，在 `cycle_callback` 中做周期逻辑；与线程池之间通过无锁队列等交换数据（见示例 `realtime_can`）。
- **GPU 场景**：`register_gpu_executor` + `submit_gpu`，kernel 与内存/流由 `IGpuExecutor` 管理（见示例 `gpu_basic`、`gpu_multi_device`，设计 [gpu_executor.md](design/gpu_executor.md)）。

更多示例见 [examples/](examples/) 与 [设计文档](design/executor.md)。
