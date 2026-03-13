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
void shutdown(bool wait_for_tasks = true);      // 关闭所有执行器
void wait_for_completion();                     // 等待已提交的异步任务完成
```

- **懒初始化**：若不调用 `initialize(config)`，首次提交任务时会使用默认配置自动初始化（不抛异常）。需要自定义线程数、队列容量等时，请在首次提交前显式调用 `initialize(config)`。
- **退出时自动关闭（单例）**：使用单例时，若未显式调用 `shutdown()`，进程退出时会自动关闭所有执行器。若需在退出前等待未完成任务完成，请在业务逻辑中显式调用 `shutdown(true)`。
- `shutdown(true)` 会等待队列中任务完成后再退出。

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
- **`submit_batch_no_future`**：批量提交任务，不返回 future（fire-and-forget），性能更高。

#### 性能特性

| 场景 | 性能提升 | 推荐使用 |
|------|---------|---------|
| 单线程提交 500+ 任务 | **5-16x** | ✅ `submit_batch_no_future` |
| 单线程提交 < 500 任务 | 2-5x | ✅ `submit_batch_no_future` |
| 多线程并发提交 | 0.6-1.2x | ⚠️ 使用循环 `submit()` |

#### 适用场景

**✅ 推荐使用批量提交**：
- 单线程需要提交大量任务（500+ 个）
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

// 使用无 future 版本，性能最佳（5-16x 加速）
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

单线程批量提交性能（使用 `submit_batch_no_future`）：

| 任务数 | 循环 submit | submit_batch_no_future | 加速比 |
|--------|-------------|------------------------|--------|
| 500    | 2757 μs     | 549 μs                 | 5.02x  |
| 1000   | 5307 μs     | 1246 μs                | 4.26x  |
| 2000   | 11003 μs    | 668 μs                 | 16.47x |

多线程并发提交性能（使用 `submit_batch`）：

| 线程数 | 每线程任务数 | 循环 submit | submit_batch | 加速比 |
|--------|-------------|-------------|--------------|--------|
| 2      | 5000        | 37 ms       | 37 ms        | 1.00x  |
| 4      | 2500        | 23 ms       | 38 ms        | 0.61x  |
| 8      | 1250        | 45 ms       | 38 ms        | 1.18x  |

**结论**：批量提交在单线程场景下性能提升显著，多线程场景下建议使用循环 `submit()`。

### 3.4 延迟与周期任务

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
bool start_realtime_task(const std::string& name);
void stop_realtime_task(const std::string& name);
```

- 每个 `name` 对应一个专用实时线程，按 `RealtimeThreadConfig` 周期执行 `cycle_callback`。
- 先 `register_realtime_task`，再 `start_realtime_task`；`stop_realtime_task` 停止该线程。

### 4.2 获取执行器与列表

```cpp
IRealtimeExecutor* get_realtime_executor(const std::string& name);
std::vector<std::string> get_realtime_task_list() const;
```

- `get_realtime_executor`：用于 `push_task` 等底层操作；若不存在返回 `nullptr`。
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
    explicit LockFreeTaskExecutor(size_t queue_capacity = 1024);
    ~LockFreeTaskExecutor();

    bool start();                                    // 启动消费者线程
    void stop();                                     // 停止并等待
    bool is_running() const;                         // 检查运行状态

    bool push_task(std::function<void()> task);      // 提交任务（线程安全）

    size_t pending_count() const;                    // 队列中待处理任务数（近似值）
    uint64_t processed_count() const;                // 已处理任务总数
};
```

### 5.5 性能特性

#### 单生产者性能（SPSC）

| 指标 | 值 |
|------|-----|
| 平均延迟 | 284 ns |
| P50 延迟 | 171 ns |
| P99 延迟 | 1,276 ns |
| 吞吐量 | 762万 ops/s |

#### 多生产者性能（MPSC）

| 生产者数 | 总吞吐量 | 单生产者吞吐量 | 效率 |
|---------|---------|---------------|------|
| 1       | 762万/s | 762万/s       | 100% |
| 2       | 528万/s | 264万/s       | 35%  |
| 4       | 360万/s | 90万/s        | 12%  |
| 8       | 281万/s | 35万/s        | 5%   |

**性能说明**：
- 单生产者性能接近理论最优（仅 3% CAS 开销）
- 2个生产者性能下降可控（35% 效率）
- 4+ 生产者因 CAS 竞争导致效率显著下降

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
ThreadPool pool(16);
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

AsyncExecutorStatus get_async_executor_status() const;
RealtimeExecutorStatus get_realtime_executor_status(const std::string& name) const;

TaskStatistics get_task_statistics(const std::string& task_type) const;
std::map<std::string, TaskStatistics> get_all_task_statistics() const;
```

- `enable_monitoring`：开启/关闭任务监控（默认可在 `ExecutorConfig::enable_monitoring` 配置）。
- `get_async_executor_status`：线程池名称、运行状态、活跃/完成/失败任务数、队列大小、平均任务时间等。
- `get_realtime_executor_status`：实时线程名称、运行状态、周期、周期计数、超时计数、平均/最大周期时间等。
- `get_task_statistics` / `get_all_task_statistics`：按 `task_type` 或全部的成功/失败/超时次数及执行时间统计。

---

## 7. 配置与类型

### 6.1 ExecutorConfig（初始化线程池）

用于 `Executor::initialize()`：

| 字段 | 类型 | 说明 |
|------|------|------|
| `min_threads` | `size_t` | 最小线程数，默认 4 |
| `max_threads` | `size_t` | 最大线程数，默认 16 |
| `queue_capacity` | `size_t` | 任务队列容量，默认 1000 |
| `thread_priority` | `int` | 线程优先级（如 Linux -20..19） |
| `cpu_affinity` | `std::vector<int>` | CPU 亲和性（核心编号） |
| `task_timeout_ms` | `int64_t` | 任务超时（毫秒），0 表示不超时 |
| `enable_work_stealing` | `bool` | 是否启用工作窃取 |
| `enable_monitoring` | `bool` | 是否启用监控，默认 true |

### 6.2 RealtimeThreadConfig（实时线程）

用于 `register_realtime_task()`：

| 字段 | 类型 | 说明 |
|------|------|------|
| `thread_name` | `std::string` | 线程名称 |
| `cycle_period_ns` | `int64_t` | 周期（纳秒），如 2 000 000 表示 2 ms |
| `thread_priority` | `int` | 线程优先级（如 SCHED_FIFO 1–99） |
| `cpu_affinity` | `std::vector<int>` | CPU 亲和性 |
| `cycle_callback` | `std::function<void()>` | 每周期执行的回调 |
| `cycle_manager` | `ICycleManager*` | 可选，外部周期管理器；默认 nullptr 使用内置周期 |

### 6.3 状态与统计类型

- **AsyncExecutorStatus**：`name`、`is_running`、`active_tasks`、`completed_tasks`、`failed_tasks`、`queue_size`、`avg_task_time_ms`。
- **RealtimeExecutorStatus**：`name`、`is_running`、`cycle_period_ns`、`cycle_count`、`cycle_timeout_count`、`avg_cycle_time_ns`、`max_cycle_time_ns`。
- **TaskStatistics**：`total_count`、`success_count`、`fail_count`、`timeout_count`、`total_execution_time_ns`、`max_`/`min_execution_time_ns`。

### 6.4 TaskPriority

```cpp
enum class TaskPriority { LOW = 0, NORMAL = 1, HIGH = 2, CRITICAL = 3 };
```

与 `submit_priority(int priority, ...)` 的整型对应。

---

## 8. GPU 执行器 API（可选，需 EXECUTOR_ENABLE_GPU）

GPU 执行器与 CPU 执行器接口分离，通过 `Executor` 注册与提交 GPU kernel，详见 [GPU 执行器设计](design/gpu_executor.md)。

### 7.1 注册与任务提交

```cpp
bool register_gpu_executor(const std::string& name,
                            const gpu::GpuExecutorConfig& config);

template<typename KernelFunc>
auto submit_gpu(const std::string& executor_name,
               KernelFunc&& kernel,
               const gpu::GpuTaskConfig& config)
    -> std::future<void>;
```

- `register_gpu_executor`：按配置创建并注册 GPU 执行器（当前支持 `GpuBackend::CUDA`）。
- `submit_gpu`：向指定 GPU 执行器提交 kernel；kernel 可为 `void()` 或 `void(void*)`（流句柄，CUDA 下为 `cudaStream_t`）。

### 7.2 查询与状态

```cpp
IGpuExecutor* get_gpu_executor(const std::string& name);
std::vector<std::string> get_gpu_executor_names() const;
gpu::GpuExecutorStatus get_gpu_executor_status(const std::string& name) const;
```

### 7.3 GPU 执行器接口（IGpuExecutor）

通过 `get_gpu_executor(name)` 获取指针后，可调用：

- **内存**：`allocate_device_memory`、`free_device_memory`；`copy_to_device`、`copy_to_host`、`copy_device_to_device`（均支持异步与流 ID）
- **流**：`create_stream`、`destroy_stream`、`synchronize_stream`、`add_stream_callback`
- **执行**：`submit_kernel(kernel, config)`（返回 `std::future<void>`）、`synchronize`、`wait_for_completion`
- **状态**：`get_name`、`get_device_info`、`get_status`、`start`、`stop`

### 7.4 配置与类型

- **GpuExecutorConfig**：`name`、`backend`（如 CUDA/OpenCL）、`device_id`、`max_queue_size`、`memory_pool_size`、`default_stream_count`、`enable_monitoring`
- **GpuTaskConfig**：`grid_size`、`block_size`、`shared_memory_bytes`、`stream_id`、`async`；可选 `priority`
- **GpuDeviceInfo**：设备名称、后端、设备 ID、厂商、总/空闲内存、计算能力等
- **GpuExecutorStatus**：名称、运行状态、活跃/完成/失败 kernel 数、队列大小、平均 kernel 时间、内存使用等

多 GPU 设备间 P2P 拷贝（`copy_from_peer`）为实验性功能。示例见 [examples/gpu_basic.cpp](../examples/gpu_basic.cpp)、[examples/gpu_multi_device.cpp](../examples/gpu_multi_device.cpp)、[examples/gpu_opencl.cpp](../examples/gpu_opencl.cpp)。

### 7.5 GPU 设备查询 API

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

### 8.1 接口定义

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

### 8.2 使用场景

**内置周期（默认）**：
- 使用 `std::this_thread::sleep_until` 实现简单周期控制
- 适合大多数场景，无需额外实现
- 配置 `RealtimeThreadConfig::cycle_manager = nullptr`（默认）

**ICycleManager（可选）**：
- 需要更精确的周期控制（如硬件定时器、RTOS 周期管理）
- 需要统一的周期监控和统计（多个实时线程共享同一周期管理器）
- 需要自定义周期超时检测和恢复策略
- 需要与外部周期管理系统集成

### 8.3 实现示例

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

### 8.4 注入到实时线程配置

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

### 8.5 工作流程

1. **注册阶段**：`RealtimeThreadExecutor::start()` 调用 `cycle_manager->register_cycle(name, period_ns, callback)`，注册周期任务。
2. **启动阶段**：`RealtimeThreadExecutor::start()` 调用 `cycle_manager->start_cycle(name)`，周期管理器开始按周期调用回调。
3. **执行阶段**：周期管理器在每个周期调用 `callback`（即 `RealtimeThreadConfig::cycle_callback`），实时线程在此回调中执行周期逻辑。
4. **停止阶段**：`RealtimeThreadExecutor::stop()` 调用 `cycle_manager->stop_cycle(name)`，周期管理器停止周期循环。

### 8.6 注意事项

- **生命周期管理**：`cycle_manager` 指针的生命周期需由用户管理，确保在实时线程运行期间有效。
- **线程安全**：`ICycleManager` 的实现需保证线程安全（如使用互斥锁保护内部状态）。
- **统计信息**：`get_statistics()` 可用于监控周期执行时间、超时次数等，便于性能分析。
- **多实例共享**：一个 `ICycleManager` 实例可管理多个实时线程的周期，便于统一监控和管理。

### 8.7 完整示例

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
