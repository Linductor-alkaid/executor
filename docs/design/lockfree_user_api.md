# 无锁队列用户接口设计

## 背景

项目已实现 SPSC 无锁队列 (`LockFreeQueue<T>`)，但仅在内部使用。用户需要处理高性能、低延迟的数据传递场景时，应提供友好的接口。

## 使用场景

1. **高频数据采集**：传感器数据、网络包、日志等需要快速入队
2. **实时通信**：CAN 总线、串口等需要低延迟传递
3. **生产者-消费者模式**：单生产者单消费者的高性能场景
4. **避免锁竞争**：性能敏感路径上避免互斥锁开销

## 设计方案

### 方案 1：直接暴露无锁队列（最简单）

**优点**：
- 实现成本低，只需移动头文件位置
- 用户完全控制，灵活性最高
- 零额外开销

**缺点**：
- 用户需自行管理消费者线程
- 需要理解 SPSC 语义和内存序
- 缺少生命周期管理

**实现**：
```cpp
// 将 lockfree_queue.hpp 移至 include/executor/util/
#include <executor/util/lockfree_queue.hpp>

// 用户代码
executor::util::LockFreeQueue<int> queue(1024);

// 生产者线程
queue.push(42);

// 消费者线程
int value;
if (queue.pop(value)) {
    // 处理 value
}
```

### 方案 2：无锁任务执行器（推荐）

**优点**：
- 封装完整，用户无需管理线程
- 提供启动/停止/监控接口
- 与现有 Executor API 风格一致

**缺点**：
- 仅支持任务（`std::function<void()>`），不支持自定义数据类型
- 增加一层封装开销（但可忽略）

**实现**：
```cpp
// 新增 LockFreeTaskExecutor 类
class LockFreeTaskExecutor {
public:
    explicit LockFreeTaskExecutor(size_t queue_capacity = 1024);

    bool start();  // 启动消费者线程
    void stop();   // 停止并等待

    bool push_task(std::function<void()> task);  // 提交任务

    size_t pending_count() const;  // 队列中待处理任务数
};

// 用户代码
executor::LockFreeTaskExecutor exec(1024);
exec.start();

exec.push_task([]() {
    // 高性能任务
});

exec.stop();
```

### 方案 3：泛型无锁执行器（最灵活）

**优点**：
- 支持任意数据类型
- 用户自定义处理逻辑
- 适合复杂数据流场景

**缺点**：
- 实现复杂度较高
- 模板代码可能影响编译时间

**实现**：
```cpp
// 泛型无锁执行器
template<typename T>
class LockFreeExecutor {
public:
    using Handler = std::function<void(const T&)>;

    LockFreeExecutor(size_t capacity, Handler handler);

    bool start();
    void stop();
    bool push(const T& item);
};

// 用户代码
struct SensorData {
    int64_t timestamp;
    float value;
};

executor::LockFreeExecutor<SensorData> exec(1024, [](const SensorData& data) {
    // 处理传感器数据
});

exec.start();
exec.push({get_timestamp(), read_sensor()});
```

## 推荐实现路径

**阶段 1**：暴露基础无锁队列（立即可用）
- 移动 `lockfree_queue.hpp` 到 `include/executor/util/`
- 添加文档和示例

**阶段 2**：实现无锁任务执行器（推荐使用）
- 新增 `LockFreeTaskExecutor` 类
- 集成到 `Executor` Facade（可选）

**阶段 3**：泛型无锁执行器（高级场景）
- 实现 `LockFreeExecutor<T>` 模板类
- 支持批量处理、背压控制等高级特性

## API 设计细节

### 基础无锁队列 API

```cpp
namespace executor {
namespace util {

template<typename T>
class LockFreeQueue {
public:
    explicit LockFreeQueue(size_t capacity);

    bool push(const T& item);
    bool pop(T& item);
    bool empty() const;
    size_t size() const;
    size_t capacity() const;
};

} // namespace util
} // namespace executor
```

### 无锁任务执行器 API

```cpp
namespace executor {

class LockFreeTaskExecutor {
public:
    explicit LockFreeTaskExecutor(size_t queue_capacity = 1024);
    ~LockFreeTaskExecutor();

    bool start();
    void stop();
    bool is_running() const;

    bool push_task(std::function<void()> task);

    size_t pending_count() const;
    uint64_t processed_count() const;
};

} // namespace executor
```

### 集成到 Executor Facade（可选）

```cpp
class Executor {
public:
    // 注册无锁任务执行器
    bool register_lockfree_executor(const std::string& name, size_t capacity = 1024);

    // 提交无锁任务
    bool submit_lockfree(const std::string& name, std::function<void()> task);

    // 获取无锁执行器
    LockFreeTaskExecutor* get_lockfree_executor(const std::string& name);
};
```

## 使用示例

### 示例 1：高频日志收集

```cpp
#include <executor/lockfree_task_executor.hpp>

executor::LockFreeTaskExecutor logger(4096);
logger.start();

// 在性能敏感路径上
logger.push_task([msg = get_log_message()]() {
    write_to_file(msg);
});
```

### 示例 2：传感器数据处理

```cpp
#include <executor/util/lockfree_queue.hpp>

struct SensorData { int64_t ts; float value; };

executor::util::LockFreeQueue<SensorData> queue(1024);

// 生产者（中断处理或高频采集）
std::thread producer([&]() {
    while (running) {
        SensorData data = read_sensor();
        if (!queue.push(data)) {
            // 队列满，处理背压
        }
    }
});

// 消费者
std::thread consumer([&]() {
    while (running) {
        SensorData data;
        if (queue.pop(data)) {
            process(data);
        }
    }
});
```

### 示例 3：与实时线程集成

```cpp
// 使用现有 RealtimeThreadExecutor
auto& ex = executor::Executor::instance();
ex.initialize(config);

RealtimeThreadConfig rt_cfg;
rt_cfg.cycle_period_ns = 1000000;  // 1ms
rt_cfg.cycle_callback = []() noexcept {
    // 周期回调
};

ex.register_realtime_task("sensor", rt_cfg);
ex.start_realtime_task("sensor");

auto* rt_exec = ex.get_realtime_executor("sensor");

// 从其他线程推送任务到实时线程
rt_exec->push_task([]() {
    // 在实时线程中执行
});
```

## 性能考虑

1. **队列容量**：必须是 2 的幂，建议 1024-8192
2. **数据类型**：必须是 trivially copyable
3. **内存序**：已优化为 acquire-release 语义
4. **缓存行**：考虑添加 padding 避免 false sharing

## 安全性

1. **SPSC 限制**：严格单生产者单消费者，多生产者会导致数据竞争
2. **队列满处理**：`push()` 返回 false 时需要背压策略
3. **生命周期**：确保队列在生产者/消费者之前创建，之后销毁

## 文档和示例

需要添加：
- API 文档（Doxygen）
- 使用指南（docs/API.md）
- 示例代码（examples/lockfree_*.cpp）
- 性能基准测试（tests/benchmark_lockfree_*.cpp）
