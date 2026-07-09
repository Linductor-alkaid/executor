#pragma once

#include "config.hpp"
#include "types.hpp"
#include "interfaces.hpp"
#include "executor_manager.hpp"
#include "gpu/gpu_scheduler.hpp"
#include <future>
#include <functional>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>
#include <type_traits>
#include <map>
#include <queue>
#include <deque>
#include <optional>

namespace executor {

/**
 * @brief Executor Facade
 * 
 * 提供统一的高级 API，内部委托给 ExecutorManager。
 * 支持单例模式和实例化模式。
 * 
 * 功能：
 * - 任务提交（submit, submit_priority, submit_delayed, submit_periodic）
 * - 实时任务管理（register_realtime_task, start_realtime_task, stop_realtime_task）
 * - 监控查询（get_async_executor_status, get_realtime_executor_status）
 */
class Executor {
public:
    /**
     * @brief 获取单例实例
     * 
     * 使用全局 ExecutorManager 单例，同一进程内共享。
     * 
     * @return Executor 单例引用
     */
    static Executor& instance();

    /**
     * @brief 构造函数（实例化模式）
     * 
     * 创建独立的 Executor 实例，内部创建独立的 ExecutorManager 实例。
     * 用于资源隔离场景。
     */
    Executor();

    /**
     * @brief 析构函数（RAII）
     * 
     * 自动关闭定时器线程，ExecutorManager 析构时会自动释放所有执行器。
     */
    ~Executor();

    // 禁止拷贝和赋值
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;

    /**
     * @brief 初始化执行器
     * 
     * 初始化默认异步执行器（线程池）。
     * 
     * @param config 执行器配置
     * @return 是否初始化成功
     */
    bool initialize(const ExecutorConfig& config);

    /**
     * @brief 初始化执行器并返回可诊断结果
     */
    ExecutorResult initialize_ex(const ExecutorConfig& config);

    /**
     * @brief 关闭执行器
     * 
     * 关闭所有执行器（异步执行器和实时执行器）。
     * 
     * @param wait_for_tasks 是否等待任务完成（默认：true）
     */
    void shutdown(bool wait_for_tasks = true);

    /**
     * @brief 设置定时器线程工厂（仅用于测试）
     *
     * 允许测试注入线程创建失败，验证 start_timer_thread() 的异常回滚行为。
     */
    void set_timer_thread_factory_for_test(
        std::function<std::thread(std::function<void()>)> factory);

    /**
     * @brief 提交任务（使用默认线程池）
     * 
     * @tparam F 可调用对象类型
     * @tparam Args 参数类型
     * @param f 可调用对象
     * @param args 参数
     * @return std::future 任务执行结果的 future
     */
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>;

    /**
     * @brief 提交优先级任务
     * 
     * @tparam F 可调用对象类型
     * @tparam Args 参数类型
     * @param priority 优先级（0=LOW, 1=NORMAL, 2=HIGH, 3=CRITICAL）
     * @param f 可调用对象
     * @param args 参数
     * @return std::future 任务执行结果的 future
     */
    template<typename F, typename... Args>
    auto submit_priority(int priority, F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>;

    /**
     * @brief 提交延迟任务
     * 
     * 任务将在指定延迟时间后执行。
     * 
     * @tparam F 可调用对象类型
     * @tparam Args 参数类型
     * @param delay_ms 延迟时间（毫秒）
     * @param f 可调用对象
     * @param args 参数
     * @return std::future 任务执行结果的 future
     */
    template<typename F, typename... Args>
    auto submit_delayed(int64_t delay_ms, F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>;

    /**
     * @brief 提交周期性任务
     * 
     * 任务将按指定周期重复执行。
     * 
     * @param period_ms 周期（毫秒）
     * @param task 任务函数
     * @return 任务 ID（可用于取消任务）
     */
    std::string submit_periodic(int64_t period_ms, std::function<void()> task);

    /**
     * @brief 取消任务
     *
     * 取消指定的周期性任务。
     *
     * @param task_id 任务 ID
     * @return 是否取消成功
     */
    bool cancel_task(const std::string& task_id);

    /**
     * @brief 查询单个周期任务状态
     *
     * 返回 std::nullopt 表示任务不存在或已取消。
     */
    std::optional<PeriodicTaskStatus> get_periodic_task_status(
        const std::string& task_id) const;

    /**
     * @brief 查询所有当前注册的周期任务状态
     */
    std::vector<PeriodicTaskStatus> get_all_periodic_task_status() const;

    /**
     * @brief 批量提交任务
     *
     * 批量提交多个任务，可减少重复提交路径开销。
     * 实际性能收益取决于任务数量、任务体、线程数、硬件和构建配置。
     *
     * @tparam F 可调用对象类型
     * @param tasks 任务列表
     * @return std::vector<std::future<void>> 任务执行结果的 future 列表
     *
     * @note 不承诺固定加速比；需要性能结论时请运行本地 benchmark。
     *
     * 示例：
     * @code
     * std::vector<std::function<void()>> tasks;
     * for (int i = 0; i < 1000; ++i) {
     *     tasks.push_back([i]() { process(i); });
     * }
     * auto futures = executor.submit_batch(tasks);
     * @endcode
     */
    template<typename F>
    std::vector<std::future<void>> submit_batch(const std::vector<F>& tasks);

    /**
     * @brief 批量提交任务（无返回值版本）
     *
     * 批量提交多个任务，不返回 future，省去逐个 future 的管理开销。
     * 适用于不需要等待任务完成的场景（fire-and-forget）。
     *
     * @tparam F 可调用对象类型
     * @param tasks 任务列表
     *
     * @note 相比返回 future 的版本，避免了 packaged_task 的开销
     *
     * 示例：
     * @code
     * std::vector<std::function<void()>> tasks;
     * for (int i = 0; i < 1000; ++i) {
     *     tasks.push_back([i]() { process(i); });
     * }
     * executor.submit_batch_no_future(tasks);
     * @endcode
     */
    template<typename F>
    void submit_batch_no_future(const std::vector<F>& tasks);

    /**
     * @brief 批量提交优先级任务
     *
     * 批量提交多个优先级任务。
     *
     * @tparam F 可调用对象类型
     * @param priority 优先级（0=LOW, 1=NORMAL, 2=HIGH, 3=CRITICAL）
     * @param tasks 任务列表
     * @return std::vector<std::future<void>> 任务执行结果的 future 列表
     */
    template<typename F>
    std::vector<std::future<void>> submit_batch_priority(
        int priority,
        const std::vector<F>& tasks);

    /**
     * @brief 注册实时任务
     * 
     * 创建并注册实时执行器（专用实时线程）。
     * 
     * @param name 任务名称
     * @param config 实时线程配置
     * @return 是否注册成功
     */
    bool register_realtime_task(const std::string& name,
                               const RealtimeThreadConfig& config);

    /**
     * @brief 注册实时任务并返回可诊断结果
     */
    ExecutorResult register_realtime_task_ex(const std::string& name,
                                             const RealtimeThreadConfig& config);

    /**
     * @brief 启动实时任务
     * 
     * @param name 任务名称
     * @return 是否启动成功
     */
    bool start_realtime_task(const std::string& name);

    /**
     * @brief 启动实时任务并返回可诊断结果
     */
    ExecutorResult start_realtime_task_ex(const std::string& name);

    /**
     * @brief 停止实时任务
     * 
     * @param name 任务名称
     */
    void stop_realtime_task(const std::string& name);

    /**
     * @brief 通过 facade 推送任务到指定实时执行器
     *
     * 失败会同时通过返回值、RealtimeExecutorStatus 计数和 facade failure event 可见。
     */
    bool push_realtime_task(const std::string& name, std::function<void()> task);

    /**
     * @brief push_realtime_task 的显式 try 命名别名
     */
    bool try_push_realtime_task(const std::string& name, std::function<void()> task);

    /**
     * @brief 获取实时执行器
     * 
     * @param name 执行器名称
     * @return 实时执行器指针，如果不存在则返回 nullptr
     */
    IRealtimeExecutor* get_realtime_executor(const std::string& name);

    /**
     * @brief 获取所有实时任务列表
     * 
     * @return 实时任务名称列表
     */
    std::vector<std::string> get_realtime_task_list() const;

    /**
     * @brief 获取异步执行器状态
     * 
     * @return 异步执行器状态
     */
    AsyncExecutorStatus get_async_executor_status() const;

    /**
     * @brief 获取实时执行器状态
     * 
     * @param name 执行器名称
     * @return 实时执行器状态
     */
    RealtimeExecutorStatus get_realtime_executor_status(const std::string& name) const;

    /**
     * @brief 设置 facade 失败事件回调
     *
     * 未设置回调时，失败事件仍会进入状态计数和最近事件缓冲。
     * callback 自身抛出的异常会被隔离，不会杀死 worker 或后台线程。
     */
    void set_failure_callback(ExecutorFailureCallback callback);

    /**
     * @brief 获取累计失败状态
     */
    ExecutorFailureStatus get_failure_status() const;

    /**
     * @brief 获取最近失败事件
     *
     * @param max_count 最多返回事件数；0 表示返回当前缓冲区内全部事件。
     * @return 按发生时间从旧到新排序的失败事件列表
     */
    std::vector<ExecutorFailureEvent> get_recent_failures(size_t max_count = 0) const;

    /**
     * @brief 清空最近失败事件
     *
     * 只清空 ring buffer，不重置累计计数。
     */
    void clear_recent_failures();

    /**
     * @brief 设置最近失败事件缓冲容量
     *
     * 容量为 0 时不保留最近事件，但累计状态和 callback 仍生效。
     */
    void set_recent_failure_capacity(size_t capacity);

    /**
     * @brief 启用或禁用任务监控
     */
    void enable_monitoring(bool enable);

    /**
     * @brief 设置监控采样率
     * @param rate 采样率 (0.0-1.0)，0.01 表示 1% 采样
     */
    void set_monitoring_sampling_rate(double rate);

    /**
     * @brief 按 task_type 获取任务统计
     */
    TaskStatistics get_task_statistics(const std::string& task_type) const;

    /**
     * @brief 获取全部 task_type 的任务统计
     */
    std::map<std::string, TaskStatistics> get_all_task_statistics() const;

    /**
     * @brief 等待所有已提交的异步任务完成
     *
     * 兼容旧调用方，最多等待 kDefaultWaitForCompletionTimeout。
     * 超时时不抛异常，但会记录 FailureKind::WaitTimeout。
     */
    void wait_for_completion();

    /**
     * @brief 等待所有已提交的异步任务完成并返回是否完成
     *
     * @param timeout 最长等待时间
     * @return true 表示所有任务在 timeout 内完成；false 表示等待超时。
     *         超时时记录 FailureKind::WaitTimeout，可通过 get_failure_status()
     *         观察 wait_timeout_count。
     */
    bool try_wait_for_completion(std::chrono::milliseconds timeout);

    /**
     * @brief 等待所有已提交的异步任务完成并返回是否完成
     */
    template<typename Rep, typename Period>
    bool wait_for_completion_for(
        const std::chrono::duration<Rep, Period>& timeout);

    /**
     * @brief 等待所有已提交的异步任务完成并返回诊断结果
     */
    WaitResult wait_for_completion_ex(std::chrono::milliseconds timeout);

    /**
     * @brief 当前默认异步执行器是否没有排队或执行中的任务
     */
    bool is_idle() const;

    /**
     * @brief 获取默认异步执行器完成状态快照
     */
    CompletionStatus get_completion_status() const;

    /**
     * @brief 注册 GPU 执行器
     * 
     * 创建并注册 GPU 执行器。
     * 
     * @param name 执行器名称
     * @param config GPU 执行器配置
     * @return 是否注册成功
     */
    bool register_gpu_executor(const std::string& name,
                              const gpu::GpuExecutorConfig& config);

    /**
     * @brief 注册 GPU 执行器并返回可诊断结果
     */
    ExecutorResult register_gpu_executor_ex(const std::string& name,
                                            const gpu::GpuExecutorConfig& config);

    /**
     * @brief 提交 GPU kernel 任务
     * 
     * @tparam KernelFunc GPU kernel 函数类型
     * @param executor_name GPU 执行器名称
     * @param kernel GPU kernel 函数
     * @param config GPU 任务配置
     * @return std::future<void> 任务执行结果的 future
     */
    template<typename KernelFunc>
    auto submit_gpu(const std::string& executor_name,
                   KernelFunc&& kernel,
                   const gpu::GpuTaskConfig& config)
        -> std::future<void>;

    /**
     * @brief 获取 GPU 执行器
     * 
     * @param name 执行器名称
     * @return GPU 执行器指针，如果不存在则返回 nullptr
     */
    IGpuExecutor* get_gpu_executor(const std::string& name);

    /**
     * @brief 获取所有 GPU 执行器名称
     * 
     * @return GPU 执行器名称列表
     */
    std::vector<std::string> get_gpu_executor_names() const;

    /**
     * @brief 获取 GPU 执行器状态
     * 
     * @param name 执行器名称
     * @return GPU 执行器状态
     */
    gpu::GpuExecutorStatus get_gpu_executor_status(const std::string& name) const;

    /**
     * @brief 获取所有 GPU 执行器状态（监控查询）
     *
     * @return 执行器名称到状态的映射
     */
    std::map<std::string, gpu::GpuExecutorStatus> get_all_gpu_executor_status() const;

    /**
     * @brief 自动选择 CPU/GPU 执行器提交任务
     *
     * 根据任务特征自动选择 CPU 或 GPU 执行器。
     * 如果选择 GPU，调用 submit_gpu()；如果选择 CPU，在 CPU 线程池执行。
     *
     * @tparam KernelFunc GPU kernel 函数类型
     * @param characteristics 任务特征（数据大小、计算强度等）
     * @param gpu_executor_name GPU 执行器名称（GPU 被选中时使用）
     * @param kernel GPU kernel 函数（需支持 nullptr stream 用于 CPU 执行）
     * @param gpu_config GPU 任务配置（GPU 被选中时使用）
     * @return std::future<void> 任务执行结果的 future
     */
    template<typename KernelFunc>
    auto submit_auto(
        const gpu::TaskCharacteristics& characteristics,
        const std::string& gpu_executor_name,
        KernelFunc&& kernel,
        const gpu::GpuTaskConfig& gpu_config)
        -> std::future<void>;

    /**
     * @brief 更新调度器配置
     *
     * @param config 调度器配置
     */
    void update_scheduler_config(const gpu::GpuScheduler::Config& config);

    /**
     * @brief 获取调度器配置
     *
     * @return 当前调度器配置
     */
    gpu::GpuScheduler::Config get_scheduler_config() const;

private:
    /**
     * @brief 单例模式构造函数（私有）
     * 
     * @param manager ExecutorManager 单例引用
     */
    Executor(ExecutorManager& manager);

    /**
     * @brief 定时器线程函数
     * 
     * 处理延迟任务和周期性任务。
     */
    void timer_thread_func();

    /**
     * @brief 启动定时器线程
     */
    void start_timer_thread();

    /**
     * @brief 停止定时器线程
     */
    void stop_timer_thread();

    /**
     * @brief 记录 facade 失败事件
     */
    void record_failure(ExecutorFailureEvent event);

    void record_result_failure(const ExecutorResult& result,
                               FailureKind kind,
                               const std::string& executor_name,
                               const std::string& task_id);

    void record_submit_rejected(const std::string& executor_name,
                                const std::string& task_id,
                                const std::string& message,
                                std::exception_ptr exception = nullptr);

    void record_task_exception(const std::string& executor_name,
                               const std::string& task_id,
                               const std::string& message,
                               std::exception_ptr exception);

    void record_task_timeout(const std::string& executor_name,
                             const std::string& task_id,
                             const std::string& message,
                             std::exception_ptr exception);

    void record_realtime_drop(const std::string& executor_name,
                              const std::string& task_id,
                              const std::string& message,
                              std::exception_ptr exception = nullptr);

    void record_periodic_task_success(const std::string& task_id);

    void record_periodic_task_exception(const std::string& executor_name,
                                        const std::string& task_id,
                                        const std::string& message,
                                        std::exception_ptr exception);

    void record_periodic_submit_rejected(const std::string& executor_name,
                                         const std::string& task_id,
                                         const std::string& message,
                                         std::exception_ptr exception = nullptr);

    /**
     * @brief 当前 facade 最近失败事件缓冲容量
     */
    size_t recent_failure_capacity() const;

    // ExecutorManager 指针（单例或实例）
    ExecutorManager* manager_;

    // 实例化模式时拥有的 ExecutorManager
    std::unique_ptr<ExecutorManager> owned_manager_;

    // 延迟任务结构（使用类型擦除）
    struct DelayedTask {
        std::string task_id;
        std::chrono::steady_clock::time_point execute_time;
        std::function<void()> task;
        std::function<void(std::exception_ptr)> on_timeout;
        std::function<void(std::exception_ptr)> on_rejected;
    };

    // 延迟任务比较器（用于 priority_queue，最早执行的在顶部）
    struct DelayedTaskComparator {
        bool operator()(const DelayedTask& a, const DelayedTask& b) const {
            return a.execute_time > b.execute_time;  // 早的在顶部（最小堆）
        }
    };

    // 周期性任务结构
    struct PeriodicTask {
        PeriodicTaskStatus status;
        std::function<void()> task;
        bool cancelled = false;  // 使用普通 bool，由 periodic_tasks_mutex_ 保护
    };

    // 延迟任务优先级队列（按执行时间排序，最早的在顶部）
    std::priority_queue<DelayedTask, std::vector<DelayedTask>, DelayedTaskComparator> delayed_tasks_;
    std::mutex delayed_tasks_mutex_;

    // 周期性任务列表
    std::unordered_map<std::string, PeriodicTask> periodic_tasks_;
    mutable std::mutex periodic_tasks_mutex_;

    // 定时器线程
    std::thread timer_thread_;
    std::atomic<bool> timer_running_{false};
    std::function<std::thread(std::function<void()>)> timer_thread_factory_for_test_;

    static constexpr size_t kDefaultRecentFailureCapacity = 128;

    mutable std::mutex failure_mutex_;
    ExecutorFailureStatus failure_status_;
    std::deque<ExecutorFailureEvent> recent_failures_;
    size_t recent_failure_capacity_ = kDefaultRecentFailureCapacity;
    ExecutorFailureCallback failure_callback_;

    // GPU 调度器
    gpu::GpuScheduler scheduler_;
};

// 模板方法实现
template<typename Rep, typename Period>
bool Executor::wait_for_completion_for(
    const std::chrono::duration<Rep, Period>& timeout) {
    return wait_for_completion_ex(
        std::chrono::duration_cast<std::chrono::milliseconds>(timeout)).completed;
}

template<typename F, typename... Args>
auto Executor::submit(F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type> {
    using return_type = typename std::invoke_result<F, Args...>::type;

    auto* executor = manager_->get_default_async_executor();
    const std::string executor_name = executor ? executor->get_name() : "default";
    const std::string task_id = "facade_submit";
    if (!executor) {
        record_submit_rejected(
            executor_name,
            task_id,
            "Async executor not initialized. Call initialize() first.");
        throw std::runtime_error("Async executor not initialized. Call initialize() first.");
    }

    auto promise = std::make_shared<std::promise<return_type>>();
    auto promise_ready = std::make_shared<std::atomic_bool>(false);
    auto future = promise->get_future();
    auto bound_task = std::make_shared<decltype(std::bind(std::forward<F>(f), std::forward<Args>(args)...))>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    auto task_wrapper = [this, executor_name, task_id, promise, promise_ready, bound_task]() mutable {
        try {
            if constexpr (std::is_void_v<return_type>) {
                std::invoke(*bound_task);
                promise->set_value();
            } else {
                promise->set_value(std::invoke(*bound_task));
            }
            promise_ready->store(true, std::memory_order_release);
        } catch (...) {
            auto exception = std::current_exception();
            promise->set_exception(exception);
            promise_ready->store(true, std::memory_order_release);
            record_task_exception(
                executor_name,
                task_id,
                "Async task threw an exception",
                exception);
            throw;
        }
    };

    auto on_timeout = [this, executor_name, task_id, promise, promise_ready](
                          std::exception_ptr exception) {
        bool expected = false;
        if (promise_ready->compare_exchange_strong(expected, true)) {
            promise->set_exception(exception);
            record_task_timeout(
                executor_name,
                task_id,
                "Async task timed out before execution",
                exception);
        }
    };

    if (!executor->try_submit_task(std::move(task_wrapper), std::move(on_timeout))) {
        auto exception = std::make_exception_ptr(
            std::runtime_error("Async executor rejected task submission"));
        bool expected = false;
        if (promise_ready->compare_exchange_strong(expected, true)) {
            promise->set_exception(exception);
            record_submit_rejected(
                executor_name,
                task_id,
                "Async executor rejected task submission",
                exception);
        }
    }

    return future;
}

template<typename F, typename... Args>
auto Executor::submit_priority(int priority, F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type> {
    using return_type = typename std::invoke_result<F, Args...>::type;

    auto* executor = manager_->get_default_async_executor();
    const std::string executor_name = executor ? executor->get_name() : "default";
    const std::string task_id = "facade_submit_priority";
    if (!executor) {
        record_submit_rejected(
            executor_name,
            task_id,
            "Async executor not initialized. Call initialize() first.");
        throw std::runtime_error("Async executor not initialized. Call initialize() first.");
    }

    auto promise = std::make_shared<std::promise<return_type>>();
    auto promise_ready = std::make_shared<std::atomic_bool>(false);
    auto future = promise->get_future();
    auto bound_task = std::make_shared<decltype(std::bind(std::forward<F>(f), std::forward<Args>(args)...))>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    auto task_wrapper = [this, executor_name, task_id, promise, promise_ready, bound_task]() mutable {
        try {
            if constexpr (std::is_void_v<return_type>) {
                std::invoke(*bound_task);
                promise->set_value();
            } else {
                promise->set_value(std::invoke(*bound_task));
            }
            promise_ready->store(true, std::memory_order_release);
        } catch (...) {
            auto exception = std::current_exception();
            promise->set_exception(exception);
            promise_ready->store(true, std::memory_order_release);
            record_task_exception(
                executor_name,
                task_id,
                "Priority async task threw an exception",
                exception);
            throw;
        }
    };

    auto on_timeout = [this, executor_name, task_id, promise, promise_ready](
                          std::exception_ptr exception) {
        bool expected = false;
        if (promise_ready->compare_exchange_strong(expected, true)) {
            promise->set_exception(exception);
            record_task_timeout(
                executor_name,
                task_id,
                "Priority async task timed out before execution",
                exception);
        }
    };

    if (!executor->try_submit_priority_task(
            priority, std::move(task_wrapper), std::move(on_timeout))) {
        auto exception = std::make_exception_ptr(
            std::runtime_error("Async executor rejected priority task submission"));
        bool expected = false;
        if (promise_ready->compare_exchange_strong(expected, true)) {
            promise->set_exception(exception);
            record_submit_rejected(
                executor_name,
                task_id,
                "Async executor rejected priority task submission",
                exception);
        }
    }

    return future;
}

template<typename F, typename... Args>
auto Executor::submit_delayed(int64_t delay_ms, F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type> {
    using return_type = typename std::invoke_result<F, Args...>::type;

    auto* executor = manager_->get_default_async_executor();
    const std::string executor_name = executor ? executor->get_name() : "default";
    const std::string task_id = "facade_submit_delayed";
    if (!executor) {
        record_submit_rejected(
            executor_name,
            task_id,
            "Async executor not initialized. Call initialize() first.");
        throw std::runtime_error("Async executor not initialized. Call initialize() first.");
    }

    auto promise = std::make_shared<std::promise<return_type>>();
    auto promise_ready = std::make_shared<std::atomic_bool>(false);
    auto future = promise->get_future();

    auto execute_time = std::chrono::steady_clock::now() +
                       std::chrono::milliseconds(delay_ms);

    std::function<void()> task_wrapper = [this,
                                         executor_name,
                                         task_id,
                                         f = std::forward<F>(f),
                                         args_tuple = std::make_tuple(std::forward<Args>(args)...),
                                         promise,
                                         promise_ready]() mutable {
        try {
            if constexpr (std::is_void_v<return_type>) {
                std::apply(f, std::move(args_tuple));
                promise->set_value();
            } else {
                auto result = std::apply(f, std::move(args_tuple));
                promise->set_value(std::move(result));
            }
            promise_ready->store(true, std::memory_order_release);
        } catch (...) {
            auto exception = std::current_exception();
            bool expected = false;
            if (promise_ready->compare_exchange_strong(expected, true)) {
                promise->set_exception(exception);
            }
            record_task_exception(
                executor_name,
                task_id,
                "Delayed async task threw an exception",
                exception);
            throw;
        }
    };

    DelayedTask delayed_task;
    delayed_task.task_id = task_id;
    delayed_task.execute_time = execute_time;
    delayed_task.task = std::move(task_wrapper);
    delayed_task.on_timeout = [this, executor_name, task_id, promise, promise_ready](
                                  std::exception_ptr exception) {
        bool expected = false;
        if (promise_ready->compare_exchange_strong(expected, true)) {
            promise->set_exception(exception);
            record_task_timeout(
                executor_name,
                task_id,
                "Delayed async task timed out before execution",
                exception);
        }
    };
    delayed_task.on_rejected = [this, executor_name, task_id, promise, promise_ready](
                                   std::exception_ptr exception) {
        bool expected = false;
        if (promise_ready->compare_exchange_strong(expected, true)) {
            promise->set_exception(exception);
            record_submit_rejected(
                executor_name,
                task_id,
                "Async executor rejected delayed task submission",
                exception);
        }
    };

    try {
        if (!timer_running_.load()) {
            start_timer_thread();
        }
    } catch (...) {
        auto exception = std::current_exception();
        bool expected = false;
        if (promise_ready->compare_exchange_strong(expected, true)) {
            promise->set_exception(exception);
        }
        record_submit_rejected(
            executor_name,
            task_id,
            "Timer thread creation failed for delayed task",
            exception);
        throw;
    }

    {
        std::lock_guard<std::mutex> lock(delayed_tasks_mutex_);
        delayed_tasks_.push(std::move(delayed_task));
    }

    return future;
}

// 批量任务提交模板方法实现
template<typename F>
std::vector<std::future<void>> Executor::submit_batch(const std::vector<F>& tasks) {
    auto* executor = manager_->get_default_async_executor();
    const std::string executor_name = executor ? executor->get_name() : "default";
    if (!executor) {
        record_submit_rejected(
            executor_name,
            "facade_submit_batch",
            "Async executor not initialized. Call initialize() first.");
        throw std::runtime_error("Async executor not initialized. Call initialize() first.");
    }

    std::vector<std::function<void()>> task_wrappers;
    std::vector<std::function<void(std::exception_ptr)>> timeout_handlers;
    std::vector<std::future<void>> futures;
    std::vector<std::shared_ptr<std::promise<void>>> promises;
    std::vector<std::shared_ptr<std::atomic_bool>> promise_ready_flags;

    task_wrappers.reserve(tasks.size());
    timeout_handlers.reserve(tasks.size());
    futures.reserve(tasks.size());
    promises.reserve(tasks.size());
    promise_ready_flags.reserve(tasks.size());

    for (size_t i = 0; i < tasks.size(); ++i) {
        auto promise = std::make_shared<std::promise<void>>();
        auto promise_ready = std::make_shared<std::atomic_bool>(false);
        futures.push_back(promise->get_future());
        promises.push_back(promise);
        promise_ready_flags.push_back(promise_ready);

        std::string task_id = "facade_submit_batch[" + std::to_string(i) + "]";

        task_wrappers.push_back([this, executor_name, task_id, promise, promise_ready, task = tasks[i]]() mutable {
            try {
                task();
                promise->set_value();
                promise_ready->store(true, std::memory_order_release);
            } catch (...) {
                auto exception = std::current_exception();
                promise->set_exception(exception);
                promise_ready->store(true, std::memory_order_release);
                record_task_exception(
                    executor_name,
                    task_id,
                    "Batch async task threw an exception",
                    exception);
                throw;
            }
        });
        timeout_handlers.push_back(
            [this, executor_name, task_id, promise, promise_ready](
                std::exception_ptr exception) {
                bool expected = false;
                if (promise_ready->compare_exchange_strong(expected, true)) {
                    promise->set_exception(exception);
                    record_task_timeout(
                        executor_name,
                        task_id,
                        "Batch async task timed out before execution",
                        exception);
                }
            });
    }

    if (!executor->try_submit_batch_tasks(
            std::move(task_wrappers), std::move(timeout_handlers))) {
        auto exception = std::make_exception_ptr(
            std::runtime_error("Async executor rejected batch task submission"));
        bool marked_any = false;
        for (size_t i = 0; i < promises.size(); ++i) {
            bool expected = false;
            if (promise_ready_flags[i]->compare_exchange_strong(expected, true)) {
                promises[i]->set_exception(exception);
                marked_any = true;
            }
        }
        if (marked_any || tasks.empty()) {
            record_submit_rejected(
                executor_name,
                "facade_submit_batch",
                tasks.empty()
                    ? "Async executor rejected empty batch task submission"
                    : "Async executor rejected batch task submission",
                exception);
        }
    }

    return futures;
}

template<typename F>
std::vector<std::future<void>> Executor::submit_batch_priority(
    int priority,
    const std::vector<F>& tasks) {
    auto* executor = manager_->get_default_async_executor();
    const std::string executor_name = executor ? executor->get_name() : "default";
    if (!executor) {
        record_submit_rejected(
            executor_name,
            "facade_submit_batch_priority",
            "Async executor not initialized. Call initialize() first.");
        throw std::runtime_error("Async executor not initialized. Call initialize() first.");
    }

    std::vector<std::future<void>> futures;
    futures.reserve(tasks.size());

    for (const auto& task : tasks) {
        futures.push_back(submit_priority(priority, task));
    }

    return futures;
}

template<typename F>
void Executor::submit_batch_no_future(const std::vector<F>& tasks) {
    auto* executor = manager_->get_default_async_executor();
    const std::string executor_name = executor ? executor->get_name() : "default";
    if (!executor) {
        record_submit_rejected(
            executor_name,
            "facade_submit_batch_no_future",
            "Async executor not initialized. Call initialize() first.");
        throw std::runtime_error("Async executor not initialized. Call initialize() first.");
    }

    std::vector<std::function<void()>> task_wrappers;
    task_wrappers.reserve(tasks.size());
    auto execution_failure_seen = std::make_shared<std::atomic_bool>(false);

    for (size_t i = 0; i < tasks.size(); ++i) {
        std::string task_id =
            "facade_submit_batch_no_future[" + std::to_string(i) + "]";

        task_wrappers.push_back([this, executor_name, task_id, execution_failure_seen, task = tasks[i]]() mutable {
            try {
                task();
            } catch (...) {
                auto exception = std::current_exception();
                execution_failure_seen->store(true, std::memory_order_release);
                record_task_exception(
                    executor_name,
                    task_id,
                    "Fire-and-forget batch async task threw an exception",
                    exception);
                throw;
            }
        });
    }

    if (!executor->try_submit_batch_tasks(std::move(task_wrappers))) {
        auto exception = std::make_exception_ptr(
            std::runtime_error("Async executor rejected fire-and-forget batch task submission"));
        if (!execution_failure_seen->load(std::memory_order_acquire)) {
            record_submit_rejected(
                executor_name,
                "facade_submit_batch_no_future",
                tasks.empty()
                    ? "Async executor rejected empty fire-and-forget batch task submission"
                    : "Async executor rejected fire-and-forget batch task submission",
                exception);
        }
    }
}

// GPU 任务提交模板方法实现
template<typename KernelFunc>
auto Executor::submit_gpu(const std::string& executor_name,
                         KernelFunc&& kernel,
                         const gpu::GpuTaskConfig& config)
    -> std::future<void> {
    auto* executor = manager_->get_gpu_executor(executor_name);
    if (!executor) {
        const std::string message =
            "submit_gpu: no GPU executor registered with name " + executor_name;
        record_submit_rejected(executor_name, "facade_submit_gpu", message);
        throw std::runtime_error("GPU executor '" + executor_name + "' not found. Call register_gpu_executor() first.");
    }
    return executor->submit_kernel(std::forward<KernelFunc>(kernel), config);
}

// 智能调度模板方法实现
template<typename KernelFunc>
auto Executor::submit_auto(
    const gpu::TaskCharacteristics& characteristics,
    const std::string& gpu_executor_name,
    KernelFunc&& kernel,
    const gpu::GpuTaskConfig& gpu_config)
    -> std::future<void> {

    auto choice = scheduler_.decide(characteristics);

    if (choice == gpu::ExecutorChoice::GPU) {
        return submit_gpu(gpu_executor_name, std::forward<KernelFunc>(kernel), gpu_config);
    } else {
        // CPU fallback: execute kernel with nullptr stream
        return submit([kernel = std::forward<KernelFunc>(kernel)]() mutable {
            kernel(nullptr);
        });
    }
}

} // namespace executor
