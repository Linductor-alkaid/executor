#pragma once

#include "config.hpp"
#include "types.hpp"
#include "interfaces.hpp"
#include "executor_manager.hpp"
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
     * @brief 关闭执行器
     * 
     * 关闭所有执行器（异步执行器和实时执行器）。
     * 
     * @param wait_for_tasks 是否等待任务完成（默认：true）
     */
    void shutdown(bool wait_for_tasks = true);

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
     * @brief 启动实时任务
     * 
     * @param name 任务名称
     * @return 是否启动成功
     */
    bool start_realtime_task(const std::string& name);

    /**
     * @brief 停止实时任务
     * 
     * @param name 任务名称
     */
    void stop_realtime_task(const std::string& name);

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
     * @brief 启用或禁用任务监控
     */
    void enable_monitoring(bool enable);

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
     */
    void wait_for_completion();

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

    // ExecutorManager 指针（单例或实例）
    ExecutorManager* manager_;

    // 实例化模式时拥有的 ExecutorManager
    std::unique_ptr<ExecutorManager> owned_manager_;

    // 延迟任务结构（使用类型擦除）
    struct DelayedTask {
        std::chrono::steady_clock::time_point execute_time;
        std::function<void()> task;
        std::function<void()> on_complete;  // 完成回调（用于设置 promise）
    };

    // 周期性任务结构
    struct PeriodicTask {
        std::string task_id;
        int64_t period_ms;
        std::chrono::steady_clock::time_point next_execute_time;
        std::function<void()> task;
        bool cancelled = false;  // 使用普通 bool，由 periodic_tasks_mutex_ 保护
    };

    // 延迟任务列表
    std::vector<DelayedTask> delayed_tasks_;
    std::mutex delayed_tasks_mutex_;

    // 周期性任务列表
    std::unordered_map<std::string, PeriodicTask> periodic_tasks_;
    std::mutex periodic_tasks_mutex_;

    // 定时器线程
    std::thread timer_thread_;
    std::atomic<bool> timer_running_{false};
};

// 模板方法实现
template<typename F, typename... Args>
auto Executor::submit(F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type> {
    auto* executor = manager_->get_default_async_executor();
    if (!executor) {
        throw std::runtime_error("Async executor not initialized. Call initialize() first.");
    }
    return executor->submit(std::forward<F>(f), std::forward<Args>(args)...);
}

template<typename F, typename... Args>
auto Executor::submit_priority(int priority, F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type> {
    auto* executor = manager_->get_default_async_executor();
    if (!executor) {
        throw std::runtime_error("Async executor not initialized. Call initialize() first.");
    }
    
    // 使用 IAsyncExecutor 接口的 submit_priority 方法
    return executor->submit_priority(priority, std::forward<F>(f), std::forward<Args>(args)...);
}

template<typename F, typename... Args>
auto Executor::submit_delayed(int64_t delay_ms, F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type> {
    using return_type = typename std::invoke_result<F, Args...>::type;
    
    auto* executor = manager_->get_default_async_executor();
    if (!executor) {
        throw std::runtime_error("Async executor not initialized. Call initialize() first.");
    }
    
    // 创建 promise 和 future
    auto promise = std::make_shared<std::promise<return_type>>();
    auto future = promise->get_future();
    
    // 计算执行时间
    auto execute_time = std::chrono::steady_clock::now() + 
                       std::chrono::milliseconds(delay_ms);
    
    // 创建延迟任务包装器
    std::function<void()> task_wrapper = [f = std::forward<F>(f), 
                                         args_tuple = std::make_tuple(std::forward<Args>(args)...),
                                         promise]() mutable {
        try {
            if constexpr (std::is_void_v<return_type>) {
                std::apply(f, std::move(args_tuple));
                promise->set_value();
            } else {
                auto result = std::apply(f, std::move(args_tuple));
                promise->set_value(std::move(result));
            }
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    };
    
    // 创建延迟任务
    DelayedTask delayed_task;
    delayed_task.execute_time = execute_time;
    delayed_task.task = std::move(task_wrapper);
    delayed_task.on_complete = []() {};  // 延迟任务不需要额外的完成回调
    
    // 添加到延迟任务列表
    {
        std::lock_guard<std::mutex> lock(delayed_tasks_mutex_);
        delayed_tasks_.push_back(std::move(delayed_task));
    }
    
    // 确保定时器线程运行
    if (!timer_running_.load()) {
        start_timer_thread();
    }
    
    return future;
}

} // namespace executor
