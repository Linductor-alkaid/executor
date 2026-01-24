#pragma once

#include "executor/config.hpp"
#include "executor/types.hpp"
#include "priority_scheduler.hpp"
#include "load_balancer.hpp"
#include "task_dispatcher.hpp"
#include "worker_local_queue.hpp"
#include "thread_pool_resizer.hpp"
#include "../util/exception_handler.hpp"
#include "../util/thread_utils.hpp"
#include "../task/task.hpp"
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <future>
#include <chrono>
#include <functional>
#include <type_traits>
#include <random>

namespace executor {

/**
 * @brief 线程池核心类
 * 
 * 管理工作线程的生命周期，从PriorityScheduler获取任务并分发给工作线程执行。
 * 支持任务提交、优先级调度、状态监控和优雅关闭。
 * 支持工作窃取、负载均衡和动态扩缩容。
 */
class ThreadPool {
public:
    /**
     * @brief 构造函数
     */
    ThreadPool();

    /**
     * @brief 析构函数
     * 
     * 自动关闭线程池并等待所有任务完成
     */
    ~ThreadPool();

    // 禁止拷贝和移动
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    /**
     * @brief 初始化线程池
     * 
     * 根据配置创建工作线程，设置线程优先级和CPU亲和性。
     * 
     * @param config 线程池配置
     * @return 如果初始化成功返回true，否则返回false
     */
    bool initialize(const ThreadPoolConfig& config);

    /**
     * @brief 提交任务（返回Future）
     * 
     * 将任务提交到线程池，使用NORMAL优先级。
     * 
     * @tparam F 可调用对象类型
     * @tparam Args 参数类型
     * @param f 可调用对象
     * @param args 参数
     * @return std::future 任务执行结果的future
     */
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>;

    /**
     * @brief 提交优先级任务
     * 
     * 将任务提交到线程池，使用指定的优先级。
     * 
     * @tparam F 可调用对象类型
     * @tparam Args 参数类型
     * @param priority 优先级（0=LOW, 1=NORMAL, 2=HIGH, 3=CRITICAL）
     * @param f 可调用对象
     * @param args 参数
     * @return std::future 任务执行结果的future
     */
    template<typename F, typename... Args>
    auto submit_priority(int priority, F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>;

    /**
     * @brief 获取线程池状态
     * 
     * @return 线程池状态信息
     */
    ThreadPoolStatus get_status() const;

    /**
     * @brief 关闭线程池
     * 
     * @param wait_for_tasks 是否等待所有任务完成（默认true）
     */
    void shutdown(bool wait_for_tasks = true);

    /**
     * @brief 等待所有任务完成
     * 
     * 阻塞直到所有已提交的任务执行完成。
     */
    void wait_for_completion();

    /**
     * @brief 检查线程池是否已停止
     * 
     * @return 如果线程池已停止返回true，否则返回false
     */
    bool is_stopped() const;

private:
    /**
     * @brief 工作线程函数
     * 
     * 循环从本地队列、工作窃取或全局调度器获取任务并执行。
     * 
     * @param worker_id 工作线程ID
     */
    void worker_thread(size_t worker_id);

    /**
     * @brief 执行任务
     * 
     * 执行任务并处理异常，更新统计信息。
     * 
     * @param task 任务对象
     */
    void execute_task(const Task& task);

    /**
     * @brief 更新统计信息
     * 
     * @param execution_time_ns 任务执行时间（纳秒）
     * @param success 是否成功
     */
    void update_statistics(int64_t execution_time_ns, bool success);

    /**
     * @brief 尝试工作窃取
     * 
     * 当本地队列为空时，从其他线程的本地队列窃取任务。
     * 
     * @param worker_id 当前工作线程ID
     * @param task 用于接收窃取的任务
     * @return 成功窃取返回 true
     */
    bool try_steal_task(size_t worker_id, Task& task);

    /**
     * @brief 检查线程是否需要退出（用于缩容）
     * 
     * @param worker_id 工作线程ID
     * @return 需要退出返回 true
     */
    bool should_exit(size_t worker_id) const;

    /**
     * @brief 监控线程函数（用于动态扩缩容）
     */
    void resize_monitor_thread();

    /**
     * @brief 创建新的工作线程
     * 
     * @param worker_id 工作线程ID
     */
    void create_worker_thread(size_t worker_id);

    // 配置信息
    ThreadPoolConfig config_;

    // 优先级调度器
    PriorityScheduler scheduler_;

    // 负载均衡器
    std::unique_ptr<LoadBalancer> load_balancer_;

    // 工作线程本地队列
    std::vector<WorkerLocalQueue> local_queues_;

    // 任务分发器
    std::unique_ptr<TaskDispatcher> dispatcher_;

    // 动态扩缩容控制器
    std::unique_ptr<ThreadPoolResizer> resizer_;

    // 工作线程
    std::vector<std::thread> workers_;

    // 工作线程ID映射（用于跟踪线程）
    std::vector<size_t> worker_ids_;

    // 停止标志
    std::atomic<bool> stop_{false};

    // 统计信息
    mutable std::mutex stats_mutex_;
    std::atomic<size_t> total_tasks_{0};
    std::atomic<size_t> completed_tasks_{0};
    std::atomic<size_t> failed_tasks_{0};
    std::atomic<int64_t> total_execution_time_ns_{0};
    std::atomic<size_t> active_threads_{0};

    // 条件变量：用于工作线程等待任务
    std::condition_variable condition_;

    // 互斥锁：保护共享状态
    mutable std::mutex mutex_;

    // 异常处理器
    util::ExceptionHandler exception_handler_;

    // 初始化标志
    std::atomic<bool> initialized_{false};

    // 待退出的线程ID集合（用于缩容）
    mutable std::mutex exit_threads_mutex_;
    std::vector<size_t> exit_threads_;

    // 监控线程（用于动态扩缩容）
    std::thread resize_monitor_thread_;
    std::atomic<bool> resize_monitor_stop_{false};
};

// 模板方法实现
template<typename F, typename... Args>
auto ThreadPool::submit(F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type> {
    
    using return_type = typename std::invoke_result<F, Args...>::type;
    
    // 创建packaged_task
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> result = task->get_future();
    
    // 创建Task对象
    Task executor_task;
    executor_task.task_id = generate_task_id();
    executor_task.priority = TaskPriority::NORMAL;
    executor_task.function = [task]() { (*task)(); };
    executor_task.submit_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
    executor_task.timeout_ms = config_.task_timeout_ms;
    
    // 提交到调度器；持锁期间分发并 notify，避免错过唤醒
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_.load()) {
            std::promise<return_type> promise;
            promise.set_exception(std::make_exception_ptr(
                std::runtime_error("ThreadPool is stopped")
            ));
            return promise.get_future();
        }
        scheduler_.enqueue(executor_task);
        total_tasks_.fetch_add(1, std::memory_order_relaxed);
        if (dispatcher_) {
            dispatcher_->dispatch_batch(1);
        }
        condition_.notify_all();
    }
    
    return result;
}

template<typename F, typename... Args>
auto ThreadPool::submit_priority(int priority, F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type> {
    
    using return_type = typename std::invoke_result<F, Args...>::type;
    
    // 创建packaged_task
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> result = task->get_future();
    
    // 将优先级值转换为TaskPriority枚举
    TaskPriority task_priority = TaskPriority::NORMAL;
    if (priority <= 0) {
        task_priority = TaskPriority::LOW;
    } else if (priority == 1) {
        task_priority = TaskPriority::NORMAL;
    } else if (priority == 2) {
        task_priority = TaskPriority::HIGH;
    } else {
        task_priority = TaskPriority::CRITICAL;
    }
    
    // 创建Task对象
    Task executor_task;
    executor_task.task_id = generate_task_id();
    executor_task.priority = task_priority;
    executor_task.function = [task]() { (*task)(); };
    executor_task.submit_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
    executor_task.timeout_ms = config_.task_timeout_ms;
    
    // 提交到调度器；持锁期间分发并 notify，避免错过唤醒
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_.load()) {
            std::promise<return_type> promise;
            promise.set_exception(std::make_exception_ptr(
                std::runtime_error("ThreadPool is stopped")
            ));
            return promise.get_future();
        }
        scheduler_.enqueue(executor_task);
        total_tasks_.fetch_add(1, std::memory_order_relaxed);
        if (dispatcher_) {
            dispatcher_->dispatch_batch(1);
        }
        condition_.notify_all();
    }
    
    return result;
}

} // namespace executor
