#pragma once

#include "executor/config.hpp"
#include "executor/types.hpp"
#include "priority_scheduler.hpp"
#include "load_balancer.hpp"
#include "task_dispatcher.hpp"

// 使用无锁队列优化（可通过 -DUSE_LOCKFREE_WORKER_QUEUE=ON 启用）
#ifdef USE_LOCKFREE_WORKER_QUEUE
#include "lockfree_worker_queue.hpp"
#else
#include "worker_local_queue.hpp"
#endif

#include "thread_pool_resizer.hpp"
#include "../util/exception_handler.hpp"
#include "../util/thread_utils.hpp"
#include "../task/task.hpp"
#include "../monitor/task_monitor.hpp"
#include <thread>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <future>
#include <chrono>
#include <functional>
#include <type_traits>
#include <random>
#include <stdexcept>

namespace executor {

#ifdef USE_LOCKFREE_WORKER_QUEUE
using WorkerQueueConcrete = LockFreeWorkerQueue;
#else
using WorkerQueueConcrete = WorkerLocalQueue;
#endif

class WorkerQueueImpl {
public:
    explicit WorkerQueueImpl(size_t capacity = 0)
        : queue_(std::make_unique<WorkerQueueConcrete>(capacity)) {}

    ~WorkerQueueImpl() = default;

    WorkerQueueImpl(const WorkerQueueImpl&) = delete;
    WorkerQueueImpl& operator=(const WorkerQueueImpl&) = delete;
    WorkerQueueImpl(WorkerQueueImpl&&) noexcept = default;
    WorkerQueueImpl& operator=(WorkerQueueImpl&&) noexcept = default;

    bool push(const Task& task) {
        return queue_->push(task);
    }

    bool push(Task&& task) {
        return queue_->push(std::move(task));
    }

    size_t push_batch(const Task* tasks, size_t n) {
        return queue_->push_batch(tasks, n);
    }

    bool pop(Task& task) {
        return queue_->pop(task);
    }

    bool steal(Task& task) {
        return queue_->steal(task);
    }

    size_t size() const {
        return queue_->size();
    }

    bool empty() const {
        return queue_->empty();
    }

    void clear() {
        queue_->clear();
    }

private:
    std::unique_ptr<WorkerQueueConcrete> queue_;
};

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
     * @note 实现内部调用 try_submit，若线程池已停止则返回带 std::runtime_error("ThreadPool is stopped") 异常的 future。
     */
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>;

    /**
     * @brief 提交任务，并报告是否被线程池接受
     *
     * @param task 任务函数
     * @return true 表示任务已入队；false 表示线程池已停止
     */
    bool try_submit(std::function<void()> task);

    bool try_submit(std::function<void()> task,
                    std::function<void(std::exception_ptr)> on_timeout);

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
     * @brief 提交优先级任务，并报告是否被线程池接受
     *
     * @param priority 优先级（0=LOW, 1=NORMAL, 2=HIGH, 3=CRITICAL）
     * @param task 任务函数
     * @return true 表示任务已入队；false 表示线程池已停止
     */
    bool try_submit_priority(int priority, std::function<void()> task);

    bool try_submit_priority(int priority,
                             std::function<void()> task,
                             std::function<void(std::exception_ptr)> on_timeout);

    /**
     * @brief 批量提交任务
     *
     * 批量提交多个任务，一次获取锁，减少锁竞争开销。
     *
     * @param tasks 任务列表
     */
    void submit_batch(std::vector<std::function<void()>> tasks);

    /**
     * @brief 批量提交任务，并报告是否被线程池接受
     *
     * @param tasks 任务列表
     * @return true 表示任务已入队；false 表示线程池已停止或任务为空
     */
    bool try_submit_batch(std::vector<std::function<void()>> tasks);

    bool try_submit_batch(
        std::vector<std::function<void()>> tasks,
        std::vector<std::function<void(std::exception_ptr)>> on_timeout_handlers);

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
     * @brief 重建工作线程本地队列（动态扩缩容 API）
     *
     * local_queues_ 通过 shared_ptr 发布新 vector。resize 路径持
     * unique_lock(local_queues_mutex_)，所有 worker / dispatcher 访问路径持
     * shared_lock 并先取得 shared_ptr 快照；旧 vector 会在最后一个快照释放后
     * 析构，不会因指针发布导致悬空引用。发布前会把旧队列中未执行任务放回
     * scheduler_，避免 shrink / replace 丢任务。
     *
     * @param new_num_queues 新的本地队列数量（与 worker 数一致）
     * @return 成功返回 true
     */
    bool resize_local_queues(size_t new_num_queues);

    /**
     * @brief 等待所有任务完成
     * 
     * 最多阻塞 kDefaultWaitForCompletionTimeout。
     */
    void wait_for_completion();

    /**
     * @brief 等待所有任务完成并返回是否完成
     *
     * @param timeout 最长等待时间
     * @return true 表示所有任务在 timeout 内完成；false 表示等待超时
     */
    bool try_wait_for_completion(std::chrono::milliseconds timeout);

    /**
     * @brief 检查线程池是否已停止
     * 
     * @return 如果线程池已停止返回true，否则返回false
     */
    bool is_stopped() const;

    /**
     * @brief 设置任务监控器（可选）
     *
     * 设置后，execute_task 前后将调用 record_task_start / record_task_complete。
     * ThreadPool 不拥有该指针；调用方必须保证 monitor 对象在所有可能
     * 已取得该指针快照的任务完成前保持存活。
     * @param m 监控器指针，可为 nullptr 表示禁用
     */
    void set_task_monitor(monitor::TaskMonitor* m);

    /**
     * @brief 获取超时任务计数
     *
     * 返回因软超时（elapsed >= timeout_ms at execution start）被跳过的任务数。
     * @return 超时任务计数
     */
    size_t get_timeout_count() const {
        return static_cast<size_t>(timeout_count_.load(std::memory_order_relaxed));
    }

#ifdef EXECUTOR_THREAD_POOL_TEST_HOOKS
    void set_worker_thread_start_hook_for_test(std::function<void(size_t)> hook) {
        worker_thread_start_hook_for_test_ = std::move(hook);
    }
#endif

private:
    /**
     * @brief RAII guard that increments active_threads_ on construction
     *        and decrements on destruction. Guarantees that any exception
     *        escaping the guarded scope (including monitor callbacks inside
     *        execute_task) cannot leave active_threads_ permanently inflated
     *        and hang wait_for_completion().
     *
     * P-001 (2026-06-22): replaces the prior fetch_add/fetch_sub pair around
     * the execute_task call in worker_thread, which leaked the decrement on
     * monitor-callback exception.
     */
    class ActiveCounter {
    public:
        explicit ActiveCounter(ThreadPool& pool) noexcept
            : pool_(pool) {
            pool_.active_threads_.fetch_add(1, std::memory_order_relaxed);
        }
        ~ActiveCounter() {
            pool_.active_threads_.fetch_sub(1, std::memory_order_relaxed);
            pool_.notify_completion_waiters();
        }
        ActiveCounter(const ActiveCounter&) = delete;
        ActiveCounter& operator=(const ActiveCounter&) = delete;
        ActiveCounter(ActiveCounter&&) = delete;
        ActiveCounter& operator=(ActiveCounter&&) = delete;
    private:
        ThreadPool& pool_;
    };

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
     * @param timed_out 是否因软超时而被跳过（P024: 超时不算 failed, 但算 completed）
     */
    void update_statistics(int64_t execution_time_ns, bool success, bool timed_out = false);

    /**
     * @brief 检查 wait_for_completion 的完整完成条件
     */
    bool is_completion_ready() const;

    /**
     * @brief 唤醒等待任务完成的调用方
     */
    void notify_completion_waiters();

    /**
     * @brief Dispatch pending scheduler tasks if the dispatcher is still alive.
     */
    size_t dispatch_pending_tasks(size_t max_tasks);

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
     * @brief try_steal_task 的内部实现
     *
     * P-260617-002: 假设调用方已持 shared_lock(local_queues_mutex_)。
     * worker_thread 的 condition_.wait 谓词需要"已持 shared_lock 时
     * 也能调窃取逻辑"，因此拆出本函数避免 std::shared_mutex 重入 UB。
     */
    bool try_steal_task_impl(size_t worker_id, Task& task);

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
     * @brief Roll back a failed initialize() attempt.
     */
    void rollback_initialization_failure();

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
    std::shared_ptr<std::vector<WorkerQueueImpl>> local_queues_;
    // 260610P012: 专门保护 local_queues_ 的 shared_mutex
    // - steal / worker 持 shared_lock(并发读)
    // - resize / shutdown 持 unique_lock(排他写)
    // 这样既消除 UAF,又允许多个 steal 线程并发。
    mutable std::shared_mutex local_queues_mutex_;

    // 任务分发器（模板化以匹配 WorkerQueueImpl）
    std::unique_ptr<TaskDispatcher<WorkerQueueImpl>> dispatcher_;
    mutable std::mutex dispatcher_mutex_;

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
    std::atomic<int64_t> timeout_count_{0};  // P024: 软超时跳过的任务计数

    // 条件变量：用于工作线程等待任务
    std::condition_variable condition_;

    // 条件变量：用于 wait_for_completion 等待所有任务完成
    std::condition_variable completion_cv_;
    mutable std::mutex completion_mutex_;

    // 互斥锁：保护共享状态
    mutable std::mutex mutex_;

    // P-008: shutdown() 并发调用的幂等性保护
    // std::call_once 用 flag,保证 shutdown 逻辑全生命周期只执行一次,
    // 避免两个线程同时通过 stop_.load() 早返回检查后 double-join 触发 UB。
    mutable std::once_flag shutdown_once_flag_;

    // 异常处理器
    util::ExceptionHandler exception_handler_;

    // 可选任务监控器（不拥有）。允许运行中 set/unset；execute_task 每个任务
    // acquire-load 一次快照，set_task_monitor release-store 发布新指针。
    std::atomic<monitor::TaskMonitor*> monitor_{nullptr};

    // 初始化标志
    std::atomic<bool> initialized_{false};

    // 待退出的线程ID集合（用于缩容）
    mutable std::mutex exit_threads_mutex_;
    std::vector<size_t> exit_threads_;

    // 监控线程（用于动态扩缩容）
    std::thread resize_monitor_thread_;
    std::atomic<bool> resize_monitor_stop_{false};
    std::condition_variable resize_monitor_cv_;
    mutable std::mutex resize_monitor_mutex_;

    std::function<void(size_t)> worker_thread_start_hook_for_test_;
};

// 模板方法实现
template<typename F, typename... Args>
auto ThreadPool::submit(F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type> {
    
    using return_type = typename std::invoke_result<F, Args...>::type;
    
    auto promise = std::make_shared<std::promise<return_type>>();
    auto promise_ready = std::make_shared<std::atomic_bool>(false);
    std::future<return_type> result = promise->get_future();

    if constexpr (std::is_same_v<
                      std::remove_cv_t<std::remove_reference_t<F>>,
                      std::function<return_type(Args...)>>) {
        if (!f) {
            promise_ready->store(true, std::memory_order_release);
            promise->set_exception(
                std::make_exception_ptr(std::invalid_argument("empty task")));
            return result;
        }
    }

    auto bound_task = std::make_shared<decltype(std::bind(std::forward<F>(f), std::forward<Args>(args)...))>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    auto task = [promise, promise_ready, bound_task]() mutable {
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
            bool expected = false;
            if (promise_ready->compare_exchange_strong(expected, true)) {
                promise->set_exception(exception);
            }
            throw;
        }
    };

    auto on_timeout = [promise, promise_ready](std::exception_ptr exception) {
        bool expected = false;
        if (promise_ready->compare_exchange_strong(expected, true)) {
            promise->set_exception(exception);
        }
    };
    
    if (!try_submit(std::move(task), std::move(on_timeout))) {
        auto exception = std::make_exception_ptr(
            std::runtime_error("ThreadPool is stopped"));
        bool expected = false;
        if (promise_ready->compare_exchange_strong(expected, true)) {
            promise->set_exception(exception);
        }
    }
    
    return result;
}

template<typename F, typename... Args>
auto ThreadPool::submit_priority(int priority, F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type> {
    
    using return_type = typename std::invoke_result<F, Args...>::type;
    
    auto promise = std::make_shared<std::promise<return_type>>();
    auto promise_ready = std::make_shared<std::atomic_bool>(false);
    std::future<return_type> result = promise->get_future();

    if constexpr (std::is_same_v<
                      std::remove_cv_t<std::remove_reference_t<F>>,
                      std::function<return_type(Args...)>>) {
        if (!f) {
            promise_ready->store(true, std::memory_order_release);
            promise->set_exception(
                std::make_exception_ptr(std::invalid_argument("empty task")));
            return result;
        }
    }

    auto bound_task = std::make_shared<decltype(std::bind(std::forward<F>(f), std::forward<Args>(args)...))>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    auto task = [promise, promise_ready, bound_task]() mutable {
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
            bool expected = false;
            if (promise_ready->compare_exchange_strong(expected, true)) {
                promise->set_exception(exception);
            }
            throw;
        }
    };

    auto on_timeout = [promise, promise_ready](std::exception_ptr exception) {
        bool expected = false;
        if (promise_ready->compare_exchange_strong(expected, true)) {
            promise->set_exception(exception);
        }
    };
    
    if (!try_submit_priority(priority, std::move(task), std::move(on_timeout))) {
        auto exception = std::make_exception_ptr(
            std::runtime_error("ThreadPool is stopped"));
        bool expected = false;
        if (promise_ready->compare_exchange_strong(expected, true)) {
            promise->set_exception(exception);
        }
    }
    
    return result;
}

} // namespace executor
