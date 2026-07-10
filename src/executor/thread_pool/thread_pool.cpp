#include "thread_pool.hpp"
#include "../task/task.hpp"
#include <stdexcept>
#include <algorithm>
#include <cstdio>
#include <random>
#include <new>
#include <vector>

namespace executor {

// thread_local 变量会在首次使用时自动初始化

ThreadPool::ThreadPool() : stop_(false), initialized_(false) {
}

ThreadPool::~ThreadPool() {
    shutdown(true);
}

bool ThreadPool::initialize(const ThreadPoolConfig& config) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    if (initialized_.load()) {
        return false;  // 已经初始化过
    }
    
    // 验证配置
    if (config.min_threads == 0 || config.min_threads > config.max_threads) {
        return false;
    }
    
    config_ = config;

    try {
        stop_.store(false);
        resize_monitor_stop_.store(false);

        // 初始化负载均衡器
        load_balancer_ = std::make_unique<LoadBalancer>(config_.min_threads);

        // 设置负载均衡策略
        if (config_.enable_work_stealing) {
            load_balancer_->set_strategy(LoadBalancer::Strategy::LEAST_TASKS);
        } else {
            load_balancer_->set_strategy(LoadBalancer::Strategy::ROUND_ROBIN);
        }

        // 初始化工作线程本地队列
        auto new_queues = std::make_shared<std::vector<WorkerQueueImpl>>();
        new_queues->reserve(config_.min_threads);
        for (size_t i = 0; i < config_.min_threads; ++i) {
            new_queues->emplace_back(config_.queue_capacity);
        }
        std::atomic_store_explicit(&local_queues_, new_queues, std::memory_order_release);

        // 初始化任务分发器（TaskDispatcher 是模板类，需要显式指定实例化类型）
        // P-260617-002: 传入 local_queues_mutex_ 指针,dispatcher 内部 dispatch
        // 路径会持 shared_lock，与 resize 路径的 unique_lock 配对防 UAF。
        {
            std::lock_guard<std::mutex> dispatcher_lock(dispatcher_mutex_);
            dispatcher_ = std::make_unique<TaskDispatcher<WorkerQueueImpl>>(
                *load_balancer_, scheduler_, &local_queues_, &local_queues_mutex_
            );
        }

        // 初始化动态扩缩容控制器
        resizer_ = std::make_unique<ThreadPoolResizer>(*this, config_);

        // 创建工作线程
        workers_.reserve(config_.min_threads);
        worker_ids_.reserve(config_.min_threads);
        for (size_t i = 0; i < config_.min_threads; ++i) {
            worker_ids_.push_back(i);
            create_worker_thread(i);
        }

        // 启动监控线程（用于动态扩缩容）
        if (config_.max_threads > config_.min_threads) {
            resize_monitor_thread_ = std::thread(&ThreadPool::resize_monitor_thread, this);
        }
    } catch (...) {
        lock.unlock();
        rollback_initialization_failure();
        return false;
    }

    initialized_.store(true);
    return true;
}

void ThreadPool::rollback_initialization_failure() {
    initialized_.store(false);
    stop_.store(true);
    {
        std::lock_guard<std::mutex> lock(resize_monitor_mutex_);
        resize_monitor_stop_.store(true, std::memory_order_release);
    }
    resize_monitor_cv_.notify_all();
    condition_.notify_all();

    if (resize_monitor_thread_.joinable()) {
        resize_monitor_thread_.join();
    }

    for (size_t i = 0; i < workers_.size(); ++i) {
        auto& worker = workers_[i];
        if (worker.joinable()) {
            worker.join();
        }
    }

    workers_.clear();
    worker_ids_.clear();
    {
        std::lock_guard<std::mutex> dispatcher_lock(dispatcher_mutex_);
        dispatcher_.reset();
    }
    resizer_.reset();
    load_balancer_.reset();
    scheduler_.clear();

    {
        std::lock_guard<std::mutex> exit_lock(exit_threads_mutex_);
        exit_threads_.clear();
    }

    {
        std::unique_lock<std::shared_mutex> lq_lock(local_queues_mutex_);
        auto empty_queues = std::make_shared<std::vector<WorkerQueueImpl>>();
        std::atomic_store_explicit(&local_queues_, empty_queues, std::memory_order_release);
    }

    resize_monitor_stop_.store(false);
    stop_.store(false);
    notify_completion_waiters();
}

void ThreadPool::worker_thread(size_t worker_id) {
    while (true) {
        Task task;
        bool has_task = false;

        if ((stop_.load(std::memory_order_acquire) &&
             !initialized_.load(std::memory_order_acquire)) ||
            should_exit(worker_id)) {
            break;
        }

        // 1. 优先从本地队列获取任务
        // P-260617-002: 持 shared_lock(local_queues_mutex_)，与 resize/shutdown
        // 路径的 unique_lock 配对，防止 vector reallocation 期间悬空访问。
        {
            std::shared_lock<std::shared_mutex> lq_lock(local_queues_mutex_);
            auto queues = std::atomic_load_explicit(&local_queues_, std::memory_order_acquire);
            if (queues && worker_id < queues->size() && (*queues)[worker_id].pop(task)) {
                has_task = true;
            }
        }
        // 2. 如果本地队列为空，尝试工作窃取
        // try_steal_task 内部已自行持 shared_lock(local_queues_mutex_)，
        // 这里的 shared_lock 已释放，无嵌套问题。
        if (!has_task && config_.enable_work_stealing) {
            has_task = try_steal_task(worker_id, task);
        }

        // 3. 若无任务，加锁后等待；谓词内再次检查本地队列、窃取、全局队列、退出条件
        // P-260617-002: 谓词持 shared_lock(local_queues_mutex_)。try_steal_task
        // 拆为公开入口(持 shared_lock)与 impl(假设调用方已持)两个版本，谓词
        // 调 impl 避免 std::shared_mutex 重入 UB。unique_lock(mutex_) 与
        // shared_lock(shared_mutex) 是不同互斥量，不存在互感知。
        if (!has_task) {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this, &has_task, &task, worker_id]() {
                if ((stop_.load(std::memory_order_acquire) &&
                     !initialized_.load(std::memory_order_acquire)) ||
                    should_exit(worker_id)) {
                    return true;
                }

                std::shared_lock<std::shared_mutex> lq_lock(local_queues_mutex_);
                auto queues = std::atomic_load_explicit(&local_queues_, std::memory_order_acquire);
                if (queues && worker_id < queues->size() && (*queues)[worker_id].pop(task)) {
                    has_task = true;
                    return true;
                }
                if (config_.enable_work_stealing) {
                    // 调 impl: 当前线程已持 shared_lock，impl 不会再获取
                    has_task = try_steal_task_impl(worker_id, task);
                    if (has_task) return true;
                }
                has_task = scheduler_.dequeue(task);
                if (has_task) return true;
                if (stop_.load() || should_exit(worker_id)) return true;
                return false;
            });
            // 被 notify 唤醒但谓词未取到任务时，再试一次本地队列（可能刚被分发）
            // 即使 stop_ 为 true，也要检查本地队列以排空任务
            // P-260617-002: predicate 内的 lq_lock 已随 lambda 析构，
            // 此处重新加 shared_lock 后再访问。
            if (!has_task) {
                std::shared_lock<std::shared_mutex> lq_lock(local_queues_mutex_);
                auto queues = std::atomic_load_explicit(&local_queues_, std::memory_order_acquire);
                if (queues && worker_id < queues->size()) {
                    has_task = (*queues)[worker_id].pop(task);
                }
            }
        }

        // 执行任务（即使 stop_ 为 true，也要执行已获取的任务以排空队列）
        if (has_task) {
            // 检查任务是否已取消
            if (!is_task_cancelled(task)) {
                // P-001 (2026-06-22): RAII guard guarantees active_threads_
                // is decremented even if monitor_ callbacks throw inside
                // execute_task. Previously a fetch_add/fetch_sub pair leaked
                // the decrement on monitor exception, killing the worker
                // and hanging wait_for_completion().
                ThreadPool::ActiveCounter active_guard(*this);
                execute_task(task);
            } else {
                // 任务被取消，也需要更新统计信息
                completed_tasks_.fetch_add(1, std::memory_order_relaxed);
                notify_completion_waiters();
            }

            // 更新负载信息
            // P-260617-002: size() 必须持 shared_lock 访问 local_queues_
            if (load_balancer_) {
                std::shared_lock<std::shared_mutex> lq_lock(local_queues_mutex_);
                auto queues = std::atomic_load_explicit(&local_queues_, std::memory_order_acquire);
                if (queues && worker_id < queues->size()) {
                    size_t queue_size = (*queues)[worker_id].size();
                    load_balancer_->update_load(worker_id, queue_size, 0);
                }
            }
        }

        // 检查是否需要退出（缩容时）
        if (should_exit(worker_id) || (stop_.load() && !has_task)) {
            break;
        }

        // 触发任务分发（从全局调度器分发到本地队列）
        // P-260617-002: dispatcher 内部 dispatch_batch 自身已持 shared_lock，
        // 此处不能再加 shared_lock（std::shared_mutex 不可重入 -> UB）。
        if (!stop_.load()) {
            size_t dispatched = dispatch_pending_tasks(5);  // 批量分发，减少锁竞争
            // 如果成功分发了任务，唤醒等待的线程
            if (dispatched > 0) {
                std::lock_guard<std::mutex> lock(mutex_);
                condition_.notify_all();
            }
        }
    }
}

void ThreadPool::execute_task(const Task& task) {
    // P-001 (2026-06-22): catch-all guards against exceptions escaping
    // monitor_->record_task_start/complete callbacks, which previously
    // propagated out of execute_task, killed the worker thread, and
    // leaked the active_threads_ decrement (now fixed by ActiveCounter
    // RAII in worker_thread).
    //
    // start_time is declared BEFORE the try so the catch-all can still
    // reference it when computing execution_time_ns for the recovery
    // update_statistics() call below.
    auto start_time = std::chrono::steady_clock::now();

    try {
        auto* monitor = monitor_.load(std::memory_order_acquire);
        if (monitor && monitor->is_enabled()) {
            monitor->record_task_start(task.task_id, "default");
        }

    // P024 soft timeout: check elapsed time BEFORE execution.
    // If elapsed >= timeout at execution start, skip the task entirely and
    // record it as timed out.  This is a pre-execution check only — C++ has
    // no safe mechanism to forcefully kill a running thread, so in-progress
    // tasks are never interrupted.
    bool timed_out = false;
    if (task.timeout_ms > 0 && task.submit_time_ns > 0) {
        int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count();
        int64_t timeout_ns = task.timeout_ms * 1'000'000;
        if ((now_ns - task.submit_time_ns) >= timeout_ns) {
            timed_out = true;
        }
    }

    bool success = false;

    if (!timed_out) {
        try {
            // 执行任务
            if (task.function) {
                task.function();
                success = true;
            }
        } catch (...) {
            // 捕获所有异常，通过ExceptionHandler处理
            exception_handler_.handle_task_exception("ThreadPool", std::current_exception());
            success = false;
        }
    } else {
        // 软超时：跳过执行，记录超时计数
        timeout_count_.fetch_add(1, std::memory_order_relaxed);
        auto timeout_exception = std::make_exception_ptr(TimedOutException(
            "Task timed out after " + std::to_string(task.timeout_ms) + "ms"));
        if (task.on_timeout) {
            try {
                task.on_timeout(timeout_exception);
            } catch (...) {
                exception_handler_.handle_task_exception(
                    "ThreadPool::execute_task timeout callback",
                    std::current_exception());
            }
        }
        if (monitor && monitor->is_enabled()) {
            monitor->record_task_timeout(task.task_id);
        }
        // 标记 timed_out 用于 update_statistics: completed++ 但不 failed++。
        // wait_for_completion() 以 completed 覆盖全部已结束任务，failed 是其子集。
    }

    auto end_time = std::chrono::steady_clock::now();
    int64_t execution_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_time - start_time
    ).count();

    if (!timed_out && monitor && monitor->is_enabled()) {
        monitor->record_task_complete(task.task_id, success, execution_time_ns);
    }
    update_statistics(execution_time_ns, success, timed_out);
    } catch (...) {
        // Monitor callbacks are user-supplied; their exceptions must not
        // kill the worker. Report via the configured exception handler so
        // operators see them, then return normally — ActiveCounter in
        // worker_thread will still decrement active_threads_.
        //
        // CRITICAL: also fire update_statistics here. Otherwise
        // completed_tasks_ never increments and wait_for_completion()
        // hangs forever (or up to 300s) waiting for total == completed.
        exception_handler_.handle_task_exception("ThreadPool::execute_task",
                                                 std::current_exception());
        // Best-effort stats: counted as completed but NOT failed (a
        // monitor exception is not a task failure — the user code
        // didn't even run). This matches the soft-timeout branch
        // (timed_out=true) which also counts as completed without
        // incrementing failed_.
        int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now() - start_time).count();
        update_statistics(ns, /*success=*/false, /*timed_out=*/true);
    }
}

void ThreadPool::update_statistics(int64_t execution_time_ns, bool success, bool timed_out) {
    completed_tasks_.fetch_add(1, std::memory_order_relaxed);

    if (!success && !timed_out) {
        // 软超时不计入 failed (有专门的 timeout_count); 只有异常/失败计入 failed
        failed_tasks_.fetch_add(1, std::memory_order_relaxed);
    }

    // 更新总执行时间（使用原子操作累加）
    total_execution_time_ns_.fetch_add(execution_time_ns, std::memory_order_relaxed);

    notify_completion_waiters();
}

ThreadPoolStatus ThreadPool::get_status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    ThreadPoolStatus status;
    status.total_threads = workers_.size();
    status.active_threads = active_threads_.load(std::memory_order_relaxed);
    // Guard against size_t underflow: active_threads_ is a relaxed atomic
    // that may briefly exceed workers_.size() during resize() (e.g. when
    // workers are being torn down but the counter has not yet been
    // decremented). Saturate to 0 instead of wrapping to a huge value.
    status.idle_threads = (status.active_threads <= status.total_threads)
                              ? (status.total_threads - status.active_threads)
                              : 0;
    
    // 队列大小 = 全局调度器 + 所有本地队列
    // P-260617-002: 持 shared_lock 防止与并发 resize 数据竞争
    size_t local_queue_size = 0;
    {
        std::shared_lock<std::shared_mutex> lq_lock(local_queues_mutex_);
        auto queues = std::atomic_load_explicit(&local_queues_, std::memory_order_acquire);
        if (queues) {
            for (const auto& queue : *queues) {
                local_queue_size += queue.size();
            }
        }
    }
    status.queue_size = scheduler_.size() + local_queue_size;
    
    status.total_tasks = total_tasks_.load(std::memory_order_relaxed);
    status.completed_tasks = completed_tasks_.load(std::memory_order_relaxed);
    status.failed_tasks = failed_tasks_.load(std::memory_order_relaxed);
    
    // 计算平均任务执行时间
    size_t completed = completed_tasks_.load(std::memory_order_relaxed);
    if (completed > 0) {
        int64_t total_time = total_execution_time_ns_.load(std::memory_order_relaxed);
        status.avg_task_time_ms = (static_cast<double>(total_time) / static_cast<double>(completed)) / 1e6;
    } else {
        status.avg_task_time_ms = 0.0;
    }
    
    // CPU使用率暂不实现（需要系统调用）
    status.cpu_usage_percent = 0.0;
    
    return status;
}

void ThreadPool::shutdown(bool wait_for_tasks) {
    // P-008 修复: 用 std::call_once 保证 shutdown 逻辑只执行一次,
    // 避免两个线程同时通过 stop_.load() 检查后 double-join resize_monitor_thread_
    // 或 workers_ 中已 joinable 的线程(触发 std::system_error, UB)。
    // call_once 的好处是: 持锁窗口极短(只覆盖 stop_ 标志翻转),
    // join() 在锁外执行,不会与工作线程持 mutex_ 死锁。
    std::call_once(shutdown_once_flag_, [this, wait_for_tasks]() {
        // 如果需要等待任务完成，先等待（此时 stop_ 仍为 false，工作线程继续运行）
        if (wait_for_tasks) {
            wait_for_completion();
        }

        // 设置停止标志
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_.store(true);
        }

        // 停止监控线程
        {
            std::lock_guard<std::mutex> lock(resize_monitor_mutex_);
            resize_monitor_stop_.store(true, std::memory_order_release);
        }
        resize_monitor_cv_.notify_all();
        if (resize_monitor_thread_.joinable()) {
            resize_monitor_thread_.join();
        }

        // 唤醒所有等待的线程
        condition_.notify_all();

        // 等待所有工作线程退出
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }

        workers_.clear();
        worker_ids_.clear();
        {
            std::lock_guard<std::mutex> dispatcher_lock(dispatcher_mutex_);
            dispatcher_.reset();
        }
        // P-260617-002: shutdown 时所有 worker 已 join 完毕，无并发 reader，
        // 仍持 unique_lock 清空 local_queues_ 以与 worker 路径的 shared_lock
        // 保持配对语义（reader 一律持 shared_lock，写者一律持 unique_lock）。
        {
            std::unique_lock<std::shared_mutex> lq_lock(local_queues_mutex_);
            auto empty_queues = std::make_shared<std::vector<WorkerQueueImpl>>();
            std::atomic_store_explicit(&local_queues_, empty_queues, std::memory_order_release);
        }
    });
}

bool ThreadPool::resize_local_queues(size_t new_num_queues) {
    if (new_num_queues == 0) {
        return false;
    }

    auto new_queues = std::make_shared<std::vector<WorkerQueueImpl>>();
    new_queues->reserve(new_num_queues);
    for (size_t i = 0; i < new_num_queues; ++i) {
        new_queues->emplace_back(config_.queue_capacity);
    }

    std::unique_lock<std::shared_mutex> lq_lock(local_queues_mutex_);
    auto old_queues = std::atomic_load_explicit(&local_queues_, std::memory_order_acquire);
    if (old_queues) {
        for (auto& queue : *old_queues) {
            Task task;
            while (queue.pop(task)) {
                scheduler_.enqueue(task);
            }
        }
    }

    std::atomic_store_explicit(&local_queues_, new_queues, std::memory_order_release);

    if (load_balancer_) {
        load_balancer_->resize(new_num_queues);
    }

    lq_lock.unlock();
    condition_.notify_all();
    notify_completion_waiters();
    return true;
}

void ThreadPool::wait_for_completion() {
    (void)try_wait_for_completion(kDefaultWaitForCompletionTimeout);
}

bool ThreadPool::try_wait_for_completion(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(completion_mutex_);
    return completion_cv_.wait_for(lock, timeout, [this]() {
        return is_completion_ready();
    });
}

bool ThreadPool::is_completion_ready() const {
    // 等待所有任务完成需要同时满足：
    // 1. 全局调度器 + 所有本地队列为空（没有待执行的任务）
    // 2. 没有活跃线程（没有正在执行的任务）
    // 3. 所有已提交的任务都已完成（total == completed）。
    // failed_tasks_ 是 completed_tasks_ 的失败子集，不再参与完成等式。
    bool scheduler_empty = scheduler_.empty();

    size_t local_queue_total = 0;
    {
        std::shared_lock<std::shared_mutex> lq_lock(local_queues_mutex_);
        auto queues = std::atomic_load_explicit(&local_queues_, std::memory_order_acquire);
        if (queues) {
            for (const auto& queue : *queues) {
                local_queue_total += queue.size();
            }
        }
    }

    size_t total = total_tasks_.load(std::memory_order_acquire);
    size_t completed = completed_tasks_.load(std::memory_order_acquire);
    size_t active = active_threads_.load(std::memory_order_acquire);

    return scheduler_empty &&
           local_queue_total == 0 &&
           active == 0 &&
           total == completed;
}

void ThreadPool::notify_completion_waiters() {
    completion_cv_.notify_all();
}

size_t ThreadPool::dispatch_pending_tasks(size_t max_tasks) {
    std::lock_guard<std::mutex> lock(dispatcher_mutex_);
    if (!dispatcher_) {
        return 0;
    }
    return dispatcher_->dispatch_batch(max_tasks);
}

bool ThreadPool::is_stopped() const {
    return stop_.load(std::memory_order_relaxed);
}

void ThreadPool::set_task_monitor(monitor::TaskMonitor* m) {
    monitor_.store(m, std::memory_order_release);
}

bool ThreadPool::try_steal_task(size_t worker_id, Task& task) {
    // P-260617-002: 公开入口持 shared_lock，内部实现 try_steal_task_impl
    // 假设调用方已持 shared_lock，避免 worker_thread 谓词中重入 shared_lock
    // (std::shared_mutex 不可重入 -> UB)。
    std::shared_lock<std::shared_mutex> lock(local_queues_mutex_);
    return try_steal_task_impl(worker_id, task);
}

bool ThreadPool::try_steal_task_impl(size_t worker_id, Task& task) {
    // P-260617-002: 调用方必须已持 shared_lock(local_queues_mutex_)。
    // 内部不再获取该锁。worker_thread 谓词中已持 shared_lock 时调用此函数
    // 不会重入，避免 std::shared_mutex 重入 UB。
    auto queues = std::atomic_load_explicit(&local_queues_, std::memory_order_acquire);
    if (!queues || queues->size() <= 1) {
        return false;  // 只有一个线程，无法窃取
    }

    // 尝试使用基于负载的智能窃取策略
    if (load_balancer_) {
        // 获取所有线程的负载信息
        std::vector<LoadBalancer::WorkerLoad> loads = load_balancer_->get_all_loads();

        if (loads.size() == queues->size()) {
            // 创建线程ID和负载的配对，用于排序
            std::vector<std::pair<size_t, size_t>> worker_loads;
            worker_loads.reserve(queues->size());

            for (size_t i = 0; i < queues->size(); ++i) {
                if (i != worker_id) {  // 跳过自己
                    // 计算总负载：队列大小 + 活跃任务数
                    size_t total_load = loads[i].queue_size + loads[i].active_tasks;
                    worker_loads.emplace_back(i, total_load);
                }
            }

            // 检查是否所有线程负载相同
            bool all_same_load = true;
            if (!worker_loads.empty()) {
                size_t first_load = worker_loads[0].second;
                for (const auto& wl : worker_loads) {
                    if (wl.second != first_load) {
                        all_same_load = false;
                        break;
                    }
                }
            }

            // 如果负载不同，按负载从高到低排序，优先窃取负载高的线程
            if (!all_same_load && !worker_loads.empty()) {
                std::sort(worker_loads.begin(), worker_loads.end(),
                    [](const std::pair<size_t, size_t>& a, const std::pair<size_t, size_t>& b) {
                        return a.second > b.second;  // 降序排序
                    });

                // 按排序顺序尝试窃取
                for (const auto& wl : worker_loads) {
                    size_t target_id = wl.first;
                    if ((*queues)[target_id].steal(task)) {
                        // 更新目标线程的负载信息
                        size_t queue_size = (*queues)[target_id].size();
                        load_balancer_->update_load(target_id, queue_size, 0);
                        return true;
                    }
                }
            }
        }
    }

    // 回退到随机策略（如果无法获取负载信息或所有线程负载相同）
    // 使用 thread_local 随机数生成器（首次使用时会自动初始化）
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, queues->size() - 1);
    size_t start_index = dist(rng);

    // 尝试从其他线程窃取任务
    for (size_t i = 0; i < queues->size(); ++i) {
        size_t target_id = (start_index + i) % queues->size();

        // 跳过自己
        if (target_id == worker_id) {
            continue;
        }

        // 尝试窃取
        if ((*queues)[target_id].steal(task)) {
            // 更新目标线程的负载信息
            if (load_balancer_) {
                size_t queue_size = (*queues)[target_id].size();
                load_balancer_->update_load(target_id, queue_size, 0);
            }
            return true;
        }
    }

    return false;
}

bool ThreadPool::should_exit(size_t worker_id) const {
    std::lock_guard<std::mutex> lock(exit_threads_mutex_);
    return std::find(exit_threads_.begin(), exit_threads_.end(), worker_id) 
           != exit_threads_.end();
}

void ThreadPool::resize_monitor_thread() {
    while (true) {
        std::unique_lock<std::mutex> lock(resize_monitor_mutex_);
        if (resize_monitor_cv_.wait_for(lock, std::chrono::seconds(1), [this]() {
                return resize_monitor_stop_.load(std::memory_order_acquire);
            })) {
            break;
        }
        lock.unlock();
        
        // 更新线程池状态信息
        ThreadPoolStatus status = get_status();
        
        // 计算平均等待时间（简化实现，使用队列大小估算）
        double avg_wait_time_ms = 0.0;
        if (status.queue_size > 0 && status.total_threads > 0) {
            // 假设每个任务平均执行时间，估算等待时间
            avg_wait_time_ms = (static_cast<double>(status.queue_size) * status.avg_task_time_ms) 
                               / static_cast<double>(status.total_threads);
        }
        
        if (resizer_) {
            resizer_->update_status(
                status.queue_size,
                status.active_threads,
                status.total_threads,
                avg_wait_time_ms
            );
            
            // 检查并执行扩缩容
            resizer_->check_and_resize();
        }
    }
}

void ThreadPool::create_worker_thread(size_t worker_id) {
    if (worker_thread_start_hook_for_test_) {
        worker_thread_start_hook_for_test_(worker_id);
    }

    workers_.emplace_back(&ThreadPool::worker_thread, this, worker_id);

    // 设置线程优先级
    if (config_.thread_priority != 0) {
        util::set_thread_priority(workers_.back().native_handle(),
                                 config_.thread_priority);
    }

    // 设置CPU亲和性
    if (!config_.cpu_affinity.empty()) {
        int cpu_id = config_.cpu_affinity[worker_id % config_.cpu_affinity.size()];
        util::set_cpu_affinity(workers_.back().native_handle(), {cpu_id});
    }
}

bool ThreadPool::try_submit(std::function<void()> task) {
    return try_submit(std::move(task), {});
}

bool ThreadPool::try_submit(std::function<void()> task,
                            std::function<void(std::exception_ptr)> on_timeout) {
    if (!task) {
        if (on_timeout) {
            on_timeout(std::make_exception_ptr(std::invalid_argument("empty task")));
        }
        return false;
    }

    Task executor_task;
    executor_task.task_id = generate_task_id();
    executor_task.priority = TaskPriority::NORMAL;
    executor_task.function = std::move(task);
    executor_task.on_timeout = std::move(on_timeout);
    executor_task.submit_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
    executor_task.timeout_ms = config_.task_timeout_ms;

    // Keep mutex_ scoped to the ThreadPool state change. Dispatching may take
    // local queue / load-balancer / scheduler locks, so doing it after releasing
    // mutex_ avoids lock-order inversions with workers and shutdown.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_.load()) {
            return false;
        }

        scheduler_.enqueue(executor_task);
        total_tasks_.fetch_add(1, std::memory_order_relaxed);
    }

    dispatch_pending_tasks(1);
    condition_.notify_all();

    return true;
}

bool ThreadPool::try_submit_priority(int priority, std::function<void()> task) {
    return try_submit_priority(priority, std::move(task), {});
}

bool ThreadPool::try_submit_priority(
    int priority,
    std::function<void()> task,
    std::function<void(std::exception_ptr)> on_timeout) {
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

    if (!task) {
        if (on_timeout) {
            on_timeout(std::make_exception_ptr(std::invalid_argument("empty task")));
        }
        return false;
    }

    Task executor_task;
    executor_task.task_id = generate_task_id();
    executor_task.priority = task_priority;
    executor_task.function = std::move(task);
    executor_task.on_timeout = std::move(on_timeout);
    executor_task.submit_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
    executor_task.timeout_ms = config_.task_timeout_ms;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_.load()) {
            return false;
        }

        scheduler_.enqueue(executor_task);
        total_tasks_.fetch_add(1, std::memory_order_relaxed);
    }

    dispatch_pending_tasks(1);
    condition_.notify_all();

    return true;
}

void ThreadPool::submit_batch(std::vector<std::function<void()>> tasks) {
    (void)try_submit_batch(std::move(tasks));
}

bool ThreadPool::try_submit_batch(std::vector<std::function<void()>> tasks) {
    return try_submit_batch(std::move(tasks), {});
}

bool ThreadPool::try_submit_batch(
    std::vector<std::function<void()>> tasks,
    std::vector<std::function<void(std::exception_ptr)>> on_timeout_handlers) {
    if (tasks.empty()) {
        return false;
    }

    for (const auto& task : tasks) {
        if (!task) {
            auto exception = std::make_exception_ptr(std::invalid_argument("empty task"));
            for (auto& on_timeout : on_timeout_handlers) {
                if (on_timeout) {
                    on_timeout(exception);
                }
            }
            return false;
        }
    }

    std::vector<std::string> task_ids;
    task_ids.reserve(tasks.size());
    for (size_t i = 0; i < tasks.size(); ++i) {
        task_ids.push_back(generate_task_id());
    }

    size_t batch_size = tasks.size();

    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (stop_.load()) {
            return false;  // 线程池已停止，拒绝任务
        }

        // 批量创建并入队任务
        for (size_t i = 0; i < tasks.size(); ++i) {
            Task executor_task;
            executor_task.task_id = std::move(task_ids[i]);
            executor_task.priority = TaskPriority::NORMAL;
            executor_task.function = std::move(tasks[i]);
            if (i < on_timeout_handlers.size()) {
                executor_task.on_timeout = std::move(on_timeout_handlers[i]);
            }
            executor_task.submit_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()
            ).count();
            executor_task.timeout_ms = config_.task_timeout_ms;

            scheduler_.enqueue(executor_task);
        }

        total_tasks_.fetch_add(batch_size, std::memory_order_relaxed);
    }

    // 批量分发
    dispatch_pending_tasks(batch_size);

    // 唤醒所有等待的工作线程
    condition_.notify_all();

    return true;
}

} // namespace executor
