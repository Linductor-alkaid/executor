#include "thread_pool.hpp"
#include "../task/task.hpp"
#include <stdexcept>
#include <algorithm>
#include <random>

namespace executor {

// thread_local 变量会在首次使用时自动初始化

ThreadPool::ThreadPool() : stop_(false), initialized_(false) {
}

ThreadPool::~ThreadPool() {
    shutdown(true);
}

bool ThreadPool::initialize(const ThreadPoolConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (initialized_.load()) {
        return false;  // 已经初始化过
    }
    
    // 验证配置
    if (config.min_threads == 0 || config.min_threads > config.max_threads) {
        return false;
    }
    
    config_ = config;
    
    // 初始化负载均衡器
    load_balancer_ = std::make_unique<LoadBalancer>(config_.min_threads);
    
    // 设置负载均衡策略
    if (config_.enable_work_stealing) {
        load_balancer_->set_strategy(LoadBalancer::Strategy::LEAST_TASKS);
    } else {
        load_balancer_->set_strategy(LoadBalancer::Strategy::ROUND_ROBIN);
    }
    
    // 初始化工作线程本地队列
    // 注意：WorkerLocalQueue 包含不可移动的 mutex，不能使用 reserve/emplace_back/resize
    // 使用 vector 的构造函数直接构造新 vector，然后使用 placement new 重新构造每个元素
    // 先销毁现有元素（如果有）
    for (auto& queue : local_queues_) {
        queue.~WorkerLocalQueue();
    }
    
    // 使用 vector 的构造函数直接分配内存（默认构造所有元素）
    // 使用 swap 避免赋值操作可能触发的移动
    std::vector<WorkerLocalQueue> new_queues(config_.min_threads);
    local_queues_.swap(new_queues);
    
    // 使用 placement new 重新构造每个元素，设置正确的 capacity
    for (size_t i = 0; i < config_.min_threads; ++i) {
        local_queues_[i].~WorkerLocalQueue();
        new (&local_queues_[i]) WorkerLocalQueue(config_.queue_capacity);
    }
    
    // 初始化任务分发器
    dispatcher_ = std::make_unique<TaskDispatcher>(
        *load_balancer_, scheduler_, local_queues_
    );
    
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
        resize_monitor_stop_.store(false);
        resize_monitor_thread_ = std::thread(&ThreadPool::resize_monitor_thread, this);
    }
    
    initialized_.store(true);
    return true;
}

void ThreadPool::worker_thread(size_t worker_id) {
    while (true) {
        Task task;
        bool has_task = false;
        
        // 1. 优先从本地队列获取任务
        if (worker_id < local_queues_.size() && local_queues_[worker_id].pop(task)) {
            has_task = true;
        }
        // 2. 如果本地队列为空，尝试工作窃取
        else if (config_.enable_work_stealing) {
            has_task = try_steal_task(worker_id, task);
        }
        
        // 3. 若无任务，加锁后等待；谓词内再次检查本地队列、窃取、全局队列、退出条件
        if (!has_task) {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this, &has_task, &task, worker_id]() {
                if (worker_id < local_queues_.size() && local_queues_[worker_id].pop(task)) {
                    has_task = true;
                    return true;
                }
                if (config_.enable_work_stealing) {
                    has_task = try_steal_task(worker_id, task);
                    if (has_task) return true;
                }
                has_task = scheduler_.dequeue(task);
                if (has_task) return true;
                if (should_exit(worker_id) || stop_.load()) return true;
                return false;
            });
            // 被 notify 唤醒但谓词未取到任务时，再试一次本地队列（可能刚被分发）
            // 即使 stop_ 为 true，也要检查本地队列以排空任务
            if (!has_task && worker_id < local_queues_.size()) {
                has_task = local_queues_[worker_id].pop(task);
            }
        }
        
        // 执行任务（即使 stop_ 为 true，也要执行已获取的任务以排空队列）
        if (has_task) {
            // 检查任务是否已取消
            if (!is_task_cancelled(task)) {
                active_threads_.fetch_add(1, std::memory_order_relaxed);
                execute_task(task);
                active_threads_.fetch_sub(1, std::memory_order_relaxed);
            } else {
                // 任务被取消，也需要更新统计信息
                completed_tasks_.fetch_add(1, std::memory_order_relaxed);
            }
            
            // 更新负载信息
            if (worker_id < local_queues_.size() && load_balancer_) {
                size_t queue_size = local_queues_[worker_id].size();
                load_balancer_->update_load(worker_id, queue_size, 0);
            }
        }
        
        // 检查是否需要退出（缩容时）
        if (should_exit(worker_id) || (stop_.load() && !has_task)) {
            break;
        }
        
        // 触发任务分发（从全局调度器分发到本地队列）
        if (dispatcher_ && !stop_.load()) {
            size_t dispatched = dispatcher_->dispatch_batch(5);  // 批量分发，减少锁竞争
            // 如果成功分发了任务，唤醒等待的线程
            if (dispatched > 0) {
                std::lock_guard<std::mutex> lock(mutex_);
                condition_.notify_all();
            }
        }
    }
}

void ThreadPool::execute_task(const Task& task) {
    if (monitor_ && monitor_->is_enabled()) {
        monitor_->record_task_start(task.task_id, "default");
    }
    auto start_time = std::chrono::steady_clock::now();
    bool success = false;

    try {
        // 检查超时（如果设置了超时时间）
        if (task.timeout_ms > 0) {
            // 注意：这里只做简单的超时检查，实际超时控制需要更复杂的机制
            // 阶段4暂不实现完整的超时机制
        }

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

    auto end_time = std::chrono::steady_clock::now();
    int64_t execution_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_time - start_time
    ).count();

    if (monitor_ && monitor_->is_enabled()) {
        monitor_->record_task_complete(task.task_id, success, execution_time_ns);
    }
    update_statistics(execution_time_ns, success);
}

void ThreadPool::update_statistics(int64_t execution_time_ns, bool success) {
    completed_tasks_.fetch_add(1, std::memory_order_relaxed);
    
    if (!success) {
        failed_tasks_.fetch_add(1, std::memory_order_relaxed);
    }
    
    // 更新总执行时间（使用原子操作累加）
    total_execution_time_ns_.fetch_add(execution_time_ns, std::memory_order_relaxed);
}

ThreadPoolStatus ThreadPool::get_status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    ThreadPoolStatus status;
    status.total_threads = workers_.size();
    status.active_threads = active_threads_.load(std::memory_order_relaxed);
    status.idle_threads = status.total_threads - status.active_threads;
    
    // 队列大小 = 全局调度器 + 所有本地队列
    size_t local_queue_size = 0;
    for (const auto& queue : local_queues_) {
        local_queue_size += queue.size();
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
    {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (stop_.load()) {
            return;  // 已经关闭
        }
        
        stop_.store(true);
    }
    
    // 停止监控线程
    resize_monitor_stop_.store(true);
    if (resize_monitor_thread_.joinable()) {
        resize_monitor_thread_.join();
    }
    
    // 唤醒所有等待的线程
    condition_.notify_all();
    
    // 如果需要等待任务完成
    if (wait_for_tasks) {
        wait_for_completion();
    }
    
    // 等待所有工作线程退出
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    
    workers_.clear();
    worker_ids_.clear();
    local_queues_.clear();
}

void ThreadPool::wait_for_completion() {
    // 等待所有任务完成
    // 需要同时满足：
    // 1. 全局调度器 + 所有本地队列为空（没有待执行的任务）
    // 2. 没有活跃线程（没有正在执行的任务）
    // 3. 所有已提交的任务都已完成（total == completed + failed）
    
    const int max_iterations = 10000;  // 最多等待100秒（10000 * 10ms）
    int iterations = 0;
    
    while (iterations < max_iterations) {
        // 先不加锁读取原子变量（避免长时间持锁）
        size_t total = total_tasks_.load(std::memory_order_relaxed);
        size_t completed = completed_tasks_.load(std::memory_order_relaxed);
        size_t failed = failed_tasks_.load(std::memory_order_relaxed);
        size_t active = active_threads_.load(std::memory_order_relaxed);
        
        // 如果任务计数不匹配或仍有活跃线程，需要检查队列
        if (total != completed + failed || active > 0) {
            std::unique_lock<std::mutex> lock(mutex_);
            bool scheduler_empty = scheduler_.empty();
            
            // 检查所有本地队列是否为空
            size_t local_queue_total = 0;
            for (const auto& queue : local_queues_) {
                local_queue_total += queue.size();
            }
            bool all_queues_empty = scheduler_empty && (local_queue_total == 0);
            
            // 重新读取（因为可能已经变化）
            total = total_tasks_.load(std::memory_order_relaxed);
            completed = completed_tasks_.load(std::memory_order_relaxed);
            failed = failed_tasks_.load(std::memory_order_relaxed);
            active = active_threads_.load(std::memory_order_relaxed);
            
            if (all_queues_empty && active == 0 && total == completed + failed) {
                break;
            }
            lock.unlock();
        } else {
            // 任务计数匹配且没有活跃线程，再检查一次队列（需要加锁）
            std::unique_lock<std::mutex> lock(mutex_);
            bool scheduler_empty = scheduler_.empty();
            
            // 检查所有本地队列是否为空
            size_t local_queue_total = 0;
            for (const auto& queue : local_queues_) {
                local_queue_total += queue.size();
            }
            bool all_queues_empty = scheduler_empty && (local_queue_total == 0);
            
            if (all_queues_empty && active == 0 && total == completed + failed) {
                break;
            }
            lock.unlock();
        }
        
        // 短暂等待后重试
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ++iterations;
    }
}

bool ThreadPool::is_stopped() const {
    return stop_.load(std::memory_order_relaxed);
}

void ThreadPool::set_task_monitor(monitor::TaskMonitor* m) {
    monitor_ = m;
}

bool ThreadPool::try_steal_task(size_t worker_id, Task& task) {
    if (local_queues_.size() <= 1) {
        return false;  // 只有一个线程，无法窃取
    }
    
    // 尝试使用基于负载的智能窃取策略
    if (load_balancer_) {
        // 获取所有线程的负载信息
        std::vector<LoadBalancer::WorkerLoad> loads = load_balancer_->get_all_loads();
        
        if (loads.size() == local_queues_.size()) {
            // 创建线程ID和负载的配对，用于排序
            std::vector<std::pair<size_t, size_t>> worker_loads;
            worker_loads.reserve(local_queues_.size());
            
            for (size_t i = 0; i < local_queues_.size(); ++i) {
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
                    if (local_queues_[target_id].steal(task)) {
                        // 更新目标线程的负载信息
                        size_t queue_size = local_queues_[target_id].size();
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
    std::uniform_int_distribution<size_t> dist(0, local_queues_.size() - 1);
    size_t start_index = dist(rng);
    
    // 尝试从其他线程窃取任务
    for (size_t i = 0; i < local_queues_.size(); ++i) {
        size_t target_id = (start_index + i) % local_queues_.size();
        
        // 跳过自己
        if (target_id == worker_id) {
            continue;
        }
        
        // 尝试窃取
        if (local_queues_[target_id].steal(task)) {
            // 更新目标线程的负载信息
            if (load_balancer_) {
                size_t queue_size = local_queues_[target_id].size();
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
    while (!resize_monitor_stop_.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        if (resize_monitor_stop_.load()) {
            break;
        }
        
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

} // namespace executor
