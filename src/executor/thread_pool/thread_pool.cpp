#include "thread_pool.hpp"
#include "../task/task.hpp"
#include <stdexcept>
#include <algorithm>

namespace executor {

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
    
    // 创建工作线程
    workers_.reserve(config_.min_threads);
    for (size_t i = 0; i < config_.min_threads; ++i) {
        workers_.emplace_back(&ThreadPool::worker_thread, this);
        
        // 设置线程优先级
        if (config_.thread_priority != 0) {
            util::set_thread_priority(workers_.back().native_handle(), 
                                     config_.thread_priority);
        }
        
        // 设置CPU亲和性
        if (!config_.cpu_affinity.empty()) {
            // 如果指定了多个CPU，循环分配
            int cpu_id = config_.cpu_affinity[i % config_.cpu_affinity.size()];
            util::set_cpu_affinity(workers_.back().native_handle(), {cpu_id});
        }
    }
    
    initialized_.store(true);
    return true;
}

void ThreadPool::worker_thread() {
    int task_count = 0;
    while (true) {
        Task task;
        bool has_task = false;
        
        {
            std::unique_lock<std::mutex> lock(mutex_);
            
            // 等待任务或停止信号
            // 注意：即使stop_为true，也要尝试获取队列中的任务
            condition_.wait(lock, [this, &has_task, &task]() {
                // 先尝试获取任务（即使stop_为true，也要处理队列中的剩余任务）
                has_task = scheduler_.dequeue(task);
                if (has_task) {
                    return true;  // 有任务，退出等待
                }
                // 没有任务，如果已停止则退出等待
                return stop_.load();
            });
            
            // 如果没有任务且已停止，退出循环
            if (!has_task && stop_.load()) {
                break;
            }
        }
        
        // 如果有任务，执行任务（即使stop_为true，也要执行队列中的任务）
        if (has_task) {
            ++task_count;
            // 检查任务是否已取消
            if (!is_task_cancelled(task)) {
                active_threads_.fetch_add(1, std::memory_order_relaxed);
                execute_task(task);
                active_threads_.fetch_sub(1, std::memory_order_relaxed);
            } else {
                // 任务被取消，也需要更新统计信息，否则wait_for_completion会卡住
                // 被取消的任务不计入失败，只计入完成（因为没有被执行）
                completed_tasks_.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
}

void ThreadPool::execute_task(const Task& task) {
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
    
    // 计算执行时间
    auto end_time = std::chrono::steady_clock::now();
    auto execution_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_time - start_time
    ).count();
    
    // 更新统计信息
    update_statistics(execution_time, success);
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
    status.queue_size = scheduler_.size();
    status.total_tasks = total_tasks_.load(std::memory_order_relaxed);
    status.completed_tasks = completed_tasks_.load(std::memory_order_relaxed);
    status.failed_tasks = failed_tasks_.load(std::memory_order_relaxed);
    
    // 计算平均任务执行时间
    size_t completed = completed_tasks_.load(std::memory_order_relaxed);
    if (completed > 0) {
        int64_t total_time = total_execution_time_ns_.load(std::memory_order_relaxed);
        status.avg_task_time_ms = (total_time / static_cast<double>(completed)) / 1e6;
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
}

void ThreadPool::wait_for_completion() {
    // 等待所有任务完成
    // 需要同时满足三个条件：
    // 1. 队列为空（没有待执行的任务）
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
        
        // 如果任务计数不匹配，需要检查队列
        if (total != completed + failed || active > 0) {
            // 需要检查队列，必须加锁
            std::unique_lock<std::mutex> lock(mutex_);
            bool queue_empty = scheduler_.empty();
            // 重新读取（因为可能已经变化）
            total = total_tasks_.load(std::memory_order_relaxed);
            completed = completed_tasks_.load(std::memory_order_relaxed);
            failed = failed_tasks_.load(std::memory_order_relaxed);
            active = active_threads_.load(std::memory_order_relaxed);
            
            // 所有条件必须同时满足
            if (queue_empty && active == 0 && total == completed + failed) {
                break;
            }
            lock.unlock();
        } else {
            // 任务计数匹配且没有活跃线程，再检查一次队列（需要加锁）
            std::unique_lock<std::mutex> lock(mutex_);
            bool queue_empty = scheduler_.empty();
            
            if (queue_empty && active == 0 && total == completed + failed) {
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

} // namespace executor
