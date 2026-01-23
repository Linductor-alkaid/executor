#include "thread_pool_executor.hpp"
#include <stdexcept>

namespace executor {

ThreadPoolExecutor::ThreadPoolExecutor(const std::string& name, const ThreadPoolConfig& config)
    : name_(name), config_(config) {
}

ThreadPoolExecutor::~ThreadPoolExecutor() {
    stop();
}

std::string ThreadPoolExecutor::get_name() const {
    return name_;
}

AsyncExecutorStatus ThreadPoolExecutor::get_status() const {
    AsyncExecutorStatus status;
    status.name = name_;
    
    // 获取线程池状态
    ThreadPoolStatus pool_status = thread_pool_.get_status();
    
    // 映射线程池状态到异步执行器状态
    // 如果total_threads == 0,说明ThreadPool还没有初始化,不应该被认为是运行中
    status.is_running = (pool_status.total_threads > 0) && !thread_pool_.is_stopped();
    status.active_tasks = pool_status.active_threads;  // 活跃线程数作为活跃任务数
    status.completed_tasks = pool_status.completed_tasks;
    status.failed_tasks = pool_status.failed_tasks;
    status.queue_size = pool_status.queue_size;
    status.avg_task_time_ms = pool_status.avg_task_time_ms;
    
    return status;
}

bool ThreadPoolExecutor::start() {
    // 初始化线程池
    return thread_pool_.initialize(config_);
}

void ThreadPoolExecutor::stop() {
    // 关闭线程池，等待所有任务完成
    thread_pool_.shutdown(true);
}

void ThreadPoolExecutor::wait_for_completion() {
    // 等待所有任务完成
    thread_pool_.wait_for_completion();
}

void ThreadPoolExecutor::submit_impl(std::function<void()> task) {
    // 将任务提交到线程池（使用NORMAL优先级）
    thread_pool_.submit(std::move(task));
}

void ThreadPoolExecutor::submit_priority_impl(int priority, std::function<void()> task) {
    // 将任务提交到线程池（使用指定优先级）
    thread_pool_.submit_priority(priority, std::move(task));
}

} // namespace executor
