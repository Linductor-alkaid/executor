#include "executor/executor.hpp"
#include "thread_pool_executor.hpp"
#include "thread_pool/thread_pool.hpp"
#include "task/task.hpp"
#include <stdexcept>
#include <algorithm>
#include <chrono>

namespace executor {

// 单例模式实现
Executor& Executor::instance() {
    static Executor inst(ExecutorManager::instance());
    return inst;
}

// 单例模式构造函数（私有）
Executor::Executor(ExecutorManager& manager)
    : manager_(&manager)
    , owned_manager_(nullptr)
    , timer_running_(false) {
}

// 实例化模式构造函数
Executor::Executor()
    : manager_(nullptr)
    , owned_manager_(std::make_unique<ExecutorManager>())
    , timer_running_(false) {
    manager_ = owned_manager_.get();
}

// 析构函数
Executor::~Executor() {
    stop_timer_thread();
    // owned_manager_ 析构时会自动释放所有执行器（RAII）
}

// 初始化执行器
bool Executor::initialize(const ExecutorConfig& config) {
    return manager_->initialize_async_executor(config);
}

// 关闭执行器
void Executor::shutdown(bool wait_for_tasks) {
    stop_timer_thread();
    manager_->shutdown(wait_for_tasks);
}

// 提交周期性任务
std::string Executor::submit_periodic(int64_t period_ms, std::function<void()> task) {
    if (period_ms <= 0) {
        throw std::invalid_argument("period_ms must be greater than 0");
    }
    if (!task) {
        throw std::invalid_argument("task must not be null");
    }
    
    auto* executor = manager_->get_default_async_executor();
    if (!executor) {
        throw std::runtime_error("Async executor not initialized. Call initialize() first.");
    }
    
    // 生成任务 ID
    std::string task_id = generate_task_id();
    
    // 创建周期性任务
    PeriodicTask periodic_task;
    periodic_task.task_id = task_id;
    periodic_task.period_ms = period_ms;
    periodic_task.next_execute_time = std::chrono::steady_clock::now() + 
                                     std::chrono::milliseconds(period_ms);
    periodic_task.task = [executor, task]() {
        executor->submit(task);
    };
    periodic_task.cancelled = false;
    
    // 添加到周期性任务列表
    {
        std::lock_guard<std::mutex> lock(periodic_tasks_mutex_);
        periodic_tasks_[task_id] = std::move(periodic_task);
    }
    
    // 确保定时器线程运行
    if (!timer_running_.load()) {
        start_timer_thread();
    }
    
    return task_id;
}

// 取消任务
bool Executor::cancel_task(const std::string& task_id) {
    std::lock_guard<std::mutex> lock(periodic_tasks_mutex_);
    
    auto it = periodic_tasks_.find(task_id);
    if (it == periodic_tasks_.end()) {
        return false;  // 任务不存在
    }
    
    // 标记为已取消
    it->second.cancelled = true;
    
    // 从列表中移除
    periodic_tasks_.erase(it);
    
    return true;
}

// 注册实时任务
bool Executor::register_realtime_task(const std::string& name,
                                     const RealtimeThreadConfig& config) {
    // 创建实时执行器
    auto executor = manager_->create_realtime_executor(name, config);
    if (!executor) {
        return false;  // 创建失败
    }
    
    // 注册执行器
    return manager_->register_realtime_executor(name, std::move(executor));
}

// 启动实时任务
bool Executor::start_realtime_task(const std::string& name) {
    auto* executor = manager_->get_realtime_executor(name);
    if (!executor) {
        return false;  // 执行器不存在
    }
    
    return executor->start();
}

// 停止实时任务
void Executor::stop_realtime_task(const std::string& name) {
    auto* executor = manager_->get_realtime_executor(name);
    if (executor) {
        executor->stop();
    }
}

// 获取实时执行器
IRealtimeExecutor* Executor::get_realtime_executor(const std::string& name) {
    return manager_->get_realtime_executor(name);
}

// 获取所有实时任务列表
std::vector<std::string> Executor::get_realtime_task_list() const {
    return manager_->get_realtime_executor_names();
}

// 获取异步执行器状态
AsyncExecutorStatus Executor::get_async_executor_status() const {
    auto* executor = manager_->get_default_async_executor();
    if (!executor) {
        AsyncExecutorStatus status;
        status.name = "default";
        status.is_running = false;
        return status;
    }
    
    return executor->get_status();
}

// 获取实时执行器状态
RealtimeExecutorStatus Executor::get_realtime_executor_status(const std::string& name) const {
    auto* executor = manager_->get_realtime_executor(name);
    if (!executor) {
        RealtimeExecutorStatus status;
        status.name = name;
        status.is_running = false;
        return status;
    }
    
    return executor->get_status();
}

void Executor::enable_monitoring(bool enable) {
    manager_->enable_monitoring(enable);
}

TaskStatistics Executor::get_task_statistics(const std::string& task_type) const {
    return manager_->get_task_statistics(task_type);
}

std::map<std::string, TaskStatistics> Executor::get_all_task_statistics() const {
    return manager_->get_all_task_statistics();
}

void Executor::wait_for_completion() {
    auto* ex = manager_->get_default_async_executor();
    if (ex) ex->wait_for_completion();
}

// 注册 GPU 执行器
bool Executor::register_gpu_executor(const std::string& name,
                                     const gpu::GpuExecutorConfig& config) {
    // 创建 GPU 执行器
    auto executor = manager_->create_gpu_executor(config);
    if (!executor) {
        return false;  // 创建失败
    }
    
    // 启动执行器（必须在注册前启动）
    if (!executor->start()) {
        return false;  // 启动失败
    }
    
    // 注册执行器
    return manager_->register_gpu_executor(name, std::move(executor));
}

// 获取 GPU 执行器
IGpuExecutor* Executor::get_gpu_executor(const std::string& name) {
    return manager_->get_gpu_executor(name);
}

// 获取所有 GPU 执行器名称
std::vector<std::string> Executor::get_gpu_executor_names() const {
    return manager_->get_gpu_executor_names();
}

// 获取 GPU 执行器状态
gpu::GpuExecutorStatus Executor::get_gpu_executor_status(const std::string& name) const {
    auto* executor = manager_->get_gpu_executor(name);
    if (!executor) {
        gpu::GpuExecutorStatus status;
        status.name = name;
        status.is_running = false;
        status.backend = gpu::GpuBackend::CUDA;  // 默认值
        status.device_id = 0;
        return status;
    }
    
    return executor->get_status();
}

// 获取所有 GPU 执行器状态
std::map<std::string, gpu::GpuExecutorStatus> Executor::get_all_gpu_executor_status() const {
    return manager_->get_all_gpu_executor_statuses();
}

// 启动定时器线程
void Executor::start_timer_thread() {
    if (timer_running_.exchange(true)) {
        return;  // 已经在运行
    }
    
    timer_thread_ = std::thread([this]() {
        timer_thread_func();
    });
}

// 停止定时器线程
void Executor::stop_timer_thread() {
    if (!timer_running_.exchange(false)) {
        return;  // 已经停止
    }
    
    if (timer_thread_.joinable()) {
        timer_thread_.join();
    }
}

// 定时器线程函数
void Executor::timer_thread_func() {
    const auto check_interval = std::chrono::milliseconds(10);  // 检查间隔：10ms
    
    while (timer_running_.load(std::memory_order_acquire)) {
        auto now = std::chrono::steady_clock::now();
        
        // 处理延迟任务
        {
            std::lock_guard<std::mutex> lock(delayed_tasks_mutex_);
            
            // 只检查并处理到期的任务（队列顶部是最早执行的任务）
            auto* executor = manager_->get_default_async_executor();
            if (executor) {
                while (!delayed_tasks_.empty()) {
                    const auto& top = delayed_tasks_.top();
                    if (now >= top.execute_time) {
                        // 执行任务（提交到执行器）
                        // 注意：top() 返回 const 引用，需要 const_cast 来移动 task
                        DelayedTask task = std::move(const_cast<DelayedTask&>(top));
                        delayed_tasks_.pop();
                        
                        executor->submit(std::move(task.task));
                        if (task.on_complete) {
                            task.on_complete();
                        }
                    } else {
                        // 后续任务更晚，无需检查
                        break;
                    }
                }
            }
        }
        
        // 处理周期性任务
        {
            std::lock_guard<std::mutex> lock(periodic_tasks_mutex_);
            
            for (auto it = periodic_tasks_.begin(); it != periodic_tasks_.end();) {
                auto& periodic_task = it->second;
                
                // 检查是否已取消
                if (periodic_task.cancelled) {
                    it = periodic_tasks_.erase(it);
                    continue;
                }
                
                // 检查是否到了执行时间
                if (now >= periodic_task.next_execute_time) {
                    // 执行任务
                    periodic_task.task();
                    
                    // 更新下次执行时间
                    periodic_task.next_execute_time = now + 
                        std::chrono::milliseconds(periodic_task.period_ms);
                }
                
                ++it;
            }
        }
        
        // 休眠一段时间
        std::this_thread::sleep_for(check_interval);
    }
}

} // namespace executor
