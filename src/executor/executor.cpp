#include "executor/executor.hpp"
#include "thread_pool_executor.hpp"
#include "thread_pool/thread_pool.hpp"
#include "task/task.hpp"
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <iterator>

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
    bool initialized = manager_->initialize_async_executor(config);
    if (!initialized) {
        ExecutorFailureEvent event;
        event.kind = FailureKind::SubmitRejected;
        event.executor_name = "default";
        event.message = "Async executor initialization failed or was rejected";
        record_failure(std::move(event));
    }
    return initialized;
}

// 关闭执行器
void Executor::shutdown(bool wait_for_tasks) {
    stop_timer_thread();
    manager_->shutdown(wait_for_tasks);
}

void Executor::set_timer_thread_factory_for_test(
    std::function<std::thread(std::function<void()>)> factory) {
    timer_thread_factory_for_test_ = std::move(factory);
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
        ExecutorFailureEvent event;
        event.kind = FailureKind::SubmitRejected;
        event.executor_name = name;
        event.message = "Realtime executor creation failed";
        record_failure(std::move(event));
        return false;  // 创建失败
    }
    
    // 注册执行器
    bool registered = manager_->register_realtime_executor(name, std::move(executor));
    if (!registered) {
        ExecutorFailureEvent event;
        event.kind = FailureKind::SubmitRejected;
        event.executor_name = name;
        event.message = "Realtime executor registration failed or duplicate name";
        record_failure(std::move(event));
    }
    return registered;
}

// 启动实时任务
bool Executor::start_realtime_task(const std::string& name) {
    auto* executor = manager_->get_realtime_executor(name);
    if (!executor) {
        ExecutorFailureEvent event;
        event.kind = FailureKind::SubmitRejected;
        event.executor_name = name;
        event.message = "Realtime executor not found";
        record_failure(std::move(event));
        return false;  // 执行器不存在
    }
    
    bool started = executor->start();
    if (!started) {
        ExecutorFailureEvent event;
        event.kind = FailureKind::SubmitRejected;
        event.executor_name = name;
        event.message = "Realtime executor start failed";
        record_failure(std::move(event));
    }
    return started;
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

void Executor::set_failure_callback(ExecutorFailureCallback callback) {
    std::lock_guard<std::mutex> lock(failure_mutex_);
    failure_callback_ = std::move(callback);
}

ExecutorFailureStatus Executor::get_failure_status() const {
    std::lock_guard<std::mutex> lock(failure_mutex_);
    return failure_status_;
}

std::vector<ExecutorFailureEvent> Executor::get_recent_failures(size_t max_count) const {
    std::lock_guard<std::mutex> lock(failure_mutex_);

    const size_t available = recent_failures_.size();
    const size_t count = (max_count == 0 || max_count > available)
                             ? available
                             : max_count;

    std::vector<ExecutorFailureEvent> result;
    result.reserve(count);

    const size_t start = available - count;
    auto it = recent_failures_.begin();
    std::advance(it, static_cast<std::ptrdiff_t>(start));
    for (; it != recent_failures_.end(); ++it) {
        result.push_back(*it);
    }

    return result;
}

void Executor::clear_recent_failures() {
    std::lock_guard<std::mutex> lock(failure_mutex_);
    recent_failures_.clear();
}

void Executor::set_recent_failure_capacity(size_t capacity) {
    std::lock_guard<std::mutex> lock(failure_mutex_);
    recent_failure_capacity_ = capacity;
    while (recent_failures_.size() > recent_failure_capacity_) {
        recent_failures_.pop_front();
    }
}

size_t Executor::recent_failure_capacity() const {
    std::lock_guard<std::mutex> lock(failure_mutex_);
    return recent_failure_capacity_;
}

void Executor::record_failure(ExecutorFailureEvent event) {
    ExecutorFailureCallback callback;

    {
        std::lock_guard<std::mutex> lock(failure_mutex_);

        ++failure_status_.total_count;
        switch (event.kind) {
        case FailureKind::TaskException:
            ++failure_status_.task_exception_count;
            break;
        case FailureKind::SubmitRejected:
            ++failure_status_.submit_rejected_count;
            break;
        case FailureKind::TaskTimeout:
            ++failure_status_.timeout_count;
            break;
        case FailureKind::RealtimeDrop:
            ++failure_status_.realtime_drop_count;
            break;
        case FailureKind::GpuFailure:
            ++failure_status_.gpu_failure_count;
            break;
        case FailureKind::WaitTimeout:
            ++failure_status_.wait_timeout_count;
            break;
        case FailureKind::TuningFallback:
            ++failure_status_.tuning_fallback_count;
            break;
        default:
            break;
        }

        if (recent_failure_capacity_ > 0) {
            while (recent_failures_.size() >= recent_failure_capacity_) {
                recent_failures_.pop_front();
            }
            recent_failures_.push_back(event);
        }

        callback = failure_callback_;
    }

    if (callback) {
        try {
            callback(event);
        } catch (...) {
            // Failure observation must never become a new worker/background failure.
        }
    }
}

void Executor::record_submit_rejected(const std::string& executor_name,
                                      const std::string& task_id,
                                      const std::string& message,
                                      std::exception_ptr exception) {
    ExecutorFailureEvent event;
    event.kind = FailureKind::SubmitRejected;
    event.executor_name = executor_name;
    event.task_id = task_id;
    event.message = message;
    event.exception = exception;
    record_failure(std::move(event));
}

void Executor::record_task_exception(const std::string& executor_name,
                                     const std::string& task_id,
                                     const std::string& message,
                                     std::exception_ptr exception) {
    ExecutorFailureEvent event;
    event.kind = FailureKind::TaskException;
    event.executor_name = executor_name;
    event.task_id = task_id;
    event.message = message;
    event.exception = exception;
    record_failure(std::move(event));
}

void Executor::enable_monitoring(bool enable) {
    manager_->enable_monitoring(enable);
}

void Executor::set_monitoring_sampling_rate(double rate) {
    manager_->set_monitoring_sampling_rate(rate);
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
        ExecutorFailureEvent event;
        event.kind = FailureKind::GpuFailure;
        event.executor_name = name;
        event.message = "GPU executor creation failed";
        record_failure(std::move(event));
        return false;  // 创建失败
    }
    
    // 启动执行器（必须在注册前启动）
    if (!executor->start()) {
        ExecutorFailureEvent event;
        event.kind = FailureKind::GpuFailure;
        event.executor_name = name;
        event.message = "GPU executor start failed";
        record_failure(std::move(event));
        return false;  // 启动失败
    }
    
    // 注册执行器
    bool registered = manager_->register_gpu_executor(name, std::move(executor));
    if (!registered) {
        ExecutorFailureEvent event;
        event.kind = FailureKind::GpuFailure;
        event.executor_name = name;
        event.message = "GPU executor registration failed or duplicate name";
        record_failure(std::move(event));
    }
    return registered;
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

// 更新调度器配置
void Executor::update_scheduler_config(const gpu::GpuScheduler::Config& config) {
    scheduler_.update_config(config);
}

// 获取调度器配置
gpu::GpuScheduler::Config Executor::get_scheduler_config() const {
    return scheduler_.get_config();
}

// 启动定时器线程
void Executor::start_timer_thread() {
    if (timer_running_.exchange(true)) {
        return;  // 已经在运行
    }
    
    try {
        auto timer_entry = [this]() {
            timer_thread_func();
        };
        if (timer_thread_factory_for_test_) {
            timer_thread_ = timer_thread_factory_for_test_(std::move(timer_entry));
        } else {
            timer_thread_ = std::thread(std::move(timer_entry));
        }
    } catch (...) {
        timer_running_.store(false);
        throw;
    }
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

namespace {

constexpr int64_t kTimerMaxSleepMs = 10;  // 无待处理任务时的最大休眠间隔（ms）

}  // namespace

// 定时器线程函数
void Executor::timer_thread_func() {
    using clock = std::chrono::steady_clock;
    const auto max_interval = std::chrono::milliseconds(kTimerMaxSleepMs);

    while (timer_running_.load(std::memory_order_acquire)) {
        auto now = clock::now();
        auto next_wake = now + max_interval;

        // 处理延迟任务
        {
            std::lock_guard<std::mutex> lock(delayed_tasks_mutex_);

            auto* executor = manager_->get_default_async_executor();
            if (executor) {
                while (!delayed_tasks_.empty()) {
                    const auto& top = delayed_tasks_.top();
                    if (now >= top.execute_time) {
                        DelayedTask task = std::move(const_cast<DelayedTask&>(top));
                        delayed_tasks_.pop();

                        executor->submit(std::move(task.task));
                        if (task.on_complete) {
                            task.on_complete();
                        }
                    } else {
                        break;
                    }
                }
            }

            if (!delayed_tasks_.empty()) {
                auto ed = delayed_tasks_.top().execute_time;
                if (ed < next_wake) next_wake = ed;
            }
        }

        // 处理周期性任务
        {
            std::lock_guard<std::mutex> lock(periodic_tasks_mutex_);

            for (auto it = periodic_tasks_.begin(); it != periodic_tasks_.end();) {
                auto& periodic_task = it->second;

                if (periodic_task.cancelled) {
                    it = periodic_tasks_.erase(it);
                    continue;
                }

                if (now >= periodic_task.next_execute_time) {
                    if (!periodic_task.cancelled) {
                        periodic_task.task();
                        periodic_task.next_execute_time =
                            now + std::chrono::milliseconds(periodic_task.period_ms);
                    }
                }

                if (periodic_task.next_execute_time < next_wake)
                    next_wake = periodic_task.next_execute_time;
                ++it;
            }
        }

        std::this_thread::sleep_until(next_wake);
    }
}

} // namespace executor
