#include "executor/executor.hpp"

#include "executor/monitor/executor_snapshot_formatter.hpp"
#include "executor/monitor/executor_monitor.hpp"
#include "thread_pool_executor.hpp"
#include "thread_pool/thread_pool.hpp"
#include "task/task.hpp"
#include "task/task_dependency_manager.hpp"
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <iterator>
#include <memory>

namespace executor {

namespace {

ExecutorResult make_failure(ExecutorErrorCode code, const std::string& message) {
    return ExecutorResult::failure(code, message);
}

ExecutorResult validate_executor_config(const ExecutorConfig& config) {
    if (config.min_threads != 0 && config.max_threads != 0 &&
        config.min_threads > config.max_threads) {
        return make_failure(
            ExecutorErrorCode::InvalidConfig,
            "ExecutorConfig invalid: min_threads must be <= max_threads");
    }
    return ExecutorResult::success();
}

ExecutorResult validate_realtime_config(const std::string& name,
                                        const RealtimeThreadConfig& config) {
    if (name.empty()) {
        return make_failure(
            ExecutorErrorCode::InvalidConfig,
            "Realtime executor name must not be empty");
    }
    if (config.thread_name.empty()) {
        return make_failure(
            ExecutorErrorCode::InvalidConfig,
            "RealtimeThreadConfig invalid: thread_name must not be empty");
    }
    if (config.cycle_period_ns <= 0) {
        return make_failure(
            ExecutorErrorCode::InvalidConfig,
            "RealtimeThreadConfig invalid: cycle_period_ns must be greater than 0");
    }
    return ExecutorResult::success();
}

ExecutorResult validate_blocking_io_config(const std::string& name,
                                           const BlockingIoConfig& config,
                                           const IBlockingIoWorker* worker) {
    if (name.empty()) {
        return make_failure(ExecutorErrorCode::InvalidConfig,
                            "Blocking I/O executor name must not be empty");
    }
    if (config.thread_name.empty()) {
        return make_failure(ExecutorErrorCode::InvalidConfig,
                            "BlockingIoConfig invalid: thread_name must not be empty");
    }
    if (config.startup_timeout.count() < 0) {
        return make_failure(ExecutorErrorCode::InvalidConfig,
                            "BlockingIoConfig invalid: startup_timeout must not be negative");
    }
    if (!worker) {
        return make_failure(ExecutorErrorCode::InvalidConfig,
                            "Blocking I/O worker must not be null");
    }
    return ExecutorResult::success();
}

ExecutorResult validate_gpu_config_for_facade(
    const std::string& name,
    const gpu::GpuExecutorConfig& config) {
    if (name.empty()) {
        return make_failure(
            ExecutorErrorCode::InvalidConfig,
            "GPU executor name must not be empty");
    }
    if (config.name.empty()) {
        return make_failure(
            ExecutorErrorCode::InvalidConfig,
            "GpuExecutorConfig invalid: config.name must not be empty");
    }
    if (config.max_queue_size == 0) {
        return make_failure(
            ExecutorErrorCode::InvalidConfig,
            "GpuExecutorConfig invalid: max_queue_size must be greater than 0");
    }
    if (config.device_id < 0) {
        return make_failure(
            ExecutorErrorCode::InvalidConfig,
            "GpuExecutorConfig invalid: device_id must be non-negative");
    }
    if (config.default_stream_count < 1) {
        return make_failure(
            ExecutorErrorCode::InvalidConfig,
            "GpuExecutorConfig invalid: default_stream_count must be at least 1");
    }
    return ExecutorResult::success();
}

ExecutorResult check_gpu_backend_available(const gpu::GpuExecutorConfig& config) {
#ifndef EXECUTOR_ENABLE_GPU
    (void)config;
    return make_failure(
        ExecutorErrorCode::BackendUnavailable,
        "GPU support is not enabled in this build");
#else
    switch (config.backend) {
    case gpu::GpuBackend::CUDA:
#ifndef EXECUTOR_ENABLE_CUDA
        return make_failure(
            ExecutorErrorCode::BackendUnavailable,
            "CUDA backend is not enabled in this build");
#else
        return ExecutorResult::success();
#endif
    case gpu::GpuBackend::OPENCL:
#ifndef EXECUTOR_ENABLE_OPENCL
        return make_failure(
            ExecutorErrorCode::BackendUnavailable,
            "OpenCL backend is not enabled in this build");
#else
        return ExecutorResult::success();
#endif
    case gpu::GpuBackend::SYCL:
        return make_failure(
            ExecutorErrorCode::BackendUnavailable,
            "SYCL backend is not implemented in this build");
    case gpu::GpuBackend::HIP:
        return make_failure(
            ExecutorErrorCode::BackendUnavailable,
            "HIP backend is not implemented in this build");
    default:
        return make_failure(
            ExecutorErrorCode::BackendUnavailable,
            "Requested GPU backend is unavailable");
    }
#endif
}

}  // namespace

// 单例模式实现
Executor& Executor::instance() {
    static Executor inst(ExecutorManager::instance());
    return inst;
}

// 单例模式构造函数（私有）
Executor::Executor(ExecutorManager& manager)
    : manager_(&manager)
    , owned_manager_(nullptr)
    , timer_running_(false)
    , task_dependencies_(std::make_unique<TaskDependencyManager>()) {
    monitor_ = std::make_unique<monitor::ExecutorMonitor>(
        *manager_, lifecycle_state_,
        [this]() { return get_completion_status(); },
        [this]() { return get_failure_status(); },
        [this]() { return get_recent_failures(); },
        [this]() { return get_all_task_statistics(); },
        [this]() { return manager_->get_in_flight_task_diagnostics(); });
}

// 实例化模式构造函数
Executor::Executor()
    : manager_(nullptr)
    , owned_manager_(std::make_unique<ExecutorManager>())
    , timer_running_(false)
    , task_dependencies_(std::make_unique<TaskDependencyManager>()) {
    manager_ = owned_manager_.get();
    monitor_ = std::make_unique<monitor::ExecutorMonitor>(
        *manager_, lifecycle_state_,
        [this]() { return get_completion_status(); },
        [this]() { return get_failure_status(); },
        [this]() { return get_recent_failures(); },
        [this]() { return get_all_task_statistics(); },
        [this]() { return manager_->get_in_flight_task_diagnostics(); });
}

// 析构函数
Executor::~Executor() {
    stop_timer_thread();
    // owned_manager_ 析构时会自动释放所有执行器（RAII）
}

// 初始化执行器
bool Executor::initialize(const ExecutorConfig& config) {
    return initialize_ex(config).ok;
}

ExecutorResult Executor::initialize_ex(const ExecutorConfig& config) {
    if (auto validation = validate_executor_config(config); !validation.ok) {
        lifecycle_state_.store(ExecutorLifecycleState::Failed, std::memory_order_release);
        record_result_failure(
            validation, FailureKind::SubmitRejected, "default", "facade_initialize");
        return validation;
    }

    if (manager_->is_default_async_shutdown()) {
        auto result = make_failure(
            ExecutorErrorCode::AlreadyShutdown,
            "Async executor has already been shutdown");
        record_result_failure(
            result, FailureKind::SubmitRejected, "default", "facade_initialize");
        return result;
    }

    if (manager_->has_default_async_executor()) {
        auto result = make_failure(
            ExecutorErrorCode::AlreadyInitialized,
            "Async executor is already initialized");
        record_result_failure(
            result, FailureKind::SubmitRejected, "default", "facade_initialize");
        return result;
    }

    lifecycle_state_.store(ExecutorLifecycleState::Initializing, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(task_graph_mutex_);
        task_graph_retention_capacity_ = config.task_graph_retention_capacity;
        trim_task_graph_retention_locked();
    }
    if (!manager_->initialize_async_executor(config)) {
        auto code = manager_->is_default_async_shutdown()
                        ? ExecutorErrorCode::AlreadyShutdown
                        : manager_->has_default_async_executor()
                              ? ExecutorErrorCode::AlreadyInitialized
                              : ExecutorErrorCode::StartFailed;
        auto result = make_failure(
            code,
            code == ExecutorErrorCode::StartFailed
                ? "Async executor initialization failed"
                : "Async executor initialization was rejected");
        record_result_failure(
            result, FailureKind::SubmitRejected, "default", "facade_initialize");
        if (code == ExecutorErrorCode::StartFailed) {
            lifecycle_state_.store(ExecutorLifecycleState::Failed, std::memory_order_release);
        }
        return result;
    }

    lifecycle_state_.store(ExecutorLifecycleState::Running, std::memory_order_release);
    return ExecutorResult::success("Async executor initialized");
}

// 关闭执行器
ShutdownResult Executor::shutdown(bool wait_for_tasks) {
    stop_timer_thread();
    lifecycle_state_.store(ExecutorLifecycleState::Draining, std::memory_order_release);
    const auto async_executor = manager_->get_default_async_executor_snapshot();
    if (async_executor && async_executor->is_current_worker_thread()) {
        const auto result = manager_->shutdown(wait_for_tasks);
        if (result == ShutdownResult::Completed) {
            lifecycle_state_.store(ExecutorLifecycleState::Stopped, std::memory_order_release);
        }
        return result;
    }
    if (wait_for_tasks && manager_->has_default_async_executor()) {
        const auto wait_result = wait_for_completion_ex(kDefaultWaitForCompletionTimeout);
        const auto result = manager_->shutdown(wait_result.completed);
        if (result == ShutdownResult::Completed) {
            lifecycle_state_.store(ExecutorLifecycleState::Stopped, std::memory_order_release);
        }
        return result;
    }

    const auto result = manager_->shutdown(wait_for_tasks);
    if (result == ShutdownResult::Completed) {
        lifecycle_state_.store(ExecutorLifecycleState::Stopped, std::memory_order_release);
    }
    return result;
}

void Executor::set_timer_thread_factory_for_test(
    std::function<std::thread(std::function<void()>)> factory) {
    timer_thread_factory_for_test_ = std::move(factory);
}

TaskHandle Executor::allocate_task_handle() {
    TaskHandle handle(generate_task_id());
    {
        std::lock_guard<std::mutex> lock(task_graph_mutex_);
        task_graph_nodes_.emplace(handle.id(), TaskGraphNode{});
    }
    manager_->record_in_flight_task_pending(handle.id(), "task_graph", "default");
    return handle;
}

void Executor::set_task_graph_retention_capacity(size_t capacity) {
    std::lock_guard<std::mutex> lock(task_graph_mutex_);
    task_graph_retention_capacity_ = capacity;
    trim_task_graph_retention_locked();
}

size_t Executor::task_graph_retention_capacity() const {
    std::lock_guard<std::mutex> lock(task_graph_mutex_);
    return task_graph_retention_capacity_;
}

bool Executor::task_handle_known_locked(const TaskHandle& handle) const {
    return handle.valid() && task_graph_nodes_.find(handle.id()) != task_graph_nodes_.end();
}

bool Executor::register_task_graph_dependencies(
    const TaskHandle& handle,
    const std::vector<TaskHandle>& dependencies,
    std::string& error_message) {
    std::lock_guard<std::mutex> lock(task_graph_mutex_);
    for (const auto& dependency : dependencies) {
        if (!task_handle_known_locked(dependency)) {
            error_message = "submit_after dependency handle is invalid";
            return false;
        }
        if (!task_dependencies_->add_dependency(handle.id(), dependency.id())) {
            error_message = "submit_after dependency graph contains a cycle or invalid edge";
            return false;
        }
        task_graph_dependents_[dependency.id()].push_back(handle.id());
        task_graph_nodes_[handle.id()].dependencies.push_back(dependency.id());
    }
    return true;
}

std::exception_ptr Executor::dependency_failure_locked(
    const std::vector<TaskHandle>& dependencies) const {
    for (const auto& dependency : dependencies) {
        auto it = task_graph_nodes_.find(dependency.id());
        if (it == task_graph_nodes_.end()) {
            return make_dependency_exception("dependency handle is invalid");
        }
        if (it->second.state == TaskGraphState::Failed) {
            if (it->second.exception) {
                return it->second.exception;
            }
            return make_dependency_exception(
                it->second.error_message.empty()
                    ? "dependency failed"
                    : it->second.error_message);
        }
    }
    return nullptr;
}

bool Executor::dependencies_succeeded_locked(
    const std::vector<TaskHandle>& dependencies) const {
    for (const auto& dependency : dependencies) {
        auto it = task_graph_nodes_.find(dependency.id());
        if (it == task_graph_nodes_.end() ||
            it->second.state != TaskGraphState::Succeeded) {
            return false;
        }
    }
    return true;
}

void Executor::mark_task_graph_running(const TaskHandle& handle) {
    std::lock_guard<std::mutex> lock(task_graph_mutex_);
    auto it = task_graph_nodes_.find(handle.id());
    if (it != task_graph_nodes_.end() && it->second.state == TaskGraphState::Pending) {
        it->second.state = TaskGraphState::Running;
    }
    manager_->record_in_flight_task_state(handle.id(), TaskLifecycleState::Running);
}

void Executor::mark_task_graph_succeeded(const TaskHandle& handle) {
    {
        std::lock_guard<std::mutex> lock(task_graph_mutex_);
        auto it = task_graph_nodes_.find(handle.id());
        if (it != task_graph_nodes_.end()) {
            it->second.state = TaskGraphState::Succeeded;
            it->second.exception = nullptr;
            it->second.error_message.clear();
            task_dependencies_->mark_completed(handle.id());
            resolve_task_graph_dependents_locked(handle.id());
            finalize_task_graph_node_locked(handle.id());
        }
    }
    manager_->record_in_flight_task_terminal(handle.id());
    task_graph_cv_.notify_all();
}

void Executor::mark_task_graph_failed(const TaskHandle& handle,
                                      std::exception_ptr exception,
                                      std::string message) {
    {
        std::lock_guard<std::mutex> lock(task_graph_mutex_);
        auto it = task_graph_nodes_.find(handle.id());
        if (it != task_graph_nodes_.end()) {
            it->second.state = TaskGraphState::Failed;
            it->second.exception = exception;
            it->second.error_message = std::move(message);
            resolve_task_graph_dependents_locked(handle.id());
            finalize_task_graph_node_locked(handle.id());
        }
    }
    manager_->record_in_flight_task_terminal(handle.id());
    task_graph_cv_.notify_all();
}

void Executor::resolve_task_graph_dependents_locked(const std::string& task_id) {
    std::vector<std::string> ready_ids{task_id};
    std::vector<std::string> terminal_ids;

    while (!ready_ids.empty()) {
        const std::string current_id = std::move(ready_ids.back());
        ready_ids.pop_back();

        auto dependents_it = task_graph_dependents_.find(current_id);
        if (dependents_it == task_graph_dependents_.end()) {
            continue;
        }

        const auto dependent_ids = dependents_it->second;
        for (const auto& dependent_id : dependent_ids) {
            auto node_it = task_graph_nodes_.find(dependent_id);
            if (node_it == task_graph_nodes_.end() ||
                node_it->second.state != TaskGraphState::WhenAll) {
                continue;
            }

            std::vector<TaskHandle> dependencies;
            const auto& dependent_node = node_it->second;
            for (const auto& dependency_id : dependent_node.dependencies) {
                dependencies.emplace_back(dependency_id);
            }

            if (auto dependency_exception = dependency_failure_locked(dependencies)) {
                node_it->second.state = TaskGraphState::Failed;
                node_it->second.exception = dependency_exception;
                node_it->second.error_message = "when_all dependency failed";
                ready_ids.push_back(dependent_id);
                terminal_ids.push_back(dependent_id);
                manager_->record_in_flight_task_terminal(dependent_id);
            } else if (dependencies_succeeded_locked(dependencies)) {
                node_it->second.state = TaskGraphState::Succeeded;
                node_it->second.exception = nullptr;
                node_it->second.error_message.clear();
                task_dependencies_->mark_completed(dependent_id);
                ready_ids.push_back(dependent_id);
                terminal_ids.push_back(dependent_id);
                manager_->record_in_flight_task_terminal(dependent_id);
            }
        }
    }

    for (const auto& terminal_id : terminal_ids) {
        finalize_task_graph_node_locked(terminal_id);
    }
}

void Executor::finalize_task_graph_node_locked(const std::string& task_id) {
    auto node_it = task_graph_nodes_.find(task_id);
    if (node_it == task_graph_nodes_.end()) {
        return;
    }

    auto& node = node_it->second;
    if (node.state != TaskGraphState::Succeeded &&
        node.state != TaskGraphState::Failed) {
        return;
    }

    // Remove this node from every dependency's reverse edge.  A dependency
    // becomes evictable only after all active dependents have finished.
    for (const auto& dependency_id : node.dependencies) {
        auto dependents_it = task_graph_dependents_.find(dependency_id);
        if (dependents_it != task_graph_dependents_.end()) {
            auto& dependents = dependents_it->second;
            dependents.erase(
                std::remove(dependents.begin(), dependents.end(), task_id),
                dependents.end());
            if (dependents.empty()) {
                task_graph_dependents_.erase(dependents_it);
            }
        }
        task_dependencies_->remove_dependency(task_id, dependency_id);
    }
    node.dependencies.clear();
    task_graph_terminal_order_.push_back(task_id);
    trim_task_graph_retention_locked();
}

void Executor::trim_task_graph_retention_locked() {
    while (task_graph_terminal_order_.size() > task_graph_retention_capacity_) {
        auto candidate = std::find_if(
            task_graph_terminal_order_.begin(), task_graph_terminal_order_.end(),
            [this](const std::string& task_id) {
                const auto it = task_graph_dependents_.find(task_id);
                return it == task_graph_dependents_.end() || it->second.empty();
            });
        if (candidate == task_graph_terminal_order_.end()) {
            // Every old terminal node is still needed by an active dependent.
            // The active graph is allowed to exceed the terminal cache bound.
            break;
        }

        const std::string task_id = *candidate;
        task_graph_terminal_order_.erase(candidate);
        task_graph_nodes_.erase(task_id);
        task_graph_dependents_.erase(task_id);
        task_dependencies_->prune(task_id);
    }
}

std::exception_ptr Executor::make_dependency_exception(const std::string& message) const {
    return std::make_exception_ptr(std::runtime_error(message));
}

TaskHandle Executor::when_all(std::vector<TaskHandle> dependencies) {
    TaskHandle handle = allocate_task_handle();

    bool dependencies_valid = true;
    bool terminal = false;
    std::string validation_error;
    {
        std::lock_guard<std::mutex> lock(task_graph_mutex_);
        for (const auto& dependency : dependencies) {
            if (!task_handle_known_locked(dependency)) {
                dependencies_valid = false;
                validation_error = "when_all dependency handle is invalid";
                break;
            }
            if (!task_dependencies_->add_dependency(handle.id(), dependency.id())) {
                dependencies_valid = false;
                validation_error = "when_all dependency graph contains a cycle or invalid edge";
                break;
            }
            task_graph_dependents_[dependency.id()].push_back(handle.id());
            task_graph_nodes_[handle.id()].dependencies.push_back(dependency.id());
        }
        if (dependencies_valid) {
            auto& node = task_graph_nodes_[handle.id()];
            if (auto dependency_exception = dependency_failure_locked(dependencies)) {
                node.state = TaskGraphState::Failed;
                node.exception = dependency_exception;
                node.error_message = "when_all dependency failed";
                terminal = true;
            } else if (dependencies_succeeded_locked(dependencies)) {
                node.state = TaskGraphState::Succeeded;
                task_dependencies_->mark_completed(handle.id());
                terminal = true;
            } else {
                node.state = TaskGraphState::WhenAll;
            }
            if (terminal) {
                finalize_task_graph_node_locked(handle.id());
            }
        }
    }

    if (!dependencies_valid) {
        auto exception = make_dependency_exception(validation_error);
        mark_task_graph_failed(handle, exception, validation_error);
        record_submit_rejected("default", handle.id(), validation_error, exception);
        return handle;
    }

    if (terminal) {
        manager_->record_in_flight_task_terminal(handle.id());
    } else {
        manager_->record_in_flight_task_state(
            handle.id(), TaskLifecycleState::DependencyBlocked);
    }

    task_graph_cv_.notify_all();

    return handle;
}

// 提交周期性任务
std::string Executor::submit_periodic(int64_t period_ms, std::function<void()> task) {
    if (period_ms <= 0) {
        throw std::invalid_argument("period_ms must be greater than 0");
    }
    if (!task) {
        throw std::invalid_argument("task must not be null");
    }

    auto executor = manager_->get_default_async_executor_snapshot();
    const std::string executor_name = executor ? executor->get_name() : "default";
    if (!executor) {
        record_submit_rejected(
            executor_name,
            "facade_submit_periodic",
            "Async executor not initialized. Call initialize() first.");
        throw std::runtime_error("Async executor not initialized. Call initialize() first.");
    }

    std::string task_id = generate_task_id();

    PeriodicTask periodic_task;
    periodic_task.status.task_id = task_id;
    periodic_task.status.period_ms = period_ms;
    periodic_task.status.is_running = true;
    periodic_task.status.next_execute_time = std::chrono::steady_clock::now() +
                                            std::chrono::milliseconds(period_ms);
    periodic_task.task = std::move(task);
    periodic_task.cancelled = false;

    {
        std::lock_guard<std::mutex> lock(periodic_tasks_mutex_);
        periodic_tasks_[task_id] = std::move(periodic_task);
    }

    try {
        if (!timer_running_.load()) {
            start_timer_thread();
        }
    } catch (...) {
        auto exception = std::current_exception();
        {
            std::lock_guard<std::mutex> lock(periodic_tasks_mutex_);
            periodic_tasks_.erase(task_id);
        }
        record_submit_rejected(
            executor_name,
            task_id,
            "Timer thread creation failed for periodic task",
            exception);
        throw;
    }

    return task_id;
}

// 取消任务
bool Executor::cancel_task(const std::string& task_id) {
    {
        std::lock_guard<std::mutex> lock(periodic_tasks_mutex_);

        auto it = periodic_tasks_.find(task_id);
        if (it != periodic_tasks_.end()) {
            it->second.cancelled = true;
            periodic_tasks_.erase(it);
            return true;
        }
    }

    record_submit_rejected(
        "default",
        task_id,
        "Periodic task cancellation failed: task not found");
    return false;
}

std::optional<PeriodicTaskStatus> Executor::get_periodic_task_status(
    const std::string& task_id) const {
    std::lock_guard<std::mutex> lock(periodic_tasks_mutex_);
    auto it = periodic_tasks_.find(task_id);
    if (it == periodic_tasks_.end()) {
        return std::nullopt;
    }
    return it->second.status;
}

std::vector<PeriodicTaskStatus> Executor::get_all_periodic_task_status() const {
    std::lock_guard<std::mutex> lock(periodic_tasks_mutex_);
    std::vector<PeriodicTaskStatus> statuses;
    statuses.reserve(periodic_tasks_.size());
    for (const auto& entry : periodic_tasks_) {
        statuses.push_back(entry.second.status);
    }
    return statuses;
}

// 注册实时任务
bool Executor::register_realtime_task(const std::string& name,
                                     const RealtimeThreadConfig& config) {
    return register_realtime_task_ex(name, config).ok;
}

ExecutorResult Executor::register_realtime_task_ex(
    const std::string& name,
    const RealtimeThreadConfig& config) {
    if (auto validation = validate_realtime_config(name, config); !validation.ok) {
        record_result_failure(
            validation, FailureKind::SubmitRejected, name, "facade_register_realtime_task");
        return validation;
    }

    auto executor = manager_->create_realtime_executor(name, config);
    if (!executor) {
        auto result = make_failure(
            ExecutorErrorCode::InvalidConfig,
            "Realtime executor creation failed");
        record_result_failure(
            result, FailureKind::SubmitRejected, name, "facade_register_realtime_task");
        return result;
    }

    if (!manager_->register_realtime_executor(name, std::move(executor))) {
        auto result = make_failure(
            ExecutorErrorCode::DuplicateName,
            "Realtime executor registration failed or duplicate name");
        record_result_failure(
            result, FailureKind::SubmitRejected, name, "facade_register_realtime_task");
        return result;
    }

    return ExecutorResult::success("Realtime executor registered");
}

// 启动实时任务
bool Executor::start_realtime_task(const std::string& name) {
    return start_realtime_task_ex(name).ok;
}

ExecutorResult Executor::start_realtime_task_ex(const std::string& name) {
    if (name.empty()) {
        auto result = make_failure(
            ExecutorErrorCode::InvalidConfig,
            "Realtime executor name must not be empty");
        record_result_failure(
            result, FailureKind::SubmitRejected, name, "facade_start_realtime_task");
        return result;
    }

    auto executor = manager_->get_realtime_executor_snapshot(name);
    if (!executor) {
        auto result = make_failure(
            ExecutorErrorCode::NotFound,
            "Realtime executor '" + name + "' not found");
        record_result_failure(
            result, FailureKind::SubmitRejected, name, "facade_start_realtime_task");
        return result;
    }

    if (!executor->start()) {
        const auto status = executor->get_status();
        auto code = status.is_running
                        ? ExecutorErrorCode::AlreadyInitialized
                        : ExecutorErrorCode::StartFailed;
        auto result = make_failure(
            code,
            status.is_running
                ? "Realtime executor '" + name + "' is already running"
                : "Realtime executor '" + name + "' start failed");
        record_result_failure(
            result, FailureKind::SubmitRejected, name, "facade_start_realtime_task");
        return result;
    }

    return ExecutorResult::success("Realtime executor started");
}

// 停止实时任务
void Executor::stop_realtime_task(const std::string& name) {
    auto executor = manager_->get_realtime_executor_snapshot(name);
    if (executor) {
        executor->stop();
    }
}

bool Executor::register_blocking_io_worker(
    const std::string& name,
    const BlockingIoConfig& config,
    std::unique_ptr<IBlockingIoWorker> worker) {
    return register_blocking_io_worker_ex(name, config, std::move(worker)).ok;
}

ExecutorResult Executor::register_blocking_io_worker_ex(
    const std::string& name,
    const BlockingIoConfig& config,
    std::unique_ptr<IBlockingIoWorker> worker) {
    if (auto validation = validate_blocking_io_config(name, config, worker.get()); !validation.ok) {
        record_result_failure(
            validation, FailureKind::SubmitRejected, name, "facade_register_blocking_io_worker");
        return validation;
    }
    auto executor = manager_->create_blocking_io_executor(name, config, std::move(worker));
    if (!executor) {
        auto result = make_failure(ExecutorErrorCode::StartFailed,
                                   "Blocking I/O executor creation failed");
        record_result_failure(
            result, FailureKind::SubmitRejected, name, "facade_register_blocking_io_worker");
        return result;
    }
    if (!manager_->register_blocking_io_executor(name, std::move(executor))) {
        auto result = make_failure(ExecutorErrorCode::DuplicateName,
                                   "Blocking I/O executor registration failed or duplicate name");
        record_result_failure(
            result, FailureKind::SubmitRejected, name, "facade_register_blocking_io_worker");
        return result;
    }
    return ExecutorResult::success("Blocking I/O executor registered");
}

bool Executor::start_blocking_io_worker(const std::string& name) {
    return start_blocking_io_worker_ex(name).ok;
}

ExecutorResult Executor::start_blocking_io_worker_ex(const std::string& name) {
    if (name.empty()) {
        auto result = make_failure(ExecutorErrorCode::InvalidConfig,
                                   "Blocking I/O executor name must not be empty");
        record_result_failure(
            result, FailureKind::SubmitRejected, name, "facade_start_blocking_io_worker");
        return result;
    }
    auto executor = manager_->get_blocking_io_executor_snapshot(name);
    if (!executor) {
        auto result = make_failure(ExecutorErrorCode::NotFound,
                                   "Blocking I/O executor '" + name + "' not found");
        record_result_failure(
            result, FailureKind::SubmitRejected, name, "facade_start_blocking_io_worker");
        return result;
    }
    if (!executor->start()) {
        const auto status = executor->get_status();
        const auto code = status.is_running ? ExecutorErrorCode::AlreadyInitialized
                                            : ExecutorErrorCode::StartFailed;
        auto result = make_failure(
            code,
            status.is_running ? "Blocking I/O executor '" + name + "' is already running"
                              : "Blocking I/O executor '" + name + "' start failed");
        record_result_failure(
            result, FailureKind::SubmitRejected, name, "facade_start_blocking_io_worker");
        return result;
    }
    return ExecutorResult::success("Blocking I/O executor started");
}

void Executor::stop_blocking_io_worker(const std::string& name) {
    manager_->stop_blocking_io_executor(name);
}

BlockingIoExecutorStatus Executor::get_blocking_io_worker_status(const std::string& name) const {
    return manager_->get_blocking_io_executor_status(name);
}

std::vector<std::string> Executor::get_blocking_io_worker_list() const {
    return manager_->get_blocking_io_executor_names();
}

WorkerHandle Executor::start_worker(BlockingWorkerSpec spec) {
    const std::string name = spec.name;
    auto result = register_blocking_io_worker_ex(
        spec.name, spec.config, std::move(spec.worker));
    if (result.ok) {
        result = start_blocking_io_worker_ex(name);
    }
    return WorkerHandle(manager_, name, std::move(result));
}

void WorkerHandle::request_stop() noexcept {
    if (manager_) {
        manager_->request_stop_blocking_io_executor(name_);
    }
}

void WorkerHandle::stop() {
    if (manager_) {
        manager_->stop_blocking_io_executor(name_);
    }
}

BlockingIoExecutorStatus WorkerHandle::status() const {
    if (manager_) {
        return manager_->get_blocking_io_executor_status(name_);
    }
    BlockingIoExecutorStatus status;
    status.name = name_;
    return status;
}

bool Executor::push_realtime_task(const std::string& name, std::function<void()> task) {
    auto executor = manager_->get_realtime_executor_snapshot(name);
    if (!executor) {
        record_submit_rejected(
            name,
            "facade_push_realtime_task",
            "Realtime executor not found");
        return false;
    }
    const auto before = executor->get_status();
    const bool accepted = executor->push_task_ex(std::move(task));
    if (accepted) {
        return true;
    }

    const auto after = executor->get_status();
    std::string message = "Realtime task push rejected";
    if (after.rejected_not_running_count > before.rejected_not_running_count) {
        message = "Realtime task push rejected: executor is not running";
    } else if (after.rejected_empty_task_count > before.rejected_empty_task_count) {
        message = "Realtime task push rejected: task is empty";
    } else if (after.pool_exhausted_count > before.pool_exhausted_count) {
        message = "Realtime task push rejected: task object pool exhausted";
    } else if (after.queue_full_count > before.queue_full_count ||
               after.failed_pushes > before.failed_pushes) {
        message = "Realtime task push rejected: queue is full";
    }

    record_realtime_drop(
        executor->get_name(),
        "facade_push_realtime_task",
        message);
    return false;
}

bool Executor::try_push_realtime_task(const std::string& name, std::function<void()> task) {
    return push_realtime_task(name, std::move(task));
}

// 获取实时执行器
IRealtimeExecutor* Executor::get_realtime_executor(const std::string& name) {
    return manager_->get_realtime_executor(name);
}

// 获取所有实时任务列表
std::vector<std::string> Executor::get_realtime_task_list() const {
    return manager_->get_realtime_executor_names();
}

bool Executor::register_lockfree_executor(
    const std::string& name,
    std::unique_ptr<LockFreeTaskExecutor> executor) {
    if (name.empty() || !executor || !manager_->register_lockfree_executor(name, std::move(executor))) {
        record_submit_rejected(name, "facade_register_lockfree_executor",
                               "Lock-free executor registration failed or duplicate name");
        return false;
    }
    return true;
}

bool Executor::start_lockfree_executor(const std::string& name) {
    if (!manager_->start_lockfree_executor(name)) {
        record_submit_rejected(name, "facade_start_lockfree_executor",
                               "Lock-free executor not found, already running, or stopped");
        return false;
    }
    return true;
}

void Executor::stop_lockfree_executor(const std::string& name) {
    manager_->stop_lockfree_executor(name);
}

std::vector<std::string> Executor::get_lockfree_executor_names() const {
    return manager_->get_lockfree_executor_names();
}

RoutingDecision Executor::route_dispatch(const TaskOptions& options) const {
    return task_router_.route_dispatch(options, manager_->get_executor_capabilities());
}

DispatchResult Executor::dispatch_auto(TaskOptions options, std::function<void()> task) {
    DispatchResult result;
    result.decision = route_dispatch(options);
    result.backend = result.decision.selected_backend;
    result.executor_name = result.decision.selected_executor_name;

    if (!task) {
        result.decision.reason = RoutingReason::Rejected;
        result.decision.detail = "dispatch task is empty";
        result.message = result.decision.detail;
        record_routing_decision(result.decision);
        record_submit_rejected(result.executor_name, result.decision.task_name, result.message);
        return result;
    }

    if (result.decision.reason == RoutingReason::Rejected ||
        result.decision.reason == RoutingReason::BackendUnavailable ||
        result.decision.reason == RoutingReason::BackendNotRunning ||
        result.decision.reason == RoutingReason::CapacityPressure) {
        result.message = result.decision.detail;
        record_routing_decision(result.decision);
        record_submit_rejected(result.executor_name, result.decision.task_name, result.message);
        return result;
    }

    result.accepted = result.backend == ExecutionBackend::LockFree
                          ? manager_->try_push_lockfree_task(result.executor_name, std::move(task))
                          : manager_->try_push_realtime_task(result.executor_name, std::move(task));
    if (!result.accepted) {
        result.decision.reason = RoutingReason::Rejected;
        result.decision.detail = "bounded executor rejected dispatch (stopped, full, or object pool exhausted)";
        result.message = result.decision.detail;
        record_submit_rejected(result.executor_name, result.decision.task_name, result.message);
    }
    record_routing_decision(result.decision);
    return result;
}

std::vector<ExecutorCapability> Executor::get_executor_capabilities() const {
    return manager_->get_executor_capabilities();
}

// 获取异步执行器状态
AsyncExecutorStatus Executor::get_async_executor_status() const {
    auto executor = manager_->get_default_async_executor_snapshot();
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
    auto executor = manager_->get_realtime_executor_snapshot(name);
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

void Executor::set_snapshot_diagnostic_callback(ExecutorSnapshotCallback callback) {
    std::lock_guard<std::mutex> lock(snapshot_diagnostic_mutex_);
    snapshot_diagnostic_callback_ = std::move(callback);
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

std::optional<RoutingDecision> Executor::get_last_routing_decision() const {
    std::lock_guard<std::mutex> lock(routing_mutex_);
    if (recent_routing_decisions_.empty()) {
        return std::nullopt;
    }
    return recent_routing_decisions_.back();
}

std::vector<RoutingDecision> Executor::get_recent_routing_decisions(size_t max_count) const {
    std::lock_guard<std::mutex> lock(routing_mutex_);
    const size_t count = max_count == 0
                             ? recent_routing_decisions_.size()
                             : std::min(max_count, recent_routing_decisions_.size());
    return {recent_routing_decisions_.end() - static_cast<std::ptrdiff_t>(count),
            recent_routing_decisions_.end()};
}

void Executor::clear_recent_routing_decisions() {
    std::lock_guard<std::mutex> lock(routing_mutex_);
    recent_routing_decisions_.clear();
}

void Executor::set_recent_routing_capacity(size_t capacity) {
    std::lock_guard<std::mutex> lock(routing_mutex_);
    recent_routing_capacity_ = capacity;
    while (recent_routing_decisions_.size() > recent_routing_capacity_) {
        recent_routing_decisions_.pop_front();
    }
}

void Executor::set_routing_callback(std::function<void(const RoutingDecision&)> callback) {
    std::lock_guard<std::mutex> lock(routing_mutex_);
    routing_callback_ = std::move(callback);
}

RoutingDecision Executor::route_task(const TaskOptions& options,
                                     bool cpu_gpu_task,
                                     std::optional<bool> gpu_selected) const {
    return task_router_.route(
        TaskRouter::Request{options, cpu_gpu_task, gpu_selected},
        manager_->get_executor_capabilities());
}

void Executor::record_routing_decision(RoutingDecision decision) {
    std::function<void(const RoutingDecision&)> callback;
    {
        std::lock_guard<std::mutex> lock(routing_mutex_);
        if (recent_routing_capacity_ > 0) {
            while (recent_routing_decisions_.size() >= recent_routing_capacity_) {
                recent_routing_decisions_.pop_front();
            }
            recent_routing_decisions_.push_back(decision);
        }
        callback = routing_callback_;
    }
    if (callback) {
        try {
            callback(decision);
        } catch (...) {
            // Routing observation must not affect submission or worker threads.
        }
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

void Executor::record_result_failure(const ExecutorResult& result,
                                     FailureKind kind,
                                     const std::string& executor_name,
                                     const std::string& task_id) {
    if (result.ok) {
        return;
    }

    ExecutorFailureEvent event;
    event.kind = kind;
    event.executor_name = executor_name;
    event.task_id = task_id;
    event.message = std::string(executor_error_code_to_string(result.error_code)) +
                    ": " + result.message;
    record_failure(std::move(event));
    emit_snapshot_diagnostic();
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

void Executor::record_task_timeout(const std::string& executor_name,
                                   const std::string& task_id,
                                   const std::string& message,
                                   std::exception_ptr exception) {
    ExecutorFailureEvent event;
    event.kind = FailureKind::TaskTimeout;
    event.executor_name = executor_name;
    event.task_id = task_id;
    event.message = message;
    event.exception = exception;
    record_failure(std::move(event));
}

void Executor::record_realtime_drop(const std::string& executor_name,
                                    const std::string& task_id,
                                    const std::string& message,
                                    std::exception_ptr exception) {
    ExecutorFailureEvent event;
    event.kind = FailureKind::RealtimeDrop;
    event.executor_name = executor_name;
    event.task_id = task_id;
    event.message = message;
    event.exception = exception;
    record_failure(std::move(event));
}

void Executor::record_periodic_task_success(const std::string& task_id) {
    std::lock_guard<std::mutex> lock(periodic_tasks_mutex_);
    auto it = periodic_tasks_.find(task_id);
    if (it == periodic_tasks_.end()) {
        return;
    }

    auto& status = it->second.status;
    ++status.execution_count;
    status.consecutive_failure_count = 0;
    status.last_error_message.clear();
}

void Executor::record_periodic_task_exception(const std::string& executor_name,
                                              const std::string& task_id,
                                              const std::string& message,
                                              std::exception_ptr exception) {
    {
        std::lock_guard<std::mutex> lock(periodic_tasks_mutex_);
        auto it = periodic_tasks_.find(task_id);
        if (it != periodic_tasks_.end()) {
            auto& status = it->second.status;
            ++status.execution_count;
            ++status.failed_count;
            ++status.consecutive_failure_count;
            status.last_error_message = message;
            status.last_failure_time = std::chrono::steady_clock::now();
        }
    }

    record_task_exception(executor_name, task_id, message, exception);
}

void Executor::record_periodic_submit_rejected(const std::string& executor_name,
                                               const std::string& task_id,
                                               const std::string& message,
                                               std::exception_ptr exception) {
    {
        std::lock_guard<std::mutex> lock(periodic_tasks_mutex_);
        auto it = periodic_tasks_.find(task_id);
        if (it != periodic_tasks_.end()) {
            auto& status = it->second.status;
            ++status.failed_count;
            ++status.consecutive_failure_count;
            status.last_error_message = message;
            status.last_failure_time = std::chrono::steady_clock::now();
        }
    }

    record_submit_rejected(executor_name, task_id, message, exception);
}

void Executor::enable_monitoring(bool enable) {
    manager_->enable_monitoring(enable);
}

void Executor::set_monitoring_sampling_rate(double rate) {
    manager_->set_monitoring_sampling_rate(rate);
}

void Executor::set_in_flight_task_capacity(size_t capacity) {
    manager_->set_in_flight_task_capacity(capacity);
}

void Executor::set_in_flight_task_sampling_rate(double rate) {
    manager_->set_in_flight_task_sampling_rate(rate);
}

TaskStatistics Executor::get_task_statistics(const std::string& task_type) const {
    return manager_->get_task_statistics(task_type);
}

std::map<std::string, TaskStatistics> Executor::get_all_task_statistics() const {
    return manager_->get_all_task_statistics();
}

void Executor::wait_for_completion() {
    (void)wait_for_completion_ex(kDefaultWaitForCompletionTimeout);
}

bool Executor::try_wait_for_completion(std::chrono::milliseconds timeout) {
    return wait_for_completion_ex(timeout).completed;
}

WaitResult Executor::wait_for_completion_ex(std::chrono::milliseconds timeout) {
    WaitResult result;
    result.timeout = timeout;

    // Waiting for an absent backend is complete; it must not lazily create one.
    if (!manager_->has_default_async_executor()) {
        result.completed = true;
        result.timed_out = false;
        result.status = get_completion_status();
        result.message = "Async executor is not initialized";
        return result;
    }

    auto ex = manager_->get_default_async_executor_snapshot();
    if (!ex) {
        result.completed = true;
        result.timed_out = false;
        result.status = get_completion_status();
        result.message = "Async executor is not initialized";
        return result;
    }

    result.completed = ex->try_wait_for_completion(timeout);
    result.timed_out = !result.completed;
    result.status = get_completion_status();

    if (result.completed) {
        result.message = "All async tasks completed";
        return result;
    }

    result.message = "wait_for_completion timed out before all tasks completed";

    ExecutorFailureEvent event;
    event.kind = FailureKind::WaitTimeout;
    event.executor_name = result.status.executor_name;
    event.task_id = "facade_wait_for_completion";
    event.message = result.message + ": active=" +
                    std::to_string(result.status.active_tasks) +
                    ", queued=" + std::to_string(result.status.queued_tasks) +
                    ", pending=" + std::to_string(result.status.pending_tasks);
    record_failure(std::move(event));
    result.diagnostic_snapshot = get_snapshot();
    emit_snapshot_diagnostic(*result.diagnostic_snapshot);
    return result;
}

bool Executor::is_idle() const {
    return get_completion_status().is_idle;
}

CompletionStatus Executor::get_completion_status() const {
    CompletionStatus completion;
    if (!manager_->has_default_async_executor()) {
        return completion;
    }

    auto ex = manager_->get_default_async_executor_snapshot();
    if (!ex) {
        return completion;
    }

    const auto status = ex->get_status();
    completion.executor_name = status.name;
    completion.is_initialized = true;
    completion.is_running = status.is_running;
    completion.active_tasks = status.active_tasks;
    completion.queued_tasks = status.queue_size;
    completion.pending_tasks = status.active_tasks + status.queue_size;
    completion.completed_tasks = status.completed_tasks;
    completion.failed_tasks = status.failed_tasks;
    completion.is_idle = completion.pending_tasks == 0;
    return completion;
}

ExecutorSnapshot Executor::get_snapshot() const {
    return monitor_->collect();
}

std::string Executor::get_snapshot_text() const {
    return monitor::format_executor_snapshot(get_snapshot());
}

void Executor::emit_snapshot_diagnostic() const {
    emit_snapshot_diagnostic(get_snapshot());
}

void Executor::emit_snapshot_diagnostic(const ExecutorSnapshot& snapshot) const {
    ExecutorSnapshotCallback callback;
    {
        std::lock_guard<std::mutex> lock(snapshot_diagnostic_mutex_);
        callback = snapshot_diagnostic_callback_;
    }
    if (!callback) {
        return;
    }
    try {
        callback(snapshot);
    } catch (...) {
        // Diagnostics must not change facade results or lifecycle behavior.
    }
}

// 注册 GPU 执行器
bool Executor::register_gpu_executor(const std::string& name,
                                     const gpu::GpuExecutorConfig& config) {
    return register_gpu_executor_ex(name, config).ok;
}

ExecutorResult Executor::register_gpu_executor_ex(
    const std::string& name,
    const gpu::GpuExecutorConfig& config) {
    if (auto validation = validate_gpu_config_for_facade(name, config); !validation.ok) {
        record_result_failure(
            validation, FailureKind::GpuFailure, name, "facade_register_gpu_executor");
        return validation;
    }

    if (auto backend = check_gpu_backend_available(config); !backend.ok) {
        record_result_failure(
            backend, FailureKind::GpuFailure, name, "facade_register_gpu_executor");
        return backend;
    }

    auto executor = manager_->create_gpu_executor(config);
    if (!executor) {
        auto result = make_failure(
            ExecutorErrorCode::BackendUnavailable,
            "GPU executor creation failed");
        record_result_failure(
            result, FailureKind::GpuFailure, name, "facade_register_gpu_executor");
        return result;
    }

    if (!executor->start()) {
        auto status = executor->get_status();
        auto result = make_failure(
            ExecutorErrorCode::StartFailed,
            status.last_error_message.empty()
                ? "GPU executor start failed"
                : "GPU executor start failed: " + status.last_error_message);
        record_result_failure(
            result, FailureKind::GpuFailure, name, "facade_register_gpu_executor");
        return result;
    }

    if (!manager_->register_gpu_executor(name, std::move(executor))) {
        auto result = make_failure(
            ExecutorErrorCode::DuplicateName,
            "GPU executor registration failed or duplicate name");
        record_result_failure(
            result, FailureKind::GpuFailure, name, "facade_register_gpu_executor");
        return result;
    }

    return ExecutorResult::success("GPU executor registered");
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
    auto executor = manager_->get_gpu_executor_snapshot(name);
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

    std::vector<DelayedTask> pending_delayed_tasks;
    {
        std::lock_guard<std::mutex> lock(delayed_tasks_mutex_);
        while (!delayed_tasks_.empty()) {
            DelayedTask task = std::move(const_cast<DelayedTask&>(delayed_tasks_.top()));
            delayed_tasks_.pop();
            pending_delayed_tasks.push_back(std::move(task));
        }
    }

    for (auto& task : pending_delayed_tasks) {
        auto exception = std::make_exception_ptr(std::runtime_error(
            "Timer stopped before delayed task execution"));
        if (task.on_rejected) {
            task.on_rejected(exception);
        }
    }

    std::vector<std::string> pending_periodic_task_ids;
    {
        std::lock_guard<std::mutex> lock(periodic_tasks_mutex_);
        pending_periodic_task_ids.reserve(periodic_tasks_.size());
        for (auto& entry : periodic_tasks_) {
            entry.second.status.is_running = false;
            pending_periodic_task_ids.push_back(entry.first);
        }
    }

    for (const auto& task_id : pending_periodic_task_ids) {
        record_periodic_submit_rejected(
            "default",
            task_id,
            "Timer stopped before periodic task execution");
    }
}

namespace {

constexpr int64_t kTimerMaxSleepMs = 10;  // 无待处理任务时的最大休眠间隔（ms）

struct DuePeriodicTask {
    std::string task_id;
    int64_t period_ms = 0;
    std::function<void()> task;
};

}  // namespace

// 定时器线程函数
void Executor::timer_thread_func() {
    using clock = std::chrono::steady_clock;
    const auto max_interval = std::chrono::milliseconds(kTimerMaxSleepMs);

    while (timer_running_.load(std::memory_order_acquire)) {
        auto now = clock::now();
        auto next_wake = now + max_interval;
        auto executor = manager_->get_default_async_executor_snapshot();
        const std::string executor_name = executor ? executor->get_name() : "default";

        {
            std::lock_guard<std::mutex> lock(delayed_tasks_mutex_);

            while (!delayed_tasks_.empty()) {
                const auto& top = delayed_tasks_.top();
                if (now < top.execute_time) {
                    break;
                }

                DelayedTask task = std::move(const_cast<DelayedTask&>(top));
                delayed_tasks_.pop();

                if (!executor) {
                    auto exception = std::make_exception_ptr(std::runtime_error(
                        "Async executor unavailable for delayed task"));
                    if (task.on_rejected) {
                        task.on_rejected(exception);
                    }
                    continue;
                }

                if (!executor->try_submit_task(
                        std::move(task.task), std::move(task.on_timeout))) {
                    auto exception = std::make_exception_ptr(std::runtime_error(
                        "Async executor rejected delayed task submission"));
                    if (task.on_rejected) {
                        task.on_rejected(exception);
                    }
                }
            }

            if (!delayed_tasks_.empty()) {
                auto ed = delayed_tasks_.top().execute_time;
                if (ed < next_wake) next_wake = ed;
            }
        }

        std::vector<DuePeriodicTask> due_periodic_tasks;
        {
            std::lock_guard<std::mutex> lock(periodic_tasks_mutex_);

            for (auto it = periodic_tasks_.begin(); it != periodic_tasks_.end();) {
                auto& periodic_task = it->second;

                if (periodic_task.cancelled) {
                    it = periodic_tasks_.erase(it);
                    continue;
                }

                auto& status = periodic_task.status;
                if (now >= status.next_execute_time) {
                    due_periodic_tasks.push_back(
                        DuePeriodicTask{status.task_id, status.period_ms, periodic_task.task});
                    status.next_execute_time =
                        now + std::chrono::milliseconds(status.period_ms);
                }

                if (status.next_execute_time < next_wake)
                    next_wake = status.next_execute_time;
                ++it;
            }
        }

        for (auto& due : due_periodic_tasks) {
            if (!executor) {
                record_periodic_submit_rejected(
                    executor_name,
                    due.task_id,
                    "Async executor unavailable for periodic task");
                continue;
            }

            auto wrapped_task = [this,
                                 executor_name,
                                 task_id = due.task_id,
                                 task = std::move(due.task)]() mutable {
                try {
                    task();
                    record_periodic_task_success(task_id);
                } catch (...) {
                    auto exception = std::current_exception();
                    record_periodic_task_exception(
                        executor_name,
                        task_id,
                        "Periodic task threw an exception",
                        exception);
                    throw;
                }
            };

            if (!executor->try_submit_task(std::move(wrapped_task))) {
                auto exception = std::make_exception_ptr(std::runtime_error(
                    "Async executor rejected periodic task submission"));
                record_periodic_submit_rejected(
                    executor_name,
                    due.task_id,
                    "Async executor rejected periodic task submission",
                    exception);
            }
        }

        std::this_thread::sleep_until(next_wake);
    }
}

} // namespace executor
