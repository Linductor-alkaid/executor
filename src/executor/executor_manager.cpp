#include "executor/executor_manager.hpp"
#include "thread_pool_executor.hpp"
#include "realtime_thread_executor.hpp"
#include "executor/monitor/statistics_collector.hpp"
#include "executor/interfaces.hpp"
#include "executor/config.hpp"
#include <cstdlib>
#include <mutex>
#include <algorithm>

#ifdef EXECUTOR_ENABLE_GPU
#include "executor/gpu/cuda_executor.hpp"
#endif

namespace executor {

// 静态成员变量定义
ExecutorManager* ExecutorManager::instance_ = nullptr;
std::once_flag ExecutorManager::once_flag_;

// 单例模式：获取单例实例
ExecutorManager& ExecutorManager::instance() {
    std::call_once(once_flag_, []() {
        instance_ = new ExecutorManager();
        std::atexit(&ExecutorManager::atexit_shutdown);
    });
    return *instance_;
}

// 退出时自动关闭：atexit 回调，仅单例创建时注册；不等待未完成任务，不抛异常
void ExecutorManager::atexit_shutdown() {
    try {
        ExecutorManager::instance().shutdown(false);
    } catch (...) {
        // 吞掉异常，避免 atexit 回调抛异常导致 std::terminate
    }
}

// 构造函数（实例化模式）
ExecutorManager::ExecutorManager()
    : default_async_executor_(nullptr)
    , statistics_collector_(std::make_unique<monitor::StatisticsCollector>()) {
    statistics_collector_->set_gpu_status_provider(
        [this]() { return get_all_gpu_executor_statuses(); });
}

// 析构函数（RAII）
ExecutorManager::~ExecutorManager() {
    shutdown(true);  // 等待所有任务完成
}

// 初始化默认异步执行器（线程池）
bool ExecutorManager::initialize_async_executor(const ExecutorConfig& config) {
    // 检查是否已经初始化
    if (default_async_executor_ != nullptr) {
        return false;  // 已经初始化过
    }

    // 将 ExecutorConfig 转换为 ThreadPoolConfig
    ThreadPoolConfig pool_config;
    pool_config.min_threads = config.min_threads;
    pool_config.max_threads = config.max_threads;
    pool_config.queue_capacity = config.queue_capacity;
    pool_config.thread_priority = config.thread_priority;
    pool_config.cpu_affinity = config.cpu_affinity;
    pool_config.task_timeout_ms = config.task_timeout_ms;
    pool_config.enable_work_stealing = config.enable_work_stealing;

    // 创建 ThreadPoolExecutor
    auto executor = std::make_unique<ThreadPoolExecutor>("default", pool_config);
    executor->set_task_monitor(&statistics_collector_->get_task_monitor());
    statistics_collector_->get_task_monitor().set_enabled(config.enable_monitoring);

    // 启动执行器
    if (!executor->start()) {
        return false;  // 启动失败
    }

    // 保存执行器
    default_async_executor_ = std::move(executor);
    return true;
}

// 获取默认异步执行器（线程池）
// 若尚未初始化，则使用默认配置懒初始化一次（线程安全由 std::call_once 保证）
// shutdown 后不再懒初始化，直接返回 nullptr
IAsyncExecutor* ExecutorManager::get_default_async_executor() {
    if (default_async_shutdown_) {
        return nullptr;
    }
    if (default_async_executor_ == nullptr) {
        std::call_once(default_init_once_, [this] {
            ExecutorConfig default_config{};
            initialize_async_executor(default_config);
        });
    }
    return default_async_executor_.get();
}

// 注册实时执行器
bool ExecutorManager::register_realtime_executor(const std::string& name,
                                                 std::unique_ptr<IRealtimeExecutor> executor) {
    if (name.empty() || executor == nullptr) {
        return false;
    }

    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    // 检查名称是否已存在
    if (realtime_executors_.find(name) != realtime_executors_.end()) {
        return false;  // 名称已存在
    }

    // 注册执行器
    realtime_executors_[name] = std::move(executor);
    return true;
}

// 获取已注册的实时执行器
IRealtimeExecutor* ExecutorManager::get_realtime_executor(const std::string& name) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    auto it = realtime_executors_.find(name);
    if (it == realtime_executors_.end()) {
        return nullptr;  // 不存在
    }
    
    return it->second.get();
}

// 创建实时执行器（便捷方法）
std::unique_ptr<IRealtimeExecutor> ExecutorManager::create_realtime_executor(
    const std::string& name,
    const RealtimeThreadConfig& config) {
    if (name.empty()) {
        return nullptr;
    }

    try {
        // 创建 RealtimeThreadExecutor 实例
        auto executor = std::make_unique<RealtimeThreadExecutor>(name, config);
        return executor;
    } catch (const std::exception&) {
        // 创建失败（如配置无效），返回 nullptr
        return nullptr;
    }
}

// 获取所有实时执行器名称
std::vector<std::string> ExecutorManager::get_realtime_executor_names() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    std::vector<std::string> names;
    names.reserve(realtime_executors_.size());
    
    for (const auto& pair : realtime_executors_) {
        names.push_back(pair.first);
    }
    
    return names;
}

// 注册 GPU 执行器
bool ExecutorManager::register_gpu_executor(const std::string& name,
                                             std::unique_ptr<IGpuExecutor> executor) {
    if (name.empty() || executor == nullptr) {
        return false;
    }

    std::unique_lock<std::shared_mutex> lock(gpu_mutex_);
    
    // 检查名称是否已存在
    if (gpu_executors_.find(name) != gpu_executors_.end()) {
        return false;  // 名称已存在
    }

    // 注册执行器
    gpu_executors_[name] = std::move(executor);
    return true;
}

// 获取已注册的 GPU 执行器
IGpuExecutor* ExecutorManager::get_gpu_executor(const std::string& name) {
    std::shared_lock<std::shared_mutex> lock(gpu_mutex_);
    
    auto it = gpu_executors_.find(name);
    if (it == gpu_executors_.end()) {
        return nullptr;  // 不存在
    }
    
    return it->second.get();
}

// 创建 GPU 执行器（便捷方法）
std::unique_ptr<IGpuExecutor> ExecutorManager::create_gpu_executor(
    const gpu::GpuExecutorConfig& config) {
#ifdef EXECUTOR_ENABLE_GPU
    // 验证配置
    if (!gpu::validate_gpu_config(config)) {
        return nullptr;
    }

    try {
        // 根据后端类型创建对应的执行器
        if (config.backend == gpu::GpuBackend::CUDA) {
#ifdef EXECUTOR_ENABLE_CUDA
            return std::make_unique<gpu::CudaExecutor>(config.name, config);
#else
            return nullptr;  // CUDA 支持未启用
#endif
        }
        // 其他后端（OpenCL、SYCL）待后续实现
        return nullptr;
    } catch (const std::exception&) {
        // 创建失败（如配置无效），返回 nullptr
        return nullptr;
    }
#else
    // GPU 支持未启用
    return nullptr;
#endif
}

// 获取所有 GPU 执行器名称
std::vector<std::string> ExecutorManager::get_gpu_executor_names() const {
    std::shared_lock<std::shared_mutex> lock(gpu_mutex_);
    
    std::vector<std::string> names;
    names.reserve(gpu_executors_.size());
    
    for (const auto& pair : gpu_executors_) {
        names.push_back(pair.first);
    }
    
    return names;
}

// 获取所有 GPU 执行器状态
std::map<std::string, gpu::GpuExecutorStatus>
ExecutorManager::get_all_gpu_executor_statuses() const {
    std::shared_lock<std::shared_mutex> lock(gpu_mutex_);
    std::map<std::string, gpu::GpuExecutorStatus> result;
    for (const auto& pair : gpu_executors_) {
        if (pair.second) {
            result[pair.first] = pair.second->get_status();
        }
    }
    return result;
}

void ExecutorManager::enable_monitoring(bool enable) {
    if (statistics_collector_) {
        statistics_collector_->get_task_monitor().set_enabled(enable);
    }
}

TaskStatistics ExecutorManager::get_task_statistics(
    const std::string& task_type) const {
    return statistics_collector_
           ? statistics_collector_->get_task_statistics(task_type)
           : TaskStatistics{};
}

std::map<std::string, TaskStatistics>
ExecutorManager::get_all_task_statistics() const {
    return statistics_collector_
           ? statistics_collector_->get_all_task_statistics()
           : std::map<std::string, TaskStatistics>{};
}

// 关闭所有执行器
void ExecutorManager::shutdown(bool wait_for_tasks) {
    // 停止所有实时执行器
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        for (auto& pair : realtime_executors_) {
            if (pair.second) {
                pair.second->stop();
                if (wait_for_tasks) {
                    // 注意：IRealtimeExecutor 没有 wait_for_completion 方法
                    // 实时执行器的任务在周期回调中执行，stop() 已经会等待线程结束
                }
            }
        }
        realtime_executors_.clear();
    }
    
    // 停止所有 GPU 执行器
    {
        std::unique_lock<std::shared_mutex> lock(gpu_mutex_);
        for (auto& pair : gpu_executors_) {
            if (pair.second) {
                pair.second->stop();
                if (wait_for_tasks) {
                    pair.second->wait_for_completion();
                }
            }
        }
        gpu_executors_.clear();
    }
    
    // 停止异步执行器
    if (default_async_executor_) {
        default_async_executor_->stop();
        if (wait_for_tasks) {
            default_async_executor_->wait_for_completion();
        }
        default_async_executor_.reset();
        default_async_shutdown_ = true;
    }
}

} // namespace executor
