#pragma once

#include "interfaces.hpp"
#include "config.hpp"
#include "types.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <map>
#include <shared_mutex>
#include <mutex>

namespace executor {
namespace monitor { class StatisticsCollector; }

/**
 * @brief 执行器管理器
 * 
 * 统一管理线程池执行器和专用实时线程执行器
 * 支持单例模式和实例化模式
 * 
 * 生命周期管理：
 * - 所有执行器的生命周期由 ExecutorManager 统一管理
 * - 使用 RAII 模式，ExecutorManager 析构时自动释放所有执行器
 * - 单例模式：ExecutorManager 单例的生命周期与程序相同
 * - 实例化模式：每个 ExecutorManager 实例拥有独立的执行器，析构时自动释放
 */
class ExecutorManager {
public:
    /**
     * @brief 获取单例实例
     * @return ExecutorManager 单例引用
     */
    static ExecutorManager& instance();

    /**
     * @brief 构造函数（实例化模式）
     * 
     * 创建独立的 ExecutorManager 实例，用于资源隔离
     */
    ExecutorManager();

    /**
     * @brief 析构函数（RAII）
     * 
     * 自动关闭并释放所有执行器
     */
    ~ExecutorManager();

    // 禁止拷贝和赋值
    ExecutorManager(const ExecutorManager&) = delete;
    ExecutorManager& operator=(const ExecutorManager&) = delete;

    /**
     * @brief 初始化默认异步执行器（线程池）
     * 
     * @param config 执行器配置
     * @return 是否初始化成功
     */
    bool initialize_async_executor(const ExecutorConfig& config);

    /**
     * @brief 获取默认异步执行器（线程池）
     * 
     * @return 异步执行器指针，如果未初始化则返回 nullptr
     */
    IAsyncExecutor* get_default_async_executor();

    /**
     * @brief 注册实时执行器
     * 
     * @param name 执行器名称
     * @param executor 执行器指针（所有权转移）
     * @return 是否注册成功（如果名称已存在则返回 false）
     */
    bool register_realtime_executor(const std::string& name,
                                   std::unique_ptr<IRealtimeExecutor> executor);

    /**
     * @brief 获取已注册的实时执行器
     * 
     * @param name 执行器名称
     * @return 实时执行器指针，如果不存在则返回 nullptr
     */
    IRealtimeExecutor* get_realtime_executor(const std::string& name);

    /**
     * @brief 创建实时执行器（便捷方法）
     * 
     * 注意：此方法仅创建执行器对象，不会自动注册
     * 需要调用 register_realtime_executor() 进行注册
     * 
     * @param name 执行器名称
     * @param config 实时线程配置
     * @return 执行器指针，如果创建失败则返回 nullptr
     */
    std::unique_ptr<IRealtimeExecutor> create_realtime_executor(
        const std::string& name,
        const RealtimeThreadConfig& config);

    /**
     * @brief 获取所有实时执行器名称
     * 
     * @return 实时执行器名称列表
     */
    std::vector<std::string> get_realtime_executor_names() const;

    /**
     * @brief 注册 GPU 执行器
     * 
     * @param name 执行器名称
     * @param executor 执行器指针（所有权转移）
     * @return 是否注册成功（如果名称已存在则返回 false）
     */
    bool register_gpu_executor(const std::string& name,
                               std::unique_ptr<IGpuExecutor> executor);

    /**
     * @brief 获取已注册的 GPU 执行器
     * 
     * @param name 执行器名称
     * @return GPU 执行器指针，如果不存在则返回 nullptr
     */
    IGpuExecutor* get_gpu_executor(const std::string& name);

    /**
     * @brief 创建 GPU 执行器（便捷方法）
     * 
     * 注意：此方法仅创建执行器对象，不会自动注册
     * 需要调用 register_gpu_executor() 进行注册
     * 
     * @param config GPU 执行器配置
     * @return 执行器指针，如果创建失败则返回 nullptr
     */
    std::unique_ptr<IGpuExecutor> create_gpu_executor(
        const gpu::GpuExecutorConfig& config);

    /**
     * @brief 获取所有 GPU 执行器名称
     * 
     * @return GPU 执行器名称列表
     */
    std::vector<std::string> get_gpu_executor_names() const;

    /**
     * @brief 获取所有 GPU 执行器状态（用于监控查询）
     * 
     * @return 执行器名称到状态的映射
     */
    std::map<std::string, gpu::GpuExecutorStatus> get_all_gpu_executor_statuses() const;

    /**
     * @brief 关闭所有执行器
     * 
     * @param wait_for_tasks 是否等待任务完成（默认：true）
     */
    void shutdown(bool wait_for_tasks = true);

    /**
     * @brief 启用或禁用任务监控
     */
    void enable_monitoring(bool enable);

    /**
     * @brief 按 task_type 获取任务统计
     */
    TaskStatistics get_task_statistics(const std::string& task_type) const;

    /**
     * @brief 获取全部 task_type 的任务统计
     */
    std::map<std::string, TaskStatistics> get_all_task_statistics() const;

private:
    // 默认异步执行器（线程池）
    std::unique_ptr<IAsyncExecutor> default_async_executor_;

    // 已关闭标记：shutdown 后不再懒初始化，get_default_async_executor() 直接返回 nullptr
    bool default_async_shutdown_ = false;

    // 懒初始化用：保证多线程首次调用 get_default_async_executor() 时只初始化一次
    std::once_flag default_init_once_;

    // 实时执行器注册表
    std::unordered_map<std::string, std::unique_ptr<IRealtimeExecutor>> realtime_executors_;

    // 读写锁（保护实时执行器注册表）
    mutable std::shared_mutex mutex_;

    // GPU 执行器注册表
    std::unordered_map<std::string, std::unique_ptr<IGpuExecutor>> gpu_executors_;

    // 读写锁（保护 GPU 执行器注册表）
    mutable std::shared_mutex gpu_mutex_;

    // 统计收集器（任务监控）
    std::unique_ptr<monitor::StatisticsCollector> statistics_collector_;

    // 退出时自动关闭：atexit 回调（仅单例创建时注册，无参无返回、不抛异常）
    static void atexit_shutdown();

    // 单例实例（线程安全初始化）
    static ExecutorManager* instance_;
    static std::once_flag once_flag_;
};

} // namespace executor
