#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <functional>

namespace executor {

// 前向声明
class ICycleManager;

/**
 * @brief 线程池配置结构
 */
struct ThreadPoolConfig {
    size_t min_threads = 4;              // 最小线程数（默认：4）
    size_t max_threads = 16;             // 最大线程数（默认：16）
    size_t queue_capacity = 1000;        // 任务队列容量
    int thread_priority = 0;             // 线程优先级（-20到19，Linux；Windows使用SetThreadPriority）
    std::vector<int> cpu_affinity;       // CPU亲和性（绑定到特定核心）
    int64_t task_timeout_ms = 0;         // 任务超时时间（毫秒），0表示不超时
    bool enable_work_stealing = false;   // 启用工作窃取
};

/**
 * @brief 实时线程配置结构
 */
struct RealtimeThreadConfig {
    std::string thread_name;                              // 线程名称
    int64_t cycle_period_ns = 0;                          // 周期（纳秒），如2000000表示2ms
    int thread_priority = 0;                              // 线程优先级（SCHED_FIFO: 1-99，Linux）
    std::vector<int> cpu_affinity;                        // CPU亲和性（绑定到特定核心）
    std::function<void()> cycle_callback;                 // 周期回调函数
    ICycleManager* cycle_manager = nullptr;               // 可选的周期管理器接口（用于更精确的周期控制）
};

/**
 * @brief 统一执行器配置结构
 * 
 * 用于 Executor::initialize() 和 ExecutorManager::initialize_async_executor()
 * 可内嵌或映射为 ThreadPoolConfig
 */
struct ExecutorConfig {
    size_t min_threads = 4;              // 最小线程数
    size_t max_threads = 16;             // 最大线程数
    size_t queue_capacity = 1000;        // 任务队列容量
    int thread_priority = 0;              // 线程优先级
    std::vector<int> cpu_affinity;       // CPU亲和性
    int64_t task_timeout_ms = 0;         // 任务超时时间（毫秒）
    bool enable_work_stealing = false;   // 启用工作窃取
};

/**
 * @brief 线程池状态结构（用于监控）
 */
struct ThreadPoolStatus {
    size_t total_threads = 0;            // 总线程数
    size_t active_threads = 0;            // 活跃线程数
    size_t idle_threads = 0;             // 空闲线程数
    size_t queue_size = 0;               // 队列中任务数
    size_t total_tasks = 0;               // 总任务数
    size_t completed_tasks = 0;          // 已完成任务数
    size_t failed_tasks = 0;             // 失败任务数
    double avg_task_time_ms = 0.0;       // 平均任务执行时间（毫秒）
    double cpu_usage_percent = 0.0;       // CPU使用率
};

} // namespace executor
