#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <atomic>
#include <functional>
#include <map>

namespace executor {

/**
 * @brief 任务优先级枚举
 */
enum class TaskPriority {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    CRITICAL = 3
};

/**
 * @brief 任务结构体
 */
struct Task {
    std::string task_id;                          // 任务ID
    TaskPriority priority = TaskPriority::NORMAL; // 任务优先级
    std::function<void()> function;              // 任务函数
    int64_t submit_time_ns = 0;                   // 提交时间（纳秒）
    int64_t timeout_ms = 0;                       // 超时时间（毫秒），0表示不超时
    std::vector<std::string> dependencies;       // 依赖任务ID
    std::atomic<bool> cancelled{false};           // 取消标志
};

/**
 * @brief 异步执行器状态结构
 */
struct AsyncExecutorStatus {
    std::string name;                             // 执行器名称
    bool is_running = false;                      // 是否运行中
    size_t active_tasks = 0;                      // 活跃任务数
    size_t completed_tasks = 0;                   // 已完成任务数
    size_t failed_tasks = 0;                      // 失败任务数
    size_t queue_size = 0;                        // 队列大小
    double avg_task_time_ms = 0.0;                // 平均任务执行时间（毫秒）
};

/**
 * @brief 实时执行器状态结构
 */
struct RealtimeExecutorStatus {
    std::string name;                             // 执行器名称
    bool is_running = false;                      // 是否运行中
    int64_t cycle_period_ns = 0;                  // 周期（纳秒）
    int64_t cycle_count = 0;                      // 周期计数
    int64_t cycle_timeout_count = 0;              // 周期超时计数
    double avg_cycle_time_ns = 0.0;                // 平均周期执行时间（纳秒）
    double max_cycle_time_ns = 0.0;               // 最大周期执行时间（纳秒）
};

/**
 * @brief 任务统计信息（用于监控）
 */
struct TaskStatistics {
    int64_t total_count = 0;                     // 总任务数
    int64_t success_count = 0;                    // 成功任务数
    int64_t fail_count = 0;                       // 失败任务数
    int64_t timeout_count = 0;                    // 超时任务数
    int64_t total_execution_time_ns = 0;          // 总执行时间（纳秒）
    int64_t max_execution_time_ns = 0;            // 最大执行时间（纳秒）
    int64_t min_execution_time_ns = 0;            // 最小执行时间（纳秒）
};

/**
 * @brief 周期统计信息（用于ICycleManager）
 */
struct CycleStatistics {
    std::string name;                             // 周期任务名称
    int64_t period_ns = 0;                        // 周期（纳秒）
    int64_t cycle_count = 0;                      // 周期计数
    int64_t timeout_count = 0;                    // 超时计数
    double avg_cycle_time_ns = 0.0;               // 平均周期执行时间（纳秒）
    double max_cycle_time_ns = 0.0;               // 最大周期执行时间（纳秒）
    bool is_running = false;                      // 是否运行中
};

} // namespace executor
