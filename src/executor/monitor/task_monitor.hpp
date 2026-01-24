#pragma once

#include "executor/types.hpp"
#include <string>
#include <map>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <cstdint>

namespace executor {
namespace monitor {

/**
 * @brief 任务监控器
 *
 * 接收任务生命周期事件（start/complete/timeout），按 task_type 聚合统计。
 * 供 ThreadPool 在 execute_task 前后钩子调用。
 */
class TaskMonitor {
public:
    TaskMonitor() = default;
    ~TaskMonitor() = default;

    TaskMonitor(const TaskMonitor&) = delete;
    TaskMonitor& operator=(const TaskMonitor&) = delete;
    TaskMonitor(TaskMonitor&&) = delete;
    TaskMonitor& operator=(TaskMonitor&&) = delete;

    /**
     * @brief 记录任务开始
     * @param task_id 任务 ID
     * @param task_type 任务类型（默认 "default"），用于聚合统计
     */
    void record_task_start(const std::string& task_id,
                           const std::string& task_type = "default");

    /**
     * @brief 记录任务完成
     * @param task_id 任务 ID
     * @param success 是否成功
     * @param execution_time_ns 执行时间（纳秒）
     */
    void record_task_complete(const std::string& task_id,
                              bool success,
                              int64_t execution_time_ns);

    /**
     * @brief 记录任务超时
     * @param task_id 任务 ID
     */
    void record_task_timeout(const std::string& task_id);

    /**
     * @brief 按 task_type 获取聚合统计
     * @param task_type 任务类型
     * @return 统计信息；若不存在则返回全 0
     */
    TaskStatistics get_statistics(const std::string& task_type) const;

    /**
     * @brief 获取全部 task_type 的聚合统计
     */
    std::map<std::string, TaskStatistics> get_all_statistics() const;

    void set_enabled(bool enabled);
    bool is_enabled() const;

private:
    mutable std::mutex mutex_;
    std::atomic<bool> enabled_{true};

    /// task_id -> task_type，用于 complete/timeout 时查找
    std::unordered_map<std::string, std::string> task_id_to_type_;

    /// 按 task_type 聚合的统计（内部存储，与 TaskStatistics 一致）
    struct Stats {
        int64_t total_count = 0;
        int64_t success_count = 0;
        int64_t fail_count = 0;
        int64_t timeout_count = 0;
        int64_t total_execution_time_ns = 0;
        int64_t max_execution_time_ns = 0;
        int64_t min_execution_time_ns = 0;  /// 0 表示尚未有样本
    };
    std::map<std::string, Stats> type_stats_;
};

}  // namespace monitor
}  // namespace executor
