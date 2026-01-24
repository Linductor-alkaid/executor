#pragma once

#include "executor/types.hpp"
#include "executor/monitor/task_monitor.hpp"
#include <string>
#include <map>

namespace executor {
namespace monitor {

/**
 * @brief 统计收集器
 *
 * 聚合与查询门面，持有 TaskMonitor，对外提供 get_task_statistics /
 * get_all_task_statistics。供 ExecutorManager / Executor 暴露监控 API。
 */
class StatisticsCollector {
public:
    StatisticsCollector() = default;
    ~StatisticsCollector() = default;

    StatisticsCollector(const StatisticsCollector&) = delete;
    StatisticsCollector& operator=(const StatisticsCollector&) = delete;
    StatisticsCollector(StatisticsCollector&&) = delete;
    StatisticsCollector& operator=(StatisticsCollector&&) = delete;

    /**
     * @brief 获取 TaskMonitor，供 ThreadPool / 实时扩展写入
     */
    TaskMonitor& get_task_monitor();

    /**
     * @brief 按 task_type 获取聚合统计
     */
    TaskStatistics get_task_statistics(const std::string& task_type) const;

    /**
     * @brief 获取全部 task_type 的聚合统计
     */
    std::map<std::string, TaskStatistics> get_all_task_statistics() const;

private:
    TaskMonitor task_monitor_;
};

}  // namespace monitor
}  // namespace executor
