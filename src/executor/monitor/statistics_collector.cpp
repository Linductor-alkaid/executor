#include "executor/monitor/statistics_collector.hpp"

namespace executor {
namespace monitor {

TaskMonitor& StatisticsCollector::get_task_monitor() {
    return task_monitor_;
}

TaskStatistics StatisticsCollector::get_task_statistics(
    const std::string& task_type) const {
    return task_monitor_.get_statistics(task_type);
}

std::map<std::string, TaskStatistics>
StatisticsCollector::get_all_task_statistics() const {
    return task_monitor_.get_all_statistics();
}

}  // namespace monitor
}  // namespace executor
