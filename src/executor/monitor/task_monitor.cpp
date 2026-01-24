#include "executor/monitor/task_monitor.hpp"
#include <algorithm>
#include <atomic>

namespace executor {
namespace monitor {

void TaskMonitor::record_task_start(const std::string& task_id,
                                    const std::string& task_type) {
    if (!enabled_.load(std::memory_order_relaxed)) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    task_id_to_type_[task_id] = task_type;
}

void TaskMonitor::record_task_complete(const std::string& task_id,
                                       bool success,
                                       int64_t execution_time_ns) {
    if (!enabled_.load(std::memory_order_relaxed)) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = task_id_to_type_.find(task_id);
    if (it == task_id_to_type_.end()) {
        return;  // 未 record_task_start，忽略
    }
    std::string task_type = std::move(it->second);
    task_id_to_type_.erase(it);

    Stats& s = type_stats_[task_type];
    s.total_count += 1;
    if (success) {
        s.success_count += 1;
    } else {
        s.fail_count += 1;
    }
    s.total_execution_time_ns += execution_time_ns;
    if (execution_time_ns > s.max_execution_time_ns) {
        s.max_execution_time_ns = execution_time_ns;
    }
    if (s.min_execution_time_ns == 0) {
        s.min_execution_time_ns = execution_time_ns;
    } else {
        s.min_execution_time_ns =
            std::min(s.min_execution_time_ns, execution_time_ns);
    }
}

void TaskMonitor::record_task_timeout(const std::string& task_id) {
    if (!enabled_.load(std::memory_order_relaxed)) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = task_id_to_type_.find(task_id);
    if (it == task_id_to_type_.end()) {
        return;
    }
    std::string task_type = std::move(it->second);
    task_id_to_type_.erase(it);

    Stats& s = type_stats_[task_type];
    s.total_count += 1;
    s.timeout_count += 1;
}

TaskStatistics TaskMonitor::get_statistics(const std::string& task_type) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = type_stats_.find(task_type);
    if (it == type_stats_.end()) {
        return {};
    }
    const Stats& s = it->second;
    TaskStatistics out;
    out.total_count = s.total_count;
    out.success_count = s.success_count;
    out.fail_count = s.fail_count;
    out.timeout_count = s.timeout_count;
    out.total_execution_time_ns = s.total_execution_time_ns;
    out.max_execution_time_ns = s.max_execution_time_ns;
    out.min_execution_time_ns = s.min_execution_time_ns;
    return out;
}

std::map<std::string, TaskStatistics> TaskMonitor::get_all_statistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<std::string, TaskStatistics> result;
    for (const auto& [k, s] : type_stats_) {
        TaskStatistics t;
        t.total_count = s.total_count;
        t.success_count = s.success_count;
        t.fail_count = s.fail_count;
        t.timeout_count = s.timeout_count;
        t.total_execution_time_ns = s.total_execution_time_ns;
        t.max_execution_time_ns = s.max_execution_time_ns;
        t.min_execution_time_ns = s.min_execution_time_ns;
        result[k] = t;
    }
    return result;
}

void TaskMonitor::set_enabled(bool enabled) {
    enabled_.store(enabled, std::memory_order_relaxed);
}

bool TaskMonitor::is_enabled() const {
    return enabled_.load(std::memory_order_relaxed);
}

}  // namespace monitor
}  // namespace executor
