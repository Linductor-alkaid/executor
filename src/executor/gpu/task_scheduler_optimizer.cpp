#include "executor/gpu/task_scheduler_optimizer.hpp"
#include <algorithm>

namespace executor {
namespace gpu {

TaskSchedulerOptimizer::TaskSchedulerOptimizer() : config_() {}

TaskSchedulerOptimizer::TaskSchedulerOptimizer(const Config& config) : config_(config) {}

// --- 任务提交 ---

bool TaskSchedulerOptimizer::add_task(const GpuTaskNode& task) {
    std::unique_lock lock(graph_mutex_);
    if (task_graph_.size() >= config_.max_pending_tasks) {
        return false;
    }
    if (task_graph_.count(task.task_id)) {
        return false;  // 重复 task_id
    }
    task_graph_[task.task_id] = task;

    if (config_.enable_priority_inheritance) {
        apply_priority_inheritance();
    }
    return true;
}

bool TaskSchedulerOptimizer::remove_task(const std::string& task_id) {
    std::unique_lock lock(graph_mutex_);
    return task_graph_.erase(task_id) > 0;
}

void TaskSchedulerOptimizer::mark_completed(const std::string& task_id) {
    std::unique_lock lock(graph_mutex_);
    task_graph_.erase(task_id);
    completed_tasks_.insert(task_id);

    {
        std::lock_guard slock(stats_mutex_);
        ++total_scheduled_;
    }
}

size_t TaskSchedulerOptimizer::pending_count() const {
    std::shared_lock lock(graph_mutex_);
    return task_graph_.size();
}

// --- 依赖图优化 ---

std::vector<GpuTaskNode> TaskSchedulerOptimizer::get_ready_tasks() {
    std::unique_lock lock(graph_mutex_);
    std::vector<GpuTaskNode> ready;

    for (const auto& [task_id, task] : task_graph_) {
        bool all_deps_met = true;
        for (const auto& dep : task.dependencies) {
            if (!completed_tasks_.count(dep) && task_graph_.count(dep)) {
                all_deps_met = false;
                {
                    std::lock_guard slock(stats_mutex_);
                    ++dependency_waits_;
                }
                break;
            }
        }
        if (all_deps_met) {
            ready.push_back(task);
        }
    }

    // 按优先级排序（高优先级在前）
    std::sort(ready.begin(), ready.end(), [](const GpuTaskNode& a, const GpuTaskNode& b) {
        return a.priority > b.priority;
    });

    return ready;
}

bool TaskSchedulerOptimizer::has_cycle() const {
    std::shared_lock lock(graph_mutex_);
    std::unordered_set<std::string> visiting;
    std::unordered_set<std::string> visited;

    for (const auto& [task_id, _] : task_graph_) {
        if (!visited.count(task_id)) {
            if (dfs_has_cycle(task_id, visiting, visited)) {
                return true;
            }
        }
    }
    return false;
}

bool TaskSchedulerOptimizer::dfs_has_cycle(const std::string& node,
                                            std::unordered_set<std::string>& visiting,
                                            std::unordered_set<std::string>& visited) const {
    visiting.insert(node);
    auto it = task_graph_.find(node);
    if (it != task_graph_.end()) {
        for (const auto& dep : it->second.dependencies) {
            if (visiting.count(dep)) {
                return true;  // 环
            }
            if (!visited.count(dep) && task_graph_.count(dep)) {
                if (dfs_has_cycle(dep, visiting, visited)) {
                    return true;
                }
            }
        }
    }
    visiting.erase(node);
    visited.insert(node);
    return false;
}

void TaskSchedulerOptimizer::apply_priority_inheritance() {
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto& [task_id, task] : task_graph_) {
            for (const auto& dep : task.dependencies) {
                auto dep_it = task_graph_.find(dep);
                if (dep_it != task_graph_.end() && dep_it->second.priority < task.priority) {
                    dep_it->second.priority = task.priority;
                    changed = true;
                    {
                        std::lock_guard slock(stats_mutex_);
                        ++priority_promotions_;
                    }
                }
            }
        }
    }
}

// --- 负载均衡 ---

void TaskSchedulerOptimizer::update_device_load(const DeviceLoad& load) {
    std::lock_guard lock(device_mutex_);
    device_loads_[load.device_id] = load;
}

int TaskSchedulerOptimizer::select_best_device(const GpuTaskNode& task) const {
    if (!config_.enable_load_balancing) {
        return 0;
    }

    std::lock_guard lock(device_mutex_);
    if (device_loads_.empty()) {
        return 0;
    }

    int best_device = device_loads_.begin()->first;
    size_t min_cost = device_loads_.begin()->second.estimated_total_cost;

    for (const auto& [device_id, load] : device_loads_) {
        size_t total_cost = load.estimated_total_cost + task.estimated_cost;
        if (total_cost < min_cost) {
            min_cost = total_cost;
            best_device = device_id;
        }
    }

    {
        std::lock_guard slock(stats_mutex_);
        ++load_balance_moves_;
    }

    return best_device;
}

std::vector<DeviceLoad> TaskSchedulerOptimizer::get_device_loads() const {
    std::lock_guard lock(device_mutex_);
    std::vector<DeviceLoad> loads;
    loads.reserve(device_loads_.size());
    for (const auto& [_, load] : device_loads_) {
        loads.push_back(load);
    }
    return loads;
}

// --- 统计 ---

TaskSchedulingStats TaskSchedulerOptimizer::get_stats() const {
    std::lock_guard lock(stats_mutex_);
    TaskSchedulingStats stats;
    stats.total_tasks_scheduled = total_scheduled_;
    stats.priority_promotions = priority_promotions_;
    stats.dependency_waits = dependency_waits_;
    stats.load_balance_moves = load_balance_moves_;
    stats.avg_wait_time_us = (total_scheduled_ > 0) ? total_wait_us_ / total_scheduled_ : 0.0;
    return stats;
}

void TaskSchedulerOptimizer::reset_stats() {
    std::lock_guard lock(stats_mutex_);
    total_scheduled_ = 0;
    priority_promotions_ = 0;
    dependency_waits_ = 0;
    load_balance_moves_ = 0;
    total_wait_us_ = 0.0;
}

TaskSchedulerOptimizer::Config TaskSchedulerOptimizer::get_config() const {
    std::shared_lock lock(graph_mutex_);
    return config_;
}

void TaskSchedulerOptimizer::update_config(const Config& config) {
    std::unique_lock lock(graph_mutex_);
    config_ = config;
}

} // namespace gpu
} // namespace executor