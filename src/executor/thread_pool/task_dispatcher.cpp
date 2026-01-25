#include "task_dispatcher.hpp"
#include "priority_scheduler.hpp"
#include "load_balancer.hpp"
#include "worker_local_queue.hpp"
#include "executor/types.hpp"
#include <memory>
#include <tuple>

namespace executor {

namespace {
void copy_task(Task& dest, const Task& src) {
    dest.task_id = src.task_id;
    dest.priority = src.priority;
    dest.function = src.function;
    dest.submit_time_ns = src.submit_time_ns;
    dest.timeout_ms = src.timeout_ms;
    dest.dependencies = src.dependencies;
    dest.cancelled.store(src.cancelled.load(std::memory_order_acquire), std::memory_order_release);
}
}  // namespace

TaskDispatcher::TaskDispatcher(LoadBalancer& balancer,
                               PriorityScheduler& scheduler,
                               std::vector<WorkerLocalQueue>& local_queues)
    : balancer_(balancer)
    , scheduler_(scheduler)
    , local_queues_(local_queues) {
}

bool TaskDispatcher::dispatch() {
    Task task;
    
    // 从调度器获取任务
    if (!scheduler_.dequeue(task)) {
        return false;
    }
    
    // 使用负载均衡器选择目标工作线程
    size_t worker_id = balancer_.select_worker();
    
    // 检查 worker_id 是否有效
    if (worker_id >= local_queues_.size()) {
        // 如果 worker_id 无效，将任务重新放回调度器（这里简化处理，实际应该重试）
        // 注意：PriorityScheduler 不支持重新入队，这里只能丢弃或记录错误
        return false;
    }
    
    // 分发任务到选定线程的本地队列
    bool success = local_queues_[worker_id].push(task);
    
    if (success) {
        // 更新负载信息
        size_t queue_size = local_queues_[worker_id].size();
        balancer_.update_load(worker_id, queue_size, 0);
    }
    
    return success;
}

bool TaskDispatcher::dispatch_task(const Task& task) {
    // 使用负载均衡器选择目标工作线程
    size_t worker_id = balancer_.select_worker();
    
    // 检查 worker_id 是否有效
    if (worker_id >= local_queues_.size()) {
        return false;
    }
    
    // 分发任务到选定线程的本地队列
    bool success = local_queues_[worker_id].push(task);
    
    if (success) {
        // 更新负载信息
        size_t queue_size = local_queues_[worker_id].size();
        balancer_.update_load(worker_id, queue_size, 0);
    }
    
    return success;
}

size_t TaskDispatcher::dispatch_batch(size_t max_tasks) {
    if (max_tasks == 0 || local_queues_.empty()) return 0;

    std::unique_ptr<Task[]> batch(new Task[max_tasks]);
    const size_t n = scheduler_.dequeue_batch(batch.get(), max_tasks);
    if (n == 0) return 0;

    const size_t num_workers = local_queues_.size();
    std::vector<std::vector<size_t>> by_worker(num_workers);

    for (size_t i = 0; i < n; ++i) {
        size_t worker_id = balancer_.select_worker();
        if (worker_id >= num_workers) worker_id = 0;
        by_worker[worker_id].push_back(i);
    }

    std::unique_ptr<Task[]> tmp(new Task[n]);
    std::vector<std::tuple<size_t, size_t, size_t>> load_updates;
    size_t total_dispatched = 0;

    for (size_t w = 0; w < num_workers; ++w) {
        const std::vector<size_t>& indices = by_worker[w];
        if (indices.empty()) continue;

        for (size_t j = 0; j < indices.size(); ++j)
            copy_task(tmp[j], batch[indices[j]]);

        size_t pushed = local_queues_[w].push_batch(tmp.get(), indices.size());
        total_dispatched += pushed;
        load_updates.emplace_back(w, local_queues_[w].size(), 0);
    }

    if (!load_updates.empty())
        balancer_.update_load_batch(load_updates);

    return total_dispatched;
}

size_t TaskDispatcher::pending_tasks() const {
    return scheduler_.size();
}

} // namespace executor
