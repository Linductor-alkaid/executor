#include "task_dispatcher.hpp"
#include "priority_scheduler.hpp"

namespace executor {

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
    size_t dispatched = 0;
    
    for (size_t i = 0; i < max_tasks; ++i) {
        if (!dispatch()) {
            break;
        }
        ++dispatched;
    }
    
    return dispatched;
}

size_t TaskDispatcher::pending_tasks() const {
    return scheduler_.size();
}

} // namespace executor
