#pragma once

#include "load_balancer.hpp"
#include "priority_scheduler.hpp"
#include "../task/task.hpp"
#include <vector>
#include <cstddef>
#include <memory>
#include <tuple>

namespace executor {

// 前向声明
class PriorityScheduler;

namespace detail {

// 将 src Task 复制到 dst Task，保持两套队列实现共享同一份拷贝语义。
// 之前位于 task_dispatcher.cpp 的匿名命名空间，模板化后需要可见。
inline void copy_task_fields(Task& dst, const Task& src) {
    dst.task_id = src.task_id;
    dst.priority = src.priority;
    dst.function = src.function;
    dst.submit_time_ns = src.submit_time_ns;
    dst.timeout_ms = src.timeout_ms;
    dst.dependencies = src.dependencies;
    dst.cancelled.store(src.cancelled.load(std::memory_order_acquire),
                        std::memory_order_release);
}

}  // namespace detail

/**
 * @brief 任务分发器（模板化以支持多种工作线程本地队列实现）
 *
 * QueueT 必须提供与 WorkerLocalQueue 兼容的接口：
 *   - bool push(const Task&)
 *   - size_t push_batch(const Task*, size_t)
 *   - size_t size() const
 *
 * 既支持默认的 WorkerLocalQueue，也支持 USE_LOCKFREE_WORKER_QUEUE 模式下的
 * LockFreeWorkerQueue。
 */
template<typename QueueT>
class TaskDispatcher {
public:
    /**
     * @brief 构造函数
     *
     * @param balancer 负载均衡器引用
     * @param scheduler 优先级调度器引用
     * @param local_queues 工作线程本地队列数组引用
     */
    TaskDispatcher(LoadBalancer& balancer,
                   PriorityScheduler& scheduler,
                   std::vector<QueueT>& local_queues)
        : balancer_(balancer)
        , scheduler_(scheduler)
        , local_queues_(local_queues) {}

    /**
     * @brief 析构函数
     */
    ~TaskDispatcher() = default;

    // 禁止拷贝和移动
    TaskDispatcher(const TaskDispatcher&) = delete;
    TaskDispatcher& operator=(const TaskDispatcher&) = delete;
    TaskDispatcher(TaskDispatcher&&) = delete;
    TaskDispatcher& operator=(TaskDispatcher&&) = delete;

    /**
     * @brief 分发任务（从调度器获取并分发）
     *
     * 从 PriorityScheduler 获取一个任务，使用 LoadBalancer
     * 选择目标工作线程，将任务分发到该线程的本地队列。
     *
     * @return 成功分发返回 true，调度器为空返回 false
     */
    bool dispatch() {
        Task task;

        // 从调度器获取任务
        if (!scheduler_.dequeue(task)) {
            return false;
        }

        // 使用负载均衡器选择目标工作线程
        size_t worker_id = balancer_.select_worker();

        // 检查 worker_id 是否有效
        if (worker_id >= local_queues_.size()) {
            // 260610P009: resize 期间 LoadBalancer 可能返回已被移除的 worker_id。
            // 修复前: 直接丢弃,任务从 scheduler 出队后永久丢失(高并发场景下可能频繁发生)。
            // 修复后: 将任务重新 enqueue 回 scheduler,等待下一轮 dispatch。
            // 这样保证 total == completed + failed(无任务丢失)。
            // 注: PriorityScheduler::enqueue 是 const Task&,所以这里传 task 走 copy 而非 move
            // (Task 应当支持轻量 copy —— 若 Task 变重,再加 move overload)
            scheduler_.enqueue(task);
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

    /**
     * @brief 分发指定任务
     *
     * @param task 要分发的任务
     * @return 成功分发返回 true
     */
    bool dispatch_task(const Task& task) {
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

    /**
     * @brief 批量分发任务
     *
     * 从调度器批量获取任务并分发，减少锁竞争。
     *
     * @param max_tasks 最多分发的任务数
     * @return 实际分发的任务数
     */
    size_t dispatch_batch(size_t max_tasks = 10) {
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
                detail::copy_task_fields(tmp[j], batch[indices[j]]);

            size_t pushed = local_queues_[w].push_batch(tmp.get(), indices.size());
            total_dispatched += pushed;

            // 将未能推送的任务放回调度器
            for (size_t j = pushed; j < indices.size(); ++j) {
                scheduler_.enqueue(tmp[j]);
            }

            load_updates.emplace_back(w, local_queues_[w].size(), 0);
        }

        if (!load_updates.empty())
            balancer_.update_load_batch(load_updates);

        return total_dispatched;
    }

    /**
     * @brief 获取待分发任务数量（调度器中的任务数）
     */
    size_t pending_tasks() const {
        return scheduler_.size();
    }

private:
    LoadBalancer& balancer_;                      // 负载均衡器
    PriorityScheduler& scheduler_;                // 优先级调度器
    std::vector<QueueT>& local_queues_;           // 工作线程本地队列数组
};

} // namespace executor
