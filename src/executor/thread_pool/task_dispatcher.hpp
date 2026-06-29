#pragma once

#include "load_balancer.hpp"
#include "priority_scheduler.hpp"
#include "../task/task.hpp"
#include <vector>
#include <cstddef>
#include <memory>
#include <tuple>
#include <shared_mutex>

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
     * @param local_queues_mutex 保护 local_queues_ 的 shared_mutex 指针
     *        (P-260617-002: dispatch 路径持 shared_lock，与 resize 路径
     *        的 unique_lock 配对，防 vector element 重建期间悬空访问)
     */
    TaskDispatcher(LoadBalancer& balancer,
                   PriorityScheduler& scheduler,
                   std::vector<QueueT>& local_queues,
                   std::shared_mutex* local_queues_mutex = nullptr)
        : balancer_(balancer)
        , scheduler_(scheduler)
        , local_queues_slot_(nullptr)
        , legacy_local_queues_(&local_queues)
        , local_queues_snapshot_()
        , local_queues_mutex_(local_queues_mutex) {}

    TaskDispatcher(LoadBalancer& balancer,
                   PriorityScheduler& scheduler,
                   const std::shared_ptr<std::vector<QueueT>>* local_queues_slot,
                   std::shared_mutex* local_queues_mutex = nullptr)
        : balancer_(balancer)
        , scheduler_(scheduler)
        , local_queues_slot_(local_queues_slot)
        , legacy_local_queues_(nullptr)
        , local_queues_snapshot_()
        , local_queues_mutex_(local_queues_mutex) {}

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
        // P-260617-002: 持 shared_lock 保护 local_queues_ 访问，与 resize 路径的
        // unique_lock 配对。注意：RAII wrapper 保证所有 return 路径都释放锁。
        std::unique_ptr<std::shared_lock<std::shared_mutex>> lq_lock;
        if (local_queues_mutex_) {
            lq_lock = std::make_unique<std::shared_lock<std::shared_mutex>>(*local_queues_mutex_);
        }
        auto local_queues = queues_snapshot_locked();

        Task task;

        // 从调度器获取任务
        if (!scheduler_.dequeue(task)) {
            return false;
        }

        // 使用负载均衡器选择目标工作线程
        size_t worker_id = balancer_.select_worker();

        // 检查 worker_id 是否有效
        if (!local_queues || worker_id >= local_queues->size()) {
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
        bool success = (*local_queues)[worker_id].push(task);

        if (success) {
            // 更新负载信息
            size_t queue_size = (*local_queues)[worker_id].size();
            balancer_.update_load(worker_id, queue_size, 0);
        } else {
            // P-260623-001: 推送失败时(本地队列满)重新入队 scheduler,
            // 避免任务从 scheduler 出队后既不在本地队列也不在 scheduler 而被永久丢弃。
            // 镜像 dispatch_batch 中的回 enqueue 模式 (dispatch_batch 会回 enqueue 未推送的任务)。
            // PriorityScheduler::enqueue 接受 const Task&,所以这里传 task 走 copy。
            scheduler_.enqueue(task);
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
        // P-260617-002: 持 shared_lock 保护 local_queues_ 访问
        std::unique_ptr<std::shared_lock<std::shared_mutex>> lq_lock;
        if (local_queues_mutex_) {
            lq_lock = std::make_unique<std::shared_lock<std::shared_mutex>>(*local_queues_mutex_);
        }
        auto local_queues = queues_snapshot_locked();

        auto enqueue_fallback = [this, &task]() {
            try {
                scheduler_.enqueue(task);
            } catch (...) {
                return false;
            }
            return false;
        };

        // 使用负载均衡器选择目标工作线程
        size_t worker_id = balancer_.select_worker();

        // 检查 worker_id 是否有效
        if (!local_queues || worker_id >= local_queues->size()) {
            // 260610P009 / 260625-007: resize 期间 LoadBalancer 可能返回已越界的
            // worker_id。dispatch_task 不从 scheduler 出队,但调用方已提交 task；
            // fallback 回 scheduler,避免静默丢任务。
            return enqueue_fallback();
        }

        // 分发任务到选定线程的本地队列
        bool success = (*local_queues)[worker_id].push(task);

        if (success) {
            // 更新负载信息
            size_t queue_size = (*local_queues)[worker_id].size();
            balancer_.update_load(worker_id, queue_size, 0);
        } else {
            // P-260623-001 / 260625-007: 本地队列满时回 enqueue 到 scheduler,
            // 与 dispatch()/dispatch_batch() 的无任务丢失契约保持一致。
            return enqueue_fallback();
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
        if (max_tasks == 0) return 0;

        // P-260617-002: 持 shared_lock 保护 local_queues_ 访问整个函数体。
        // 早返回的 0 路径不需要锁（仅读 size，无 race 风险）。
        std::unique_ptr<std::shared_lock<std::shared_mutex>> lq_lock;
        if (local_queues_mutex_) {
            lq_lock = std::make_unique<std::shared_lock<std::shared_mutex>>(*local_queues_mutex_);
        }
        auto local_queues = queues_snapshot_locked();
        if (!local_queues || local_queues->empty()) return 0;

        std::unique_ptr<Task[]> batch(new Task[max_tasks]);
        const size_t n = scheduler_.dequeue_batch(batch.get(), max_tasks);
        if (n == 0) return 0;

        // P-260626-004: num_workers 是 shared_lock 内的本地快照，但
        // resize() 持 unique_lock(local_queues_mutex_) 重建 vector，因此
        // 持锁期间 num_workers 与 local_queues_ 一致 — 然而 L192 的早返回
        // 仅在 *函数入口* 检测 local_queues_ 非空；存在一条 race: 调用方
        // 在 L192 检查通过后, 持锁前被 resize(0) 抢先, 进入本函数时
        // local_queues_ 为空, 但 batch 已从 scheduler 出队。统一防线:
        // 持锁后再断言 num_workers > 0, 否则把已出队的 batch 全部回 enqueue
        // 到 scheduler, 避免钳位到 0 后的越界访问 (local_queues_[w]) 与静默丢任务。
        const size_t num_workers = local_queues->size();
        if (num_workers == 0) {
            for (size_t i = 0; i < n; ++i) {
                scheduler_.enqueue(batch[i]);
            }
            return 0;
        }
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

            size_t pushed = (*local_queues)[w].push_batch(tmp.get(), indices.size());
            total_dispatched += pushed;

            // 将未能推送的任务放回调度器
            for (size_t j = pushed; j < indices.size(); ++j) {
                scheduler_.enqueue(tmp[j]);
            }

            load_updates.emplace_back(w, (*local_queues)[w].size(), 0);
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
    std::shared_ptr<std::vector<QueueT>> queues_snapshot_locked() const {
        if (local_queues_slot_) {
            return std::atomic_load_explicit(local_queues_slot_,
                                             std::memory_order_acquire);
        }
        if (legacy_local_queues_) {
            if (!local_queues_snapshot_) {
                local_queues_snapshot_ =
                    std::shared_ptr<std::vector<QueueT>>(legacy_local_queues_,
                                                         [](std::vector<QueueT>*) {});
            }
            return local_queues_snapshot_;
        }
        return {};
    }

    LoadBalancer& balancer_;                      // 负载均衡器
    PriorityScheduler& scheduler_;                // 优先级调度器
    const std::shared_ptr<std::vector<QueueT>>* local_queues_slot_;
    std::vector<QueueT>* legacy_local_queues_;
    mutable std::shared_ptr<std::vector<QueueT>> local_queues_snapshot_;
    std::shared_mutex* local_queues_mutex_;       // P-260617-002: 保护 local_queues_ 的 shared_mutex
};

} // namespace executor
