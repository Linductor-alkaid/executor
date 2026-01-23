#pragma once

#include "load_balancer.hpp"
#include "worker_local_queue.hpp"
#include "priority_scheduler.hpp"
#include "../task/task.hpp"
#include <vector>
#include <cstddef>

namespace executor {

// 前向声明
class PriorityScheduler;

/**
 * @brief 任务分发器
 * 
 * 从 PriorityScheduler 获取任务，根据 LoadBalancer 的策略
 * 选择目标工作线程，将任务分发给选定的工作线程。
 */
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
                   std::vector<WorkerLocalQueue>& local_queues);

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
    bool dispatch();

    /**
     * @brief 分发指定任务
     * 
     * 将指定的任务分发到选定的工作线程。
     * 
     * @param task 要分发的任务
     * @return 成功分发返回 true
     */
    bool dispatch_task(const Task& task);

    /**
     * @brief 批量分发任务
     * 
     * 从调度器批量获取任务并分发，减少锁竞争。
     * 
     * @param max_tasks 最多分发的任务数
     * @return 实际分发的任务数
     */
    size_t dispatch_batch(size_t max_tasks = 10);

    /**
     * @brief 获取待分发任务数量（调度器中的任务数）
     * 
     * @return 调度器中待分发的任务数
     */
    size_t pending_tasks() const;

private:
    LoadBalancer& balancer_;                      // 负载均衡器
    PriorityScheduler& scheduler_;                // 优先级调度器
    std::vector<WorkerLocalQueue>& local_queues_; // 工作线程本地队列数组
};

} // namespace executor
