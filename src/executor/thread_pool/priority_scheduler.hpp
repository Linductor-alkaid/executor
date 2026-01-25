#pragma once

#include "executor/types.hpp"
#include "../task/task.hpp"
#include <queue>
#include <mutex>
#include <functional>
#include <memory>
#include <cstddef>

namespace executor {

/**
 * @brief 优先级调度器
 * 
 * 管理4个优先级队列（CRITICAL, HIGH, NORMAL, LOW），
 * 提供线程安全的enqueue和dequeue接口。
 * 按优先级顺序调度任务，同优先级任务按提交时间（FIFO）调度。
 * 
 * 注意：使用std::shared_ptr<Task>存储任务，因为Task包含std::atomic<bool>不可复制。
 */
class PriorityScheduler {
public:
    /**
     * @brief 构造函数
     */
    PriorityScheduler() = default;

    /**
     * @brief 析构函数
     */
    ~PriorityScheduler() = default;

    // 禁止拷贝和移动
    PriorityScheduler(const PriorityScheduler&) = delete;
    PriorityScheduler& operator=(const PriorityScheduler&) = delete;
    PriorityScheduler(PriorityScheduler&&) = delete;
    PriorityScheduler& operator=(PriorityScheduler&&) = delete;

    /**
     * @brief 添加任务到优先级队列
     * 
     * 根据任务的优先级将任务放入对应的队列。
     * 
     * @param task 任务对象（会被复制为shared_ptr）
     */
    void enqueue(const Task& task);

    /**
     * @brief 从优先级队列获取任务（按优先级顺序）
     * 
     * 按优先级从高到低（CRITICAL -> HIGH -> NORMAL -> LOW）获取任务。
     * 同优先级任务按提交时间（FIFO）获取。
     * 
     * @param task 用于接收任务的引用
     * @return 如果成功获取任务返回true，队列为空返回false
     */
    bool dequeue(Task& task);

    /**
     * @brief 获取队列总大小
     * 
     * @return 所有优先级队列中的任务总数
     */
    size_t size() const;

    /**
     * @brief 检查队列是否为空
     * 
     * @return 如果所有队列都为空返回true，否则返回false
     */
    bool empty() const;

    /**
     * @brief 清空所有队列
     */
    void clear();

private:
    // shared_ptr比较器：比较Task对象（通过解引用）
    // priority_queue是最大堆，顶部是"最大"的元素
    // Task::operator< 返回true表示lhs优先级低于rhs，所以rhs应该在顶部
    // 因此TaskPtrCompare应该返回*lhs < *rhs，这样rhs（优先级高）会在顶部
    struct TaskPtrCompare {
        bool operator()(const std::shared_ptr<Task>& lhs, const std::shared_ptr<Task>& rhs) const {
            if (!lhs || !rhs) return false;
            // 使用Task的operator<进行比较
            // 如果*lhs < *rhs为真，表示lhs优先级低于rhs，rhs应该在顶部
            return *lhs < *rhs;
        }
    };

    // 使用priority_queue存储shared_ptr<Task>
    // priority_queue是最大堆，使用TaskPtrCompare比较器
    // TaskPtrCompare返回*lhs < *rhs时，rhs会在顶部（优先级高的在顶部）
    using TaskQueue = std::priority_queue<std::shared_ptr<Task>, 
                                          std::vector<std::shared_ptr<Task>>, 
                                          TaskPtrCompare>;

    TaskQueue critical_queue_;  // CRITICAL优先级队列
    TaskQueue high_queue_;      // HIGH优先级队列
    TaskQueue normal_queue_;    // NORMAL优先级队列
    TaskQueue low_queue_;       // LOW优先级队列

    // 每队列独立锁（细粒度锁，减少 enqueue/dequeue 竞争）
    mutable std::mutex critical_mutex_;
    mutable std::mutex high_mutex_;
    mutable std::mutex normal_mutex_;
    mutable std::mutex low_mutex_;

    /** 从 shared_ptr<Task> 拷贝到 Task&，供 dequeue 复用 */
    static void copy_task_out(const std::shared_ptr<Task>& src, Task& out);
};

} // namespace executor
