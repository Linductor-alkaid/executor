#pragma once

#include "executor/types.hpp"
#include "../task/task.hpp"
#include <vector>
#include <algorithm>
#include <memory>
#include <mutex>
#include <cstddef>

namespace executor {

/**
 * @brief 优先级调度器
 * 
 * 管理4个优先级队列（CRITICAL, HIGH, NORMAL, LOW），
 * 提供线程安全的enqueue和dequeue接口。
 * 按优先级顺序调度任务，同优先级任务按提交时间（FIFO）调度。
 * 
 * 使用std::vector<std::unique_ptr<Task>> + 堆操作存储任务，
 * 相比shared_ptr减少引用计数开销和控制块内存分配。
 * Task包含std::atomic<bool>不可复制和移动，因此使用unique_ptr管理。
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
     * @param task 任务对象（会被复制为unique_ptr）
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
     * @brief 批量从优先级队列获取任务（按优先级顺序）
     * 
     * 按优先级从高到低依次从各队列取任务，每个优先级队列仅加锁一次，
     * 最多取出 max_tasks 个任务。调用方必须保证 out 指向至少 max_tasks 个
     * 已构造的 Task 对象（如 vector::resize 后传入 data()）。
     * 
     * @param out 用于接收任务的缓冲区，写入 out[0..return-1]
     * @param max_tasks 最多取出的任务数
     * @return 实际取出的任务数
     */
    size_t dequeue_batch(Task* out, size_t max_tasks);

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
    // unique_ptr比较器：比较Task对象（通过解引用）
    // 使用std::greater确保优先级高的任务在堆顶
    struct TaskPtrCompare {
        bool operator()(const std::unique_ptr<Task>& lhs, const std::unique_ptr<Task>& rhs) const {
            if (!lhs || !rhs) return false;
            // Task::operator< 返回true表示lhs优先级低于rhs
            // 对于std::greater，返回*lhs < *rhs时，rhs（优先级高）会在堆顶
            return *lhs < *rhs;
        }
    };

    // 使用vector存储unique_ptr<Task>，配合堆操作维护优先级顺序
    using TaskQueue = std::vector<std::unique_ptr<Task>>;

    TaskQueue critical_queue_;  // CRITICAL优先级队列
    TaskQueue high_queue_;      // HIGH优先级队列
    TaskQueue normal_queue_;    // NORMAL优先级队列
    TaskQueue low_queue_;       // LOW优先级队列

    // 每队列独立锁（细粒度锁，减少 enqueue/dequeue 竞争）
    mutable std::mutex critical_mutex_;
    mutable std::mutex high_mutex_;
    mutable std::mutex normal_mutex_;
    mutable std::mutex low_mutex_;

    /** 从 unique_ptr<Task> 拷贝到 Task&，供 dequeue 复用 */
    static void copy_task_out(const std::unique_ptr<Task>& src, Task& out);
};

} // namespace executor
