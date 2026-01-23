#include "executor/types.hpp"
#include <chrono>

namespace executor {

/**
 * @brief Task 比较操作符（用于优先级队列）
 * 
 * 优先级高的任务优先（CRITICAL > HIGH > NORMAL > LOW）
 * 同优先级时，提交时间早的任务优先
 */
bool operator<(const Task& lhs, const Task& rhs) {
    // 优先级高的优先（数值大的优先）
    if (lhs.priority != rhs.priority) {
        return static_cast<int>(lhs.priority) < static_cast<int>(rhs.priority);
    }
    // 同优先级，提交时间早的优先（submit_time_ns 小的优先）
    return lhs.submit_time_ns > rhs.submit_time_ns;
}

/**
 * @brief Task 大于比较操作符
 */
bool operator>(const Task& lhs, const Task& rhs) {
    return rhs < lhs;
}

/**
 * @brief 创建任务ID（辅助函数）
 * 
 * 基于时间戳生成唯一的任务ID
 */
std::string generate_task_id() {
    auto now = std::chrono::steady_clock::now();
    auto duration = now.time_since_epoch();
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    return "task_" + std::to_string(nanoseconds);
}

/**
 * @brief 检查任务是否已取消
 */
bool is_task_cancelled(const Task& task) {
    return task.cancelled.load(std::memory_order_acquire);
}

/**
 * @brief 取消任务
 */
void cancel_task(Task& task) {
    task.cancelled.store(true, std::memory_order_release);
}

} // namespace executor
