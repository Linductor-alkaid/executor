#pragma once

#include "executor/types.hpp"
#include <string>

namespace executor {

/**
 * @brief Task 比较操作符（用于优先级队列）
 * 
 * 优先级高的任务优先（CRITICAL > HIGH > NORMAL > LOW）
 * 同优先级时，提交时间早的任务优先
 */
bool operator<(const Task& lhs, const Task& rhs);

/**
 * @brief Task 大于比较操作符
 */
bool operator>(const Task& lhs, const Task& rhs);

/**
 * @brief 创建任务ID（辅助函数）
 * 
 * 基于时间戳生成唯一的任务ID
 * 
 * @return 唯一的任务ID字符串
 */
std::string generate_task_id();

/**
 * @brief 检查任务是否已取消
 * 
 * @param task 任务对象
 * @return 如果任务已取消，返回 true
 */
bool is_task_cancelled(const Task& task);

/**
 * @brief 取消任务
 * 
 * @param task 任务对象
 */
void cancel_task(Task& task);

} // namespace executor
