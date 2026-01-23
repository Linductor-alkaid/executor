#pragma once

#include <thread>
#include <vector>
#include <cstdint>

namespace executor {
namespace util {

/**
 * @brief 设置线程优先级
 * 
 * @param handle 线程原生句柄
 * @param priority 优先级
 *                Linux: SCHED_FIFO优先级范围1-99，普通优先级范围-20到19
 *                Windows: THREAD_PRIORITY_* 常量（如THREAD_PRIORITY_NORMAL）
 * @return 成功返回true，失败返回false
 */
bool set_thread_priority(std::thread::native_handle_type handle, int priority);

/**
 * @brief 设置CPU亲和性
 * 
 * 将线程绑定到指定的CPU核心上。
 * 
 * @param handle 线程原生句柄
 * @param cpu_ids CPU核心ID列表（从0开始）
 * @return 成功返回true，失败返回false
 */
bool set_cpu_affinity(std::thread::native_handle_type handle,
                      const std::vector<int>& cpu_ids);

/**
 * @brief 获取当前线程的优先级
 * 
 * @return 当前线程的优先级值，失败返回0
 */
int get_current_thread_priority();

/**
 * @brief 获取当前线程的CPU亲和性
 * 
 * @return CPU核心ID列表，失败返回空列表
 */
std::vector<int> get_current_thread_affinity();

} // namespace util
} // namespace executor
