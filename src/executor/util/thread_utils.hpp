#pragma once

#include <thread>
#include <vector>
#include <cstdint>
#include <string>

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

/**
 * @brief 把当前线程锁在物理内存里，防止分页到 swap
 *
 * 在 Linux 上调用 mlockall(MCL_CURRENT|MCL_FUTURE)，避免实时线程因缺页/换页
 * 引入毫秒级抖动。需要 CAP_IPC_LOCK 权限（或足够的 RLIMIT_MEMLOCK）。
 * 失败时（如无权限）静默返回 false，由调用方决定是否告警，不抛异常。
 *
 * @return 成功返回true，失败返回false（Windows 上始终返回false）
 */
bool try_mlock_current_thread();

/**
 * @brief 把当前线程名设进内核，便于 top/htop/perf 看到
 *
 * Linux 上线程名最长 15 字符（pthread_setname_np 限制），超出会自动截断。
 *
 * @param name 线程名
 */
void set_current_thread_name(const std::string& name);

/**
 * @brief 设置当前线程的 timer slack（纳秒）
 *
 * Linux 默认 timer slack 为 50us，会给定时唤醒带来额外抖动。设为 1 几乎无 slack。
 * Windows 上为空实现。
 *
 * @param slack_ns timer slack（纳秒）
 */
void set_current_thread_timer_slack_ns(uint64_t slack_ns);

} // namespace util
} // namespace executor
