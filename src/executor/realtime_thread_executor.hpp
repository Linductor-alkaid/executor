#pragma once

#include "executor/interfaces.hpp"
#include "executor/config.hpp"
#include "executor/types.hpp"
#include "util/lockfree_queue.hpp"
#include "util/exception_handler.hpp"
#include "util/thread_utils.hpp"
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <functional>
#include <unordered_map>
#include <mutex>
#include <cstdint>

namespace executor {

/**
 * @brief 实时线程执行器
 * 
 * 实现 IRealtimeExecutor 接口，提供专用实时线程执行功能。
 * 用于处理高实时性任务（如实时通信、传感器采集），支持：
 * - 高优先级线程
 * - CPU 亲和性设置
 * - 精确周期控制（使用 std::this_thread::sleep_until）
 * - 无锁队列任务传递
 * - 周期统计和监控
 * - 异常处理
 */
class RealtimeThreadExecutor : public IRealtimeExecutor {
public:
    /**
     * @brief 构造函数
     * 
     * @param name 执行器名称
     * @param config 实时线程配置
     */
    RealtimeThreadExecutor(const std::string& name, const RealtimeThreadConfig& config);

    /**
     * @brief 析构函数
     * 
     * 自动停止执行器并等待线程结束
     */
    ~RealtimeThreadExecutor();

    // 禁止拷贝和移动
    RealtimeThreadExecutor(const RealtimeThreadExecutor&) = delete;
    RealtimeThreadExecutor& operator=(const RealtimeThreadExecutor&) = delete;
    RealtimeThreadExecutor(RealtimeThreadExecutor&&) = delete;
    RealtimeThreadExecutor& operator=(RealtimeThreadExecutor&&) = delete;

    /**
     * @brief 启动实时线程
     * 
     * 创建实时线程，设置优先级和 CPU 亲和性，开始周期循环。
     * 
     * @return 如果启动成功返回 true，否则返回 false
     */
    bool start() override;

    /**
     * @brief 停止实时线程
     * 
     * 停止周期循环并等待线程结束。
     */
    void stop() override;

    /**
     * @brief 推送任务到无锁队列（在周期回调中处理）
     * 
     * 任务通过无锁队列传递，在实时线程的下一个周期回调中执行。
     * 
     * @param task 任务函数
     */
    void push_task(std::function<void()> task) override;

    /**
     * @brief 获取执行器名称
     * @return 执行器名称
     */
    std::string get_name() const override;

    /**
     * @brief 获取执行器状态
     * @return 实时执行器状态
     */
    RealtimeExecutorStatus get_status() const override;

private:
    /**
     * @brief 简单周期循环（内置默认实现）
     * 
     * 使用 std::this_thread::sleep_until 实现精确周期控制。
     * 在每个周期中：
     * 1. 执行周期回调函数
     * 2. 处理无锁队列中的任务
     * 3. 更新统计信息
     * 4. 等待下一个周期
     */
    void simple_cycle_loop();

    /**
     * @brief 处理无锁队列中的任务
     * 
     * 从无锁队列中弹出所有任务并执行，捕获异常。
     */
    void process_tasks();

    /**
     * @brief 更新周期统计信息
     * 
     * @param cycle_time_ns 当前周期执行时间（纳秒）
     */
    void update_statistics(int64_t cycle_time_ns);

    std::string name_;                              // 执行器名称
    RealtimeThreadConfig config_;                   // 实时线程配置
    std::thread thread_;                            // 实时线程
    std::atomic<bool> running_{false};              // 运行状态标志
    
    // 无锁队列（用于任务传递）
    // 注意：std::function 不是 trivially copyable，所以使用任务 ID（uint64_t）代替
    // 任务实际存储在 task_map_ 中
    util::LockFreeQueue<uint64_t> lockfree_queue_;
    
    // 任务存储（使用 map 存储任务，key 为任务 ID）
    std::unordered_map<uint64_t, std::function<void()>> task_map_;
    std::mutex task_map_mutex_;
    std::atomic<uint64_t> next_task_id_{1};  // 下一个任务 ID
    
    // 异常处理器
    util::ExceptionHandler exception_handler_;
    
    // 统计信息（使用原子变量支持并发访问）
    std::atomic<int64_t> cycle_count_{0};          // 周期计数
    std::atomic<int64_t> cycle_timeout_count_{0};   // 周期超时计数
    std::atomic<double> avg_cycle_time_ns_{0.0};    // 平均周期执行时间（纳秒）
    std::atomic<double> max_cycle_time_ns_{0.0};   // 最大周期执行时间（纳秒）
};

} // namespace executor
