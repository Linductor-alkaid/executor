#pragma once

#include <functional>
#include <thread>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace executor {

// Forward declaration
namespace util {
template<typename T> class LockFreeQueue;
template<typename T> class ObjectPool;
}

/**
 * @brief 无锁任务执行器
 *
 * 封装无锁队列和消费者线程，提供高性能任务提交接口。
 * 支持多个线程并发调用 push_task()，单个消费者线程处理任务（MPSC模式）。
 *
 * 使用场景：
 * - 高频日志收集
 * - 异步事件处理
 * - 性能敏感路径的任务分发
 * - 多线程环境下的任务聚合
 */
class LockFreeTaskExecutor {
public:
    /**
     * @brief 构造函数
     * @param queue_capacity 队列容量（必须是2的幂，如果不是会自动调整）
     * @param backoff_multiplier CAS 退避倍数（默认2，适合中等竞争场景）
     * @param enable_stats 是否启用性能统计（默认false）
     */
    explicit LockFreeTaskExecutor(size_t queue_capacity = 1024, size_t backoff_multiplier = 2, bool enable_stats = false);

    /**
     * @brief 析构函数（自动停止）
     */
    ~LockFreeTaskExecutor();

    // 禁止拷贝和移动
    LockFreeTaskExecutor(const LockFreeTaskExecutor&) = delete;
    LockFreeTaskExecutor& operator=(const LockFreeTaskExecutor&) = delete;

    /**
     * @brief 启动消费者线程
     * @return 成功返回true，已启动返回false
     */
    bool start();

    /**
     * @brief 停止消费者线程并等待
     */
    void stop();

    /**
     * @brief 检查是否正在运行
     */
    bool is_running() const;

    /**
     * @brief 提交任务到无锁队列（线程安全，支持多线程并发调用）
     * @param task 任务函数
     * @return 成功返回true，队列满返回false
     */
    bool push_task(std::function<void()> task);

    /**
     * @brief 批量提交任务
     * @param tasks 任务数组
     * @param count 任务数量
     * @param pushed 实际提交的任务数量（输出参数）
     * @return 成功返回true，队列满返回false
     */
    bool push_tasks_batch(const std::function<void()>* tasks, size_t count, size_t& pushed);

    /**
     * @brief 获取队列中待处理任务数（近似值）
     */
    size_t pending_count() const;

    /**
     * @brief 获取已处理任务总数
     */
    uint64_t processed_count() const;

    /**
     * @brief 获取队列性能统计
     */
    struct QueueStats {
        uint64_t total_pushes;
        uint64_t failed_pushes;
        uint64_t total_pops;
        uint64_t empty_pops;
        uint64_t batch_pushes;
        uint64_t batch_pops;
        uint64_t current_size;
        uint64_t peak_size;
        double success_rate;
    };
    QueueStats get_queue_stats() const;

private:
    struct TaskWrapper {
        std::function<void()> func;
    };

    void worker_thread();

    util::LockFreeQueue<TaskWrapper*>* queue_;
    util::ObjectPool<TaskWrapper>* task_pool_;

    std::thread worker_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> processed_count_{0};
};

} // namespace executor
