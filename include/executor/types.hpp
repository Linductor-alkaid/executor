#pragma once

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>
#include <atomic>
#include <functional>
#include <map>

namespace executor {

inline constexpr std::chrono::seconds kDefaultWaitForCompletionTimeout{300};

class TimedOutException : public std::runtime_error {
public:
    explicit TimedOutException(const std::string& message)
        : std::runtime_error(message) {}
};

/**
 * @brief 任务优先级枚举
 */
enum class TaskPriority {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    CRITICAL = 3
};

/**
 * @brief Executor facade 统一失败事件类型
 *
 * 自动调优类失败允许安全回退；任务运行、提交拒绝、超时等状态必须通过事件、
 * 状态计数或 future 保持可观察。
 */
enum class FailureKind {
    TaskException,     // 用户任务抛异常
    SubmitRejected,    // 任务提交被拒绝
    TaskTimeout,       // 任务执行超时
    RealtimeDrop,      // 实时队列丢任务/推送失败
    GpuFailure,        // GPU 执行失败
    WaitTimeout,       // 等待完成超时
    TuningFallback     // 平台调优失败并安全回退
};

/**
 * @brief Executor facade 失败事件
 */
struct ExecutorFailureEvent {
    FailureKind kind = FailureKind::TaskException;
    std::string executor_name;
    std::string task_id;
    std::string message;
    std::exception_ptr exception;
    std::chrono::steady_clock::time_point timestamp =
        std::chrono::steady_clock::now();
};

/**
 * @brief Executor facade 累计失败状态
 */
struct ExecutorFailureStatus {
    uint64_t task_exception_count = 0;
    uint64_t submit_rejected_count = 0;
    uint64_t timeout_count = 0;
    uint64_t realtime_drop_count = 0;
    uint64_t gpu_failure_count = 0;
    uint64_t wait_timeout_count = 0;
    uint64_t tuning_fallback_count = 0;
    uint64_t total_count = 0;
};

/**
 * @brief Executor facade 周期任务状态
 */
struct PeriodicTaskStatus {
    std::string task_id;
    int64_t period_ms = 0;
    bool is_running = false;
    uint64_t execution_count = 0;
    uint64_t failed_count = 0;
    uint64_t consecutive_failure_count = 0;
    std::string last_error_message;
    std::chrono::steady_clock::time_point next_execute_time{};
    std::chrono::steady_clock::time_point last_failure_time{};
};

using ExecutorFailureCallback = std::function<void(const ExecutorFailureEvent&)>;

/**
 * @brief 任务结构体
 */
struct Task {
    std::string task_id;                          // 任务ID
    TaskPriority priority = TaskPriority::NORMAL; // 任务优先级
    std::function<void()> function;              // 任务函数
    std::function<void(std::exception_ptr)> on_timeout; // 软超时时显式满足 future
    int64_t submit_time_ns = 0;                   // 提交时间（纳秒）
    int64_t timeout_ms = 0;                       // 超时时间（毫秒），0表示不超时
    std::vector<std::string> dependencies;       // 依赖任务ID
    std::atomic<bool> cancelled{false};           // 取消标志
};

/**
 * @brief 异步执行器状态结构
 */
struct AsyncExecutorStatus {
    std::string name;                             // 执行器名称
    bool is_running = false;                      // 是否运行中
    size_t active_tasks = 0;                      // 活跃任务数
    size_t completed_tasks = 0;                   // 已完成任务数
    size_t failed_tasks = 0;                      // 失败任务数
    size_t queue_size = 0;                        // 队列大小
    double avg_task_time_ms = 0.0;                // 平均任务执行时间（毫秒）
};

/**
 * @brief 实时执行器状态结构
 *
 * P-001 (260615): 新增 dropped_task_count / failed_pushes / peak_queue_size /
 * queue_capacity 字段，承载背压下静默丢任务的可见性。
 * - dropped_task_count:  push_task() 因 未运行、空任务、队列满 或 对象池耗尽
 *                       而被丢弃的累计数
 *                       (即使 enable_stats=false 也会累计, 是背压可见性的核心指标)
 * - failed_pushes:       仅 enable_stats=true 时由 LockFreeQueue 统计的入队失败数
 *                       (== 队列满 触发的丢弃, 与 dropped_task_count 的子集)
 * - peak_queue_size:     仅 enable_stats=true 时由 LockFreeQueue 统计的峰值队列长度
 * - queue_capacity:      RT 无锁队列的固定容量 (>= dropped 阈值), 用于比率分析
 */
struct RealtimeExecutorStatus {
    std::string name;                             // 执行器名称
    bool is_running = false;                      // 是否运行中
    int64_t cycle_period_ns = 0;                  // 周期（纳秒）
    int64_t cycle_count = 0;                      // 周期计数
    int64_t cycle_timeout_count = 0;              // 周期超时计数
    double avg_cycle_time_ns = 0.0;                // 平均周期执行时间（纳秒）
    double max_cycle_time_ns = 0.0;               // 最大周期执行时间（纳秒）
    // P-001 (260615): 背压可见性字段
    uint64_t dropped_task_count = 0;              // 累计丢任务数 (未运行+空任务+队列满+池耗尽, 始终累计)
    uint64_t failed_pushes = 0;                   // LockFreeQueue 失败入队数 (仅 enable_stats=true)
    uint64_t peak_queue_size = 0;                 // 队列峰值 (仅 enable_stats=true)
    uint64_t queue_capacity = 0;                  // 队列固定容量
    uint64_t rejected_not_running_count = 0;      // 未运行/已停止时拒绝的累计数
    uint64_t rejected_empty_task_count = 0;       // 空任务拒绝累计数
    uint64_t pool_exhausted_count = 0;            // 对象池耗尽拒绝累计数
    uint64_t queue_full_count = 0;                // 队列满拒绝累计数
};

/**
 * @brief 任务统计信息（用于监控）
 */
struct TaskStatistics {
    int64_t total_count = 0;                     // 总任务数
    int64_t success_count = 0;                    // 成功任务数
    int64_t fail_count = 0;                       // 失败任务数
    int64_t timeout_count = 0;                    // 超时任务数
    int64_t total_execution_time_ns = 0;          // 总执行时间（纳秒）
    int64_t max_execution_time_ns = 0;            // 最大执行时间（纳秒）
    int64_t min_execution_time_ns = 0;            // 最小执行时间（纳秒）
};

/**
 * @brief 周期统计信息（用于ICycleManager）
 */
struct CycleStatistics {
    std::string name;                             // 周期任务名称
    int64_t period_ns = 0;                        // 周期（纳秒）
    int64_t cycle_count = 0;                      // 周期计数
    int64_t timeout_count = 0;                    // 超时计数
    double avg_cycle_time_ns = 0.0;               // 平均周期执行时间（纳秒）
    double max_cycle_time_ns = 0.0;               // 最大周期执行时间（纳秒）
    bool is_running = false;                      // 是否运行中
};

/**
 * @brief GPU 相关类型定义
 */
namespace gpu {

/**
 * @brief GPU 后端类型
 */
enum class GpuBackend {
    CUDA,      // NVIDIA CUDA
    OPENCL,    // OpenCL (跨平台)
    SYCL,      // SYCL (Intel/AMD)
    HIP        // AMD ROCm
};

/**
 * @brief GPU 设备信息
 */
struct GpuDeviceInfo {
    std::string name;                             // 设备名称
    GpuBackend backend;                            // 后端类型
    int device_id = 0;                             // 设备ID
    std::string vendor;                            // 厂商（NVIDIA/AMD/Intel）
    size_t total_memory_bytes = 0;                // 总内存（字节）
    size_t free_memory_bytes = 0;                 // 空闲内存（字节）
    int compute_capability_major = 0;              // 计算能力主版本（CUDA）
    int compute_capability_minor = 0;              // 计算能力次版本（CUDA）
    size_t max_threads_per_block = 0;              // 每个Block最大线程数
    size_t max_blocks_per_grid[3] = {0, 0, 0};    // Grid最大维度
};

/**
 * @brief GPU 执行器状态
 */
struct GpuExecutorStatus {
    std::string name;                             // 执行器名称
    bool is_running = false;                      // 是否运行中
    GpuBackend backend;                            // 后端类型
    int device_id = 0;                             // 设备ID
    size_t active_kernels = 0;                    // 活跃kernel数
    size_t completed_kernels = 0;                 // 已完成kernel数
    size_t failed_kernels = 0;                    // 失败kernel数
    size_t queue_size = 0;                        // 任务队列大小
    double avg_kernel_time_ms = 0.0;              // 平均kernel执行时间（毫秒）
    size_t memory_used_bytes = 0;                  // 已使用内存（字节）
    size_t memory_total_bytes = 0;                 // 总内存（字节）
    double memory_usage_percent = 0.0;            // 内存使用率
    std::string last_error_message;                // 最近一次启动/运行失败原因（空表示无错误）
};

/**
 * @brief GPU 任务配置
 */
struct GpuTaskConfig {
    size_t grid_size[3] = {1, 1, 1};              // Grid维度
    size_t block_size[3] = {1, 1, 1};            // Block维度
    size_t shared_memory_bytes = 0;               // 共享内存大小（字节）
    int stream_id = 0;                            // 流ID（0表示默认流）
    bool async = true;                            // 是否异步执行
    int priority = 1;                             // 优先级（0=LOW, 1=NORMAL, 2=HIGH, 3=CRITICAL），与 CPU submit_priority 对齐
};

} // namespace gpu

} // namespace executor
