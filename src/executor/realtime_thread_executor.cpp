#include "realtime_thread_executor.hpp"
#include <stdexcept>
#include <algorithm>
#include <mutex>
#include <unordered_map>
#ifdef _WIN32
#include <windows.h>
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")
#endif

namespace executor {

RealtimeThreadExecutor::RealtimeThreadExecutor(const std::string& name, const RealtimeThreadConfig& config)
    : name_(name)
    , config_(config)
    , lockfree_queue_(1024)  // 默认队列容量为 1024
{
    // 验证配置
    if (config_.cycle_period_ns <= 0) {
        throw std::invalid_argument("cycle_period_ns must be greater than 0");
    }
    if (config_.thread_name.empty()) {
        throw std::invalid_argument("thread_name must not be empty");
    }
}

RealtimeThreadExecutor::~RealtimeThreadExecutor() {
    stop();
}

bool RealtimeThreadExecutor::start() {
    // 检查是否已在运行
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return false;  // 已经在运行
    }

    // 创建实时线程
    thread_ = std::thread([this]() {
#ifdef _WIN32
        // 在Windows上提高定时器精度（对于短周期很重要）
        // 将定时器精度设置为1ms（默认是15.6ms）
        // 注意：这会增加系统功耗，但提高定时精度
        if (config_.cycle_period_ns < 20000000) {  // 如果周期小于20ms
            timer_period_ms_ = 1;
            timeBeginPeriod(1);
        }
#endif
        
        // 设置线程优先级
        if (config_.thread_priority != 0) {
            util::set_thread_priority(thread_.native_handle(), config_.thread_priority);
        }

        // 设置 CPU 亲和性
        if (!config_.cpu_affinity.empty()) {
            util::set_cpu_affinity(thread_.native_handle(), config_.cpu_affinity);
        }

        // 如果提供了外部周期管理器，使用它进行精确周期控制
        if (config_.cycle_manager) {
            // 注册周期任务：回调函数是 cycle_loop()
            if (!config_.cycle_manager->register_cycle(
                    name_,
                    config_.cycle_period_ns,
                    [this]() { cycle_loop(); })) {
                // 注册失败，回退到内置实现
                simple_cycle_loop();
                return;
            }

            // 启动周期任务（阻塞在此，直到 stop_cycle 被调用）
            if (!config_.cycle_manager->start_cycle(name_)) {
                // 启动失败，回退到内置实现
                simple_cycle_loop();
                return;
            }
        } else {
            // 使用内置的简单周期实现
            simple_cycle_loop();
        }
    });

    return true;
}

void RealtimeThreadExecutor::stop() {
    // 设置停止标志
    bool expected = true;
    if (running_.compare_exchange_strong(expected, false)) {
        // 如果使用了周期管理器，需要先停止周期任务以解除阻塞
        if (config_.cycle_manager) {
            config_.cycle_manager->stop_cycle(name_);
        }

        // 等待线程结束
        if (thread_.joinable()) {
            thread_.join();
        }
        
#ifdef _WIN32
        // 恢复Windows定时器精度
        if (timer_period_ms_ > 0) {
            timeEndPeriod(timer_period_ms_);
            timer_period_ms_ = 0;
        }
#endif
    }
}

std::string RealtimeThreadExecutor::get_name() const {
    return name_;
}

void RealtimeThreadExecutor::push_task(std::function<void()> task) {
    if (task) {
        // 生成任务 ID
        uint64_t task_id = next_task_id_.fetch_add(1, std::memory_order_relaxed);
        
        // 将任务存储到 task_map_ 中
        {
            std::lock_guard<std::mutex> lock(task_map_mutex_);
            task_map_[task_id] = std::move(task);
        }
        
        // 尝试推送到无锁队列（推送任务 ID）
        // 如果队列满，任务会被丢弃（可以根据需要调整策略）
        if (!lockfree_queue_.push(task_id)) {
            // 队列满，从 task_map_ 中移除任务
            std::lock_guard<std::mutex> lock(task_map_mutex_);
            task_map_.erase(task_id);
        }
    }
}

void RealtimeThreadExecutor::simple_cycle_loop() {
    // 计算下一个周期时间点
    auto next_cycle_time = std::chrono::steady_clock::now();
    const auto period_ns = std::chrono::nanoseconds(config_.cycle_period_ns);

    while (running_.load(std::memory_order_acquire)) {
        // 记录周期开始时间
        auto cycle_start = std::chrono::steady_clock::now();

        // 执行周期回调函数
        if (config_.cycle_callback) {
            try {
                config_.cycle_callback();
            } catch (...) {
                // 捕获周期回调中的异常，防止影响周期执行
                exception_handler_.handle_task_exception(name_, std::current_exception());
            }
        }

        // 处理无锁队列中的任务
        process_tasks();

        // 记录周期结束时间
        auto cycle_end = std::chrono::steady_clock::now();
        auto cycle_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            cycle_end - cycle_start
        ).count();

        // 更新统计信息
        update_statistics(cycle_time_ns);

        // 计算下一个周期时间点
        next_cycle_time += period_ns;

        // 等待下一个周期（使用 sleep_until 实现精确周期控制）
        std::this_thread::sleep_until(next_cycle_time);
    }
}

void RealtimeThreadExecutor::cycle_loop() {
    // 记录周期开始时间
    auto cycle_start = std::chrono::steady_clock::now();

    // 执行周期回调函数
    if (config_.cycle_callback) {
        try {
            config_.cycle_callback();
        } catch (...) {
            exception_handler_.handle_task_exception(name_, std::current_exception());
        }
    }

    // 处理无锁队列中的任务
    process_tasks();

    // 记录周期结束时间并更新统计信息
    auto cycle_end = std::chrono::steady_clock::now();
    auto cycle_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        cycle_end - cycle_start
    ).count();
    update_statistics(cycle_time_ns);

    // 注意：不在这里执行 sleep，周期管理器负责等待下一个周期
}

void RealtimeThreadExecutor::process_tasks() {
    uint64_t task_id = 0;
    
    // 从无锁队列中弹出所有任务并执行
    // 注意：这里会处理队列中的所有任务，直到队列为空
    while (lockfree_queue_.pop(task_id)) {
        // 从 task_map_ 中获取任务
        std::function<void()> task;
        {
            std::lock_guard<std::mutex> lock(task_map_mutex_);
            auto it = task_map_.find(task_id);
            if (it != task_map_.end()) {
                task = std::move(it->second);
                task_map_.erase(it);
            }
        }
        
        // 执行任务
        if (task) {
            try {
                task();
            } catch (...) {
                // 捕获任务执行中的异常，不影响其他任务和周期执行
                exception_handler_.handle_task_exception(name_, std::current_exception());
            }
        }
    }
}

void RealtimeThreadExecutor::update_statistics(int64_t cycle_time_ns) {
    // 增加周期计数
    cycle_count_.fetch_add(1, std::memory_order_relaxed);

    // 更新平均周期时间（使用指数移动平均 EMA）
    const double alpha = 0.1;  // 平滑因子
    double old_avg = avg_cycle_time_ns_.load(std::memory_order_relaxed);
    double new_avg;
    
    if (old_avg == 0.0) {
        // 第一次更新，直接使用当前值
        new_avg = static_cast<double>(cycle_time_ns);
    } else {
        // 指数移动平均
        new_avg = alpha * static_cast<double>(cycle_time_ns) + (1.0 - alpha) * old_avg;
    }
    
    avg_cycle_time_ns_.store(new_avg, std::memory_order_relaxed);

    // 更新最大周期时间（使用 CAS 操作）
    double old_max = max_cycle_time_ns_.load(std::memory_order_relaxed);
    double new_max = static_cast<double>(cycle_time_ns);
    
    while (new_max > old_max) {
        if (max_cycle_time_ns_.compare_exchange_weak(old_max, new_max, 
                                                      std::memory_order_relaxed,
                                                      std::memory_order_relaxed)) {
            break;
        }
        // CAS 失败，重新读取 old_max
        old_max = max_cycle_time_ns_.load(std::memory_order_relaxed);
        new_max = static_cast<double>(cycle_time_ns);
    }

    // 检查是否超时（周期执行时间 > 周期时间）
    if (cycle_time_ns > config_.cycle_period_ns) {
        cycle_timeout_count_.fetch_add(1, std::memory_order_relaxed);
    }
}

RealtimeExecutorStatus RealtimeThreadExecutor::get_status() const {
    RealtimeExecutorStatus status;
    status.name = name_;
    status.is_running = running_.load(std::memory_order_acquire);
    status.cycle_period_ns = config_.cycle_period_ns;
    status.cycle_count = cycle_count_.load(std::memory_order_acquire);
    status.cycle_timeout_count = cycle_timeout_count_.load(std::memory_order_acquire);
    status.avg_cycle_time_ns = avg_cycle_time_ns_.load(std::memory_order_acquire);
    status.max_cycle_time_ns = max_cycle_time_ns_.load(std::memory_order_acquire);
    return status;
}

} // namespace executor
