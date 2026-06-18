#include "realtime_thread_executor.hpp"
#include <stdexcept>
#include <algorithm>
#include <atomic>

// Round-robin CPU hint: incremented each time a new RT thread auto-selects its core.
// Wraps at hw_concurrency so threads spread across all available CPUs.
static std::atomic<unsigned> g_next_rt_cpu_hint{0};

#ifdef _WIN32
#include <windows.h>
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")
#endif

namespace executor {

RealtimeThreadExecutor::RealtimeThreadExecutor(const std::string& name,
                                             const RealtimeThreadConfig& config,
                                             bool enable_stats,
                                             size_t queue_capacity)
    : name_(name)
    , config_(config)
    , lockfree_queue_(queue_capacity, 1, enable_stats)
    , task_pool_(queue_capacity)
    , enable_stats_(enable_stats)
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
            timer_period_ms_.store(1, std::memory_order_relaxed);
            timeBeginPeriod(1);
        }
#endif
        
        // 自适应 priority: 用户未显式设 (== 0) 时, 按周期建议
        // 使用 pthread_self()/GetCurrentThread() 而非 thread_.native_handle()，
        // 避免与主线程的 thread_ move-assign 产生 data race.
#ifdef _WIN32
        auto self_handle = static_cast<std::thread::native_handle_type>(GetCurrentThread());
#else
        auto self_handle = pthread_self();
#endif
        if (config_.thread_priority == 0 && config_.cycle_period_ns > 0) {
            int auto_priority = 0;
            if (config_.cycle_period_ns <= 1'000'000) {        // <= 1ms
                auto_priority = 80;  // 硬实时, 短周期
            } else if (config_.cycle_period_ns <= 10'000'000) {  // <= 10ms
                auto_priority = 50;  // 软实时, 中周期
            }  // > 10ms 保持 0 (普通调度够用)
            if (auto_priority > 0) {
                util::set_thread_priority(self_handle, auto_priority);
            }
        } else if (config_.thread_priority != 0) {
            // 用户显式设了, 尊重覆盖
            util::set_thread_priority(self_handle, config_.thread_priority);
        }

        // CPU 亲和性: 空 = 自适应 sentinel, 用 round-robin 跨核分布; 显式设值尊重覆盖
        if (config_.cpu_affinity.empty()) {
            unsigned hw = std::thread::hardware_concurrency();
            if (hw >= 2) {
                // P-005 round-robin: 每个新 RT 线程取下一个 CPU 核号, 避免全部挤在核 0.
                // hw == 1 或探测失败(0) 时不绑, 避免争抢唯一核心. 多 rt 线程场景用户可显式覆盖 cpu_affinity.
                unsigned cpu = g_next_rt_cpu_hint.fetch_add(1, std::memory_order_relaxed) % hw;
                util::set_cpu_affinity(self_handle, {static_cast<int>(cpu)});
            }
        } else {
            util::set_cpu_affinity(self_handle, config_.cpu_affinity);
        }

        // 锁定内存，避免分页到 swap 引入抖动（默认开启，失败静默回退；用户可显式设 false 关闭）
        if (config_.enable_memory_lock) {
            util::try_mlock_current_thread();
        }

        // 设置线程名，便于 top/htop/perf 调试
        if (!config_.thread_name.empty()) {
            util::set_current_thread_name(config_.thread_name);
        }

        // 降低 timer slack，减少定时唤醒抖动（默认 1ns；0 为显式 opt-out，保留内核默认）
        if (config_.timer_slack_ns > 0) {
            util::set_current_thread_timer_slack_ns(config_.timer_slack_ns);
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
        if (timer_period_ms_.load(std::memory_order_relaxed) > 0) {
            timeEndPeriod(timer_period_ms_.load(std::memory_order_relaxed));
            timer_period_ms_.store(0, std::memory_order_relaxed);
        }
#endif
    }
}

std::string RealtimeThreadExecutor::get_name() const {
    return name_;
}

void RealtimeThreadExecutor::push_task(std::function<void()> task) {
    // P-001 (260615): 保留 void 接口, 调用方应改用 push_task_ex 或 get_status().
    (void)push_task_ex(std::move(task));
}

bool RealtimeThreadExecutor::push_task_ex(std::function<void()> task) {
    // P-001 (260615): 三条失败路径全部计入 dropped_task_count_ —
    //   (1) task 为空 (无效输入)
    //   (2) 对象池耗尽 (task_pool_.acquire() == nullptr)
    //   (3) 队列满 (lockfree_queue_.push() == false)
    // 此计数器独立于 enable_stats, 是背压可见性的核心契约.
    if (!task) {
        dropped_task_count_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    // 从对象池获取任务对象
    TaskWrapper* task_wrapper = task_pool_.acquire();
    if (!task_wrapper) {
        // 对象池耗尽: 任务被静默丢弃
        dropped_task_count_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    task_wrapper->func = std::move(task);

    // 推送到无锁队列
    if (!lockfree_queue_.push(task_wrapper)) {
        // 队列满: 释放回对象池, 任务被静默丢弃
        task_pool_.release(task_wrapper);
        dropped_task_count_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    return true;
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

        // P-260618-004: skip-late. If a callback ran long and next_cycle_time
        // has fallen into the past, sleeping until it would return immediately
        // and we'd burn through every missed phase with zero sleep. Instead,
        // re-phase to "now + period" so the thread guarantees at least one
        // real sleep and a fresh period. Documented behavior is "skip
        // missed phases" rather than "catch up" — the latter produced
        // jitter storms under load.
        {
            const auto now = std::chrono::steady_clock::now();
            if (now > next_cycle_time) {
                next_cycle_time = now + period_ns;
            }
        }

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
    TaskWrapper* task_wrapper = nullptr;

    // P-260618-002: 单周期任务预算. 每周期最多处理 max_tasks_per_cycle 个任务,
    // 避免生产速率短暂超过消费速率时单周期一口气耗尽整条队列, 打破"周期确定性"契约
    // (cycle_time 爆涨 / cycle_timeout_count 尖刺). 剩余任务自然滚到下一周期处理
    // (MPSC 无锁队列, 无需额外锁). max_tasks_per_cycle == 0 表示不限 (保留旧行为,
    // 向后兼容) — 此时循环条件第一段恒真, 仅靠 pop() 返回 false 退出.
    const uint64_t budget = config_.max_tasks_per_cycle;
    for (uint64_t processed = 0; budget == 0 || processed < budget; ++processed) {
        if (!lockfree_queue_.pop(task_wrapper)) {
            break;  // 队列为空
        }

        if (task_wrapper && task_wrapper->func) {
            try {
                task_wrapper->func();
            } catch (...) {
                exception_handler_.handle_task_exception(name_, std::current_exception());
            }
        }

        // 释放回对象池
        task_pool_.release(task_wrapper);
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
    // P-001 (260615): 背压可见性
    status.dropped_task_count = dropped_task_count_.load(std::memory_order_acquire);
    // failed_pushes / peak_queue_size 仅在 enable_stats=true 时有意义;
    // LockFreeQueue 内部在 stats 关闭时 get_stats() 返回零结构.
    if (enable_stats_) {
        auto qstats = lockfree_queue_.get_stats();
        status.failed_pushes = qstats.failed_pushes;
        status.peak_queue_size = qstats.peak_size;
    }
    // queue_capacity 永远等于构造时的固定容量; 暴露以方便比率分析
    // (dropped / capacity 间接表达"队列满导致丢弃"的最小次数, 因对象池
    //  容量 == 队列容量, 池耗尽与队列满通常同时发生).
    status.queue_capacity = lockfree_queue_.capacity();
    return status;
}

} // namespace executor
