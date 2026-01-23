#include "thread_pool_resizer.hpp"
#include "thread_pool.hpp"
#include <algorithm>
#include <chrono>

namespace executor {

ThreadPoolResizer::ThreadPoolResizer(ThreadPool& pool, const ThreadPoolConfig& config)
    : pool_(pool)
    , config_(config)
    , last_resize_time_(std::chrono::steady_clock::now())
    , last_idle_start_time_(std::chrono::steady_clock::now()) {
}

void ThreadPoolResizer::check_and_resize() {
    if (!enabled_.load()) {
        return;
    }
    
    // 防止频繁扩缩容，至少间隔1秒
    auto now = std::chrono::steady_clock::now();
    auto time_since_last_resize = std::chrono::duration_cast<std::chrono::seconds>(
        now - last_resize_time_
    ).count();
    
    if (time_since_last_resize < 1) {
        return;
    }
    
    size_t current_threads = total_threads_.load();
    size_t queue_size = queue_size_.load();
    double avg_wait_time = avg_wait_time_ms_.load();
    
    // 检查是否需要扩容
    if (should_expand()) {
        size_t max_threads = config_.max_threads;
        if (current_threads < max_threads) {
            size_t threads_to_add = std::min(size_t(1), max_threads - current_threads);
            expand(threads_to_add);
            last_resize_time_ = now;
            return;
        }
    }
    
    // 检查是否需要缩容
    if (should_shrink()) {
        size_t min_threads = config_.min_threads;
        if (current_threads > min_threads) {
            size_t threads_to_remove = std::min(size_t(1), current_threads - min_threads);
            shrink(threads_to_remove);
            last_resize_time_ = now;
        }
    }
}

bool ThreadPoolResizer::should_expand() const {
    size_t queue_size = queue_size_.load();
    double avg_wait_time = avg_wait_time_ms_.load();
    size_t current_threads = total_threads_.load();
    size_t max_threads = config_.max_threads;
    
    // 扩容条件（需要同时满足）：
    // 1. 队列中任务数 > 队列容量的 80%
    // 2. 平均任务等待时间 > 100ms
    // 3. 当前线程数 < max_threads
    
    bool queue_high = (queue_size > config_.queue_capacity * 0.8);
    bool wait_time_high = (avg_wait_time > 100.0);
    bool can_expand = (current_threads < max_threads);
    
    return queue_high && wait_time_high && can_expand;
}

bool ThreadPoolResizer::should_shrink() {
    size_t queue_size = queue_size_.load();
    size_t active_threads = active_threads_.load();
    size_t total_threads = total_threads_.load();
    size_t min_threads = config_.min_threads;
    
    // 缩容条件（需要同时满足）：
    // 1. 空闲线程数 > 总线程数的 50%
    // 2. 队列中任务数 < 队列容量的 20%
    // 3. 持续空闲时间 > 60秒
    // 4. 当前线程数 > min_threads
    
    size_t idle_threads = total_threads - active_threads;
    bool idle_high = (idle_threads > total_threads * 0.5);
    bool queue_low = (queue_size < config_.queue_capacity * 0.2);
    bool can_shrink = (total_threads > min_threads);
    
    // 检查持续空闲时间
    auto now = std::chrono::steady_clock::now();
    auto idle_duration = std::chrono::duration_cast<std::chrono::seconds>(
        now - last_idle_start_time_
    ).count();
    bool idle_long_enough = (idle_duration > 60);
    
    // 更新空闲状态
    if (idle_high && queue_low) {
        if (!is_idle_.load()) {
            is_idle_.store(true);
            last_idle_start_time_ = now;
        }
    } else {
        is_idle_.store(false);
    }
    
    return idle_high && queue_low && idle_long_enough && can_shrink;
}

bool ThreadPoolResizer::expand(size_t num_threads) {
    // 实际扩容逻辑在 ThreadPool 中实现
    // 这里只是接口，ThreadPool 需要提供 expand_threads 方法
    // 暂时返回 true，实际实现需要在 ThreadPool 中完成
    return true;
}

bool ThreadPoolResizer::shrink(size_t num_threads) {
    // 实际缩容逻辑在 ThreadPool 中实现
    // 这里只是接口，ThreadPool 需要提供 shrink_threads 方法
    // 暂时返回 true，实际实现需要在 ThreadPool 中完成
    return true;
}

void ThreadPoolResizer::set_enabled(bool enabled) {
    enabled_.store(enabled);
}

bool ThreadPoolResizer::is_enabled() const {
    return enabled_.load();
}

void ThreadPoolResizer::update_status(size_t queue_size,
                                      size_t active_threads,
                                      size_t total_threads,
                                      double avg_wait_time_ms) {
    queue_size_.store(queue_size);
    active_threads_.store(active_threads);
    total_threads_.store(total_threads);
    avg_wait_time_ms_.store(avg_wait_time_ms);
}

void ThreadPoolResizer::mark_thread_for_exit(size_t thread_id) {
    std::lock_guard<std::mutex> lock(exit_threads_mutex_);
    exit_threads_.push_back(thread_id);
}

} // namespace executor
