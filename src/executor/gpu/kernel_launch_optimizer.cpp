#include "executor/gpu/kernel_launch_optimizer.hpp"
#include <algorithm>

namespace executor {
namespace gpu {

KernelLaunchOptimizer::KernelLaunchOptimizer() : config_() {}

KernelLaunchOptimizer::KernelLaunchOptimizer(const Config& config) : config_(config) {}

// --- 参数缓存 ---

bool KernelLaunchOptimizer::lookup_params(const std::string& kernel_name,
                                           KernelParamCacheEntry& out) {
    if (!config_.enable_param_cache) {
        return false;
    }

    std::unique_lock lock(cache_mutex_);
    auto it = param_cache_.find(kernel_name);
    if (it == param_cache_.end()) {
        std::lock_guard slock(stats_mutex_);
        ++cache_misses_;
        return false;
    }

    it->second.last_access_ns =
        std::chrono::steady_clock::now().time_since_epoch().count();
    ++it->second.hit_count;
    out = it->second;

    {
        std::lock_guard slock(stats_mutex_);
        ++cache_hits_;
    }
    return true;
}

void KernelLaunchOptimizer::store_params(const std::string& kernel_name,
                                          const KernelParamCacheEntry& entry) {
    if (!config_.enable_param_cache) {
        return;
    }

    std::unique_lock lock(cache_mutex_);
    evict_lru_if_needed();

    auto& cached = param_cache_[kernel_name];
    cached = entry;
    cached.last_access_ns =
        std::chrono::steady_clock::now().time_since_epoch().count();
}

void KernelLaunchOptimizer::invalidate_params(const std::string& kernel_name) {
    std::unique_lock lock(cache_mutex_);
    param_cache_.erase(kernel_name);
}

void KernelLaunchOptimizer::clear_cache() {
    std::unique_lock lock(cache_mutex_);
    param_cache_.clear();
}

size_t KernelLaunchOptimizer::cache_size() const {
    std::shared_lock lock(cache_mutex_);
    return param_cache_.size();
}

void KernelLaunchOptimizer::evict_lru_if_needed() {
    if (param_cache_.size() < config_.max_cache_entries) {
        return;
    }

    auto oldest = param_cache_.begin();
    for (auto it = param_cache_.begin(); it != param_cache_.end(); ++it) {
        if (it->second.last_access_ns < oldest->second.last_access_ns) {
            oldest = it;
        }
    }
    param_cache_.erase(oldest);
}

// --- 批量化 ---

size_t KernelLaunchOptimizer::enqueue(BatchedKernelRequest request) {
    if (!config_.enable_batching) {
        return 0;
    }

    std::lock_guard lock(batch_mutex_);
    if (batch_queue_.empty()) {
        batch_window_start_ns_ =
            std::chrono::steady_clock::now().time_since_epoch().count();
    }
    batch_queue_.push_back(std::move(request));
    return batch_queue_.size();
}

std::vector<BatchedKernelRequest> KernelLaunchOptimizer::flush_if_ready() {
    if (!config_.enable_batching) {
        return {};
    }

    std::lock_guard lock(batch_mutex_);
    if (batch_queue_.empty()) {
        return {};
    }

    bool should_flush = false;
    if (batch_queue_.size() >= config_.batch_threshold) {
        should_flush = true;
    } else {
        auto now_ns = std::chrono::steady_clock::now().time_since_epoch().count();
        auto elapsed_us = (now_ns - batch_window_start_ns_) / 1000;
        if (elapsed_us >= config_.batch_window_us) {
            should_flush = true;
        }
    }

    if (!should_flush) {
        return {};
    }

    std::vector<BatchedKernelRequest> result(batch_queue_.begin(), batch_queue_.end());
    batch_queue_.clear();

    {
        std::lock_guard slock(stats_mutex_);
        ++batched_launches_;
    }

    return result;
}

std::vector<BatchedKernelRequest> KernelLaunchOptimizer::flush_all() {
    std::lock_guard lock(batch_mutex_);
    if (batch_queue_.empty()) {
        return {};
    }

    std::vector<BatchedKernelRequest> result(batch_queue_.begin(), batch_queue_.end());
    batch_queue_.clear();

    if (config_.enable_batching) {
        std::lock_guard slock(stats_mutex_);
        ++batched_launches_;
    }

    return result;
}

size_t KernelLaunchOptimizer::pending_count() const {
    std::lock_guard lock(batch_mutex_);
    return batch_queue_.size();
}

// --- 延迟跟踪 ---

void KernelLaunchOptimizer::record_launch_latency(double latency_us) {
    if (!config_.track_latency) {
        return;
    }

    std::lock_guard lock(stats_mutex_);
    total_latency_us_ += latency_us;
    ++total_launches_;
    min_latency_us_ = std::min(min_latency_us_, latency_us);
    max_latency_us_ = std::max(max_latency_us_, latency_us);
}

KernelLaunchStats KernelLaunchOptimizer::get_stats() const {
    std::lock_guard lock(stats_mutex_);
    KernelLaunchStats stats;
    stats.total_launches = total_launches_;
    stats.cache_hits = cache_hits_;
    stats.cache_misses = cache_misses_;
    stats.batched_launches = batched_launches_;
    stats.min_launch_latency_us = (total_launches_ > 0) ? min_latency_us_ : 0.0;
    stats.max_launch_latency_us = max_latency_us_;
    stats.avg_launch_latency_us = (total_launches_ > 0) ? total_latency_us_ / total_launches_ : 0.0;
    return stats;
}

void KernelLaunchOptimizer::reset_stats() {
    std::lock_guard lock(stats_mutex_);
    total_latency_us_ = 0.0;
    min_latency_us_ = 1e18;
    max_latency_us_ = 0.0;
    total_launches_ = 0;
    cache_hits_ = 0;
    cache_misses_ = 0;
    batched_launches_ = 0;
}

KernelLaunchOptimizer::Config KernelLaunchOptimizer::get_config() const {
    std::shared_lock lock(cache_mutex_);
    return config_;
}

void KernelLaunchOptimizer::update_config(const Config& config) {
    std::unique_lock lock(cache_mutex_);
    config_ = config;
}

} // namespace gpu
} // namespace executor
