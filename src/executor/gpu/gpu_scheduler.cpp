#include "executor/gpu/gpu_scheduler.hpp"
#include <mutex>

namespace executor {
namespace gpu {

GpuScheduler::GpuScheduler()
    : config_() {}

GpuScheduler::GpuScheduler(const Config& config)
    : config_(config) {}

ExecutorChoice GpuScheduler::decide(const TaskCharacteristics& characteristics) const {
    std::shared_lock lock(mutex_);

    // User preference hint
    if (characteristics.prefer_gpu) {
        return ExecutorChoice::GPU;
    }

    // Heuristic: GPU if both data size and compute intensity exceed thresholds
    bool large_data = characteristics.data_size_bytes >= config_.data_size_threshold;
    bool compute_heavy = characteristics.compute_intensity >= config_.compute_intensity_threshold;

    if (large_data && compute_heavy) {
        return ExecutorChoice::GPU;
    }

    return ExecutorChoice::CPU;
}

void GpuScheduler::update_config(const Config& config) {
    std::unique_lock lock(mutex_);
    config_ = config;
}

GpuScheduler::Config GpuScheduler::get_config() const {
    std::shared_lock lock(mutex_);
    return config_;
}

} // namespace gpu
} // namespace executor
