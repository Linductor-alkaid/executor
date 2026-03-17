#pragma once

#include <cstddef>
#include <shared_mutex>

namespace executor {
namespace gpu {

/**
 * @brief Task characteristics for scheduling decisions
 */
struct TaskCharacteristics {
    size_t data_size_bytes = 0;        ///< Total data size to process
    float compute_intensity = 1.0f;     ///< Compute intensity hint (1.0=baseline, >1.0=compute-heavy)
    bool prefer_gpu = false;            ///< User preference hint
};

/**
 * @brief Executor choice result
 */
enum class ExecutorChoice {
    CPU,    ///< Use CPU thread pool
    GPU     ///< Use GPU executor
};

/**
 * @brief GPU scheduler for automatic CPU/GPU task routing
 *
 * Uses heuristic-based decision making to choose between CPU and GPU execution.
 * Considers data size and compute intensity.
 */
class GpuScheduler {
public:
    /**
     * @brief Scheduler configuration
     */
    struct Config {
        size_t data_size_threshold = 1024 * 1024;  ///< Data size threshold (bytes, default 1MB)
        float compute_intensity_threshold = 2.0f;   ///< Compute intensity threshold (default 2.0x)
    };

    /**
     * @brief Construct scheduler with default config
     */
    GpuScheduler();

    /**
     * @brief Construct scheduler with config
     */
    explicit GpuScheduler(const Config& config);

    /**
     * @brief Decide which executor to use
     *
     * Decision logic:
     * - If prefer_gpu hint is set, choose GPU
     * - If data_size >= threshold AND compute_intensity >= threshold, choose GPU
     * - Otherwise, choose CPU
     *
     * @param characteristics Task characteristics
     * @return ExecutorChoice CPU or GPU
     */
    ExecutorChoice decide(const TaskCharacteristics& characteristics) const;

    /**
     * @brief Update scheduler configuration
     */
    void update_config(const Config& config);

    /**
     * @brief Get current configuration
     */
    Config get_config() const;

private:
    Config config_;
    mutable std::shared_mutex mutex_;
};

} // namespace gpu
} // namespace executor
