#include <cassert>
#include <iostream>
#include <executor/gpu/gpu_scheduler.hpp>

using namespace executor;
using namespace executor::gpu;

// Test helper macro
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "FAILED: " << message << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

// Test 1: Default configuration
bool test_default_config() {
    GpuScheduler scheduler;
    auto config = scheduler.get_config();

    TEST_ASSERT(config.data_size_threshold == 1024 * 1024, "Default data size threshold should be 1MB");
    TEST_ASSERT(config.compute_intensity_threshold == 2.0f, "Default compute intensity threshold should be 2.0");

    std::cout << "PASSED: test_default_config" << std::endl;
    return true;
}

// Test 2: Small data -> CPU
bool test_small_data_cpu() {
    GpuScheduler scheduler;

    TaskCharacteristics chars;
    chars.data_size_bytes = 512 * 1024;  // 512KB < 1MB threshold
    chars.compute_intensity = 3.0f;       // High compute, but small data

    auto choice = scheduler.decide(chars);
    TEST_ASSERT(choice == ExecutorChoice::CPU, "Small data should choose CPU");

    std::cout << "PASSED: test_small_data_cpu" << std::endl;
    return true;
}

// Test 3: Large data + high compute -> GPU
bool test_large_data_high_compute_gpu() {
    GpuScheduler scheduler;

    TaskCharacteristics chars;
    chars.data_size_bytes = 10 * 1024 * 1024;  // 10MB > 1MB threshold
    chars.compute_intensity = 5.0f;             // 5.0 > 2.0 threshold

    auto choice = scheduler.decide(chars);
    TEST_ASSERT(choice == ExecutorChoice::GPU, "Large data + high compute should choose GPU");

    std::cout << "PASSED: test_large_data_high_compute_gpu" << std::endl;
    return true;
}

// Test 4: Large data + low compute -> CPU
bool test_large_data_low_compute_cpu() {
    GpuScheduler scheduler;

    TaskCharacteristics chars;
    chars.data_size_bytes = 10 * 1024 * 1024;  // 10MB > threshold
    chars.compute_intensity = 1.0f;             // 1.0 < 2.0 threshold

    auto choice = scheduler.decide(chars);
    TEST_ASSERT(choice == ExecutorChoice::CPU, "Large data but low compute should choose CPU");

    std::cout << "PASSED: test_large_data_low_compute_cpu" << std::endl;
    return true;
}

// Test 5: User preference hint
bool test_prefer_gpu_hint() {
    GpuScheduler scheduler;

    TaskCharacteristics chars;
    chars.data_size_bytes = 100;  // Very small data
    chars.compute_intensity = 0.5f;  // Low compute
    chars.prefer_gpu = true;  // But user prefers GPU

    auto choice = scheduler.decide(chars);
    TEST_ASSERT(choice == ExecutorChoice::GPU, "prefer_gpu hint should override heuristics");

    std::cout << "PASSED: test_prefer_gpu_hint" << std::endl;
    return true;
}

// Test 6: Configuration update
bool test_config_update() {
    GpuScheduler scheduler;

    GpuScheduler::Config new_config;
    new_config.data_size_threshold = 512 * 1024;  // 512KB
    new_config.compute_intensity_threshold = 1.5f;

    scheduler.update_config(new_config);
    auto config = scheduler.get_config();

    TEST_ASSERT(config.data_size_threshold == 512 * 1024, "Config should be updated");
    TEST_ASSERT(config.compute_intensity_threshold == 1.5f, "Config should be updated");

    // Test with new thresholds
    TaskCharacteristics chars;
    chars.data_size_bytes = 600 * 1024;  // 600KB > 512KB
    chars.compute_intensity = 2.0f;       // 2.0 > 1.5

    auto choice = scheduler.decide(chars);
    TEST_ASSERT(choice == ExecutorChoice::GPU, "Should choose GPU with updated thresholds");

    std::cout << "PASSED: test_config_update" << std::endl;
    return true;
}

int main() {
    std::cout << "Running GPU Scheduler tests..." << std::endl;

    bool all_passed = true;
    all_passed &= test_default_config();
    all_passed &= test_small_data_cpu();
    all_passed &= test_large_data_high_compute_gpu();
    all_passed &= test_large_data_low_compute_cpu();
    all_passed &= test_prefer_gpu_hint();
    all_passed &= test_config_update();

    if (all_passed) {
        std::cout << "\nAll tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cerr << "\nSome tests FAILED!" << std::endl;
        return 1;
    }
}
