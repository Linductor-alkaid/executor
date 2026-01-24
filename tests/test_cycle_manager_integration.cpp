#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>
#include <atomic>

#include <executor/config.hpp>
#include <executor/types.hpp>
#include <executor/interfaces.hpp>
#include "executor/realtime_thread_executor.hpp"
#include "mock_cycle_manager.hpp"

using namespace executor;
using namespace executor::test;

#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "FAILED: " << message << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

bool test_realtime_executor_with_cycle_manager() {
    std::cout << "Testing RealtimeThreadExecutor with ICycleManager..." << std::endl;

    MockCycleManager mock_manager;

    RealtimeThreadConfig config;
    config.thread_name = "test_cycle_manager";
    config.cycle_period_ns = 10000000;  // 10ms
    config.thread_priority = 0;
    config.cycle_manager = &mock_manager;
    config.cycle_callback = []() {};

    RealtimeThreadExecutor executor("test_executor", config);

    TEST_ASSERT(executor.get_name() == "test_executor", "Executor name should match");
    TEST_ASSERT(executor.start(), "Executor should start successfully");

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    auto status = executor.get_status();
    TEST_ASSERT(status.is_running == true, "Executor should be running with cycle manager");
    TEST_ASSERT(status.cycle_count > 0, "Cycle count should increase");

    executor.stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    status = executor.get_status();
    TEST_ASSERT(status.is_running == false, "Executor should stop cleanly");

    std::cout << "  RealtimeThreadExecutor with ICycleManager: PASSED" << std::endl;
    return true;
}

bool test_realtime_executor_cycle_manager_stop() {
    std::cout << "Testing RealtimeThreadExecutor cycle manager stop..." << std::endl;

    MockCycleManager mock_manager;

    RealtimeThreadConfig config;
    config.thread_name = "test_stop";
    config.cycle_period_ns = 5000000;  // 5ms
    config.thread_priority = 0;
    config.cycle_manager = &mock_manager;
    config.cycle_callback = []() {};

    RealtimeThreadExecutor executor("test_stop_executor", config);
    TEST_ASSERT(executor.start(), "Start should succeed");

    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    executor.stop();

    auto status = executor.get_status();
    TEST_ASSERT(status.is_running == false, "Executor should be stopped");
    TEST_ASSERT(status.cycle_count > 0, "Should have run at least one cycle");

    std::cout << "  Cycle manager stop: PASSED" << std::endl;
    return true;
}

bool test_realtime_executor_cycle_loop_execution() {
    std::cout << "Testing RealtimeThreadExecutor cycle_loop execution..." << std::endl;

    MockCycleManager mock_manager;
    std::atomic<int> cycle_count{0};

    RealtimeThreadConfig config;
    config.thread_name = "test_cycle_loop";
    config.cycle_period_ns = 5000000;  // 5ms
    config.thread_priority = 0;
    config.cycle_manager = &mock_manager;
    config.cycle_callback = [&cycle_count]() { cycle_count.fetch_add(1); };

    RealtimeThreadExecutor executor("test_cycle_loop_executor", config);
    TEST_ASSERT(executor.start(), "Start should succeed");

    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    int count_before_stop = cycle_count.load();
    executor.stop();

    TEST_ASSERT(count_before_stop > 0, "Cycle callback should have been called");
    auto status = executor.get_status();
    TEST_ASSERT(status.cycle_count == count_before_stop,
                "Executor cycle count should match callback invocations");

    std::cout << "  Cycle_loop execution: PASSED" << std::endl;
    return true;
}

bool test_realtime_executor_fallback_to_simple() {
    std::cout << "Testing RealtimeThreadExecutor fallback to simple..." << std::endl;

    MockCycleManager mock_manager;
    mock_manager.fail_register = true;

    std::atomic<int> fallback_cycles{0};

    RealtimeThreadConfig config;
    config.thread_name = "test_fallback";
    config.cycle_period_ns = 5000000;
    config.thread_priority = 0;
    config.cycle_manager = &mock_manager;
    config.cycle_callback = [&fallback_cycles]() { fallback_cycles.fetch_add(1); };

    RealtimeThreadExecutor executor("test_fallback_executor", config);
    TEST_ASSERT(executor.start(), "Start should succeed despite register failure");

    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    int count = fallback_cycles.load();
    executor.stop();

    TEST_ASSERT(count > 0, "Should run simple_cycle_loop when register fails");
    auto status = executor.get_status();
    TEST_ASSERT(status.cycle_count > 0, "Executor should have cycle count from fallback");

    std::cout << "  Fallback to simple: PASSED" << std::endl;
    return true;
}

bool test_realtime_executor_fallback_start_fails() {
    std::cout << "Testing RealtimeThreadExecutor fallback when start_cycle fails..." << std::endl;

    MockCycleManager mock_manager;
    mock_manager.fail_start = true;

    std::atomic<int> fallback_cycles{0};

    RealtimeThreadConfig config;
    config.thread_name = "test_fallback_start";
    config.cycle_period_ns = 5000000;
    config.thread_priority = 0;
    config.cycle_manager = &mock_manager;
    config.cycle_callback = [&fallback_cycles]() { fallback_cycles.fetch_add(1); };

    RealtimeThreadExecutor executor("test_fallback_start_executor", config);
    TEST_ASSERT(executor.start(), "Start should succeed");

    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    int count = fallback_cycles.load();
    executor.stop();

    TEST_ASSERT(count > 0, "Should run simple_cycle_loop when start_cycle fails");

    std::cout << "  Fallback when start_cycle fails: PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "========== ICycleManager 集成测试 ==========" << std::endl;
    std::cout << std::endl;

    bool all_passed = true;
    all_passed &= test_realtime_executor_with_cycle_manager();
    all_passed &= test_realtime_executor_cycle_manager_stop();
    all_passed &= test_realtime_executor_cycle_loop_execution();
    all_passed &= test_realtime_executor_fallback_to_simple();
    all_passed &= test_realtime_executor_fallback_start_fails();

    std::cout << std::endl;
    if (all_passed) {
        std::cout << "========== 所有 ICycleManager 集成测试通过 ==========" << std::endl;
        return 0;
    } else {
        std::cout << "========== 部分测试失败 ==========" << std::endl;
        return 1;
    }
}
