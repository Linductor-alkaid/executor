#include <cassert>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <future>
#include <algorithm>
#include <string>

// 包含 RealtimeThreadExecutor 的头文件
#include <executor/config.hpp>
#include <executor/types.hpp>
#include <executor/interfaces.hpp>
#include "executor/realtime_thread_executor.hpp"

using namespace executor;

// 测试辅助宏
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "FAILED: " << message << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

// ========== 基本功能测试 ==========

bool test_realtime_executor_basic() {
    std::cout << "Testing RealtimeThreadExecutor basic operations..." << std::endl;
    
    // 创建配置
    RealtimeThreadConfig config;
    config.thread_name = "test_realtime_thread";
    config.cycle_period_ns = 10000000;  // 10ms
    config.thread_priority = 0;  // 普通优先级
    config.cycle_callback = []() {
        // 空回调
    };
    
    // 创建执行器
    RealtimeThreadExecutor executor("test_executor", config);
    
    // 测试获取名称
    TEST_ASSERT(executor.get_name() == "test_executor", "Executor name should match");
    
    // 测试状态（启动前应该不是运行状态）
    auto status = executor.get_status();
    TEST_ASSERT(status.name == "test_executor", "Status name should match");
    TEST_ASSERT(status.is_running == false, "Executor should not be running before start");
    TEST_ASSERT(status.cycle_period_ns == 10000000, "Cycle period should match");
    TEST_ASSERT(status.cycle_count == 0, "Cycle count should be 0 initially");
    
    // 测试启动
    TEST_ASSERT(executor.start(), "Executor should start successfully");
    
    // 等待一小段时间让线程启动
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // 测试状态（启动后应该是运行状态）
    status = executor.get_status();
    TEST_ASSERT(status.is_running == true, "Executor should be running after start");
    
    // 测试停止
    executor.stop();
    
    // 等待线程结束
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // 测试状态（停止后应该不是运行状态）
    status = executor.get_status();
    TEST_ASSERT(status.is_running == false, "Executor should not be running after stop");
    
    std::cout << "  RealtimeThreadExecutor basic operations: PASSED" << std::endl;
    return true;
}

bool test_realtime_executor_double_start() {
    std::cout << "Testing RealtimeThreadExecutor double start..." << std::endl;
    
    RealtimeThreadConfig config;
    config.thread_name = "test_thread";
    config.cycle_period_ns = 10000000;  // 10ms
    config.cycle_callback = []() {};
    
    RealtimeThreadExecutor executor("test_executor", config);
    
    // 第一次启动应该成功
    TEST_ASSERT(executor.start(), "First start should succeed");
    
    // 等待线程启动
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // 第二次启动应该失败
    TEST_ASSERT(!executor.start(), "Second start should fail");
    
    executor.stop();
    
    std::cout << "  Double start: PASSED" << std::endl;
    return true;
}

// ========== 周期循环测试 ==========

bool test_realtime_executor_cycle_loop() {
    std::cout << "Testing RealtimeThreadExecutor cycle loop..." << std::endl;
    
    std::atomic<int> cycle_count(0);
    const int64_t period_ns = 20000000;  // 20ms
    
    RealtimeThreadConfig config;
    config.thread_name = "test_thread";
    config.cycle_period_ns = period_ns;
    config.cycle_callback = [&cycle_count]() {
        cycle_count.fetch_add(1);
    };
    
    RealtimeThreadExecutor executor("test_executor", config);
    TEST_ASSERT(executor.start(), "Executor should start successfully");
    
    // 等待几个周期
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // 检查周期计数
    int count = cycle_count.load();
    TEST_ASSERT(count > 0, "Cycle callback should be called");
    TEST_ASSERT(count >= 4, "Should have at least 4 cycles in 100ms (20ms period)");
    
    // 检查状态中的周期计数
    // 注意：由于时序问题，状态中的周期计数可能与回调计数不完全一致
    // 在 Windows 上，由于线程调度精度问题，允许一定的差异
    auto status = executor.get_status();
    TEST_ASSERT(status.cycle_count > 0, "Status cycle count should be greater than 0");
    // 允许状态计数与回调计数有 ±1 的差异（由于时序问题）
    int64_t count_diff = (status.cycle_count > static_cast<int64_t>(count)) ? 
                         (status.cycle_count - static_cast<int64_t>(count)) : 
                         (static_cast<int64_t>(count) - status.cycle_count);
    TEST_ASSERT(count_diff <= 1, 
                "Status cycle count should be approximately equal to actual count (diff: " + 
                std::to_string(count_diff) + ")");
    
    executor.stop();
    
    std::cout << "  Cycle loop: PASSED (count: " << count << ")" << std::endl;
    return true;
}

bool test_realtime_executor_cycle_period_accuracy() {
    std::cout << "Testing RealtimeThreadExecutor cycle period accuracy..." << std::endl;
    
    std::vector<std::chrono::steady_clock::time_point> cycle_times;
    const int64_t period_ns = 10000000;  // 10ms
    const int num_cycles = 10;
    
    RealtimeThreadConfig config;
    config.thread_name = "test_thread";
    config.cycle_period_ns = period_ns;
    config.cycle_callback = [&cycle_times]() {
        cycle_times.push_back(std::chrono::steady_clock::now());
    };
    
    RealtimeThreadExecutor executor("test_executor", config);
    TEST_ASSERT(executor.start(), "Executor should start successfully");
    
    // 等待足够的周期
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    
    executor.stop();
    
    // 检查周期时间间隔
    // 注意：这是一个精度测试，用于验证周期控制的准确性
    // 在Windows上，即使使用了高精度定时器，系统调度仍可能影响精度
    if (cycle_times.size() >= num_cycles) {
        int64_t expected = period_ns;
        
        // 跳过前几个间隔（系统稳定期）
        const size_t skip_initial = 2;
        
        // 计算平均间隔和标准差
        int64_t total_interval = 0;
        size_t interval_count = 0;
        std::vector<int64_t> intervals;
        
        for (size_t i = skip_initial + 1; i < cycle_times.size(); ++i) {
            auto interval = std::chrono::duration_cast<std::chrono::nanoseconds>(
                cycle_times[i] - cycle_times[i-1]
            ).count();
            intervals.push_back(interval);
            total_interval += interval;
            interval_count++;
        }
        
        if (interval_count > 0) {
            // 计算平均间隔
            int64_t avg_interval = total_interval / interval_count;
            
            // 计算平均误差百分比
            double avg_error_percent = std::abs(static_cast<double>(avg_interval - expected)) / expected * 100.0;
            
#ifdef _WIN32
            // Windows: 使用高精度定时器后，平均误差应该在30%以内
            // 允许个别间隔有较大偏差（系统调度影响），但平均值应该接近预期
            TEST_ASSERT(avg_error_percent <= 30.0,
                       "Average cycle interval error too large: " + 
                       std::to_string(avg_error_percent) + "% (expected < 30%)");
            
            // 检查至少70%的间隔在合理范围内（±50%下限，+100%上限）
            int64_t tolerance_lower = expected / 2;
            int64_t tolerance_upper = expected * 2;
            size_t valid_count = 0;
            for (auto interval : intervals) {
                if (interval >= expected - tolerance_lower && interval <= expected + tolerance_upper) {
                    valid_count++;
                }
            }
            double valid_rate = static_cast<double>(valid_count) / interval_count;
            TEST_ASSERT(valid_rate >= 0.7,
                       "Too many intervals out of tolerance: " + 
                       std::to_string(valid_count) + "/" + std::to_string(interval_count) + 
                       " within tolerance (expected >= 70%)");
#else
            // Linux: 保持严格标准（±20%下限，+60%上限，平均误差<10%）
            TEST_ASSERT(avg_error_percent <= 10.0,
                       "Average cycle interval error too large: " + 
                       std::to_string(avg_error_percent) + "% (expected < 10%)");
            
            int64_t tolerance_lower = expected / 5;
            int64_t tolerance_upper = expected / 5 * 3;
            size_t valid_count = 0;
            for (auto interval : intervals) {
                if (interval >= expected - tolerance_lower && interval <= expected + tolerance_upper) {
                    valid_count++;
                }
            }
            TEST_ASSERT(valid_count == interval_count,
                       "All intervals should be within tolerance: " + 
                       std::to_string(valid_count) + "/" + std::to_string(interval_count));
#endif
        }
    } else {
        // 如果没有足够的周期数据，至少检查是否有周期执行
        TEST_ASSERT(cycle_times.size() > 0, "At least some cycles should be executed");
    }
    
    std::cout << "  Cycle period accuracy: PASSED" << std::endl;
    return true;
}

// ========== 任务处理测试 ==========

bool test_realtime_executor_push_task() {
    std::cout << "Testing RealtimeThreadExecutor push_task..." << std::endl;
    
    std::atomic<int> task_count(0);
    const int64_t period_ns = 10000000;  // 10ms
    
    RealtimeThreadConfig config;
    config.thread_name = "test_thread";
    config.cycle_period_ns = period_ns;
    config.cycle_callback = []() {
        // 空回调
    };
    
    RealtimeThreadExecutor executor("test_executor", config);
    TEST_ASSERT(executor.start(), "Executor should start successfully");
    
    // 推送多个任务
    const int num_tasks = 10;
    for (int i = 0; i < num_tasks; ++i) {
        executor.push_task([&task_count, i]() {
            task_count.fetch_add(1);
        });
    }
    
    // 等待任务执行（需要等待几个周期）
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    
    // 检查任务是否都执行了
    int count = task_count.load();
    TEST_ASSERT(count == num_tasks, "All tasks should be executed");
    
    executor.stop();
    
    std::cout << "  Push task: PASSED (executed " << count << " tasks)" << std::endl;
    return true;
}

bool test_realtime_executor_task_order() {
    std::cout << "Testing RealtimeThreadExecutor task order..." << std::endl;
    
    std::vector<int> execution_order;
    std::mutex order_mutex;
    const int64_t period_ns = 10000000;  // 10ms
    
    RealtimeThreadConfig config;
    config.thread_name = "test_thread";
    config.cycle_period_ns = period_ns;
    config.cycle_callback = []() {};
    
    RealtimeThreadExecutor executor("test_executor", config);
    TEST_ASSERT(executor.start(), "Executor should start successfully");
    
    // 推送多个任务，每个任务记录执行顺序
    const int num_tasks = 5;
    for (int i = 0; i < num_tasks; ++i) {
        executor.push_task([&execution_order, &order_mutex, i]() {
            std::lock_guard<std::mutex> lock(order_mutex);
            execution_order.push_back(i);
        });
    }
    
    // 等待任务执行
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    
    executor.stop();
    
    // 检查任务执行顺序（应该是 FIFO）
    TEST_ASSERT(execution_order.size() == num_tasks, "All tasks should be executed");
    for (int i = 0; i < num_tasks; ++i) {
        TEST_ASSERT(execution_order[i] == i, "Tasks should execute in order");
    }
    
    std::cout << "  Task order: PASSED" << std::endl;
    return true;
}

// ========== 统计信息测试 ==========

bool test_realtime_executor_statistics() {
    std::cout << "Testing RealtimeThreadExecutor statistics..." << std::endl;
    
    const int64_t period_ns = 10000000;  // 10ms
    
    RealtimeThreadConfig config;
    config.thread_name = "test_thread";
    config.cycle_period_ns = period_ns;
    config.cycle_callback = []() {
        // 模拟一些工作
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    };
    
    RealtimeThreadExecutor executor("test_executor", config);
    TEST_ASSERT(executor.start(), "Executor should start successfully");
    
    // 等待几个周期
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    
    auto status = executor.get_status();
    
    // 检查统计信息
    TEST_ASSERT(status.cycle_count > 0, "Cycle count should be greater than 0");
    TEST_ASSERT(status.avg_cycle_time_ns > 0, "Average cycle time should be greater than 0");
    TEST_ASSERT(status.max_cycle_time_ns > 0, "Max cycle time should be greater than 0");
    TEST_ASSERT(status.max_cycle_time_ns >= status.avg_cycle_time_ns, 
                "Max cycle time should be >= average cycle time");
    
    executor.stop();
    
    std::cout << "  Statistics: PASSED (cycles: " << status.cycle_count 
              << ", avg: " << status.avg_cycle_time_ns / 1000000.0 << "ms"
              << ", max: " << status.max_cycle_time_ns / 1000000.0 << "ms)" << std::endl;
    return true;
}

bool test_realtime_executor_timeout_count() {
    std::cout << "Testing RealtimeThreadExecutor timeout count..." << std::endl;
    
    const int64_t period_ns = 10000000;  // 10ms
    
    RealtimeThreadConfig config;
    config.thread_name = "test_thread";
    config.cycle_period_ns = period_ns;
    config.cycle_callback = []() {
        // 模拟长时间工作，导致周期超时
        std::this_thread::sleep_for(std::chrono::milliseconds(15));  // 超过周期时间
    };
    
    RealtimeThreadExecutor executor("test_executor", config);
    TEST_ASSERT(executor.start(), "Executor should start successfully");
    
    // 等待几个周期
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    auto status = executor.get_status();
    
    // 检查超时计数
    TEST_ASSERT(status.cycle_timeout_count > 0, "Timeout count should be greater than 0");
    
    executor.stop();
    
    std::cout << "  Timeout count: PASSED (timeouts: " << status.cycle_timeout_count << ")" << std::endl;
    return true;
}

// ========== 异常处理测试 ==========

bool test_realtime_executor_cycle_callback_exception() {
    std::cout << "Testing RealtimeThreadExecutor cycle callback exception..." << std::endl;
    
    std::atomic<int> cycle_count(0);
    const int64_t period_ns = 10000000;  // 10ms
    
    RealtimeThreadConfig config;
    config.thread_name = "test_thread";
    config.cycle_period_ns = period_ns;
    config.cycle_callback = [&cycle_count]() {
        cycle_count.fetch_add(1);
        if (cycle_count.load() == 1) {
            // 第一次周期抛出异常
            throw std::runtime_error("Test exception");
        }
    };
    
    RealtimeThreadExecutor executor("test_executor", config);
    TEST_ASSERT(executor.start(), "Executor should start successfully");
    
    // 等待几个周期
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // 检查周期是否继续执行（异常不应该影响周期执行）
    int count = cycle_count.load();
    TEST_ASSERT(count > 1, "Cycle should continue after exception");
    
    // 检查执行器是否仍在运行
    auto status = executor.get_status();
    TEST_ASSERT(status.is_running == true, "Executor should still be running after exception");
    
    executor.stop();
    
    std::cout << "  Cycle callback exception: PASSED (cycles: " << count << ")" << std::endl;
    return true;
}

bool test_realtime_executor_task_exception() {
    std::cout << "Testing RealtimeThreadExecutor task exception..." << std::endl;
    
    std::atomic<int> task_count(0);
    const int64_t period_ns = 10000000;  // 10ms
    
    RealtimeThreadConfig config;
    config.thread_name = "test_thread";
    config.cycle_period_ns = period_ns;
    config.cycle_callback = []() {};
    
    RealtimeThreadExecutor executor("test_executor", config);
    TEST_ASSERT(executor.start(), "Executor should start successfully");
    
    // 推送一个会抛出异常的任务
    executor.push_task([&task_count]() {
        task_count.fetch_add(1);
        throw std::runtime_error("Task exception");
    });
    
    // 推送一个正常任务
    executor.push_task([&task_count]() {
        task_count.fetch_add(1);
    });
    
    // 等待任务执行
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // 检查两个任务是否都执行了（异常不应该影响其他任务）
    int count = task_count.load();
    TEST_ASSERT(count == 2, "Both tasks should be executed");
    
    executor.stop();
    
    std::cout << "  Task exception: PASSED" << std::endl;
    return true;
}

// ========== 并发测试 ==========

bool test_realtime_executor_concurrent_push_task() {
    std::cout << "Testing RealtimeThreadExecutor concurrent push_task..." << std::endl;
    
    std::atomic<int> task_count(0);
    const int64_t period_ns = 10000000;  // 10ms
    
    RealtimeThreadConfig config;
    config.thread_name = "test_thread";
    config.cycle_period_ns = period_ns;
    config.cycle_callback = []() {};
    
    RealtimeThreadExecutor executor("test_executor", config);
    TEST_ASSERT(executor.start(), "Executor should start successfully");
    
    // 多个线程同时推送任务
    const int num_threads = 5;
    const int tasks_per_thread = 10;
    std::vector<std::thread> threads;
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&executor, &task_count, tasks_per_thread]() {
            for (int j = 0; j < tasks_per_thread; ++j) {
                executor.push_task([&task_count]() {
                    task_count.fetch_add(1);
                });
            }
        });
    }
    
    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }
    
    // 等待任务执行
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // 检查所有任务是否都执行了
    int count = task_count.load();
    TEST_ASSERT(count == num_threads * tasks_per_thread, 
                "All tasks should be executed");
    
    executor.stop();
    
    std::cout << "  Concurrent push_task: PASSED (executed " << count << " tasks)" << std::endl;
    return true;
}

bool test_realtime_executor_concurrent_get_status() {
    std::cout << "Testing RealtimeThreadExecutor concurrent get_status..." << std::endl;
    
    const int64_t period_ns = 10000000;  // 10ms
    
    RealtimeThreadConfig config;
    config.thread_name = "test_thread";
    config.cycle_period_ns = period_ns;
    config.cycle_callback = []() {};
    
    RealtimeThreadExecutor executor("test_executor", config);
    TEST_ASSERT(executor.start(), "Executor should start successfully");
    
    // 多个线程同时获取状态
    const int num_threads = 10;
    std::vector<std::thread> threads;
    std::atomic<int> success_count(0);
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&executor, &success_count]() {
            for (int j = 0; j < 100; ++j) {
                auto status = executor.get_status();
                if (status.is_running && status.cycle_period_ns == 10000000) {
                    success_count.fetch_add(1);
                }
            }
        });
    }
    
    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }
    
    executor.stop();
    
    // 检查所有调用是否成功
    int count = success_count.load();
    TEST_ASSERT(count == num_threads * 100, "All status calls should succeed");
    
    std::cout << "  Concurrent get_status: PASSED" << std::endl;
    return true;
}

// ========== 主函数 ==========

int main() {
    std::cout << "========== RealtimeThreadExecutor 集成测试 ==========" << std::endl;
    std::cout << std::endl;
    
    bool all_passed = true;
    
    // 基本功能测试
    all_passed &= test_realtime_executor_basic();
    all_passed &= test_realtime_executor_double_start();
    
    // 周期循环测试
    all_passed &= test_realtime_executor_cycle_loop();
    all_passed &= test_realtime_executor_cycle_period_accuracy();
    
    // 任务处理测试
    all_passed &= test_realtime_executor_push_task();
    all_passed &= test_realtime_executor_task_order();
    
    // 统计信息测试
    all_passed &= test_realtime_executor_statistics();
    all_passed &= test_realtime_executor_timeout_count();
    
    // 异常处理测试
    all_passed &= test_realtime_executor_cycle_callback_exception();
    all_passed &= test_realtime_executor_task_exception();
    
    // 并发测试
    all_passed &= test_realtime_executor_concurrent_push_task();
    all_passed &= test_realtime_executor_concurrent_get_status();
    
    std::cout << std::endl;
    if (all_passed) {
        std::cout << "========== 所有测试通过 ==========" << std::endl;
        return 0;
    } else {
        std::cout << "========== 部分测试失败 ==========" << std::endl;
        return 1;
    }
}
