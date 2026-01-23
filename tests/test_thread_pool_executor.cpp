#include <cassert>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <future>
#include <algorithm>

// 包含 ThreadPoolExecutor 的头文件
#include <executor/config.hpp>
#include <executor/types.hpp>
#include <executor/interfaces.hpp>
#include "executor/thread_pool_executor.hpp"

using namespace executor;

// 测试辅助宏
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "FAILED: " << message << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

// 测试函数前向声明
bool test_thread_pool_executor_basic();
bool test_thread_pool_executor_multiple_tasks();
bool test_thread_pool_executor_concurrent_submit();
bool test_thread_pool_executor_exception_handling();
bool test_thread_pool_executor_status();
bool test_thread_pool_executor_wait_for_completion();
bool test_thread_pool_executor_stop_behavior();

// ========== ThreadPoolExecutor 基本功能测试 ==========

bool test_thread_pool_executor_basic() {
    std::cout << "Testing ThreadPoolExecutor basic operations..." << std::endl;
    
    // 创建配置
    ThreadPoolConfig config;
    config.min_threads = 2;
    config.max_threads = 4;
    config.queue_capacity = 100;
    
    // 创建执行器
    ThreadPoolExecutor executor("test_executor", config);
    
    // 测试获取名称
    TEST_ASSERT(executor.get_name() == "test_executor", "Executor name should match");
    
    // 测试启动
    TEST_ASSERT(executor.start(), "Executor should start successfully");
    
    // 测试状态（启动后应该是运行状态）
    auto status = executor.get_status();
    TEST_ASSERT(status.name == "test_executor", "Status name should match");
    TEST_ASSERT(status.is_running == true, "Executor should be running");
    
    // 测试提交任务
    auto future = executor.submit([]() noexcept {
        return 42;
    });
    
    int result = future.get();
    TEST_ASSERT(result == 42, "Task result should be 42");
    
    // 测试停止
    executor.stop();
    
    // 测试状态（停止后应该不是运行状态）
    status = executor.get_status();
    TEST_ASSERT(status.is_running == false, "Executor should not be running after stop");
    
    std::cout << "  ThreadPoolExecutor basic operations: PASSED" << std::endl;
    return true;
}

bool test_thread_pool_executor_multiple_tasks() {
    std::cout << "Testing ThreadPoolExecutor multiple tasks..." << std::endl;
    
    ThreadPoolConfig config;
    config.min_threads = 4;
    config.max_threads = 8;
    config.queue_capacity = 100;
    
    ThreadPoolExecutor executor("test_executor", config);
    TEST_ASSERT(executor.start(), "Executor should start successfully");
    
    // 提交多个任务
    const int num_tasks = 100;
    std::vector<std::future<int>> futures;
    
    for (int i = 0; i < num_tasks; ++i) {
        auto future = executor.submit([i]() noexcept {
            return i * 2;
        });
        futures.push_back(std::move(future));
    }
    
    // 等待所有任务完成并验证结果
    for (size_t i = 0; i < futures.size(); ++i) {
        int result = futures[i].get();
        TEST_ASSERT(result == static_cast<int>(i) * 2, "Task result should match");
    }
    
    // 检查状态
    auto status = executor.get_status();
    TEST_ASSERT(status.completed_tasks >= num_tasks, "Completed tasks should be at least num_tasks");
    
    executor.stop();
    
    std::cout << "  ThreadPoolExecutor multiple tasks: PASSED" << std::endl;
    return true;
}

bool test_thread_pool_executor_concurrent_submit() {
    std::cout << "Testing ThreadPoolExecutor concurrent submit..." << std::endl;
    
    ThreadPoolConfig config;
    config.min_threads = 4;
    config.max_threads = 8;
    config.queue_capacity = 1000;
    
    ThreadPoolExecutor executor("test_executor", config);
    TEST_ASSERT(executor.start(), "Executor should start successfully");
    
    // 从多个线程并发提交任务
    const int num_threads = 10;
    const int tasks_per_thread = 50;
    std::atomic<int> completed_count{0};
    std::vector<std::thread> threads;
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&executor, &completed_count, tasks_per_thread, t]() {
            for (int i = 0; i < tasks_per_thread; ++i) {
                auto future = executor.submit([&completed_count]() noexcept {
                    completed_count.fetch_add(1, std::memory_order_relaxed);
                    return 1;
                });
                future.wait();
            }
        });
    }
    
    // 等待所有提交线程完成
    for (auto& thread : threads) {
        thread.join();
    }
    
    // 等待所有任务完成
    executor.wait_for_completion();
    
    // 验证所有任务都完成了
    TEST_ASSERT(completed_count.load() == num_threads * tasks_per_thread, 
                "All tasks should be completed");
    
    auto status = executor.get_status();
    TEST_ASSERT(status.completed_tasks >= num_threads * tasks_per_thread,
                "Status should reflect completed tasks");
    
    executor.stop();
    
    std::cout << "  ThreadPoolExecutor concurrent submit: PASSED" << std::endl;
    return true;
}

bool test_thread_pool_executor_exception_handling() {
    std::cout << "Testing ThreadPoolExecutor exception handling..." << std::endl;
    
    ThreadPoolConfig config;
    config.min_threads = 2;
    config.max_threads = 4;
    config.queue_capacity = 100;
    
    ThreadPoolExecutor executor("test_executor", config);
    TEST_ASSERT(executor.start(), "Executor should start successfully");
    
    // 提交一个会抛出异常的任务
    auto future = executor.submit([]() {
        throw std::runtime_error("Test exception");
        return 42;
    });
    
    // 验证异常被传播到future
    bool exception_caught = false;
    try {
        future.get();
    } catch (const std::runtime_error& e) {
        exception_caught = true;
        TEST_ASSERT(std::string(e.what()) == "Test exception", "Exception message should match");
    }
    
    TEST_ASSERT(exception_caught, "Exception should be caught");
    
    // 执行器应该仍然运行
    auto status = executor.get_status();
    TEST_ASSERT(status.is_running == true, "Executor should still be running after exception");
    
    executor.stop();
    
    std::cout << "  ThreadPoolExecutor exception handling: PASSED" << std::endl;
    return true;
}

bool test_thread_pool_executor_status() {
    std::cout << "Testing ThreadPoolExecutor status..." << std::endl;
    
    ThreadPoolConfig config;
    config.min_threads = 2;
    config.max_threads = 4;
    config.queue_capacity = 100;
    
    ThreadPoolExecutor executor("test_executor", config);
    
    // 测试初始状态（未启动）
    auto status = executor.get_status();
    TEST_ASSERT(status.name == "test_executor", "Status name should match");
    TEST_ASSERT(status.is_running == false, "Executor should not be running initially");
    TEST_ASSERT(status.active_tasks == 0, "Active tasks should be 0 initially");
    TEST_ASSERT(status.completed_tasks == 0, "Completed tasks should be 0 initially");
    TEST_ASSERT(status.queue_size == 0, "Queue size should be 0 initially");
    
    // 启动执行器
    TEST_ASSERT(executor.start(), "Executor should start successfully");
    
    // 测试运行状态
    status = executor.get_status();
    TEST_ASSERT(status.is_running == true, "Executor should be running");
    
    // 提交一些任务
    const int num_tasks = 10;
    std::vector<std::future<void>> futures;
    for (int i = 0; i < num_tasks; ++i) {
        auto future = executor.submit([]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        });
        futures.push_back(std::move(future));
    }
    
    // 等待所有任务完成
    for (auto& future : futures) {
        future.wait();
    }
    
    executor.wait_for_completion();
    
    // 检查最终状态
    status = executor.get_status();
    TEST_ASSERT(status.completed_tasks >= num_tasks, "Completed tasks should be at least num_tasks");
    TEST_ASSERT(status.queue_size == 0, "Queue should be empty after completion");
    
    executor.stop();
    
    std::cout << "  ThreadPoolExecutor status: PASSED" << std::endl;
    return true;
}

bool test_thread_pool_executor_wait_for_completion() {
    std::cout << "Testing ThreadPoolExecutor wait_for_completion..." << std::endl;
    
    ThreadPoolConfig config;
    config.min_threads = 2;
    config.max_threads = 4;
    config.queue_capacity = 100;
    
    ThreadPoolExecutor executor("test_executor", config);
    TEST_ASSERT(executor.start(), "Executor should start successfully");
    
    // 提交一些需要时间的任务
    const int num_tasks = 20;
    std::atomic<int> completed{0};
    
    for (int i = 0; i < num_tasks; ++i) {
        executor.submit([&completed]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            completed.fetch_add(1, std::memory_order_relaxed);
        });
    }
    
    // 等待所有任务完成
    executor.wait_for_completion();
    
    // 验证所有任务都完成了
    TEST_ASSERT(completed.load() == num_tasks, "All tasks should be completed");
    
    executor.stop();
    
    std::cout << "  ThreadPoolExecutor wait_for_completion: PASSED" << std::endl;
    return true;
}

bool test_thread_pool_executor_stop_behavior() {
    std::cout << "Testing ThreadPoolExecutor stop behavior..." << std::endl;
    
    ThreadPoolConfig config;
    config.min_threads = 2;
    config.max_threads = 4;
    config.queue_capacity = 100;
    
    ThreadPoolExecutor executor("test_executor", config);
    TEST_ASSERT(executor.start(), "Executor should start successfully");
    
    // 提交一些任务
    const int num_tasks = 10;
    std::vector<std::future<void>> futures;
    
    for (int i = 0; i < num_tasks; ++i) {
        auto future = executor.submit([]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        });
        futures.push_back(std::move(future));
    }
    
    // 停止执行器（应该等待任务完成）
    executor.stop();
    
    // 验证所有任务都完成了
    for (auto& future : futures) {
        future.wait();  // 应该不会阻塞，因为任务已经完成
    }
    
    // 验证状态
    auto status = executor.get_status();
    TEST_ASSERT(status.is_running == false, "Executor should not be running after stop");
    
    std::cout << "  ThreadPoolExecutor stop behavior: PASSED" << std::endl;
    return true;
}

// ========== 主函数 ==========

int main() {
    std::cout << "========== ThreadPoolExecutor Integration Tests ==========" << std::endl;
    std::cout << std::endl;
    
    bool all_passed = true;
    
    // 运行所有测试
    all_passed &= test_thread_pool_executor_basic();
    all_passed &= test_thread_pool_executor_multiple_tasks();
    all_passed &= test_thread_pool_executor_concurrent_submit();
    all_passed &= test_thread_pool_executor_exception_handling();
    all_passed &= test_thread_pool_executor_status();
    all_passed &= test_thread_pool_executor_wait_for_completion();
    all_passed &= test_thread_pool_executor_stop_behavior();
    
    std::cout << std::endl;
    if (all_passed) {
        std::cout << "========== All ThreadPoolExecutor tests PASSED ==========" << std::endl;
        return 0;
    } else {
        std::cout << "========== Some ThreadPoolExecutor tests FAILED ==========" << std::endl;
        return 1;
    }
}
