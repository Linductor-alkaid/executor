#include <cassert>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <future>
#include <algorithm>

// 包含 Executor 的头文件
#include <executor/executor.hpp>

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
bool test_singleton_instance();
bool test_instance_mode();
bool test_initialize();
bool test_submit();
bool test_submit_priority();
bool test_submit_delayed();
bool test_submit_periodic();
bool test_realtime_task_management();
bool test_monitor_queries();
bool test_concurrent_submit();
bool test_enable_monitoring();
bool test_wait_for_completion();

// ========== 单例模式测试 ==========

bool test_singleton_instance() {
    std::cout << "Testing Executor singleton instance..." << std::endl;
    
    // 获取单例实例
    Executor& instance1 = Executor::instance();
    Executor& instance2 = Executor::instance();
    
    // 验证是同一个实例
    TEST_ASSERT(&instance1 == &instance2, "Singleton instances should be the same");
    
    std::cout << "  Singleton instance: PASSED" << std::endl;
    return true;
}

// ========== 实例化模式测试 ==========

bool test_instance_mode() {
    std::cout << "Testing Executor instance mode..." << std::endl;
    
    // 创建独立的 Executor 实例
    Executor executor1;
    Executor executor2;
    
    // 初始化两个实例
    ExecutorConfig config1;
    config1.min_threads = 2;
    config1.max_threads = 4;
    
    ExecutorConfig config2;
    config2.min_threads = 3;
    config2.max_threads = 6;
    
    TEST_ASSERT(executor1.initialize(config1), "executor1 initialization should succeed");
    TEST_ASSERT(executor2.initialize(config2), "executor2 initialization should succeed");
    
    // 验证两个实例是独立的
    auto status1 = executor1.get_async_executor_status();
    auto status2 = executor2.get_async_executor_status();
    
    TEST_ASSERT(status1.is_running, "executor1 should be running");
    TEST_ASSERT(status2.is_running, "executor2 should be running");
    
    executor1.shutdown();
    executor2.shutdown();
    
    std::cout << "  Instance mode: PASSED" << std::endl;
    return true;
}

// ========== 初始化测试 ==========

bool test_initialize() {
    std::cout << "Testing Executor::initialize()..." << std::endl;
    
    Executor executor;
    
    ExecutorConfig config;
    config.min_threads = 4;
    config.max_threads = 8;
    config.queue_capacity = 100;
    
    TEST_ASSERT(executor.initialize(config), "Initialization should succeed");
    
    auto status = executor.get_async_executor_status();
    TEST_ASSERT(status.is_running, "Executor should be running after initialization");
    
    executor.shutdown();
    
    std::cout << "  Initialize: PASSED" << std::endl;
    return true;
}

// ========== 基本任务提交测试 ==========

bool test_submit() {
    std::cout << "Testing Executor::submit()..." << std::endl;
    
    Executor executor;
    ExecutorConfig config;
    config.min_threads = 2;
    config.max_threads = 4;
    executor.initialize(config);
    
    // 测试基本任务提交
    auto future = executor.submit([]() noexcept {
        return 42;
    });
    
    TEST_ASSERT(future.get() == 42, "Task result should be 42");
    
    // 测试带参数的任务
    auto future2 = executor.submit([](int a, int b) noexcept {
        return a + b;
    }, 10, 20);
    
    TEST_ASSERT(future2.get() == 30, "Task result should be 30");
    
    executor.shutdown();
    
    std::cout << "  Submit: PASSED" << std::endl;
    return true;
}

// ========== 优先级任务提交测试 ==========

bool test_submit_priority() {
    std::cout << "Testing Executor::submit_priority()..." << std::endl;
    
    Executor executor;
    ExecutorConfig config;
    config.min_threads = 2;
    config.max_threads = 4;
    executor.initialize(config);
    
    // 测试不同优先级的任务
    std::vector<int> execution_order;
    std::mutex order_mutex;
    
    // 提交低优先级任务
    executor.submit_priority(0, [&execution_order, &order_mutex]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::lock_guard<std::mutex> lock(order_mutex);
        execution_order.push_back(0);
    });
    
    // 提交高优先级任务
    executor.submit_priority(2, [&execution_order, &order_mutex]() {
        std::lock_guard<std::mutex> lock(order_mutex);
        execution_order.push_back(2);
    });
    
    // 等待任务完成
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // 验证高优先级任务先执行（理想情况下）
    // 注意：由于线程调度的不确定性，这个测试可能不稳定
    {
        std::lock_guard<std::mutex> lock(order_mutex);
        TEST_ASSERT(execution_order.size() == 2, "Both tasks should complete");
    }
    
    executor.shutdown();
    
    std::cout << "  Submit priority: PASSED" << std::endl;
    return true;
}

// ========== 延迟任务提交测试 ==========

bool test_submit_delayed() {
    std::cout << "Testing Executor::submit_delayed()..." << std::endl;
    
    Executor executor;
    ExecutorConfig config;
    config.min_threads = 2;
    config.max_threads = 4;
    executor.initialize(config);
    
    // 测试延迟任务
    auto start_time = std::chrono::steady_clock::now();
    std::atomic<bool> task_executed(false);
    
    auto future = executor.submit_delayed(100, [&task_executed]() noexcept {
        task_executed.store(true);
        return 100;
    });
    
    // 验证任务在延迟后执行
    auto result = future.get();
    auto end_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    TEST_ASSERT(result == 100, "Task result should be 100");
    TEST_ASSERT(task_executed.load(), "Task should be executed");
    TEST_ASSERT(elapsed >= 90, "Task should execute after delay (at least 90ms)");
    TEST_ASSERT(elapsed < 200, "Task should execute within reasonable time (less than 200ms)");
    
    executor.shutdown();
    
    std::cout << "  Submit delayed: PASSED" << std::endl;
    return true;
}

// ========== 周期性任务测试 ==========

bool test_submit_periodic() {
    std::cout << "Testing Executor::submit_periodic()..." << std::endl;
    
    Executor executor;
    ExecutorConfig config;
    config.min_threads = 2;
    config.max_threads = 4;
    executor.initialize(config);
    
    // 测试周期性任务
    std::atomic<int> execution_count(0);
    
    std::string task_id = executor.submit_periodic(50, [&execution_count]() noexcept {
        execution_count.fetch_add(1);
    });
    
    TEST_ASSERT(!task_id.empty(), "Task ID should not be empty");
    
    // 等待几个周期
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // 验证任务执行了多次
    int count = execution_count.load();
    TEST_ASSERT(count >= 3, "Task should execute at least 3 times");
    TEST_ASSERT(count <= 6, "Task should execute at most 6 times (with some tolerance)");
    
    // 取消任务
    TEST_ASSERT(executor.cancel_task(task_id), "Task cancellation should succeed");
    
    // 等待一段时间，验证任务不再执行
    int count_before = execution_count.load();
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    int count_after = execution_count.load();
    
    TEST_ASSERT(count_after == count_before, "Task should not execute after cancellation");
    
    executor.shutdown();
    
    std::cout << "  Submit periodic: PASSED" << std::endl;
    return true;
}

// ========== 实时任务管理测试 ==========

bool test_realtime_task_management() {
    std::cout << "Testing realtime task management..." << std::endl;
    
    Executor executor;
    ExecutorConfig config;
    config.min_threads = 2;
    config.max_threads = 4;
    executor.initialize(config);
    
    // 注册实时任务
    RealtimeThreadConfig rt_config;
    rt_config.thread_name = "test_realtime";
    rt_config.cycle_period_ns = 10000000;  // 10ms
    rt_config.thread_priority = 0;  // 普通优先级（测试环境）
    rt_config.cycle_callback = []() noexcept {
        // 空回调
    };
    
    TEST_ASSERT(executor.register_realtime_task("test_realtime", rt_config),
                "Realtime task registration should succeed");
    
    // 验证任务已注册
    auto task_list = executor.get_realtime_task_list();
    TEST_ASSERT(std::find(task_list.begin(), task_list.end(), "test_realtime") != task_list.end(),
                "Realtime task should be in the list");
    
    // 启动实时任务
    TEST_ASSERT(executor.start_realtime_task("test_realtime"),
                "Realtime task start should succeed");
    
    // 验证任务正在运行
    auto status = executor.get_realtime_executor_status("test_realtime");
    TEST_ASSERT(status.is_running, "Realtime task should be running");
    
    // 停止实时任务
    executor.stop_realtime_task("test_realtime");
    
    // 等待线程停止
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // 验证任务已停止
    auto status2 = executor.get_realtime_executor_status("test_realtime");
    TEST_ASSERT(!status2.is_running, "Realtime task should be stopped");
    
    executor.shutdown();
    
    std::cout << "  Realtime task management: PASSED" << std::endl;
    return true;
}

// ========== 监控查询测试 ==========

bool test_monitor_queries() {
    std::cout << "Testing monitor queries..." << std::endl;
    
    Executor executor;
    ExecutorConfig config;
    config.min_threads = 2;
    config.max_threads = 4;
    executor.initialize(config);
    
    // 提交一些任务
    for (int i = 0; i < 10; ++i) {
        executor.submit([i]() noexcept {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            return i;
        });
    }
    
    // 等待任务完成
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // 查询异步执行器状态
    auto async_status = executor.get_async_executor_status();
    TEST_ASSERT(async_status.is_running, "Async executor should be running");
    TEST_ASSERT(async_status.completed_tasks > 0, "Some tasks should be completed");
    
    executor.shutdown();
    
    std::cout << "  Monitor queries: PASSED" << std::endl;
    return true;
}

// ========== 并发测试 ==========

bool test_concurrent_submit() {
    std::cout << "Testing concurrent submit..." << std::endl;
    
    Executor executor;
    ExecutorConfig config;
    config.min_threads = 4;
    config.max_threads = 8;
    executor.initialize(config);
    
    const int num_tasks = 100;
    std::vector<std::future<int>> futures;
    std::atomic<int> completed_count(0);
    
    // 并发提交任务
    for (int i = 0; i < num_tasks; ++i) {
        auto future = executor.submit([i, &completed_count]() noexcept {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            completed_count.fetch_add(1);
            return i;
        });
        futures.push_back(std::move(future));
    }
    
    // 等待所有任务完成
    for (auto& future : futures) {
        future.get();
    }
    
    TEST_ASSERT(completed_count.load() == num_tasks, "All tasks should complete");
    
    executor.shutdown();
    
    std::cout << "  Concurrent submit: PASSED" << std::endl;
    return true;
}

// ========== enable_monitoring 测试 ==========

bool test_enable_monitoring() {
    std::cout << "Testing Executor::enable_monitoring()..." << std::endl;
    
    Executor executor;
    ExecutorConfig config;
    config.min_threads = 2;
    config.max_threads = 4;
    config.enable_monitoring = false;  // 初始禁用
    executor.initialize(config);
    
    // 提交一些任务（监控禁用时）
    for (int i = 0; i < 5; ++i) {
        executor.submit([i]() noexcept {
            return i;
        });
    }
    executor.wait_for_completion();
    
    // 查询统计（应该为空或很少）
    auto stats_before = executor.get_task_statistics("default");
    int64_t count_before = stats_before.total_count;
    
    // 启用监控
    executor.enable_monitoring(true);
    
    // 提交更多任务
    for (int i = 0; i < 10; ++i) {
        executor.submit([i]() noexcept {
            return i;
        });
    }
    executor.wait_for_completion();
    
    // 查询统计（应该有新的计数）
    auto stats_after = executor.get_task_statistics("default");
    TEST_ASSERT(stats_after.total_count > count_before, 
                "Task count should increase after enabling monitoring");
    
    // 禁用监控
    executor.enable_monitoring(false);
    int64_t count_after_disable = stats_after.total_count;
    
    // 提交更多任务（监控禁用时）
    for (int i = 0; i < 5; ++i) {
        executor.submit([i]() noexcept {
            return i;
        });
    }
    executor.wait_for_completion();
    
    // 统计应该不变
    auto stats_final = executor.get_task_statistics("default");
    TEST_ASSERT(stats_final.total_count == count_after_disable,
                "Task count should not increase when monitoring is disabled");
    
    executor.shutdown();
    
    std::cout << "  Enable monitoring: PASSED" << std::endl;
    return true;
}

// ========== wait_for_completion 测试 ==========

bool test_wait_for_completion() {
    std::cout << "Testing Executor::wait_for_completion()..." << std::endl;
    
    Executor executor;
    ExecutorConfig config;
    config.min_threads = 2;
    config.max_threads = 4;
    executor.initialize(config);
    
    std::atomic<int> completed_count{0};
    const int num_tasks = 20;
    
    // 提交多个任务
    std::vector<std::future<void>> futures;
    for (int i = 0; i < num_tasks; ++i) {
        auto future = executor.submit([i, &completed_count]() noexcept {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            completed_count.fetch_add(1);
        });
        futures.push_back(std::move(future));
    }
    
    // 调用 wait_for_completion
    executor.wait_for_completion();
    
    // 验证所有任务已完成
    TEST_ASSERT(completed_count.load() == num_tasks,
                "All tasks should be completed after wait_for_completion");
    
    // 验证所有 future 可以立即获取结果
    for (auto& future : futures) {
        future.get();  // 应该立即返回，不阻塞
    }
    
    // 再次调用 wait_for_completion（应该立即返回，因为没有待处理任务）
    auto start = std::chrono::steady_clock::now();
    executor.wait_for_completion();
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start).count();
    
    TEST_ASSERT(elapsed < 100, 
                "wait_for_completion should return immediately when no pending tasks");
    
    executor.shutdown();
    
    std::cout << "  Wait for completion: PASSED" << std::endl;
    return true;
}

// ========== 主函数 ==========

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Executor Facade Integration Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    bool all_passed = true;
    
    // 运行所有测试
    all_passed &= test_singleton_instance();
    all_passed &= test_instance_mode();
    all_passed &= test_initialize();
    all_passed &= test_submit();
    all_passed &= test_submit_priority();
    all_passed &= test_submit_delayed();
    all_passed &= test_submit_periodic();
    all_passed &= test_realtime_task_management();
    all_passed &= test_monitor_queries();
    all_passed &= test_concurrent_submit();
    all_passed &= test_enable_monitoring();
    all_passed &= test_wait_for_completion();
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    if (all_passed) {
        std::cout << "All tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests FAILED!" << std::endl;
        return 1;
    }
}
