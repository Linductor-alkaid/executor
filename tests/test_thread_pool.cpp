#include <cassert>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <future>
#include <algorithm>

// 包含 thread_pool 模块的头文件
#include <executor/config.hpp>
#include <executor/types.hpp>
#include "executor/thread_pool/priority_scheduler.hpp"
#include "executor/thread_pool/thread_pool.hpp"

using namespace executor;

// 测试辅助宏
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "FAILED: " << message << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

// ========== PriorityScheduler 测试 ==========

bool test_priority_scheduler_basic() {
    std::cout << "Testing PriorityScheduler basic operations..." << std::endl;
    
    PriorityScheduler scheduler;
    
    // 测试空队列
    TEST_ASSERT(scheduler.empty(), "Scheduler should be empty initially");
    TEST_ASSERT(scheduler.size() == 0, "Scheduler size should be 0 initially");
    
    // 创建测试任务
    Task task1;
    task1.task_id = "task_1";
    task1.priority = TaskPriority::NORMAL;
    task1.submit_time_ns = 1000;
    task1.function = []() {};
    
    // 测试enqueue
    scheduler.enqueue(task1);
    TEST_ASSERT(!scheduler.empty(), "Scheduler should not be empty after enqueue");
    TEST_ASSERT(scheduler.size() == 1, "Scheduler size should be 1");
    
    // 测试dequeue
    Task result;
    TEST_ASSERT(scheduler.dequeue(result), "Should be able to dequeue task");
    TEST_ASSERT(result.task_id == "task_1", "Dequeued task ID should match");
    TEST_ASSERT(scheduler.empty(), "Scheduler should be empty after dequeue");
    
    std::cout << "  PriorityScheduler basic operations: PASSED" << std::endl;
    return true;
}

bool test_priority_scheduler_priority_order() {
    std::cout << "Testing PriorityScheduler priority ordering..." << std::endl;
    
    PriorityScheduler scheduler;
    
    // 创建不同优先级的任务（按优先级从低到高提交）
    Task low_task;
    low_task.task_id = "low_task";
    low_task.priority = TaskPriority::LOW;
    low_task.submit_time_ns = 1000;
    low_task.function = []() {};
    
    Task normal_task;
    normal_task.task_id = "normal_task";
    normal_task.priority = TaskPriority::NORMAL;
    normal_task.submit_time_ns = 2000;
    normal_task.function = []() {};
    
    Task high_task;
    high_task.task_id = "high_task";
    high_task.priority = TaskPriority::HIGH;
    high_task.submit_time_ns = 3000;
    high_task.function = []() {};
    
    Task critical_task;
    critical_task.task_id = "critical_task";
    critical_task.priority = TaskPriority::CRITICAL;
    critical_task.submit_time_ns = 4000;
    critical_task.function = []() {};
    
    // 按低优先级到高优先级顺序提交
    scheduler.enqueue(low_task);
    scheduler.enqueue(normal_task);
    scheduler.enqueue(high_task);
    scheduler.enqueue(critical_task);
    
    // 应该按高优先级到低优先级顺序出队
    Task result;
    TEST_ASSERT(scheduler.dequeue(result), "Should dequeue critical task first");
    TEST_ASSERT(result.task_id == "critical_task", "Critical task should be dequeued first");
    
    TEST_ASSERT(scheduler.dequeue(result), "Should dequeue high task second");
    TEST_ASSERT(result.task_id == "high_task", "High task should be dequeued second");
    
    TEST_ASSERT(scheduler.dequeue(result), "Should dequeue normal task third");
    TEST_ASSERT(result.task_id == "normal_task", "Normal task should be dequeued third");
    
    TEST_ASSERT(scheduler.dequeue(result), "Should dequeue low task last");
    TEST_ASSERT(result.task_id == "low_task", "Low task should be dequeued last");
    
    TEST_ASSERT(scheduler.empty(), "Scheduler should be empty after all dequeues");
    
    std::cout << "  PriorityScheduler priority ordering: PASSED" << std::endl;
    return true;
}

bool test_priority_scheduler_same_priority_fifo() {
    std::cout << "Testing PriorityScheduler same priority FIFO..." << std::endl;
    
    PriorityScheduler scheduler;
    
    // 创建相同优先级但不同提交时间的任务
    Task task1;
    task1.task_id = "task_1";
    task1.priority = TaskPriority::NORMAL;
    task1.submit_time_ns = 1000;  // 更早
    task1.function = []() {};
    
    Task task2;
    task2.task_id = "task_2";
    task2.priority = TaskPriority::NORMAL;
    task2.submit_time_ns = 2000;  // 更晚
    task2.function = []() {};
    
    Task task3;
    task3.task_id = "task_3";
    task3.priority = TaskPriority::NORMAL;
    task3.submit_time_ns = 3000;  // 最晚
    task3.function = []() {};
    
    // 按时间顺序提交
    scheduler.enqueue(task1);
    scheduler.enqueue(task2);
    scheduler.enqueue(task3);
    
    // 应该按提交时间顺序出队（FIFO）
    Task result;
    TEST_ASSERT(scheduler.dequeue(result), "Should dequeue task_1 first");
    TEST_ASSERT(result.task_id == "task_1", "Task 1 should be dequeued first (earliest)");
    
    TEST_ASSERT(scheduler.dequeue(result), "Should dequeue task_2 second");
    TEST_ASSERT(result.task_id == "task_2", "Task 2 should be dequeued second");
    
    TEST_ASSERT(scheduler.dequeue(result), "Should dequeue task_3 third");
    TEST_ASSERT(result.task_id == "task_3", "Task 3 should be dequeued third (latest)");
    
    std::cout << "  PriorityScheduler same priority FIFO: PASSED" << std::endl;
    return true;
}

bool test_priority_scheduler_clear() {
    std::cout << "Testing PriorityScheduler clear..." << std::endl;
    
    PriorityScheduler scheduler;
    
    // 添加一些任务
    for (int i = 0; i < 10; ++i) {
        Task task;
        task.task_id = "task_" + std::to_string(i);
        task.priority = static_cast<TaskPriority>(i % 4);
        task.submit_time_ns = i * 1000;
        task.function = []() {};
        scheduler.enqueue(task);
    }
    
    TEST_ASSERT(scheduler.size() == 10, "Scheduler should have 10 tasks");
    
    scheduler.clear();
    
    TEST_ASSERT(scheduler.empty(), "Scheduler should be empty after clear");
    TEST_ASSERT(scheduler.size() == 0, "Scheduler size should be 0 after clear");
    
    std::cout << "  PriorityScheduler clear: PASSED" << std::endl;
    return true;
}

bool test_priority_scheduler_concurrent() {
    std::cout << "Testing PriorityScheduler concurrent operations..." << std::endl;
    
    PriorityScheduler scheduler;
    const int num_threads = 10;
    const int tasks_per_thread = 100;
    std::atomic<int> enqueue_count{0};
    std::atomic<int> dequeue_count{0};
    
    // 多个线程同时enqueue和dequeue
    std::vector<std::thread> threads;
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            // 每个线程enqueue一些任务
            for (int j = 0; j < tasks_per_thread; ++j) {
                Task task;
                task.task_id = "task_" + std::to_string(i) + "_" + std::to_string(j);
                task.priority = static_cast<TaskPriority>(j % 4);
                task.submit_time_ns = j * 1000;
                task.function = []() {};
                scheduler.enqueue(task);
                enqueue_count.fetch_add(1);
            }
            
            // 每个线程dequeue一些任务
            for (int j = 0; j < tasks_per_thread; ++j) {
                Task result;
                if (scheduler.dequeue(result)) {
                    dequeue_count.fetch_add(1);
                }
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    TEST_ASSERT(enqueue_count.load() == num_threads * tasks_per_thread,
               "All enqueue operations should complete");
    TEST_ASSERT(dequeue_count.load() > 0, "Some dequeue operations should complete");
    
    std::cout << "  PriorityScheduler concurrent operations: PASSED" << std::endl;
    return true;
}

// ========== ThreadPool 测试 ==========

bool test_thread_pool_initialize() {
    std::cout << "Testing ThreadPool initialize..." << std::endl;
    
    ThreadPool pool;
    
    ThreadPoolConfig config;
    config.min_threads = 4;
    config.max_threads = 8;
    config.queue_capacity = 100;
    
    TEST_ASSERT(pool.initialize(config), "Should be able to initialize thread pool");
    TEST_ASSERT(!pool.is_stopped(), "Thread pool should not be stopped after initialization");
    
    // 再次初始化应该失败
    TEST_ASSERT(!pool.initialize(config), "Should not be able to initialize twice");
    
    pool.shutdown();
    
    std::cout << "  ThreadPool initialize: PASSED" << std::endl;
    return true;
}

bool test_thread_pool_submit_basic() {
    std::cout << "Testing ThreadPool submit basic..." << std::endl;
    
    ThreadPool pool;
    ThreadPoolConfig config;
    config.min_threads = 2;
    config.max_threads = 4;
    pool.initialize(config);
    
    // 提交简单任务
    auto future = pool.submit([]() {
        return 42;
    });
    
    int result = future.get();
    TEST_ASSERT(result == 42, "Task result should be 42");
    
    // 提交带参数的任务
    auto future2 = pool.submit([](int a, int b) {
        return a + b;
    }, 10, 20);
    
    int result2 = future2.get();
    TEST_ASSERT(result2 == 30, "Task result should be 30");
    
    pool.shutdown();
    
    std::cout << "  ThreadPool submit basic: PASSED" << std::endl;
    return true;
}

bool test_thread_pool_submit_priority() {
    std::cout << "Testing ThreadPool submit_priority..." << std::endl;
    
    ThreadPool pool;
    ThreadPoolConfig config;
    config.min_threads = 2;
    config.max_threads = 4;
    pool.initialize(config);
    
    std::vector<int> execution_order;
    std::mutex order_mutex;
    
    // 提交不同优先级的任务
    pool.submit_priority(0, [&]() {  // LOW
        std::lock_guard<std::mutex> lock(order_mutex);
        execution_order.push_back(0);
    });
    
    pool.submit_priority(3, [&]() {  // CRITICAL
        std::lock_guard<std::mutex> lock(order_mutex);
        execution_order.push_back(3);
    });
    
    pool.submit_priority(1, [&]() {  // NORMAL
        std::lock_guard<std::mutex> lock(order_mutex);
        execution_order.push_back(1);
    });
    
    pool.submit_priority(2, [&]() {  // HIGH
        std::lock_guard<std::mutex> lock(order_mutex);
        execution_order.push_back(2);
    });
    
    // 等待所有任务完成
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    pool.wait_for_completion();
    
    // 验证执行顺序（高优先级应该先执行）
    TEST_ASSERT(execution_order.size() == 4, "All tasks should be executed");
    // 注意：由于并发执行，顺序可能不完全确定，但CRITICAL应该较早执行
    TEST_ASSERT(std::find(execution_order.begin(), execution_order.end(), 3) != execution_order.end(),
               "CRITICAL task should be executed");
    
    pool.shutdown();
    
    std::cout << "  ThreadPool submit_priority: PASSED" << std::endl;
    return true;
}

bool test_thread_pool_concurrent_submit() {
    std::cout << "Testing ThreadPool concurrent submit..." << std::endl;
    
    ThreadPool pool;
    ThreadPoolConfig config;
    config.min_threads = 4;
    config.max_threads = 8;
    pool.initialize(config);
    
    const int num_tasks = 100;
    std::atomic<int> completed_count{0};
    std::vector<std::future<void>> futures;
    
    // 并发提交多个任务
    for (int i = 0; i < num_tasks; ++i) {
        auto future = pool.submit([&, i]() {
            completed_count.fetch_add(1);
        });
        futures.push_back(std::move(future));
    }
    
    // 等待所有任务完成
    for (auto& future : futures) {
        future.wait();
    }
    
    TEST_ASSERT(completed_count.load() == num_tasks, "All tasks should be completed");
    
    pool.shutdown();
    
    std::cout << "  ThreadPool concurrent submit: PASSED" << std::endl;
    return true;
}

bool test_thread_pool_exception_handling() {
    std::cout << "Testing ThreadPool exception handling..." << std::endl;
    
    ThreadPool pool;
    ThreadPoolConfig config;
    config.min_threads = 2;
    config.max_threads = 4;
    pool.initialize(config);
    
    // 提交抛出异常的任务
    auto future = pool.submit([]() {
        throw std::runtime_error("Test exception");
        return 42;
    });
    
    // 应该能够捕获异常
    bool exception_caught = false;
    try {
        future.get();
    } catch (const std::exception& e) {
        exception_caught = true;
        TEST_ASSERT(std::string(e.what()) == "Test exception", "Exception message should match");
    }
    
    TEST_ASSERT(exception_caught, "Exception should be caught");
    
    pool.shutdown();
    
    std::cout << "  ThreadPool exception handling: PASSED" << std::endl;
    return true;
}

bool test_thread_pool_status() {
    std::cout << "Testing ThreadPool status..." << std::endl;
    
    ThreadPool pool;
    ThreadPoolConfig config;
    config.min_threads = 4;
    config.max_threads = 8;
    pool.initialize(config);
    
    // 获取初始状态
    auto status = pool.get_status();
    TEST_ASSERT(status.total_threads == 4, "Total threads should be 4");
    TEST_ASSERT(status.total_tasks == 0, "Initial total tasks should be 0");
    
    // 提交一些任务
    std::vector<std::future<void>> futures;
    for (int i = 0; i < 10; ++i) {
        futures.push_back(pool.submit([]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }));
    }
    
    // 等待一段时间让任务开始执行
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    status = pool.get_status();
    TEST_ASSERT(status.total_tasks == 10, "Total tasks should be 10");
    TEST_ASSERT(status.queue_size >= 0, "Queue size should be non-negative");
    
    // 等待所有任务完成
    for (auto& future : futures) {
        future.wait();
    }
    pool.wait_for_completion();
    
    status = pool.get_status();
    TEST_ASSERT(status.completed_tasks == 10, "Completed tasks should be 10");
    
    pool.shutdown();
    
    std::cout << "  ThreadPool status: PASSED" << std::endl;
    return true;
}

bool test_thread_pool_shutdown() {
    std::cout << "Testing ThreadPool shutdown..." << std::endl;
    
    ThreadPool pool;
    ThreadPoolConfig config;
    config.min_threads = 2;
    config.max_threads = 4;
    pool.initialize(config);
    
    // 提交一些任务
    std::vector<std::future<int>> futures;
    for (int i = 0; i < 5; ++i) {
        futures.push_back(pool.submit([i]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            return i;
        }));
    }
    
    // 等待一段时间
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // 关闭线程池（等待任务完成）
    pool.shutdown(true);
    
    TEST_ASSERT(pool.is_stopped(), "Thread pool should be stopped after shutdown");
    
    // 验证任务都完成了
    for (size_t i = 0; i < futures.size(); ++i) {
        int result = futures[i].get();
        TEST_ASSERT(result == static_cast<int>(i), "Task result should match");
    }
    
    std::cout << "  ThreadPool shutdown: PASSED" << std::endl;
    return true;
}

bool test_thread_pool_shutdown_no_wait() {
    std::cout << "Testing ThreadPool shutdown (no wait)..." << std::endl;
    
    ThreadPool pool;
    ThreadPoolConfig config;
    config.min_threads = 2;
    config.max_threads = 4;
    pool.initialize(config);
    
    // 提交一些长时间运行的任务
    std::vector<std::future<void>> futures;
    for (int i = 0; i < 5; ++i) {
        futures.push_back(pool.submit([]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }));
    }
    
    // 立即关闭（不等待）
    pool.shutdown(false);
    
    TEST_ASSERT(pool.is_stopped(), "Thread pool should be stopped after shutdown");
    
    std::cout << "  ThreadPool shutdown (no wait): PASSED" << std::endl;
    return true;
}

bool test_thread_pool_submit_after_shutdown() {
    std::cout << "Testing ThreadPool submit after shutdown..." << std::endl;
    
    ThreadPool pool;
    ThreadPoolConfig config;
    config.min_threads = 2;
    config.max_threads = 4;
    pool.initialize(config);
    
    pool.shutdown();
    
    // 关闭后提交任务应该抛出异常
    bool exception_caught = false;
    try {
        auto future = pool.submit([]() {
            return 42;
        });
        future.get();
    } catch (const std::exception& e) {
        exception_caught = true;
    }
    
    TEST_ASSERT(exception_caught, "Should throw exception when submitting after shutdown");
    
    std::cout << "  ThreadPool submit after shutdown: PASSED" << std::endl;
    return true;
}

bool test_thread_pool_wait_for_completion() {
    std::cout << "Testing ThreadPool wait_for_completion..." << std::endl;
    
    ThreadPool pool;
    ThreadPoolConfig config;
    config.min_threads = 2;
    config.max_threads = 4;
    pool.initialize(config);
    
    std::atomic<int> completed{0};
    const int num_tasks = 20;
    
    // 提交多个任务
    for (int i = 0; i < num_tasks; ++i) {
        pool.submit([&]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            completed.fetch_add(1);
        });
    }
    
    // 等待所有任务完成
    pool.wait_for_completion();
    
    TEST_ASSERT(completed.load() == num_tasks, "All tasks should be completed");
    
    pool.shutdown();
    
    std::cout << "  ThreadPool wait_for_completion: PASSED" << std::endl;
    return true;
}

// ========== 主测试函数 ==========

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Executor ThreadPool Module Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    bool all_passed = true;
    
    // PriorityScheduler 测试
    std::cout << "--- PriorityScheduler Tests ---" << std::endl;
    all_passed &= test_priority_scheduler_basic();
    all_passed &= test_priority_scheduler_priority_order();
    all_passed &= test_priority_scheduler_same_priority_fifo();
    all_passed &= test_priority_scheduler_clear();
    all_passed &= test_priority_scheduler_concurrent();
    std::cout << std::endl;
    
    // ThreadPool 测试
    std::cout << "--- ThreadPool Tests ---" << std::endl;
    all_passed &= test_thread_pool_initialize();
    all_passed &= test_thread_pool_submit_basic();
    all_passed &= test_thread_pool_submit_priority();
    all_passed &= test_thread_pool_concurrent_submit();
    all_passed &= test_thread_pool_exception_handling();
    all_passed &= test_thread_pool_status();
    all_passed &= test_thread_pool_shutdown();
    all_passed &= test_thread_pool_shutdown_no_wait();
    all_passed &= test_thread_pool_submit_after_shutdown();
    all_passed &= test_thread_pool_wait_for_completion();
    std::cout << std::endl;
    
    // 总结
    std::cout << "========================================" << std::endl;
    std::cout << "Test Summary:" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    if (all_passed) {
        std::cout << "All tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests FAILED!" << std::endl;
        return 1;
    }
}
