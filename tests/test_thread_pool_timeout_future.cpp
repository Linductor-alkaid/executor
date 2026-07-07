#include <atomic>
#include <chrono>
#include <exception>
#include <future>
#include <iostream>
#include <string>
#include <thread>

#include <executor/config.hpp>
#include <executor/executor.hpp>
#include <executor/monitor/task_monitor.hpp>
#include <executor/thread_pool/thread_pool.hpp>
#include <executor/types.hpp>

using namespace executor;

#define TEST_ASSERT(condition, message)                                      \
    do {                                                                     \
        if (!(condition)) {                                                   \
            std::cerr << "FAILED: " << message << " at " << __FILE__       \
                      << ":" << __LINE__ << std::endl;                       \
            return false;                                                     \
        }                                                                    \
    } while (0)

namespace {

bool future_throws_timed_out(std::future<int>& future) {
    try {
        (void)future.get();
    } catch (const TimedOutException& ex) {
        const std::string message = ex.what();
        return message.find("Task timed out after ") != std::string::npos;
    } catch (const std::future_error& ex) {
        std::cerr << "future.get() threw std::future_error instead: "
                  << ex.what() << std::endl;
        return false;
    } catch (const std::exception& ex) {
        std::cerr << "future.get() threw unexpected exception: "
                  << ex.what() << std::endl;
        return false;
    }
    std::cerr << "future.get() did not throw" << std::endl;
    return false;
}

bool test_thread_pool_timeout_satisfies_future_and_monitor() {
    std::cout << "Testing ThreadPool timeout future and monitor stats..."
              << std::endl;

    ThreadPool pool;
    ThreadPoolConfig config;
    config.min_threads = 1;
    config.max_threads = 1;
    config.task_timeout_ms = 20;
    TEST_ASSERT(pool.initialize(config), "thread pool should initialize");

    monitor::TaskMonitor monitor;
    pool.set_task_monitor(&monitor);

    auto blocker = pool.submit([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
        return 1;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    std::atomic<bool> timed_out_task_ran{false};
    auto timed_out = pool.submit([&timed_out_task_ran]() {
        timed_out_task_ran.store(true, std::memory_order_release);
        return 7;
    });

    TEST_ASSERT(blocker.get() == 1, "blocker task should complete normally");
    TEST_ASSERT(future_throws_timed_out(timed_out),
                "timed-out ThreadPool future should throw TimedOutException");

    pool.wait_for_completion();

    TEST_ASSERT(!timed_out_task_ran.load(std::memory_order_acquire),
                "timed-out task function should not run");
    TEST_ASSERT(pool.get_timeout_count() == 1,
                "ThreadPool timeout_count should be 1");

    auto stats = monitor.get_statistics("default");
    TEST_ASSERT(stats.timeout_count == 1,
                "TaskMonitor timeout_count should be 1");
    TEST_ASSERT(stats.fail_count == 0,
                "TaskMonitor fail_count should not include timeout");

    pool.shutdown();

    std::cout << "  ThreadPool timeout future and monitor stats: PASSED"
              << std::endl;
    return true;
}

bool test_executor_timeout_satisfies_future_and_failure_status() {
    std::cout << "Testing Executor timeout future and failure status..."
              << std::endl;

    Executor executor;
    ExecutorConfig config;
    config.min_threads = 1;
    config.max_threads = 1;
    config.task_timeout_ms = 20;
    TEST_ASSERT(executor.initialize(config), "executor should initialize");

    auto blocker = executor.submit([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
        return 1;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    std::atomic<bool> timed_out_task_ran{false};
    auto timed_out = executor.submit([&timed_out_task_ran]() {
        timed_out_task_ran.store(true, std::memory_order_release);
        return 9;
    });

    TEST_ASSERT(blocker.get() == 1, "facade blocker should complete normally");
    TEST_ASSERT(future_throws_timed_out(timed_out),
                "timed-out Executor future should throw TimedOutException");

    executor.wait_for_completion();

    TEST_ASSERT(!timed_out_task_ran.load(std::memory_order_acquire),
                "timed-out facade task function should not run");

    auto failure_status = executor.get_failure_status();
    TEST_ASSERT(failure_status.timeout_count == 1,
                "ExecutorFailureStatus timeout_count should be 1");
    TEST_ASSERT(failure_status.task_exception_count == 0,
                "timeout should not increment task_exception_count");

    auto async_status = executor.get_async_executor_status();
    TEST_ASSERT(async_status.failed_tasks == 0,
                "timeout should not increment async failed_tasks");

    executor.shutdown();

    std::cout << "  Executor timeout future and failure status: PASSED"
              << std::endl;
    return true;
}

}  // namespace

int main() {
    bool all_passed = true;
    all_passed &= test_thread_pool_timeout_satisfies_future_and_monitor();
    all_passed &= test_executor_timeout_satisfies_future_and_failure_status();

    if (all_passed) {
        std::cout << "All thread pool timeout future tests passed."
                  << std::endl;
        return 0;
    }

    std::cerr << "Some thread pool timeout future tests failed." << std::endl;
    return 1;
}
