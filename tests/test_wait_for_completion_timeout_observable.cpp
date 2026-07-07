#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <thread>

#include <executor/config.hpp>
#include <executor/executor.hpp>
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

bool test_try_wait_for_completion_returns_true_when_complete() {
    std::cout << "Testing Executor try_wait_for_completion completes..."
              << std::endl;

    Executor executor;
    ExecutorConfig config;
    config.min_threads = 1;
    config.max_threads = 1;
    TEST_ASSERT(executor.initialize(config), "executor should initialize");

    std::atomic<bool> ran{false};
    auto future = executor.submit([&ran]() {
        ran.store(true, std::memory_order_release);
    });

    TEST_ASSERT(executor.try_wait_for_completion(std::chrono::seconds(1)),
                "try_wait_for_completion should return true after completion");
    future.get();
    TEST_ASSERT(ran.load(std::memory_order_acquire), "task should run");

    auto status = executor.get_failure_status();
    TEST_ASSERT(status.wait_timeout_count == 0,
                "completed wait should not increment wait_timeout_count");

    executor.shutdown();

    std::cout << "  Executor try_wait_for_completion completes: PASSED"
              << std::endl;
    return true;
}

bool test_try_wait_for_completion_returns_false_on_timeout() {
    std::cout << "Testing Executor try_wait_for_completion timeout..."
              << std::endl;

    Executor executor;
    ExecutorConfig config;
    config.min_threads = 1;
    config.max_threads = 1;
    TEST_ASSERT(executor.initialize(config), "executor should initialize");

    auto future = executor.submit([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    });

    const bool completed =
        executor.try_wait_for_completion(std::chrono::milliseconds(50));
    TEST_ASSERT(!completed,
                "try_wait_for_completion should return false on timeout");

    future.get();
    TEST_ASSERT(executor.try_wait_for_completion(std::chrono::seconds(1)),
                "try_wait_for_completion should later return true");

    auto status = executor.get_failure_status();
    TEST_ASSERT(status.wait_timeout_count == 1,
                "timeout should increment wait_timeout_count");
    TEST_ASSERT(status.total_count == 1,
                "timeout should increment total failure count");

    executor.shutdown();

    std::cout << "  Executor try_wait_for_completion timeout: PASSED"
              << std::endl;
    return true;
}

bool test_wait_for_completion_records_wait_timeout_on_timeout_path() {
    std::cout << "Testing wait timeout failure event details..." << std::endl;

    Executor executor;
    ExecutorConfig config;
    config.min_threads = 1;
    config.max_threads = 1;
    TEST_ASSERT(executor.initialize(config), "executor should initialize");

    std::atomic<int> callback_count{0};
    executor.set_failure_callback([&callback_count](const ExecutorFailureEvent& event) {
        if (event.kind == FailureKind::WaitTimeout) {
            callback_count.fetch_add(1, std::memory_order_relaxed);
        }
    });

    auto future = executor.submit([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    });

    TEST_ASSERT(!executor.try_wait_for_completion(std::chrono::milliseconds(50)),
                "timeout path shared by wait_for_completion should be observable");

    auto status = executor.get_failure_status();
    TEST_ASSERT(status.wait_timeout_count == 1,
                "WaitTimeout should be counted");
    TEST_ASSERT(status.task_exception_count == 0,
                "WaitTimeout should not be a task exception");
    TEST_ASSERT(callback_count.load(std::memory_order_relaxed) == 1,
                "WaitTimeout should invoke failure callback");

    auto recent = executor.get_recent_failures(1);
    TEST_ASSERT(recent.size() == 1, "recent failure should contain timeout");
    TEST_ASSERT(recent[0].kind == FailureKind::WaitTimeout,
                "recent failure kind should be WaitTimeout");
    TEST_ASSERT(recent[0].executor_name == "default",
                "recent failure should name default executor");

    future.get();
    executor.shutdown();

    auto async_status = executor.get_async_executor_status();
    TEST_ASSERT(!async_status.is_running,
                "executor should not be running after shutdown");

    std::cout << "  Wait timeout failure event details: PASSED" << std::endl;
    return true;
}

bool test_thread_pool_try_wait_for_completion_basic() {
    std::cout << "Testing ThreadPool try_wait_for_completion..." << std::endl;

    ThreadPool pool;
    ThreadPoolConfig config;
    config.min_threads = 1;
    config.max_threads = 1;
    TEST_ASSERT(pool.initialize(config), "thread pool should initialize");

    auto slow = pool.submit([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    });
    TEST_ASSERT(!pool.try_wait_for_completion(std::chrono::milliseconds(50)),
                "ThreadPool try_wait_for_completion should return false on timeout");

    slow.get();
    TEST_ASSERT(pool.try_wait_for_completion(std::chrono::seconds(1)),
                "ThreadPool try_wait_for_completion should return true after completion");

    pool.shutdown();

    std::cout << "  ThreadPool try_wait_for_completion: PASSED" << std::endl;
    return true;
}

}  // namespace

int main() {
    bool all_passed = true;
    all_passed &= test_try_wait_for_completion_returns_true_when_complete();
    all_passed &= test_try_wait_for_completion_returns_false_on_timeout();
    all_passed &= test_wait_for_completion_records_wait_timeout_on_timeout_path();
    all_passed &= test_thread_pool_try_wait_for_completion_basic();

    if (all_passed) {
        std::cout << "All wait_for_completion timeout observability tests passed."
                  << std::endl;
        return 0;
    }

    std::cerr << "Some wait_for_completion timeout observability tests failed."
              << std::endl;
    return 1;
}
