#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <executor/executor.hpp>

using namespace executor;

#define TEST_ASSERT(condition, message)                                      \
    do {                                                                    \
        if (!(condition)) {                                                  \
            std::cerr << "FAILED: " << message << " at " << __FILE__      \
                      << ":" << __LINE__ << std::endl;                     \
            return false;                                                    \
        }                                                                   \
    } while (0)

bool test_failure_status_recent_events_and_callback() {
    std::cout << "Testing Executor failure status and recent events..."
              << std::endl;

    Executor executor;
    executor.set_recent_failure_capacity(2);

    std::atomic<int> callback_count{0};
    std::atomic<bool> callback_saw_expected_kind{true};
    executor.set_failure_callback([&callback_count, &callback_saw_expected_kind](
                                      const ExecutorFailureEvent& event) {
        if (event.kind != FailureKind::SubmitRejected) {
            callback_saw_expected_kind.store(false, std::memory_order_relaxed);
        }
        callback_count.fetch_add(1, std::memory_order_relaxed);
        throw std::runtime_error("observer failure should be isolated");
    });

    TEST_ASSERT(!executor.start_realtime_task("missing_rt_a"),
                "starting missing realtime task should fail visibly");
    TEST_ASSERT(!executor.start_realtime_task("missing_rt_b"),
                "starting second missing realtime task should fail visibly");
    TEST_ASSERT(!executor.start_realtime_task("missing_rt_c"),
                "starting third missing realtime task should fail visibly");

    auto status = executor.get_failure_status();
    TEST_ASSERT(status.total_count == 3, "total failure count should be 3");
    TEST_ASSERT(status.submit_rejected_count == 3,
                "submit rejection count should be 3");
    TEST_ASSERT(status.task_exception_count == 0,
                "task exception count should remain 0 in phase 1 plumbing test");
    TEST_ASSERT(callback_count.load(std::memory_order_relaxed) == 3,
                "callback should be invoked for every event");
    TEST_ASSERT(callback_saw_expected_kind.load(std::memory_order_relaxed),
                "callback should receive submit rejection events");

    auto recent = executor.get_recent_failures();
    TEST_ASSERT(recent.size() == 2, "recent failure buffer should keep capacity");
    TEST_ASSERT(recent[0].executor_name == "missing_rt_b",
                "oldest retained event should be missing_rt_b");
    TEST_ASSERT(recent[1].executor_name == "missing_rt_c",
                "newest retained event should be missing_rt_c");
    TEST_ASSERT(recent[0].timestamp <= recent[1].timestamp,
                "recent events should be ordered from old to new");

    auto latest_one = executor.get_recent_failures(1);
    TEST_ASSERT(latest_one.size() == 1, "max_count should limit result size");
    TEST_ASSERT(latest_one[0].executor_name == "missing_rt_c",
                "limited recent query should return newest event");

    executor.clear_recent_failures();
    TEST_ASSERT(executor.get_recent_failures().empty(),
                "clear_recent_failures should clear only recent events");
    status = executor.get_failure_status();
    TEST_ASSERT(status.total_count == 3,
                "clear_recent_failures should not reset counters");

    std::cout << "  Failure status and recent events: PASSED" << std::endl;
    return true;
}

bool test_recent_failure_capacity_zero_keeps_counters() {
    std::cout << "Testing Executor recent failure capacity zero..." << std::endl;

    Executor executor;
    executor.set_recent_failure_capacity(0);

    TEST_ASSERT(!executor.start_realtime_task("missing_rt"),
                "starting missing realtime task should fail visibly");

    auto status = executor.get_failure_status();
    TEST_ASSERT(status.total_count == 1, "counter should update at capacity 0");
    TEST_ASSERT(status.submit_rejected_count == 1,
                "submit rejection counter should update at capacity 0");
    TEST_ASSERT(executor.get_recent_failures().empty(),
                "capacity 0 should disable recent event retention");

    std::cout << "  Recent failure capacity zero: PASSED" << std::endl;
    return true;
}

int main() {
    bool all_passed = true;
    all_passed &= test_failure_status_recent_events_and_callback();
    all_passed &= test_recent_failure_capacity_zero_keeps_counters();

    if (all_passed) {
        std::cout << "All executor failure observability tests passed."
                  << std::endl;
        return 0;
    }

    std::cerr << "Some executor failure observability tests failed."
              << std::endl;
    return 1;
}
