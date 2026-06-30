#include "interleaving_harness.hpp"
#include "operation_history.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using executor::test::harness::Operation;
using executor::test::harness::OperationHistory;
using executor::test::harness::Stepper;
using namespace std::chrono_literals;

TEST(StepperTest, ArriveWaitRelease) {
    Stepper stepper(500ms);
    std::atomic<bool> released{false};

    std::thread worker([&] {
        stepper.arrive("worker_ready");
        stepper.wait_for("allow_worker");
        released.store(true, std::memory_order_release);
        stepper.arrive("worker_done");
    });

    EXPECT_TRUE(stepper.wait_for("worker_ready", 500ms));
    EXPECT_FALSE(released.load(std::memory_order_acquire));

    stepper.release("allow_worker");
    stepper.wait_for("worker_done");
    worker.join();

    EXPECT_TRUE(released.load(std::memory_order_acquire));
}

TEST(StepperTest, WaitTimeoutReturnsFalseAndThrowingWaitNamesPoint) {
    Stepper stepper(20ms);

    EXPECT_FALSE(stepper.wait_for("never_arrives", 20ms));

    try {
        stepper.wait_for("missing_point");
        FAIL() << "wait_for without explicit timeout should throw on timeout";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("missing_point"), std::string::npos);
    }
}

TEST(StepperTest, ReleaseAllWakesMultipleWaiters) {
    Stepper stepper(500ms);
    std::atomic<int> ready{0};
    std::atomic<int> passed{0};
    std::vector<std::thread> waiters;

    for (int i = 0; i < 3; ++i) {
        waiters.emplace_back([&] {
            ready.fetch_add(1, std::memory_order_acq_rel);
            stepper.wait_for("shared_gate");
            passed.fetch_add(1, std::memory_order_acq_rel);
        });
    }

    while (ready.load(std::memory_order_acquire) != 3) {
        std::this_thread::yield();
    }

    stepper.release_all("shared_gate");

    for (std::thread& waiter : waiters) {
        waiter.join();
    }

    EXPECT_EQ(passed.load(std::memory_order_acquire), 3);
}

TEST(StepperTest, WaitForArrivalCountAndReset) {
    Stepper stepper(500ms);

    std::thread worker([&] {
        stepper.arrive("batch_point");
        stepper.arrive("batch_point");
        stepper.arrive("batch_point");
    });

    EXPECT_TRUE(stepper.wait_for_arrivals("batch_point", 3, 500ms));
    worker.join();

    EXPECT_TRUE(stepper.has_arrived("batch_point"));
    EXPECT_EQ(stepper.arrival_count("batch_point"), 3U);

    stepper.reset("batch_point");
    EXPECT_FALSE(stepper.has_arrived("batch_point"));
    EXPECT_EQ(stepper.arrival_count("batch_point"), 0U);

    stepper.arrive("other_point");
    stepper.reset_all();
    EXPECT_FALSE(stepper.has_arrived("other_point"));
}

TEST(StepperTest, WaitForArrivalCountTimeoutNamesPoint) {
    Stepper stepper(20ms);

    EXPECT_FALSE(stepper.wait_for_arrivals("short_batch", 2, 20ms));

    try {
        stepper.wait_for_arrivals("named_batch", 1);
        FAIL() << "wait_for_arrivals without explicit timeout should throw on timeout";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("named_batch"), std::string::npos);
        EXPECT_NE(message.find("1"), std::string::npos);
    }
}

TEST(OperationHistoryTest, RecordsOperations) {
    OperationHistory history;
    const auto start = std::chrono::steady_clock::now();
    const auto end = start + 1ms;

    history.record(Operation{
        .thread_id = 7,
        .op_id = 42,
        .type = Operation::Type::Send,
        .value = 123,
        .success = true,
        .start = start,
        .end = end,
    });

    ASSERT_EQ(history.size(), 1U);
    const Operation operation = history.snapshot().front();
    EXPECT_EQ(operation.thread_id, 7U);
    EXPECT_EQ(operation.op_id, 42U);
    EXPECT_EQ(operation.type, Operation::Type::Send);
    EXPECT_EQ(operation.value, 123U);
    EXPECT_TRUE(operation.success);
    EXPECT_EQ(operation.start, start);
    EXPECT_EQ(operation.end, end);
    EXPECT_EQ(history.count(Operation::Type::Send), 1U);
    EXPECT_EQ(std::string(OperationHistory::type_name(Operation::Type::Send)), "send");
}

TEST(OperationHistoryTest, ChecksNoDuplicateAndNoPhantomReceives) {
    OperationHistory good_history;
    good_history.record(Operation{.thread_id = 1, .op_id = 1, .type = Operation::Type::Send, .value = 10, .success = true});
    good_history.record(Operation{.thread_id = 2, .op_id = 2, .type = Operation::Type::Recv, .value = 10, .success = true});

    EXPECT_TRUE(good_history.check_no_duplicate_successful_receives().ok);
    EXPECT_TRUE(good_history.check_no_phantom_successful_receives().ok);
    EXPECT_TRUE(good_history.check_basic_channel_invariants().ok);

    OperationHistory duplicate_history;
    duplicate_history.record(Operation{.thread_id = 1, .op_id = 1, .type = Operation::Type::Send, .value = 11, .success = true});
    duplicate_history.record(Operation{.thread_id = 2, .op_id = 2, .type = Operation::Type::Recv, .value = 11, .success = true});
    duplicate_history.record(Operation{.thread_id = 3, .op_id = 3, .type = Operation::Type::Recv, .value = 11, .success = true});

    const auto duplicate_result = duplicate_history.check_no_duplicate_successful_receives();
    EXPECT_FALSE(duplicate_result.ok);
    ASSERT_FALSE(duplicate_result.failures.empty());
    EXPECT_NE(duplicate_result.failures.front().find("11"), std::string::npos);

    OperationHistory phantom_history;
    phantom_history.record(Operation{.thread_id = 4, .op_id = 4, .type = Operation::Type::Recv, .value = 99, .success = true});

    const auto phantom_result = phantom_history.check_no_phantom_successful_receives();
    EXPECT_FALSE(phantom_result.ok);
    ASSERT_FALSE(phantom_result.failures.empty());
    EXPECT_NE(phantom_result.failures.front().find("99"), std::string::npos);
}

TEST(OperationHistoryTest, ChecksUniqueOperationIds) {
    OperationHistory good_history;
    good_history.record(Operation{.thread_id = 1, .op_id = 1, .type = Operation::Type::Publish, .value = 7, .success = true});
    good_history.record(Operation{.thread_id = 1, .op_id = 2, .type = Operation::Type::Read, .value = 7, .success = true});
    good_history.record(Operation{.thread_id = 2, .op_id = 1, .type = Operation::Type::Read, .value = 7, .success = true});

    EXPECT_TRUE(good_history.check_unique_operation_ids().ok);
    EXPECT_EQ(good_history.count(Operation::Type::Read), 2U);
    EXPECT_EQ(std::string(OperationHistory::type_name(Operation::Type::Publish)), "publish");

    OperationHistory duplicate_history;
    duplicate_history.record(Operation{.thread_id = 4, .op_id = 8, .type = Operation::Type::Send, .value = 1, .success = true});
    duplicate_history.record(Operation{.thread_id = 4, .op_id = 8, .type = Operation::Type::Recv, .value = 1, .success = true});

    const auto duplicate_result = duplicate_history.check_unique_operation_ids();
    EXPECT_FALSE(duplicate_result.ok);
    ASSERT_FALSE(duplicate_result.failures.empty());
    EXPECT_NE(duplicate_result.failures.front().find("thread=4"), std::string::npos);
    EXPECT_NE(duplicate_result.failures.front().find("op=8"), std::string::npos);
    EXPECT_FALSE(duplicate_history.check_basic_channel_invariants().ok);
}
