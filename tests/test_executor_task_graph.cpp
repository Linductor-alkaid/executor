#include <executor/executor.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std::chrono_literals;

namespace {

executor::ExecutorConfig config() {
    executor::ExecutorConfig cfg;
    cfg.min_threads = 2;
    cfg.max_threads = 2;
    cfg.queue_capacity = 64;
    return cfg;
}

TEST(ExecutorTaskGraphTest, SubmitAfterRunsAfterDependency) {
    executor::Executor executor;
    ASSERT_TRUE(executor.initialize(config()));

    std::atomic<int> order{0};
    auto root = executor.submit_with_handle([&] {
        EXPECT_EQ(order.fetch_add(1, std::memory_order_acq_rel), 0);
        return 21;
    });

    auto dependent = executor.submit_after(root.handle, [&] {
        EXPECT_EQ(order.fetch_add(1, std::memory_order_acq_rel), 1);
        return 42;
    });

    EXPECT_EQ(root.future.get(), 21);
    EXPECT_EQ(dependent.get(), 42);
    EXPECT_EQ(order.load(std::memory_order_acquire), 2);

    executor.shutdown();
}

TEST(ExecutorTaskGraphTest, WhenAllWaitsForAllDependencies) {
    executor::Executor executor;
    ASSERT_TRUE(executor.initialize(config()));

    std::atomic<int> completed{0};
    auto first = executor.submit_with_handle([&] {
        completed.fetch_add(1, std::memory_order_acq_rel);
        return 1;
    });
    auto second = executor.submit_with_handle([&] {
        completed.fetch_add(1, std::memory_order_acq_rel);
        return 2;
    });

    auto all = executor.when_all({first.handle, second.handle});
    auto after_all = executor.submit_after(all, [&] {
        EXPECT_EQ(completed.load(std::memory_order_acquire), 2);
        return 3;
    });

    EXPECT_EQ(first.future.get(), 1);
    EXPECT_EQ(second.future.get(), 2);
    EXPECT_EQ(after_all.get(), 3);

    executor.shutdown();
}

TEST(ExecutorTaskGraphTest, NestedWhenAllPropagatesCompletion) {
    executor::Executor executor;
    ASSERT_TRUE(executor.initialize(config()));

    auto first = executor.submit_with_handle([] {
        return 1;
    });
    auto second = executor.submit_with_handle([] {
        return 2;
    });

    auto inner = executor.when_all({first.handle, second.handle});
    auto outer = executor.when_all({inner});

    auto after_outer = executor.submit_after(outer, [] {
        return 3;
    });

    EXPECT_EQ(first.future.get(), 1);
    EXPECT_EQ(second.future.get(), 2);
    EXPECT_EQ(after_outer.get(), 3);

    executor.shutdown();
}

TEST(ExecutorTaskGraphTest, DependencyFailureSkipsDependentTask) {
    executor::Executor executor;
    ASSERT_TRUE(executor.initialize(config()));

    auto failing = executor.submit_with_handle([]() -> int {
        throw std::runtime_error("root failed");
    });

    std::atomic<bool> ran{false};
    auto dependent = executor.submit_after(failing.handle, [&] {
        ran.store(true, std::memory_order_release);
        return 7;
    });

    EXPECT_THROW(failing.future.get(), std::runtime_error);
    EXPECT_THROW(dependent.get(), std::runtime_error);
    EXPECT_FALSE(ran.load(std::memory_order_acquire));
    EXPECT_GE(executor.get_failure_status().task_exception_count, 1U);

    executor.shutdown();
}

TEST(ExecutorTaskGraphTest, InvalidHandleReturnsReadyExceptionalFuture) {
    executor::Executor executor;
    ASSERT_TRUE(executor.initialize(config()));

    std::atomic<bool> ran{false};
    auto future = executor.submit_after(executor::TaskHandle{}, [&] {
        ran.store(true, std::memory_order_release);
    });

    EXPECT_THROW(future.get(), std::runtime_error);
    EXPECT_FALSE(ran.load(std::memory_order_acquire));
    EXPECT_GE(executor.get_failure_status().submit_rejected_count, 1U);

    executor.shutdown();
}

TEST(ExecutorTaskGraphTest, ShutdownMakesPendingDependencyObservable) {
    executor::Executor executor;
    ASSERT_TRUE(executor.initialize(config()));

    executor::TaskHandle invalid;
    auto future = executor.submit_after(invalid, [] {
        return 1;
    });

    executor.shutdown(false);
    EXPECT_THROW(future.get(), std::runtime_error);
}

} // namespace
