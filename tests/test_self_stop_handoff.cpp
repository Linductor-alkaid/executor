#include <gtest/gtest.h>

#include <executor/config.hpp>
#include <executor/lockfree_task_executor.hpp>
#include "executor/realtime_thread_executor.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <future>
#include <memory>
#include <thread>

namespace {

using namespace std::chrono_literals;

// LockFreeTaskExecutor 自停止路径:任务 lambda 中调用 stop() + stop_and_join(),
// 期望 stop_and_join() 返回 false(已请求停止但未 join),外部线程随后 stop_and_join()
// 返回 true 正常 join。
TEST(SelfStopHandoff, SelfStopDoesNotTerminate_LockFreeTaskExecutor) {
    auto executor = std::make_unique<executor::LockFreeTaskExecutor>(8);
    std::promise<bool> self_stop_result;
    auto self_stop_future = self_stop_result.get_future();

    ASSERT_TRUE(executor->push_task([&] {
        executor->stop();
        self_stop_result.set_value(executor->stop_and_join());
    }));
    ASSERT_TRUE(executor->start());

    ASSERT_EQ(self_stop_future.wait_for(1s), std::future_status::ready);
    EXPECT_FALSE(self_stop_future.get());
    EXPECT_TRUE(executor->stop_and_join());
    executor.reset();
}

TEST(SelfStopHandoff, SelfStopDoesNotTerminate_RealtimeThreadExecutor) {
    executor::RealtimeThreadConfig config;
    config.thread_name = "external_stop_rt";
    config.cycle_period_ns = 1'000'000;
    std::atomic<executor::RealtimeThreadExecutor*> executor_ptr{nullptr};
    std::atomic<bool> callback_invoked{false};
    std::promise<bool> self_stop_result;
    auto self_stop_future = self_stop_result.get_future();
    config.cycle_callback = [&] {
        auto* executor = executor_ptr.load(std::memory_order_acquire);
        if (executor && !callback_invoked.exchange(true, std::memory_order_acq_rel)) {
            executor->stop();
            self_stop_result.set_value(executor->stop_and_join());
        }
    };

    auto executor = std::make_unique<executor::RealtimeThreadExecutor>(
        "external_stop_rt", config);
    executor_ptr.store(executor.get(), std::memory_order_release);
    ASSERT_TRUE(executor->start());

    ASSERT_EQ(self_stop_future.wait_for(1s), std::future_status::ready);
    EXPECT_FALSE(self_stop_future.get());
    EXPECT_TRUE(executor->stop_and_join());
    executor_ptr.store(nullptr, std::memory_order_release);
    executor.reset();
}

// 并发外部线程 stop_and_join:两个外部线程同时调用 stop_and_join,断言两个都返回
// true 且无死锁。stop_mutex_ 应串行化 join,确保只有一个 thread 真正 join。
TEST(SelfStopHandoff, ConcurrentStopFromTwoExternalThreads) {
    executor::LockFreeTaskExecutor executor(8);
    ASSERT_TRUE(executor.start());

    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    bool first_result = false;
    bool second_result = false;
    auto stop_from_external_thread = [&](bool& result, const char* who) {
        ready.fetch_add(1, std::memory_order_release);
        while (!go.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        std::fprintf(stderr, "[%s] before stop_and_join\n", who);
        result = executor.stop_and_join();
        std::fprintf(stderr, "[%s] after stop_and_join result=%d\n", who, result);
    };

    std::thread first(stop_from_external_thread, std::ref(first_result), "first");
    std::thread second(stop_from_external_thread, std::ref(second_result), "second");
    while (ready.load(std::memory_order_acquire) != 2) {
        std::this_thread::yield();
    }
    go.store(true, std::memory_order_release);
    first.join();
    second.join();

    EXPECT_TRUE(first_result);
    EXPECT_TRUE(second_result);
}

// 自停止后剩余任务被丢弃:任务 A 在内部调 stop(),后续任务 B 不应再执行,
// processed_count 应只统计已执行的任务(A)。
TEST(SelfStopHandoff, DrainOnStopRejectedWhenSelfStop) {
    executor::LockFreeTaskExecutor executor(8);
    std::promise<void> self_stop_done;
    auto self_stop_future = self_stop_done.get_future();
    std::atomic<int> executed{0};

    ASSERT_TRUE(executor.push_task([&] {
        executed.fetch_add(1, std::memory_order_relaxed);
        executor.stop();
        self_stop_done.set_value();
    }));
    ASSERT_TRUE(executor.push_task([&] {
        executed.fetch_add(1, std::memory_order_relaxed);
    }));
    ASSERT_TRUE(executor.start());

    ASSERT_EQ(self_stop_future.wait_for(1s), std::future_status::ready);
    EXPECT_TRUE(executor.stop_and_join());
    EXPECT_EQ(executed.load(std::memory_order_relaxed), 1);
    EXPECT_EQ(executor.processed_count(), 1u);
}

} // namespace
