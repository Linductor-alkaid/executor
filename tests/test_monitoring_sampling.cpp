#include <gtest/gtest.h>
#include "executor/monitor/task_monitor.hpp"
#include "executor/lockfree_task_executor.hpp"
#include <thread>
#include <chrono>

using namespace executor;
using namespace executor::monitor;

TEST(MonitoringSamplingTest, DefaultFullSampling) {
    TaskMonitor monitor;

    for (int i = 0; i < 100; ++i) {
        monitor.record_task_start("task_" + std::to_string(i), "test");
        monitor.record_task_complete("task_" + std::to_string(i), true, 1000);
    }

    auto stats = monitor.get_statistics("test");
    EXPECT_EQ(stats.total_count, 100);
    EXPECT_EQ(stats.success_count, 100);
}

TEST(MonitoringSamplingTest, OnePctSampling) {
    TaskMonitor monitor;
    monitor.set_sampling_rate(0.01);

    EXPECT_DOUBLE_EQ(monitor.get_sampling_rate(), 0.01);

    for (int i = 0; i < 10000; ++i) {
        monitor.record_task_start("task_" + std::to_string(i), "test");
        monitor.record_task_complete("task_" + std::to_string(i), true, 1000);
    }

    auto stats = monitor.get_statistics("test");
    EXPECT_GT(stats.total_count, 50);
    EXPECT_LT(stats.total_count, 150);
}

TEST(MonitoringSamplingTest, ZeroSampling) {
    TaskMonitor monitor;
    monitor.set_sampling_rate(0.0);

    for (int i = 0; i < 100; ++i) {
        monitor.record_task_start("task_" + std::to_string(i), "test");
        monitor.record_task_complete("task_" + std::to_string(i), true, 1000);
    }

    auto stats = monitor.get_statistics("test");
    EXPECT_EQ(stats.total_count, 0);
}

TEST(LockFreeQueueStatsTest, BasicStats) {
    LockFreeTaskExecutor executor(1024, 2, true);
    executor.start();

    std::atomic<int> counter{0};
    for (int i = 0; i < 100; ++i) {
        executor.push_task([&counter]() { counter++; });
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    auto stats = executor.get_queue_stats();
    EXPECT_EQ(stats.total_pushes, 100);
    EXPECT_GE(stats.total_pops, 90);
    EXPECT_GE(stats.success_rate, 0.99);

    executor.stop();
}

TEST(LockFreeQueueStatsTest, BatchStats) {
    LockFreeTaskExecutor executor(1024, 2, true);
    executor.start();

    std::function<void()> tasks[50];
    for (int i = 0; i < 50; ++i) {
        tasks[i] = []() {};
    }

    size_t pushed;
    executor.push_tasks_batch(tasks, 50, pushed);
    EXPECT_EQ(pushed, 50);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    auto stats = executor.get_queue_stats();
    EXPECT_EQ(stats.total_pushes, 50);
    EXPECT_EQ(stats.batch_pushes, 1);
    EXPECT_GE(stats.batch_pops, 1);

    executor.stop();
}
