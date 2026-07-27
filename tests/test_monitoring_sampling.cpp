#include <gtest/gtest.h>
#include "executor/monitor/task_monitor.hpp"
#include "executor/lockfree_task_executor.hpp"
#include "executor/util/lockfree_queue.hpp"
#include <thread>
#include <chrono>
#include <atomic>

using namespace executor;
using namespace executor::monitor;

namespace {

struct ReservationPause {
    std::atomic<bool> entered{false};
    std::atomic<bool> release{false};
};

void pause_before_publish(void* context) {
    auto* pause = static_cast<ReservationPause*>(context);
    pause->entered.store(true, std::memory_order_release);
    while (!pause->release.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
}

} // namespace

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
    // P-260623-004 fixed push_tasks_batch to actually call queue_->push_batch()
    // (the previous implementation looped queue_->push() per item, which kept
    // batch_pushes at 0 and broke monitoring that keyed on it). With the fix:
    //   - batch_pushes increments by 1 per push_tasks_batch() call
    //   - total_pushes still reflects the number of wrappers handed to the queue
    //   - batch_pops is independent of push path; worker_thread uses pop_batch
    EXPECT_EQ(stats.total_pushes, 50);
    EXPECT_EQ(stats.batch_pushes, 1u);
    // worker_thread still uses pop_batch, so batch_pops > 0.
    EXPECT_GE(stats.batch_pops, 1);

    executor.stop();
}

TEST(LockFreeQueueStatsTest, ReportsContentionReservationsAndSnapshot) {
    util::LockFreeQueue<int> queue(4, 1, true);
    ASSERT_TRUE(queue.push(1));
    ASSERT_TRUE(queue.push(2));
    ASSERT_TRUE(queue.push(3));
    EXPECT_FALSE(queue.push(4));
    EXPECT_GT(queue.get_stats().contention_rejection, 0u);

    util::LockFreeQueue<int> stalled_queue(8, 1, true);
    ReservationPause pause;
    stalled_queue.set_before_publish_hook(pause_before_publish, &pause);
    std::thread producer([&]() { EXPECT_TRUE(stalled_queue.push(1)); });
    while (!pause.entered.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    EXPECT_EQ(stalled_queue.get_stats().reserved_count, 1u);
    pause.release.store(true, std::memory_order_release);
    producer.join();

    LockFreeTaskExecutor executor(8, 1, true);
    ASSERT_TRUE(executor.push_task([] {}));
    const auto snapshot = executor.get_status_snapshot();
    EXPECT_EQ(snapshot.queue_capacity, 8u);
    EXPECT_GT(snapshot.current_size, 0u);
    EXPECT_GT(snapshot.total_pushes, 0u);
}
