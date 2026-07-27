#include <gtest/gtest.h>

#include "executor/lockfree_task_executor.hpp"

#include <atomic>
#include <chrono>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

using executor::LockFreeTaskExecutor;

static_assert(std::is_copy_constructible<LockFreeTaskExecutor::QueueStats>::value,
              "QueueStats must remain a value-typed snapshot");
static_assert(std::is_copy_assignable<LockFreeTaskExecutor::QueueStats>::value,
              "QueueStats must remain assignable for monitoring clients");

TEST(LockFreeQueueStatusSnapshotTest, ReportsPushPathsAndCopiesByValue) {
    LockFreeTaskExecutor executor(7, 1, true);
    ASSERT_TRUE(executor.start());

    ASSERT_TRUE(executor.push_task([] {}));
    std::function<void()> batch[] = {[] {}, [] {}};
    size_t pushed = 0;
    ASSERT_TRUE(executor.push_tasks_batch(batch, 2, pushed));
    EXPECT_EQ(pushed, 2u);
    EXPECT_FALSE(executor.push_task({}));

    const auto snapshot = executor.get_status_snapshot();
    const auto copied_snapshot = snapshot;
    LockFreeTaskExecutor::QueueStats assigned_snapshot{};
    assigned_snapshot = copied_snapshot;

    EXPECT_EQ(assigned_snapshot.queue_capacity, 8u);
    EXPECT_GE(assigned_snapshot.total_pushes, 3u);
    EXPECT_EQ(assigned_snapshot.rejected_empty_count, 1u);
    EXPECT_GE(assigned_snapshot.submission_rejection, 1u);
    EXPECT_LE(assigned_snapshot.current_size, assigned_snapshot.queue_capacity);

    executor.stop();
    EXPECT_FALSE(executor.push_task([] {}));
    EXPECT_GE(executor.get_status_snapshot().submission_rejection, 2u);
}

TEST(LockFreeQueueStatusSnapshotTest, DoesNotBlockProducersUnderConcurrentTraffic) {
    constexpr size_t kProducerCount = 4;
    constexpr size_t kTasksPerProducer = 20000;
    constexpr size_t kExpectedPushes = kProducerCount * kTasksPerProducer;

    LockFreeTaskExecutor executor(1024, 1, true);
    ASSERT_TRUE(executor.start());

    std::atomic<bool> begin{false};
    std::atomic<size_t> producers_finished{0};
    std::vector<std::thread> producers;
    producers.reserve(kProducerCount);
    for (size_t producer_index = 0; producer_index < kProducerCount; ++producer_index) {
        producers.emplace_back([&] {
            while (!begin.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (size_t task_index = 0; task_index < kTasksPerProducer; ++task_index) {
                while (!executor.push_task([] {})) {
                    std::this_thread::yield();
                }
            }
            producers_finished.fetch_add(1, std::memory_order_release);
        });
    }

    begin.store(true, std::memory_order_release);
    size_t samples = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (producers_finished.load(std::memory_order_acquire) != kProducerCount &&
           std::chrono::steady_clock::now() < deadline) {
        const auto snapshot = executor.get_status_snapshot();
        const auto copied_snapshot = snapshot;
        EXPECT_EQ(copied_snapshot.queue_capacity, 1024u);
        EXPECT_LE(copied_snapshot.current_size, copied_snapshot.queue_capacity);
        ++samples;
    }

    EXPECT_EQ(producers_finished.load(std::memory_order_acquire), kProducerCount)
        << "status sampling blocked producer progress";
    for (auto& producer : producers) {
        producer.join();
    }

    executor.stop();
    const auto final_snapshot = executor.get_status_snapshot();
    EXPECT_GT(samples, 0u);
    EXPECT_EQ(final_snapshot.total_pushes, kExpectedPushes);
    EXPECT_EQ(executor.processed_count(), kExpectedPushes);
}

} // namespace
