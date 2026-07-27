#include <gtest/gtest.h>

#include "executor/lockfree_task_executor.hpp"

#include <atomic>
#include <thread>
#include <vector>

namespace {

TEST(LockFreeQueueStatusSnapshotTest, IsCopyableAndNonBlockingUnderConcurrentTraffic) {
    constexpr size_t kProducerCount = 4;
    constexpr size_t kTasksPerProducer = 250000;
    constexpr size_t kTaskCount = kProducerCount * kTasksPerProducer;

    executor::LockFreeTaskExecutor executor(4096, 1, true);
    ASSERT_TRUE(executor.start());

    std::atomic<bool> producers_started{false};
    std::atomic<size_t> producers_finished{0};
    std::vector<std::thread> producers;
    producers.reserve(kProducerCount);
    for (size_t producer_index = 0; producer_index < kProducerCount; ++producer_index) {
        producers.emplace_back([&]() {
            while (!producers_started.load(std::memory_order_acquire)) {
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

    producers_started.store(true, std::memory_order_release);
    uint64_t snapshot_count = 0;
    while (producers_finished.load(std::memory_order_acquire) != kProducerCount) {
        const auto first = executor.get_status_snapshot();
        const auto copied = first;
        EXPECT_EQ(copied.queue_capacity, 4096u);
        EXPECT_LE(copied.reserved_count + copied.ready_count, copied.queue_capacity);
        ++snapshot_count;
    }
    for (auto& producer : producers) {
        producer.join();
    }

    executor.stop();
    const auto final_snapshot = executor.get_status_snapshot();
    EXPECT_GT(snapshot_count, 0u);
    EXPECT_EQ(final_snapshot.total_pushes, kTaskCount);
    EXPECT_EQ(executor.processed_count(), kTaskCount);

    // A snapshot is assembled from independent atomics. During this 1M-task
    // burst, instantaneous queue fields can drift by at most one queue window
    // (4096 slots), so the assertions above intentionally avoid exact
    // cross-field equality while traffic is active.
    EXPECT_LE(final_snapshot.current_size, final_snapshot.queue_capacity);
}

} // namespace
