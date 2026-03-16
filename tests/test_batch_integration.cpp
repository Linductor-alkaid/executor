/**
 * 批量操作集成测试
 */

#include "executor/lockfree_task_executor.hpp"
#include <gtest/gtest.h>
#include <atomic>
#include <vector>

using namespace executor;

TEST(BatchIntegration, PushTasksBatch) {
    LockFreeTaskExecutor executor(1024);
    ASSERT_TRUE(executor.start());

    std::atomic<int> counter{0};
    const size_t batch_size = 100;

    std::vector<std::function<void()>> tasks(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        tasks[i] = [&counter]() { counter.fetch_add(1); };
    }

    size_t pushed;
    ASSERT_TRUE(executor.push_tasks_batch(tasks.data(), batch_size, pushed));
    EXPECT_EQ(pushed, batch_size);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_EQ(counter.load(), batch_size);
    EXPECT_EQ(executor.processed_count(), batch_size);
}

TEST(BatchIntegration, WorkerUsesBatchPop) {
    LockFreeTaskExecutor executor(1024);
    ASSERT_TRUE(executor.start());

    std::atomic<int> counter{0};
    const size_t total_tasks = 1000;

    for (size_t i = 0; i < total_tasks; ++i) {
        ASSERT_TRUE(executor.push_task([&counter]() { counter.fetch_add(1); }));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_EQ(counter.load(), total_tasks);
    EXPECT_EQ(executor.processed_count(), total_tasks);
}
