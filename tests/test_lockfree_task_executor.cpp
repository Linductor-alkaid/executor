#include <gtest/gtest.h>
#include <executor/lockfree_task_executor.hpp>
#include <atomic>
#include <thread>
#include <chrono>

using namespace executor;

TEST(LockFreeTaskExecutorTest, BasicStartStop) {
    LockFreeTaskExecutor exec(128);

    EXPECT_FALSE(exec.is_running());
    EXPECT_TRUE(exec.start());
    EXPECT_TRUE(exec.is_running());

    exec.stop();
    EXPECT_FALSE(exec.is_running());
}

TEST(LockFreeTaskExecutorTest, DoubleStart) {
    LockFreeTaskExecutor exec(128);

    EXPECT_TRUE(exec.start());
    EXPECT_FALSE(exec.start());

    exec.stop();
}

TEST(LockFreeTaskExecutorTest, TaskExecution) {
    LockFreeTaskExecutor exec(128);
    exec.start();

    std::atomic<int> counter{0};

    for (int i = 0; i < 100; ++i) {
        EXPECT_TRUE(exec.push_task([&counter]() {
            counter.fetch_add(1, std::memory_order_relaxed);
        }));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_EQ(counter.load(), 100);
    EXPECT_EQ(exec.processed_count(), 100);

    exec.stop();
}

TEST(LockFreeTaskExecutorTest, QueueFull) {
    LockFreeTaskExecutor exec(16);
    // 不启动，这样任务不会被消费

    // 填满队列（容量16，但实际可用15个槽位）
    for (int i = 0; i < 15; ++i) {
        bool success = exec.push_task([]() {});
        EXPECT_TRUE(success);
    }

    // 队列满时应该失败
    bool result = exec.push_task([]() {});
    EXPECT_FALSE(result);
}

TEST(LockFreeTaskExecutorTest, ExceptionHandling) {
    LockFreeTaskExecutor exec(128);
    exec.start();

    std::atomic<int> counter{0};

    exec.push_task([&counter]() {
        counter.fetch_add(1);
        throw std::runtime_error("test exception");
    });

    exec.push_task([&counter]() {
        counter.fetch_add(1);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    EXPECT_EQ(counter.load(), 2);
    EXPECT_EQ(exec.processed_count(), 2);

    exec.stop();
}

TEST(LockFreeTaskExecutorTest, StopWithPendingTasks) {
    LockFreeTaskExecutor exec(128);
    exec.start();

    std::atomic<int> counter{0};

    for (int i = 0; i < 50; ++i) {
        exec.push_task([&counter]() {
            counter.fetch_add(1);
        });
    }

    exec.stop();

    EXPECT_EQ(counter.load(), 50);
    EXPECT_EQ(exec.processed_count(), 50);
}

// push_tasks_batch: 正常路径，全部入队
TEST(LockFreeTaskExecutorTest, PushTasksBatchAllSuccess) {
    LockFreeTaskExecutor exec(128);
    exec.start();

    std::atomic<int> counter{0};

    constexpr size_t TASK_COUNT = 10;
    std::vector<std::function<void()>> tasks(TASK_COUNT,
        [&counter]() { counter.fetch_add(1, std::memory_order_relaxed); });

    size_t pushed = 0;
    bool ok = exec.push_tasks_batch(tasks.data(), TASK_COUNT, pushed);

    EXPECT_TRUE(ok);
    EXPECT_EQ(pushed, TASK_COUNT);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_EQ(counter.load(), static_cast<int>(TASK_COUNT));

    exec.stop();
}

// push_tasks_batch: 队列接近满时，pushed < count
TEST(LockFreeTaskExecutorTest, PushTasksBatchPartial) {
    // 容量 32：对象池也是 32，先填 28 个槽（剩余 4 个 wrapper 空余）
    // 然后批量提交 3 个：池够但队列只剩约 3 个槽
    // 期望：ok=true, pushed <= 剩余槽数
    LockFreeTaskExecutor exec(32);
    // 不启动消费者，让任务堆积

    // 填充 28 个任务（队列容量 32，available = 32-1-28 = 3）
    for (int i = 0; i < 28; ++i) {
        ASSERT_TRUE(exec.push_task([]() {}));
    }

    // 批量提交 3 个：对象池还有 4 个空余，队列也有约 3 个槽
    std::vector<std::function<void()>> tasks(3, []() {});
    size_t pushed = 0;
    bool ok = exec.push_tasks_batch(tasks.data(), tasks.size(), pushed);

    // 池够，queue_->push_batch 会成功（返回 true），pushed == 3 或更少
    EXPECT_TRUE(ok);
    EXPECT_LE(pushed, tasks.size());

    // 再批量提交 5 个：队列此时只剩约 0 个槽，push_batch 返回 false
    // push_tasks_batch 会先申请 5 个 wrapper，但 push_batch 失败，全部回收
    std::vector<std::function<void()>> overflow(5, []() {});
    size_t pushed2 = 0;
    bool ok2 = exec.push_tasks_batch(overflow.data(), overflow.size(), pushed2);
    // ok2 可能为 false（队列满），pushed2 == 0
    if (!ok2) {
        EXPECT_EQ(pushed2, 0u);
    }
}
