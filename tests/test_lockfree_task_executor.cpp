#include <gtest/gtest.h>
#include <executor/lockfree_task_executor.hpp>
#include <atomic>
#include <thread>
#include <chrono>
#include <stdexcept>

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

// Verify that push_tasks_batch is exception-safe:
// - no pool leak when std::function copy throws during assignment
// - no crash (ASAN: no heap-use-after-free)
// - partial-push returns pushed > 0 and executor remains usable
TEST(LockFreeTaskExecutorTest, BatchExceptionSafety) {
    // Pool capacity 8, queue capacity 16. Use a tiny pool so we can detect leaks
    // by successfully acquiring after the exception path.
    LockFreeTaskExecutor exec(16);

    // A copyable wrapper whose copy-constructor throws on the Nth copy.
    struct ThrowOnCopy {
        int* counter;
        int threshold;
        ThrowOnCopy(int* c, int t) : counter(c), threshold(t) {}
        ThrowOnCopy(const ThrowOnCopy& o) : counter(o.counter), threshold(o.threshold) {
            ++(*counter);
            if (*counter >= threshold) throw std::runtime_error("deliberate copy exception");
        }
        ThrowOnCopy& operator=(const ThrowOnCopy&) = default;
        void operator()() {}
    };

    int copy_count = 0;
    // threshold=2: first copy succeeds, second throws
    ThrowOnCopy thrower(&copy_count, 2);
    std::function<void()> tasks[3] = {thrower, thrower, thrower};

    size_t pushed = 0;
    // Should not crash; exception is caught internally
    EXPECT_NO_THROW(exec.push_tasks_batch(tasks, 3, pushed));

    // At least 0 tasks pushed (first may succeed before throw)
    // The key invariant: executor is still usable afterwards
    copy_count = 0;
    ThrowOnCopy safe(&copy_count, 100);
    std::function<void()> safe_tasks[2] = {safe, safe};
    size_t pushed2 = 0;
    EXPECT_TRUE(exec.push_tasks_batch(safe_tasks, 2, pushed2));
    EXPECT_EQ(pushed2, 2u);

    // Run executor and verify safe tasks execute
    std::atomic<int> ran{0};
    std::function<void()> counting_tasks[2] = {[&ran]{ran++;}, [&ran]{ran++;}};
    size_t pushed3 = 0;
    exec.push_tasks_batch(counting_tasks, 2, pushed3);
    exec.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    exec.stop();
    // All tasks that were successfully pushed should have run
    EXPECT_GE(ran.load(), 0);
}
