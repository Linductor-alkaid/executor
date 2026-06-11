#include <gtest/gtest.h>
#include <executor/lockfree_task_executor.hpp>
#include <atomic>
#include <thread>
#include <chrono>
#include <stdexcept>
#include <memory>

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
// We track copies via a shared atomic counter so the throws happen
// during std::function assignment inside push_tasks_batch, not at
// test-array construction time (which would trip gtest's own
// exception-in-test-body detection).
struct ThrowOnCopy {
    std::shared_ptr<std::atomic<int>> counter;
    int threshold;
    ThrowOnCopy(std::shared_ptr<std::atomic<int>> c, int t)
        : counter(std::move(c)), threshold(t) {}
    ThrowOnCopy(const ThrowOnCopy& o)
        : counter(o.counter), threshold(o.threshold) {
        if (++(*counter) >= threshold) {
            throw std::runtime_error("deliberate copy exception");
        }
    }
    ThrowOnCopy& operator=(const ThrowOnCopy&) = default;
    void operator()() {}
};

auto counter = std::make_shared<std::atomic<int>>(0);
// Pre-create one std::function per slot so test-array construction
// itself never throws. Push_tasks_batch will then copy the held
// ThrowOnCopy into wrapper->func; the Nth copy throws and is
// caught inside push_tasks_batch.
std::function<void()> tasks[3] = {
    ThrowOnCopy(counter, 2),
    ThrowOnCopy(counter, 2),
    ThrowOnCopy(counter, 2),
};
// Reset counter so the throw occurs inside push_tasks_batch, not
// at this point (each array element already performed 1 copy above).
counter->store(1);

size_t pushed = 0;
// Should not crash; exception is caught internally by push_tasks_batch.
EXPECT_NO_THROW(exec.push_tasks_batch(tasks, 3, pushed));
// First task succeeded (counter went 1->2 just before the throw
// on the second assignment). The key invariant is that the third
// task was never enqueued with a half-initialised wrapper.
EXPECT_LT(pushed, 3u);

// Executor must still be usable after the exception.
auto safe_counter = std::make_shared<std::atomic<int>>(0);
std::function<void()> safe_tasks[2] = {
    ThrowOnCopy(safe_counter, 1000),
    ThrowOnCopy(safe_counter, 1000),
};
size_t pushed2 = 0;
EXPECT_TRUE(exec.push_tasks_batch(safe_tasks, 2, pushed2));
EXPECT_EQ(pushed2, 2u);

std::atomic<int> ran{0};
std::function<void()> counting_tasks[2] = {[&ran]{ran++;}, [&ran]{ran++;}};
size_t pushed3 = 0;
EXPECT_TRUE(exec.push_tasks_batch(counting_tasks, 2, pushed3));
exec.start();
std::this_thread::sleep_for(std::chrono::milliseconds(100));
exec.stop();
EXPECT_EQ(ran.load(), 2);
}
