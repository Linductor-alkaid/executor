#include <gtest/gtest.h>
#include "executor/util/lockfree_queue.hpp"
#include <executor/lockfree_task_executor.hpp>

#include <cstddef>
#include <limits>
#include <stdexcept>

using executor::LockFreeTaskExecutor;
using executor::util::LockFreeQueue;

TEST(LockFreeQueueBackoffValidationTest, test_lockfree_queue_rejects_zero_backoff) {
    EXPECT_THROW((LockFreeQueue<int>(64, 0)), std::invalid_argument);
}

TEST(LockFreeQueueBackoffValidationTest, test_lockfree_queue_huge_backoff_is_clamped) {
    LockFreeQueue<int> q(64, std::numeric_limits<size_t>::max());

    int out = 0;
    EXPECT_TRUE(q.push(7));
    EXPECT_TRUE(q.push(11));
    EXPECT_TRUE(q.pop(out));
    EXPECT_EQ(out, 7);
    EXPECT_TRUE(q.pop(out));
    EXPECT_EQ(out, 11);
    EXPECT_FALSE(q.pop(out));
}

TEST(LockFreeQueueBackoffValidationTest, test_lockfree_task_executor_invalid_backoff_throws) {
    EXPECT_THROW((LockFreeTaskExecutor(64, 0)), std::invalid_argument);
}

TEST(LockFreeQueueBackoffValidationTest, test_lockfree_queue_normal_backoff_works) {
    LockFreeQueue<int> q(64, 2);

    int out = 0;
    for (int i = 0; i < 4; ++i) {
        ASSERT_TRUE(q.push(i));
    }
    for (int i = 0; i < 4; ++i) {
        ASSERT_TRUE(q.pop(out));
        EXPECT_EQ(out, i);
    }
    EXPECT_TRUE(q.empty());
}
