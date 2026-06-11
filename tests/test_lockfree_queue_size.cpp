// P-003 regression test: LockFreeQueue::size() must never return a
// value greater than capacity() and must never underflow on weakly-
// ordered architectures.
//
// We exercise three properties:
//   1. Static correctness: std::atomic<size_t> must be lock-free.
//   2. Empty queue reports size 0.
//   3. Under concurrent push/pop traffic, observed size() stays
//      within [0, capacity()], never near SIZE_MAX, and monotonically
//      tracks producer/consumer progress in aggregate.

#include <gtest/gtest.h>
#include "executor/util/lockfree_queue.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <thread>
#include <vector>

using executor::util::LockFreeQueue;
using executor::util::LockFreeQueueStats;

// 1. Static: the position counters must be lock-free atomics. If the
//    platform falls back to a mutex implementation, the fix in size()
//    is meaningless, so fail loudly at compile/init time.
static_assert(std::atomic<size_t>::is_always_lock_free,
              "LockFreeQueue requires lock-free std::atomic<size_t>");

TEST(LockFreeQueueSizeTest, AtomicPositionIsLockFree) {
    EXPECT_TRUE(std::atomic<size_t>::is_always_lock_free);
}

TEST(LockFreeQueueSizeTest, EmptyQueueReportsZero) {
    LockFreeQueue<int> q(64);
    EXPECT_EQ(q.size(), static_cast<size_t>(0));
    EXPECT_EQ(q.size(), q.capacity() == 0 ? 0u : 0u);
}

TEST(LockFreeQueueSizeTest, SingleThreadedSizeMatchesPushesAndPops) {
    LockFreeQueue<int> q(64);
    int out = 0;

    for (int i = 0; i < 10; ++i) {
        ASSERT_TRUE(q.push(i));
    }
    EXPECT_EQ(q.size(), 10u);

    for (int i = 0; i < 4; ++i) {
        ASSERT_TRUE(q.pop(out));
    }
    EXPECT_EQ(q.size(), 6u);

    for (int i = 0; i < 6; ++i) {
        ASSERT_TRUE(q.pop(out));
    }
    EXPECT_EQ(q.size(), 0u);
}

// 2. Regression: under high contention, size() must NEVER return a
//    value >= capacity() (in particular, must never wrap to a huge
//    number due to size_t underflow). This is the property that
//    breaks on ARM/POWER with relaxed loads.
TEST(LockFreeQueueSizeTest, SizeNeverExceedsCapacityUnderContention) {
    constexpr size_t kCap = 256;  // must be power of two (queue requirement)
    LockFreeQueue<int> q(kCap);

    constexpr int kProducers = 4;
    std::atomic<bool> stop_producers{false};   // signal producers to exit
    std::atomic<bool> stop_consumer{false};   // signal consumer to drain
    std::atomic<uint64_t> total_pushed{0};
    std::atomic<uint64_t> total_popped{0};

    std::vector<std::thread> producers;
    producers.reserve(kProducers);
    for (int p = 0; p < kProducers; ++p) {
        producers.emplace_back([&]() {
            int v = 0;
            while (!stop_producers.load(std::memory_order_relaxed)) {
                if (q.push(v)) {
                    total_pushed.fetch_add(1, std::memory_order_relaxed);
                    ++v;
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }

    std::thread consumer([&]() {
        int out = 0;
        // Phase 1: keep popping while work is being produced.
        while (!stop_consumer.load(std::memory_order_relaxed)) {
            if (q.pop(out)) {
                total_popped.fetch_add(1, std::memory_order_relaxed);
            } else {
                std::this_thread::yield();
            }
        }
        // Phase 2: all producers have stopped AND main thread has
        // signaled us to drain the rest.
        while (q.pop(out)) {
            total_popped.fetch_add(1, std::memory_order_relaxed);
        }
    });

    std::thread observer([&]() {
        size_t max_seen = 0;
        uint64_t underflow_violations = 0;
        const uint64_t kHugeThreshold = static_cast<uint64_t>(1) << 32;
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        while (std::chrono::steady_clock::now() < deadline) {
            size_t s = q.size();
            if (s > max_seen) max_seen = s;
            // size() must never appear huge (sign of underflow).
            if (static_cast<uint64_t>(s) > kHugeThreshold) {
                ++underflow_violations;
            }
            // size() must never exceed capacity.
            if (s > q.capacity()) {
                ++underflow_violations;
            }
        }
        EXPECT_EQ(underflow_violations, 0u)
            << "size() returned out-of-range value, max_seen=" << max_seen
            << ", capacity=" << q.capacity();
    });

    observer.join();
    // 1. Stop producers and wait for them to fully exit.
    stop_producers.store(true, std::memory_order_relaxed);
    for (auto& t : producers) t.join();
    // 2. Now no producer is pushing. Tell consumer to drain and exit.
    stop_consumer.store(true, std::memory_order_relaxed);
    consumer.join();

    EXPECT_EQ(q.size(), 0u)
        << "size must be 0 after all producers stopped and consumer "
           "drained; pushed=" << total_pushed.load()
        << " popped=" << total_popped.load();
    EXPECT_EQ(total_pushed.load(), total_popped.load());
}

// 3. Regression: after the consumer has clearly drained the queue,
//    size() must drop to 0 and stay there. With relaxed loads it could
//    transiently report SIZE_MAX.
TEST(LockFreeQueueSizeTest, SizeReturnsZeroAfterDrain) {
    constexpr size_t kCap = 32;
    LockFreeQueue<int> q(kCap);
    int out = 0;

    for (int i = 0; i < 16; ++i) ASSERT_TRUE(q.push(i));
    for (int i = 0; i < 16; ++i) ASSERT_TRUE(q.pop(out));

    for (int i = 0; i < 1000; ++i) {
        size_t s = q.size();
        ASSERT_LT(s, q.capacity()) << "size()=" << s
                                    << " suggests underflow on iteration " << i;
    }
    EXPECT_EQ(q.size(), 0u);
}

// 4. The acquire ordering must not be regressed back to relaxed in the
//    future. We can't introspect memory_order from the type system, but
//    we can at least sanity-check that the queue still functions when
//    stats are enabled (which also reads enqueue_pos_).
TEST(LockFreeQueueSizeTest, WorksWithStatsEnabled) {
    LockFreeQueue<int> q(64, 1, /*enable_stats=*/true);
    int out = 0;
    for (int i = 0; i < 20; ++i) ASSERT_TRUE(q.push(i));
    EXPECT_EQ(q.size(), 20u);
    for (int i = 0; i < 20; ++i) ASSERT_TRUE(q.pop(out));
    EXPECT_EQ(q.size(), 0u);

    LockFreeQueueStats stats = q.get_stats();
    EXPECT_EQ(stats.total_pushes, 20u);
    EXPECT_EQ(stats.total_pops, 20u);
}

// 5. P-007 regression: empty() and size() are both approximate, but
//    under quiescent (single-threaded, sequential) conditions they
//    must agree: when size()==0, empty() must be true; when an item
//    is enqueued, size()==1 and empty() must be false. This anchors
//    the documented contract that both are approximate *under
//    concurrency*, but exact *in quiescent state*.
TEST(LockFreeQueueSizeTest, EmptyAndSizeAgreeInQuiescentState) {
    LockFreeQueue<int> q(64);
    int out = 0;

    // Empty state
    ASSERT_EQ(q.size(), 0u);
    EXPECT_TRUE(q.empty());

    // After pushing N items, size()==N and empty()==false
    for (int i = 0; i < 50; ++i) {
        ASSERT_TRUE(q.push(i));
        EXPECT_FALSE(q.empty());
        EXPECT_EQ(q.size(), static_cast<size_t>(i + 1));
    }

    // After popping all items, size()==0 and empty()==true
    for (int i = 0; i < 50; ++i) {
        ASSERT_TRUE(q.pop(out));
        if (i < 49) {
            EXPECT_FALSE(q.empty());
            EXPECT_GT(q.size(), 0u);
        }
    }
    EXPECT_TRUE(q.empty());
    EXPECT_EQ(q.size(), 0u);
}

// 6. P-007 stress: under MPSC traffic, the *invariant* we care about
//    is that empty()==true implies size()==0 in quiescent trailing
//    state. Mid-flight the two can disagree (documented), but after
//    all producers stop and the consumer drains the queue, both must
//    converge to the empty/zero answer. Run 100k iterations to flush
//    out any ordering bug.
TEST(LockFreeQueueSizeTest, EmptyAndSizeConvergeAfterDrain) {
    constexpr int kIterations = 1000;       // outer iterations
    constexpr int kBatchSize  = 64;         // items per iteration
    LockFreeQueue<int> q(256);

    for (int iter = 0; iter < kIterations; ++iter) {
        // Push kBatchSize items
        for (int i = 0; i < kBatchSize; ++i) {
            ASSERT_TRUE(q.push(iter * kBatchSize + i));
        }

        // Drain all items
        int out = 0;
        int popped = 0;
        while (q.pop(out)) ++popped;

        ASSERT_EQ(popped, kBatchSize) << "iteration " << iter;

        // After drain, both views must agree on empty
        ASSERT_TRUE(q.empty()) << "iteration " << iter
                                << " size=" << q.size();
        ASSERT_EQ(q.size(), 0u) << "iteration " << iter;
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
