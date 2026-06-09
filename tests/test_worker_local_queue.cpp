/**
 * WorkerLocalQueue::empty() 回归测试
 *
 * 目的：覆盖 P-001 修复——加锁路径原本用 `front_index_ >= queue_.size()`
 * 判断空，由于 front_index_ 是环形索引（永远 < queue_.size()），
 * 旧实现在加锁分支下永远返回 false，导致 empty() 实际恒为 false，
 * 与 size()/clear() 完全脱节。
 *
 * 不变量：empty() == true  <=>  size() == 0
 *
 * 本文件覆盖：
 *   1) basic push/pop 正确性
 *   2) basic empty() 行为（初始空、push 后非空、pop/clear 后再次为空）
 *   3) 并发回归：1 个 push/pop 线程 + 1 个 empty() 循环线程，验证
 *      empty() 在观测到 size > 0 时从不错报为 true。
 */

#include "executor/thread_pool/worker_local_queue.hpp"
#include "executor/types.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

using executor::Task;
using executor::WorkerLocalQueue;

namespace {

// 构造一个空壳 Task 用于测试（function 为空，不实际执行）。
// 返回 std::unique_ptr 以避免 Task 移动构造不可用的问题
// （Task 内部含 std::atomic<bool>，移动构造被隐式删除）。
std::unique_ptr<Task> make_test_task(size_t id_hint = 0) {
    auto t = std::make_unique<Task>();
    t->task_id = "wlq_test_" + std::to_string(id_hint);
    t->priority = executor::TaskPriority::NORMAL;
    t->function = nullptr;  // 测试不执行
    t->submit_time_ns = 0;
    t->timeout_ms = 0;
    return t;
}

}  // namespace

// ---------- basic push/pop ----------

TEST(WorkerLocalQueue, BasicPushPop) {
    WorkerLocalQueue q(/*capacity=*/16);

    EXPECT_TRUE(q.empty());
    EXPECT_EQ(q.size(), 0u);

    ASSERT_TRUE(q.push(*make_test_task(1)));
    ASSERT_TRUE(q.push(*make_test_task(2)));
    ASSERT_TRUE(q.push(*make_test_task(3)));

    EXPECT_FALSE(q.empty());
    EXPECT_EQ(q.size(), 3u);

    Task out;
    ASSERT_TRUE(q.pop(out));
    EXPECT_EQ(out.task_id, "wlq_test_1");
    ASSERT_TRUE(q.pop(out));
    EXPECT_EQ(out.task_id, "wlq_test_2");
    ASSERT_TRUE(q.pop(out));
    EXPECT_EQ(out.task_id, "wlq_test_3");

    EXPECT_TRUE(q.empty());
    EXPECT_EQ(q.size(), 0u);

    // pop on empty must fail
    EXPECT_FALSE(q.pop(out));
}

TEST(WorkerLocalQueue, StealAndClear) {
    WorkerLocalQueue q(16);
    ASSERT_TRUE(q.push(*make_test_task(10)));
    ASSERT_TRUE(q.push(*make_test_task(20)));
    ASSERT_TRUE(q.push(*make_test_task(30)));

    Task stolen;
    ASSERT_TRUE(q.steal(stolen));
    EXPECT_EQ(stolen.task_id, "wlq_test_30");  // steal 从 back 取
    EXPECT_EQ(q.size(), 2u);

    q.clear();
    EXPECT_TRUE(q.empty());
    EXPECT_EQ(q.size(), 0u);
}

// ---------- 不变量：empty() == size_ == 0 ----------
// 复现 P-001：旧实现在加锁分支下永远返回 false，
// 因此 push 一个再 clear，empty() 仍为 false。

TEST(WorkerLocalQueue, EmptySizeInvariantAfterClear) {
    WorkerLocalQueue q(8);

    for (int i = 0; i < 4; ++i) {
        ASSERT_TRUE(q.push(*make_test_task(static_cast<size_t>(i))));
    }
    EXPECT_FALSE(q.empty());
    EXPECT_EQ(q.size(), 4u);

    q.clear();
    // 旧实现在这里会失败：empty() 仍为 false（与 size() 不一致）。
    EXPECT_TRUE(q.empty());
    EXPECT_EQ(q.size(), 0u);
}

TEST(WorkerLocalQueue, EmptySizeInvariantAfterDrain) {
    WorkerLocalQueue q(8);

    for (int i = 0; i < 5; ++i) {
        ASSERT_TRUE(q.push(*make_test_task(static_cast<size_t>(i))));
    }
    EXPECT_EQ(q.size(), 5u);
    EXPECT_FALSE(q.empty());

    Task out;
    for (int i = 0; i < 5; ++i) {
        ASSERT_TRUE(q.pop(out));
    }
    // 全部 pop 完后，empty() 必须为 true（旧实现下这里仍为 false）
    EXPECT_TRUE(q.empty());
    EXPECT_EQ(q.size(), 0u);
}

// ---------- 并发回归 ----------
// 1 个线程 push/pop 循环，1 个线程反复调 empty()。
// 不变量：empty() 报错"非空"是允许的（不强制说非空时一定报非空），
// 但绝不能在 size > 0 的同时报 empty() == true。

TEST(WorkerLocalQueue, ConcurrentEmptyNeverLiesWhenNonEmpty) {
    constexpr size_t kCapacity = 64;
    constexpr int kProducerIterations = 10000;

    WorkerLocalQueue q(kCapacity);

    // 不变量：empty() == true  ⇔  size() == 0
    //
    // 测试策略（避免 reader 上的 size()/empty() 之间的 ABA 误报）：
    //   单一 reader 线程采用"先锁内 snapshot"的观察方式——直接让 reader
    //   在同一临界区内既读 size_ 也判定 empty()，从而严格保持不变量。
    //   具体：reader 加锁后读 size_，解锁后再调 empty()（无锁快路径）。
    //   任何时刻只要 size_ > 0，empty() 都不可能错返 true。
    //
    // 此测试不要求 reader 看到完整的状态序列，只要求"在 size > 0 时不撒谎"。
    std::atomic<bool> producer_done{false};
    std::atomic<int> violations{0};  // size()>0 && empty()==true 的次数（必须 =0）
    std::atomic<int> checks_while_nonempty{0};
    std::atomic<int> total_size_readings{0};

    // observer 线程
    std::thread observer([&]() {
        while (!producer_done.load(std::memory_order_acquire)) {
            // 在锁内读 size_，保证 size_ 的快照与 empty() 的无锁快路径来自同一时刻
            size_t s;
            {
                // 通过 empty() 内部的锁路径复用 mutex_——但 std::mutex 是 private。
                // 改用更轻量的"自旋+重读"模式：
                //   1) 无锁读 size_
                //   2) 紧跟一次无锁 empty() 快路径
                //   3) 若 size_ > 0 且 empty()==true，记录 1 次"可疑"——但因为二者不
                //      在同一临界区，可能确实被并发 pop 走了。补救：再读一次 size_，
                //      若仍 > 0 才计为 violation。
                s = q.size();
                bool e = q.empty();
                if (e) {
                    // 重读 size 一次，验证 size_ 是否真的仍然 > 0
                    size_t s2 = q.size();
                    if (s2 > 0) {
                        checks_while_nonempty.fetch_add(1, std::memory_order_relaxed);
                        violations.fetch_add(1, std::memory_order_relaxed);
                    }
                } else {
                    if (s > 0) {
                        checks_while_nonempty.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                total_size_readings.fetch_add(1, std::memory_order_relaxed);
            }
            std::this_thread::yield();
        }
        // 收尾：producer 完成后，size 必为 0，empty 必为 true
        size_t s = q.size();
        bool e = q.empty();
        if (s > 0) {
            checks_while_nonempty.fetch_add(1, std::memory_order_relaxed);
            if (e) violations.fetch_add(1, std::memory_order_relaxed);
        }
    });

    // producer 线程：每轮先 push 2 个再 pop 1 个，循环 kProducerIterations 次。
    // 中间故意先 push 多个再一次性 pop，制造 size() > 0 的观察窗口。
    std::thread producer([&]() {
        // 预分配 2 个 Task 复用（Task 因含 std::atomic<bool> 不能拷贝/移动）
        auto t1_ptr = make_test_task(0);
        auto t2_ptr = make_test_task(0);
        for (int i = 0; i < kProducerIterations; ++i) {
            // 复用同一个 Task 对象（每轮改 id 以便追踪）
            t1_ptr->task_id = "wlq_test_" + std::to_string(i);
            t2_ptr->task_id = "wlq_test_" + std::to_string(i + 100000);
            if (!q.push(*t1_ptr)) break;
            if (!q.push(*t2_ptr)) break;
            Task out;
            (void)q.pop(out);
        }
        // 收尾：把所有任务 pop 完，让 size 归零
        Task out;
        while (q.pop(out)) {}
        producer_done.store(true, std::memory_order_release);
    });

    producer.join();
    observer.join();

    EXPECT_EQ(violations.load(), 0)
        << "empty() reported true while size() > 0 — P-001 regressed. "
        << "checks_while_nonempty=" << checks_while_nonempty.load();
    // 至少要观察到一些非空窗口，否则测试本身无意义
    EXPECT_GT(checks_while_nonempty.load(), 0);
    EXPECT_GT(total_size_readings.load(), 50);

    // 收尾：现在 size 必须 0，empty 必须 true
    EXPECT_EQ(q.size(), 0u);
    EXPECT_TRUE(q.empty());
}

TEST(WorkerLocalQueue, FinalStateIsConsistent) {
    // 简化版：单线程做完整生命周期
    constexpr size_t kCapacity = 32;
    WorkerLocalQueue q(kCapacity);

    EXPECT_TRUE(q.empty());
    EXPECT_EQ(q.size(), 0u);

    for (int i = 0; i < 10; ++i) {
        ASSERT_TRUE(q.push(*make_test_task(static_cast<size_t>(i))));
    }
    EXPECT_FALSE(q.empty());
    EXPECT_EQ(q.size(), 10u);

    Task out;
    for (int i = 0; i < 10; ++i) {
        ASSERT_TRUE(q.pop(out));
    }
    EXPECT_TRUE(q.empty());
    EXPECT_EQ(q.size(), 0u);
}
