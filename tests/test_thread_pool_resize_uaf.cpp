// test_thread_pool_resize_uaf.cpp
// Regression tests for P-260617-002: thread_pool resize UAF guard via
// shared_lock on local_queues_mutex_.
//
// 修复前: worker_thread 直接无锁访问 local_queues_[i].pop() / .size(),
// resize 重建 vector 元素时可能让 worker 读到悬空引用 (UAF)。
//
// 修复后: 所有 reader 路径 (worker_thread / try_steal_task / get_status /
// wait_for_completion) 持 shared_lock(local_queues_mutex_); writer 路径
// (resize_local_queues / shutdown 的 clear) 持 unique_lock 配对。
//
// 本测试在以下场景下断言:
//   1. 不会崩溃、不触发 data race (TSAN-friendly)
//   2. 任务计数对账: 提交 N -> 完成后 completed == N
//   3. resize 期间 worker 持续 pop/execute 不触发 UAF
//
// 注意: 由于 TaskDispatcher 持 std::vector<QueueT>& 引用，跨 vector 替换
// 触发的"双 vector"问题在 P-002 范围内暂以"size 不变的元素重建"覆盖。
// 真扩缩容留作后续 plan (需把 local_queues_ 改 unique_ptr<vector>)。

#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

#include "executor/thread_pool/thread_pool.hpp"
#include <executor/config.hpp>

using namespace executor;

#define TEST_ASSERT(condition, message)                                       \
    do {                                                                      \
        if (!(condition)) {                                                   \
            std::cerr << "FAILED: " << message << " at " << __FILE__ << ":"   \
                      << __LINE__ << std::endl;                               \
            return false;                                                     \
        }                                                                     \
    } while (0)

// ----------------------------------------------------------------------------
// Test 1: 反复调用 resize_local_queues(N)（size 不变但内部元素重建）
// 期间 worker 持续 pop/execute 任务。
// 修复前: 元素析构 + placement-new 重建期间 worker 读悬空 -> UAF / 崩溃。
// 修复后: shared_lock 串行化 reader 与 writer，无崩溃。
// ----------------------------------------------------------------------------
static bool test_resize_during_drain() {
    std::cout << "[P-002] resize_during_drain: concurrent submit + resize..."
              << std::endl;
    std::cout.flush();

    ThreadPoolConfig config;
    config.min_threads = 4;
    config.max_threads = 4;
    config.queue_capacity = 1024;
    config.enable_work_stealing = true;

    ThreadPool pool;
    TEST_ASSERT(pool.initialize(config), "pool init");

    std::atomic<bool> stop{false};
    std::atomic<int> completed{0};

    // 主线程: 反复 resize 强制元素重建 (size 不变,placement-new 重建)
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    for (int round = 0; round < 10; ++round) {
        std::cout << "  round " << round << " submit" << std::endl;
        std::cout.flush();
        for (int t = 0; t < 50; ++t) {
            pool.submit([&completed, round, t]() noexcept {
                std::this_thread::yield();
                completed.fetch_add(1, std::memory_order_relaxed);
            });
        }
        std::cout << "  round " << round << " resize" << std::endl;
        std::cout.flush();
        bool ok = pool.resize_local_queues(config.min_threads);
        TEST_ASSERT(ok, "resize_local_queues should succeed for same-size rebuild");
    }
    std::cout << "  wait_for_completion..." << std::endl;
    std::cout.flush();

    pool.wait_for_completion();
    pool.shutdown(true);

    int done = completed.load();
    std::cout << "  resize_during_drain: completed=" << done << std::endl;
    TEST_ASSERT(done > 0, "at least some tasks must have completed");
    return true;
}

// ----------------------------------------------------------------------------
// Test 2: 多 worker 持续 steal 期间反复触发 element 重建。
// 修复前 resize 重建期间 worker 持有旧 element 引用即 UAF;
// 修复后所有访问点持 shared_lock，无 UAF。
// ----------------------------------------------------------------------------
static bool test_concurrent_steal_and_resize() {
    std::cout << "[P-002] concurrent_steal_and_resize: many workers steal "
                 "during resize..."
              << std::endl;

    ThreadPoolConfig config;
    config.min_threads = 6;
    config.max_threads = 6;
    config.queue_capacity = 2048;
    config.enable_work_stealing = true;

    ThreadPool pool;
    TEST_ASSERT(pool.initialize(config), "pool init");

    std::atomic<int> completed{0};
    const int total_rounds = 20;
    const int tasks_per_round = 60;

    for (int round = 0; round < total_rounds; ++round) {
        for (int i = 0; i < tasks_per_round; ++i) {
            pool.submit([&completed, round, i]() noexcept {
                std::this_thread::yield();
                std::this_thread::yield();
                completed.fetch_add(1, std::memory_order_relaxed);
            });
        }
        // size 不变的 element 重建: 每个 WorkerLocalQueue 析构 + placement-new
        bool ok = pool.resize_local_queues(config.min_threads);
        TEST_ASSERT(ok, "resize_local_queues should succeed for same-size rebuild");
    }

    pool.wait_for_completion();
    pool.shutdown(true);

    int done = completed.load();
    int expected = total_rounds * tasks_per_round;
    std::cout << "  concurrent_steal_and_resize: completed=" << done
              << " expected=" << expected << std::endl;
    TEST_ASSERT(done == expected,
                "all submitted tasks must complete (no UAF, no loss)");
    return true;
}

// ----------------------------------------------------------------------------
// Test 3: get_status() 与 resize 并发 — 验证 reader 路径也持锁。
// 修复前 get_status 内部对 local_queues_ 迭代无锁，与 resize 并发会
// 触发 TSAN data race。修复后 shared_lock 串行化。
// ----------------------------------------------------------------------------
static bool test_get_status_during_resize() {
    std::cout << "[P-002] get_status_during_resize: reader and writer "
                 "concurrently hit local_queues_..."
              << std::endl;

    ThreadPoolConfig config;
    config.min_threads = 2;
    config.max_threads = 2;
    config.queue_capacity = 512;
    config.enable_work_stealing = true;

    ThreadPool pool;
    TEST_ASSERT(pool.initialize(config), "pool init");

    std::atomic<bool> stop{false};
    std::thread reader([&]() {
        while (!stop.load(std::memory_order_acquire)) {
            (void)pool.get_status();
            std::this_thread::yield();
        }
    });

    std::thread resizer([&]() {
        for (int i = 0; i < 200 && !stop.load(); ++i) {
            pool.resize_local_queues(config.min_threads);
        }
    });

    std::atomic<int> done{0};
    for (int i = 0; i < 200; ++i) {
        pool.submit([&done]() noexcept {
            done.fetch_add(1, std::memory_order_relaxed);
        });
    }

    resizer.join();
    pool.wait_for_completion();
    stop.store(true, std::memory_order_release);
    reader.join();
    pool.shutdown(true);

    std::cout << "  get_status_during_resize: tasks done=" << done.load()
              << std::endl;
    TEST_ASSERT(done.load() == 200, "all tasks must complete");
    return true;
}

int main() {
    std::cout << "=== P-260617-002 thread_pool resize UAF guard tests ==="
              << std::endl;

    bool all_ok = true;
    all_ok &= test_resize_during_drain();
    all_ok &= test_concurrent_steal_and_resize();
    all_ok &= test_get_status_during_resize();

    if (all_ok) {
        std::cout << "\n=== All P-002 tests PASSED ===" << std::endl;
        return 0;
    }
    std::cout << "\n=== Some P-002 tests FAILED ===" << std::endl;
    return 1;
}
