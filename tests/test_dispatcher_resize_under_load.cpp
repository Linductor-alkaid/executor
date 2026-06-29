#include <executor/config.hpp>
#include "executor/thread_pool/thread_pool.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <future>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

using namespace executor;

#define TEST_ASSERT(condition, message)                                       \
    do {                                                                      \
        if (!(condition)) {                                                   \
            std::cerr << "FAILED: " << message << " at " << __FILE__ << ":"   \
                      << __LINE__ << std::endl;                               \
            return false;                                                     \
        }                                                                     \
    } while (0)

static bool test_dispatcher_resize_under_load() {
    std::cout << "[P-260629-001] dispatcher resize under load..." << std::endl;

    ThreadPoolConfig config;
    config.min_threads = 4;
    config.max_threads = 4;
    config.queue_capacity = 4096;
    config.enable_work_stealing = true;

    ThreadPool pool;
    TEST_ASSERT(pool.initialize(config), "pool initialize");

    std::atomic<bool> stop_resize{false};
    std::atomic<size_t> resize_calls{0};
    std::atomic<size_t> resize_failures{0};
    std::thread resizer([&]() {
        std::mt19937 rng(0x260629);
        std::uniform_int_distribution<int> coin(0, 1);
        size_t current = config.min_threads;

        while (!stop_resize.load(std::memory_order_acquire)) {
            current = coin(rng) == 0 ? current * 2 : (current + 1) / 2;
            if (current < 1) current = 1;
            if (current > 8) current = 8;

            if (pool.resize_local_queues(current)) {
                resize_calls.fetch_add(1, std::memory_order_relaxed);
            } else {
                resize_failures.fetch_add(1, std::memory_order_relaxed);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    constexpr size_t kTasks = 10000;
    std::atomic<size_t> completed{0};
    std::vector<std::future<void>> futures;
    futures.reserve(kTasks);

    const auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < kTasks; ++i) {
        futures.emplace_back(pool.submit([&completed]() noexcept {
            completed.fetch_add(1, std::memory_order_relaxed);
        }));
    }

    while (std::chrono::steady_clock::now() - start < std::chrono::seconds(5)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    stop_resize.store(true, std::memory_order_release);
    resizer.join();

    size_t future_timeouts = 0;
    for (auto& future : futures) {
        if (future.wait_for(std::chrono::seconds(10)) == std::future_status::ready) {
            future.get();
        } else {
            ++future_timeouts;
        }
    }

    pool.wait_for_completion();
    ThreadPoolStatus status = pool.get_status();
    pool.shutdown(true);

    std::cout << "  submitted=" << kTasks
              << " completed=" << completed.load(std::memory_order_relaxed)
              << " future_timeouts=" << future_timeouts
              << " resize_calls=" << resize_calls.load(std::memory_order_relaxed)
              << " resize_failures=" << resize_failures.load(std::memory_order_relaxed)
              << std::endl;

    TEST_ASSERT(resize_calls.load(std::memory_order_relaxed) > 0,
                "background resize should run");
    TEST_ASSERT(future_timeouts == 0, "no future timeout");
    TEST_ASSERT(completed.load(std::memory_order_relaxed) == kTasks,
                "completed task count must equal submitted count");
    TEST_ASSERT(status.completed_tasks == kTasks,
                "ThreadPool completed_tasks must equal submitted count");
    return true;
}

int main() {
    bool ok = test_dispatcher_resize_under_load();
    if (ok) {
        std::cout << "\n=== test_dispatcher_resize_under_load PASSED ==="
                  << std::endl;
        return 0;
    }
    std::cout << "\n=== test_dispatcher_resize_under_load FAILED ==="
              << std::endl;
    return 1;
}
