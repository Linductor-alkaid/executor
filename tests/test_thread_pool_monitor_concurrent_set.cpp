#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <executor/config.hpp>
#include <executor/monitor/task_monitor.hpp>
#include <executor/thread_pool/thread_pool.hpp>

using namespace executor;
using namespace std::chrono_literals;

#define TEST_ASSERT(condition, message)                                      \
    do {                                                                     \
        if (!(condition)) {                                                   \
            std::cerr << "FAILED: " << message << " at " << __FILE__       \
                      << ":" << __LINE__ << std::endl;                       \
            return false;                                                     \
        }                                                                    \
    } while (0)

namespace {

class CountingMonitor : public monitor::TaskMonitor {
public:
    void record_task_start(const std::string& task_id,
                           const std::string& task_type) override {
        starts_.fetch_add(1, std::memory_order_relaxed);
        monitor::TaskMonitor::record_task_start(task_id, task_type);
    }

    void record_task_complete(const std::string& task_id,
                              bool success,
                              int64_t execution_time_ns) override {
        completes_.fetch_add(1, std::memory_order_relaxed);
        monitor::TaskMonitor::record_task_complete(task_id, success,
                                                   execution_time_ns);
    }

    int starts() const {
        return starts_.load(std::memory_order_relaxed);
    }

    int completes() const {
        return completes_.load(std::memory_order_relaxed);
    }

private:
    std::atomic<int> starts_{0};
    std::atomic<int> completes_{0};
};

bool test_concurrent_monitor_set_unset() {
    ThreadPool pool;
    ThreadPoolConfig config;
    config.min_threads = 4;
    config.max_threads = 4;
    config.queue_capacity = 4096;
    TEST_ASSERT(pool.initialize(config), "thread pool should initialize");

    CountingMonitor monitor;
    std::atomic<bool> setter_ready{false};
    std::atomic<bool> stop_setter{false};
    std::atomic<int> ran{0};
    std::atomic<int> setter_iterations{0};

    std::thread setter([&]() {
        setter_ready.store(true, std::memory_order_release);
        while (!stop_setter.load(std::memory_order_acquire)) {
            pool.set_task_monitor(&monitor);
            pool.set_task_monitor(nullptr);
            setter_iterations.fetch_add(1, std::memory_order_relaxed);
        }
    });

    constexpr int kTaskCount = 2000;
    for (int i = 0; i < kTaskCount; ++i) {
        pool.submit([&]() {
            while (!setter_ready.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            ran.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::sleep_for(50us);
        });
    }

    pool.wait_for_completion();
    stop_setter.store(true, std::memory_order_release);
    setter.join();

    pool.set_task_monitor(nullptr);
    pool.shutdown();

    TEST_ASSERT(ran.load(std::memory_order_relaxed) == kTaskCount,
                "all submitted tasks should run");
    TEST_ASSERT(setter_iterations.load(std::memory_order_relaxed) > 0,
                "setter thread should race with task execution");

    std::cout << "  concurrent set_task_monitor set/unset: PASSED"
              << " (tasks=" << ran.load(std::memory_order_relaxed)
              << ", setter_iterations="
              << setter_iterations.load(std::memory_order_relaxed)
              << ", starts=" << monitor.starts()
              << ", completes=" << monitor.completes() << ")"
              << std::endl;
    return true;
}

}  // namespace

int main() {
    std::cout << "Testing ThreadPool concurrent task monitor replacement..."
              << std::endl;
    if (!test_concurrent_monitor_set_unset()) {
        std::cerr << "TEST FAILED" << std::endl;
        return 1;
    }
    std::cout << "ALL TESTS PASSED" << std::endl;
    return 0;
}
