#include <executor/lockfree_task_executor.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <atomic>
#include <algorithm>
#include <numeric>

using namespace executor;
using namespace std::chrono;

struct BenchResult {
    double avg_latency_ns = 0;
    double p50_latency_ns = 0;
    double p99_latency_ns = 0;
    double throughput_ops_per_sec = 0;
};

BenchResult benchmark_task_submission(size_t num_tasks) {
    LockFreeTaskExecutor exec(8192);
    exec.start();

    std::atomic<size_t> completed{0};
    std::vector<double> latencies_ns;
    latencies_ns.reserve(num_tasks);

    auto t0 = steady_clock::now();

    for (size_t i = 0; i < num_tasks; ++i) {
        auto submit_start = steady_clock::now();
        exec.push_task([&completed]() {
            completed.fetch_add(1, std::memory_order_relaxed);
        });
        auto submit_end = steady_clock::now();
        latencies_ns.push_back(duration<double, std::nano>(submit_end - submit_start).count());
    }

    auto t1 = steady_clock::now();

    // 等待任务完成
    while (completed.load() < num_tasks) {
        std::this_thread::sleep_for(microseconds(100));
    }

    exec.stop();

    double total_ns = duration<double, std::nano>(t1 - t0).count();

    std::sort(latencies_ns.begin(), latencies_ns.end());
    size_t p50_idx = latencies_ns.size() / 2;
    size_t p99_idx = static_cast<size_t>(0.99 * latencies_ns.size());

    BenchResult result;
    result.avg_latency_ns = std::accumulate(latencies_ns.begin(), latencies_ns.end(), 0.0) / latencies_ns.size();
    result.p50_latency_ns = latencies_ns[p50_idx];
    result.p99_latency_ns = latencies_ns[p99_idx];
    result.throughput_ops_per_sec = num_tasks * 1e9 / total_ns;

    return result;
}

BenchResult benchmark_throughput(size_t num_tasks) {
    LockFreeTaskExecutor exec(8192);
    exec.start();

    std::atomic<size_t> completed{0};

    auto t0 = steady_clock::now();

    for (size_t i = 0; i < num_tasks; ++i) {
        while (!exec.push_task([&completed]() {
            completed.fetch_add(1, std::memory_order_relaxed);
        })) {
            // 重试
        }
    }

    while (completed.load() < num_tasks) {
        std::this_thread::sleep_for(microseconds(10));
    }

    auto t1 = steady_clock::now();

    exec.stop();

    double total_ns = duration<double, std::nano>(t1 - t0).count();

    BenchResult result;
    result.throughput_ops_per_sec = num_tasks * 1e9 / total_ns;

    return result;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "LockFreeTaskExecutor Performance Benchmark\n";
    std::cout << "========================================\n\n";

    std::cout << std::fixed << std::setprecision(2);

    // 测试1: 任务提交延迟
    std::cout << "Test 1: Task Submission Latency (10,000 tasks)\n";
    auto result1 = benchmark_task_submission(10000);
    std::cout << "  Average latency: " << result1.avg_latency_ns << " ns\n";
    std::cout << "  p50 latency: " << result1.p50_latency_ns << " ns\n";
    std::cout << "  p99 latency: " << result1.p99_latency_ns << " ns\n";
    std::cout << "  Throughput: " << result1.throughput_ops_per_sec << " ops/s\n\n";

    // 测试2: 吞吐量
    std::cout << "Test 2: Throughput (100,000 tasks)\n";
    auto result2 = benchmark_throughput(100000);
    std::cout << "  Throughput: " << result2.throughput_ops_per_sec << " ops/s\n\n";

    std::cout << "========================================\n";
    std::cout << "Benchmark complete\n";
    std::cout << "========================================\n";

    return 0;
}
