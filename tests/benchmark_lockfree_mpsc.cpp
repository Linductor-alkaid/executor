/**
 * MPSC 性能基准测试（简化版）
 *
 * 测试不同生产者数量下的性能表现
 */

#include "executor/lockfree_task_executor.hpp"
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <iostream>
#include <iomanip>

using namespace executor;
using namespace std::chrono;

// 基准测试：吞吐量测试
void benchmark_throughput(int num_producers) {
    const int test_duration_ms = 1000;
    LockFreeTaskExecutor executor(16384);

    if (!executor.start()) {
        std::cerr << "Failed to start executor\n";
        return;
    }

    std::atomic<uint64_t> total_tasks{0};
    std::atomic<bool> stop_flag{false};
    std::vector<std::thread> producers;

    auto start = steady_clock::now();

    // 启动生产者线程
    for (int i = 0; i < num_producers; ++i) {
        producers.emplace_back([&]() {
            uint64_t local_count = 0;
            while (!stop_flag.load(std::memory_order_relaxed)) {
                bool success = executor.push_task([]() {
                    // 空任务，只测试队列性能
                });

                if (success) {
                    ++local_count;
                } else {
                    std::this_thread::yield();
                }
            }
            total_tasks.fetch_add(local_count, std::memory_order_relaxed);
        });
    }

    // 运行指定时间
    std::this_thread::sleep_for(milliseconds(test_duration_ms));
    stop_flag.store(true);

    // 等待生产者完成
    for (auto& t : producers) {
        t.join();
    }

    auto end = steady_clock::now();
    double duration_ms = duration_cast<milliseconds>(end - start).count();

    // 等待消费者处理完
    std::this_thread::sleep_for(milliseconds(200));

    uint64_t tasks = total_tasks.load();
    uint64_t processed = executor.processed_count();
    double throughput = tasks * 1000.0 / duration_ms;

    executor.stop();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "生产者数量: " << num_producers << "\n";
    std::cout << "  提交任务数: " << tasks << "\n";
    std::cout << "  处理任务数: " << processed << "\n";
    std::cout << "  测试时长: " << duration_ms << " ms\n";
    std::cout << "  吞吐量: " << throughput << " tasks/sec\n";
    std::cout << "\n";
}

int main() {
    std::cout << "\n=== MPSC 性能基准测试 ===\n\n";

    std::cout << "1. 吞吐量测试\n";
    std::cout << "----------------------------------------\n";

    benchmark_throughput(1);   // SPSC 基线
    benchmark_throughput(2);   // 2 生产者
    benchmark_throughput(4);   // 4 生产者
    benchmark_throughput(8);   // 8 生产者
    benchmark_throughput(16);  // 16 生产者

    std::cout << "测试完成！\n";
    return 0;
}
