/**
 * 多线程测试 - 定位段错误问题
 */

#include "executor/lockfree_task_executor.hpp"
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>

using namespace executor;

int main() {
    std::cout << "测试: 多线程提交任务\n";

    LockFreeTaskExecutor executor(4096);
    executor.start();

    std::atomic<int> submitted{0};
    std::vector<std::thread> threads;

    // 启动4个线程，每个提交1000个任务
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < 1000; ++i) {
                bool success = executor.push_task([]() {
                    // 空任务
                });
                if (success) {
                    submitted.fetch_add(1);
                }
            }
            std::cout << "线程 " << t << " 完成\n";
        });
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }

    std::cout << "提交任务数: " << submitted.load() << "\n";

    // 等待任务处理完成
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    std::cout << "处理任务数: " << executor.processed_count() << "\n";

    executor.stop();

    std::cout << "测试完成！\n";
    return 0;
}
