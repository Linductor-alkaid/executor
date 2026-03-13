#include <executor/executor.hpp>
#include <iostream>
#include <atomic>
#include <vector>

using namespace executor;

int main() {
    std::cout << "开始测试...\n";

    Executor executor;
    std::atomic<int> counter{0};

    std::cout << "准备 5000 个任务...\n";
    std::vector<std::function<void()>> tasks;
    for (int i = 0; i < 5000; ++i) {
        tasks.push_back([&counter]() {
            counter++;
        });
    }

    std::cout << "批量提交任务...\n";
    executor.submit_batch_no_future(tasks);

    std::cout << "调用 shutdown(true)...\n";
    executor.shutdown(true);  // 等待任务完成

    std::cout << "完成！计数器: " << counter.load() << "\n";
    return 0;
}
