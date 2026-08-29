#include <executor/executor.hpp>

#include <atomic>
#include <cassert>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <vector>

int main() {
    executor::Executor ex;
    executor::ExecutorConfig config;
    config.min_threads = 2;
    config.max_threads = 2;
    assert(ex.initialize(config));

    executor::SerialExecutionContext context;
    std::mutex mutex;
    std::vector<int> order;
    auto first = ex.submit_on(context, [&] { std::lock_guard<std::mutex> lock(mutex); order.push_back(1); });
    auto second = ex.submit_on(context, [&] { std::lock_guard<std::mutex> lock(mutex); order.push_back(2); });
    first.get();
    second.get();
    assert((order == std::vector<int>{1, 2}));

    auto failed = ex.submit_on(context, []() -> int { throw std::runtime_error("context failure"); });
    bool saw_failure = false;
    try { (void)failed.get(); } catch (const std::runtime_error&) { saw_failure = true; }
    assert(saw_failure);

    context.shutdown();
    auto rejected = ex.submit_on(context, [] { return 7; });
    bool saw_stopped = false;
    try { (void)rejected.get(); } catch (const executor::ExecutorStopping&) { saw_stopped = true; }
    assert(saw_stopped);
    ex.shutdown();
    return 0;
}
