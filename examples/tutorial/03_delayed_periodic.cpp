#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

#include <executor/executor.hpp>

using namespace std::chrono_literals;

int main() {
    executor::Executor executor;
    executor::ExecutorConfig config;
    config.min_threads = 1;
    config.max_threads = 1;
    if (!executor.initialize_ex(config)) {
        return 1;
    }

    auto retry = executor.submit_delayed(1, [] { return "retry complete"; });
    std::atomic<int> health_checks{0};
    const auto task_id = executor.submit_periodic(5, [&] { ++health_checks; });

    std::this_thread::sleep_for(30ms);
    const auto status = executor.get_periodic_task_status(task_id);
    const bool cancelled = executor.cancel_task(task_id);

    std::cout << retry.get() << '\n';
    std::cout << "health checks=" << health_checks.load() << '\n';
    std::cout << "periodic status=" << (status && status->execution_count > 0 ? "running" : "missing")
              << ", cancelled=" << (cancelled ? "yes" : "no") << '\n';

    executor.shutdown();
    return status && status->execution_count > 0 && cancelled ? 0 : 1;
}
