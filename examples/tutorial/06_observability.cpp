#include <atomic>
#include <exception>
#include <iostream>
#include <stdexcept>

#include <executor/executor.hpp>

int main() {
    executor::Executor executor;
    std::atomic<int> callbacks{0};
    executor.set_failure_callback([&](const executor::ExecutorFailureEvent&) { ++callbacks; });

    auto failed = executor.submit([]() -> int {
        throw std::runtime_error("expected observability failure");
    });

    try {
        static_cast<void>(failed.get());
    } catch (const std::exception&) {
    }

    const auto status = executor.get_failure_status();
    const auto recent = executor.get_recent_failures();
    std::cout << "failures=" << status.task_exception_count
              << ", callback=" << callbacks.load()
              << ", recent=" << recent.size() << '\n';

    executor.clear_recent_failures();
    executor.shutdown();
    return status.task_exception_count == 1 && callbacks == 1 && recent.size() == 1 ? 0 : 1;
}
