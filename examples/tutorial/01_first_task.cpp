#include <exception>
#include <iostream>
#include <stdexcept>

#include <executor/executor.hpp>

int main() {
    auto& executor = executor::Executor::instance();

    auto answer = executor.submit([] { return 42; });
    std::cout << "answer=" << answer.get() << '\n';

    auto failing_task = executor.submit([]() -> int {
        throw std::runtime_error("expected tutorial failure");
    });

    try {
        static_cast<void>(failing_task.get());
    } catch (const std::exception& error) {
        std::cout << "task failed: " << error.what() << '\n';
    }

    executor.shutdown();
    return 0;
}
