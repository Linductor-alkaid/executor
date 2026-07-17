#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

#include <executor/executor.hpp>

using namespace std::chrono_literals;

int main() {
    executor::Executor executor;
    std::atomic<int> cycles{0};
    std::atomic<int> commands{0};

    executor::RealtimeThreadConfig config;
    config.thread_name = "tutorial_rt";
    config.cycle_period_ns = 5'000'000;
    config.thread_priority = 0;
    config.enable_memory_lock = false;
    config.timer_slack_ns = 0;
    config.cycle_callback = [&] { ++cycles; };

    const auto registered = executor.register_realtime_task_ex("tutorial_rt", config);
    const auto started = registered ? executor.start_realtime_task_ex("tutorial_rt")
                                  : executor::ExecutorResult{};
    if (!registered || !started) {
        std::cerr << "realtime start failed\n";
        return 1;
    }

    const bool pushed = executor.try_push_realtime_task("tutorial_rt", [&] { ++commands; });
    std::this_thread::sleep_for(30ms);
    const auto status = executor.get_realtime_executor_status("tutorial_rt");
    executor.stop_realtime_task("tutorial_rt");

    std::cout << "realtime started=yes, command=" << (pushed ? "queued" : "rejected")
              << ", cycles=" << (status.cycle_count > 0 ? "observed" : "missing")
              << ", command ran=" << (commands.load() == 1 ? "yes" : "no") << '\n';
    executor.shutdown();
    return pushed && status.cycle_count > 0 && commands == 1 ? 0 : 1;
}
