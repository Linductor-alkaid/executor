/**
 * @brief Robot control pipeline using executor::comm facade.
 *
 * Scenario:
 * - A sensor thread publishes frames to a planner through MpscChannel.
 * - A configuration thread publishes the latest control config through LatestMailbox.
 * - The planner sends bounded commands to the real-time control loop through RealtimeChannel.
 * - The planner publishes complete system snapshots through DoubleBuffer for monitors.
 * - Startup ordering uses PhaseGate.
 * - CPU bootstrap tasks use TaskHandle / when_all before the pipeline starts.
 */

#include <executor/comm.hpp>
#include <executor/executor.hpp>

#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

using namespace std::chrono_literals;

namespace {

struct SensorFrame {
    int sequence = 0;
    double speed_mps = 0.0;
};

struct ControlConfig {
    double gain = 1.0;
    bool enabled = true;
};

struct ControlCommand {
    int frame_sequence = 0;
    double throttle = 0.0;
};

struct SystemState {
    int last_frame = -1;
    double speed_mps = 0.0;
    double throttle = 0.0;
};

void print_stats(const char* name, const executor::comm::CommStats& stats) {
    std::cout << name
              << ": sent=" << stats.sent_count
              << ", received=" << stats.received_count
              << ", dropped=" << stats.dropped_count
              << ", stale=" << stats.stale_read_count
              << ", missed_phase=" << stats.missed_phase_count
              << ", depth=" << stats.current_depth
              << ", peak=" << stats.peak_depth
              << ", producer_lag=" << stats.producer_lag
              << ", consumer_lag=" << stats.consumer_lag
              << ", max_latency_ns=" << stats.max_latency.count()
              << "\n";
}

} // namespace

int main() {
    executor::comm::ChannelOptions frame_options;
    frame_options.capacity = 16;
    frame_options.name = "sensor_frames";

    executor::comm::RealtimeChannelOptions command_options;
    command_options.capacity = 8;
    command_options.max_items_per_cycle = 2;
    command_options.name = "control_commands";

    executor::comm::MpscChannel<SensorFrame> sensor_frames(frame_options);
    executor::comm::LatestMailbox<ControlConfig> control_config("control_config");
    executor::comm::RealtimeChannel<ControlCommand> control_commands(command_options);
    executor::comm::PhaseGate startup("startup");
    executor::comm::DoubleBuffer<SystemState> system_state(SystemState{}, "system_state");

    auto comm_event_logger = [](const executor::comm::CommEvent& event) {
        std::cout << "[comm] " << event.component_name << ": "
                  << executor::comm::comm_event_kind_to_string(event.kind)
                  << " (" << event.message << ")\n";
    };
    sensor_frames.set_event_callback(comm_event_logger);
    control_config.set_event_callback(comm_event_logger);
    control_commands.set_event_callback(comm_event_logger);
    startup.set_event_callback(comm_event_logger);
    system_state.set_event_callback(comm_event_logger);

    executor::Executor executor;
    executor::ExecutorConfig executor_config;
    executor_config.min_threads = 2;
    executor_config.max_threads = 2;
    executor_config.queue_capacity = 64;
    executor.initialize(executor_config);

    auto load_map = executor.submit_with_handle([] {
        std::this_thread::sleep_for(2ms);
        return std::string{"map-ready"};
    });
    auto calibrate = executor.submit_with_handle([] {
        std::this_thread::sleep_for(2ms);
        return std::string{"calibration-ready"};
    });
    auto prerequisites = executor.when_all({load_map.handle, calibrate.handle});
    auto bootstrap = executor.submit_after(prerequisites, [&] {
        control_config.publish(ControlConfig{1.4, true});
        startup.advance_to(1);
        return load_map.future.get() + ", " + calibrate.future.get();
    });

    std::atomic<bool> planner_done{false};

    std::thread sensor_thread([&] {
        if (!startup.wait_for(1, 1s)) {
            return;
        }

        for (int i = 0; i < 8; ++i) {
            while (!sensor_frames.try_send(SensorFrame{i, 2.0 + i * 0.25})) {
                std::this_thread::yield();
            }
            std::this_thread::sleep_for(1ms);
        }
        sensor_frames.close();
    });

    std::thread planner_thread([&] {
        if (!startup.wait_for(1, 1s)) {
            planner_done.store(true, std::memory_order_release);
            return;
        }

        SensorFrame frame;
        while (sensor_frames.receive_for(frame, 100ms)) {
            ControlConfig config;
            uint64_t config_sequence = 0;
            if (!control_config.try_load_newer_than(0, config, config_sequence)) {
                config = ControlConfig{};
            }

            const double throttle = config.enabled ? frame.speed_mps * config.gain : 0.0;
            control_commands.try_send(ControlCommand{frame.sequence, throttle});
            system_state.publish(SystemState{frame.sequence, frame.speed_mps, throttle});
        }

        planner_done.store(true, std::memory_order_release);
    });

    std::thread realtime_thread([&] {
        if (!startup.wait_for(1, 1s)) {
            return;
        }

        while (!planner_done.load(std::memory_order_acquire) || !control_commands.empty()) {
            control_commands.drain_for_cycle([](const ControlCommand& command) {
                std::cout << "[rt] frame=" << command.frame_sequence
                          << ", throttle=" << command.throttle << "\n";
            });
            std::this_thread::sleep_for(1ms);
        }
    });

    std::thread monitor_thread([&] {
        if (!startup.wait_for(1, 1s)) {
            return;
        }

        uint64_t last_seen = 0;
        executor::comm::Snapshot<SystemState> snapshot;
        while (!planner_done.load(std::memory_order_acquire)) {
            if (system_state.load_newer_than(last_seen, snapshot)) {
                last_seen = snapshot.sequence;
                std::cout << "[monitor] frame=" << snapshot.value.last_frame
                          << ", speed=" << snapshot.value.speed_mps
                          << ", throttle=" << snapshot.value.throttle << "\n";
            }
            std::this_thread::sleep_for(2ms);
        }
    });

    const auto bootstrap_result = bootstrap.get();
    std::cout << "bootstrap: " << bootstrap_result << "\n";

    sensor_thread.join();
    planner_thread.join();
    realtime_thread.join();
    monitor_thread.join();

    print_stats("sensor_frames", sensor_frames.stats());
    print_stats("control_config", control_config.stats());
    print_stats("control_commands", control_commands.stats());
    print_stats("startup", startup.stats());
    print_stats("system_state", system_state.stats());

    executor.shutdown();
    return 0;
}
