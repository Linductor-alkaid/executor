#include <executor/comm.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

namespace {

struct SensorFrame {
    int sequence = 0;
    double value = 0.0;
};

TEST(FacadeCommUsage, SensorProducerPlannerConsumer) {
    executor::comm::ChannelOptions options;
    options.capacity = 16;
    executor::comm::MpscChannel<SensorFrame> frames(options);

    std::thread sensor([&] {
        for (int i = 0; i < 8; ++i) {
            while (!frames.try_send(SensorFrame{.sequence = i, .value = i * 0.5})) {
                std::this_thread::yield();
            }
        }
        frames.close();
    });

    std::vector<int> planned_sequences;
    SensorFrame frame;
    while (true) {
        const auto result = frames.receive_for(frame, std::chrono::milliseconds(500));
        if (!result) {
            EXPECT_EQ(result.error_code, executor::comm::CommErrorCode::Closed);
            break;
        }
        planned_sequences.push_back(frame.sequence);
    }

    sensor.join();

    ASSERT_EQ(planned_sequences.size(), 8U);
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(planned_sequences[static_cast<size_t>(i)], i);
    }
    EXPECT_EQ(frames.stats().sent_count, 8U);
    EXPECT_EQ(frames.stats().received_count, 8U);
}

TEST(FacadeCommUsage, ConfigThreadRealtimeControlThread) {
    struct ControlConfig {
        int gain = 0;
        bool enabled = false;
    };

    executor::comm::LatestMailbox<ControlConfig> config_box("control_config");
    config_box.publish(ControlConfig{.gain = 1, .enabled = true});
    config_box.publish(ControlConfig{.gain = 3, .enabled = true});

    uint64_t seen_sequence = 0;
    ControlConfig active_config;
    ASSERT_TRUE(config_box.try_load_newer_than(seen_sequence, active_config, seen_sequence));

    EXPECT_EQ(active_config.gain, 3);
    EXPECT_TRUE(active_config.enabled);
    EXPECT_EQ(seen_sequence, 2U);
    EXPECT_FALSE(config_box.try_load_newer_than(seen_sequence, active_config, seen_sequence));
    EXPECT_EQ(config_box.stats().overwritten_count, 1U);
}

TEST(FacadeCommUsage, InitThreadWorkerThread) {
    executor::comm::PhaseGate startup_gate("startup");
    std::atomic<bool> worker_started{false};
    std::atomic<bool> worker_completed{false};

    std::thread worker([&] {
        worker_started.store(true, std::memory_order_release);
        const auto result = startup_gate.wait_for(1, std::chrono::milliseconds(500));
        ASSERT_TRUE(result);
        worker_completed.store(true, std::memory_order_release);
    });

    while (!worker_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    EXPECT_FALSE(worker_completed.load(std::memory_order_acquire));
    EXPECT_TRUE(startup_gate.advance());

    worker.join();
    EXPECT_TRUE(worker_completed.load(std::memory_order_acquire));
    EXPECT_EQ(startup_gate.current_phase(), 1U);
    EXPECT_EQ(startup_gate.stats().received_count, 1U);
}

TEST(FacadeCommUsage, StateWriterMonitorReader) {
    struct SystemState {
        int tick = 0;
        int checksum = 0;
    };

    executor::comm::DoubleBuffer<SystemState> states(SystemState{.tick = 0, .checksum = 0});

    states.publish(SystemState{.tick = 1, .checksum = 17});
    states.update([](SystemState& state) {
        state.tick = 2;
        state.checksum = 34;
    });

    executor::comm::Snapshot<SystemState> snapshot;
    ASSERT_TRUE(states.load_newer_than(0, snapshot));
    EXPECT_EQ(snapshot.sequence, 2U);
    EXPECT_EQ(snapshot.value.tick, 2);
    EXPECT_EQ(snapshot.value.checksum, snapshot.value.tick * 17);

    EXPECT_FALSE(states.load_newer_than(snapshot.sequence, snapshot));
    EXPECT_EQ(states.stats().stale_read_count, 1U);
}

TEST(FacadeCommUsage, RealtimeCycleDrainsMessages) {
    executor::comm::RealtimeChannelOptions options;
    options.capacity = 8;
    options.max_items_per_cycle = 2;

    executor::comm::RealtimeChannel<int> commands(options);
    ASSERT_TRUE(commands.try_send(10));
    ASSERT_TRUE(commands.try_send(20));
    ASSERT_TRUE(commands.try_send(30));

    int applied_sum = 0;
    const size_t first_cycle = commands.drain_for_cycle([&](int command) {
        applied_sum += command;
    });

    EXPECT_EQ(first_cycle, 2U);
    EXPECT_EQ(applied_sum, 30);
    EXPECT_EQ(commands.stats().current_depth, 1U);

    const size_t second_cycle = commands.drain_for_cycle([&](int command) {
        applied_sum += command;
    });
    EXPECT_EQ(second_cycle, 1U);
    EXPECT_EQ(applied_sum, 60);
    EXPECT_TRUE(commands.empty());
}

} // namespace
