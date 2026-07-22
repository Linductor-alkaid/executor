#include <executor/comm.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace {

using executor::comm::CommEventKind;
using executor::comm::DoubleBuffer;
using executor::comm::Snapshot;

struct State {
    int version = 0;
    int checksum = 0;
    std::string label;
};

State make_state(int version) {
    return State{
        .version = version,
        .checksum = version * 17,
        .label = "state-" + std::to_string(version),
    };
}

void expect_consistent(const State& state) {
    EXPECT_EQ(state.checksum, state.version * 17);
    EXPECT_EQ(state.label, "state-" + std::to_string(state.version));
}

TEST(CommDoubleBufferTest, LoadReturnsInitialSnapshot) {
    DoubleBuffer<State> buffer(make_state(1), "state");

    const Snapshot<State> snapshot = buffer.load();
    EXPECT_EQ(snapshot.sequence, 0U);
    expect_consistent(snapshot.value);
    EXPECT_EQ(snapshot.value.version, 1);
    EXPECT_EQ(buffer.sequence(), 0U);
    EXPECT_EQ(buffer.stats().received_count, 1U);
}

TEST(CommDoubleBufferTest, PublishReturnsCompleteSnapshots) {
    DoubleBuffer<State> buffer(make_state(1));

    const uint64_t sequence = buffer.publish(make_state(2));
    EXPECT_EQ(sequence, 1U);

    const auto snapshot = buffer.load();
    EXPECT_EQ(snapshot.sequence, 1U);
    EXPECT_EQ(snapshot.value.version, 2);
    expect_consistent(snapshot.value);

    const auto stats = buffer.stats();
    EXPECT_EQ(stats.sent_count, 1U);
    EXPECT_EQ(stats.received_count, 1U);
    EXPECT_EQ(stats.capacity, 2U);
    EXPECT_EQ(stats.current_depth, 1U);
}

TEST(CommDoubleBufferTest, UpdateMutatesInactiveCopyThenPublishesOnce) {
    DoubleBuffer<State> buffer(make_state(3));

    const uint64_t sequence = buffer.update([](State& state) {
        state.version = 4;
        state.checksum = state.version * 17;
        state.label = "state-4";
    });

    EXPECT_EQ(sequence, 1U);
    const auto snapshot = buffer.load();
    EXPECT_EQ(snapshot.value.version, 4);
    expect_consistent(snapshot.value);
}

TEST(CommDoubleBufferTest, LoadNewerThanAvoidsDuplicateConsumption) {
    DoubleBuffer<State> buffer(make_state(1), "state_buffer");
    int stale_events = 0;
    buffer.set_event_callback([&](const executor::comm::CommEvent& event) noexcept {
        if (event.kind == CommEventKind::StaleRead) {
            ++stale_events;
            EXPECT_EQ(event.component_name, "state_buffer");
        }
    });

    Snapshot<State> snapshot;
    EXPECT_FALSE(buffer.load_newer_than(0, snapshot));
    EXPECT_EQ(buffer.stats().stale_read_count, 1U);
    EXPECT_EQ(stale_events, 1);

    const uint64_t sequence = buffer.publish(make_state(2));
    ASSERT_TRUE(buffer.load_newer_than(0, snapshot));
    EXPECT_EQ(snapshot.sequence, sequence);
    EXPECT_EQ(snapshot.value.version, 2);
    expect_consistent(snapshot.value);

    EXPECT_FALSE(buffer.load_newer_than(snapshot.sequence, snapshot));
    EXPECT_EQ(buffer.stats().stale_read_count, 2U);
}

TEST(CommDoubleBufferTest, MultipleReadersSeeConsistentSnapshots) {
    DoubleBuffer<State> buffer(make_state(0));
    std::atomic<bool> stop{false};
    std::atomic<int> readers_started{0};
    std::atomic<int> checked{0};
    std::vector<std::thread> readers;

    for (int i = 0; i < 4; ++i) {
        readers.emplace_back([&] {
            readers_started.fetch_add(1, std::memory_order_release);
            while (!stop.load(std::memory_order_acquire)) {
                const auto snapshot = buffer.load();
                EXPECT_EQ(snapshot.value.checksum, snapshot.value.version * 17);
                EXPECT_EQ(snapshot.value.label,
                          "state-" + std::to_string(snapshot.value.version));
                checked.fetch_add(1, std::memory_order_acq_rel);
            }
        });
    }

    while (readers_started.load(std::memory_order_acquire) < 4) {
        std::this_thread::yield();
    }
    while (checked.load(std::memory_order_acquire) == 0) {
        std::this_thread::yield();
    }

    for (int version = 1; version <= 200; ++version) {
        buffer.publish(make_state(version));
    }

    stop.store(true, std::memory_order_release);
    for (auto& reader : readers) {
        reader.join();
    }

    EXPECT_GT(checked.load(std::memory_order_acquire), 0);
    const auto latest = buffer.load();
    EXPECT_EQ(latest.sequence, 200U);
    EXPECT_EQ(latest.value.version, 200);
    expect_consistent(latest.value);
}

TEST(CommDoubleBufferTest, HighFrequencyWriterNeverExposesHalfUpdatedState) {
    DoubleBuffer<State> buffer(make_state(0));
    std::atomic<bool> stop{false};
    std::atomic<int> inconsistent{0};

    std::thread reader([&] {
        while (!stop.load(std::memory_order_acquire)) {
            const auto snapshot = buffer.load();
            if (snapshot.value.checksum != snapshot.value.version * 17 ||
                snapshot.value.label != "state-" + std::to_string(snapshot.value.version)) {
                inconsistent.fetch_add(1, std::memory_order_acq_rel);
            }
        }
    });

    for (int version = 1; version <= 500; ++version) {
        buffer.update([version](State& state) {
            state.version = version;
            state.checksum = version * 17;
            state.label = "state-" + std::to_string(version);
        });
    }

    stop.store(true, std::memory_order_release);
    reader.join();

    EXPECT_EQ(inconsistent.load(std::memory_order_acquire), 0);
    EXPECT_EQ(buffer.sequence(), 500U);
}

} // namespace
