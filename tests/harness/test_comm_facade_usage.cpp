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

TEST(FacadeCommUsage, DISABLED_ConfigThreadRealtimeControlThread) {
    GTEST_SKIP() << "TODO: enable when executor::comm::LatestMailbox<T> is introduced.";
}

TEST(FacadeCommUsage, DISABLED_InitThreadWorkerThread) {
    GTEST_SKIP() << "TODO: enable when executor::comm::PhaseGate is introduced.";
}

TEST(FacadeCommUsage, DISABLED_StateWriterMonitorReader) {
    GTEST_SKIP() << "TODO: enable when executor::comm::DoubleBuffer<T> is introduced.";
}

TEST(FacadeCommUsage, DISABLED_RealtimeCycleDrainsMessages) {
    GTEST_SKIP() << "TODO: enable when executor::comm::RealtimeChannel<T> is introduced.";
}

} // namespace
