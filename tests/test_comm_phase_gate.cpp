#include <executor/comm.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace {

using executor::comm::CommErrorCode;
using executor::comm::CommEventKind;
using executor::comm::PhaseGate;
using executor::comm::Sequencer;

TEST(CommPhaseGateTest, WaitBeforeAdvanceWakesWhenPhaseReached) {
    PhaseGate gate("startup");
    std::atomic<bool> waiter_ready{false};
    std::atomic<bool> waiter_done{false};

    std::thread waiter([&] {
        waiter_ready.store(true, std::memory_order_release);
        const auto result = gate.wait_for(2, 1s);
        EXPECT_TRUE(result);
        waiter_done.store(true, std::memory_order_release);
    });

    while (!waiter_ready.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    EXPECT_TRUE(gate.advance());
    EXPECT_FALSE(waiter_done.load(std::memory_order_acquire));
    EXPECT_TRUE(gate.advance());

    waiter.join();
    EXPECT_TRUE(waiter_done.load(std::memory_order_acquire));
    EXPECT_EQ(gate.current_phase(), 2U);
    EXPECT_EQ(gate.stats().sent_count, 2U);
    EXPECT_EQ(gate.stats().received_count, 1U);
}

TEST(CommPhaseGateTest, WaitAfterAdvanceReturnsImmediately) {
    PhaseGate gate;
    EXPECT_TRUE(gate.advance_to(3));

    const auto result = gate.wait_for(2, 10ms);
    EXPECT_TRUE(result);
    EXPECT_TRUE(gate.has_reached(3));
    EXPECT_EQ(gate.stats().received_count, 1U);
}

TEST(CommPhaseGateTest, CloseWakesWaiter) {
    PhaseGate gate;
    std::atomic<bool> waiter_done{false};

    std::thread waiter([&] {
        const auto result = gate.wait_for(1, 1s);
        EXPECT_FALSE(result);
        EXPECT_EQ(result.error_code, CommErrorCode::Closed);
        waiter_done.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(20ms);
    EXPECT_FALSE(waiter_done.load(std::memory_order_acquire));
    EXPECT_TRUE(gate.close());
    waiter.join();
    EXPECT_TRUE(waiter_done.load(std::memory_order_acquire));
    EXPECT_TRUE(gate.is_closed());
}

TEST(CommPhaseGateTest, AdvanceToRejectsRollbackAndCountsMissedPhase) {
    PhaseGate gate("phase");
    int missed_events = 0;
    gate.set_event_callback([&](const executor::comm::CommEvent& event) noexcept {
        if (event.kind == CommEventKind::MissedPhase) {
            ++missed_events;
            EXPECT_EQ(event.component_name, "phase");
        }
    });

    EXPECT_TRUE(gate.advance_to(4));

    const auto result = gate.advance_to(3);
    EXPECT_FALSE(result);
    EXPECT_EQ(result.error_code, CommErrorCode::MissedPhase);
    EXPECT_EQ(gate.current_phase(), 4U);
    EXPECT_EQ(gate.stats().missed_phase_count, 1U);
    EXPECT_EQ(missed_events, 1);
}

TEST(CommPhaseGateTest, ExactWaitReportsMissedPhase) {
    PhaseGate gate;
    EXPECT_TRUE(gate.advance_to(5));

    const auto result = gate.wait_for_exact(3, 10ms);
    EXPECT_FALSE(result);
    EXPECT_EQ(result.error_code, CommErrorCode::MissedPhase);
    EXPECT_EQ(gate.stats().missed_phase_count, 1U);
}

TEST(CommPhaseGateTest, WaitTimeoutIsObservable) {
    PhaseGate gate;

    const auto result = gate.wait_for(1, 5ms);
    EXPECT_FALSE(result);
    EXPECT_EQ(result.error_code, CommErrorCode::Timeout);
    EXPECT_EQ(gate.stats().timeout_count, 1U);
}

TEST(CommPhaseGateTest, ConcurrentWaitersAllWake) {
    PhaseGate gate;
    constexpr int kWaiterCount = 8;
    std::atomic<int> completed{0};
    std::vector<std::thread> waiters;
    waiters.reserve(kWaiterCount);

    for (int i = 0; i < kWaiterCount; ++i) {
        waiters.emplace_back([&] {
            EXPECT_TRUE(gate.wait_for(1, 1s));
            completed.fetch_add(1, std::memory_order_acq_rel);
        });
    }

    std::this_thread::sleep_for(20ms);
    EXPECT_TRUE(gate.advance());

    for (auto& waiter : waiters) {
        waiter.join();
    }

    EXPECT_EQ(completed.load(std::memory_order_acquire), kWaiterCount);
    EXPECT_EQ(gate.stats().received_count, static_cast<uint64_t>(kWaiterCount));
}

TEST(CommSequencerTest, PublishesTicketsAndWaitsForExactTicket) {
    Sequencer sequencer("steps");
    const uint64_t first = sequencer.next_ticket();
    const uint64_t second = sequencer.next_ticket();
    EXPECT_EQ(first, 1U);
    EXPECT_EQ(second, 2U);

    std::atomic<bool> waiter_done{false};
    std::thread waiter([&] {
        const auto result = sequencer.wait_until_published(second, 1s);
        EXPECT_TRUE(result);
        waiter_done.store(true, std::memory_order_release);
    });

    EXPECT_TRUE(sequencer.publish(first));
    EXPECT_FALSE(waiter_done.load(std::memory_order_acquire));
    EXPECT_TRUE(sequencer.publish(second));

    waiter.join();
    EXPECT_TRUE(waiter_done.load(std::memory_order_acquire));
    EXPECT_TRUE(sequencer.is_published(second));
    EXPECT_EQ(sequencer.published_ticket(), second);
}

TEST(CommSequencerTest, MissedTicketReturnsMissedPhase) {
    Sequencer sequencer;
    EXPECT_TRUE(sequencer.publish(3));

    const auto result = sequencer.wait_until_published(2, 10ms);
    EXPECT_FALSE(result);
    EXPECT_EQ(result.error_code, CommErrorCode::MissedPhase);
    EXPECT_EQ(sequencer.stats().missed_phase_count, 1U);
}

TEST(CommSequencerTest, CloseWakesWaiter) {
    Sequencer sequencer;
    const uint64_t ticket = sequencer.next_ticket();
    std::atomic<bool> waiter_done{false};

    std::thread waiter([&] {
        const auto result = sequencer.wait_until_published(ticket, 1s);
        EXPECT_FALSE(result);
        EXPECT_EQ(result.error_code, CommErrorCode::Closed);
        waiter_done.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(20ms);
    EXPECT_FALSE(waiter_done.load(std::memory_order_acquire));
    EXPECT_TRUE(sequencer.close());
    waiter.join();
    EXPECT_TRUE(waiter_done.load(std::memory_order_acquire));
}

TEST(CommSequencerTest, RejectsInvalidAndDuplicateTickets) {
    Sequencer sequencer;

    const auto invalid = sequencer.wait_until_published(0, 1ms);
    EXPECT_FALSE(invalid);
    EXPECT_EQ(invalid.error_code, CommErrorCode::InvalidArgument);

    EXPECT_TRUE(sequencer.publish(1));
    const auto duplicate = sequencer.publish(1);
    EXPECT_FALSE(duplicate);
    EXPECT_EQ(duplicate.error_code, CommErrorCode::MissedPhase);
    EXPECT_EQ(sequencer.stats().missed_phase_count, 1U);
}

} // namespace
