#include <executor/comm.hpp>
#include <executor/realtime_thread_executor.hpp>

#include <atomic>
#include <chrono>
#include <gtest/gtest.h>

#include <string>
#include <thread>

namespace {

TEST(CommRealtimeMemoryTest, GuardIsExplicitAndReportsItsBuildMode) {
    executor::comm::RealtimeAllocationGuard::reset_current_thread_stats();
    {
        executor::comm::RealtimeAllocationGuard guard("control_loop", "drain");
        const auto stats = executor::comm::RealtimeAllocationGuard::current_thread_stats();
        if (executor::comm::RealtimeAllocationGuard::is_enabled()) {
            EXPECT_EQ(stats.component, "control_loop");
            EXPECT_EQ(stats.phase, "drain");
        } else {
            EXPECT_EQ(stats.allocation_count, 0U);
        }
    }
}

TEST(CommRealtimeMemoryTest, GuardedAllocationIsRecordedWhenEnabled) {
    executor::comm::RealtimeAllocationGuard::reset_current_thread_stats();
    {
        executor::comm::RealtimeAllocationGuard guard("let_mailbox", "publish");
        std::string allocation(128, 'x');
        (void)allocation;
    }
    const auto stats = executor::comm::RealtimeAllocationGuard::current_thread_stats();
    if (executor::comm::RealtimeAllocationGuard::is_enabled()) {
        EXPECT_GE(stats.allocation_count, 1U);
        EXPECT_GE(stats.allocated_bytes, 128U);
        EXPECT_EQ(stats.component, "let_mailbox");
        EXPECT_EQ(stats.phase, "publish");
    } else {
        EXPECT_EQ(stats.allocation_count, 0U);
    }
}

TEST(CommRealtimeMemoryTest, NestedGuardPreservesOuterContextAndCounts) {
    executor::comm::RealtimeAllocationGuard::reset_current_thread_stats();
    {
        executor::comm::RealtimeAllocationGuard outer("control_loop", "cycle");
        std::string outer_allocation(128, 'o');
        {
            executor::comm::RealtimeAllocationGuard inner("mailbox", "publish");
            std::string inner_allocation(128, 'i');
            (void)inner_allocation;
        }
        (void)outer_allocation;
    }
    const auto stats = executor::comm::RealtimeAllocationGuard::current_thread_stats();
    if (executor::comm::RealtimeAllocationGuard::is_enabled()) {
        EXPECT_EQ(stats.component, "control_loop");
        EXPECT_EQ(stats.phase, "cycle");
        EXPECT_GE(stats.allocation_count, 2U);
    } else {
        EXPECT_EQ(stats.allocation_count, 0U);
    }
}

TEST(CommRealtimeMemoryTest, AbortPolicyTerminatesOnAllocationWhenEnabled) {
    if (!executor::comm::RealtimeAllocationGuard::is_enabled()) {
        GTEST_SKIP() << "allocation guard is disabled in this build";
    }
#ifdef EXECUTOR_ENABLE_REALTIME_ALLOCATION_GUARD
    EXPECT_DEATH(
        {
            executor::comm::RealtimeAllocationGuard guard(
                "control_loop", "cycle",
                executor::comm::RealtimeAllocationViolationPolicy::Abort);
            volatile auto* allocation = new int(1);
            delete allocation;
        },
        "");
#endif
}

TEST(CommRealtimeMemoryTest, RealtimeThreadCanExplicitlyAttachGuardToCallback) {
    std::atomic<bool> callback_seen{false};
    std::atomic<bool> context_matches{false};
    executor::RealtimeThreadConfig config;
    config.thread_name = "allocation_guard_rt";
    config.cycle_period_ns = 1'000'000;
    config.enable_allocation_guard = true;
    config.cycle_callback = [&] {
        const auto stats = executor::comm::RealtimeAllocationGuard::current_thread_stats();
        if (executor::comm::RealtimeAllocationGuard::is_enabled()) {
            context_matches.store(stats.component == "allocation_guard_rt" &&
                                      stats.phase == "cycle_callback",
                                  std::memory_order_release);
        }
        callback_seen.store(true, std::memory_order_release);
    };

    executor::RealtimeThreadExecutor realtime("allocation_guard_rt", config);
    ASSERT_TRUE(realtime.start());
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
    while (!callback_seen.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    realtime.stop();
    ASSERT_TRUE(callback_seen.load(std::memory_order_acquire));
    if (executor::comm::RealtimeAllocationGuard::is_enabled()) {
        EXPECT_TRUE(context_matches.load(std::memory_order_acquire));
    }
}

} // namespace
