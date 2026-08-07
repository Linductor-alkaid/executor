#include <executor/comm.hpp>

#include <gtest/gtest.h>

#include <string>

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

} // namespace
