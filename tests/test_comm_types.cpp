#include <executor/comm.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <type_traits>

namespace {

using namespace executor::comm;

TEST(CommTypesTest, CommResultDefaultsToSuccess) {
    CommResult result;

    EXPECT_TRUE(result);
    EXPECT_TRUE(result.ok);
    EXPECT_EQ(result.error_code, CommErrorCode::Ok);
    EXPECT_TRUE(result.message.empty());
}

TEST(CommTypesTest, CommResultFactoriesSetBoolAndDiagnostics) {
    const CommResult success = CommResult::success("closed cleanly");
    EXPECT_TRUE(success);
    EXPECT_EQ(success.error_code, CommErrorCode::Ok);
    EXPECT_EQ(success.message, "closed cleanly");

    const CommResult failure = CommResult::failure(CommErrorCode::Timeout, "receive timed out");
    EXPECT_FALSE(failure);
    EXPECT_FALSE(failure.ok);
    EXPECT_EQ(failure.error_code, CommErrorCode::Timeout);
    EXPECT_EQ(failure.message, "receive timed out");
}

TEST(CommTypesTest, ErrorCodesStringifyToStableNames) {
    EXPECT_STREQ(comm_error_code_to_string(CommErrorCode::Ok), "Ok");
    EXPECT_STREQ(comm_error_code_to_string(CommErrorCode::Closed), "Closed");
    EXPECT_STREQ(comm_error_code_to_string(CommErrorCode::Full), "Full");
    EXPECT_STREQ(comm_error_code_to_string(CommErrorCode::Empty), "Empty");
    EXPECT_STREQ(comm_error_code_to_string(CommErrorCode::Timeout), "Timeout");
    EXPECT_STREQ(comm_error_code_to_string(CommErrorCode::Stale), "Stale");
    EXPECT_STREQ(comm_error_code_to_string(CommErrorCode::MissedPhase), "MissedPhase");
    EXPECT_STREQ(comm_error_code_to_string(CommErrorCode::InvalidArgument), "InvalidArgument");
    EXPECT_STREQ(comm_error_code_to_string(CommErrorCode::NotReady), "NotReady");
    EXPECT_STREQ(comm_error_code_to_string(CommErrorCode::Unknown), "Unknown");
}

TEST(CommTypesTest, StringifyUnknownEnumValues) {
    EXPECT_STREQ(comm_error_code_to_string(static_cast<CommErrorCode>(999)), "Unknown");
    EXPECT_STREQ(drop_policy_to_string(static_cast<DropPolicy>(999)), "Unknown");
    EXPECT_STREQ(comm_event_kind_to_string(static_cast<CommEventKind>(999)), "Unknown");
}

TEST(CommTypesTest, DropPolicyAndEventKindStringifyToStableNames) {
    EXPECT_STREQ(drop_policy_to_string(DropPolicy::RejectNewest), "RejectNewest");
    EXPECT_STREQ(drop_policy_to_string(DropPolicy::DropOldest), "DropOldest");
    EXPECT_STREQ(drop_policy_to_string(DropPolicy::KeepLatest), "KeepLatest");

    EXPECT_STREQ(comm_event_kind_to_string(CommEventKind::Dropped), "Dropped");
    EXPECT_STREQ(comm_event_kind_to_string(CommEventKind::Overwritten), "Overwritten");
    EXPECT_STREQ(comm_event_kind_to_string(CommEventKind::ClosedSend), "ClosedSend");
    EXPECT_STREQ(comm_event_kind_to_string(CommEventKind::Timeout), "Timeout");
    EXPECT_STREQ(comm_event_kind_to_string(CommEventKind::StaleRead), "StaleRead");
    EXPECT_STREQ(comm_event_kind_to_string(CommEventKind::MissedPhase), "MissedPhase");
    EXPECT_STREQ(comm_event_kind_to_string(CommEventKind::ProducerLag), "ProducerLag");
    EXPECT_STREQ(comm_event_kind_to_string(CommEventKind::ConsumerLag), "ConsumerLag");
    EXPECT_STREQ(comm_event_kind_to_string(CommEventKind::LatencyHigh), "LatencyHigh");
    EXPECT_STREQ(comm_event_kind_to_string(CommEventKind::HandlerException), "HandlerException");
}

TEST(CommTypesTest, StatsDefaultToZeroLatencyAndCounts) {
    CommStats stats;

    EXPECT_EQ(stats.sent_count, 0U);
    EXPECT_EQ(stats.received_count, 0U);
    EXPECT_EQ(stats.dropped_count, 0U);
    EXPECT_EQ(stats.overwritten_count, 0U);
    EXPECT_EQ(stats.stale_read_count, 0U);
    EXPECT_EQ(stats.closed_send_count, 0U);
    EXPECT_EQ(stats.timeout_count, 0U);
    EXPECT_EQ(stats.handler_exception_count, 0U);
    EXPECT_EQ(stats.missed_phase_count, 0U);
    EXPECT_EQ(stats.current_depth, 0U);
    EXPECT_EQ(stats.peak_depth, 0U);
    EXPECT_EQ(stats.capacity, 0U);
    EXPECT_EQ(stats.producer_lag, 0U);
    EXPECT_EQ(stats.consumer_lag, 0U);
    EXPECT_EQ(stats.max_latency, std::chrono::nanoseconds{0});
    EXPECT_EQ(stats.avg_latency, std::chrono::nanoseconds{0});
}

TEST(CommTypesTest, EventDefaultsAreUsableAndCallbackIsInvocable) {
    CommEvent event;
    event.component_name = "channel";
    event.message = "drop";
    event.sequence = 42;

    bool called = false;
    CommEventCallback callback = [&](const CommEvent& observed) {
        called = true;
        EXPECT_EQ(observed.kind, CommEventKind::Dropped);
        EXPECT_EQ(observed.component_name, "channel");
        EXPECT_EQ(observed.message, "drop");
        EXPECT_EQ(observed.sequence, 42U);
    };

    callback(event);
    EXPECT_TRUE(called);
}

TEST(CommTypesTest, AggregatedHeaderExposesNamespaceTypes) {
    static_assert(std::is_same_v<decltype(CommStats{}.sent_count), uint64_t>);
    static_assert(std::is_same_v<CommEventCallback, std::function<void(const CommEvent&)>>);
}

} // namespace
