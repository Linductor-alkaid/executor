#include <executor/comm.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using executor::comm::CommEventKind;
using executor::comm::DropPolicy;
using executor::comm::LatestMailbox;
using executor::comm::RealtimeChannel;
using executor::comm::RealtimeChannelOptions;

RealtimeChannelOptions realtime_options(
    size_t capacity,
    size_t max_items_per_cycle = 64,
    DropPolicy drop_policy = DropPolicy::RejectNewest) {
    RealtimeChannelOptions options;
    options.capacity = capacity;
    options.max_items_per_cycle = max_items_per_cycle;
    options.drop_policy = drop_policy;
    return options;
}

TEST(CommMailboxTest, TryLoadReturnsFalseBeforeFirstPublish) {
    LatestMailbox<int> mailbox("config");

    int value = 0;
    EXPECT_FALSE(mailbox.try_load(value));
    EXPECT_EQ(mailbox.sequence(), 0U);
    EXPECT_EQ(mailbox.stats().sent_count, 0U);
}

TEST(CommMailboxTest, PublishKeepsOnlyLatestValueAndSequence) {
    LatestMailbox<std::string> mailbox("config");

    mailbox.publish(std::string{"slow"});
    mailbox.publish(std::string{"fast"});

    std::string value;
    EXPECT_TRUE(mailbox.try_load(value));
    EXPECT_EQ(value, "fast");
    EXPECT_EQ(mailbox.sequence(), 2U);

    const auto stats = mailbox.stats();
    EXPECT_EQ(stats.sent_count, 2U);
    EXPECT_EQ(stats.received_count, 1U);
    EXPECT_EQ(stats.overwritten_count, 1U);
    EXPECT_EQ(stats.capacity, 1U);
    EXPECT_EQ(stats.current_depth, 1U);
}

TEST(CommMailboxTest, TryLoadNewerThanAvoidsDuplicateConsumption) {
    LatestMailbox<int> mailbox;

    mailbox.publish(10);

    int value = 0;
    uint64_t sequence = 0;
    EXPECT_TRUE(mailbox.try_load_newer_than(0, value, sequence));
    EXPECT_EQ(value, 10);
    EXPECT_EQ(sequence, 1U);

    EXPECT_FALSE(mailbox.try_load_newer_than(sequence, value, sequence));
    EXPECT_EQ(mailbox.stats().stale_read_count, 1U);

    mailbox.publish(20);
    EXPECT_TRUE(mailbox.try_load_newer_than(sequence, value, sequence));
    EXPECT_EQ(value, 20);
    EXPECT_EQ(sequence, 2U);
}

TEST(CommMailboxTest, EmitsOverwriteAndStaleEventsOutsideLock) {
    LatestMailbox<int> mailbox("control_config");
    std::vector<CommEventKind> events;
    mailbox.set_event_callback([&](const executor::comm::CommEvent& event) {
        EXPECT_EQ(event.component_name, "control_config");
        events.push_back(event.kind);
    });

    mailbox.publish(1);
    mailbox.publish(2);

    int value = 0;
    uint64_t sequence = 0;
    EXPECT_FALSE(mailbox.try_load_newer_than(2, value, sequence));

    ASSERT_EQ(events.size(), 2U);
    EXPECT_EQ(events[0], CommEventKind::Overwritten);
    EXPECT_EQ(events[1], CommEventKind::StaleRead);
}

TEST(CommRealtimeChannelTest, DrainForCycleUsesConfiguredBudget) {
    RealtimeChannel<int> channel(realtime_options(8, 3));
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(channel.try_send(i));
    }

    std::vector<int> drained;
    const size_t count = channel.drain_for_cycle([&](int value) {
        drained.push_back(value);
    });

    EXPECT_EQ(count, 3U);
    ASSERT_EQ(drained.size(), 3U);
    EXPECT_EQ(drained[0], 0);
    EXPECT_EQ(drained[1], 1);
    EXPECT_EQ(drained[2], 2);
    EXPECT_EQ(channel.stats().received_count, 3U);
    EXPECT_EQ(channel.stats().current_depth, 2U);
}

TEST(CommRealtimeChannelTest, DrainForCycleCanOverrideBudget) {
    RealtimeChannel<int> channel(realtime_options(8, 1));
    EXPECT_TRUE(channel.try_send(1));
    EXPECT_TRUE(channel.try_send(2));

    int sum = 0;
    const size_t count = channel.drain_for_cycle([&](int value) {
        sum += value;
    }, 2);

    EXPECT_EQ(count, 2U);
    EXPECT_EQ(sum, 3);
    EXPECT_TRUE(channel.empty());
}

TEST(CommRealtimeChannelTest, ZeroConfiguredBudgetDrainsUntilEmpty) {
    RealtimeChannel<int> channel(realtime_options(8, 0));
    EXPECT_TRUE(channel.try_send(1));
    EXPECT_TRUE(channel.try_send(2));
    EXPECT_TRUE(channel.try_send(3));

    int sum = 0;
    const size_t count = channel.drain_for_cycle([&](int value) {
        sum += value;
    });

    EXPECT_EQ(count, 3U);
    EXPECT_EQ(sum, 6);
    EXPECT_TRUE(channel.empty());
}

TEST(CommRealtimeChannelTest, RejectNewestCountsDroppedMessages) {
    RealtimeChannel<int> channel(realtime_options(2, 64));

    EXPECT_TRUE(channel.try_send(1));
    EXPECT_TRUE(channel.try_send(2));
    EXPECT_FALSE(channel.try_send(3));

    const auto stats = channel.stats();
    EXPECT_EQ(stats.sent_count, 2U);
    EXPECT_EQ(stats.dropped_count, 1U);
    EXPECT_EQ(stats.current_depth, 2U);
}

TEST(CommRealtimeChannelTest, DropOldestKeepsNewerMessages) {
    RealtimeChannel<int> channel(realtime_options(2, 64, DropPolicy::DropOldest));

    EXPECT_TRUE(channel.try_send(1));
    EXPECT_TRUE(channel.try_send(2));
    EXPECT_TRUE(channel.try_send(3));

    std::vector<int> drained;
    EXPECT_EQ(channel.drain_for_cycle([&](int value) { drained.push_back(value); }), 2U);

    ASSERT_EQ(drained.size(), 2U);
    EXPECT_EQ(drained[0], 2);
    EXPECT_EQ(drained[1], 3);
    EXPECT_EQ(channel.stats().dropped_count, 1U);
}

TEST(CommRealtimeChannelTest, KeepLatestOverwritesBacklog) {
    RealtimeChannel<int> channel(realtime_options(2, 64, DropPolicy::KeepLatest));

    EXPECT_TRUE(channel.try_send(1));
    EXPECT_TRUE(channel.try_send(2));
    EXPECT_TRUE(channel.try_send(3));

    int value = 0;
    EXPECT_EQ(channel.drain_for_cycle([&](int item) { value = item; }), 1U);
    EXPECT_EQ(value, 3);
    EXPECT_EQ(channel.stats().overwritten_count, 1U);
}

TEST(CommRealtimeChannelTest, CloseRejectsNewSendsButDrainsBufferedMessages) {
    RealtimeChannel<int> channel(realtime_options(2));
    EXPECT_TRUE(channel.try_send(7));

    channel.close();
    EXPECT_TRUE(channel.is_closed());
    EXPECT_FALSE(channel.try_send(8));

    int value = 0;
    EXPECT_EQ(channel.drain_for_cycle([&](int item) { value = item; }), 1U);
    EXPECT_EQ(value, 7);
    EXPECT_TRUE(channel.empty());
    EXPECT_EQ(channel.stats().closed_send_count, 1U);
}

TEST(CommRealtimeChannelTest, SupportsMoveOnlyPayloads) {
    RealtimeChannel<std::unique_ptr<int>> channel(realtime_options(2));

    EXPECT_TRUE(channel.try_send(std::make_unique<int>(42)));

    int value = 0;
    EXPECT_EQ(channel.drain_for_cycle([&](std::unique_ptr<int> item) {
        ASSERT_NE(item, nullptr);
        value = *item;
    }), 1U);
    EXPECT_EQ(value, 42);
}

TEST(CommRealtimeChannelTest, HandlerExceptionStopsCurrentDrainAndRethrows) {
    RealtimeChannel<int> channel(realtime_options(4));
    channel.try_send(1);
    channel.try_send(2);

    std::atomic<int> exception_events{0};
    channel.set_event_callback([&](const executor::comm::CommEvent& event) noexcept {
        if (event.kind == CommEventKind::HandlerException) {
            exception_events.fetch_add(1, std::memory_order_relaxed);
        }
    });

    EXPECT_THROW(
        channel.drain_for_cycle([](int) {
            throw std::runtime_error("handler failed");
        }),
        std::runtime_error);

    EXPECT_EQ(exception_events.load(std::memory_order_relaxed), 1);
    EXPECT_EQ(channel.stats().handler_exception_count, 1U);
    EXPECT_EQ(channel.stats().received_count, 1U);
    EXPECT_EQ(channel.stats().current_depth, 1U);
}

} // namespace
