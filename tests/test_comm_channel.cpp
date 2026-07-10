#include <executor/comm.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace std::chrono_literals;

namespace {

using executor::comm::ChannelOptions;
using executor::comm::CommErrorCode;
using executor::comm::CommEventKind;
using executor::comm::DropPolicy;
using executor::comm::MpscChannel;
using executor::comm::SpscChannel;

ChannelOptions options(size_t capacity,
                       DropPolicy drop_policy = DropPolicy::RejectNewest) {
    ChannelOptions opts;
    opts.capacity = capacity;
    opts.drop_policy = drop_policy;
    return opts;
}

TEST(CommChannelTest, SingleProducerSingleConsumerKeepsFifoOrder) {
    MpscChannel<int> channel(options(4));

    EXPECT_TRUE(channel.try_send(1));
    EXPECT_TRUE(channel.try_send(2));
    EXPECT_TRUE(channel.try_send(3));

    int value = 0;
    EXPECT_TRUE(channel.try_receive(value));
    EXPECT_EQ(value, 1);
    EXPECT_TRUE(channel.try_receive(value));
    EXPECT_EQ(value, 2);
    EXPECT_TRUE(channel.try_receive(value));
    EXPECT_EQ(value, 3);
    EXPECT_FALSE(channel.try_receive(value));

    const auto stats = channel.stats();
    EXPECT_EQ(stats.sent_count, 3U);
    EXPECT_EQ(stats.received_count, 3U);
    EXPECT_EQ(stats.peak_depth, 3U);
    EXPECT_EQ(stats.current_depth, 0U);
    EXPECT_EQ(channel.capacity(), 4U);
    EXPECT_TRUE(channel.empty());
}

TEST(CommChannelTest, MultipleProducersSingleConsumerReceivesEachValueOnce) {
    constexpr int kProducerCount = 4;
    constexpr int kItemsPerProducer = 250;
    constexpr int kTotalItems = kProducerCount * kItemsPerProducer;

    MpscChannel<int> channel(options(128));
    std::atomic<int> producers_done{0};
    std::vector<std::thread> producers;

    for (int producer = 0; producer < kProducerCount; ++producer) {
        producers.emplace_back([&, producer] {
            for (int i = 0; i < kItemsPerProducer; ++i) {
                const int value = producer * kItemsPerProducer + i;
                while (!channel.try_send(value)) {
                    std::this_thread::yield();
                }
            }
            producers_done.fetch_add(1, std::memory_order_acq_rel);
        });
    }

    std::unordered_set<int> received;
    received.reserve(kTotalItems);
    while (static_cast<int>(received.size()) < kTotalItems) {
        int value = -1;
        if (channel.try_receive(value)) {
            received.insert(value);
            continue;
        }
        EXPECT_LT(producers_done.load(std::memory_order_acquire), kProducerCount);
        std::this_thread::yield();
    }

    for (auto& producer : producers) {
        producer.join();
    }

    EXPECT_EQ(received.size(), static_cast<size_t>(kTotalItems));
    for (int value = 0; value < kTotalItems; ++value) {
        EXPECT_EQ(received.count(value), 1U);
    }
    EXPECT_EQ(channel.stats().sent_count, static_cast<uint64_t>(kTotalItems));
    EXPECT_EQ(channel.stats().received_count, static_cast<uint64_t>(kTotalItems));
}

TEST(CommChannelTest, RejectNewestReportsFullAndDropStats) {
    MpscChannel<int> channel(options(2));
    int dropped_events = 0;
    channel.set_event_callback([&](const executor::comm::CommEvent& event) noexcept {
        if (event.kind == CommEventKind::Dropped) {
            ++dropped_events;
        }
    });

    EXPECT_TRUE(channel.try_send(10));
    EXPECT_TRUE(channel.try_send(11));
    EXPECT_FALSE(channel.try_send(12));

    auto stats = channel.stats();
    EXPECT_EQ(stats.sent_count, 2U);
    EXPECT_EQ(stats.dropped_count, 1U);
    EXPECT_EQ(stats.current_depth, 2U);
    EXPECT_EQ(dropped_events, 1);

    int value = 0;
    EXPECT_TRUE(channel.try_receive(value));
    EXPECT_EQ(value, 10);
    EXPECT_TRUE(channel.try_receive(value));
    EXPECT_EQ(value, 11);
}

TEST(CommChannelTest, DropOldestKeepsNewerValuesAndCountsDrop) {
    MpscChannel<int> channel(options(2, DropPolicy::DropOldest));

    EXPECT_TRUE(channel.try_send(1));
    EXPECT_TRUE(channel.try_send(2));
    EXPECT_TRUE(channel.try_send(3));

    int value = 0;
    EXPECT_TRUE(channel.try_receive(value));
    EXPECT_EQ(value, 2);
    EXPECT_TRUE(channel.try_receive(value));
    EXPECT_EQ(value, 3);
    EXPECT_FALSE(channel.try_receive(value));
    EXPECT_EQ(channel.stats().dropped_count, 1U);
}

TEST(CommChannelTest, KeepLatestOverwritesBufferedValues) {
    MpscChannel<int> channel(options(2, DropPolicy::KeepLatest));

    EXPECT_TRUE(channel.try_send(1));
    EXPECT_TRUE(channel.try_send(2));
    EXPECT_TRUE(channel.try_send(3));

    int value = 0;
    EXPECT_TRUE(channel.try_receive(value));
    EXPECT_EQ(value, 3);
    EXPECT_FALSE(channel.try_receive(value));
    EXPECT_EQ(channel.stats().overwritten_count, 1U);
}

TEST(CommChannelTest, CloseRejectsNewSendsButAllowsDrainingBufferedValues) {
    MpscChannel<int> channel(options(2));

    EXPECT_TRUE(channel.try_send(7));
    channel.close();
    EXPECT_TRUE(channel.is_closed());
    EXPECT_FALSE(channel.try_send(8));

    int value = 0;
    EXPECT_TRUE(channel.try_receive(value));
    EXPECT_EQ(value, 7);
    EXPECT_FALSE(channel.try_receive(value));
    EXPECT_EQ(channel.stats().closed_send_count, 1U);
}

TEST(CommChannelTest, ReceiveForTimesOutAndCloseWakesWaiter) {
    MpscChannel<int> channel(options(1));

    int value = 0;
    const auto timeout = channel.receive_for(value, 10ms);
    EXPECT_FALSE(timeout);
    EXPECT_EQ(timeout.error_code, CommErrorCode::Timeout);
    EXPECT_EQ(channel.stats().timeout_count, 1U);

    std::atomic<bool> waiter_done{false};
    std::thread waiter([&] {
        const auto result = channel.receive_for(value, 2s);
        EXPECT_FALSE(result);
        EXPECT_EQ(result.error_code, CommErrorCode::Closed);
        waiter_done.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(20ms);
    EXPECT_FALSE(waiter_done.load(std::memory_order_acquire));
    channel.close();
    waiter.join();
    EXPECT_TRUE(waiter_done.load(std::memory_order_acquire));
}

TEST(CommChannelTest, SendForWaitsForCapacity) {
    MpscChannel<int> channel(options(1));
    EXPECT_TRUE(channel.try_send(1));

    std::thread consumer([&] {
        std::this_thread::sleep_for(20ms);
        int value = 0;
        EXPECT_TRUE(channel.try_receive(value));
        EXPECT_EQ(value, 1);
    });

    const auto result = channel.send_for(2, 1s);
    EXPECT_TRUE(result);
    consumer.join();

    int value = 0;
    EXPECT_TRUE(channel.try_receive(value));
    EXPECT_EQ(value, 2);
}

TEST(CommChannelTest, SendForTimesOutWhenRejectNewestChannelStaysFull) {
    MpscChannel<int> channel(options(1));
    EXPECT_TRUE(channel.try_send(1));

    const auto result = channel.send_for(2, 10ms);
    EXPECT_FALSE(result);
    EXPECT_EQ(result.error_code, CommErrorCode::Timeout);
    EXPECT_EQ(channel.stats().timeout_count, 1U);
}

TEST(CommChannelTest, SupportsNonTrivialAndMoveOnlyPayloads) {
    MpscChannel<std::string> strings(options(2));
    EXPECT_TRUE(strings.try_send(std::string{"sensor"}));

    std::string text;
    EXPECT_TRUE(strings.try_receive(text));
    EXPECT_EQ(text, "sensor");

    MpscChannel<std::unique_ptr<int>> pointers(options(1));
    EXPECT_TRUE(pointers.try_send(std::make_unique<int>(42)));

    std::unique_ptr<int> pointer;
    EXPECT_TRUE(pointers.try_receive(pointer));
    ASSERT_NE(pointer, nullptr);
    EXPECT_EQ(*pointer, 42);
}

TEST(CommChannelTest, SpscAliasUsesMpscChannelImplementation) {
    SpscChannel<int> channel(options(1));
    EXPECT_TRUE(channel.try_send(5));

    int value = 0;
    EXPECT_TRUE(channel.try_receive(value));
    EXPECT_EQ(value, 5);
}

} // namespace
