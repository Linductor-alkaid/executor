#pragma once

#include <executor/comm/types.hpp>

#include <chrono>
#include <cstddef>
#include <deque>
#include <optional>
#include <string>
#include <utility>

namespace executor::comm {

template <class T>
class BoundedQueue {
public:
    struct Item {
        T value;
        std::chrono::steady_clock::time_point enqueued_at;
    };

    BoundedQueue(size_t capacity, DropPolicy drop_policy, bool enable_stats,
                 std::string component_name, std::string event_prefix)
        : capacity_(capacity == 0 ? 1 : capacity), drop_policy_(drop_policy),
          enable_stats_(enable_stats), component_name_(std::move(component_name)),
          event_prefix_(std::move(event_prefix)) {}

    template <class U>
    bool enqueue(U&& value, std::optional<CommEvent>& event) {
        if (closed_) {
            record_closed_send(event);
            return false;
        }
        if (queue_.size() >= capacity_) {
            if (drop_policy_ == DropPolicy::DropOldest) {
                queue_.pop_front();
                record_drop(event);
            } else if (drop_policy_ == DropPolicy::KeepLatest) {
                queue_.clear();
                record_overwrite(event);
            } else {
                record_drop(event);
                return false;
            }
        }
        queue_.push_back(Item{std::forward<U>(value), std::chrono::steady_clock::now()});
        record_send();
        return true;
    }

    std::optional<Item> try_pop() {
        if (queue_.empty()) {
            return std::nullopt;
        }
        Item item = std::move(queue_.front());
        queue_.pop_front();
        record_receive(item.enqueued_at);
        return item;
    }

    void close() { closed_ = true; }
    bool is_closed() const { return closed_; }
    bool empty() const { return queue_.empty(); }
    size_t size() const { return queue_.size(); }
    size_t capacity() const { return capacity_; }

    CommStats stats() const {
        CommStats snapshot = stats_;
        snapshot.current_depth = queue_.size();
        snapshot.capacity = capacity_;
        if (enable_stats_) {
            snapshot.consumer_lag = queue_.size();
            snapshot.producer_lag = stats_.sent_count >= stats_.received_count
                                        ? stats_.sent_count - stats_.received_count
                                        : 0;
        }
        return snapshot;
    }

    void record_timeout(std::optional<CommEvent>& event) {
        if (enable_stats_) ++stats_.timeout_count;
        event = make_event(CommEventKind::Timeout, event_prefix_ + " operation timed out");
    }

    void record_handler_exception(std::optional<CommEvent>& event) {
        if (enable_stats_) ++stats_.handler_exception_count;
        event = make_event(CommEventKind::HandlerException,
                           event_prefix_ + " handler threw");
    }

    CommEventCallback callback() const { return event_callback_; }
    void set_event_callback(CommEventCallback callback) { event_callback_ = std::move(callback); }

private:
    void record_send() {
        if (!enable_stats_) return;
        ++stats_.sent_count;
        stats_.current_depth = queue_.size();
        if (stats_.current_depth > stats_.peak_depth) stats_.peak_depth = stats_.current_depth;
    }

    void record_receive(std::chrono::steady_clock::time_point enqueued_at) {
        if (!enable_stats_) return;
        ++stats_.received_count;
        stats_.current_depth = queue_.size();
        update_latency_stats(stats_, total_latency_,
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - enqueued_at));
    }

    void record_drop(std::optional<CommEvent>& event) {
        if (enable_stats_) { ++stats_.dropped_count; stats_.current_depth = queue_.size(); }
        event = make_event(CommEventKind::Dropped, event_prefix_ + " message dropped");
    }
    void record_overwrite(std::optional<CommEvent>& event) {
        if (enable_stats_) { ++stats_.overwritten_count; stats_.current_depth = queue_.size(); }
        event = make_event(CommEventKind::Overwritten, event_prefix_ + " messages overwritten");
    }
    void record_closed_send(std::optional<CommEvent>& event) {
        if (enable_stats_) ++stats_.closed_send_count;
        event = make_event(CommEventKind::ClosedSend, event_prefix_ + " send rejected after close");
    }
    std::optional<CommEvent> make_event(CommEventKind kind, std::string message) const {
        if (!event_callback_) return std::nullopt;
        return CommEvent{kind, component_name_, std::move(message), stats_.sent_count};
    }

    size_t capacity_;
    DropPolicy drop_policy_;
    bool enable_stats_;
    std::string component_name_;
    std::string event_prefix_;
    std::deque<Item> queue_;
    bool closed_ = false;
    CommStats stats_;
    std::chrono::nanoseconds total_latency_{0};
    CommEventCallback event_callback_;
};

} // namespace executor::comm
