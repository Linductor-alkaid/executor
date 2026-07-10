#pragma once

#include <executor/comm/fwd.hpp>
#include <executor/comm/types.hpp>

#include <algorithm>
#include <cstddef>
#include <deque>
#include <exception>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

namespace executor::comm {

template <class T>
class LatestMailbox {
public:
    explicit LatestMailbox(std::string name = {})
        : name_(std::move(name)) {}

    void publish(const T& value) {
        publish_impl(value);
    }

    void publish(T&& value) {
        publish_impl(std::move(value));
    }

    bool try_load(T& out) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!value_) {
            return false;
        }

        out = *value_;
        if (enable_stats_) {
            ++stats_.received_count;
            stats_.consumer_lag = sequence_;
        }
        return true;
    }

    bool try_load_newer_than(uint64_t last_seen_sequence,
                             T& out,
                             uint64_t& new_sequence) const {
        std::optional<CommEvent> event;
        CommEventCallback callback;
        bool loaded = false;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!value_ || sequence_ <= last_seen_sequence) {
                record_stale_read_locked(event);
                callback = event_callback_;
            } else {
                out = *value_;
                new_sequence = sequence_;
                if (enable_stats_) {
                    ++stats_.received_count;
                    stats_.consumer_lag = sequence_ - last_seen_sequence;
                }
                loaded = true;
            }
        }

        if (event && callback) {
            callback(*event);
        }
        return loaded;
    }

    uint64_t sequence() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return sequence_;
    }

    CommStats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        CommStats snapshot = stats_;
        snapshot.current_depth = value_ ? 1U : 0U;
        snapshot.peak_depth = value_ ? std::max<uint64_t>(snapshot.peak_depth, 1U)
                                     : snapshot.peak_depth;
        snapshot.capacity = 1;
        if (enable_stats_) {
            snapshot.producer_lag = sequence_;
        }
        return snapshot;
    }

    void set_event_callback(CommEventCallback callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        event_callback_ = std::move(callback);
    }

private:
    template <class U>
    void publish_impl(U&& value) {
        std::optional<CommEvent> event;
        CommEventCallback callback;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (value_) {
                record_overwrite_locked(event);
            }
            value_.emplace(std::forward<U>(value));
            ++sequence_;
            if (enable_stats_) {
                ++stats_.sent_count;
                stats_.current_depth = 1;
                stats_.peak_depth = 1;
                stats_.producer_lag = sequence_;
            }
            callback = event_callback_;
        }

        if (event && callback) {
            callback(*event);
        }
    }

    void record_overwrite_locked(std::optional<CommEvent>& event) {
        if (enable_stats_) {
            ++stats_.overwritten_count;
        }
        event = make_event_locked(CommEventKind::Overwritten, "mailbox value overwritten");
    }

    void record_stale_read_locked(std::optional<CommEvent>& event) const {
        if (enable_stats_) {
            ++stats_.stale_read_count;
        }
        event = make_event_locked(CommEventKind::StaleRead, "mailbox has no newer value");
    }

    std::optional<CommEvent> make_event_locked(CommEventKind kind, std::string message) const {
        if (!event_callback_) {
            return std::nullopt;
        }

        CommEvent event;
        event.kind = kind;
        event.component_name = name_;
        event.message = std::move(message);
        event.sequence = sequence_;
        return event;
    }

    std::string name_;
    mutable std::mutex mutex_;
    std::optional<T> value_;
    uint64_t sequence_ = 0;
    bool enable_stats_ = true;
    mutable CommStats stats_;
    CommEventCallback event_callback_;
};

template <class T>
class RealtimeChannel {
public:
    explicit RealtimeChannel(RealtimeChannelOptions options = {})
        : options_(normalize_options(std::move(options))) {}

    bool try_send(const T& value) {
        return try_send_impl(value);
    }

    bool try_send(T&& value) {
        return try_send_impl(std::move(value));
    }

    template <class Fn>
    size_t drain_for_cycle(Fn&& handler, size_t max_items = 0) {
        const size_t budget = max_items == 0 ? options_.max_items_per_cycle : max_items;
        const bool unlimited = budget == 0;

        size_t drained = 0;
        while (unlimited || drained < budget) {
            std::optional<T> item;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (queue_.empty()) {
                    break;
                }

                item.emplace(std::move(queue_.front()));
                queue_.pop_front();
                record_receive_locked();
            }

            try {
                invoke_handler(handler, *item);
            } catch (...) {
                record_handler_exception_event();
                throw;
            }
            ++drained;
        }

        return drained;
    }

    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
    }

    bool is_closed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    size_t size_approx() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    size_t capacity() const {
        return options_.capacity;
    }

    size_t max_items_per_cycle() const {
        return options_.max_items_per_cycle;
    }

    CommStats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        CommStats snapshot = stats_;
        snapshot.current_depth = queue_.size();
        snapshot.capacity = options_.capacity;
        if (options_.enable_stats) {
            snapshot.consumer_lag = queue_.size();
            snapshot.producer_lag =
                (stats_.sent_count >= stats_.received_count)
                    ? (stats_.sent_count - stats_.received_count)
                    : 0;
        }
        return snapshot;
    }

    void set_event_callback(CommEventCallback callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        event_callback_ = std::move(callback);
    }

private:
    static RealtimeChannelOptions normalize_options(RealtimeChannelOptions options) {
        if (options.capacity == 0) {
            options.capacity = 1;
        }
        return options;
    }

    template <class U>
    bool try_send_impl(U&& value) {
        std::optional<CommEvent> event;
        CommEventCallback callback;
        bool sent = false;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            sent = enqueue_locked(std::forward<U>(value), event);
            callback = event_callback_;
        }

        if (event && callback) {
            callback(*event);
        }
        return sent;
    }

    template <class Fn>
    static void invoke_handler(Fn& handler, T& item) {
        if constexpr (requires { handler(item); }) {
            handler(item);
        } else {
            handler(std::move(item));
        }
    }

    template <class U>
    bool enqueue_locked(U&& value, std::optional<CommEvent>& event) {
        if (closed_) {
            record_closed_send_locked(event);
            return false;
        }

        if (queue_.size() >= options_.capacity) {
            if (options_.drop_policy == DropPolicy::DropOldest) {
                queue_.pop_front();
                record_drop_locked(event);
            } else if (options_.drop_policy == DropPolicy::KeepLatest) {
                queue_.clear();
                record_overwrite_locked(event);
            } else {
                record_drop_locked(event);
                return false;
            }
        }

        queue_.push_back(std::forward<U>(value));
        record_send_locked();
        return true;
    }

    void record_send_locked() {
        if (!options_.enable_stats) {
            return;
        }
        ++stats_.sent_count;
        stats_.current_depth = queue_.size();
        if (stats_.current_depth > stats_.peak_depth) {
            stats_.peak_depth = stats_.current_depth;
        }
    }

    void record_receive_locked() {
        if (!options_.enable_stats) {
            return;
        }
        ++stats_.received_count;
        stats_.current_depth = queue_.size();
    }

    void record_drop_locked(std::optional<CommEvent>& event) {
        if (options_.enable_stats) {
            ++stats_.dropped_count;
            stats_.current_depth = queue_.size();
        }
        event = make_event_locked(CommEventKind::Dropped, "realtime channel message dropped");
    }

    void record_overwrite_locked(std::optional<CommEvent>& event) {
        if (options_.enable_stats) {
            ++stats_.overwritten_count;
            stats_.current_depth = queue_.size();
        }
        event = make_event_locked(CommEventKind::Overwritten,
                                  "realtime channel messages overwritten");
    }

    void record_closed_send_locked(std::optional<CommEvent>& event) {
        if (options_.enable_stats) {
            ++stats_.closed_send_count;
        }
        event = make_event_locked(CommEventKind::ClosedSend,
                                  "send rejected after realtime channel close");
    }

    void record_handler_exception_event() {
        std::optional<CommEvent> event;
        CommEventCallback callback;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (options_.enable_stats) {
                ++stats_.handler_exception_count;
            }
            event = make_event_locked(CommEventKind::HandlerException,
                                      "realtime channel handler threw");
            callback = event_callback_;
        }

        if (event && callback) {
            callback(*event);
        }
    }

    std::optional<CommEvent> make_event_locked(CommEventKind kind, std::string message) const {
        if (!event_callback_) {
            return std::nullopt;
        }

        CommEvent event;
        event.kind = kind;
        event.component_name = options_.name;
        event.message = std::move(message);
        event.sequence = stats_.sent_count;
        return event;
    }

    RealtimeChannelOptions options_;
    mutable std::mutex mutex_;
    std::deque<T> queue_;
    bool closed_ = false;
    CommStats stats_;
    CommEventCallback event_callback_;
};

} // namespace executor::comm
