#pragma once

#include <executor/comm/fwd.hpp>
#include <executor/comm/types.hpp>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

namespace executor::comm {

template <class T>
class MpscChannel {
public:
    explicit MpscChannel(ChannelOptions options = {})
        : options_(normalize_options(std::move(options))) {}

    bool try_send(const T& value) {
        return try_send_impl(value);
    }

    bool try_send(T&& value) {
        return try_send_impl(std::move(value));
    }

    template <class Rep, class Period>
    CommResult send_for(T value, std::chrono::duration<Rep, Period> timeout) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        std::optional<CommEvent> event;
        CommEventCallback callback;
        bool notify_not_empty = false;
        CommResult result;
        bool done = false;

        {
            std::unique_lock<std::mutex> lock(mutex_);

            if (closed_) {
                record_closed_send_locked(event);
                callback = event_callback_;
                result = CommResult::failure(CommErrorCode::Closed, "channel is closed");
            } else {
                while (!closed_ && queue_.size() >= options_.capacity &&
                       options_.drop_policy == DropPolicy::RejectNewest) {
                    if (not_full_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
                        record_timeout_locked(event);
                        callback = event_callback_;
                        result = CommResult::failure(CommErrorCode::Timeout, "channel send timed out");
                        done = true;
                        break;
                    }
                }

                if (done) {
                    // Result already set by timeout.
                } else if (closed_) {
                    record_closed_send_locked(event);
                    callback = event_callback_;
                    result = CommResult::failure(CommErrorCode::Closed, "channel is closed");
                } else if (!enqueue_locked(std::move(value), event)) {
                    callback = event_callback_;
                    result = CommResult::failure(CommErrorCode::Full, "channel is full");
                } else {
                    callback = event_callback_;
                    notify_not_empty = true;
                    result = CommResult::success();
                }
            }
        }

        emit_comm_event_noexcept(callback, event);
        if (notify_not_empty) {
            not_empty_cv_.notify_one();
        }
        return result;
    }

    bool try_receive(T& out) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }

        out = std::move(queue_.front().value);
        const auto enqueued_at = queue_.front().enqueued_at;
        queue_.pop_front();
        record_receive_locked(enqueued_at);
        lock.unlock();
        not_full_cv_.notify_one();
        return true;
    }

    template <class Rep, class Period>
    CommResult receive_for(T& out, std::chrono::duration<Rep, Period> timeout) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        std::optional<CommEvent> event;
        CommEventCallback callback;
        CommResult result;
        bool done = false;

        {
            std::unique_lock<std::mutex> lock(mutex_);

            while (queue_.empty() && !closed_) {
                if (not_empty_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
                    record_timeout_locked(event);
                    callback = event_callback_;
                    result = CommResult::failure(CommErrorCode::Timeout, "channel receive timed out");
                    done = true;
                    break;
                }
            }

            if (done) {
                // Return after releasing the lock and emitting the timeout event.
            } else if (queue_.empty()) {
                return CommResult::failure(CommErrorCode::Closed, "channel is closed");
            } else {
                out = std::move(queue_.front().value);
                const auto enqueued_at = queue_.front().enqueued_at;
                queue_.pop_front();
                record_receive_locked(enqueued_at);
                lock.unlock();
                not_full_cv_.notify_one();
                return CommResult::success();
            }
        }

        emit_comm_event_noexcept(callback, event);
        return result;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        not_empty_cv_.notify_all();
        not_full_cv_.notify_all();
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
    static ChannelOptions normalize_options(ChannelOptions options) {
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
            std::unique_lock<std::mutex> lock(mutex_);
            sent = enqueue_locked(std::forward<U>(value), event);
            callback = event_callback_;
        }

        emit_comm_event_noexcept(callback, event);
        if (sent) {
            not_empty_cv_.notify_one();
        }
        return sent;
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

        queue_.push_back(QueuedItem{std::forward<U>(value), std::chrono::steady_clock::now()});
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

    void record_receive_locked(std::chrono::steady_clock::time_point enqueued_at) {
        if (!options_.enable_stats) {
            return;
        }
        ++stats_.received_count;
        stats_.current_depth = queue_.size();
        update_latency_stats(
            stats_,
            total_latency_,
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - enqueued_at));
    }

    void record_drop_locked(std::optional<CommEvent>& event) {
        if (options_.enable_stats) {
            ++stats_.dropped_count;
            stats_.current_depth = queue_.size();
        }
        event = make_event_locked(CommEventKind::Dropped, "channel message dropped");
    }

    void record_overwrite_locked(std::optional<CommEvent>& event) {
        if (options_.enable_stats) {
            ++stats_.overwritten_count;
            stats_.current_depth = queue_.size();
        }
        event = make_event_locked(CommEventKind::Overwritten, "channel message overwritten");
    }

    void record_closed_send_locked(std::optional<CommEvent>& event) {
        if (options_.enable_stats) {
            ++stats_.closed_send_count;
        }
        event = make_event_locked(CommEventKind::ClosedSend, "send rejected after channel close");
    }

    void record_timeout_locked(std::optional<CommEvent>& event) {
        if (options_.enable_stats) {
            ++stats_.timeout_count;
        }
        event = make_event_locked(CommEventKind::Timeout, "channel operation timed out");
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

    ChannelOptions options_;
    struct QueuedItem {
        T value;
        std::chrono::steady_clock::time_point enqueued_at;
    };
    mutable std::mutex mutex_;
    std::condition_variable not_empty_cv_;
    std::condition_variable not_full_cv_;
    std::deque<QueuedItem> queue_;
    bool closed_ = false;
    CommStats stats_;
    std::chrono::nanoseconds total_latency_{0};
    CommEventCallback event_callback_;
};

} // namespace executor::comm
