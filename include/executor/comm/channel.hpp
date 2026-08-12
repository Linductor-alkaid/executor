#pragma once

#include <executor/comm/bounded_queue.hpp>
#include <executor/comm/fwd.hpp>
#include <executor/comm/types.hpp>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <utility>

#if defined(__has_feature)
#  if __has_feature(thread_sanitizer)
#    define EXECUTOR_COMM_CHANNEL_HAS_TSAN 1
#  endif
#endif
#if defined(__SANITIZE_THREAD__)
#  define EXECUTOR_COMM_CHANNEL_HAS_TSAN 1
#endif
#ifndef EXECUTOR_COMM_CHANNEL_HAS_TSAN
#  define EXECUTOR_COMM_CHANNEL_HAS_TSAN 0
#endif

namespace executor::comm {

template <class T>
class MpscChannel {
#if EXECUTOR_COMM_CHANNEL_HAS_TSAN
    // GCC 11 TSAN does not model pthread_cond_clockwait mutex handoff.
    // Its system-clock overload uses the intercepted pthread_cond_timedwait.
    using WaitClock = std::chrono::system_clock;
#else
    using WaitClock = std::chrono::steady_clock;
#endif

public:
    explicit MpscChannel(ChannelOptions options = {})
        : options_(normalize_options(std::move(options))),
          queue_(options_.capacity, options_.drop_policy, options_.enable_stats,
                 options_.name, "channel") {}

    bool try_send(const T& value) { return try_send_impl(value); }
    bool try_send(T&& value) { return try_send_impl(std::move(value)); }

    template <class Rep, class Period>
    CommResult send_for(T value, std::chrono::duration<Rep, Period> timeout) {
        const auto deadline = WaitClock::now() + timeout;
        std::optional<CommEvent> event;
        CommEventCallback callback;
        bool notify_not_empty = false;
        CommResult result;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            while (!queue_.is_closed() && queue_.size() >= queue_.capacity() &&
                   options_.drop_policy == DropPolicy::RejectNewest) {
                if (not_full_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
                    queue_.record_timeout(event);
                    callback = queue_.callback();
                    result = CommResult::failure(CommErrorCode::Timeout, "channel send timed out");
                    lock.unlock();
                    emit_comm_event_noexcept(callback, event);
                    return result;
                }
            }
            if (queue_.is_closed()) {
                queue_.enqueue(std::move(value), event);
                result = CommResult::failure(CommErrorCode::Closed, "channel is closed");
            } else if (!queue_.enqueue(std::move(value), event)) {
                result = CommResult::failure(CommErrorCode::Full, "channel is full");
            } else {
                notify_not_empty = true;
                result = CommResult::success();
            }
            callback = queue_.callback();
        }
        emit_comm_event_noexcept(callback, event);
        if (notify_not_empty) not_empty_cv_.notify_one();
        return result;
    }

    bool try_receive(T& out) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto item = queue_.try_pop();
        if (!item) return false;
        out = std::move(item->value);
        lock.unlock();
        not_full_cv_.notify_one();
        return true;
    }

    template <class Rep, class Period>
    CommResult receive_for(T& out, std::chrono::duration<Rep, Period> timeout) {
        const auto deadline = WaitClock::now() + timeout;
        std::optional<CommEvent> event;
        CommEventCallback callback;
        std::unique_lock<std::mutex> lock(mutex_);
        while (queue_.empty() && !queue_.is_closed()) {
            if (not_empty_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
                queue_.record_timeout(event);
                callback = queue_.callback();
                lock.unlock();
                emit_comm_event_noexcept(callback, event);
                return CommResult::failure(CommErrorCode::Timeout, "channel receive timed out");
            }
        }
        auto item = queue_.try_pop();
        if (!item) return CommResult::failure(CommErrorCode::Closed, "channel is closed");
        out = std::move(item->value);
        lock.unlock();
        not_full_cv_.notify_one();
        return CommResult::success();
    }

    void close() {
        { std::lock_guard<std::mutex> lock(mutex_); queue_.close(); }
        not_empty_cv_.notify_all();
        not_full_cv_.notify_all();
    }
    bool is_closed() const { std::lock_guard<std::mutex> lock(mutex_); return queue_.is_closed(); }
    bool empty() const { std::lock_guard<std::mutex> lock(mutex_); return queue_.empty(); }
    size_t size_approx() const { std::lock_guard<std::mutex> lock(mutex_); return queue_.size(); }
    size_t capacity() const { return queue_.capacity(); }
    CommStats stats() const { std::lock_guard<std::mutex> lock(mutex_); return queue_.stats(); }
    void set_event_callback(CommEventCallback callback) {
        std::lock_guard<std::mutex> lock(mutex_); queue_.set_event_callback(std::move(callback));
    }

private:
    static ChannelOptions normalize_options(ChannelOptions options) {
        if (options.capacity == 0) options.capacity = 1;
        return options;
    }
    template <class U>
    bool try_send_impl(U&& value) {
        std::optional<CommEvent> event;
        CommEventCallback callback;
        bool sent;
        { std::lock_guard<std::mutex> lock(mutex_);
          sent = queue_.enqueue(std::forward<U>(value), event); callback = queue_.callback(); }
        emit_comm_event_noexcept(callback, event);
        if (sent) not_empty_cv_.notify_one();
        return sent;
    }

    ChannelOptions options_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_cv_;
    std::condition_variable not_full_cv_;
    BoundedQueue<T> queue_;
};

} // namespace executor::comm

#undef EXECUTOR_COMM_CHANNEL_HAS_TSAN
