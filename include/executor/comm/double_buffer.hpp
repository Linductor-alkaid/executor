#pragma once

#include <executor/comm/fwd.hpp>
#include <executor/comm/types.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

namespace executor::comm {

template <class T>
struct Snapshot {
    T value;
    uint64_t sequence = 0;
    std::chrono::steady_clock::time_point timestamp =
        std::chrono::steady_clock::now();
};

template <class T>
class DoubleBuffer {
public:
    explicit DoubleBuffer(T initial = {}, std::string name = {})
        : name_(std::move(name)) {
        buffers_[0] = std::move(initial);
        buffers_[1] = buffers_[0];
        timestamps_[0] = std::chrono::steady_clock::now();
        timestamps_[1] = timestamps_[0];
        stats_.current_depth = 1;
        stats_.peak_depth = 1;
        stats_.capacity = 2;
    }

    uint64_t publish(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t write_index = inactive_index_locked();
        buffers_[write_index] = std::move(value);
        timestamps_[write_index] = std::chrono::steady_clock::now();
        active_index_ = write_index;
        ++sequence_;
        record_publish_locked();
        return sequence_;
    }

    template <class Fn>
    uint64_t update(Fn&& writer) {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t write_index = inactive_index_locked();
        buffers_[write_index] = buffers_[active_index_];
        writer(buffers_[write_index]);
        timestamps_[write_index] = std::chrono::steady_clock::now();
        active_index_ = write_index;
        ++sequence_;
        record_publish_locked();
        return sequence_;
    }

    Snapshot<T> load() const {
        std::lock_guard<std::mutex> lock(mutex_);
        record_load_locked();
        return make_snapshot_locked();
    }

    bool load_newer_than(uint64_t last_seen_sequence, Snapshot<T>& out) const {
        std::optional<CommEvent> event;
        CommEventCallback callback;
        bool loaded = false;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (sequence_ <= last_seen_sequence) {
                record_stale_read_locked(event);
                callback = event_callback_;
            } else {
                record_load_locked();
                out = make_snapshot_locked();
                loaded = true;
            }
        }

        emit_comm_event_noexcept(callback, event);
        return loaded;
    }

    uint64_t sequence() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return sequence_;
    }

    CommStats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        CommStats snapshot = stats_;
        snapshot.current_depth = 1;
        snapshot.peak_depth = snapshot.peak_depth == 0 ? 1 : snapshot.peak_depth;
        snapshot.capacity = 2;
        snapshot.producer_lag = sequence_;
        return snapshot;
    }

    void set_event_callback(CommEventCallback callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        event_callback_ = std::move(callback);
    }

private:
    size_t inactive_index_locked() const {
        return active_index_ == 0 ? 1U : 0U;
    }

    Snapshot<T> make_snapshot_locked() const {
        Snapshot<T> snapshot;
        snapshot.value = buffers_[active_index_];
        snapshot.sequence = sequence_;
        snapshot.timestamp = timestamps_[active_index_];
        return snapshot;
    }

    void record_publish_locked() {
        ++stats_.sent_count;
        stats_.current_depth = 1;
        stats_.peak_depth = 1;
        stats_.capacity = 2;
        stats_.producer_lag = sequence_;
    }

    void record_load_locked() const {
        ++stats_.received_count;
        stats_.consumer_lag = sequence_;
        update_latency_stats(
            stats_,
            total_latency_,
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - timestamps_[active_index_]));
    }

    void record_stale_read_locked(std::optional<CommEvent>& event) const {
        ++stats_.stale_read_count;
        event = make_event_locked(CommEventKind::StaleRead, "double buffer has no newer snapshot");
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
    T buffers_[2]{};
    std::chrono::steady_clock::time_point timestamps_[2]{};
    size_t active_index_ = 0;
    uint64_t sequence_ = 0;
    mutable CommStats stats_;
    mutable std::chrono::nanoseconds total_latency_{0};
    CommEventCallback event_callback_;
};

} // namespace executor::comm
