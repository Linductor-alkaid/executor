#pragma once

#include <executor/comm/fwd.hpp>
#include <executor/comm/types.hpp>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

namespace executor::comm {

class PhaseGate {
public:
    explicit PhaseGate(std::string name = {})
        : name_(std::move(name)) {}

    uint64_t current_phase() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return current_phase_;
    }

    CommResult advance_to(uint64_t phase) {
        std::optional<CommEvent> event;
        CommEventCallback callback;
        CommResult result;
        bool advanced = false;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) {
                record_closed_locked(event);
                callback = event_callback_;
                result = CommResult::failure(CommErrorCode::Closed,
                                             "phase gate is closed");
            } else if (phase <= current_phase_) {
                record_missed_phase_locked(event);
                callback = event_callback_;
                result = CommResult::failure(CommErrorCode::MissedPhase,
                                             "phase must advance monotonically");
            } else {
                current_phase_ = phase;
                if (enable_stats_) {
                    ++stats_.sent_count;
                    stats_.producer_lag = current_phase_;
                }
                result = CommResult::success();
                advanced = true;
            }
        }

        emit_comm_event_noexcept(callback, event);
        if (advanced) {
            cv_.notify_all();
        }
        return result;
    }

    CommResult advance() {
        std::optional<CommEvent> event;
        CommEventCallback callback;
        CommResult result;
        bool advanced = false;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) {
                record_closed_locked(event);
                callback = event_callback_;
                result = CommResult::failure(CommErrorCode::Closed,
                                             "phase gate is closed");
            } else {
                ++current_phase_;
                if (enable_stats_) {
                    ++stats_.sent_count;
                    stats_.producer_lag = current_phase_;
                }
                result = CommResult::success();
                advanced = true;
            }
        }

        emit_comm_event_noexcept(callback, event);
        if (advanced) {
            cv_.notify_all();
        }
        return result;
    }

    bool has_reached(uint64_t phase) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return current_phase_ >= phase;
    }

    template <class Rep, class Period>
    CommResult wait_for(uint64_t phase,
                        std::chrono::duration<Rep, Period> timeout) {
        return wait_until_impl(phase, timeout, false);
    }

    template <class Rep, class Period>
    CommResult wait_for_exact(uint64_t phase,
                              std::chrono::duration<Rep, Period> timeout) {
        return wait_until_impl(phase, timeout, true);
    }

    CommResult close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) {
                return CommResult::failure(CommErrorCode::Closed,
                                           "phase gate is already closed");
            }
            closed_ = true;
        }
        cv_.notify_all();
        return CommResult::success();
    }

    bool is_closed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_;
    }

    CommStats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        CommStats snapshot = stats_;
        snapshot.producer_lag = current_phase_;
        snapshot.consumer_lag = waiter_count_;
        return snapshot;
    }

    void set_event_callback(CommEventCallback callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        event_callback_ = std::move(callback);
    }

private:
    template <class Rep, class Period>
    CommResult wait_until_impl(uint64_t phase,
                               std::chrono::duration<Rep, Period> timeout,
                               bool exact) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        const auto wait_started_at = std::chrono::steady_clock::now();
        std::optional<CommEvent> event;
        CommEventCallback callback;
        CommResult result;

        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (exact && current_phase_ > phase) {
                record_missed_phase_locked(event);
                callback = event_callback_;
                result = CommResult::failure(CommErrorCode::MissedPhase,
                                             "requested phase was already missed");
            } else if (current_phase_ >= phase) {
                record_receive_locked(wait_started_at);
                result = CommResult::success();
            } else if (closed_) {
                result = CommResult::failure(CommErrorCode::Closed,
                                             "phase gate is closed");
            } else {
                ++waiter_count_;
                while (current_phase_ < phase && !closed_) {
                    if (cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
                        record_timeout_locked(event);
                        callback = event_callback_;
                        result = CommResult::failure(CommErrorCode::Timeout,
                                                     "phase wait timed out");
                        break;
                    }
                    if (exact && current_phase_ > phase) {
                        record_missed_phase_locked(event);
                        callback = event_callback_;
                        result = CommResult::failure(CommErrorCode::MissedPhase,
                                                     "requested phase was skipped");
                        break;
                    }
                }
                --waiter_count_;

                if (result.error_code == CommErrorCode::Ok) {
                    if (closed_ && current_phase_ < phase) {
                        result = CommResult::failure(CommErrorCode::Closed,
                                                     "phase gate is closed");
                    } else if (exact && current_phase_ > phase) {
                        record_missed_phase_locked(event);
                        callback = event_callback_;
                        result = CommResult::failure(CommErrorCode::MissedPhase,
                                                     "requested phase was skipped");
                    } else {
                        record_receive_locked(wait_started_at);
                        result = CommResult::success();
                    }
                }
            }
        }

        emit_comm_event_noexcept(callback, event);
        return result;
    }

    void record_receive_locked(std::chrono::steady_clock::time_point wait_started_at) {
        if (enable_stats_) {
            ++stats_.received_count;
            update_latency_stats(
                stats_,
                total_latency_,
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - wait_started_at));
        }
    }

    void record_timeout_locked(std::optional<CommEvent>& event) {
        if (enable_stats_) {
            ++stats_.timeout_count;
        }
        event = make_event_locked(CommEventKind::Timeout, "phase wait timed out");
    }

    void record_missed_phase_locked(std::optional<CommEvent>& event) {
        if (enable_stats_) {
            ++stats_.missed_phase_count;
        }
        event = make_event_locked(CommEventKind::MissedPhase, "phase was missed");
    }

    void record_closed_locked(std::optional<CommEvent>& event) {
        if (enable_stats_) {
            ++stats_.closed_send_count;
        }
        event = make_event_locked(CommEventKind::ClosedSend,
                                  "phase gate operation rejected after close");
    }

    std::optional<CommEvent> make_event_locked(CommEventKind kind, std::string message) const {
        if (!event_callback_) {
            return std::nullopt;
        }

        CommEvent event;
        event.kind = kind;
        event.component_name = name_;
        event.message = std::move(message);
        event.sequence = current_phase_;
        return event;
    }

    std::string name_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    uint64_t current_phase_ = 0;
    bool closed_ = false;
    bool enable_stats_ = true;
    uint64_t waiter_count_ = 0;
    CommStats stats_;
    std::chrono::nanoseconds total_latency_{0};
    CommEventCallback event_callback_;
};

class Sequencer {
public:
    explicit Sequencer(std::string name = {})
        : name_(std::move(name)) {}

    uint64_t next_ticket() {
        std::lock_guard<std::mutex> lock(mutex_);
        return ++next_ticket_;
    }

    CommResult publish(uint64_t ticket) {
        std::optional<CommEvent> event;
        CommEventCallback callback;
        CommResult result;
        bool published = false;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) {
                record_closed_locked(event);
                callback = event_callback_;
                result = CommResult::failure(CommErrorCode::Closed,
                                             "sequencer is closed");
            } else if (ticket == 0 || ticket <= published_ticket_) {
                record_missed_phase_locked(event);
                callback = event_callback_;
                result = CommResult::failure(CommErrorCode::MissedPhase,
                                             "ticket was already published or invalid");
            } else {
                published_ticket_ = ticket;
                if (ticket > next_ticket_) {
                    next_ticket_ = ticket;
                }
                if (enable_stats_) {
                    ++stats_.sent_count;
                    stats_.producer_lag = published_ticket_;
                }
                result = CommResult::success();
                published = true;
            }
        }

        emit_comm_event_noexcept(callback, event);
        if (published) {
            cv_.notify_all();
        }
        return result;
    }

    bool is_published(uint64_t ticket) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return ticket != 0 && published_ticket_ >= ticket;
    }

    template <class Rep, class Period>
    CommResult wait_until_published(uint64_t ticket,
                                    std::chrono::duration<Rep, Period> timeout) {
        if (ticket == 0) {
            return CommResult::failure(CommErrorCode::InvalidArgument,
                                       "ticket must be greater than zero");
        }

        const auto deadline = std::chrono::steady_clock::now() + timeout;
        const auto wait_started_at = std::chrono::steady_clock::now();
        std::optional<CommEvent> event;
        CommEventCallback callback;
        CommResult result;

        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (published_ticket_ == ticket) {
                record_receive_locked(wait_started_at);
                result = CommResult::success();
            } else if (published_ticket_ > ticket) {
                record_missed_phase_locked(event);
                callback = event_callback_;
                result = CommResult::failure(CommErrorCode::MissedPhase,
                                             "ticket was already missed");
            } else if (closed_) {
                result = CommResult::failure(CommErrorCode::Closed,
                                             "sequencer is closed");
            } else {
                ++waiter_count_;
                while (published_ticket_ < ticket && !closed_) {
                    if (cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
                        record_timeout_locked(event);
                        callback = event_callback_;
                        result = CommResult::failure(CommErrorCode::Timeout,
                                                     "sequencer wait timed out");
                        break;
                    }
                }
                --waiter_count_;

                if (result.error_code == CommErrorCode::Ok) {
                    if (closed_ && published_ticket_ < ticket) {
                        result = CommResult::failure(CommErrorCode::Closed,
                                                     "sequencer is closed");
                    } else if (published_ticket_ == ticket) {
                        record_receive_locked(wait_started_at);
                        result = CommResult::success();
                    } else {
                        record_missed_phase_locked(event);
                        callback = event_callback_;
                        result = CommResult::failure(CommErrorCode::MissedPhase,
                                                     "ticket was already missed");
                    }
                }
            }
        }

        emit_comm_event_noexcept(callback, event);
        return result;
    }

    CommResult close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) {
                return CommResult::failure(CommErrorCode::Closed,
                                           "sequencer is already closed");
            }
            closed_ = true;
        }
        cv_.notify_all();
        return CommResult::success();
    }

    bool is_closed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_;
    }

    uint64_t published_ticket() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return published_ticket_;
    }

    CommStats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        CommStats snapshot = stats_;
        snapshot.producer_lag = published_ticket_;
        snapshot.consumer_lag = waiter_count_;
        return snapshot;
    }

    void set_event_callback(CommEventCallback callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        event_callback_ = std::move(callback);
    }

private:
    void record_receive_locked(std::chrono::steady_clock::time_point wait_started_at) {
        if (enable_stats_) {
            ++stats_.received_count;
            update_latency_stats(
                stats_,
                total_latency_,
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - wait_started_at));
        }
    }

    void record_timeout_locked(std::optional<CommEvent>& event) {
        if (enable_stats_) {
            ++stats_.timeout_count;
        }
        event = make_event_locked(CommEventKind::Timeout, "sequencer wait timed out");
    }

    void record_missed_phase_locked(std::optional<CommEvent>& event) {
        if (enable_stats_) {
            ++stats_.missed_phase_count;
        }
        event = make_event_locked(CommEventKind::MissedPhase, "ticket was missed");
    }

    void record_closed_locked(std::optional<CommEvent>& event) {
        if (enable_stats_) {
            ++stats_.closed_send_count;
        }
        event = make_event_locked(CommEventKind::ClosedSend,
                                  "sequencer operation rejected after close");
    }

    std::optional<CommEvent> make_event_locked(CommEventKind kind, std::string message) const {
        if (!event_callback_) {
            return std::nullopt;
        }

        CommEvent event;
        event.kind = kind;
        event.component_name = name_;
        event.message = std::move(message);
        event.sequence = published_ticket_;
        return event;
    }

    std::string name_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    uint64_t next_ticket_ = 0;
    uint64_t published_ticket_ = 0;
    bool closed_ = false;
    bool enable_stats_ = true;
    uint64_t waiter_count_ = 0;
    CommStats stats_;
    std::chrono::nanoseconds total_latency_{0};
    CommEventCallback event_callback_;
};

} // namespace executor::comm
