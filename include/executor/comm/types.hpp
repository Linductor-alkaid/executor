#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>

namespace executor::comm {

/**
 * @brief Common error codes for communication facade control operations.
 */
enum class CommErrorCode {
    Ok,
    Closed,
    Full,
    Empty,
    Timeout,
    Stale,
    MissedPhase,
    InvalidArgument,
    NotReady,
    Unknown
};

inline const char* comm_error_code_to_string(CommErrorCode code) noexcept {
    switch (code) {
    case CommErrorCode::Ok:
        return "Ok";
    case CommErrorCode::Closed:
        return "Closed";
    case CommErrorCode::Full:
        return "Full";
    case CommErrorCode::Empty:
        return "Empty";
    case CommErrorCode::Timeout:
        return "Timeout";
    case CommErrorCode::Stale:
        return "Stale";
    case CommErrorCode::MissedPhase:
        return "MissedPhase";
    case CommErrorCode::InvalidArgument:
        return "InvalidArgument";
    case CommErrorCode::NotReady:
        return "NotReady";
    case CommErrorCode::Unknown:
        return "Unknown";
    default:
        return "Unknown";
    }
}

/**
 * @brief Lightweight result for communication facade operations with diagnostics.
 */
struct CommResult {
    bool ok = true;
    CommErrorCode error_code = CommErrorCode::Ok;
    std::string message;

    explicit operator bool() const noexcept {
        return ok;
    }

    static CommResult success(std::string msg = {}) {
        CommResult result;
        result.message = std::move(msg);
        return result;
    }

    static CommResult failure(CommErrorCode code, std::string msg = {}) {
        CommResult result;
        result.ok = false;
        result.error_code = code;
        result.message = std::move(msg);
        return result;
    }
};

enum class DropPolicy {
    RejectNewest,
    DropOldest,
    KeepLatest
};

inline const char* drop_policy_to_string(DropPolicy policy) noexcept {
    switch (policy) {
    case DropPolicy::RejectNewest:
        return "RejectNewest";
    case DropPolicy::DropOldest:
        return "DropOldest";
    case DropPolicy::KeepLatest:
        return "KeepLatest";
    default:
        return "Unknown";
    }
}

struct ChannelOptions {
    size_t capacity = 1024;
    DropPolicy drop_policy = DropPolicy::RejectNewest;
    bool enable_stats = true;
    std::string name;
};

struct RealtimeChannelOptions {
    size_t capacity = 1024;
    size_t max_items_per_cycle = 64;
    DropPolicy drop_policy = DropPolicy::RejectNewest;
    bool enable_stats = true;
    std::string name;
};

/**
 * @brief Local cumulative communication statistics.
 */
struct CommStats {
    uint64_t sent_count = 0;
    uint64_t received_count = 0;
    uint64_t dropped_count = 0;
    uint64_t overwritten_count = 0;
    uint64_t stale_read_count = 0;
    uint64_t closed_send_count = 0;
    uint64_t timeout_count = 0;
    uint64_t handler_exception_count = 0;
    uint64_t missed_phase_count = 0;
    uint64_t current_depth = 0;
    uint64_t peak_depth = 0;
    uint64_t capacity = 0;
    uint64_t producer_lag = 0;
    uint64_t consumer_lag = 0;
    std::chrono::nanoseconds max_latency{0};
    std::chrono::nanoseconds avg_latency{0};
};

enum class CommEventKind {
    Dropped,
    Overwritten,
    ClosedSend,
    Timeout,
    StaleRead,
    MissedPhase,
    ProducerLag,
    ConsumerLag,
    LatencyHigh,
    HandlerException
};

inline const char* comm_event_kind_to_string(CommEventKind kind) noexcept {
    switch (kind) {
    case CommEventKind::Dropped:
        return "Dropped";
    case CommEventKind::Overwritten:
        return "Overwritten";
    case CommEventKind::ClosedSend:
        return "ClosedSend";
    case CommEventKind::Timeout:
        return "Timeout";
    case CommEventKind::StaleRead:
        return "StaleRead";
    case CommEventKind::MissedPhase:
        return "MissedPhase";
    case CommEventKind::ProducerLag:
        return "ProducerLag";
    case CommEventKind::ConsumerLag:
        return "ConsumerLag";
    case CommEventKind::LatencyHigh:
        return "LatencyHigh";
    case CommEventKind::HandlerException:
        return "HandlerException";
    default:
        return "Unknown";
    }
}

struct CommEvent {
    CommEventKind kind = CommEventKind::Dropped;
    std::string component_name;
    std::string message;
    uint64_t sequence = 0;
    std::chrono::steady_clock::time_point timestamp =
        std::chrono::steady_clock::now();
};

using CommEventCallback = std::function<void(const CommEvent&)>;

} // namespace executor::comm
