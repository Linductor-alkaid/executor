#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace executor::comm {

struct RealtimeAllocationStats {
    uint64_t allocation_count = 0;
    uint64_t allocated_bytes = 0;
    std::string_view component;
    std::string_view phase;
};

class RealtimeAllocationGuard {
public:
    RealtimeAllocationGuard(std::string_view component, std::string_view phase) noexcept;
    ~RealtimeAllocationGuard() noexcept;

    RealtimeAllocationGuard(const RealtimeAllocationGuard&) = delete;
    RealtimeAllocationGuard& operator=(const RealtimeAllocationGuard&) = delete;

    static bool is_enabled() noexcept;
    static RealtimeAllocationStats current_thread_stats() noexcept;
    static void reset_current_thread_stats() noexcept;

private:
    bool active_ = false;
};

} // namespace executor::comm
