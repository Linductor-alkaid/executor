#include <executor/comm/realtime_memory.hpp>

#include <atomic>
#include <cstdlib>
#include <new>

namespace {

struct GuardState {
    bool active = false;
    executor::comm::RealtimeAllocationStats stats;
};

thread_local GuardState guard_state;

} // namespace

namespace executor::comm {

RealtimeAllocationGuard::RealtimeAllocationGuard(std::string_view component,
                                                 std::string_view phase) noexcept {
#ifdef EXECUTOR_ENABLE_REALTIME_ALLOCATION_GUARD
    guard_state.active = true;
    guard_state.stats.component = component;
    guard_state.stats.phase = phase;
    active_ = true;
#else
    (void)component;
    (void)phase;
#endif
}

RealtimeAllocationGuard::~RealtimeAllocationGuard() noexcept {
    if (active_) {
        guard_state.active = false;
    }
}

bool RealtimeAllocationGuard::is_enabled() noexcept {
#ifdef EXECUTOR_ENABLE_REALTIME_ALLOCATION_GUARD
    return true;
#else
    return false;
#endif
}

RealtimeAllocationStats RealtimeAllocationGuard::current_thread_stats() noexcept {
    return guard_state.stats;
}

void RealtimeAllocationGuard::reset_current_thread_stats() noexcept {
    guard_state.stats = {};
}

} // namespace executor::comm

#ifdef EXECUTOR_ENABLE_REALTIME_ALLOCATION_GUARD
void* operator new(std::size_t size) {
    if (guard_state.active) {
        ++guard_state.stats.allocation_count;
        guard_state.stats.allocated_bytes += size;
    }
    if (void* pointer = std::malloc(size)) {
        return pointer;
    }
    throw std::bad_alloc();
}

void* operator new[](std::size_t size) {
    return ::operator new(size);
}

void operator delete(void* pointer) noexcept {
    std::free(pointer);
}

void operator delete[](void* pointer) noexcept {
    std::free(pointer);
}

void operator delete(void* pointer, std::size_t) noexcept {
    std::free(pointer);
}

void operator delete[](void* pointer, std::size_t) noexcept {
    std::free(pointer);
}
#endif
