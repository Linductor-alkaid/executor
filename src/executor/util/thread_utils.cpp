#include "thread_utils.hpp"

#include <cassert>
#include <mutex>

#ifdef _WIN32
    #include <windows.h>
#elif defined(__linux__)
    #include <pthread.h>
    #include <sched.h>
    #include <unistd.h>
    #include <sys/syscall.h>
    #include <sys/resource.h>
    #include <sys/mman.h>
    #include <sys/prctl.h>
    #include <errno.h>
#else
    #error "Unsupported platform"
#endif

namespace executor {
namespace util {

#ifdef _WIN32

bool set_thread_priority(std::thread::native_handle_type handle, int priority) {
    // Windows优先级映射
    // priority范围：-2到2，对应THREAD_PRIORITY_IDLE到THREAD_PRIORITY_TIME_CRITICAL
    int win_priority;
    
    if (priority <= -15) {
        win_priority = THREAD_PRIORITY_IDLE;
    } else if (priority <= -10) {
        win_priority = THREAD_PRIORITY_LOWEST;
    } else if (priority <= -5) {
        win_priority = THREAD_PRIORITY_BELOW_NORMAL;
    } else if (priority <= 0) {
        win_priority = THREAD_PRIORITY_NORMAL;
    } else if (priority <= 5) {
        win_priority = THREAD_PRIORITY_ABOVE_NORMAL;
    } else if (priority <= 10) {
        win_priority = THREAD_PRIORITY_HIGHEST;
    } else {
        win_priority = THREAD_PRIORITY_TIME_CRITICAL;
    }
    
    return SetThreadPriority(handle, win_priority) != 0;
}

bool set_cpu_affinity(std::thread::native_handle_type handle,
                      const std::vector<int>& cpu_ids) {
    if (cpu_ids.empty()) {
        return false;
    }
    
    DWORD_PTR mask = 0;
    for (int cpu_id : cpu_ids) {
        if (cpu_id < 0 || cpu_id >= 64) {  // Windows最多支持64个CPU
            return false;
        }
        mask |= (static_cast<DWORD_PTR>(1) << cpu_id);
    }
    
    return SetThreadAffinityMask(handle, mask) != 0;
}

int get_current_thread_priority() {
    HANDLE handle = GetCurrentThread();
    int win_priority = GetThreadPriority(handle);
    
    // 反向映射到通用优先级值
    switch (win_priority) {
        case THREAD_PRIORITY_IDLE:
            return -15;
        case THREAD_PRIORITY_LOWEST:
            return -10;
        case THREAD_PRIORITY_BELOW_NORMAL:
            return -5;
        case THREAD_PRIORITY_NORMAL:
            return 0;
        case THREAD_PRIORITY_ABOVE_NORMAL:
            return 5;
        case THREAD_PRIORITY_HIGHEST:
            return 10;
        case THREAD_PRIORITY_TIME_CRITICAL:
            return 15;
        default:
            return 0;
    }
}

std::vector<int> get_current_thread_affinity() {
    HANDLE handle = GetCurrentThread();
    DWORD_PTR process_mask, system_mask;
    
    if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask) == 0) {
        return {};
    }
    
    DWORD_PTR thread_mask = SetThreadAffinityMask(handle, process_mask);
    if (thread_mask == 0) {
        return {};
    }
    
    // 恢复原始亲和性
    SetThreadAffinityMask(handle, thread_mask);
    
    std::vector<int> cpu_ids;
    for (int i = 0; i < 64; ++i) {
        if (thread_mask & (static_cast<DWORD_PTR>(1) << i)) {
            cpu_ids.push_back(i);
        }
    }
    
    return cpu_ids;
}

void set_current_thread_name(const std::string& name) {
    // Windows 通过 SetThreadDescription 设置线程名（需要 Win10 1607+）
    // 将窄字符串转换为宽字符串
    std::wstring wname(name.begin(), name.end());
    SetThreadDescription(GetCurrentThread(), wname.c_str());
}

bool set_current_thread_timer_slack_ns(uint64_t /*slack_ns*/) {
    // Windows 不支持 per-thread timer slack，空实现
    return false;
}

#elif defined(__linux__)

bool set_thread_priority(std::thread::native_handle_type handle, int priority) {
    struct sched_param param;
    param.sched_priority = priority;

    // 如果优先级在1-99范围内，使用SCHED_FIFO实时调度策略
    // 否则使用SCHED_OTHER普通调度策略
    int policy = (priority >= 1 && priority <= 99) ? SCHED_FIFO : SCHED_OTHER;

    // 对于SCHED_OTHER，优先级必须为0
    if (policy == SCHED_OTHER) {
        param.sched_priority = 0;
        // 使用nice值设置优先级（-20到19）
        if (priority < -20) priority = -20;
        if (priority > 19) priority = 19;
    }

    int result = pthread_setschedparam(handle, policy, &param);
    if (result != 0) {
        return false;
    }

    // P-260618-007: previously the SCHED_OTHER branch silently dropped the
    // priority argument (the comment said "这里简化处理,只设置调度策略").
    // Callers (ThreadPool, RealtimeThreadExecutor) treated the resulting
    // "true" as "priority applied", but nice was never touched — a non-root
    // user setting thread_priority=10 got a false success and no effect.
    //
    // We now actually call setpriority(PRIO_PROCESS, ..., clamped_nice) so
    // the priority argument has a real effect. Note: Linux's setpriority is
    // process-level (PRIO_PROCESS) and applies to the entire process, not
    // a single thread — there is no per-thread nice on Linux. The caller
    // is therefore expected to set priority from a fresh forked process
    // dedicated to one thread, or accept that the entire process nice is
    // adjusted. setpriority returns 0 on success, -1 on failure
    // (EACCES/EPERM for non-root). For priority == 0 we skip this path
    // entirely (no nice adjustment is needed).
    if (policy == SCHED_OTHER && priority != 0) {
        if (setpriority(PRIO_PROCESS, 0, priority) != 0) {
            // Permission denied or other error: return false honestly
            // instead of pretending success. Callers that want "best
            // effort" can ignore the return code; callers that want strict
            // application will now be told the truth.
            return false;
        }
    }

    return true;
}

bool set_cpu_affinity(std::thread::native_handle_type handle,
                      const std::vector<int>& cpu_ids) {
    if (cpu_ids.empty()) {
        return false;
    }
    
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    
    for (int cpu_id : cpu_ids) {
        if (cpu_id < 0 || cpu_id >= CPU_SETSIZE) {
            return false;
        }
        CPU_SET(static_cast<size_t>(cpu_id), &cpuset);
    }
    
#ifdef __ANDROID__
    // bionic 没有 glibc 的 pthread_setaffinity_np 扩展；Android 上对当前线程
    // 设置 affinity 使用 gettid() + sched_setaffinity()。普通 App 仍可能因
    // cgroup/SELinux 限制而失败，调用方继续按 best-effort 处理返回值。
    (void)handle;
    return sched_setaffinity(gettid(), sizeof(cpu_set_t), &cpuset) == 0;
#else
    return pthread_setaffinity_np(handle, sizeof(cpu_set_t), &cpuset) == 0;
#endif
}

int get_current_thread_priority() {
    struct sched_param param;
    int policy;
    
    pthread_t thread = pthread_self();
    if (pthread_getschedparam(thread, &policy, &param) != 0) {
        return 0;
    }
    
    if (policy == SCHED_FIFO || policy == SCHED_RR) {
        // 实时调度策略，返回优先级值（1-99）
        return param.sched_priority;
    } else {
        // 普通调度策略，返回nice值（-20到19）
        errno = 0;
        int nice_val = getpriority(PRIO_PROCESS, 0);
        if (errno != 0) {
            return 0;
        }
        return nice_val;
    }
}

std::vector<int> get_current_thread_affinity() {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    
#ifdef __ANDROID__
    if (sched_getaffinity(gettid(), sizeof(cpu_set_t), &cpuset) != 0) {
        return {};
    }
#else
    pthread_t thread = pthread_self();
    if (pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset) != 0) {
        return {};
    }
#endif
    
    std::vector<int> cpu_ids;
    for (int i = 0; i < CPU_SETSIZE; ++i) {
        if (CPU_ISSET(static_cast<size_t>(i), &cpuset)) {
            cpu_ids.push_back(i);
        }
    }
    
    return cpu_ids;
}

void set_current_thread_name(const std::string& name) {
    // pthread_setname_np 限制线程名最长 15 字符 + '\0'，超出会失败
    // 这里主动截断到 15 字符以保证设置成功
    std::string truncated = name.substr(0, 15);
    pthread_setname_np(pthread_self(), truncated.c_str());
}

bool set_current_thread_timer_slack_ns(uint64_t slack_ns) {
    return prctl(PR_SET_TIMERSLACK, static_cast<unsigned long>(slack_ns)) == 0;
}

#endif

// ---- 进程级内存锁定：真实系统调用与引用计数租约 ----

ProcessMemoryLockResult RealProcessMemoryLockSyscalls::mlockall_current_future() {
#if defined(__linux__)
    if (::mlockall(MCL_CURRENT | MCL_FUTURE) == 0) {
        return {true, 0};
    }
    return {false, errno};
#else
    // Windows 不支持 mlockall，无对应的进程级内存锁定语义
    return {false, ERROR_NOT_SUPPORTED};
#endif
}

int RealProcessMemoryLockSyscalls::munlockall() {
#if defined(__linux__)
    return ::munlockall();
#else
    return -1;
#endif
}

ProcessMemoryLockResult try_mlock_process_memory() {
    // 无租约语义的一次性调用：仅报告结果，不参与引用计数（历史公开行为）。
    RealProcessMemoryLockSyscalls real_syscalls;
    return real_syscalls.mlockall_current_future();
}

namespace {
// 引用计数与替身接缝由同一把互斥保护，保证多执行器并发启停时
// mlockall/munlockall 仍严格按“首获取锁定 / 末释放解锁”配对。
std::mutex g_process_memory_lock_mutex;
int g_process_memory_lock_refcount = 0;
ProcessMemoryLockSyscalls* g_process_memory_lock_test_syscalls = nullptr;

ProcessMemoryLockSyscalls& process_memory_lock_syscalls() {
    static RealProcessMemoryLockSyscalls real_syscalls;
    if (g_process_memory_lock_test_syscalls != nullptr) {
        return *g_process_memory_lock_test_syscalls;
    }
    return real_syscalls;
}
}  // namespace

ProcessMemoryLockLease ProcessMemoryLockLease::try_acquire() {
    std::lock_guard<std::mutex> lock(g_process_memory_lock_mutex);
    if (g_process_memory_lock_refcount > 0) {
        ++g_process_memory_lock_refcount;
        return ProcessMemoryLockLease(true, 0);
    }
    const ProcessMemoryLockResult result =
        process_memory_lock_syscalls().mlockall_current_future();
    if (!result.applied) {
        return ProcessMemoryLockLease(false, result.error_code);
    }
    ++g_process_memory_lock_refcount;
    return ProcessMemoryLockLease(true, 0);
}

ProcessMemoryLockLease::ProcessMemoryLockLease(ProcessMemoryLockLease&& other) noexcept
    : held_(other.held_), error_code_(other.error_code_) {
    other.held_ = false;
    other.error_code_ = 0;
}

ProcessMemoryLockLease& ProcessMemoryLockLease::operator=(ProcessMemoryLockLease&& other) noexcept {
    if (this != &other) {
        release();
        held_ = other.held_;
        error_code_ = other.error_code_;
        other.held_ = false;
        other.error_code_ = 0;
    }
    return *this;
}

ProcessMemoryLockLease::~ProcessMemoryLockLease() {
    release();
}

void ProcessMemoryLockLease::release() {
    if (!held_) {
        return;
    }
    held_ = false;
    std::lock_guard<std::mutex> lock(g_process_memory_lock_mutex);
    if (--g_process_memory_lock_refcount == 0) {
        (void)process_memory_lock_syscalls().munlockall();
    }
}

void ProcessMemoryLockLease::set_syscalls_for_test(ProcessMemoryLockSyscalls* syscalls) {
    std::lock_guard<std::mutex> lock(g_process_memory_lock_mutex);
    assert(g_process_memory_lock_refcount == 0);
    g_process_memory_lock_test_syscalls = syscalls;
}

} // namespace util
} // namespace executor
