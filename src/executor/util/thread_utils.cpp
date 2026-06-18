#include "thread_utils.hpp"

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

bool try_mlock_current_thread() {
    // Windows 不支持 mlockall，无对应的进程级内存锁定语义，直接返回 false
    return false;
}

void set_current_thread_name(const std::string& name) {
    // Windows 通过 SetThreadDescription 设置线程名（需要 Win10 1607+）
    // 将窄字符串转换为宽字符串
    std::wstring wname(name.begin(), name.end());
    SetThreadDescription(GetCurrentThread(), wname.c_str());
}

void set_current_thread_timer_slack_ns(uint64_t /*slack_ns*/) {
    // Windows 不支持 per-thread timer slack，空实现
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
        CPU_SET(cpu_id, &cpuset);
    }
    
    return pthread_setaffinity_np(handle, sizeof(cpu_set_t), &cpuset) == 0;
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
    
    pthread_t thread = pthread_self();
    if (pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset) != 0) {
        return {};
    }
    
    std::vector<int> cpu_ids;
    for (int i = 0; i < CPU_SETSIZE; ++i) {
        if (CPU_ISSET(i, &cpuset)) {
            cpu_ids.push_back(i);
        }
    }
    
    return cpu_ids;
}

bool try_mlock_current_thread() {
    // 锁定当前及未来的所有页面，防止分页到 swap 引入抖动
    // 失败（如无 CAP_IPC_LOCK 权限）时静默返回 false，不抛异常
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == 0) {
        return true;
    }
    return false;
}

void set_current_thread_name(const std::string& name) {
    // pthread_setname_np 限制线程名最长 15 字符 + '\0'，超出会失败
    // 这里主动截断到 15 字符以保证设置成功
    std::string truncated = name.substr(0, 15);
    pthread_setname_np(pthread_self(), truncated.c_str());
}

void set_current_thread_timer_slack_ns(uint64_t slack_ns) {
    prctl(PR_SET_TIMERSLACK, static_cast<unsigned long>(slack_ns));
}

#endif

} // namespace util
} // namespace executor
