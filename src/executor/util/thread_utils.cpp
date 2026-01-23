#include "thread_utils.hpp"

#ifdef _WIN32
    #include <windows.h>
#elif defined(__linux__)
    #include <pthread.h>
    #include <sched.h>
    #include <unistd.h>
    #include <sys/syscall.h>
    #include <sys/resource.h>
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
        // 注意：nice值需要通过setpriority设置，这里只设置调度策略
    }
    
    int result = pthread_setschedparam(handle, policy, &param);
    if (result == 0 && policy == SCHED_OTHER && priority != 0) {
        // 对于普通优先级，还需要设置nice值
        // 注意：这需要在调用线程中设置，不能直接设置其他线程的nice值
        // 这里简化处理，只设置调度策略
    }
    
    return result == 0;
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

#endif

} // namespace util
} // namespace executor
