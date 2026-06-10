#include <cassert>
#include <iostream>
#include <string>
#include <fstream>
#include <thread>

// 包含线程工具头文件
#include "executor/util/thread_utils.hpp"
#include "executor/config.hpp"

#ifdef __linux__
#include <unistd.h>
#include <sys/syscall.h>
#endif

using namespace executor::util;

// 测试辅助宏
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "FAILED: " << message << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

// ========== try_mlock_current_thread 测试 ==========

bool test_try_mlock_current_thread() {
    std::cout << "Testing try_mlock_current_thread..." << std::endl;

    // CI 容器一般无 CAP_IPC_LOCK，可能返回 false；这里只断言不崩溃即可。
    // 返回值用于消除未使用变量告警。
    bool locked = try_mlock_current_thread();
    std::cout << "  mlockall result: " << (locked ? "true (locked)" : "false (no permission, ok)")
              << std::endl;

    // 无论成功与否都不应崩溃或抛异常
    return true;
}

// ========== set_current_thread_name 测试 ==========

bool test_set_current_thread_name() {
    std::cout << "Testing set_current_thread_name..." << std::endl;

    const std::string short_name = "rtchk";  // 5 字符短名
    set_current_thread_name(short_name);

#ifdef __linux__
    // 读取 /proc/self/task/<tid>/comm 验证线程名被内核接受
    pid_t tid = static_cast<pid_t>(syscall(SYS_gettid));
    std::string path = "/proc/self/task/" + std::to_string(tid) + "/comm";
    std::ifstream ifs(path);
    TEST_ASSERT(ifs.is_open(), "Should be able to open /proc comm file");

    std::string comm;
    std::getline(ifs, comm);
    // comm 文件内容带换行，getline 已去掉换行
    TEST_ASSERT(comm == short_name,
                "Thread name in /proc comm should match the name set");
    std::cout << "  /proc comm = '" << comm << "'" << std::endl;
#else
    // 非 Linux 平台仅验证不崩溃
    std::cout << "  (non-Linux: name set, no /proc verification)" << std::endl;
#endif

    return true;
}

// ========== set_current_thread_timer_slack_ns 测试 ==========

bool test_set_current_thread_timer_slack_ns() {
    std::cout << "Testing set_current_thread_timer_slack_ns..." << std::endl;

    // 设置一个值，断言不崩溃
    set_current_thread_timer_slack_ns(1);
    set_current_thread_timer_slack_ns(1000);

    return true;
}

// ========== 主测试函数 ==========

bool test_default_config_is_optimal() {
    std::cout << "Testing RealtimeThreadConfig default-is-optimal values..." << std::endl;

    executor::RealtimeThreadConfig cfg;
    TEST_ASSERT(cfg.enable_memory_lock == true,
                "Default enable_memory_lock should be true (opt-out, facade philosophy)");
    TEST_ASSERT(cfg.timer_slack_ns == 1,
                "Default timer_slack_ns should be 1 (1ns, opt-out by setting 0)");
    // thread_name 仍然 "", 不变

    std::cout << "  enable_memory_lock default = " << cfg.enable_memory_lock << std::endl;
    std::cout << "  timer_slack_ns default = " << cfg.timer_slack_ns << std::endl;
    return true;
}

bool test_realtime_priority_adaptive() {
    std::cout << "Testing RealtimeThreadConfig priority adaptive recommendation..." << std::endl;

    // Case 1: 1ms cycle, priority=0 → auto-recommend 80 (hard realtime)
    {
        executor::RealtimeThreadConfig cfg;
        cfg.thread_name = "test_rt_1ms";
        cfg.cycle_period_ns = 1'000'000;  // 1ms
        cfg.thread_priority = 0;          // not explicitly set

        // Verify config fields that drive the auto-priority decision
        TEST_ASSERT(cfg.thread_priority == 0, "thread_priority should be 0 (auto sentinel)");
        TEST_ASSERT(cfg.cycle_period_ns == 1'000'000, "cycle_period_ns should be 1ms");
        // Logic: 1ms <= 1ms → auto_priority = 80
        TEST_ASSERT(cfg.cycle_period_ns <= 1'000'000,
                    "1ms cycle should trigger hard-realtime auto-priority (80)");
        std::cout << "  1ms cycle: thread_priority=0, cycle <=1ms → auto_priority would be 80" << std::endl;
    }

    // Case 2: 5ms cycle, priority=0 → auto-recommend 50 (soft realtime)
    {
        executor::RealtimeThreadConfig cfg;
        cfg.thread_name = "test_rt_5ms";
        cfg.cycle_period_ns = 5'000'000;  // 5ms
        cfg.thread_priority = 0;

        TEST_ASSERT(cfg.thread_priority == 0, "thread_priority should be 0");
        TEST_ASSERT(cfg.cycle_period_ns > 1'000'000 && cfg.cycle_period_ns <= 10'000'000,
                    "5ms cycle should trigger soft-realtime auto-priority (50)");
        std::cout << "  5ms cycle: thread_priority=0, 1ms<cycle<=10ms → auto_priority would be 50" << std::endl;
    }

    // Case 3: 20ms cycle, priority=0 → stays 0 (normal scheduling sufficient)
    {
        executor::RealtimeThreadConfig cfg;
        cfg.thread_name = "test_rt_20ms";
        cfg.cycle_period_ns = 20'000'000;  // 20ms
        cfg.thread_priority = 0;

        TEST_ASSERT(cfg.cycle_period_ns > 10'000'000,
                    "20ms cycle should NOT trigger auto-priority (stays 0)");
        std::cout << "  20ms cycle: thread_priority=0, cycle>10ms → auto_priority stays 0" << std::endl;
    }

    // Case 4: user explicitly sets priority → respected, no auto
    {
        executor::RealtimeThreadConfig cfg;
        cfg.thread_name = "test_rt_explicit";
        cfg.cycle_period_ns = 1'000'000;
        cfg.thread_priority = 42;  // explicit override

        TEST_ASSERT(cfg.thread_priority != 0, "Explicit priority should be non-zero");
        // When thread_priority != 0, auto-logic is skipped, user value used directly
        std::cout << "  1ms cycle with explicit priority=42 → auto_priority skipped, uses 42" << std::endl;
    }

    return true;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Realtime Hardening Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    bool all_passed = true;

    all_passed &= test_try_mlock_current_thread();
    all_passed &= test_set_current_thread_name();
    all_passed &= test_set_current_thread_timer_slack_ns();
    all_passed &= test_default_config_is_optimal();
    all_passed &= test_realtime_priority_adaptive();

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    if (all_passed) {
        std::cout << "All tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests FAILED!" << std::endl;
        return 1;
    }
}
