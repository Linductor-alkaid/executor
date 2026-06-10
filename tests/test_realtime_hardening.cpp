#include <cassert>
#include <iostream>
#include <string>
#include <fstream>
#include <thread>

// 包含线程工具头文件
#include "executor/util/thread_utils.hpp"

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

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Realtime Hardening Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    bool all_passed = true;

    all_passed &= test_try_mlock_current_thread();
    all_passed &= test_set_current_thread_name();
    all_passed &= test_set_current_thread_timer_slack_ns();

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
