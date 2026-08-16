#include <atomic>
#include <iostream>
#include <thread>
#include <type_traits>

#include <executor/stop_token.hpp>

#if !defined(__ANDROID__) || defined(__cpp_lib_jthread)
static_assert(std::is_same_v<executor::StopToken, std::stop_token>,
              "executor::StopToken must alias std::stop_token on desktop/standard-lib path");
#endif

#define TEST_ASSERT(condition, message)                                      \
    do {                                                                     \
        if (!(condition)) {                                                  \
            std::cerr << "FAILED: " << message << " at " << __FILE__      \
                      << ':' << __LINE__ << '\n';                          \
            return false;                                                    \
        }                                                                    \
    } while (0)

bool test_stop_source_request_once() {
    executor::StopSource source;
    const executor::StopToken token = source.get_token();

    TEST_ASSERT(token.stop_possible(), "stop token must be possible from a source");
    TEST_ASSERT(!token.stop_requested(), "stop token must not be requested initially");
    TEST_ASSERT(source.request_stop(), "first request_stop must report a new request");
    TEST_ASSERT(token.stop_requested(), "stop token must observe request_stop");
    TEST_ASSERT(!source.request_stop(), "repeated request_stop must report no new request");
    return true;
}

bool test_jthread_passes_stop_token_and_joins() {
    std::atomic<bool> entered{false};
    std::atomic<bool> left{false};

    executor::detail::JThread thread([&](executor::StopToken token) {
        entered.store(true, std::memory_order_release);
        while (!token.stop_requested()) {
            std::this_thread::yield();
        }
        left.store(true, std::memory_order_release);
    });

    while (!entered.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    TEST_ASSERT(thread.joinable(), "jthread must be joinable while running");
    TEST_ASSERT(thread.request_stop(), "jthread request_stop must report a new request");
    thread.join();
    TEST_ASSERT(left.load(std::memory_order_acquire), "jthread entry must observe stop token");
    TEST_ASSERT(!thread.joinable(), "jthread must not be joinable after join");
    return true;
}

int main() {
    bool ok = true;
    ok &= test_stop_source_request_once();
    ok &= test_jthread_passes_stop_token_and_joins();
    std::cout << (ok ? "All stop_token compatibility tests PASSED\n"
                     : "stop_token compatibility tests FAILED\n");
    return ok ? 0 : 1;
}
