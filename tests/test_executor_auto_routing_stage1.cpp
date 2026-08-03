#include <executor/executor.hpp>

#include <atomic>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace executor;

#define TEST_ASSERT(condition, message)                                      \
    do {                                                                     \
        if (!(condition)) {                                                  \
            std::cerr << "FAILED: " << message << " at " << __FILE__       \
                      << ':' << __LINE__ << std::endl;                      \
            return false;                                                    \
        }                                                                    \
    } while (0)

namespace {

bool future_throws(std::future<void> future) {
    try {
        future.get();
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

bool test_plain_submit_auto_uses_default_async_executor() {
    Executor executor;
    auto future = executor.submit_auto([] { return 42; });
    TEST_ASSERT(future.get() == 42, "plain submit_auto should fulfill CPU future");
    executor.shutdown();
    return true;
}

bool test_unavailable_gpu_allow_cpu_falls_back() {
    Executor executor;
    std::atomic<int> cpu_runs{0};
    std::atomic<int> gpu_runs{0};
    auto future = executor.submit_auto(
        cpu_gpu_task(
            [&] { ++cpu_runs; },
            [&](void*) { ++gpu_runs; })
            .name("fallback-cpu")
            .preferred_executor("missing-gpu")
            .fallback(FallbackPolicy::AllowCpu)
            .prefer_gpu());
    future.get();

    TEST_ASSERT(cpu_runs == 1, "AllowCpu should execute CPU path when GPU is unavailable");
    TEST_ASSERT(gpu_runs == 0, "unavailable GPU path must not execute");
    TEST_ASSERT(executor.get_failure_status().submit_rejected_count == 0,
                "allowed CPU fallback is not a submission failure");
    executor.shutdown();
    return true;
}

bool test_unavailable_gpu_without_fallback_rejects() {
    Executor executor;
    auto future = executor.submit_auto(
        cpu_gpu_task([] {}, [](void*) {})
            .name("reject-missing-gpu")
            .preferred_executor("missing-gpu")
            .prefer_gpu());
    TEST_ASSERT(future_throws(std::move(future)),
                "NoFallback should reject unavailable GPU");
    TEST_ASSERT(executor.get_failure_status().submit_rejected_count == 1,
                "rejected automatic submission should be observable");
    executor.shutdown();
    return true;
}

bool test_require_requested_backend_requires_running_gpu() {
    Executor executor;
    auto future = executor.submit_auto(
        cpu_gpu_task([] {}, [](void*) {})
            .name("require-gpu")
            .preferred_executor("missing-gpu")
            .fallback(FallbackPolicy::RequireRequestedBackend));
    TEST_ASSERT(future_throws(std::move(future)),
                "required GPU backend should reject when unavailable");
    TEST_ASSERT(executor.get_failure_status().submit_rejected_count == 1,
                "required backend rejection should be observable");
    executor.shutdown();
    return true;
}

bool test_unsupported_generic_intent_rejects() {
    Executor executor;
    auto future = executor.submit_auto(task([] {}).intent(ExecutionIntent::LowLatency));
    TEST_ASSERT(future_throws(std::move(future)),
                "generic submit_auto should reject LowLatency intent");
    TEST_ASSERT(executor.get_failure_status().submit_rejected_count == 1,
                "typed API rejection should be observable");
    executor.shutdown();
    return true;
}

}  // namespace

int main() {
    bool passed = true;
    passed &= test_plain_submit_auto_uses_default_async_executor();
    passed &= test_unavailable_gpu_allow_cpu_falls_back();
    passed &= test_unavailable_gpu_without_fallback_rejects();
    passed &= test_require_requested_backend_requires_running_gpu();
    passed &= test_unsupported_generic_intent_rejects();
    return passed ? 0 : 1;
}
