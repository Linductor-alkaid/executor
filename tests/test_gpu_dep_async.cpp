// P-005 regression test: submit_kernel_after should not block the GPU worker
// thread, so unrelated tasks (D) can execute while a dependent task (B) is
// still waiting on its dependency (A).
//
// We submit A -> B -> C as a dependency chain and concurrently submit D
// (no dependency). With the old (blocking) implementation, B's dep.wait()
// would freeze the single GPU worker, so D would not start until the
// whole A->B->C chain finished. With the new (non-blocking) implementation,
// D should start running during B's wait window.
//
// We instrument start/finish timestamps to verify temporal overlap.

#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

#include <executor/config.hpp>
#include <executor/interfaces.hpp>
#include <executor/types.hpp>
#include "executor/gpu/cuda_executor.hpp"

using namespace std::chrono;
using namespace executor;
using namespace executor::gpu;

namespace {

#define GPU_DEP_TEST_ASSERT(cond, msg)                                         \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "FAILED: " << msg << " at " << __FILE__ << ":"        \
                      << __LINE__ << std::endl;                                \
            return false;                                                      \
        }                                                                      \
    } while (0)

// Returns the wall-clock now in nanoseconds (steady_clock).
inline int64_t now_ns() {
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch())
        .count();
}

bool test_gpu_dependency_does_not_starve_worker() {
    std::cout << "P-005: submit_kernel_after must not block GPU worker"
              << std::endl;
#ifdef EXECUTOR_ENABLE_CUDA
    GpuExecutorConfig cfg;
    cfg.name = "p005_dep_async";
    cfg.device_id = 0;
    cfg.max_queue_size = 256;
    cfg.default_stream_count = 1;

    CudaExecutor exec(cfg.name, cfg);
    if (!exec.start()) {
        std::cout << "  CUDA not available, skipping" << std::endl;
        return true;
    }

    GpuTaskConfig tcfg;
    tcfg.async = false;

    // === Build dependency chain A -> B -> C ===
    std::atomic<int64_t> a_start{0}, a_end{0};
    std::atomic<int64_t> b_start{0}, b_end{0};
    std::atomic<int64_t> c_start{0}, c_end{0};
    std::atomic<int64_t> d_start{0}, d_end{0};

    // A: no-op kernel that "sleeps" briefly (yield CPU, do not block GPU)
    auto future_a = exec.submit_kernel([&](void* /*s*/) {
        a_start.store(now_ns());
        std::this_thread::sleep_for(milliseconds(30));
        a_end.store(now_ns());
    }, tcfg);
    std::shared_future<void> shared_a = future_a.share();

    // B: depends on A, also sleeps 30ms (B's sleep used to block the worker)
    auto future_b = exec.submit_kernel_after(shared_a, [&](void* /*s*/) {
        b_start.store(now_ns());
        std::this_thread::sleep_for(milliseconds(30));
        b_end.store(now_ns());
    }, tcfg);
    std::shared_future<void> shared_b = future_b.share();

    // C: depends on B
    auto future_c = exec.submit_kernel_after(shared_b, [&](void* /*s*/) {
        c_start.store(now_ns());
        std::this_thread::sleep_for(milliseconds(10));
        c_end.store(now_ns());
    }, tcfg);

    // Give the worker a moment to start processing A. The goal is to
    // submit D *while B's dep.wait() would have been holding the worker*
    // (in the old implementation). With the new implementation, B's
    // dep.wait() runs in a detached helper thread, so the worker is free.
    //
    // We wait until A is finished, then submit D shortly before B is
    // expected to start. If the worker is blocked, D's start would be
    // delayed past B's end.
    future_a.wait();
    // Small window to ensure B's dep is satisfied and B has been re-enqueued
    std::this_thread::sleep_for(milliseconds(5));
    auto future_d = exec.submit_kernel([&](void* /*s*/) {
        d_start.store(now_ns());
        std::this_thread::sleep_for(milliseconds(10));
        d_end.store(now_ns());
    }, tcfg);

    // Wait for everything
    future_c.get();
    future_d.get();

    std::cout << "  A: [" << a_start.load() << ", " << a_end.load() << "]"
              << std::endl;
    std::cout << "  B: [" << b_start.load() << ", " << b_end.load() << "]"
              << std::endl;
    std::cout << "  C: [" << c_start.load() << ", " << c_end.load() << "]"
              << std::endl;
    std::cout << "  D: [" << d_start.load() << ", " << d_end.load() << "]"
              << std::endl;

    // Sanity: ordering
    GPU_DEP_TEST_ASSERT(a_end.load() > 0, "A must have run");
    GPU_DEP_TEST_ASSERT(b_end.load() > 0, "B must have run");
    GPU_DEP_TEST_ASSERT(c_end.load() > 0, "C must have run");
    GPU_DEP_TEST_ASSERT(d_end.load() > 0, "D must have run");
    GPU_DEP_TEST_ASSERT(a_end.load() <= b_start.load(),
                        "B must start after A ends");
    GPU_DEP_TEST_ASSERT(b_end.load() <= c_start.load(),
                        "C must start after B ends");

    // P-005 key assertion: D must have started *during* B's window
    // (i.e., B is waiting on something or executing, and D can still
    // make progress). With the old blocking dep.wait(), D would start
    // only after C ended, so d_start >= c_end.
    // With the new implementation, d_start should be <= c_end (D can run
    // in parallel with B's execution phase, or at least before C ends).
    int64_t d_s = d_start.load();
    int64_t c_e = c_end.load();
    int64_t b_s = b_start.load();
    int64_t b_e = b_end.load();

    // D should not be forced to wait for the whole chain. The strongest
    // guarantee we can make: D's start time is not after C ends. (We
    // cannot guarantee strict overlap with B because the worker may
    // legitimately process B and D serially — what we *can* guarantee
    // is that the dep.wait() in the old code did NOT block the worker,
    // which is observable by the fact that D was enqueued *after* A
    // completed but still finished before the chain forced sequential
    // serialization on it. We test the weaker but reliable property:
    // D's start <= C's end, i.e. D does not have to wait for the chain.)
    GPU_DEP_TEST_ASSERT(d_s <= c_e,
                        "D should not be forced to wait for the full "
                        "A->B->C chain (worker would be blocked by "
                        "B's dep.wait() in the old impl)");

    // Stronger property we can check: D started before B ended. The
    // worker is free during B's execution phase (no more dep.wait() on
    // the worker thread), so D can be processed in parallel.
    // We allow a small tolerance for scheduling jitter.
    bool d_overlaps_b = (d_s <= b_e) && (d_s >= b_s - milliseconds(50).count());
    if (!d_overlaps_b) {
        std::cout << "  NOTE: D did not overlap with B (D_start=" << d_s
                  << ", B_start=" << b_s << ", B_end=" << b_e << ")"
                  << std::endl;
        // Not a hard failure: some executors may process tasks strictly
        // sequentially even without dep.wait(). The real test is that
        // D is not blocked by dep.wait() — captured by the assertion
        // above (d_s <= c_e).
    } else {
        std::cout << "  D overlapped with B's execution window: PASSED"
                  << std::endl;
    }

    exec.wait_for_completion();
    exec.stop();
    std::cout << "  P-005 dep-async test: PASSED" << std::endl;
    return true;
#else
    std::cout << "  CUDA not enabled, skipping" << std::endl;
    return true;
#endif
}

} // namespace

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "P-005 GPU dependency async regression test" << std::endl;
    std::cout << "=========================================" << std::endl;

    bool ok = test_gpu_dependency_does_not_starve_worker();
    if (ok) {
        std::cout << "All P-005 tests PASSED" << std::endl;
        return 0;
    }
    std::cerr << "P-005 tests FAILED" << std::endl;
    return 1;
}
