#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <thread>

#include <executor/executor.hpp>
#include <executor/executor_manager.hpp>
#include <executor/interfaces.hpp>
#include <executor/monitor/executor_monitor.hpp>

using namespace executor;

#define TEST_ASSERT(condition, message)                                      \
    do {                                                                     \
        if (!(condition)) {                                                  \
            std::cerr << "FAILED: " << message << " at " << __FILE__       \
                      << ":" << __LINE__ << std::endl;                     \
            return false;                                                    \
        }                                                                    \
    } while (0)

class IdleBlockingWorker final : public IBlockingIoWorker {
public:
    void run(std::stop_token stop_token) override {
        while (!stop_token.stop_requested()) {
            std::this_thread::yield();
        }
    }

    void wakeup() noexcept override {}
};

class SnapshotMockGpuExecutor final : public IGpuExecutor {
public:
    explicit SnapshotMockGpuExecutor(std::string name) : name_(std::move(name)) {}

    std::string get_name() const override { return name_; }
    gpu::GpuDeviceInfo get_device_info() const override {
        gpu::GpuDeviceInfo info;
        info.name = name_ + "_device";
        info.backend = gpu::GpuBackend::CUDA;
        return info;
    }
    gpu::GpuExecutorStatus get_status() const override {
        gpu::GpuExecutorStatus status;
        status.name = name_;
        status.backend = gpu::GpuBackend::CUDA;
        status.is_running = running_.load(std::memory_order_acquire);
        status.active_kernels = 2;
        status.queue_size = 3;
        status.failed_kernels = 1;
        return status;
    }
    bool start() override {
        running_.store(true, std::memory_order_release);
        return true;
    }
    void stop() override { running_.store(false, std::memory_order_release); }
    void wait_for_completion() override {}
    void* allocate_device_memory(size_t) override { return nullptr; }
    void free_device_memory(void*) override {}
    bool copy_to_device(void*, const void*, size_t, bool, int) override { return false; }
    bool copy_to_host(void*, const void*, size_t, bool, int) override { return false; }
    bool copy_device_to_device(void*, const void*, size_t, bool, int) override { return false; }
    void synchronize() override {}
    void synchronize_stream(int) override {}
    int create_stream() override { return -1; }
    void destroy_stream(int) override {}
    bool add_stream_callback(int, std::function<void()>) override { return false; }

protected:
    std::future<void> submit_kernel_impl(
        std::function<void(void*)>, const gpu::GpuTaskConfig&) override {
        std::promise<void> promise;
        promise.set_value();
        return promise.get_future();
    }

private:
    std::string name_;
    std::atomic<bool> running_{false};
};

bool test_snapshot_does_not_lazy_initialize() {
    Executor executor;
    const auto before = executor.get_snapshot();
    const auto after = executor.get_snapshot();

    TEST_ASSERT(before.lifecycle == ExecutorLifecycleState::Created,
                "uninitialized executor must report Created");
    TEST_ASSERT(!before.completion.is_initialized,
                "snapshot must not initialize the default async executor");
    TEST_ASSERT(!after.async.is_running,
                "default async executor must remain stopped after snapshot");
    TEST_ASSERT(after.snapshot_sequence == before.snapshot_sequence + 1,
                "snapshot sequence must increase monotonically");
    return true;
}

bool test_snapshot_reports_async_work_and_shutdown() {
    Executor executor;
    ExecutorConfig config;
    config.min_threads = 1;
    config.max_threads = 1;
    TEST_ASSERT(executor.initialize(config), "executor initialization must succeed");

    const auto running = executor.get_snapshot();
    TEST_ASSERT(running.lifecycle == ExecutorLifecycleState::Running,
                "initialized executor must report Running");
    TEST_ASSERT(running.async.is_running,
                "snapshot must include the running async backend");
    TEST_ASSERT(running.async.is_running == executor.get_async_executor_status().is_running,
                "snapshot async status must agree with the existing status API");
    TEST_ASSERT(running.running_backend_count >= 1,
                "running backend count must include async backend");

    std::promise<void> release;
    auto gate = release.get_future().share();
    std::promise<void> started;
    auto task = executor.submit([gate, &started]() {
        started.set_value();
        gate.wait();
    });
    started.get_future().wait();

    const auto active = executor.get_snapshot();
    TEST_ASSERT(active.active_task_count >= 1,
                "snapshot must include active async tasks");
    TEST_ASSERT(active.completion.pending_tasks >= 1,
                "completion snapshot must agree that work is pending");

    release.set_value();
    task.get();
    executor.wait_for_completion();
    executor.shutdown();

    const auto stopped = executor.get_snapshot();
    TEST_ASSERT(stopped.lifecycle == ExecutorLifecycleState::Stopped,
                "completed shutdown must report Stopped");
    TEST_ASSERT(!stopped.async.is_running,
                "async backend must be stopped after shutdown");
    return true;
}

bool test_snapshot_reports_failure_and_failed_initialization() {
    Executor invalid_executor;
    ExecutorConfig invalid_config;
    invalid_config.min_threads = 2;
    invalid_config.max_threads = 1;
    TEST_ASSERT(!invalid_executor.initialize(invalid_config),
                "invalid configuration must fail initialization");
    const auto failed = invalid_executor.get_snapshot();
    TEST_ASSERT(failed.lifecycle == ExecutorLifecycleState::Failed,
                "initialization failure must report Failed");
    TEST_ASSERT(failed.failures.submit_rejected_count >= 1,
                "initialization failure must be visible in failure counters");
    TEST_ASSERT(!failed.recent_failures.empty(),
                "initialization failure must retain a recent failure event");

    Executor executor;
    TEST_ASSERT(!executor.start_realtime_task("missing"),
                "missing realtime backend must fail visibly");
    const auto snapshot = executor.get_snapshot();
    TEST_ASSERT(snapshot.failures.submit_rejected_count >= 1,
                "snapshot must include facade failure status");
    TEST_ASSERT(!snapshot.recent_failures.empty(),
                "snapshot must include recent facade failures");
    return true;
}

bool test_snapshot_includes_registered_backends() {
    Executor executor;
    RealtimeThreadConfig realtime_config;
    realtime_config.thread_name = "snapshot-rt";
    realtime_config.cycle_period_ns = 1'000'000;
    TEST_ASSERT(executor.register_realtime_task("snapshot_rt", realtime_config),
                "realtime backend registration must succeed");

    BlockingIoConfig blocking_config;
    blocking_config.thread_name = "snapshot-io";
    TEST_ASSERT(executor.register_blocking_io_worker(
                    "snapshot_io", blocking_config,
                    std::make_unique<IdleBlockingWorker>()),
                "blocking I/O backend registration must succeed");

    const auto snapshot = executor.get_snapshot();
    TEST_ASSERT(snapshot.realtime.contains("snapshot_rt"),
                "snapshot must include registered realtime backend");
    TEST_ASSERT(snapshot.blocking_io.contains("snapshot_io"),
                "snapshot must include registered Blocking I/O backend");
    executor.shutdown();
    return true;
}

bool test_monitor_includes_registered_gpu_backend() {
    ExecutorManager manager;
    auto gpu_executor = std::make_unique<SnapshotMockGpuExecutor>("snapshot_gpu");
    TEST_ASSERT(gpu_executor->start(), "mock GPU backend must start");
    TEST_ASSERT(manager.register_gpu_executor("snapshot_gpu", std::move(gpu_executor)),
                "mock GPU backend registration must succeed");

    std::atomic<ExecutorLifecycleState> lifecycle{ExecutorLifecycleState::Running};
    monitor::ExecutorMonitor monitor(
        manager, lifecycle,
        [] { return CompletionStatus{}; },
        [] { return ExecutorFailureStatus{}; },
        [] { return std::vector<ExecutorFailureEvent>{}; },
        [] { return std::map<std::string, TaskStatistics>{}; });
    const auto snapshot = monitor.collect();

    const auto gpu = snapshot.gpu.find("snapshot_gpu");
    TEST_ASSERT(gpu != snapshot.gpu.end(),
                "snapshot must include the registered GPU backend");
    TEST_ASSERT(gpu->second.is_running,
                "snapshot must preserve GPU running state");
    TEST_ASSERT(snapshot.active_task_count == 2 && snapshot.queued_task_count == 3,
                "GPU work counters must contribute to aggregate counters");
    TEST_ASSERT(snapshot.failed_task_count == 1,
                "GPU failed kernels must contribute to aggregate counters");
    manager.shutdown();
    return true;
}

bool test_snapshot_is_safe_during_shutdown() {
    Executor executor;
    ExecutorConfig config;
    config.min_threads = 1;
    config.max_threads = 1;
    TEST_ASSERT(executor.initialize(config), "executor initialization must succeed");

    std::atomic<bool> stop{false};
    std::thread reader([&executor, &stop]() {
        while (!stop.load(std::memory_order_acquire)) {
            (void)executor.get_snapshot();
        }
    });

    executor.shutdown();
    stop.store(true, std::memory_order_release);
    reader.join();

    TEST_ASSERT(executor.get_snapshot().lifecycle == ExecutorLifecycleState::Stopped,
                "snapshot must remain usable after shutdown");
    return true;
}

bool test_snapshot_observes_draining() {
    Executor executor;
    ExecutorConfig config;
    config.min_threads = 1;
    config.max_threads = 1;
    TEST_ASSERT(executor.initialize(config), "executor initialization must succeed");

    std::promise<void> release;
    auto gate = release.get_future().share();
    std::promise<void> started;
    auto task = executor.submit([gate, &started]() {
        started.set_value();
        gate.wait();
    });
    started.get_future().wait();

    std::atomic<bool> shutdown_complete{false};
    std::thread shutdown_thread([&executor, &shutdown_complete]() {
        executor.shutdown();
        shutdown_complete.store(true, std::memory_order_release);
    });

    bool observed_draining = false;
    for (int attempt = 0; attempt < 100 && !shutdown_complete.load(std::memory_order_acquire);
         ++attempt) {
        observed_draining |= executor.get_snapshot().lifecycle == ExecutorLifecycleState::Draining;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    release.set_value();
    task.get();
    shutdown_thread.join();

    TEST_ASSERT(observed_draining,
                "snapshot must expose Draining while shutdown waits for accepted work");
    return true;
}

bool test_snapshot_concurrent_registration_stop_and_query() {
    Executor executor;
    std::atomic<bool> keep_querying{true};
    std::thread reader([&executor, &keep_querying]() {
        while (keep_querying.load(std::memory_order_acquire)) {
            (void)executor.get_snapshot();
        }
    });

    std::thread registrar([&executor]() {
        for (int i = 0; i < 32; ++i) {
            RealtimeThreadConfig config;
            config.thread_name = "snapshot-race-" + std::to_string(i);
            config.cycle_period_ns = 1'000'000;
            (void)executor.register_realtime_task("snapshot_race_" + std::to_string(i), config);
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    executor.shutdown(false);
    registrar.join();
    keep_querying.store(false, std::memory_order_release);
    reader.join();

    TEST_ASSERT(executor.get_snapshot().lifecycle == ExecutorLifecycleState::Stopped,
                "snapshot must remain valid after concurrent registration and shutdown");
    return true;
}

bool test_snapshot_safe_with_concurrent_shared_ownership_destruction() {
    auto executor = std::make_shared<Executor>();
    std::weak_ptr<Executor> weak_executor = executor;
    std::atomic<bool> keep_querying{true};
    std::atomic<uint64_t> snapshots{0};
    std::thread reader([&weak_executor, &keep_querying, &snapshots]() {
        while (keep_querying.load(std::memory_order_acquire)) {
            if (auto current = weak_executor.lock()) {
                (void)current->get_snapshot();
                snapshots.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    executor.reset();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    keep_querying.store(false, std::memory_order_release);
    reader.join();

    TEST_ASSERT(snapshots.load(std::memory_order_relaxed) != 0,
                "reader must successfully collect snapshots before destruction");
    TEST_ASSERT(weak_executor.expired(),
                "executor must be destroyed after the reader releases its last snapshot owner");
    return true;
}

int main() {
    bool success = true;
    success &= test_snapshot_does_not_lazy_initialize();
    success &= test_snapshot_reports_async_work_and_shutdown();
    success &= test_snapshot_reports_failure_and_failed_initialization();
    success &= test_snapshot_includes_registered_backends();
    success &= test_monitor_includes_registered_gpu_backend();
    success &= test_snapshot_is_safe_during_shutdown();
    success &= test_snapshot_observes_draining();
    success &= test_snapshot_concurrent_registration_stop_and_query();
    success &= test_snapshot_safe_with_concurrent_shared_ownership_destruction();
    return success ? 0 : 1;
}
