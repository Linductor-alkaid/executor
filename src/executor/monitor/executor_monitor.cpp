#include "executor/monitor/executor_monitor.hpp"

#include "executor/executor_manager.hpp"

namespace executor {
namespace monitor {
namespace {

void mark_partial(ExecutorSnapshot& snapshot, const char* provider) {
    snapshot.partial = true;
    if (!snapshot.consistency_note.empty()) {
        snapshot.consistency_note += "; ";
    }
    snapshot.consistency_note += provider;
}

} // namespace

ExecutorMonitor::ExecutorMonitor(
    const ExecutorManager& manager,
    const std::atomic<ExecutorLifecycleState>& lifecycle,
    CompletionProvider completion_provider,
    FailureStatusProvider failure_status_provider,
    RecentFailuresProvider recent_failures_provider,
    TaskStatisticsProvider task_statistics_provider)
    : manager_(manager)
    , lifecycle_(lifecycle)
    , completion_provider_(std::move(completion_provider))
    , failure_status_provider_(std::move(failure_status_provider))
    , recent_failures_provider_(std::move(recent_failures_provider))
    , task_statistics_provider_(std::move(task_statistics_provider)) {
}

ExecutorSnapshot ExecutorMonitor::collect() const {
    ExecutorSnapshot snapshot;
    snapshot.captured_at = std::chrono::steady_clock::now();
    snapshot.lifecycle = lifecycle_.load(std::memory_order_acquire);

    try {
        snapshot.completion = completion_provider_();
        snapshot.async.name = snapshot.completion.executor_name;
        snapshot.async.is_running = snapshot.completion.is_running;
        snapshot.async.active_tasks = snapshot.completion.active_tasks;
        snapshot.async.queue_size = snapshot.completion.queued_tasks;
        snapshot.async.completed_tasks = snapshot.completion.completed_tasks;
        snapshot.async.failed_tasks = snapshot.completion.failed_tasks;
    } catch (...) {
        mark_partial(snapshot, "completion");
    }

    try {
        snapshot.realtime = manager_.get_all_realtime_executor_statuses();
    } catch (...) {
        mark_partial(snapshot, "realtime");
    }
    try {
        snapshot.blocking_io = manager_.get_all_blocking_io_executor_statuses();
    } catch (...) {
        mark_partial(snapshot, "blocking_io");
    }
    try {
        snapshot.gpu = manager_.get_all_gpu_executor_statuses();
    } catch (...) {
        mark_partial(snapshot, "gpu");
    }
    try {
        snapshot.failures = failure_status_provider_();
    } catch (...) {
        mark_partial(snapshot, "failures");
    }
    try {
        snapshot.recent_failures = recent_failures_provider_();
    } catch (...) {
        mark_partial(snapshot, "recent_failures");
    }
    try {
        snapshot.task_statistics = task_statistics_provider_();
    } catch (...) {
        mark_partial(snapshot, "task_statistics");
    }

    snapshot.running_backend_count = snapshot.async.is_running ? 1 : 0;
    snapshot.stopping_backend_count = 0;
    snapshot.active_task_count = snapshot.async.active_tasks;
    snapshot.queued_task_count = snapshot.async.queue_size;
    snapshot.failed_task_count = snapshot.async.failed_tasks;

    for (const auto& [name, status] : snapshot.realtime) {
        (void)name;
        snapshot.running_backend_count += status.is_running ? 1 : 0;
        snapshot.dropped_work_count += status.dropped_task_count;
    }
    for (const auto& [name, status] : snapshot.blocking_io) {
        (void)name;
        snapshot.running_backend_count += status.is_running ? 1 : 0;
        snapshot.stopping_backend_count += status.stop_requested ? 1 : 0;
    }
    for (const auto& [name, status] : snapshot.gpu) {
        (void)name;
        snapshot.running_backend_count += status.is_running ? 1 : 0;
        snapshot.active_task_count += status.active_kernels;
        snapshot.queued_task_count += status.queue_size;
        snapshot.failed_task_count += status.failed_kernels;
    }

    if (snapshot.lifecycle == ExecutorLifecycleState::Created &&
        snapshot.running_backend_count != 0) {
        snapshot.lifecycle = ExecutorLifecycleState::Running;
    }
    snapshot.snapshot_sequence = next_sequence_.fetch_add(1, std::memory_order_relaxed) + 1;
    snapshot.collection_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - snapshot.captured_at);
    return snapshot;
}

} // namespace monitor
} // namespace executor
