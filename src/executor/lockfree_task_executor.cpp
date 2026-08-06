#include "executor/lockfree_task_executor.hpp"
#include "util/lockfree_queue.hpp"
#include "util/object_pool.hpp"
#include <chrono>
#include <thread>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  if defined(_MSC_VER)
#    include <intrin.h>
#  else
#    include <immintrin.h>
#  endif
#  define EXECUTOR_PAUSE() _mm_pause()
#else
#  define EXECUTOR_PAUSE() std::this_thread::yield()
#endif

namespace executor {

LockFreeTaskExecutor::LockFreeTaskExecutor(size_t queue_capacity, size_t backoff_multiplier, bool enable_stats)
    : queue_(std::make_unique<util::LockFreeQueue<TaskWrapper*>>(queue_capacity, backoff_multiplier, enable_stats))
    , task_pool_(std::make_unique<util::ObjectPool<TaskWrapper>>(queue_capacity))
    , task_pool_capacity_(queue_capacity) {
}

LockFreeTaskExecutor::~LockFreeTaskExecutor() {
    stop();
}

bool LockFreeTaskExecutor::start() {
    std::lock_guard<std::mutex> lock(stop_mutex_);
    if (stopped_.load(std::memory_order_acquire)) {
        return false;
    }

    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return false;
    }

    try {
        worker_ = create_worker_thread();
    } catch (...) {
        running_.store(false, std::memory_order_release);
        worker_id_ = std::thread::id{};
        return false;
    }
    return true;
}

std::thread LockFreeTaskExecutor::create_worker_thread() {
    return std::thread(&LockFreeTaskExecutor::worker_thread, this);
}

void LockFreeTaskExecutor::stop() {
    (void)stop_and_join();
}

bool LockFreeTaskExecutor::stop_and_join() {
    std::thread joiner;
    {
        std::lock_guard<std::mutex> lock(stop_mutex_);
        stopped_.store(true, std::memory_order_release);
        if (std::this_thread::get_id() == worker_id_) {
            self_stop_requested_.store(true, std::memory_order_release);
            running_.store(false, std::memory_order_release);
            return false;
        }

        while (active_pushes_.load(std::memory_order_acquire) != 0) {
            std::this_thread::yield();
        }
        running_.store(false, std::memory_order_release);
        if (worker_.joinable()) {
            joiner = std::move(worker_);
        }
    }

    if (joiner.joinable()) {
        joiner.join();
    }
    return true;
}

bool LockFreeTaskExecutor::is_running() const {
    return running_.load(std::memory_order_acquire);
}

bool LockFreeTaskExecutor::push_task(std::function<void()> task) {
    if (!task) {
        rejected_empty_count_.fetch_add(1, std::memory_order_relaxed);
        submission_rejection_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    if (!enter_push()) {
        submission_rejection_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    auto* wrapper = task_pool_->acquire();
    if (!wrapper) {
        submission_rejection_.fetch_add(1, std::memory_order_relaxed);
        leave_push();
        return false;
    }

    wrapper->func = std::move(task);

    if (!queue_->push(wrapper)) {
        task_pool_->release(wrapper);
        leave_push();
        return false;
    }

    leave_push();
    return true;
}

bool LockFreeTaskExecutor::push_tasks_batch(const std::function<void()>* tasks, size_t count, size_t& pushed) {
    pushed = 0;
    if (!tasks) {
        rejected_empty_count_.fetch_add(1, std::memory_order_relaxed);
        submission_rejection_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (count == 0) {
        return true;
    }
    // The bounded ring always reserves one slot, so no batch this large can
    // succeed. Reject it before scanning caller memory or allocating the
    // temporary pointer array.
    if (count > task_pool_capacity_ || count >= queue_->capacity()) {
        submission_rejection_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        if (!tasks[i]) {
            rejected_empty_count_.fetch_add(1, std::memory_order_relaxed);
            submission_rejection_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
    }

    if (!enter_push()) {
        submission_rejection_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    struct PushGuard {
        LockFreeTaskExecutor& executor;

        ~PushGuard() {
            executor.leave_push();
        }
    } push_guard{*this};

    // P-260623-004: keep batch monitoring meaningful by bulk-acquiring
    // wrappers, populating them, then dispatching the whole array in one exact
    // batch call so the queue records a single batch_pushes++ and a single CAS
    // reservation, instead of N independent push() calls.
    std::vector<TaskWrapper*> ptrs;
    size_t acquired = 0;

    try {
        if (before_batch_allocation_hook_) {
            before_batch_allocation_hook_(before_batch_allocation_context_);
        }
        ptrs.assign(count, nullptr);

        // 1) Bulk-acquire wrappers. If the pool cannot hand out `count` in one
        //    pass we must report a hard failure (matches the previous behaviour
        //    where the first acquire() returning null aborted the whole batch).
        for (size_t i = 0; i < count; ++i) {
            auto* wrapper = task_pool_->acquire();
            if (!wrapper) {
                submission_rejection_.fetch_add(1, std::memory_order_relaxed);
                for (size_t j = 0; j < acquired; ++j) {
                    task_pool_->release(ptrs[j]);
                }
                return false;
            }
            ptrs[i] = wrapper;
            ++acquired;
        }

        // 2) Populate wrappers. An exception while copying a std::function must
        //    release every acquired wrapper so the pool does not leak. We do not
        //    have to undo any queue mutation because exact enqueue happens after.
        for (size_t i = 0; i < count; ++i) {
            ptrs[i]->func = tasks[i];
        }

        // 3) Single exact batched enqueue. LockFreeTaskExecutor exposes atomic
        //    batch semantics: either all wrappers are handed to the queue, or none
        //    are. The queue-level exact helper refuses to reserve a smaller prefix.
        bool ok = queue_->push_batch_exact(ptrs.data(), count);
        if (ok) {
            pushed = count;
        } else {
            // An unsuccessful exact batch is completely non-observable, so
            // every acquired wrapper remains ours to return to the pool.
            for (size_t i = 0; i < count; ++i) {
                task_pool_->release(ptrs[i]);
            }
        }

        return ok;
    } catch (...) {
        for (size_t i = 0; i < acquired; ++i) {
            task_pool_->release(ptrs[i]);
        }
        submission_rejection_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
}

size_t LockFreeTaskExecutor::pending_count() const {
    return queue_->size();
}

uint64_t LockFreeTaskExecutor::processed_count() const {
    return processed_count_.load(std::memory_order_relaxed);
}

uint64_t LockFreeTaskExecutor::exception_count() const {
    return exception_count_.load(std::memory_order_relaxed);
}

uint64_t LockFreeTaskExecutor::rejected_empty_count() const {
    return rejected_empty_count_.load(std::memory_order_relaxed);
}

void LockFreeTaskExecutor::set_exception_handler(std::function<void(std::exception_ptr)> handler) {
    std::lock_guard<std::mutex> lock(exception_handler_mutex_);
    exception_handler_ = std::move(handler);
}

LockFreeTaskExecutor::QueueStats LockFreeTaskExecutor::get_queue_stats() const {
    return make_queue_stats(queue_->get_stats());
}

LockFreeTaskExecutor::QueueStats LockFreeTaskExecutor::expensive_diagnostic_snapshot() const {
    return make_queue_stats(queue_->expensive_diagnostic_snapshot());
}

LockFreeTaskExecutor::QueueStats LockFreeTaskExecutor::make_queue_stats(
    const util::LockFreeQueueStats& raw) const {
    QueueStats result;
    result.total_pushes = raw.total_pushes;
    result.failed_pushes = raw.failed_pushes;
    result.queue_full_rejections = raw.queue_full_rejections;
    result.total_pops = raw.total_pops;
    result.empty_pops = raw.empty_pops;
    result.batch_pushes = raw.batch_pushes;
    result.batch_pops = raw.batch_pops;
    result.current_size = raw.current_size;
    result.peak_size = raw.peak_size;
    result.queue_capacity = queue_->capacity();
    result.reserved_count = raw.reserved_count;
    result.reservation_count = raw.reservation_count;
    result.ready_count = raw.ready_count;
    result.contention_rejection = raw.contention_rejection;
    result.reservation_cancelled_rejections = raw.reservation_cancelled_rejections;
    result.cancelled_reservation_count = raw.cancelled_reservation_count;
    result.submission_rejection = submission_rejection_.load(std::memory_order_relaxed);
    result.reservation_wait_yields = raw.reservation_wait_yields;
    switch (raw.fail_reason) {
    case util::LockFreeQueueFailReason::QueueFull:
        result.fail_reason = QueueFailReason::QueueFull;
        break;
    case util::LockFreeQueueFailReason::Contention:
        result.fail_reason = QueueFailReason::Contention;
        break;
    case util::LockFreeQueueFailReason::ReservationCancelled:
        result.fail_reason = QueueFailReason::ReservationCancelled;
        break;
    case util::LockFreeQueueFailReason::None:
    default:
        result.fail_reason = QueueFailReason::None;
        break;
    }
    // P-260618-006: expose the exception count alongside the existing queue
    // stats so monitoring code can correlate exceptions with queue state.
    result.exception_count = exception_count_.load(std::memory_order_relaxed);
    result.rejected_empty_count = rejected_empty_count_.load(std::memory_order_relaxed);
    const double total_attempts =
        static_cast<double>(raw.total_pushes) + static_cast<double>(raw.failed_pushes);
    result.success_rate = total_attempts > 0.0
        ? static_cast<double>(raw.total_pushes) / total_attempts
        : 0.0;
    return result;
}

LockFreeTaskExecutor::QueueStats LockFreeTaskExecutor::get_status_snapshot() const {
    return get_queue_stats();
}

void LockFreeTaskExecutor::set_before_publish_hook(BeforePublishHook hook, void* context) {
    queue_->set_before_publish_hook(hook, context);
}

void LockFreeTaskExecutor::set_before_batch_allocation_hook_for_test(
    BeforeBatchAllocationHook hook, void* context) {
    before_batch_allocation_context_ = context;
    before_batch_allocation_hook_ = hook;
}

bool LockFreeTaskExecutor::enter_push() {
    if (stopped_.load(std::memory_order_acquire)) {
        return false;
    }

    active_pushes_.fetch_add(1, std::memory_order_acq_rel);
    if (stopped_.load(std::memory_order_acquire)) {
        leave_push();
        return false;
    }

    return true;
}

void LockFreeTaskExecutor::leave_push() {
    active_pushes_.fetch_sub(1, std::memory_order_acq_rel);
}

void LockFreeTaskExecutor::worker_thread() {
    {
        std::lock_guard<std::mutex> lock(stop_mutex_);
        worker_id_ = std::this_thread::get_id();
    }

    constexpr size_t BATCH_SIZE = 32;
    std::vector<TaskWrapper*> batch(BATCH_SIZE);

    while (true) {
        size_t popped = queue_->pop_batch(batch.data(), BATCH_SIZE);

        if (popped > 0) {
            bool self_stop_interrupted_batch = false;
            for (size_t i = 0; i < popped; ++i) {
                try {
                    batch[i]->func();
                } catch (...) {
                    // P-260618-006: surface task exceptions via the
                    // exception_count counter and (optionally) a registered
                    // handler. Default behavior is "count only" — no rethrow,
                    // no crash — preserving back-compat.
                    exception_count_.fetch_add(1, std::memory_order_relaxed);
                    std::function<void(std::exception_ptr)> handler;
                    {
                        std::lock_guard<std::mutex> lock(exception_handler_mutex_);
                        handler = exception_handler_;
                    }
                    if (handler) {
                        try {
                            handler(std::current_exception());
                        } catch (...) {
                            // Swallow exceptions from the handler itself;
                            // the worker must keep draining the queue.
                        }
                    }
                }
                task_pool_->release(batch[i]);
                if (!running_.load(std::memory_order_acquire) &&
                    self_stop_requested_.load(std::memory_order_acquire)) {
                    for (size_t remaining = i + 1; remaining < popped; ++remaining) {
                        task_pool_->release(batch[remaining]);
                    }
                    processed_count_.fetch_add(i + 1, std::memory_order_relaxed);
                    self_stop_interrupted_batch = true;
                    break;
                }
            }
            if (!self_stop_interrupted_batch) {
                processed_count_.fetch_add(popped, std::memory_order_relaxed);
            }
        } else {
            if (!running_.load(std::memory_order_acquire)) {
                break;
            }

            // Hybrid backoff: PAUSE spin → yield → 1µs sleep
            static constexpr uint32_t kPauseSpins  = 32;
            static constexpr uint32_t kSleepThresh = 1000;
            idle_count_++;
            if (idle_count_ <= kPauseSpins) {
                EXECUTOR_PAUSE();
            } else if (idle_count_ <= kSleepThresh) {
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
            continue;
        }
        idle_count_ = 0;
    }

    if (self_stop_requested_.load(std::memory_order_acquire)) {
        return;
    }

    // 处理剩余任务
    size_t popped;
    while ((popped = queue_->pop_batch(batch.data(), BATCH_SIZE)) > 0) {
        for (size_t i = 0; i < popped; ++i) {
            try {
                batch[i]->func();
            } catch (...) {
                // P-260618-006: same handling as in the running loop.
                exception_count_.fetch_add(1, std::memory_order_relaxed);
                std::function<void(std::exception_ptr)> handler;
                {
                    std::lock_guard<std::mutex> lock(exception_handler_mutex_);
                    handler = exception_handler_;
                }
                if (handler) {
                    try {
                        handler(std::current_exception());
                    } catch (...) {
                    }
                }
            }
            task_pool_->release(batch[i]);
        }
        processed_count_.fetch_add(popped, std::memory_order_relaxed);
    }
}

} // namespace executor
