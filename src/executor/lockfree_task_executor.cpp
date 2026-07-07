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
    , task_pool_(std::make_unique<util::ObjectPool<TaskWrapper>>(queue_capacity)) {
}

LockFreeTaskExecutor::~LockFreeTaskExecutor() {
    stop();
}

bool LockFreeTaskExecutor::start() {
    if (stopped_.load(std::memory_order_acquire)) {
        return false;
    }

    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return false;
    }

    worker_ = std::thread(&LockFreeTaskExecutor::worker_thread, this);
    return true;
}

void LockFreeTaskExecutor::stop() {
    stopped_.store(true, std::memory_order_release);
    while (active_pushes_.load(std::memory_order_acquire) != 0) {
        std::this_thread::yield();
    }

    if (running_.exchange(false, std::memory_order_acq_rel) && worker_.joinable()) {
        worker_.join();
    }
}

bool LockFreeTaskExecutor::is_running() const {
    return running_.load(std::memory_order_acquire);
}

bool LockFreeTaskExecutor::push_task(std::function<void()> task) {
    if (!task) {
        rejected_empty_count_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    if (!enter_push()) {
        return false;
    }

    auto* wrapper = task_pool_->acquire();
    if (!wrapper) {
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
        return false;
    }
    if (count == 0) {
        return true;
    }
    for (size_t i = 0; i < count; ++i) {
        if (!tasks[i]) {
            rejected_empty_count_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
    }

    if (!enter_push()) {
        return false;
    }

    // P-260623-004: keep batch monitoring meaningful by bulk-acquiring
    // wrappers, populating them, then dispatching the whole array in one exact
    // batch call so the queue records a single batch_pushes++ and a single CAS
    // reservation, instead of N independent push() calls.
    std::vector<TaskWrapper*> ptrs(count, nullptr);
    size_t acquired = 0;

    // 1) Bulk-acquire wrappers. If the pool cannot hand out `count` in one
    //    pass we must report a hard failure (matches the previous behaviour
    //    where the first acquire() returning null aborted the whole batch).
    for (size_t i = 0; i < count; ++i) {
        auto* wrapper = task_pool_->acquire();
        if (!wrapper) {
            for (size_t j = 0; j < acquired; ++j) {
                task_pool_->release(ptrs[j]);
            }
            leave_push();
            return false;
        }
        ptrs[i] = wrapper;
        ++acquired;
    }

    // 2) Populate wrappers. An exception while copying a std::function must
    //    release every acquired wrapper so the pool does not leak. We do not
    //    have to undo any queue mutation because exact enqueue happens after.
    for (size_t i = 0; i < count; ++i) {
        try {
            ptrs[i]->func = tasks[i];
        } catch (...) {
            for (size_t j = 0; j < count; ++j) {
                task_pool_->release(ptrs[j]);
            }
            // pushed stays 0; nothing reached the queue.
            leave_push();
            return false;
        }
    }

    // 3) Single exact batched enqueue. LockFreeTaskExecutor exposes atomic
    //    batch semantics: either all wrappers are handed to the queue, or none
    //    are. The queue-level exact helper refuses to reserve a smaller prefix.
    bool ok = queue_->push_batch_exact(ptrs.data(), count);
    if (ok) {
        pushed = count;
    } else {
        for (size_t i = 0; i < count; ++i) {
            task_pool_->release(ptrs[i]);
        }
    }

    leave_push();
    return ok;
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
    auto raw = queue_->get_stats();
    QueueStats result;
    result.total_pushes = raw.total_pushes;
    result.failed_pushes = raw.failed_pushes;
    result.total_pops = raw.total_pops;
    result.empty_pops = raw.empty_pops;
    result.batch_pushes = raw.batch_pushes;
    result.batch_pops = raw.batch_pops;
    result.current_size = raw.current_size;
    result.peak_size = raw.peak_size;
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
    constexpr size_t BATCH_SIZE = 32;
    std::vector<TaskWrapper*> batch(BATCH_SIZE);

    while (true) {
        size_t popped = queue_->pop_batch(batch.data(), BATCH_SIZE);

        if (popped > 0) {
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
            }
            processed_count_.fetch_add(popped, std::memory_order_relaxed);
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
