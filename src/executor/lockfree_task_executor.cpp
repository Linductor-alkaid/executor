#include "executor/lockfree_task_executor.hpp"
#include "util/lockfree_queue.hpp"
#include "util/object_pool.hpp"
#include <chrono>
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
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return false;
    }

    worker_ = std::thread(&LockFreeTaskExecutor::worker_thread, this);
    return true;
}

void LockFreeTaskExecutor::stop() {
    if (!running_.exchange(false)) {
        return;
    }

    if (worker_.joinable()) {
        worker_.join();
    }
}

bool LockFreeTaskExecutor::is_running() const {
    return running_.load(std::memory_order_acquire);
}

bool LockFreeTaskExecutor::push_task(std::function<void()> task) {
    auto* wrapper = task_pool_->acquire();
    if (!wrapper) {
        return false;
    }

    wrapper->func = std::move(task);

    if (!queue_->push(wrapper)) {
        task_pool_->release(wrapper);
        return false;
    }

    return true;
}

bool LockFreeTaskExecutor::push_tasks_batch(const std::function<void()>* tasks, size_t count, size_t& pushed) {
    pushed = 0;
    if (count == 0) return true;

    for (size_t i = 0; i < count; ++i) {
        auto* wrapper = task_pool_->acquire();
        if (!wrapper) {
            return pushed > 0;
        }

        // func assignment before push: exception here releases this wrapper and returns
        try {
            wrapper->func = tasks[i];
        } catch (...) {
            task_pool_->release(wrapper);
            return pushed > 0;
        }

        if (!queue_->push(wrapper)) {
            task_pool_->release(wrapper);
            return pushed > 0;
        }

        ++pushed;
    }

    return true;
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

void LockFreeTaskExecutor::set_exception_handler(std::function<void(std::exception_ptr)> handler) {
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
    result.success_rate = raw.total_pushes > 0
        ? static_cast<double>(raw.total_pushes - raw.failed_pushes) / static_cast<double>(raw.total_pushes)
        : 0.0;
    return result;
}

void LockFreeTaskExecutor::worker_thread() {
    constexpr size_t BATCH_SIZE = 32;
    std::vector<TaskWrapper*> batch(BATCH_SIZE);

    while (running_.load(std::memory_order_acquire)) {
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
                    if (exception_handler_) {
                        try {
                            exception_handler_(std::current_exception());
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
                if (exception_handler_) {
                    try {
                        exception_handler_(std::current_exception());
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
