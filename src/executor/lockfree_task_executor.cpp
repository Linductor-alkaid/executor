#include "executor/lockfree_task_executor.hpp"
#include "util/lockfree_queue.hpp"
#include "util/object_pool.hpp"
#include <chrono>

namespace executor {

LockFreeTaskExecutor::LockFreeTaskExecutor(size_t queue_capacity)
    : queue_(new util::LockFreeQueue<TaskWrapper*>(queue_capacity))
    , task_pool_(new util::ObjectPool<TaskWrapper>(queue_capacity)) {
}

LockFreeTaskExecutor::~LockFreeTaskExecutor() {
    stop();
    delete queue_;
    delete task_pool_;
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

    std::vector<TaskWrapper*> wrappers(count);
    for (size_t i = 0; i < count; ++i) {
        wrappers[i] = task_pool_->acquire();
        if (!wrappers[i]) {
            for (size_t j = 0; j < i; ++j) {
                task_pool_->release(wrappers[j]);
            }
            return false;
        }
        wrappers[i]->func = tasks[i];
    }

    if (!queue_->push_batch(wrappers.data(), count, pushed)) {
        for (size_t i = 0; i < count; ++i) {
            task_pool_->release(wrappers[i]);
        }
        return false;
    }

    if (pushed < count) {
        for (size_t i = pushed; i < count; ++i) {
            task_pool_->release(wrappers[i]);
        }
    }

    return true;
}

size_t LockFreeTaskExecutor::pending_count() const {
    return queue_->size();
}

uint64_t LockFreeTaskExecutor::processed_count() const {
    return processed_count_.load(std::memory_order_relaxed);
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
                }
                task_pool_->release(batch[i]);
            }
            processed_count_.fetch_add(popped, std::memory_order_relaxed);
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    // 处理剩余任务
    size_t popped;
    while ((popped = queue_->pop_batch(batch.data(), BATCH_SIZE)) > 0) {
        for (size_t i = 0; i < popped; ++i) {
            try {
                batch[i]->func();
            } catch (...) {
            }
            task_pool_->release(batch[i]);
        }
        processed_count_.fetch_add(popped, std::memory_order_relaxed);
    }
}

} // namespace executor
