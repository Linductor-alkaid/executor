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

size_t LockFreeTaskExecutor::pending_count() const {
    return queue_->size();
}

uint64_t LockFreeTaskExecutor::processed_count() const {
    return processed_count_.load(std::memory_order_relaxed);
}

void LockFreeTaskExecutor::worker_thread() {
    while (running_.load(std::memory_order_acquire)) {
        TaskWrapper* wrapper = nullptr;

        if (queue_->pop(wrapper)) {
            try {
                wrapper->func();
            } catch (...) {
                // 忽略异常，继续处理
            }

            task_pool_->release(wrapper);
            processed_count_.fetch_add(1, std::memory_order_relaxed);
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    // 处理剩余任务
    TaskWrapper* wrapper = nullptr;
    while (queue_->pop(wrapper)) {
        try {
            wrapper->func();
        } catch (...) {
        }
        task_pool_->release(wrapper);
        processed_count_.fetch_add(1, std::memory_order_relaxed);
    }
}

} // namespace executor
