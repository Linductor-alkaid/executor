#pragma once

#include "../task/task.hpp"
#include "../util/lockfree_queue.hpp"
#include <atomic>
#include <mutex>
#include <vector>
#include <cstdint>

namespace executor {

/**
 * @brief 无锁工作线程本地队列
 *
 * 使用 LockFreeQueue (MPSC) + unique_ptr 包装实现高性能 push/pop。
 * Task 包含 std::function 和 atomic，不是 trivially copyable，
 * 因此使用 uintptr_t 存储指针，绕过类型限制。
 */
class LockFreeWorkerQueue {
public:
    explicit LockFreeWorkerQueue(size_t capacity = 1024)
        : main_queue_(capacity > 0 ? capacity : 1024) {}

    bool push(const Task& task) {
        auto* task_ptr = new Task();
        copy_task(*task_ptr, task);
        uintptr_t ptr = reinterpret_cast<uintptr_t>(task_ptr);
        if (!main_queue_.push(ptr)) {
            delete task_ptr;
            return false;
        }
        return true;
    }

    bool push(Task&& task) {
        return push(task);
    }

    size_t push_batch(const Task* tasks, size_t n) {
        if (!tasks || n == 0) return 0;

        std::vector<uintptr_t> ptrs;
        ptrs.reserve(n);

        for (size_t i = 0; i < n; ++i) {
            auto* task_ptr = new Task();
            copy_task(*task_ptr, tasks[i]);
            ptrs.push_back(reinterpret_cast<uintptr_t>(task_ptr));
        }

        size_t pushed = 0;
        if (!main_queue_.push_batch(ptrs.data(), n, pushed)) {
            // 清理未推入的任务
            for (size_t i = pushed; i < n; ++i) {
                delete reinterpret_cast<Task*>(ptrs[i]);
            }
        }
        return pushed;
    }

    bool pop(Task& task) {
        uintptr_t ptr;
        if (!main_queue_.pop(ptr)) {
            return false;
        }
        auto* task_ptr = reinterpret_cast<Task*>(ptr);
        copy_task(task, *task_ptr);
        delete task_ptr;
        return true;
    }

    bool steal(Task& task) {
        std::lock_guard<std::mutex> lock(steal_mutex_);

        if (steal_buffer_.empty()) {
            constexpr size_t STEAL_BATCH = 16;
            uintptr_t ptrs[STEAL_BATCH];
            size_t stolen = main_queue_.pop_batch(ptrs, STEAL_BATCH);

            if (stolen == 0) {
                return false;
            }

            for (size_t i = 0; i < stolen; ++i) {
                steal_buffer_.push_back(ptrs[i]);
            }
        }

        uintptr_t ptr = steal_buffer_.back();
        steal_buffer_.pop_back();
        auto* task_ptr = reinterpret_cast<Task*>(ptr);
        copy_task(task, *task_ptr);
        delete task_ptr;
        return true;
    }

    size_t size() const {
        return main_queue_.size();
    }

    bool empty() const {
        return main_queue_.empty();
    }

    void clear() {
        uintptr_t ptr;
        while (main_queue_.pop(ptr)) {
            delete reinterpret_cast<Task*>(ptr);
        }
        std::lock_guard<std::mutex> lock(steal_mutex_);
        for (auto p : steal_buffer_) {
            delete reinterpret_cast<Task*>(p);
        }
        steal_buffer_.clear();
    }

    ~LockFreeWorkerQueue() {
        clear();
    }

private:
    static void copy_task(Task& dst, const Task& src) {
        dst.task_id = src.task_id;
        dst.priority = src.priority;
        dst.function = src.function;
        dst.submit_time_ns = src.submit_time_ns;
        dst.timeout_ms = src.timeout_ms;
        dst.dependencies = src.dependencies;
        dst.cancelled.store(src.cancelled.load(std::memory_order_acquire),
                           std::memory_order_release);
    }

    util::LockFreeQueue<uintptr_t> main_queue_;

    mutable std::mutex steal_mutex_;
    std::vector<uintptr_t> steal_buffer_;
};

} // namespace executor
