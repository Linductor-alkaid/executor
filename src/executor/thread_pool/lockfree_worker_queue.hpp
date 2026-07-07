#pragma once

#include "../task/task.hpp"
#include "../util/lockfree_queue.hpp"
#include <atomic>
#include <utility>
#include <memory>
#include <mutex>
#include <vector>
#include <cstdint>
#include <exception>
#include <stdexcept>

namespace executor {

/**
 * @brief 无锁工作线程本地队列
 *
 * 使用 LockFreeQueue (MPSC) + unique_ptr 包装实现高性能 push/pop。
 * Task 包含 std::function 和 atomic，不是 trivially copyable，
 * 因此使用 uintptr_t 存储指针，绕过类型限制。
 *
 * LockFreeQueue 是 MPSC 队列，消费端没有 CAS 抢占语义。为避免 owner pop
 * 和 work-stealing steal 同时消费同一槽位，pop/steal 在本包装层用 mutex
 * 串行化；push 路径仍不加该锁。
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
        auto* task_ptr = new Task();
        move_task(*task_ptr, std::move(task));
        uintptr_t ptr = reinterpret_cast<uintptr_t>(task_ptr);
        if (!main_queue_.push(ptr)) {
            delete task_ptr;
            return false;
        }
        return true;
    }

    size_t push_batch(const Task* tasks, size_t n) {
        if (!tasks || n == 0) return 0;

        std::vector<uintptr_t> ptrs;
        ptrs.reserve(n);

        for (size_t i = 0; i < n; ++i) {
            Task* task_ptr = nullptr;
            try {
                task_ptr = new Task();
                copy_task(*task_ptr, tasks[i]);
            } catch (...) {
                if (task_ptr) {
                    delete task_ptr;
                }
                for (size_t k = 0; k < ptrs.size(); ++k) {
                    delete reinterpret_cast<Task*>(ptrs[k]);
                }
                ptrs.clear();
                throw;
            }
            ptrs.push_back(reinterpret_cast<uintptr_t>(task_ptr));
        }

        size_t pushed = 0;
        const bool ok = main_queue_.push_batch(ptrs.data(), n, pushed);
        (void)ok;

        // 清理未推入的任务；底层部分成功时也可能返回 true 且 pushed < n。
        for (size_t i = pushed; i < n; ++i) {
            delete reinterpret_cast<Task*>(ptrs[i]);
        }
        return pushed;
    }

    bool pop(Task& task) {
        std::lock_guard<std::mutex> lock(consume_mx_);

        if (!steal_buffer_.empty()) {
            uintptr_t ptr = steal_buffer_.back();
            steal_buffer_.pop_back();
            std::unique_ptr<Task> task_ptr(reinterpret_cast<Task*>(ptr));
            move_task(task, std::move(*task_ptr));
            return true;
        }

        uintptr_t ptr;
        if (!main_queue_.pop(ptr)) {
            return false;
        }
        std::unique_ptr<Task> task_ptr(reinterpret_cast<Task*>(ptr));
        move_task(task, std::move(*task_ptr));
        return true;
    }

    bool steal(Task& task) {
        std::lock_guard<std::mutex> lock(consume_mx_);

        if (steal_buffer_.empty()) {
            constexpr size_t STEAL_BATCH = 16;
            uintptr_t ptrs[STEAL_BATCH];
            size_t stolen = main_queue_.pop_batch(ptrs, STEAL_BATCH);

            if (stolen == 0) {
                return false;
            }

            for (size_t i = stolen; i > 0; --i) {
                steal_buffer_.push_back(ptrs[i - 1]);
            }
        }

        uintptr_t ptr = steal_buffer_.back();
        steal_buffer_.pop_back();
        std::unique_ptr<Task> task_ptr(reinterpret_cast<Task*>(ptr));
        move_task(task, std::move(*task_ptr));
        return true;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(consume_mx_);
        return main_queue_.size() + steal_buffer_.size();
    }

    bool empty() const {
        return size() == 0;
    }

    void clear() {
        std::vector<std::unique_ptr<Task>> discarded;

        {
            std::lock_guard<std::mutex> lock(consume_mx_);

            uintptr_t ptr;
            while (main_queue_.pop(ptr)) {
                discarded.emplace_back(reinterpret_cast<Task*>(ptr));
            }
            for (auto p : steal_buffer_) {
                discarded.emplace_back(reinterpret_cast<Task*>(p));
            }
            steal_buffer_.clear();
        }

        for (auto& task : discarded) {
            discard_task(*task);
        }
    }

    /**
     * @brief 析构函数
     */
    ~LockFreeWorkerQueue() {
        clear();
    }

private:
    static void copy_task(Task& dst, const Task& src) {
        dst.task_id = src.task_id;
        dst.priority = src.priority;
        dst.function = src.function;
        dst.on_timeout = src.on_timeout;
        dst.submit_time_ns = src.submit_time_ns;
        dst.timeout_ms = src.timeout_ms;
        dst.dependencies = src.dependencies;
        dst.cancelled.store(src.cancelled.load(std::memory_order_acquire),
                           std::memory_order_release);
    }

    static void move_task(Task& dst, Task&& src) {
        dst.task_id = std::move(src.task_id);
        dst.priority = src.priority;
        dst.function = std::move(src.function);
        dst.on_timeout = std::move(src.on_timeout);
        dst.submit_time_ns = src.submit_time_ns;
        dst.timeout_ms = src.timeout_ms;
        dst.dependencies = std::move(src.dependencies);
        dst.cancelled.store(src.cancelled.load(std::memory_order_acquire),
                           std::memory_order_release);
    }

    static void discard_task(Task& task) noexcept {
        if (!task.on_timeout) {
            return;
        }

        try {
            task.on_timeout(std::make_exception_ptr(std::runtime_error(
                "Task discarded before execution")));
        } catch (...) {
        }
    }

    util::LockFreeQueue<uintptr_t> main_queue_;

    mutable std::mutex consume_mx_;
    std::vector<uintptr_t> steal_buffer_;
};

} // namespace executor
