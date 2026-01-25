#include "priority_scheduler.hpp"
#include <algorithm>
#include <memory>
#include <mutex>

namespace executor {

void PriorityScheduler::copy_task_out(const std::unique_ptr<Task>& src, Task& out) {
    if (!src) return;
    out.task_id = src->task_id;
    out.priority = src->priority;
    out.function = src->function;
    out.submit_time_ns = src->submit_time_ns;
    out.timeout_ms = src->timeout_ms;
    out.dependencies = src->dependencies;
    out.cancelled.store(src->cancelled.load(std::memory_order_acquire), std::memory_order_release);
}

void PriorityScheduler::enqueue(const Task& task) {
    // 创建unique_ptr<Task>（复制字段）
    auto task_ptr = std::make_unique<Task>();
    task_ptr->task_id = task.task_id;
    task_ptr->priority = task.priority;
    task_ptr->function = task.function;
    task_ptr->submit_time_ns = task.submit_time_ns;
    task_ptr->timeout_ms = task.timeout_ms;
    task_ptr->dependencies = task.dependencies;
    task_ptr->cancelled.store(task.cancelled.load(std::memory_order_acquire), std::memory_order_release);

    TaskPtrCompare cmp;

    switch (task.priority) {
        case TaskPriority::CRITICAL: {
            std::lock_guard<std::mutex> lock(critical_mutex_);
            critical_queue_.push_back(std::move(task_ptr));
            std::push_heap(critical_queue_.begin(), critical_queue_.end(), cmp);
            break;
        }
        case TaskPriority::HIGH: {
            std::lock_guard<std::mutex> lock(high_mutex_);
            high_queue_.push_back(std::move(task_ptr));
            std::push_heap(high_queue_.begin(), high_queue_.end(), cmp);
            break;
        }
        case TaskPriority::NORMAL: {
            std::lock_guard<std::mutex> lock(normal_mutex_);
            normal_queue_.push_back(std::move(task_ptr));
            std::push_heap(normal_queue_.begin(), normal_queue_.end(), cmp);
            break;
        }
        case TaskPriority::LOW: {
            std::lock_guard<std::mutex> lock(low_mutex_);
            low_queue_.push_back(std::move(task_ptr));
            std::push_heap(low_queue_.begin(), low_queue_.end(), cmp);
            break;
        }
        default:
            break;
    }
}

bool PriorityScheduler::dequeue(Task& task) {
    TaskPtrCompare cmp;
    std::unique_ptr<Task> task_ptr;

    {
        std::lock_guard<std::mutex> lock(critical_mutex_);
        if (!critical_queue_.empty()) {
            std::pop_heap(critical_queue_.begin(), critical_queue_.end(), cmp);
            task_ptr = std::move(critical_queue_.back());
            critical_queue_.pop_back();
        }
    }
    if (task_ptr) {
        copy_task_out(task_ptr, task);
        return true;
    }

    {
        std::lock_guard<std::mutex> lock(high_mutex_);
        if (!high_queue_.empty()) {
            std::pop_heap(high_queue_.begin(), high_queue_.end(), cmp);
            task_ptr = std::move(high_queue_.back());
            high_queue_.pop_back();
        }
    }
    if (task_ptr) {
        copy_task_out(task_ptr, task);
        return true;
    }

    {
        std::lock_guard<std::mutex> lock(normal_mutex_);
        if (!normal_queue_.empty()) {
            std::pop_heap(normal_queue_.begin(), normal_queue_.end(), cmp);
            task_ptr = std::move(normal_queue_.back());
            normal_queue_.pop_back();
        }
    }
    if (task_ptr) {
        copy_task_out(task_ptr, task);
        return true;
    }

    {
        std::lock_guard<std::mutex> lock(low_mutex_);
        if (!low_queue_.empty()) {
            std::pop_heap(low_queue_.begin(), low_queue_.end(), cmp);
            task_ptr = std::move(low_queue_.back());
            low_queue_.pop_back();
        }
    }
    if (task_ptr) {
        copy_task_out(task_ptr, task);
        return true;
    }

    return false;
}

size_t PriorityScheduler::dequeue_batch(Task* out, size_t max_tasks) {
    if (max_tasks == 0 || !out) return 0;
    TaskPtrCompare cmp;
    size_t count = 0;

    auto drain = [&](TaskQueue& queue, std::mutex& m) {
        std::lock_guard<std::mutex> lock(m);
        while (!queue.empty() && count < max_tasks) {
            std::pop_heap(queue.begin(), queue.end(), cmp);
            std::unique_ptr<Task> ptr = std::move(queue.back());
            queue.pop_back();
            copy_task_out(ptr, out[count]);
            ++count;
        }
    };

    drain(critical_queue_, critical_mutex_);
    drain(high_queue_, high_mutex_);
    drain(normal_queue_, normal_mutex_);
    drain(low_queue_, low_mutex_);

    return count;
}

size_t PriorityScheduler::size() const {
    std::scoped_lock lock(critical_mutex_, high_mutex_, normal_mutex_, low_mutex_);
    return critical_queue_.size() + high_queue_.size() +
           normal_queue_.size() + low_queue_.size();
}

bool PriorityScheduler::empty() const {
    std::scoped_lock lock(critical_mutex_, high_mutex_, normal_mutex_, low_mutex_);
    if (!critical_queue_.empty()) return false;
    if (!high_queue_.empty()) return false;
    if (!normal_queue_.empty()) return false;
    if (!low_queue_.empty()) return false;
    return true;
}

void PriorityScheduler::clear() {
    std::scoped_lock lock(critical_mutex_, high_mutex_, normal_mutex_, low_mutex_);
    critical_queue_.clear();
    high_queue_.clear();
    normal_queue_.clear();
    low_queue_.clear();
}

} // namespace executor
