#include "priority_scheduler.hpp"
#include <algorithm>
#include <memory>
#include <mutex>

namespace executor {

void PriorityScheduler::copy_task_out(const std::shared_ptr<Task>& src, Task& out) {
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
    auto task_ptr = std::make_shared<Task>();
    task_ptr->task_id = task.task_id;
    task_ptr->priority = task.priority;
    task_ptr->function = task.function;
    task_ptr->submit_time_ns = task.submit_time_ns;
    task_ptr->timeout_ms = task.timeout_ms;
    task_ptr->dependencies = task.dependencies;
    task_ptr->cancelled.store(task.cancelled.load(std::memory_order_acquire), std::memory_order_release);

    switch (task.priority) {
        case TaskPriority::CRITICAL: {
            std::lock_guard<std::mutex> lock(critical_mutex_);
            critical_queue_.push(std::move(task_ptr));
            break;
        }
        case TaskPriority::HIGH: {
            std::lock_guard<std::mutex> lock(high_mutex_);
            high_queue_.push(std::move(task_ptr));
            break;
        }
        case TaskPriority::NORMAL: {
            std::lock_guard<std::mutex> lock(normal_mutex_);
            normal_queue_.push(std::move(task_ptr));
            break;
        }
        case TaskPriority::LOW: {
            std::lock_guard<std::mutex> lock(low_mutex_);
            low_queue_.push(std::move(task_ptr));
            break;
        }
    }
}

bool PriorityScheduler::dequeue(Task& task) {
    std::shared_ptr<Task> task_ptr;

    {
        std::lock_guard<std::mutex> lock(critical_mutex_);
        if (!critical_queue_.empty()) {
            task_ptr = critical_queue_.top();
            critical_queue_.pop();
        }
    }
    if (task_ptr) {
        copy_task_out(task_ptr, task);
        return true;
    }

    {
        std::lock_guard<std::mutex> lock(high_mutex_);
        if (!high_queue_.empty()) {
            task_ptr = high_queue_.top();
            high_queue_.pop();
        }
    }
    if (task_ptr) {
        copy_task_out(task_ptr, task);
        return true;
    }

    {
        std::lock_guard<std::mutex> lock(normal_mutex_);
        if (!normal_queue_.empty()) {
            task_ptr = normal_queue_.top();
            normal_queue_.pop();
        }
    }
    if (task_ptr) {
        copy_task_out(task_ptr, task);
        return true;
    }

    {
        std::lock_guard<std::mutex> lock(low_mutex_);
        if (!low_queue_.empty()) {
            task_ptr = low_queue_.top();
            low_queue_.pop();
        }
    }
    if (task_ptr) {
        copy_task_out(task_ptr, task);
        return true;
    }

    return false;
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
    while (!critical_queue_.empty()) critical_queue_.pop();
    while (!high_queue_.empty()) high_queue_.pop();
    while (!normal_queue_.empty()) normal_queue_.pop();
    while (!low_queue_.empty()) low_queue_.pop();
}

} // namespace executor
