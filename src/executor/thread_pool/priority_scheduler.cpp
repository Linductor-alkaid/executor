#include "priority_scheduler.hpp"
#include <algorithm>
#include <memory>

namespace executor {

void PriorityScheduler::enqueue(const Task& task) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 创建shared_ptr包装Task
    auto task_ptr = std::make_shared<Task>();
    task_ptr->task_id = task.task_id;
    task_ptr->priority = task.priority;
    task_ptr->function = task.function;
    task_ptr->submit_time_ns = task.submit_time_ns;
    task_ptr->timeout_ms = task.timeout_ms;
    task_ptr->dependencies = task.dependencies;
    task_ptr->cancelled.store(task.cancelled.load(std::memory_order_acquire), std::memory_order_release);
    
    switch (task.priority) {
        case TaskPriority::CRITICAL:
            critical_queue_.push(task_ptr);
            break;
        case TaskPriority::HIGH:
            high_queue_.push(task_ptr);
            break;
        case TaskPriority::NORMAL:
            normal_queue_.push(task_ptr);
            break;
        case TaskPriority::LOW:
            low_queue_.push(task_ptr);
            break;
    }
}

bool PriorityScheduler::dequeue(Task& task) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::shared_ptr<Task> task_ptr;
    
    // 按优先级顺序从高到低获取任务
    if (!critical_queue_.empty()) {
        task_ptr = critical_queue_.top();
        critical_queue_.pop();
    } else if (!high_queue_.empty()) {
        task_ptr = high_queue_.top();
        high_queue_.pop();
    } else if (!normal_queue_.empty()) {
        task_ptr = normal_queue_.top();
        normal_queue_.pop();
    } else if (!low_queue_.empty()) {
        task_ptr = low_queue_.top();
        low_queue_.pop();
    } else {
        return false;
    }
    
    // 复制Task内容到输出参数
    if (task_ptr) {
        task.task_id = task_ptr->task_id;
        task.priority = task_ptr->priority;
        task.function = task_ptr->function;
        task.submit_time_ns = task_ptr->submit_time_ns;
        task.timeout_ms = task_ptr->timeout_ms;
        task.dependencies = task_ptr->dependencies;
        task.cancelled.store(task_ptr->cancelled.load(std::memory_order_acquire), std::memory_order_release);
        return true;
    }
    
    return false;
}

size_t PriorityScheduler::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return critical_queue_.size() + high_queue_.size() + 
           normal_queue_.size() + low_queue_.size();
}

bool PriorityScheduler::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return critical_queue_.empty() && high_queue_.empty() && 
           normal_queue_.empty() && low_queue_.empty();
}

void PriorityScheduler::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // priority_queue没有clear方法，需要逐个pop
    while (!critical_queue_.empty()) {
        critical_queue_.pop();
    }
    while (!high_queue_.empty()) {
        high_queue_.pop();
    }
    while (!normal_queue_.empty()) {
        normal_queue_.pop();
    }
    while (!low_queue_.empty()) {
        low_queue_.pop();
    }
}

} // namespace executor
