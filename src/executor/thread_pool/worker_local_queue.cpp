#include "worker_local_queue.hpp"

namespace executor {

WorkerLocalQueue::WorkerLocalQueue(size_t capacity)
    : front_index_(0)
    , back_index_(0)
    , capacity_(capacity)
    , size_(0) {
    // 使用固定大小的环形缓冲区
    size_t buffer_size = (capacity_ > 0) ? capacity_ : 100;
    // TaskWrapper 可以默认构造和拷贝，所以 resize 是安全的
    queue_.resize(buffer_size);
}

bool WorkerLocalQueue::push(const Task& task) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t current_size = size_.load(std::memory_order_relaxed);
    size_t max_size = queue_.size();
    
    // 检查容量限制（环形缓冲区已满）
    if (current_size >= max_size) {
        return false;
    }
    
    // 使用环形缓冲区，在 back_index_ 位置构造（TaskWrapper 可拷贝）
    queue_[back_index_] = TaskWrapper(task);
    
    // 更新后端索引（环形）
    back_index_ = (back_index_ + 1) % queue_.size();
    size_.store(current_size + 1, std::memory_order_relaxed);
    return true;
}

bool WorkerLocalQueue::push(Task&& task) {
    return push(task);  // 委托给 const 版本
}

bool WorkerLocalQueue::pop(Task& task) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (size_.load(std::memory_order_relaxed) == 0) {
        return false;
    }
    
    // 从队列前端弹出（使用环形缓冲区）
    // 手动复制 Task 字段（因为 Task 包含 atomic，不能直接拷贝）
    const Task& front_task = queue_[front_index_].task;
    task.task_id = front_task.task_id;
    task.priority = front_task.priority;
    task.function = front_task.function;
    task.submit_time_ns = front_task.submit_time_ns;
    task.timeout_ms = front_task.timeout_ms;
    task.dependencies = front_task.dependencies;
    task.cancelled.store(front_task.cancelled.load(std::memory_order_acquire), std::memory_order_release);
    
    // 更新前端索引（环形）
    front_index_ = (front_index_ + 1) % queue_.size();
    size_.store(size_.load(std::memory_order_relaxed) - 1, std::memory_order_relaxed);
    return true;
}

bool WorkerLocalQueue::steal(Task& task) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (size_.load(std::memory_order_relaxed) == 0) {
        return false;
    }
    
    // 从队列后端弹出（使用环形缓冲区，从 back_index_ - 1 位置读取）
    // 计算后端位置（需要处理环形回绕）
    size_t steal_index = (back_index_ == 0) ? (queue_.size() - 1) : (back_index_ - 1);
    const Task& back_task = queue_[steal_index].task;
    task.task_id = back_task.task_id;
    task.priority = back_task.priority;
    task.function = back_task.function;
    task.submit_time_ns = back_task.submit_time_ns;
    task.timeout_ms = back_task.timeout_ms;
    task.dependencies = back_task.dependencies;
    task.cancelled.store(back_task.cancelled.load(std::memory_order_acquire), std::memory_order_release);
    
    // 更新后端索引（环形，向前移动）
    back_index_ = steal_index;
    size_.store(size_.load(std::memory_order_relaxed) - 1, std::memory_order_relaxed);
    return true;
}

size_t WorkerLocalQueue::size() const {
    // 使用原子变量快速查询，可能不是完全准确的
    return size_.load(std::memory_order_relaxed);
}

bool WorkerLocalQueue::empty() const {
    // 快速检查（使用原子变量）
    if (size_.load(std::memory_order_relaxed) == 0) {
        return true;
    }
    
    // 如果原子变量显示非空，需要加锁确认
    std::lock_guard<std::mutex> lock(mutex_);
    return front_index_ >= queue_.size();
}

void WorkerLocalQueue::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    front_index_ = 0;
    back_index_ = 0;
    size_.store(0, std::memory_order_relaxed);
}

} // namespace executor
