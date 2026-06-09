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

size_t WorkerLocalQueue::push_batch(const Task* tasks, size_t n) {
    if (!tasks || n == 0) return 0;
    std::lock_guard<std::mutex> lock(mutex_);
    size_t max_size = queue_.size();
    size_t pushed = 0;
    for (size_t i = 0; i < n; ++i) {
        size_t current_size = size_.load(std::memory_order_relaxed);
        if (current_size >= max_size) break;
        queue_[back_index_] = TaskWrapper(tasks[i]);
        back_index_ = (back_index_ + 1) % queue_.size();
        size_.store(current_size + 1, std::memory_order_relaxed);
        ++pushed;
    }
    return pushed;
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
    // 关键不变量：empty() 必须与 size_.load() == 0 严格等价。
    //
    // 修复说明（2026-06-09, P-001）：
    //   旧实现在加锁路径上使用 `return front_index_ >= queue_.size();`，
    //   这是错误的——front_index_ 是环形缓冲区下标，由
    //   `front_index_ = (front_index_ + 1) % queue_.size()` 维护，
    //   其取值范围恒为 [0, queue_.size())，因此 `front_index_ >= queue_.size()`
    //   永远为 false。也就是说，旧版本在 size_>0 路径下加锁后永远返回 false，
    //   即 empty() 永远报告"非空"——与 size()/clear() 等其他方法的状态完全脱节。
    //
    // 正确做法：empty() 统一基于原子 size_ 判断，与 size() 保持一致语义。
    // 第一行无锁快速路径在绝大多数调用下直接返回，避免每次都加锁；
    // 仅在 size_ 暂态为非 0 时加锁复核（防 ABA / pop 竞态导致错误返回 true）。
    if (size_.load(std::memory_order_relaxed) == 0) {
        return true;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    return size_.load(std::memory_order_relaxed) == 0;
}

void WorkerLocalQueue::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    front_index_ = 0;
    back_index_ = 0;
    size_.store(0, std::memory_order_relaxed);
}

} // namespace executor
