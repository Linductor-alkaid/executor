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

    // 在持锁状态下一次性快照 size_，避免后续多次 load 的不一致风险
    // （即便在持锁情况下，依赖 back_index_ 推算位置也容易出错，详见下方注释）
    size_t current_size = size_.load(std::memory_order_relaxed);
    if (current_size == 0) {
        return false;
    }

    // ============================================================
    // 修复 P-002：基于 size_ 重新计算窃取位置，而不是 back_index_ - 1
    // ============================================================
    //
    // 原实现（bug 路径）:steal_index = (back_index_ == 0) ? size-1 : back_index_ - 1
    //
    // 在有锁实现下，因为 push/pop/steal 都在 mutex 内串行执行，
    // back_index_ 与 size_ 在持锁期间是一致的，看起来"碰巧"工作。
    // 但这种实现存在以下隐患：
    //
    //  1. 索引计算依赖 back_index_ 与 size_ 的隐式一致性契约，
    //     任何未来重构（例如拆分锁、引入无锁路径、或在 push_batch 中
    //     对 back_index_ 的多次推进）都可能打破这个契约。
    //
    //  2. plan P-002 描述的更严重路径是：在无锁 / CAS 设计下，
    //     "old_size = size_; CAS 减小 size_ 后返回 queue_[old_size-1]"
    //     会因为 owner 线程在中间推进 front_index_ 而返回已被 pop 的元素，
    //     导致任务被重复执行或窃取位置错误。即使我们目前是有锁实现，
    //     也要避免使用这种"用旧的逻辑位置索引"作为窃取位置的反模式。
    //
    // 修复方法：用 size_ 重新计算 steal 位置，从"前一个"元素的角度推出
    // 当前 back_index_ - 1 应该指向哪个物理槽位：
    //   * back_index_ 指向"下一个 push 要写的位置"（即队列中
    //     第一个空槽的下标）
    //   * 最后一个有效元素的物理位置 = back_index_ 向前回退 1 步
    //     （环形回绕）
    //
    // 由于我们在锁内，back_index_ 和 size_ 都已是最新值，
    // 下面这个公式在所有情况下都正确：
    //   steal_index = (back_index_ + queue_.size() - 1) % queue_.size()
    //
    // 用"加 queue_.size() 再模"避免 back_index_ == 0 时的下界溢出。
    // ============================================================
    const size_t buf_size = queue_.size();
    size_t steal_index = (back_index_ + buf_size - 1) % buf_size;

    const Task& back_task = queue_[steal_index].task;
    task.task_id = back_task.task_id;
    task.priority = back_task.priority;
    task.function = back_task.function;
    task.submit_time_ns = back_task.submit_time_ns;
    task.timeout_ms = back_task.timeout_ms;
    task.dependencies = back_task.dependencies;
    task.cancelled.store(back_task.cancelled.load(std::memory_order_acquire), std::memory_order_release);

    // 更新后端索引（环形回退一步）
    back_index_ = steal_index;
    // 用本函数开头缓存的 current_size 做减法，保证读改写一致
    size_.store(current_size - 1, std::memory_order_relaxed);
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
