#pragma once

#include <atomic>
#include <cstddef>
#include <vector>
#include <type_traits>
#include <memory>
#include <mutex>

namespace executor {
namespace util {

/**
 * @brief 线程安全队列（MPSC - 多生产者单消费者）
 *
 * 使用互斥锁保证线程安全，支持多个生产者并发写入，单个消费者读取。
 *
 * @tparam T 队列元素类型，必须是可平凡复制的（trivially copyable）
 */
template<typename T>
class LockFreeQueue {
    static_assert(std::is_trivially_copyable_v<T>,
                  "LockFreeQueue requires trivially copyable type");

public:
    explicit LockFreeQueue(size_t capacity)
        : capacity_(capacity > 0 ? capacity : 1024) {
        buffer_.reserve(capacity_);
    }

    bool push(const T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (buffer_.size() >= capacity_) {
            return false;
        }
        buffer_.push_back(item);
        return true;
    }

    bool pop(T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (buffer_.empty()) {
            return false;
        }
        item = buffer_.front();
        buffer_.erase(buffer_.begin());
        return true;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffer_.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffer_.size();
    }

    size_t capacity() const {
        return capacity_;
    }

private:
    const size_t capacity_;
    std::vector<T> buffer_;
    mutable std::mutex mutex_;
};

} // namespace util
} // namespace executor
