#pragma once

#include <atomic>
#include <cstddef>
#include <vector>
#include <type_traits>
#include <memory>

namespace executor {
namespace util {

/**
 * @brief 无锁队列（SPSC - 单生产者单消费者）
 * 
 * 使用环形缓冲区实现，适用于单生产者单消费者场景。
 * 使用 std::atomic 和内存序保证线程安全。
 * 
 * @tparam T 队列元素类型，必须是可平凡复制的（trivially copyable）
 */
template<typename T>
class LockFreeQueue {
    static_assert(std::is_trivially_copyable_v<T>, 
                  "LockFreeQueue requires trivially copyable type");

public:
    /**
     * @brief 构造函数
     * @param capacity 队列容量（必须是2的幂，如果不是会自动调整为2的幂）
     */
    explicit LockFreeQueue(size_t capacity)
        : capacity_(round_to_power_of_two(capacity))
        , mask_(capacity_ - 1)
        , buffer_(capacity_)
        , write_pos_(0)
        , read_pos_(0) {
    }

    /**
     * @brief 推入元素到队列
     * @param item 要推入的元素
     * @return 成功返回true，队列满时返回false
     */
    bool push(const T& item) {
        const size_t current_write = write_pos_.load(std::memory_order_relaxed);
        const size_t next_write = (current_write + 1) & mask_;
        const size_t current_read = read_pos_.load(std::memory_order_acquire);

        // 检查队列是否满
        if (next_write == current_read) {
            return false;
        }

        // 写入数据
        buffer_[current_write] = item;

        // 更新写位置（使用release语义确保数据写入完成）
        write_pos_.store(next_write, std::memory_order_release);
        return true;
    }

    /**
     * @brief 从队列弹出元素
     * @param item 用于接收弹出元素的引用
     * @return 成功返回true，队列空时返回false
     */
    bool pop(T& item) {
        const size_t current_read = read_pos_.load(std::memory_order_relaxed);
        const size_t current_write = write_pos_.load(std::memory_order_acquire);

        // 检查队列是否空
        if (current_read == current_write) {
            return false;
        }

        // 读取数据
        item = buffer_[current_read];

        // 更新读位置（使用release语义确保数据读取完成）
        const size_t next_read = (current_read + 1) & mask_;
        read_pos_.store(next_read, std::memory_order_release);
        return true;
    }

    /**
     * @brief 检查队列是否为空
     * @return 队列为空返回true，否则返回false
     */
    bool empty() const {
        const size_t current_read = read_pos_.load(std::memory_order_acquire);
        const size_t current_write = write_pos_.load(std::memory_order_acquire);
        return current_read == current_write;
    }

    /**
     * @brief 获取队列中元素数量（近似值）
     * @return 队列中元素数量
     */
    size_t size() const {
        const size_t current_read = read_pos_.load(std::memory_order_acquire);
        const size_t current_write = write_pos_.load(std::memory_order_acquire);
        
        if (current_write >= current_read) {
            return current_write - current_read;
        } else {
            // 处理环形缓冲区回绕情况
            return (capacity_ - current_read) + current_write;
        }
    }

    /**
     * @brief 获取队列容量
     * @return 队列容量
     */
    size_t capacity() const {
        return capacity_;
    }

private:
    /**
     * @brief 将数字向上舍入到最近的2的幂
     * @param n 输入数字
     * @return 2的幂
     */
    static size_t round_to_power_of_two(size_t n) {
        if (n == 0) {
            return 1;
        }
        if ((n & (n - 1)) == 0) {
            return n;  // 已经是2的幂
        }
        
        // 找到最高位并左移1位
        size_t power = 1;
        while (power < n) {
            power <<= 1;
        }
        return power;
    }

    const size_t capacity_;           // 队列容量（2的幂）
    const size_t mask_;               // 掩码（capacity - 1），用于快速取模
    std::vector<T> buffer_;          // 环形缓冲区
    std::atomic<size_t> write_pos_;   // 写位置
    std::atomic<size_t> read_pos_;    // 读位置
};

} // namespace util
} // namespace executor
