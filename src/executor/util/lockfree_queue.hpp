#pragma once

#include <atomic>
#include <cstddef>
#include <vector>
#include <type_traits>
#include <memory>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <emmintrin.h>
#define PAUSE_INSTRUCTION() _mm_pause()
#else
#define PAUSE_INSTRUCTION() do {} while(0)
#endif

namespace executor {
namespace util {

/**
 * @brief 无锁队列（MPSC - 多生产者单消费者）
 *
 * 使用序列号跟踪每个槽位状态，保证线程安全。
 *
 * @tparam T 队列元素类型，必须是可平凡复制的（trivially copyable）
 */
template<typename T>
class LockFreeQueue {
    static_assert(std::is_trivially_copyable_v<T>,
                  "LockFreeQueue requires trivially copyable type");

public:
    explicit LockFreeQueue(size_t capacity, size_t backoff_multiplier = 1)
        : capacity_(round_to_power_of_two(capacity))
        , mask_(capacity_ - 1)
        , buffer_(capacity_)
        , sequences_(capacity_)
        , enqueue_pos_(0)
        , dequeue_pos_(0)
        , backoff_multiplier_(backoff_multiplier) {
        // 初始化序列号
        for (size_t i = 0; i < capacity_; ++i) {
            sequences_[i].store(i, std::memory_order_relaxed);
        }
    }

    bool push(const T& item) {
        size_t pos;
        size_t backoff = 1;
        constexpr size_t MAX_BACKOFF = 16;
        constexpr int MAX_RETRIES = 64;

        for (int retry = 0; retry < MAX_RETRIES; ++retry) {
            pos = enqueue_pos_.load(std::memory_order_relaxed);

            // 检查队列是否已满（保留一个空槽位）
            size_t deq = dequeue_pos_.load(std::memory_order_acquire);
            if (pos - deq >= capacity_ - 1) {
                return false;
            }

            size_t index = pos & mask_;
            size_t seq = sequences_[index].load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);

            if (diff == 0) {
                // 槽位可用，尝试预留
                if (enqueue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    // 写入数据
                    buffer_[index] = item;
                    // 更新序列号，标记数据已就绪
                    sequences_[index].store(pos + 1, std::memory_order_release);
                    return true;
                }
                // CAS 失败，指数退避（应用退避倍数）
                size_t scaled_backoff = backoff * backoff_multiplier_;
                for (size_t i = 0; i < scaled_backoff; ++i) {
                    PAUSE_INSTRUCTION();
                }
                backoff = backoff < MAX_BACKOFF ? backoff * 2 : MAX_BACKOFF;
            } else if (diff < 0) {
                // 队列满
                return false;
            }
            // diff > 0: 其他线程正在操作，重试
        }
        return false;
    }

    bool pop(T& item) {
        size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
        size_t index = pos & mask_;
        size_t seq = sequences_[index].load(std::memory_order_acquire);
        intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);

        if (diff == 0) {
            // 数据已就绪
            item = buffer_[index];
            sequences_[index].store(pos + capacity_, std::memory_order_release);
            dequeue_pos_.store(pos + 1, std::memory_order_release);
            return true;
        }
        // 队列空
        return false;
    }

    bool push_batch(const T* items, size_t count, size_t& pushed) {
        pushed = 0;
        if (count == 0) return true;

        size_t pos;
        size_t backoff = 1;
        constexpr size_t MAX_BACKOFF = 16;
        constexpr int MAX_RETRIES = 64;

        for (int retry = 0; retry < MAX_RETRIES; ++retry) {
            pos = enqueue_pos_.load(std::memory_order_relaxed);
            size_t deq = dequeue_pos_.load(std::memory_order_acquire);
            size_t available = capacity_ - 1 - (pos - deq);

            if (available == 0) return false;

            size_t batch_size = (count < available) ? count : available;
            if (enqueue_pos_.compare_exchange_weak(pos, pos + batch_size, std::memory_order_relaxed)) {
                for (size_t i = 0; i < batch_size; ++i) {
                    size_t index = (pos + i) & mask_;
                    buffer_[index] = items[i];
                    sequences_[index].store(pos + i + 1, std::memory_order_release);
                }
                pushed = batch_size;
                return true;
            }
            // CAS 失败，指数退避（应用退避倍数）
            size_t scaled_backoff = backoff * backoff_multiplier_;
            for (size_t i = 0; i < scaled_backoff; ++i) {
                PAUSE_INSTRUCTION();
            }
            backoff = backoff < MAX_BACKOFF ? backoff * 2 : MAX_BACKOFF;
        }
        return false;
    }

    size_t pop_batch(T* items, size_t max_count) {
        size_t popped = 0;
        size_t pos = dequeue_pos_.load(std::memory_order_relaxed);

        for (size_t i = 0; i < max_count; ++i) {
            size_t index = (pos + i) & mask_;
            size_t seq = sequences_[index].load(std::memory_order_acquire);

            if (seq != pos + i + 1) break;

            items[i] = buffer_[index];
            sequences_[index].store(pos + i + capacity_, std::memory_order_release);
            popped++;
        }

        if (popped > 0) {
            dequeue_pos_.store(pos + popped, std::memory_order_release);
        }
        return popped;
    }

    bool empty() const {
        size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
        size_t index = pos & mask_;
        size_t seq = sequences_[index].load(std::memory_order_acquire);
        return seq != pos + 1;
    }

    size_t size() const {
        size_t enq = enqueue_pos_.load(std::memory_order_relaxed);
        size_t deq = dequeue_pos_.load(std::memory_order_relaxed);
        return enq - deq;
    }

    size_t capacity() const {
        return capacity_;
    }

private:
    static size_t round_to_power_of_two(size_t n) {
        if (n == 0) return 1;
        if ((n & (n - 1)) == 0) return n;
        size_t power = 1;
        while (power < n) power <<= 1;
        return power;
    }

    const size_t capacity_;
    const size_t mask_;
    std::vector<T> buffer_;
    std::vector<std::atomic<size_t>> sequences_;
    alignas(64) std::atomic<size_t> enqueue_pos_;
    alignas(64) std::atomic<size_t> dequeue_pos_;
    const size_t backoff_multiplier_;
};

} // namespace util
} // namespace executor
