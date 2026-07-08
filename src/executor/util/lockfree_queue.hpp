// Lock-free queue implementation.
#pragma once

#include <atomic>
#include <cstddef>
#include <limits>
#include <vector>
#include <type_traits>
#include <memory>
#include <stdexcept>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <emmintrin.h>
#define PAUSE_INSTRUCTION() _mm_pause()
#elif defined(__aarch64__) || defined(__arm__)
#define PAUSE_INSTRUCTION() __asm__ volatile("yield" ::: "memory")
#else
#define PAUSE_INSTRUCTION() do {} while(0)
#endif

namespace executor {
namespace util {

/**
 * @brief 无锁队列性能统计
 */
struct LockFreeQueueStats {
    uint64_t total_pushes = 0;
    uint64_t failed_pushes = 0;
    uint64_t total_pops = 0;
    uint64_t empty_pops = 0;
    uint64_t batch_pushes = 0;
    uint64_t batch_pops = 0;
    uint64_t current_size = 0;
    uint64_t peak_size = 0;
};

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
    explicit LockFreeQueue(size_t capacity, size_t backoff_multiplier = 1, bool enable_stats = false)
        : capacity_(round_to_power_of_two(capacity))
        , mask_(capacity_ - 1)
        , buffer_(capacity_)
        , sequences_(capacity_)
        , enqueue_pos_(0)
        , dequeue_pos_(0)
        , backoff_multiplier_(backoff_multiplier)
        , stats_enabled_(enable_stats) {
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
            pos = enqueue_pos_.load(std::memory_order_acquire);

            // 检查队列是否已满（保留一个空槽位）
            size_t deq = dequeue_pos_.load(std::memory_order_acquire);
            size_t in_flight = (pos >= deq) ? (pos - deq) : 0;
            if (in_flight >= capacity_ - 1) {
                // 260610P013: relaxed load
        if (stats_enabled_.load(std::memory_order_relaxed)) stats_.failed_pushes.fetch_add(1, std::memory_order_relaxed);
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
                    // 260610P013: relaxed load
                    if (stats_enabled_.load(std::memory_order_relaxed)) {
                        stats_.total_pushes.fetch_add(1, std::memory_order_relaxed);
                        update_peak_size();
                    }
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
                // 260610P013: relaxed load
        if (stats_enabled_.load(std::memory_order_relaxed)) stats_.failed_pushes.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            // diff > 0: 其他线程正在操作，重试
        }
        // 260610P013: relaxed load
        if (stats_enabled_.load(std::memory_order_relaxed)) stats_.failed_pushes.fetch_add(1, std::memory_order_relaxed);
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
            // 260610P013: relaxed load
            if (stats_enabled_.load(std::memory_order_relaxed)) stats_.total_pops.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        // 队列空
        // 260610P013: relaxed load
        if (stats_enabled_.load(std::memory_order_relaxed)) stats_.empty_pops.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    /**
     * @brief 批量入队。
     *
     * 当 count > 0 时，items 必须指向至少 count 个元素；空指针视为失败。
     */
    bool push_batch(const T* items, size_t count, size_t& pushed) {
        pushed = 0;
        if (count == 0) return true;
        if (items == nullptr) return false;

        size_t pos;
        size_t backoff = 1;
        constexpr size_t MAX_BACKOFF = 16;
        constexpr int MAX_RETRIES = 64;

        for (int retry = 0; retry < MAX_RETRIES; ++retry) {
            pos = enqueue_pos_.load(std::memory_order_acquire);
            size_t deq = dequeue_pos_.load(std::memory_order_acquire);
            size_t in_flight = (pos >= deq) ? (pos - deq) : 0;
            size_t available = (in_flight < capacity_ - 1) ? (capacity_ - 1 - in_flight) : 0;

            if (available == 0) {
                // 260610P013: relaxed load
        if (stats_enabled_.load(std::memory_order_relaxed)) stats_.failed_pushes.fetch_add(1, std::memory_order_relaxed);
                return false;
            }

            size_t batch_size = (count < available) ? count : available;

            // 260626P003: CAS 之前预校验本次 batch 拟占用的全部槽位
            // sequences_[index] == pos + i (即「待写入」状态)。该不变式
            // 与单条 push() 一致:任何被其他线程占用 (seq > pos+i) 或
            // 已经被消费者推进后回填的槽位 (seq < pos+i) 都视为不可用,
            // 放弃本轮 CAS 预留,退避后重试读取最新 pos/available。
            // 修复了 P-260626-003: CAS 成功后再覆盖式写入可能损坏已被
            // 消费者推进的槽位或与并发 push 写入交叉的不一致风险。
            bool slots_available = true;
            for (size_t i = 0; i < batch_size; ++i) {
                size_t index = (pos + i) & mask_;
                size_t seq = sequences_[index].load(std::memory_order_acquire);
                if (seq != pos + i) { slots_available = false; break; }
            }
            if (!slots_available) {
                size_t scaled_backoff = backoff * backoff_multiplier_;
                for (size_t i = 0; i < scaled_backoff; ++i) {
                    PAUSE_INSTRUCTION();
                }
                backoff = backoff < MAX_BACKOFF ? backoff * 2 : MAX_BACKOFF;
                continue;
            }

            if (enqueue_pos_.compare_exchange_weak(pos, pos + batch_size, std::memory_order_relaxed)) {
                for (size_t i = 0; i < batch_size; ++i) {
                    size_t index = (pos + i) & mask_;
                    buffer_[index] = items[i];
                    sequences_[index].store(pos + i + 1, std::memory_order_release);
                }
                pushed = batch_size;
                // 260610P013: relaxed load
                if (stats_enabled_.load(std::memory_order_relaxed)) {
                    stats_.total_pushes.fetch_add(batch_size, std::memory_order_relaxed);
                    stats_.batch_pushes.fetch_add(1, std::memory_order_relaxed);
                    update_peak_size();
                }
                return true;
            }
            // CAS 失败，指数退避（应用退避倍数）
            size_t scaled_backoff = backoff * backoff_multiplier_;
            for (size_t i = 0; i < scaled_backoff; ++i) {
                PAUSE_INSTRUCTION();
            }
            backoff = backoff < MAX_BACKOFF ? backoff * 2 : MAX_BACKOFF;
        }
        // 260610P013: relaxed load
        if (stats_enabled_.load(std::memory_order_relaxed)) stats_.failed_pushes.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    /**
     * @brief 精确批量入队，只有全部 count 个元素可写入时才成功。
     *
     * 当 count > 0 时，items 必须指向至少 count 个元素；空指针视为失败。
     */
    bool push_batch_exact(const T* items, size_t count) {
        if (count == 0) return true;
        if (items == nullptr) return false;

        size_t pos;
        size_t backoff = 1;
        constexpr size_t MAX_BACKOFF = 16;
        constexpr int MAX_RETRIES = 64;

        for (int retry = 0; retry < MAX_RETRIES; ++retry) {
            pos = enqueue_pos_.load(std::memory_order_acquire);
            size_t deq = dequeue_pos_.load(std::memory_order_acquire);
            size_t in_flight = (pos >= deq) ? (pos - deq) : 0;
            size_t available = (in_flight < capacity_ - 1) ? (capacity_ - 1 - in_flight) : 0;

            if (available < count) {
                if (stats_enabled_.load(std::memory_order_relaxed)) {
                    stats_.failed_pushes.fetch_add(1, std::memory_order_relaxed);
                }
                return false;
            }

            bool slots_available = true;
            for (size_t i = 0; i < count; ++i) {
                size_t index = (pos + i) & mask_;
                size_t seq = sequences_[index].load(std::memory_order_acquire);
                if (seq != pos + i) { slots_available = false; break; }
            }
            if (!slots_available) {
                size_t scaled_backoff = backoff * backoff_multiplier_;
                for (size_t i = 0; i < scaled_backoff; ++i) {
                    PAUSE_INSTRUCTION();
                }
                backoff = backoff < MAX_BACKOFF ? backoff * 2 : MAX_BACKOFF;
                continue;
            }

            if (enqueue_pos_.compare_exchange_weak(pos, pos + count, std::memory_order_relaxed)) {
                for (size_t i = 0; i < count; ++i) {
                    size_t index = (pos + i) & mask_;
                    buffer_[index] = items[i];
                    sequences_[index].store(pos + i + 1, std::memory_order_release);
                }
                if (stats_enabled_.load(std::memory_order_relaxed)) {
                    stats_.total_pushes.fetch_add(count, std::memory_order_relaxed);
                    stats_.batch_pushes.fetch_add(1, std::memory_order_relaxed);
                    update_peak_size();
                }
                return true;
            }

            size_t scaled_backoff = backoff * backoff_multiplier_;
            for (size_t i = 0; i < scaled_backoff; ++i) {
                PAUSE_INSTRUCTION();
            }
            backoff = backoff < MAX_BACKOFF ? backoff * 2 : MAX_BACKOFF;
        }

        if (stats_enabled_.load(std::memory_order_relaxed)) {
            stats_.failed_pushes.fetch_add(1, std::memory_order_relaxed);
        }
        return false;
    }

    /**
     * @brief 批量出队。
     *
     * 当 max_count > 0 时，items 必须指向至少 max_count 个元素；空指针返回 0。
     */
    size_t pop_batch(T* items, size_t max_count) {
        if (max_count > 0 && items == nullptr) return 0;

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
            // 260610P013: relaxed load
            if (stats_enabled_.load(std::memory_order_relaxed)) {
                stats_.total_pops.fetch_add(popped, std::memory_order_relaxed);
                stats_.batch_pops.fetch_add(1, std::memory_order_relaxed);
            }
            // 260610P013: relaxed load
        } else if (stats_enabled_.load(std::memory_order_relaxed)) {
            stats_.empty_pops.fetch_add(1, std::memory_order_relaxed);
        }
        return popped;
    }

    /**
     * @brief 判断队列是否为空（消费者就绪检查，近似值）
     *
     * 实现基于 sequences_ 视角：检查下一个待消费槽位的序列号。
     * 在 push 完成后（sequences_[i] 已 release store 新值）且该槽位尚未被 pop 消费时，
     * 此方法返回 false；在 pop 完成（sequences_[i] 被回填为 pos+capacity）后返回 true。
     *
     * **与 size() 的语义差异（260611P007 文档化）**：
     *   - `empty()` 是消费者就绪检查（用于 pop 前判空），基于 sequences_ 视角
     *   - `size()`  是容量统计（基于 enqueue_pos_ - dequeue_pos_ 计数）
     *   - 两者均为近似值，在并发下可能短暂给出不一致结论（empty() 为 true 时 size() 仍 > 0，
     *     或反之），不可用于精确同步。若需严格判空，应使用 `pop()` 的返回值而非 `empty()`。
     */
    bool empty() const {
        size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
        size_t index = pos & mask_;
        size_t seq = sequences_[index].load(std::memory_order_acquire);
        return seq != pos + 1;
    }

    /**
     * @brief 返回队列中当前的元素数量（近似值）
     *
     * 本方法在多线程并发下读取 enqueue_pos_ 和 dequeue_pos_ 两个原子变量。
     * 在 ARM/POWER 等弱内存序架构上,如果使用 relaxed 加载,编译器/CPU 可能把
     * dequeue_pos_ 的读取重排到 enqueue_pos_ 之前,导致在 deq 刚刚被推进的
     * 瞬间出现 enq < deq 的情况,触发 size_t 下溢,size() 返回一个巨大的
     * 接近 SIZE_MAX 的值,引发调用方分配/拷贝异常甚至安全断言。
     *
     * 修复:两个加载都改用 acquire,确保 enqueue_pos_ 看到的是 dequeue_pos_
     * 推进之后的"最新"视图(以及相反),并用饱和减法保护下溢。
     *
     * 注意:此方法返回的是某瞬时快照,不是精确值(并发环境下两值仍可能小幅
     * 不一致),仅供监控/统计用途,不可作为容量判定依据。
     *
     * 在弱序架构 (ARM/POWER) 上的端到端正确性需要 CI 验证,本机 x86_64
     * TSO 模型下能掩盖此问题。
     *
     * **与 empty() 的语义差异（260611P007 文档化）**：本方法基于 enqueue_pos_/dequeue_pos_
     * 计数（容量统计视角），而 `empty()` 基于 sequences_（消费者就绪视角）。两者均为近似值，
     * 并发下可能短暂不一致。LockFreeTaskExecutor::pending_count() 本质上即调用本方法，
     * 故 pending_count() 也是近似值，不可用于精确同步。
     */
    size_t size() const {
        size_t enq = enqueue_pos_.load(std::memory_order_acquire);
        size_t deq = dequeue_pos_.load(std::memory_order_acquire);
        return (enq >= deq) ? (enq - deq) : 0;
    }

    size_t capacity() const {
        return capacity_;
    }

    void enable_stats(bool enable) {
        // 260610P013: relaxed store
        stats_enabled_.store(enable, std::memory_order_relaxed);
    }

    LockFreeQueueStats get_stats() const {
        // 260610P013: relaxed load
        if (!stats_enabled_.load(std::memory_order_relaxed)) return {};
        LockFreeQueueStats result;
        result.total_pushes = stats_.total_pushes.load(std::memory_order_relaxed);
        result.failed_pushes = stats_.failed_pushes.load(std::memory_order_relaxed);
        result.total_pops = stats_.total_pops.load(std::memory_order_relaxed);
        result.empty_pops = stats_.empty_pops.load(std::memory_order_relaxed);
        result.batch_pushes = stats_.batch_pushes.load(std::memory_order_relaxed);
        result.batch_pops = stats_.batch_pops.load(std::memory_order_relaxed);
        result.current_size = size();
        result.peak_size = stats_.peak_size.load(std::memory_order_relaxed);
        return result;
    }

    // 260611P004: helpers for callers that perform a logical "batch"
    // operation by composing per-item push/pop (e.g. the exception-safe
    // push_tasks_batch path in LockFreeTaskExecutor). This keeps the
    // stats signal meaningful regardless of which underlying entry
    // point was used.
    void record_batch_push() {
        if (stats_enabled_.load(std::memory_order_relaxed)) {
            stats_.batch_pushes.fetch_add(1, std::memory_order_relaxed);
        }
    }
    void record_batch_pop() {
        if (stats_enabled_.load(std::memory_order_relaxed)) {
            stats_.batch_pops.fetch_add(1, std::memory_order_relaxed);
        }
    }

private:
    void update_peak_size() {
        size_t current = size();
        uint64_t peak = stats_.peak_size.load(std::memory_order_relaxed);
        while (current > peak) {
            if (stats_.peak_size.compare_exchange_weak(peak, current, std::memory_order_relaxed)) {
                break;
            }
        }
    }
    static size_t round_to_power_of_two(size_t n) {
        if (n < 2) {
            throw std::invalid_argument("LockFreeQueue capacity must be at least 2");
        }

        constexpr size_t max_power_of_two = (std::numeric_limits<size_t>::max() / 2) + 1;
        if (n > max_power_of_two) {
            throw std::invalid_argument("LockFreeQueue capacity is too large to round to a power of two");
        }

        size_t power = 1;
        while (power < n) power <<= 1;

        if (power > std::vector<T>().max_size() ||
            power > std::vector<std::atomic<size_t>>().max_size()) {
            throw std::invalid_argument("LockFreeQueue capacity is too large to allocate");
        }

        return power;
    }

    const size_t capacity_;
    const size_t mask_;
    std::vector<T> buffer_;
    std::vector<std::atomic<size_t>> sequences_;
    alignas(64) std::atomic<size_t> enqueue_pos_;
    alignas(64) std::atomic<size_t> dequeue_pos_;
    const size_t backoff_multiplier_;
    // 260610P013: std::atomic<bool> 替换裸 bool 字段
    // enable_stats() 可从任意线程写入,热路径 (push/pop) 频繁读取 — C++ data race (UB)
    // relaxed ordering: 统计开关不与其它内存构成 happens-before 关系
    std::atomic<bool> stats_enabled_;

    struct Stats {
        alignas(64) std::atomic<uint64_t> total_pushes{0};
        alignas(64) std::atomic<uint64_t> failed_pushes{0};
        alignas(64) std::atomic<uint64_t> total_pops{0};
        alignas(64) std::atomic<uint64_t> empty_pops{0};
        alignas(64) std::atomic<uint64_t> batch_pushes{0};
        alignas(64) std::atomic<uint64_t> batch_pops{0};
        alignas(64) std::atomic<uint64_t> peak_size{0};
    };
    Stats stats_;
};

} // namespace util
} // namespace executor
