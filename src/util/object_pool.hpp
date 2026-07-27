#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace executor {
namespace util {

/**
 * @brief Object pool for task allocation
 *
 * Pre-allocates objects and provides mutex-protected acquire/release operations.
 * Uses a free-list approach. Mutex protection eliminates ABA problem present
 * in the previous lock-free CAS implementation.
 *
 * All nodes live in a single contiguous array (owned by one unique_ptr) so that
 * release() can recover a node from its data pointer in O(1) via pointer
 * arithmetic instead of scanning storage_. Contiguity is essential: the index
 * of a node is just (byte offset from the first node) / sizeof(Node), which only
 * holds when nodes are equally spaced in one allocation.
 */
template<typename T>
class ObjectPool {
public:
    explicit ObjectPool(size_t capacity = 1024) : capacity_(capacity) {
        // Reject zero capacity: the free-list construction below would
        // underflow (`capacity - 1` wraps to SIZE_MAX) and dereference
        // `storage_[0]` / `storage_[SIZE_MAX]`, triggering UB and very
        // likely a segfault before acquire()/release() are ever called.
        if (capacity == 0) {
            throw std::invalid_argument("ObjectPool capacity must be > 0");
        }

        // Allocate every node in ONE contiguous array. Contiguity is what
        // makes release() O(1): a node's index is recovered from its data
        // pointer by pure pointer arithmetic. A vector of individually
        // heap-allocated nodes would scatter them across the heap and force
        // release() back into an O(n) scan.
        storage_ = std::make_unique<Node[]>(capacity);

        // Every node starts on the free list; mark it so and chain them.
        for (size_t i = 0; i < capacity; ++i) {
            storage_[i].in_free_list = true;
            storage_[i].next = (i + 1 < capacity) ? &storage_[i + 1] : nullptr;
        }
        free_list_ = &storage_[0];
    }

    /**
     * @brief Acquire an object from the pool
     * @return Pointer to object, or nullptr if pool is exhausted
     */
    T* acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!free_list_) return nullptr;
        Node* node = free_list_;
        free_list_ = node->next;
        node->in_free_list = false;  // handed out to a caller
        return &node->data;
    }

    /**
     * @brief Release an object back to the pool
     * @param obj Pointer to object to release
     *
     * O(1) in capacity: the node index is recovered from the data pointer by
     * pointer arithmetic on the contiguous node array (data is the first member
     * of Node, so consecutive nodes are exactly sizeof(Node) apart), and
     * double-release is caught with a per-node flag instead of a free-list scan.
     */
    void release(T* obj) {
        if (!obj) return;

        std::lock_guard<std::mutex> lock(mutex_);

        // Validate the integer address before subtracting it.  A foreign
        // pointer can be farther than ptrdiff_t can represent from storage_,
        // so signed address subtraction would be undefined behavior.
        const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(obj);
        const std::uintptr_t storage_begin =
            reinterpret_cast<std::uintptr_t>(&storage_[0].data);
        const std::uintptr_t storage_size = capacity_ * sizeof(Node);
        if (storage_size > std::numeric_limits<std::uintptr_t>::max() - storage_begin
            || address < storage_begin
            || address > storage_begin + storage_size) {
            throw std::logic_error(
                std::string("ObjectPool: release of foreign pointer ")
                + std::to_string(address));
        }

        const std::uintptr_t offset = address - storage_begin;
        if (offset % sizeof(Node) != 0) {
            throw std::logic_error(
                std::string("ObjectPool: release of foreign pointer ")
                + std::to_string(address));
        }

        const size_t index = static_cast<size_t>(offset / sizeof(Node));
        if (index >= capacity_) {
            throw std::logic_error(
                std::string("ObjectPool: release of foreign pointer ")
                + std::to_string(address));
        }
        Node* node = &storage_[index];

        // O(1) double-release detection via a per-node flag, set when the node
        // re-enters the free list and cleared when it is acquired.
        if (node->in_free_list) {
            throw std::logic_error(
                std::string("ObjectPool: double release of node ")
                + std::to_string(reinterpret_cast<std::uintptr_t>(node)));
        }

        node->in_free_list = true;
        node->next = free_list_;
        free_list_ = node;
    }

private:
    struct Node {
        T data;
        Node* next{nullptr};
        bool in_free_list{false};
    };

    size_t capacity_;
    std::unique_ptr<Node[]> storage_;  // contiguous; enables O(1) release()
    std::mutex mutex_;
    Node* free_list_{nullptr};
};

} // namespace util
} // namespace executor
