#pragma once

#include <atomic>
#include <memory>
#include <vector>

namespace executor {
namespace util {

/**
 * @brief Lock-free object pool for task allocation
 *
 * Pre-allocates objects and provides lock-free acquire/release operations.
 * Uses a free-list approach with atomic operations.
 */
template<typename T>
class ObjectPool {
public:
    explicit ObjectPool(size_t capacity = 1024) : capacity_(capacity) {
        // Pre-allocate all objects
        storage_.reserve(capacity);
        for (size_t i = 0; i < capacity; ++i) {
            storage_.emplace_back(std::make_unique<Node>());
        }

        // Build free list
        for (size_t i = 0; i < capacity - 1; ++i) {
            storage_[i]->next.store(storage_[i + 1].get(), std::memory_order_relaxed);
        }
        storage_[capacity - 1]->next.store(nullptr, std::memory_order_relaxed);

        free_list_.store(storage_[0].get(), std::memory_order_relaxed);
    }

    /**
     * @brief Acquire an object from the pool
     * @return Pointer to object, or nullptr if pool is exhausted
     */
    T* acquire() {
        Node* node = free_list_.load(std::memory_order_acquire);
        while (node != nullptr) {
            Node* next = node->next.load(std::memory_order_relaxed);
            if (free_list_.compare_exchange_weak(node, next,
                                                  std::memory_order_release,
                                                  std::memory_order_acquire)) {
                return &node->data;
            }
        }
        return nullptr;
    }

    /**
     * @brief Release an object back to the pool
     * @param obj Pointer to object to release
     */
    void release(T* obj) {
        if (!obj) return;

        Node* node = reinterpret_cast<Node*>(
            reinterpret_cast<char*>(obj) - offsetof(Node, data)
        );

        Node* old_head = free_list_.load(std::memory_order_relaxed);
        do {
            node->next.store(old_head, std::memory_order_relaxed);
        } while (!free_list_.compare_exchange_weak(old_head, node,
                                                     std::memory_order_release,
                                                     std::memory_order_relaxed));
    }

private:
    struct Node {
        T data;
        std::atomic<Node*> next{nullptr};
    };

    size_t capacity_;
    std::vector<std::unique_ptr<Node>> storage_;
    std::atomic<Node*> free_list_{nullptr};
};

} // namespace util
} // namespace executor
