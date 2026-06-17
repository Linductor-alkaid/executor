#pragma once

#include <cassert>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace executor {
namespace util {

/**
 * @brief Object pool for task allocation
 *
 * Pre-allocates objects and provides mutex-protected acquire/release operations.
 * Uses a free-list approach. Mutex protection eliminates ABA problem present
 * in the previous lock-free CAS implementation.
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

        // Pre-allocate all objects
        storage_.reserve(capacity);
        for (size_t i = 0; i < capacity; ++i) {
            storage_.emplace_back(std::make_unique<Node>());
        }

        // Build free list
        for (size_t i = 0; i < capacity - 1; ++i) {
            storage_[i]->next = storage_[i + 1].get();
        }
        storage_[capacity - 1]->next = nullptr;

        free_list_ = storage_[0].get();
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
        return &node->data;
    }

    /**
     * @brief Release an object back to the pool
     * @param obj Pointer to object to release
     */
    void release(T* obj) {
        if (!obj) return;

        std::lock_guard<std::mutex> lock(mutex_);
        Node* node = nullptr;
        for (const auto& stored_node : storage_) {
            if (&stored_node->data == obj) {
                node = stored_node.get();
                break;
            }
        }
        assert(node && "ObjectPool::release called with a foreign pointer");
        if (!node) return;

        bool already_free = false;
        for (Node* current = free_list_; current; current = current->next) {
            if (current == node) {
                already_free = true;
                break;
            }
        }
        assert(!already_free && "ObjectPool::release called more than once");
        if (already_free) return;

        node->next = free_list_;
        free_list_ = node;
    }

private:
    struct Node {
        T data;
        Node* next{nullptr};
    };

    size_t capacity_;
    std::vector<std::unique_ptr<Node>> storage_;
    std::mutex mutex_;
    Node* free_list_{nullptr};
};

} // namespace util
} // namespace executor
