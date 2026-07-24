#pragma once

#include <cstddef>
#include <string>

namespace executor {
namespace gpu {

inline bool validate_host_transfer_buffer(const char* operation,
                                          const char* buffer_name,
                                          const void* buffer,
                                          size_t requested_size,
                                          std::string* error) {
    if (requested_size == 0 || buffer != nullptr) {
        return true;
    }

    *error = std::string(operation) + " rejected: " + buffer_name +
             " host buffer is null for requested size " + std::to_string(requested_size);
    return false;
}

inline bool validate_device_transfer_allocation(const char* operation,
                                                const char* allocation_name,
                                                bool is_registered,
                                                size_t capacity,
                                                size_t requested_size,
                                                std::string* error) {
    if (requested_size == 0) {
        return true;
    }
    if (!is_registered) {
        *error = std::string(operation) + " rejected: " + allocation_name +
                 " device allocation is not managed by this executor; requested size " +
                 std::to_string(requested_size) + ", capacity 0";
        return false;
    }
    if (requested_size <= capacity) {
        return true;
    }

    *error = std::string(operation) + " rejected: requested size " +
             std::to_string(requested_size) + " exceeds " + allocation_name +
             " device allocation capacity " + std::to_string(capacity);
    return false;
}

}  // namespace gpu
}  // namespace executor
