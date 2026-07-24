#include <gtest/gtest.h>

#include "executor/gpu/transfer_validation.hpp"

#include <cstddef>
#include <string>

namespace {

class MockDriver {
public:
    bool submit(bool valid) {
        if (!valid) {
            return false;
        }
        ++calls;
        return true;
    }

    int calls = 0;
};

bool validate_h2d(const void* host_source, bool destination_registered,
                  size_t destination_capacity, size_t size, std::string* error) {
    using namespace executor::gpu;
    return validate_host_transfer_buffer("H2D", "source", host_source, size, error) &&
           validate_device_transfer_allocation("H2D", "destination", destination_registered,
                                               destination_capacity, size, error);
}

bool validate_d2h(void* host_destination, bool source_registered,
                  size_t source_capacity, size_t size, std::string* error) {
    using namespace executor::gpu;
    return validate_host_transfer_buffer("D2H", "destination", host_destination, size, error) &&
           validate_device_transfer_allocation("D2H", "source", source_registered,
                                               source_capacity, size, error);
}

bool validate_d2d(bool source_registered, size_t source_capacity,
                  bool destination_registered, size_t destination_capacity,
                  size_t size, std::string* error) {
    using namespace executor::gpu;
    return validate_device_transfer_allocation("D2D", "source", source_registered,
                                               source_capacity, size, error) &&
           validate_device_transfer_allocation("D2D", "destination", destination_registered,
                                               destination_capacity, size, error);
}

}  // namespace

TEST(GpuTransferValidationTest, RejectsOversizeAndNullHostBuffers) {
    constexpr size_t kCapacity = 16;
    int host_buffer[4] = {};
    std::string error;
    MockDriver driver;

    EXPECT_FALSE(driver.submit(validate_h2d(host_buffer, true, kCapacity, kCapacity + 1, &error)));
    EXPECT_NE(error.find("H2D"), std::string::npos);
    EXPECT_NE(error.find("17"), std::string::npos);
    EXPECT_NE(error.find("16"), std::string::npos);

    EXPECT_FALSE(driver.submit(validate_h2d(nullptr, true, kCapacity, kCapacity, &error)));
    EXPECT_NE(error.find("source host buffer is null"), std::string::npos);

    EXPECT_FALSE(driver.submit(validate_d2h(host_buffer, true, kCapacity - 1, kCapacity, &error)));
    EXPECT_NE(error.find("D2H"), std::string::npos);

    EXPECT_FALSE(driver.submit(validate_d2h(nullptr, true, kCapacity, kCapacity, &error)));
    EXPECT_NE(error.find("destination host buffer is null"), std::string::npos);

    EXPECT_FALSE(driver.submit(validate_d2d(true, kCapacity - 1, true, kCapacity,
                                             kCapacity, &error)));
    EXPECT_NE(error.find("source"), std::string::npos);

    EXPECT_FALSE(driver.submit(validate_d2d(true, kCapacity, true, kCapacity - 1,
                                             kCapacity, &error)));
    EXPECT_NE(error.find("destination"), std::string::npos);
    EXPECT_EQ(driver.calls, 0);
}

TEST(GpuTransferValidationTest, AllowsExactCapacityAndZeroByteNoOp) {
    constexpr size_t kCapacity = 16;
    int host_buffer[4] = {};
    std::string error;
    MockDriver driver;

    EXPECT_TRUE(driver.submit(validate_h2d(host_buffer, true, kCapacity, kCapacity, &error)));
    EXPECT_TRUE(driver.submit(validate_d2h(host_buffer, true, kCapacity, kCapacity, &error)));
    EXPECT_TRUE(driver.submit(validate_d2d(true, kCapacity, true, kCapacity, kCapacity, &error)));
    EXPECT_TRUE(validate_h2d(nullptr, false, 0, 0, &error));
    EXPECT_EQ(driver.calls, 3);
}
