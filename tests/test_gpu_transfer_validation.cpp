#include <gtest/gtest.h>

#include "executor/gpu/transfer_validation.hpp"

#include <cstddef>
#include <string>

namespace {

using executor::gpu::validate_host_transfer_buffer;
using executor::gpu::validate_device_transfer_allocation;

// Direct unit tests for the two transfer_validation helpers. The integration
// with the real CudaExecutor::copy_to_device / OpenCLExecutor::copy_to_device
// entry points is covered indirectly by tests/test_cuda_executor.cpp and
// tests/test_opencl_executor.cpp (those run the real copy paths in this
// repository's CI). Running the helpers directly here keeps the boundary
// contract pinned without depending on a CUDA runtime in the build env.

TEST(GpuTransferValidationTest, HostBufferNullRejectedForNonZeroSize) {
    std::string error;
    EXPECT_FALSE(validate_host_transfer_buffer("op", "source", nullptr, 16, &error));
    EXPECT_NE(error.find("source host buffer is null"), std::string::npos);
    EXPECT_NE(error.find("16"), std::string::npos);
}

TEST(GpuTransferValidationTest, HostBufferNullAllowedForZeroSize) {
    std::string error;
    EXPECT_TRUE(validate_host_transfer_buffer("op", "source", nullptr, 0, &error));
    EXPECT_TRUE(error.empty());
}

TEST(GpuTransferValidationTest, HostBufferNonNullAlwaysAllowed) {
    int buffer = 0;
    std::string error;
    EXPECT_TRUE(validate_host_transfer_buffer("op", "source", &buffer, 16, &error));
    EXPECT_TRUE(validate_host_transfer_buffer("op", "source", &buffer, 0, &error));
}

TEST(GpuTransferValidationTest, DeviceAllocationUnregisteredRejected) {
    std::string error;
    EXPECT_FALSE(validate_device_transfer_allocation("op", "destination",
                                                    /*is_registered=*/false,
                                                    /*capacity=*/0, 16, &error));
    EXPECT_NE(error.find("not managed"), std::string::npos);
    EXPECT_NE(error.find("16"), std::string::npos);
}

TEST(GpuTransferValidationTest, DeviceAllocationOversizeRejected) {
    std::string error;
    EXPECT_FALSE(validate_device_transfer_allocation("op", "destination",
                                                    /*is_registered=*/true,
                                                    /*capacity=*/16, 17, &error));
    EXPECT_NE(error.find("17"), std::string::npos);
    EXPECT_NE(error.find("16"), std::string::npos);
}

TEST(GpuTransferValidationTest, DeviceAllocationExactCapacityAllowed) {
    std::string error;
    EXPECT_TRUE(validate_device_transfer_allocation("op", "destination",
                                                   /*is_registered=*/true,
                                                   /*capacity=*/16, 16, &error));
    EXPECT_TRUE(error.empty());
}

TEST(GpuTransferValidationTest, DeviceAllocationZeroSizeAlwaysAllowed) {
    std::string error;
    EXPECT_TRUE(validate_device_transfer_allocation("op", "destination",
                                                   /*is_registered=*/false,
                                                   /*capacity=*/0, 0, &error));
    EXPECT_TRUE(validate_device_transfer_allocation("op", "destination",
                                                   /*is_registered=*/true,
                                                   /*capacity=*/16, 0, &error));
}

}  // namespace
