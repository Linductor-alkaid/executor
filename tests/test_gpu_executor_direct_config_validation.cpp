#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <stdexcept>
#include <string>

#include <executor/config.hpp>
#include <executor/types.hpp>
#ifdef EXECUTOR_ENABLE_CUDA
#include "executor/gpu/cuda_executor.hpp"
#endif
#ifdef EXECUTOR_ENABLE_OPENCL
#include "executor/gpu/opencl_executor.hpp"
#endif

#ifdef EXECUTOR_ENABLE_CUDA
using executor::gpu::CudaExecutor;
#endif
using executor::gpu::GpuBackend;
using executor::gpu::GpuExecutorConfig;
using executor::gpu::GpuTaskConfig;
#ifdef EXECUTOR_ENABLE_OPENCL
using executor::gpu::OpenCLExecutor;
#endif

namespace {

GpuExecutorConfig make_config(GpuBackend backend, const std::string& name) {
    GpuExecutorConfig config;
    config.name = name;
    config.backend = backend;
    config.device_id = 0;
    config.max_queue_size = 0;
    config.default_stream_count = 1;
    return config;
}

template <typename Executor>
void expect_submit_fails_without_blocking(Executor& executor) {
    GpuTaskConfig task_config;
    auto future = executor.submit_kernel([](void*) {}, task_config);

    EXPECT_EQ(future.wait_for(std::chrono::milliseconds(100)),
              std::future_status::ready);
    EXPECT_THROW(future.get(), std::runtime_error);
}

bool has_queue_size_error(const std::string& message) {
    return message.find("max_queue_size") != std::string::npos &&
           message.find("> 0") != std::string::npos;
}

}  // namespace

TEST(GpuExecutorDirectConfigValidation, CudaRejectsZeroQueueCapacity) {
#ifdef EXECUTOR_ENABLE_CUDA
    auto config = make_config(GpuBackend::CUDA, "cuda_zero_queue_direct");
    CudaExecutor executor(config.name, config);

    EXPECT_FALSE(executor.start());
    EXPECT_TRUE(has_queue_size_error(executor.get_status().last_error_message))
        << executor.get_status().last_error_message;
    expect_submit_fails_without_blocking(executor);
#else
    GTEST_SKIP() << "CUDA support is not enabled";
#endif
}

TEST(GpuExecutorDirectConfigValidation, OpenCLRejectsZeroQueueCapacity) {
#ifdef EXECUTOR_ENABLE_OPENCL
    auto config = make_config(GpuBackend::OPENCL, "opencl_zero_queue_direct");
    OpenCLExecutor executor(config.name, config);

    EXPECT_FALSE(executor.start());
    EXPECT_TRUE(has_queue_size_error(executor.get_status().last_error_message))
        << executor.get_status().last_error_message;
    expect_submit_fails_without_blocking(executor);
#else
    GTEST_SKIP() << "OpenCL support is not enabled";
#endif
}
