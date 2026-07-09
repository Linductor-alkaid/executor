#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>
#include <future>
#include <stdexcept>
#include <string>

#define private public
#include "executor/gpu/opencl_executor.hpp"
#include "executor/gpu/opencl_loader.hpp"
#undef private

using executor::gpu::GpuBackend;
using executor::gpu::GpuExecutorConfig;
using executor::gpu::GpuTaskConfig;
using executor::gpu::OpenCLExecutor;
using executor::gpu::OpenCLFunctionPointers;
using executor::gpu::OpenCLLoader;

namespace {

cl_int fake_cl_get_platform_ids(cl_uint num_entries,
                                cl_platform_id* platforms,
                                cl_uint* num_platforms) {
    if (num_platforms != nullptr) {
        *num_platforms = 1;
    }
    if (platforms != nullptr && num_entries > 0) {
        platforms[0] = reinterpret_cast<cl_platform_id>(0x1);
    }
    return CL_SUCCESS;
}

cl_int fake_cl_get_device_ids(cl_platform_id,
                              cl_device_type,
                              cl_uint num_entries,
                              cl_device_id* devices,
                              cl_uint* num_devices) {
    if (num_devices != nullptr) {
        *num_devices = 1;
    }
    if (devices != nullptr && num_entries > 0) {
        devices[0] = reinterpret_cast<cl_device_id>(0x2);
    }
    return CL_SUCCESS;
}

cl_int fake_cl_get_device_info(cl_device_id,
                               cl_device_info,
                               size_t param_value_size,
                               void* param_value,
                               size_t* param_value_size_ret) {
    constexpr cl_ulong kMemorySize = 1024 * 1024;
    if (param_value != nullptr && param_value_size >= sizeof(kMemorySize)) {
        std::memcpy(param_value, &kMemorySize, sizeof(kMemorySize));
    }
    if (param_value_size_ret != nullptr) {
        *param_value_size_ret = sizeof(kMemorySize);
    }
    return CL_SUCCESS;
}

cl_context fake_cl_create_context(const cl_context_properties*,
                                  cl_uint,
                                  const cl_device_id*,
                                  void (*)(const char*, const void*, size_t, void*),
                                  void*,
                                  cl_int* errcode_ret) {
    if (errcode_ret != nullptr) {
        *errcode_ret = CL_SUCCESS;
    }
    return reinterpret_cast<cl_context>(0x3);
}

cl_int fake_cl_release_context(cl_context) {
    return CL_SUCCESS;
}

cl_command_queue fake_cl_create_command_queue(cl_context,
                                              cl_device_id,
                                              cl_command_queue_properties,
                                              cl_int* errcode_ret) {
    if (errcode_ret != nullptr) {
        *errcode_ret = CL_SUCCESS;
    }
    return reinterpret_cast<cl_command_queue>(0x4);
}

cl_int fake_cl_release_command_queue(cl_command_queue) {
    return CL_SUCCESS;
}

cl_mem fake_cl_create_buffer(cl_context, cl_mem_flags, size_t, void*, cl_int* errcode_ret) {
    if (errcode_ret != nullptr) {
        *errcode_ret = CL_SUCCESS;
    }
    return reinterpret_cast<cl_mem>(0x5);
}

cl_int fake_cl_release_mem_object(cl_mem) {
    return CL_SUCCESS;
}

cl_int fake_cl_enqueue_read_buffer(cl_command_queue,
                                   cl_mem,
                                   cl_bool,
                                   size_t,
                                   size_t,
                                   void*,
                                   cl_uint,
                                   const cl_event*,
                                   cl_event*) {
    return CL_SUCCESS;
}

cl_int fake_cl_enqueue_write_buffer(cl_command_queue,
                                    cl_mem,
                                    cl_bool,
                                    size_t,
                                    size_t,
                                    const void*,
                                    cl_uint,
                                    const cl_event*,
                                    cl_event*) {
    return CL_SUCCESS;
}

cl_int fake_cl_enqueue_copy_buffer(cl_command_queue,
                                   cl_mem,
                                   cl_mem,
                                   size_t,
                                   size_t,
                                   size_t,
                                   cl_uint,
                                   const cl_event*,
                                   cl_event*) {
    return CL_SUCCESS;
}

cl_int fake_cl_finish(cl_command_queue) {
    return CL_SUCCESS;
}

cl_int fake_cl_flush(cl_command_queue) {
    return CL_SUCCESS;
}

class FakeOpenCLLoaderScope {
public:
    FakeOpenCLLoaderScope()
        : loader_(OpenCLLoader::instance())
        , original_is_loaded_(loader_.is_loaded_)
        , original_dll_handle_(loader_.dll_handle_)
        , original_dll_path_(loader_.dll_path_)
        , original_functions_(loader_.functions_) {
        OpenCLFunctionPointers functions;
        functions.clGetPlatformIDs = fake_cl_get_platform_ids;
        functions.clGetDeviceIDs = fake_cl_get_device_ids;
        functions.clGetDeviceInfo = fake_cl_get_device_info;
        functions.clCreateContext = fake_cl_create_context;
        functions.clReleaseContext = fake_cl_release_context;
        functions.clCreateCommandQueue = fake_cl_create_command_queue;
        functions.clReleaseCommandQueue = fake_cl_release_command_queue;
        functions.clCreateBuffer = fake_cl_create_buffer;
        functions.clReleaseMemObject = fake_cl_release_mem_object;
        functions.clEnqueueReadBuffer = fake_cl_enqueue_read_buffer;
        functions.clEnqueueWriteBuffer = fake_cl_enqueue_write_buffer;
        functions.clEnqueueCopyBuffer = fake_cl_enqueue_copy_buffer;
        functions.clFinish = fake_cl_finish;
        functions.clFlush = fake_cl_flush;

        loader_.is_loaded_ = true;
        loader_.dll_handle_ = nullptr;
        loader_.dll_path_ = "fake-opencl";
        loader_.functions_ = functions;
    }

    ~FakeOpenCLLoaderScope() {
        loader_.functions_ = original_functions_;
        loader_.dll_path_ = original_dll_path_;
        loader_.dll_handle_ = original_dll_handle_;
        loader_.is_loaded_ = original_is_loaded_;
    }

    FakeOpenCLLoaderScope(const FakeOpenCLLoaderScope&) = delete;
    FakeOpenCLLoaderScope& operator=(const FakeOpenCLLoaderScope&) = delete;

private:
    OpenCLLoader& loader_;
    bool original_is_loaded_;
    void* original_dll_handle_;
    std::string original_dll_path_;
    OpenCLFunctionPointers original_functions_;
};

GpuExecutorConfig make_opencl_config() {
    GpuExecutorConfig config;
    config.name = "opencl_kernel_last_error";
    config.backend = GpuBackend::OPENCL;
    config.device_id = 0;
    config.default_stream_count = 1;
    config.max_queue_size = 16;
    return config;
}

bool contains(const std::string& message, const std::string& needle) {
    return message.find(needle) != std::string::npos;
}

}  // namespace

TEST(OpenCLKernelExceptionLastError, InvalidStreamSetsLastError) {
    FakeOpenCLLoaderScope fake_loader;
    OpenCLExecutor executor("opencl_kernel_last_error", make_opencl_config());
    ASSERT_TRUE(executor.start()) << executor.get_status().last_error_message;

    GpuTaskConfig task_config;
    task_config.stream_id = 9999;

    auto future = executor.submit_kernel([](void*) {}, task_config);
    EXPECT_THROW(future.get(), std::runtime_error);

    const auto status = executor.get_status();
    EXPECT_EQ(status.failed_kernels, 1U);
    EXPECT_TRUE(contains(status.last_error_message, "submit_kernel"))
        << status.last_error_message;
    EXPECT_TRUE(contains(status.last_error_message, "invalid stream_id 9999"))
        << status.last_error_message;
}

TEST(OpenCLKernelExceptionLastError, KernelFuncExceptionSetsLastError) {
    FakeOpenCLLoaderScope fake_loader;
    OpenCLExecutor executor("opencl_kernel_last_error", make_opencl_config());
    ASSERT_TRUE(executor.start()) << executor.get_status().last_error_message;

    GpuTaskConfig task_config;
    task_config.stream_id = 0;

    auto future = executor.submit_kernel([](void*) {
        throw std::runtime_error("user-fault");
    }, task_config);

    try {
        future.get();
        FAIL() << "Expected kernel exception";
    } catch (const std::runtime_error& ex) {
        EXPECT_TRUE(contains(ex.what(), "user-fault")) << ex.what();
    } catch (...) {
        FAIL() << "Expected std::runtime_error";
    }

    const auto status = executor.get_status();
    EXPECT_EQ(status.failed_kernels, 1U);
    EXPECT_TRUE(contains(status.last_error_message, "submit_kernel: kernel_func exception"))
        << status.last_error_message;
    EXPECT_TRUE(contains(status.last_error_message, "user-fault"))
        << status.last_error_message;
}
