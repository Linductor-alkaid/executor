#include <gtest/gtest.h>
#include "executor/gpu/opencl_executor.hpp"
#include "executor/gpu/opencl_loader.hpp"

using namespace executor::gpu;

class OpenCLExecutorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 尝试加载 OpenCL
        auto& loader = OpenCLLoader::instance();
        opencl_available_ = loader.load();

        if (opencl_available_) {
            GpuExecutorConfig config;
            config.name = "test_opencl";
            config.backend = GpuBackend::OPENCL;
            config.device_id = 0;
            config.default_stream_count = 2;

            executor_ = std::make_unique<OpenCLExecutor>("test_opencl", config);
        }
    }

    void TearDown() override {
        executor_.reset();
    }

    bool opencl_available_ = false;
    std::unique_ptr<OpenCLExecutor> executor_;
};

TEST_F(OpenCLExecutorTest, LoaderTest) {
    auto& loader = OpenCLLoader::instance();

    if (loader.is_available()) {
        EXPECT_FALSE(loader.get_dll_path().empty());
        const auto& funcs = loader.get_functions();
        EXPECT_TRUE(funcs.is_complete());
    }
}

TEST_F(OpenCLExecutorTest, ExecutorCreation) {
    if (!opencl_available_) {
        GTEST_SKIP() << "OpenCL not available";
    }

    ASSERT_NE(executor_, nullptr);
    EXPECT_TRUE(executor_->start());
    EXPECT_EQ(executor_->get_name(), "test_opencl");
}

TEST_F(OpenCLExecutorTest, DeviceInfo) {
    if (!opencl_available_) {
        GTEST_SKIP() << "OpenCL not available";
    }

    ASSERT_TRUE(executor_->start());

    auto info = executor_->get_device_info();
    EXPECT_EQ(info.backend, GpuBackend::OPENCL);
    EXPECT_FALSE(info.name.empty());
}

TEST_F(OpenCLExecutorTest, MemoryAllocation) {
    if (!opencl_available_) {
        GTEST_SKIP() << "OpenCL not available";
    }

    ASSERT_TRUE(executor_->start());

    const size_t size = 1024 * sizeof(float);
    void* ptr = executor_->allocate_device_memory(size);
    ASSERT_NE(ptr, nullptr);

    executor_->free_device_memory(ptr);
}

TEST_F(OpenCLExecutorTest, MemoryCopy) {
    if (!opencl_available_) {
        GTEST_SKIP() << "OpenCL not available";
    }

    ASSERT_TRUE(executor_->start());

    const size_t size = 1024 * sizeof(float);
    std::vector<float> host_data(1024, 1.0f);
    std::vector<float> result_data(1024, 0.0f);

    void* device_ptr = executor_->allocate_device_memory(size);
    ASSERT_NE(device_ptr, nullptr);

    EXPECT_TRUE(executor_->copy_to_device(device_ptr, host_data.data(), size));
    EXPECT_TRUE(executor_->copy_to_host(result_data.data(), device_ptr, size));

    for (size_t i = 0; i < 1024; ++i) {
        EXPECT_FLOAT_EQ(result_data[i], 1.0f);
    }

    executor_->free_device_memory(device_ptr);
}

TEST_F(OpenCLExecutorTest, StreamManagement) {
    if (!opencl_available_) {
        GTEST_SKIP() << "OpenCL not available";
    }

    ASSERT_TRUE(executor_->start());

    int stream_id = executor_->create_stream();
    EXPECT_GE(stream_id, 0);

    executor_->synchronize_stream(stream_id);
    executor_->destroy_stream(stream_id);
}

TEST_F(OpenCLExecutorTest, KernelSubmission) {
    if (!opencl_available_) {
        GTEST_SKIP() << "OpenCL not available";
    }

    ASSERT_TRUE(executor_->start());

    GpuTaskConfig config;
    config.stream_id = 0;

    bool executed = false;
    auto future = executor_->submit_kernel(
        [&executed](void*) { executed = true; },
        config
    );

    future.wait();
    EXPECT_TRUE(executed);
}

TEST_F(OpenCLExecutorTest, Status) {
    if (!opencl_available_) {
        GTEST_SKIP() << "OpenCL not available";
    }

    ASSERT_TRUE(executor_->start());

    auto status = executor_->get_status();
    EXPECT_EQ(status.name, "test_opencl");
    EXPECT_TRUE(status.is_running);
    EXPECT_EQ(status.backend, GpuBackend::OPENCL);
}
