#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <string>

#define private public
#include "executor/gpu/cuda_loader.hpp"
#include "executor/gpu/opencl_loader.hpp"
#undef private

using executor::gpu::CudaLoader;
using executor::gpu::OpenCLLoader;

namespace {

class LoaderLibraryPathScope {
public:
    LoaderLibraryPathScope() {
        const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
        directory_ = std::filesystem::temp_directory_path() /
            ("executor_loader_failure_" + std::to_string(timestamp));
        std::filesystem::create_directories(directory_ / "lib64");
        std::filesystem::create_directories(directory_ / "lib");
        std::filesystem::create_symlink("/lib/x86_64-linux-gnu/libc.so.6",
                                        directory_ / "lib64/libcudart.so");
        std::filesystem::create_symlink("/lib/x86_64-linux-gnu/libc.so.6",
                                        directory_ / "lib/libOpenCL.so");
        setenv("CUDA_PATH", directory_.c_str(), 1);
        setenv("OPENCL_PATH", directory_.c_str(), 1);
    }

    ~LoaderLibraryPathScope() {
        std::error_code error;
        std::filesystem::remove_all(directory_, error);
    }

private:
    std::filesystem::path directory_;
};

template <typename Loader>
void verify_failed_load_can_retry(Loader& loader) {
    loader.unload();
    LoaderLibraryPathScope library_path;
    loader.function_resolver_ = [](const char*) { return nullptr; };

    auto failed_load = std::async(std::launch::async, [&loader] { return loader.load(); });
    ASSERT_EQ(failed_load.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_FALSE(failed_load.get());
    EXPECT_EQ(loader.dll_handle_, nullptr);
    EXPECT_FALSE(loader.get_functions().is_complete());

    loader.function_resolver_ = [](const char*) { return reinterpret_cast<void*>(&std::rand); };
    EXPECT_TRUE(loader.load());
    loader.unload();
    loader.function_resolver_ = {};
}

}  // namespace

TEST(CudaLoaderTest, LoaderLoadFunctionsFailureDoesNotDeadlock) {
    verify_failed_load_can_retry(CudaLoader::instance());
}

TEST(OpenCLLoaderTest, LoaderLoadFunctionsFailureDoesNotDeadlock) {
    verify_failed_load_can_retry(OpenCLLoader::instance());
}
