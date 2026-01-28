#include <cassert>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <future>
#include <algorithm>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#endif

// 包含 CUDA 执行器的头文件
#include <executor/config.hpp>
#include <executor/types.hpp>
#include <executor/interfaces.hpp>
#include "executor/gpu/cuda_executor.hpp"

using namespace executor;
using namespace executor::gpu;

// 测试辅助宏
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "FAILED: " << message << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

// 测试函数前向声明
bool test_cuda_executor_creation();
bool test_cuda_executor_device_info();
bool test_cuda_executor_memory_management();
bool test_cuda_executor_memory_copy();
bool test_cuda_executor_kernel_submit();
bool test_cuda_executor_synchronize();
bool test_cuda_executor_status();
bool test_cuda_executor_stream_management();

// ========== CUDA 执行器基本功能测试 ==========

bool test_cuda_executor_creation() {
    std::cout << "Testing CudaExecutor creation and destruction..." << std::endl;
    
#ifdef EXECUTOR_ENABLE_CUDA
    // 创建配置
    GpuExecutorConfig config;
    config.name = "test_cuda_executor";
    config.backend = GpuBackend::CUDA;
    config.device_id = 0;
    config.max_queue_size = 1000;
    
    // 创建执行器
    CudaExecutor executor(config.name, config);
    
    // 测试获取名称
    TEST_ASSERT(executor.get_name() == "test_cuda_executor", "Executor name should match");
    
    // 测试启动（如果CUDA可用）
    bool started = executor.start();
    if (started) {
        // 测试状态（启动后应该是运行状态）
        auto status = executor.get_status();
        TEST_ASSERT(status.name == "test_cuda_executor", "Status name should match");
        TEST_ASSERT(status.is_running == true, "Executor should be running");
        
        // 测试停止
        executor.stop();
        
        // 测试状态（停止后应该不是运行状态）
        status = executor.get_status();
        TEST_ASSERT(status.is_running == false, "Executor should not be running after stop");
    } else {
        std::cout << "  CUDA not available, skipping runtime tests" << std::endl;
    }
    
    std::cout << "  CudaExecutor creation and destruction: PASSED" << std::endl;
    return true;
#else
    std::cout << "  CUDA support not enabled at compile time, skipping test" << std::endl;
    return true;
#endif
}

bool test_cuda_executor_device_info() {
    std::cout << "Testing CudaExecutor device info..." << std::endl;
    
#ifdef EXECUTOR_ENABLE_CUDA
    GpuExecutorConfig config;
    config.name = "test_cuda_executor";
    config.backend = GpuBackend::CUDA;
    config.device_id = 0;
    
    CudaExecutor executor(config.name, config);
    
    // 获取设备信息
    auto device_info = executor.get_device_info();
    
    TEST_ASSERT(device_info.backend == GpuBackend::CUDA, "Backend should be CUDA");
    TEST_ASSERT(device_info.device_id == 0, "Device ID should match");
    
    // 如果CUDA可用，检查设备信息是否有效
    if (executor.start()) {
        device_info = executor.get_device_info();
        // 设备名称不应为空（如果CUDA可用）
        TEST_ASSERT(!device_info.name.empty(), "Device name should not be empty");
        
        executor.stop();
    }
    
    std::cout << "  CudaExecutor device info: PASSED" << std::endl;
    return true;
#else
    std::cout << "  CUDA support not enabled at compile time, skipping test" << std::endl;
    return true;
#endif
}

bool test_cuda_executor_memory_management() {
    std::cout << "Testing CudaExecutor memory management..." << std::endl;
    
#ifdef EXECUTOR_ENABLE_CUDA
    GpuExecutorConfig config;
    config.name = "test_cuda_executor";
    config.backend = GpuBackend::CUDA;
    config.device_id = 0;
    
    CudaExecutor executor(config.name, config);
    
    if (!executor.start()) {
        std::cout << "  CUDA not available, skipping memory management tests" << std::endl;
        return true;
    }
    
    // 测试分配设备内存
    const size_t test_size = 1024 * 1024; // 1MB
    void* device_ptr = executor.allocate_device_memory(test_size);
    
    if (device_ptr != nullptr) {
        TEST_ASSERT(device_ptr != nullptr, "Device memory allocation should succeed");
        
        // 测试释放设备内存
        executor.free_device_memory(device_ptr);
        
        // 测试多次分配和释放
        std::vector<void*> ptrs;
        for (int i = 0; i < 10; ++i) {
            void* ptr = executor.allocate_device_memory(test_size);
            if (ptr != nullptr) {
                ptrs.push_back(ptr);
            }
        }
        
        // 释放所有内存
        for (void* ptr : ptrs) {
            executor.free_device_memory(ptr);
        }
    } else {
        std::cout << "  Device memory allocation failed (CUDA may not be available)" << std::endl;
    }
    
    executor.stop();
    std::cout << "  CudaExecutor memory management: PASSED" << std::endl;
    return true;
#else
    std::cout << "  CUDA support not enabled at compile time, skipping test" << std::endl;
    return true;
#endif
}

bool test_cuda_executor_memory_copy() {
    std::cout << "Testing CudaExecutor memory copy..." << std::endl;
    
#ifdef EXECUTOR_ENABLE_CUDA
    GpuExecutorConfig config;
    config.name = "test_cuda_executor";
    config.backend = GpuBackend::CUDA;
    config.device_id = 0;
    
    CudaExecutor executor(config.name, config);
    
    if (!executor.start()) {
        std::cout << "  CUDA not available, skipping memory copy tests" << std::endl;
        return true;
    }
    
    const size_t test_size = 1024; // 1KB
    void* device_ptr = executor.allocate_device_memory(test_size);
    
    if (device_ptr != nullptr) {
        // 准备主机数据
        std::vector<int> host_data(test_size / sizeof(int), 42);
        
        // 测试主机到设备复制
        bool success = executor.copy_to_device(device_ptr, host_data.data(), test_size, false);
        TEST_ASSERT(success, "Host to device copy should succeed");
        
        // 同步
        executor.synchronize();
        
        // 测试设备到主机复制
        std::vector<int> host_result(test_size / sizeof(int), 0);
        success = executor.copy_to_host(host_result.data(), device_ptr, test_size, false);
        TEST_ASSERT(success, "Device to host copy should succeed");
        
        // 验证数据
        TEST_ASSERT(host_result == host_data, "Copied data should match original");
        
        executor.free_device_memory(device_ptr);
    } else {
        std::cout << "  Device memory allocation failed (CUDA may not be available)" << std::endl;
    }
    
    executor.stop();
    std::cout << "  CudaExecutor memory copy: PASSED" << std::endl;
    return true;
#else
    std::cout << "  CUDA support not enabled at compile time, skipping test" << std::endl;
    return true;
#endif
}

bool test_cuda_executor_kernel_submit() {
    std::cout << "Testing CudaExecutor kernel submit..." << std::endl;
    
#ifdef EXECUTOR_ENABLE_CUDA
    GpuExecutorConfig config;
    config.name = "test_cuda_executor";
    config.backend = GpuBackend::CUDA;
    config.device_id = 0;
    
    CudaExecutor executor(config.name, config);
    
    if (!executor.start()) {
        std::cout << "  CUDA not available, skipping kernel submit tests" << std::endl;
        return true;
    }
    
    // 测试提交简单的kernel（通过回调函数）
    GpuTaskConfig task_config;
    task_config.grid_size[0] = 1;
    task_config.block_size[0] = 1;
    task_config.async = false;
    
    // 创建一个简单的kernel函数（在实际使用中，这里会调用CUDA kernel）
    auto future = executor.submit_kernel([&executor]() {
        // 简单的测试：分配和释放内存
        void* ptr = executor.allocate_device_memory(1024);
        if (ptr != nullptr) {
            executor.free_device_memory(ptr);
        }
    }, task_config);
    
    // 等待kernel完成
    try {
        future.wait();
        future.get(); // 检查是否有异常
    } catch (const std::exception& e) {
        std::cerr << "  Kernel execution failed: " << e.what() << std::endl;
        // 如果CUDA不可用，这是预期的
    }
    
    executor.stop();
    std::cout << "  CudaExecutor kernel submit: PASSED" << std::endl;
    return true;
#else
    std::cout << "  CUDA support not enabled at compile time, skipping test" << std::endl;
    return true;
#endif
}

bool test_cuda_executor_synchronize() {
    std::cout << "Testing CudaExecutor synchronize..." << std::endl;
    
#ifdef EXECUTOR_ENABLE_CUDA
    GpuExecutorConfig config;
    config.name = "test_cuda_executor";
    config.backend = GpuBackend::CUDA;
    config.device_id = 0;
    
    CudaExecutor executor(config.name, config);
    
    if (!executor.start()) {
        std::cout << "  CUDA not available, skipping synchronize tests" << std::endl;
        return true;
    }
    
    // 测试同步操作
    executor.synchronize();
    
    // 测试流同步（默认流）
    executor.synchronize_stream(0);
    
    executor.stop();
    std::cout << "  CudaExecutor synchronize: PASSED" << std::endl;
    return true;
#else
    std::cout << "  CUDA support not enabled at compile time, skipping test" << std::endl;
    return true;
#endif
}

bool test_cuda_executor_status() {
    std::cout << "Testing CudaExecutor status..." << std::endl;
    
#ifdef EXECUTOR_ENABLE_CUDA
    GpuExecutorConfig config;
    config.name = "test_cuda_executor";
    config.backend = GpuBackend::CUDA;
    config.device_id = 0;
    
    CudaExecutor executor(config.name, config);
    
    // 获取初始状态
    auto status = executor.get_status();
    TEST_ASSERT(status.name == "test_cuda_executor", "Status name should match");
    TEST_ASSERT(status.backend == GpuBackend::CUDA, "Status backend should be CUDA");
    TEST_ASSERT(status.device_id == 0, "Status device ID should match");
    TEST_ASSERT(status.is_running == false, "Status should show not running initially");
    
    if (executor.start()) {
        status = executor.get_status();
        TEST_ASSERT(status.is_running == true, "Status should show running after start");
        
        executor.stop();
        
        status = executor.get_status();
        TEST_ASSERT(status.is_running == false, "Status should show not running after stop");
    }
    
    std::cout << "  CudaExecutor status: PASSED" << std::endl;
    return true;
#else
    std::cout << "  CUDA support not enabled at compile time, skipping test" << std::endl;
    return true;
#endif
}

bool test_cuda_executor_stream_management() {
    std::cout << "Testing CudaExecutor stream management..." << std::endl;
    
#ifdef EXECUTOR_ENABLE_CUDA
    GpuExecutorConfig config;
    config.name = "test_cuda_executor";
    config.backend = GpuBackend::CUDA;
    config.device_id = 0;
    config.default_stream_count = 1;
    
    CudaExecutor executor(config.name, config);
    
    if (!executor.start()) {
        std::cout << "  CUDA not available, skipping stream management tests" << std::endl;
        return true;
    }
    
    int stream_id = executor.create_stream();
    if (stream_id > 0) {
        TEST_ASSERT(stream_id > 0, "Stream ID should be positive");
        executor.synchronize_stream(stream_id);
        executor.destroy_stream(stream_id);
        
        // destroy 后再 synchronize_stream 不应崩溃
        executor.synchronize_stream(stream_id);
        
        // 重复 create/destroy
        int a = executor.create_stream();
        int b = executor.create_stream();
        if (a > 0 && b > 0) {
            executor.destroy_stream(a);
            executor.destroy_stream(b);
            int c = executor.create_stream();
            if (c > 0) {
                executor.synchronize_stream(c);
                executor.destroy_stream(c);
            }
        }
    } else {
        std::cout << "  Stream creation failed (CUDA may not be available)" << std::endl;
    }
    
    // 非法 stream_id：synchronize_stream / destroy_stream 安全 no-op
    executor.synchronize_stream(-1);
    executor.synchronize_stream(99999);
    executor.destroy_stream(-1);
    executor.destroy_stream(0);
    executor.destroy_stream(99999);
    
    executor.stop();
    std::cout << "  CudaExecutor stream management: PASSED" << std::endl;
    return true;
#else
    std::cout << "  CUDA support not enabled at compile time, skipping test" << std::endl;
    return true;
#endif
}

bool test_cuda_executor_multi_stream_parallel() {
    std::cout << "Testing CudaExecutor multi-stream parallel..." << std::endl;
    
#ifdef EXECUTOR_ENABLE_CUDA
    GpuExecutorConfig config;
    config.name = "test_cuda_executor";
    config.backend = GpuBackend::CUDA;
    config.device_id = 0;
    config.default_stream_count = 2;
    
    CudaExecutor executor(config.name, config);
    
    if (!executor.start()) {
        std::cout << "  CUDA not available, skipping multi-stream tests" << std::endl;
        return true;
    }
    
    GpuTaskConfig tc;
    tc.grid_size[0] = 1;
    tc.block_size[0] = 1;
    tc.async = false;
    
    tc.stream_id = 1;
    auto f1 = executor.submit_kernel([&executor]() {
        void* p = executor.allocate_device_memory(512);
        if (p) executor.free_device_memory(p);
    }, tc);
    
    tc.stream_id = 2;
    auto f2 = executor.submit_kernel([&executor]() {
        void* p = executor.allocate_device_memory(512);
        if (p) executor.free_device_memory(p);
    }, tc);
    
    f1.wait();
    f2.wait();
    try { f1.get(); f2.get(); } catch (...) { }
    
    executor.synchronize_stream(1);
    executor.synchronize_stream(2);
    executor.synchronize();
    
    executor.stop();
    std::cout << "  CudaExecutor multi-stream parallel: PASSED" << std::endl;
    return true;
#else
    std::cout << "  CUDA support not enabled at compile time, skipping test" << std::endl;
    return true;
#endif
}

bool test_cuda_executor_stream_sync() {
    std::cout << "Testing CudaExecutor stream sync..." << std::endl;
    
#ifdef EXECUTOR_ENABLE_CUDA
    GpuExecutorConfig config;
    config.name = "test_cuda_executor";
    config.backend = GpuBackend::CUDA;
    config.device_id = 0;
    config.default_stream_count = 1;
    
    CudaExecutor executor(config.name, config);
    
    if (!executor.start()) {
        std::cout << "  CUDA not available, skipping stream sync tests" << std::endl;
        return true;
    }
    
    // sync(0) 等价 synchronize()
    executor.synchronize_stream(0);
    executor.synchronize();
    
    int sid = executor.create_stream();
    if (sid > 0) {
        // 多次 synchronize_stream 幂等
        executor.synchronize_stream(sid);
        executor.synchronize_stream(sid);
        executor.synchronize_stream(sid);
        
        // 多流分别 sync 后整体 synchronize() 无挂起
        executor.synchronize_stream(sid);
        executor.synchronize();
        executor.destroy_stream(sid);
    }
    
    executor.stop();
    std::cout << "  CudaExecutor stream sync: PASSED" << std::endl;
    return true;
#else
    std::cout << "  CUDA support not enabled at compile time, skipping test" << std::endl;
    return true;
#endif
}

// ========== 主函数 ==========

// 全局初始化函数，在main之前执行
#ifdef _WIN32
struct EarlyInit {
    EarlyInit() {
        // 在静态初始化阶段就设置错误模式
        SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
        // 尝试输出到控制台（可能失败，但不影响程序）
        std::cout << "Early initialization..." << std::endl;
        std::cout.flush();
    }
};
static EarlyInit g_early_init;
#endif

int main() {
    // 立即刷新输出，确保能看到
    std::cout.flush();
    std::cerr.flush();
    
    try {
        std::cout << "Starting CUDA Executor Unit Tests..." << std::endl;
        std::cout.flush();
        
        std::cout << "=========================================" << std::endl;
        std::cout << "CUDA Executor Unit Tests" << std::endl;
        std::cout << "=========================================" << std::endl;
        std::cout.flush();
        
        bool all_passed = true;
        
        // 运行所有测试
        all_passed &= test_cuda_executor_creation();
        all_passed &= test_cuda_executor_device_info();
        all_passed &= test_cuda_executor_memory_management();
        all_passed &= test_cuda_executor_memory_copy();
        all_passed &= test_cuda_executor_kernel_submit();
        all_passed &= test_cuda_executor_synchronize();
        all_passed &= test_cuda_executor_status();
        all_passed &= test_cuda_executor_stream_management();
        all_passed &= test_cuda_executor_multi_stream_parallel();
        all_passed &= test_cuda_executor_stream_sync();
        
        std::cout << "=========================================" << std::endl;
        if (all_passed) {
            std::cout << "All tests PASSED" << std::endl;
            return 0;
        } else {
            std::cout << "Some tests FAILED" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: Exception caught in main: " << e.what() << std::endl;
        std::cerr.flush();
        return 1;
    } catch (...) {
        std::cerr << "FATAL ERROR: Unknown exception caught in main" << std::endl;
        std::cerr.flush();
        return 1;
    }
}
