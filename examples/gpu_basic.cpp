/**
 * @file gpu_basic.cpp
 * @brief GPU 执行器基础示例：演示执行器注册、任务提交与内存管理
 *
 * 需使用 EXECUTOR_ENABLE_GPU=ON 且 EXECUTOR_ENABLE_CUDA=ON 构建。
 * 实际 kernel 启动由用户在 submit_gpu 的 lambda 内编写。
 */

#include <iostream>
#include <vector>
#include <executor/executor.hpp>

using namespace executor;

#ifdef EXECUTOR_ENABLE_GPU

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "GPU Executor Basic Example" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // ---------- 1. 初始化 Executor ----------
    ExecutorConfig config;
    config.min_threads = 2;
    config.max_threads = 4;
    config.queue_capacity = 1000;

    auto& exec = Executor::instance();
    if (!exec.initialize(config)) {
        std::cerr << "Failed to initialize executor" << std::endl;
        return 1;
    }
    std::cout << "Executor initialized successfully" << std::endl;
    std::cout << std::endl;

    // ---------- 2. GPU 执行器注册 ----------
    // 配置 CUDA 执行器：名称、后端、设备 ID、队列与流数量等
    gpu::GpuExecutorConfig gpu_config;
    gpu_config.name = "cuda0";
    gpu_config.backend = gpu::GpuBackend::CUDA;
    gpu_config.device_id = 0;
    gpu_config.max_queue_size = 1000;
    gpu_config.default_stream_count = 1;

    if (!exec.register_gpu_executor("cuda0", gpu_config)) {
        std::cout << "GPU not available or registration failed. Skipping GPU examples." << std::endl;
        exec.shutdown();
        return 0;
    }
    std::cout << "GPU executor \"cuda0\" registered successfully" << std::endl;
    std::cout << std::endl;

    // ---------- 示例一：基本 GPU 任务提交 ----------
    std::cout << "Example 1: Basic GPU task submission" << std::endl;
    {
        gpu::GpuTaskConfig task_config;
        task_config.grid_size[0] = 1;
        task_config.block_size[0] = 1;
        task_config.async = false;

        // 提交无参 callable；在实际应用中，此处应填入在 GPU 上执行的 kernel 调用逻辑
        auto future = exec.submit_gpu("cuda0",
            []() {
                // 占位：实际使用时在此调用 CUDA kernel，如
                //   my_kernel<<<grid, block, 0, stream>>>(...);
            },
            task_config);

        future.wait();
        try {
            future.get();
        } catch (const std::exception& e) {
            std::cerr << "  GPU task exception: " << e.what() << std::endl;
        }
        std::cout << "  GPU task completed" << std::endl;
    }
    std::cout << std::endl;

    // ---------- 示例二：GPU 内存管理 ----------
    std::cout << "Example 2: GPU memory management" << std::endl;
    {
        auto* gpu_exec = exec.get_gpu_executor("cuda0");
        if (!gpu_exec) {
            std::cerr << "  GPU executor not found" << std::endl;
            exec.shutdown();
            return 1;
        }

        const size_t n = 1024;
        const size_t size_bytes = n * sizeof(float);
        std::vector<float> host_src(n, 1.0f);
        std::vector<float> host_dst(n, 0.0f);

        // 分配设备内存
        void* d_ptr = gpu_exec->allocate_device_memory(size_bytes);
        if (!d_ptr) {
            std::cerr << "  Failed to allocate device memory" << std::endl;
            exec.shutdown();
            return 1;
        }

        // 主机 -> 设备（同步复制）
        if (!gpu_exec->copy_to_device(d_ptr, host_src.data(), size_bytes, false)) {
            std::cerr << "  copy_to_device failed" << std::endl;
            gpu_exec->free_device_memory(d_ptr);
            exec.shutdown();
            return 1;
        }

        // 提交“内核”（此处为占位，实际应在此执行 GPU kernel）
        gpu::GpuTaskConfig task_config;
        task_config.grid_size[0] = 1;
        task_config.block_size[0] = 1;
        task_config.async = false;

        auto future = exec.submit_gpu("cuda0",
            [d_ptr, size_bytes, gpu_exec]() {
                (void)d_ptr;
                (void)size_bytes;
                (void)gpu_exec;
                // 占位：在实际应用中在此启动 kernel 处理 d_ptr 指向的设备数据
            },
            task_config);

        future.wait();
        try {
            future.get();
        } catch (const std::exception& e) {
            std::cerr << "  GPU task exception: " << e.what() << std::endl;
            gpu_exec->free_device_memory(d_ptr);
            exec.shutdown();
            return 1;
        }

        // 设备 -> 主机
        if (!gpu_exec->copy_to_host(host_dst.data(), d_ptr, size_bytes, false)) {
            std::cerr << "  copy_to_host failed" << std::endl;
            gpu_exec->free_device_memory(d_ptr);
            exec.shutdown();
            return 1;
        }

        gpu_exec->free_device_memory(d_ptr);

        std::cout << "  Allocate -> copy_to_device -> submit_gpu -> copy_to_host -> free done"
                  << std::endl;
        std::cout << "  host_dst[0] = " << host_dst[0] << " (expected 1.0)" << std::endl;
    }
    std::cout << std::endl;

    // ---------- 3. 状态查询 ----------
    std::cout << "Example 3: GPU executor status" << std::endl;
    {
        auto status = exec.get_gpu_executor_status("cuda0");
        std::cout << "  Name: " << status.name << std::endl;
        std::cout << "  Is running: " << (status.is_running ? "true" : "false") << std::endl;
        std::cout << "  Device ID: " << status.device_id << std::endl;
        std::cout << "  Queue size: " << status.queue_size << std::endl;
        std::cout << "  Memory used: " << status.memory_used_bytes << " B" << std::endl;
        std::cout << "  Memory total: " << status.memory_total_bytes << " B" << std::endl;
    }
    std::cout << std::endl;

    exec.shutdown();
    std::cout << "Example completed successfully." << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}

#else

int main() {
    std::cout << "GPU support not enabled. Build with EXECUTOR_ENABLE_GPU=ON and "
                 "EXECUTOR_ENABLE_CUDA=ON." << std::endl;
    return 0;
}

#endif
