#pragma once

#include "../../../include/executor/interfaces.hpp"
#include "../../../include/executor/config.hpp"
#include "../../../include/executor/types.hpp"
#include "cuda_loader.hpp"
#include "gpu_memory_manager.hpp"
#include "../util/exception_handler.hpp"
#include <exception>
#include <memory>
#include <string>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <future>
#include <functional>
#include <queue>
#include <condition_variable>

#ifdef EXECUTOR_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace executor {
namespace gpu {

/**
 * @brief CUDA 执行器实现
 * 
 * 实现 IGpuExecutor 接口，提供 CUDA GPU 任务执行功能。
 * 支持构建时和运行时的 CUDA 可用性检测。
 */
class CudaExecutor : public IGpuExecutor {
public:
    /**
     * @brief 构造函数
     * 
     * @param name 执行器名称
     * @param config GPU 执行器配置
     */
    CudaExecutor(const std::string& name, const GpuExecutorConfig& config);

    /**
     * @brief 析构函数
     */
    ~CudaExecutor();

    // 禁止拷贝和移动
    CudaExecutor(const CudaExecutor&) = delete;
    CudaExecutor& operator=(const CudaExecutor&) = delete;
    CudaExecutor(CudaExecutor&&) = delete;
    CudaExecutor& operator=(CudaExecutor&&) = delete;

    // 实现 IGpuExecutor 接口
    void* allocate_device_memory(size_t size) override;
    void free_device_memory(void* ptr) override;
    bool copy_to_device(void* dst, const void* src, size_t size, bool async = false, int stream_id = 0) override;
    bool copy_to_host(void* dst, const void* src, size_t size, bool async = false, int stream_id = 0) override;
    bool copy_device_to_device(void* dst, const void* src, size_t size, bool async = false, int stream_id = 0) override;
    bool copy_from_peer(IGpuExecutor* src_executor, const void* src_ptr, void* dst_ptr,
                       size_t size, bool async = false, int stream_id = 0) override;
    bool add_stream_callback(int stream_id, std::function<void()> callback) override;
    void synchronize() override;
    void synchronize_stream(int stream_id) override;
    int create_stream() override;
    void destroy_stream(int stream_id) override;
    std::string get_name() const override;
    GpuDeviceInfo get_device_info() const override;
    GpuExecutorStatus get_status() const override;
    bool start() override;
    void stop() override;
    void wait_for_completion() override;

    /**
     * @brief 设置任务异常回调（与 CPU 执行器一致，用于监控/日志）
     * @param callback 回调函数，参数为执行器名称和异常指针
     */
    void set_exception_callback(
        std::function<void(const std::string&, std::exception_ptr)> callback);

protected:
    /**
     * @brief 提交 GPU kernel 实现（内部方法）
     */
    std::future<void> submit_kernel_impl(
        std::function<void(void*)> kernel_func,
        const GpuTaskConfig& config) override;

private:
    /**
     * @brief 检查 CUDA 运行时是否可用
     * @return 是否可用
     */
    bool check_cuda_available();

    /**
     * @brief 初始化 CUDA 设备
     * @return 是否初始化成功
     */
    bool initialize_device();

    /**
     * @brief 检查 CUDA 错误并记录
     * @param error_code CUDA 错误码
     * @param operation 操作名称
     * @return 是否成功
     */
#ifdef EXECUTOR_ENABLE_CUDA
    bool check_cuda_error(cudaError_t error_code, const char* operation) const;
#else
    bool check_cuda_error(int error_code, const char* operation) const;
#endif

    /**
     * @brief 获取默认流
     * @return 默认流句柄
     */
#ifdef EXECUTOR_ENABLE_CUDA
    cudaStream_t get_default_stream() const;
#endif

    /**
     * @brief 根据 stream_id 解析流句柄（0=默认流，1..N=streams_[id-1]）
     * @param stream_id 流ID
     * @return 流句柄，无效时返回 nullptr（表示默认流）
     */
#ifdef EXECUTOR_ENABLE_CUDA
    cudaStream_t get_stream(int stream_id) const;
#endif

    /**
     * @brief 创建单个流（不加锁，由调用方持有 streams_mutex_ 时使用）
     * @return 新流句柄，失败返回 nullptr
     */
#ifdef EXECUTOR_ENABLE_CUDA
    cudaStream_t create_one_stream();
#endif

    /**
     * @brief 确保当前 CUDA 上下文为本执行器对应设备（多 GPU 安全）
     * @return 是否成功
     */
    bool ensure_device_context() const;

    /**
     * @brief 底层设备内存分配（不经过内存池，不写入 allocated_memory_）
     * @param size 字节数
     * @return 设备指针，失败返回 nullptr
     */
    void* raw_allocate_device_memory(size_t size);

    /**
     * @brief 底层设备内存释放（不操作 allocated_memory_）
     * @param ptr 设备指针
     */
    void raw_free_device_memory(void* ptr);

    /**
     * @brief 将 CUDA 错误码转换为标准异常（用于 ExceptionHandler / future）
     * @param error_code CUDA 错误码
     * @param operation 操作名称
     * @return 异常指针，可传递给 handle_task_exception 或 promise::set_exception
     */
#ifdef EXECUTOR_ENABLE_CUDA
    std::exception_ptr make_cuda_exception_ptr(cudaError_t error_code, const char* operation) const;
#endif

private:
    std::string name_;                          // 执行器名称
    GpuExecutorConfig config_;                  // 配置
    int device_id_;                            // 设备ID
    bool is_available_;                        // CUDA是否可用
    std::atomic<bool> is_running_{false};      // 是否运行中
    CudaLoader* loader_;                       // CUDA加载器（单例引用）

#ifdef EXECUTOR_ENABLE_CUDA
    cudaDeviceProp device_prop_;               // 设备属性
    cudaStream_t default_stream_;              // 默认流
    std::vector<cudaStream_t> streams_;        // 流列表
    mutable std::mutex streams_mutex_;          // 流列表互斥锁（mutable用于const方法）
#endif

    // 内存管理
    std::unique_ptr<GpuMemoryManager> memory_manager_;    // 内存池（memory_pool_size > 0 时使用）
    std::unordered_map<void*, size_t> allocated_memory_;  // 已分配内存映射（仅未使用池时）
    mutable std::mutex memory_mutex_;                     // 内存映射互斥锁（mutable用于const方法）

    // 统计信息
    std::atomic<size_t> active_kernels_{0};     // 活跃kernel数
    std::atomic<size_t> completed_kernels_{0}; // 已完成kernel数
    std::atomic<size_t> failed_kernels_{0};    // 失败kernel数
    std::atomic<int64_t> total_kernel_time_ns_{0}; // 总kernel执行时间（纳秒）

    util::ExceptionHandler exception_handler_;  // 任务异常处理，与 CPU 执行器一致
};

} // namespace gpu
} // namespace executor
