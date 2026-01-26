#include "cuda_executor.hpp"
#include <chrono>
#include <stdexcept>
#include <thread>

#ifdef EXECUTOR_ENABLE_CUDA
#include <cuda_runtime.h>
#ifdef _WIN32
#include <windows.h>
#endif
#endif

namespace executor {
namespace gpu {

CudaExecutor::CudaExecutor(const std::string& name, const GpuExecutorConfig& config)
    : name_(name)
    , config_(config)
    , device_id_(config.device_id)
    , is_available_(false)
#ifdef EXECUTOR_ENABLE_CUDA
    , default_stream_(nullptr)
#endif
{
    try {
        // 检查 CUDA 是否可用
        if (!check_cuda_available()) {
            is_available_ = false;
            return;
        }

        // 初始化设备
        if (!initialize_device()) {
            is_available_ = false;
            return;
        }

        is_available_ = true;
    } catch (...) {
        // 捕获所有异常，确保构造函数不会抛出
        is_available_ = false;
    }
}

CudaExecutor::~CudaExecutor() {
    stop();

#ifdef EXECUTOR_ENABLE_CUDA
    // 释放所有已分配的内存
    {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        for (auto& [ptr, size] : allocated_memory_) {
            if (ptr != nullptr) {
                cudaFree(ptr);
            }
        }
        allocated_memory_.clear();
    }

    // 销毁所有流（除了默认流）
    {
        std::lock_guard<std::mutex> lock(streams_mutex_);
        for (auto stream : streams_) {
            if (stream != nullptr) {
                cudaStreamDestroy(stream);
            }
        }
        streams_.clear();
    }
#endif
}

bool CudaExecutor::check_cuda_available() {
#ifdef EXECUTOR_ENABLE_CUDA
#ifdef _WIN32
    // Windows: 先检查DLL是否可用（延迟加载时，DLL可能不存在）
    // 使用LoadLibrary检查DLL是否存在
    HMODULE cuda_dll = nullptr;
    // 尝试加载常见的CUDA DLL名称
    const char* cuda_dll_names[] = {
        "cudart64_12.dll",
        "cudart64_11.dll",
        "cudart64_10.dll",
        "cudart64_9.dll"
    };
    
    bool dll_found = false;
    for (const char* dll_name : cuda_dll_names) {
        cuda_dll = LoadLibraryA(dll_name);
        if (cuda_dll != nullptr) {
            dll_found = true;
            FreeLibrary(cuda_dll);  // 释放，让延迟加载机制接管
            break;
        }
    }
    
    if (!dll_found) {
        // DLL不存在，CUDA不可用
        return false;
    }
#endif  // _WIN32

    try {
        // 尝试初始化 CUDA 运行时（通过调用 cudaFree(0)）
        // 注意：即使DLL存在，如果CUDA驱动不可用，这个调用也可能失败
        // 延迟加载失败时，这个调用可能会抛出异常或导致程序崩溃
        // 我们已经通过LoadLibrary检查了DLL是否存在，所以这里应该相对安全
        cudaError_t error = cudaFree(0);
        if (error != cudaSuccess) {
            return false;
        }

        // 检查设备数量
        int device_count = 0;
        error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0) {
            return false;
        }

        return true;
    } catch (...) {
        // 捕获所有C++异常（包括可能的延迟加载异常）
        return false;
    }
#else
    return false;
#endif
}

bool CudaExecutor::initialize_device() {
#ifdef EXECUTOR_ENABLE_CUDA
    try {
        // 检查设备ID是否有效
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess) {
            return false;
        }

        if (device_id_ < 0 || device_id_ >= device_count) {
            return false;
        }

        // 设置当前设备
        error = cudaSetDevice(device_id_);
        if (!check_cuda_error(error, "cudaSetDevice")) {
            return false;
        }

        // 获取设备属性
        error = cudaGetDeviceProperties(&device_prop_, device_id_);
        if (!check_cuda_error(error, "cudaGetDeviceProperties")) {
            return false;
        }

        // 默认流就是 nullptr（CUDA 默认流）
        default_stream_ = nullptr;

        return true;
    } catch (...) {
        // 捕获所有C++异常（包括可能的延迟加载异常）
        return false;
    }
#else
    return false;
#endif
}

#ifdef EXECUTOR_ENABLE_CUDA
bool CudaExecutor::check_cuda_error(cudaError_t error_code, const char* operation) {
    (void)operation;  // 暂时未使用，保留用于未来错误日志
    if (error_code != cudaSuccess) {
        // 可以在这里记录错误日志
        // 暂时不抛出异常，返回 false 表示失败
        return false;
    }
    return true;
}
#else
bool CudaExecutor::check_cuda_error(int error_code, const char* operation) {
    (void)error_code;
    (void)operation;
    return false;
}
#endif

bool CudaExecutor::start() {
    if (!is_available_) {
        return false;
    }

    bool expected = false;
    if (!is_running_.compare_exchange_strong(expected, true)) {
        return false;  // 已经在运行
    }

    return true;
}

void CudaExecutor::stop() {
    if (!is_available_) {
        return;
    }

    is_running_.store(false);
    
    // 等待所有任务完成
    wait_for_completion();
}

void CudaExecutor::wait_for_completion() {
    if (!is_available_) {
        return;
    }

#ifdef EXECUTOR_ENABLE_CUDA
    synchronize();
#endif
}

void* CudaExecutor::allocate_device_memory(size_t size) {
    if (!is_available_ || !is_running_.load()) {
        return nullptr;
    }

#ifdef EXECUTOR_ENABLE_CUDA
    void* ptr = nullptr;
    cudaError_t error = cudaMalloc(&ptr, size);
    if (!check_cuda_error(error, "cudaMalloc") || ptr == nullptr) {
        return nullptr;
    }

    // 记录已分配的内存
    {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        allocated_memory_[ptr] = size;
    }

    return ptr;
#else
    (void)size;
    return nullptr;
#endif
}

void CudaExecutor::free_device_memory(void* ptr) {
    if (!is_available_ || ptr == nullptr) {
        return;
    }

#ifdef EXECUTOR_ENABLE_CUDA
    // 检查是否是我们分配的内存
    {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        if (allocated_memory_.find(ptr) == allocated_memory_.end()) {
            return;  // 不是我们分配的内存，忽略
        }
        allocated_memory_.erase(ptr);
    }

    cudaError_t error = cudaFree(ptr);
    check_cuda_error(error, "cudaFree");
#else
    (void)ptr;
#endif
}

bool CudaExecutor::copy_to_device(void* dst, const void* src, size_t size, bool async) {
    if (!is_available_ || !is_running_.load() || dst == nullptr || src == nullptr) {
        return false;
    }

#ifdef EXECUTOR_ENABLE_CUDA
    cudaError_t error;
    if (async) {
        cudaStream_t stream = get_default_stream();
        error = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
    } else {
        error = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    }
    return check_cuda_error(error, "cudaMemcpy (H2D)");
#else
    (void)dst;
    (void)src;
    (void)size;
    (void)async;
    return false;
#endif
}

bool CudaExecutor::copy_to_host(void* dst, const void* src, size_t size, bool async) {
    if (!is_available_ || !is_running_.load() || dst == nullptr || src == nullptr) {
        return false;
    }

#ifdef EXECUTOR_ENABLE_CUDA
    cudaError_t error;
    if (async) {
        cudaStream_t stream = get_default_stream();
        error = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
    } else {
        error = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    }
    return check_cuda_error(error, "cudaMemcpy (D2H)");
#else
    (void)dst;
    (void)src;
    (void)size;
    (void)async;
    return false;
#endif
}

bool CudaExecutor::copy_device_to_device(void* dst, const void* src, size_t size, bool async) {
    if (!is_available_ || !is_running_.load() || dst == nullptr || src == nullptr) {
        return false;
    }

#ifdef EXECUTOR_ENABLE_CUDA
    cudaError_t error;
    if (async) {
        cudaStream_t stream = get_default_stream();
        error = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
    } else {
        error = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    }
    return check_cuda_error(error, "cudaMemcpy (D2D)");
#else
    (void)dst;
    (void)src;
    (void)size;
    (void)async;
    return false;
#endif
}

void CudaExecutor::synchronize() {
    if (!is_available_) {
        return;
    }

#ifdef EXECUTOR_ENABLE_CUDA
    cudaError_t error = cudaDeviceSynchronize();
    check_cuda_error(error, "cudaDeviceSynchronize");
#endif
}

void CudaExecutor::synchronize_stream(int stream_id) {
    if (!is_available_) {
        return;
    }

#ifdef EXECUTOR_ENABLE_CUDA
    if (stream_id == 0) {
        // 默认流
        synchronize();
        return;
    }

    std::lock_guard<std::mutex> lock(streams_mutex_);
    if (stream_id > 0 && static_cast<size_t>(stream_id) <= streams_.size()) {
        cudaStream_t stream = streams_[stream_id - 1];
        if (stream != nullptr) {
            cudaError_t error = cudaStreamSynchronize(stream);
            check_cuda_error(error, "cudaStreamSynchronize");
        }
    }
#else
    (void)stream_id;
#endif
}

int CudaExecutor::create_stream() {
    if (!is_available_) {
        return -1;
    }

#ifdef EXECUTOR_ENABLE_CUDA
    cudaStream_t stream = nullptr;
    cudaError_t error = cudaStreamCreate(&stream);
    if (!check_cuda_error(error, "cudaStreamCreate") || stream == nullptr) {
        return -1;
    }

    std::lock_guard<std::mutex> lock(streams_mutex_);
    streams_.push_back(stream);
    // 返回流ID（从1开始，0是默认流）
    return static_cast<int>(streams_.size());
#else
    return -1;
#endif
}

void CudaExecutor::destroy_stream(int stream_id) {
    if (!is_available_ || stream_id <= 0) {
        return;
    }

#ifdef EXECUTOR_ENABLE_CUDA
    std::lock_guard<std::mutex> lock(streams_mutex_);
    if (static_cast<size_t>(stream_id) <= streams_.size()) {
        cudaStream_t stream = streams_[stream_id - 1];
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
            streams_[stream_id - 1] = nullptr;
        }
    }
#else
    (void)stream_id;
#endif
}

std::string CudaExecutor::get_name() const {
    return name_;
}

GpuDeviceInfo CudaExecutor::get_device_info() const {
    GpuDeviceInfo info;
    info.name = name_;
    info.backend = GpuBackend::CUDA;
    info.device_id = device_id_;

    if (!is_available_) {
        return info;
    }

#ifdef EXECUTOR_ENABLE_CUDA
    info.name = device_prop_.name;
    info.compute_capability_major = device_prop_.major;
    info.compute_capability_minor = device_prop_.minor;
    info.max_threads_per_block = device_prop_.maxThreadsPerBlock;
    info.max_blocks_per_grid[0] = device_prop_.maxGridSize[0];
    info.max_blocks_per_grid[1] = device_prop_.maxGridSize[1];
    info.max_blocks_per_grid[2] = device_prop_.maxGridSize[2];

    // 获取内存信息
    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    info.free_memory_bytes = free_mem;
    info.total_memory_bytes = total_mem;
#endif

    return info;
}

GpuExecutorStatus CudaExecutor::get_status() const {
    GpuExecutorStatus status;
    status.name = name_;
    status.is_running = is_running_.load();
    status.backend = GpuBackend::CUDA;
    status.device_id = device_id_;
    status.active_kernels = active_kernels_.load();
    status.completed_kernels = completed_kernels_.load();
    status.failed_kernels = failed_kernels_.load();

    if (!is_available_) {
        return status;
    }

#ifdef EXECUTOR_ENABLE_CUDA
    // 获取内存使用情况
    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    status.memory_total_bytes = total_mem;
    
    // 计算已使用内存
    {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        size_t allocated = 0;
        for (const auto& [ptr, size] : allocated_memory_) {
            allocated += size;
        }
        status.memory_used_bytes = allocated;
    }

    if (total_mem > 0) {
        status.memory_usage_percent = 
            (static_cast<double>(status.memory_used_bytes) / total_mem) * 100.0;
    }

    // 计算平均kernel执行时间
    size_t completed = completed_kernels_.load();
    if (completed > 0) {
        int64_t total_time = total_kernel_time_ns_.load();
        status.avg_kernel_time_ms = (static_cast<double>(total_time) / completed) / 1000000.0;
    }
#endif

    return status;
}

std::future<void> CudaExecutor::submit_kernel_impl(
    std::function<void()> kernel_func,
    const GpuTaskConfig& config) {
    
    auto promise = std::make_shared<std::promise<void>>();
    auto future = promise->get_future();

    if (!is_available_ || !is_running_.load()) {
        promise->set_exception(std::make_exception_ptr(
            std::runtime_error("CudaExecutor is not available or not running")));
        return future;
    }

    // 增加活跃kernel计数
    active_kernels_++;

    // 在后台线程中执行kernel（基础版本，使用默认流）
    std::thread([this, kernel_func = std::move(kernel_func), config, promise]() mutable {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
#ifdef EXECUTOR_ENABLE_CUDA
            // 设置当前设备
            cudaSetDevice(device_id_);
            
            // 执行kernel函数
            kernel_func();
            
            // 同步（确保kernel完成）
            if (!config.async) {
                synchronize();
            }
#endif
            promise->set_value();
            
            // 更新统计信息
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time).count();
            total_kernel_time_ns_ += duration;
            completed_kernels_++;
        } catch (...) {
            promise->set_exception(std::current_exception());
            failed_kernels_++;
        }
        
        active_kernels_--;
    }).detach();

    return future;
}

#ifdef EXECUTOR_ENABLE_CUDA
cudaStream_t CudaExecutor::get_default_stream() const {
    return default_stream_;  // nullptr 表示默认流
}
#endif

} // namespace gpu
} // namespace executor
