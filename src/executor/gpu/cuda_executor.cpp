#include "cuda_executor.hpp"
#include "gpu_memory_manager.hpp"
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <functional>

#ifdef EXECUTOR_ENABLE_CUDA
#include <cuda_runtime.h>
#ifdef _WIN32
#include <windows.h>
#endif
#endif

namespace {

struct StreamCallbackContext {
    std::function<void()> callback;
};

void stream_host_callback(void* userData) {
    StreamCallbackContext* ctx = static_cast<StreamCallbackContext*>(userData);
    if (ctx && ctx->callback) {
        ctx->callback();
    }
    delete ctx;
}

}  // namespace

namespace executor {
namespace gpu {

CudaExecutor::CudaExecutor(const std::string& name, const GpuExecutorConfig& config)
    : name_(name)
    , config_(config)
    , device_id_(config.device_id)
    , is_available_(false)
    , loader_(&CudaLoader::instance())
#ifdef EXECUTOR_ENABLE_CUDA
    , default_stream_(nullptr)
#endif
{
    try {
        // 尝试加载CUDA DLL
        if (!loader_->load()) {
            is_available_ = false;
            return;
        }

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

    memory_manager_.reset();

#ifdef EXECUTOR_ENABLE_CUDA
    if (is_available_ && loader_->is_available()) {
        (void)ensure_device_context();  // 多 GPU 场景下确保操作本设备
        const auto& funcs = loader_->get_functions();
        
        // 释放所有已分配的内存（仅未使用内存池时 allocated_memory_ 非空）
        {
            std::lock_guard<std::mutex> lock(memory_mutex_);
            for (auto& [ptr, size] : allocated_memory_) {
                if (ptr != nullptr && funcs.cudaFree != nullptr) {
                    funcs.cudaFree(ptr);
                }
            }
            allocated_memory_.clear();
        }

        // 销毁所有流（除了默认流）
        {
            std::lock_guard<std::mutex> lock(streams_mutex_);
            for (auto stream : streams_) {
                if (stream != nullptr && funcs.cudaStreamDestroy != nullptr) {
                    funcs.cudaStreamDestroy(stream);
                }
            }
            streams_.clear();
        }
    }
#endif
}

bool CudaExecutor::check_cuda_available() {
#ifdef EXECUTOR_ENABLE_CUDA
    if (!loader_->is_available()) {
        return false;
    }

    const auto& funcs = loader_->get_functions();
    if (!funcs.is_complete()) {
        return false;
    }

    try {
        // 尝试初始化 CUDA 运行时（通过调用 cudaFree(0)）
        cudaError_t error = funcs.cudaFree(0);
        if (error != cudaSuccess) {
            return false;
        }

        // 检查设备数量
        int device_count = 0;
        error = funcs.cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0) {
            return false;
        }

        return true;
    } catch (...) {
        // 捕获所有C++异常
        return false;
    }
#else
    return false;
#endif
}

bool CudaExecutor::initialize_device() {
#ifdef EXECUTOR_ENABLE_CUDA
    if (!loader_->is_available()) {
        return false;
    }

    const auto& funcs = loader_->get_functions();
    if (!funcs.is_complete()) {
        return false;
    }

    try {
        // 检查设备ID是否有效
        int device_count = 0;
        cudaError_t error = funcs.cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess) {
            return false;
        }

        if (device_id_ < 0 || device_id_ >= device_count) {
            return false;
        }

        // 设置当前设备
        error = funcs.cudaSetDevice(device_id_);
        if (!check_cuda_error(error, "cudaSetDevice")) {
            return false;
        }

        // 获取设备属性
        error = funcs.cudaGetDeviceProperties(&device_prop_, device_id_);
        if (!check_cuda_error(error, "cudaGetDeviceProperties")) {
            return false;
        }

        // 默认流就是 nullptr（CUDA 默认流）
        default_stream_ = nullptr;

        return true;
    } catch (...) {
        // 捕获所有C++异常
        return false;
    }
#else
    return false;
#endif
}

#ifdef EXECUTOR_ENABLE_CUDA
bool CudaExecutor::check_cuda_error(cudaError_t error_code, const char* operation) const {
    (void)operation;  // 暂时未使用，保留用于未来错误日志
    if (error_code != cudaSuccess) {
        // 可以在这里记录错误日志
        // 暂时不抛出异常，返回 false 表示失败
        return false;
    }
    return true;
}
#else
bool CudaExecutor::check_cuda_error(int error_code, const char* operation) const {
    (void)error_code;
    (void)operation;
    return false;
}
#endif

bool CudaExecutor::ensure_device_context() const {
#ifdef EXECUTOR_ENABLE_CUDA
    if (!loader_->is_available()) {
        return false;
    }
    const auto& funcs = loader_->get_functions();
    if (funcs.cudaSetDevice == nullptr) {
        return false;
    }
    cudaError_t err = funcs.cudaSetDevice(device_id_);
    return check_cuda_error(err, "cudaSetDevice");
#else
    return false;
#endif
}

bool CudaExecutor::start() {
    if (!is_available_) {
        return false;
    }

    bool expected = false;
    if (!is_running_.compare_exchange_strong(expected, true)) {
        return false;  // 已经在运行
    }

#ifdef EXECUTOR_ENABLE_CUDA
    if (!loader_->is_available()) {
        is_running_.store(false);
        return false;
    }

    const auto& funcs = loader_->get_functions();
    if (funcs.cudaStreamCreate == nullptr || funcs.cudaStreamDestroy == nullptr) {
        is_running_.store(false);
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(streams_mutex_);
        if (streams_.empty() && config_.default_stream_count > 0) {
            for (int i = 0; i < config_.default_stream_count; ++i) {
                cudaStream_t s = create_one_stream();
                if (s == nullptr) {
                    for (cudaStream_t t : streams_) {
                        if (t != nullptr) {
                            funcs.cudaStreamDestroy(t);
                        }
                    }
                    streams_.clear();
                    is_running_.store(false);
                    return false;
                }
                streams_.push_back(s);
            }
        }
    }

    if (config_.memory_pool_size > 0) {
        memory_manager_ = std::make_unique<GpuMemoryManager>(
            [this](size_t sz) { return raw_allocate_device_memory(sz); },
            [this](void* p) { raw_free_device_memory(p); },
            config_.memory_pool_size);
    }
#endif

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

void* CudaExecutor::raw_allocate_device_memory(size_t size) {
#ifdef EXECUTOR_ENABLE_CUDA
    if (!is_available_ || !is_running_.load() || !loader_->is_available()) {
        return nullptr;
    }
    if (!ensure_device_context()) {
        return nullptr;
    }
    const auto& funcs = loader_->get_functions();
    if (funcs.cudaMalloc == nullptr) {
        return nullptr;
    }
    void* ptr = nullptr;
    cudaError_t error = funcs.cudaMalloc(&ptr, size);
    if (!check_cuda_error(error, "cudaMalloc") || ptr == nullptr) {
        return nullptr;
    }
    return ptr;
#else
    (void)size;
    return nullptr;
#endif
}

void CudaExecutor::raw_free_device_memory(void* ptr) {
#ifdef EXECUTOR_ENABLE_CUDA
    if (!is_available_ || ptr == nullptr || !loader_->is_available()) {
        return;
    }
    if (!ensure_device_context()) {
        return;
    }
    const auto& funcs = loader_->get_functions();
    if (funcs.cudaFree != nullptr) {
        cudaError_t error = funcs.cudaFree(ptr);
        check_cuda_error(error, "cudaFree");
    }
#else
    (void)ptr;
#endif
}

void* CudaExecutor::allocate_device_memory(size_t size) {
    if (!is_available_ || !is_running_.load() || !loader_->is_available()) {
        return nullptr;
    }

    if (memory_manager_) {
        return memory_manager_->allocate(size);
    }

#ifdef EXECUTOR_ENABLE_CUDA
    if (!ensure_device_context()) {
        return nullptr;
    }
    const auto& funcs = loader_->get_functions();
    if (funcs.cudaMalloc == nullptr) {
        return nullptr;
    }

    void* ptr = nullptr;
    cudaError_t error = funcs.cudaMalloc(&ptr, size);
    if (!check_cuda_error(error, "cudaMalloc") || ptr == nullptr) {
        return nullptr;
    }

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
    if (!is_available_ || ptr == nullptr || !loader_->is_available()) {
        return;
    }

    if (memory_manager_) {
        memory_manager_->free(ptr);
        return;
    }

#ifdef EXECUTOR_ENABLE_CUDA
    if (!ensure_device_context()) {
        return;
    }
    const auto& funcs = loader_->get_functions();
    if (funcs.cudaFree == nullptr) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        if (allocated_memory_.find(ptr) == allocated_memory_.end()) {
            return;
        }
        allocated_memory_.erase(ptr);
    }

    cudaError_t error = funcs.cudaFree(ptr);
    check_cuda_error(error, "cudaFree");
#else
    (void)ptr;
#endif
}

bool CudaExecutor::copy_to_device(void* dst, const void* src, size_t size, bool async, int stream_id) {
    if (!is_available_ || !is_running_.load() || dst == nullptr || src == nullptr || !loader_->is_available()) {
        return false;
    }

#ifdef EXECUTOR_ENABLE_CUDA
    if (!ensure_device_context()) {
        return false;
    }
    const auto& funcs = loader_->get_functions();
    cudaError_t error;
    if (async) {
        if (funcs.cudaMemcpyAsync == nullptr) {
            return false;
        }
        cudaStream_t stream = (stream_id == 0) ? get_default_stream() : get_stream(stream_id);
        if (stream_id != 0 && stream == nullptr) {
            return false;  // 无效 stream_id
        }
        error = funcs.cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
    } else {
        if (funcs.cudaMemcpy == nullptr) {
            return false;
        }
        error = funcs.cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    }
    return check_cuda_error(error, "cudaMemcpy (H2D)");
#else
    (void)dst;
    (void)src;
    (void)size;
    (void)async;
    (void)stream_id;
    return false;
#endif
}

bool CudaExecutor::copy_to_host(void* dst, const void* src, size_t size, bool async, int stream_id) {
    if (!is_available_ || !is_running_.load() || dst == nullptr || src == nullptr || !loader_->is_available()) {
        return false;
    }

#ifdef EXECUTOR_ENABLE_CUDA
    if (!ensure_device_context()) {
        return false;
    }
    const auto& funcs = loader_->get_functions();
    cudaError_t error;
    if (async) {
        if (funcs.cudaMemcpyAsync == nullptr) {
            return false;
        }
        cudaStream_t stream = (stream_id == 0) ? get_default_stream() : get_stream(stream_id);
        if (stream_id != 0 && stream == nullptr) {
            return false;  // 无效 stream_id
        }
        error = funcs.cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
    } else {
        if (funcs.cudaMemcpy == nullptr) {
            return false;
        }
        error = funcs.cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    }
    return check_cuda_error(error, "cudaMemcpy (D2H)");
#else
    (void)dst;
    (void)src;
    (void)size;
    (void)async;
    (void)stream_id;
    return false;
#endif
}

bool CudaExecutor::copy_device_to_device(void* dst, const void* src, size_t size, bool async, int stream_id) {
    if (!is_available_ || !is_running_.load() || dst == nullptr || src == nullptr || !loader_->is_available()) {
        return false;
    }

#ifdef EXECUTOR_ENABLE_CUDA
    if (!ensure_device_context()) {
        return false;
    }
    const auto& funcs = loader_->get_functions();
    cudaError_t error;
    if (async) {
        if (funcs.cudaMemcpyAsync == nullptr) {
            return false;
        }
        cudaStream_t stream = (stream_id == 0) ? get_default_stream() : get_stream(stream_id);
        if (stream_id != 0 && stream == nullptr) {
            return false;  // 无效 stream_id
        }
        error = funcs.cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
    } else {
        if (funcs.cudaMemcpy == nullptr) {
            return false;
        }
        error = funcs.cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    }
    return check_cuda_error(error, "cudaMemcpy (D2D)");
#else
    (void)dst;
    (void)src;
    (void)size;
    (void)async;
    (void)stream_id;
    return false;
#endif
}

bool CudaExecutor::copy_from_peer(IGpuExecutor* src_executor, const void* src_ptr, void* dst_ptr,
                                  size_t size, bool async, int stream_id) {
    if (!is_available_ || !is_running_.load() || !loader_->is_available() ||
        src_executor == nullptr || src_executor == this || src_ptr == nullptr || dst_ptr == nullptr) {
        return false;
    }

    const int dst_device = device_id_;
    const int src_device = src_executor->get_device_info().device_id;
    if (src_device == dst_device) {
        return false;  /* 同设备请使用 copy_device_to_device */
    }

#ifdef EXECUTOR_ENABLE_CUDA
    const auto& funcs = loader_->get_functions();
    auto p2p_log = [&funcs](const char* step, bool use_cuda_err) {
        if (funcs.cudaGetLastError && funcs.cudaGetErrorString) {
            cudaError_t e = funcs.cudaGetLastError();
            std::fprintf(stderr, "P2P copy_from_peer failed at %s: %s\n",
                step, use_cuda_err ? funcs.cudaGetErrorString(e) : "(see above)");
        }
    };

    if (!funcs.is_p2p_available()) {
        std::fprintf(stderr, "P2P copy_from_peer failed: P2P symbols not loaded (cudaMemcpyPeer etc.)\n");
        return false;
    }

    int can = 0;
    cudaError_t err = funcs.cudaDeviceCanAccessPeer(&can, dst_device, src_device);
    if (!check_cuda_error(err, "cudaDeviceCanAccessPeer")) {
        p2p_log("cudaDeviceCanAccessPeer", true);
        return false;
    }
    if (can != 1) {
        std::fprintf(stderr, "P2P copy_from_peer failed: device %d cannot access peer %d (CanAccessPeer=0)\n",
            dst_device, src_device);
        return false;
    }

    if (!ensure_device_context()) {
        p2p_log("ensure_device_context", true);
        return false;
    }

    err = funcs.cudaDeviceEnablePeerAccess(src_device, 0);
    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
        p2p_log("cudaDeviceEnablePeerAccess", true);
        return false;
    }

    if (async) {
        if (funcs.cudaMemcpyPeerAsync == nullptr) {
            std::fprintf(stderr, "P2P copy_from_peer failed: cudaMemcpyPeerAsync not available\n");
            return false;
        }
        cudaStream_t stream = (stream_id == 0) ? get_default_stream() : get_stream(stream_id);
        if (stream_id != 0 && stream == nullptr) {
            std::fprintf(stderr, "P2P copy_from_peer failed: invalid stream_id %d\n", stream_id);
            return false;
        }
        err = funcs.cudaMemcpyPeerAsync(dst_ptr, dst_device, src_ptr, src_device, size, stream);
    } else {
        if (funcs.cudaMemcpyPeer == nullptr) {
            std::fprintf(stderr, "P2P copy_from_peer failed: cudaMemcpyPeer not available\n");
            return false;
        }
        err = funcs.cudaMemcpyPeer(dst_ptr, dst_device, src_ptr, src_device, size);
    }
    if (!check_cuda_error(err, "cudaMemcpyPeer")) {
        p2p_log("cudaMemcpyPeer", true);
        return false;
    }
    return true;
#else
    (void)size;
    (void)async;
    (void)stream_id;
    return false;
#endif
}

bool CudaExecutor::add_stream_callback(int stream_id, std::function<void()> callback) {
    if (!is_available_ || !is_running_.load() || !callback || !loader_->is_available()) {
        return false;
    }

#ifdef EXECUTOR_ENABLE_CUDA
    if (!ensure_device_context()) {
        return false;
    }
    const auto& funcs = loader_->get_functions();
    if (funcs.cudaLaunchHostFunc == nullptr) {
        return false;
    }
    cudaStream_t stream = (stream_id == 0) ? get_default_stream() : get_stream(stream_id);
    if (stream_id != 0 && stream == nullptr) {
        return false;  // 无效 stream_id
    }
    StreamCallbackContext* ctx = new (std::nothrow) StreamCallbackContext{std::move(callback)};
    if (ctx == nullptr) {
        return false;
    }
    cudaError_t error = funcs.cudaLaunchHostFunc(stream, &stream_host_callback, ctx);
    if (!check_cuda_error(error, "cudaLaunchHostFunc")) {
        delete ctx;
        return false;
    }
    return true;
#else
    (void)stream_id;
    (void)callback;
    return false;
#endif
}

void CudaExecutor::synchronize() {
    if (!is_available_ || !loader_->is_available()) {
        return;
    }

#ifdef EXECUTOR_ENABLE_CUDA
    if (!ensure_device_context()) {
        return;
    }
    const auto& funcs = loader_->get_functions();
    if (funcs.cudaDeviceSynchronize != nullptr) {
        cudaError_t error = funcs.cudaDeviceSynchronize();
        check_cuda_error(error, "cudaDeviceSynchronize");
    }
#endif
}

void CudaExecutor::synchronize_stream(int stream_id) {
    if (!is_available_ || !loader_->is_available()) {
        return;
    }

#ifdef EXECUTOR_ENABLE_CUDA
    if (!ensure_device_context()) {
        return;
    }
    const auto& funcs = loader_->get_functions();
    if (funcs.cudaStreamSynchronize == nullptr) {
        return;
    }

    if (stream_id == 0) {
        // 默认流
        synchronize();
        return;
    }

    std::lock_guard<std::mutex> lock(streams_mutex_);
    if (stream_id > 0 && static_cast<size_t>(stream_id) <= streams_.size()) {
        cudaStream_t stream = streams_[stream_id - 1];
        if (stream != nullptr) {
            cudaError_t error = funcs.cudaStreamSynchronize(stream);
            check_cuda_error(error, "cudaStreamSynchronize");
        }
    }
#else
    (void)stream_id;
#endif
}

int CudaExecutor::create_stream() {
    if (!is_available_ || !loader_->is_available()) {
        return -1;
    }

#ifdef EXECUTOR_ENABLE_CUDA
    cudaStream_t stream = create_one_stream();
    if (stream == nullptr) {
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
    if (!is_available_ || stream_id <= 0 || !loader_->is_available()) {
        return;
    }

#ifdef EXECUTOR_ENABLE_CUDA
    if (!ensure_device_context()) {
        return;
    }
    const auto& funcs = loader_->get_functions();
    if (funcs.cudaStreamDestroy == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(streams_mutex_);
    if (static_cast<size_t>(stream_id) <= streams_.size()) {
        cudaStream_t stream = streams_[stream_id - 1];
        if (stream != nullptr) {
            funcs.cudaStreamDestroy(stream);
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
    if (loader_->is_available() && ensure_device_context()) {
        const auto& funcs = loader_->get_functions();
        if (funcs.cudaMemGetInfo != nullptr) {
            size_t free_mem = 0;
            size_t total_mem = 0;
            cudaError_t error = funcs.cudaMemGetInfo(&free_mem, &total_mem);
            if (error == cudaSuccess) {
                info.free_memory_bytes = free_mem;
                info.total_memory_bytes = total_mem;
            }
        }
    }
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
    status.queue_size = 0;  // 当前模型无待执行队列

    if (!is_available_) {
        return status;
    }

#ifdef EXECUTOR_ENABLE_CUDA
    if (loader_->is_available() && ensure_device_context()) {
        const auto& funcs = loader_->get_functions();
        
        // 获取内存使用情况
        if (funcs.cudaMemGetInfo != nullptr) {
            size_t free_mem = 0;
            size_t total_mem = 0;
            cudaError_t error = funcs.cudaMemGetInfo(&free_mem, &total_mem);
            if (error == cudaSuccess) {
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
            }
        }

        // 计算平均kernel执行时间
        size_t completed = completed_kernels_.load();
        if (completed > 0) {
            int64_t total_time = total_kernel_time_ns_.load();
            status.avg_kernel_time_ms = (static_cast<double>(total_time) / completed) / 1000000.0;
        }
    }
#endif

    return status;
}

std::future<void> CudaExecutor::submit_kernel_impl(
    std::function<void(void*)> kernel_func,
    const GpuTaskConfig& config) {
    
    auto promise = std::make_shared<std::promise<void>>();
    auto future = promise->get_future();

    if (!is_available_ || !is_running_.load()) {
        promise->set_exception(std::make_exception_ptr(
            std::runtime_error("CudaExecutor is not available or not running")));
        return future;
    }

    void* stream_ptr = nullptr;
#ifdef EXECUTOR_ENABLE_CUDA
    cudaStream_t s = get_stream(config.stream_id);
    stream_ptr = static_cast<void*>(s);
#endif

    active_kernels_++;

    std::thread([this, kernel_func = std::move(kernel_func), config, promise, stream_ptr]() mutable {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
#ifdef EXECUTOR_ENABLE_CUDA
            if (!ensure_device_context()) {
                promise->set_exception(std::make_exception_ptr(
                    std::runtime_error("CudaExecutor: ensure_device_context failed")));
                active_kernels_--;
                return;
            }
#endif
            kernel_func(stream_ptr);
#ifdef EXECUTOR_ENABLE_CUDA
            if (loader_->is_available() && !config.async) {
                synchronize_stream(config.stream_id);
            }
#endif
            promise->set_value();
            
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

cudaStream_t CudaExecutor::get_stream(int stream_id) const {
    if (stream_id == 0) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(streams_mutex_);
    if (stream_id < 1 || static_cast<size_t>(stream_id) > streams_.size()) {
        return nullptr;
    }
    return streams_[static_cast<size_t>(stream_id) - 1];
}

cudaStream_t CudaExecutor::create_one_stream() {
    if (!loader_->is_available()) {
        return nullptr;
    }
    if (!ensure_device_context()) {
        return nullptr;
    }
    const auto& funcs = loader_->get_functions();
    if (funcs.cudaStreamCreate == nullptr) {
        return nullptr;
    }
    cudaStream_t stream = nullptr;
    cudaError_t error = funcs.cudaStreamCreate(&stream);
    if (!check_cuda_error(error, "cudaStreamCreate") || stream == nullptr) {
        return nullptr;
    }
    return stream;
}
#endif

} // namespace gpu
} // namespace executor
