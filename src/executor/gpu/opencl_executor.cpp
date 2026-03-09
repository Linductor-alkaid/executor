#include "opencl_executor.hpp"
#include <chrono>
#include <sstream>

namespace executor {
namespace gpu {

OpenCLExecutor::OpenCLExecutor(const std::string& name, const GpuExecutorConfig& config)
    : name_(name)
    , config_(config)
    , is_available_(false)
    , loader_(&OpenCLLoader::instance())
    , platform_(nullptr)
    , device_(nullptr)
    , context_(nullptr) {
}

OpenCLExecutor::~OpenCLExecutor() {
    stop();
    cleanup();
}

bool OpenCLExecutor::start() {
    if (running_) {
        return true;
    }

    if (!loader_->load()) {
        return false;
    }

    if (!initialize_opencl()) {
        return false;
    }

    running_ = true;
    worker_ = std::thread(&OpenCLExecutor::worker_thread, this);
    return true;
}

void OpenCLExecutor::stop() {
    if (!running_) {
        return;
    }

    running_ = false;
    queue_cv_.notify_all();

    if (worker_.joinable()) {
        worker_.join();
    }
}

void OpenCLExecutor::wait_for_completion() {
    synchronize();
}

bool OpenCLExecutor::initialize_opencl() {
    auto& funcs = loader_->get_functions();
    cl_int err;

    // 获取平台
    cl_uint num_platforms;
    err = funcs.clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        return false;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    err = funcs.clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        return false;
    }
    platform_ = platforms[0];

    // 获取GPU设备
    cl_uint num_devices;
    err = funcs.clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
        return false;
    }

    std::vector<cl_device_id> devices(num_devices);
    err = funcs.clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
    if (err != CL_SUCCESS) {
        return false;
    }

    if (config_.device_id >= static_cast<int>(num_devices)) {
        return false;
    }
    device_ = devices[config_.device_id];

    // 创建上下文
    context_ = funcs.clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        return false;
    }

    // 创建默认命令队列
    for (int i = 0; i < config_.default_stream_count; ++i) {
        auto queue_wrapper = std::make_unique<CommandQueueWrapper>();
        queue_wrapper->queue = funcs.clCreateCommandQueue(context_, device_, 0, &err);
        if (err != CL_SUCCESS) {
            cleanup();
            return false;
        }
        queues_.push_back(std::move(queue_wrapper));
    }

    is_available_ = true;
    return true;
}

void OpenCLExecutor::cleanup() {
    auto& funcs = loader_->get_functions();

    // 释放内存
    {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        for (auto& [ptr, mem] : memory_map_) {
            if (mem) {
                funcs.clReleaseMemObject(mem);
            }
        }
        memory_map_.clear();
    }

    // 释放命令队列
    {
        std::lock_guard<std::mutex> lock(queues_mutex_);
        for (auto& queue_wrapper : queues_) {
            if (queue_wrapper && queue_wrapper->queue) {
                funcs.clReleaseCommandQueue(queue_wrapper->queue);
            }
        }
        queues_.clear();
    }

    // 释放上下文
    if (context_) {
        funcs.clReleaseContext(context_);
        context_ = nullptr;
    }

    device_ = nullptr;
    platform_ = nullptr;
    is_available_ = false;
}

void* OpenCLExecutor::allocate_device_memory(size_t size) {
    if (!is_available_) {
        return nullptr;
    }

    auto& funcs = loader_->get_functions();
    cl_int err;
    cl_mem buffer = funcs.clCreateBuffer(context_, CL_MEM_READ_WRITE, size, nullptr, &err);

    if (err != CL_SUCCESS || !buffer) {
        return nullptr;
    }

    void* ptr = reinterpret_cast<void*>(buffer);

    std::lock_guard<std::mutex> lock(memory_mutex_);
    memory_map_[ptr] = buffer;

    return ptr;
}

void OpenCLExecutor::free_device_memory(void* ptr) {
    if (!ptr || !is_available_) {
        return;
    }

    std::lock_guard<std::mutex> lock(memory_mutex_);
    auto it = memory_map_.find(ptr);
    if (it != memory_map_.end()) {
        auto& funcs = loader_->get_functions();
        funcs.clReleaseMemObject(it->second);
        memory_map_.erase(it);
    }
}

bool OpenCLExecutor::copy_to_device(void* dst, const void* src, size_t size, bool async, int stream_id) {
    if (!is_available_) {
        return false;
    }

    auto* queue_wrapper = get_queue(stream_id);
    if (!queue_wrapper) {
        return false;
    }

    std::lock_guard<std::mutex> lock(memory_mutex_);
    auto it = memory_map_.find(dst);
    if (it == memory_map_.end()) {
        return false;
    }

    auto& funcs = loader_->get_functions();
    std::lock_guard<std::mutex> queue_lock(queue_wrapper->mutex);
    cl_int err = funcs.clEnqueueWriteBuffer(
        queue_wrapper->queue, it->second, async ? CL_FALSE : CL_TRUE,
        0, size, src, 0, nullptr, nullptr);

    return check_opencl_error(err, "clEnqueueWriteBuffer");
}

bool OpenCLExecutor::copy_to_host(void* dst, const void* src, size_t size, bool async, int stream_id) {
    if (!is_available_) {
        return false;
    }

    auto* queue_wrapper = get_queue(stream_id);
    if (!queue_wrapper) {
        return false;
    }

    std::lock_guard<std::mutex> lock(memory_mutex_);
    auto it = memory_map_.find(const_cast<void*>(src));
    if (it == memory_map_.end()) {
        return false;
    }

    auto& funcs = loader_->get_functions();
    std::lock_guard<std::mutex> queue_lock(queue_wrapper->mutex);
    cl_int err = funcs.clEnqueueReadBuffer(
        queue_wrapper->queue, it->second, async ? CL_FALSE : CL_TRUE,
        0, size, dst, 0, nullptr, nullptr);

    return check_opencl_error(err, "clEnqueueReadBuffer");
}

bool OpenCLExecutor::copy_device_to_device(void* dst, const void* src, size_t size, bool async, int stream_id) {
    if (!is_available_) {
        return false;
    }

    auto* queue_wrapper = get_queue(stream_id);
    if (!queue_wrapper) {
        return false;
    }

    std::lock_guard<std::mutex> lock(memory_mutex_);
    auto src_it = memory_map_.find(const_cast<void*>(src));
    auto dst_it = memory_map_.find(dst);

    if (src_it == memory_map_.end() || dst_it == memory_map_.end()) {
        return false;
    }

    auto& funcs = loader_->get_functions();
    std::lock_guard<std::mutex> queue_lock(queue_wrapper->mutex);
    cl_int err = funcs.clEnqueueCopyBuffer(
        queue_wrapper->queue, src_it->second, dst_it->second,
        0, 0, size, 0, nullptr, nullptr);

    return check_opencl_error(err, "clEnqueueCopyBuffer");
}

bool OpenCLExecutor::copy_from_peer(IGpuExecutor* src_executor, const void* src_ptr,
                                    void* dst_ptr, size_t size, bool async, int stream_id) {
    // OpenCL不直接支持P2P，需要通过主机内存中转
    return false;
}

void OpenCLExecutor::synchronize() {
    if (!is_available_) {
        return;
    }

    auto& funcs = loader_->get_functions();
    std::lock_guard<std::mutex> lock(queues_mutex_);

    for (auto& queue_wrapper : queues_) {
        if (queue_wrapper && queue_wrapper->queue) {
            std::lock_guard<std::mutex> queue_lock(queue_wrapper->mutex);
            funcs.clFinish(queue_wrapper->queue);
        }
    }
}

void OpenCLExecutor::synchronize_stream(int stream_id) {
    auto* queue_wrapper = get_queue(stream_id);
    if (!queue_wrapper) {
        return;
    }

    auto& funcs = loader_->get_functions();
    std::lock_guard<std::mutex> lock(queue_wrapper->mutex);
    funcs.clFinish(queue_wrapper->queue);
}

int OpenCLExecutor::create_stream() {
    if (!is_available_) {
        return -1;
    }

    auto& funcs = loader_->get_functions();
    cl_int err;

    auto queue_wrapper = std::make_unique<CommandQueueWrapper>();
    queue_wrapper->queue = funcs.clCreateCommandQueue(context_, device_, 0, &err);

    if (err != CL_SUCCESS) {
        return -1;
    }

    std::lock_guard<std::mutex> lock(queues_mutex_);
    int stream_id = static_cast<int>(queues_.size());
    queues_.push_back(std::move(queue_wrapper));

    return stream_id;
}

void OpenCLExecutor::destroy_stream(int stream_id) {
    if (stream_id < 0 || stream_id >= static_cast<int>(config_.default_stream_count)) {
        std::lock_guard<std::mutex> lock(queues_mutex_);
        if (stream_id >= 0 && stream_id < static_cast<int>(queues_.size())) {
            auto& queue_wrapper = queues_[stream_id];
            if (queue_wrapper && queue_wrapper->queue) {
                auto& funcs = loader_->get_functions();
                std::lock_guard<std::mutex> queue_lock(queue_wrapper->mutex);
                funcs.clReleaseCommandQueue(queue_wrapper->queue);
                queue_wrapper->queue = nullptr;
            }
        }
    }
}

bool OpenCLExecutor::add_stream_callback(int stream_id, std::function<void()> callback) {
    // OpenCL 1.x不直接支持回调，需要通过事件轮询实现
    return false;
}

std::string OpenCLExecutor::get_name() const {
    return name_;
}

GpuDeviceInfo OpenCLExecutor::get_device_info() const {
    GpuDeviceInfo info;
    info.name = name_;
    info.backend = GpuBackend::OPENCL;
    info.device_id = config_.device_id;

    if (is_available_ && device_) {
        auto& funcs = loader_->get_functions();
        char device_name[256] = {0};
        funcs.clGetDeviceInfo(device_, 0x102B, sizeof(device_name), device_name, nullptr); // CL_DEVICE_NAME
        info.name = device_name;

        cl_ulong mem_size = 0;
        funcs.clGetDeviceInfo(device_, 0x101F, sizeof(mem_size), &mem_size, nullptr); // CL_DEVICE_GLOBAL_MEM_SIZE
        info.total_memory_bytes = static_cast<size_t>(mem_size);
    }

    return info;
}

GpuExecutorStatus OpenCLExecutor::get_status() const {
    GpuExecutorStatus status;
    status.name = name_;
    status.is_running = running_;
    status.backend = GpuBackend::OPENCL;
    status.device_id = config_.device_id;
    status.active_kernels = active_kernels_.load();
    status.completed_kernels = completed_kernels_.load();
    status.failed_kernels = failed_kernels_.load();

    if (completed_kernels_ > 0) {
        status.avg_kernel_time_ms = total_kernel_time_ns_.load() / 1000000.0 / completed_kernels_.load();
    }

    return status;
}

std::future<void> OpenCLExecutor::submit_kernel_impl(
    std::function<void(void*)> kernel_func,
    const GpuTaskConfig& config) {

    std::packaged_task<void()> task([this, kernel_func, config]() {
        auto start = std::chrono::high_resolution_clock::now();
        active_kernels_++;

        try {
            auto* queue_wrapper = get_queue(config.stream_id);
            if (queue_wrapper) {
                kernel_func(queue_wrapper->queue);
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            total_kernel_time_ns_ += duration;
            completed_kernels_++;
        } catch (...) {
            failed_kernels_++;
        }

        active_kernels_--;
    });

    auto future = task.get_future();

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        task_queue_.push(std::move(task));
    }
    queue_cv_.notify_one();

    return future;
}

void OpenCLExecutor::worker_thread() {
    while (running_) {
        std::packaged_task<void()> task;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return !task_queue_.empty() || !running_; });

            if (!running_ && task_queue_.empty()) {
                break;
            }

            if (!task_queue_.empty()) {
                task = std::move(task_queue_.front());
                task_queue_.pop();
            }
        }

        if (task.valid()) {
            task();
        }
    }
}

OpenCLExecutor::CommandQueueWrapper* OpenCLExecutor::get_queue(int stream_id) {
    std::lock_guard<std::mutex> lock(queues_mutex_);

    if (stream_id < 0 || stream_id >= static_cast<int>(queues_.size())) {
        return queues_.empty() ? nullptr : queues_[0].get();
    }

    return queues_[stream_id].get();
}

bool OpenCLExecutor::check_opencl_error(cl_int error, const char* operation) {
    if (error == CL_SUCCESS) {
        return true;
    }
    return false;
}

} // namespace gpu
} // namespace executor
