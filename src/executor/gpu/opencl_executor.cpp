#include "opencl_executor.hpp"
#include <chrono>
#include <sstream>
#include <stdexcept>

namespace executor {
namespace gpu {

namespace {

constexpr const char* kInvalidOpenCLDeviceIdMessage = "OpenCL device_id must be >= 0";

const char* opencl_error_name(cl_int error) {
    switch (error) {
        case CL_SUCCESS:
            return "CL_SUCCESS";
        case -1:
            return "CL_DEVICE_NOT_FOUND";
        case -2:
            return "CL_DEVICE_NOT_AVAILABLE";
        case -3:
            return "CL_COMPILER_NOT_AVAILABLE";
        case -4:
            return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5:
            return "CL_OUT_OF_RESOURCES";
        case -6:
            return "CL_OUT_OF_HOST_MEMORY";
        case -30:
            return "CL_INVALID_VALUE";
        case -36:
            return "CL_INVALID_COMMAND_QUEUE";
        case -38:
            return "CL_INVALID_MEM_OBJECT";
        case -48:
            return "CL_INVALID_KERNEL";
        case -52:
            return "CL_INVALID_KERNEL_ARGS";
        case -54:
            return "CL_INVALID_WORK_GROUP_SIZE";
        case -57:
            return "CL_INVALID_EVENT_WAIT_LIST";
        case -61:
            return "CL_INVALID_BUFFER_SIZE";
        default:
            return "CL_UNKNOWN_ERROR";
    }
}

}  // namespace

OpenCLExecutor::OpenCLExecutor(const std::string& name, const GpuExecutorConfig& config)
    : name_(name)
    , config_(config)
    , is_available_(false)
    , loader_(&OpenCLLoader::instance())
    , platform_(nullptr)
    , device_(nullptr)
    , context_(nullptr) {
    validate_config();
}

OpenCLExecutor::~OpenCLExecutor() {
    stop();
    cleanup();
}

bool OpenCLExecutor::start() {
    if (!validate_config()) {
        return false;
    }

    if (running_) {
        return true;
    }

    clear_last_error();

    if (!loader_->load()) {
        set_last_error("OpenCL loader is unavailable");
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
    queue_not_full_cv_.notify_all();

    // P-260625-002 fix: signal + join all submit_kernel_after waiter threads
    // before tearing down internal state. Mirrors the same P-002 fix on
    // CudaExecutor (commit 159ab55) to eliminate the UAF where a detached
    // waiter accesses this->submit_kernel_impl() after ~OpenCLExecutor()
    // releases context_/queues_/streams_ via cleanup().
    join_pending_waiters();

    if (worker_.joinable()) {
        worker_.join();
    }
}

void OpenCLExecutor::wait_for_completion() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_drained_cv_.wait(lock, [this] {
            return task_queue_.empty() && active_kernels_.load() == 0;
        });
    }

    synchronize();
}

bool OpenCLExecutor::initialize_opencl() {
    if (!validate_config()) {
        return false;
    }

    auto funcs = loader_->get_functions();
    cl_int err;

    // 获取平台
    cl_uint num_platforms;
    err = funcs.clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        set_last_error("OpenCL platform enumeration failed");
        return false;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    err = funcs.clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        set_last_error("OpenCL platform list retrieval failed");
        return false;
    }
    platform_ = platforms[0];

    // 获取GPU设备
    cl_uint num_devices;
    err = funcs.clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
        set_last_error("OpenCL GPU device enumeration failed");
        return false;
    }

    std::vector<cl_device_id> devices(num_devices);
    err = funcs.clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
    if (err != CL_SUCCESS) {
        set_last_error("OpenCL GPU device list retrieval failed");
        return false;
    }

    if (config_.device_id < 0) {
        set_last_error(kInvalidOpenCLDeviceIdMessage);
        return false;
    }

    if (config_.device_id >= static_cast<int>(num_devices)) {
        set_last_error("OpenCL device_id is out of range");
        return false;
    }
    device_ = devices[config_.device_id];

    // 创建上下文
    context_ = funcs.clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        set_last_error("OpenCL context creation failed");
        return false;
    }

    // 创建默认命令队列
    for (int i = 0; i < config_.default_stream_count; ++i) {
        auto queue_wrapper = std::make_shared<CommandQueueWrapper>();
        queue_wrapper->queue = funcs.clCreateCommandQueue(context_, device_, 0, &err);
        if (err != CL_SUCCESS) {
            cleanup();
            set_last_error("OpenCL command queue creation failed");
            return false;
        }
        queues_.push_back(std::move(queue_wrapper));
    }

    is_available_ = true;
    clear_last_error();
    return true;
}

bool OpenCLExecutor::validate_config() {
    if (config_.device_id < 0) {
        set_last_error(kInvalidOpenCLDeviceIdMessage);
        return false;
    }
    return true;
}

void OpenCLExecutor::clear_last_error() {
    std::lock_guard<std::mutex> lock(error_mutex_);
    last_error_message_.clear();
}

void OpenCLExecutor::set_last_error(const std::string& message) {
    std::lock_guard<std::mutex> lock(error_mutex_);
    last_error_message_ = message;
}

std::string OpenCLExecutor::get_last_error() const {
    std::lock_guard<std::mutex> lock(error_mutex_);
    return last_error_message_;
}

void OpenCLExecutor::cleanup() {
    auto funcs = loader_->get_functions();
    std::vector<std::shared_ptr<CommandQueueWrapper>> detached_queues;

    // 释放内存
    {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        for (auto& [ptr, allocation] : memory_map_) {
            if (allocation.buffer) {
                funcs.clReleaseMemObject(allocation.buffer);
            }
        }
        memory_map_.clear();
    }

    // 释放命令队列
    {
        std::lock_guard<std::mutex> lock(queues_mutex_);
        detached_queues.swap(queues_);
    }

    for (auto& queue_wrapper : detached_queues) {
        if (!queue_wrapper) {
            continue;
        }
        std::lock_guard<std::mutex> queue_lock(queue_wrapper->mutex);
        if (queue_wrapper->queue) {
            funcs.clReleaseCommandQueue(queue_wrapper->queue);
            queue_wrapper->queue = nullptr;
        }
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

    auto funcs = loader_->get_functions();
    cl_int err;
    cl_mem buffer = funcs.clCreateBuffer(context_, CL_MEM_READ_WRITE, size, nullptr, &err);

    if (err != CL_SUCCESS || !buffer) {
        return nullptr;
    }

    void* ptr = reinterpret_cast<void*>(buffer);

    std::lock_guard<std::mutex> lock(memory_mutex_);
    memory_map_[ptr] = MemoryAllocation{buffer, size};

    return ptr;
}

void OpenCLExecutor::free_device_memory(void* ptr) {
    if (!ptr || !is_available_) {
        return;
    }

    std::lock_guard<std::mutex> lock(memory_mutex_);
    auto it = memory_map_.find(ptr);
    if (it != memory_map_.end()) {
        auto funcs = loader_->get_functions();
        funcs.clReleaseMemObject(it->second.buffer);
        memory_map_.erase(it);
    }
}

bool OpenCLExecutor::copy_to_device(void* dst, const void* src, size_t size, bool async, int stream_id) {
    if (!is_available_) {
        return false;
    }

    auto queue_wrapper = get_queue(stream_id);
    if (!queue_wrapper) {
        return false;
    }

    std::lock_guard<std::mutex> lock(memory_mutex_);
    auto it = memory_map_.find(dst);
    if (it == memory_map_.end()) {
        return false;
    }

    auto funcs = loader_->get_functions();
    std::lock_guard<std::mutex> queue_lock(queue_wrapper->mutex);
    if (!queue_wrapper->queue) {
        return false;
    }
    cl_int err = funcs.clEnqueueWriteBuffer(
        queue_wrapper->queue, it->second.buffer, async ? CL_FALSE : CL_TRUE,
        0, size, src, 0, nullptr, nullptr);

    return check_opencl_error(err, "clEnqueueWriteBuffer");
}

bool OpenCLExecutor::copy_to_host(void* dst, const void* src, size_t size, bool async, int stream_id) {
    if (!is_available_) {
        return false;
    }

    auto queue_wrapper = get_queue(stream_id);
    if (!queue_wrapper) {
        return false;
    }

    std::lock_guard<std::mutex> lock(memory_mutex_);
    auto it = memory_map_.find(const_cast<void*>(src));
    if (it == memory_map_.end()) {
        return false;
    }

    auto funcs = loader_->get_functions();
    std::lock_guard<std::mutex> queue_lock(queue_wrapper->mutex);
    if (!queue_wrapper->queue) {
        return false;
    }
    cl_int err = funcs.clEnqueueReadBuffer(
        queue_wrapper->queue, it->second.buffer, async ? CL_FALSE : CL_TRUE,
        0, size, dst, 0, nullptr, nullptr);

    return check_opencl_error(err, "clEnqueueReadBuffer");
}

bool OpenCLExecutor::copy_device_to_device(void* dst, const void* src, size_t size, bool async, int stream_id) {
    if (!is_available_) {
        return false;
    }

    auto queue_wrapper = get_queue(stream_id);
    if (!queue_wrapper) {
        return false;
    }

    std::lock_guard<std::mutex> lock(memory_mutex_);
    auto src_it = memory_map_.find(const_cast<void*>(src));
    auto dst_it = memory_map_.find(dst);

    if (src_it == memory_map_.end() || dst_it == memory_map_.end()) {
        return false;
    }

    auto funcs = loader_->get_functions();
    std::lock_guard<std::mutex> queue_lock(queue_wrapper->mutex);
    if (!queue_wrapper->queue) {
        return false;
    }
    cl_int err = funcs.clEnqueueCopyBuffer(
        queue_wrapper->queue, src_it->second.buffer, dst_it->second.buffer,
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

    auto funcs = loader_->get_functions();
    std::vector<std::shared_ptr<CommandQueueWrapper>> queue_snapshot;

    {
        std::lock_guard<std::mutex> lock(queues_mutex_);
        queue_snapshot = queues_;
    }

    for (auto& queue_wrapper : queue_snapshot) {
        if (queue_wrapper) {
            std::lock_guard<std::mutex> queue_lock(queue_wrapper->mutex);
            if (!queue_wrapper->queue) {
                continue;
            }
            check_opencl_error(funcs.clFinish(queue_wrapper->queue), "clFinish");
        }
    }
}

void OpenCLExecutor::synchronize_stream(int stream_id) {
    auto queue_wrapper = get_queue(stream_id);
    if (!queue_wrapper) {
        return;
    }

    auto funcs = loader_->get_functions();
    std::lock_guard<std::mutex> lock(queue_wrapper->mutex);
    if (!queue_wrapper->queue) {
        return;
    }
    check_opencl_error(funcs.clFinish(queue_wrapper->queue), "clFinish");
}

int OpenCLExecutor::create_stream() {
    if (!is_available_) {
        return -1;
    }

    auto funcs = loader_->get_functions();
    cl_int err;

    auto queue_wrapper = std::make_shared<CommandQueueWrapper>();
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
    if (stream_id < static_cast<int>(config_.default_stream_count)) {
        return;
    }

    std::shared_ptr<CommandQueueWrapper> queue_wrapper;

    {
        std::lock_guard<std::mutex> lock(queues_mutex_);
        if (stream_id >= 0 && stream_id < static_cast<int>(queues_.size())) {
            queue_wrapper = std::move(queues_[stream_id]);
            queues_[stream_id].reset();
        }
    }

    if (queue_wrapper) {
        auto funcs = loader_->get_functions();
        std::lock_guard<std::mutex> queue_lock(queue_wrapper->mutex);
        if (queue_wrapper->queue) {
            funcs.clReleaseCommandQueue(queue_wrapper->queue);
            queue_wrapper->queue = nullptr;
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
        auto funcs = loader_->get_functions();
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
    status.last_error_message = get_last_error();
    status.active_kernels = active_kernels_.load();
    status.completed_kernels = completed_kernels_.load();
    status.failed_kernels = failed_kernels_.load();
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        status.queue_size = task_queue_.size();
    }
    {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        for (const auto& [ptr, allocation] : memory_map_) {
            (void)ptr;
            status.memory_used_bytes += allocation.size;
        }
    }

    if (is_available_ && device_) {
        auto funcs = loader_->get_functions();
        cl_ulong mem_size = 0;
        cl_int err = funcs.clGetDeviceInfo(
            device_, 0x101F, sizeof(mem_size), &mem_size, nullptr); // CL_DEVICE_GLOBAL_MEM_SIZE
        if (err == CL_SUCCESS) {
            status.memory_total_bytes = static_cast<size_t>(mem_size);
            if (status.memory_total_bytes > 0) {
                status.memory_usage_percent =
                    (static_cast<double>(status.memory_used_bytes) /
                     static_cast<double>(status.memory_total_bytes)) * 100.0;
            }
        }
    }

    size_t completed = completed_kernels_.load();
    if (completed > 0) {
        status.avg_kernel_time_ms =
            static_cast<double>(total_kernel_time_ns_.load()) / 1000000.0 /
            static_cast<double>(completed);
    }

    return status;
}

std::future<void> OpenCLExecutor::submit_kernel_impl(
    std::function<void(void*)> kernel_func,
    const GpuTaskConfig& config) {

    std::packaged_task<void()> task([this, kernel_func, config]() {
        auto start = std::chrono::high_resolution_clock::now();

        try {
            auto queue_wrapper = get_queue(config.stream_id);
            if (!queue_wrapper) {
                throw std::runtime_error("OpenCLExecutor: invalid stream_id");
            }

            std::lock_guard<std::mutex> queue_lock(queue_wrapper->mutex);
            if (!queue_wrapper->queue) {
                throw std::runtime_error("OpenCLExecutor: invalid stream_id");
            }
            kernel_func(queue_wrapper->queue);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            total_kernel_time_ns_ += duration;
            completed_kernels_++;
        } catch (...) {
            failed_kernels_++;
            throw;
        }
    });

    auto future = task.get_future();

    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_not_full_cv_.wait(lock, [this] {
            return !running_.load(std::memory_order_acquire) ||
                   task_queue_.size() < config_.max_queue_size;
        });
        if (!running_) {
            std::promise<void> promise;
            promise.set_exception(std::make_exception_ptr(
                std::runtime_error("OpenCLExecutor is not running")));
            return promise.get_future();
        }

        task_queue_.push(std::move(task));
    }
    queue_cv_.notify_one();

    return future;
}

void OpenCLExecutor::worker_thread() {
    while (true) {
        std::packaged_task<void()> task;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return !task_queue_.empty() || !running_; });

            if (!running_ && task_queue_.empty()) {
                queue_drained_cv_.notify_all();
                break;
            }

            task = std::move(task_queue_.front());
            task_queue_.pop();
            active_kernels_++;
            queue_not_full_cv_.notify_one();
        }

        if (task.valid()) {
            task();
        }

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            active_kernels_--;
            if (task_queue_.empty() && active_kernels_.load() == 0) {
                queue_drained_cv_.notify_all();
            }
        }
    }
}

std::shared_ptr<OpenCLExecutor::CommandQueueWrapper> OpenCLExecutor::get_queue(int stream_id) {
    std::lock_guard<std::mutex> lock(queues_mutex_);

    if (stream_id < 0 || stream_id >= static_cast<int>(queues_.size())) {
        return nullptr;
    }

    return queues_[stream_id];
}

bool OpenCLExecutor::check_opencl_error(cl_int error, const char* operation) {
    if (error == CL_SUCCESS) {
        return true;
    }

    std::ostringstream oss;
    oss << (operation && operation[0] != '\0' ? operation : "OpenCL operation")
        << ": " << opencl_error_name(error)
        << " (" << error << ")";
    set_last_error(oss.str());
    return false;
}

} // namespace gpu
} // namespace executor
