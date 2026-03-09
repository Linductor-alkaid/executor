#pragma once

#include "../../../include/executor/interfaces.hpp"
#include "../../../include/executor/config.hpp"
#include "../../../include/executor/types.hpp"
#include "opencl_loader.hpp"
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
#include <thread>

namespace executor {
namespace gpu {

class OpenCLExecutor : public IGpuExecutor {
public:
    OpenCLExecutor(const std::string& name, const GpuExecutorConfig& config);
    ~OpenCLExecutor();

    OpenCLExecutor(const OpenCLExecutor&) = delete;
    OpenCLExecutor& operator=(const OpenCLExecutor&) = delete;

    // IGpuExecutor 接口实现
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

protected:
    std::future<void> submit_kernel_impl(
        std::function<void(void*)> kernel_func,
        const GpuTaskConfig& config) override;

private:
    struct CommandQueueWrapper {
        cl_command_queue queue = nullptr;
        std::mutex mutex;
    };

    bool initialize_opencl();
    void cleanup();
    bool check_opencl_error(cl_int error, const char* operation);
    CommandQueueWrapper* get_queue(int stream_id);
    void worker_thread();

    std::string name_;
    GpuExecutorConfig config_;
    bool is_available_;
    OpenCLLoader* loader_;

    cl_platform_id platform_;
    cl_device_id device_;
    cl_context context_;
    std::vector<std::unique_ptr<CommandQueueWrapper>> queues_;
    mutable std::mutex queues_mutex_;

    std::unordered_map<void*, cl_mem> memory_map_;
    mutable std::mutex memory_mutex_;

    std::atomic<size_t> active_kernels_{0};
    std::atomic<size_t> completed_kernels_{0};
    std::atomic<size_t> failed_kernels_{0};
    std::atomic<int64_t> total_kernel_time_ns_{0};

    std::atomic<bool> running_{false};
    std::thread worker_;
    std::queue<std::packaged_task<void()>> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
};

} // namespace gpu
} // namespace executor
