#pragma once

#include "types.hpp"
#include <string>
#include <future>
#include <functional>
#include <type_traits>

namespace executor {

/**
 * @brief 异步执行器接口（用于线程池）
 * 
 * 提供任务提交、状态查询、生命周期管理等功能
 */
class IAsyncExecutor {
public:
    virtual ~IAsyncExecutor() = default;

    /**
     * @brief 提交任务（返回Future）
     * 
     * @tparam F 可调用对象类型
     * @tparam Args 参数类型
     * @param f 可调用对象
     * @param args 参数
     * @return std::future 任务执行结果的future
     */
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;
        
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> result = task->get_future();
        submit_impl([task]() { (*task)(); });
        return result;
    }

    /**
     * @brief 获取执行器名称
     * @return 执行器名称
     */
    virtual std::string get_name() const = 0;

    /**
     * @brief 获取执行器状态
     * @return 异步执行器状态
     */
    virtual AsyncExecutorStatus get_status() const = 0;

    /**
     * @brief 启动执行器
     * @return 是否启动成功
     */
    virtual bool start() = 0;

    /**
     * @brief 停止执行器
     */
    virtual void stop() = 0;

    /**
     * @brief 等待所有任务完成
     */
    virtual void wait_for_completion() = 0;

    /**
     * @brief 提交优先级任务（返回Future）
     * 
     * @tparam F 可调用对象类型
     * @tparam Args 参数类型
     * @param priority 优先级（0=LOW, 1=NORMAL, 2=HIGH, 3=CRITICAL）
     * @param f 可调用对象
     * @param args 参数
     * @return std::future 任务执行结果的future
     */
    template<typename F, typename... Args>
    auto submit_priority(int priority, F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;
        
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> result = task->get_future();
        submit_priority_impl(priority, [task]() { (*task)(); });
        return result;
    }

protected:
    /**
     * @brief 提交任务实现（内部方法）
     * @param task 任务函数
     */
    virtual void submit_impl(std::function<void()> task) = 0;

    /**
     * @brief 提交优先级任务实现（内部方法）
     * @param priority 优先级（0=LOW, 1=NORMAL, 2=HIGH, 3=CRITICAL）
     * @param task 任务函数
     */
    virtual void submit_priority_impl(int /*priority*/, std::function<void()> task) {
        // 默认实现：忽略优先级，使用普通提交
        submit_impl(std::move(task));
    }
};

/**
 * @brief 实时执行器接口（用于专用实时线程）
 * 
 * 提供实时线程的启动、停止、任务推送等功能
 * 注意：实时执行器不提供 submit() 接口，因为实时线程是周期执行的
 */
class IRealtimeExecutor {
public:
    virtual ~IRealtimeExecutor() = default;

    /**
     * @brief 启动实时线程
     * @return 是否启动成功
     */
    virtual bool start() = 0;

    /**
     * @brief 停止实时线程
     */
    virtual void stop() = 0;

    /**
     * @brief 推送任务到无锁队列（在周期回调中处理）
     * 
     * 任务通过无锁队列传递，在实时线程的下一个周期回调中执行
     * @param task 任务函数
     */
    virtual void push_task(std::function<void()> task) = 0;

    /**
     * @brief 获取执行器名称
     * @return 执行器名称
     */
    virtual std::string get_name() const = 0;

    /**
     * @brief 获取执行器状态
     * @return 实时执行器状态
     */
    virtual RealtimeExecutorStatus get_status() const = 0;
};

/**
 * @brief 周期管理器接口（可选，用于更精确的周期控制和监控）
 * 
 * 如果不提供周期管理器，executor使用内置的简单周期实现（sleep_until）
 * 对于需要精确周期控制的场景，可以实现此接口并注入到RealtimeThreadConfig中
 */
class ICycleManager {
public:
    virtual ~ICycleManager() = default;

    /**
     * @brief 注册周期任务
     * 
     * @param name 周期任务名称
     * @param period_ns 周期（纳秒）
     * @param callback 周期回调函数
     * @return 是否注册成功
     */
    virtual bool register_cycle(const std::string& name,
                                int64_t period_ns,
                                std::function<void()> callback) = 0;

    /**
     * @brief 启动周期任务
     * 
     * @param name 周期任务名称
     * @return 是否启动成功
     */
    virtual bool start_cycle(const std::string& name) = 0;

    /**
     * @brief 停止周期任务
     * 
     * @param name 周期任务名称
     */
    virtual void stop_cycle(const std::string& name) = 0;

    /**
     * @brief 获取周期统计信息（可选）
     * 
     * @param name 周期任务名称
     * @return 周期统计信息
     */
    virtual CycleStatistics get_statistics(const std::string& name) const = 0;
};

/**
 * @brief GPU 执行器接口
 * 
 * 提供 GPU kernel 任务提交、内存管理、流管理等功能
 */
class IGpuExecutor {
public:
    virtual ~IGpuExecutor() = default;

    /**
     * @brief 提交 GPU kernel 任务
     * 
     * @tparam KernelFunc GPU kernel 函数类型
     * @param kernel GPU kernel 函数（类型擦除，通过回调执行）
     * @param config GPU 任务配置
     * @return std::future<void> 任务执行结果的 future
     */
    template<typename KernelFunc>
    auto submit_kernel(KernelFunc&& kernel, const gpu::GpuTaskConfig& config)
        -> std::future<void>;

    /**
     * @brief 分配设备内存
     * 
     * @param size 内存大小（字节）
     * @return 设备内存指针，失败返回 nullptr
     */
    virtual void* allocate_device_memory(size_t size) = 0;

    /**
     * @brief 释放设备内存
     * 
     * @param ptr 设备内存指针
     */
    virtual void free_device_memory(void* ptr) = 0;

    /**
     * @brief 从主机内存复制到设备内存
     * 
     * @param dst 目标设备内存指针
     * @param src 源主机内存指针
     * @param size 复制大小（字节）
     * @param async 是否异步复制
     * @param stream_id 流ID（async 时使用，0=默认流）
     * @return 是否成功
     */
    virtual bool copy_to_device(void* dst, const void* src, size_t size, bool async = false, int stream_id = 0) = 0;

    /**
     * @brief 从设备内存复制到主机内存
     * 
     * @param dst 目标主机内存指针
     * @param src 源设备内存指针
     * @param size 复制大小（字节）
     * @param async 是否异步复制
     * @param stream_id 流ID（async 时使用，0=默认流）
     * @return 是否成功
     */
    virtual bool copy_to_host(void* dst, const void* src, size_t size, bool async = false, int stream_id = 0) = 0;

    /**
     * @brief 在设备内存之间复制
     * 
     * @param dst 目标设备内存指针
     * @param src 源设备内存指针
     * @param size 复制大小（字节）
     * @param async 是否异步复制
     * @param stream_id 流ID（async 时使用，0=默认流）
     * @return 是否成功
     */
    virtual bool copy_device_to_device(void* dst, const void* src, size_t size, bool async = false, int stream_id = 0) = 0;

    /**
     * @brief 同步所有操作
     * 
     * 等待所有已提交的 GPU 操作完成
     */
    virtual void synchronize() = 0;

    /**
     * @brief 同步指定流
     * 
     * @param stream_id 流ID
     */
    virtual void synchronize_stream(int stream_id) = 0;

    /**
     * @brief 创建新的流
     * 
     * @return 流ID，失败返回 -1
     */
    virtual int create_stream() = 0;

    /**
     * @brief 销毁流
     * 
     * @param stream_id 流ID
     */
    virtual void destroy_stream(int stream_id) = 0;

    /**
     * @brief 在指定流上注册主机回调
     *
     * 当该流上此前排队的操作（包括异步 copy）都完成后，在适当时机调用 callback。
     *
     * @param stream_id 流ID（0=默认流）
     * @param callback 回调函数
     * @return 成功返回 true，无效 stream_id 或不可用时返回 false
     */
    virtual bool add_stream_callback(int stream_id, std::function<void()> callback) = 0;

    /**
     * @brief 从 peer 执行器所在设备拷贝到本执行器设备（P2P 设备间拷贝）
     *
     * dst_ptr 须由本执行器分配，src_ptr 须由 src_executor 分配。
     * 同设备请使用 copy_device_to_device。
     *
     * @param src_executor 源设备对应执行器
     * @param src_ptr 源设备内存指针
     * @param dst_ptr 本设备目标内存指针
     * @param size 拷贝字节数
     * @param async 是否异步
     * @param stream_id 流ID（async 时使用，0=默认流）
     * @return 是否成功；默认实现返回 false
     */
    virtual bool copy_from_peer(IGpuExecutor* src_executor, const void* src_ptr, void* dst_ptr,
                               size_t size, bool async = false, int stream_id = 0) {
        (void)src_executor;
        (void)src_ptr;
        (void)dst_ptr;
        (void)size;
        (void)async;
        (void)stream_id;
        return false;
    }

    /**
     * @brief 获取执行器名称
     * @return 执行器名称
     */
    virtual std::string get_name() const = 0;

    /**
     * @brief 获取 GPU 设备信息
     * @return GPU 设备信息
     */
    virtual gpu::GpuDeviceInfo get_device_info() const = 0;

    /**
     * @brief 获取执行器状态
     * @return GPU 执行器状态
     */
    virtual gpu::GpuExecutorStatus get_status() const = 0;

    /**
     * @brief 启动执行器
     * @return 是否启动成功
     */
    virtual bool start() = 0;

    /**
     * @brief 停止执行器
     */
    virtual void stop() = 0;

    /**
     * @brief 等待所有任务完成
     */
    virtual void wait_for_completion() = 0;

protected:
    /**
     * @brief 提交 GPU kernel 实现（内部方法）
     *
     * kernel_func 接收流句柄 void*：nullptr 表示默认流，非空表示后端流（如 cudaStream_t）。
     *
     * @param kernel_func GPU kernel 函数（类型擦除，接收 void* 流句柄）
     * @param config GPU 任务配置
     * @return std::future<void> 任务执行结果的 future
     */
    virtual std::future<void> submit_kernel_impl(
        std::function<void(void*)> kernel_func,
        const gpu::GpuTaskConfig& config) = 0;
};

// 模板方法实现
template<typename KernelFunc>
auto IGpuExecutor::submit_kernel(KernelFunc&& kernel, const gpu::GpuTaskConfig& config)
    -> std::future<void> {
    std::function<void(void*)> kernel_func;
    if constexpr (std::is_invocable_v<KernelFunc, void*>) {
        kernel_func = [kernel = std::forward<KernelFunc>(kernel)](void* stream) mutable {
            kernel(stream);
        };
    } else {
        kernel_func = [kernel = std::forward<KernelFunc>(kernel)](void* stream) mutable {
            (void)stream;
            kernel();
        };
    }
    return submit_kernel_impl(std::move(kernel_func), config);
}

} // namespace executor
