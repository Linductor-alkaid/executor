#pragma once

#include "types.hpp"
#include <string>
#include <future>
#include <functional>
#include <type_traits>
#include <vector>
#include <thread>
#include <memory>
#include <mutex>
#include <atomic>
#include <stdexcept>
#include <utility>

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

        auto promise = std::make_shared<std::promise<return_type>>();
        auto bound_task = std::make_shared<decltype(std::bind(std::forward<F>(f), std::forward<Args>(args)...))>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> result = promise->get_future();
        auto task = [promise, bound_task]() mutable {
            try {
                if constexpr (std::is_void_v<return_type>) {
                    std::invoke(*bound_task);
                    promise->set_value();
                } else {
                    promise->set_value(std::invoke(*bound_task));
                }
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        };

        if (!try_submit_impl(std::move(task))) {
            promise->set_exception(std::make_exception_ptr(
                std::runtime_error("Executor is stopped")
            ));
        }

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

    /**
     * @brief 批量提交任务
     *
     * 批量提交多个任务，内部优化减少锁竞争。
     *
     * @tparam F 可调用对象类型
     * @param tasks 任务列表
     * @return std::vector<std::future<void>> 任务执行结果的 future 列表
     */
    template<typename F>
    std::vector<std::future<void>> submit_batch(const std::vector<F>& tasks) {
        std::vector<std::function<void()>> task_wrappers;
        std::vector<std::future<void>> futures;
        std::vector<std::shared_ptr<std::promise<void>>> promises;

        task_wrappers.reserve(tasks.size());
        futures.reserve(tasks.size());
        promises.reserve(tasks.size());

        // 准备所有任务和 future
        for (const auto& task : tasks) {
            auto promise = std::make_shared<std::promise<void>>();
            futures.push_back(promise->get_future());
            promises.push_back(promise);
            task_wrappers.push_back([promise, task]() {
                try {
                    task();
                    promise->set_value();
                } catch (...) {
                    promise->set_exception(std::current_exception());
                }
            });
        }

        // 批量提交（减少锁竞争）
        if (!try_submit_batch_impl(std::move(task_wrappers))) {
            for (auto& promise : promises) {
                promise->set_exception(std::make_exception_ptr(
                    std::runtime_error("ThreadPool is stopped")
                ));
            }
        }

        return futures;
    }

    /**
     * @brief 批量提交任务（无 future 版本）
     *
     * 批量提交多个任务，不返回 future，性能更高。
     *
     * @tparam F 可调用对象类型
     * @param tasks 任务列表
     */
    template<typename F>
    void submit_batch_no_future(const std::vector<F>& tasks) {
        std::vector<std::function<void()>> task_wrappers;
        task_wrappers.reserve(tasks.size());

        for (const auto& task : tasks) {
            task_wrappers.push_back(task);
        }

        submit_batch_impl(std::move(task_wrappers));
    }

protected:
    /**
     * @brief 提交任务实现（内部方法）
     * @param task 任务函数
     */
    virtual void submit_impl(std::function<void()> task) = 0;

    /**
     * @brief 提交任务实现（可报告拒绝）
     * @param task 任务函数
     * @return true 表示任务已被接受；false 表示执行器已拒绝任务
     */
    virtual bool try_submit_impl(std::function<void()> task) {
        try {
            submit_impl(std::move(task));
            return true;
        } catch (...) {
            return false;
        }
    }

    /**
     * @brief 提交优先级任务实现（内部方法）
     * @param priority 优先级（0=LOW, 1=NORMAL, 2=HIGH, 3=CRITICAL）
     * @param task 任务函数
     */
    virtual void submit_priority_impl(int priority, std::function<void()> task) {
        // 默认实现：忽略优先级，使用普通提交
        submit_impl(std::move(task));
    }

    /**
     * @brief 批量提交任务实现（内部方法）
     * @param tasks 任务列表
     */
    virtual void submit_batch_impl(std::vector<std::function<void()>> tasks) {
        // 默认实现：逐个提交
        for (auto& task : tasks) {
            submit_impl(std::move(task));
        }
    }

    /**
     * @brief 批量提交任务实现（可报告拒绝）
     * @param tasks 任务列表
     * @return true 表示任务已被接受；false 表示执行器已拒绝整批任务
     */
    virtual bool try_submit_batch_impl(std::vector<std::function<void()>> tasks) {
        submit_batch_impl(std::move(tasks));
        return true;
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
     * 任务通过无锁队列传递，在实时线程的下一个周期回调中执行。
     *
     * P-001 (260615) 破坏性约束说明: 接口签名保持 void 不变以保证 ABI/调用方兼容
     * (plan 提议的 bool 返回是破坏性的, 这里走"次优"方案 — 通过 get_status() 暴露
     *  dropped_task_count_, 并保留 push_task() 的 void 形态). 任务是否被丢弃
     * 必须通过 RealtimeExecutorStatus::dropped_task_count 与 failed_pushes 观察.
     *
     * @param task 任务函数
     */
    virtual void push_task(std::function<void()> task) = 0;

    /**
     * @brief 推送任务并回传是否成功 (P-001 260615, 非破坏扩展)
     *
     * 默认实现回退到 push_task(), 返回 true 仅表示任务已交给旧接口处理。
     * 派生类应 override 以直接返回 push 路径的实际成功/失败结果。
     *
     * @param task 任务函数
     * @return true 表示任务已被接受或已交给旧接口; false 表示派生类确认拒绝。
     */
    virtual bool push_task_ex(std::function<void()> task) {
        // 默认实现保持 ABI/源码兼容: 交给 void push_task(), 无法精确报告失败。
        // 派生类 override 可返回队列 push 的实际结果。
        push_task(std::move(task));
        return true;
    }

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
     * @brief 批量提交 GPU kernel 任务
     *
     * 在同一把锁内连续入队，减少多次 submit 时的锁竞争。
     *
     * @param tasks 任务列表，每项为 (kernel 函数, 配置)
     * @return 与 tasks 一一对应的 future 列表
     */
    virtual std::vector<std::future<void>> submit_kernels_batch(
        const std::vector<std::pair<std::function<void(void*)>, gpu::GpuTaskConfig>>& tasks);

    /**
     * @brief 在依赖完成后提交 GPU kernel 任务
     *
     * dependency 完成后才执行 kernel，priority 仍来自 config。
     *
     * @tparam KernelFunc GPU kernel 函数类型
     * @param dependency 依赖的 future，需先完成
     * @param kernel GPU kernel 函数
     * @param config GPU 任务配置
     * @return std::future<void> 任务执行结果的 future
     */
    template<typename KernelFunc>
    auto submit_kernel_after(std::shared_future<void> dependency, KernelFunc&& kernel,
                             const gpu::GpuTaskConfig& config) -> std::future<void>;

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
     * @brief 分配统一内存（Unified Memory）
     *
     * 统一内存可被 CPU 和 GPU 同时访问，无需显式传输。
     * 需要硬件和驱动支持（CUDA 6.0+）。
     *
     * @param size 内存大小（字节）
     * @return 统一内存指针，失败或不支持返回 nullptr
     */
    virtual void* allocate_unified_memory(size_t size) {
        (void)size;
        return nullptr;  // 默认不支持
    }

    /**
     * @brief 释放统一内存
     *
     * @param ptr 统一内存指针
     */
    virtual void free_unified_memory(void* ptr) {
        (void)ptr;
        // 默认不支持，什么也不做
    }

    /**
     * @brief 预取内存到指定设备
     *
     * 将统一内存预取到指定设备，优化访问性能。
     * device_id = cudaCpuDeviceId (-1) 表示预取到主机。
     *
     * @param ptr 统一内存指针
     * @param size 预取大小（字节）
     * @param device_id 目标设备ID（-1 表示主机）
     * @param stream_id 流ID（0=默认流）
     * @return 是否成功
     */
    virtual bool prefetch_memory(const void* ptr, size_t size, int device_id, int stream_id = 0) {
        (void)ptr;
        (void)size;
        (void)device_id;
        (void)stream_id;
        return false;  // 默认不支持
    }

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
     * @note EXPERIMENTAL - Not tested on real hardware.
     *       If you have a multi-GPU setup, please test and report issues.
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

    /**
     * @brief 等待并清理所有 submit_kernel_after 产生的等待线程。
     *
     * 派生类的 stop() 应在销毁内部状态前调用此方法，防止析构后
     * waiter 线程访问已销毁成员（UAF）。
     */
    void join_pending_waiters() {
        stopping_.store(true, std::memory_order_release);
        std::vector<std::thread> to_join;
        {
            std::lock_guard<std::mutex> lk(pending_waiters_mutex_);
            to_join.swap(pending_waiters_);
        }
        for (auto& t : to_join) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    // P-002 fix: tracks detached-waiter threads; joined in join_pending_waiters()
    std::mutex              pending_waiters_mutex_;
    std::vector<std::thread> pending_waiters_;
    std::atomic<bool>       stopping_{false};
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

inline std::vector<std::future<void>> IGpuExecutor::submit_kernels_batch(
    const std::vector<std::pair<std::function<void(void*)>, gpu::GpuTaskConfig>>& tasks) {
    std::vector<std::future<void>> result;
    result.reserve(tasks.size());
    for (const auto& p : tasks) {
        result.push_back(submit_kernel_impl(p.first, p.second));
    }
    return result;
}

template<typename KernelFunc>
auto IGpuExecutor::submit_kernel_after(std::shared_future<void> dependency, KernelFunc&& kernel,
                                       const gpu::GpuTaskConfig& config) -> std::future<void> {
    // P-005 fix: 旧实现在 worker 线程的 lambda 中直接 dep.wait(),
    // 会阻塞单一 GPU worker,后续无依赖任务全部饿死。
    //
    // 新实现: dep.wait() 移到独立 std::thread,等依赖完成后再通过
    // submit_kernel_impl 把真正的 kernel 重新入队,worker 线程立即
    // 可处理其它任务。
    //
    // 限制 (选项 B, 最小可行修复):
    //   - 每个 submit_kernel_after 启一个 std::thread。线程创建有
    //     固定开销 (~数十 us + 8MB stack),大规模依赖链 (>>千级)
    //     短暂 burst 场景下会比"阻塞 worker"更糟。计划在后续 P
    //     中用 CUDA event + 共享等待线程池(选项 A)替换。
    //   - 独立 thread 中调用 submit_kernel_impl 需要类成员上下文
    //     (protected 虚函数),此处合法,因为 lambda 继承自 IGpuExecutor。
    //   - 失败/超时: dep.wait() 在 std::future 上无超时 API;
    //     若 dep 永远不完成,独立线程会永久存活,但不会影响 worker。
    //     调用方应避免传入永远不完成的 future。
    std::shared_future<void> dep = std::move(dependency);

    // 构造"去 dep 等待"的 kernel lambda,这样重新入队时 worker 线程
    // 不会再次阻塞。
    std::function<void(void*)> kernel_func;
    if constexpr (std::is_invocable_v<KernelFunc, void*>) {
        kernel_func = [kernel = std::forward<KernelFunc>(kernel)](void* stream) mutable {
            kernel(stream);
        };
    } else {
        kernel_func = [kernel = std::forward<KernelFunc>(kernel)](void* /*stream*/) mutable {
            kernel();
        };
    }

    // 创建外层 promise/future — 调用方拿到的 future
    auto promise = std::make_shared<std::promise<void>>();
    auto result_future = promise->get_future();

    // 启动独立线程等待 dep,然后把 kernel 重新入队
    // P-002 fix: 线程存入 pending_waiters_,由 join_pending_waiters()
    // 在 stop() 时 join,防止析构后访问 this（UAF）。
    // waiter 在 dep.wait_for 循环中定期检查 stopping_,使 stop() 不会
    // 永久阻塞在永不完成的 dependency 上。
    std::thread waiter([dep, kernel_func = std::move(kernel_func), config,
                        promise, this]() mutable {
        try {
            // Poll-wait: 每 10ms 检查一次 stopping_ 标志，避免永久阻塞
            while (dep.wait_for(std::chrono::milliseconds(10)) !=
                   std::future_status::ready) {
                if (stopping_.load(std::memory_order_acquire)) {
                    promise->set_exception(std::make_exception_ptr(
                        std::runtime_error("executor stopped before dependency completed")));
                    return;
                }
            }
            if (stopping_.load(std::memory_order_acquire)) {
                promise->set_exception(std::make_exception_ptr(
                    std::runtime_error("executor stopped before dependency completed")));
                return;
            }
            auto inner_future = submit_kernel_impl(std::move(kernel_func), config);
            inner_future.get();
            promise->set_value();
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });
    {
        std::lock_guard<std::mutex> lk(pending_waiters_mutex_);
        pending_waiters_.push_back(std::move(waiter));
    }
    return result_future;
}

} // namespace executor
