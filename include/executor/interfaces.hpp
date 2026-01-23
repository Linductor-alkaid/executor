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

protected:
    /**
     * @brief 提交任务实现（内部方法）
     * @param task 任务函数
     */
    virtual void submit_impl(std::function<void()> task) = 0;
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

} // namespace executor
