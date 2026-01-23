#pragma once

#include "executor/interfaces.hpp"
#include "executor/config.hpp"
#include "executor/types.hpp"
#include "thread_pool/thread_pool.hpp"
#include <string>
#include <memory>

namespace executor {

/**
 * @brief 线程池执行器
 * 
 * 实现 IAsyncExecutor 接口，封装 ThreadPool，提供异步任务执行功能。
 * 用于处理普通并发任务，支持任务提交、优先级调度、状态监控等。
 */
class ThreadPoolExecutor : public IAsyncExecutor {
public:
    /**
     * @brief 构造函数
     * 
     * @param name 执行器名称
     * @param config 线程池配置
     */
    ThreadPoolExecutor(const std::string& name, const ThreadPoolConfig& config);

    /**
     * @brief 析构函数
     * 
     * 自动停止执行器并等待所有任务完成
     */
    ~ThreadPoolExecutor();

    // 禁止拷贝和移动
    ThreadPoolExecutor(const ThreadPoolExecutor&) = delete;
    ThreadPoolExecutor& operator=(const ThreadPoolExecutor&) = delete;
    ThreadPoolExecutor(ThreadPoolExecutor&&) = delete;
    ThreadPoolExecutor& operator=(ThreadPoolExecutor&&) = delete;

    /**
     * @brief 获取执行器名称
     * @return 执行器名称
     */
    std::string get_name() const override;

    /**
     * @brief 获取执行器状态
     * @return 异步执行器状态
     */
    AsyncExecutorStatus get_status() const override;

    /**
     * @brief 启动执行器
     * 
     * 初始化线程池并启动工作线程。
     * 
     * @return 如果启动成功返回true，否则返回false
     */
    bool start() override;

    /**
     * @brief 停止执行器
     * 
     * 关闭线程池，默认等待所有任务完成。
     */
    void stop() override;

    /**
     * @brief 等待所有任务完成
     * 
     * 阻塞直到所有已提交的任务执行完成。
     */
    void wait_for_completion() override;

protected:
    /**
     * @brief 提交任务实现（内部方法）
     * 
     * 将任务提交到线程池执行。
     * 
     * @param task 任务函数
     */
    void submit_impl(std::function<void()> task) override;

    /**
     * @brief 提交优先级任务实现（内部方法）
     * 
     * 将任务提交到线程池执行，使用指定优先级。
     * 
     * @param priority 优先级（0=LOW, 1=NORMAL, 2=HIGH, 3=CRITICAL）
     * @param task 任务函数
     */
    void submit_priority_impl(int priority, std::function<void()> task) override;

private:
    std::string name_;              // 执行器名称
    ThreadPoolConfig config_;       // 线程池配置
    ThreadPool thread_pool_;       // 线程池实例
};

} // namespace executor
