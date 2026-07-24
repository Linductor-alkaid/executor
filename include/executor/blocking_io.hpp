#pragma once

#include "config.hpp"
#include "types.hpp"

#include <memory>
#include <stop_token>
#include <string>

namespace executor {

/**
 * @brief 应用实现的专属、可中断阻塞 I/O 循环。
 *
 * run() 可以阻塞等待 transport 事件，但必须在 wakeup() 后尽快返回并检查
 * stop_token。stop_token 本身不能中断第三方库的 read/poll/handle 调用。
 */
class IBlockingIoWorker {
public:
    virtual ~IBlockingIoWorker() = default;

    virtual void run(std::stop_token stop_token) = 0;
    virtual void wakeup() noexcept = 0;
};

/** @brief 专属阻塞 I/O worker 的 executor 接口。 */
class IBlockingIoExecutor {
public:
    virtual ~IBlockingIoExecutor() = default;

    virtual bool start() = 0;
    virtual void request_stop() noexcept = 0;
    virtual void stop() = 0;
    virtual std::string get_name() const = 0;
    virtual BlockingIoExecutorStatus get_status() const = 0;
};

} // namespace executor
