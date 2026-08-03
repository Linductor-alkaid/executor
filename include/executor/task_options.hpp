#pragma once

#include "types.hpp"

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace executor {

/**
 * @brief 用户声明的任务执行意图。
 *
 * 路由器只能依据此声明和后端能力决定投递位置，不会推断 callable 的内容。
 */
enum class ExecutionIntent : uint8_t {
    Auto,
    GeneralCpu,
    CpuOrGpu,
    LowLatency,
    RealtimeQueue,
    BlockingWorker
};

/**
 * @brief 首选后端不可提交时的处理策略。
 */
enum class FallbackPolicy : uint8_t {
    NoFallback,
    AllowCpu,
    RequireRequestedBackend
};

/**
 * @brief 自动路由使用的后端类别。
 */
enum class ExecutionBackend : uint8_t {
    DefaultAsync,
    Gpu,
    LockFree,
    Realtime,
    BlockingIo
};

/**
 * @brief 路由决定的主要依据。
 */
enum class RoutingReason : uint8_t {
    DefaultPolicy,
    ExplicitIntent,
    PreferredExecutor,
    GpuHeuristic,
    AdaptiveHistory,
    BackendUnavailable,
    BackendNotRunning,
    CapacityPressure,
    FallbackPolicy,
    Rejected
};

/**
 * @brief 自动路由的不可变输入选项。
 *
 * `deadline` 仅供路由与诊断使用，不表示中断已开始执行的任务，也不改变
 * ThreadPoolConfig::task_timeout_ms 的软超时语义。
 */
struct TaskOptions {
    std::string name;
    TaskPriority priority = TaskPriority::NORMAL;
    ExecutionIntent intent = ExecutionIntent::Auto;
    std::optional<std::string> preferred_executor;
    FallbackPolicy fallback = FallbackPolicy::NoFallback;
    std::optional<std::chrono::steady_clock::time_point> deadline;
};

/**
 * @brief 将 callable 与自动路由选项组合的按值 builder。
 *
 * 此类型只表达任务意图；实际投递由后续 `Executor::submit_auto()` 重载完成。
 */
template <typename Function>
class TaskBuilder {
public:
    explicit TaskBuilder(Function function)
        : function_(std::move(function)) {}

    TaskBuilder& name(std::string value) {
        options_.name = std::move(value);
        return *this;
    }

    TaskBuilder& priority(TaskPriority value) noexcept {
        options_.priority = value;
        return *this;
    }

    TaskBuilder& intent(ExecutionIntent value) noexcept {
        options_.intent = value;
        return *this;
    }

    TaskBuilder& preferred_executor(std::string value) {
        options_.preferred_executor = std::move(value);
        return *this;
    }

    TaskBuilder& fallback(FallbackPolicy value) noexcept {
        options_.fallback = value;
        return *this;
    }

    TaskBuilder& deadline(std::chrono::steady_clock::time_point value) noexcept {
        options_.deadline = value;
        return *this;
    }

    const TaskOptions& options() const noexcept {
        return options_;
    }

    TaskOptions& options() noexcept {
        return options_;
    }

    const Function& function() const& noexcept {
        return function_;
    }

    Function& function() & noexcept {
        return function_;
    }

    Function&& function() && noexcept {
        return std::move(function_);
    }

private:
    Function function_;
    TaskOptions options_;
};

/**
 * @brief 创建可配置自动路由意图的 callable 包装。
 */
template <typename Function>
auto task(Function&& function) -> TaskBuilder<std::decay_t<Function>> {
    return TaskBuilder<std::decay_t<Function>>(std::forward<Function>(function));
}

}  // namespace executor
