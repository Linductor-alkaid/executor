#pragma once

#include "config.hpp"
#include "types.hpp"
#include "task_options.hpp"
#include "task_router.hpp"
#include "task_cancellation.hpp"
#include "timer.hpp"
#include "serial_execution_context.hpp"
#include "interfaces.hpp"
#include "executor_manager.hpp"
#include "blocking_io.hpp"
#include "lockfree_task_executor.hpp"
#include "gpu/gpu_scheduler.hpp"
#include <future>
#include <functional>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <chrono>
#include <concepts>
#include <type_traits>
#include <tuple>
#include <map>
#include <queue>
#include <deque>
#include <optional>

namespace executor {

class TaskDependencyManager;
namespace monitor { class ExecutorMonitor; }

/**
 * @brief Executor Facade
 * 
 * 提供统一的高级 API，内部委托给 ExecutorManager。
 * 支持单例模式和实例化模式。
 * 
 * 功能：
 * - 任务提交（submit, submit_priority, submit_delayed, submit_periodic）
 * - 实时任务管理（register_realtime_task, start_realtime_task, stop_realtime_task）
 * - 监控查询（get_async_executor_status, get_realtime_executor_status）
 */
class Executor {
public:
    /**
     * @brief 获取单例实例
     * 
     * 使用全局 ExecutorManager 单例，同一进程内共享。
     * 
     * @return Executor 单例引用
     */
    static Executor& instance();

    /**
     * @brief 构造函数（实例化模式）
     * 
     * 创建独立的 Executor 实例，内部创建独立的 ExecutorManager 实例。
     * 用于资源隔离场景。
     */
    Executor();

    /**
     * @brief 析构函数（RAII）
     * 
     * 自动关闭定时器线程，ExecutorManager 析构时会自动释放所有执行器。
     */
    ~Executor();

    // 禁止拷贝和赋值
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;

    /**
     * @brief 初始化执行器
     * 
     * 初始化默认异步执行器（线程池）。
     * 
     * @param config 执行器配置
     * @return 是否初始化成功
     */
    bool initialize(const ExecutorConfig& config);

    /**
     * @brief 初始化执行器并返回可诊断结果
     */
    ExecutorResult initialize_ex(const ExecutorConfig& config);

    /**
     * @brief 关闭执行器
     * 
     * 关闭所有执行器（异步执行器和实时执行器）。
     * 
     * @param wait_for_tasks 是否等待任务完成（默认：true）
     */
    ShutdownResult shutdown(bool wait_for_tasks = true);

    /**
     * @brief 设置定时器线程工厂（仅用于测试）
     *
     * 允许测试注入线程创建失败，验证 start_timer_thread() 的异常回滚行为。
     */
    void set_timer_thread_factory_for_test(
        std::function<std::thread(std::function<void()>)> factory);

    /**
     * @brief 提交任务（使用默认线程池）
     * 
     * @tparam F 可调用对象类型
     * @tparam Args 参数类型
     * @param f 可调用对象
     * @param args 参数
     * @return std::future 任务执行结果的 future
     */
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>;

    template<typename F, typename... Args>
    auto submit_with_handle(F&& f, Args&&... args)
        -> TaskSubmission<typename std::invoke_result<F, Args...>::type>;

    /** Submit a task through a FIFO serialized context while retaining facade admission. */
    template<typename F, typename... Args>
    auto submit_on(SerialExecutionContext& context, F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>;

    template<typename F, typename... Args>
    auto submit_on_with_handle(SerialExecutionContext& context, F&& f, Args&&... args)
        -> TaskSubmission<typename std::invoke_result<F, Args...>::type>;

    template<typename F, typename... Args>
    auto submit_after(const TaskHandle& dependency, F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>;

    template<typename F, typename... Args>
    auto submit_after(const std::vector<TaskHandle>& dependencies, F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>;

    template<typename F, typename... Args>
    auto submit_after_with_handle(const TaskHandle& dependency, F&& f, Args&&... args)
        -> TaskSubmission<typename std::invoke_result<F, Args...>::type>;

    template<typename F, typename... Args>
    auto submit_after_with_handle(const std::vector<TaskHandle>& dependencies, F&& f, Args&&... args)
        -> TaskSubmission<typename std::invoke_result<F, Args...>::type>;

    TaskHandle when_all(std::vector<TaskHandle> dependencies);

    /**
     * @brief Configure how many terminal task-graph handles remain usable.
     *
     * A terminal handle can be used to create a later dependent task while it
     * remains retained.  Evicted handles are rejected as expired.  Active
     * dependency chains are never evicted early.
     */
    void set_task_graph_retention_capacity(size_t capacity);
    size_t task_graph_retention_capacity() const;

    /**
     * @brief 提交优先级任务
     * 
     * @tparam F 可调用对象类型
     * @tparam Args 参数类型
     * @param priority 优先级（0=LOW, 1=NORMAL, 2=HIGH, 3=CRITICAL）
     * @param f 可调用对象
     * @param args 参数
     * @return std::future 任务执行结果的 future
     */
    template<typename F, typename... Args>
    auto submit_priority(int priority, F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>;

    /**
     * @brief 提交延迟任务
     * 
     * 任务将在指定延迟时间后执行。
     * 
     * @tparam F 可调用对象类型
     * @tparam Args 参数类型
     * @param delay_ms 延迟时间（毫秒）
     * @param f 可调用对象
     * @param args 参数
     * @return std::future 任务执行结果的 future
     */
    template<typename F, typename... Args>
    auto submit_delayed(int64_t delay_ms, F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>;

    /**
     * @brief 提交周期性任务
     * 
     * 任务将按指定周期重复执行。
     * 
     * @param period_ms 周期（毫秒）
     * @param task 任务函数
     * @return 任务 ID（可用于取消任务）
     */
    std::string submit_periodic(int64_t period_ms, std::function<void()> task);

    /**
     * @brief 取消任务
     *
     * 取消指定的周期性任务。
     *
     * @param task_id 任务 ID
     * @return 是否取消成功
     */
    bool cancel_task(const std::string& task_id);

    // ------------------------------------------------------------------
    // 任务级协作取消（C1）
    // ------------------------------------------------------------------

    /**
     * @brief 请求取消一个带句柄的任务（排队取消或运行中协作请求）。
     *
     * - 排队中（未开始执行，含依赖未满足）：任务不再执行，future 以
     *   TaskCancelled(Explicit) 就绪，不产生 failure 事件；
     * - 运行中：只置位任务的协作停止 token（不抢占、不中断）；任务通过
     *   submit_cancellable* 收到的 StopToken 轮询退出；
     * - 重复/过期句柄幂等：返回 AlreadyRequested / AlreadyCompleted /
     *   NotFound，不写 failure。
     *
     * @param handle submit_with_handle/submit_after_with_handle/
     *               submit_cancellable* 返回的句柄
     * @return 取消请求结果
     */
    TaskCancellationResponse request_task_cancel(const TaskHandle& handle) noexcept;

    /**
     * @brief 获取取消生命周期独立计数（不并入 ExecutorFailureStatus）。
     */
    CancellationStatus get_cancellation_status() const;

    /**
     * @brief 设置按句柄取消 registry 的容量（active 与 tombstone 各自上限）。
     *
     * 容量耗尽时新的可取消提交被明确拒绝（SubmitRejected 诊断），
     * 不会退化成"提交成功但无法取消"。
     */
    void set_cancellation_registry_capacity(size_t capacity);

    size_t cancellation_registry_capacity() const;

    /**
     * @brief 提交可协作取消的任务；executor 注入 StopToken 作为首参数。
     *
     * token 是任务与 executor 之间唯一的协作取消通道：request_task_cancel()
     * 在任务运行中只置位 token，任务负责轮询 stop_requested() 并自行退出。
     * 阻塞在无 wakeup 机制的调用上时，取消不会打断该调用。
     *
     * @return 句柄 + future；句柄可用于 request_task_cancel()
     */
    template<typename F, typename... Args>
    auto submit_cancellable(F&& f, Args&&... args)
        -> TaskSubmission<typename std::invoke_result<F, StopToken, Args...>::type>;

    /** @brief submit_cancellable 的优先级版本。 */
    template<typename F, typename... Args>
    auto submit_cancellable_priority(int priority, F&& f, Args&&... args)
        -> TaskSubmission<typename std::invoke_result<F, StopToken, Args...>::type>;

    /** @brief submit_cancellable 的依赖图版本（单个依赖句柄）。 */
    template<typename F, typename... Args>
    auto submit_cancellable_after(const TaskHandle& dependency, F&& f, Args&&... args)
        -> TaskSubmission<typename std::invoke_result<F, StopToken, Args...>::type>;

    /** @brief submit_cancellable 的依赖图版本（等待全部依赖）。 */
    template<typename F, typename... Args>
    auto submit_cancellable_after(const std::vector<TaskHandle>& dependencies,
                                  F&& f, Args&&... args)
        -> TaskSubmission<typename std::invoke_result<F, StopToken, Args...>::type>;

    // ------------------------------------------------------------------
    // 定时句柄（T1）
    // ------------------------------------------------------------------

    /**
     * @brief 提交带句柄的延迟任务。
     *
     * 与 submit_delayed() 的 future 语义一致，额外返回可取消/可重排的
     * TimerHandle（析构不取消）。调度线程停止（shutdown）时未到期任务以
     * TaskCancelled(Shutdown) 就绪，不产生 failure 事件。
     */
    template<typename F, typename... Args>
    auto submit_delayed_with_handle(int64_t delay_ms, F&& f, Args&&... args)
        -> TimerSubmission<typename std::invoke_result<F, Args...>::type>;

    /**
     * @brief 提交带句柄、可协作取消的延迟任务（StopToken 作为首参数注入）。
     *
     * 取消在到期前生效时任务不执行；到期派发后取消继续向排队/运行中的
     * 任务传播（CancellationRequestedAfterDispatch）。
     */
    template<typename F, typename... Args>
    auto submit_delayed_cancellable_with_handle(int64_t delay_ms, F&& f, Args&&... args)
        -> TimerSubmission<typename std::invoke_result<F, StopToken, Args...>::type>;

    /**
     * @brief 提交带句柄的周期任务。
     *
     * 与 submit_periodic() 的诊断语义一致（tick 异常进入 failure 体系与
     * PeriodicTaskStatus），额外返回 TimerHandle：cancel 阻止后续 tick，
     * reschedule_after 只改下一次到期时间、不改周期。
     */
    TimerHandle submit_periodic_with_handle(int64_t period_ms,
                                            std::function<void()> task);

    /**
     * @brief 提交带句柄、可协作取消的周期任务（StopToken 作为首参数注入）。
     *
     * 每个 tick 独立注入 token；cancel 原子阻止后续 tick 并对在途 tick
     * 请求排队/协作取消，但不撤回已取得执行权的 callback，也不等待完成。
     */
    TimerHandle submit_periodic_cancellable_with_handle(
        int64_t period_ms, std::function<void(StopToken)> task);

    /** @brief 定时任务计数快照（pending/executed/cancelled）。 */
    TimerStatusSummary get_timer_status_summary() const;


    /**
     * @brief 查询单个周期任务状态
     *
     * 返回 std::nullopt 表示任务不存在或已取消。
     */
    std::optional<PeriodicTaskStatus> get_periodic_task_status(
        const std::string& task_id) const;

    /**
     * @brief 查询所有当前注册的周期任务状态
     */
    std::vector<PeriodicTaskStatus> get_all_periodic_task_status() const;

    /**
     * @brief 批量提交任务
     *
     * 批量提交多个任务，可减少重复提交路径开销。
     * 实际性能收益取决于任务数量、任务体、线程数、硬件和构建配置。
     *
     * @tparam F 可调用对象类型
     * @param tasks 任务列表
     * @return std::vector<std::future<void>> 任务执行结果的 future 列表
     *
     * @note 不承诺固定加速比；需要性能结论时请运行本地 benchmark。
     *
     * 示例：
     * @code
     * std::vector<std::function<void()>> tasks;
     * for (int i = 0; i < 1000; ++i) {
     *     tasks.push_back([i]() { process(i); });
     * }
     * auto futures = executor.submit_batch(tasks);
     * @endcode
     */
    template<typename F>
    std::vector<std::future<void>> submit_batch(const std::vector<F>& tasks);

    /**
     * @brief 批量提交任务（无返回值版本）
     *
     * 批量提交多个任务，不返回 future，省去逐个 future 的管理开销。
     * 适用于不需要等待任务完成的场景（fire-and-forget）。
     *
     * @tparam F 可调用对象类型
     * @param tasks 任务列表
     *
     * @note 相比返回 future 的版本，避免了 packaged_task 的开销
     *
     * 示例：
     * @code
     * std::vector<std::function<void()>> tasks;
     * for (int i = 0; i < 1000; ++i) {
     *     tasks.push_back([i]() { process(i); });
     * }
     * executor.submit_batch_no_future(tasks);
     * @endcode
     */
    template<typename F>
    void submit_batch_no_future(const std::vector<F>& tasks);

    /**
     * @brief 批量提交优先级任务
     *
     * 批量提交多个优先级任务。
     *
     * @tparam F 可调用对象类型
     * @param priority 优先级（0=LOW, 1=NORMAL, 2=HIGH, 3=CRITICAL）
     * @param tasks 任务列表
     * @return std::vector<std::future<void>> 任务执行结果的 future 列表
     */
    template<typename F>
    std::vector<std::future<void>> submit_batch_priority(
        int priority,
        const std::vector<F>& tasks);

    /**
     * @brief 注册实时任务
     * 
     * 创建并注册实时执行器（专用实时线程）。
     * 
     * @param name 任务名称
     * @param config 实时线程配置
     * @return 是否注册成功
     */
    bool register_realtime_task(const std::string& name,
                               const RealtimeThreadConfig& config);

    /**
     * @brief 注册实时任务并返回可诊断结果
     */
    ExecutorResult register_realtime_task_ex(const std::string& name,
                                             const RealtimeThreadConfig& config);

    /**
     * @brief 启动实时任务
     * 
     * @param name 任务名称
     * @return 是否启动成功
     */
    bool start_realtime_task(const std::string& name);

    /**
     * @brief 启动实时任务并返回可诊断结果
     */
    ExecutorResult start_realtime_task_ex(const std::string& name);

    /**
     * @brief 停止实时任务
     * 
     * @param name 任务名称
     */
    void stop_realtime_task(const std::string& name);

    bool register_blocking_io_worker(const std::string& name,
                                     const BlockingIoConfig& config,
                                     std::unique_ptr<IBlockingIoWorker> worker);

    ExecutorResult register_blocking_io_worker_ex(
        const std::string& name,
        const BlockingIoConfig& config,
        std::unique_ptr<IBlockingIoWorker> worker);

    bool start_blocking_io_worker(const std::string& name);
    ExecutorResult start_blocking_io_worker_ex(const std::string& name);
    void stop_blocking_io_worker(const std::string& name);
    BlockingIoExecutorStatus get_blocking_io_worker_status(const std::string& name) const;
    std::vector<std::string> get_blocking_io_worker_list() const;

    /**
     * @brief Register and start a dedicated Blocking I/O worker in one facade call.
     *
     * The returned handle controls the worker lifecycle; it does not represent
     * completion of a one-shot task or transfer worker ownership.
     */
    WorkerHandle start_worker(BlockingWorkerSpec spec);

    /**
     * @brief 通过 facade 推送任务到指定实时执行器
     *
     * 失败会同时通过返回值、RealtimeExecutorStatus 计数和 facade failure event 可见。
     */
    bool push_realtime_task(const std::string& name, std::function<void()> task);

    /**
     * @brief push_realtime_task 的显式 try 命名别名
     */
    bool try_push_realtime_task(const std::string& name, std::function<void()> task);

    /**
     * @brief 获取实时执行器的非持有裸指针
     *
     * 高级逃生口。返回值不延长生命周期，不能跨或并发于 shutdown() 使用；
     * 普通任务推送请使用 push_realtime_task()。
     *
     * @param name 执行器名称
     * @return 实时执行器指针，如果不存在则返回 nullptr
     */
    IRealtimeExecutor* get_realtime_executor(const std::string& name);

    /**
     * @brief 获取所有实时任务列表
     * 
     * @return 实时任务名称列表
     */
    std::vector<std::string> get_realtime_task_list() const;

    bool register_lockfree_executor(const std::string& name,
                                    std::unique_ptr<LockFreeTaskExecutor> executor);
    bool start_lockfree_executor(const std::string& name);
    void stop_lockfree_executor(const std::string& name);
    std::vector<std::string> get_lockfree_executor_names() const;

    /**
     * @brief Dispatch to an explicitly selected bounded, fire-and-forget backend.
     *
     * `accepted` reports queue admission only; it never represents task
     * completion. `LowLatency` requires a running named lock-free executor.
     */
    DispatchResult dispatch_auto(TaskOptions options, std::function<void()> task);

    /** @brief Enumerate advisory state snapshots for every registered backend. */
    std::vector<ExecutorCapability> get_executor_capabilities() const;

    /**
     * @brief 获取异步执行器状态
     * 
     * @return 异步执行器状态
     */
    AsyncExecutorStatus get_async_executor_status() const;

    /**
     * @brief 获取实时执行器状态
     * 
     * @param name 执行器名称
     * @return 实时执行器状态
     */
    RealtimeExecutorStatus get_realtime_executor_status(const std::string& name) const;

    /**
     * @brief 设置 facade 失败事件回调
     *
     * 未设置回调时，失败事件仍会进入状态计数和最近事件缓冲。
     * callback 自身抛出的异常会被隔离，不会杀死 worker 或后台线程。
     */
    void set_failure_callback(ExecutorFailureCallback callback);

    /**
     * @brief 获取累计失败状态
     */
    ExecutorFailureStatus get_failure_status() const;

    /**
     * @brief 获取最近失败事件
     *
     * @param max_count 最多返回事件数；0 表示返回当前缓冲区内全部事件。
     * @return 按发生时间从旧到新排序的失败事件列表
     */
    std::vector<ExecutorFailureEvent> get_recent_failures(size_t max_count = 0) const;

    /**
     * @brief 清空最近失败事件
     *
     * 只清空 ring buffer，不重置累计计数。
     */
    void clear_recent_failures();

    /**
     * @brief 设置最近失败事件缓冲容量
     *
     * 容量为 0 时不保留最近事件，但累计状态和 callback 仍生效。
     */
    void set_recent_failure_capacity(size_t capacity);

    std::optional<RoutingDecision> get_last_routing_decision() const;
    std::vector<RoutingDecision> get_recent_routing_decisions(size_t max_count = 0) const;
    void clear_recent_routing_decisions();
    void set_recent_routing_capacity(size_t capacity);
    void set_routing_callback(std::function<void(const RoutingDecision&)> callback);

    /**
     * @brief 启用或禁用任务监控
     */
    void enable_monitoring(bool enable);

    /**
     * @brief 设置监控采样率
     * @param rate 采样率 (0.0-1.0)，0.01 表示 1% 采样
     */
    void set_monitoring_sampling_rate(double rate);

    /**
     * @brief Limit sampled queued/running task diagnostics retained by snapshots.
     *
     * A capacity of 0 disables in-flight retention. It does not disable the
     * existing aggregate TaskStatistics monitor.
     */
    void set_in_flight_task_capacity(size_t capacity);

    /** Set the independent sampling rate for in-flight task diagnostics. */
    void set_in_flight_task_sampling_rate(double rate);

    /**
     * @brief 按 task_type 获取任务统计
     */
    TaskStatistics get_task_statistics(const std::string& task_type) const;

    /**
     * @brief 获取全部 task_type 的任务统计
     */
    std::map<std::string, TaskStatistics> get_all_task_statistics() const;

    /**
     * @brief 等待默认异步后端已提交的 future 型任务完成
     *
     * 兼容旧调用方，最多等待 kDefaultWaitForCompletionTimeout。
     * 超时时不抛异常，但会记录 FailureKind::WaitTimeout。
     */
    void wait_for_completion();

    /**
     * @brief 等待默认异步后端已提交的 future 型任务完成并返回是否完成
     *
     * @param timeout 最长等待时间
     * @return true 表示所有任务在 timeout 内完成；false 表示等待超时。
     *         超时时记录 FailureKind::WaitTimeout，可通过 get_failure_status()
     *         观察 wait_timeout_count。
     */
    bool try_wait_for_completion(std::chrono::milliseconds timeout);

    /**
     * @brief 等待默认异步后端已提交的 future 型任务完成并返回是否完成
     */
    template<typename Rep, typename Period>
    bool wait_for_completion_for(
        const std::chrono::duration<Rep, Period>& timeout);

    /**
     * @brief 等待默认异步后端已提交的 future 型任务完成并返回诊断结果
     */
    WaitResult wait_for_completion_ex(std::chrono::milliseconds timeout);

    /**
     * @brief 当前默认异步执行器是否没有排队或执行中的任务
     */
    bool is_idle() const;

    /**
     * @brief 获取默认异步执行器完成状态快照
     */
    CompletionStatus get_completion_status() const;

    /**
     * @brief 获取 Executor 的完整生命周期诊断快照。
     *
     * 这是低频、best-effort 的只读诊断接口。查询不会创建默认异步执行器，
     * 不承诺跨后端事务级一致性，也不应在实时周期中调用。
     */
    ExecutorSnapshot get_snapshot() const;

    /**
     * @brief 返回稳定的行式生命周期快照文本，适用于日志和故障支持包。
     */
    std::string get_snapshot_text() const;

    /**
     * @brief 设置低频故障现场回调。
     *
     * 回调在 wait 超时及 facade 生命周期/注册/启动失败的调用线程执行；
     * 回调异常被隔离，且不得从实时周期或任务热路径调用此 API。
     */
    void set_snapshot_diagnostic_callback(ExecutorSnapshotCallback callback);

    /**
     * @brief 注册 GPU 执行器
     * 
     * 创建并注册 GPU 执行器。
     * 
     * @param name 执行器名称
     * @param config GPU 执行器配置
     * @return 是否注册成功
     */
    bool register_gpu_executor(const std::string& name,
                              const gpu::GpuExecutorConfig& config);

    /**
     * @brief 注册 GPU 执行器并返回可诊断结果
     */
    ExecutorResult register_gpu_executor_ex(const std::string& name,
                                            const gpu::GpuExecutorConfig& config);

    /**
     * @brief 提交 GPU kernel 任务
     * 
     * @tparam KernelFunc GPU kernel 函数类型
     * @param executor_name GPU 执行器名称
     * @param kernel GPU kernel 函数
     * @param config GPU 任务配置
     * @return std::future<void> 任务执行结果的 future
     */
    template<typename KernelFunc>
    auto submit_gpu(const std::string& executor_name,
                   KernelFunc&& kernel,
                   const gpu::GpuTaskConfig& config)
        -> std::future<void>;

    /**
     * @brief 获取 GPU 执行器的非持有裸指针
     *
     * 高级逃生口。返回值不延长生命周期，不能跨或并发于 shutdown() 使用。
     *
     * @param name 执行器名称
     * @return GPU 执行器指针，如果不存在则返回 nullptr
     */
    IGpuExecutor* get_gpu_executor(const std::string& name);

    /**
     * @brief 获取所有 GPU 执行器名称
     * 
     * @return GPU 执行器名称列表
     */
    std::vector<std::string> get_gpu_executor_names() const;

    /**
     * @brief 获取 GPU 执行器状态
     * 
     * @param name 执行器名称
     * @return GPU 执行器状态
     */
    gpu::GpuExecutorStatus get_gpu_executor_status(const std::string& name) const;

    /**
     * @brief 获取所有 GPU 执行器状态（监控查询）
     *
     * @return 执行器名称到状态的映射
     */
    std::map<std::string, gpu::GpuExecutorStatus> get_all_gpu_executor_status() const;

    /**
     * @brief 自动选择 CPU/GPU 执行器提交任务（legacy overload）
     *
     * 根据任务特征自动选择 CPU 或 GPU 执行器。
     * 如果选择 GPU，调用 submit_gpu()；如果选择 CPU，在 CPU 线程池执行。
     *
     * @deprecated 迁移期内保持现有语义：CPU 路径会以 nullptr stream 调用
     * kernel，GPU 不可用时不会隐式回退。新代码应使用 cpu_gpu_task()，由两条
     * 明确 callable 表达 CPU 与 GPU 路径。
     *
     * @tparam KernelFunc GPU kernel 函数类型
     * @param characteristics 任务特征（数据大小、计算强度等）
     * @param gpu_executor_name GPU 执行器名称（GPU 被选中时使用）
     * @param kernel GPU kernel 函数（需支持 nullptr stream 用于 CPU 执行）
     * @param gpu_config GPU 任务配置（GPU 被选中时使用）
     * @return std::future<void> 任务执行结果的 future
     */
    template<typename KernelFunc>
    auto submit_auto(
        const gpu::TaskCharacteristics& characteristics,
        const std::string& gpu_executor_name,
        KernelFunc&& kernel,
        const gpu::GpuTaskConfig& gpu_config)
        -> std::future<void>;

    /**
     * @brief 提交一般 CPU 任务到自动路由入口。
     *
     * 首版 `Auto` 只选择默认异步线程池；此 overload 与 `submit()` 保持相同的
     * future 完成语义，为后续路由诊断提供稳定入口。
     */
    template<typename F, typename... Args>
    auto submit_auto(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>;

    /**
     * @brief 提交带任务意图的普通 callable。
     *
     * 阶段一仅接受 `Auto` 或 `GeneralCpu`，其他意图必须使用对应 typed API。
     */
    template<typename Function>
    auto submit_auto(TaskBuilder<Function> task)
        -> std::future<typename std::invoke_result<Function&>::type>;

    /**
     * @brief 提交 CPU/GPU 双路径任务。
     */
    template<typename CpuFunction, typename GpuFunction>
    std::future<void> submit_auto(CpuGpuTask<CpuFunction, GpuFunction> task);

    /**
     * @brief 更新调度器配置
     *
     * @param config 调度器配置
     */
    void update_scheduler_config(const gpu::GpuScheduler::Config& config);

    /**
     * @brief 获取调度器配置
     *
     * @return 当前调度器配置
     */
    gpu::GpuScheduler::Config get_scheduler_config() const;

private:
    // 定时器调度器（registry + generation heap + 调度线程）以 shared_ptr
    // 持有：TimerHandle 只持有 weak 锚点，Executor 析构后句柄安全失效。
    // 线程启停/代际管理封装在 detail::TimerScheduler 内。
    Executor(ExecutorManager& manager);

    /**
     * @brief 记录 facade 失败事件
     */
    void record_failure(ExecutorFailureEvent event);
    void record_routing_decision(RoutingDecision decision);
    RoutingDecision route_task(const TaskOptions& options,
                               bool cpu_gpu_task,
                               std::optional<bool> gpu_selected = std::nullopt) const;
    RoutingDecision route_dispatch(const TaskOptions& options) const;

    void record_result_failure(const ExecutorResult& result,
                               FailureKind kind,
                               const std::string& executor_name,
                               const std::string& task_id);

    void record_submit_rejected(const std::string& executor_name,
                                const std::string& task_id,
                                const std::string& message,
                                std::exception_ptr exception = nullptr);

    void record_task_exception(const std::string& executor_name,
                               const std::string& task_id,
                               const std::string& message,
                               std::exception_ptr exception);

    void record_task_timeout(const std::string& executor_name,
                             const std::string& task_id,
                             const std::string& message,
                             std::exception_ptr exception);

    void record_realtime_drop(const std::string& executor_name,
                              const std::string& task_id,
                              const std::string& message,
                              std::exception_ptr exception = nullptr);

    void record_periodic_task_exception(const std::string& executor_name,
                                        const std::string& task_id,
                                        const std::string& message,
                                        std::exception_ptr exception);

    void record_periodic_submit_rejected(const std::string& executor_name,
                                         const std::string& task_id,
                                         const std::string& message,
                                         std::exception_ptr exception = nullptr);

    /**
     * @brief submit() 的内部实现，额外暴露"提交被拒绝/未送达"观察者。
     *
     * 任务图句柄路径必须知道提交是否真正送达执行器：若提交被拒绝（执行器
     * 已停止、提交路径抛异常等），wrapper 永远不会运行，对应的任务图节点
     * 会停留在 Pending。没有这个通知，任何依赖该句柄的 submit_after /
     * when_all 都会在 worker 线程上无限期等待，耗尽线程池并挂死 shutdown。
     * on_rejected 在设置拒绝异常的同一位置被调用（可能为 null）。
     */
    template<typename F, typename... Args>
    std::future<typename std::invoke_result<F, Args...>::type>
    submit_with_rejection_observer(
        std::function<void(std::exception_ptr)> on_rejected,
        F&& f,
        Args&&... args);

    // ------------------------------------------------------------------
    // 协作取消内部管道（C1）
    // ------------------------------------------------------------------

    // 推导带/不带 token 注入的 tracked 任务返回类型。
    template <bool kInjectToken, typename F, typename... Args>
    struct tracked_invoke_result
        : std::conditional_t<
              kInjectToken,
              std::invoke_result<F, StopToken, Args...>,
              std::invoke_result<F, Args...>> {};

    /**
     * @brief 带句柄 + 共享取消状态的统一提交路径。
     *
     * 覆盖 submit_with_handle / submit_after_with_handle / submit_cancellable*
     * 的公共语义：句柄与取消 state 一一对应，排队取消、运行中协作取消、
     * queued soft timeout、提交拒绝与 worker 完成通过同一 phase CAS 仲裁，
     * promise 恰好满足一次。
     *
     * @param priority 优先级；nullopt 走普通队列
     * @param dependencies 依赖句柄；空指针表示无依赖
     * @param kInjectToken 是否向 callable 首位注入 StopToken
     */
    template <bool kInjectToken, typename F, typename... Args>
    auto submit_tracked(
        std::optional<int> priority,
        std::shared_ptr<const std::vector<TaskHandle>> dependencies,
        F&& f,
        Args&&... args)
        -> TaskSubmission<typename tracked_invoke_result<kInjectToken, F, Args...>::type>;

    /**
     * @brief 取消状态传播核心（排队取消 / 运行中协作请求 / 终态判定）。
     *
     * request_task_cancel()（按 registry 句柄）与定时器 cancel 的传播
     * hook（按已持有 state）共用。graph_handle 非空时同步推进任务图终态
     * 并唤醒依赖等待者。
     */
    TaskCancellationResponse propagate_cancel_state(
        const std::string& task_id,
        const std::shared_ptr<TaskCancellationState>& state,
        const TaskHandle* graph_handle) noexcept;

    /** 定时器 hook：向已派发的 delayed/periodic tick 任务传播取消。 */
    void propagate_timer_task_cancel(
        const std::string& task_state_id,
        const std::shared_ptr<TaskCancellationState>& state) noexcept;

    /** 依赖异常按设计归类：依赖被取消时改写为 DependencyCancelled。 */
    std::exception_ptr reclassify_dependency_exception(
        std::exception_ptr exception) const;

    // ------------------------------------------------------------------
    // 定时器内部管道（T1）
    // ------------------------------------------------------------------

    /** 懒创建定时器调度器（进程内单实例、稳定地址，供句柄锚定）。 */
    detail::TimerScheduler& ensure_timers();

    /** 给调度器安装取消传播 hook（构造与防御性重建时调用）。 */
    void configure_timer_scheduler_hooks();

    /** 确保调度线程已启动；线程创建失败按 legacy 语义向上抛出。 */
    void start_timer_thread();

    /** 停止调度线程并按 TaskCancelled(Shutdown) 清理未到期任务。 */
    void stop_timer_thread();

    /**
     * @brief submit_delayed 系列的统一实现。
     *
     * kInjectToken 为 true 时向 callable 首位注入 StopToken。返回句柄 +
     * future；legacy submit_delayed() 丢弃句柄保持旧返回类型。
     */
    template <bool kInjectToken, typename F, typename... Args>
    auto submit_delayed_impl(int64_t delay_ms, F&& f, Args&&... args)
        -> TimerSubmission<typename tracked_invoke_result<kInjectToken, F, Args...>::type>;

    enum class TaskGraphState {
        Pending,
        Running,
        Succeeded,
        Failed,
        WhenAll
    };

    struct TaskGraphNode {
        TaskGraphState state = TaskGraphState::Pending;
        std::exception_ptr exception;
        std::string error_message;
        std::vector<std::string> dependencies;
    };

    TaskHandle allocate_task_handle();
    bool task_handle_known_locked(const TaskHandle& handle) const;
    bool register_task_graph_dependencies(const TaskHandle& handle,
                                          const std::vector<TaskHandle>& dependencies,
                                          std::string& error_message);
    std::exception_ptr dependency_failure_locked(const std::vector<TaskHandle>& dependencies) const;
    bool dependencies_succeeded_locked(const std::vector<TaskHandle>& dependencies) const;
    void mark_task_graph_running(const TaskHandle& handle);
    void mark_task_graph_succeeded(const TaskHandle& handle);
    void mark_task_graph_failed(const TaskHandle& handle,
                                std::exception_ptr exception,
                                std::string message);
    void resolve_task_graph_dependents_locked(const std::string& task_id);
    void finalize_task_graph_node_locked(const std::string& task_id);
    void trim_task_graph_retention_locked();
    std::exception_ptr make_dependency_exception(const std::string& message) const;

    /**
     * @brief 当前 facade 最近失败事件缓冲容量
     */
    size_t recent_failure_capacity() const;
    void emit_snapshot_diagnostic() const;
    void emit_snapshot_diagnostic(const ExecutorSnapshot& snapshot) const;

    // ExecutorManager 指针（单例或实例）
    ExecutorManager* manager_;

    // 实例化模式时拥有的 ExecutorManager
    std::unique_ptr<ExecutorManager> owned_manager_;

    // 仅由 facade 生命周期边界写入；Monitor 对运行后端状态作保守补充。
    std::atomic<ExecutorLifecycleState> lifecycle_state_{ExecutorLifecycleState::Created};
    std::unique_ptr<monitor::ExecutorMonitor> monitor_;

    mutable std::mutex snapshot_diagnostic_mutex_;
    ExecutorSnapshotCallback snapshot_diagnostic_callback_;

    // 按句柄取消 registry（active + 有界 tombstone + lifecycle 计数）。
    std::unique_ptr<TaskCancellationRegistry> cancellation_registry_;

    // 定时器调度器（delayed/periodic registry + generation heap + 线程）。
    // 懒创建：地址在 Executor 生命周期内稳定，TimerHandle 经 weak_ptr 锚定。
    std::shared_ptr<detail::TimerScheduler> timers_;
    mutable std::mutex timers_mutex_;

    static constexpr size_t kDefaultRecentFailureCapacity = 128;
    static constexpr size_t kDefaultRecentRoutingCapacity = 128;

    mutable std::mutex failure_mutex_;
    ExecutorFailureStatus failure_status_;
    std::deque<ExecutorFailureEvent> recent_failures_;
    size_t recent_failure_capacity_ = kDefaultRecentFailureCapacity;
    ExecutorFailureCallback failure_callback_;

    mutable std::mutex routing_mutex_;
    std::deque<RoutingDecision> recent_routing_decisions_;
    size_t recent_routing_capacity_ = kDefaultRecentRoutingCapacity;
    std::function<void(const RoutingDecision&)> routing_callback_;
    TaskRouter task_router_;

    mutable std::mutex task_graph_mutex_;
    std::condition_variable task_graph_cv_;
    std::unique_ptr<TaskDependencyManager> task_dependencies_;
    std::unordered_map<std::string, TaskGraphNode> task_graph_nodes_;
    std::unordered_map<std::string, std::vector<std::string>> task_graph_dependents_;
    std::deque<std::string> task_graph_terminal_order_;
    size_t task_graph_retention_capacity_ = 1024;

    // GPU 调度器
    gpu::GpuScheduler scheduler_;
};

// 模板方法实现
template<typename Rep, typename Period>
bool Executor::wait_for_completion_for(
    const std::chrono::duration<Rep, Period>& timeout) {
    return wait_for_completion_ex(
        std::chrono::duration_cast<std::chrono::milliseconds>(timeout)).completed;
}

template<typename F, typename... Args>
auto Executor::submit(F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type> {
    return submit_with_rejection_observer(
        nullptr, std::forward<F>(f), std::forward<Args>(args)...);
}

template<typename F, typename... Args>
auto Executor::submit_on(SerialExecutionContext& context, F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type> {
    return submit_on_with_handle(context, std::forward<F>(f), std::forward<Args>(args)...).future;
}

template<typename F, typename... Args>
auto Executor::submit_on_with_handle(SerialExecutionContext& context, F&& f, Args&&... args)
    -> TaskSubmission<typename std::invoke_result<F, Args...>::type> {
    using return_type = typename std::invoke_result<F, Args...>::type;
    auto bound = std::make_shared<decltype(std::bind(std::forward<F>(f), std::forward<Args>(args)...))>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    auto gate = std::make_shared<std::promise<return_type>>();
    auto gate_future = std::make_shared<std::future<return_type>>(gate->get_future());
    auto wrapper = [&context, bound, gate, gate_future]() mutable -> return_type {
        std::mutex mutex;
        std::condition_variable cv;
        bool finished = false;
        bool accepted = context.post([bound, gate, &mutex, &cv, &finished]() mutable {
            try {
                if constexpr (std::is_void_v<return_type>) {
                    std::invoke(*bound);
                    gate->set_value();
                } else {
                    gate->set_value(std::invoke(*bound));
                }
            } catch (...) {
                try { gate->set_exception(std::current_exception()); } catch (...) {}
            }
            { std::lock_guard<std::mutex> lock(mutex); finished = true; }
            cv.notify_one();
        });
        if (!accepted) {
            throw ExecutorStopping("Serial execution context is stopped");
        }
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&finished] { return finished; });
        return gate_future->get();
    };
    return submit_with_handle(std::move(wrapper));
}

template<typename F, typename... Args>
auto Executor::submit_with_rejection_observer(
    std::function<void(std::exception_ptr)> on_rejected,
    F&& f,
    Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type> {
    using return_type = typename std::invoke_result<F, Args...>::type;

    auto executor = manager_->get_default_async_executor_snapshot();
    const std::string executor_name = executor ? executor->get_name() : "default";
    const std::string task_id = "facade_submit";
    if (!executor) {
        record_submit_rejected(
            executor_name,
            task_id,
            "Async executor not initialized. Call initialize() first.");
        auto exception = std::make_exception_ptr(std::runtime_error(
            "Async executor not initialized. Call initialize() first."));
        if (on_rejected) {
            on_rejected(exception);
        }
        throw std::runtime_error(
            "Async executor not initialized. Call initialize() first.");
    }

    auto promise = std::make_shared<std::promise<return_type>>();
    auto promise_ready = std::make_shared<std::atomic_bool>(false);
    auto future = promise->get_future();

    if constexpr (sizeof...(Args) == 0) {
        if (detail::is_empty_std_function(f)) {
            auto exception = std::make_exception_ptr(std::invalid_argument("empty task"));
            promise_ready->store(true, std::memory_order_release);
            promise->set_exception(exception);
            record_submit_rejected(
                executor_name,
                task_id,
                "Async executor rejected empty task submission",
                exception);
            if (on_rejected) {
                on_rejected(exception);
            }
            return future;
        }
    }

    auto bound_task = std::make_shared<decltype(std::bind(std::forward<F>(f), std::forward<Args>(args)...))>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    auto task_wrapper = [this, executor_name, task_id, promise, promise_ready, bound_task]() mutable {
        try {
            if constexpr (std::is_void_v<return_type>) {
                std::invoke(*bound_task);
                promise->set_value();
            } else {
                promise->set_value(std::invoke(*bound_task));
            }
            promise_ready->store(true, std::memory_order_release);
        } catch (...) {
            auto exception = std::current_exception();
            promise->set_exception(exception);
            promise_ready->store(true, std::memory_order_release);
            record_task_exception(
                executor_name,
                task_id,
                "Async task threw an exception",
                exception);
            throw;
        }
    };

    auto on_timeout = [this, executor_name, task_id, promise, promise_ready](
                          std::exception_ptr exception) {
        bool expected = false;
        if (promise_ready->compare_exchange_strong(expected, true)) {
            promise->set_exception(exception);
            record_task_timeout(
                executor_name,
                task_id,
                "Async task timed out before execution",
                exception);
        }
    };

    if (!executor->try_submit_task(std::move(task_wrapper), std::move(on_timeout))) {
        auto exception = std::make_exception_ptr(
            std::runtime_error("Async executor rejected task submission"));
        bool expected = false;
        if (promise_ready->compare_exchange_strong(expected, true)) {
            promise->set_exception(exception);
            record_submit_rejected(
                executor_name,
                task_id,
                "Async executor rejected task submission",
                exception);
        }
        // wrapper 不会运行：通知任务图路径把句柄置为 Failed，避免依赖方
        // 在 worker 线程上等待一个永不到来的终态。
        if (on_rejected) {
            on_rejected(exception);
        }
    }

    return future;
}

template<typename F, typename... Args>
auto Executor::submit_with_handle(F&& f, Args&&... args)
    -> TaskSubmission<typename std::invoke_result<F, Args...>::type> {
    return submit_tracked<false>(
        std::nullopt, nullptr, std::forward<F>(f), std::forward<Args>(args)...);
}

template<typename F, typename... Args>
auto Executor::submit_cancellable(F&& f, Args&&... args)
    -> TaskSubmission<typename std::invoke_result<F, StopToken, Args...>::type> {
    return submit_tracked<true>(
        std::nullopt, nullptr, std::forward<F>(f), std::forward<Args>(args)...);
}

template<typename F, typename... Args>
auto Executor::submit_cancellable_priority(int priority, F&& f, Args&&... args)
    -> TaskSubmission<typename std::invoke_result<F, StopToken, Args...>::type> {
    return submit_tracked<true>(
        priority, nullptr, std::forward<F>(f), std::forward<Args>(args)...);
}

template<typename F, typename... Args>
auto Executor::submit_cancellable_after(const TaskHandle& dependency,
                                        F&& f,
                                        Args&&... args)
    -> TaskSubmission<typename std::invoke_result<F, StopToken, Args...>::type> {
    auto dependencies =
        std::make_shared<const std::vector<TaskHandle>>(std::vector<TaskHandle>{dependency});
    return submit_tracked<true>(
        std::nullopt, std::move(dependencies), std::forward<F>(f),
        std::forward<Args>(args)...);
}

template<typename F, typename... Args>
auto Executor::submit_cancellable_after(const std::vector<TaskHandle>& dependencies,
                                        F&& f,
                                        Args&&... args)
    -> TaskSubmission<typename std::invoke_result<F, StopToken, Args...>::type> {
    auto dependencies_copy =
        std::make_shared<const std::vector<TaskHandle>>(dependencies);
    return submit_tracked<true>(
        std::nullopt, std::move(dependencies_copy), std::forward<F>(f),
        std::forward<Args>(args)...);
}

template<typename F, typename... Args>
auto Executor::submit_after(const TaskHandle& dependency, F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type> {
    std::vector<TaskHandle> dependencies{dependency};
    return submit_after(std::move(dependencies), std::forward<F>(f), std::forward<Args>(args)...);
}

template<typename F, typename... Args>
auto Executor::submit_after(const std::vector<TaskHandle>& dependencies, F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type> {
    return submit_after_with_handle(dependencies, std::forward<F>(f), std::forward<Args>(args)...).future;
}

template<typename F, typename... Args>
auto Executor::submit_after_with_handle(const TaskHandle& dependency, F&& f, Args&&... args)
    -> TaskSubmission<typename std::invoke_result<F, Args...>::type> {
    std::vector<TaskHandle> dependencies{dependency};
    return submit_after_with_handle(std::move(dependencies), std::forward<F>(f), std::forward<Args>(args)...);
}

template<typename F, typename... Args>
auto Executor::submit_after_with_handle(const std::vector<TaskHandle>& dependencies, F&& f, Args&&... args)
    -> TaskSubmission<typename std::invoke_result<F, Args...>::type> {
    auto dependencies_copy =
        std::make_shared<const std::vector<TaskHandle>>(dependencies);
    return submit_tracked<false>(
        std::nullopt, std::move(dependencies_copy), std::forward<F>(f),
        std::forward<Args>(args)...);
}

// submit_tracked：带句柄 + 共享取消状态的统一提交路径。
//
// 取消/超时/开始执行通过 TaskCancellationState 的单一 phase CAS 仲裁：
// - 取消先赢：任务不调用 callable，future 由 completion sink 立即以
//   TaskCancelled 就绪，任务图节点由取消方落 Failed 终态并唤醒依赖等待；
// - 超时先赢（queued soft timeout）：on_timeout 处理器完成 TimedOut 终态，
//   future 以 TimedOutException 就绪并计入 TaskTimeout 诊断；
// - worker 先赢（进入 Running）：取消只置位 StopToken（协作），callable
//   抛出的 TaskCancelled 在已请求取消时按取消归类，不触发 failure。
template <bool kInjectToken, typename F, typename... Args>
auto Executor::submit_tracked(
    std::optional<int> priority,
    std::shared_ptr<const std::vector<TaskHandle>> dependencies,
    F&& f,
    Args&&... args)
    -> TaskSubmission<typename tracked_invoke_result<kInjectToken, F, Args...>::type> {
    using return_type =
        typename tracked_invoke_result<kInjectToken, F, Args...>::type;

    TaskSubmission<return_type> submission;
    submission.handle = allocate_task_handle();
    TaskHandle handle = submission.handle;

    // 依赖登记（依赖图变体）。
    std::string validation_error;
    if (dependencies && !dependencies->empty()) {
        const bool dependencies_valid =
            register_task_graph_dependencies(handle, *dependencies, validation_error);
        if (!dependencies_valid) {
            auto exception = make_dependency_exception(validation_error);
            mark_task_graph_failed(handle, exception, validation_error);
            auto promise = std::make_shared<std::promise<return_type>>();
            submission.future = promise->get_future();
            promise->set_exception(exception);
            record_submit_rejected("default", handle.id(), validation_error, exception);
            return submission;
        }
        manager_->record_in_flight_task_state(
            handle.id(), TaskLifecycleState::DependencyBlocked);
    }

    auto promise = std::make_shared<std::promise<return_type>>();
    auto promise_ready = std::make_shared<std::atomic_bool>(false);
    submission.future = promise->get_future();

    auto state = std::make_shared<TaskCancellationState>();
    state->set_completion_sink(
        [promise, promise_ready](std::exception_ptr exception) {
            bool expected = false;
            if (promise_ready->compare_exchange_strong(
                    expected, true, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                promise->set_exception(exception);
            }
        });

    // registry admission：容量耗尽按提交拒绝处理，不静默失去取消能力。
    if (!cancellation_registry_->register_state(handle.id(), state)) {
        auto exception = std::make_exception_ptr(std::runtime_error(
            "Cancellation registry capacity exhausted; tracked task rejected"));
        state->try_reject();
        bool expected = false;
        if (promise_ready->compare_exchange_strong(
                expected, true, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            promise->set_exception(exception);
        }
        mark_task_graph_failed(handle, exception, "Tracked task submission rejected");
        record_submit_rejected(
            "default", handle.id(),
            "Cancellation registry capacity exhausted for tracked task", exception);
        return submission;
    }

    // 注意：与 submit() 不同，这里不做 empty-std::function 预检。旧语义是
    // 空 std::function 真正入队、执行时抛 bad_function_call，任务图节点经
    // 执行异常路径落 Failed（见 test_task_graph_rejected_dependency）。

    auto executor_snapshot = manager_->get_default_async_executor_snapshot();
    const std::string executor_name =
        executor_snapshot ? executor_snapshot->get_name() : "default";
    if (!executor_snapshot) {
        const std::string message =
            "Async executor not initialized. Call initialize() first.";
        auto exception = std::make_exception_ptr(std::runtime_error(message));
        state->try_reject();
        mark_task_graph_failed(handle, exception, message);
        record_submit_rejected(executor_name, handle.id(), message, exception);
        cancellation_registry_->finalize(handle.id());
        throw std::runtime_error(message);
    }

    // 提交被拒时 wrapper 永不运行：节点必须落 Failed，否则依赖它的后续
    // 任务会在 worker 线程上等待 Pending 节点直到挂死。
    auto on_rejected = [this, handle, state, promise, promise_ready](
                           std::exception_ptr exception) {
        state->try_reject();
        bool expected = false;
        if (promise_ready->compare_exchange_strong(
                expected, true, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            promise->set_exception(exception);
        }
        mark_task_graph_failed(handle, exception, "Tracked task submission rejected");
        cancellation_registry_->finalize(handle.id());
    };

    // queued soft timeout：与取消经同一 phase CAS 仲裁，只赢一次。
    auto on_timeout = [this, handle, state, executor_name, promise, promise_ready](
                          std::exception_ptr exception) {
        if (!state->try_timeout_before_start()) {
            return;  // 取消已赢或已开始执行
        }
        bool expected = false;
        if (promise_ready->compare_exchange_strong(
                expected, true, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            promise->set_exception(exception);
        }
        record_task_timeout(
            executor_name, handle.id(),
            "Tracked async task timed out before execution", exception);
        mark_task_graph_failed(
            handle, exception, "Tracked async task timed out before execution");
        cancellation_registry_->finalize(handle.id());
    };

    auto invoke_user_callable =
        [f = std::forward<F>(f),
         args_tuple = std::make_tuple(std::forward<Args>(args)...),
         state](StopToken token) mutable -> return_type {
        if constexpr (kInjectToken) {
            return std::apply(
                [&f, &token](auto&&... unpacked) -> return_type {
                    return f(token, std::forward<decltype(unpacked)>(unpacked)...);
                },
                std::move(args_tuple));
        } else {
            (void)token;
            return std::apply(std::move(f), std::move(args_tuple));
        }
    };

    auto task_wrapper = [this,
                         handle,
                         state,
                         executor_name,
                         dependencies,
                         promise,
                         promise_ready,
                         invoke = std::move(invoke_user_callable)]() mutable {
        // 依赖图变体：依赖未满足前保持 Pending/Queued，取消可赢排队仲裁。
        if (dependencies && !dependencies->empty()) {
            std::exception_ptr dependency_exception;
            bool dependencies_ready = false;
            {
                std::unique_lock<std::mutex> lock(task_graph_mutex_);
                task_graph_cv_.wait(lock, [&] {
                    dependency_exception =
                        dependency_failure_locked(*dependencies);
                    if (dependency_exception) {
                        return true;
                    }
                    if (dependencies_succeeded_locked(*dependencies)) {
                        dependencies_ready = true;
                        return true;
                    }
                    return state->cancel_requested();
                });
            }

            if (dependency_exception) {
                dependency_exception =
                    reclassify_dependency_exception(dependency_exception);
                mark_task_graph_failed(
                    handle,
                    dependency_exception,
                    "Dependency failed before dependent task execution");
                if (state->try_reject()) {
                    bool expected = false;
                    if (promise_ready->compare_exchange_strong(
                            expected, true, std::memory_order_acq_rel,
                            std::memory_order_acquire)) {
                        promise->set_exception(dependency_exception);
                    }
                }
                cancellation_registry_->finalize(handle.id());
                std::rethrow_exception(dependency_exception);
            }

            if (!dependencies_ready) {
                // 被自身取消唤醒：取消方已满足 future 并落图终态。
                return;
            }
        }

        // 开始执行仲裁（单一 CAS 线性化点）。
        if (!state->try_begin_execution()) {
            return;  // 取消/超时已先赢，future 已满足
        }

        mark_task_graph_running(handle);
        try {
            // 观测不变式：future 就绪之前，取消生命周期计数必须已最终化
            // （否则等待 future 后立即读 get_cancellation_status() 会与
            // worker 侧计数竞态）。
            if constexpr (std::is_void_v<return_type>) {
                invoke(state->stop_token());
                mark_task_graph_succeeded(handle);
                if (state->cancel_requested()) {
                    // 运行中收到停止请求后仍正常完成：保留业务结果，只计数。
                    cancellation_registry_->on_completed_after_request();
                }
                state->try_finish_running(TaskCancellationState::Phase::Succeeded);
                promise->set_value();
            } else {
                auto result = invoke(state->stop_token());
                mark_task_graph_succeeded(handle);
                if (state->cancel_requested()) {
                    cancellation_registry_->on_completed_after_request();
                }
                state->try_finish_running(TaskCancellationState::Phase::Succeeded);
                promise->set_value(std::move(result));
            }
            promise_ready->store(true, std::memory_order_release);
            cancellation_registry_->finalize(handle.id());
        } catch (...) {
            auto exception = std::current_exception();
            // 判定是否为"已请求取消后的协作退出"：只有 stop state 已被请求
            // 时 TaskCancelled 才按取消归类；无取消请求时主动抛出的
            // TaskCancelled 仍按任务异常处理，防止绕过 failure 统计。
            bool cooperative_cancel = false;
            try {
                std::rethrow_exception(exception);
            } catch (const TaskCancelled&) {
                cooperative_cancel = state->cancel_requested();
            } catch (...) {
                cooperative_cancel = false;
            }

            if (cooperative_cancel) {
                // 运行中协作取消：任务记为 Cancelled，不触发 failure。
                state->try_finish_running(
                    TaskCancellationState::Phase::Cancelled);
                bool expected = false;
                if (promise_ready->compare_exchange_strong(
                        expected, true, std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    promise->set_exception(exception);
                }
                mark_task_graph_failed(
                    handle, exception, "Task cancelled during execution");
                manager_->record_in_flight_task_state(
                    handle.id(), TaskLifecycleState::Cancelled);
                cancellation_registry_->finalize(handle.id());
                return;
            }

            state->try_finish_running(TaskCancellationState::Phase::Failed);
            bool expected = false;
            if (promise_ready->compare_exchange_strong(
                    expected, true, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                promise->set_exception(exception);
            }
            mark_task_graph_failed(handle, exception, "Tracked task failed");
            if (state->cancel_requested()) {
                cancellation_registry_->on_completed_after_request();
            }
            cancellation_registry_->finalize(handle.id());
            record_task_exception(
                executor_name, handle.id(),
                "Tracked async task threw an exception", exception);
            throw;
        }
    };

    try {
        bool accepted = false;
        if (priority) {
            accepted = executor_snapshot->try_submit_priority_task(
                *priority, std::move(task_wrapper), std::move(on_timeout));
        } else {
            accepted = executor_snapshot->try_submit_task(
                std::move(task_wrapper), std::move(on_timeout));
        }
        if (!accepted) {
            auto exception = std::make_exception_ptr(std::runtime_error(
                "Async executor rejected task submission"));
            on_rejected(exception);
            record_submit_rejected(
                executor_name, handle.id(),
                "Async executor rejected tracked task submission", exception);
        }
    } catch (...) {
        auto exception = std::current_exception();
        mark_task_graph_failed(handle, exception, "Tracked task submission failed");
        cancellation_registry_->finalize(handle.id());
        throw;
    }

    return submission;
}

template<typename F, typename... Args>
auto Executor::submit_priority(int priority, F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type> {
    using return_type = typename std::invoke_result<F, Args...>::type;

    auto executor = manager_->get_default_async_executor_snapshot();
    const std::string executor_name = executor ? executor->get_name() : "default";
    const std::string task_id = "facade_submit_priority";
    if (!executor) {
        record_submit_rejected(
            executor_name,
            task_id,
            "Async executor not initialized. Call initialize() first.");
        throw std::runtime_error("Async executor not initialized. Call initialize() first.");
    }

    auto promise = std::make_shared<std::promise<return_type>>();
    auto promise_ready = std::make_shared<std::atomic_bool>(false);
    auto future = promise->get_future();

    if constexpr (sizeof...(Args) == 0) {
        if (detail::is_empty_std_function(f)) {
            auto exception = std::make_exception_ptr(std::invalid_argument("empty task"));
            promise_ready->store(true, std::memory_order_release);
            promise->set_exception(exception);
            record_submit_rejected(
                executor_name,
                task_id,
                "Async executor rejected empty priority task submission",
                exception);
            return future;
        }
    }

    auto bound_task = std::make_shared<decltype(std::bind(std::forward<F>(f), std::forward<Args>(args)...))>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    auto task_wrapper = [this, executor_name, task_id, promise, promise_ready, bound_task]() mutable {
        try {
            if constexpr (std::is_void_v<return_type>) {
                std::invoke(*bound_task);
                promise->set_value();
            } else {
                promise->set_value(std::invoke(*bound_task));
            }
            promise_ready->store(true, std::memory_order_release);
        } catch (...) {
            auto exception = std::current_exception();
            promise->set_exception(exception);
            promise_ready->store(true, std::memory_order_release);
            record_task_exception(
                executor_name,
                task_id,
                "Priority async task threw an exception",
                exception);
            throw;
        }
    };

    auto on_timeout = [this, executor_name, task_id, promise, promise_ready](
                          std::exception_ptr exception) {
        bool expected = false;
        if (promise_ready->compare_exchange_strong(expected, true)) {
            promise->set_exception(exception);
            record_task_timeout(
                executor_name,
                task_id,
                "Priority async task timed out before execution",
                exception);
        }
    };

    if (!executor->try_submit_priority_task(
            priority, std::move(task_wrapper), std::move(on_timeout))) {
        auto exception = std::make_exception_ptr(
            std::runtime_error("Async executor rejected priority task submission"));
        bool expected = false;
        if (promise_ready->compare_exchange_strong(expected, true)) {
            promise->set_exception(exception);
            record_submit_rejected(
                executor_name,
                task_id,
                "Async executor rejected priority task submission",
                exception);
        }
    }

    return future;
}

template<typename F, typename... Args>
auto Executor::submit_delayed(int64_t delay_ms, F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type> {
    // legacy 变体：保持只返回 future；句柄被丢弃，因此不可按句柄取消。
    return submit_delayed_impl<false>(
        delay_ms, std::forward<F>(f), std::forward<Args>(args)...).future;
}

template<typename F, typename... Args>
auto Executor::submit_delayed_with_handle(int64_t delay_ms, F&& f, Args&&... args)
    -> TimerSubmission<typename std::invoke_result<F, Args...>::type> {
    return submit_delayed_impl<false>(
        delay_ms, std::forward<F>(f), std::forward<Args>(args)...);
}

template<typename F, typename... Args>
auto Executor::submit_delayed_cancellable_with_handle(
    int64_t delay_ms, F&& f, Args&&... args)
    -> TimerSubmission<typename std::invoke_result<F, StopToken, Args...>::type> {
    return submit_delayed_impl<true>(
        delay_ms, std::forward<F>(f), std::forward<Args>(args)...);
}

// submit_delayed_impl：delayed 系列的统一实现。
//
// - 到期前（Scheduled）cancel：任务不执行，future 以 TaskCancelled(Explicit)
//   就绪（TimerScheduler 在锁内完成状态仲裁后回调 on_cancelled）；
// - 到期派发后 cancel：经共享 TaskCancellationState 继续向排队/运行中的
//   任务传播（CancellationRequestedAfterDispatch）；
// - shutdown 清理未到期任务：future 以 TaskCancelled(Shutdown) 就绪，
//   不产生 failure 事件（生命周期事件，非失败）；
// - queued soft timeout / 派发被拒：保持既有 TimedOutException /
//   SubmitRejected 诊断语义。
template <bool kInjectToken, typename F, typename... Args>
auto Executor::submit_delayed_impl(int64_t delay_ms, F&& f, Args&&... args)
    -> TimerSubmission<typename tracked_invoke_result<kInjectToken, F, Args...>::type> {
    using return_type =
        typename tracked_invoke_result<kInjectToken, F, Args...>::type;

    TimerSubmission<return_type> submission;

    // legacy 语义：默认异步执行器未初始化时，提交立即失败（而非等到期）。
    {
        auto executor = manager_->get_default_async_executor_snapshot();
        if (!executor) {
            record_submit_rejected(
                "default",
                "facade_submit_delayed",
                "Async executor not initialized. Call initialize() first.");
            throw std::runtime_error(
                "Async executor not initialized. Call initialize() first.");
        }
    }

    auto promise = std::make_shared<std::promise<return_type>>();
    auto promise_ready = std::make_shared<std::atomic_bool>(false);
    submission.future = promise->get_future();

    // cancellable 变体：任务取消状态从登记时刻即存在，关闭"到期派发瞬间
    // cancel 找不到传播目标"的竞争窗口。
    std::shared_ptr<TaskCancellationState> state;
    if constexpr (kInjectToken) {
        state = std::make_shared<TaskCancellationState>();
        state->set_completion_sink(
            [promise, promise_ready](std::exception_ptr exception) {
                bool expected = false;
                if (promise_ready->compare_exchange_strong(
                        expected, true, std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    promise->set_exception(exception);
                }
            });
    }

    const std::string timer_id = generate_task_id();
    const std::string task_state_id = timer_id + "#run";
    const std::string task_id = kInjectToken ? task_state_id
                                             : std::string("facade_submit_delayed");

    // 到期派发闭包：把已包装任务提交到默认异步执行器。
    auto dispatch = [this, state, task_state_id, task_id, promise, promise_ready,
                     f = std::forward<F>(f),
                     args_tuple = std::make_tuple(std::forward<Args>(args)...)]() mutable {
        auto executor = manager_->get_default_async_executor_snapshot();
        const std::string executor_name =
            executor ? executor->get_name() : "default";

        auto reject = [this, state, executor_name, task_id, promise,
                       promise_ready](const char* message) {
            auto exception = std::make_exception_ptr(std::runtime_error(message));
            if (state) {
                state->try_reject();
            }
            bool expected = false;
            if (promise_ready->compare_exchange_strong(
                    expected, true, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                promise->set_exception(exception);
            }
            record_submit_rejected(executor_name, task_id, message, exception);
        };

        if (!executor) {
            reject("Async executor unavailable for delayed task");
            return;
        }

        auto pool_task = [this, state, executor_name, task_id, promise,
                          promise_ready, f = std::move(f),
                          args_tuple = std::move(args_tuple)]() mutable {
            if (state && !state->try_begin_execution()) {
                return;  // 已取消/已超时，future 已满足
            }
            try {
                if constexpr (std::is_void_v<return_type>) {
                    if constexpr (kInjectToken) {
                        std::apply(
                            [&f, token = state->stop_token()](auto&&... unpacked) {
                                f(token,
                                  std::forward<decltype(unpacked)>(unpacked)...);
                            },
                            std::move(args_tuple));
                    } else {
                        std::apply(std::move(f), std::move(args_tuple));
                    }
                    promise->set_value();
                } else {
                    if constexpr (kInjectToken) {
                        auto result = std::apply(
                            [&f, token = state->stop_token()](
                                auto&&... unpacked) -> return_type {
                                return f(
                                    token,
                                    std::forward<decltype(unpacked)>(unpacked)...);
                            },
                            std::move(args_tuple));
                        promise->set_value(std::move(result));
                    } else {
                        auto result = std::apply(
                            std::move(f), std::move(args_tuple));
                        promise->set_value(std::move(result));
                    }
                }
                if (state) {
                    // 观测不变式：future 就绪前完成取消后完成计数（与
                    // submit_tracked 一致）。
                    if (state->cancel_requested()) {
                        cancellation_registry_->on_completed_after_request();
                    }
                    state->try_finish_running(
                        TaskCancellationState::Phase::Succeeded);
                }
                promise_ready->store(true, std::memory_order_release);
            } catch (const TaskCancelled&) {
                if (!state || !state->cancel_requested()) {
                    throw;  // 无取消请求：按任务异常处理
                }
                state->try_finish_running(TaskCancellationState::Phase::Cancelled);
                auto exception = std::current_exception();
                bool expected = false;
                if (promise_ready->compare_exchange_strong(
                        expected, true, std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    promise->set_exception(exception);
                }
                // 协作取消是生命周期事件，不记 failure。
            } catch (...) {
                auto exception = std::current_exception();
                if (state) {
                    state->try_finish_running(
                        TaskCancellationState::Phase::Failed);
                }
                bool expected = false;
                if (promise_ready->compare_exchange_strong(
                        expected, true, std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    promise->set_exception(exception);
                }
                record_task_exception(
                    executor_name,
                    task_id,
                    "Delayed async task threw an exception",
                    exception);
                throw;
            }
        };

        auto on_timeout = [this, state, executor_name, task_id, promise,
                           promise_ready](std::exception_ptr exception) {
            if (state && !state->try_timeout_before_start()) {
                return;  // 取消已赢或已开始执行
            }
            bool expected = false;
            if (promise_ready->compare_exchange_strong(
                    expected, true, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                promise->set_exception(exception);
            }
            record_task_timeout(
                executor_name,
                task_id,
                "Delayed async task timed out before execution",
                exception);
        };

        if (!executor->try_submit_task(std::move(pool_task), std::move(on_timeout))) {
            reject("Async executor rejected delayed task submission");
        }
    };

    // 派发前取消回调：TimerScheduler 在锁内完成 Scheduled -> Cancelled 仲裁
    // 后调用；future 以 TaskCancelled(Explicit) 就绪，无 failure 事件。
    auto on_cancelled = [state, promise, promise_ready](
                            std::exception_ptr exception) {
        bool expected = false;
        if (state) {
            if (!state->try_cancel_before_start()) {
                return;  // 任务侧已终态（并发窗口），future 已满足
            }
            state->notify_cancelled(exception);
            return;
        }
        if (promise_ready->compare_exchange_strong(
                expected, true, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            promise->set_exception(exception);
        }
    };

    const std::string executor_check_name = "default";
    try {
        start_timer_thread();
    } catch (...) {
        auto exception = std::current_exception();
        bool expected = false;
        if (promise_ready->compare_exchange_strong(
                expected, true, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            promise->set_exception(exception);
        }
        record_submit_rejected(
            executor_check_name,
            task_id,
            "Timer thread creation failed for delayed task",
            exception);
        throw;
    }

    const std::string scheduled_id = ensure_timers().schedule_once(
        delay_ms, timer_id, state, task_state_id, std::move(dispatch),
        std::move(on_cancelled));
    if (scheduled_id.empty()) {
        // 定时器已停止（并发 shutdown 窗口）：按 TaskCancelled(Shutdown)
        // 满足 future；生命周期事件，不写 failure。注意 on_cancelled 已被
        // 移入调度器，这里直接满足 promise，不能调用移空后的闭包。
        auto exception = std::make_exception_ptr(TaskCancelled(
            TaskCancellationReason::Shutdown,
            "Timer stopped before delayed task execution"));
        if (state) {
            if (state->try_cancel_before_start()) {
                state->notify_cancelled(exception);
            }
        } else {
            bool expected = false;
            if (promise_ready->compare_exchange_strong(
                    expected, true, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                promise->set_exception(exception);
            }
        }
    } else {
        submission.handle = TimerHandle(timer_id, timers_);
    }

    return submission;
}

// 批量任务提交模板方法实现
template<typename F>
std::vector<std::future<void>> Executor::submit_batch(const std::vector<F>& tasks) {
    auto executor = manager_->get_default_async_executor_snapshot();
    const std::string executor_name = executor ? executor->get_name() : "default";
    if (!executor) {
        record_submit_rejected(
            executor_name,
            "facade_submit_batch",
            "Async executor not initialized. Call initialize() first.");
        throw std::runtime_error("Async executor not initialized. Call initialize() first.");
    }

    std::vector<std::function<void()>> task_wrappers;
    std::vector<std::function<void(std::exception_ptr)>> timeout_handlers;
    std::vector<std::future<void>> futures;
    std::vector<std::shared_ptr<std::promise<void>>> promises;
    std::vector<std::shared_ptr<std::atomic_bool>> promise_ready_flags;

    task_wrappers.reserve(tasks.size());
    timeout_handlers.reserve(tasks.size());
    futures.reserve(tasks.size());
    promises.reserve(tasks.size());
    promise_ready_flags.reserve(tasks.size());

    bool has_empty_task = false;
    for (size_t i = 0; i < tasks.size(); ++i) {
        auto promise = std::make_shared<std::promise<void>>();
        auto promise_ready = std::make_shared<std::atomic_bool>(false);
        futures.push_back(promise->get_future());
        promises.push_back(promise);
        promise_ready_flags.push_back(promise_ready);

        std::string task_id = "facade_submit_batch[" + std::to_string(i) + "]";

        if (detail::is_empty_std_function(tasks[i])) {
            has_empty_task = true;
        }
        task_wrappers.push_back([this, executor_name, task_id, promise, promise_ready, task = tasks[i]]() mutable {
            try {
                task();
                promise->set_value();
                promise_ready->store(true, std::memory_order_release);
            } catch (...) {
                auto exception = std::current_exception();
                promise->set_exception(exception);
                promise_ready->store(true, std::memory_order_release);
                record_task_exception(
                    executor_name,
                    task_id,
                    "Batch async task threw an exception",
                    exception);
                throw;
            }
        });
        timeout_handlers.push_back(
            [this, executor_name, task_id, promise, promise_ready](
                std::exception_ptr exception) {
                bool expected = false;
                if (promise_ready->compare_exchange_strong(expected, true)) {
                    promise->set_exception(exception);
                    record_task_timeout(
                        executor_name,
                        task_id,
                        "Batch async task timed out before execution",
                        exception);
                }
            });
    }

    if (has_empty_task) {
        auto exception = std::make_exception_ptr(std::invalid_argument("empty task"));
        for (size_t i = 0; i < promises.size(); ++i) {
            promise_ready_flags[i]->store(true, std::memory_order_release);
            promises[i]->set_exception(exception);
        }
        record_submit_rejected(
            executor_name,
            "facade_submit_batch",
            "Async executor rejected batch task submission with empty task",
            exception);
        return futures;
    }

    if (!executor->try_submit_batch_tasks(
            std::move(task_wrappers), std::move(timeout_handlers))) {
        auto exception = std::make_exception_ptr(
            std::runtime_error("Async executor rejected batch task submission"));
        bool marked_any = false;
        for (size_t i = 0; i < promises.size(); ++i) {
            bool expected = false;
            if (promise_ready_flags[i]->compare_exchange_strong(expected, true)) {
                promises[i]->set_exception(exception);
                marked_any = true;
            }
        }
        if (marked_any || tasks.empty()) {
            record_submit_rejected(
                executor_name,
                "facade_submit_batch",
                tasks.empty()
                    ? "Async executor rejected empty batch task submission"
                    : "Async executor rejected batch task submission",
                exception);
        }
    }

    return futures;
}

template<typename F>
std::vector<std::future<void>> Executor::submit_batch_priority(
    int priority,
    const std::vector<F>& tasks) {
    auto executor = manager_->get_default_async_executor_snapshot();
    const std::string executor_name = executor ? executor->get_name() : "default";
    if (!executor) {
        record_submit_rejected(
            executor_name,
            "facade_submit_batch_priority",
            "Async executor not initialized. Call initialize() first.");
        throw std::runtime_error("Async executor not initialized. Call initialize() first.");
    }

    std::vector<std::future<void>> futures;
    futures.reserve(tasks.size());

    for (const auto& task : tasks) {
        futures.push_back(submit_priority(priority, task));
    }

    return futures;
}

template<typename F>
void Executor::submit_batch_no_future(const std::vector<F>& tasks) {
    auto executor = manager_->get_default_async_executor_snapshot();
    const std::string executor_name = executor ? executor->get_name() : "default";
    if (!executor) {
        record_submit_rejected(
            executor_name,
            "facade_submit_batch_no_future",
            "Async executor not initialized. Call initialize() first.");
        throw std::runtime_error("Async executor not initialized. Call initialize() first.");
    }

    std::vector<std::function<void()>> task_wrappers;
    task_wrappers.reserve(tasks.size());
    auto execution_failure_seen = std::make_shared<std::atomic_bool>(false);

    for (size_t i = 0; i < tasks.size(); ++i) {
        std::string task_id =
            "facade_submit_batch_no_future[" + std::to_string(i) + "]";

        task_wrappers.push_back([this, executor_name, task_id, execution_failure_seen, task = tasks[i]]() mutable {
            try {
                task();
            } catch (...) {
                auto exception = std::current_exception();
                execution_failure_seen->store(true, std::memory_order_release);
                record_task_exception(
                    executor_name,
                    task_id,
                    "Fire-and-forget batch async task threw an exception",
                    exception);
                throw;
            }
        });
    }

    if (!executor->try_submit_batch_tasks(std::move(task_wrappers))) {
        auto exception = std::make_exception_ptr(
            std::runtime_error("Async executor rejected fire-and-forget batch task submission"));
        if (!execution_failure_seen->load(std::memory_order_acquire)) {
            record_submit_rejected(
                executor_name,
                "facade_submit_batch_no_future",
                tasks.empty()
                    ? "Async executor rejected empty fire-and-forget batch task submission"
                    : "Async executor rejected fire-and-forget batch task submission",
                exception);
        }
    }
}

// GPU 任务提交模板方法实现
template<typename KernelFunc>
auto Executor::submit_gpu(const std::string& executor_name,
                         KernelFunc&& kernel,
                         const gpu::GpuTaskConfig& config)
    -> std::future<void> {
    auto executor = manager_->get_gpu_executor_snapshot(executor_name);
    if (!executor) {
        const std::string message =
            "submit_gpu: no GPU executor registered with name " + executor_name;
        record_submit_rejected(executor_name, "facade_submit_gpu", message);
        throw std::runtime_error("GPU executor '" + executor_name + "' not found. Call register_gpu_executor() first.");
    }
    return executor->submit_kernel(std::forward<KernelFunc>(kernel), config);
}

// 智能调度模板方法实现
template<typename KernelFunc>
auto Executor::submit_auto(
    const gpu::TaskCharacteristics& characteristics,
    const std::string& gpu_executor_name,
    KernelFunc&& kernel,
    const gpu::GpuTaskConfig& gpu_config)
    -> std::future<void> {

    TaskOptions routing_options;
    routing_options.name = "facade_submit_auto_legacy";
    routing_options.intent = ExecutionIntent::CpuOrGpu;
    routing_options.preferred_executor = gpu_executor_name;
    record_routing_decision(route_task(
        routing_options,
        true,
        scheduler_.decide(characteristics) == gpu::ExecutorChoice::GPU));

    auto choice = scheduler_.decide(characteristics);

    if (choice == gpu::ExecutorChoice::GPU) {
        return submit_gpu(gpu_executor_name, std::forward<KernelFunc>(kernel), gpu_config);
    } else {
        // CPU fallback: execute kernel with nullptr stream
        return submit([kernel = std::forward<KernelFunc>(kernel)]() mutable {
            kernel(nullptr);
        });
    }
}

template<typename F, typename... Args>
auto Executor::submit_auto(F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type> {
    TaskOptions options;
    record_routing_decision(route_task(options, false));
    return submit(std::forward<F>(f), std::forward<Args>(args)...);
}

template<typename Function>
auto Executor::submit_auto(TaskBuilder<Function> task)
    -> std::future<typename std::invoke_result<Function&>::type> {
    const auto& options = task.options();
    const auto decision = route_task(options, false);
    record_routing_decision(decision);
    if (decision.reason == RoutingReason::Rejected) {
        const std::string message = "submit_auto: " + decision.detail;
        record_submit_rejected("default", options.name, message);
        std::promise<typename std::invoke_result<Function&>::type> promise;
        promise.set_exception(std::make_exception_ptr(std::runtime_error(message)));
        return promise.get_future();
    }
    return submit_priority(static_cast<int>(options.priority), std::move(task).function());
}

template<typename CpuFunction, typename GpuFunction>
std::future<void> Executor::submit_auto(CpuGpuTask<CpuFunction, GpuFunction> task) {
    static_assert(requires(CpuFunction& cpu) {
                      { cpu() } -> std::same_as<void>;
                  },
                  "CpuGpuTask CPU callable must be invocable with no arguments and return void");
    static_assert(requires(GpuFunction& gpu) {
                      { gpu(static_cast<void*>(nullptr)) } -> std::same_as<void>;
                  } || requires(GpuFunction& gpu) {
                      { gpu() } -> std::same_as<void>;
                  },
                  "CpuGpuTask GPU callable must be invocable with void* stream or no arguments and return void");

    const auto& options = task.options();
    const auto task_name = options.name.empty() ? "facade_submit_auto" : options.name;
    auto decision = route_task(options, true);
    if (decision.selected_backend == ExecutionBackend::Gpu &&
        options.fallback != FallbackPolicy::RequireRequestedBackend) {
        decision = route_task(
            options, true, scheduler_.decide(task.characteristics()) == gpu::ExecutorChoice::GPU);
    }
    record_routing_decision(decision);

    const auto reject = [this, &task_name, &decision](const std::string& message) {
        record_submit_rejected(decision.selected_executor_name.empty()
                                   ? "gpu" : decision.selected_executor_name,
                               task_name, message);
        std::promise<void> promise;
        promise.set_exception(std::make_exception_ptr(std::runtime_error(message)));
        return promise.get_future();
    };

    if (decision.reason == RoutingReason::Rejected ||
        (decision.selected_backend == ExecutionBackend::DefaultAsync && !decision.fell_back &&
         options.fallback != FallbackPolicy::AllowCpu)) {
        return reject("submit_auto: " + decision.detail);
    }

    if (decision.selected_backend == ExecutionBackend::DefaultAsync) {
        return submit(std::move(task).take_cpu());
    }

    const auto& gpu_name = decision.selected_executor_name;
    try {
        auto gpu_config = task.gpu_config();
        return submit_gpu(gpu_name, std::move(task).take_gpu(), gpu_config);
    } catch (const std::exception& error) {
        if (options.fallback == FallbackPolicy::AllowCpu) {
            RoutingDecision fallback = decision;
            fallback.selected_backend = ExecutionBackend::DefaultAsync;
            fallback.selected_executor_name = "default";
            fallback.reason = RoutingReason::FallbackPolicy;
            fallback.fell_back = true;
            fallback.detail = std::string("GPU submission rejected; falling back to CPU: ") + error.what();
            fallback.timestamp = std::chrono::steady_clock::now();
            record_routing_decision(std::move(fallback));
            return submit(std::move(task).take_cpu());
        }
        return reject(std::string("submit_auto: GPU submission rejected: ") + error.what());
    }
}

} // namespace executor
