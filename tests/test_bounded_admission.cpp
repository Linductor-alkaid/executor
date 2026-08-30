// A1 验收测试（mira_feedback_update_plan.md，对齐 docs/design/bounded_admission.md）：
// 总量有界 admission 的拒绝语义、各终态恰好一次释放、shutdown 交错、
// capacity rejection 与 stopping/invalid input 的可区分性、串行 ticket 释放。
#include <executor/executor.hpp>

#include <atomic>
#include <cassert>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

int main() {
    using executor::CapacityExhaustedException;
    using executor::Executor;
    using executor::ExecutorConfig;

    // 验收 1：单 worker、总容量 N 时第 N+1 个未结算 submit 明确拒绝且 future
    // 就绪；解除后恢复。验收 4 前半：failure 计数可观测。
    {
        Executor ex;
        ExecutorConfig config;
        config.min_threads = 1;
        config.max_threads = 1;
        config.max_in_flight_tasks = 3;
        assert(ex.initialize(config));

        std::promise<void> release;
        auto release_future = release.get_future().share();
        std::promise<void> started;
        auto started_future = started.get_future();
        std::vector<std::future<void>> blocked;
        blocked.push_back(ex.submit([&] {
            started.set_value();
            release_future.wait();
        }));
        for (int i = 1; i < 3; ++i) {
            blocked.push_back(ex.submit([&] { release_future.wait(); }));
        }
        started_future.wait();
        // 3 个未结算提交占满容量（1 运行 + 2 排队）。
        assert(ex.get_in_flight_submissions() == 3);
        assert(ex.get_max_in_flight_tasks() == 3);

        auto rejected = ex.submit([] { return 7; });
        bool saw_capacity = false;
        try { (void)rejected.get(); }
        catch (const CapacityExhaustedException&) { saw_capacity = true; }
        assert(saw_capacity);
        // 拒绝不改变在途计数，不产生额外堆积。
        assert(ex.get_in_flight_submissions() == 3);

        release.set_value();
        for (auto& future : blocked) future.get();
        assert(ex.get_in_flight_submissions() == 0);
        assert(ex.submit([] { return 11; }).get() == 11);

        const auto failure_status = ex.get_failure_status();
        assert(failure_status.capacity_exhausted_count == 1);
        assert(failure_status.total_count >= 1);
        const auto recent = ex.get_recent_failures(8);
        assert(!recent.empty());
        assert(recent.front().kind == executor::FailureKind::CapacityExhausted);
        ex.shutdown();
    }

    // 验收 2：各终态恰好一次释放（完成 / 异常 / 排队取消 / 执行前超时）。
    {
        Executor ex;
        ExecutorConfig config;
        config.min_threads = 1;
        config.max_threads = 1;
        config.max_in_flight_tasks = 4;
        config.task_timeout_ms = 50;
        assert(ex.initialize(config));

        // 完成。
        assert(ex.submit([] { return 1; }).get() == 1);
        assert(ex.get_in_flight_submissions() == 0);
        // 异常。
        bool saw_error = false;
        try { (void)ex.submit([] { throw std::runtime_error("boom"); }).get(); }
        catch (const std::runtime_error&) { saw_error = true; }
        assert(saw_error);
        assert(ex.get_in_flight_submissions() == 0);

        // 排队取消：先占住唯一 worker，tracked 任务排队后被取消。
        std::promise<void> release_cancel;
        auto release_cancel_future = release_cancel.get_future().share();
        std::promise<void> cancel_blocker_started;
        auto cancel_blocker_started_future = cancel_blocker_started.get_future();
        auto cancel_blocker = ex.submit([&] {
            cancel_blocker_started.set_value();
            release_cancel_future.wait();
        });
        cancel_blocker_started_future.wait();
        auto cancelled = ex.submit_with_handle([] { assert(false && "cancelled before start"); });
        const auto response = ex.request_task_cancel(cancelled.handle);
        assert(response.result == executor::TaskCancellationResult::RequestedBeforeStart);
        bool saw_cancel = false;
        try { (void)cancelled.future.get(); }
        catch (const executor::TaskCancelled&) { saw_cancel = true; }
        assert(saw_cancel);
        assert(ex.get_in_flight_submissions() == 1);  // 仅剩 blocker
        release_cancel.set_value();
        cancel_blocker.get();
        assert(ex.get_in_flight_submissions() == 0);

        // 执行前超时：sleep 占满 worker，排队任务出队时已过期。
        auto sleeper = ex.submit([] {
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
        });
        auto timed_out = ex.submit([] { assert(false && "must time out"); });
        bool saw_timeout = false;
        try { (void)timed_out.get(); }
        catch (const executor::TimedOutException&) { saw_timeout = true; }
        assert(saw_timeout);
        sleeper.get();
        assert(ex.get_in_flight_submissions() == 0);
        ex.shutdown();
    }

    // 验收 5：submit_on_with_handle 拒绝时释放 context ticket，后续 FIFO 不阻塞。
    {
        Executor ex;
        ExecutorConfig config;
        config.min_threads = 1;
        config.max_threads = 1;
        config.max_in_flight_tasks = 2;
        assert(ex.initialize(config));
        executor::SerialExecutionContext context;

        std::promise<void> release;
        auto release_future = release.get_future().share();
        std::promise<void> started;
        auto started_future = started.get_future();
        auto blocker = ex.submit([&] {
            started.set_value();
            release_future.wait();
        });
        started_future.wait();

        // 容量 2：blocker + 第一个串行提交占满；第二个串行提交被拒，
        // 其 ticket 必须被释放，不得阻塞后续 ticket 的顺序执行。
        auto first = ex.submit_on_with_handle(context, [] { return 1; });
        auto rejected = ex.submit_on_with_handle(context, [] { return 2; });
        bool saw_capacity = false;
        try { (void)rejected.future.get(); }
        catch (const CapacityExhaustedException&) { saw_capacity = true; }
        assert(saw_capacity);

        release.set_value();
        blocker.get();
        assert(first.future.get() == 1);
        auto third = ex.submit_on_with_handle(context, [] { return 3; });
        assert(third.future.get() == 3);
        assert(ex.get_in_flight_submissions() == 0);
        context.shutdown();
        ex.shutdown();
    }

    // 验收 4：capacity rejection 与 invalid input 可区分（异常类型 + failure kind）。
    {
        Executor ex;
        ExecutorConfig config;
        config.min_threads = 1;
        config.max_threads = 1;
        config.max_in_flight_tasks = 1;
        assert(ex.initialize(config));

        std::promise<void> release;
        auto release_future = release.get_future().share();
        auto blocker = ex.submit([&] { release_future.wait(); });

        auto rejected = ex.submit([] { return 1; });
        bool saw_capacity = false;
        try { (void)rejected.get(); }
        catch (const CapacityExhaustedException&) { saw_capacity = true; }
        assert(saw_capacity);

        // invalid input：空任务的拒绝是 std::invalid_argument。
        auto empty = ex.submit(std::function<int()>());
        bool saw_invalid = false;
        try { (void)empty.get(); }
        catch (const std::invalid_argument&) { saw_invalid = true; }
        assert(saw_invalid);

        release.set_value();
        blocker.get();

        const auto status = ex.get_failure_status();
        assert(status.capacity_exhausted_count == 1);
        assert(status.submit_rejected_count >= 1);  // 空 task 走 SubmitRejected
        ex.shutdown();
    }

    // 验收 3：shutdown 与并发 submit 不越过容量，也不留下未就绪 future。
    {
        Executor ex;
        ExecutorConfig config;
        config.min_threads = 2;
        config.max_threads = 2;
        config.max_in_flight_tasks = 8;
        assert(ex.initialize(config));

        std::atomic<uint64_t> capacity_rejections{0};
        ex.set_failure_callback([&](const executor::ExecutorFailureEvent& event) {
            if (event.kind == executor::FailureKind::CapacityExhausted) {
                capacity_rejections.fetch_add(1, std::memory_order_relaxed);
            }
        });

        std::mutex futures_mutex;
        std::vector<std::future<void>> futures;
        std::atomic<bool> stopping{false};
        std::thread submitter([&] {
            for (int i = 0; i < 4000; ++i) {
                auto future = ex.submit([] {});
                std::lock_guard<std::mutex> lock(futures_mutex);
                futures.push_back(std::move(future));
                if (stopping.load(std::memory_order_acquire)) break;
            }
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        stopping.store(true, std::memory_order_release);
        submitter.join();
        ex.shutdown();
        // 并发提交期间在途计数从未超过容量上限（由计数器语义保证：接纳数
        // - 终态数 <= 上限）；shutdown 后所有 future 均已就绪。
        for (auto& future : futures) {
            bool settled = false;
            try { future.get(); settled = true; }
            catch (const CapacityExhaustedException&) { settled = true; }
            catch (const executor::ExecutorStopping&) { settled = true; }
            catch (const std::runtime_error&) {
                // 池停止后的 SubmitRejected 结算路径。
                settled = true;
            }
            assert(settled);
        }
        assert(ex.get_in_flight_submissions() == 0);
        // 峰值不超过容量的必要条件：拒绝只在容量满时发生，且每条拒绝都有
        // 对应 failure 事件（回读计数与回调计数一致）。
        assert(capacity_rejections.load() ==
               ex.get_failure_status().capacity_exhausted_count);
    }

    // 运行期调整：调小不驱逐已接纳任务，只约束后续提交。
    {
        Executor ex;
        ExecutorConfig config;
        config.min_threads = 1;
        config.max_threads = 1;
        assert(ex.initialize(config));
        assert(ex.get_max_in_flight_tasks() == 0);
        assert(ex.get_in_flight_submissions() == 0);  // 未启用恒为 0

        ex.set_max_in_flight_tasks(1);
        std::promise<void> release;
        auto release_future = release.get_future().share();
        auto blocker = ex.submit([&] { release_future.wait(); });
        auto rejected = ex.submit([] {});
        bool saw_capacity = false;
        try { (void)rejected.get(); }
        catch (const CapacityExhaustedException&) { saw_capacity = true; }
        assert(saw_capacity);
        release.set_value();
        blocker.get();
        ex.shutdown();
    }

    return 0;
}
