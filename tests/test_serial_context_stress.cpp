// W1 压测（mira_feedback_update_plan.md）：多 worker 突发 FIFO、worker 数扫描、
// 突发中取消/超时/context shutdown/executor shutdown 的交错收敛。
// 旧实现（阻塞 wrapper）在两 worker × 10,000 突发下首 future 30s 超时
//（Mira 台账 EXE-20260830-002 复现口径），本测试是回归门。
#include <executor/executor.hpp>

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

// 单纯提交并消费 kCount 个串行提交，校验 ticket FIFO 与有界时长。
void run_burst_fifo(unsigned workers, int count, std::chrono::seconds budget) {
    executor::Executor ex;
    executor::ExecutorConfig config;
    config.min_threads = workers;
    config.max_threads = workers;
    assert(ex.initialize(config));
    executor::SerialExecutionContext context;

    std::vector<int> executed;
    std::vector<std::future<void>> futures;
    futures.reserve(static_cast<size_t>(count));

    const auto start = Clock::now();
    for (int i = 0; i < count; ++i) {
        futures.push_back(ex.submit_on(context, [i, &executed] {
            // 串行线程按 ticket 顺序执行：第 i 个提交的回调必须看到第 i 个槽位。
            executed.push_back(i);
        }));
    }
    for (auto& future : futures) future.get();
    const auto elapsed = Clock::now() - start;

    assert(executed.size() == static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        assert(executed[static_cast<size_t>(i)] == i);
    }
    assert(elapsed < budget);
    context.shutdown();
    ex.shutdown();
}

}  // namespace

int main() {
    // 台账复现口径：两 worker、10,000 次快速回调。旧实现约 60s 失败；
    // 非阻塞发布下与 worker 数无关（Mira 单 worker 基线约 0.51s）。
    run_burst_fifo(2, 10000, std::chrono::seconds(30));

    // worker 数 1..4 扫描：later wrapper 先启动不能阻止 earlier ticket。
    for (unsigned workers = 1; workers <= 4; ++workers) {
        run_burst_fifo(workers, 2000, std::chrono::seconds(20));
    }

    // 突发中的排队取消：被取消提交的 future 以 TaskCancelled 就绪，
    // 未取消提交仍按 FIFO 完成（ticket 被释放，后续不阻塞）。
    {
        executor::Executor ex;
        executor::ExecutorConfig config;
        config.min_threads = 2;
        config.max_threads = 2;
        assert(ex.initialize(config));
        executor::SerialExecutionContext context;

        constexpr int kCount = 4000;
        std::vector<executor::TaskSubmission<void>> submissions;
        submissions.reserve(kCount);
        std::atomic<int> executed{0};
        for (int i = 0; i < kCount; ++i) {
            submissions.push_back(ex.submit_on_with_handle(
                context, [&executed] { executed.fetch_add(1, std::memory_order_relaxed); }));
        }
        int cancelled_seen = 0;
        for (int i = kCount / 4; i < kCount / 2; ++i) {
            const auto response = ex.request_task_cancel(submissions[static_cast<size_t>(i)].handle);
            // 突发下任务可能已执行：只统计排队取消赢家，语义两者皆合法。
            if (response.result == executor::TaskCancellationResult::RequestedBeforeStart) {
                ++cancelled_seen;
            }
        }
        for (int i = 0; i < kCount; ++i) {
            auto& submission = submissions[static_cast<size_t>(i)];
            bool settled = false;
            try {
                submission.future.get();
                settled = true;
            } catch (const executor::TaskCancelled&) {
                settled = true;
            }
            assert(settled);
        }
        assert(executed.load() + cancelled_seen <= kCount);
        assert(executed.load() >= kCount / 2);  // 取消窗口之后的提交全部执行
        context.shutdown();
        ex.shutdown();
    }

    // queued soft timeout：派发任务在队列中过期，被 worker 出队时跳过执行，
    // future 以 TimedOutException 就绪且释放 ticket，后续提交不被拖死。
    // 注意超时在 worker 出队时评估：先用两个 sleep 占满 worker，让派发任务
    // 在队列里停留超过 timeout 窗口。
    {
        executor::Executor ex;
        executor::ExecutorConfig config;
        config.min_threads = 2;
        config.max_threads = 2;
        config.task_timeout_ms = 50;
        assert(ex.initialize(config));
        executor::SerialExecutionContext context;

        auto sleeper1 = ex.submit([] { std::this_thread::sleep_for(std::chrono::milliseconds(150)); });
        auto sleeper2 = ex.submit([] { std::this_thread::sleep_for(std::chrono::milliseconds(150)); });

        auto timed_out = ex.submit_on_with_handle(context, [] { assert(false && "must time out"); });
        bool saw_timeout = false;
        try { (void)timed_out.future.get(); }
        catch (const executor::TimedOutException&) { saw_timeout = true; }
        assert(saw_timeout);
        // 超时已释放 ticket：后续提交不被拖死。
        auto after_timeout = ex.submit_on_with_handle(context, [] { return 42; });
        sleeper1.get();
        sleeper2.get();
        assert(after_timeout.future.get() == 42);
        context.shutdown();
        ex.shutdown();
    }

    // 突发中并发 context shutdown：所有 future 在有界时间内结算
    //（已发布的回调运行至返回，未发布的以 ExecutorStopping 就绪）。
    {
        executor::Executor ex;
        executor::ExecutorConfig config;
        config.min_threads = 2;
        config.max_threads = 2;
        assert(ex.initialize(config));
        executor::SerialExecutionContext context;

        constexpr int kCount = 4000;
        std::vector<std::future<int>> futures;
        futures.reserve(kCount);
        std::atomic<bool> shutdown_started{false};
        std::thread shutdown_thread([&] {
            shutdown_started.store(true, std::memory_order_release);
            context.shutdown();
        });
        while (!shutdown_started.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        for (int i = 0; i < kCount; ++i) {
            futures.push_back(ex.submit_on(context, [i] { return i; }));
        }
        shutdown_thread.join();
        int values = 0;
        int stopped = 0;
        for (auto& future : futures) {
            try {
                values += future.get() >= 0 ? 1 : 0;
            } catch (const executor::ExecutorStopping&) {
                ++stopped;
            }
        }
        assert(values + stopped == kCount);
        ex.shutdown();
    }

    // executor shutdown（drain）不留下未就绪 future：提交后立即关闭，
    // 派发任务全部被执行，串行回调在上下文排空中完成。
    {
        executor::Executor ex;
        executor::ExecutorConfig config;
        config.min_threads = 2;
        config.max_threads = 2;
        assert(ex.initialize(config));
        executor::SerialExecutionContext context;

        constexpr int kCount = 2000;
        std::vector<std::future<void>> futures;
        futures.reserve(kCount);
        for (int i = 0; i < kCount; ++i) {
            futures.push_back(ex.submit_on(context, [] {}));
        }
        ex.shutdown();
        for (auto& future : futures) future.get();
        context.shutdown();
    }

    return 0;
}
