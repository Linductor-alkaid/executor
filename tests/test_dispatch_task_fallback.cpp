#include "executor/thread_pool/task_dispatcher.hpp"
#include "executor/thread_pool/thread_pool.hpp"
#include "executor/thread_pool/load_balancer.hpp"
#include "executor/thread_pool/priority_scheduler.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <string>
#include <vector>

using namespace executor;

namespace {

void make_task(Task& task, const std::string& id, std::atomic<int>& completed) {
    task.task_id = id;
    task.priority = TaskPriority::NORMAL;
    task.function = [&completed] {
        completed.fetch_add(1, std::memory_order_relaxed);
    };
}

std::vector<WorkerQueueImpl> make_queues(size_t count, size_t capacity) {
    std::vector<WorkerQueueImpl> queues;
    queues.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        queues.emplace_back(capacity);
    }
    return queues;
}

}  // namespace

TEST(DispatchTaskFallbackTest, LocalQueueFullFallbackReenqueuesTask) {
    LoadBalancer balancer(1);
    PriorityScheduler scheduler;
    auto queues = make_queues(1, 1);
    TaskDispatcher<WorkerQueueImpl> dispatcher(balancer, scheduler, queues);
    std::atomic<int> completed{0};

    Task resident;
    make_task(resident, "resident", completed);
    ASSERT_TRUE(queues[0].push(resident));
    Task fallback;
    make_task(fallback, "queue-full-fallback", completed);

    EXPECT_FALSE(dispatcher.dispatch_task(fallback));
    EXPECT_EQ(queues[0].size(), 1U);
    ASSERT_EQ(scheduler.size(), 1U);

    Task recovered;
    ASSERT_TRUE(scheduler.dequeue(recovered));
    EXPECT_EQ(recovered.task_id, "queue-full-fallback");
    ASSERT_TRUE(recovered.function);
    recovered.function();
    EXPECT_EQ(completed.load(std::memory_order_relaxed), 1);
}

TEST(DispatchTaskFallbackTest, OutOfRangeWorkerIdFallbackReenqueuesTask) {
    // The balancer still believes two workers exist while only one local queue
    // is available, matching the transient resize state guarded by dispatch_task.
    LoadBalancer balancer(2);
    PriorityScheduler scheduler;
    auto queues = make_queues(1, 1);
    TaskDispatcher<WorkerQueueImpl> dispatcher(balancer, scheduler, queues);
    std::atomic<int> completed{0};

    ASSERT_EQ(balancer.select_worker(), 0U);
    Task fallback;
    make_task(fallback, "out-of-range-fallback", completed);

    EXPECT_FALSE(dispatcher.dispatch_task(fallback));
    EXPECT_TRUE(queues[0].empty());
    ASSERT_EQ(scheduler.size(), 1U);

    Task recovered;
    ASSERT_TRUE(scheduler.dequeue(recovered));
    EXPECT_EQ(recovered.task_id, "out-of-range-fallback");
    ASSERT_TRUE(recovered.function);
    recovered.function();
    EXPECT_EQ(completed.load(std::memory_order_relaxed), 1);
}
