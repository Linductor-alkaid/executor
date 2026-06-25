#include "executor/thread_pool/lockfree_worker_queue.hpp"
#include "executor/task/task.hpp"
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>

using namespace executor;

bool test_basic_operations() {
    std::cout << "Testing basic push/pop..." << std::endl;

    LockFreeWorkerQueue queue(100);

    Task task1;
    task1.task_id = "task_1";
    task1.priority = TaskPriority::NORMAL;
    task1.function = []() {};

    if (!queue.push(task1)) {
        std::cerr << "FAILED: push failed" << std::endl;
        return false;
    }

    if (queue.size() != 1) {
        std::cerr << "FAILED: size should be 1, got " << queue.size() << std::endl;
        return false;
    }

    Task popped;
    if (!queue.pop(popped)) {
        std::cerr << "FAILED: pop failed" << std::endl;
        return false;
    }

    if (popped.task_id != "task_1") {
        std::cerr << "FAILED: task_id mismatch" << std::endl;
        return false;
    }

    std::cout << "PASSED: basic operations" << std::endl;
    return true;
}

bool test_batch_operations() {
    std::cout << "Testing batch push..." << std::endl;

    LockFreeWorkerQueue queue(100);

    std::vector<Task> tasks(10);
    for (size_t i = 0; i < 10; ++i) {
        tasks[i].task_id = "task_" + std::to_string(i);
        tasks[i].priority = TaskPriority::NORMAL;
        tasks[i].function = []() {};
    }

    size_t pushed = queue.push_batch(tasks.data(), 10);
    if (pushed != 10) {
        std::cerr << "FAILED: pushed " << pushed << " tasks, expected 10" << std::endl;
        return false;
    }

    if (queue.size() != 10) {
        std::cerr << "FAILED: size should be 10, got " << queue.size() << std::endl;
        return false;
    }

    std::cout << "PASSED: batch operations" << std::endl;
    return true;
}

// P-260625-001: 回归测试。push_batch 部分成功时必须清理未入队的 Task。
// 在 ASan 下，该测试也充当泄漏断言；反复跑多次仍应零泄漏。
bool test_push_batch_partial_fill_no_leak() {
    std::cout << "Testing push_batch partial-fill cleanup (P-260625-001)..." << std::endl;

    constexpr size_t CAP = 8;
    constexpr size_t PRESEED = 6;
    constexpr size_t BATCH = 5;
    constexpr size_t ROUNDS = 200;

    for (size_t round = 0; round < ROUNDS; ++round) {
        LockFreeWorkerQueue queue(CAP);

        for (size_t i = 0; i < PRESEED; ++i) {
            Task t;
            t.task_id = "seed_" + std::to_string(i);
            t.priority = TaskPriority::NORMAL;
            t.function = []() {};
            if (!queue.push(t)) {
                std::cerr << "FAILED round " << round
                          << ": seed push " << i << " returned false" << std::endl;
                return false;
            }
        }
        if (queue.size() != PRESEED) {
            std::cerr << "FAILED round " << round
                      << ": size after seed = " << queue.size()
                      << ", expected " << PRESEED << std::endl;
            return false;
        }

        std::vector<Task> batch(BATCH);
        for (size_t i = 0; i < BATCH; ++i) {
            batch[i].task_id = "batch_" + std::to_string(i);
            batch[i].priority = TaskPriority::NORMAL;
            batch[i].function = []() {};
        }

        size_t pushed = queue.push_batch(batch.data(), BATCH);
        if (pushed > 1) {
            std::cerr << "FAILED round " << round
                      << ": pushed = " << pushed
                      << " > available 1" << std::endl;
            return false;
        }

        if (queue.size() != PRESEED + pushed) {
            std::cerr << "FAILED round " << round
                      << ": size after batch = " << queue.size()
                      << ", expected " << (PRESEED + pushed) << std::endl;
            return false;
        }
    }

    std::cout << "PASSED: push_batch partial-fill cleanup"
              << " (" << ROUNDS << " rounds)" << std::endl;
    return true;
}

bool test_steal() {
    std::cout << "Testing steal..." << std::endl;

    LockFreeWorkerQueue queue(100);

    for (int i = 0; i < 5; ++i) {
        Task task;
        task.task_id = "task_" + std::to_string(i);
        task.priority = TaskPriority::NORMAL;
        task.function = []() {};
        queue.push(task);
    }

    Task stolen;
    if (!queue.steal(stolen)) {
        std::cerr << "FAILED: steal failed" << std::endl;
        return false;
    }

    std::cout << "PASSED: steal operation" << std::endl;
    return true;
}

bool test_concurrent_push_pop() {
    std::cout << "Testing concurrent push/pop..." << std::endl;

    LockFreeWorkerQueue queue(10000);
    std::atomic<int> push_count{0};
    std::atomic<int> pop_count{0};

    constexpr int NUM_TASKS = 1000;

    // 多个生产者线程
    std::vector<std::thread> producers;
    for (int t = 0; t < 4; ++t) {
        producers.emplace_back([&, t]() {
            for (int i = 0; i < NUM_TASKS / 4; ++i) {
                Task task;
                task.task_id = "task_" + std::to_string(t * 1000 + i);
                task.priority = TaskPriority::NORMAL;
                task.function = []() {};
                if (queue.push(task)) {
                    push_count.fetch_add(1);
                }
            }
        });
    }

    // 单个消费者线程
    std::thread consumer([&]() {
        Task task;
        while (pop_count.load() < NUM_TASKS) {
            if (queue.pop(task)) {
                pop_count.fetch_add(1);
            }
        }
    });

    for (auto& t : producers) t.join();
    consumer.join();

    if (pop_count.load() != NUM_TASKS) {
        std::cerr << "FAILED: popped " << pop_count.load() << " tasks, expected " << NUM_TASKS << std::endl;
        return false;
    }

    std::cout << "PASSED: concurrent push/pop" << std::endl;
    return true;
}

bool test_steal_no_double_consume() {
    std::cout << "Testing pop/steal no double consume..." << std::endl;

    constexpr int ITERATIONS = 10000;

    for (int i = 0; i < ITERATIONS; ++i) {
        LockFreeWorkerQueue queue(8);

        Task task;
        task.task_id = "race_task_" + std::to_string(i);
        task.priority = TaskPriority::NORMAL;
        task.function = []() {};

        if (!queue.push(task)) {
            std::cerr << "FAILED: push failed at iteration " << i << std::endl;
            return false;
        }

        std::atomic<int> ready{0};
        std::atomic<bool> start{false};
        std::atomic<bool> pop_result{false};
        std::atomic<bool> steal_result{false};

        std::thread owner([&]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            Task popped;
            pop_result.store(queue.pop(popped), std::memory_order_release);
        });

        std::thread stealer([&]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            Task stolen;
            steal_result.store(queue.steal(stolen), std::memory_order_release);
        });

        while (ready.load(std::memory_order_acquire) != 2) {
            std::this_thread::yield();
        }
        start.store(true, std::memory_order_release);

        owner.join();
        stealer.join();

        const int consumed = (pop_result.load(std::memory_order_acquire) ? 1 : 0) +
                             (steal_result.load(std::memory_order_acquire) ? 1 : 0);
        if (consumed != 1) {
            std::cerr << "FAILED: iteration " << i << " consumed " << consumed
                      << " tasks; pop=" << pop_result.load()
                      << " steal=" << steal_result.load() << std::endl;
            return false;
        }
    }

    std::cout << "PASSED: pop/steal no double consume" << std::endl;
    return true;
}

int main() {
    bool all_passed = true;

    all_passed &= test_basic_operations();
    all_passed &= test_batch_operations();
    all_passed &= test_steal();
    all_passed &= test_concurrent_push_pop();
    all_passed &= test_steal_no_double_consume();
    all_passed &= test_push_batch_partial_fill_no_leak();

    if (all_passed) {
        std::cout << "\n✓ All tests passed!" << std::endl;
        return 0;
    } else {
        std::cerr << "\n✗ Some tests failed!" << std::endl;
        return 1;
    }
}
