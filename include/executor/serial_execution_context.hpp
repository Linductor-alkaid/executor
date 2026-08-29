#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

namespace executor {

/** A small FIFO, single-threaded context usable with Executor::submit_on. */
class SerialExecutionContext {
public:
    SerialExecutionContext() : worker_([this] { run(); }) {}
    ~SerialExecutionContext() { shutdown(); }

    SerialExecutionContext(const SerialExecutionContext&) = delete;
    SerialExecutionContext& operator=(const SerialExecutionContext&) = delete;

    bool post(std::function<void()> task) {
        if (!task) return false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) return false;
            queue_.push(std::move(task));
        }
        cv_.notify_one();
        return true;
    }

    void shutdown() noexcept {
        bool should_join = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                should_join = true;
            } else {
                stopping_ = true;
            }
        }
        if (!should_join) cv_.notify_all();
        if (worker_.joinable() && std::this_thread::get_id() != worker_.get_id())
            worker_.join();
    }

    bool is_stopped() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return stopping_;
    }

private:
    void run() noexcept {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
                if (stopping_ && queue_.empty()) return;
                task = std::move(queue_.front());
                queue_.pop();
            }
            try { task(); } catch (...) { /* submit_on owns exception delivery */ }
        }
    }

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<std::function<void()>> queue_;
    bool stopping_ = false;
    std::thread worker_;
};

} // namespace executor
