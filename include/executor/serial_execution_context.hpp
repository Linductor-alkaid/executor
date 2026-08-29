#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <map>
#include <optional>
#include <set>
#include <thread>

namespace executor {

/** A small FIFO, single-threaded context usable with Executor::submit_on. */
class SerialExecutionContext {
public:
    using Ticket = uint64_t;

    SerialExecutionContext() : worker_([this] { run(); }) {}
    ~SerialExecutionContext() { shutdown(); }

    SerialExecutionContext(const SerialExecutionContext&) = delete;
    SerialExecutionContext& operator=(const SerialExecutionContext&) = delete;

    bool post(std::function<void()> task) {
        auto ticket = reserve();
        return ticket && post_reserved(*ticket, std::move(task));
    }

    std::optional<Ticket> reserve() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) return std::nullopt;
        const Ticket ticket = next_ticket_++;
        reserved_.insert(ticket);
        return ticket;
    }

    bool post_reserved(Ticket ticket, std::function<void()> task) {
        if (!task) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (reserved_.find(ticket) == reserved_.end()) return false;
            abandon_locked(ticket);
            cv_.notify_one();
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                // shutdown() normally clears all reservations while holding
                // this mutex.  Preserve already-published callbacks if a
                // late publisher arrives after that point.
                if (reserved_.find(ticket) != reserved_.end()) {
                    abandon_locked(ticket);
                }
                return false;
            }
            if (reserved_.erase(ticket) == 0) {
                // A ticket can only be published once.  In particular, do
                // not erase the callback accepted by an earlier publisher.
                return false;
            }
            pending_.emplace(ticket, std::move(task));
            release_ready_locked();
        }
        cv_.notify_one();
        return true;
    }

    void abandon(Ticket ticket) noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            abandon_locked(ticket);
        }
        cv_.notify_one();
    }

    void shutdown() noexcept {
        bool should_join = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                should_join = true;
            } else {
                stopping_ = true;
                // Reservations belong to facade wrappers that may still be
                // waiting for a worker to publish their callback.  Skipping
                // them here prevents one such wrapper from blocking later
                // already-published work during shutdown.
                for (const auto ticket : reserved_) {
                    skipped_.insert(ticket);
                }
                reserved_.clear();
                release_ready_locked();
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

    void abandon_locked(Ticket ticket) noexcept {
        reserved_.erase(ticket);
        pending_.erase(ticket);
        // Ignore duplicate/late abandonment after the ticket has already
        // crossed the ready watermark.
        if (ticket >= next_ready_ && ticket < next_ticket_) {
            skipped_.insert(ticket);
        }
        release_ready_locked();
    }

    void release_ready_locked() {
        for (;;) {
            auto skipped = skipped_.find(next_ready_);
            if (skipped != skipped_.end()) {
                skipped_.erase(skipped);
                ++next_ready_;
                continue;
            }
            auto pending = pending_.find(next_ready_);
            if (pending == pending_.end()) return;
            queue_.push(std::move(pending->second));
            pending_.erase(pending);
            ++next_ready_;
        }
    }

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<std::function<void()>> queue_;
    std::map<Ticket, std::function<void()>> pending_;
    std::set<Ticket> reserved_;
    std::set<Ticket> skipped_;
    Ticket next_ticket_ = 0;
    Ticket next_ready_ = 0;
    bool stopping_ = false;
    std::thread worker_;
};

} // namespace executor
