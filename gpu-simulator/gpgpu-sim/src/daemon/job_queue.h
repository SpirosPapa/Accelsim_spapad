// gpu-simulator/gpgpu-sim/src/daemon/job_queue.h
#pragma once
#include <mutex>
#include <condition_variable>
#include <queue>
#include <string>
#include <utility>

// job_queue.h
struct Job {
    std::string id;
    std::string trace_dir;
};

class JobQueue {
public:
    void push(Job j) {
        {
            std::lock_guard<std::mutex> lk(mx_);
            if (closed_) return;               // ignore pushes after shutdown (optional)
            q_.push(std::move(j));
        }
        cv_.notify_one();
    }

    // returns false if queue was closed and empty (caller should exit)
    bool pop_blocking(Job& out) {
        std::unique_lock<std::mutex> lk(mx_);
        cv_.wait(lk, [&]{ return closed_ || !q_.empty(); });
        if (q_.empty()) return false;          // closed and empty
        out = std::move(q_.front());
        q_.pop();
        return true;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lk(mx_);
        return q_.size();
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lk(mx_);
            closed_ = true;
        }
        cv_.notify_all();
    }

private:
    mutable std::mutex mx_;
    std::condition_variable cv_;
    std::queue<Job> q_;
    bool closed_ = false;
};

