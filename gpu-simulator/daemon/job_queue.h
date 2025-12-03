#pragma once
#include <mutex>
#include <condition_variable>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

struct Job {
    std::string id;
    std::string trace_dir;
    std::string out_dir;   // per-job output directory
    bool use_all_sms = true;
    std::vector<unsigned> sm_ids; // SM affinity mask for this job
};

class JobQueue {
public:
    // Push a new job into the queue (called from IPC handler thread)
    void push(Job j) {
        {
            std::lock_guard<std::mutex> lk(mx_);
            if (closed_) return;
            q_.push(std::move(j));
        }
        cv_.notify_one();
    }

    bool pop_blocking(Job& out) {
        std::unique_lock<std::mutex> lk(mx_);
        cv_.wait(lk, [&]{ return closed_ || !q_.empty(); });
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop();
        return true;
    }

    // Non-blocking pop. Reurtns true if a job was popped, false if queue is empty.
    bool try_pop(Job& out) {
        std::lock_guard<std::mutex> lk(mx_);
        if (q_.empty()) return false;
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

    // Take a snapshot of the current job 
    std::vector<Job> snapshot(size_t max_items = 32) const {
        std::lock_guard<std::mutex> lk(mx_);
        std::vector<Job> v;
        v.reserve(std::min(max_items, q_.size()));
        std::queue<Job> tmp = q_; 
        while (!tmp.empty() && v.size() < max_items) {
            v.push_back(tmp.front());
            tmp.pop();
        }
        return v;
    }

private:
    mutable std::mutex mx_;
    std::condition_variable cv_;
    std::queue<Job> q_;
    bool closed_ = false;
};
