#pragma once
#include <fstream>
#include <mutex>
#include <string>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

class DaemonLogger {
public:
    explicit DaemonLogger(const std::string& path)
        : out_(path, std::ios::app) {}

    template <typename... Args>
    void logf(const char* fmt, Args&&... args) {
        std::lock_guard<std::mutex> lk(mx_);
        if (!out_) return;
        out_ << ts() << " " << format(fmt, std::forward<Args>(args)...) << "\n";
        out_.flush();
    }

private:
    std::string ts() {
        using namespace std::chrono;
        auto now = system_clock::now();
        auto t = system_clock::to_time_t(now);
        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        char buf[64];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
        return std::string("[") + buf + "]";
    }

    template <typename... Args>
    static std::string format(const char* fmt, Args&&... args) {
        int size = std::snprintf(nullptr, 0, fmt, args...) + 1;
        std::string buf;
        buf.resize(size);
        std::snprintf(&buf[0], size, fmt, args...);
        if (!buf.empty() && buf.back() == '\0') buf.pop_back();
        return buf;
    }

    std::ofstream out_;
    std::mutex mx_;
};
