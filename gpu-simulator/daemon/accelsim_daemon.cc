// src/daemon/accelsim_daemon.cc
#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

#include "context.h"
#include "execution_engine.h"
#include "ipc_server.h"
#include "job_queue.h"

namespace fs = std::filesystem;

static std::atomic<bool> g_stop{false};

static void on_sigint(int) { g_stop = true; }

static std::string now_compact() {
    using namespace std::chrono;
    auto t = system_clock::now();
    auto tt = system_clock::to_time_t(t);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%04d%02d%02dT%02d%02d%02d",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec);
    return buf;
}

struct DaemonState {
    JobQueue queue;
    std::atomic<bool> draining{false};
    std::atomic<bool> running{false};
    std::string current_job;
};

static std::string trim(const std::string &s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    if (a == std::string::npos) return std::string();
    return s.substr(a, b - a + 1);
}

// Protocol:
//  SUBMIT <abs_trace_dir>
//  STATUS
//  DRAIN_AND_EXIT
static std::string handle_line(DaemonState &st, const std::string &line) {
    std::istringstream iss(line);
    std::string cmd;
    iss >> cmd;

    if (cmd == "SUBMIT") {
        std::string trace_dir;
        std::getline(iss, trace_dir);
        trace_dir = trim(trace_dir);

        if (trace_dir.empty())
            return "{\"ok\":false,\"error\":\"missing trace_dir\"}";
        if (!fs::exists(trace_dir))
            return "{\"ok\":false,\"error\":\"trace_dir not found\"}";

        // be a bit tolerant: if user gives parent dir, but kernelslist.g is in ./traces
        fs::path td(trace_dir);
        fs::path kl = td / "kernelslist.g";
        if (!fs::exists(kl)) {
            fs::path td2 = td / "traces";
            if (fs::exists(td2 / "kernelslist.g")) {
                trace_dir = td2.string();
            } else {
                return "{\"ok\":false,\"error\":\"kernelslist.g missing\"}";
            }
        }

        static std::atomic<uint64_t> counter{0};
        std::ostringstream id;
        id << now_compact() << "-" << counter.fetch_add(1);
        st.queue.push(Job{id.str(), trace_dir});

        std::ostringstream resp;
        resp << "{\"ok\":true,\"job_id\":\"" << id.str() << "\"}";
        return resp.str();
    }

    if (cmd == "STATUS") {
        std::ostringstream resp;
        resp << "{\"ok\":true"
             << ",\"queue_len\":" << st.queue.size()
             << ",\"running\":" << (st.running ? "true" : "false")
             << ",\"current\":\"" << st.current_job << "\"}";
        return resp.str();
    }

    if (cmd == "DRAIN_AND_EXIT") {
        st.draining = true;
        return "{\"ok\":true}";
    }

    return "{\"ok\":false,\"error\":\"unknown cmd\"}";
}

int main(int argc, char **argv) {
    std::string cfg_arg;
    std::string sock_path = "/tmp/accelsimd.sock";
    std::string out_root = "results";

    // -------- parse args --------
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto grab = [&](const char *flag, std::string &dst) {
            if (a == flag && i + 1 < argc) {
                dst = argv[++i];
                return true;
            }
            return false;
        };
        if (grab("--config", cfg_arg)) continue;
        if (grab("--socket", sock_path)) continue;
        if (grab("--out", out_root)) continue;
        if (a == "-h" || a == "--help") {
            std::cerr << "Usage: accelsim-daemon "
                         "--config <gpgpusim.config|config-dir> "
                         "[--socket /tmp/accelsim.sock] "
                         "[--out results]\n";
            return 0;
        }
    }

    if (cfg_arg.empty()) {
        std::cerr << "[daemon] ERROR: --config is required\n";
        return 1;
    }

    std::signal(SIGINT, on_sigint);
    std::signal(SIGTERM, on_sigint);

    // -------- normalize config path --------
    fs::path cfgp(cfg_arg);
    fs::path cfgdir;
    fs::path cfgfile;

    if (fs::is_directory(cfgp)) {
        cfgdir = cfgp;
        cfgfile = cfgp / "gpgpusim.config";
    } else {
        cfgdir = cfgp.parent_path();
        cfgfile = cfgp;
    }

    std::cerr << "[daemon] cfgdir  = " << cfgdir << "\n";
    std::cerr << "[daemon] cfgfile = " << cfgfile << "\n";

    if (!fs::exists(cfgfile)) {
        std::cerr << "[daemon] ERROR: config file not found: " << cfgfile << "\n";
        return 2;
    }

    // -------- create output root --------
    try {
        fs::create_directories(out_root);
    } catch (const std::exception &e) {
        std::cerr << "[daemon] ERROR: cannot create out dir '" << out_root
                  << "': " << e.what() << "\n";
        return 2;
    }

    // -------- chdir so that gpgpu-sim relative files work --------
    if (!cfgdir.empty()) {
        if (chdir(cfgdir.c_str()) != 0) {
            std::perror("[daemon] chdir to config dir failed");
            return 2;
        }
    }
    std::cerr << "[daemon] cwd     = " << fs::current_path() << "\n";

    // -------- initialize engine --------
    std::cerr << "[daemon] initializing execution engine...\n";
    ExecutionEngine engine;
    if (!engine.initialize(cfgfile.string())) {
        std::cerr << "[daemon] ERROR: engine.initialize() failed for "
                  << cfgfile << "\n";
        return 3;
    }
    std::cerr << "[daemon] execution engine initialized\n";

    // -------- init IPC --------
    DaemonState st;
    IpcServer server(sock_path);
    std::cerr << "[daemon] starting IPC server on " << sock_path << " ...\n";
    if (!server.start([&](const std::string &line) { return handle_line(st, line); })) {
        std::cerr << "[daemon] ERROR: failed to listen on socket: " << sock_path << "\n";
        return 4;
    }
    std::cerr << "[daemon] listening on " << sock_path << "\n";

    // -------- main loop --------
    while (!g_stop) {
        if (st.draining && st.queue.size() == 0 && !st.running) break;

        Job j = st.queue.pop_blocking();  // blocks until a SUBMIT arrives
        st.running = true;
        st.current_job = j.id;

        fs::path job_out = fs::path(out_root) / j.id;
        fs::create_directories(job_out);
        engine.set_output_root(job_out.string());

        std::cerr << "[daemon] running job id=" << j.id
                  << " trace=" << j.trace_dir << "\n";

        bool ok = engine.reset()
               && engine.bind(GPUContext{j.id, j.trace_dir})
               && engine.run_to_completion();

        engine.unbind();
        std::cerr << "[daemon] job " << j.id << (ok ? " OK" : " FAIL") << "\n";

        st.running = false;
        st.current_job.clear();
    }

    std::cerr << "[daemon] shutting down...\n";
    server.stop();
    engine.shutdown();
    std::cerr << "[daemon] bye\n";
    return 0;
}
