#include "accel-sim.h"
#include "./daemon/job_queue.h"
#include "./daemon/ipc_server.h"

#include <thread>
#include <atomic>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <ctime>
#include <filesystem>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

namespace fs = std::filesystem;

struct DaemonStatus {
  std::atomic<bool> has_job{false};
  std::mutex mx;
  std::string current_id;
  std::string current_trace;
  std::string current_out;
};

// --- helpers ---------------------------------------------------------------

static bool ensure_dir(const std::string& path) {
  std::error_code ec;
  fs::create_directories(path, ec);
  return !ec;
}

struct FdRedirect {
  int old_out{-1}, old_err{-1};
  int new_out{-1}, new_err{-1};

  bool start(const std::string& out_dir) {
    // open log files
    std::string out_log = out_dir + "/stdout.log";
    std::string err_log = out_dir + "/stderr.log";

    new_out = ::open(out_log.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (new_out < 0) return false;
    new_err = ::open(err_log.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (new_err < 0) { ::close(new_out); new_out = -1; return false; }

    // duplicate originals
    old_out = ::dup(STDOUT_FILENO);
    old_err = ::dup(STDERR_FILENO);
    if (old_out < 0 || old_err < 0) return false;

    // redirect
    if (::dup2(new_out, STDOUT_FILENO) < 0) return false;
    if (::dup2(new_err, STDERR_FILENO) < 0) return false;
    return true;
  }

  void stop() {
    // flush
    fflush(stdout);
    fflush(stderr);
    // restore
    if (old_out >= 0) { ::dup2(old_out, STDOUT_FILENO); ::close(old_out); old_out = -1; }
    if (old_err >= 0) { ::dup2(old_err, STDERR_FILENO); ::close(old_err); old_err = -1; }
    // close new
    if (new_out >= 0) { ::close(new_out); new_out = -1; }
    if (new_err >= 0) { ::close(new_err); new_err = -1; }
  }
};

static std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  size_t b = s.find_last_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  return s.substr(a, b - a + 1);
}

static std::vector<std::string> split_ws(const std::string& s) {
  std::istringstream is(s);
  std::vector<std::string> v;
  std::string tok;
  while (is >> tok) v.push_back(tok);
  return v;
}

// Parse: "SUBMIT <trace_path> [out=/path]"
static std::string handle_line(JobQueue& q, DaemonStatus& st, const std::string& line) {
  if (line.rfind("SUBMIT ", 0) == 0) {
    auto payload = trim(line.substr(7));
    auto toks = split_ws(payload);
    if (toks.empty()) return "{\"ok\":false,\"error\":\"missing trace path\"}";

    Job j;
    j.trace_dir = toks[0];
    // id based on content + time
    j.id = std::to_string(std::hash<std::string>{}(j.trace_dir + std::to_string(std::time(nullptr))));

    // optional out=...
    j.out_dir.clear();
    for (size_t i = 1; i < toks.size(); ++i) {
      const auto& t = toks[i];
      if (t.rfind("out=", 0) == 0) {
        j.out_dir = t.substr(4);
      }
    }
    if (j.out_dir.empty()) {
      j.out_dir = "/tmp/accelsim_job_" + j.id;
    }

    q.push(std::move(j));
    return "{\"ok\":true,\"accepted\":true}";
  }

  if (line == "PING") return "{\"ok\":true,\"pong\":true}";
  if (line == "SHUTDOWN") { q.shutdown(); return "{\"ok\":true,\"shutdown\":true}"; }

  if (line == "STATUS") {
    bool running = st.has_job.load();
    std::string id, trace, out;
    {
      std::lock_guard<std::mutex> lk(st.mx);
      id = st.current_id; trace = st.current_trace; out = st.current_out;
    }
    std::ostringstream os;
    os << "{\"ok\":true"
       << ",\"running\":" << (running ? "true" : "false")
       << ",\"current\":{"
         << "\"id\":\"" << id << "\","
         << "\"trace\":\"" << trace << "\","
         << "\"out\":\"" << out << "\"}"
       << ",\"queued\":" << q.size()
       << "}";
    return os.str();
  }

  if (line == "QUEUE") {
    auto v = q.snapshot(20);
    std::ostringstream os;
    os << "{\"ok\":true,\"queue\":[";
    for (size_t i = 0; i < v.size(); ++i) {
      if (i) os << ",";
      os << "{\"id\":\"" << v[i].id
         << "\",\"trace\":\"" << v[i].trace_dir
         << "\",\"out\":\"" << v[i].out_dir << "\"}";
    }
    os << "],\"queued\":" << v.size() << "}";
    return os.str();
  }

  return "{\"ok\":false,\"error\":\"unknown command\"}";
}

int main(int argc, const char** argv) {
  // Build the GPU once from -config
  accel_sim_framework fw;
  fw.build_gpu_once(argc, argv);

  JobQueue jq;
  DaemonStatus st;

  IpcServer srv("/tmp/accelsim.sock");
  if (!srv.start([&](const std::string& s){ return handle_line(jq, st, s); })) {
    std::cerr << "Failed to start IPC server\n";
    return 1;
  }

  std::thread worker([&](){
    Job j;
    while (jq.pop_blocking(j)) {
      // Record status
      {
        std::lock_guard<std::mutex> lk(st.mx);
        st.current_id = j.id;
        st.current_trace = j.trace_dir;
        st.current_out = j.out_dir;
      }
      st.has_job.store(true);

      // Ensure out dir exists and redirect logs
      ensure_dir(j.out_dir);
      FdRedirect redir;
      bool redir_ok = redir.start(j.out_dir);
      if (!redir_ok) {
        std::cerr << "WARN: failed to redirect logs to " << j.out_dir << "\n";
      }

      // Run the job
      fw.soft_reset_for_next_job();
      fw.load_trace(j.trace_dir);
      fw.run_one_job();

      // Restore logs
      redir.stop();

      // Clear status
      st.has_job.store(false);
      {
        std::lock_guard<std::mutex> lk(st.mx);
        st.current_id.clear();
        st.current_trace.clear();
        st.current_out.clear();
      }
    }
  });

  worker.join();
  srv.stop();
  return 0;
}
