#include "accel-sim.h"
#include "./daemon/ipc_server.h"

#include <atomic>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

// ------------------- Job + daemon state -------------------------------------

struct Job {
  std::string id;
  std::string trace_dir;
  std::string out_dir;
  bool use_all_sms = true;            // true => job can use all SMs
  std::vector<unsigned> sm_ids;       // empty + use_all_sms=true => all SMs
};

enum class JobState { Pending, Running, Finished };

struct JobRecord {
  Job job;
  JobState state = JobState::Pending;
};

struct DaemonState {
  std::mutex mx;
  std::vector<JobRecord> jobs;        // all jobs, in arrival order
  std::atomic<bool> shutdown_requested{false};
};

// ------------------- helpers ------------------------------------------------

static bool ensure_dir(const std::string &path) {
  if (path.empty()) return true;
  std::error_code ec;
  fs::create_directories(path, ec);
  return !ec;
}

static std::string trim(const std::string &s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

static std::vector<std::string> split_ws(const std::string &s) {
  std::istringstream is(s);
  std::vector<std::string> v;
  std::string tok;
  while (is >> tok) v.push_back(tok);
  return v;
}

// Parse "sms=0,1,2" or "sms=-1" into out_ids.
// Returns true on success. On failure, fills err_json with a JSON error reply.
static bool parse_sm_list(const std::string &spec,
                          unsigned total_sms,
                          std::vector<unsigned> &out_ids,
                          std::string &err_json) {
  out_ids.clear();

  // "-1" or empty => use all SMs
  if (spec.empty() || spec == "-1") {
    return true;
  }

  std::istringstream ss(spec);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item = trim(item);
    if (item.empty()) continue;

    int v = 0;
    try {
      size_t pos = 0;
      v = std::stoi(item, &pos);
      if (pos != item.size()) throw std::invalid_argument("junk");
    } catch (...) {
      std::ostringstream os;
      os << "{\"ok\":false,\"error\":\"invalid SM id '" << item << "'\"}";
      err_json = os.str();
      out_ids.clear();
      return false;
    }

    if (v == -1) {
      err_json =
          "{\"ok\":false,\"error\":\"-1 (all SMs) cannot be mixed with explicit SM ids\"}";
      out_ids.clear();
      return false;
    }

    if (v < 0 || static_cast<unsigned>(v) >= total_sms) {
      std::ostringstream os;
      os << "{\"ok\":false,\"error\":\"THE GPU HAS " << total_sms
         << " SM WITH ID'S 0-" << (total_sms ? (total_sms - 1) : 0)
         << " PLEASE GIVE ID'S THAT MATCH WITH THESE OR -1 TO RUN ON ALL SM\"}";
      err_json = os.str();
      out_ids.clear();
      return false;
    }

    out_ids.push_back(static_cast<unsigned>(v));
  }

  return true;
}

// Helper to count jobs by state (for STATUS)
static void count_jobs(const DaemonState &st,
                       size_t &pending,
                       size_t &running,
                       size_t &finished) {
  pending = running = finished = 0;
  for (const auto &jr : st.jobs) {
    switch (jr.state) {
      case JobState::Pending:  ++pending;  break;
      case JobState::Running:  ++running;  break;
      case JobState::Finished: ++finished; break;
    }
  }
}

// ------------------- IPC command handler ------------------------------------

// Parse: "SUBMIT <trace_path> [out=/path] [sms=...]"
static std::string handle_line(DaemonState &st,
                               unsigned total_sms,
                               const std::string &line) {
  const std::string cmd = trim(line);

  // ---- SUBMIT --------------------------------------------------------------
  if (cmd.rfind("SUBMIT ", 0) == 0) {
    auto payload = trim(cmd.substr(7));
    auto toks = split_ws(payload);
    if (toks.empty())
      return "{\"ok\":false,\"error\":\"missing trace path\"}";

    Job job;
    job.trace_dir = toks[0];

    // id based on content + time
    job.id = std::to_string(
        std::hash<std::string>{}(job.trace_dir + std::to_string(std::time(nullptr))));

    job.out_dir.clear();
    job.sm_ids.clear();
    job.use_all_sms = true;

    // optional args: out=..., sms=...
    for (size_t i = 1; i < toks.size(); ++i) {
      const auto &t = toks[i];
      if (t.rfind("out=", 0) == 0) {
        job.out_dir = t.substr(4);
      } else if (t.rfind("sms=", 0) == 0) {
        std::string spec = t.substr(4);
        std::string err_json;
        std::vector<unsigned> sm_ids;
        if (!parse_sm_list(spec, total_sms, sm_ids, err_json)) {
          // invalid SM list => reject submission
          return err_json;
        }
        if (!sm_ids.empty()) {
          job.use_all_sms = false;
          job.sm_ids = std::move(sm_ids);
        } else {
          // "-1" / empty => all SMs
          job.use_all_sms = true;
          job.sm_ids.clear();
        }
      }
    }

    if (job.out_dir.empty()) {
      job.out_dir = "/tmp/accelsim_job_" + job.id;
    }

    {
      std::lock_guard<std::mutex> lk(st.mx);
      JobRecord jr;
      jr.job = std::move(job);
      jr.state = JobState::Pending;
      st.jobs.push_back(std::move(jr));
      std::cout << "[daemon] queued job id=" << st.jobs.back().job.id
                << " trace=" << st.jobs.back().job.trace_dir << "\n";
    }

    return "{\"ok\":true,\"accepted\":true}";
  }

  // ---- PING ----------------------------------------------------------------
  if (cmd == "PING") {
    return "{\"ok\":true,\"pong\":true}";
  }

  // ---- SHUTDOWN ------------------------------------------------------------
  if (cmd == "SHUTDOWN") {
    st.shutdown_requested.store(true);
    return "{\"ok\":true,\"shutdown\":true}";
  }

  // ---- STATUS --------------------------------------------------------------
  if (cmd == "STATUS") {
    DaemonState snapshot;
    {
      std::lock_guard<std::mutex> lk(st.mx);
      snapshot.jobs = st.jobs;
    }
    size_t pending = 0, running = 0, finished = 0;
    count_jobs(snapshot, pending, running, finished);

    std::ostringstream os;
    os << "{\"ok\":true"
       << ",\"pending\":"  << pending
       << ",\"running\":"  << running
       << ",\"finished\":" << finished
       << "}";
    return os.str();
  }

  // ---- QUEUE: list pending jobs -------------------------------------------
  if (cmd == "QUEUE") {
    std::vector<JobRecord> snapshot;
    {
      std::lock_guard<std::mutex> lk(st.mx);
      snapshot = st.jobs;
    }

    std::ostringstream os;
    os << "{\"ok\":true,\"queue\":[";
    bool first = true;
    for (const auto &jr : snapshot) {
      if (jr.state != JobState::Pending) continue;
      if (!first) os << ",";
      first = false;
      os << "{\"id\":\""   << jr.job.id
         << "\",\"trace\":\"" << jr.job.trace_dir
         << "\",\"out\":\""   << jr.job.out_dir << "\"}";
    }
    os << "]}";
    return os.str();
  }

  return "{\"ok\":false,\"error\":\"unknown command\"}";
}

// ------------------- Job scheduling with SM reservation ---------------------

// Build per-job SM mask (vector<bool> of length total_sms)
static std::vector<bool> build_sm_mask(const Job &job, unsigned total_sms) {
  std::vector<bool> mask(total_sms, false);
  if (job.use_all_sms || job.sm_ids.empty()) {
    std::fill(mask.begin(), mask.end(), true);
  } else {
    for (unsigned sm_id : job.sm_ids) {
      if (sm_id < total_sms) mask[sm_id] = true;
    }
  }
  return mask;
}

// Reservation policy: explained in comments in previous messages.
static std::vector<size_t> pick_jobs_to_start(const std::vector<JobRecord> &jobs,
                                              unsigned total_sms) {
  std::vector<size_t> to_start;
  if (total_sms == 0) return to_start;

  std::vector<bool> used_or_reserved(total_sms, false);

  for (size_t idx = 0; idx < jobs.size(); ++idx) {
    const JobRecord &jr = jobs[idx];
    if (jr.state == JobState::Finished) continue;

    std::vector<bool> mask = build_sm_mask(jr.job, total_sms);

    bool overlap = false;
    for (unsigned sm = 0; sm < total_sms; ++sm) {
      if (mask[sm] && used_or_reserved[sm]) {
        overlap = true;
        break;
      }
    }

    if (jr.state == JobState::Running) {
      for (unsigned sm = 0; sm < total_sms; ++sm) {
        if (mask[sm]) used_or_reserved[sm] = true;
      }
    } else if (jr.state == JobState::Pending) {
      if (overlap) {
        for (unsigned sm = 0; sm < total_sms; ++sm) {
          if (mask[sm]) used_or_reserved[sm] = true;
        }
      } else {
        to_start.push_back(idx);
        for (unsigned sm = 0; sm < total_sms; ++sm) {
          if (mask[sm]) used_or_reserved[sm] = true;
        }
      }
    }
  }

  return to_start;
}

// ------------------- main: daemon + simulation loop -------------------------

int main(int argc, const char **argv) {
  // Build the GPU once from -config (no trace yet).
  accel_sim_framework fw;
  fw.build_gpu_once(argc, argv);

  const unsigned total_sms = fw.get_num_sms();

  DaemonState state;

  IpcServer srv("/tmp/accelsim.sock");
  if (!srv.start([&](const std::string &s) {
        return handle_line(state, total_sms, s);
      })) {
    std::cerr << "Failed to start IPC server\n";
    return 1;
  }

  std::cout << "Accel-Sim daemon listening on /tmp/accelsim.sock\n";

  while (true) {
    // --- 1. Snapshot jobs under lock ----------------------------------------
    std::vector<JobRecord> snapshot;
    {
      std::lock_guard<std::mutex> lk(state.mx);
      snapshot = state.jobs;
    }

    // --- 2. Decide which pending jobs to start according to reservation policy
    std::vector<size_t> to_start = pick_jobs_to_start(snapshot, total_sms);

    // --- 3. Start those jobs in the real state + forward to fw --------------
    {
      std::lock_guard<std::mutex> lk(state.mx);
      for (size_t idx : to_start) {
        if (idx >= state.jobs.size()) continue;
        JobRecord &jr = state.jobs[idx];
        if (jr.state != JobState::Pending) continue;

        ensure_dir(jr.job.out_dir);

        std::cout << "[daemon] starting job id=" << jr.job.id
                  << " trace=" << jr.job.trace_dir << "\n";

        fw.start_job(jr.job.trace_dir,
                     jr.job.out_dir,
                     jr.job.use_all_sms,
                     jr.job.sm_ids,
                     jr.job.id);

        jr.state = JobState::Running;
      }
    }

    // --- 4. Always advance simulation by one cycle --------------------------
    fw.step_one_cycle();

    // --- 5. See which jobs finished in this step ----------------------------
    {
      std::vector<std::string> finished_ids = fw.collect_finished_jobs();
      if (!finished_ids.empty()) {
        std::lock_guard<std::mutex> lk(state.mx);
        for (const auto &fid : finished_ids) {
          for (auto &jr : state.jobs) {
            if (jr.job.id == fid && jr.state == JobState::Running) {
              jr.state = JobState::Finished;
              std::cout << "[daemon] job finished id=" << jr.job.id << "\n";
            }
          }
        }
      }
    }

    // --- 6. Exit / idle logic -----------------------------------------------
    bool shutdown = state.shutdown_requested.load();
    size_t pending = 0, running = 0, finished = 0;
    {
      std::lock_guard<std::mutex> lk(state.mx);
      count_jobs(state, pending, running, finished);
    }
    bool fw_active = fw.has_active_work();

    if (shutdown && pending == 0 && running == 0 && !fw_active) {
      break;
    }

    if (!fw_active && pending == 0 && running == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }

  srv.stop();
  std::cout << "Daemon shutdown cleanly.\n";
  return 0;
}
