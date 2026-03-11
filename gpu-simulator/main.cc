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

// Job + daemon state 

struct Job {
  std::string id;
  std::string trace_dir;
  std::string out_dir;
  bool use_all_sms = true;            // true => job can use all SMs
  std::vector<unsigned> sm_ids;       // empty + use_all_sms=true => all SMs
  int slice_id = -1;                  // -1 means "no MIG slice"
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

  bool mig_enabled = false;
  std::vector<std::vector<unsigned>> mig_slice_sms;
  std::vector<std::string> mig_slice_profiles;
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

static bool parse_sm_list(const std::string &spec,
                          unsigned total_sms,
                          std::vector<unsigned> &out_ids,
                          std::string &err_json) {
  out_ids.clear();

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

// Parse: "SUBMIT <trace_path> out=/path sms"
static std::string handle_line(DaemonState &st,
                               unsigned total_sms,
                               const std::string &line) {
  const std::string cmd = trim(line);

  if (cmd.rfind("SUBMIT ", 0) == 0) {
    auto payload = trim(cmd.substr(7));
    auto toks = split_ws(payload);
    if (toks.empty())
      return "{\"ok\":false,\"error\":\"missing trace path\"}";

    Job job;
    job.trace_dir = toks[0];
    job.id = std::to_string(
        std::hash<std::string>{}(job.trace_dir + std::to_string(std::time(nullptr))));

    job.out_dir.clear();
    job.sm_ids.clear();
    job.use_all_sms = true;
    job.slice_id = -1;

    bool saw_sms = false;
    bool saw_slice = false;

    for (size_t i = 1; i < toks.size(); ++i) {
      const auto &t = toks[i];
      if (t.rfind("out=", 0) == 0) {
        job.out_dir = t.substr(4);
      } else if (t.rfind("sms=", 0) == 0) {
        saw_sms = true;
        std::string spec = t.substr(4);
        std::string err_json;
        std::vector<unsigned> sm_ids;
        if (!parse_sm_list(spec, total_sms, sm_ids, err_json)) {
          return err_json;
        }
        if (!sm_ids.empty()) {
          job.use_all_sms = false;
          job.sm_ids = std::move(sm_ids);
        } else {
          job.use_all_sms = true;
          job.sm_ids.clear();
        }
      } else if (t.rfind("slice=", 0) == 0) {
        saw_slice = true;
        try {
          size_t pos = 0;
          int v = std::stoi(t.substr(6), &pos);
          if (pos != t.substr(6).size()) throw std::invalid_argument("junk");
          if (v < 0) throw std::invalid_argument("negative");
          job.slice_id = v;
        } catch (...) {
          return "{\"ok\":false,\"error\":\"invalid slice id\"}";
        }
      }
    }

    if (st.mig_enabled) {
      if (!saw_slice) {
        return "{\"ok\":false,\"error\":\"MIG is enabled; SUBMIT must include slice=<id>\"}";
      }
      if (saw_sms) {
        return "{\"ok\":false,\"error\":\"when MIG is enabled, do not pass sms=; the slice defines the SM set\"}";
      }
      if (job.slice_id < 0 ||
          static_cast<size_t>(job.slice_id) >= st.mig_slice_sms.size()) {
        return "{\"ok\":false,\"error\":\"slice id out of range\"}";
      }

      job.use_all_sms = false;
      job.sm_ids = st.mig_slice_sms[job.slice_id];
    } else {
      if (saw_slice) {
        return "{\"ok\":false,\"error\":\"slice= requires startup with -mig\"}";
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

  if (cmd == "PING") {
    return "{\"ok\":true,\"pong\":true}";
  }

  if (cmd == "SHUTDOWN") {
    st.shutdown_requested.store(true);
    return "{\"ok\":true,\"shutdown\":true}";
  }

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


int main(int argc, const char **argv) {
  accel_sim_framework fw;
  fw.build_gpu_once(argc, argv);

  const unsigned total_sms = fw.get_num_sms();

  DaemonState state;
  state.mig_enabled = fw.mig_enabled();
  for (size_t i = 0; i < fw.mig_num_slices(); ++i) {
    state.mig_slice_sms.push_back(fw.mig_slice_sms(i));
    state.mig_slice_profiles.push_back(fw.mig_slice_profile(i));
  }

  IpcServer srv("/tmp/accelsim.sock");
  if (!srv.start([&](const std::string &s) {
        return handle_line(state, total_sms, s);
      })) {
    std::cerr << "Failed to start IPC server\n";
    return 1;
  }

  std::cout << "Accel-Sim daemon listening on /tmp/accelsim.sock\n";
  // in each iteration we check based on the current snpashot/jobs_waiting wich job to start(if possible)
  // if a job is to be launched create the out_dir and inform terminal
  // After we start the job we execute one gpu cycle
  // in the end we check wether any job is finished
  while (true) {
    std::vector<JobRecord> snapshot;
    {
      std::lock_guard<std::mutex> lk(state.mx);
      snapshot = state.jobs;
    }

    std::vector<size_t> to_start = pick_jobs_to_start(snapshot, total_sms);

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
                     jr.job.slice_id,
                     jr.job.id);

        jr.state = JobState::Running;
      }
    }

    fw.step_one_cycle();

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

    //(shutdown option)
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
