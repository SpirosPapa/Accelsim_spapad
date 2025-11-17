#include "accel-sim.h"
#include "./daemon/job_queue.h"
#include "./daemon/ipc_server.h"
#include <thread>
#include <atomic>
#include <iostream>

static std::string handle_line(JobQueue& q, const std::string& line) {
  // Simple line protocol:
  //   SUBMIT <path-to-trace-dir-or-commandlist>
  //   PING
  //   SHUTDOWN
  if (line.rfind("SUBMIT ", 0) == 0) {
    Job j;
    j.id = std::to_string(std::hash<std::string>{}(line));
    j.trace_dir = line.substr(7);
    q.push(std::move(j));
    return "{\"ok\":true,\"accepted\":true}";
  }
  if (line == "PING") return "{\"ok\":true,\"pong\":true}";
  if (line == "SHUTDOWN") { q.shutdown(); return "{\"ok\":true,\"shutdown\":true}"; }
  return "{\"ok\":false,\"error\":\"unknown command\"}";
}

int main(int argc, const char** argv) {
  // Start the sim once (argv must include -config /path/to/gpgpu_sim.config)
  accel_sim_framework fw;
  fw.build_gpu_once(argc, argv);

  JobQueue jq;
  IpcServer srv("/tmp/accelsim.sock");
  if (!srv.start([&](const std::string& s){ return handle_line(jq, s); })) {
    std::cerr << "Failed to start IPC server\n";
    return 1;
  }

  std::thread worker([&](){
    Job j;
    while (jq.pop_blocking(j)) {
      fw.soft_reset_for_next_job();
      fw.load_trace(j.trace_dir);   // trace.config is read here (per job)
      fw.run_one_job();             // blocking until the trace finishes
      // Optional: write per-job stats to a file tagged by j.id
    }
  });

  worker.join();
  srv.stop();
  return 0;
}
