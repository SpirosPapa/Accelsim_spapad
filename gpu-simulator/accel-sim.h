#pragma once

#include <math.h>
#include <stdio.h>
#include <time.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../ISA_Def/trace_opcode.h"
#include "../trace-parser/trace_parser.h"
#include "abstract_hardware_model.h"
#include "cuda-sim/cuda-sim.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu-sim/icnt_wrapper.h"
#include "gpgpu_context.h"
#include "gpgpusim_entrypoint.h"
#include "option_parser.h"
#include "trace_driven.h"

class accel_sim_framework {
 public:
  accel_sim_framework();  // empty constructor (daemon mode etc.)
  accel_sim_framework(int argc, const char **argv);
  accel_sim_framework(std::string config_file, std::string trace_file);

  // ----- legacy single-job API (CLI-style) ----------------------------------
  void init() {
    active = false;
    sim_cycles = false;
    window_size = 0;
    commandlist_index = 0;

    assert(m_gpgpu_context);
    assert(m_gpgpu_sim);

    concurrent_kernel_sm =
        m_gpgpu_sim->getShaderCoreConfig()->gpgpu_concurrent_kernel_sm;
    window_size = concurrent_kernel_sm
                      ? m_gpgpu_sim->get_config().get_max_concurrent_kernel()
                      : 1;
    assert(window_size > 0);
    commandlist = tracer.parse_commandlist_file();

    kernels_info.reserve(window_size);
  }

  // Build GPU once (from -config); do NOT bind a trace here (daemon mode).
  void build_gpu_once(int argc, const char **argv);

  void simulation_loop();
  void parse_commandlist();
  void cleanup(unsigned finished_kernel);
  unsigned simulate();
  trace_kernel_info_t *create_kernel_info(kernel_trace_t *kernel_trace_info,
                                          gpgpu_context *m_gpgpu_context,
                                          trace_config *config,
                                          trace_parser *parser);
  gpgpu_sim *gpgpu_trace_sim_init_perf_model(int argc, const char *argv[],
                                             gpgpu_context *m_gpgpu_context,
                                             trace_config *m_config);

  // Legacy per-job operations used in older daemon prototype
  void load_trace(const std::string &trace_path);  // points tracer to a job
  void run_one_job();                               // blocks until job done
  void soft_reset_for_next_job();                   // clears runtime state only
  unsigned get_num_sms() const;
  void configure_sm_mask_for_next_job(
      bool use_all_sms, const std::vector<unsigned> &sm_ids);

  // ----- NEW multi-job daemon API -------------------------------------------
  //   - each SUBMIT becomes one job here
  //   - main loop repeatedly calls step_one_cycle()
  //   - finished jobs are reported via collect_finished_jobs()
  void start_job(const std::string &trace_dir,
                 const std::string &out_dir,
                 bool use_all_sms,
                 const std::vector<unsigned> &sm_ids,
                 const std::string &job_id);

  // Is there any remaining work (jobs not done OR GPU still active)?
  bool has_active_work() const;

  // Advance simulation by:
  //   1) feeding new kernels for all active jobs (respecting per-job window)
  //   2) doing exactly one gpgpu_sim cycle()
  //   3) cleaning up at most one finished kernel
  void step_one_cycle();

  // Return and clear the list of job_ids that finished since last call
  std::vector<std::string> collect_finished_jobs();

  struct KernelStatSnapshot {
    unsigned long long start_cycle_total     = 0;
    unsigned long long start_insn_total      = 0;
    unsigned long long start_tot_issued_cta  = 0;
  };

 private:
  // Per-job runtime state for the new concurrent daemon
  struct JobRuntime {
    std::string job_id;
    std::string trace_dir;
    std::string out_dir;

    bool use_all_sms = true;
    std::vector<unsigned> sm_ids;

    std::unique_ptr<trace_parser> parser;
    std::vector<trace_command> commandlist;
    unsigned commandlist_index = 0;

    bool concurrent_kernel_sm = false;
    unsigned window_size = 1;

    std::vector<trace_kernel_info_t *> kernels_info;
    std::vector<unsigned long long> busy_streams;

    // NEW: local_stream_id -> global_stream_id
    std::unordered_map<unsigned long long, unsigned long long> stream_remap;

    bool done = false;

    JobRuntime() = default;
  };

  unsigned long long remap_stream_id(JobRuntime &job,
                                     unsigned long long local_sid);

  // Global counter for assigning new stream ids
  unsigned long long next_global_stream_id_ = 1;

  
  void init_job_state_();  // legacy (single-job) helper

  // NEW helpers for multi-job scheduling
  void parse_and_launch_for_job(size_t job_index);
  void cleanup_finished_kernel(unsigned finished_kernel_uid);

  // Global GPU / trace state (shared by all jobs)
  gpgpu_context *m_gpgpu_context;
  trace_config tconfig;
  trace_parser tracer;  // legacy single-job parser
  gpgpu_sim *m_gpgpu_sim;

  // Legacy single-job fields (used by simulation_loop & friends)
  bool concurrent_kernel_sm;
  bool active;
  bool sim_cycles;
  unsigned window_size;
  unsigned commandlist_index;
  std::vector<unsigned long long> busy_streams;
  std::vector<trace_kernel_info_t *> kernels_info;
  std::vector<trace_command> commandlist;

  // Per-job SM restriction for legacy path
  bool job_use_all_sms_ = true;
  std::vector<unsigned> job_sm_ids_;

  // NEW multi-job state
  std::vector<JobRuntime> jobs_;                    // all jobs known to fw
  std::vector<std::string> finished_job_ids_;       // jobs finished since last
  std::unordered_map<unsigned, size_t> kernel_uid_to_job_;  // kernel uid -> job idx

  bool sim_cycles_any_ = false;   // did we advance at least one cycle this step?
  bool trace_config_parsed_ = false;
  bool stopped_due_to_limit_ = false;
  std::unordered_map<unsigned , KernelStatSnapshot> kernel_start_stats_;
};
