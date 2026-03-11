#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

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
  uint32_t next_job_uid_ = 1;
  new_addr_type next_job_global_offset_ = (1ULL << 40);   // start at 1 TiB
  static constexpr new_addr_type kJobAddrStride = (1ULL << 40); // 1 TiB/job
 public:
  accel_sim_framework();
  accel_sim_framework(int argc, const char **argv);
  accel_sim_framework(std::string config_file, std::string trace_file);

  // ----- legacy API ----------------------------------------------------------
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

  // Build GPU once (from -config).
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

  // Legacy per-job operations (older daemon prototype)
  void load_trace(const std::string &trace_path);
  void run_one_job();
  void soft_reset_for_next_job();

  unsigned get_num_sms() const;
  void configure_sm_mask_for_next_job(bool use_all_sms,
                                      const std::vector<unsigned> &sm_ids);
  // MIG SLICE CREATION 
  bool mig_enabled() const { return mig_enabled_; }
  size_t mig_num_slices() const { return mig_slice_sms_.size(); }

  const std::vector<unsigned> &mig_slice_sms(size_t slice_id) const {
    assert(slice_id < mig_slice_sms_.size());
    return mig_slice_sms_[slice_id];
  }

  const std::string &mig_slice_profile(size_t slice_id) const {
    assert(slice_id < mig_slice_profiles_.size());
    return mig_slice_profiles_[slice_id];
  } 
  // ----- NEW multi-job daemon API -------------------------------------------
  void start_job(const std::string &trace_dir,
                const std::string &out_dir,
                bool use_all_sms,
                const std::vector<unsigned> &sm_ids,
                int slice_id,
                const std::string &job_id);

  bool has_active_work() const;

  // Advance simulation by:
  //   1) feed new kernels for all active jobs (respect per-job window)
  //   2) do exactly one gpgpu_sim cycle()
  //   3) cleanup at most one finished kernel
  void step_one_cycle();

  std::vector<std::string> collect_finished_jobs();

 private:
  // Per-job runtime state for daemon
  struct JobRuntime {
    uint32_t job_uid = 0;     // numeric id used inside sim
    std::string job_id;
    std::string trace_dir;
    std::string out_dir;

    bool use_all_sms = true;
    std::vector<unsigned> sm_ids;
    
    int slice_id = -1;   // NEW
    new_addr_type global_mem_offset = 0;  // unique global address-space base for this job

    std::unique_ptr<trace_parser> parser;
    std::vector<trace_command> commandlist;
    unsigned commandlist_index = 0;

    bool concurrent_kernel_sm = false;
    unsigned window_size = 1;

    std::vector<trace_kernel_info_t *> kernels_info;
    std::vector<unsigned long long> busy_streams;

    // Local stream id -> global stream id remap (global stream ids must be unique
    // across jobs so gpgpu_sim can attribute events correctly).
    std::unordered_map<unsigned long long, unsigned long long> stream_remap;

    bool done = false;

    JobRuntime() = default;
    JobRuntime(const JobRuntime &) = delete;
    JobRuntime &operator=(const JobRuntime &) = delete;
    JobRuntime(JobRuntime &&) = default;
    JobRuntime &operator=(JobRuntime &&) = default;
  };

  // Assign a global stream id for a (job, local_stream_id) pair.
  unsigned long long remap_stream_id(JobRuntime &job,
                                     unsigned long long local_sid);

  // Legacy helper (load_trace path)
  void init_job_state_();

  // Multi-job scheduling helpers
  void parse_and_launch_for_job(size_t job_index);
  void cleanup_finished_kernel(unsigned finished_kernel_uid);

 private:
  // -------------------------------------------------------------------------
  // Global GPU / trace state (shared by all jobs)
  // -------------------------------------------------------------------------
  gpgpu_context *m_gpgpu_context = nullptr;
  trace_config tconfig;
  trace_parser tracer;      // legacy single-job parser
  gpgpu_sim *m_gpgpu_sim = nullptr;

  // -------------------------------------------------------------------------
  // Legacy fields for single-job simulation_loop()
  // -------------------------------------------------------------------------
  bool concurrent_kernel_sm = false;
  bool active = false;
  bool sim_cycles = false;
  unsigned window_size = 0;
  unsigned commandlist_index = 0;

  std::vector<unsigned long long> busy_streams;
  std::vector<trace_kernel_info_t *> kernels_info;
  std::vector<trace_command> commandlist;

  // Legacy per-job SM mask (applied in create_kernel_info on legacy path)
  bool job_use_all_sms_ = true;
  std::vector<unsigned> job_sm_ids_;
  // MIG startup configuration (fixed for the lifetime of the daemon)
  bool mig_enabled_ = false;
  std::vector<std::vector<unsigned>> mig_slice_sms_;
  std::vector<std::string> mig_slice_profiles_;
  // -------------------------------------------------------------------------
  // Multi-job daemon state
  // -------------------------------------------------------------------------
  std::vector<JobRuntime> jobs_;
  std::vector<std::string> finished_job_ids_;

  // kernel launch UID -> job index (so finished kernels are cleaned up correctly)
  std::unordered_map<unsigned, size_t> kernel_uid_to_job_;

  // monotonically increasing global stream ids
  unsigned long long next_global_stream_id_ = 1;

  // bookkeeping flags
  bool sim_cycles_any_ = false;
  bool trace_config_parsed_ = false;
  bool stopped_due_to_limit_ = false;
};
