// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, Vijay Kandiah,
// Nikos Hardavellas Mahmoud Khairy, Junrui Pan, Timothy G. Rogers The
// University of British Columbia, Northwestern University, Purdue University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef GPU_SIM_H
#define GPU_SIM_H

#include <stdint.h>
#include <stdio.h>
#include <vector> 
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <unordered_map>
#include <cmath>
#include <algorithm> 
#include <limits>

#include "../abstract_hardware_model.h"
#include "../option_parser.h"
#include "../trace.h"
#include "addrdec.h"
#include "gpu-cache.h"
#include "shader.h"

// constants for statistics printouts
#define GPU_RSTAT_SHD_INFO 0x1
#define GPU_RSTAT_BW_STAT 0x2
#define GPU_RSTAT_WARP_DIS 0x4
#define GPU_RSTAT_DWF_MAP 0x8
#define GPU_RSTAT_L1MISS 0x10
#define GPU_RSTAT_PDOM 0x20
#define GPU_RSTAT_SCHED 0x40
#define GPU_MEMLATSTAT_MC 0x2

// constants for configuring merging of coalesced scatter-gather requests
#define TEX_MSHR_MERGE 0x4
#define CONST_MSHR_MERGE 0x2
#define GLOBAL_MSHR_MERGE 0x1

// clock constants
#define MhZ *1000000

#define CREATELOG 111
#define SAMPLELOG 222
#define DUMPLOG 333

class gpgpu_context;

extern tr1_hash_map<new_addr_type, unsigned> address_random_interleaving;

// SST communication functions
/**
 * @brief Check if SST requests buffer is full
 *
 * @param core_id
 * @return true
 * @return false
 */
extern bool is_SST_buffer_full(unsigned core_id);
__attribute__((weak)) bool is_SST_buffer_full(unsigned core_id) {
  return false;
}

/**
 * @brief Send loads to SST memory backend
 *
 * @param core_id
 * @param address
 * @param size
 * @param mem_req
 */
extern void send_read_request_SST(unsigned core_id, uint64_t address,
                                  size_t size, void *mem_req);
__attribute__((weak)) void send_read_request_SST(unsigned core_id,
                                                 uint64_t address, size_t size,
                                                 void *mem_req) {}
/**
 * @brief Send stores to SST memory backend
 *
 * @param core_id
 * @param address
 * @param size
 * @param mem_req
 */
extern void send_write_request_SST(unsigned core_id, uint64_t address,
                                   size_t size, void *mem_req);
__attribute__((weak)) void send_write_request_SST(unsigned core_id,
                                                  uint64_t address, size_t size,
                                                  void *mem_req) {}

enum dram_ctrl_t { DRAM_FIFO = 0, DRAM_FRFCFS = 1 };

enum hw_perf_t {
  HW_BENCH_NAME = 0,
  HW_KERNEL_NAME,
  HW_L1_RH,
  HW_L1_RM,
  HW_L1_WH,
  HW_L1_WM,
  HW_CC_ACC,
  HW_SHRD_ACC,
  HW_DRAM_RD,
  HW_DRAM_WR,
  HW_L2_RH,
  HW_L2_RM,
  HW_L2_WH,
  HW_L2_WM,
  HW_NOC,
  HW_PIPE_DUTY,
  HW_NUM_SM_IDLE,
  HW_CYCLES,
  HW_VOLTAGE,
  HW_TOTAL_STATS
};

struct power_config {
  power_config() { m_valid = true; }
  void init() {
    // initialize file name if it is not set
    time_t curr_time;
    time(&curr_time);
    char *date = ctime(&curr_time);
    char *s = date;
    while (*s) {
      if (*s == ' ' || *s == '\t' || *s == ':') *s = '-';
      if (*s == '\n' || *s == '\r') *s = 0;
      s++;
    }
    char buf1[1024];
    // snprintf(buf1, 1024, "accelwattch_power_report__%s.log", date);
    snprintf(buf1, 1024, "accelwattch_power_report.log");
    g_power_filename = strdup(buf1);
    char buf2[1024];
    snprintf(buf2, 1024, "gpgpusim_power_trace_report__%s.log.gz", date);
    g_power_trace_filename = strdup(buf2);
    char buf3[1024];
    snprintf(buf3, 1024, "gpgpusim_metric_trace_report__%s.log.gz", date);
    g_metric_trace_filename = strdup(buf3);
    char buf4[1024];
    snprintf(buf4, 1024, "gpgpusim_steady_state_tracking_report__%s.log.gz",
             date);
    g_steady_state_tracking_filename = strdup(buf4);
    // for(int i =0; i< hw_perf_t::HW_TOTAL_STATS; i++){
    //   accelwattch_hybrid_configuration[i] = 0;
    // }

    if (g_steady_power_levels_enabled) {
      sscanf(gpu_steady_state_definition, "%lf:%lf",
             &gpu_steady_power_deviation, &gpu_steady_min_period);
    }

    // NOTE: After changing the nonlinear model to only scaling idle core,
    // NOTE: The min_inc_per_active_sm is not used any more
    // if (g_use_nonlinear_model)
    //   sscanf(gpu_nonlinear_model_config, "%lf:%lf", &gpu_idle_core_power,
    //          &gpu_min_inc_per_active_sm);
  }
  void reg_options(class OptionParser *opp);

  char *g_power_config_name;

  bool m_valid;
  bool g_power_simulation_enabled;
  bool g_power_trace_enabled;
  bool g_steady_power_levels_enabled;
  bool g_power_per_cycle_dump;
  bool g_power_simulator_debug;
  char *g_power_filename;
  char *g_power_trace_filename;
  char *g_metric_trace_filename;
  char *g_steady_state_tracking_filename;
  int g_power_trace_zlevel;
  char *gpu_steady_state_definition;
  double gpu_steady_power_deviation;
  double gpu_steady_min_period;

  char *g_hw_perf_file_name;
  char *g_hw_perf_bench_name;
  int g_power_simulation_mode;
  bool g_dvfs_enabled;
  bool g_aggregate_power_stats;
  bool accelwattch_hybrid_configuration[hw_perf_t::HW_TOTAL_STATS];

  // Nonlinear power model
  bool g_use_nonlinear_model;
  char *gpu_nonlinear_model_config;
  double gpu_idle_core_power;
  double gpu_min_inc_per_active_sm;
};
// -----------------------------------------------------------------------------
// Per-kernel statistics 
// -----------------------------------------------------------------------------
struct cache_cnt_t {
  unsigned long long access = 0;
  unsigned long long miss = 0;
  unsigned long long pending_hit = 0;
  unsigned long long resfail = 0;
};
class traffic_breakdown;

struct kernel_stats_accum_t {
    kernel_stats_accum_t()
      : outgoing_traffic("core_to_mem"),
        incoming_traffic("mem_to_core") {}
  // core execution
  unsigned long long sim_cycle   = 0;
  unsigned long long sim_insn    = 0;
  unsigned long long issued_cta  = 0;

  // stalls
  unsigned long long stall_dramfull = 0;
  unsigned long long stall_icnt2sh  = 0;

  // memory traffic
  unsigned long long l2_reqs    = 0;
  unsigned long long l2_bytes   = 0;
  unsigned long long dram_reqs  = 0;
  unsigned long long dram_bytes = 0;

  cache_cnt_t l1i, l1d, l1c, l1t;

  std::vector<cache_cnt_t> l1i_sm, l1d_sm, l1c_sm, l1t_sm;

  unsigned long long thrd_icount = 0;   //  gpgpu_n_tot_thrd_icount
  unsigned long long warp_icount = 0;   //  gpgpu_n_tot_w_icount

  unsigned long long n_load_insn  = 0;
  unsigned long long n_store_insn = 0;

  unsigned long long n_mem_read_local   = 0;
  unsigned long long n_mem_write_local  = 0;
  unsigned long long n_mem_read_global  = 0;
  unsigned long long n_mem_write_global = 0;
  unsigned long long n_mem_texture      = 0;
  unsigned long long n_mem_const        = 0;

  unsigned long long n_shmem_insn  = 0;
  unsigned long long n_sstarr_insn = 0;
  unsigned long long n_tex_insn    = 0;
  unsigned long long n_const_insn  = 0;
  unsigned long long n_param_insn  = 0;

  std::vector<unsigned long long> shader_cycle_distro; 

  std::vector<unsigned long long> single_issue_nums; 
  std::vector<unsigned long long> dual_issue_nums;   

  unsigned long long gpgpu_n_stall_shd_mem = 0;

  unsigned long long gpgpu_n_shmem_bkconflict = 0;
  unsigned long long gpgpu_n_l1cache_bkconflict = 0;
  unsigned long long gpgpu_n_intrawarp_mshr_merge = 0;
  unsigned long long gpgpu_n_cmem_portconflict = 0;
  unsigned long long gpu_reg_bank_conflict_stalls = 0;

  unsigned long long gpgpu_stall_shd_mem_cmem_resource_stall = 0;
  unsigned long long gpgpu_stall_shd_mem_smem_bk_conf = 0;
  unsigned long long gpgpu_stall_shd_mem_glmem_resource_stall = 0;
  unsigned long long gpgpu_stall_shd_mem_glmem_coal_stall = 0;
  unsigned long long gpgpu_stall_shd_mem_glmem_data_port_stall = 0;

  traffic_breakdown outgoing_traffic;
  traffic_breakdown incoming_traffic;

  unsigned long long memlat_num_mfs = 0;

  unsigned long long memlat_mf_total_lat = 0;
  unsigned long long memlat_tot_icnt2mem_lat = 0;
  unsigned long long memlat_tot_icnt2sh_lat  = 0;

  unsigned memlat_max_mf_lat = 0;
  unsigned memlat_max_icnt2mem_lat = 0;
  unsigned memlat_max_mrq_lat = 0;
  unsigned memlat_max_icnt2sh_lat = 0;
  
  unsigned memlat_mrq_lat_table[32]      = {0};
  unsigned memlat_dq_lat_table[32]       = {0};
  unsigned memlat_mf_lat_table[32]       = {0};
  unsigned memlat_icnt2mem_lat_table[24] = {0};
  unsigned memlat_icnt2sh_lat_table[24]  = {0};
  unsigned memlat_mf_lat_pw_table[32]    = {0};

  unsigned long long memlat_mf_tot_lat_pw = 0;
  unsigned long long memlat_mf_num_lat_pw = 0;

  std::vector<unsigned> memlat_max_conc_access2samerow;   // (n_mem*nbk)
  std::vector<unsigned> memlat_max_servicetime2samerow;   // (n_mem*nbk)
  std::vector<unsigned long long> memlat_row_access;     // total served row accesses per (dram,bank)
  std::vector<unsigned long long> memlat_num_activates;  // #row-episodes participated per (dram,bank)
  std::vector<unsigned long long> memlat_totalbankreads;  // (n_mem*nbk)
  std::vector<unsigned long long> memlat_totalbankwrites;  // (n_mem*nbk)
  std::vector<unsigned long long> totalbankaccesses;
  std::vector<unsigned long long> memlat_mf_total_laten;
  std::vector<unsigned long long> memlat_max_mf_laten;
  //DRAM

  std::vector<unsigned long long> dram_n_cmd;
  std::vector<unsigned long long> dram_n_nop;
  std::vector<unsigned long long> dram_n_act;
  std::vector<unsigned long long> dram_n_pre;
  std::vector<unsigned long long> dram_n_activity;
  std::vector<unsigned long long> dram_n_ref_event;
  std::vector<unsigned long long> dram_n_req;
  std::vector<unsigned long long> dram_n_rd;
  std::vector<unsigned long long> dram_n_rd_L2_A;
  std::vector<unsigned long long> dram_n_wr;
  std::vector<unsigned long long> dram_n_wr_WB;
  std::vector<unsigned long long> dram_bwutil;
  //DRAM per-bank stats (flattened: idx = dram_id * nbk + bank)
  std::vector<unsigned long long> dram_bk_n_access;
  std::vector<unsigned long long> dram_bk_n_idle;

  // ---- DRAM BLP / locality (per dram_id) ----
  std::vector<unsigned long long> dram_banks_1time;               // sum(memory_pending)
  std::vector<unsigned long long> dram_banks_access_total;        // cycles with memory_pending>0

  std::vector<unsigned long long> dram_banks_time_rw;             // sum(memory_pending_rw)
  std::vector<unsigned long long> dram_banks_access_rw_total;      // cycles with memory_pending_rw>0

  std::vector<unsigned long long> dram_banks_time_ready;          // sum(memory_Pending_ready)
  std::vector<unsigned long long> dram_banks_access_ready_total;   // cycles with memory_Pending_ready>0

  std::vector<unsigned long long> dram_w2r_ratio_sum_1e6;          // sum( write/(write+read) ) * 1e6 over rw cycles
  std::vector<unsigned long long> dram_bkgrp_parallsim_rw;         // sum(bankgrp_count) over rw cycles


  std::vector<unsigned long long> dram_access_num;
  std::vector<unsigned long long> dram_hits_num;
  std::vector<unsigned long long> dram_read_num;
  std::vector<unsigned long long> dram_hits_read_num;
  std::vector<unsigned long long> dram_write_num;
  std::vector<unsigned long long> dram_hits_write_num;


  // BW classification (per dram_id): mirrors util_bw / wasted_bw_col / wasted_bw_row / idle_bw
  std::vector<unsigned long long> dram_util_bw;
  std::vector<unsigned long long> dram_wasted_bw_col;
  std::vector<unsigned long long> dram_wasted_bw_row;
  std::vector<unsigned long long> dram_idle_bw;

  std::vector<unsigned long long> dram_RCDc_limit;
  std::vector<unsigned long long> dram_RCDWRc_limit;
  std::vector<unsigned long long> dram_WTRc_limit;
  std::vector<unsigned long long> dram_RTWc_limit;
  std::vector<unsigned long long> dram_CCDLc_limit;
  std::vector<unsigned long long> dram_rwq_limit;
  std::vector<unsigned long long> dram_CCDLc_limit_alone;
  std::vector<unsigned long long> dram_WTRc_limit_alone;
  std::vector<unsigned long long> dram_RTWc_limit_alone;

  std::vector<unsigned long long> dram_issued_total_row;
  std::vector<unsigned long long> dram_issued_total_col;
  std::vector<unsigned long long> dram_issued_total;
  std::vector<unsigned long long> dram_issued_two;
  std::vector<unsigned long long> dram_ave_mrqs_sum;

  // --- DRAM queue / bins (per dram_id) ---
  std::vector<unsigned long long> dram_max_mrqs;     // size = n_mem

  // util/eff bins per dram_id (flattened: dram_id*10 + bin)
  std::vector<unsigned long long> dram_util_bins;    // size = n_mem*10
  std::vector<unsigned long long> dram_eff_bins;     // size = n_mem*10

  // snapshots for interval-binning (per dram_id)
  std::vector<unsigned long long> dram_last_n_cmd;      // size = n_mem
  std::vector<unsigned long long> dram_last_n_activity; // size = n_mem
  std::vector<unsigned long long> dram_last_bwutil;     // size = n_mem


};

struct kernel_stats_view_t {
  // "Unset" sentinels 
  static constexpr long long kUnset  = -1;
  static constexpr double   kUnsetD = -1.0;

  long long gpu_sim_cycle = kUnset;
  long long gpu_sim_insn  = kUnset;
  double    gpu_ipc       = kUnsetD;

  long long gpu_tot_sim_cycle  = kUnset;
  long long gpu_tot_sim_insn   = kUnset;
  double    gpu_tot_ipc        = kUnsetD;
  long long gpu_tot_issued_cta = kUnset;

  long long gpu_stall_dramfull = kUnset;
  long long gpu_stall_icnt2sh  = kUnset;

  double L2_BW       = kUnsetD;
  double L2_BW_total = kUnsetD;

  double gpu_occupancy_percent      = kUnsetD;
  double gpu_tot_occupancy_percent  = kUnsetD;

  double partiton_level_parallism            = kUnsetD;
  double partiton_level_parallism_total      = kUnsetD;
  double partiton_level_parallism_util       = kUnsetD;
  double partiton_level_parallism_util_total = kUnsetD;

  cache_cnt_t l1i{}, l1d{}, l1c{}, l1t{};
  std::vector<cache_cnt_t> l1i_sm, l1d_sm, l1c_sm, l1t_sm;


  long long l1d_port_available_cycles = kUnset;
  long long l1d_data_port_busy_cycles = kUnset;
  long long l1d_fill_port_busy_cycles = kUnset;

  long long ctas_completed_for_kernel = kUnset;
  int scheduler_sampling_core = -1;             
  std::vector<unsigned> warp_slot_issue_distro; 

  long long gpgpu_n_tot_thrd_icount = kUnset;
  long long gpgpu_n_tot_w_icount    = kUnset;

  long long gpgpu_n_load_insn  = kUnset;
  long long gpgpu_n_store_insn = kUnset;

  long long gpgpu_n_mem_read_local   = kUnset;
  long long gpgpu_n_mem_write_local  = kUnset;
  long long gpgpu_n_mem_read_global  = kUnset;
  long long gpgpu_n_mem_write_global = kUnset;
  long long gpgpu_n_mem_texture      = kUnset;
  long long gpgpu_n_mem_const        = kUnset;

  long long gpgpu_n_shmem_insn  = kUnset;
  long long gpgpu_n_sstarr_insn = kUnset;
  long long gpgpu_n_tex_insn    = kUnset;
  long long gpgpu_n_const_mem_insn = kUnset;
  long long gpgpu_n_param_mem_insn = kUnset;

  std::vector<unsigned long long> shader_cycle_distro;
  std::vector<unsigned long long> single_issue_nums;
  std::vector<unsigned long long> dual_issue_nums;


  unsigned long long gpgpu_n_stall_shd_mem = kUnset;

  unsigned long long gpgpu_n_shmem_bkconflict = kUnset;
  unsigned long long gpgpu_n_l1cache_bkconflict = kUnset;
  unsigned long long gpgpu_n_intrawarp_mshr_merge = kUnset;
  unsigned long long gpgpu_n_cmem_portconflict = kUnset;
  unsigned long long gpu_reg_bank_conflict_stalls = kUnset;

  unsigned long long gpgpu_stall_shd_mem_cmem_resource_stall = kUnset;
  unsigned long long gpgpu_stall_shd_mem_smem_bk_conf = kUnset;
  unsigned long long gpgpu_stall_shd_mem_glmem_resource_stall = kUnset;
  unsigned long long gpgpu_stall_shd_mem_glmem_coal_stall = kUnset;
  unsigned long long gpgpu_stall_shd_mem_glmem_data_port_stall = kUnset;

  const traffic_breakdown *outgoing_traffic = nullptr;
  const traffic_breakdown *incoming_traffic = nullptr;


  unsigned long long memlat_num_mfs = kUnset;

  unsigned long long memlat_mf_total_lat = kUnset;
  unsigned long long memlat_tot_icnt2mem_lat = kUnset;
  unsigned long long memlat_tot_icnt2sh_lat  = kUnset;

  unsigned memlat_max_mf_lat = kUnset;
  unsigned memlat_max_icnt2mem_lat = kUnset;
  unsigned memlat_max_mrq_lat = kUnset;
  unsigned memlat_max_icnt2sh_lat = kUnset;

  unsigned memlat_mrq_lat_table[32]      = {0};
  unsigned memlat_dq_lat_table[32]       = {0};
  unsigned memlat_mf_lat_table[32]       = {0};
  unsigned memlat_icnt2mem_lat_table[24] = {0};
  unsigned memlat_icnt2sh_lat_table[24]  = {0};
  unsigned memlat_mf_lat_pw_table[32]    = {0};

  const unsigned *memlat_max_conc_access2samerow = nullptr;
  const unsigned *memlat_max_servicetime2samerow = nullptr;
  unsigned memlat_rowstats_n_mem = 0;
  unsigned memlat_rowstats_n_bk  = 0;

  const unsigned long long *memlat_row_access    = nullptr;
  const unsigned long long *memlat_num_activates = nullptr;
  const unsigned long long *memlat_totalbankreads = nullptr;
  const unsigned long long *memlat_totalbankwrites = nullptr;
  const unsigned long long *totalbankaccesses = nullptr;
  const unsigned long long *memlat_mf_total_laten = nullptr;
  const unsigned long long *memlat_max_mf_laten = nullptr;

  // DRAM

  const unsigned long long *dram_n_cmd      = nullptr;
  const unsigned long long *dram_n_nop      = nullptr;
  const unsigned long long *dram_n_activity = nullptr;
  const unsigned long long *dram_n_act      = nullptr;
  const unsigned long long *dram_n_pre      = nullptr;
  const unsigned long long *dram_n_req       = nullptr; 
  const unsigned long long *dram_n_ref_event = nullptr; 
  const unsigned long long *dram_n_rd       = nullptr;
  const unsigned long long *dram_n_rd_L2_A  = nullptr;
  const unsigned long long *dram_n_wr       = nullptr;
  const unsigned long long *dram_n_wr_WB    = nullptr;
  const unsigned long long *dram_bwutil = nullptr;
  const unsigned long long *dram_bk_n_access = nullptr;
  const unsigned long long *dram_bk_n_idle   = nullptr;
  // ---- DRAM BLP / locality (per dram_id) ----
  const unsigned long long *dram_banks_1time = nullptr;
  const unsigned long long *dram_banks_access_total = nullptr;

  const unsigned long long *dram_banks_time_rw = nullptr;
  const unsigned long long *dram_banks_access_rw_total = nullptr;

  const unsigned long long *dram_banks_time_ready = nullptr;
  const unsigned long long *dram_banks_access_ready_total = nullptr;

  const unsigned long long *dram_w2r_ratio_sum_1e6 = nullptr;
  const unsigned long long *dram_bkgrp_parallsim_rw = nullptr;

  const unsigned long long *dram_access_num = nullptr;
  const unsigned long long *dram_hits_num   = nullptr;
  const unsigned long long *dram_read_num   = nullptr;
  const unsigned long long *dram_hits_read_num  = nullptr;
  const unsigned long long *dram_write_num  = nullptr;
  const unsigned long long *dram_hits_write_num = nullptr;

  const unsigned long long *dram_bk_access = nullptr;


  const unsigned long long *dram_util_bw        = nullptr;
  const unsigned long long *dram_wasted_bw_col  = nullptr;
  const unsigned long long *dram_wasted_bw_row  = nullptr;
  const unsigned long long *dram_idle_bw        = nullptr;

  const unsigned long long *dram_RCDc_limit          = nullptr;
  const unsigned long long *dram_RCDWRc_limit        = nullptr;
  const unsigned long long *dram_WTRc_limit          = nullptr;
  const unsigned long long *dram_RTWc_limit          = nullptr;
  const unsigned long long *dram_CCDLc_limit         = nullptr;
  const unsigned long long *dram_rwq_limit           = nullptr;
  const unsigned long long *dram_CCDLc_limit_alone   = nullptr;
  const unsigned long long *dram_WTRc_limit_alone    = nullptr;
  const unsigned long long *dram_RTWc_limit_alone    = nullptr;

  const unsigned long long *dram_issued_total_row = nullptr;
  const unsigned long long *dram_issued_total_col = nullptr;
  const unsigned long long *dram_issued_total     = nullptr;
  const unsigned long long *dram_issued_two       = nullptr;
  const unsigned long long *dram_ave_mrqs_sum     = nullptr;

  const unsigned long long *dram_max_mrqs   = nullptr;
  const unsigned long long *dram_util_bins = nullptr; // n_mem*10
  const unsigned long long *dram_eff_bins  = nullptr; // n_mem*10

};


// -----------------------------------------------------------------------------
// Assumes cluster == SM when n_simt_cores_per_cluster == 1.
// -----------------------------------------------------------------------------
struct port_snap_t {
  unsigned long long avail = 0;
  unsigned long long data  = 0;
  unsigned long long fill  = 0;
};

struct l1d_ports_rec_t {
  bool have_begin = false;
  bool have_end   = false;
  std::vector<port_snap_t> begin_per_cluster;
  std::vector<port_snap_t> end_per_cluster;
  std::vector<bool> allowed_sm; // size=n_simt_clusters, true if kernel can run there
};




class gpgpu_sim;  //forward decl
class memory_config {
 public:
  memory_config(gpgpu_context *ctx) {
    m_valid = false;
    gpgpu_dram_timing_opt = NULL;
    gpgpu_L2_queue_config = NULL;
    gpgpu_ctx = ctx;
  }
  void set_gpu(gpgpu_sim *gpu) { m_gpu = gpu; }
  gpgpu_sim *get_gpu() const { return m_gpu; }
  void init() {
    assert(gpgpu_dram_timing_opt);
    if (strchr(gpgpu_dram_timing_opt, '=') == NULL) {
      // dram timing option in ordered variables (legacy)
      // Disabling bank groups if their values are not specified
      nbkgrp = 1;
      tCCDL = 0;
      tRTPL = 0;
      sscanf(gpgpu_dram_timing_opt, "%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",
             &nbk, &tCCD, &tRRD, &tRCD, &tRAS, &tRP, &tRC, &CL, &WL, &tCDLR,
             &tWR, &nbkgrp, &tCCDL, &tRTPL);
    } else {
      // named dram timing options (unordered)
      option_parser_t dram_opp = option_parser_create();

      option_parser_register(dram_opp, "nbk", OPT_UINT32, &nbk,
                             "number of banks", "");
      option_parser_register(dram_opp, "CCD", OPT_UINT32, &tCCD,
                             "column to column delay", "");
      option_parser_register(
          dram_opp, "RRD", OPT_UINT32, &tRRD,
          "minimal delay between activation of rows in different banks", "");
      option_parser_register(dram_opp, "RCD", OPT_UINT32, &tRCD,
                             "row to column delay", "");
      option_parser_register(dram_opp, "RAS", OPT_UINT32, &tRAS,
                             "time needed to activate row", "");
      option_parser_register(dram_opp, "RP", OPT_UINT32, &tRP,
                             "time needed to precharge (deactivate) row", "");
      option_parser_register(dram_opp, "RC", OPT_UINT32, &tRC, "row cycle time",
                             "");
      option_parser_register(dram_opp, "CDLR", OPT_UINT32, &tCDLR,
                             "switching from write to read (changes tWTR)", "");
      option_parser_register(dram_opp, "WR", OPT_UINT32, &tWR,
                             "last data-in to row precharge", "");

      option_parser_register(dram_opp, "CL", OPT_UINT32, &CL, "CAS latency",
                             "");
      option_parser_register(dram_opp, "WL", OPT_UINT32, &WL, "Write latency",
                             "");

      // Disabling bank groups if their values are not specified
      option_parser_register(dram_opp, "nbkgrp", OPT_UINT32, &nbkgrp,
                             "number of bank groups", "1");
      option_parser_register(
          dram_opp, "CCDL", OPT_UINT32, &tCCDL,
          "column to column delay between accesses to different bank groups",
          "0");
      option_parser_register(
          dram_opp, "RTPL", OPT_UINT32, &tRTPL,
          "read to precharge delay between accesses to different bank groups",
          "0");

      option_parser_delimited_string(dram_opp, gpgpu_dram_timing_opt, "=:;");
      fprintf(stdout, "DRAM Timing Options:\n");
      option_parser_print(dram_opp, stdout);
      option_parser_destroy(dram_opp);
    }

    int nbkt = nbk / nbkgrp;
    unsigned i;
    for (i = 0; nbkt > 0; i++) {
      nbkt = nbkt >> 1;
    }
    bk_tag_length = i - 1;
    assert(nbkgrp > 0 && "Number of bank groups cannot be zero");
    tRCDWR = tRCD - (WL + 1);
    if (elimnate_rw_turnaround) {
      tRTW = 0;
      tWTR = 0;
    } else {
      tRTW = (CL + (BL / data_command_freq_ratio) + 2 - WL);
      tWTR = (WL + (BL / data_command_freq_ratio) + tCDLR);
    }
    tWTP = (WL + (BL / data_command_freq_ratio) + tWR);
    dram_atom_size =
        BL * busW * gpu_n_mem_per_ctrlr;  // burst length x bus width x # chips
                                          // per partition

    assert(m_n_sub_partition_per_memory_channel > 0);
    assert((nbk % m_n_sub_partition_per_memory_channel == 0) &&
           "Number of DRAM banks must be a perfect multiple of memory sub "
           "partition");
    m_n_mem_sub_partition = m_n_mem * m_n_sub_partition_per_memory_channel;
    fprintf(stdout, "Total number of memory sub partition = %u\n",
            m_n_mem_sub_partition);

    m_address_mapping.init(m_n_mem, m_n_sub_partition_per_memory_channel);
    m_L2_config.init(&m_address_mapping);

    m_valid = true;

    sscanf(write_queue_size_opt, "%d:%d:%d",
           &gpgpu_frfcfs_dram_write_queue_size, &write_high_watermark,
           &write_low_watermark);
  }
  void reg_options(class OptionParser *opp);

  /**
   * @brief Check if the config script is in SST mode
   *
   * @return true
   * @return false
   */
  bool is_SST_mode() const { return SST_mode; }

  bool m_valid;
  mutable l2_cache_config m_L2_config;
  bool m_L2_texure_only;

  char *gpgpu_dram_timing_opt;
  char *gpgpu_L2_queue_config;
  bool l2_ideal;
  unsigned gpgpu_frfcfs_dram_sched_queue_size;
  unsigned gpgpu_dram_return_queue_size;
  enum dram_ctrl_t scheduler_type;
  bool gpgpu_memlatency_stat;
  unsigned m_n_mem;
  unsigned m_n_sub_partition_per_memory_channel;
  unsigned m_n_mem_sub_partition;
  unsigned gpu_n_mem_per_ctrlr;

  unsigned rop_latency;
  unsigned dram_latency;

  // DRAM parameters

  unsigned tCCDL;  // column to column delay when bank groups are enabled
  unsigned tRTPL;  // read to precharge delay when bank groups are enabled for
                   // GDDR5 this is identical to RTPS, if for other DRAM this is
                   // different, you will need to split them in two

  unsigned tCCD;    // column to column delay
  unsigned tRRD;    // minimal time required between activation of rows in
                    // different banks
  unsigned tRCD;    // row to column delay - time required to activate a row
                    // before a read
  unsigned tRCDWR;  // row to column delay for a write command
  unsigned tRAS;    // time needed to activate row
  unsigned tRP;     // row precharge ie. deactivate row
  unsigned
      tRC;  // row cycle time ie. precharge current, then activate different row
  unsigned tCDLR;  // Last data-in to Read command (switching from write to
                   // read)
  unsigned tWR;    // Last data-in to Row precharge

  unsigned CL;    // CAS latency
  unsigned WL;    // WRITE latency
  unsigned BL;    // Burst Length in bytes (4 in GDDR3, 8 in GDDR5)
  unsigned tRTW;  // time to switch from read to write
  unsigned tWTR;  // time to switch from write to read
  unsigned tWTP;  // time to switch from write to precharge in the same bank
  unsigned busW;

  unsigned nbkgrp;  // number of bank groups (has to be power of 2)
  unsigned
      bk_tag_length;  // number of bits that define a bank inside a bank group

  unsigned nbk;

  bool elimnate_rw_turnaround;

  unsigned
      data_command_freq_ratio;  // frequency ratio between DRAM data bus and
                                // command bus (2 for GDDR3, 4 for GDDR5)
  unsigned
      dram_atom_size;  // number of bytes transferred per read or write command

  linear_to_raw_address_translation m_address_mapping;

  unsigned icnt_flit_size;

  unsigned dram_bnk_indexing_policy;
  unsigned dram_bnkgrp_indexing_policy;
  bool dual_bus_interface;

  bool seperate_write_queue_enabled;
  char *write_queue_size_opt;
  unsigned gpgpu_frfcfs_dram_write_queue_size;
  unsigned write_high_watermark;
  unsigned write_low_watermark;
  bool m_perf_sim_memcpy;
  bool simple_dram_model;
  bool SST_mode;
  gpgpu_context *gpgpu_ctx;
  private:
      gpgpu_sim *m_gpu = nullptr;
};

extern bool g_interactive_debugger_enabled;

class gpgpu_sim_config : public power_config,
                         public gpgpu_functional_sim_config {
 public:
  gpgpu_sim_config(gpgpu_context *ctx)
      : m_shader_config(ctx), m_memory_config(ctx) {
    m_valid = false;
    gpgpu_ctx = ctx;
  }
  void reg_options(class OptionParser *opp);
  void init() {
    gpu_stat_sample_freq = 10000;
    gpu_runtime_stat_flag = 0;
    sscanf(gpgpu_runtime_stat, "%d:%x", &gpu_stat_sample_freq,
           &gpu_runtime_stat_flag);
    m_shader_config.init();
    ptx_set_tex_cache_linesize(m_shader_config.m_L1T_config.get_line_sz());
    m_memory_config.init();
    init_clock_domains();
    power_config::init();
    Trace::init();

    // initialize file name if it is not set
    time_t curr_time;
    time(&curr_time);
    char *date = ctime(&curr_time);
    char *s = date;
    while (*s) {
      if (*s == ' ' || *s == '\t' || *s == ':') *s = '-';
      if (*s == '\n' || *s == '\r') *s = 0;
      s++;
    }
    char buf[1024];
    snprintf(buf, 1024, "gpgpusim_visualizer__%s.log.gz", date);
    g_visualizer_filename = strdup(buf);

    m_valid = true;
  }
  unsigned get_core_freq() const { return core_freq; }
  unsigned num_shader() const { return m_shader_config.num_shader(); }
  unsigned num_cluster() const { return m_shader_config.n_simt_clusters; }
  unsigned get_max_concurrent_kernel() const { return max_concurrent_kernel; }

  /**
   * @brief Check if we are in SST mode
   *
   * @return true
   * @return false
   */
  bool is_SST_mode() const { return m_memory_config.SST_mode; }

  unsigned checkpoint_option;

  size_t stack_limit() const { return stack_size_limit; }
  size_t heap_limit() const { return heap_size_limit; }
  size_t sync_depth_limit() const { return runtime_sync_depth_limit; }
  size_t pending_launch_count_limit() const {
    return runtime_pending_launch_count_limit;
  }

  bool flush_l1() const { return gpgpu_flush_l1_cache; }

 private:
  void init_clock_domains(void);

  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  bool m_valid;
  shader_core_config m_shader_config;
  memory_config m_memory_config;
  // clock domains - frequency
  double core_freq;
  double icnt_freq;
  double dram_freq;
  double l2_freq;
  double core_period;
  double icnt_period;
  double dram_period;
  double l2_period;

  // GPGPU-Sim timing model options
  unsigned long long gpu_max_cycle_opt;
  unsigned long long gpu_max_insn_opt;
  unsigned gpu_max_cta_opt;
  unsigned gpu_max_completed_cta_opt;
  char *gpgpu_runtime_stat;
  bool gpgpu_flush_l1_cache;
  bool gpgpu_flush_l2_cache;
  bool gpu_deadlock_detect;
  int gpgpu_frfcfs_dram_sched_queue_size;
  int gpgpu_cflog_interval;
  char *gpgpu_clock_domains;
  unsigned max_concurrent_kernel;

  // visualizer
  bool g_visualizer_enabled;
  char *g_visualizer_filename;
  int g_visualizer_zlevel;

  // statistics collection
  int gpu_stat_sample_freq;
  int gpu_runtime_stat_flag;

  // Device Limits
  size_t stack_size_limit;
  size_t heap_size_limit;
  size_t runtime_sync_depth_limit;
  size_t runtime_pending_launch_count_limit;

  // gpu compute capability options
  unsigned int gpgpu_compute_capability_major;
  unsigned int gpgpu_compute_capability_minor;
  unsigned long long liveness_message_freq;

  friend class gpgpu_sim;
  friend class sst_gpgpu_sim;
};

struct occupancy_stats {
  occupancy_stats()
      : aggregate_warp_slot_filled(0), aggregate_theoretical_warp_slots(0) {}
  occupancy_stats(unsigned long long wsf, unsigned long long tws)
      : aggregate_warp_slot_filled(wsf),
        aggregate_theoretical_warp_slots(tws) {}

  unsigned long long aggregate_warp_slot_filled;
  unsigned long long aggregate_theoretical_warp_slots;

  float get_occ_fraction() const {
    return float(aggregate_warp_slot_filled) /
           float(aggregate_theoretical_warp_slots);
  }

  occupancy_stats &operator+=(const occupancy_stats &rhs) {
    aggregate_warp_slot_filled += rhs.aggregate_warp_slot_filled;
    aggregate_theoretical_warp_slots += rhs.aggregate_theoretical_warp_slots;
    return *this;
  }

  occupancy_stats operator+(const occupancy_stats &rhs) const {
    return occupancy_stats(
        aggregate_warp_slot_filled + rhs.aggregate_warp_slot_filled,
        aggregate_theoretical_warp_slots +
            rhs.aggregate_theoretical_warp_slots);
  }
};

class gpgpu_context;
class ptx_instruction;

class watchpoint_event {
 public:
  watchpoint_event() {
    m_thread = NULL;
    m_inst = NULL;
  }
  watchpoint_event(const ptx_thread_info *thd, const ptx_instruction *pI) {
    m_thread = thd;
    m_inst = pI;
  }
  const ptx_thread_info *thread() const { return m_thread; }
  const ptx_instruction *inst() const { return m_inst; }

 private:
  const ptx_thread_info *m_thread;
  const ptx_instruction *m_inst;
};

class warp_inst_t; // forward decl

class gpgpu_sim : public gpgpu_t {
 public:
  gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx);

  long long get_gpu_sim_cycle() const        { return (long long)gpu_sim_cycle; }
  long long get_gpu_sim_insn() const         { return (long long)gpu_sim_insn; }
  long long get_gpu_tot_sim_cycle() const    { return (long long)(gpu_tot_sim_cycle + gpu_sim_cycle); }
  long long get_gpu_tot_sim_insn() const     { return (long long)(gpu_tot_sim_insn + gpu_sim_insn); }
  long long get_gpu_tot_issued_cta() const   { return (long long)(gpu_tot_issued_cta + m_total_cta_launched); }


  void set_prop(struct cudaDeviceProp *prop);

  void launch(kernel_info_t *kinfo);
  bool can_start_kernel();
  unsigned finished_kernel();
  void set_kernel_done(kernel_info_t *kernel);
  void stop_all_running_kernels();


    // --- Per-kernel / per-job accounting helpers (NEW) --------------------
  // These are implemented in gpu-sim.cc and are used by the daemon / driver
  // code to keep proper per-kernel stats even under concurrency.
  void note_kernel_launch(kernel_info_t *kinfo);
  void note_kernel_completion(kernel_info_t *kinfo);

  // Build a stats view for a specific kernel UID from the internal
  // per-kernel maps (kernel_inst_count, kernel_mem_reply_bytes, etc.).
  kernel_stats_view_t make_kernel_stats_view(unsigned kernel_uid) const;

  // Clear all per-kernel bookkeeping for a given kernel UID once we have
  // printed its stats.
  void clear_kernel_stats(unsigned kernel_uid);


  enum class l1_kind_t { L1I, L1D, L1C, L1T };

  void record_kernel_l1_access(unsigned kernel_uid,
                              unsigned smid,
                              l1_kind_t kind,
                              cache_request_status status);


    // Event recorders (must be called at event sites: shader/L2/DRAM/ICNT)
  void record_kernel_cycle(unsigned kernel_uid, unsigned long long ncycles = 1);
  void record_kernel_inst(unsigned kernel_uid, unsigned long long ninst = 1);
  void record_kernel_issued_cta(unsigned kernel_uid, unsigned long long ncta = 1);

  void record_kernel_stall_dramfull(unsigned kernel_uid,
                                    unsigned long long ncycles = 1);
  void record_kernel_stall_icnt2sh(unsigned kernel_uid,
                                   unsigned long long ncycles = 1);

  void record_kernel_l2_request(unsigned kernel_uid, unsigned long long nreq = 1);
  void record_kernel_l2_bytes(unsigned kernel_uid, unsigned long long nbytes);
  void record_kernel_dram_request(unsigned kernel_uid, unsigned long long nreq = 1);
  void record_kernel_dram_bytes(unsigned kernel_uid, unsigned long long nbytes);
  inline void record_kernel_l1d(unsigned kernel_uid, unsigned smid, cache_request_status s) {
    record_kernel_l1_access(kernel_uid, smid, l1_kind_t::L1D, s);
  }
  inline void record_kernel_l1i(unsigned kernel_uid, unsigned smid, cache_request_status s) {
    record_kernel_l1_access(kernel_uid, smid, l1_kind_t::L1I, s);
  }
  inline void record_kernel_l1c(unsigned kernel_uid, unsigned smid, cache_request_status s) {
    record_kernel_l1_access(kernel_uid, smid, l1_kind_t::L1C, s);
  }
  inline void record_kernel_l1t(unsigned kernel_uid, unsigned smid, cache_request_status s) {
    record_kernel_l1_access(kernel_uid, smid, l1_kind_t::L1T, s);
  }
  void record_kernel_icnt_stats(unsigned kid, const mem_fetch *mf);
  void record_kernel_mem_inst_commit(unsigned kernel_uid,const warp_inst_t &inst);
  unsigned kernel_uid_from_stream(unsigned long long streamID) const;

  void record_kernel_warp_issue_distro(unsigned smid,unsigned warp_id,unsigned long long streamID);

  void record_kernel_shader_active_count_bucket(unsigned kernel_uid,unsigned active_count);

  void record_kernel_shader_idle_cycle(unsigned kernel_uid);
  void record_kernel_shader_scoreboard_cycle(unsigned kernel_uid);
  void record_kernel_shader_stall_cycle(unsigned kernel_uid);

  void record_kernel_scheduler_issue(unsigned kernel_uid,unsigned sched_id,unsigned issued_count);
  void record_kernel_inst_commit(unsigned int kernel_uid, unsigned int active_lanes);
  unsigned sole_active_kernel_uid() const;

  void shader_print_scheduler_stat(FILE *fout,bool print_dynamic_info,const kernel_stats_view_t *view) const;
  void record_kernel_mem_read(unsigned kernel_uid,const warp_inst_t &inst);

  // void record_kernel_stall_shd_mem_cmem_resource(unsigned kid);
  // void record_kernel_stall_shd_mem_smem_bkconf(unsigned kid);
  // void record_kernel_stall_shd_mem_gl_resource(unsigned kid);
  // void record_kernel_stall_shd_mem_gl_coal(unsigned kid);
  // void record_kernel_stall_shd_mem_gl_data_port(unsigned kid);

  void record_kernel_shmem_bkconflict(unsigned kid);
  void record_kernel_l1cache_bkconflict(unsigned kid);
  void record_kernel_intrawarp_mshr_merge(unsigned kid);
  void record_kernel_cmem_portconflict(unsigned kid);
  void record_kernel_reg_bank_conflict_stall(unsigned kid);
  void record_kernel_stall_shd_mem(unsigned kernel_uid,unsigned mem_space, unsigned stall_reason);
  void record_kernel_outgoing_traffic(unsigned kid, mem_fetch *mf, unsigned sz);
  void record_kernel_incoming_traffic(unsigned kid, mem_fetch *mf, unsigned sz);
  void record_kernel_memlat(unsigned kid,unsigned mf_lat,unsigned icnt2mem_lat,unsigned mrq_lat,unsigned icnt2sh_lat);
  void record_kernel_mrq_lat_bucket(unsigned kid, unsigned lat);
  void record_kernel_icnt2mem_lat_bucket(unsigned kid, unsigned lat);
  void record_kernel_icnt2sh_lat_bucket(unsigned kid, unsigned lat);
  void record_kernel_mf_lat_bucket(unsigned kid, unsigned lat);

  void record_kernel_mf_lat_pw_accum(unsigned kid, unsigned mf_lat);
  void flush_kernel_mf_lat_pw_tables();
  inline unsigned flat_idx(unsigned dram, unsigned bank, unsigned nbk) {
    return dram * nbk + bank;
  }

  void record_kernel_row_episode_access(unsigned kid,unsigned dram,unsigned bank,unsigned cnt);
  void record_kernel_row_episode_servicetime(unsigned kid,unsigned dram,unsigned bank,unsigned srv);

  void record_kernel_row_access(unsigned kid,unsigned dram,unsigned bank);
  void record_kernel_row_activate(unsigned kid,unsigned dram,unsigned bank);
  void record_kernel_totalbankread(unsigned kid,unsigned dram,unsigned bank,unsigned long long inc);
  void record_kernel_mf_bank_lat_sum(unsigned kid,unsigned dram,unsigned bank,unsigned mf_latency);
  void record_max_mf_lat_per_bank(unsigned kid,unsigned dram,unsigned bank,unsigned mf_latency);

  void record_kernel_dram_cycle_counters(unsigned kid,unsigned dram_id,unsigned long long inc_cmd,
                                                  unsigned long long inc_nop,
                                                  unsigned long long inc_act);
  void record_kernel_dram_row_cmd_counters(unsigned kid,
                                         unsigned dram_id,
                                         unsigned long long inc_act,
                                         unsigned long long inc_pre);
  void record_kernel_dram_req_ref_event(unsigned kid,
                                        unsigned dram_id,
                                        unsigned long long inc_req,
                                        unsigned long long inc_ref);
  void record_kernel_dram_rw_counters(unsigned kid, unsigned dram_id,
                                    unsigned long long inc_rd,
                                    unsigned long long inc_rd_l2a,
                                    unsigned long long inc_wr,
                                    unsigned long long inc_wr_wb);
  void record_kernel_dram_bwutil(unsigned kid, unsigned dram_id,
                                unsigned long long inc_bwutil);
                            
  void record_kernel_dram_bank_access(unsigned kid, unsigned dram_id, unsigned bank,
                                      unsigned long long inc_access);

  void record_kernel_dram_bank_idle(unsigned kid, unsigned dram_id, unsigned bank,
                                    unsigned long long inc_idle);


  void record_kernel_dram_rowbuf_locality(unsigned kid, unsigned dram_id,
                                          bool is_write, bool is_row_hit);

  void record_kernel_dram_blp_stats(unsigned kid, unsigned dram_id,
                                    unsigned long long inc_banks_1time,
                                    unsigned long long inc_banks_access_total,
                                    unsigned long long inc_banks_time_rw,
                                    unsigned long long inc_banks_access_rw_total,
                                    unsigned long long inc_banks_time_ready,
                                    unsigned long long inc_banks_access_ready_total,
                                    unsigned long long inc_w2r_ratio_sum_1e6,
                                    unsigned long long inc_bkgrp_parallsim_rw);      
                                  
  void record_kernel_dram_bw_class(unsigned kid, unsigned dram_id,
                                 unsigned long long inc_util,
                                 unsigned long long inc_wcol,
                                 unsigned long long inc_wrow,
                                 unsigned long long inc_idle);

  void record_kernel_dram_bw_bottlenecks(unsigned kid, unsigned dram_id,
                                        unsigned long long inc_RCDc,
                                        unsigned long long inc_RCDWRc,
                                        unsigned long long inc_WTRc,
                                        unsigned long long inc_RTWc,
                                        unsigned long long inc_CCDLc,
                                        unsigned long long inc_rwq,
                                        unsigned long long inc_CCDLc_alone,
                                        unsigned long long inc_WTRc_alone,
                                        unsigned long long inc_RTWc_alone);

  void record_kernel_dram_issue_stats(unsigned kid, unsigned dram_id,
                                    unsigned long long inc_row,
                                    unsigned long long inc_col,
                                    unsigned long long inc_total,
                                    unsigned long long inc_two,
                                    unsigned long long inc_ave_mrqs);

  void record_kernel_dram_max_mrqs(unsigned kid, unsigned dram_id,
                                  unsigned long long qlen);

  // called once per “stat interval” (or at kernel end) to bin util/eff
  void record_kernel_dram_util_eff_bins_interval(unsigned kid, unsigned dram_id);
  // SM -> kernel uid owner (0 = none)
  std::vector<unsigned> m_sm_owner_kid;

  unsigned kernel_uid_from_smid(unsigned sid) const;
  void bind_kernel_to_allowed_sms(const kernel_info_t *k);
  void unbind_kernel_from_sms(unsigned kid);


  void init();
  void cycle();
  bool active();
  bool cycle_insn_cta_max_hit() {
    return (m_config.gpu_max_cycle_opt && (gpu_tot_sim_cycle + gpu_sim_cycle) >=
                                              m_config.gpu_max_cycle_opt) ||
           (m_config.gpu_max_insn_opt &&
            (gpu_tot_sim_insn + gpu_sim_insn) >= m_config.gpu_max_insn_opt) ||
           (m_config.gpu_max_cta_opt &&
            (gpu_tot_issued_cta >= m_config.gpu_max_cta_opt)) ||
           (m_config.gpu_max_completed_cta_opt &&
            (gpu_completed_cta >= m_config.gpu_max_completed_cta_opt));
  }
  void print_stats(unsigned long long streamID);
  // new (daemon path)
  void print_stats(unsigned long long streamID,
                  const kernel_stats_view_t *view);
  void print_stats(unsigned long long streamID,
                  const kernel_stats_view_t *view,
                  const char *single_kernel_name,
                  int single_kernel_uid);



  void update_stats();
  void deadlock_check();
  void inc_completed_cta() { gpu_completed_cta++; }
  void get_pdom_stack_top_info(unsigned sid, unsigned tid, unsigned *pc,
                               unsigned *rpc);

  int shared_mem_size() const;
  int shared_mem_per_block() const;
  int compute_capability_major() const;
  int compute_capability_minor() const;
  int num_registers_per_core() const;
  int num_registers_per_block() const;
  int wrp_size() const;
  int shader_clock() const;
  int max_cta_per_core() const;
  int get_max_cta(const kernel_info_t &k) const;
  const struct cudaDeviceProp *get_prop() const;
  enum divergence_support_t simd_model() const;

  unsigned threads_per_core() const;
  bool get_more_cta_left() const;
  bool kernel_more_cta_left(kernel_info_t *kernel) const;
  bool hit_max_cta_count() const;
  kernel_info_t *select_kernel();
  PowerscalingCoefficients *get_scaling_coeffs();
  void decrement_kernel_latency();

  const gpgpu_sim_config &get_config() const { return m_config; }
  void gpu_print_stat(unsigned long long streamID,
                      const kernel_stats_view_t *view,
                      const char *single_kernel_name,
                      int single_kernel_uid);
  void dump_pipeline(int mask, int s, int m) const;
  void perf_memcpy_to_gpu(size_t dst_start_addr, size_t count);

  // The next three functions added to be used by the functional simulation
  // function

  //! Get shader core configuration
  /*!
   * Returning the configuration of the shader core, used by the functional
   * simulation only so far
   */
  const shader_core_config *getShaderCoreConfig();

  //! Get shader core Memory Configuration
  /*!
   * Returning the memory configuration of the shader core, used by the
   * functional simulation only so far
   */
  const memory_config *getMemoryConfig();

  //! Get shader core SIMT cluster
  /*!
   * Returning the cluster of of the shader core, used by the functional
   * simulation so far
   */
  simt_core_cluster *getSIMTCluster();

  void hit_watchpoint(unsigned watchpoint_num, ptx_thread_info *thd,
                      const ptx_instruction *pI);

  /**
   * @brief Check if we are in SST mode
   *
   * @return true
   * @return false
   */
  bool is_SST_mode() { return m_config.is_SST_mode(); }

  // backward pointer
  class gpgpu_context *gpgpu_ctx;

 protected:
  // clocks
  void reinit_clock_domains(void);
  int next_clock_domain(void);
  void issue_block2core();
  void print_dram_stats(FILE *fout) const;
  void shader_print_runtime_stat(FILE *fout);
  void shader_print_l1_miss_stat(FILE *fout) const;
  void shader_print_cache_stats(FILE *fout) const;
  void shader_print_cache_stats(FILE *fout, const kernel_stats_view_t *view) const;
  void shader_print_scheduler_stat(FILE *fout, bool print_dynamic_info) const;
  void visualizer_printstat();
  void print_shader_cycle_distro(FILE *fout) const;

  void gpgpu_debug();

 protected:
  ///// data /////
  class simt_core_cluster **m_cluster;
  class memory_partition_unit **m_memory_partition_unit;
  class memory_sub_partition **m_memory_sub_partition;

  std::vector<kernel_info_t *> m_running_kernels;
  unsigned m_last_issued_kernel;

  std::list<unsigned> m_finished_kernel;
  // m_total_cta_launched == per-kernel count. gpu_tot_issued_cta == global
  // count.
  unsigned long long m_total_cta_launched;
  unsigned long long gpu_tot_issued_cta;
  unsigned gpu_completed_cta;

  unsigned m_last_cluster_issue;
  float *average_pipeline_duty_cycle;
  float *active_sms;
  // time of next rising edge
  double core_time;
  double icnt_time;
  double dram_time;
  double l2_time;

  // debug
  bool gpu_deadlock;

  //// configuration parameters ////
  const gpgpu_sim_config &m_config;

  const struct cudaDeviceProp *m_cuda_properties;
  const shader_core_config *m_shader_config;
  const memory_config *m_memory_config;

  // stats
  class shader_core_stats *m_shader_stats;
  class memory_stats_t *m_memory_stats;
  class power_stat_t *m_power_stats;
  class gpgpu_sim_wrapper *m_gpgpusim_wrapper;
  unsigned long long last_gpu_sim_insn;

  unsigned long long last_liveness_message_time;

  std::map<std::string, FuncCache> m_special_cache_config;

  std::vector<std::string>
      m_executed_kernel_names;  //< names of kernel for stat printout
  std::vector<unsigned>
      m_executed_kernel_uids;  //< uids of kernel launches for stat printout
  std::map<unsigned, watchpoint_event> g_watchpoint_hits;

  std::string executed_kernel_info_string();  //< format the kernel information
                                              // into a string for stat printout
  std::string executed_kernel_name();
  void clear_executed_kernel_info();  //< clear the kernel information after
                                      // stat printout
  void snapshot_l1d_ports_per_cluster(std::vector<port_snap_t>& out) const;
  virtual void createSIMTCluster() = 0;

 public:
  unsigned long long gpu_sim_insn;
  unsigned long long gpu_tot_sim_insn;
  unsigned long long gpu_sim_insn_last_update;
  unsigned gpu_sim_insn_last_update_sid;
  occupancy_stats gpu_occupancy;
  occupancy_stats gpu_tot_occupancy;

  typedef struct {
    unsigned long long start_cycle;
    unsigned long long end_cycle;
  } kernel_time_t;
  std::map<unsigned long long, std::map<unsigned, kernel_time_t>>
      gpu_kernel_time;
  unsigned long long last_streamID;
  unsigned long long last_uid;
  cache_stats aggregated_l1_stats;
  cache_stats aggregated_l2_stats;

  // performance counter for stalls due to congestion.
  // performance counter for stalls due to congestion.
  unsigned int gpu_stall_dramfull;
  unsigned int gpu_stall_icnt2sh;
  unsigned long long partiton_reqs_in_parallel;
  unsigned long long partiton_reqs_in_parallel_total;
  unsigned long long partiton_reqs_in_parallel_util;
  unsigned long long partiton_reqs_in_parallel_util_total;
  unsigned long long gpu_sim_cycle_parition_util;
  unsigned long long gpu_tot_sim_cycle_parition_util;
  unsigned long long partiton_replys_in_parallel;
  unsigned long long partiton_replys_in_parallel_total;
  // --- Per-kernel / per-job statistics (keyed by kernel launch UID) ---

  // Total instructions executed that we attribute to this kernel.
  // (Will be used for per-kernel gpu_sim_insn / gpu_tot_sim_insn if we
  //  decide not to rely only on the external view struct.)
  std::map<unsigned, unsigned long long> kernel_inst_count;

  // Per-kernel memory / L2 bandwidth stats.
  // We will increment these when L2 sends replies for mem_fetch objects
  // belonging to a given kernel (via mem_fetch->get_kernel()->get_uid()).
  //
  //   kernel_mem_reply_bytes[k]        : total bytes returned to SMs
  //   kernel_mem_concurrency_sum[k]    : sum over cycles of concurrent replies
  //   kernel_mem_busy_cycles[k]        : #cycles where this kernel saw >=1 reply
  std::map<unsigned, unsigned long long> kernel_mem_reply_bytes;
  std::map<unsigned, unsigned long long> kernel_mem_concurrency_sum;
  std::map<unsigned, unsigned long long> kernel_mem_busy_cycles;

  // Per-kernel launch geometry / CTAs.
  // We will fill these when the kernel is launched.
  std::map<unsigned, unsigned> kernel_cta_count;        // total CTAs of the kernel
  std::map<unsigned, unsigned> kernel_threads_per_cta;  // threads per CTA

  // Per-kernel congestion-related stalls (optional, but useful for per-job view).
  std::map<unsigned, unsigned> kernel_stall_dramfull;   // DRAM-full stalls per kernel
  std::map<unsigned, unsigned> kernel_stall_icnt2sh;    // interconnect-to-shader stalls per kernel

  // Map global streamID -> current kernel UID running on that stream.
  // This lets us attribute some events (like stalls) to "the kernel that owns
  // this stream" when we only have a streamID handy.
  std::map<unsigned long long, unsigned> stream_to_kernel_map;


  FuncCache get_cache_config(std::string kernel_name);
  void set_cache_config(std::string kernel_name, FuncCache cacheConfig);
  bool has_special_cache_config(std::string kernel_name);
  void change_cache_config(FuncCache cache_config);
  void set_cache_config(std::string kernel_name);
  int pick_sampling_core_for_kernel_(const kernel_info_t *k) const;
  void record_kernel_warp_issue(unsigned smid, unsigned warp_id,unsigned sch_id,unsigned active_count,unsigned long long streamID,const warp_inst_t *inst);

  // Jin: functional simulation for CDP
 protected:
 //MY ADDITION
 port_snap_t snap_l1d_ports_for_cluster_(unsigned cluster_id) const;
  // set by stream operation every time a functoinal simulation is done
  bool m_functional_sim;
  kernel_info_t *m_functional_sim_kernel;

  // -------------------------------------------------------------------------
  // Concurrency-safe per-kernel accumulators (keyed by kernel launch UID)
  // -------------------------------------------------------------------------
  kernel_stats_accum_t &kernel_stats_mut_(unsigned kernel_uid);
  const kernel_stats_accum_t *kernel_stats_find_(unsigned kernel_uid) const;

  std::unordered_map<unsigned, kernel_stats_accum_t> m_kernel_stats_;
  std::unordered_map<unsigned, l1d_ports_rec_t> m_l1d_ports_rec;

  // -------------------------------------------------------------------------
  // Per-kernel scheduler issue distro recorder 
  // -------------------------------------------------------------------------
  struct sched_issue_rec_t {
    int sampling_core = -1;              // which SM we count on
    std::vector<unsigned> distro;        // distro[warp_id]++
  };

  std::unordered_map<unsigned, sched_issue_rec_t> m_sched_issue_rec_;



 public:
  bool is_functional_sim() { return m_functional_sim; }
  kernel_info_t *get_functional_kernel() { return m_functional_sim_kernel; }
  std::vector<kernel_info_t *> get_running_kernels() {
    return m_running_kernels;
  }
  void functional_launch(kernel_info_t *k) {
    m_functional_sim = true;
    m_functional_sim_kernel = k;
  }
  void finish_functional_sim(kernel_info_t *k) {
    assert(m_functional_sim);
    assert(m_functional_sim_kernel == k);
    m_functional_sim = false;
    m_functional_sim_kernel = NULL;
  }
};

class exec_gpgpu_sim : public gpgpu_sim {
 public:
  exec_gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx)
      : gpgpu_sim(config, ctx) {
    createSIMTCluster();
  }

  virtual void createSIMTCluster();
};

/**
 * @brief A GPGPUSim class customized to SST Balar interfacing
 *
 */
class sst_gpgpu_sim : public gpgpu_sim {
 public:
  sst_gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx)
      : gpgpu_sim(config, ctx) {
    createSIMTCluster();
  }

  // SST memory handling
  std::vector<std::deque<mem_fetch *>>
      SST_gpgpu_reply_buffer; /** SST mem response queue */

  /**
   * @brief Receive mem request's response from SST and put
   *        it in a buffer (SST_gpgpu_reply_buffer)
   *
   * @param core_id
   * @param mem_req
   */
  void SST_receive_mem_reply(unsigned core_id, void *mem_req);

  /**
   * @brief Pop the head of the buffer queue to get the
   *        memory response
   *
   * @param core_id
   * @return mem_fetch*
   */
  mem_fetch *SST_pop_mem_reply(unsigned core_id);

  virtual void createSIMTCluster();

  // SST Balar interfacing
  /**
   * @brief Advance core and collect stats
   *
   */
  void SST_cycle();

  /**
   * @brief Wrapper of SST_cycle()
   *
   */
  void cycle();

  /**
   * @brief Whether the GPU is active, removed test for
   *        memory system since that is handled in SST
   *
   * @return true
   * @return false
   */
  bool active();

  /**
   * @brief SST mode use SST memory system instead, so the memcpy
   *        is empty here
   *
   * @param dst_start_addr
   * @param count
   */
  void perf_memcpy_to_gpu(size_t dst_start_addr, size_t count) {};

  /**
   * @brief Check if the SST config matches up with the
   *        gpgpusim.config in core number
   *
   * @param sst_numcores SST core count
   */
  void SST_gpgpusim_numcores_equal_check(unsigned sst_numcores);
};



#endif
