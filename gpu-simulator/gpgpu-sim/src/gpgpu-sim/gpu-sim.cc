// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, George L. Yuan,
// Ali Bakhoda, Andrew Turner, Ivan Sham, Vijay Kandiah, Nikos Hardavellas,
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue
// University All rights reserved.
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

#include "gpu-sim.h"
#include "shader.h"
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include "zlib.h"
#include <map>

#include "dram.h"
#include "mem_fetch.h"
#include "shader.h"
#include "shader_trace.h"

#include <time.h>
#include "addrdec.h"
#include "delayqueue.h"
#include "dram.h"
#include "gpu-cache.h"
#include "gpu-misc.h"
#include "icnt_wrapper.h"
#include "l2cache.h"
#include "shader.h"
#include "stat-tool.h"

#include "../../libcuda/gpgpu_context.h"
#include "../abstract_hardware_model.h"
#include "../cuda-sim/cuda-sim.h"
#include "../cuda-sim/cuda_device_runtime.h"
#include "../cuda-sim/ptx-stats.h"
#include "../cuda-sim/ptx_ir.h"
#include "../debug.h"
#include "../gpgpusim_entrypoint.h"
#include "../statwrapper.h"
#include "../trace.h"
#include "mem_latency_stat.h"
#include "power_stat.h"
#include "stats.h"
#include "visualizer.h"

#ifdef GPGPUSIM_POWER_MODEL
#include "power_interface.h"
#else
class gpgpu_sim_wrapper {};
#endif

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>

// #define MAX(a, b) (((a) > (b)) ? (a) : (b)) //redefined

bool g_interactive_debugger_enabled = false;

tr1_hash_map<new_addr_type, unsigned> address_random_interleaving;

/* Clock Domains */

#define CORE 0x01
#define L2 0x02
#define DRAM 0x04
#define ICNT 0x08

#define MEM_LATENCY_STAT_IMPL

#include "mem_latency_stat.h"

void power_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-accelwattch_xml_file", OPT_CSTR,
                         &g_power_config_name, "AccelWattch XML file",
                         "accelwattch_sass_sim.xml");

  option_parser_register(opp, "-power_simulation_enabled", OPT_BOOL,
                         &g_power_simulation_enabled,
                         "Turn on power simulator (1=On, 0=Off)", "0");

  option_parser_register(opp, "-power_per_cycle_dump", OPT_BOOL,
                         &g_power_per_cycle_dump,
                         "Dump detailed power output each cycle", "0");

  option_parser_register(opp, "-hw_perf_file_name", OPT_CSTR,
                         &g_hw_perf_file_name,
                         "Hardware Performance Statistics file", "hw_perf.csv");

  option_parser_register(
      opp, "-hw_perf_bench_name", OPT_CSTR, &g_hw_perf_bench_name,
      "Kernel Name in Hardware Performance Statistics file", "");

  option_parser_register(opp, "-power_simulation_mode", OPT_INT32,
                         &g_power_simulation_mode,
                         "Switch performance counter input for power "
                         "simulation (0=Sim, 1=HW, 2=HW-Sim Hybrid)",
                         "0");

  option_parser_register(opp, "-dvfs_enabled", OPT_BOOL, &g_dvfs_enabled,
                         "Turn on DVFS for power model", "0");
  option_parser_register(opp, "-aggregate_power_stats", OPT_BOOL,
                         &g_aggregate_power_stats,
                         "Accumulate power across all kernels", "0");

  // Accelwattch Hyrbid Configuration

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L1_RH", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L1_RH],
      "Get L1 Read Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L1_RM", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L1_RM],
      "Get L1 Read Misses for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L1_WH", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L1_WH],
      "Get L1 Write Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L1_WM", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L1_WM],
      "Get L1 Write Misses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L2_RH", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L2_RH],
      "Get L2 Read Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L2_RM", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L2_RM],
      "Get L2 Read Misses for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L2_WH", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L2_WH],
      "Get L2 Write Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L2_WM", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L2_WM],
      "Get L2 Write Misses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_CC_ACC", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_CC_ACC],
      "Get Constant Cache Acesses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_SHARED_ACC", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_SHRD_ACC],
      "Get Shared Memory Acesses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(opp, "-accelwattch_hybrid_perfsim_DRAM_RD", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_DRAM_RD],
                         "Get DRAM Reads for Accelwattch-Hybrid from Accel-Sim",
                         "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_DRAM_WR", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_DRAM_WR],
      "Get DRAM Writes for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_NOC", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_NOC],
      "Get Interconnect Acesses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_PIPE_DUTY", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_PIPE_DUTY],
      "Get Pipeline Duty Cycle Acesses for Accelwattch-Hybrid from Accel-Sim",
      "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_NUM_SM_IDLE", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_NUM_SM_IDLE],
      "Get Number of Idle SMs for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_CYCLES", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_CYCLES],
      "Get Executed Cycles for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_VOLTAGE", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_VOLTAGE],
      "Get Chip Voltage for Accelwattch-Hybrid from Accel-Sim", "0");

  // Output Data Formats
  option_parser_register(
      opp, "-power_trace_enabled", OPT_BOOL, &g_power_trace_enabled,
      "produce a file for the power trace (1=On, 0=Off)", "0");

  option_parser_register(
      opp, "-power_trace_zlevel", OPT_INT32, &g_power_trace_zlevel,
      "Compression level of the power trace output log (0=no comp, 9=highest)",
      "6");

  option_parser_register(
      opp, "-steady_power_levels_enabled", OPT_BOOL,
      &g_steady_power_levels_enabled,
      "produce a file for the steady power levels (1=On, 0=Off)", "0");

  option_parser_register(opp, "-steady_state_definition", OPT_CSTR,
                         &gpu_steady_state_definition,
                         "allowed deviation:number of samples", "8:4");
}

void memory_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-gpgpu_perf_sim_memcpy", OPT_BOOL,
                         &m_perf_sim_memcpy, "Fill the L2 cache on memcpy",
                         "1");
  option_parser_register(opp, "-gpgpu_simple_dram_model", OPT_BOOL,
                         &simple_dram_model,
                         "simple_dram_model with fixed latency and BW", "0");
  option_parser_register(opp, "-gpgpu_dram_scheduler", OPT_INT32,
                         &scheduler_type, "0 = fifo, 1 = FR-FCFS (defaul)",
                         "1");
  option_parser_register(opp, "-gpgpu_dram_partition_queues", OPT_CSTR,
                         &gpgpu_L2_queue_config, "i2$:$2d:d2$:$2i", "8:8:8:8");

  option_parser_register(opp, "-l2_ideal", OPT_BOOL, &l2_ideal,
                         "Use a ideal L2 cache that always hit", "0");
  option_parser_register(
      opp, "-gpgpu_cache:dl2", OPT_CSTR, &m_L2_config.m_config_string,
      "unified banked L2 data cache config "
      " {<sector?>:<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>:<set_"
      "index_fn>,<mshr>:<N>:<merge>,<mq>:<fifo_entry>,<data_port_width>",
      "S:32:128:24,L:B:m:L:P,A:192:4,32:0,32");
  option_parser_register(opp, "-gpgpu_cache:dl2_texture_only", OPT_BOOL,
                         &m_L2_texure_only, "L2 cache used for texture only",
                         "1");
  option_parser_register(
      opp, "-gpgpu_n_mem", OPT_UINT32, &m_n_mem,
      "number of memory modules (e.g. memory controllers) in gpu", "8");
  option_parser_register(opp, "-gpgpu_n_sub_partition_per_mchannel", OPT_UINT32,
                         &m_n_sub_partition_per_memory_channel,
                         "number of memory subpartition in each memory module",
                         "1");
  option_parser_register(opp, "-gpgpu_n_mem_per_ctrlr", OPT_UINT32,
                         &gpu_n_mem_per_ctrlr,
                         "number of memory chips per memory controller", "1");
  option_parser_register(opp, "-gpgpu_memlatency_stat", OPT_INT32,
                         &gpgpu_memlatency_stat,
                         "track and display latency statistics 0x2 enables MC, "
                         "0x4 enables queue logs",
                         "0");
  option_parser_register(opp, "-gpgpu_frfcfs_dram_sched_queue_size", OPT_INT32,
                         &gpgpu_frfcfs_dram_sched_queue_size,
                         "0 = unlimited (default); # entries per chip", "0");
  option_parser_register(opp, "-gpgpu_dram_return_queue_size", OPT_INT32,
                         &gpgpu_dram_return_queue_size,
                         "0 = unlimited (default); # entries per chip", "0");
  option_parser_register(opp, "-gpgpu_dram_buswidth", OPT_UINT32, &busW,
                         "default = 4 bytes (8 bytes per cycle at DDR)", "4");
  option_parser_register(
      opp, "-gpgpu_dram_burst_length", OPT_UINT32, &BL,
      "Burst length of each DRAM request (default = 4 data bus cycle)", "4");
  option_parser_register(opp, "-dram_data_command_freq_ratio", OPT_UINT32,
                         &data_command_freq_ratio,
                         "Frequency ratio between DRAM data bus and command "
                         "bus (default = 2 times, i.e. DDR)",
                         "2");
  option_parser_register(
      opp, "-gpgpu_dram_timing_opt", OPT_CSTR, &gpgpu_dram_timing_opt,
      "DRAM timing parameters = "
      "{nbk:tCCD:tRRD:tRCD:tRAS:tRP:tRC:CL:WL:tCDLR:tWR:nbkgrp:tCCDL:tRTPL}",
      "4:2:8:12:21:13:34:9:4:5:13:1:0:0");
  option_parser_register(opp, "-gpgpu_l2_rop_latency", OPT_UINT32, &rop_latency,
                         "ROP queue latency (default 85)", "85");
  option_parser_register(opp, "-dram_latency", OPT_UINT32, &dram_latency,
                         "DRAM latency (default 30)", "30");
  option_parser_register(opp, "-dram_dual_bus_interface", OPT_UINT32,
                         &dual_bus_interface,
                         "dual_bus_interface (default = 0) ", "0");
  option_parser_register(opp, "-dram_bnk_indexing_policy", OPT_UINT32,
                         &dram_bnk_indexing_policy,
                         "dram_bnk_indexing_policy (0 = normal indexing, 1 = "
                         "Xoring with the higher bits) (Default = 0)",
                         "0");
  option_parser_register(opp, "-dram_bnkgrp_indexing_policy", OPT_UINT32,
                         &dram_bnkgrp_indexing_policy,
                         "dram_bnkgrp_indexing_policy (0 = take higher bits, 1 "
                         "= take lower bits) (Default = 0)",
                         "0");
  option_parser_register(opp, "-dram_seperate_write_queue_enable", OPT_BOOL,
                         &seperate_write_queue_enabled,
                         "Seperate_Write_Queue_Enable", "0");
  option_parser_register(opp, "-dram_write_queue_size", OPT_CSTR,
                         &write_queue_size_opt, "Write_Queue_Size", "32:28:16");
  option_parser_register(
      opp, "-dram_elimnate_rw_turnaround", OPT_BOOL, &elimnate_rw_turnaround,
      "elimnate_rw_turnaround i.e set tWTR and tRTW = 0", "0");
  option_parser_register(opp, "-icnt_flit_size", OPT_UINT32, &icnt_flit_size,
                         "icnt_flit_size", "32");
  // SST mode activate
  option_parser_register(opp, "-SST_mode", OPT_BOOL, &SST_mode, "SST mode",
                         "0");
  m_address_mapping.addrdec_setoption(opp);
}

void shader_core_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-gpgpu_simd_model", OPT_INT32, &model,
                         "1 = post-dominator", "1");
  option_parser_register(
      opp, "-gpgpu_shader_core_pipeline", OPT_CSTR,
      &gpgpu_shader_core_pipeline_opt,
      "shader core pipeline config, i.e., {<nthread>:<warpsize>}", "1024:32");
  option_parser_register(opp, "-gpgpu_tex_cache:l1", OPT_CSTR,
                         &m_L1T_config.m_config_string,
                         "per-shader L1 texture cache  (READ-ONLY) config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>:<rf>}",
                         "8:128:5,L:R:m:N,F:128:4,128:2");
  option_parser_register(
      opp, "-gpgpu_const_cache:l1", OPT_CSTR, &m_L1C_config.m_config_string,
      "per-shader L1 constant memory cache  (READ-ONLY) config "
      " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<"
      "merge>,<mq>} ",
      "64:64:2,L:R:f:N,A:2:32,4");
  option_parser_register(
      opp, "-gpgpu_cache:il1", OPT_CSTR, &m_L1I_config.m_config_string,
      "shader L1 instruction cache config "
      " {<sector?>:<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
      "alloc>:<set_index_fn>,<mshr>:<N>:<merge>,<mq>} ",
      "N:64:128:16,L:R:f:N:L,S:2:48,4");
  option_parser_register(opp, "-gpgpu_cache:dl1", OPT_CSTR,
                         &m_L1D_config.m_config_string,
                         "per-shader L1 data cache config "
                         " {<sector?>:<nsets>:<bsize>:<assoc>,<rep>:<wr>:<"
                         "alloc>:<wr_alloc>:<set_index_fn>,<mshr>:<N>:<merge>,<"
                         "mq>:<fifo_entry>,<data_port_width> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_l1_cache_write_ratio", OPT_UINT32,
                         &m_L1D_config.m_wr_percent, "L1D write ratio", "0");
  option_parser_register(opp, "-gpgpu_l1_banks", OPT_UINT32,
                         &m_L1D_config.l1_banks, "The number of L1 cache banks",
                         "1");
  option_parser_register(opp, "-gpgpu_l1_banks_byte_interleaving", OPT_UINT32,
                         &m_L1D_config.l1_banks_byte_interleaving,
                         "l1 banks byte interleaving granularity", "32");
  option_parser_register(opp, "-gpgpu_l1_banks_hashing_function", OPT_UINT32,
                         &m_L1D_config.l1_banks_hashing_function,
                         "l1 banks hashing function", "0");
  option_parser_register(opp, "-gpgpu_l1_latency", OPT_UINT32,
                         &m_L1D_config.l1_latency, "L1 Hit Latency", "1");
  option_parser_register(opp, "-gpgpu_smem_latency", OPT_UINT32, &smem_latency,
                         "smem Latency", "3");
  option_parser_register(opp, "-gpgpu_cache:dl1PrefL1", OPT_CSTR,
                         &m_L1D_config.m_config_stringPrefL1,
                         "per-shader L1 data cache config "
                         " {<sector?>:<nsets>:<bsize>:<assoc>,<rep>:<wr>:<"
                         "alloc>:<wr_alloc>:<set_index_fn>,<mshr>:<N>:<merge>,<"
                         "mq>:<fifo_entry>,<data_port_width> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_cache:dl1PrefShared", OPT_CSTR,
                         &m_L1D_config.m_config_stringPrefShared,
                         "per-shader L1 data cache config "
                         " {<sector?>:<nsets>:<bsize>:<assoc>,<rep>:<wr>:<"
                         "alloc>:<wr_alloc>:<set_index_fn>,<mshr>:<N>:<merge>,<"
                         "mq>:<fifo_entry>,<data_port_width> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_gmem_skip_L1D", OPT_BOOL, &gmem_skip_L1D,
                         "global memory access skip L1D cache (implements "
                         "-Xptxas -dlcm=cg, default=no skip)",
                         "0");

  option_parser_register(opp, "-gpgpu_perfect_mem", OPT_BOOL,
                         &gpgpu_perfect_mem,
                         "enable perfect memory mode (no cache miss)", "0");
  option_parser_register(
      opp, "-n_regfile_gating_group", OPT_UINT32, &n_regfile_gating_group,
      "group of lanes that should be read/written together)", "4");
  option_parser_register(
      opp, "-gpgpu_clock_gated_reg_file", OPT_BOOL, &gpgpu_clock_gated_reg_file,
      "enable clock gated reg file for power calculations", "0");
  option_parser_register(
      opp, "-gpgpu_clock_gated_lanes", OPT_BOOL, &gpgpu_clock_gated_lanes,
      "enable clock gated lanes for power calculations", "0");
  option_parser_register(opp, "-gpgpu_shader_registers", OPT_UINT32,
                         &gpgpu_shader_registers,
                         "Number of registers per shader core. Limits number "
                         "of concurrent CTAs. (default 8192)",
                         "8192");
  option_parser_register(
      opp, "-gpgpu_registers_per_block", OPT_UINT32, &gpgpu_registers_per_block,
      "Maximum number of registers per CTA. (default 8192)", "8192");
  option_parser_register(opp, "-gpgpu_ignore_resources_limitation", OPT_BOOL,
                         &gpgpu_ignore_resources_limitation,
                         "gpgpu_ignore_resources_limitation (default 0)", "0");
  option_parser_register(
      opp, "-gpgpu_shader_cta", OPT_UINT32, &max_cta_per_core,
      "Maximum number of concurrent CTAs in shader (default 32)", "32");
  option_parser_register(
      opp, "-gpgpu_num_cta_barriers", OPT_UINT32, &max_barriers_per_cta,
      "Maximum number of named barriers per CTA (default 16)", "16");
  option_parser_register(opp, "-gpgpu_n_clusters", OPT_UINT32, &n_simt_clusters,
                         "number of processing clusters", "10");
  option_parser_register(opp, "-gpgpu_n_cores_per_cluster", OPT_UINT32,
                         &n_simt_cores_per_cluster,
                         "number of simd cores per cluster", "3");
  option_parser_register(opp, "-gpgpu_n_cluster_ejection_buffer_size",
                         OPT_UINT32, &n_simt_ejection_buffer_size,
                         "number of packets in ejection buffer", "8");
  option_parser_register(
      opp, "-gpgpu_n_ldst_response_buffer_size", OPT_UINT32,
      &ldst_unit_response_queue_size,
      "number of response packets in ld/st unit ejection buffer", "2");
  option_parser_register(
      opp, "-gpgpu_shmem_per_block", OPT_UINT32, &gpgpu_shmem_per_block,
      "Size of shared memory per thread block or CTA (default 48kB)", "49152");
  option_parser_register(
      opp, "-gpgpu_shmem_size", OPT_UINT32, &gpgpu_shmem_size,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(opp, "-gpgpu_shmem_option", OPT_CSTR,
                         &gpgpu_shmem_option,
                         "Option list of shared memory sizes", "0");
  option_parser_register(
      opp, "-gpgpu_unified_l1d_size", OPT_UINT32,
      &m_L1D_config.m_unified_cache_size,
      "Size of unified data cache(L1D + shared memory) in KB", "0");
  option_parser_register(opp, "-gpgpu_adaptive_cache_config", OPT_BOOL,
                         &adaptive_cache_config, "adaptive_cache_config", "0");
  option_parser_register(
      opp, "-gpgpu_shmem_sizeDefault", OPT_UINT32, &gpgpu_shmem_sizeDefault,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(
      opp, "-gpgpu_shmem_size_PrefL1", OPT_UINT32, &gpgpu_shmem_sizePrefL1,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(opp, "-gpgpu_shmem_size_PrefShared", OPT_UINT32,
                         &gpgpu_shmem_sizePrefShared,
                         "Size of shared memory per shader core (default 16kB)",
                         "16384");
  option_parser_register(
      opp, "-gpgpu_shmem_num_banks", OPT_UINT32, &num_shmem_bank,
      "Number of banks in the shared memory in each shader core (default 16)",
      "16");
  option_parser_register(
      opp, "-gpgpu_shmem_limited_broadcast", OPT_BOOL, &shmem_limited_broadcast,
      "Limit shared memory to do one broadcast per cycle (default on)", "1");
  option_parser_register(opp, "-gpgpu_shmem_warp_parts", OPT_INT32,
                         &mem_warp_parts,
                         "Number of portions a warp is divided into for shared "
                         "memory bank conflict check ",
                         "2");
  option_parser_register(
      opp, "-gpgpu_mem_unit_ports", OPT_INT32, &mem_unit_ports,
      "The number of memory transactions allowed per core cycle", "1");
  option_parser_register(opp, "-gpgpu_shmem_warp_parts", OPT_INT32,
                         &mem_warp_parts,
                         "Number of portions a warp is divided into for shared "
                         "memory bank conflict check ",
                         "2");
  option_parser_register(
      opp, "-gpgpu_warpdistro_shader", OPT_INT32, &gpgpu_warpdistro_shader,
      "Specify which shader core to collect the warp size distribution from",
      "-1");
  option_parser_register(
      opp, "-gpgpu_warp_issue_shader", OPT_INT32, &gpgpu_warp_issue_shader,
      "Specify which shader core to collect the warp issue distribution from",
      "0");
  option_parser_register(opp, "-gpgpu_local_mem_map", OPT_BOOL,
                         &gpgpu_local_mem_map,
                         "Mapping from local memory space address to simulated "
                         "GPU physical address space (default = enabled)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_reg_banks", OPT_INT32,
                         &gpgpu_num_reg_banks,
                         "Number of register banks (default = 8)", "8");
  option_parser_register(
      opp, "-gpgpu_reg_bank_use_warp_id", OPT_BOOL, &gpgpu_reg_bank_use_warp_id,
      "Use warp ID in mapping registers to banks (default = off)", "0");
  option_parser_register(opp, "-gpgpu_sub_core_model", OPT_BOOL,
                         &sub_core_model,
                         "Sub Core Volta/Pascal model (default = off)", "0");
  option_parser_register(opp, "-gpgpu_enable_specialized_operand_collector",
                         OPT_BOOL, &enable_specialized_operand_collector,
                         "enable_specialized_operand_collector", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_units_sp,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_units_dp,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_units_sfu,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_int",
                         OPT_INT32, &gpgpu_operand_collector_num_units_int,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_tensor_core",
                         OPT_INT32,
                         &gpgpu_operand_collector_num_units_tensor_core,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_units_mem,
                         "number of collector units (default = 2)", "2");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_units_gen,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_sp,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_dp,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_sfu,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_int",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_int,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_operand_collector_num_in_ports_tensor_core", OPT_INT32,
      &gpgpu_operand_collector_num_in_ports_tensor_core,
      "number of collector unit in ports (default = 1)", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_mem,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_gen,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_sp,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_dp,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_sfu,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_int",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_int,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_operand_collector_num_out_ports_tensor_core", OPT_INT32,
      &gpgpu_operand_collector_num_out_ports_tensor_core,
      "number of collector unit in ports (default = 1)", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_mem,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_gen,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_coalesce_arch", OPT_INT32,
                         &gpgpu_coalesce_arch,
                         "Coalescing arch (GT200 = 13, Fermi = 20)", "13");
  option_parser_register(opp, "-gpgpu_num_sched_per_core", OPT_INT32,
                         &gpgpu_num_sched_per_core,
                         "Number of warp schedulers per core", "1");
  option_parser_register(opp, "-gpgpu_max_insn_issue_per_warp", OPT_INT32,
                         &gpgpu_max_insn_issue_per_warp,
                         "Max number of instructions that can be issued per "
                         "warp in one cycle by scheduler (either 1 or 2)",
                         "2");
  option_parser_register(opp, "-gpgpu_dual_issue_diff_exec_units", OPT_BOOL,
                         &gpgpu_dual_issue_diff_exec_units,
                         "should dual issue use two different execution unit "
                         "resources (Default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_simt_core_sim_order", OPT_INT32,
                         &simt_core_sim_order,
                         "Select the simulation order of cores in a cluster "
                         "(0=Fix, 1=Round-Robin)",
                         "1");
  option_parser_register(
      opp, "-gpgpu_pipeline_widths", OPT_CSTR, &pipeline_widths_string,
      "Pipeline widths "
      "ID_OC_SP,ID_OC_DP,ID_OC_INT,ID_OC_SFU,ID_OC_MEM,OC_EX_SP,OC_EX_DP,OC_EX_"
      "INT,OC_EX_SFU,OC_EX_MEM,EX_WB,ID_OC_TENSOR_CORE,OC_EX_TENSOR_CORE",
      "1,1,1,1,1,1,1,1,1,1,1,1,1");
  option_parser_register(opp, "-gpgpu_tensor_core_avail", OPT_UINT32,
                         &gpgpu_tensor_core_avail,
                         "Tensor Core Available (default=0)", "0");
  option_parser_register(opp, "-gpgpu_num_sp_units", OPT_UINT32,
                         &gpgpu_num_sp_units, "Number of SP units (default=1)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_dp_units", OPT_UINT32,
                         &gpgpu_num_dp_units, "Number of DP units (default=0)",
                         "0");
  option_parser_register(opp, "-gpgpu_num_int_units", OPT_UINT32,
                         &gpgpu_num_int_units,
                         "Number of INT units (default=0)", "0");
  option_parser_register(opp, "-gpgpu_num_sfu_units", OPT_UINT32,
                         &gpgpu_num_sfu_units, "Number of SF units (default=1)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_tensor_core_units", OPT_UINT32,
                         &gpgpu_num_tensor_core_units,
                         "Number of tensor_core units (default=1)", "0");
  option_parser_register(
      opp, "-gpgpu_num_mem_units", OPT_UINT32, &gpgpu_num_mem_units,
      "Number if ldst units (default=1) WARNING: not hooked up to anything",
      "1");
  option_parser_register(
      opp, "-gpgpu_scheduler", OPT_CSTR, &gpgpu_scheduler_string,
      "Scheduler configuration: < lrr | gto | two_level_active > "
      "If "
      "two_level_active:<num_active_warps>:<inner_prioritization>:<outer_"
      "prioritization>"
      "For complete list of prioritization values see shader.h enum "
      "scheduler_prioritization_type"
      "Default: gto",
      "gto");

  option_parser_register(
      opp, "-gpgpu_concurrent_kernel_sm", OPT_BOOL, &gpgpu_concurrent_kernel_sm,
      "Support concurrent kernels on a SM (default = disabled)", "0");
  option_parser_register(opp, "-gpgpu_perfect_inst_const_cache", OPT_BOOL,
                         &perfect_inst_const_cache,
                         "perfect inst and const cache mode, so all inst and "
                         "const hits in the cache(default = disabled)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_inst_fetch_throughput", OPT_INT32, &inst_fetch_throughput,
      "the number of fetched intruction per warp each cycle", "1");
  option_parser_register(opp, "-gpgpu_reg_file_port_throughput", OPT_INT32,
                         &reg_file_port_throughput,
                         "the number ports of the register file", "1");

  for (unsigned j = 0; j < SPECIALIZED_UNIT_NUM; ++j) {
    std::stringstream ss;
    ss << "-specialized_unit_" << j + 1;
    option_parser_register(opp, ss.str().c_str(), OPT_CSTR,
                           &specialized_unit_string[j],
                           "specialized unit config"
                           " {<enabled>,<num_units>:<latency>:<initiation>,<ID_"
                           "OC_SPEC>:<OC_EX_SPEC>,<NAME>}",
                           "0,4,4,4,4,BRA");
  }
}

void gpgpu_sim_config::reg_options(option_parser_t opp) {
  gpgpu_functional_sim_config::reg_options(opp);
  m_shader_config.reg_options(opp);
  m_memory_config.reg_options(opp);
  power_config::reg_options(opp);
  option_parser_register(opp, "-gpgpu_max_cycle", OPT_INT64, &gpu_max_cycle_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(opp, "-gpgpu_max_insn", OPT_INT64, &gpu_max_insn_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(opp, "-gpgpu_max_cta", OPT_INT32, &gpu_max_cta_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(opp, "-gpgpu_max_completed_cta", OPT_INT32,
                         &gpu_max_completed_cta_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(
      opp, "-gpgpu_runtime_stat", OPT_CSTR, &gpgpu_runtime_stat,
      "display runtime statistics such as dram utilization {<freq>:<flag>}",
      "10000:0");
  option_parser_register(opp, "-liveness_message_freq", OPT_INT64,
                         &liveness_message_freq,
                         "Minimum number of seconds between simulation "
                         "liveness messages (0 = always print)",
                         "1");
  option_parser_register(opp, "-gpgpu_compute_capability_major", OPT_UINT32,
                         &gpgpu_compute_capability_major,
                         "Major compute capability version number", "7");
  option_parser_register(opp, "-gpgpu_compute_capability_minor", OPT_UINT32,
                         &gpgpu_compute_capability_minor,
                         "Minor compute capability version number", "0");
  option_parser_register(opp, "-gpgpu_flush_l1_cache", OPT_BOOL,
                         &gpgpu_flush_l1_cache,
                         "Flush L1 cache at the end of each kernel call", "0");
  option_parser_register(opp, "-gpgpu_flush_l2_cache", OPT_BOOL,
                         &gpgpu_flush_l2_cache,
                         "Flush L2 cache at the end of each kernel call", "0");
  option_parser_register(
      opp, "-gpgpu_deadlock_detect", OPT_BOOL, &gpu_deadlock_detect,
      "Stop the simulation at deadlock (1=on (default), 0=off)", "1");
  option_parser_register(
      opp, "-gpgpu_ptx_instruction_classification", OPT_INT32,
      &(gpgpu_ctx->func_sim->gpgpu_ptx_instruction_classification),
      "if enabled will classify ptx instruction types per kernel (Max 255 "
      "kernels now)",
      "0");
  option_parser_register(
      opp, "-gpgpu_ptx_sim_mode", OPT_INT32,
      &(gpgpu_ctx->func_sim->g_ptx_sim_mode),
      "Select between Performance (default) or Functional simulation (1)", "0");
  option_parser_register(opp, "-gpgpu_clock_domains", OPT_CSTR,
                         &gpgpu_clock_domains,
                         "Clock Domain Frequencies in MhZ {<Core Clock>:<ICNT "
                         "Clock>:<L2 Clock>:<DRAM Clock>}",
                         "500.0:2000.0:2000.0:2000.0");
  option_parser_register(
      opp, "-gpgpu_max_concurrent_kernel", OPT_INT32, &max_concurrent_kernel,
      "maximum kernels that can run concurrently on GPU, set this value "
      "according to max resident grids for your compute capability",
      "32");
  option_parser_register(
      opp, "-gpgpu_cflog_interval", OPT_INT32, &gpgpu_cflog_interval,
      "Interval between each snapshot in control flow logger", "0");
  option_parser_register(opp, "-visualizer_enabled", OPT_BOOL,
                         &g_visualizer_enabled,
                         "Turn on visualizer output (1=On, 0=Off)", "1");
  option_parser_register(opp, "-visualizer_outputfile", OPT_CSTR,
                         &g_visualizer_filename,
                         "Specifies the output log file for visualizer", NULL);
  option_parser_register(
      opp, "-visualizer_zlevel", OPT_INT32, &g_visualizer_zlevel,
      "Compression level of the visualizer output log (0=no comp, 9=highest)",
      "6");
  option_parser_register(opp, "-gpgpu_stack_size_limit", OPT_INT32,
                         &stack_size_limit, "GPU thread stack size", "1024");
  option_parser_register(opp, "-gpgpu_heap_size_limit", OPT_INT32,
                         &heap_size_limit, "GPU malloc heap size ", "8388608");
  option_parser_register(opp, "-gpgpu_runtime_sync_depth_limit", OPT_INT32,
                         &runtime_sync_depth_limit,
                         "GPU device runtime synchronize depth", "2");
  option_parser_register(opp, "-gpgpu_runtime_pending_launch_count_limit",
                         OPT_INT32, &runtime_pending_launch_count_limit,
                         "GPU device runtime pending launch count", "2048");
  option_parser_register(opp, "-trace_enabled", OPT_BOOL, &Trace::enabled,
                         "Turn on traces", "0");
  option_parser_register(opp, "-trace_components", OPT_CSTR, &Trace::config_str,
                         "comma seperated list of traces to enable. "
                         "Complete list found in trace_streams.tup. "
                         "Default none",
                         "none");
  option_parser_register(
      opp, "-trace_sampling_core", OPT_INT32, &Trace::sampling_core,
      "The core which is printed using CORE_DPRINTF. Default 0", "0");
  option_parser_register(opp, "-trace_sampling_memory_partition", OPT_INT32,
                         &Trace::sampling_memory_partition,
                         "The memory partition which is printed using "
                         "MEMPART_DPRINTF. Default -1 (i.e. all)",
                         "-1");
  gpgpu_ctx->stats->ptx_file_line_stats_options(opp);

  // Jin: kernel launch latency
  option_parser_register(opp, "-gpgpu_kernel_launch_latency", OPT_INT32,
                         &(gpgpu_ctx->device_runtime->g_kernel_launch_latency),
                         "Kernel launch latency in cycles. Default: 0", "0");
  option_parser_register(opp, "-gpgpu_cdp_enabled", OPT_BOOL,
                         &(gpgpu_ctx->device_runtime->g_cdp_enabled),
                         "Turn on CDP", "0");

  option_parser_register(opp, "-gpgpu_TB_launch_latency", OPT_INT32,
                         &(gpgpu_ctx->device_runtime->g_TB_launch_latency),
                         "thread block launch latency in cycles. Default: 0",
                         "0");
}

/////////////////////////////////////////////////////////////////////////////

void increment_x_then_y_then_z(dim3 &i, const dim3 &bound) {
  i.x++;
  if (i.x >= bound.x) {
    i.x = 0;
    i.y++;
    if (i.y >= bound.y) {
      i.y = 0;
      if (i.z < bound.z) i.z++;
    }
  }
}


// ------------------------------
// Internal helpers (NEW)
// ------------------------------
kernel_stats_accum_t &gpgpu_sim::kernel_stats_mut_(unsigned kernel_uid) {
  return m_kernel_stats_[kernel_uid]; // default-construct if missing
}

const kernel_stats_accum_t *gpgpu_sim::kernel_stats_find_(unsigned kernel_uid) const {
  auto it = m_kernel_stats_.find(kernel_uid);
  if (it == m_kernel_stats_.end()) return nullptr;
  return &it->second;
}

// ------------------------------
// Event recorders (NEW)
// ------------------------------
void gpgpu_sim::record_kernel_cycle(unsigned kernel_uid, unsigned long long ncycles) {
  kernel_stats_mut_(kernel_uid).sim_cycle += ncycles;
}

void gpgpu_sim::record_kernel_inst(unsigned kernel_uid, unsigned long long ninst) {
  kernel_stats_mut_(kernel_uid).sim_insn += ninst;
}

void gpgpu_sim::record_kernel_issued_cta(unsigned kernel_uid, unsigned long long ncta) {
  kernel_stats_mut_(kernel_uid).issued_cta += ncta;
}

void gpgpu_sim::record_kernel_stall_dramfull(unsigned kernel_uid, unsigned long long ncycles) {
  kernel_stats_mut_(kernel_uid).stall_dramfull += ncycles;
}

void gpgpu_sim::record_kernel_stall_icnt2sh(unsigned kernel_uid, unsigned long long ncycles) {
  kernel_stats_mut_(kernel_uid).stall_icnt2sh += ncycles;
}

void gpgpu_sim::record_kernel_l2_request(unsigned kernel_uid, unsigned long long nreq) {
  kernel_stats_mut_(kernel_uid).l2_reqs += nreq;
}

void gpgpu_sim::record_kernel_l2_bytes(unsigned kernel_uid, unsigned long long nbytes) {
  kernel_stats_mut_(kernel_uid).l2_bytes += nbytes;
}

void gpgpu_sim::record_kernel_dram_request(unsigned kernel_uid, unsigned long long nreq) {
  kernel_stats_mut_(kernel_uid).dram_reqs += nreq;
}

void gpgpu_sim::record_kernel_dram_bytes(unsigned kernel_uid, unsigned long long nbytes) {
  kernel_stats_mut_(kernel_uid).dram_bytes += nbytes;
}



////
// ---- per-kernel L1 helpers -----------------------------------------
using l1_kind_t = gpgpu_sim::l1_kind_t;

static inline void bump_cache_cnt(cache_cnt_t &c, cache_request_status st) {
  switch (st) {
    case RESERVATION_FAIL:
      c.resfail++;
      return; 

    case HIT:
      c.access++;
      return;

    case MISS:
    case SECTOR_MISS:
      c.access++;
      c.miss++;
      return;

    case HIT_RESERVED:
    case MSHR_HIT:
      c.access++;
      c.pending_hit++;
      return;

    default:
      c.access++; // safest fallback
      return;
  }
}

static inline cache_cnt_t &tot_for(kernel_stats_accum_t &a, l1_kind_t k) {
  switch (k) {
    case l1_kind_t::L1I: return a.l1i;
    case l1_kind_t::L1D: return a.l1d;
    case l1_kind_t::L1C: return a.l1c;
    case l1_kind_t::L1T: return a.l1t;
  }
  return a.l1d; // unreachable fallback
}

static inline std::vector<cache_cnt_t> &vec_for(kernel_stats_accum_t &a, l1_kind_t k) {
  switch (k) {
    case l1_kind_t::L1I: return a.l1i_sm;
    case l1_kind_t::L1D: return a.l1d_sm;
    case l1_kind_t::L1C: return a.l1c_sm;
    case l1_kind_t::L1T: return a.l1t_sm;
  }
  return a.l1d_sm; // unreachable fallback
}


///


void gpgpu_sim::record_kernel_l1_access(unsigned kernel_uid,
                                       unsigned smid,
                                       l1_kind_t kind,
                                       cache_request_status status) {
  if (!kernel_uid) return;

  auto &a = kernel_stats_mut_(kernel_uid);

  auto &v = vec_for(a, kind);
  if (v.empty()) v.resize(m_shader_config->num_shader());
  if (smid >= v.size()) return;

  bump_cache_cnt(tot_for(a, kind), status);
  bump_cache_cnt(v[smid], status);
}


// ------------------------------
// View + clear (NEW)
// ------------------------------
kernel_stats_view_t gpgpu_sim::make_kernel_stats_view(unsigned kernel_uid) const {
  kernel_stats_view_t v{};
  const kernel_stats_accum_t *a = kernel_stats_find_(kernel_uid);
  if (!a) return v;
  
  v.gpu_sim_cycle = a->sim_cycle;
  v.gpu_sim_insn  = a->sim_insn;
  v.gpu_ipc       = (a->sim_cycle ? (double)a->sim_insn / (double)a->sim_cycle : 0.0);

  // In per-kernel view, "tot" == the kernel itself
  v.gpu_tot_sim_cycle  = v.gpu_sim_cycle;
  v.gpu_tot_sim_insn   = v.gpu_sim_insn;
  v.gpu_tot_ipc        = v.gpu_ipc;
  v.gpu_tot_issued_cta = a->issued_cta;

  v.gpu_stall_dramfull = a->stall_dramfull;
  v.gpu_stall_icnt2sh  = a->stall_icnt2sh;

  // Bandwidth from bytes + kernel-attributed cycles
  // core_freq is private in config but gpgpu_sim is a friend, so we can use it.
  const double core_freq_hz = m_config.core_freq; 
  // seconds = cycles * seconds_per_cycle
  const double secs = (a->sim_cycle ? ((double)a->sim_cycle * m_config.core_period) : 0.0);

  if (secs > 0.0) {
    const double bytes = (double)a->l2_reqs * 32.0;

    const double bw = bytes / secs / 1e9; // GB/s
    v.L2_BW       = bw;
    v.L2_BW_total = bw;

    // v.L2_BW       = ((double)a->l2_bytes / secs) / 1e9; // GB/s
    // v.L2_BW_total = v.L2_BW;
  }


  // initialize the for l1 stats
  v.l1i = a->l1i; v.l1d = a->l1d; v.l1c = a->l1c; v.l1t = a->l1t;
  v.l1i_sm = a->l1i_sm; v.l1d_sm = a->l1d_sm; v.l1c_sm = a->l1c_sm; v.l1t_sm = a->l1t_sm;

  // initialize l1 ports
  auto it = m_l1d_ports_rec.find(kernel_uid);
  if (it != m_l1d_ports_rec.end()) {
    const auto &rec = it->second;
    if (rec.have_begin && rec.have_end) {
      const unsigned n = std::min(rec.begin_per_cluster.size(),
                                  rec.end_per_cluster.size());

      unsigned long long avail=0, data=0, fill=0;

      for (unsigned c = 0; c < n; ++c) {
        if (!rec.allowed_sm.empty()) {
          if (c >= rec.allowed_sm.size()) continue;
          if (!rec.allowed_sm[c]) continue;     // cluster==SM id
        }

        const auto &b = rec.begin_per_cluster[c];
        const auto &e = rec.end_per_cluster[c];

        if (e.avail >= b.avail) avail += (e.avail - b.avail);
        if (e.data  >= b.data ) data  += (e.data  - b.data );
        if (e.fill  >= b.fill ) fill  += (e.fill  - b.fill );
      }

      v.l1d_port_available_cycles = avail;
      v.l1d_data_port_busy_cycles = data;
      v.l1d_fill_port_busy_cycles = fill;
    }
  }
  auto sit = m_sched_issue_rec_.find(kernel_uid);
  if (sit != m_sched_issue_rec_.end()) {
    v.scheduler_sampling_core = sit->second.sampling_core;
    v.warp_slot_issue_distro  = sit->second.distro;
  }
  v.ctas_completed_for_kernel = a->issued_cta;   
  v.gpgpu_n_tot_thrd_icount = a->thrd_icount;
  v.gpgpu_n_tot_w_icount    = a->warp_icount;

  v.gpgpu_n_load_insn  = a->n_load_insn;
  v.gpgpu_n_store_insn = a->n_store_insn;

  v.gpgpu_n_mem_read_local   = a->n_mem_read_local;
  v.gpgpu_n_mem_write_local  = a->n_mem_write_local;
  v.gpgpu_n_mem_read_global  = a->n_mem_read_global;
  v.gpgpu_n_mem_write_global = a->n_mem_write_global;
  v.gpgpu_n_mem_texture      = a->n_mem_texture;
  v.gpgpu_n_mem_const        = a->n_mem_const;

  v.gpgpu_n_shmem_insn       = a->n_shmem_insn;
  v.gpgpu_n_sstarr_insn      = a->n_sstarr_insn;
  v.gpgpu_n_tex_insn         = a->n_tex_insn;
  v.gpgpu_n_const_mem_insn   = a->n_const_insn;
  v.gpgpu_n_param_mem_insn   = a->n_param_insn;

  v.shader_cycle_distro = a->shader_cycle_distro;
  v.single_issue_nums   = a->single_issue_nums;
  v.dual_issue_nums     = a->dual_issue_nums;

  v.gpgpu_n_shmem_bkconflict       = a->gpgpu_n_shmem_bkconflict;
  v.gpgpu_n_l1cache_bkconflict     = a->gpgpu_n_l1cache_bkconflict;
  v.gpgpu_n_intrawarp_mshr_merge   = a->gpgpu_n_intrawarp_mshr_merge;
  v.gpgpu_n_cmem_portconflict      = a->gpgpu_n_cmem_portconflict;
  v.gpu_reg_bank_conflict_stalls   = a->gpu_reg_bank_conflict_stalls;

  v.gpgpu_n_stall_shd_mem = a->gpgpu_n_stall_shd_mem;

  v.gpgpu_stall_shd_mem_cmem_resource_stall   = a->gpgpu_stall_shd_mem_cmem_resource_stall;
  v.gpgpu_stall_shd_mem_smem_bk_conf          = a->gpgpu_stall_shd_mem_smem_bk_conf;
  v.gpgpu_stall_shd_mem_glmem_resource_stall  = a->gpgpu_stall_shd_mem_glmem_resource_stall;
  v.gpgpu_stall_shd_mem_glmem_coal_stall      = a->gpgpu_stall_shd_mem_glmem_coal_stall;
  v.gpgpu_stall_shd_mem_glmem_data_port_stall = a->gpgpu_stall_shd_mem_glmem_data_port_stall;  

  v.outgoing_traffic = &a->outgoing_traffic;
  v.incoming_traffic = &a->incoming_traffic;

  v.memlat_num_mfs           = a->memlat_num_mfs;
  v.memlat_mf_total_lat      = a->memlat_mf_total_lat;
  v.memlat_tot_icnt2mem_lat  = a->memlat_tot_icnt2mem_lat;
  v.memlat_tot_icnt2sh_lat   = a->memlat_tot_icnt2sh_lat;

  v.memlat_max_mf_lat        = a->memlat_max_mf_lat;
  v.memlat_max_icnt2mem_lat  = a->memlat_max_icnt2mem_lat;
  v.memlat_max_mrq_lat       = a->memlat_max_mrq_lat;
  v.memlat_max_icnt2sh_lat   = a->memlat_max_icnt2sh_lat;

  for (int i = 0; i < 32; i++) {
    v.memlat_mrq_lat_table[i]   = a->memlat_mrq_lat_table[i];
    v.memlat_dq_lat_table[i]    = a->memlat_dq_lat_table[i];
    v.memlat_mf_lat_table[i]    = a->memlat_mf_lat_table[i];
    v.memlat_mf_lat_pw_table[i] = a->memlat_mf_lat_pw_table[i];
  }
  for (int i = 0; i < 24; i++) {
    v.memlat_icnt2mem_lat_table[i] = a->memlat_icnt2mem_lat_table[i];
    v.memlat_icnt2sh_lat_table[i]  = a->memlat_icnt2sh_lat_table[i];
  }

  v.memlat_rowstats_n_mem = m_memory_config->m_n_mem;
  v.memlat_rowstats_n_bk  = m_memory_config->nbk;

  v.memlat_max_conc_access2samerow =
      a->memlat_max_conc_access2samerow.empty()
          ? nullptr
          : a->memlat_max_conc_access2samerow.data();

  v.memlat_max_servicetime2samerow =a->memlat_max_servicetime2samerow.empty()
          ? nullptr
          : a->memlat_max_servicetime2samerow.data();

  v.memlat_row_access = a->memlat_row_access.empty() ? nullptr : a->memlat_row_access.data();
  v.memlat_num_activates = a->memlat_num_activates.empty() ? nullptr : a->memlat_num_activates.data();
  v.memlat_totalbankreads =a->memlat_totalbankreads.empty() ? nullptr: a->memlat_totalbankreads.data();
  v.memlat_totalbankwrites =a->memlat_totalbankwrites.empty() ? nullptr: a->memlat_totalbankwrites.data();  
  v.totalbankaccesses =a->totalbankaccesses.empty() ? nullptr: a->totalbankaccesses.data();
  v.memlat_mf_total_laten = a->memlat_mf_total_laten.empty() ? nullptr : a->memlat_mf_total_laten.data();
  v.memlat_max_mf_laten = a->memlat_max_mf_laten.empty() ? nullptr : a->memlat_max_mf_laten.data();
  

  //DRAM

  v.dram_n_cmd      = a->dram_n_cmd.empty()      ? nullptr : a->dram_n_cmd.data();
  v.dram_n_nop      = a->dram_n_nop.empty()      ? nullptr : a->dram_n_nop.data();
  v.dram_n_activity = a->dram_n_activity.empty() ? nullptr : a->dram_n_activity.data();
  v.dram_n_act      = a->dram_n_act.empty()      ? nullptr : a->dram_n_act.data();
  v.dram_n_pre      = a->dram_n_pre.empty()      ? nullptr : a->dram_n_pre.data();
  v.dram_n_req       = a->dram_n_req.empty()       ? nullptr : a->dram_n_req.data();
  v.dram_n_ref_event = a->dram_n_ref_event.empty() ? nullptr : a->dram_n_ref_event.data();
  v.dram_n_rd      = a->dram_n_rd.empty()      ? nullptr : a->dram_n_rd.data();
  v.dram_n_rd_L2_A = a->dram_n_rd_L2_A.empty() ? nullptr : a->dram_n_rd_L2_A.data();
  v.dram_n_wr      = a->dram_n_wr.empty()      ? nullptr : a->dram_n_wr.data();
  v.dram_n_wr_WB   = a->dram_n_wr_WB.empty()   ? nullptr : a->dram_n_wr_WB.data();
  v.dram_bwutil = a->dram_bwutil.empty() ? nullptr : a->dram_bwutil.data();
  v.dram_bk_n_access = a->dram_bk_n_access.empty() ? nullptr : a->dram_bk_n_access.data();
  v.dram_bk_n_idle   = a->dram_bk_n_idle.empty()   ? nullptr : a->dram_bk_n_idle.data();


  // BLP stats
  v.dram_banks_1time             = a->dram_banks_1time.empty()             ? nullptr : a->dram_banks_1time.data();
  v.dram_banks_access_total      = a->dram_banks_access_total.empty()      ? nullptr : a->dram_banks_access_total.data();

  v.dram_banks_time_rw           = a->dram_banks_time_rw.empty()           ? nullptr : a->dram_banks_time_rw.data();
  v.dram_banks_access_rw_total   = a->dram_banks_access_rw_total.empty()   ? nullptr : a->dram_banks_access_rw_total.data();

  v.dram_banks_time_ready        = a->dram_banks_time_ready.empty()        ? nullptr : a->dram_banks_time_ready.data();
  v.dram_banks_access_ready_total= a->dram_banks_access_ready_total.empty()? nullptr : a->dram_banks_access_ready_total.data();

  v.dram_w2r_ratio_sum_1e6       = a->dram_w2r_ratio_sum_1e6.empty()       ? nullptr : a->dram_w2r_ratio_sum_1e6.data();
  v.dram_bkgrp_parallsim_rw      = a->dram_bkgrp_parallsim_rw.empty()      ? nullptr : a->dram_bkgrp_parallsim_rw.data();

  v.dram_access_num = a->dram_access_num.empty() ? nullptr : a->dram_access_num.data();
  v.dram_hits_num = a->dram_hits_num.empty() ? nullptr : a->dram_hits_num.data();
  v.dram_read_num = a->dram_read_num.empty() ? nullptr : a->dram_read_num.data();
  v.dram_write_num = a->dram_write_num.empty() ? nullptr : a->dram_write_num.data();
  v.dram_hits_write_num = a->dram_hits_write_num.empty() ? nullptr : a->dram_hits_write_num.data();
  v.dram_hits_read_num = a->dram_hits_read_num.empty() ? nullptr : a->dram_hits_read_num.data();
  

  v.dram_util_bw       = a->dram_util_bw.empty()       ? nullptr : a->dram_util_bw.data();
  v.dram_wasted_bw_col = a->dram_wasted_bw_col.empty() ? nullptr : a->dram_wasted_bw_col.data();
  v.dram_wasted_bw_row = a->dram_wasted_bw_row.empty() ? nullptr : a->dram_wasted_bw_row.data();
  v.dram_idle_bw       = a->dram_idle_bw.empty()       ? nullptr : a->dram_idle_bw.data();

  v.dram_RCDc_limit        = a->dram_RCDc_limit.empty()        ? nullptr : a->dram_RCDc_limit.data();
  v.dram_RCDWRc_limit      = a->dram_RCDWRc_limit.empty()      ? nullptr : a->dram_RCDWRc_limit.data();
  v.dram_WTRc_limit        = a->dram_WTRc_limit.empty()        ? nullptr : a->dram_WTRc_limit.data();
  v.dram_RTWc_limit        = a->dram_RTWc_limit.empty()        ? nullptr : a->dram_RTWc_limit.data();
  v.dram_CCDLc_limit       = a->dram_CCDLc_limit.empty()       ? nullptr : a->dram_CCDLc_limit.data();
  v.dram_rwq_limit         = a->dram_rwq_limit.empty()         ? nullptr : a->dram_rwq_limit.data();
  v.dram_CCDLc_limit_alone = a->dram_CCDLc_limit_alone.empty() ? nullptr : a->dram_CCDLc_limit_alone.data();
  v.dram_WTRc_limit_alone  = a->dram_WTRc_limit_alone.empty()  ? nullptr : a->dram_WTRc_limit_alone.data();
  v.dram_RTWc_limit_alone  = a->dram_RTWc_limit_alone.empty()  ? nullptr : a->dram_RTWc_limit_alone.data();

  v.dram_issued_total_row = a->dram_issued_total_row.empty() ? nullptr : a->dram_issued_total_row.data();
  v.dram_issued_total_col = a->dram_issued_total_col.empty() ? nullptr : a->dram_issued_total_col.data();
  v.dram_issued_total     = a->dram_issued_total.empty()     ? nullptr : a->dram_issued_total.data();
  v.dram_issued_two       = a->dram_issued_two.empty()       ? nullptr : a->dram_issued_two.data();
  v.dram_ave_mrqs_sum     = a->dram_ave_mrqs_sum.empty()     ? nullptr : a->dram_ave_mrqs_sum.data();

  v.dram_max_mrqs   = a->dram_max_mrqs.empty()   ? nullptr : a->dram_max_mrqs.data();
  v.dram_util_bins  = a->dram_util_bins.empty() ? nullptr : a->dram_util_bins.data();
  v.dram_eff_bins   = a->dram_eff_bins.empty()  ? nullptr : a->dram_eff_bins.data();
  
  return v;
}

void gpgpu_sim::clear_kernel_stats(unsigned kernel_uid) {
  m_kernel_stats_.erase(kernel_uid);
  m_l1d_ports_rec.erase(kernel_uid);

  // 
  kernel_inst_count.erase(kernel_uid);
  kernel_mem_reply_bytes.erase(kernel_uid);
  kernel_mem_concurrency_sum.erase(kernel_uid);
  kernel_mem_busy_cycles.erase(kernel_uid);
  kernel_cta_count.erase(kernel_uid);
  kernel_threads_per_cta.erase(kernel_uid);
  kernel_stall_dramfull.erase(kernel_uid);
  kernel_stall_icnt2sh.erase(kernel_uid);
  //
  m_sched_issue_rec_.erase(kernel_uid);
}








void gpgpu_sim::launch(kernel_info_t *kinfo) {
  unsigned kernelID = kinfo->get_uid();
  unsigned long long streamID = kinfo->get_streamID();
  (void)kernel_stats_mut_(kernelID);

  kernel_time_t kernel_time = {gpu_tot_sim_cycle + gpu_sim_cycle, 0};
  if (gpu_kernel_time.find(streamID) == gpu_kernel_time.end()) {
    std::map<unsigned, kernel_time_t> new_val;
    new_val.insert(std::pair<unsigned, kernel_time_t>(kernelID, kernel_time));
    gpu_kernel_time.insert(
        std::pair<unsigned long long, std::map<unsigned, kernel_time_t>>(
            streamID, new_val));
  } else {
    gpu_kernel_time.at(streamID).insert(
        std::pair<unsigned, kernel_time_t>(kernelID, kernel_time));
    ////////// assume same kernel ID do not appear more than once
  }

  unsigned cta_size = kinfo->threads_per_cta();
  if (cta_size > m_shader_config->n_thread_per_shader) {
    printf(
        "Execution error: Shader kernel CTA (block) size is too large for "
        "microarch config.\n");
    printf("                 CTA size (x*y*z) = %u, max supported = %u\n",
           cta_size, m_shader_config->n_thread_per_shader);
    printf(
        "                 => either change -gpgpu_shader argument in "
        "gpgpusim.config file or\n");
    printf(
        "                 modify the CUDA source to decrease the kernel block "
        "size.\n");
    abort();
  }
  unsigned n = 0;
  for (n = 0; n < m_running_kernels.size(); n++) {
    if ((NULL == m_running_kernels[n]) || m_running_kernels[n]->done()) {
      m_running_kernels[n] = kinfo;
      break;
    }
  }
  assert(n < m_running_kernels.size());
}

bool gpgpu_sim::can_start_kernel() {
  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if ((NULL == m_running_kernels[n]) || m_running_kernels[n]->done())
      return true;
  }
  return false;
}

bool gpgpu_sim::hit_max_cta_count() const {
  if (m_config.gpu_max_cta_opt != 0) {
    if ((gpu_tot_issued_cta + m_total_cta_launched) >= m_config.gpu_max_cta_opt)
      return true;
  }
  return false;
}

bool gpgpu_sim::kernel_more_cta_left(kernel_info_t *kernel) const {
  if (hit_max_cta_count()) return false;

  if (kernel && !kernel->no_more_ctas_to_run()) return true;

  return false;
}

bool gpgpu_sim::get_more_cta_left() const {
  if (hit_max_cta_count()) return false;

  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if (m_running_kernels[n] && !m_running_kernels[n]->no_more_ctas_to_run())
      return true;
  }
  return false;
}

void gpgpu_sim::decrement_kernel_latency() {
  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if (m_running_kernels[n] && m_running_kernels[n]->m_kernel_TB_latency)
      m_running_kernels[n]->m_kernel_TB_latency--;
  }
}

kernel_info_t *gpgpu_sim::select_kernel() {
  if (m_running_kernels[m_last_issued_kernel] &&
      !m_running_kernels[m_last_issued_kernel]->no_more_ctas_to_run() &&
      !m_running_kernels[m_last_issued_kernel]->m_kernel_TB_latency) {
    unsigned launch_uid = m_running_kernels[m_last_issued_kernel]->get_uid();
    if (std::find(m_executed_kernel_uids.begin(), m_executed_kernel_uids.end(),
                  launch_uid) == m_executed_kernel_uids.end()) {
      m_running_kernels[m_last_issued_kernel]->start_cycle =
          gpu_sim_cycle + gpu_tot_sim_cycle;
      m_executed_kernel_uids.push_back(launch_uid);
      m_executed_kernel_names.push_back(
          m_running_kernels[m_last_issued_kernel]->name());
    }
    return m_running_kernels[m_last_issued_kernel];
  }

  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    unsigned idx =
        (n + m_last_issued_kernel + 1) % m_config.max_concurrent_kernel;
    if (kernel_more_cta_left(m_running_kernels[idx]) &&
        !m_running_kernels[idx]->m_kernel_TB_latency) {
      m_last_issued_kernel = idx;
      m_running_kernels[idx]->start_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
      // record this kernel for stat print if it is the first time this kernel
      // is selected for execution
      unsigned launch_uid = m_running_kernels[idx]->get_uid();
      assert(std::find(m_executed_kernel_uids.begin(),
                       m_executed_kernel_uids.end(),
                       launch_uid) == m_executed_kernel_uids.end());
      m_executed_kernel_uids.push_back(launch_uid);
      m_executed_kernel_names.push_back(m_running_kernels[idx]->name());

      return m_running_kernels[idx];
    }
  }
  return NULL;
}

unsigned gpgpu_sim::finished_kernel() {
  if (m_finished_kernel.empty()) {
    last_streamID = -1;
    return 0;
  }
  unsigned result = m_finished_kernel.front();
  m_finished_kernel.pop_front();
  return result;
}

void gpgpu_sim::set_kernel_done(kernel_info_t *kernel) {
  unsigned uid = kernel->get_uid();
  last_uid = uid;
  unsigned long long streamID = kernel->get_streamID();
  last_streamID = streamID;
  gpu_kernel_time.at(streamID).at(uid).end_cycle =
      gpu_tot_sim_cycle + gpu_sim_cycle;
  m_finished_kernel.push_back(uid);
  std::vector<kernel_info_t *>::iterator k;
  for (k = m_running_kernels.begin(); k != m_running_kernels.end(); k++) {
    if (*k == kernel) {
      kernel->end_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
      *k = NULL;
      break;
    }
  }
  assert(k != m_running_kernels.end());
}

void gpgpu_sim::stop_all_running_kernels() {
  std::vector<kernel_info_t *>::iterator k;
  for (k = m_running_kernels.begin(); k != m_running_kernels.end(); ++k) {
    if (*k != NULL) {       // If a kernel is active
      set_kernel_done(*k);  // Stop the kernel
      assert(*k == NULL);
    }
  }
}

void exec_gpgpu_sim::createSIMTCluster() {
  m_cluster = new simt_core_cluster *[m_shader_config->n_simt_clusters];
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i] =
        new exec_simt_core_cluster(this, i, m_shader_config, m_memory_config,
                                   m_shader_stats, m_memory_stats);
}

// SST get its own simt_cluster
void sst_gpgpu_sim::createSIMTCluster() {
  m_cluster = new simt_core_cluster *[m_shader_config->n_simt_clusters];
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i] =
        new sst_simt_core_cluster(this, i, m_shader_config, m_memory_config,
                                  m_shader_stats, m_memory_stats);
  SST_gpgpu_reply_buffer.resize(m_shader_config->n_simt_clusters);
}

gpgpu_sim::gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx)
    : gpgpu_t(config, ctx), m_config(config) {
  gpgpu_ctx = ctx;
  m_shader_config = &m_config.m_shader_config;
  m_memory_config = &m_config.m_memory_config;
  const_cast<memory_config*>(m_memory_config)->set_gpu(this);

  ctx->ptx_parser->set_ptx_warp_size(m_shader_config);
  ptx_file_line_stats_create_exposed_latency_tracker(m_config.num_shader());

#ifdef GPGPUSIM_POWER_MODEL
  m_gpgpusim_wrapper = new gpgpu_sim_wrapper(
      config.g_power_simulation_enabled, config.g_power_config_name,
      config.g_power_simulation_mode, config.g_dvfs_enabled);
#endif

  m_shader_stats = new shader_core_stats(m_shader_config);
  m_memory_stats = new memory_stats_t(m_config.num_shader(), m_shader_config,
                                      m_memory_config, this);
  average_pipeline_duty_cycle = (float *)malloc(sizeof(float));
  active_sms = (float *)malloc(sizeof(float));
  m_power_stats =
      new power_stat_t(m_shader_config, average_pipeline_duty_cycle, active_sms,
                       m_shader_stats, m_memory_config, m_memory_stats);

  gpu_sim_insn = 0;
  gpu_tot_sim_insn = 0;
  gpu_tot_issued_cta = 0;
  gpu_completed_cta = 0;
  m_total_cta_launched = 0;
  gpu_deadlock = false;

  gpu_stall_dramfull = 0;
  gpu_stall_icnt2sh = 0;
  partiton_reqs_in_parallel = 0;
  partiton_reqs_in_parallel_total = 0;
  partiton_reqs_in_parallel_util = 0;
  partiton_reqs_in_parallel_util_total = 0;
  gpu_sim_cycle_parition_util = 0;
  gpu_tot_sim_cycle_parition_util = 0;
  partiton_replys_in_parallel = 0;
  partiton_replys_in_parallel_total = 0;
  last_streamID = -1;

  gpu_kernel_time.clear();

  // TODO: somehow move this logic to the sst_gpgpu_sim constructor?
  if (!m_config.is_SST_mode()) {
    // Init memory if not in SST mode
    m_memory_partition_unit =
        new memory_partition_unit *[m_memory_config->m_n_mem];
    m_memory_sub_partition =
        new memory_sub_partition *[m_memory_config->m_n_mem_sub_partition];
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      m_memory_partition_unit[i] =
          new memory_partition_unit(i, m_memory_config, m_memory_stats, this);
      for (unsigned p = 0;
           p < m_memory_config->m_n_sub_partition_per_memory_channel; p++) {
        unsigned submpid =
            i * m_memory_config->m_n_sub_partition_per_memory_channel + p;
        m_memory_sub_partition[submpid] =
            m_memory_partition_unit[i]->get_sub_partition(p);
      }
    }

    icnt_wrapper_init();
    icnt_create(m_shader_config->n_simt_clusters,
                m_memory_config->m_n_mem_sub_partition);
  }
  time_vector_create(NUM_MEM_REQ_STAT);
  fprintf(stdout,
          "GPGPU-Sim uArch: performance model initialization complete.\n");

  m_running_kernels.resize(config.max_concurrent_kernel, NULL);
  m_last_issued_kernel = 0;
  m_last_cluster_issue = m_shader_config->n_simt_clusters -
                         1;  // this causes first launch to use simt cluster 0
  *average_pipeline_duty_cycle = 0;
  *active_sms = 0;

  last_liveness_message_time = 0;

  // Jin: functional simulation for CDP
  m_functional_sim = false;
  m_functional_sim_kernel = NULL;
}

void sst_gpgpu_sim::SST_receive_mem_reply(unsigned core_id, void *mem_req) {
  assert(core_id < m_shader_config->n_simt_clusters);
  mem_fetch *mf = (mem_fetch *)mem_req;

  (SST_gpgpu_reply_buffer[core_id]).push_back(mf);
}

mem_fetch *sst_gpgpu_sim::SST_pop_mem_reply(unsigned core_id) {
  if (SST_gpgpu_reply_buffer[core_id].size() > 0) {
    mem_fetch *temp = SST_gpgpu_reply_buffer[core_id].front();
    SST_gpgpu_reply_buffer[core_id].pop_front();
    return temp;
  } else
    return NULL;
}

int gpgpu_sim::shared_mem_size() const {
  return m_shader_config->gpgpu_shmem_size;
}

int gpgpu_sim::shared_mem_per_block() const {
  return m_shader_config->gpgpu_shmem_per_block;
}

int gpgpu_sim::num_registers_per_core() const {
  return m_shader_config->gpgpu_shader_registers;
}

int gpgpu_sim::num_registers_per_block() const {
  return m_shader_config->gpgpu_registers_per_block;
}

int gpgpu_sim::wrp_size() const { return m_shader_config->warp_size; }

int gpgpu_sim::shader_clock() const { return m_config.core_freq / 1000; }

int gpgpu_sim::max_cta_per_core() const {
  return m_shader_config->max_cta_per_core;
}

int gpgpu_sim::get_max_cta(const kernel_info_t &k) const {
  return m_shader_config->max_cta(k);
}

void gpgpu_sim::set_prop(cudaDeviceProp *prop) { m_cuda_properties = prop; }

int gpgpu_sim::compute_capability_major() const {
  return m_config.gpgpu_compute_capability_major;
}

int gpgpu_sim::compute_capability_minor() const {
  return m_config.gpgpu_compute_capability_minor;
}

const struct cudaDeviceProp *gpgpu_sim::get_prop() const {
  return m_cuda_properties;
}

enum divergence_support_t gpgpu_sim::simd_model() const {
  return m_shader_config->model;
}

void gpgpu_sim_config::init_clock_domains(void) {
  sscanf(gpgpu_clock_domains, "%lf:%lf:%lf:%lf", &core_freq, &icnt_freq,
         &l2_freq, &dram_freq);
  core_freq = core_freq MhZ;
  icnt_freq = icnt_freq MhZ;
  l2_freq = l2_freq MhZ;
  dram_freq = dram_freq MhZ;
  core_period = 1 / core_freq;
  icnt_period = 1 / icnt_freq;
  dram_period = 1 / dram_freq;
  l2_period = 1 / l2_freq;
  printf("GPGPU-Sim uArch: clock freqs: %lf:%lf:%lf:%lf\n", core_freq,
         icnt_freq, l2_freq, dram_freq);
  printf("GPGPU-Sim uArch: clock periods: %.20lf:%.20lf:%.20lf:%.20lf\n",
         core_period, icnt_period, l2_period, dram_period);
}

void gpgpu_sim::reinit_clock_domains(void) {
  core_time = 0;
  dram_time = 0;
  icnt_time = 0;
  l2_time = 0;
}

bool gpgpu_sim::active() {
  if (m_config.gpu_max_cycle_opt &&
      (gpu_tot_sim_cycle + gpu_sim_cycle) >= m_config.gpu_max_cycle_opt)
    return false;
  if (m_config.gpu_max_insn_opt &&
      (gpu_tot_sim_insn + gpu_sim_insn) >= m_config.gpu_max_insn_opt)
    return false;
  if (m_config.gpu_max_cta_opt &&
      (gpu_tot_issued_cta >= m_config.gpu_max_cta_opt))
    return false;
  if (m_config.gpu_max_completed_cta_opt &&
      (gpu_completed_cta >= m_config.gpu_max_completed_cta_opt))
    return false;
  if (m_config.gpu_deadlock_detect && gpu_deadlock) return false;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    if (m_cluster[i]->get_not_completed() > 0) return true;
  ;
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
    if (m_memory_partition_unit[i]->busy() > 0) return true;
  ;
  if (icnt_busy()) return true;
  if (get_more_cta_left()) return true;
  return false;
}

bool sst_gpgpu_sim::active() {
  if (m_config.gpu_max_cycle_opt &&
      (gpu_tot_sim_cycle + gpu_sim_cycle) >= m_config.gpu_max_cycle_opt)
    return false;
  if (m_config.gpu_max_insn_opt &&
      (gpu_tot_sim_insn + gpu_sim_insn) >= m_config.gpu_max_insn_opt)
    return false;
  if (m_config.gpu_max_cta_opt &&
      (gpu_tot_issued_cta >= m_config.gpu_max_cta_opt))
    return false;
  if (m_config.gpu_max_completed_cta_opt &&
      (gpu_completed_cta >= m_config.gpu_max_completed_cta_opt))
    return false;
  if (m_config.gpu_deadlock_detect && gpu_deadlock) return false;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    if (m_cluster[i]->get_not_completed() > 0) return true;
  if (get_more_cta_left()) return true;
  return false;
}

void gpgpu_sim::init() {
  // run a CUDA grid on the GPU microarchitecture simulator
  gpu_sim_cycle = 0;
  gpu_sim_insn = 0;
  last_gpu_sim_insn = 0;
  m_total_cta_launched = 0;
  gpu_completed_cta = 0;
  partiton_reqs_in_parallel = 0;
  partiton_replys_in_parallel = 0;
  partiton_reqs_in_parallel_util = 0;
  gpu_sim_cycle_parition_util = 0;

// McPAT initialization function. Called on first launch of GPU
#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    init_mcpat(m_config, m_gpgpusim_wrapper, m_config.gpu_stat_sample_freq,
               gpu_tot_sim_insn, gpu_sim_insn);
  }
#endif

  reinit_clock_domains();
  gpgpu_ctx->func_sim->set_param_gpgpu_num_shaders(m_config.num_shader());
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i]->reinit();
  m_shader_stats->new_grid();
  // initialize the control-flow, memory access, memory latency logger
  if (m_config.g_visualizer_enabled) {
    create_thread_CFlogger(gpgpu_ctx, m_config.num_shader(),
                           m_shader_config->n_thread_per_shader, 0,
                           m_config.gpgpu_cflog_interval);
  }
  shader_CTA_count_create(m_config.num_shader(), m_config.gpgpu_cflog_interval);
  if (m_config.gpgpu_cflog_interval != 0) {
    insn_warp_occ_create(m_config.num_shader(), m_shader_config->warp_size);
    shader_warp_occ_create(m_config.num_shader(), m_shader_config->warp_size,
                           m_config.gpgpu_cflog_interval);
    shader_mem_acc_create(m_config.num_shader(), m_memory_config->m_n_mem, 4,
                          m_config.gpgpu_cflog_interval);
    shader_mem_lat_create(m_config.num_shader(), m_config.gpgpu_cflog_interval);
    shader_cache_access_create(m_config.num_shader(), 3,
                               m_config.gpgpu_cflog_interval);
    set_spill_interval(m_config.gpgpu_cflog_interval * 40);
  }

  if (g_network_mode) icnt_init();
}

void gpgpu_sim::update_stats() {
  m_memory_stats->memlatstat_lat_pw();
  gpu_tot_sim_cycle += gpu_sim_cycle;
  gpu_tot_sim_insn += gpu_sim_insn;
  gpu_tot_issued_cta += m_total_cta_launched;
  partiton_reqs_in_parallel_total += partiton_reqs_in_parallel;
  partiton_replys_in_parallel_total += partiton_replys_in_parallel;
  partiton_reqs_in_parallel_util_total += partiton_reqs_in_parallel_util;
  gpu_tot_sim_cycle_parition_util += gpu_sim_cycle_parition_util;
  gpu_tot_occupancy += gpu_occupancy;

  gpu_sim_cycle = 0;
  partiton_reqs_in_parallel = 0;
  partiton_replys_in_parallel = 0;
  partiton_reqs_in_parallel_util = 0;
  gpu_sim_cycle_parition_util = 0;
  gpu_sim_insn = 0;
  m_total_cta_launched = 0;
  gpu_completed_cta = 0;
  gpu_occupancy = occupancy_stats();
}

PowerscalingCoefficients *gpgpu_sim::get_scaling_coeffs() {
  return m_gpgpusim_wrapper->get_scaling_coeffs();
}

void gpgpu_sim::print_stats(unsigned long long streamID) {
  // Legacy entry point: no per-kernel override, no explicit kernel name/uid.
  gpgpu_ctx->stats->ptx_file_line_stats_write_file();
  gpu_print_stat(streamID, /*view=*/nullptr,
                 /*single_kernel_name=*/nullptr,
                 /*single_kernel_uid=*/-1);

  if (g_network_mode) {
    printf("----------------------------Interconnect-DETAILS------------------------------\n");
    icnt_display_stats();
    icnt_display_overall_stats();
    printf("----------------------------END-of-Interconnect-DETAILS-----------------------\n");
  }
}

void gpgpu_sim::print_stats(unsigned long long streamID,
                            const kernel_stats_view_t *view) {
  // Daemon entry point with override for top counters, but no explicit name/uid.
  gpgpu_ctx->stats->ptx_file_line_stats_write_file();
  gpu_print_stat(streamID, view,
                 /*single_kernel_name=*/nullptr,
                 /*single_kernel_uid=*/-1);

  if (g_network_mode) {
    printf("----------------------------Interconnect-DETAILS------------------------------\n");
    icnt_display_stats();
    icnt_display_overall_stats();
    printf("----------------------------END-of-Interconnect-DETAILS-----------------------\n");
  }
}

void gpgpu_sim::print_stats(unsigned long long streamID,
                            const kernel_stats_view_t *view,
                            const char *single_kernel_name,
                            int single_kernel_uid) {
  // Full daemon path: top counters override + explicit kernel name/uid.
  gpgpu_ctx->stats->ptx_file_line_stats_write_file();
  gpu_print_stat(streamID, view, single_kernel_name, single_kernel_uid);

  if (g_network_mode) {
    printf("----------------------------Interconnect-DETAILS------------------------------\n");
    icnt_display_stats();
    icnt_display_overall_stats();
    printf("----------------------------END-of-Interconnect-DETAILS-----------------------\n");
  }
}



void gpgpu_sim::deadlock_check() {
  if (m_config.gpu_deadlock_detect && gpu_deadlock) {
    fflush(stdout);
    printf(
        "\n\nGPGPU-Sim uArch: ERROR ** deadlock detected: last writeback core "
        "%u @ gpu_sim_cycle %u (+ gpu_tot_sim_cycle %u) (%u cycles ago)\n",
        gpu_sim_insn_last_update_sid, (unsigned)gpu_sim_insn_last_update,
        (unsigned)(gpu_tot_sim_cycle - gpu_sim_cycle),
        (unsigned)(gpu_sim_cycle - gpu_sim_insn_last_update));
    unsigned num_cores = 0;
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      unsigned not_completed = m_cluster[i]->get_not_completed();
      if (not_completed) {
        if (!num_cores) {
          printf(
              "GPGPU-Sim uArch: DEADLOCK  shader cores no longer committing "
              "instructions [core(# threads)]:\n");
          printf("GPGPU-Sim uArch: DEADLOCK  ");
          m_cluster[i]->print_not_completed(stdout);
        } else if (num_cores < 8) {
          m_cluster[i]->print_not_completed(stdout);
        } else if (num_cores >= 8) {
          printf(" + others ... ");
        }
        num_cores += m_shader_config->n_simt_cores_per_cluster;
      }
    }
    printf("\n");
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      bool busy = m_memory_partition_unit[i]->busy();
      if (busy)
        printf("GPGPU-Sim uArch DEADLOCK:  memory partition %u busy\n", i);
    }
    if (icnt_busy()) {
      printf("GPGPU-Sim uArch DEADLOCK:  iterconnect contains traffic\n");
      icnt_display_state(stdout);
    }
    printf(
        "\nRe-run the simulator in gdb and use debug routines in .gdbinit to "
        "debug this\n");
    fflush(stdout);
    abort();
  }
}

/// printing the names and uids of a set of executed kernels (usually there is
/// only one)
std::string gpgpu_sim::executed_kernel_info_string() {
  std::stringstream statout;

  statout << "kernel_name = ";
  for (unsigned int k = 0; k < m_executed_kernel_names.size(); k++) {
    statout << m_executed_kernel_names[k] << " ";
  }
  statout << std::endl;
  statout << "kernel_launch_uid = ";
  for (unsigned int k = 0; k < m_executed_kernel_uids.size(); k++) {
    statout << m_executed_kernel_uids[k] << " ";
  }
  statout << std::endl;

  return statout.str();
}

std::string gpgpu_sim::executed_kernel_name() {
  std::stringstream statout;
  if (m_executed_kernel_names.size() == 1)
    statout << m_executed_kernel_names[0];
  else {
    for (unsigned int k = 0; k < m_executed_kernel_names.size(); k++) {
      statout << m_executed_kernel_names[k] << " ";
    }
  }
  return statout.str();
}
void gpgpu_sim::set_cache_config(std::string kernel_name,
                                 FuncCache cacheConfig) {
  m_special_cache_config[kernel_name] = cacheConfig;
}

FuncCache gpgpu_sim::get_cache_config(std::string kernel_name) {
  for (std::map<std::string, FuncCache>::iterator iter =
           m_special_cache_config.begin();
       iter != m_special_cache_config.end(); iter++) {
    std::string kernel = iter->first;
    if (kernel_name.compare(kernel) == 0) {
      return iter->second;
    }
  }
  return (FuncCache)0;
}

bool gpgpu_sim::has_special_cache_config(std::string kernel_name) {
  for (std::map<std::string, FuncCache>::iterator iter =
           m_special_cache_config.begin();
       iter != m_special_cache_config.end(); iter++) {
    std::string kernel = iter->first;
    if (kernel_name.compare(kernel) == 0) {
      return true;
    }
  }
  return false;
}

void gpgpu_sim::set_cache_config(std::string kernel_name) {
  if (has_special_cache_config(kernel_name)) {
    change_cache_config(get_cache_config(kernel_name));
  } else {
    change_cache_config(FuncCachePreferNone);
  }
}

void gpgpu_sim::change_cache_config(FuncCache cache_config) {
  if (cache_config != m_shader_config->m_L1D_config.get_cache_status()) {
    printf("FLUSH L1 Cache at configuration change between kernels\n");
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      m_cluster[i]->cache_invalidate();
    }
  }

  switch (cache_config) {
    case FuncCachePreferNone:
      m_shader_config->m_L1D_config.init(
          m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
      m_shader_config->gpgpu_shmem_size =
          m_shader_config->gpgpu_shmem_sizeDefault;
      break;
    case FuncCachePreferL1:
      if ((m_shader_config->m_L1D_config.m_config_stringPrefL1 == NULL) ||
          (m_shader_config->gpgpu_shmem_sizePrefL1 == (unsigned)-1)) {
        printf("WARNING: missing Preferred L1 configuration\n");
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizeDefault;

      } else {
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_stringPrefL1,
            FuncCachePreferL1);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizePrefL1;
      }
      break;
    case FuncCachePreferShared:
      if ((m_shader_config->m_L1D_config.m_config_stringPrefShared == NULL) ||
          (m_shader_config->gpgpu_shmem_sizePrefShared == (unsigned)-1)) {
        printf("WARNING: missing Preferred L1 configuration\n");
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizeDefault;
      } else {
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_stringPrefShared,
            FuncCachePreferShared);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizePrefShared;
      }
      break;
    default:
      break;
  }
}

void gpgpu_sim::clear_executed_kernel_info() {
  m_executed_kernel_names.clear();
  m_executed_kernel_uids.clear();
}

//helper for stat printing
void gpgpu_sim::snapshot_l1d_ports_per_cluster(std::vector<port_snap_t>& out) const {
  const unsigned n = m_shader_config->n_simt_clusters;
  out.assign(n, {});

  for (unsigned cid = 0; cid < n; ++cid) {
    cache_sub_stats css;
    css.clear();
    m_cluster[cid]->get_L1D_sub_stats(css);

    out[cid].avail = css.port_available_cycles;
    out[cid].data  = css.data_port_busy_cycles;
    out[cid].fill  = css.fill_port_busy_cycles;
  }
}

int gpgpu_sim::pick_sampling_core_for_kernel_(const kernel_info_t *k) const {
  int fallback = (int)m_shader_config->gpgpu_warp_issue_shader;

  for (unsigned sm = 0; sm < m_shader_config->n_simt_clusters; ++sm) {
    if (k->is_sm_allowed(sm)) return (int)sm;
  }
  return fallback;
}



void gpgpu_sim::gpu_print_stat(unsigned long long streamID,
                               const kernel_stats_view_t *view,
                               const char *single_kernel_name,
                               int single_kernel_uid) {
  FILE *statfout = stdout;

  // -------- Kernel header (name + uid) --------
  std::string kernel_info_str;

  if (single_kernel_name != nullptr) {
    // Daemon/per-kernel path
    kernel_info_str  = "kernel_name = ";
    kernel_info_str += single_kernel_name;
    kernel_info_str += "\n";

    kernel_info_str += "kernel_launch_uid = ";
    if (single_kernel_uid >= 0) {
      kernel_info_str += std::to_string(single_kernel_uid);
    }
    kernel_info_str += "\n";
  } else {
    // Legacy path
    kernel_info_str = executed_kernel_info_string();
  }

  fprintf(statfout, "%s", kernel_info_str.c_str());

  // -------- Top-level counters (cycles, insn, IPC, occupancy) --------
  printf("kernel_stream_id = %llu\n", streamID);

  auto choose_ll = [&](long long ov, long long fb) { return (view && ov != -1) ? ov : fb; };
  auto choose_f  = [&](float ov, float fb)        { return (view && ov >= 0.0f) ? ov : fb; };
  auto choose_d  = [&](double ov, double fb)      { return (view && ov >= 0.0) ? ov : fb; };


  long long sim_cycle = choose_ll(
      view ? view->gpu_sim_cycle : -1,
      (long long)gpu_sim_cycle);

  long long sim_insn = choose_ll(
      view ? view->gpu_sim_insn : -1,
      (long long)gpu_sim_insn);

  long long tot_cycle = choose_ll(
      view ? view->gpu_tot_sim_cycle : -1,
      (long long)(gpu_tot_sim_cycle + gpu_sim_cycle));

  long long tot_insn = choose_ll(
      view ? view->gpu_tot_sim_insn : -1,
      (long long)(gpu_tot_sim_insn + gpu_sim_insn));

  long long tot_cta = choose_ll(
      view ? view->gpu_tot_issued_cta : -1,
      (long long)(gpu_tot_issued_cta + m_total_cta_launched));

  float occ = choose_f(
      view ? view->gpu_occupancy_percent : -1.0f,
      gpu_occupancy.get_occ_fraction() * 100.0f);

  float occ_tot = choose_f(
      view ? view->gpu_tot_occupancy_percent : -1.0f,
      (gpu_occupancy + gpu_tot_occupancy).get_occ_fraction() * 100.0f);

  if (sim_cycle < 1)  sim_cycle  = 1;   
  if (tot_cycle < 1)  tot_cycle  = 1;

  printf("gpu_sim_cycle = %lld\n", sim_cycle);
  printf("gpu_sim_insn = %lld\n", sim_insn);
  printf("gpu_sim_insn calculated by classic = %lld\n", gpu_sim_insn);
  printf("gpu_ipc = %12.4f\n", (float)sim_insn / (float)sim_cycle);
  printf("gpu_tot_sim_cycle = %lld\n", tot_cycle);
  printf("gpu_tot_sim_insn = %lld\n", tot_insn);
  printf("gpu_tot_ipc = %12.4f\n", (float)tot_insn / (float)tot_cycle);
  printf("gpu_tot_issued_cta = %lld\n", tot_cta);
  printf("gpu_occupancy = %.4f%% \n", occ);
  printf("gpu_tot_occupancy = %.4f%% \n", occ_tot);

  fprintf(statfout, "max_total_param_size = %llu\n",
          gpgpu_ctx->device_runtime->g_max_total_param_size);


  long long dram_stall = choose_ll(view ? view->gpu_stall_dramfull : -1,
                                  (long long)gpu_stall_dramfull);
  long long icnt_stall = choose_ll(view ? view->gpu_stall_icnt2sh  : -1,
                                  (long long)gpu_stall_icnt2sh);

  printf("gpu_stall_dramfull = %lld\n", dram_stall);
  printf("gpu_stall_icnt2sh    = %lld\n", icnt_stall);


  // -------- Partition-level parallelism + L2_BW --------
  // partition-level parallelism stays global 
  long long safe_sim_cycle = sim_cycle;
  long long safe_tot_cycle = tot_cycle;
  if (safe_sim_cycle < 1) safe_sim_cycle = 1;
  if (safe_tot_cycle < 1) safe_tot_cycle = 1;

  double pll =
      (double)partiton_reqs_in_parallel / (double)safe_sim_cycle;
  double pll_tot =
      (double)(partiton_reqs_in_parallel + partiton_reqs_in_parallel_total) /
      (double)safe_tot_cycle;

  unsigned long long safe_sim_cycle_part =
      gpu_sim_cycle_parition_util ? gpu_sim_cycle_parition_util : 1ULL;
  unsigned long long safe_tot_cycle_part =
      (gpu_sim_cycle_parition_util + gpu_tot_sim_cycle_parition_util)
          ? (gpu_sim_cycle_parition_util + gpu_tot_sim_cycle_parition_util)
          : 1ULL;

  double pll_util =
      (double)partiton_reqs_in_parallel_util / (double)safe_sim_cycle_part;
  double pll_util_tot =
      (double)(partiton_reqs_in_parallel_util +
               partiton_reqs_in_parallel_util_total) /
      (double)safe_tot_cycle_part;

  double l2_bw =
      (view ? view->L2_BW
            : ((((double)partiton_replys_in_parallel * 32.0) /
                ((double)safe_sim_cycle * m_config.core_period)) /
               1e9));

  double l2_bw_tot =
      (view ? view->L2_BW_total
            : ((((double)(partiton_replys_in_parallel +
                          partiton_replys_in_parallel_total) *
                 32.0) /
                ((double)safe_tot_cycle * m_config.core_period)) /
               1e9));

  printf("partiton_level_parallism = %12.4f\n", (float)pll);
  printf("partiton_level_parallism_total  = %12.4f\n", (float)pll_tot);
  printf("partiton_level_parallism_util = %12.4f\n", (float)pll_util);
  printf("partiton_level_parallism_util_total  = %12.4f\n", (float)pll_util_tot);
  printf("L2_BW  = %12.4f GB/Sec\n", (float)l2_bw);
  printf("L2_BW_total  = %12.4f GB/Sec\n", (float)l2_bw_tot);

  // printf("partiton_level_parallism calculated_by_classic = %12.4f\n", (float)pll);
  // printf("partiton_level_parallism_total calculated_by_classic  = %12.4f\n", (float)pll_tot);
  // printf("partiton_level_parallism_util calculated_by_classic= %12.4f\n", (float)pll_util);
  // printf("partiton_level_parallism_util_total  calculated_by_classic= %12.4f\n", (float)pll_util_tot);
  printf("L2_BW  calculated_by_classic= %12.4f GB/Sec\n", (float)((((double)partiton_replys_in_parallel * 32.0) /
                ((double)safe_sim_cycle * m_config.core_period)) /
               1e9));
  printf("L2_BW_total  calculated_by_classic= %12.4f GB/Sec\n", (float)((((double)(partiton_replys_in_parallel +
                          partiton_replys_in_parallel_total) *
                 32.0) /
                ((double)safe_tot_cycle * m_config.core_period)) /
               1e9));



  time_t curr_time;
  time(&curr_time);
  unsigned long long elapsed_time =
      MAX(curr_time - gpgpu_ctx->the_gpgpusim->g_simulation_starttime, 1);
  printf("gpu_total_sim_rate=%u\n",
         (unsigned)((gpu_tot_sim_insn + gpu_sim_insn) / elapsed_time));

  //shader_print_cache_stats(stdout);
  // CALL THE OVERLOAD
  shader_print_cache_stats(stdout, view);


  cache_stats core_cache_stats;
  core_cache_stats.clear();
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    m_cluster[i]->get_cache_stats(core_cache_stats);
  }
  printf("\nTotal_core_cache_stats:\n");
  core_cache_stats.print_stats(stdout, streamID,
                               "Total_core_cache_stats_breakdown");
  printf("\nTotal_core_cache_fail_stats:\n");
  core_cache_stats.print_fail_stats(stdout, streamID,
                                    "Total_core_cache_fail_stats_breakdown");
  //shader_print_scheduler_stat(stdout, false);
  // call the overload
  shader_print_scheduler_stat(stdout, false, view);

  //m_shader_stats->print(stdout);
  //call overload
  m_shader_stats->print(stdout, view);

#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    if (m_config.g_power_simulation_mode > 0) {
      // if(!m_config.g_aggregate_power_stats)
      mcpat_reset_perf_count(m_gpgpusim_wrapper);
      calculate_hw_mcpat(m_config, getShaderCoreConfig(), m_gpgpusim_wrapper,
                         m_power_stats, m_config.gpu_stat_sample_freq,
                         gpu_tot_sim_cycle, gpu_sim_cycle, gpu_tot_sim_insn,
                         gpu_sim_insn, m_config.g_power_simulation_mode,
                         m_config.g_dvfs_enabled, m_config.g_hw_perf_file_name,
                         m_config.g_hw_perf_bench_name, executed_kernel_name(),
                         m_config.accelwattch_hybrid_configuration,
                         m_config.g_aggregate_power_stats);
    }
    m_gpgpusim_wrapper->print_power_kernel_stats(
        gpu_sim_cycle, gpu_tot_sim_cycle, gpu_tot_sim_insn + gpu_sim_insn,
        kernel_info_str, true);
    // if(!m_config.g_aggregate_power_stats)
    mcpat_reset_perf_count(m_gpgpusim_wrapper);
  }
#endif
  //call overload
  // performance counter that are not local to one shader
  m_memory_stats->memlatstat_print(m_memory_config->m_n_mem,
                                   m_memory_config->nbk,view);
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
    m_memory_partition_unit[i]->print(stdout,single_kernel_uid,view);

  // L2 cache stats
  if (!m_memory_config->m_L2_config.disabled()) {
    cache_stats l2_stats;
    struct cache_sub_stats l2_css;
    struct cache_sub_stats total_l2_css;
    l2_stats.clear();
    l2_css.clear();
    total_l2_css.clear();

    printf("\n========= L2 cache stats =========\n");
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      m_memory_sub_partition[i]->accumulate_L2cache_stats(l2_stats);
      m_memory_sub_partition[i]->get_L2cache_sub_stats(l2_css);

      fprintf(stdout,
              "L2_cache_bank[%d]: Access = %llu, Miss = %llu, Miss_rate = "
              "%.3lf, Pending_hits = %llu, Reservation_fails = %llu\n",
              i, l2_css.accesses, l2_css.misses,
              (double)l2_css.misses / (double)l2_css.accesses,
              l2_css.pending_hits, l2_css.res_fails);

      total_l2_css += l2_css;
    }
    if (!m_memory_config->m_L2_config.disabled() &&
        m_memory_config->m_L2_config.get_num_lines()) {
      // L2c_print_cache_stat();
      printf("L2_total_cache_accesses = %llu\n", total_l2_css.accesses);
      printf("L2_total_cache_misses = %llu\n", total_l2_css.misses);
      if (total_l2_css.accesses > 0)
        printf("L2_total_cache_miss_rate = %.4lf\n",
               (double)total_l2_css.misses / (double)total_l2_css.accesses);
      printf("L2_total_cache_pending_hits = %llu\n", total_l2_css.pending_hits);
      printf("L2_total_cache_reservation_fails = %llu\n",
             total_l2_css.res_fails);
      printf("L2_total_cache_breakdown:\n");
      l2_stats.print_stats(stdout, streamID, "L2_cache_stats_breakdown");
      printf("L2_total_cache_reservation_fail_breakdown:\n");
      l2_stats.print_fail_stats(stdout, streamID,
                                "L2_cache_stats_fail_breakdown");
      total_l2_css.print_port_stats(stdout, "L2_cache");
    }
  }

  if (m_config.gpgpu_cflog_interval != 0) {
    spill_log_to_file(stdout, 1, gpu_sim_cycle);
    insn_warp_occ_print(stdout);
  }
  if (gpgpu_ctx->func_sim->gpgpu_ptx_instruction_classification) {
    StatDisp(gpgpu_ctx->func_sim->g_inst_classification_stat
                 [gpgpu_ctx->func_sim->g_ptx_kernel_count]);
    StatDisp(gpgpu_ctx->func_sim->g_inst_op_classification_stat
                 [gpgpu_ctx->func_sim->g_ptx_kernel_count]);
  }

#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    m_gpgpusim_wrapper->detect_print_steady_state(
        1, gpu_tot_sim_insn + gpu_sim_insn);
  }
#endif

  long total_simt_to_mem = 0;
  long total_mem_to_simt = 0;
  long temp_stm = 0;
  long temp_mts = 0;
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    m_cluster[i]->get_icnt_stats(temp_stm, temp_mts);
    total_simt_to_mem += temp_stm;
    total_mem_to_simt += temp_mts;
  }
  printf("\nicnt_total_pkts_mem_to_simt=%ld\n", total_mem_to_simt);
  printf("icnt_total_pkts_simt_to_mem=%ld\n", total_simt_to_mem);

  time_vector_print();
  fflush(stdout);

  clear_executed_kernel_info();
}


// performance counter that are not local to one shader
unsigned gpgpu_sim::threads_per_core() const {
  return m_shader_config->n_thread_per_shader;
}

void shader_core_ctx::mem_instruction_stats(const warp_inst_t &inst) {
  unsigned active_count = inst.active_count();
  // this breaks some encapsulation: the is_[space] functions, if you change
  // those, change this.
  switch (inst.space.get_type()) {
    case undefined_space:
    case reg_space:
      break;
    case shared_space:
      m_stats->gpgpu_n_shmem_insn += active_count;
      break;
    case sstarr_space:
      m_stats->gpgpu_n_sstarr_insn += active_count;
      break;
    case const_space:
      m_stats->gpgpu_n_const_insn += active_count;
      break;
    case param_space_kernel:
    case param_space_local:
      m_stats->gpgpu_n_param_insn += active_count;
      break;
    case tex_space:
      m_stats->gpgpu_n_tex_insn += active_count;
      break;
    case global_space:
    case local_space:
      if (inst.is_store())
        m_stats->gpgpu_n_store_insn += active_count;
      else
        m_stats->gpgpu_n_load_insn += active_count;
      break;
    default:
      abort();
  }
}
bool shader_core_ctx::can_issue_1block(kernel_info_t &kernel) {
  // SPAPAD
  if (!kernel.is_sm_allowed(m_sid))
    return false;
  // Jin: concurrent kernels on one SM
  if (m_config->gpgpu_concurrent_kernel_sm) {
    if (m_config->max_cta(kernel) < 1) return false;

    return occupy_shader_resource_1block(kernel, false);
  } else {
    return (get_n_active_cta() < m_config->max_cta(kernel));
  }
}

int shader_core_ctx::find_available_hwtid(unsigned int cta_size, bool occupy) {
  unsigned int step;
  for (step = 0; step < m_config->n_thread_per_shader; step += cta_size) {
    unsigned int hw_tid;
    for (hw_tid = step; hw_tid < step + cta_size; hw_tid++) {
      if (m_occupied_hwtid.test(hw_tid)) break;
    }
    if (hw_tid == step + cta_size)  // consecutive non-active
      break;
  }
  if (step >= m_config->n_thread_per_shader)  // didn't find
    return -1;
  else {
    if (occupy) {
      for (unsigned hw_tid = step; hw_tid < step + cta_size; hw_tid++)
        m_occupied_hwtid.set(hw_tid);
    }
    return step;
  }
}

bool shader_core_ctx::occupy_shader_resource_1block(kernel_info_t &k,
                                                    bool occupy) {
  unsigned threads_per_cta = k.threads_per_cta();
  const class function_info *kernel = k.entry();
  unsigned int padded_cta_size = threads_per_cta;
  unsigned int warp_size = m_config->warp_size;
  if (padded_cta_size % warp_size)
    padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

  if (m_occupied_n_threads + padded_cta_size > m_config->n_thread_per_shader)
    return false;

  if (find_available_hwtid(padded_cta_size, false) == -1) return false;

  const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);

  if (m_occupied_shmem + kernel_info->smem > m_config->gpgpu_shmem_size)
    return false;

  unsigned int used_regs = padded_cta_size * ((kernel_info->regs + 3) & ~3);
  if (m_occupied_regs + used_regs > m_config->gpgpu_shader_registers)
    return false;

  if (m_occupied_ctas + 1 > m_config->max_cta_per_core) return false;

  if (occupy) {
    m_occupied_n_threads += padded_cta_size;
    m_occupied_shmem += kernel_info->smem;
    m_occupied_regs += (padded_cta_size * ((kernel_info->regs + 3) & ~3));
    m_occupied_ctas++;

    SHADER_DPRINTF(LIVENESS,
                   "GPGPU-Sim uArch: Occupied %u threads, %u shared mem, %u "
                   "registers, %u ctas, on shader %d\n",
                   m_occupied_n_threads, m_occupied_shmem, m_occupied_regs,
                   m_occupied_ctas, m_sid);
  }

  return true;
}

void shader_core_ctx::release_shader_resource_1block(unsigned hw_ctaid,
                                                     kernel_info_t &k) {
  if (m_config->gpgpu_concurrent_kernel_sm) {
    unsigned threads_per_cta = k.threads_per_cta();
    const class function_info *kernel = k.entry();
    unsigned int padded_cta_size = threads_per_cta;
    unsigned int warp_size = m_config->warp_size;
    if (padded_cta_size % warp_size)
      padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

    assert(m_occupied_n_threads >= padded_cta_size);
    m_occupied_n_threads -= padded_cta_size;

    int start_thread = m_occupied_cta_to_hwtid[hw_ctaid];

    for (unsigned hwtid = start_thread; hwtid < start_thread + padded_cta_size;
         hwtid++)
      m_occupied_hwtid.reset(hwtid);
    m_occupied_cta_to_hwtid.erase(hw_ctaid);

    const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);

    assert(m_occupied_shmem >= (unsigned int)kernel_info->smem);
    m_occupied_shmem -= kernel_info->smem;

    unsigned int used_regs = padded_cta_size * ((kernel_info->regs + 3) & ~3);
    assert(m_occupied_regs >= used_regs);
    m_occupied_regs -= used_regs;

    assert(m_occupied_ctas >= 1);
    m_occupied_ctas--;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Launches a cooperative thread array (CTA).
 *
 * @param kernel
 *    object that tells us which kernel to ask for a CTA from
 */

unsigned exec_shader_core_ctx::sim_init_thread(
    kernel_info_t &kernel, ptx_thread_info **thread_info, int sid, unsigned tid,
    unsigned threads_left, unsigned num_threads, core_t *core,
    unsigned hw_cta_id, unsigned hw_warp_id, gpgpu_t *gpu) {
  return ptx_sim_init_thread(kernel, thread_info, sid, tid, threads_left,
                             num_threads, core, hw_cta_id, hw_warp_id, gpu);
}

void shader_core_ctx::issue_block2core(kernel_info_t &kernel) {
  //
  // NEW: sanity check – we should never issue a CTA for a kernel
  // onto an SM that is not in that kernel's allowed set.
  // The *real* filtering happens in can_issue_1block(), this is just a guard.
  //
  assert(kernel.is_sm_allowed(m_sid) && "issue_block2core called on an SM that is not allowed for this kernel");

  if (!m_config->gpgpu_concurrent_kernel_sm)
    set_max_cta(kernel);
  else
    assert(occupy_shader_resource_1block(kernel, true));

  kernel.inc_running();

  // find a free CTA context
  unsigned free_cta_hw_id = (unsigned)-1;

  unsigned max_cta_per_core;
  if (!m_config->gpgpu_concurrent_kernel_sm)
    max_cta_per_core = kernel_max_cta_per_shader;
  else
    max_cta_per_core = m_config->max_cta_per_core;
  for (unsigned i = 0; i < max_cta_per_core; i++) {
    if (m_cta_status[i] == 0) {
      free_cta_hw_id = i;
      break;
    }
  }
  assert(free_cta_hw_id != (unsigned)-1);

  // determine hardware threads and warps that will be used for this CTA
  int cta_size = kernel.threads_per_cta();

  // hw warp id = hw thread id mod warp size, so we need to find a range
  // of hardware thread ids corresponding to an integral number of hardware
  // thread ids
  int padded_cta_size = cta_size;
  if (cta_size % m_config->warp_size)
    padded_cta_size =
        ((cta_size / m_config->warp_size) + 1) * (m_config->warp_size);

  unsigned int start_thread, end_thread;

  if (!m_config->gpgpu_concurrent_kernel_sm) {
    start_thread = free_cta_hw_id * padded_cta_size;
    end_thread = start_thread + cta_size;
  } else {
    start_thread = find_available_hwtid(padded_cta_size, true);
    assert((int)start_thread != -1);
    end_thread = start_thread + cta_size;
    assert(m_occupied_cta_to_hwtid.find(free_cta_hw_id) ==
           m_occupied_cta_to_hwtid.end());
    m_occupied_cta_to_hwtid[free_cta_hw_id] = start_thread;
  }

  // reset the microarchitecture state of the selected hardware thread and warp
  // contexts
  reinit(start_thread, end_thread, false);

  // initialize scalar threads and determine which hardware warps they are
  // allocated to bind functional simulation state of threads to hardware
  // resources (simulation)
  warp_set_t warps;
  unsigned nthreads_in_block = 0;
  function_info *kernel_func_info = kernel.entry();
  symbol_table *symtab = kernel_func_info->get_symtab();
  unsigned ctaid = kernel.get_next_cta_id_single();
  checkpoint *g_checkpoint = new checkpoint();
  for (unsigned i = start_thread; i < end_thread; i++) {
    m_threadState[i].m_cta_id = free_cta_hw_id;
    unsigned warp_id = i / m_config->warp_size;
    nthreads_in_block += sim_init_thread(
        kernel, &m_thread[i], m_sid, i, cta_size - (i - start_thread),
        m_config->n_thread_per_shader, this, free_cta_hw_id, warp_id,
        m_cluster->get_gpu());
    m_threadState[i].m_active = true;
    // load thread local memory and register file
    if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
        ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
      char fname[2048];
      snprintf(fname, 2048, "checkpoint_files/thread_%d_%d_reg.txt",
               i % cta_size, ctaid);
      m_thread[i]->resume_reg_thread(fname, symtab);
      char f1name[2048];
      snprintf(f1name, 2048, "checkpoint_files/local_mem_thread_%d_%d_reg.txt",
               i % cta_size, ctaid);
      g_checkpoint->load_global_mem(m_thread[i]->m_local_mem, f1name);
    }
    //
    warps.set(warp_id);
  }
  assert(nthreads_in_block > 0 &&
         nthreads_in_block <=
             m_config->n_thread_per_shader);  // should be at least one, but
                                              // less than max
  m_cta_status[free_cta_hw_id] = nthreads_in_block;

  if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
      ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
    char f1name[2048];
    snprintf(f1name, 2048, "checkpoint_files/shared_mem_%d.txt", ctaid);

    g_checkpoint->load_global_mem(m_thread[start_thread]->m_shared_mem, f1name);
  }
  // now that we know which warps are used in this CTA, we can allocate
  // resources for use in CTA-wide barrier operations
  m_barriers.allocate_barrier(free_cta_hw_id, warps);

  // initialize the SIMT stacks and fetch hardware
  init_warps(free_cta_hw_id, start_thread, end_thread, ctaid, cta_size, kernel);
  m_n_active_cta++;

  shader_CTA_count_log(m_sid, 1);
  // NEW: per-kernel CTA counter
  m_gpu->record_kernel_issued_cta(kernel.get_uid(), 1);
  SHADER_DPRINTF(LIVENESS,
                 "GPGPU-Sim uArch: cta:%2u, start_tid:%4u, end_tid:%4u, "
                 "initialized @(%lld,%lld), kernel_uid:%u, kernel_name:%s\n",
                 free_cta_hw_id, start_thread, end_thread, m_gpu->gpu_sim_cycle,
                 m_gpu->gpu_tot_sim_cycle, kernel.get_uid(),
                 kernel.get_name().c_str());
}

///////////////////////////////////////////////////////////////////////////////////////////

void dram_t::dram_log(int task) {
  if (task == SAMPLELOG) {
    StatAddSample(mrqq_Dist, que_length());
  } else if (task == DUMPLOG) {
    printf("Queue Length DRAM[%d] ", id);
    StatDisp(mrqq_Dist);
  }
}

// Find next clock domain and increment its time
int gpgpu_sim::next_clock_domain(void) {
  double smallest = min3(core_time, icnt_time, dram_time);
  int mask = 0x00;
  if (l2_time <= smallest) {
    smallest = l2_time;
    mask |= L2;
    l2_time += m_config.l2_period;
  }
  if (icnt_time <= smallest) {
    mask |= ICNT;
    icnt_time += m_config.icnt_period;
  }
  if (dram_time <= smallest) {
    mask |= DRAM;
    dram_time += m_config.dram_period;
  }
  if (core_time <= smallest) {
    mask |= CORE;
    core_time += m_config.core_period;
  }
  return mask;
}

void gpgpu_sim::issue_block2core() {
  unsigned last_issued = m_last_cluster_issue;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    unsigned idx = (i + last_issued + 1) % m_shader_config->n_simt_clusters;
    unsigned num = m_cluster[idx]->issue_block2core();
    if (num) {
      m_last_cluster_issue = idx;
      m_total_cta_launched += num;
    }
  }
}

unsigned long long g_single_step =
    0;  // set this in gdb to single step the pipeline

void gpgpu_sim::cycle() {
  int clock_mask = next_clock_domain();

  if (clock_mask & CORE) {
    // shader core loading (pop from ICNT into core) follows CORE clock
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
      m_cluster[i]->icnt_cycle();
  }
  unsigned partiton_replys_in_parallel_per_cycle = 0;
  if (clock_mask & ICNT) {
    // pop from memory controller to interconnect
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      mem_fetch *mf = m_memory_sub_partition[i]->top();
      if (mf) {
        unsigned response_size =
            mf->get_is_write() ? mf->get_ctrl_size() : mf->size();

        if (::icnt_has_buffer(m_shader_config->mem2device(i), response_size)) {
          mf->set_return_timestamp(gpu_sim_cycle + gpu_tot_sim_cycle);
          mf->set_status(IN_ICNT_TO_SHADER, gpu_sim_cycle + gpu_tot_sim_cycle);

          // NEW: per-kernel reply attribution (use for L2_BW)
          if (mf && mf->has_kernel_uid()) {
            unsigned kid = mf->get_kernel_uid();
            record_kernel_l2_bytes(kid, response_size);
            record_kernel_l2_request(kid, 1);   // also count requests
          }

          ::icnt_push(m_shader_config->mem2device(i), mf->get_tpc(), mf, response_size);
          m_memory_sub_partition[i]->pop();
          partiton_replys_in_parallel_per_cycle++;
        } else {
          gpu_stall_icnt2sh++;

          // NEW: per-kernel stall attribution
          if (mf->has_kernel_uid()) {
            record_kernel_stall_icnt2sh(mf->get_kernel_uid(), 1);
          }
        }
      } else {
        m_memory_sub_partition[i]->pop();
      }
    }
  }
  partiton_replys_in_parallel += partiton_replys_in_parallel_per_cycle;

  if (clock_mask & DRAM) {
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      if (m_memory_config->simple_dram_model)
        m_memory_partition_unit[i]->simple_dram_model_cycle();
      else
        m_memory_partition_unit[i]
            ->dram_cycle();  // Issue the dram command (scheduler + delay model)
      // Update performance counters for DRAM
      if (m_config.g_power_simulation_enabled) {
        m_memory_partition_unit[i]->set_dram_power_stats(
            m_power_stats->pwr_mem_stat->n_cmd[CURRENT_STAT_IDX][i],
            m_power_stats->pwr_mem_stat->n_activity[CURRENT_STAT_IDX][i],
            m_power_stats->pwr_mem_stat->n_nop[CURRENT_STAT_IDX][i],
            m_power_stats->pwr_mem_stat->n_act[CURRENT_STAT_IDX][i],
            m_power_stats->pwr_mem_stat->n_pre[CURRENT_STAT_IDX][i],
            m_power_stats->pwr_mem_stat->n_rd[CURRENT_STAT_IDX][i],
            m_power_stats->pwr_mem_stat->n_wr[CURRENT_STAT_IDX][i],
            m_power_stats->pwr_mem_stat->n_wr_WB[CURRENT_STAT_IDX][i],
            m_power_stats->pwr_mem_stat->n_req[CURRENT_STAT_IDX][i]);
      }
    }
  }

  // L2 operations follow L2 clock domain
  unsigned partiton_reqs_in_parallel_per_cycle = 0;
  if (clock_mask & L2) {
    m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].clear();
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      // move memory request from interconnect into memory partition (if not
      // backed up) Note:This needs to be called in DRAM clock domain if there
      // is no L2 cache in the system In the worst case, we may need to push
      // SECTOR_CHUNCK_SIZE requests, so ensure you have enough buffer for them
      if (m_memory_sub_partition[i]->full(SECTOR_CHUNCK_SIZE)) {
        gpu_stall_dramfull++;

        // NEW: attribute the stall to the kernel that is head-of-line at this memory node
        mem_fetch *blocked = (mem_fetch*) icnt_peek(m_shader_config->mem2device(i));
        if (blocked && blocked->has_kernel_uid()) {
          record_kernel_stall_dramfull(blocked->get_kernel_uid(), 1);
        }
      } else {
        mem_fetch *mf = (mem_fetch *)icnt_pop(m_shader_config->mem2device(i));
        m_memory_sub_partition[i]->push(mf, gpu_sim_cycle + gpu_tot_sim_cycle);
        if (mf) partiton_reqs_in_parallel_per_cycle++;
      }

      m_memory_sub_partition[i]->cache_cycle(gpu_sim_cycle + gpu_tot_sim_cycle);
      if (m_config.g_power_simulation_enabled) {
        m_memory_sub_partition[i]->accumulate_L2cache_stats(
            m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX]);
      }
    }
  }
  partiton_reqs_in_parallel += partiton_reqs_in_parallel_per_cycle;
  if (partiton_reqs_in_parallel_per_cycle > 0) {
    partiton_reqs_in_parallel_util += partiton_reqs_in_parallel_per_cycle;
    gpu_sim_cycle_parition_util++;
  }

  if (clock_mask & ICNT) {
    icnt_transfer();
  }

  if (clock_mask & CORE) {
    // L1 cache + shader core pipeline stages
    m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].clear();
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      if (m_cluster[i]->get_not_completed() || get_more_cta_left()) {
        m_cluster[i]->core_cycle();
        *active_sms += m_cluster[i]->get_n_active_sms();
      }
      // Update core icnt/cache stats for AccelWattch
      if (m_config.g_power_simulation_enabled) {
        m_cluster[i]->get_icnt_stats(
            m_power_stats->pwr_mem_stat->n_simt_to_mem[CURRENT_STAT_IDX][i],
            m_power_stats->pwr_mem_stat->n_mem_to_simt[CURRENT_STAT_IDX][i]);
        m_cluster[i]->get_cache_stats(
            m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX]);
      }
      m_cluster[i]->get_current_occupancy(
          gpu_occupancy.aggregate_warp_slot_filled,
          gpu_occupancy.aggregate_theoretical_warp_slots);
    }
    float temp = 0;
    for (unsigned i = 0; i < m_shader_config->num_shader(); i++) {
      temp += m_shader_stats->m_pipeline_duty_cycle[i];
    }
    temp = temp / m_shader_config->num_shader();
    *average_pipeline_duty_cycle = ((*average_pipeline_duty_cycle) + temp);
    // cout<<"Average pipeline duty cycle:
    // "<<*average_pipeline_duty_cycle<<endl;

    if (g_single_step &&
        ((gpu_sim_cycle + gpu_tot_sim_cycle) >= g_single_step)) {
      raise(SIGTRAP);  // Debug breakpoint
    }
    if (g_interactive_debugger_enabled) gpgpu_debug();

    // NEW: overlap-safe per-kernel cycle attribution
    // 1 sim "core cycle" is charged to every kernel that is currently running.
    for (auto *k : m_running_kernels) {
      if (k && !k->done()) {
        record_kernel_cycle(k->get_uid(), 1);
      }
    }

    gpu_sim_cycle++;


      // McPAT main cycle (interface with McPAT)
#ifdef GPGPUSIM_POWER_MODEL
    if (m_config.g_power_simulation_enabled) {
      if (m_config.g_power_simulation_mode == 0) {
        mcpat_cycle(m_config, getShaderCoreConfig(), m_gpgpusim_wrapper,
                    m_power_stats, m_config.gpu_stat_sample_freq,
                    gpu_tot_sim_cycle, gpu_sim_cycle, gpu_tot_sim_insn,
                    gpu_sim_insn, m_config.g_dvfs_enabled);
      }
    }
#endif

    issue_block2core();
    decrement_kernel_latency();

    // Depending on configuration, invalidate the caches once all of threads are
    // completed.
    int all_threads_complete = 1;
    if (m_config.gpgpu_flush_l1_cache) {
      for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
        if (m_cluster[i]->get_not_completed() == 0)
          m_cluster[i]->cache_invalidate();
        else
          all_threads_complete = 0;
      }
    }

    if (m_config.gpgpu_flush_l2_cache) {
      if (!m_config.gpgpu_flush_l1_cache) {
        for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
          if (m_cluster[i]->get_not_completed() != 0) {
            all_threads_complete = 0;
            break;
          }
        }
      }

      if (all_threads_complete && !m_memory_config->m_L2_config.disabled()) {
        printf("Flushed L2 caches...\n");
        if (m_memory_config->m_L2_config.get_num_lines()) {
          int dlc = 0;
          for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
            dlc = m_memory_sub_partition[i]->flushL2();
            assert(dlc == 0);  // TODO: need to model actual writes to DRAM here
            printf("Dirty lines flushed from L2 %d is %d\n", i, dlc);
          }
        }
      }
    }

    if (!(gpu_sim_cycle % m_config.gpu_stat_sample_freq)) {
      time_t days, hrs, minutes, sec;
      time_t curr_time;
      time(&curr_time);
      unsigned long long elapsed_time =
          MAX(curr_time - gpgpu_ctx->the_gpgpusim->g_simulation_starttime, 1);
      if ((elapsed_time - last_liveness_message_time) >=
              m_config.liveness_message_freq &&
          DTRACE(LIVENESS)) {
        days = elapsed_time / (3600 * 24);
        hrs = elapsed_time / 3600 - 24 * days;
        minutes = elapsed_time / 60 - 60 * (hrs + 24 * days);
        sec = elapsed_time - 60 * (minutes + 60 * (hrs + 24 * days));

        unsigned long long active = 0, total = 0;
        for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
          m_cluster[i]->get_current_occupancy(active, total);
        }
        DPRINTFG(LIVENESS,
                 "uArch: inst.: %lld (ipc=%4.1f, occ=%0.4f%% [%llu / %llu]) "
                 "sim_rate=%u (inst/sec) elapsed = %u:%u:%02u:%02u / %s",
                 gpu_tot_sim_insn + gpu_sim_insn,
                 (double)gpu_sim_insn / (double)gpu_sim_cycle,
                 float(active) / float(total) * 100, active, total,
                 (unsigned)((gpu_tot_sim_insn + gpu_sim_insn) / elapsed_time),
                 (unsigned)days, (unsigned)hrs, (unsigned)minutes,
                 (unsigned)sec, ctime(&curr_time));
        fflush(stdout);
        last_liveness_message_time = elapsed_time;
      }
      visualizer_printstat();
      m_memory_stats->memlatstat_lat_pw();
      if (m_config.gpgpu_runtime_stat &&
          (m_config.gpu_runtime_stat_flag != 0)) {
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_BW_STAT) {
          for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
            m_memory_partition_unit[i]->print_stat(stdout);
          printf("maxmrqlatency = %d \n", m_memory_stats->max_mrq_latency);
          printf("maxmflatency = %d \n", m_memory_stats->max_mf_latency);
        }
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SHD_INFO)
          shader_print_runtime_stat(stdout);
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_L1MISS)
          shader_print_l1_miss_stat(stdout);
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SCHED)
          shader_print_scheduler_stat(stdout, false);
      }
    }

    if (!(gpu_sim_cycle % 50000)) {
      // deadlock detection
      if (m_config.gpu_deadlock_detect && gpu_sim_insn == last_gpu_sim_insn) {
        gpu_deadlock = true;
      } else {
        last_gpu_sim_insn = gpu_sim_insn;
      }
    }
    try_snap_shot(gpu_sim_cycle);
    spill_log_to_file(stdout, 0, gpu_sim_cycle);

#if (CUDART_VERSION >= 5000)
    // launch device kernel
    gpgpu_ctx->device_runtime->launch_one_device_kernel();
#endif
  }
}

void sst_gpgpu_sim::cycle() {
  SST_cycle();
  return;
}

void shader_core_ctx::dump_warp_state(FILE *fout) const {
  fprintf(fout, "\n");
  fprintf(fout, "per warp functional simulation status:\n");
  for (unsigned w = 0; w < m_config->max_warps_per_shader; w++)
    m_warp[w]->print(fout);
}

void gpgpu_sim::perf_memcpy_to_gpu(size_t dst_start_addr, size_t count) {
  if (m_memory_config->m_perf_sim_memcpy) {
    // if(!m_config.trace_driven_mode)    //in trace-driven mode, CUDA runtime
    // can start nre data structure at any position 	assert (dst_start_addr %
    // 32
    //== 0);

    for (unsigned counter = 0; counter < count; counter += 32) {
      const unsigned wr_addr = dst_start_addr + counter;
      addrdec_t raw_addr;
      mem_access_sector_mask_t mask;
      mask.set(wr_addr % 128 / 32);
      m_memory_config->m_address_mapping.addrdec_tlx(wr_addr, &raw_addr);
      const unsigned partition_id =
          raw_addr.sub_partition /
          m_memory_config->m_n_sub_partition_per_memory_channel;
      m_memory_partition_unit[partition_id]->handle_memcpy_to_gpu(
          wr_addr, raw_addr.sub_partition, mask);
    }
  }
}



void gpgpu_sim::dump_pipeline(int mask, int s, int m) const {
  /*
     You may want to use this function while running GPGPU-Sim in gdb.
     One way to do that is add the following to your .gdbinit file:

        define dp
           call g_the_gpu.dump_pipeline_impl((0x40|0x4|0x1),$arg0,0)
        end

     Then, typing "dp 3" will show the contents of the pipeline for shader
     core 3.
  */

  printf("Dumping pipeline state...\n");
  if (!mask) mask = 0xFFFFFFFF;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    if (s != -1) {
      i = s;
    }
    if (mask & 1)
      m_cluster[m_shader_config->sid_to_cluster(i)]->display_pipeline(
          i, stdout, 1, mask & 0x2E);
    if (s != -1) {
      break;
    }
  }
  if (mask & 0x10000) {
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      if (m != -1) {
        i = m;
      }
      printf("DRAM / memory controller %u:\n", i);
      if (mask & 0x100000) m_memory_partition_unit[i]->print_stat(stdout);
      if (mask & 0x1000000) m_memory_partition_unit[i]->visualize();
      if (mask & 0x10000000) m_memory_partition_unit[i]->print(stdout);
      if (m != -1) {
        break;
      }
    }
  }
  fflush(stdout);
}

const shader_core_config *gpgpu_sim::getShaderCoreConfig() {
  return m_shader_config;
}

const memory_config *gpgpu_sim::getMemoryConfig() { return m_memory_config; }

simt_core_cluster *gpgpu_sim::getSIMTCluster() { return *m_cluster; }

void sst_gpgpu_sim::SST_gpgpusim_numcores_equal_check(unsigned sst_numcores) {
  if (m_shader_config->n_simt_clusters != sst_numcores) {
    assert(
        "\nSST core is not equal the GPGPU-sim cores. Open gpgpu-sim.config "
        "file and ensure n_simt_clusters"
        "is the same as SST gpu cores.\n" &&
        0);
  } else {
    printf("\nSST GPU core is equal the GPGPU-sim cores = %d\n", sst_numcores);
  }
}

void sst_gpgpu_sim::SST_cycle() {
  // shader core loading (pop from ICNT into core) follows CORE clock
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    static_cast<sst_simt_core_cluster *>(m_cluster[i])->icnt_cycle_SST();

  // L1 cache + shader core pipeline stages
  m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].clear();
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    if (m_cluster[i]->get_not_completed() || get_more_cta_left()) {
      m_cluster[i]->core_cycle();
      *active_sms += m_cluster[i]->get_n_active_sms();
    }
    // Update core icnt/cache stats for GPUWattch
    m_cluster[i]->get_icnt_stats(
        m_power_stats->pwr_mem_stat->n_simt_to_mem[CURRENT_STAT_IDX][i],
        m_power_stats->pwr_mem_stat->n_mem_to_simt[CURRENT_STAT_IDX][i]);
    m_cluster[i]->get_cache_stats(
        m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX]);
  }
  float temp = 0;
  for (unsigned i = 0; i < m_shader_config->num_shader(); i++) {
    temp += m_shader_stats->m_pipeline_duty_cycle[i];
  }
  temp = temp / m_shader_config->num_shader();
  *average_pipeline_duty_cycle = ((*average_pipeline_duty_cycle) + temp);
  // cout<<"Average pipeline duty cycle: "<<*average_pipeline_duty_cycle<<endl;

  if (g_single_step && ((gpu_sim_cycle + gpu_tot_sim_cycle) >= g_single_step)) {
    asm("int $03");
  }
  gpu_sim_cycle++;
  if (g_interactive_debugger_enabled) gpgpu_debug();

    // McPAT main cycle (interface with McPAT)
#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    mcpat_cycle(m_config, getShaderCoreConfig(), m_gpgpusim_wrapper,
                m_power_stats, m_config.gpu_stat_sample_freq, gpu_tot_sim_cycle,
                gpu_sim_cycle, gpu_tot_sim_insn, gpu_sim_insn,
                m_config.g_dvfs_enabled);
  }
#endif

  issue_block2core();

  if (!(gpu_sim_cycle % m_config.gpu_stat_sample_freq)) {
    time_t days, hrs, minutes, sec;
    time_t curr_time;
    time(&curr_time);
    unsigned long long elapsed_time =
        MAX(curr_time - gpgpu_ctx->the_gpgpusim->g_simulation_starttime, 1);
    if ((elapsed_time - last_liveness_message_time) >=
        m_config.liveness_message_freq) {
      days = elapsed_time / (3600 * 24);
      hrs = elapsed_time / 3600 - 24 * days;
      minutes = elapsed_time / 60 - 60 * (hrs + 24 * days);
      sec = elapsed_time - 60 * (minutes + 60 * (hrs + 24 * days));

      last_liveness_message_time = elapsed_time;
    }
    visualizer_printstat();
    m_memory_stats->memlatstat_lat_pw();
    if (m_config.gpgpu_runtime_stat && (m_config.gpu_runtime_stat_flag != 0)) {
      if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_BW_STAT) {
        for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
          m_memory_partition_unit[i]->print_stat(stdout);
        printf("maxmrqlatency = %d \n", m_memory_stats->max_mrq_latency);
        printf("maxmflatency = %d \n", m_memory_stats->max_mf_latency);
      }
      if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SHD_INFO)
        shader_print_runtime_stat(stdout);
      if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_L1MISS)
        shader_print_l1_miss_stat(stdout);
      if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SCHED)
        shader_print_scheduler_stat(stdout, false);
    }
  }

  if (!(gpu_sim_cycle % 20000)) {
    // deadlock detection
    if (m_config.gpu_deadlock_detect && gpu_sim_insn == last_gpu_sim_insn) {
      gpu_deadlock = true;
    } else {
      last_gpu_sim_insn = gpu_sim_insn;
    }
  }
  try_snap_shot(gpu_sim_cycle);
  spill_log_to_file(stdout, 0, gpu_sim_cycle);

#if (CUDART_VERSION >= 5000)
  // launch device kernel
  gpgpu_ctx->device_runtime->launch_one_device_kernel();
#endif
}

// void gpgpu_sim::kernel_add_inst(unsigned kernel_uid, unsigned long long n) {
//   record_kernel_inst(kernel_uid, n);
// }

// void gpgpu_sim::kernel_add_mem_reply(unsigned kernel_uid,
//                                      unsigned bytes,
//                                      unsigned long long /*concurrent_replies_this_cycle*/,
//                                      bool /*mem_busy_this_cycle*/) {
//   // Minimal safe behavior: record bytes (bandwidth). We'll add exact reqs/concurrency later.
//   record_kernel_l2_bytes(kernel_uid, bytes);
//   record_kernel_l2_request(kernel_uid, 1);
// }

// void gpgpu_sim::kernel_add_dram_stall(unsigned kernel_uid) {
//   record_kernel_stall_dramfull(kernel_uid, 1);
// }

// void gpgpu_sim::kernel_add_icnt_stall(unsigned kernel_uid) {
//   record_kernel_stall_icnt2sh(kernel_uid, 1);
// }



void gpgpu_sim::note_kernel_launch(kernel_info_t* k) {
  if (!k) return;
  bind_kernel_to_allowed_sms(k);
  const unsigned long long streamID = k->get_streamID();
  const unsigned kid = k->get_uid();

  // Track "current kernel per stream"
  stream_to_kernel_map[streamID] = kid;
  auto &rec = m_sched_issue_rec_[kid];
  rec.sampling_core = -1;
  rec.distro.clear();
}

void gpgpu_sim::note_kernel_completion(kernel_info_t* k) {
  if (!k) return;
  unbind_kernel_from_sms(k->get_uid());
  const unsigned long long streamID = k->get_streamID();
  const unsigned kid = k->get_uid();

  auto it = stream_to_kernel_map.find(streamID);
  if (it != stream_to_kernel_map.end() && it->second == kid) {
    stream_to_kernel_map.erase(it);
  }
}

port_snap_t gpgpu_sim::snap_l1d_ports_for_cluster_(unsigned c) const {
  cache_sub_stats css;
  css.clear();
  m_cluster[c]->get_L1D_sub_stats(css); // cumulative counters up to "now"
  port_snap_t s;
  s.avail = css.port_available_cycles;
  s.data  = css.data_port_busy_cycles;
  s.fill  = css.fill_port_busy_cycles;
  return s;
}

void gpgpu_sim::record_kernel_warp_issue(unsigned smid,
                                        unsigned warp_id,
                                        unsigned ,
                                        unsigned ,
                                        unsigned long long streamID,
                                        const warp_inst_t * ) {
  // Map stream -> kernel uid
  auto it = stream_to_kernel_map.find(streamID);
  if (it == stream_to_kernel_map.end()) return;
  const unsigned kid = it->second;

  auto rit = m_sched_issue_rec_.find(kid);
  if (rit == m_sched_issue_rec_.end()) return;

  auto &rec = rit->second;

  // choose sampling core once
  if (rec.sampling_core < 0) rec.sampling_core = (int)smid;

  // only record distro on sampling core (like classic print behavior)
  if (rec.sampling_core != (int)smid) return;

  // warp issue distro
  if (rec.distro.size() <= warp_id) rec.distro.resize(warp_id + 1, 0);
  rec.distro[warp_id]++;
}




// NEW commit recorders
void gpgpu_sim::record_kernel_inst_commit(unsigned kernel_uid,
                                         unsigned active_lanes)
{
  auto &a = kernel_stats_mut_(kernel_uid);
  a.warp_icount++;

  if (!m_shader_config->gpgpu_clock_gated_lanes) {
    a.thrd_icount += m_shader_config->warp_size;
  } else {
    a.thrd_icount += active_lanes;
  }
}

void gpgpu_sim::record_kernel_icnt_stats(unsigned kid, const mem_fetch *mf) {
  if (!kid || !mf) return;

  auto &a = kernel_stats_mut_(kid);

  switch (mf->get_access_type()) {
    case CONST_ACC_R:
      a.n_mem_const++;
      break;

    case TEXTURE_ACC_R:
      a.n_mem_texture++;
      break;

    case GLOBAL_ACC_R:
      a.n_mem_read_global++;
      break;

    case GLOBAL_ACC_W:
      a.n_mem_write_global++;
      break;

    case LOCAL_ACC_R:
      a.n_mem_read_local++;
      break;

    case LOCAL_ACC_W:
      a.n_mem_write_local++;
      break;

    case L1_WRBK_ACC:
      a.n_mem_write_global++;
      break;
    default:
      break;
  }
  ////
}

void gpgpu_sim::record_kernel_mem_inst_commit(unsigned kernel_uid,
                                              const warp_inst_t &inst) {
  if (!kernel_uid) return;
  auto &a = kernel_stats_mut_(kernel_uid);
  const unsigned active_count = inst.active_count();

  switch (inst.space.get_type()) {
    case undefined_space:
    case reg_space:
      break;

    case shared_space:
      a.n_shmem_insn += active_count;
      break;

    case sstarr_space:
      a.n_sstarr_insn += active_count;
      break;

    case const_space:
      a.n_const_insn += active_count;
      break;

    case param_space_kernel:
    case param_space_local:
      a.n_param_insn += active_count;
      break;

    case tex_space:
      a.n_tex_insn += active_count;
      break;

    case global_space:
    case local_space:
      if (inst.is_store())
        a.n_store_insn += active_count;
      else
        a.n_load_insn += active_count;
      break;

    default:
      abort();
  }
  ////
}




// void gpgpu_sim::record_kernel_mem_read_L1_MISS(unsigned kernel_uid,
//                                        const warp_inst_t &inst) {
//   if (!kernel_uid) return;
//   if (!inst.is_load()) return;   

//   auto &a = kernel_stats_mut_(kernel_uid);

//   switch (inst.space.get_type()) {
//     case global_space: a.n_mem_read_global++; break;
//     case local_space:  a.n_mem_read_local++;  break;
//     default: break; 
//   }
// }



unsigned gpgpu_sim::kernel_uid_from_stream(unsigned long long streamID) const {
  auto it = stream_to_kernel_map.find(streamID);
  if (it == stream_to_kernel_map.end()) return 0;
  return it->second;
}

void gpgpu_sim::record_kernel_warp_issue_distro(unsigned smid,
                                               unsigned warp_id,
                                               unsigned long long streamID) {
  const unsigned kid = kernel_uid_from_stream(streamID);
  if (!kid) return;

  auto rit = m_sched_issue_rec_.find(kid);
  if (rit == m_sched_issue_rec_.end()) return;

  auto &rec = rit->second;

  if (rec.sampling_core < 0) rec.sampling_core = (int)smid;

  if (rec.sampling_core != (int)smid) return;

  if (rec.distro.size() <= warp_id) rec.distro.resize(warp_id + 1, 0);
  rec.distro[warp_id]++;
}

void gpgpu_sim::record_kernel_shader_active_count_bucket(unsigned kernel_uid,
                                                        unsigned active_count) {
  if (!kernel_uid) return;

  auto &a = kernel_stats_mut_(kernel_uid);

  if (a.shader_cycle_distro.empty()) {
    a.shader_cycle_distro.assign(m_shader_config->warp_size + 3, 0);
  }

  const unsigned idx = 2 + active_count; // W1..W32 (since index 2 is Stall)
  if (idx < a.shader_cycle_distro.size()) a.shader_cycle_distro[idx]++;
}

static inline void ensure_shader_cycle_distro(kernel_stats_accum_t &a,
                                             unsigned warp_size) {
  if (a.shader_cycle_distro.empty()) {
    a.shader_cycle_distro.assign(warp_size + 3, 0);
  }
}

void gpgpu_sim::record_kernel_shader_idle_cycle(unsigned kernel_uid) {
  if (!kernel_uid) return;
  auto &a = kernel_stats_mut_(kernel_uid);
  ensure_shader_cycle_distro(a, m_shader_config->warp_size);
  a.shader_cycle_distro[0]++; // W0_Idle
}

void gpgpu_sim::record_kernel_shader_scoreboard_cycle(unsigned kernel_uid) {
  if (!kernel_uid) return;
  auto &a = kernel_stats_mut_(kernel_uid);
  ensure_shader_cycle_distro(a, m_shader_config->warp_size);
  a.shader_cycle_distro[1]++; // W0_Scoreboard
}

void gpgpu_sim::record_kernel_shader_stall_cycle(unsigned kernel_uid) {
  if (!kernel_uid) return;
  auto &a = kernel_stats_mut_(kernel_uid);
  ensure_shader_cycle_distro(a, m_shader_config->warp_size);
  a.shader_cycle_distro[2]++; // Stall
}

void gpgpu_sim::record_kernel_scheduler_issue(unsigned kernel_uid,
                                             unsigned sched_id,
                                             unsigned issued_count) {
  if (!kernel_uid) return;

  auto &a = kernel_stats_mut_(kernel_uid);

  if (a.single_issue_nums.empty()) {
    a.single_issue_nums.assign(m_shader_config->gpgpu_num_sched_per_core, 0);
    a.dual_issue_nums.assign(m_shader_config->gpgpu_num_sched_per_core, 0);
  }

  if (sched_id >= a.single_issue_nums.size()) return;

  if (issued_count == 1) a.single_issue_nums[sched_id]++;
  else if (issued_count > 1) a.dual_issue_nums[sched_id]++;
}

unsigned gpgpu_sim::kernel_uid_from_smid(unsigned sid) const {
  if (sid >= m_sm_owner_kid.size()) return 0;
  return m_sm_owner_kid[sid];
}

void gpgpu_sim::bind_kernel_to_allowed_sms(const kernel_info_t *k) {
  if (!k) return;
  const unsigned kid = k->get_uid();
  if (!kid) return;

  if (m_sm_owner_kid.empty())
    m_sm_owner_kid.assign(m_shader_config->num_shader(), 0);

  for (unsigned sid = 0; sid < m_shader_config->num_shader(); ++sid) {
    if (!k->is_sm_allowed(sid)) continue;

    if (m_sm_owner_kid[sid] != 0 && m_sm_owner_kid[sid] != kid) {
    }
    m_sm_owner_kid[sid] = kid;
  }
}

void gpgpu_sim::unbind_kernel_from_sms(unsigned kid) {
  if (!kid) return;
  for (auto &x : m_sm_owner_kid)
    if (x == kid) x = 0;
}


void gpgpu_sim::record_kernel_shmem_bkconflict(unsigned kid) {
  if (!kid) return;
  kernel_stats_mut_(kid).gpgpu_n_shmem_bkconflict++;
}

void gpgpu_sim::record_kernel_l1cache_bkconflict(unsigned kid) {
  if (!kid) return;
  kernel_stats_mut_(kid).gpgpu_n_l1cache_bkconflict++;
}

void gpgpu_sim::record_kernel_intrawarp_mshr_merge(unsigned kid) {
  if (!kid) return;
  kernel_stats_mut_(kid).gpgpu_n_intrawarp_mshr_merge++;
}

void gpgpu_sim::record_kernel_cmem_portconflict(unsigned kid) {
  if (!kid) return;
  kernel_stats_mut_(kid).gpgpu_n_cmem_portconflict++;
}

void gpgpu_sim::record_kernel_reg_bank_conflict_stall(unsigned kid) {
  if (!kid) return;
  kernel_stats_mut_(kid).gpu_reg_bank_conflict_stalls++;
}


void gpgpu_sim::record_kernel_stall_shd_mem(unsigned kid,
                                            unsigned access_type,
                                            unsigned stall_type) {
  if (!kid) return;

  auto &a = kernel_stats_mut_(kid);

  a.gpgpu_n_stall_shd_mem++;

  // ---- matches classic printing semantics ----
  // classic prints:
  //   c_mem][resource_stall]  -> breakdown[C_MEM][BK_CONF]
  //   s_mem][bk_conf]         -> breakdown[S_MEM][BK_CONF]
  //   gl_mem][resource_stall] -> sum of *_MEM_* [BK_CONF]
  //   gl_mem][coal_stall]     -> sum of *_MEM_* [COAL_STALL]
  //   gl_mem][data_port_stall]-> sum of *_MEM_* [DATA_PORT_STALL]

  if (access_type == C_MEM && stall_type == BK_CONF) {
    a.gpgpu_stall_shd_mem_cmem_resource_stall++;
    return;
  }

  if (access_type == S_MEM && stall_type == BK_CONF) {
    a.gpgpu_stall_shd_mem_smem_bk_conf++;
    return;
  }

  const bool is_glmem =
      (access_type == G_MEM_LD || access_type == G_MEM_ST ||
       access_type == L_MEM_LD || access_type == L_MEM_ST);

  if (!is_glmem) return;

  if (stall_type == BK_CONF) {
    a.gpgpu_stall_shd_mem_glmem_resource_stall++;
  } else if (stall_type == COAL_STALL) {
    a.gpgpu_stall_shd_mem_glmem_coal_stall++;
  } else if (stall_type == DATA_PORT_STALL) {
    a.gpgpu_stall_shd_mem_glmem_data_port_stall++;
  }
}

void gpgpu_sim::record_kernel_outgoing_traffic(unsigned kid,
                                               mem_fetch *mf,
                                               unsigned sz) {
  if (!kid) return;
  kernel_stats_mut_(kid).outgoing_traffic.record_traffic(mf, sz);
}

void gpgpu_sim::record_kernel_incoming_traffic(unsigned kid,
                                               mem_fetch *mf,
                                               unsigned sz) {
  if (!kid) return;
  kernel_stats_mut_(kid).incoming_traffic.record_traffic(mf, sz);
}



void gpgpu_sim::record_kernel_memlat(unsigned kid,
                                    unsigned mf_lat,
                                    unsigned icnt2mem_lat,
                                    unsigned mrq_lat,
                                    unsigned icnt2sh_lat) {
  if (!kid) return;
  auto &a = kernel_stats_mut_(kid);

  a.memlat_num_mfs++;

  a.memlat_mf_total_lat       += mf_lat;
  a.memlat_tot_icnt2mem_lat   += icnt2mem_lat;
  a.memlat_tot_icnt2sh_lat    += icnt2sh_lat;

  if (mf_lat > a.memlat_max_mf_lat) a.memlat_max_mf_lat = mf_lat;
  if (icnt2mem_lat > a.memlat_max_icnt2mem_lat) a.memlat_max_icnt2mem_lat = icnt2mem_lat;
  if (mrq_lat > a.memlat_max_mrq_lat) a.memlat_max_mrq_lat = mrq_lat;
  if (icnt2sh_lat > a.memlat_max_icnt2sh_lat) a.memlat_max_icnt2sh_lat = icnt2sh_lat;
}


static inline unsigned clamp_bucket(unsigned b, unsigned max) {
  return (b >= max) ? (max - 1) : b;
}

void gpgpu_sim::record_kernel_mrq_lat_bucket(unsigned kid, unsigned mrq_lat) {
  if (!kid || !mrq_lat) return;
  auto &a = kernel_stats_mut_(kid);
  unsigned b = clamp_bucket(LOGB2(mrq_lat), 32);
  a.memlat_mrq_lat_table[b]++;
}

void gpgpu_sim::record_kernel_icnt2mem_lat_bucket(unsigned kid, unsigned lat) {
  if (!kid || !lat) return;
  auto &a = kernel_stats_mut_(kid);
  unsigned b = clamp_bucket(LOGB2(lat), 24);
  a.memlat_icnt2mem_lat_table[b]++;
}

void gpgpu_sim::record_kernel_icnt2sh_lat_bucket(unsigned kid, unsigned lat) {
  if (!kid || !lat) return;
  auto &a = kernel_stats_mut_(kid);
  unsigned b = clamp_bucket(LOGB2(lat), 24);
  a.memlat_icnt2sh_lat_table[b]++;
}

void gpgpu_sim::record_kernel_mf_lat_bucket(unsigned kid, unsigned mf_lat) {
  if (!kid || !mf_lat) return;
  auto &a = kernel_stats_mut_(kid);
  unsigned b = clamp_bucket(LOGB2(mf_lat), 32);
  a.memlat_mf_lat_table[b]++;
}


void gpgpu_sim::record_kernel_mf_lat_pw_accum(unsigned kid, unsigned mf_lat) {
  if (!kid || !mf_lat) return;
  auto &a = kernel_stats_mut_(kid);
  a.memlat_mf_num_lat_pw++;
  a.memlat_mf_tot_lat_pw += mf_lat;
}

void gpgpu_sim::flush_kernel_mf_lat_pw_tables() {
  for (auto &kv : m_kernel_stats_) {
    auto &a = kv.second;
    if (!a.memlat_mf_num_lat_pw) continue;
    unsigned avg = (unsigned)(a.memlat_mf_tot_lat_pw / a.memlat_mf_num_lat_pw);
    unsigned b = clamp_bucket(LOGB2(avg), 32);
    a.memlat_mf_lat_pw_table[b]++;
    a.memlat_mf_tot_lat_pw = 0;
    a.memlat_mf_num_lat_pw = 0;
  }
}


void gpgpu_sim::record_kernel_row_episode_access(unsigned kid,
                                                 unsigned dram,
                                                 unsigned bank,
                                                 unsigned cnt) {
  if (!kid || !cnt) return;
  auto &a = kernel_stats_mut_(kid);

  unsigned n_mem = m_memory_config->m_n_mem;
  unsigned nbk   = m_memory_config->nbk;
  if (dram >= n_mem || bank >= nbk) return;

  // [ADDED] lazy init
  if (a.memlat_max_conc_access2samerow.empty()) {
    a.memlat_max_conc_access2samerow.assign(n_mem * nbk, 0);
  }

  unsigned idx = dram * nbk + bank;
  if (cnt > a.memlat_max_conc_access2samerow[idx])
    a.memlat_max_conc_access2samerow[idx] = cnt;
}

// [ADDED]
void gpgpu_sim::record_kernel_row_episode_servicetime(unsigned kid,
                                                      unsigned dram,
                                                      unsigned bank,
                                                      unsigned srv) {
  if (!kid || !srv) return;

  auto &a = kernel_stats_mut_(kid);

  unsigned n_mem = m_memory_config->m_n_mem;
  unsigned nbk   = m_memory_config->nbk;
  if (dram >= n_mem || bank >= nbk) return;

  // lazy init
  if (a.memlat_max_servicetime2samerow.empty()) {
    a.memlat_max_servicetime2samerow.assign(n_mem * nbk, 0);
  }

  unsigned idx = dram * nbk + bank;
  if (srv > a.memlat_max_servicetime2samerow[idx])
    a.memlat_max_servicetime2samerow[idx] = srv;
}


void gpgpu_sim::record_kernel_row_access(unsigned kid,
                                         unsigned dram,
                                         unsigned bank) {
  if (!kid) return;

  auto &a = kernel_stats_mut_(kid);

  unsigned n_mem = m_memory_config->m_n_mem;
  unsigned nbk   = m_memory_config->nbk;
  if (dram >= n_mem || bank >= nbk) return;

  if (a.memlat_row_access.empty())
    a.memlat_row_access.assign(n_mem * nbk, 0);

  unsigned idx = flat_idx(dram, bank, nbk);
  a.memlat_row_access[idx] += 1ULL;
}

void gpgpu_sim::record_kernel_row_activate(unsigned kid,
                                           unsigned dram,
                                           unsigned bank) {
  if (!kid) return;

  auto &a = kernel_stats_mut_(kid);

  unsigned n_mem = m_memory_config->m_n_mem;
  unsigned nbk   = m_memory_config->nbk;
  if (dram >= n_mem || bank >= nbk) return;

  if (a.memlat_num_activates.empty())
    a.memlat_num_activates.assign(n_mem * nbk, 0);

  unsigned idx = flat_idx(dram, bank, nbk);
  a.memlat_num_activates[idx] += 1ULL;
}


void gpgpu_sim::record_kernel_totalbankread(unsigned kid,unsigned dram,unsigned bank,unsigned long long inc) {
  if (!kid || !inc) return;

  auto &a = kernel_stats_mut_(kid);

  unsigned n_mem = m_memory_config->m_n_mem;
  unsigned nbk   = m_memory_config->nbk;
  if (dram >= n_mem || bank >= nbk) return;

  if (a.memlat_totalbankreads.empty())
    a.memlat_totalbankreads.assign(n_mem * nbk, 0ULL);

  unsigned idx = dram * nbk + bank;
  a.memlat_totalbankreads[idx] += inc;
}

void gpgpu_sim::record_kernel_mf_bank_lat_sum(unsigned kid,unsigned dram,unsigned bank,unsigned mf_latency) {
  if (!kid || !mf_latency) return;

  const unsigned n_mem = m_memory_config->m_n_mem;
  const unsigned nbk   = m_memory_config->nbk;
  if (dram >= n_mem || bank >= nbk) return;

  auto &a = kernel_stats_mut_(kid);

  const unsigned sz  = n_mem * nbk;
  const unsigned idx = dram * nbk + bank;

  if (a.memlat_mf_total_laten.empty())
    a.memlat_mf_total_laten.assign(sz, 0ULL);

  a.memlat_mf_total_laten[idx] += (unsigned long long)mf_latency;
}

void gpgpu_sim::record_max_mf_lat_per_bank(unsigned kid,unsigned dram,unsigned bank,unsigned mf_latency){
  if(!kid||!mf_latency) return;

  const unsigned n_mem = m_memory_config->m_n_mem;
  const unsigned nbk   = m_memory_config->nbk;
  if (dram >= n_mem || bank >= nbk) return;
  auto &a = kernel_stats_mut_(kid);
  const unsigned sz  = n_mem * nbk;
  const unsigned idx = dram * nbk + bank;
  if (a.memlat_max_mf_laten.empty())
    a.memlat_max_mf_laten.assign(sz, 0ULL);

  if(a.memlat_max_mf_laten[idx] < mf_latency) a.memlat_max_mf_laten[idx] = mf_latency;
}



void gpgpu_sim::record_kernel_dram_cycle_counters(unsigned kid,
                                                  unsigned dram_id,
                                                  unsigned long long inc_cmd,
                                                  unsigned long long inc_nop,
                                                  unsigned long long inc_act) {
  if (!kid) return;

  const unsigned n_mem = m_memory_config->m_n_mem;
  if (dram_id >= n_mem) return;

  auto &a = kernel_stats_mut_(kid);

  if (a.dram_n_cmd.empty())      a.dram_n_cmd.assign(n_mem, 0ULL);
  if (a.dram_n_nop.empty())      a.dram_n_nop.assign(n_mem, 0ULL);
  if (a.dram_n_activity.empty()) a.dram_n_activity.assign(n_mem, 0ULL);

  a.dram_n_cmd[dram_id]      += inc_cmd;
  a.dram_n_nop[dram_id]      += inc_nop;
  a.dram_n_activity[dram_id] += inc_act;
}

void gpgpu_sim::record_kernel_dram_row_cmd_counters(unsigned kid,
                                                    unsigned dram_id,
                                                    unsigned long long inc_act,
                                                    unsigned long long inc_pre) {
  if (!kid) return;

  const unsigned n_mem = m_memory_config->m_n_mem;
  if (dram_id >= n_mem) return;

  auto &a = kernel_stats_mut_(kid);

  if (a.dram_n_act.empty()) a.dram_n_act.assign(n_mem, 0ULL);
  if (a.dram_n_pre.empty()) a.dram_n_pre.assign(n_mem, 0ULL);

  a.dram_n_act[dram_id] += inc_act;
  a.dram_n_pre[dram_id] += inc_pre;
}

void gpgpu_sim::record_kernel_dram_req_ref_event(unsigned kid,
                                                 unsigned dram_id,
                                                 unsigned long long inc_req,
                                                 unsigned long long inc_ref) {
  if (!kid) return;
  const unsigned n_mem = m_memory_config->m_n_mem;
  if (dram_id >= n_mem) return;

  auto &a = kernel_stats_mut_(kid);

  if (a.dram_n_req.empty())       a.dram_n_req.assign(n_mem, 0ULL);
  if (a.dram_n_ref_event.empty()) a.dram_n_ref_event.assign(n_mem, 0ULL);

  a.dram_n_req[dram_id]       += inc_req;
  a.dram_n_ref_event[dram_id] += inc_ref;
}

void gpgpu_sim::record_kernel_dram_rw_counters(unsigned kid, unsigned dram_id,
                                              unsigned long long inc_rd,
                                              unsigned long long inc_rd_l2a,
                                              unsigned long long inc_wr,
                                              unsigned long long inc_wr_wb) {
  if (!kid) return;
  const unsigned n_mem = m_memory_config->m_n_mem;
  if (dram_id >= n_mem) return;

  auto &a = kernel_stats_mut_(kid);

  if (a.dram_n_rd.empty())      a.dram_n_rd.assign(n_mem, 0ULL);
  if (a.dram_n_rd_L2_A.empty()) a.dram_n_rd_L2_A.assign(n_mem, 0ULL);
  if (a.dram_n_wr.empty())      a.dram_n_wr.assign(n_mem, 0ULL);
  if (a.dram_n_wr_WB.empty())   a.dram_n_wr_WB.assign(n_mem, 0ULL);

  a.dram_n_rd[dram_id]      += inc_rd;
  a.dram_n_rd_L2_A[dram_id] += inc_rd_l2a;
  a.dram_n_wr[dram_id]      += inc_wr;
  a.dram_n_wr_WB[dram_id]   += inc_wr_wb;
}

void gpgpu_sim::record_kernel_dram_bwutil(unsigned kid,
                                         unsigned dram_id,
                                         unsigned long long inc_bwutil) {
  if (!kid) return;
  if (!inc_bwutil) return;

  auto &a = kernel_stats_mut_(kid);

  if (a.dram_bwutil.empty()) {
    a.dram_bwutil.resize(m_memory_config->m_n_mem, 0ULL);
  }
  if (dram_id >= a.dram_bwutil.size()) return;

  a.dram_bwutil[dram_id] += inc_bwutil;
}



void gpgpu_sim::record_kernel_dram_bank_access(unsigned kid, unsigned dram_id,
                                               unsigned bank,
                                               unsigned long long inc_access) {
  if (!kid) return;

  const unsigned n_mem = m_memory_config->m_n_mem;
  const unsigned nbk   = m_memory_config->nbk;
  if (dram_id >= n_mem || bank >= nbk) return;

  const unsigned idx = dram_id * nbk + bank;

  auto &a = kernel_stats_mut_(kid);
  if (a.dram_bk_n_access.empty()) a.dram_bk_n_access.assign(n_mem * nbk, 0ULL);
  a.dram_bk_n_access[idx] += inc_access;
}

void gpgpu_sim::record_kernel_dram_bank_idle(unsigned kid, unsigned dram_id,
                                             unsigned bank,
                                             unsigned long long inc_idle) {
  if (!kid) return;

  const unsigned n_mem = m_memory_config->m_n_mem;
  const unsigned nbk   = m_memory_config->nbk;
  if (dram_id >= n_mem || bank >= nbk) return;

  const unsigned idx = dram_id * nbk + bank;

  auto &a = kernel_stats_mut_(kid);
  if (a.dram_bk_n_idle.empty()) a.dram_bk_n_idle.assign(n_mem * nbk, 0ULL);
  a.dram_bk_n_idle[idx] += inc_idle;
}


void gpgpu_sim::record_kernel_dram_rowbuf_locality(unsigned kid,
                                                   unsigned dram_id,
                                                   bool is_write,
                                                   bool is_row_hit) {
  if (!kid) return;
  const unsigned n_mem = m_memory_config->m_n_mem;
  if (dram_id >= n_mem) return;

  auto &a = kernel_stats_mut_(kid);

  if (a.dram_access_num.empty())     a.dram_access_num.assign(n_mem, 0ULL);
  if (a.dram_hits_num.empty())       a.dram_hits_num.assign(n_mem, 0ULL);
  if (a.dram_read_num.empty())       a.dram_read_num.assign(n_mem, 0ULL);
  if (a.dram_hits_read_num.empty())  a.dram_hits_read_num.assign(n_mem, 0ULL);
  if (a.dram_write_num.empty())      a.dram_write_num.assign(n_mem, 0ULL);
  if (a.dram_hits_write_num.empty()) a.dram_hits_write_num.assign(n_mem, 0ULL);

  a.dram_access_num[dram_id] += 1ULL;
  if (is_row_hit) a.dram_hits_num[dram_id] += 1ULL;

  if (is_write) {
    a.dram_write_num[dram_id] += 1ULL;
    if (is_row_hit) a.dram_hits_write_num[dram_id] += 1ULL;
  } else {
    a.dram_read_num[dram_id] += 1ULL;
    if (is_row_hit) a.dram_hits_read_num[dram_id] += 1ULL;
  }
}

void gpgpu_sim::record_kernel_dram_blp_stats(unsigned kid, unsigned dram_id,
                                             unsigned long long inc_banks_1time,
                                             unsigned long long inc_banks_access_total,
                                             unsigned long long inc_banks_time_rw,
                                             unsigned long long inc_banks_access_rw_total,
                                             unsigned long long inc_banks_time_ready,
                                             unsigned long long inc_banks_access_ready_total,
                                             unsigned long long inc_w2r_ratio_sum_1e6,
                                             unsigned long long inc_bkgrp_parallsim_rw) {
  if (!kid) return;
  const unsigned n_mem = m_memory_config->m_n_mem;
  if (dram_id >= n_mem) return;

  auto &a = kernel_stats_mut_(kid);

  if (a.dram_banks_1time.empty())              a.dram_banks_1time.assign(n_mem, 0ULL);
  if (a.dram_banks_access_total.empty())       a.dram_banks_access_total.assign(n_mem, 0ULL);

  if (a.dram_banks_time_rw.empty())            a.dram_banks_time_rw.assign(n_mem, 0ULL);
  if (a.dram_banks_access_rw_total.empty())    a.dram_banks_access_rw_total.assign(n_mem, 0ULL);

  if (a.dram_banks_time_ready.empty())         a.dram_banks_time_ready.assign(n_mem, 0ULL);
  if (a.dram_banks_access_ready_total.empty()) a.dram_banks_access_ready_total.assign(n_mem, 0ULL);

  if (a.dram_w2r_ratio_sum_1e6.empty())        a.dram_w2r_ratio_sum_1e6.assign(n_mem, 0ULL);
  if (a.dram_bkgrp_parallsim_rw.empty())       a.dram_bkgrp_parallsim_rw.assign(n_mem, 0ULL);

  a.dram_banks_1time[dram_id]               += inc_banks_1time;
  a.dram_banks_access_total[dram_id]        += inc_banks_access_total;

  a.dram_banks_time_rw[dram_id]             += inc_banks_time_rw;
  a.dram_banks_access_rw_total[dram_id]     += inc_banks_access_rw_total;

  a.dram_banks_time_ready[dram_id]          += inc_banks_time_ready;
  a.dram_banks_access_ready_total[dram_id]  += inc_banks_access_ready_total;

  a.dram_w2r_ratio_sum_1e6[dram_id]         += inc_w2r_ratio_sum_1e6;
  a.dram_bkgrp_parallsim_rw[dram_id]        += inc_bkgrp_parallsim_rw;
}

void gpgpu_sim::record_kernel_dram_bw_class(unsigned kid, unsigned dram_id,
                                            unsigned long long inc_util,
                                            unsigned long long inc_wcol,
                                            unsigned long long inc_wrow,
                                            unsigned long long inc_idle) {
  if (!kid) return;
  const unsigned n_mem = m_memory_config->m_n_mem;
  if (dram_id >= n_mem) return;

  auto &a = kernel_stats_mut_(kid);

  if (a.dram_util_bw.empty())       a.dram_util_bw.assign(n_mem, 0ULL);
  if (a.dram_wasted_bw_col.empty()) a.dram_wasted_bw_col.assign(n_mem, 0ULL);
  if (a.dram_wasted_bw_row.empty()) a.dram_wasted_bw_row.assign(n_mem, 0ULL);
  if (a.dram_idle_bw.empty())       a.dram_idle_bw.assign(n_mem, 0ULL);

  a.dram_util_bw[dram_id]       += inc_util;
  a.dram_wasted_bw_col[dram_id] += inc_wcol;
  a.dram_wasted_bw_row[dram_id] += inc_wrow;
  a.dram_idle_bw[dram_id]       += inc_idle;
}

void gpgpu_sim::record_kernel_dram_bw_bottlenecks(unsigned kid, unsigned dram_id,
                                                  unsigned long long inc_RCDc,
                                                  unsigned long long inc_RCDWRc,
                                                  unsigned long long inc_WTRc,
                                                  unsigned long long inc_RTWc,
                                                  unsigned long long inc_CCDLc,
                                                  unsigned long long inc_rwq,
                                                  unsigned long long inc_CCDLc_alone,
                                                  unsigned long long inc_WTRc_alone,
                                                  unsigned long long inc_RTWc_alone) {
  if (!kid) return;
  const unsigned n_mem = m_memory_config->m_n_mem;
  if (dram_id >= n_mem) return;

  auto &a = kernel_stats_mut_(kid);

  if (a.dram_RCDc_limit.empty())        a.dram_RCDc_limit.assign(n_mem, 0ULL);
  if (a.dram_RCDWRc_limit.empty())      a.dram_RCDWRc_limit.assign(n_mem, 0ULL);
  if (a.dram_WTRc_limit.empty())        a.dram_WTRc_limit.assign(n_mem, 0ULL);
  if (a.dram_RTWc_limit.empty())        a.dram_RTWc_limit.assign(n_mem, 0ULL);
  if (a.dram_CCDLc_limit.empty())       a.dram_CCDLc_limit.assign(n_mem, 0ULL);
  if (a.dram_rwq_limit.empty())         a.dram_rwq_limit.assign(n_mem, 0ULL);
  if (a.dram_CCDLc_limit_alone.empty()) a.dram_CCDLc_limit_alone.assign(n_mem, 0ULL);
  if (a.dram_WTRc_limit_alone.empty())  a.dram_WTRc_limit_alone.assign(n_mem, 0ULL);
  if (a.dram_RTWc_limit_alone.empty())  a.dram_RTWc_limit_alone.assign(n_mem, 0ULL);

  a.dram_RCDc_limit[dram_id]          += inc_RCDc;
  a.dram_RCDWRc_limit[dram_id]        += inc_RCDWRc;
  a.dram_WTRc_limit[dram_id]          += inc_WTRc;
  a.dram_RTWc_limit[dram_id]          += inc_RTWc;
  a.dram_CCDLc_limit[dram_id]         += inc_CCDLc;
  a.dram_rwq_limit[dram_id]           += inc_rwq;
  a.dram_CCDLc_limit_alone[dram_id]   += inc_CCDLc_alone;
  a.dram_WTRc_limit_alone[dram_id]    += inc_WTRc_alone;
  a.dram_RTWc_limit_alone[dram_id]    += inc_RTWc_alone;
}

void gpgpu_sim::record_kernel_dram_issue_stats(unsigned kid, unsigned dram_id,
                                               unsigned long long inc_row,
                                               unsigned long long inc_col,
                                               unsigned long long inc_total,
                                               unsigned long long inc_two,
                                               unsigned long long inc_ave_mrqs) {
  if (!kid) return;
  const unsigned n_mem = m_memory_config->m_n_mem;
  if (dram_id >= n_mem) return;

  auto &a = kernel_stats_mut_(kid);

  if (a.dram_issued_total_row.empty()) a.dram_issued_total_row.assign(n_mem, 0ULL);
  if (a.dram_issued_total_col.empty()) a.dram_issued_total_col.assign(n_mem, 0ULL);
  if (a.dram_issued_total.empty())     a.dram_issued_total.assign(n_mem, 0ULL);
  if (a.dram_issued_two.empty())       a.dram_issued_two.assign(n_mem, 0ULL);
  if (a.dram_ave_mrqs_sum.empty())     a.dram_ave_mrqs_sum.assign(n_mem, 0ULL);

  a.dram_issued_total_row[dram_id] += inc_row;
  a.dram_issued_total_col[dram_id] += inc_col;
  a.dram_issued_total[dram_id]     += inc_total;
  a.dram_issued_two[dram_id]       += inc_two;
  a.dram_ave_mrqs_sum[dram_id]     += inc_ave_mrqs;
}

void gpgpu_sim::record_kernel_dram_max_mrqs(unsigned kid,
                                            unsigned dram_id,
                                            unsigned long long qlen) {
  if (!kid) return;
  const unsigned n_mem = m_memory_config->m_n_mem;
  if (dram_id >= n_mem) return;

  auto &a = kernel_stats_mut_(kid);

  if (a.dram_max_mrqs.empty()) a.dram_max_mrqs.assign(n_mem, 0ULL);

  if (qlen > a.dram_max_mrqs[dram_id]) a.dram_max_mrqs[dram_id] = qlen;
}

static inline unsigned clamp_bin_0_9(unsigned pct) {
  if (pct > 99) pct = 99;
  return pct / 10;
}

void gpgpu_sim::record_kernel_dram_util_eff_bins_interval(unsigned kid,
                                                          unsigned dram_id) {
  if (!kid) return;
  const unsigned n_mem = m_memory_config->m_n_mem;
  if (dram_id >= n_mem) return;

  auto &a = kernel_stats_mut_(kid);

  if (a.dram_last_n_cmd.empty())      a.dram_last_n_cmd.assign(n_mem, 0ULL);
  if (a.dram_last_n_activity.empty()) a.dram_last_n_activity.assign(n_mem, 0ULL);
  if (a.dram_last_bwutil.empty())     a.dram_last_bwutil.assign(n_mem, 0ULL);

  if (a.dram_util_bins.empty()) a.dram_util_bins.assign(n_mem * 10, 0ULL);
  if (a.dram_eff_bins.empty())  a.dram_eff_bins.assign(n_mem * 10, 0ULL);

  const unsigned long long cur_cmd =
      (a.dram_n_cmd.size() > dram_id) ? a.dram_n_cmd[dram_id] : 0ULL;
  const unsigned long long cur_act =
      (a.dram_n_activity.size() > dram_id) ? a.dram_n_activity[dram_id] : 0ULL;
  const unsigned long long cur_bw =
      (a.dram_bwutil.size() > dram_id) ? a.dram_bwutil[dram_id] : 0ULL;

  const unsigned long long d_cmd = cur_cmd - a.dram_last_n_cmd[dram_id];
  const unsigned long long d_act = cur_act - a.dram_last_n_activity[dram_id];
  const unsigned long long d_bw  = cur_bw  - a.dram_last_bwutil[dram_id];

  a.dram_last_n_cmd[dram_id]      = cur_cmd;
  a.dram_last_n_activity[dram_id] = cur_act;
  a.dram_last_bwutil[dram_id]     = cur_bw;

  // util% = 100 * bwutil / cmd   (same style as visualizer_print does globally)
  unsigned util_pct = 0;
  if (d_cmd) util_pct = (unsigned)((100.0 * (double)d_bw / (double)d_cmd) + 0.5);

  // eff%  = 100 * bwutil / activity
  unsigned eff_pct = 0;
  if (d_act) eff_pct = (unsigned)((100.0 * (double)d_bw / (double)d_act) + 0.5);

  a.dram_util_bins[dram_id * 10 + clamp_bin_0_9(util_pct)]++;
  a.dram_eff_bins [dram_id * 10 + clamp_bin_0_9(eff_pct)]++;
}