

// These are the configuration parameters for NVIDIA A30 (Ampere GA100)
// Sources:
//   https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/products/a30-gpu

#ifndef AMPERE_A30_DEF_H
#define AMPERE_A30_DEF_H

#include "./common/common.h"
#include "./common/deviceQuery.h"

#define L1_SIZE (192 * 1024) // 192 KB per SM (Ampere SM8.0; same for A30 and A100)
                             // https://www.techpowerup.com/gpu-specs/a30-pcie.c3792

#define CLK_FREQUENCY 1440 // Boost clock in MHz for A30 (930 MHz base → 1440 MHz boost) same source

#define ISSUE_MODEL   issue_model::single // Ampere uses a single‐issue datapath per core
#define CORE_MODEL    core_model::subcore  // subcore model (fine‐grained per‐core)  
#define DRAM_MODEL    dram_model::HBM      // A30 uses HBM2e
#define WARP_SCHEDS_PER_SM  4             // 4 warp schedulers per Ampere SM

// number of SASS HMMA per 16×16 PTX WMMA for FP16→FP32 accumulate 
// (third‐gen Tensor Core mapping; same for all Ampere GA100 devices)
#define SASS_hmma_per_PTX_wmma  2         


// see slide 24 from Nvidia at
// https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21730-inside-the-nvidia-ampere-architecture.pdf

// L2 banks and bank width (identical across Ampere generations)
#define L2_BANKS_PER_MEM_CHANNEL  2
#define L2_BANK_WIDTH_in_BYTE     32

#endif // AMPERE_A30_DEF_H
