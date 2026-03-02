// Copyright (c) 2009-2011, Tor M. Aamodt, Ali Bakhoda, George L. Yuan,
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
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

#include "dram_sched.h"
#include "../abstract_hardware_model.h"
#include "gpu-misc.h"
#include "gpu-sim.h"
#include "mem_latency_stat.h"

frfcfs_scheduler::frfcfs_scheduler(const memory_config *config, dram_t *dm,
                                   memory_stats_t *stats) {
  m_config = config;
  m_stats = stats;
  m_num_pending = 0;
  m_num_write_pending = 0;
  m_dram = dm;
  m_queue = new std::list<dram_req_t *>[m_config->nbk];
  m_bins = new std::map<
      unsigned, std::list<std::list<dram_req_t *>::iterator> >[m_config->nbk];
  m_last_row =
      new std::list<std::list<dram_req_t *>::iterator> *[m_config->nbk];
  curr_row_service_time = new unsigned[m_config->nbk];
  row_service_timestamp = new unsigned[m_config->nbk];
  for (unsigned i = 0; i < m_config->nbk; i++) {
    m_queue[i].clear();
    m_bins[i].clear();
    m_last_row[i] = NULL;
    curr_row_service_time[i] = 0;
    row_service_timestamp[i] = 0;
  }
  if (m_config->seperate_write_queue_enabled) {
    m_write_queue = new std::list<dram_req_t *>[m_config->nbk];
    m_write_bins = new std::map<
        unsigned, std::list<std::list<dram_req_t *>::iterator> >[m_config->nbk];
    m_last_write_row =
        new std::list<std::list<dram_req_t *>::iterator> *[m_config->nbk];

    for (unsigned i = 0; i < m_config->nbk; i++) {
      m_write_queue[i].clear();
      m_write_bins[i].clear();
      m_last_write_row[i] = NULL;
    }
  }
  m_mode = READ_MODE;
  //[ADDED]
  m_episode_kid_accesses.resize(m_config->nbk);
}

void frfcfs_scheduler::add_req(dram_req_t *req) {
  if (m_config->seperate_write_queue_enabled && req->data->is_write()) {
    assert(m_num_write_pending < m_config->gpgpu_frfcfs_dram_write_queue_size);
    m_num_write_pending++;
    m_write_queue[req->bk].push_front(req);
    std::list<dram_req_t *>::iterator ptr = m_write_queue[req->bk].begin();
    m_write_bins[req->bk][req->row].push_front(ptr);  // newest reqs to the
                                                      // front
  } else {
    assert(m_num_pending < m_config->gpgpu_frfcfs_dram_sched_queue_size);
    m_num_pending++;
    m_queue[req->bk].push_front(req);
    std::list<dram_req_t *>::iterator ptr = m_queue[req->bk].begin();
    m_bins[req->bk][req->row].push_front(ptr);  // newest reqs to the front
  }
}
void frfcfs_scheduler::data_collection(unsigned int bank) {
  // Bootstrap guard:
  // row_service_timestamp[bank] starts at 0, so the very first
  // "episode_srv" would become gpu_sim_cycle - 0, which is NOT
  // a real row episode (just time since sim start).
  unsigned episode_srv = 0;
  bool valid_episode = (row_service_timestamp[bank] != 0);

  if (valid_episode && m_dram->m_gpu->gpu_sim_cycle > row_service_timestamp[bank]) {
    episode_srv = m_dram->m_gpu->gpu_sim_cycle - row_service_timestamp[bank];

    if (episode_srv > m_stats->max_servicetime2samerow[m_dram->id][bank])
      m_stats->max_servicetime2samerow[m_dram->id][bank] = episode_srv;
  }

  curr_row_service_time[bank] = 0;
  row_service_timestamp[bank] = m_dram->m_gpu->gpu_sim_cycle;

  if (m_stats->concurrent_row_access[m_dram->id][bank] >
      m_stats->max_conc_access2samerow[m_dram->id][bank]) {
    m_stats->max_conc_access2samerow[m_dram->id][bank] =
        m_stats->concurrent_row_access[m_dram->id][bank];
  }

  m_stats->concurrent_row_access[m_dram->id][bank] = 0;
  m_stats->num_activates[m_dram->id][bank]++;

  // [ADDED] flush per-kernel episode contributions
  auto &mp = m_episode_kid_accesses[bank];
  if (!mp.empty()) {
    for (auto &kv : mp) {
      unsigned kid = kv.first;
      unsigned cnt = kv.second;

      m_dram->m_gpu->record_kernel_row_episode_access(kid, m_dram->id, bank, cnt);

      // Only record service time if it was a valid real episode
      if (valid_episode && episode_srv) {
        m_dram->m_gpu->record_kernel_row_episode_servicetime(
            kid, m_dram->id, bank, episode_srv);
      }

     // m_dram->m_gpu->record_kernel_row_locality_episode(kid, m_dram->id, bank, cnt);
    }
    mp.clear();
  }
}


dram_req_t *frfcfs_scheduler::schedule(unsigned bank, unsigned curr_row) {
  // row
  bool rowhit = true;
  std::list<dram_req_t *> *m_current_queue = m_queue;
  std::map<unsigned, std::list<std::list<dram_req_t *>::iterator> >
      *m_current_bins = m_bins;
  std::list<std::list<dram_req_t *>::iterator> **m_current_last_row =
      m_last_row;

  // ---------------- write-queue mode switching ----------------
  if (m_config->seperate_write_queue_enabled) {
    if (m_mode == READ_MODE &&
        (m_num_write_pending >= m_config->write_high_watermark)) {
      m_mode = WRITE_MODE;
    } else if (m_mode == WRITE_MODE &&
               (m_num_write_pending < m_config->write_low_watermark)) {
      m_mode = READ_MODE;
    }
  }

  if (m_mode == WRITE_MODE) {
    m_current_queue = m_write_queue;
    m_current_bins = m_write_bins;
    m_current_last_row = m_last_write_row;
  }

  // ---------------- pick/maintain current row bin ----------------
  if (m_current_last_row[bank] == NULL) {
    if (m_current_queue[bank].empty()) return NULL;

    std::map<unsigned, std::list<std::list<dram_req_t *>::iterator> >::iterator
        bin_ptr = m_current_bins[bank].find(curr_row);

    if (bin_ptr == m_current_bins[bank].end()) {
      // No requests for "curr_row" -> switch to the row of the oldest request
      dram_req_t *tmp = m_current_queue[bank].back();
      bin_ptr = m_current_bins[bank].find(tmp->row);
      assert(bin_ptr != m_current_bins[bank].end());  // where did the request go???

      m_current_last_row[bank] = &(bin_ptr->second);

      // Row episode ended, collect stats and start a new episode
      data_collection(bank);
      rowhit = false;
    } else {
      // Continue on current row
      m_current_last_row[bank] = &(bin_ptr->second);
      rowhit = true;
    }
  }

  // ---------------- choose next request ----------------
  std::list<dram_req_t *>::iterator next = m_current_last_row[bank]->back();
  dram_req_t *req = (*next);

  // ---------------- per-kernel attribution ----------------
  unsigned kid = 0;
  mem_fetch *mf = nullptr;
  if (req && req->data) {
    mf = req->data;
    if (mf->has_kernel_uid()) kid = mf->get_kernel_uid();
  }

  // Keep episode accounting for max_conc / servicetime flush in data_collection()
  if (kid) {
    m_episode_kid_accesses[bank][kid]++;  // +1 per scheduled request in this row-episode
  }

  // Per-kernel locality EXACTLY like vanilla:
  // - row_access++ for every scheduled request
  // - num_activates++ when a new row episode starts (rowhit == false)
  if (kid) {
    m_dram->m_gpu->record_kernel_row_access(kid, m_dram->id, bank);
    if (!rowhit) {
      m_dram->m_gpu->record_kernel_row_activate(kid, m_dram->id, bank);
    }
  }

  // ---------------- rowblp stats (global/vanilla) ----------------
  m_dram->access_num++;

  bool is_write = req->data->is_write();
  if (is_write)
    m_dram->write_num++;
  else
    m_dram->read_num++;

  if (rowhit) {
    m_dram->hits_num++;
    if (is_write)
      m_dram->hits_write_num++;
    else
      m_dram->hits_read_num++;
  }

  m_stats->concurrent_row_access[m_dram->id][bank]++;
  m_stats->row_access[m_dram->id][bank]++;

  // ---------------- remove chosen request from structures ----------------
  m_current_last_row[bank]->pop_back();
  m_current_queue[bank].erase(next);

  if (m_current_last_row[bank]->empty()) {
    m_current_bins[bank].erase(req->row);
    m_current_last_row[bank] = NULL;
  }

#ifdef DEBUG_FAST_IDEAL_SCHED
  if (req)
    printf("%08u : DRAM(%u) scheduling memory request to bank=%u, row=%u\n",
           (unsigned)gpu_sim_cycle, m_dram->id, req->bk, req->row);
#endif

  // ---------------- update pending counters ----------------
  if (m_config->seperate_write_queue_enabled && req->data->is_write()) {
    assert(req != NULL && m_num_write_pending != 0);
    m_num_write_pending--;
  } else {
    assert(req != NULL && m_num_pending != 0);
    m_num_pending--;
  }

  return req;
}

void frfcfs_scheduler::print(FILE *fp) {
  for (unsigned b = 0; b < m_config->nbk; b++) {
    printf(" %u: queue length = %u\n", b, (unsigned)m_queue[b].size());
  }
}

void dram_t::scheduler_frfcfs() {
  unsigned mrq_latency;
  frfcfs_scheduler *sched = m_frfcfs_scheduler;
  while (!mrqq->empty()) {
    dram_req_t *req = mrqq->pop();

    // Power stats
    // if(req->data->get_type() != READ_REPLY && req->data->get_type() !=
    // WRITE_ACK)
    m_stats->total_n_access++;

    if (req->data->get_type() == WRITE_REQUEST) {
      m_stats->total_n_writes++;
    } else if (req->data->get_type() == READ_REQUEST) {
      m_stats->total_n_reads++;
    }

    req->data->set_status(IN_PARTITION_MC_INPUT_QUEUE,
                          m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    sched->add_req(req);
  }

  dram_req_t *req;
  unsigned i;
  for (i = 0; i < m_config->nbk; i++) {
    unsigned b = (i + prio) % m_config->nbk;
    if (!bk[b]->mrq) {
      req = sched->schedule(b, bk[b]->curr_row);

      if (req) {
        req->data->set_status(IN_PARTITION_MC_BANK_ARB_QUEUE,
                              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        prio = (prio + 1) % m_config->nbk;
        bk[b]->mrq = req;
        if (m_config->gpgpu_memlatency_stat) {
          mrq_latency = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle -
                        bk[b]->mrq->timestamp;
          m_stats->tot_mrq_latency += mrq_latency;
          m_stats->tot_mrq_num++;
          bk[b]->mrq->timestamp =
              m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;
          mem_fetch *mf = req->data;
          m_stats->mrq_lat_table[LOGB2(mrq_latency)]++;
          //MY ADDITION
          if (mf && mf->has_kernel_uid()) {
            m_gpu->record_kernel_mrq_lat_bucket(mf->get_kernel_uid(), mrq_latency);
          }
          //
          if (mrq_latency > m_stats->max_mrq_latency) {
            m_stats->max_mrq_latency = mrq_latency;
          }
          // my addition
          if (mf) mf->set_mrq_latency(mrq_latency);
        }

        break;
      }
    }
  }
}
