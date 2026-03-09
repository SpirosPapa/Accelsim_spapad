// Copyright (c) 2019, Mahmoud Khairy
// Purdue University
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

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

#include <unordered_set>
#include "icnt_wrapper.h"

#include "local_interconnect.h"
#include "mem_fetch.h"

namespace {

static unsigned packet_kid_from_ptr(void *data) {
  const mem_fetch *mf = static_cast<const mem_fetch *>(data);
  if (!mf) return 0;
  if (!mf->has_kernel_uid()) return 0;
  return mf->get_kernel_uid();
}

static void occ_inc(std::unordered_map<unsigned, unsigned long long> &m,
                    unsigned kid,
                    unsigned long long n = 1) {
  if (!kid || !n) return;
  m[kid] += n;
}

static void occ_dec(std::unordered_map<unsigned, unsigned long long> &m,
                    unsigned kid,
                    unsigned long long n = 1) {
  if (!kid || !n) return;
  auto it = m.find(kid);
  if (it == m.end()) return;
  if (it->second > n) it->second -= n;
  else m.erase(it);
}

static void build_active_kid_sets(
    const std::unordered_map<unsigned, unsigned long long> &in_occ,
    const std::unordered_map<unsigned, unsigned long long> &out_occ,
    std::unordered_set<unsigned> &present_anywhere,
    std::unordered_set<unsigned> &present_in_input) {
  present_anywhere.clear();
  present_in_input.clear();

  for (auto &kv : in_occ) {
    if (kv.second) {
      present_anywhere.insert(kv.first);
      present_in_input.insert(kv.first);
    }
  }
  for (auto &kv : out_occ) {
    if (kv.second) {
      present_anywhere.insert(kv.first);
    }
  }
}

}  // namespace

xbar_router::xbar_router(unsigned router_id, enum Interconnect_type m_type,
                         unsigned n_shader, unsigned n_mem,
                         const struct inct_config& m_localinct_config) {
  m_id = router_id;
  router_type = m_type;
  _n_mem = n_mem;
  _n_shader = n_shader;
  total_nodes = n_shader + n_mem;
  verbose = m_localinct_config.verbose;
  grant_cycles = m_localinct_config.grant_cycles;
  grant_cycles_count = m_localinct_config.grant_cycles;
  in_buffers.resize(total_nodes);
  out_buffers.resize(total_nodes);
  next_node.resize(total_nodes, 0);
  in_buffer_limit = m_localinct_config.in_buffer_limit;
  out_buffer_limit = m_localinct_config.out_buffer_limit;
  arbit_type = m_localinct_config.arbiter_algo;
  next_node_id = 0;
  if (m_type == REQ_NET) {
    active_in_buffers = n_shader;
    active_out_buffers = n_mem;
  } else if (m_type == REPLY_NET) {
    active_in_buffers = n_mem;
    active_out_buffers = n_shader;
  }

  cycles = 0;
  conflicts = 0;
  out_buffer_full = 0;
  in_buffer_full = 0;
  out_buffer_util = 0;
  in_buffer_util = 0;
  packets_num = 0;
  conflicts_util = 0;
  cycles_util = 0;
  reqs_util = 0;
}

xbar_router::~xbar_router() {}

void xbar_router::Push(unsigned input_deviceID, unsigned output_deviceID,
                       void* data, unsigned int size) {
  assert(input_deviceID < total_nodes);
  in_buffers[input_deviceID].push(Packet(data, output_deviceID));
  packets_num++;

  if (router_type == REQ_NET) {
    const unsigned kid = packet_kid_from_ptr(data);
    if (kid) {
      icnt_record_req_net_packets(kid, 1);
      occ_inc(req_in_occ_by_kid, kid, 1);
    }
  }
  if (router_type == REPLY_NET) {
    const unsigned kid = packet_kid_from_ptr(data);
    if (kid) {
      icnt_record_reply_net_packets(kid, 1);
      occ_inc(reply_in_occ_by_kid, kid, 1);
    }
  }
}

void* xbar_router::Pop(unsigned ouput_deviceID) {
  assert(ouput_deviceID < total_nodes);
  void* data = NULL;

  if (!out_buffers[ouput_deviceID].empty()) {
    data = out_buffers[ouput_deviceID].front().data;

    if (router_type == REQ_NET) {
      const unsigned kid = packet_kid_from_ptr(data);
      if (kid) occ_dec(req_out_occ_by_kid, kid, 1);
    }
    if (router_type == REPLY_NET) {
      const unsigned kid = packet_kid_from_ptr(data);
      if (kid) occ_dec(reply_out_occ_by_kid, kid, 1);
    }
    out_buffers[ouput_deviceID].pop();
  }

  return data;
}

void* xbar_router::Peek(unsigned ouput_deviceID) const {
  assert(ouput_deviceID < total_nodes);
  if (out_buffers[ouput_deviceID].empty()) return NULL;
  return out_buffers[ouput_deviceID].front().data;
}


bool xbar_router::Has_Buffer_In(unsigned input_deviceID, unsigned size,
                                bool update_counter) {
  assert(input_deviceID < total_nodes);

  bool has_buffer =
      (in_buffers[input_deviceID].size() + size <= in_buffer_limit);
  if (update_counter && !has_buffer) in_buffer_full++;

  return has_buffer;
}

bool xbar_router::Has_Buffer_Out(unsigned output_deviceID, unsigned size) {
  return (out_buffers[output_deviceID].size() + size <= out_buffer_limit);
}

void xbar_router::Advance() {
  if (arbit_type == NAIVE_RR)
    RR_Advance();
  else if (arbit_type == iSLIP)
    iSLIP_Advance();
  else
    assert(0);
}

void xbar_router::RR_Advance() {
  bool active = false;
  vector<bool> issued(total_nodes, false);
  unsigned conflict_sub = 0;
  unsigned reqs = 0;

  if (router_type == REQ_NET) {
    std::unordered_set<unsigned> active_anywhere, active_input;
    build_active_kid_sets(req_in_occ_by_kid, req_out_occ_by_kid,
                          active_anywhere, active_input);

    for (unsigned kid : active_anywhere) {
      icnt_record_req_net_cycles(kid, 1);
    }
    for (unsigned kid : active_input) {
      icnt_record_req_net_cycles_util(kid, 1);
    }
  }

  if (router_type == REPLY_NET) {
    std::unordered_set<unsigned> active_anywhere, active_input;
    build_active_kid_sets(reply_in_occ_by_kid, reply_out_occ_by_kid,
                          active_anywhere, active_input);

    for (unsigned kid : active_anywhere) {
      icnt_record_reply_net_cycles(kid, 1);
    }
    for (unsigned kid : active_input) {
      icnt_record_reply_net_cycles_util(kid, 1);
    }
  }

  for (unsigned i = 0; i < total_nodes; ++i) {
    unsigned node_id = (i + next_node_id) % total_nodes;

    if (!in_buffers[node_id].empty()) {
      active = true;
      Packet _packet = in_buffers[node_id].front();
      const unsigned kid = packet_kid_from_ptr(_packet.data);

      if (Has_Buffer_Out(_packet.output_deviceID, 1)) {
        if (!issued[_packet.output_deviceID]) {
          out_buffers[_packet.output_deviceID].push(_packet);
          in_buffers[node_id].pop();
          issued[_packet.output_deviceID] = true;
          reqs++;

          if (router_type == REQ_NET && kid) {
            occ_dec(req_in_occ_by_kid, kid, 1);
            occ_inc(req_out_occ_by_kid, kid, 1);
            icnt_record_req_net_reqs_util(kid, 1);
          }

          if (router_type == REPLY_NET && kid) {
            occ_dec(reply_in_occ_by_kid, kid, 1);
            occ_inc(reply_out_occ_by_kid, kid, 1);
            icnt_record_reply_net_reqs_util(kid, 1);
          }
        } else {
          conflict_sub++;

          if (router_type == REQ_NET && kid) {
            icnt_record_req_net_conflicts(kid, 1);
            icnt_record_req_net_conflicts_util(kid, 1);
          }

          if (router_type == REPLY_NET && kid) {
            icnt_record_reply_net_conflicts(kid, 1);
            icnt_record_reply_net_conflicts_util(kid, 1);
          }
        }
      } else {
        out_buffer_full++;

        if (router_type == REQ_NET && kid) {
          icnt_record_req_net_out_buffer_full(kid, 1);
        }

        if (router_type == REPLY_NET && kid) {
          icnt_record_reply_net_out_buffer_full(kid, 1);
        }

        if (issued[_packet.output_deviceID]) {
          conflict_sub++;

          if (router_type == REQ_NET && kid) {
            icnt_record_req_net_conflicts(kid, 1);
            icnt_record_req_net_conflicts_util(kid, 1);
          }

          if (router_type == REPLY_NET && kid) {
            icnt_record_reply_net_conflicts(kid, 1);
            icnt_record_reply_net_conflicts_util(kid, 1);
          }
        }
      }
    }
  }

  next_node_id = next_node_id + 1;
  next_node_id = (next_node_id % total_nodes);

  conflicts += conflict_sub;
  if (active) {
    conflicts_util += conflict_sub;
    cycles_util++;
    reqs_util += reqs;
  }

  if (verbose) {
    printf("%d : cycle %llu : conflicts = %d\n", m_id, cycles, conflict_sub);
    printf("%d : cycle %llu : passing reqs = %d\n", m_id, cycles, reqs);
  }

  for (unsigned i = 0; i < total_nodes; ++i) {
    in_buffer_util += in_buffers[i].size();
    out_buffer_util += out_buffers[i].size();
  }

  if (router_type == REQ_NET) {
    for (auto &kv : req_in_occ_by_kid) {
      if (kv.second) icnt_record_req_net_in_buffer_util(kv.first, kv.second);
    }
    for (auto &kv : req_out_occ_by_kid) {
      if (kv.second) icnt_record_req_net_out_buffer_util(kv.first, kv.second);
    }
  }

  if (router_type == REPLY_NET) {
    for (auto &kv : reply_in_occ_by_kid) {
      if (kv.second) icnt_record_reply_net_in_buffer_util(kv.first, kv.second);
    }
    for (auto &kv : reply_out_occ_by_kid) {
      if (kv.second) icnt_record_reply_net_out_buffer_util(kv.first, kv.second);
    }
  }

  cycles++;
}

// iSLIP algorithm
// McKeown, Nick. "The iSLIP scheduling algorithm for input-queued switches."
// IEEE/ACM transactions on networking 2 (1999): 188-201.
// https://www.cs.rutgers.edu/~sn624/552-F18/papers/islip.pdf
void xbar_router::iSLIP_Advance() {
  bool active = false;

  unsigned conflict_sub = 0;
  unsigned reqs = 0;

  if (router_type == REQ_NET) {
    std::unordered_set<unsigned> active_anywhere, active_input;
    build_active_kid_sets(req_in_occ_by_kid, req_out_occ_by_kid,
                          active_anywhere, active_input);

    for (unsigned kid : active_anywhere) {
      icnt_record_req_net_cycles(kid, 1);
    }
    for (unsigned kid : active_input) {
      icnt_record_req_net_cycles_util(kid, 1);
    }
  }

  if (router_type == REPLY_NET) {
    std::unordered_set<unsigned> active_anywhere, active_input;
    build_active_kid_sets(reply_in_occ_by_kid, reply_out_occ_by_kid,
                          active_anywhere, active_input);

    for (unsigned kid : active_anywhere) {
      icnt_record_reply_net_cycles(kid, 1);
    }
    for (unsigned kid : active_input) {
      icnt_record_reply_net_cycles_util(kid, 1);
    }
  }

  std::set<unsigned> input_nodes;
  std::set<unsigned> destination_set;
  std::set<unsigned> seen_destinations;

  for (unsigned i = 0; i < total_nodes; ++i) {
    if (!in_buffers[i].empty()) {
      input_nodes.insert(i);
      Packet p = in_buffers[i].front();
      unsigned out_node = p.output_deviceID;

      if (seen_destinations.find(out_node) != seen_destinations.end()) {
        conflict_sub++;

        if (router_type == REQ_NET) {
          const unsigned kid = packet_kid_from_ptr(p.data);
          if (kid) {
            icnt_record_req_net_conflicts(kid, 1);
            icnt_record_req_net_conflicts_util(kid, 1);
          }
        }

        if (router_type == REPLY_NET) {
          const unsigned kid = packet_kid_from_ptr(p.data);
          if (kid) {
            icnt_record_reply_net_conflicts(kid, 1);
            icnt_record_reply_net_conflicts_util(kid, 1);
          }
        }
      }

      seen_destinations.insert(out_node);
      destination_set.insert(out_node);
      active = true;
    }
  }

  conflicts += conflict_sub;
  if (active) {
    conflicts_util += conflict_sub;
    cycles_util++;
  }

  for (auto dest : destination_set) {
    if (Has_Buffer_Out(dest, 1)) {
      unsigned start_node = next_node[dest];
      auto it = std::upper_bound(input_nodes.begin(), input_nodes.end(),
                                 start_node);

      for (unsigned j = 0; j < input_nodes.size(); j++, it++) {
        if (it == input_nodes.end()) {
          it = input_nodes.begin();
        }

        unsigned node_id = *it;
        assert(!in_buffers[node_id].empty());
        Packet _packet = in_buffers[node_id].front();

        if (_packet.output_deviceID == dest) {
          const unsigned kid = packet_kid_from_ptr(_packet.data);

          out_buffers[_packet.output_deviceID].push(_packet);
          in_buffers[node_id].pop();
          input_nodes.erase(node_id);

          if (verbose)
            printf("%d : cycle %llu : send req from %d to %d\n", m_id, cycles,
                   node_id, dest - _n_shader);

          if (grant_cycles_count == 1)
            next_node[dest] = (++node_id % total_nodes);

          if (verbose) {
            for (unsigned k = j + 1; k < total_nodes; ++k) {
              unsigned node_id2 = (k + next_node[dest]) % total_nodes;
              if (!in_buffers[node_id2].empty()) {
                Packet _packet2 = in_buffers[node_id2].front();

                if (_packet2.output_deviceID == dest)
                  printf("%d : cycle %llu : cannot send req from %d to %d\n",
                         m_id, cycles, node_id2, dest - _n_shader);
              }
            }
          }

          reqs++;

          if (router_type == REQ_NET && kid) {
            occ_dec(req_in_occ_by_kid, kid, 1);
            occ_inc(req_out_occ_by_kid, kid, 1);
            icnt_record_req_net_reqs_util(kid, 1);
          }

          if (router_type == REPLY_NET && kid) {
            occ_dec(reply_in_occ_by_kid, kid, 1);
            occ_inc(reply_out_occ_by_kid, kid, 1);
            icnt_record_reply_net_reqs_util(kid, 1);
          }

          break;
        }
      }
    } else {
      out_buffer_full++;

      if (router_type == REQ_NET) {
        unsigned blocked_kid = 0;

        unsigned start_node = next_node[dest];
        auto it = std::upper_bound(input_nodes.begin(), input_nodes.end(),
                                   start_node);

        for (unsigned j = 0; j < input_nodes.size(); j++, it++) {
          if (it == input_nodes.end()) {
            it = input_nodes.begin();
          }

          unsigned node_id = *it;
          assert(!in_buffers[node_id].empty());
          Packet _packet = in_buffers[node_id].front();

          if (_packet.output_deviceID == dest) {
            blocked_kid = packet_kid_from_ptr(_packet.data);
            break;
          }
        }

        if (blocked_kid) {
          icnt_record_req_net_out_buffer_full(blocked_kid, 1);
        }
      }

      if (router_type == REPLY_NET) {
        unsigned blocked_kid = 0;

        unsigned start_node = next_node[dest];
        auto it = std::upper_bound(input_nodes.begin(), input_nodes.end(),
                                   start_node);

        for (unsigned j = 0; j < input_nodes.size(); j++, it++) {
          if (it == input_nodes.end()) {
            it = input_nodes.begin();
          }

          unsigned node_id = *it;
          assert(!in_buffers[node_id].empty());
          Packet _packet = in_buffers[node_id].front();

          if (_packet.output_deviceID == dest) {
            blocked_kid = packet_kid_from_ptr(_packet.data);
            break;
          }
        }

        if (blocked_kid) {
          icnt_record_reply_net_out_buffer_full(blocked_kid, 1);
        }
      }
    }
  }

  if (active) {
    reqs_util += reqs;
  }

  if (verbose)
    printf("%d : cycle %llu : grant_cycles = %d\n", m_id, cycles, grant_cycles);

  if (active && grant_cycles_count == 1)
    grant_cycles_count = grant_cycles;
  else if (active)
    grant_cycles_count--;

  if (verbose) {
    printf("%d : cycle %llu : conflicts = %d\n", m_id, cycles, conflict_sub);
    printf("%d : cycle %llu : passing reqs = %d\n", m_id, cycles, reqs);
  }

  for (unsigned i = 0; i < total_nodes; ++i) {
    in_buffer_util += in_buffers[i].size();
    out_buffer_util += out_buffers[i].size();
  }

  if (router_type == REQ_NET) {
    for (auto &kv : req_in_occ_by_kid) {
      if (kv.second) icnt_record_req_net_in_buffer_util(kv.first, kv.second);
    }
    for (auto &kv : req_out_occ_by_kid) {
      if (kv.second) icnt_record_req_net_out_buffer_util(kv.first, kv.second);
    }
  }

  if (router_type == REPLY_NET) {
    for (auto &kv : reply_in_occ_by_kid) {
      if (kv.second) icnt_record_reply_net_in_buffer_util(kv.first, kv.second);
    }
    for (auto &kv : reply_out_occ_by_kid) {
      if (kv.second) icnt_record_reply_net_out_buffer_util(kv.first, kv.second);
    }
  }

  cycles++;
}

bool xbar_router::Busy() const {
  for (unsigned i = 0; i < total_nodes; ++i) {
    if (!in_buffers[i].empty()) return true;

    if (!out_buffers[i].empty()) return true;
  }
  return false;
}

////////////////////////////////////////////////////
/////////////LocalInterconnect/////////////////////

// assume all the packets are one flit
#define LOCAL_INCT_FLIT_SIZE 40

LocalInterconnect* LocalInterconnect::New(
    const struct inct_config& m_localinct_config) {
  LocalInterconnect* icnt_interface = new LocalInterconnect(m_localinct_config);

  return icnt_interface;
}

LocalInterconnect::LocalInterconnect(
    const struct inct_config& m_localinct_config)
    : m_inct_config(m_localinct_config) {
  n_shader = 0;
  n_mem = 0;
  n_subnets = m_localinct_config.subnets;
}

LocalInterconnect::~LocalInterconnect() {
  for (unsigned i = 0; i < m_inct_config.subnets; ++i) {
    delete net[i];
  }
}

void LocalInterconnect::CreateInterconnect(unsigned m_n_shader,
                                           unsigned m_n_mem) {
  n_shader = m_n_shader;
  n_mem = m_n_mem;

  net.resize(n_subnets);
  for (unsigned i = 0; i < n_subnets; ++i) {
    net[i] = new xbar_router(i, static_cast<Interconnect_type>(i), m_n_shader,
                             m_n_mem, m_inct_config);
  }
}

void LocalInterconnect::Init() {
  // empty
  // there is nothing to do
}

void LocalInterconnect::Push(unsigned input_deviceID, unsigned output_deviceID,
                             void* data, unsigned int size) {
  unsigned subnet;
  if (n_subnets == 1) {
    subnet = 0;
  } else {
    if (input_deviceID < n_shader) {
      subnet = 0;
    } else {
      subnet = 1;
    }
  }

  // it should have free buffer
  // assume all the packets have size of one
  // no flits are implemented
  assert(net[subnet]->Has_Buffer_In(input_deviceID, 1));

  net[subnet]->Push(input_deviceID, output_deviceID, data, size);
}

void* LocalInterconnect::Pop(unsigned ouput_deviceID) {
  // 0-_n_shader-1 indicates reply(network 1), otherwise request(network 0)
  int subnet = 0;
  if (ouput_deviceID < n_shader) subnet = 1;

  return net[subnet]->Pop(ouput_deviceID);
}
void* LocalInterconnect::Peek(unsigned ouput_deviceID) const {
  // 0-(n_shader-1) indicates reply(network 1), otherwise request(network 0)
  int subnet = 0;
  if (ouput_deviceID < n_shader) subnet = 1;

  return net[subnet]->Peek(ouput_deviceID);
}


void LocalInterconnect::Advance() {
  for (unsigned i = 0; i < n_subnets; ++i) {
    net[i]->Advance();
  }
}

bool LocalInterconnect::Busy() const {
  for (unsigned i = 0; i < n_subnets; ++i) {
    if (net[i]->Busy()) return true;
  }
  return false;
}

bool LocalInterconnect::HasBuffer(unsigned deviceID, unsigned int size) const {
  bool has_buffer = false;

  if ((n_subnets > 1) && deviceID >= n_shader)  // deviceID is memory node
    has_buffer = net[REPLY_NET]->Has_Buffer_In(deviceID, 1, true);
  else
    has_buffer = net[REQ_NET]->Has_Buffer_In(deviceID, 1, true);

  return has_buffer;
}

void LocalInterconnect::DisplayStats() const {
  printf("Req_Network_injected_packets_num = %lld\n",
         net[REQ_NET]->packets_num);
  printf("Req_Network_cycles = %lld\n", net[REQ_NET]->cycles);
  printf("Req_Network_injected_packets_per_cycle = %12.4f \n",
         (float)(net[REQ_NET]->packets_num) / (net[REQ_NET]->cycles));
  printf("Req_Network_conflicts_per_cycle = %12.4f\n",
         (float)(net[REQ_NET]->conflicts) / (net[REQ_NET]->cycles));
  printf("Req_Network_conflicts_per_cycle_util = %12.4f\n",
         (float)(net[REQ_NET]->conflicts_util) / (net[REQ_NET]->cycles_util));
  printf("Req_Bank_Level_Parallism = %12.4f\n",
         (float)(net[REQ_NET]->reqs_util) / (net[REQ_NET]->cycles_util));
  printf("Req_Network_in_buffer_full_per_cycle = %12.4f\n",
         (float)(net[REQ_NET]->in_buffer_full) / (net[REQ_NET]->cycles));
  printf("Req_Network_in_buffer_avg_util = %12.4f\n",
         ((float)(net[REQ_NET]->in_buffer_util) / (net[REQ_NET]->cycles) /
          net[REQ_NET]->active_in_buffers));
  printf("Req_Network_out_buffer_full_per_cycle = %12.4f\n",
         (float)(net[REQ_NET]->out_buffer_full) / (net[REQ_NET]->cycles));
  printf("Req_Network_out_buffer_avg_util = %12.4f\n",
         ((float)(net[REQ_NET]->out_buffer_util) / (net[REQ_NET]->cycles) /
          net[REQ_NET]->active_out_buffers));

  printf("\n");
  printf("Reply_Network_injected_packets_num = %lld\n",
         net[REPLY_NET]->packets_num);
  printf("Reply_Network_cycles = %lld\n", net[REPLY_NET]->cycles);
  printf("Reply_Network_injected_packets_per_cycle =  %12.4f\n",
         (float)(net[REPLY_NET]->packets_num) / (net[REPLY_NET]->cycles));
  printf("Reply_Network_conflicts_per_cycle =  %12.4f\n",
         (float)(net[REPLY_NET]->conflicts) / (net[REPLY_NET]->cycles));
  printf(
      "Reply_Network_conflicts_per_cycle_util = %12.4f\n",
      (float)(net[REPLY_NET]->conflicts_util) / (net[REPLY_NET]->cycles_util));
  printf("Reply_Bank_Level_Parallism = %12.4f\n",
         (float)(net[REPLY_NET]->reqs_util) / (net[REPLY_NET]->cycles_util));
  printf("Reply_Network_in_buffer_full_per_cycle = %12.4f\n",
         (float)(net[REPLY_NET]->in_buffer_full) / (net[REPLY_NET]->cycles));
  printf("Reply_Network_in_buffer_avg_util = %12.4f\n",
         ((float)(net[REPLY_NET]->in_buffer_util) / (net[REPLY_NET]->cycles) /
          net[REPLY_NET]->active_in_buffers));
  printf("Reply_Network_out_buffer_full_per_cycle = %12.4f\n",
         (float)(net[REPLY_NET]->out_buffer_full) / (net[REPLY_NET]->cycles));
  printf("Reply_Network_out_buffer_avg_util = %12.4f\n",
         ((float)(net[REPLY_NET]->out_buffer_util) / (net[REPLY_NET]->cycles) /
          net[REPLY_NET]->active_out_buffers));
}

void LocalInterconnect::DisplayOverallStats() const {}

unsigned LocalInterconnect::GetFlitSize() const { return LOCAL_INCT_FLIT_SIZE; }

void LocalInterconnect::DisplayState(FILE* fp) const {
  fprintf(fp, "GPGPU-Sim uArch: ICNT:Display State: Under implementation\n");
}
