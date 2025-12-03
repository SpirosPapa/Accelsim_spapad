#include "execution_engine.h"

#include <sys/stat.h>
#include <cassert>
#include <iostream>

#include "../libcuda/gpgpu_context.h"   // gpgpu_context, _cuda_device_id, CUctx_st
#include "../gpgpusim_entrypoint.h"     // gpgpu_ptx_sim_init_perf()
#include "../stream_manager.h"          // stream_manager, operation(bool*)

namespace {
inline void log(const char* s)  { std::cerr << "[engine] " << s << "\n"; }
inline void logs(const std::string& s) { log(s.c_str()); }
} // namespace

ExecutionEngine::~ExecutionEngine() = default;

bool ExecutionEngine::file_exists(const std::string& p) const {
  struct stat st{};
  return ::stat(p.c_str(), &st) == 0;
}

bool ExecutionEngine::initialize(const std::string& ) {
  logs("initialize: create global context and device");
  ctx_ = GPGPU_Context();

  gpgpu_sim* gpu = gpgpu_ptx_sim_init_perf();
  if (!gpu) {
    logs("ERROR: gpgpu_ptx_sim_init_perf() returned null");
    return false;
  }
  if (!ctx_->the_gpgpusim->the_cude_device) {
    ctx_->the_gpgpusim->the_cude_device = new _cuda_device_id(gpu);
  }

  if (!ctx_->the_gpgpusim->the_context) {
    ctx_->the_gpgpusim->the_context = new CUctx_st(ctx_->the_gpgpusim->the_cude_device);
  }
  return true;
}

bool ExecutionEngine::reset() {
  logs("reset: create a fresh CUctx_st");
  if (ctx_ && ctx_->the_gpgpusim && ctx_->the_gpgpusim->the_context) {
    delete ctx_->the_gpgpusim->the_context;
    ctx_->the_gpgpusim->the_context = nullptr;
  }

  _cuda_device_id* dev = ctx_->the_gpgpusim->the_cude_device;
  assert(dev && "_cuda_device_id must exist after initialize()");
  ctx_->the_gpgpusim->the_context = new CUctx_st(dev);
  return true;
}

bool ExecutionEngine::bind(const GPUContext& job) {
  current_trace_dir_ = job.trace_dir;
  logs(std::string("bind: ") + current_trace_dir_);

  const std::string kernels = current_trace_dir_ + "/kernelslist.g";
  if (!file_exists(kernels)) {
    logs("ERROR: kernelslist.g not found in trace_dir");
    return false;
  }

  const std::string trace_cfg_path = current_trace_dir_ + "/trace.config";
  if (file_exists(trace_cfg_path)) {
    tconf_.reset(new trace_config(trace_cfg_path.c_str()));
  } else {
    tconf_.reset(new trace_config()); 
  }
  tparser_.reset(new trace_parser(*tconf_, current_trace_dir_));

  extern stream_manager* g_stream_manager;
  size_t enq = 0;
  while (true) {
    trace_kernel_info_t* kinfo = tparser_->get_next_kernel();
    if (!kinfo) break;
    unsigned stream_id = kinfo->get_stream_id(); 
    g_stream_manager->push(kinfo, stream_id);    
    ++enq;
  }
  if (enq == 0) logs("WARNING: no kernels enqueued from trace");
  return true;
}

bool ExecutionEngine::run_to_completion() {
  logs("run_to_completion: driving stream_manager.operation()");
  extern stream_manager* g_stream_manager;

  bool sim = true;
  while (sim) {
    g_stream_manager->operation(&sim);  
  }
 
  return true;
}

void ExecutionEngine::unbind() {
  logs("unbind");
  tparser_.reset();
  tconf_.reset();
}

void ExecutionEngine::shutdown() {
  logs("shutdown");
  if (ctx_ && ctx_->the_gpgpusim && ctx_->the_gpgpusim->the_context) {
    delete ctx_->the_gpgpusim->the_context;
    ctx_->the_gpgpusim->the_context = nullptr;
  }
}
