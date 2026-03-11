#include "accel-sim.h"
#include "accelsim_version.h"

#include <cassert>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <sstream>

namespace {
  struct ParsedMigProfile {
    std::string name;
    unsigned units = 0;   // A30 quarter-GPU units: 1, 2, or 4
  };

  static std::string trim_copy(const std::string &s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    const size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
  }

  static bool parse_a30_mig_profile_token(const std::string &tok,
                                          ParsedMigProfile &out) {
    const std::string t = trim_copy(tok);
    if (t == "1g.6gb") {
      out.name = t;
      out.units = 1;
      return true;
    }
    if (t == "2g.12gb") {
      out.name = t;
      out.units = 2;
      return true;
    }
    if (t == "4g.24gb") {
      out.name = t;
      out.units = 4;
      return true;
    }
    return false;
  }

  static bool parse_a30_mig_tuple(const std::string &spec,
                                  std::vector<ParsedMigProfile> &out,
                                  std::string &err) {
    out.clear();
    err.clear();

    std::stringstream ss(spec);
    std::string item;
    unsigned total_units = 0;

    while (std::getline(ss, item, ',')) {
      item = trim_copy(item);
      if (item.empty()) continue;

      ParsedMigProfile p;
      if (!parse_a30_mig_profile_token(item, p)) {
        err = "unsupported A30 MIG profile token: '" + item + "'";
        out.clear();
        return false;
      }

      total_units += p.units;
      out.push_back(p);
    }

    if (out.empty()) {
      err = "empty -mig tuple";
      return false;
    }

    if (total_units != 4) {
      err = "invalid A30 MIG tuple: the profile units must sum to 4";
      out.clear();
      return false;
    }

    return true;
  }
  static void append_line_to_file(const std::string &path,
                                  const std::string &line) {
    std::ofstream ofs(path, std::ios::app);
    if (!ofs) return;
    ofs << line;
    if (line.empty() || line.back() != '\n') ofs << '\n';
  }
// helper temporarily redirect stdout to a file.
struct ScopedStdoutRedirect {
  int old_fd = -1;
  int new_fd = -1;
  bool active = false;

  ScopedStdoutRedirect(const std::string &path, bool append = false) {
    old_fd = ::dup(STDOUT_FILENO);
    if (old_fd < 0) return;

    int flags = O_CREAT | O_WRONLY | (append ? O_APPEND : O_TRUNC);
    new_fd = ::open(path.c_str(), flags, 0644);
    if (new_fd < 0) {
      ::close(old_fd);
      old_fd = -1;
      return;
    }
    if (::dup2(new_fd, STDOUT_FILENO) < 0) {
      ::close(new_fd);
      ::close(old_fd);
      new_fd = -1;
      old_fd = -1;
      return;
    }

    active = true;
  }

  ~ScopedStdoutRedirect() {
    if (!active) return;
    fflush(stdout);
    if (old_fd >= 0) {
      ::dup2(old_fd, STDOUT_FILENO);
      ::close(old_fd);
    }
    if (new_fd >= 0) {
      ::close(new_fd);
    }
  }
};

}  



// ---------------------------------------------------------------------------
// Existing constructors 
// ---------------------------------------------------------------------------

accel_sim_framework::accel_sim_framework(std::string config_file,
                                         std::string trace_file)
    : m_gpgpu_context(nullptr),
      tracer(""),
      m_gpgpu_sim(nullptr),
      concurrent_kernel_sm(false),
      active(false),
      sim_cycles(false),
      window_size(0),
      commandlist_index(0),
      job_use_all_sms_(true),
      sim_cycles_any_(false),
      trace_config_parsed_(false),
      stopped_due_to_limit_(false),
      next_global_stream_id_(1) {  // ensure deterministic global stream IDs
  std::cout << "Accel-Sim [build " << g_accelsim_version << "]";
  m_gpgpu_context = new gpgpu_context();

  // mimic argv
  unsigned argc = 5;
  const char *argv[] = {"accel-sim.out", "-config", config_file.c_str(),
                        "-trace", trace_file.c_str()};

  m_gpgpu_sim =
      gpgpu_trace_sim_init_perf_model(argc, argv, m_gpgpu_context, &tconfig);
  m_gpgpu_sim->init();

  tracer = trace_parser(tconfig.get_traces_filename());

  tconfig.parse_config();
  trace_config_parsed_ = true;

  init();
}

accel_sim_framework::accel_sim_framework(int argc, const char **argv)
    : m_gpgpu_context(nullptr),
      tracer(""),
      m_gpgpu_sim(nullptr),
      concurrent_kernel_sm(false),
      active(false),
      sim_cycles(false),
      window_size(0),
      commandlist_index(0),
      job_use_all_sms_(true),
      sim_cycles_any_(false),
      trace_config_parsed_(false),
      stopped_due_to_limit_(false),
      next_global_stream_id_(1) {  // <-- ADDED
  std::cout << "Accel-Sim [build " << g_accelsim_version << "]";
  m_gpgpu_context = new gpgpu_context();

  m_gpgpu_sim =
      gpgpu_trace_sim_init_perf_model(argc, argv, m_gpgpu_context, &tconfig);
  m_gpgpu_sim->init();

  tracer = trace_parser(tconfig.get_traces_filename());

  tconfig.parse_config();
  trace_config_parsed_ = true;

  init();
}

accel_sim_framework::accel_sim_framework()
    : m_gpgpu_context(nullptr),
      tracer(""),
      m_gpgpu_sim(nullptr),
      concurrent_kernel_sm(false),
      active(false),
      sim_cycles(false),
      window_size(0),
      commandlist_index(0),
      job_use_all_sms_(true),
      sim_cycles_any_(false),
      trace_config_parsed_(false),
      stopped_due_to_limit_(false),
      next_global_stream_id_(1) {}  // <-- ADDED

// ---------------------------------------------------------------------------
// Legacy single-job code
// ---------------------------------------------------------------------------

void accel_sim_framework::simulation_loop() {
  // for each kernel
  //   - load file
  //   - parse and create kernel info
  //   - launch
  //   - loop until end of kernel execution
  //   - print stats

  while (commandlist_index < commandlist.size() || !kernels_info.empty()) {
    parse_commandlist();

    // Launch all kernels within window that are on a stream that isn't already
    // running
    for (auto k : kernels_info) {
      bool stream_busy = false;
      for (auto s : busy_streams) {
        if (s == k->get_cuda_stream_id()) stream_busy = true;
      }
      if (!stream_busy && m_gpgpu_sim->can_start_kernel() &&
          !k->was_launched()) {
        std::cout << "launching kernel name: " << k->get_name()
                  << " uid: " << k->get_uid()
                  << " cuda_stream_id: " << k->get_cuda_stream_id()
                  << std::endl;
        m_gpgpu_sim->note_kernel_launch(k);   // NEW
        m_gpgpu_sim->launch(k);
        k->set_launched();
        busy_streams.push_back(k->get_cuda_stream_id());
      }
    }

    unsigned finished_kernel_uid = simulate();
    // cleanup finished kernel
    if (finished_kernel_uid || m_gpgpu_sim->cycle_insn_cta_max_hit() ||
        !m_gpgpu_sim->active()) {
      cleanup(finished_kernel_uid);
    }

    if (sim_cycles) {
      m_gpgpu_sim->update_stats();
      m_gpgpu_context->print_simulation_time();
    }

    if (m_gpgpu_sim->cycle_insn_cta_max_hit()) {
      printf(
          "GPGPU-Sim: ** break due to reaching the maximum cycles (or "
          "instructions) **\n");
      fflush(stdout);
      break;
    }
  }
}

void accel_sim_framework::parse_commandlist() {
  // Gulp up as many commands as possible - either cpu_gpu_mem_copy
  // or kernel_launch - until "kernels_info" has reached "window_size"
  // or we have read every command from "commandlist".
  while (kernels_info.size() < window_size &&
         commandlist_index < commandlist.size()) {
    trace_kernel_info_t *kernel_info = NULL;
    if (commandlist[commandlist_index].m_type ==
        command_type::cpu_gpu_mem_copy) {
      size_t addre, Bcount;
      tracer.parse_memcpy_info(
          commandlist[commandlist_index].command_string, addre, Bcount);
      std::cout << "launching memcpy command : "
                << commandlist[commandlist_index].command_string << std::endl;
      m_gpgpu_sim->perf_memcpy_to_gpu(addre, Bcount);
      commandlist_index++;
    } else if (commandlist[commandlist_index].m_type ==
               command_type::kernel_launch) {
      // Read trace header info for window_size number of kernels
      kernel_trace_t *kernel_trace_info = tracer.parse_kernel_info(
          commandlist[commandlist_index].command_string);
      kernel_info = create_kernel_info(kernel_trace_info, m_gpgpu_context,
                                       &tconfig, &tracer);
      kernels_info.push_back(kernel_info);
      std::cout << "Header info loaded for kernel command : "
                << commandlist[commandlist_index].command_string << std::endl;
      commandlist_index++;
    } else {
      // unsupported commands will fail the simulation
      assert(0 && "Undefined Command");
    }
  }
}

void accel_sim_framework::cleanup(unsigned finished_kernel) {
  trace_kernel_info_t *k = NULL;
  unsigned long long finished_kernel_cuda_stream_id = (unsigned long long)-1;
  for (unsigned j = 0; j < kernels_info.size(); j++) {
    k = kernels_info.at(j);
    if (k->get_uid() == finished_kernel ||
        m_gpgpu_sim->cycle_insn_cta_max_hit() || !m_gpgpu_sim->active()) {
      for (unsigned int l = 0; l < busy_streams.size(); l++) {
        if (busy_streams.at(l) == k->get_cuda_stream_id()) {
          finished_kernel_cuda_stream_id = k->get_cuda_stream_id();
          busy_streams.erase(busy_streams.begin() + l);
          break;
        }
      }
      tracer.kernel_finalizer(k->get_trace_info());
      delete k->entry();
      delete k;
      kernels_info.erase(kernels_info.begin() + j);
      if (!m_gpgpu_sim->cycle_insn_cta_max_hit() && m_gpgpu_sim->active())
        break;
    }
  }
  assert(k);
  m_gpgpu_sim->print_stats(finished_kernel_cuda_stream_id);
}

unsigned accel_sim_framework::simulate() {
  unsigned finished_kernel_uid = 0;
  do {
    if (!m_gpgpu_sim->active()) break;

    // performance simulation
    if (m_gpgpu_sim->active()) {
      m_gpgpu_sim->cycle();
      sim_cycles = true;
      m_gpgpu_sim->deadlock_check();
    } else {
      if (m_gpgpu_sim->cycle_insn_cta_max_hit()) {
        m_gpgpu_context->the_gpgpusim->g_stream_manager
            ->stop_all_running_kernels();
        break;
      }
    }

    active = m_gpgpu_sim->active();
    finished_kernel_uid = m_gpgpu_sim->finished_kernel();
  } while (active && !finished_kernel_uid);
  return finished_kernel_uid;
}

trace_kernel_info_t *accel_sim_framework::create_kernel_info(
    kernel_trace_t *kernel_trace_info, gpgpu_context *m_gpgpu_context,
    trace_config *config, trace_parser *parser) {
  gpgpu_ptx_sim_info info;
  info.smem = kernel_trace_info->shmem;
  info.regs = kernel_trace_info->nregs;
  dim3 gridDim(kernel_trace_info->grid_dim_x, kernel_trace_info->grid_dim_y,
               kernel_trace_info->grid_dim_z);
  dim3 blockDim(kernel_trace_info->tb_dim_x, kernel_trace_info->tb_dim_y,
                kernel_trace_info->tb_dim_z);
  trace_function_info *function_info =
      new trace_function_info(info, m_gpgpu_context);
  function_info->set_name(kernel_trace_info->kernel_name.c_str());
  trace_kernel_info_t *kernel_info = new trace_kernel_info_t(
      gridDim, blockDim, function_info, parser, config, kernel_trace_info);

  // Per-job SM restriction 
  if (!job_use_all_sms_ && !job_sm_ids_.empty()) {
    unsigned num_sms = get_num_sms();
    kernel_info->set_allowed_sms(job_sm_ids_, num_sms);
  }

  return kernel_info;
}

gpgpu_sim *accel_sim_framework::gpgpu_trace_sim_init_perf_model(
    int argc, const char *argv[], gpgpu_context *m_gpgpu_context,
    trace_config *m_config) {
  srand(1);
  print_splash();

  option_parser_t opp = option_parser_create();

  m_gpgpu_context->ptx_reg_options(opp);
  m_gpgpu_context->func_sim->ptx_opcocde_latency_options(opp);

  icnt_reg_options(opp);

  m_gpgpu_context->the_gpgpusim->g_the_gpu_config =
      new gpgpu_sim_config(m_gpgpu_context);
  m_gpgpu_context->the_gpgpusim->g_the_gpu_config->reg_options(
      opp);  // register GPU microrachitecture options
  m_config->reg_options(opp);

  option_parser_cmdline(opp, argc, argv);  // parse configuration options
  fprintf(stdout, "GPGPU-Sim: Configuration options:\n\n");
  option_parser_print(opp, stdout);
  // Set the Numeric locale to a standard locale where a decimal point is a
  // "dot" not a "comma" so it does the parsing correctly independent of the
  // system environment variables
  assert(setlocale(LC_NUMERIC, "C"));
  m_gpgpu_context->the_gpgpusim->g_the_gpu_config->init();

  m_gpgpu_context->the_gpgpusim->g_the_gpu = new trace_gpgpu_sim(
      *(m_gpgpu_context->the_gpgpusim->g_the_gpu_config), m_gpgpu_context);

  m_gpgpu_context->the_gpgpusim->g_stream_manager =
      new stream_manager((m_gpgpu_context->the_gpgpusim->g_the_gpu),
                         m_gpgpu_context->func_sim->g_cuda_launch_blocking);

  m_gpgpu_context->the_gpgpusim->g_simulation_starttime = time((time_t *)NULL);

  return m_gpgpu_context->the_gpgpusim->g_the_gpu;
}

// ---------------------------------------------------------------------------
// SM affinity implementations
// ---------------------------------------------------------------------------

unsigned accel_sim_framework::get_num_sms() const {
  assert(m_gpgpu_sim);
  // gpgpu_sim_config::num_shader() normally returns total # of SMs.
  return m_gpgpu_sim->get_config().num_shader();
}

void accel_sim_framework::build_gpu_once(int argc, const char **argv) {
  if (m_gpgpu_sim) return;

  std::cout << "Accel-Sim [build " << g_accelsim_version << "]";
  m_gpgpu_context = new gpgpu_context();

  std::string cfg_path, cfg_dir;
  bool has_trace_opt = false;
  std::string mig_spec;

  // Build a filtered argv for GPGPU-Sim itself:
  // strip our daemon-only "-mig <tuple>" option before option_parser_cmdline().
  std::vector<const char *> av;
  av.reserve(static_cast<size_t>(argc) + 4);

  for (int i = 0; i < argc; ++i) {
    const std::string arg = argv[i];

    if (arg == "-mig") {
      if (i + 1 >= argc) {
        std::cerr << "[MIG] error: -mig requires a tuple, e.g. -mig 1g.6gb,1g.6gb,2g.12gb\n";
        std::exit(1);
      }
      mig_spec = argv[i + 1];
      ++i;  // skip tuple argument too
      continue;
    }

    av.push_back(argv[i]);

    if (arg == "-config" && i + 1 < argc) {
      cfg_path = argv[i + 1];
    }
    if (arg == "-trace" || arg == "-trace_config") {
      has_trace_opt = true;
    }
  }

  if (!cfg_path.empty()) {
    auto pos = cfg_path.find_last_of("/\\");
    cfg_dir = (pos == std::string::npos) ? "." : cfg_path.substr(0, pos);
  }

  std::string trace_cfg_full;
  int argc_local = static_cast<int>(av.size());

  m_gpgpu_sim = gpgpu_trace_sim_init_perf_model(argc_local, av.data(),
                                                m_gpgpu_context, &tconfig);
  m_gpgpu_sim->init();

  // ---------------- MIG startup creation ----------------
  mig_enabled_ = false;
  mig_slice_sms_.clear();
  mig_slice_profiles_.clear();

  if (!mig_spec.empty()) {
    std::vector<ParsedMigProfile> parsed;
    std::string err;
    if (!parse_a30_mig_tuple(mig_spec, parsed, err)) {
      std::cerr << "[MIG] error: " << err << "\n";
      std::exit(1);
    }

    const unsigned total_sms = m_gpgpu_sim->get_config().num_shader();
    const memory_config *mc = m_gpgpu_sim->getMemoryConfig();
    const unsigned total_mem = mc->m_n_mem;
    const unsigned sub_per_chan = mc->m_n_sub_partition_per_memory_channel;
    const unsigned total_sub = mc->m_n_mem_sub_partition;

    if ((total_sms % 4) != 0 || (total_mem % 4) != 0 || (total_sub % 4) != 0) {
      std::cerr << "[MIG] error: current GPU config is not divisible into 4 A30 MIG units"
                << " (sms=" << total_sms
                << ", mem=" << total_mem
                << ", sub=" << total_sub << ")\n";
      std::exit(1);
    }

    const unsigned sms_per_unit = total_sms / 4;
    const unsigned mem_per_unit = total_mem / 4;

    unsigned next_sms = 0;
    unsigned next_mem = 0;

    for (size_t i = 0; i < parsed.size(); ++i) {
      const ParsedMigProfile &p = parsed[i];

      gpgpu_sim::mig_slice_desc_t s;
      s.id = static_cast<unsigned>(i);
      s.name = p.name;

      const unsigned sms_count = p.units * sms_per_unit;
      const unsigned mem_count = p.units * mem_per_unit;

      for (unsigned sid = next_sms; sid < next_sms + sms_count; ++sid) {
        s.sms.push_back(sid);
      }

      for (unsigned mp = next_mem; mp < next_mem + mem_count; ++mp) {
        s.mem_partitions.push_back(mp);
        for (unsigned sp = 0; sp < sub_per_chan; ++sp) {
          s.sub_partitions.push_back(mp * sub_per_chan + sp);
        }
      }

      m_gpgpu_sim->register_mig_slice(s);

      mig_slice_sms_.push_back(s.sms);
      mig_slice_profiles_.push_back(s.name);

      std::cout << "[MIG] slice_id=" << s.id
                << " profile=" << s.name
                << " sms={";
      for (size_t j = 0; j < s.sms.size(); ++j) {
        if (j) std::cout << ",";
        std::cout << s.sms[j];
      }
      std::cout << "} mem_partitions={";
      for (size_t j = 0; j < s.mem_partitions.size(); ++j) {
        if (j) std::cout << ",";
        std::cout << s.mem_partitions[j];
      }
      std::cout << "} sub_partitions={";
      for (size_t j = 0; j < s.sub_partitions.size(); ++j) {
        if (j) std::cout << ",";
        std::cout << s.sub_partitions[j];
      }
      std::cout << "}\n";

      next_sms += sms_count;
      next_mem += mem_count;
    }

    mig_enabled_ = true;
    std::cout << "[MIG] enabled with " << mig_slice_sms_.size()
              << " slices from tuple: " << mig_spec << "\n";
  }
}

void accel_sim_framework::load_trace(const std::string &trace_path) {
  // Bind parser to the incoming job’s trace dir / commandlist (legacy path)
  tracer = trace_parser(trace_path.c_str());
  if (!trace_config_parsed_) {
    tconfig.parse_config();
    trace_config_parsed_ = true;
  }
  init_job_state_();
}

void accel_sim_framework::init_job_state_() {
  active = false;
  sim_cycles = false;
  commandlist_index = 0;

  assert(m_gpgpu_context);
  assert(m_gpgpu_sim);

  concurrent_kernel_sm =
      m_gpgpu_sim->getShaderCoreConfig()->gpgpu_concurrent_kernel_sm;
  window_size = concurrent_kernel_sm
                    ? m_gpgpu_sim->get_config().get_max_concurrent_kernel()
                    : 1;
  assert(window_size > 0);

  busy_streams.clear();
  kernels_info.clear();
  commandlist = tracer.parse_commandlist_file();
  kernels_info.reserve(window_size);
}

void accel_sim_framework::run_one_job() {
  simulation_loop();
}

void accel_sim_framework::soft_reset_for_next_job() {
  active = false;
  sim_cycles = false;
  commandlist_index = 0;
  busy_streams.clear();
  kernels_info.clear();
  commandlist.clear();
  job_use_all_sms_ = true;
  job_sm_ids_.clear();
  // NOTE: we intentionally do not reset m_gpgpu_sim itself here.
}

void accel_sim_framework::configure_sm_mask_for_next_job(
    bool use_all_sms, const std::vector<unsigned> &sm_ids) {
  if (use_all_sms || sm_ids.empty()) {
    job_use_all_sms_ = true;
    job_sm_ids_.clear();
  } else {
    job_use_all_sms_ = false;
    job_sm_ids_ = sm_ids;
  }
}

// ---------------------------------------------------------------------------
// NEW multi-job daemon API implementation
// ---------------------------------------------------------------------------

void accel_sim_framework::start_job(const std::string &trace_dir,
                                    const std::string &out_dir,
                                    bool use_all_sms,
                                    const std::vector<unsigned> &sm_ids,
                                    int slice_id,
                                    const std::string &job_id) {
  assert(m_gpgpu_sim && "build_gpu_once must be called before start_job");

  if (!trace_config_parsed_) {
    tconfig.parse_config();
    trace_config_parsed_ = true;
  }

  JobRuntime job;
  job.job_uid = next_job_uid_++;
  job.job_id      = job_id;
  job.trace_dir   = trace_dir;
  job.out_dir     = out_dir;
  job.use_all_sms = use_all_sms;
  job.sm_ids      = sm_ids;
  job.slice_id    = slice_id;
  job.global_mem_offset = next_job_global_offset_;
  next_job_global_offset_ += kJobAddrStride;
  job.done        = false;

  std::string kernelslist_path = trace_dir;
  if (!kernelslist_path.empty() && kernelslist_path.back() != '/')
    kernelslist_path += "/";
  kernelslist_path += "kernelslist.g";

  std::cout << "[fw] start_job id=" << job.job_id
            << " trace_dir=" << job.trace_dir
            << " kernelslist=" << kernelslist_path
            << " use_all_sms=" << (job.use_all_sms ? 1 : 0)
            << " slice_id=" << job.slice_id
            << " global_mem_offset=0x" << std::hex << job.global_mem_offset << std::dec
            << " sm_ids={";
  for (size_t i = 0; i < job.sm_ids.size(); ++i) {
    if (i) std::cout << ",";
    std::cout << job.sm_ids[i];
  }
  std::cout << "}" << std::endl;
  m_gpgpu_sim->reset_cluster_round_robin();
  job.parser = std::make_unique<trace_parser>(kernelslist_path.c_str());
  job.commandlist       = job.parser->parse_commandlist_file();
  job.commandlist_index = 0;

  job.concurrent_kernel_sm =
      m_gpgpu_sim->getShaderCoreConfig()->gpgpu_concurrent_kernel_sm;
  job.window_size = job.concurrent_kernel_sm
                        ? m_gpgpu_sim->get_config().get_max_concurrent_kernel()
                        : 1;
  if (job.window_size == 0) job.window_size = 1;

  job.kernels_info.reserve(job.window_size);
  job.busy_streams.clear();
  job.stream_remap.clear();

  std::cout << "[fw] start_job id=" << job.job_id
            << " cmds=" << job.commandlist.size()
            << " window_size=" << job.window_size
            << std::endl;

  jobs_.push_back(std::move(job));
}

bool accel_sim_framework::has_active_work() const {
  if (!m_gpgpu_sim) return false;

  if (m_gpgpu_sim->active()) return true;

  // If any job is not marked done, there is still work to do
  for (const auto &job : jobs_) {
    if (!job.done) return true;
  }
  return false;
}

unsigned long long accel_sim_framework::remap_stream_id(
    JobRuntime &job, unsigned long long local_sid) {

  auto it = job.stream_remap.find(local_sid);
  if (it != job.stream_remap.end()) {
    return it->second;
  }
  unsigned long long global_sid = next_global_stream_id_++;
  job.stream_remap[local_sid] = global_sid;
  return global_sid;
}

void accel_sim_framework::parse_and_launch_for_job(size_t job_index) {
  if (job_index >= jobs_.size()) return;
  JobRuntime &job = jobs_[job_index];
  if (job.done) return;
  if (!job.parser) return;
  trace_parser &parser = *job.parser;

  // 1) Parse more commands into kernels_info up to job.window_size
  while (job.kernels_info.size() < job.window_size &&
         job.commandlist_index < job.commandlist.size()) {
    const trace_command &cmd = job.commandlist[job.commandlist_index];
    trace_kernel_info_t *kernel_info = nullptr;

    if (cmd.m_type == command_type::cpu_gpu_mem_copy) {
      size_t addre = 0, Bcount = 0;
      parser.parse_memcpy_info(cmd.command_string, addre, Bcount);
      std::cout << "[fw] job " << job.job_id
                << " launching memcpy: " << cmd.command_string
                << " offset=0x" << std::hex << job.global_mem_offset << std::dec
                << std::endl;
      m_gpgpu_sim->perf_memcpy_to_gpu(addre, Bcount,
                                      job.global_mem_offset,
                                      job.slice_id);
      job.commandlist_index++;
    } else if (cmd.m_type == command_type::kernel_launch) {
      kernel_trace_t *kernel_trace_info =
          parser.parse_kernel_info(cmd.command_string);

      kernel_info = create_kernel_info(kernel_trace_info, m_gpgpu_context,
                                      &tconfig, &parser);

      // If this job is bound to a MIG slice, register slice ownership first
      // and let the slice define the SM mask.
      // Otherwise keep the old explicit-SM behavior.
      if (job.slice_id >= 0) {
        m_gpgpu_sim->register_kernel_slice(kernel_info->get_uid(),
                                          (unsigned)job.slice_id);

        unsigned num_sms = get_num_sms();
        kernel_info->set_allowed_sms(job.sm_ids, num_sms);
      } else if (!job.use_all_sms && !job.sm_ids.empty()) {
        unsigned num_sms = get_num_sms();
        kernel_info->set_allowed_sms(job.sm_ids, num_sms);
      }
      m_gpgpu_sim->register_kernel_global_offset(kernel_info->get_uid(),
                                                 job.global_mem_offset);

      job.kernels_info.push_back(kernel_info);
      const unsigned kid = kernel_info->get_uid();
      const std::string out_path =
          job.out_dir + "/kernel_" + std::to_string(kid) + ".out";

      // Register path immediately so later low-level code can append to it.
      m_gpgpu_sim->register_kernel_log_file(kid, out_path);

      // Start clean for this kernel file.
      {
        std::ofstream trunc(out_path, std::ios::trunc);
      }

      {
        std::ostringstream oss;
        oss << "[fw] job " << job.job_id
            << " header loaded for kernel: " << cmd.command_string
            << " (uid=" << kid << ")";
        append_line_to_file(out_path, oss.str());
      }
      std::cout << "[fw] job " << job.job_id
                << " header loaded for kernel: " << cmd.command_string
                << " (uid=" << kernel_info->get_uid() << ")" << std::endl;

      job.commandlist_index++;
    } else {
      assert(0 && "Undefined Command");
    }
  }

  // 2) Try to launch kernels for this job
  for (auto *k : job.kernels_info) {
    if (k->was_launched()) continue;

    unsigned long long local_sid  = k->get_cuda_stream_id();
    unsigned long long global_sid = remap_stream_id(job, local_sid);

    bool stream_busy = false;
    for (auto s : job.busy_streams) {
      if (s == global_sid) {
        stream_busy = true;
        break;
      }
    }
    if (stream_busy) {
      std::cout << "[fw] job " << job.job_id
                << " NOT launching kernel uid=" << k->get_uid()
                << " global_stream_id=" << global_sid
                << " (stream busy)" << std::endl;
      continue;
    }

    if (!m_gpgpu_sim->can_start_kernel()) {
      std::cout << "[fw] job " << job.job_id
                << " cannot start kernel uid=" << k->get_uid()
                << " (GPU says cannot start)" << std::endl;
      break;
    }

    // Actually remap the stream id in  kernel trace
    k->set_cuda_stream_id(global_sid);
    {
      const unsigned kid = k->get_uid();
      const std::string out_path =
          job.out_dir + "/kernel_" + std::to_string(kid) + ".out";

      std::ostringstream oss;
      oss << "[fw] launching kernel (job " << job.job_id << ") name: "
          << k->get_name() << " uid: " << k->get_uid()
          << " local_stream_id: " << local_sid
          << " global_stream_id: " << global_sid;
      append_line_to_file(out_path, oss.str());
    }
    m_gpgpu_sim->note_kernel_launch(k);   // NEW (needs the *global* stream id)
    m_gpgpu_sim->launch(k);
    k->set_launched();
    job.busy_streams.push_back(global_sid);

    unsigned uid = k->get_uid();
    kernel_uid_to_job_[uid] = job_index;

  }

  // Job has only memcpys and no kernels
  if (job.commandlist_index >= job.commandlist.size() &&
      job.kernels_info.empty() &&
      job.busy_streams.empty() && !job.done) {
    std::cout << "[fw] job " << job.job_id
              << " has no kernels (only memcpys) -> marking done" << std::endl;
    job.done = true;
    finished_job_ids_.push_back(job.job_id);
  }
}

void accel_sim_framework::cleanup_finished_kernel(unsigned finished_kernel_uid) {
  if (!finished_kernel_uid) return;

  // Map kernel uid -> job index
  auto it = kernel_uid_to_job_.find(finished_kernel_uid);
  if (it == kernel_uid_to_job_.end()) return;

  size_t job_index = it->second;
  if (job_index >= jobs_.size()) {
    kernel_uid_to_job_.erase(it);
    return;
  }

  JobRuntime &job = jobs_[job_index];
  trace_kernel_info_t *k = nullptr;
  unsigned long long finished_stream_id = (unsigned long long)-1;

  // Info used for per-kernel stats printing
  std::string finished_kernel_name;
  int         finished_kernel_uid_int = -1;

  // Based on legacy cleanup().
  for (size_t j = 0; j < job.kernels_info.size(); ++j) {
    trace_kernel_info_t *cand = job.kernels_info[j];
    if (cand->get_uid() == finished_kernel_uid) {
      k = cand;

      // Capture name + uid BEFORE deleting k.
      finished_kernel_name    = k->get_name();
      finished_kernel_uid_int = (int)k->get_uid();

      // Remove the stream from job.busy_streams
      unsigned long long sid = k->get_cuda_stream_id();
      for (size_t l = 0; l < job.busy_streams.size(); ++l) {
        if (job.busy_streams[l] == sid) {
          finished_stream_id = sid;
          job.busy_streams.erase(job.busy_streams.begin() + l);
          break;
        }
      }

      job.parser->kernel_finalizer(k->get_trace_info());
      m_gpgpu_sim->note_kernel_completion(k);   // MY ADDITION
      delete k->entry();
      delete k;
      job.kernels_info.erase(job.kernels_info.begin() + j);
      break;
    }
  }

  kernel_uid_to_job_.erase(it);



  if (!finished_kernel_name.empty() &&
      finished_stream_id != (unsigned long long)-1) {
    std::string out_path = job.out_dir + "/kernel_" +
                           std::to_string(finished_kernel_uid) + ".out";

    // Redirect stdout so gpu_print_stat prints into job-specific file
    ScopedStdoutRedirect redirect(out_path, /*append=*/true);

    kernel_stats_view_t view =
        m_gpgpu_sim->make_kernel_stats_view(finished_kernel_uid);

    m_gpgpu_sim->print_stats(
        finished_stream_id,
        &view,
        finished_kernel_name.c_str(),
        finished_kernel_uid_int);

    // Duplicate the global simulation footer into this kernel file too.
    // These are still global snapshots, not kernel-private values.
    m_gpgpu_context->print_simulation_time();
    printf("GPGPU-Sim: *** simulation thread exiting ***\n");
    printf("GPGPU-Sim: *** exit detected ***\n");

    m_gpgpu_sim->clear_kernel_stats(finished_kernel_uid);
    m_gpgpu_sim->unregister_kernel_log_file(finished_kernel_uid);
    m_gpgpu_sim->unregister_kernel_global_offset(finished_kernel_uid);
  }

  // Mark job as done when everything for this job has drained
  if (job.commandlist_index >= job.commandlist.size() &&
      job.kernels_info.empty() &&
      job.busy_streams.empty() && !job.done) {
    job.done = true;
    finished_job_ids_.push_back(job.job_id);
    std::cout << "[fw] job " << job.job_id << " completed.\n";
  }
}

void accel_sim_framework::step_one_cycle() {
  if (!m_gpgpu_sim) return;

  sim_cycles_any_ = false;

  // 1) Feed new kernels for each active job
  for (size_t idx = 0; idx < jobs_.size(); ++idx) {
    parse_and_launch_for_job(idx);
  }

  // 2) Do one GPU cycle if active
  if (m_gpgpu_sim->active()) {
    m_gpgpu_sim->cycle();
    sim_cycles_any_ = true;
    m_gpgpu_sim->deadlock_check();
  } else {
    if (m_gpgpu_sim->cycle_insn_cta_max_hit() && !stopped_due_to_limit_) {
      printf(
          "GPGPU-Sim: ** break due to reaching the maximum cycles (or "
          "instructions) **\n");
      fflush(stdout);
      m_gpgpu_context->the_gpgpusim->g_stream_manager
          ->stop_all_running_kernels();
      stopped_due_to_limit_ = true;

      // Mark all remaining jobs as finished due to limit
      for (auto &job : jobs_) {
        if (!job.done) {
          job.done = true;
          finished_job_ids_.push_back(job.job_id);
        }
      }
      kernel_uid_to_job_.clear();
    }
  }

  // 3) Check for a finished kernel
  unsigned finished_kernel_uid = m_gpgpu_sim->finished_kernel();
  if (finished_kernel_uid) {
    cleanup_finished_kernel(finished_kernel_uid);
  }
}

std::vector<std::string> accel_sim_framework::collect_finished_jobs() {
  std::vector<std::string> out;
  out.swap(finished_job_ids_);
  return out;
}
