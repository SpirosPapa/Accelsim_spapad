#include "accel-sim.h"
#include "accelsim_version.h"

accel_sim_framework::accel_sim_framework(std::string config_file,
                                          std::string trace_file) {
  std::cout << "Accel-Sim [build " << g_accelsim_version << "]";
  m_gpgpu_context = new gpgpu_context();

  // mimic argv
  unsigned argc = 5;
  const char *argv[] = {"accel-sim.out", "-config", config_file.c_str(),
                        "-trace", trace_file.c_str()};

  gpgpu_sim *m_gpgpu_sim =
      gpgpu_trace_sim_init_perf_model(argc, argv, m_gpgpu_context, &tconfig);
  m_gpgpu_sim->init();

  tracer = trace_parser(tconfig.get_traces_filename());

  tconfig.parse_config();

  init();
}

accel_sim_framework::accel_sim_framework(int argc, const char **argv) {
  std::cout << "Accel-Sim [build " << g_accelsim_version << "]";
  m_gpgpu_context = new gpgpu_context();

  m_gpgpu_sim =
      gpgpu_trace_sim_init_perf_model(argc, argv, m_gpgpu_context, &tconfig);
  m_gpgpu_sim->init();

  tracer = trace_parser(tconfig.get_traces_filename());

  tconfig.parse_config();

  init();
}

void accel_sim_framework::simulation_loop() {
  // for each kernel
  // load file
  // parse and create kernel info
  // launch
  // while loop till the end of the end kernel execution
  // prints stats

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
  // gulp up as many commands as possible - either cpu_gpu_mem_copy
  // or kernel_launch - until the vector "kernels_info" has reached
  // the window_size or we have read every command from commandlist
  while (kernels_info.size() < window_size && commandlist_index < commandlist.size()) {
    trace_kernel_info_t *kernel_info = NULL;
    if (commandlist[commandlist_index].m_type == command_type::cpu_gpu_mem_copy) {
      size_t addre, Bcount;
      tracer.parse_memcpy_info(commandlist[commandlist_index].command_string, addre, Bcount);
      std::cout << "launching memcpy command : "
                << commandlist[commandlist_index].command_string << std::endl;
      m_gpgpu_sim->perf_memcpy_to_gpu(addre, Bcount);
      commandlist_index++;
    } else if (commandlist[commandlist_index].m_type == command_type::kernel_launch) {
      // Read trace header info for window_size number of kernels
      kernel_trace_t *kernel_trace_info =
          tracer.parse_kernel_info(commandlist[commandlist_index].command_string);
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
  unsigned long long finished_kernel_cuda_stream_id = -1;
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

trace_kernel_info_t *accel_sim_framework::create_kernel_info(kernel_trace_t *kernel_trace_info,
                                        gpgpu_context *m_gpgpu_context,
                                        trace_config *config,
                                        trace_parser *parser) {
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


/// NEW IMPLEMENTATIONS


// void accel_sim_framework::build_gpu_once(int argc, const char **argv) {
//   if (m_gpgpu_sim) return;               // already built

//   std::cout << "Accel-Sim [build " << g_accelsim_version << "]";
//   m_gpgpu_context = new gpgpu_context();

//   // Build the simulator from -config (argv may or may not include -trace).
//   m_gpgpu_sim =
//       gpgpu_trace_sim_init_perf_model(argc, argv, m_gpgpu_context, &tconfig);
//   m_gpgpu_sim->init();
// }

void accel_sim_framework::build_gpu_once(int argc, const char **argv) {
  if (m_gpgpu_sim) return;

  std::cout << "Accel-Sim [build " << g_accelsim_version << "]";
  m_gpgpu_context = new gpgpu_context();

  // Detect -config path and synthesize a matching -trace_config if missing
  std::string cfg_path, cfg_dir;
  bool has_trace_opt = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "-config" && i + 1 < argc) {
      cfg_path = argv[i + 1];
    }
    if (std::string(argv[i]) == "-trace" || std::string(argv[i]) == "-trace_config") {
      has_trace_opt = true;
    }
  }
  // derive directory of gpgpusim.config
  if (!cfg_path.empty()) {
    auto pos = cfg_path.find_last_of("/\\");
    cfg_dir = (pos == std::string::npos) ? "." : cfg_path.substr(0, pos);
  }

  // Build a local argv that adds -trace_config <cfg_dir>/trace.config if needed
  std::vector<const char*> av;
  av.reserve(static_cast<size_t>(argc) + 4);
  for (int i = 0; i < argc; ++i) av.push_back(argv[i]);
  std::string trace_cfg_full;
  // if (!has_trace_opt && !cfg_dir.empty()) {
  //   trace_cfg_full = cfg_dir + "/trace.config";
  //   av.push_back("-trace_config");
  //   av.push_back(trace_cfg_full.c_str());
  // }
  int argc_local = static_cast<int>(av.size());

  m_gpgpu_sim = gpgpu_trace_sim_init_perf_model(argc_local, av.data(),
                                                 m_gpgpu_context, &tconfig);
  m_gpgpu_sim->init();
}




void accel_sim_framework::load_trace(const std::string& trace_path) {
  // Bind parser to the incoming job’s trace dir / commandlist
  //tconfig.set_traces_filename(trace_path.c_str());  
  tracer = trace_parser(trace_path.c_str());
  tconfig.parse_config();      // reads trace.config associated with this trace
  init_job_state_();           // sets window, loads commandlist, clears vectors
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
  simulation_loop();           // your existing loop (launch → cycle → cleanup)
}

void accel_sim_framework::soft_reset_for_next_job() {
  // Precondition: previous job is finished and simulation_loop() returned.
  // Minimal viable reset for serial jobs:
  active = false;
  sim_cycles = false;
  commandlist_index = 0;
  busy_streams.clear();
  kernels_info.clear();
  commandlist.clear();

  // Optional (recommended): if available in your tree, also zero stats/counters
  // m_gpgpu_sim->get_stats()->reset();          // if such method exists
  // m_gpgpu_sim->reset_cycle_counters();        // if such method exists
  // m_gpgpu_context->the_gpgpusim->g_stream_manager->reset(); // if available
}

accel_sim_framework::accel_sim_framework()
  : m_gpgpu_context(nullptr),
    tracer(""),                  // or a default-constructed parser
    m_gpgpu_sim(nullptr),
    concurrent_kernel_sm(false),
    active(false),
    sim_cycles(false),
    window_size(0),
    commandlist_index(0) {
  // intentionally empty; build later via build_gpu_once()
}
