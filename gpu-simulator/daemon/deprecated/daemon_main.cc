// gpu-simulator/gpgpu-sim/src/daemon/main.cc
// Persistent Accel-Sim daemon: preloads GPU once, accepts trace jobs via IPC,
// runs them FIFO (one at a time), and "flushes everything" between jobs by
// recreating the sim + stream manager using the already-parsed config.

#include <atomic>
#include <cassert>
#include <csignal>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

#if !defined(_WIN32)
#include <unistd.h>
#endif

// Your daemon helpers
#include "job_queue.h"
#include "ipc_server.h"
#include "daemon_logger.h"   // <- uses your existing logger files

// ---- Accel-Sim / GPGPU-Sim includes (match original trace main) ----
#include "../../libcuda/gpgpu_context.h"
#include "abstract_hardware_model.h"
#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx_ir.h" 
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu-sim/icnt_wrapper.h"
#include "gpgpusim_entrypoint.h"
#include "option_parser.h"
#include "../../../ISA_Def/trace_opcode.h"
#include "../../../trace-driven/trace_driven.h"
#include "../../../trace-parser/trace_parser.h"
#include "accelsim_version.h"

// ======================================================================
// Helpers
// ======================================================================
static void ignore_sigpipe() {
#if !defined(_WIN32)
    std::signal(SIGPIPE, SIG_IGN);
#endif
}

static bool file_exists(const std::string &p) {
    struct stat st {};
    return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}
static bool dir_exists(const std::string &p) {
    struct stat st {};
    return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

static void print_banner() {
    print_splash();
    std::cout << "Accel-Sim Daemon: starting…" << std::endl;
}

// Parse daemon flags, optionally a config directory, and rewrite argv to feed
// Accel-Sim's option parser (-config/-trace_config). We also pull out --socket/--log.
struct DaemonCli {
    std::string socket_path = "/tmp/accelsim-daemon.sock";
    std::string log_path    = "accelsim-daemon.log";
    std::string used_cfg_dir;
};

static void extract_daemon_and_rewrite(
    int &argc, const char **&argv,
    DaemonCli &cli,
    std::vector<std::string> &rebuilt_args_storage
) {
    // env override for socket
    if (const char *p = std::getenv("ACCELSIM_DAEMON_SOCK")) {
        cli.socket_path = p;
    }

    // Find daemon-only flags and an optional config directory
    std::string cfgdir;
    std::vector<std::string> passthrough; // args that go to Accel-Sim parser
    passthrough.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--socket" && i + 1 < argc) {
            cli.socket_path = argv[++i];
            continue;
        }
        if (a == "--log" && i + 1 < argc) {
            cli.log_path = argv[++i];
            continue;
        }
        if (a == "--config-dir" && i + 1 < argc) {
            cfgdir = argv[++i];
            continue;
        }
        passthrough.push_back(a);
    }

    // Positional directory support: if first non-flag arg is a dir
    // and user didn't specify --config-dir, accept it as config dir.
    if (cfgdir.empty() && argc >= 2) {
        std::string maybe_dir = argv[1];
        if (!maybe_dir.empty() && maybe_dir[0] != '-' && dir_exists(maybe_dir)) {
            cfgdir = maybe_dir;
            // drop argv[1] (we'll rebuild below)
            std::vector<std::string> tmp{ passthrough[0] };
            for (size_t i = 1; i < passthrough.size(); ++i) {
                if (passthrough[i] == maybe_dir) continue;
                tmp.push_back(passthrough[i]);
            }
            passthrough.swap(tmp);
        }
    }

    // If a config directory is given, inject -config / -trace_config
    if (!cfgdir.empty()) {
        std::string gcfg = cfgdir + "/gpgpusim.config";
        std::string tcfg = cfgdir + "/trace.config";
        if (!file_exists(gcfg) || !file_exists(tcfg)) {
            std::cerr << "ERROR: config dir '" << cfgdir
                      << "' must contain gpgpusim.config and trace.config\n";
            std::exit(2);
        }

        // Remove any user-specified -config/-trace_config (we'll override)
        std::vector<std::string> cleaned;
        cleaned.push_back(passthrough[0]);
        for (size_t i = 1; i < passthrough.size(); ++i) {
            if (passthrough[i] == "-config" || passthrough[i] == "-trace_config") {
                ++i; // skip its value
                continue;
            }
            cleaned.push_back(passthrough[i]);
        }

        rebuilt_args_storage.clear();
        rebuilt_args_storage.push_back(cleaned[0]);
        rebuilt_args_storage.push_back("-config");
        rebuilt_args_storage.push_back(gcfg);
        rebuilt_args_storage.push_back("-trace_config");
        rebuilt_args_storage.push_back(tcfg);
        for (size_t i = 1; i < cleaned.size(); ++i) {
            rebuilt_args_storage.push_back(cleaned[i]);
        }

        static std::vector<const char *> ptrs;
        ptrs.clear();
        for (auto &s : rebuilt_args_storage) ptrs.push_back(s.c_str());
        argc = static_cast<int>(ptrs.size());
        argv = ptrs.data();

        cli.used_cfg_dir = cfgdir;
    }
}

// ======================================================================
// GPU init/reinit glue (based on original main.cc)
// ======================================================================
static gpgpu_sim *init_gpu_once(
    int argc, const char *argv[],
    gpgpu_context *&ctx_out,
    trace_config &tconfig_out
) {
    srand(1);
    print_banner();

    option_parser_t opp = option_parser_create();

    gpgpu_context *m_gpgpu_context = new gpgpu_context();
    m_gpgpu_context->ptx_reg_options(opp);
    m_gpgpu_context->func_sim->ptx_opcocde_latency_options(opp);
    icnt_reg_options(opp);

    m_gpgpu_context->the_gpgpusim->g_the_gpu_config =
        new gpgpu_sim_config(m_gpgpu_context);
    m_gpgpu_context->the_gpgpusim->g_the_gpu_config->reg_options(opp);
    tconfig_out.reg_options(opp);

    option_parser_cmdline(opp, argc, argv); // parse options (now resolved)
    fprintf(stdout, "GPGPU-Sim: Configuration options:\n\n");
    option_parser_print(opp, stdout);

    // Ensure numeric locale uses dot for decimals
    assert(setlocale(LC_NUMERIC, "C"));

    // Finish config init
    m_gpgpu_context->the_gpgpusim->g_the_gpu_config->init();

    // Create the GPU (trace_gpgpu_sim) and stream manager
    gpgpu_sim *sim = new trace_gpgpu_sim(
        *(m_gpgpu_context->the_gpgpusim->g_the_gpu_config),
        m_gpgpu_context);

    m_gpgpu_context->the_gpgpusim->g_the_gpu = sim;

    m_gpgpu_context->the_gpgpusim->g_stream_manager =
        new stream_manager((m_gpgpu_context->the_gpgpusim->g_the_gpu),
                           m_gpgpu_context->func_sim->g_cuda_launch_blocking);

    m_gpgpu_context->the_gpgpusim->g_simulation_starttime = time((time_t *)NULL);

    sim->init();
    tconfig_out.parse_config();

    ctx_out = m_gpgpu_context;
    return sim;
}

// Recreate sim + stream_manager with same config (flush memory/caches/stats)
static void recreate_sim(gpgpu_context *ctx, gpgpu_sim *&sim) {
    if (ctx->the_gpgpusim->g_stream_manager) {
        ctx->the_gpgpusim->g_stream_manager->stop_all_running_kernels();
    }
    delete ctx->the_gpgpusim->g_stream_manager;
    ctx->the_gpgpusim->g_stream_manager = nullptr;

    delete sim;
    sim = nullptr;

    auto *cfg = ctx->the_gpgpusim->g_the_gpu_config;
    sim = new trace_gpgpu_sim(*cfg, ctx);
    ctx->the_gpgpusim->g_the_gpu = sim;
    sim->init();

    ctx->the_gpgpusim->g_stream_manager =
        new stream_manager(sim, ctx->func_sim->g_cuda_launch_blocking);
}

// ======================================================================
/** Per-job execution: run a trace given by trace_dir.
 *  We treat trace_dir as the path the trace_parser expects (e.g., a commandlist).
 */
static bool run_trace_job(
    const std::string &trace_dir,
    gpgpu_context *ctx,
    gpgpu_sim *sim,
    trace_config &tconf,
    DaemonLogger &log
) {
    try {
        log.logf("Starting simulation for trace '%s'", trace_dir.c_str());

        // release branch: ctor takes const char*
        trace_parser parser(trace_dir.c_str());
        std::vector<trace_command> commands = parser.parse_commandlist_file();

        for (auto &cmd : commands) {
            if (cmd.m_type == command_type::cpu_gpu_mem_copy) {
                size_t addr = 0, bytes = 0;
                parser.parse_memcpy_info(cmd.command_string.c_str(), addr, bytes);
                std::cout << "launching memcpy command : " << cmd.command_string << std::endl;
                log.logf("Memcpy: %zu bytes to 0x%zx", bytes, addr);
                sim->perf_memcpy_to_gpu(addr, bytes);
            } else if (cmd.m_type == command_type::kernel_launch) {
                // release branch: returns kernel_trace_t*
                kernel_trace_t *kt = parser.parse_kernel_info(cmd.command_string.c_str());

                std::cout << "launching kernel command : " << cmd.command_string << std::endl;
                log.logf("Kernel launch: %s  grid(%u,%u,%u) block(%u,%u,%u) smem=%u regs=%u",
                         kt->kernel_name.c_str(),
                         kt->grid_dim_x, kt->grid_dim_y, kt->grid_dim_z,
                         kt->tb_dim_x,   kt->tb_dim_y,   kt->tb_dim_z,
                         kt->shmem, kt->nregs);

                gpgpu_ptx_sim_info info;
                info.smem = kt->shmem;
                info.regs = kt->nregs;
                dim3 gridDim(kt->grid_dim_x, kt->grid_dim_y, kt->grid_dim_z);
                dim3 blockDim(kt->tb_dim_x,  kt->tb_dim_y,  kt->tb_dim_z);

                trace_function_info *func = new trace_function_info(info, ctx);
                trace_kernel_info_t *kinfo =
                    new trace_kernel_info_t(gridDim, blockDim, func, &parser, &tconf, kt);

                sim->launch(kinfo);

                bool did_cycle = false;
                while (true) {
                    if (!sim->active()) break;
                    sim->cycle();
                    did_cycle = true;
                    sim->deadlock_check();
                    if (!sim->active()) {
                        if (sim->cycle_insn_cta_max_hit()) {
                            ctx->the_gpgpusim->g_stream_manager->stop_all_running_kernels();
                        }
                        break;
                    }
                }

                // cleanup & stats (note signatures on this branch)
                delete kinfo->entry();      // function_info*
                delete kinfo;
                parser.kernel_finalizer(kt);
                sim->print_stats(0ULL);

                if (did_cycle) {
                    sim->update_stats();
                    ctx->print_simulation_time();
                }
                if (sim->cycle_insn_cta_max_hit()) {
                    std::cout << "GPGPU-Sim: ** break due to reaching the maximum cycles/instructions **\n";
                    log.logf("Aborted trace due to cycle/insn limit.");
                    break;
                }
            } else {
                std::cerr << "Undefined Command; aborting job." << std::endl;
                log.logf("ERROR: Undefined command in trace; aborting job.");
                return false;
            }
        }

        // Final stats per job
        sim->print_stats(0ULL);
        log.logf("Completed simulation for '%s'", trace_dir.c_str());
        return true;
    } catch (const std::exception &e) {
        log.logf("Exception while running trace '%s': %s", trace_dir.c_str(), e.what());
        return false;
    } catch (...) {
        log.logf("Unknown exception while running trace '%s'", trace_dir.c_str());
        return false;
    }
}


// ======================================================================
// IPC handler: simple line protocol
//   ":status"   -> {"ok":true,"queue":N}
//   ":shutdown" -> {"ok":true}  (also triggers graceful shutdown)
//   any other non-empty line is treated as a trace path to enqueue
// ======================================================================
struct DaemonState {
    JobQueue *queue = nullptr;
    std::atomic<bool> *quit = nullptr;
    DaemonLogger *log = nullptr;
};

static std::string trim(const std::string &s) {
    size_t b = 0, e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r' || s[b] == '\n')) ++b;
    while (e > b && (s[e-1] == ' ' || s[e-1] == '\t' || s[e-1] == '\r' || s[e-1] == '\n')) --e;
    return s.substr(b, e - b);
}

static std::string ipc_handler_line(const std::string &line, DaemonState *st) {
    std::string cmd = trim(line);
    if (cmd.empty()) {
        return R"({"ok":false,"error":"empty"})";
    }
    if (cmd == ":status") {
        size_t n = st->queue->size();
        return std::string("{\"ok\":true,\"queue\":") + std::to_string(n) + "}";
    }
    if (cmd == ":shutdown") {
        st->log->logf("Shutdown requested via IPC.");
        st->quit->store(true);
        // Wake consumer: push sentinel job (since JobQueue has no shutdown())
        st->queue->push(Job{ "__quit__", "" });
        return R"({"ok":true})";
    }

    // Treat as trace path (also use it as job id)
    st->queue->push(Job{ cmd, cmd });
    size_t n = st->queue->size();
    st->log->logf("Added a job to the queue -> %zu job(s) (trace='%s')", n, cmd.c_str());
    return std::string("{\"ok\":true,\"queued\":") + std::to_string(n) + "}";
}

// ======================================================================
// main()
// ======================================================================
int main(int argc, const char **argv) {
    ignore_sigpipe();

    // Parse daemon flags and build Accel-Sim argv
    DaemonCli cli;
    std::vector<std::string> rebuilt_args_storage;
    extract_daemon_and_rewrite(argc, argv, cli, rebuilt_args_storage);

    // Initialize GPU once
    gpgpu_context *ctx = nullptr;
    trace_config tconf;
    gpgpu_sim *sim = init_gpu_once(argc, argv, ctx, tconf);

    DaemonLogger log(cli.log_path);
    if (!cli.used_cfg_dir.empty()) {
        log.logf("Loaded configs from dir: %s", cli.used_cfg_dir.c_str());
    }
    log.logf("Socket path: %s", cli.socket_path.c_str());
    log.logf("Daemon ready. Waiting for jobs…");

    // Infra: queue, quit flag, IPC server
    JobQueue queue;
    std::atomic<bool> quit{ false };

    IpcServer server(cli.socket_path);
    DaemonState state { &queue, &quit, &log };
    auto handler = [&state](const std::string &line) {
        return ipc_handler_line(line, &state);
    };
    if (!server.start(handler)) {
        std::cerr << "Failed to start IPC server on " << cli.socket_path << std::endl;
        return 1;
    }

    // Consumer loop: process jobs FIFO
    while (!quit.load()) {
         // blocks until a job arrives
        Job j;
        if (!queue.pop_blocking(j)) {
            if (quit.load()) break;
            continue;
        }
        if (quit.load() && (j.id == "__quit__" || j.trace_dir.empty())) {
            break; // graceful shutdown
        }

        log.logf("Dequeued job: id='%s' trace='%s' (remaining=%zu)",
                 j.id.c_str(), j.trace_dir.c_str(), queue.size());

        // Run trace
        (void)run_trace_job(j.trace_dir, ctx, sim, tconf, log);

        // "Flush everything": recreate sim + stream manager (preserve parsed config)
        recreate_sim(ctx, sim);
        log.logf("Simulator reset complete. Waiting for next job…");
    }

    // Shutdown
    server.stop();
    log.logf("Daemon stopping.");

    delete ctx->the_gpgpusim->g_stream_manager;
    delete sim;
    delete ctx;
    return 0;
}
