import opentuner
from opentuner import ConfigurationManipulator, IntegerParameter, Result
import subprocess
import re
import os
import time

ACCEL_SIM = "/accel-sim-framework"

RUN_SIM   = os.path.join(ACCEL_SIM, "util/job_launching/run_simulations.py")
WAIT_SIM  = os.path.join(ACCEL_SIM, "util/job_launching/monitor_func_test.py")
GET_STATS = os.path.join(ACCEL_SIM, "util/job_launching/get_stats.py")
PLOT_CORR = os.path.join(ACCEL_SIM, "util/plotting/plot-correlation.py")

CONFIG_FILE_SRC = os.path.join(
    ACCEL_SIM, "gpu-simulator/gpgpu-sim/configs/tested-cfgs/NVIDIA_A30_MIG_4g.24gb/gpgpusim.config"
)
CONFIG_FILE_RUN = os.path.join(ACCEL_SIM, "sim_run_11.1/gpgpusim.config")

TRACE_DIR   = os.path.join(ACCEL_SIM, "hw_run/traces/device-MIG-4c4406a9-a869-5e8a-bf65-6d1e05c46541/11.1/")
HW_RUN_DIR  = os.path.join(ACCEL_SIM, "hw_run/device-MIG-4c4406a9-a869-5e8a-bf65-6d1e05c46541/11.1/")
SIM_RUN_DIR = os.path.join(ACCEL_SIM, "sim_run_11.1")
CORREL_BASE = os.path.join(ACCEL_SIM, "util/plotting/correl-html")


class GPGPUSimTuner(opentuner.MeasurementInterface):
    def manipulator(self):
        manip = ConfigurationManipulator()

        manip.add_parameter(IntegerParameter("kernel_launch_latency", 8000, 16000))
        manip.add_parameter(IntegerParameter("l2_rop_latency", 140, 200))
        manip.add_parameter(IntegerParameter("dram_latency", 120, 250))
        manip.add_parameter(IntegerParameter("icnt_in_buf", 256, 1024))
        manip.add_parameter(IntegerParameter("icnt_out_buf", 256, 1024))
        manip.add_parameter(IntegerParameter("RCD", 15, 25))
        manip.add_parameter(IntegerParameter("CL", 12, 24))
        manip.add_parameter(IntegerParameter("WL", 2, 6))
        manip.add_parameter(IntegerParameter("WR", 10, 20))
        manip.add_parameter(IntegerParameter("RP", 12, 20))
        manip.add_parameter(IntegerParameter("RAS", 35, 50))

        return manip

    def run(self, desired_result, input, limit):
        cfg = desired_result.configuration.data
        print(f"[TUNER] Starting config: {cfg}")

        ts = int(time.time())
        run_name = f"tuning_A30_MIG_4g24gb_cfg{desired_result.id}_{ts}"

        csv_out = os.path.join(ACCEL_SIM, f"per.kernel.stats_{run_name}.csv")
        correl_file = os.path.join(
            CORREL_BASE,
            "gpc_cycles.A30_MIG_4g.24gb-SASS-LINEAR-RR-256B-FRFCFS.app.raw.csv"
        )

        self.update_config(cfg)

        try:
            subprocess.run([
                RUN_SIM,
                "-T", TRACE_DIR,
                "-C", "A30_MIG_4g.24gb-SASS-LINEAR-RR-256B-FRFCFS",
                "-N", run_name,
                "-B", "GPU_Microbenchmark",
                "--launcher", "local",
                "--cores", "7",
                "--job_mem", "4G"
            ], cwd=ACCEL_SIM, check=True)

            proc = subprocess.run([WAIT_SIM, "-v", "-N", run_name], cwd=ACCEL_SIM)
            if proc.returncode != 0:
                print(f"[TUNER] monitor_func_test exited with {proc.returncode}, continuing.")

            print(f"[TUNER] Running get_stats.py -> {csv_out}")
            subprocess.run(
                f"python3 {GET_STATS} -R -k -K -R {SIM_RUN_DIR} | tee {csv_out}",
                cwd=ACCEL_SIM,
                shell=True,
                check=True
            )

            subprocess.run([PLOT_CORR, "-c", csv_out, "-H", HW_RUN_DIR],
                           cwd=ACCEL_SIM, check=True)

            error_val = self.parse_error(correl_file)

        except subprocess.CalledProcessError as e:
            print(f"[TUNER] Simulation error: {e}")
            error_val = 999.0

        if error_val == 999.0:
            print(f"[TUNER] WARNING: failed or invalid run ({run_name})")

        print(f"[TUNER] Result: Error = {error_val:.2f}% for {cfg}")
        return Result(time=error_val, state='OK', accuracy=error_val)

    def parse_error(self, correl_file):
        if not os.path.exists(correl_file):
            return 999.0
        with open(correl_file, "r") as f:
            line = f.readline()
        match = re.search(r"Err=([\d.]+)%", line)
        return float(match.group(1)) if match else 999.0

    def update_config(self, params):
        with open(CONFIG_FILE_SRC, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            if line.startswith("-gpgpu_kernel_launch_latency"):
                new_lines.append(f"-gpgpu_kernel_launch_latency {params['kernel_launch_latency']}\n")
            elif line.startswith("-gpgpu_l2_rop_latency"):
                new_lines.append(f"-gpgpu_l2_rop_latency {params['l2_rop_latency']}\n")
            elif line.startswith("-dram_latency"):
                new_lines.append(f"-dram_latency {params['dram_latency']}\n")
            elif line.startswith("-gpgpu_dram_timing_opt"):
                new_lines.append(
                    "-gpgpu_dram_timing_opt nbk=16:CCD=1:RRD=5:"
                    f"RCD={params['RCD']}:RAS={params['RAS']}:"
                    f"RP={params['RP']}:RC={params['RAS']+params['RP']}:"
                    f"CL={params['CL']}:WL={params['WL']}:CDLR=4:WR={params['WR']}:"
                    "nbkgrp=4:CCDL=3:RTPL=5\n"
                )
            elif line.startswith("-icnt_in_buffer_limit"):
                new_lines.append(f"-icnt_in_buffer_limit {params['icnt_in_buf']}\n")
            elif line.startswith("-icnt_out_buffer_limit"):
                new_lines.append(f"-icnt_out_buffer_limit {params['icnt_out_buf']}\n")
            else:
                new_lines.append(line)

        for target in [CONFIG_FILE_SRC, CONFIG_FILE_RUN]:
            try:
                with open(target, "w") as f:
                    f.writelines(new_lines)
            except FileNotFoundError:
                pass

if __name__ == "__main__":
    args = opentuner.default_argparser().parse_args()
    GPGPUSimTuner.main(args)
