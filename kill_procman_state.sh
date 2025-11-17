#!/bin/bash
pkill -9 accel-sim.out
pkill -9 -f procman.py
rm -f /accel-sim-framework/util/job_launching/procman/procman.*.pickle*
echo "Procman and all accel-sim jobs cleared. Next run will start fresh."
