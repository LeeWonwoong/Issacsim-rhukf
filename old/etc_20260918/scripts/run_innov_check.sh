#!/usr/bin/env bash
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true; set -u
cd /home/acsl/projects/Issacsim-rhukf
export WIND_START=80 WIND_END=220 SWEEP_ATK_START=140 SWEEP_ATK_END=260
mkdir -p results_innov_check
SWEEP_ATTACK_TYPE=loe_combined setsid python3 online_rl_main.py --sweep --headless \
  --sweep-mode torque --torque-yaw-ratio 0.0 \
  --capture-mode hijack \
  --capture-disturbances "none:0,wind_turbulence:8" \
  --capture-biases "0.0,3.05" \
  --capture-patterns "hover,aggressive" \
  --episodes 4 --ramp 0.0 --speed 10 --outdir results_innov_check \
  > results_innov_check/run.log 2>&1 &
echo "innov check pid $!"
