#!/usr/bin/env bash
# fineband (2026-09-08) — 정밀 밴드: δ 0.75~0.82(0.01) × ws 6~10(1) × track/dhover3 × 3대표패턴 × 3ep
#   대표패턴 = waypoint(순한)·aggressive(급기동)·scurve(3축). 바람창 60-430.
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07 DISPLAY=:1; unset MPC_ACC_HOR
export EP_MAX_STEPS=450 SWEEP_ATTACK_TYPE=tilt
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
kill_all; OUT=results_fineband; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] fineband START" | tee isaac_fineband.log
WIND_START=60 WIND_END=430 SWEEP_ATK_START=180 SWEEP_ATK_END=380 CAPTURE_POLICIES="track,dhover3" \
timeout 57600 python3 -u online_rl_main.py --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns waypoint,aggressive,scurve \
  --capture-disturbances wind_turbulence:6,wind_turbulence:7,wind_turbulence:8,wind_turbulence:9,wind_turbulence:10 \
  --capture-biases 3.27,3.314,3.357,3.401,3.444,3.488,3.532,3.575 --episodes 3 --speed 10 --outdir $OUT > $OUT/run.log 2>&1
echo "[$(date '+%m-%d %H:%M')] fineband END rc=$? rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a isaac_fineband.log
kill_all; touch FINEBAND_DONE
