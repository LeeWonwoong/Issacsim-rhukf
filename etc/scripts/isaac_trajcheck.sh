#!/usr/bin/env bash
# trajcheck (2026-09-10) — 무공격 전 패턴 reference vs 실제 궤적 (PX4 추종 버그 점검). {무풍, ws6} × 5패턴 × track × 2ep
cd /home/acsl/projects/Issacsim-rhukf
while [ ! -f WINDZU_DONE ]; do sleep 60; done
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07 EP_MAX_STEPS=300 SWEEP_ATTACK_TYPE=tilt
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
kill_all; OUT=results_trajcheck; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] trajcheck START" | tee isaac_trajcheck.log
WIND_START=60 WIND_END=290 SWEEP_ATK_START=9999 SWEEP_ATK_END=9999 CAPTURE_POLICIES="track" \
timeout 3600 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns waypoint,circle,figure8,aggressive,scurve --capture-disturbances none:0,wind_turbulence:6 \
  --capture-biases 0 --episodes 2 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
echo "[$(date '+%m-%d %H:%M')] trajcheck END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a isaac_trajcheck.log
kill_all; touch TRAJCHECK_DONE
