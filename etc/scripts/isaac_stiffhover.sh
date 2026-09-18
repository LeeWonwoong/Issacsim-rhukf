#!/usr/bin/env bash
# stiffhover (C실험) — 제어권한 상향: MPC_ACC_HOR 3→8, TILTMAX 45→60 (INT_LIM 은 stock 0.3 유지)
#   급작 δ0.8 · hold 5s · 무풍 · 전5패턴 · track/dhover3 × 3ep
#   판정: ① hover 절대끌림 (사수되나 — 목표 <5m·정지) ② track 여전히 LOC 인가 (결과성 유지)
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07 EP_MAX_STEPS=300 SWEEP_ATTACK_TYPE=tilt
export MPC_ACC_HOR=8.0 MPC_TILTMAX=60.0
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
kill_all; OUT=results_stiffhover; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] stiffhover START (ACC_HOR=8 TILTMAX=60, 급작 δ0.8 hold5s)" | tee isaac_stiffhover.log
SWEEP_ATK_START=180 SWEEP_ATK_END=230 CAPTURE_POLICIES="track,dhover3" \
timeout 7200 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns waypoint,circle,figure8,aggressive,scurve --capture-disturbances none:0 \
  --capture-biases 3.488 --episodes 3 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
echo "[$(date '+%m-%d %H:%M')] stiffhover END rc=$? rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a isaac_stiffhover.log
kill_all; touch STIFFHOVER_DONE
