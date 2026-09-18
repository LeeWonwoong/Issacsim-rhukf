#!/usr/bin/env bash
# dh1check (2026-09-11) — "즉각 호버는 즉사형 flip 을 막나 / ws5 가 더 깔끔한가": arm 0.05 · ON30 · δ{0.76,0.78,0.80} × ws{5,6} × {track,dhover1,dhover3} × 5패턴 × 2ep = 180에피 (~1.2h)
cd /home/acsl/projects/Issacsim-rhukf
while pgrep -f "isaac_holdchec[k]" >/dev/null; do sleep 30; done
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.05 EP_MAX_STEPS=300 SWEEP_ATTACK_TYPE=tilt
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
kill_all; OUT=results/claudecodefortest/dh1check; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] dh1check START" | tee results/claudecodefortest/isaac_dh1check.log
WIND_START=60 WIND_END=290 SWEEP_ATK_START=180 SWEEP_ATK_END=210 CAPTURE_POLICIES="track,dhover1,dhover3" \
timeout 7200 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns waypoint,circle,figure8,aggressive,scurve --capture-disturbances wind_turbulence:5,wind_turbulence:6 \
  --capture-biases 3.314,3.401,3.488 --episodes 2 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
echo "[$(date '+%m-%d %H:%M')] dh1check END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a results/claudecodefortest/isaac_dh1check.log
kill_all; touch results/claudecodefortest/DH1CHECK_DONE
