#!/usr/bin/env bash
# gapfill — δ{0.78,0.80} × ws{6,7} × dhover3 × 5패턴 × 3ep = 60에피 (~30분)
#   목적: "δ0.80 + ws7 에서 호버 전생존" 확정 (밴드 상단·바람 상단 동시 확정의 마지막 칸)
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07 EP_MAX_STEPS=300 SWEEP_ATTACK_TYPE=tilt
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
kill_all; OUT=results_gapfill; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] gapfill START" | tee isaac_gapfill.log
WIND_START=60 WIND_END=280 SWEEP_ATK_START=180 SWEEP_ATK_END=230 CAPTURE_POLICIES="dhover3" \
timeout 7200 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns waypoint,circle,figure8,aggressive,scurve \
  --capture-disturbances wind_turbulence:6,wind_turbulence:7 \
  --capture-biases 3.401,3.488 --episodes 3 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
echo "[$(date '+%m-%d %H:%M')] gapfill END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a isaac_gapfill.log
kill_all; touch GAPFILL_DONE
