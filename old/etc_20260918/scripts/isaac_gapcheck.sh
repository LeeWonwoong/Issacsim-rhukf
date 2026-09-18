#!/usr/bin/env bash
# gapcheck (2026-09-11) — 약공격 상한 검증: δ{0.60,0.70} × ON30 × {무풍, ws6(arm .05)} × track × 5패턴 × 3ep = 60에피 (~25분)
#   판정: track 전원 생존이면 약공격 클래스 δ U(0.15,0.7)×ON U(5,30) 유지, 사망 있으면 상한 0.6 (또는 ON 상한 15)
cd /home/acsl/projects/Issacsim-rhukf
while [ ! -f results/claudecodefortest/BANDCAP_ARM_DONE ]; do sleep 60; done
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.05 EP_MAX_STEPS=300 SWEEP_ATTACK_TYPE=tilt
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
kill_all; OUT=results/claudecodefortest/gapcheck; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] gapcheck START (arm 0.05)" | tee results/claudecodefortest/isaac_gapcheck.log
WIND_START=60 WIND_END=290 SWEEP_ATK_START=180 SWEEP_ATK_END=210 CAPTURE_POLICIES="track" \
timeout 3600 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns waypoint,circle,figure8,aggressive,scurve --capture-disturbances none:0,wind_turbulence:6 \
  --capture-biases 2.616,3.052 --episodes 3 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
echo "[$(date '+%m-%d %H:%M')] gapcheck END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a results/claudecodefortest/isaac_gapcheck.log
kill_all; touch results/claudecodefortest/GAPCHECK_DONE
