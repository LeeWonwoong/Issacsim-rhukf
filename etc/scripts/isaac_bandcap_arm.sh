#!/usr/bin/env bash
# bandcap_arm (2026-09-10) — 바람 회전 모멘트 arm 축: WIND_MOMENT_ARM {0.05, 0.03} × ws6 만 (무풍 셀은 arm 무관)
#   5패턴 × δ{0,0.25,0.74~0.82} × {track,dhover3} × 2ep × ON30 = 140에피/arm (~1h). bandcap_on 뒤 자동.
#   출력: results/claudecodefortest/bandcap_arm{005,003}/ (+ 자동 분석). 비교 대상 = results_bandcap(arm 0.07) 의 ws6 열.
cd /home/acsl/projects/Issacsim-rhukf
while [ ! -f results/claudecodefortest/BANDCAP_ON_DONE ]; do sleep 60; done
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 EP_MAX_STEPS=300 SWEEP_ATTACK_TYPE=tilt
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
LOG=results/claudecodefortest/isaac_bandcap_arm.log
for ARM in 0.05 0.03; do
  TAG=${ARM/0./0}; kill_all; OUT=results/claudecodefortest/bandcap_arm$TAG; rm -rf $OUT; mkdir -p $OUT
  echo "[$(date '+%m-%d %H:%M')] bandcap ARM=$ARM START" | tee -a $LOG
  WIND_MOMENT_ARM=$ARM WIND_START=60 WIND_END=290 SWEEP_ATK_START=180 SWEEP_ATK_END=210 CAPTURE_POLICIES="track,dhover3" \
  timeout 10800 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
    --capture-patterns waypoint,circle,figure8,aggressive,scurve --capture-disturbances wind_turbulence:6 \
    --capture-biases 0,1.090,3.226,3.314,3.401,3.488,3.575 --episodes 2 --speed 2.5 --log-zu --outdir $OUT > $OUT/run.log 2>&1
  echo "[$(date '+%m-%d %H:%M')] bandcap ARM=$ARM END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a $LOG
  (setsid nohup python3 etc/scripts/analyze_bandcap.py $OUT 30 > $OUT/analysis.log 2>&1 < /dev/null &)
done
kill_all; touch results/claudecodefortest/BANDCAP_ARM_DONE; echo "[$(date '+%m-%d %H:%M')] bandcap_arm 완료" | tee -a $LOG
