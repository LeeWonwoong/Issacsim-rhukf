#!/usr/bin/env bash
# bandfill — 경계대 채움: δ{0.72,0.74,0.76,0.82,0.85} × {무풍,ws6,ws7} × track/dhover3(약호버)
#   × 전5패턴 × 3ep = 450에피. 급작 rise0 · hold 5s (flipband 과 동일조건 — 직접 비교 가능).
#   산출: P(치명|δ,ws,패턴) 곡선 + 호버 생존 상한(ws7? δ0.82?) → 샘플링 스펙 완성
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07 EP_MAX_STEPS=300 SWEEP_ATTACK_TYPE=tilt
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
kill_all; OUT=results_bandfill; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] bandfill START (450에피 ~2h)" | tee isaac_bandfill.log
WIND_START=60 WIND_END=280 SWEEP_ATK_START=180 SWEEP_ATK_END=230 CAPTURE_POLICIES="track,dhover3" \
timeout 14400 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns waypoint,circle,figure8,aggressive,scurve \
  --capture-disturbances none:0,wind_turbulence:6,wind_turbulence:7 \
  --capture-biases 3.139,3.226,3.314,3.575,3.706 --episodes 3 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
echo "[$(date '+%m-%d %H:%M')] bandfill END rc=$? rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a isaac_bandfill.log
kill_all; touch BANDFILL_DONE
