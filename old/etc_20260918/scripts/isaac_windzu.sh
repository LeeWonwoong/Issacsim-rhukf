#!/usr/bin/env bash
# windzu (2026-09-10) — 바람 라벨이 정확한 zu 캡처: 강풍 창(60~280) × {clean, δ0.2, δ0.4} × ON15(180~195) × track × 5패턴 × 2ep
#   목적: R_gyro 스캔(vel 보조채널 부활) 을 정확한 바람/공격 라벨로 재생 — 강풍 vs 강풍+약공격, vel 의 강풍 오탐
cd /home/acsl/projects/Issacsim-rhukf
while [ ! -f SPARSE_DONE ]; do sleep 60; done
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07 EP_MAX_STEPS=300 SWEEP_ATTACK_TYPE=tilt
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
kill_all; OUT=results_windzu; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] windzu START" | tee isaac_windzu.log
WIND_START=60 WIND_END=280 SWEEP_ATK_START=180 SWEEP_ATK_END=195 CAPTURE_POLICIES="track" \
timeout 3600 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns waypoint,circle,figure8,aggressive,scurve --capture-disturbances none:0,wind_turbulence:6,wind_turbulence:8 \
  --capture-biases 0,0.872,1.744 --episodes 2 --speed 2.5 --log-zu --outdir $OUT > $OUT/run.log 2>&1
echo "[$(date '+%m-%d %H:%M')] windzu END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a isaac_windzu.log
kill_all; touch WINDZU_DONE
