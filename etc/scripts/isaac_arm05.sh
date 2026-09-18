#!/usr/bin/env bash
# arm 0.05 검증 — ①생존 창: δ{0.75,0.8} × ws{8,10} × track/dhover3 × 3패턴
#                ②aliasing 보존: 온셋 전 평시 프레임에서 바닥/스파이크 측정 (detail 에서 후처리)
cd /home/acsl/projects/Issacsim-rhukf
while [ ! -f WS6CHK_DONE ]; do sleep 60; done
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u
export SPEED_SCALE=1.0 MPC_ACC_HOR=4.0 WIND_MOMENT_ARM=0.05
export DISPLAY=:1   # GUI 모드 (사용자 요청 09-07)
export EP_MAX_STEPS=450 SWEEP_ATK_START=180 SWEEP_ATK_END=380 SWEEP_ATTACK_TYPE=tilt
export CAPTURE_POLICIES="track,dhover3"
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
kill_all
OUT=results_arm05; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] arm05 START" | tee isaac_arm05.log
timeout 14400 python3 -u online_rl_main.py --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns waypoint,figure8,aggressive \
  --capture-disturbances wind_turbulence:8,wind_turbulence:10 \
  --capture-biases 3.270,3.488 --episodes 4 --speed 10 --outdir $OUT > $OUT/run.log 2>&1
echo "[$(date '+%m-%d %H:%M')] arm05 END rc=$? rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a isaac_arm05.log
kill_all; touch ARM05_DONE
