#!/usr/bin/env bash
# isaac_slowab — 느린 환경 A/B: SPEED_SCALE 1.0 · MPC_ACC_HOR 3.0 (나머지 v3 동일)
#   판정: ①밴드 위치·패턴 균일성 (δ0.7~1.0 × track/dhover3 × 4패턴 × 바람2)
#         ②기동 스파이크 형태학 보존 여부 (detail 온셋 전 구간에서 측정)
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u
export SPEED_SCALE=1.0 MPC_ACC_HOR=3.0
export EP_MAX_STEPS=450 SWEEP_ATK_START=180 SWEEP_ATK_END=380 SWEEP_ATTACK_TYPE=tilt
export CAPTURE_POLICIES="track,dhover3"
LOG=isaac_slowab.log
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
kill_all
OUT=results_slowab; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] slowab START (scale1.0 acc3.0)" | tee $LOG
timeout 14400 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns waypoint,circle,figure8,aggressive \
  --capture-disturbances none:0,wind_turbulence:8 \
  --capture-biases 3.052,3.488,3.924,4.360 --episodes 2 --speed 10 --outdir $OUT > $OUT/run.log 2>&1
echo "[$(date '+%m-%d %H:%M')] slowab END rc=$? rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a $LOG
kill_all; touch SLOWAB_DONE
