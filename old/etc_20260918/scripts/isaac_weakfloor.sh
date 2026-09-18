#!/usr/bin/env bash
# weakfloor (2026-09-10) — 학습 가능 하한 δ 확정용: δ{0.10,0.15,0.20} × {무풍, ws6} × ON15 × track × 5패턴 × 2ep = 60에피 (~25분), zu 로그(G2 재생용)
#   판정: 온셋 후 gyro 고원이 평시(FF 후 aggressive 포함) p95/1.0/1.2 임계를 넘는 δ 최소값 → attack_delta_range 하한
cd /home/acsl/projects/Issacsim-rhukf
while [ ! -f BANDCAP_DONE ]; do sleep 60; done
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07 EP_MAX_STEPS=300 SWEEP_ATTACK_TYPE=tilt
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
kill_all; OUT=results_weakfloor; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] weakfloor START" | tee isaac_weakfloor.log
WIND_START=60 WIND_END=280 SWEEP_ATK_START=180 SWEEP_ATK_END=195 CAPTURE_POLICIES="track" \
timeout 3600 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns waypoint,circle,figure8,aggressive,scurve --capture-disturbances none:0,wind_turbulence:6 \
  --capture-biases 0.436,0.654,0.872 --episodes 2 --speed 2.5 --log-zu --outdir $OUT > $OUT/run.log 2>&1
echo "[$(date '+%m-%d %H:%M')] weakfloor END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a isaac_weakfloor.log
kill_all; touch WEAKFLOOR_DONE
