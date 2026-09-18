#!/usr/bin/env bash
# hardhover — 강한 호버(HOVER_HARD_GAIN=사수모드: 앵커너머 가상타깃+속도FF 6m/s) 검증.
#   급작 δ{0.75,0.80,0.85} · hold 5s · 무풍 · 전5패턴 · track vs dhover3(강한호버) × 3ep
#   판정: ① 강호버 절대끌림 (기존 29m → 얼마?) ② track 여전히 LOC ③ 강호버 자세(flip 안하나)
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MC_INT_LIM MPC_TILTMAX MPC_ACC_HOR
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07 EP_MAX_STEPS=300 SWEEP_ATTACK_TYPE=tilt
export HOVER_HARD_GAIN=2.0    # 앵커 너머 2× 가상타깃 + 속도FF (강한 사수)
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
kill_all; OUT=results_hardhover; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] hardhover START (HARD_GAIN=2.0, 급작 δ0.75/0.8/0.85 hold5s)" | tee isaac_hardhover.log
SWEEP_ATK_START=180 SWEEP_ATK_END=230 CAPTURE_POLICIES="track,dhover3" \
timeout 9000 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns waypoint,circle,figure8,aggressive,scurve --capture-disturbances none:0 \
  --capture-biases 3.270,3.488,3.706 --episodes 3 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
echo "[$(date '+%m-%d %H:%M')] hardhover END rc=$? rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a isaac_hardhover.log
kill_all; touch HARDHOVER_DONE
