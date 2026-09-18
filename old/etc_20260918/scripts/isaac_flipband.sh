#!/usr/bin/env bash
# flipband — 확정 측정: 급작(rise0) δ{0.75,0.80,0.85} · hold 5초 · 무풍 · 전5패턴 · track/dhover3 × 5ep
#   판정: track flip 사망률(≥0.9) ∧ dhover 생존률(=1.0) + flip lag 분포(데드라인)
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MPC_ACC_HOR
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07 EP_MAX_STEPS=300 SWEEP_ATTACK_TYPE=tilt
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
kill_all; OUT=results_flipband; rm -rf $OUT; mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] flipband START (급작 rise0, hold 5s = 창 180-230)" | tee isaac_flipband.log
# ramp 없음(--ramp 미지정 = 급작 step), 공격창 180~230 = 5초
SWEEP_ATK_START=180 SWEEP_ATK_END=230 CAPTURE_POLICIES="track,dhover3" \
timeout 21600 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
  --capture-patterns waypoint,circle,figure8,aggressive,scurve --capture-disturbances none:0 \
  --capture-biases 3.270,3.488,3.706 --episodes 5 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
echo "[$(date '+%m-%d %H:%M')] flipband END rc=$? rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a isaac_flipband.log
kill_all; touch FLIPBAND_DONE
