#!/usr/bin/env bash
# intsweep — INT_LIM↓ 로 약공격(δ0.2~0.5)을 결과적으로 만들 수 있나 (프레임워크 B 관문 측정).
#   INT_LIM {0.15, 0.08} × δ {0.2,0.3,0.4,0.5} × {track,dhover3} × {waypoint,aggressive} × 3ep = 96에피
#   지속 공격(120~390) · 무풍 · EP 400 (drift 경로 시간 확보) · 절대좌표 pos_x/pos_y 로깅.
#   판정: ①track 치명 문턱 δ* ②호버 격리(절대좌표) ③추락 경로(flip/drift)·시간 ④δ* NIS 애매도
cd /home/acsl/projects/Issacsim-rhukf
set +u; source /opt/ros/humble/setup.bash 2>/dev/null||true; source ~/colcon_ws/install/setup.bash 2>/dev/null||true
source etc/frozen_v3.env; set -u; unset MPC_TILTMAX MPC_ACC_HOR HOVER_HARD_GAIN
export SPEED_SCALE=1.0 WIND_MOMENT_ARM=0.07 EP_MAX_STEPS=400 SWEEP_ATTACK_TYPE=tilt
kill_all(){ pkill -f MicroXRCEAgent 2>/dev/null||true
  for p in $(pgrep -f "python.sh run_sim" 2>/dev/null); do kill -9 $p 2>/dev/null||true; done
  for p in $(ps -eo pid,args|grep "isaacsim/kit"|grep -v grep|awk '{print $1}'); do kill -9 $p 2>/dev/null||true; done; sleep 6; }
LOG=isaac_intsweep.log
for IL in 0.15 0.08; do
  kill_all; OUT=results_int_${IL/0./}; rm -rf $OUT; mkdir -p $OUT
  echo "[$(date '+%m-%d %H:%M')] INT_LIM=$IL START" | tee -a $LOG
  MC_INT_LIM=$IL SWEEP_ATK_START=120 SWEEP_ATK_END=390 CAPTURE_POLICIES="track,dhover3" \
  timeout 5400 python3 -u online_rl_main.py --headless --sweep --sweep-mode torque --capture-mode hijack \
    --capture-patterns waypoint,aggressive --capture-disturbances none:0 \
    --capture-biases 0.872,1.308,1.744,2.180 --episodes 3 --speed 2.5 --outdir $OUT > $OUT/run.log 2>&1
  echo "[$(date '+%m-%d %H:%M')] INT_LIM=$IL END rows=$(wc -l < $OUT/sweep_summary.csv 2>/dev/null||echo 0)" | tee -a $LOG
done
kill_all; touch INTSWEEP_DONE
